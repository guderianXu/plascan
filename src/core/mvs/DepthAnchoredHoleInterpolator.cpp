#include "DepthAnchoredHoleInterpolator.h"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::mvs
{
namespace
{

bool isValidDepth(float depth)
{
    return std::isfinite(depth) && depth > 0.0f;
}

cv::Mat makeGuide(const cv::Mat *guideGray, const cv::Size &size)
{
    if (!guideGray || guideGray->empty())
    {
        return cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    }
    cv::Mat gray;
    if (guideGray->channels() == 1)
    {
        gray = *guideGray;
    }
    else
    {
        cv::cvtColor(*guideGray, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.size() != size)
    {
        cv::resize(gray, gray, size, 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat result;
    gray.convertTo(result, CV_32FC1);
    return result;
}

float robustRelativeSpread(std::vector<float> values)
{
    if (values.empty())
    {
        return std::numeric_limits<float>::infinity();
    }
    std::sort(values.begin(), values.end());
    const std::size_t last = values.size() - 1;
    const float lower = values[static_cast<std::size_t>(
        std::floor(0.10 * static_cast<double>(last)))];
    const float median = values[last / 2];
    const float upper = values[static_cast<std::size_t>(
        std::ceil(0.90 * static_cast<double>(last)))];
    return (upper - lower) / std::max(1.0e-6f, median);
}

} // namespace

DepthAnchoredHoleInterpolationStats interpolateAnchoredInternalDepthHoles(
    cv::Mat &depth,
    const cv::Mat &support_mask,
    const cv::Mat &strong_anchor_mask,
    const cv::Mat *guide_gray,
    const DepthAnchoredHoleInterpolationOptions &options,
    cv::Mat *confidence,
    cv::Mat *repaired_mask)
{
    DepthAnchoredHoleInterpolationStats result;
    if (!options.enabled || depth.empty() || depth.type() != CV_32FC1 ||
        support_mask.type() != CV_8UC1 || support_mask.size() != depth.size() ||
        strong_anchor_mask.type() != CV_8UC1 ||
        strong_anchor_mask.size() != depth.size())
    {
        return result;
    }

    const bool has_confidence = confidence && confidence->type() == CV_32FC1 &&
        confidence->size() == depth.size();
    const bool has_repaired_mask = repaired_mask &&
        repaired_mask->type() == CV_8UC1 && repaired_mask->size() == depth.size();
    result.anchorPixelCount = static_cast<std::uint64_t>(
        cv::countNonZero(strong_anchor_mask));
    cv::Mat invalid_supported(depth.size(), CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < depth.rows; ++row)
    {
        const float *depth_row = depth.ptr<float>(row);
        const std::uint8_t *support_row = support_mask.ptr<std::uint8_t>(row);
        std::uint8_t *invalid_row = invalid_supported.ptr<std::uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            invalid_row[column] = support_row[column] != 0 &&
                    !isValidDepth(depth_row[column])
                ? 255 : 0;
        }
    }
    if (cv::countNonZero(invalid_supported) == 0)
    {
        return result;
    }

    cv::Mat interpolation_candidates = invalid_supported;
    if (options.allowSilhouetteConnectedInterior)
    {
        cv::Mat support_distance;
        cv::distanceTransform(
            support_mask, support_distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        const int protection_radius = std::clamp(
            options.silhouetteProtectionRadiusPixels, 1, 32);
        interpolation_candidates = invalid_supported.clone();
        const cv::Mat protected_band = support_distance <=
            static_cast<float>(protection_radius);
        interpolation_candidates.setTo(0, protected_band);

        cv::Mat original_labels;
        cv::Mat original_stats;
        cv::Mat original_centroids;
        const int original_component_count = cv::connectedComponentsWithStats(
            invalid_supported,
            original_labels,
            original_stats,
            original_centroids,
            8,
            CV_32S);
        for (int label = 1; label < original_component_count; ++label)
        {
            bool touches_silhouette = false;
            bool retains_interior = false;
            const int top = original_stats.at<int>(label, cv::CC_STAT_TOP);
            const int left = original_stats.at<int>(label, cv::CC_STAT_LEFT);
            const int height = original_stats.at<int>(label, cv::CC_STAT_HEIGHT);
            const int width = original_stats.at<int>(label, cv::CC_STAT_WIDTH);
            for (int row = top;
                 row < top + height && (!touches_silhouette || !retains_interior);
                 ++row)
            {
                for (int column = left;
                     column < left + width &&
                         (!touches_silhouette || !retains_interior);
                     ++column)
                {
                    if (original_labels.at<int>(row, column) != label)
                    {
                        continue;
                    }
                    const float distance = support_distance.at<float>(row, column);
                    touches_silhouette = touches_silhouette || distance <= 1.0f;
                    retains_interior = retains_interior ||
                        interpolation_candidates.at<std::uint8_t>(row, column) != 0;
                }
            }
            if (!touches_silhouette || !retains_interior)
            {
                continue;
            }
            ++result.protectedSilhouetteComponentCount;
            const cv::Mat component_mask = original_labels == label;
            const cv::Mat retained_mask =
                component_mask & (interpolation_candidates != 0);
            result.protectedSilhouettePixelCount +=
                static_cast<std::uint64_t>(
                    cv::countNonZero(component_mask) -
                    cv::countNonZero(retained_mask));
        }
    }
    if (cv::countNonZero(interpolation_candidates) == 0)
    {
        return result;
    }

    cv::Mat labels;
    cv::Mat component_stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        interpolation_candidates,
        labels,
        component_stats,
        centroids,
        8,
        CV_32S);
    const int support_area = cv::countNonZero(support_mask);
    const int maximum_area = std::max(
        1,
        std::min(
            std::max(1, options.maximumComponentArea),
            static_cast<int>(std::lround(
                std::max(0.001f, options.maximumComponentAreaRatio) *
                static_cast<float>(support_area)))));
    const int anchor_radius = std::clamp(options.anchorSearchRadius, 1, 4);
    cv::Mat expanded_anchors;
    cv::dilate(
        strong_anchor_mask,
        expanded_anchors,
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(anchor_radius * 2 + 1, anchor_radius * 2 + 1)));
    const cv::Mat guide = makeGuide(guide_gray, depth.size());
    const float guide_sigma = std::max(1.0f, options.guideColorSigma);
    const int maximum_iterations = std::clamp(options.maximumIterations, 8, 500);
    constexpr int offsets[4][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    for (int label = 1; label < component_count; ++label)
    {
        ++result.candidateComponentCount;
        const int area = component_stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > maximum_area)
        {
            ++result.rejectedAreaComponentCount;
            continue;
        }
        const int top = component_stats.at<int>(label, cv::CC_STAT_TOP);
        const int left = component_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int height = component_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int width = component_stats.at<int>(label, cv::CC_STAT_WIDTH);
        std::vector<cv::Point> pixels;
        std::vector<float> boundary_inverse_depths;
        pixels.reserve(static_cast<std::size_t>(area));
        boundary_inverse_depths.reserve(static_cast<std::size_t>(area));
        bool touches_silhouette = false;
        int anchor_contact_pixels = 0;

        for (int row = top; row < top + height; ++row)
        {
            for (int column = left; column < left + width; ++column)
            {
                if (labels.at<int>(row, column) != label)
                {
                    continue;
                }
                pixels.emplace_back(column, row);
                if (expanded_anchors.at<std::uint8_t>(row, column) != 0)
                {
                    ++anchor_contact_pixels;
                }
                for (const auto &offset : offsets)
                {
                    const int neighbor_row = row + offset[0];
                    const int neighbor_column = column + offset[1];
                    if (neighbor_row < 0 || neighbor_row >= depth.rows ||
                        neighbor_column < 0 || neighbor_column >= depth.cols ||
                        support_mask.at<std::uint8_t>(
                            std::clamp(neighbor_row, 0, depth.rows - 1),
                            std::clamp(neighbor_column, 0, depth.cols - 1)) == 0)
                    {
                        touches_silhouette = true;
                        continue;
                    }
                    if (labels.at<int>(neighbor_row, neighbor_column) == label)
                    {
                        continue;
                    }
                    const float neighbor_depth =
                        depth.at<float>(neighbor_row, neighbor_column);
                    if (isValidDepth(neighbor_depth))
                    {
                        boundary_inverse_depths.push_back(1.0f / neighbor_depth);
                    }
                }
            }
        }
        if (touches_silhouette)
        {
            ++result.rejectedSilhouetteComponentCount;
            continue;
        }
        if (anchor_contact_pixels < options.minimumAnchorContactPixelCount ||
            static_cast<int>(boundary_inverse_depths.size()) <
                options.minimumBoundarySampleCount)
        {
            ++result.rejectedAnchorComponentCount;
            continue;
        }
        if (robustRelativeSpread(boundary_inverse_depths) >
            options.maximumBoundaryInverseDepthSpread)
        {
            ++result.rejectedBoundarySpreadComponentCount;
            continue;
        }

        std::nth_element(
            boundary_inverse_depths.begin(),
            boundary_inverse_depths.begin() +
                static_cast<std::ptrdiff_t>(boundary_inverse_depths.size() / 2),
            boundary_inverse_depths.end());
        const float initial_inverse_depth =
            boundary_inverse_depths[boundary_inverse_depths.size() / 2];
        // Components are independent and bounded by their connected-component
        // rectangle. Allocating an image-sized scratch matrix for every small
        // hole made full-resolution streaming consistency repeatedly commit
        // tens of MiB even when the component covered only a few pixels.
        cv::Mat inverse_values(height, width, CV_32FC1, cv::Scalar(0.0f));
        for (const cv::Point &pixel : pixels)
        {
            inverse_values.at<float>(pixel.y - top, pixel.x - left) =
                initial_inverse_depth;
        }
        std::vector<cv::Vec4f> guide_weights(
            pixels.size(), cv::Vec4f::all(0.0f));
        std::vector<cv::Vec4b> component_neighbors(
            pixels.size(), cv::Vec4b::all(0));
        std::vector<cv::Vec4f> boundary_inverse_depth(
            pixels.size(), cv::Vec4f::all(0.0f));
        for (std::size_t pixel_index = 0;
             pixel_index < pixels.size();
             ++pixel_index)
        {
            const cv::Point &pixel = pixels[pixel_index];
            const float center_guide = guide.at<float>(pixel.y, pixel.x);
            for (int offset_index = 0; offset_index < 4; ++offset_index)
            {
                const int neighbor_row = pixel.y + offsets[offset_index][0];
                const int neighbor_column = pixel.x + offsets[offset_index][1];
                guide_weights[pixel_index][offset_index] = std::exp(
                    -std::fabs(
                        center_guide -
                        guide.at<float>(neighbor_row, neighbor_column)) /
                    guide_sigma);
                if (labels.at<int>(neighbor_row, neighbor_column) == label)
                {
                    component_neighbors[pixel_index][offset_index] = 1;
                    continue;
                }
                const float neighbor_depth =
                    depth.at<float>(neighbor_row, neighbor_column);
                boundary_inverse_depth[pixel_index][offset_index] =
                    isValidDepth(neighbor_depth) ? 1.0f / neighbor_depth : 0.0f;
            }
        }

        for (int iteration = 0; iteration < maximum_iterations; ++iteration)
        {
            float maximum_delta = 0.0f;
            for (std::size_t pixel_index = 0;
                 pixel_index < pixels.size();
                 ++pixel_index)
            {
                const cv::Point &pixel = pixels[pixel_index];
                float weighted_sum = 0.0f;
                float weight_sum = 0.0f;
                for (int offset_index = 0; offset_index < 4; ++offset_index)
                {
                    const int neighbor_row =
                        pixel.y + offsets[offset_index][0];
                    const int neighbor_column =
                        pixel.x + offsets[offset_index][1];
                    const float neighbor_inverse_depth =
                        component_neighbors[pixel_index][offset_index] != 0
                        ? inverse_values.at<float>(
                              neighbor_row - top, neighbor_column - left)
                        : boundary_inverse_depth[pixel_index][offset_index];
                    if (neighbor_inverse_depth <= 0.0f)
                    {
                        continue;
                    }
                    const float weight = guide_weights[pixel_index][offset_index];
                    weighted_sum += weight * neighbor_inverse_depth;
                    weight_sum += weight;
                }
                if (weight_sum <= 1.0e-6f)
                {
                    continue;
                }
                float &stored = inverse_values.at<float>(
                    pixel.y - top, pixel.x - left);
                const float updated = weighted_sum / weight_sum;
                maximum_delta = std::max(maximum_delta, std::fabs(updated - stored));
                stored = updated;
            }
            if (maximum_delta <= options.convergenceTolerance)
            {
                break;
            }
        }

        for (const cv::Point &pixel : pixels)
        {
            const float inverse_depth = inverse_values.at<float>(
                pixel.y - top, pixel.x - left);
            if (!std::isfinite(inverse_depth) || inverse_depth <= 1.0e-8f)
            {
                continue;
            }
            depth.at<float>(pixel.y, pixel.x) = 1.0f / inverse_depth;
            if (has_confidence)
            {
                confidence->at<float>(pixel.y, pixel.x) =
                    std::clamp(options.interpolatedConfidence, 0.0f, 0.65f);
            }
            if (has_repaired_mask)
            {
                repaired_mask->at<std::uint8_t>(pixel.y, pixel.x) = 255;
            }
            ++result.interpolatedPixelCount;
        }
        ++result.acceptedComponentCount;
    }
    return result;
}

QJsonObject depthAnchoredHoleInterpolationStatsToJson(
    const DepthAnchoredHoleInterpolationStats &stats)
{
    QJsonObject object;
    object.insert(
        QStringLiteral("anchor_pixel_count"),
        static_cast<double>(stats.anchorPixelCount));
    object.insert(
        QStringLiteral("candidate_component_count"),
        static_cast<double>(stats.candidateComponentCount));
    object.insert(
        QStringLiteral("accepted_component_count"),
        static_cast<double>(stats.acceptedComponentCount));
    object.insert(
        QStringLiteral("interpolated_pixel_count"),
        static_cast<double>(stats.interpolatedPixelCount));
    object.insert(
        QStringLiteral("rejected_silhouette_component_count"),
        static_cast<double>(stats.rejectedSilhouetteComponentCount));
    object.insert(
        QStringLiteral("rejected_area_component_count"),
        static_cast<double>(stats.rejectedAreaComponentCount));
    object.insert(
        QStringLiteral("rejected_anchor_component_count"),
        static_cast<double>(stats.rejectedAnchorComponentCount));
    object.insert(
        QStringLiteral("rejected_boundary_spread_component_count"),
        static_cast<double>(stats.rejectedBoundarySpreadComponentCount));
    object.insert(
        QStringLiteral("protected_silhouette_component_count"),
        static_cast<double>(stats.protectedSilhouetteComponentCount));
    object.insert(
        QStringLiteral("protected_silhouette_pixel_count"),
        static_cast<double>(stats.protectedSilhouettePixelCount));
    return object;
}

} // namespace xjw::mvs
