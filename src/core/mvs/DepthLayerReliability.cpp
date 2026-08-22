#include "DepthLayerReliability.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mvs
{
namespace
{

struct SurfaceSample
{
    int x = 0;
    int y = 0;
    double inverseDepth = 0.0;
};

float quantile(std::vector<float> values, float fraction)
{
    if (values.empty())
    {
        return -1.0f;
    }
    const float bounded = std::clamp(fraction, 0.0f, 1.0f);
    const std::size_t index = static_cast<std::size_t>(
        std::lround(bounded * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

bool compatibleFloatMap(const cv::Mat &map, cv::Size size)
{
    return map.type() == CV_32FC1 && map.size() == size;
}

cv::Mat normalizedGuide(const cv::Mat &guide, cv::Size size)
{
    if (guide.empty() || guide.channels() != 1)
    {
        return {};
    }
    cv::Mat resized;
    if (guide.size() == size)
    {
        resized = guide;
    }
    else
    {
        cv::resize(guide, resized, size, 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat normalized;
    if (resized.depth() == CV_8U)
    {
        resized.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
    }
    else if (resized.depth() == CV_32F)
    {
        normalized = resized.clone();
    }
    return normalized;
}

struct SurfaceDomain
{
    double centerX = 0.0;
    double centerY = 0.0;
    double scaleX = 1.0;
    double scaleY = 1.0;
};

using SurfaceCoefficients = cv::Vec<double, 6>;

SurfaceDomain makeSurfaceDomain(const cv::Rect &bounds)
{
    SurfaceDomain domain;
    domain.centerX = bounds.x + 0.5 * std::max(1, bounds.width);
    domain.centerY = bounds.y + 0.5 * std::max(1, bounds.height);
    domain.scaleX = std::max(1.0, 0.5 * std::max(1, bounds.width));
    domain.scaleY = std::max(1.0, 0.5 * std::max(1, bounds.height));
    return domain;
}

cv::Vec<double, 6> surfaceBasis(int x,
                                int y,
                                const SurfaceDomain &domain)
{
    const double u = (static_cast<double>(x) + 0.5 - domain.centerX) /
        domain.scaleX;
    const double v = (static_cast<double>(y) + 0.5 - domain.centerY) /
        domain.scaleY;
    return {1.0, u, v, u * u, u * v, v * v};
}

double predictedInverseDepth(const SurfaceCoefficients &surface,
                             int x,
                             int y,
                             const SurfaceDomain &domain)
{
    const cv::Vec<double, 6> basis = surfaceBasis(x, y, domain);
    return surface.dot(basis);
}

bool fitInverseDepthSurface(const std::vector<SurfaceSample> &samples,
                            const SurfaceDomain &domain,
                            SurfaceCoefficients *coefficients,
                            float *fit_p90)
{
    constexpr int kCoefficientCount = 6;
    if (!coefficients || !fit_p90 ||
        samples.size() < kCoefficientCount)
    {
        return false;
    }
    auto solve_samples = [&](const std::vector<SurfaceSample> &subset,
                             SurfaceCoefficients *solution)
    {
        cv::Mat design(
            static_cast<int>(subset.size()), kCoefficientCount, CV_64FC1);
        cv::Mat observations(static_cast<int>(subset.size()), 1, CV_64FC1);
        for (int index = 0; index < static_cast<int>(subset.size()); ++index)
        {
            const SurfaceSample &sample = subset[static_cast<std::size_t>(index)];
            const cv::Vec<double, kCoefficientCount> basis = surfaceBasis(
                sample.x, sample.y, domain);
            for (int coefficient = 0;
                 coefficient < kCoefficientCount;
                 ++coefficient)
            {
                design.at<double>(index, coefficient) = basis[coefficient];
            }
            observations.at<double>(index, 0) = sample.inverseDepth;
        }
        cv::Mat solved;
        if (!cv::solve(design, observations, solved, cv::DECOMP_SVD) ||
            solved.rows != kCoefficientCount)
        {
            return false;
        }
        for (int coefficient = 0;
             coefficient < kCoefficientCount;
             ++coefficient)
        {
            (*solution)[coefficient] = solved.at<double>(coefficient);
        }
        return true;
    };

    SurfaceCoefficients initial;
    if (!solve_samples(samples, &initial))
    {
        return false;
    }
    auto residuals_for = [&](const SurfaceCoefficients &surface)
    {
        std::vector<float> residuals;
        residuals.reserve(samples.size());
        for (const SurfaceSample &sample : samples)
        {
            const double predicted = predictedInverseDepth(
                surface, sample.x, sample.y, domain);
            residuals.push_back(predicted > 0.0
                ? static_cast<float>(std::fabs(sample.inverseDepth - predicted) / predicted)
                : std::numeric_limits<float>::infinity());
        }
        return residuals;
    };

    const std::vector<float> initial_residuals = residuals_for(initial);
    const float trim_threshold = quantile(initial_residuals, 0.80f);
    std::vector<SurfaceSample> trimmed;
    trimmed.reserve(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        if (std::isfinite(initial_residuals[index]) &&
            initial_residuals[index] <= trim_threshold)
        {
            trimmed.push_back(samples[index]);
        }
    }
    SurfaceCoefficients robust = initial;
    if (trimmed.size() >= kCoefficientCount)
    {
        solve_samples(trimmed, &robust);
    }
    const std::vector<float> final_residuals = residuals_for(robust);
    *coefficients = robust;
    *fit_p90 = quantile(final_residuals, 0.90f);
    return std::isfinite(*fit_p90);
}

float predictedDepth(const SurfaceCoefficients &surface,
                     int x,
                     int y,
                     const SurfaceDomain &domain)
{
    const double inverse_depth = predictedInverseDepth(
        surface, x, y, domain);
    return inverse_depth > 0.0 && std::isfinite(inverse_depth)
        ? static_cast<float>(1.0 / inverse_depth)
        : 0.0f;
}

} // namespace

DepthLayerReliabilityResult analyzeDepthLayerReliability(
    const cv::Mat &depth,
    const cv::Mat &guideGray,
    const cv::Mat &effectiveViewCount,
    const cv::Mat &conflictRatio,
    const cv::Mat &inverseDepthRelativeSpread,
    const DepthLayerReliabilityOptions &options)
{
    DepthLayerReliabilityResult result;
    if (depth.type() != CV_32FC1 || depth.empty() ||
        !compatibleFloatMap(effectiveViewCount, depth.size()) ||
        !compatibleFloatMap(conflictRatio, depth.size()) ||
        !compatibleFloatMap(inverseDepthRelativeSpread, depth.size()))
    {
        result.errorMessage = "missing_or_incompatible_depth_geometry_evidence";
        return result;
    }
    const cv::Mat guide = normalizedGuide(guideGray, depth.size());
    if (guide.empty())
    {
        result.errorMessage = "missing_or_incompatible_reference_guide";
        return result;
    }

    const int texture_radius = std::clamp(options.textureRadiusPixels, 1, 16);
    const int texture_size = 2 * texture_radius + 1;
    cv::Mat guide_mean;
    cv::Mat guide_squared_mean;
    cv::boxFilter(guide, guide_mean, CV_32FC1,
                  cv::Size(texture_size, texture_size), cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT_101);
    cv::boxFilter(guide.mul(guide), guide_squared_mean, CV_32FC1,
                  cv::Size(texture_size, texture_size), cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT_101);
    cv::Mat variance = guide_squared_mean - guide_mean.mul(guide_mean);
    cv::max(variance, 0.0, variance);
    cv::sqrt(variance, variance);

    result.classMap = cv::Mat(
        depth.size(), CV_8UC1,
        cv::Scalar(static_cast<std::uint8_t>(
            DepthLayerReliabilityClass::Unobservable)));
    cv::Mat candidates(depth.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < depth.rows; ++y)
    {
        const float *depth_row = depth.ptr<float>(y);
        const float *texture_row = variance.ptr<float>(y);
        const float *effective_row = effectiveViewCount.ptr<float>(y);
        const float *conflict_row = conflictRatio.ptr<float>(y);
        const float *spread_row = inverseDepthRelativeSpread.ptr<float>(y);
        std::uint8_t *class_row = result.classMap.ptr<std::uint8_t>(y);
        std::uint8_t *candidate_row = candidates.ptr<std::uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x)
        {
            if (!(depth_row[x] > 0.0f) || !std::isfinite(depth_row[x]))
            {
                continue;
            }
            ++result.validPixelCount;
            class_row[x] = static_cast<std::uint8_t>(
                DepthLayerReliabilityClass::Reliable);
            const bool low_texture = std::isfinite(texture_row[x]) &&
                texture_row[x] <= options.maximumLowTextureStandardDeviation;
            const bool weak_geometry =
                std::isfinite(effective_row[x]) &&
                effective_row[x] < options.maximumWeakEffectiveViewCount &&
                ((std::isfinite(conflict_row[x]) &&
                  conflict_row[x] > options.minimumWeakConflictRatio) ||
                 (std::isfinite(spread_row[x]) &&
                  spread_row[x] > options.minimumWeakInverseDepthSpread));
            result.lowTexturePixelCount += low_texture ? 1 : 0;
            result.weakGeometryPixelCount += weak_geometry ? 1 : 0;
            if (low_texture && weak_geometry)
            {
                candidate_row[x] = 255;
                class_row[x] = static_cast<std::uint8_t>(
                    DepthLayerReliabilityClass::AmbiguousLowTexture);
                ++result.candidatePixelCount;
            }
        }
    }

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        candidates, labels, statistics, centroids, 8, CV_32S);
    const int minimum_area = std::max(1, options.minimumComponentArea);
    const int ring_radius = std::clamp(options.boundaryRingRadiusPixels, 1, 32);
    const cv::Mat ring_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * ring_radius + 1, 2 * ring_radius + 1));
    for (int component = 1; component < component_count; ++component)
    {
        const int area = statistics.at<int>(component, cv::CC_STAT_AREA);
        if (area < minimum_area)
        {
            continue;
        }
        DepthLayerReliabilityComponent record;
        record.pixelCount = area;
        record.bounds = cv::Rect(
            statistics.at<int>(component, cv::CC_STAT_LEFT),
            statistics.at<int>(component, cv::CC_STAT_TOP),
            statistics.at<int>(component, cv::CC_STAT_WIDTH),
            statistics.at<int>(component, cv::CC_STAT_HEIGHT));
        const int anchor_left = std::max(0, record.bounds.x - ring_radius);
        const int anchor_top = std::max(0, record.bounds.y - ring_radius);
        const int anchor_right = std::min(
            depth.cols, record.bounds.x + record.bounds.width + ring_radius);
        const int anchor_bottom = std::min(
            depth.rows, record.bounds.y + record.bounds.height + ring_radius);
        const cv::Rect anchor_bounds(
            anchor_left,
            anchor_top,
            anchor_right - anchor_left,
            anchor_bottom - anchor_top);
        const cv::Mat component_mask = labels(anchor_bounds) == component;
        cv::Mat dilated;
        cv::dilate(component_mask, dilated, ring_kernel);
        cv::Mat ring;
        cv::subtract(dilated, component_mask, ring);

        std::vector<SurfaceSample> anchors;
        for (int local_y = 0; local_y < anchor_bounds.height; ++local_y)
        {
            const int y = anchor_bounds.y + local_y;
            const std::uint8_t *ring_row = ring.ptr<std::uint8_t>(local_y);
            const float *depth_row = depth.ptr<float>(y);
            const float *effective_row = effectiveViewCount.ptr<float>(y);
            const float *conflict_row = conflictRatio.ptr<float>(y);
            const float *spread_row = inverseDepthRelativeSpread.ptr<float>(y);
            for (int local_x = 0; local_x < anchor_bounds.width; ++local_x)
            {
                const int x = anchor_bounds.x + local_x;
                const bool reliable_anchor = ring_row[local_x] != 0 &&
                    depth_row[x] > 0.0f && std::isfinite(depth_row[x]) &&
                    std::isfinite(effective_row[x]) &&
                    effective_row[x] >=
                        options.minimumBoundaryEffectiveViewCount &&
                    std::isfinite(conflict_row[x]) &&
                    conflict_row[x] <= options.maximumBoundaryConflictRatio &&
                    std::isfinite(spread_row[x]) &&
                    spread_row[x] <=
                        options.maximumBoundaryInverseDepthSpread;
                if (reliable_anchor)
                {
                    anchors.push_back({x, y, 1.0 / depth_row[x]});
                }
            }
        }
        record.boundaryAnchorCount = static_cast<int>(anchors.size());
        const SurfaceDomain surface_domain = makeSurfaceDomain(anchor_bounds);
        SurfaceCoefficients surface;
        if (record.boundaryAnchorCount >= options.minimumBoundaryAnchorCount &&
            fitInverseDepthSurface(anchors, surface_domain, &surface,
                                   &record.boundarySurfaceFitP90))
        {
            std::vector<float> signed_residuals;
            std::vector<float> absolute_residuals;
            signed_residuals.reserve(static_cast<std::size_t>(area));
            absolute_residuals.reserve(static_cast<std::size_t>(area));
            for (int y = record.bounds.y;
                 y < record.bounds.y + record.bounds.height;
                 ++y)
            {
                const int *label_row = labels.ptr<int>(y);
                const float *depth_row = depth.ptr<float>(y);
                for (int x = record.bounds.x;
                     x < record.bounds.x + record.bounds.width;
                     ++x)
                {
                    if (label_row[x] != component)
                    {
                        continue;
                    }
                    const float predicted = predictedDepth(
                        surface, x, y, surface_domain);
                    if (predicted > 0.0f && depth_row[x] > 0.0f)
                    {
                        const float residual =
                            (depth_row[x] - predicted) / predicted;
                        signed_residuals.push_back(residual);
                        absolute_residuals.push_back(std::fabs(residual));
                    }
                }
            }
            record.signedRelativeResidualMedian =
                quantile(signed_residuals, 0.50f);
            record.absoluteRelativeResidualMedian =
                quantile(absolute_residuals, 0.50f);
            if (record.boundarySurfaceFitP90 <=
                    options.maximumBoundarySurfaceFitP90 &&
                record.absoluteRelativeResidualMedian >=
                    options.minimumRejectedLayerRelativeResidual)
            {
                record.reliabilityClass =
                    DepthLayerReliabilityClass::RejectedLayer;
                result.classMap(anchor_bounds).setTo(
                    static_cast<std::uint8_t>(
                        DepthLayerReliabilityClass::RejectedLayer),
                    component_mask);
                result.rejectedLayerPixelCount += area;
                ++result.rejectedLayerComponentCount;
            }
        }
        result.components.push_back(record);
    }
    std::sort(
        result.components.begin(), result.components.end(),
        [](const DepthLayerReliabilityComponent &left,
           const DepthLayerReliabilityComponent &right)
        {
            return left.pixelCount > right.pixelCount;
        });
    if (static_cast<int>(result.components.size()) >
        options.maximumReportedComponents)
    {
        result.components.resize(static_cast<std::size_t>(
            std::max(0, options.maximumReportedComponents)));
    }
    result.ambiguousPixelCount = result.candidatePixelCount -
        result.rejectedLayerPixelCount;
    result.reliablePixelCount = result.validPixelCount -
        result.candidatePixelCount;
    result.validInputs = true;
    return result;
}

} // namespace xjw::mvs
