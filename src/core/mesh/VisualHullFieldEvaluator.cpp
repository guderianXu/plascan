#include "VisualHullFieldEvaluator.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mesh::detail
{
namespace
{

cv::Mat signedDistanceFromMask(const cv::Mat &mask)
{
    if (mask.type() != CV_8UC1 || mask.empty())
    {
        return {};
    }

    cv::Mat foreground;
    cv::compare(mask, 0, foreground, cv::CMP_GT);
    cv::Mat background;
    cv::bitwise_not(foreground, background);

    cv::Mat inside_distance;
    cv::Mat outside_distance;
    cv::distanceTransform(
        foreground,
        inside_distance,
        cv::DIST_L2,
        cv::DIST_MASK_5);
    cv::distanceTransform(
        background,
        outside_distance,
        cv::DIST_L2,
        cv::DIST_MASK_5);
    return inside_distance - outside_distance;
}

bool bilinearSample(
    const cv::Mat &image,
    const double pixel[2],
    float *value)
{
    if (value == nullptr || image.type() != CV_32FC1 ||
        image.empty() || pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > static_cast<double>(image.cols - 1) ||
        pixel[1] > static_cast<double>(image.rows - 1))
    {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(pixel[0]));
    const int y0 = static_cast<int>(std::floor(pixel[1]));
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float tx = static_cast<float>(pixel[0] - x0);
    const float ty = static_cast<float>(pixel[1] - y0);
    const float top =
        image.at<float>(y0, x0) * (1.0f - tx) +
        image.at<float>(y0, x1) * tx;
    const float bottom =
        image.at<float>(y1, x0) * (1.0f - tx) +
        image.at<float>(y1, x1) * tx;
    *value = top * (1.0f - ty) + bottom * ty;
    return std::isfinite(*value);
}

} // namespace

std::vector<PreparedVisualHullView> prepareVisualHullFieldViews(
    const std::vector<VisualHullView> &views)
{
    std::vector<PreparedVisualHullView> prepared;
    prepared.reserve(views.size());
    for (const VisualHullView &view : views)
    {
        PreparedVisualHullView entry;
        entry.view = &view;
        entry.signedSilhouetteDistance =
            signedDistanceFromMask(view.silhouetteMask);
        prepared.push_back(std::move(entry));
    }
    return prepared;
}

float evaluateContinuousVisualHullField(
    float worldX,
    float worldY,
    float worldZ,
    const std::vector<PreparedVisualHullView> &views,
    const VisualHullConfig &config)
{
    thread_local std::vector<float> margins;
    margins.clear();
    if (margins.capacity() < views.size())
    {
        margins.reserve(views.size());
    }

    int free_space_violations = 0;
    const double world[3] = {worldX, worldY, worldZ};
    for (const PreparedVisualHullView &prepared : views)
    {
        if (prepared.view == nullptr ||
            prepared.signedSilhouetteDistance.empty())
        {
            continue;
        }
        const VisualHullView &view = *prepared.view;
        double pixel[2] = {};
        double camera_depth = 0.0;
        if (!view.camera.projectWorldPointWithDepth(
                world, pixel, camera_depth))
        {
            continue;
        }

        float signed_pixel_distance = 0.0f;
        if (!bilinearSample(
                prepared.signedSilhouetteDistance,
                pixel,
                &signed_pixel_distance))
        {
            continue;
        }

        const double mean_focal =
            0.5 * (
                std::abs(view.camera.focalX()) +
                std::abs(view.camera.focalY()));
        if (!(mean_focal > 1.0e-9) ||
            !std::isfinite(camera_depth))
        {
            continue;
        }
        margins.push_back(static_cast<float>(
            signed_pixel_distance * camera_depth / mean_focal));

        if (config.enableDepthFreeSpaceCarving &&
            !view.depthMap.empty())
        {
            const int column = static_cast<int>(
                std::lround(pixel[0]));
            const int row = static_cast<int>(
                std::lround(pixel[1]));
            if (row >= 0 && column >= 0 &&
                row < view.depthMap.rows &&
                column < view.depthMap.cols)
            {
                const float measured_depth =
                    view.depthMap.at<float>(row, column);
                if (std::isfinite(measured_depth) &&
                    measured_depth > 0.0f)
                {
                    const float tolerance = std::max(
                        1.0e-6f,
                        measured_depth *
                            config.relativeDepthTolerance);
                    if (camera_depth <
                        measured_depth - tolerance)
                    {
                        ++free_space_violations;
                    }
                }
            }
        }
    }

    if (static_cast<int>(margins.size()) <
        config.minimumVisibleViews)
    {
        return 1.0f;
    }
    if (config.enableDepthFreeSpaceCarving &&
        free_space_violations >=
            config.minimumDepthFreeSpaceViolations)
    {
        return 1.0f;
    }

    const std::size_t allowed_violations =
        static_cast<std::size_t>(std::clamp(
            config.allowedSilhouetteViolations,
            0,
            static_cast<int>(margins.size()) - 1));
    std::nth_element(
        margins.begin(),
        margins.begin() +
            static_cast<std::ptrdiff_t>(allowed_violations),
        margins.end());
    const float support_margin =
        margins[allowed_violations];
    if (!std::isfinite(support_margin))
    {
        return 1.0f;
    }
    return -support_margin;
}

} // namespace xjw::mesh::detail
