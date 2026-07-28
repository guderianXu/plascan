#include "DepthFusionFramePolicy.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

double dot(const std::array<double, 3> &left,
           const std::array<double, 3> &right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

std::array<double, 3> subtract(const std::array<double, 3> &left,
                               const std::array<double, 3> &right)
{
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2]
    };
}

} // namespace

OrbitalCoverageStatistics DepthFusionFramePolicy::evaluateOrbitalCoverage(
    const std::vector<DepthFusionView> &views,
    const std::vector<float> &weights)
{
    OrbitalCoverageStatistics result;
    if (views.size() != weights.size() || views.size() < 3)
    {
        return result;
    }

    std::array<double, 3> centroid{};
    for (const DepthFusionView &view : views)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            centroid[axis] += view.cameraCenter[axis];
        }
    }
    for (double &coordinate : centroid)
    {
        coordinate /= static_cast<double>(views.size());
    }

    std::vector<std::array<double, 3>> active_centers;
    active_centers.reserve(views.size());
    for (std::size_t index = 0; index < views.size(); ++index)
    {
        if (weights[index] > 0.0f)
        {
            active_centers.push_back(views[index].cameraCenter);
        }
    }
    result.activeViewCount = static_cast<int>(active_centers.size());
    if (active_centers.size() < 3)
    {
        return result;
    }

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64FC1);
    for (const DepthFusionView &view : views)
    {
        const std::array<double, 3> offset =
            subtract(view.cameraCenter, centroid);
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                covariance.at<double>(row, column) +=
                    offset[row] * offset[column];
            }
        }
    }

    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    if (!cv::eigen(covariance, eigenvalues, eigenvectors) ||
        eigenvectors.rows != 3 || eigenvectors.cols != 3)
    {
        return result;
    }
    std::array<double, 3> axis_u{
        eigenvectors.at<double>(0, 0),
        eigenvectors.at<double>(0, 1),
        eigenvectors.at<double>(0, 2)
    };
    std::array<double, 3> axis_v{
        eigenvectors.at<double>(1, 0),
        eigenvectors.at<double>(1, 1),
        eigenvectors.at<double>(1, 2)
    };
    if (!std::isfinite(dot(axis_u, axis_u)) ||
        !std::isfinite(dot(axis_v, axis_v)) ||
        dot(axis_u, axis_u) <= std::numeric_limits<double>::epsilon() ||
        dot(axis_v, axis_v) <= std::numeric_limits<double>::epsilon())
    {
        return result;
    }

    std::vector<double> angles;
    angles.reserve(active_centers.size());
    for (const auto &center : active_centers)
    {
        const std::array<double, 3> offset = subtract(center, centroid);
        angles.push_back(std::atan2(dot(offset, axis_v), dot(offset, axis_u)));
    }
    std::sort(angles.begin(), angles.end());

    std::vector<double> gaps;
    gaps.reserve(angles.size());
    for (std::size_t index = 1; index < angles.size(); ++index)
    {
        gaps.push_back(angles[index] - angles[index - 1]);
    }
    gaps.push_back(angles.front() + 2.0 * 3.14159265358979323846 - angles.back());
    std::sort(gaps.begin(), gaps.end());

    const std::size_t middle = gaps.size() / 2;
    const double median_gap = gaps.size() % 2 == 0
        ? 0.5 * (gaps[middle - 1] + gaps[middle])
        : gaps[middle];
    const double maximum_gap = gaps.back();
    if (!std::isfinite(median_gap) || median_gap <= 1.0e-9 ||
        !std::isfinite(maximum_gap))
    {
        return result;
    }

    result.valid = true;
    result.medianAngularSpacingDegrees = median_gap * kRadiansToDegrees;
    result.maximumAngularGapDegrees = maximum_gap * kRadiansToDegrees;
    result.maximumAngularGapRatio = maximum_gap / median_gap;
    return result;
}

bool DepthFusionFramePolicy::canRejectWithoutCoverageGap(
    const std::vector<DepthFusionView> &views,
    const std::vector<float> &weights,
    int candidateFrameIndex,
    double maximumGapRatio,
    int minimumRetainedFrames,
    OrbitalCoverageStatistics *trialCoverage)
{
    if (views.size() != weights.size() ||
        candidateFrameIndex < 0 ||
        candidateFrameIndex >= static_cast<int>(weights.size()))
    {
        return false;
    }

    std::vector<float> trial_weights = weights;
    trial_weights[static_cast<std::size_t>(candidateFrameIndex)] = 0.0f;
    const OrbitalCoverageStatistics coverage =
        evaluateOrbitalCoverage(views, trial_weights);
    if (trialCoverage)
    {
        *trialCoverage = coverage;
    }
    if (coverage.activeViewCount < std::max(3, minimumRetainedFrames))
    {
        return false;
    }
    if (!coverage.valid)
    {
        return false;
    }
    return coverage.maximumAngularGapRatio <= std::max(1.0, maximumGapRatio);
}

} // namespace xjw::mesh
