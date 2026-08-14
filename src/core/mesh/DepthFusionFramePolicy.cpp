#include "DepthFusionFramePolicy.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace xjw::mesh
{
namespace
{

constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

double normalizeAngle(double angle)
{
    while (angle < -3.14159265358979323846)
    {
        angle += kTwoPi;
    }
    while (angle >= 3.14159265358979323846)
    {
        angle -= kTwoPi;
    }
    return angle;
}

double circularDistance(double left, double right)
{
    return std::fabs(normalizeAngle(left - right));
}

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

const char *orbitalFrameRoleId(OrbitalFrameRole role)
{
    switch (role)
    {
    case OrbitalFrameRole::GapBoundary:
        return "gap_boundary";
    case OrbitalFrameRole::GapOpposite:
        return "gap_opposite";
    case OrbitalFrameRole::NormalSector:
    default:
        return "normal_sector";
    }
}

OrbitalCoverageStatistics DepthFusionFramePolicy::evaluateOrbitalCoverage(
    const std::vector<DepthFusionView> &views,
    const std::vector<float> &weights,
    double significantGapRatio)
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

    std::vector<DepthFusionView> active_views;
    active_views.reserve(views.size());
    for (std::size_t index = 0; index < views.size(); ++index)
    {
        if (weights[index] > 0.0f)
        {
            active_views.push_back(views[index]);
        }
    }
    result.activeViewCount = static_cast<int>(active_views.size());
    if (active_views.size() < 3)
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

    struct AngularView
    {
        DepthFusionView view;
        double angle = 0.0;
    };
    cv::Mat circle_system(
        static_cast<int>(views.size()), 3, CV_64FC1);
    cv::Mat circle_rhs(
        static_cast<int>(views.size()), 1, CV_64FC1);
    for (int index = 0; index < static_cast<int>(views.size()); ++index)
    {
        const std::array<double, 3> offset =
            subtract(views[static_cast<std::size_t>(index)].cameraCenter, centroid);
        const double projected_x = dot(offset, axis_u);
        const double projected_y = dot(offset, axis_v);
        circle_system.at<double>(index, 0) = projected_x;
        circle_system.at<double>(index, 1) = projected_y;
        circle_system.at<double>(index, 2) = 1.0;
        circle_rhs.at<double>(index, 0) =
            -(projected_x * projected_x + projected_y * projected_y);
    }
    cv::Mat circle_solution;
    double circle_center_x = 0.0;
    double circle_center_y = 0.0;
    if (cv::solve(
            circle_system,
            circle_rhs,
            circle_solution,
            cv::DECOMP_SVD)
        && circle_solution.rows == 3)
    {
        circle_center_x = -0.5 * circle_solution.at<double>(0, 0);
        circle_center_y = -0.5 * circle_solution.at<double>(1, 0);
    }

    std::vector<AngularView> angular_views;
    angular_views.reserve(active_views.size());
    for (const DepthFusionView &view : active_views)
    {
        const std::array<double, 3> offset =
            subtract(view.cameraCenter, centroid);
        angular_views.push_back(
            {view,
             std::atan2(
                 dot(offset, axis_v) - circle_center_y,
                 dot(offset, axis_u) - circle_center_x)});
    }
    std::sort(
        angular_views.begin(),
        angular_views.end(),
        [](const AngularView &left, const AngularView &right)
        {
            return left.angle < right.angle;
        });

    std::vector<double> gaps;
    gaps.reserve(angular_views.size());
    std::size_t maximum_gap_start_index = 0;
    double maximum_gap = -1.0;
    for (std::size_t index = 0; index < angular_views.size(); ++index)
    {
        const std::size_t next_index = (index + 1) % angular_views.size();
        const double next_angle = next_index == 0
            ? angular_views.front().angle + kTwoPi
            : angular_views[next_index].angle;
        const double gap = next_angle - angular_views[index].angle;
        gaps.push_back(gap);
        if (gap > maximum_gap)
        {
            maximum_gap = gap;
            maximum_gap_start_index = index;
        }
    }
    std::vector<double> sorted_gaps = gaps;
    std::sort(sorted_gaps.begin(), sorted_gaps.end());

    const std::size_t middle = sorted_gaps.size() / 2;
    const double median_gap = sorted_gaps.size() % 2 == 0
        ? 0.5 * (sorted_gaps[middle - 1] + sorted_gaps[middle])
        : sorted_gaps[middle];
    if (!std::isfinite(median_gap) || median_gap <= 1.0e-9 ||
        !std::isfinite(maximum_gap))
    {
        return result;
    }

    result.valid = true;
    result.medianAngularSpacingDegrees = median_gap * kRadiansToDegrees;
    result.maximumAngularGapDegrees = maximum_gap * kRadiansToDegrees;
    result.maximumAngularGapRatio = maximum_gap / median_gap;
    result.angularGapDegreesDescending.reserve(sorted_gaps.size());
    for (auto iterator = sorted_gaps.crbegin();
         iterator != sorted_gaps.crend();
         ++iterator)
    {
        result.angularGapDegreesDescending.push_back(
            *iterator * kRadiansToDegrees);
    }
    const std::size_t maximum_gap_end_index =
        (maximum_gap_start_index + 1) % angular_views.size();
    const AngularView &gap_start = angular_views[maximum_gap_start_index];
    const AngularView &gap_end = angular_views[maximum_gap_end_index];
    result.gapStartFrameIndex = gap_start.view.frameIndex;
    result.gapStartRefIndex = gap_start.view.refIndex;
    result.gapEndFrameIndex = gap_end.view.frameIndex;
    result.gapEndRefIndex = gap_end.view.refIndex;
    result.significantGap =
        result.maximumAngularGapRatio >= std::max(1.0, significantGapRatio);

    result.frameRoles.reserve(angular_views.size());
    for (const AngularView &angular_view : angular_views)
    {
        result.frameRoles.push_back(
            {angular_view.view.frameIndex,
             angular_view.view.refIndex,
             angular_view.angle * kRadiansToDegrees,
             OrbitalFrameRole::NormalSector});
    }
    if (result.significantGap)
    {
        const double gap_midpoint = normalizeAngle(
            gap_start.angle + maximum_gap * 0.5);
        const double opposite_angle = normalizeAngle(
            gap_midpoint + 3.14159265358979323846);
        std::size_t opposite_index = 0;
        double opposite_distance = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < angular_views.size(); ++index)
        {
            const double distance = circularDistance(
                angular_views[index].angle, opposite_angle);
            if (distance < opposite_distance)
            {
                opposite_distance = distance;
                opposite_index = index;
            }
        }
        result.gapOppositeFrameIndex =
            angular_views[opposite_index].view.frameIndex;
        result.gapOppositeRefIndex =
            angular_views[opposite_index].view.refIndex;
        for (OrbitalFrameRoleAssignment &assignment : result.frameRoles)
        {
            if (assignment.frameIndex == result.gapStartFrameIndex
                || assignment.frameIndex == result.gapEndFrameIndex)
            {
                assignment.role = OrbitalFrameRole::GapBoundary;
            }
            else if (assignment.frameIndex == result.gapOppositeFrameIndex)
            {
                assignment.role = OrbitalFrameRole::GapOpposite;
            }
        }
    }
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

std::vector<int> DepthFusionFramePolicy::selectCoverageComplementaryCandidates(
    const std::vector<DepthFusionView> &fixedViews,
    const std::vector<DepthFusionView> &candidateViews,
    int maximumSelectedCount)
{
    std::vector<int> selected;
    const int selection_count = std::clamp(
        maximumSelectedCount,
        0,
        static_cast<int>(candidateViews.size()));
    if (selection_count == 0 || fixedViews.size() < 3)
    {
        return selected;
    }

    std::vector<DepthFusionView> combined_views = fixedViews;
    combined_views.insert(
        combined_views.end(), candidateViews.begin(), candidateViews.end());
    std::vector<float> weights(combined_views.size(), 0.0f);
    std::fill_n(weights.begin(), fixedViews.size(), 1.0f);
    std::vector<std::uint8_t> candidate_selected(candidateViews.size(), 0);
    selected.reserve(static_cast<std::size_t>(selection_count));

    constexpr double kComparisonEpsilon = 1.0e-9;
    const auto has_better_gap_distribution = [kComparisonEpsilon](
        const OrbitalCoverageStatistics &candidate,
        const OrbitalCoverageStatistics &current)
    {
        const std::size_t comparable_count = std::min(
            candidate.angularGapDegreesDescending.size(),
            current.angularGapDegreesDescending.size());
        for (std::size_t index = 0; index < comparable_count; ++index)
        {
            const double difference =
                candidate.angularGapDegreesDescending[index] -
                current.angularGapDegreesDescending[index];
            if (std::fabs(difference) > kComparisonEpsilon)
            {
                return difference < 0.0;
            }
        }
        return candidate.angularGapDegreesDescending.size() <
            current.angularGapDegreesDescending.size();
    };
    for (int slot = 0; slot < selection_count; ++slot)
    {
        int best_candidate = -1;
        OrbitalCoverageStatistics best_coverage;
        for (int candidate_index = 0;
             candidate_index < static_cast<int>(candidateViews.size());
             ++candidate_index)
        {
            if (candidate_selected[static_cast<std::size_t>(candidate_index)] != 0)
            {
                continue;
            }
            const std::size_t combined_index =
                fixedViews.size() + static_cast<std::size_t>(candidate_index);
            weights[combined_index] = 1.0f;
            const OrbitalCoverageStatistics coverage =
                evaluateOrbitalCoverage(combined_views, weights);
            weights[combined_index] = 0.0f;
            if (!coverage.valid)
            {
                continue;
            }

            if (best_candidate < 0 ||
                has_better_gap_distribution(coverage, best_coverage))
            {
                best_candidate = candidate_index;
                best_coverage = coverage;
            }
        }
        if (best_candidate < 0)
        {
            selected.clear();
            return selected;
        }

        candidate_selected[static_cast<std::size_t>(best_candidate)] = 1;
        weights[fixedViews.size() + static_cast<std::size_t>(best_candidate)] =
            1.0f;
        selected.push_back(
            candidateViews[static_cast<std::size_t>(best_candidate)].frameIndex);
    }
    return selected;
}

} // namespace xjw::mesh
