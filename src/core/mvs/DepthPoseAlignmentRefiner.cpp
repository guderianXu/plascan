#include "DepthPoseAlignmentRefiner.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace xjw::mvs
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool isFinite(const cv::Vec3d &value)
{
    return std::isfinite(value[0]) &&
        std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

cv::Matx33d rotationFromVector(const cv::Vec3d &rotation_vector)
{
    const double angle = cv::norm(rotation_vector);
    if (angle <= 1.0e-14)
    {
        return cv::Matx33d::eye();
    }
    const cv::Vec3d axis = rotation_vector / angle;
    const cv::Matx33d skew(
        0.0, -axis[2], axis[1],
        axis[2], 0.0, -axis[0],
        -axis[1], axis[0], 0.0);
    const cv::Matx33d outer(
        axis[0] * axis[0], axis[0] * axis[1], axis[0] * axis[2],
        axis[1] * axis[0], axis[1] * axis[1], axis[1] * axis[2],
        axis[2] * axis[0], axis[2] * axis[1], axis[2] * axis[2]);
    return std::cos(angle) * cv::Matx33d::eye() +
        (1.0 - std::cos(angle)) * outer +
        std::sin(angle) * skew;
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = fraction *
        static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return values[lower] * (1.0 - blend) + values[upper] * blend;
}

std::vector<double> residuals(
    const std::vector<const DepthPoseAlignmentSample *> &samples,
    const cv::Vec3d &pivot,
    const cv::Matx33d &rotation,
    const cv::Vec3d &translation)
{
    std::vector<double> values;
    values.reserve(samples.size());
    for (const DepthPoseAlignmentSample *sample : samples)
    {
        const cv::Vec3d point =
            rotation * (sample->sourcePointWorld - pivot) +
            pivot + translation;
        const cv::Vec3d normal =
            sample->targetNormalWorld /
            cv::norm(sample->targetNormalWorld);
        values.push_back(std::abs(
            normal.dot(point - sample->targetPointWorld)));
    }
    return values;
}

DepthPoseAlignmentCorrection refineCamera(
    int camera_index,
    const std::vector<const DepthPoseAlignmentSample *> &samples,
    const DepthPoseAlignmentOptions &options)
{
    DepthPoseAlignmentCorrection correction;
    correction.cameraIndex = camera_index;
    correction.correspondenceCount = static_cast<int>(samples.size());
    if (camera_index == options.anchorCameraIndex)
    {
        correction.reason = "anchor_camera";
        return correction;
    }
    if (correction.correspondenceCount < options.minimumCorrespondences)
    {
        correction.reason = "insufficient_correspondences";
        return correction;
    }

    for (const DepthPoseAlignmentSample *sample : samples)
    {
        correction.pivotWorld += sample->sourcePointWorld;
    }
    correction.pivotWorld *= 1.0 / static_cast<double>(samples.size());
    const std::vector<double> residuals_before =
        residuals(samples,
                  correction.pivotWorld,
                  correction.rotation,
                  correction.translation);
    correction.residualMedianBefore = percentile(residuals_before, 0.5);
    correction.residualP90Before = percentile(residuals_before, 0.9);

    const double maximum_rotation =
        options.maximumRotationDegrees * kPi / 180.0;
    for (int iteration = 0;
         iteration < std::max(1, options.maximumIterations);
         ++iteration)
    {
        cv::Matx<double, 6, 6> normal_matrix =
            cv::Matx<double, 6, 6>::zeros();
        cv::Vec<double, 6> gradient = cv::Vec<double, 6>::all(0.0);
        for (const DepthPoseAlignmentSample *sample : samples)
        {
            const cv::Vec3d point =
                correction.rotation *
                (sample->sourcePointWorld - correction.pivotWorld) +
                correction.pivotWorld + correction.translation;
            const cv::Vec3d relative = point - correction.pivotWorld;
            const cv::Vec3d normal =
                sample->targetNormalWorld /
                cv::norm(sample->targetNormalWorld);
            const double residual =
                normal.dot(point - sample->targetPointWorld);
            const double robust_weight =
                std::abs(residual) <= options.huberDelta
                ? 1.0
                : options.huberDelta / std::abs(residual);
            const double weight =
                std::max(0.0, sample->confidence) * robust_weight;
            const cv::Vec3d rotation_jacobian = relative.cross(normal);
            const cv::Vec<double, 6> jacobian(
                normal[0], normal[1], normal[2],
                rotation_jacobian[0],
                rotation_jacobian[1],
                rotation_jacobian[2]);
            normal_matrix += weight * jacobian * jacobian.t();
            gradient += weight * jacobian * residual;
        }
        for (int diagonal = 0; diagonal < 6; ++diagonal)
        {
            normal_matrix(diagonal, diagonal) += options.damping;
        }
        cv::Vec<double, 6> increment;
        if (!cv::solve(normal_matrix,
                       -gradient,
                       increment,
                       cv::DECOMP_CHOLESKY))
        {
            correction.reason = "singular_normal_equations";
            return correction;
        }
        cv::Vec3d translation_increment(
            increment[0], increment[1], increment[2]);
        cv::Vec3d rotation_increment(
            increment[3], increment[4], increment[5]);
        const double translation_norm = cv::norm(translation_increment);
        if (translation_norm > options.maximumTranslation)
        {
            translation_increment *=
                options.maximumTranslation / translation_norm;
        }
        const double rotation_norm = cv::norm(rotation_increment);
        if (rotation_norm > maximum_rotation)
        {
            rotation_increment *= maximum_rotation / rotation_norm;
        }
        const cv::Matx33d incremental_rotation =
            rotationFromVector(rotation_increment);
        correction.rotation =
            incremental_rotation * correction.rotation;
        correction.translation =
            incremental_rotation * correction.translation +
            translation_increment;
        correction.iterationCount = iteration + 1;
        if (cv::norm(correction.translation) >
            options.maximumTranslation ||
            std::acos(std::clamp(
                (cv::trace(correction.rotation) - 1.0) * 0.5,
                -1.0,
                1.0)) > maximum_rotation)
        {
            correction.reason = "correction_exceeds_safety_limit";
            return correction;
        }
        if (translation_norm <= options.convergenceTranslation &&
            rotation_norm <= options.convergenceRotationRadians)
        {
            break;
        }
    }

    const std::vector<double> residuals_after =
        residuals(samples,
                  correction.pivotWorld,
                  correction.rotation,
                  correction.translation);
    correction.residualMedianAfter = percentile(residuals_after, 0.5);
    correction.residualP90After = percentile(residuals_after, 0.9);
    correction.accepted =
        correction.residualP90After <
        correction.residualP90Before *
        options.requiredP90ImprovementRatio;
    correction.reason = correction.accepted
        ? "accepted"
        : "insufficient_residual_improvement";
    return correction;
}

} // namespace

DepthPoseAlignmentResult DepthPoseAlignmentRefiner::refine(
    const std::vector<DepthPoseAlignmentSample> &samples,
    const DepthPoseAlignmentOptions &options)
{
    DepthPoseAlignmentResult result;
    result.enabled = options.enabled;
    if (!options.enabled)
    {
        return result;
    }

    std::map<int, std::vector<const DepthPoseAlignmentSample *>> grouped;
    for (const DepthPoseAlignmentSample &sample : samples)
    {
        if (sample.cameraIndex < 0 ||
            sample.occluded ||
            sample.confidence <= 0.0 ||
            !isFinite(sample.sourcePointWorld) ||
            !isFinite(sample.targetPointWorld) ||
            !isFinite(sample.targetNormalWorld) ||
            cv::norm(sample.targetNormalWorld) <= 1.0e-12)
        {
            continue;
        }
        grouped[sample.cameraIndex].push_back(&sample);
    }
    result.corrections.reserve(grouped.size());
    for (const auto &entry : grouped)
    {
        DepthPoseAlignmentCorrection correction =
            refineCamera(entry.first, entry.second, options);
        result.acceptedAny = result.acceptedAny || correction.accepted;
        result.corrections.push_back(std::move(correction));
    }
    return result;
}

cv::Vec3d DepthPoseAlignmentRefiner::applyCorrection(
    const DepthPoseAlignmentCorrection &correction,
    const cv::Vec3d &pointWorld)
{
    if (!correction.accepted)
    {
        return pointWorld;
    }
    return correction.rotation *
        (pointWorld - correction.pivotWorld) +
        correction.pivotWorld +
        correction.translation;
}

} // namespace xjw::mvs
