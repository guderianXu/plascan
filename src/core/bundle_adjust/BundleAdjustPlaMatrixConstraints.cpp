#include "BundleAdjustPlaMatrixConstraints.h"

#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/optimization/robust_loss.h>

#include <algorithm>
#include <cmath>

namespace xjw::detail::plamatrix_ba
{
namespace
{

bool finishRobust(ConstraintLinearization* output, double huber_delta)
{
    double squared_norm = 0.0;
    for (int index = 0; index < output->residualSize; ++index)
    {
        squared_norm += output->residual[static_cast<std::size_t>(index)] *
                        output->residual[static_cast<std::size_t>(index)];
    }
    const auto robust = plamatrix::evaluateHuberLoss(squared_norm, huber_delta);
    output->normalWeight = robust.weight;
    output->robustCost = robust.cost;
    return std::isfinite(output->normalWeight) && std::isfinite(output->robustCost);
}

std::array<double, 3> rotationLog(const std::array<double, 9>& rotation)
{
    const double cosine = std::clamp(
        (rotation[0] + rotation[4] + rotation[8] - 1.0) * 0.5, -1.0, 1.0);
    const double angle = std::acos(cosine);
    const std::array<double, 3> vee{{
        rotation[7] - rotation[5],
        rotation[2] - rotation[6],
        rotation[3] - rotation[1]}};
    if (angle < 1e-10)
    {
        return {{0.5 * vee[0], 0.5 * vee[1], 0.5 * vee[2]}};
    }
    const double scale = angle / (2.0 * std::sin(angle));
    return {{scale * vee[0], scale * vee[1], scale * vee[2]}};
}

std::array<double, 3> poseRotationResidual(const FramePinholeCamera& camera,
                                           const BACameraPosePrior& prior)
{
    const auto rotation = camera.cameraToWorldRotation();
    std::array<double, 9> relative{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                relative[static_cast<std::size_t>(row * 3 + column)] +=
                    rotation[static_cast<std::size_t>(row * 3 + inner)] *
                    prior.cameraToWorldRotation[static_cast<std::size_t>(column * 3 + inner)];
            }
        }
    }
    return rotationLog(relative);
}

} // namespace

bool linearizeLaserPlane(const BALaserPlaneConstraint& constraint,
                         const std::array<double, 3>& point,
                         const BAOptions& options,
                         ConstraintLinearization* output)
{
    const double weight = options.laserPlaneWeight * constraint.weight;
    if (!output || !(weight > 0.0))
    {
        return false;
    }
    *output = {};
    output->residualSize = 1;
    const double scale = std::sqrt(weight);
    for (int axis = 0; axis < 3; ++axis)
    {
        output->residual[0] += scale * constraint.normal[static_cast<std::size_t>(axis)] *
            (point[static_cast<std::size_t>(axis)] -
             constraint.point[static_cast<std::size_t>(axis)]);
        output->pointJacobian[static_cast<std::size_t>(axis)] =
            scale * constraint.normal[static_cast<std::size_t>(axis)];
    }
    return finishRobust(output, options.laserHuberDeltaMeters * scale);
}

bool linearizeControlPoint(const BAControlPointConstraint& constraint,
                           const std::array<double, 3>& point,
                           const BAOptions& options,
                           ConstraintLinearization* output)
{
    const double weight = options.controlPointWeight * constraint.weight;
    if (!output || !(weight > 0.0))
    {
        return false;
    }
    *output = {};
    output->residualSize = 3;
    const double scale = std::sqrt(weight) / std::max(1e-9, constraint.sigmaMeters);
    for (int axis = 0; axis < 3; ++axis)
    {
        output->residual[static_cast<std::size_t>(axis)] = scale *
            (point[static_cast<std::size_t>(axis)] -
             constraint.point[static_cast<std::size_t>(axis)]);
        output->pointJacobian[static_cast<std::size_t>(axis * 3 + axis)] = scale;
    }
    return finishRobust(output, options.controlPointHuberDeltaMeters);
}

bool linearizeScaleBar(const BAScaleBarConstraint& constraint,
                       const std::array<double, 3>& point_a,
                       const std::array<double, 3>& point_b,
                       const BAOptions& options,
                       ConstraintLinearization* output)
{
    const double weight = options.scaleBarWeight * constraint.weight;
    if (!output || !(weight > 0.0))
    {
        return false;
    }
    std::array<double, 3> delta{};
    double squared_distance = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        delta[static_cast<std::size_t>(axis)] =
            point_a[static_cast<std::size_t>(axis)] - point_b[static_cast<std::size_t>(axis)];
        squared_distance += delta[static_cast<std::size_t>(axis)] *
                            delta[static_cast<std::size_t>(axis)];
    }
    const double distance = std::sqrt(squared_distance);
    if (!(distance > 1e-12))
    {
        return false;
    }
    *output = {};
    output->residualSize = 1;
    const double scale = std::sqrt(weight) / std::max(1e-9, constraint.sigmaMeters);
    output->residual[0] = scale * (distance - constraint.measuredDistanceMeters);
    for (int axis = 0; axis < 3; ++axis)
    {
        const double derivative = scale * delta[static_cast<std::size_t>(axis)] / distance;
        output->primaryJacobian[static_cast<std::size_t>(axis)] = derivative;
        output->secondaryPrimaryJacobian[static_cast<std::size_t>(axis)] = -derivative;
    }
    return finishRobust(output, options.scaleBarHuberDeltaMeters);
}

bool linearizePosePrior(const FramePinholeCamera& camera,
                        const BACameraPosePrior& prior,
                        const BAOptions& options,
                        ConstraintLinearization* output)
{
    if (!output || !prior.enabled)
    {
        return false;
    }
    *output = {};
    output->residualSize = 6;
    const double scale = std::sqrt(options.cameraPosePriorWeight);
    const double rotation_scale = scale /
        std::max(1e-9, prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0);
    const double position_scale = scale / std::max(1e-9, prior.positionSigmaMeters);
    const auto rotation_residual = poseRotationResidual(camera, prior);
    const auto center = camera.cameraCenter();
    for (int axis = 0; axis < 3; ++axis)
    {
        output->residual[static_cast<std::size_t>(axis)] =
            rotation_scale * rotation_residual[static_cast<std::size_t>(axis)];
        output->residual[static_cast<std::size_t>(axis + 3)] = position_scale *
            (center[static_cast<std::size_t>(axis)] -
             prior.cameraCenter[static_cast<std::size_t>(axis)]);
        output->primaryJacobian[static_cast<std::size_t>((axis + 3) * 9 + axis + 3)] =
            position_scale;
    }
    constexpr double epsilon = 1e-7;
    for (int parameter = 0; parameter < 3; ++parameter)
    {
        double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        delta[parameter] = epsilon;
        auto plus = camera;
        plus.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        auto minus = camera;
        minus.applyDeltaPose(delta);
        const auto plus_residual = poseRotationResidual(plus, prior);
        const auto minus_residual = poseRotationResidual(minus, prior);
        for (int row = 0; row < 3; ++row)
        {
            output->primaryJacobian[static_cast<std::size_t>(row * 9 + parameter)] =
                rotation_scale *
                (plus_residual[static_cast<std::size_t>(row)] -
                 minus_residual[static_cast<std::size_t>(row)]) /
                (2.0 * epsilon);
        }
    }
    return finishRobust(output, options.cameraPosePriorHuberDelta);
}

bool linearizeCameraPlane(const FramePinholeCamera& camera,
                          std::size_t camera_index,
                          const BAOptions& options,
                          ConstraintLinearization* output)
{
    const auto& constraint = options.cameraPlaneConstraint;
    if (!output || !constraint.enabled)
    {
        return false;
    }
    *output = {};
    output->residualSize = 1;
    const double scale = std::sqrt(constraint.weight) /
        std::max(1e-9, constraint.sigmaMeters);
    const double reference = constraint.referenceSignedDistances.empty()
        ? 0.0
        : constraint.referenceSignedDistances[camera_index];
    const auto center = camera.cameraCenter();
    for (int axis = 0; axis < 3; ++axis)
    {
        output->residual[0] += scale * constraint.normal[static_cast<std::size_t>(axis)] *
            (center[static_cast<std::size_t>(axis)] -
             constraint.point[static_cast<std::size_t>(axis)]);
        output->primaryJacobian[static_cast<std::size_t>(axis + 3)] =
            scale * constraint.normal[static_cast<std::size_t>(axis)];
    }
    output->residual[0] -= scale * reference;
    return finishRobust(output, options.cameraPlaneHuberDelta);
}

bool linearizeLaserRange(const FramePinholeCamera& camera,
                         const BALaserRangeConstraint& constraint,
                         const std::array<double, 3>& point,
                         const BAOptions& options,
                         ConstraintLinearization* output)
{
    const double weight = options.laserRangeWeight * constraint.weight;
    if (!output || !(weight > 0.0))
    {
        return false;
    }
    const auto rotation = camera.cameraToWorldRotation();
    const auto center = camera.cameraCenter();
    std::array<double, 3> lever_world{};
    std::array<double, 3> unit{};
    double squared_range = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            lever_world[static_cast<std::size_t>(row)] +=
                rotation[static_cast<std::size_t>(row * 3 + column)] *
                constraint.leverArmCameraMeters[static_cast<std::size_t>(column)];
        }
        unit[static_cast<std::size_t>(row)] = point[static_cast<std::size_t>(row)] -
            center[static_cast<std::size_t>(row)] - lever_world[static_cast<std::size_t>(row)];
        squared_range += unit[static_cast<std::size_t>(row)] *
                         unit[static_cast<std::size_t>(row)];
    }
    const double range = std::sqrt(squared_range);
    if (!(range > 1e-12))
    {
        return false;
    }
    for (double& value : unit)
    {
        value /= range;
    }
    *output = {};
    output->residualSize = 1;
    const double sqrt_weight = std::sqrt(weight);
    const double scale = sqrt_weight / constraint.sigmaRangeMeters;
    output->residual[0] = scale * (range - constraint.observedRangeMeters);
    for (int axis = 0; axis < 3; ++axis)
    {
        output->pointJacobian[static_cast<std::size_t>(axis)] =
            scale * unit[static_cast<std::size_t>(axis)];
        output->primaryJacobian[static_cast<std::size_t>(axis + 3)] =
            -scale * unit[static_cast<std::size_t>(axis)];
    }
    output->primaryJacobian[0] = scale *
        (unit[1] * lever_world[2] - unit[2] * lever_world[1]);
    output->primaryJacobian[1] = scale *
        (unit[2] * lever_world[0] - unit[0] * lever_world[2]);
    output->primaryJacobian[2] = scale *
        (unit[0] * lever_world[1] - unit[1] * lever_world[0]);
    return finishRobust(output, options.laserRangeHuberDelta * sqrt_weight);
}

bool linearizeLaserPointPrior(const BALaserRangeConstraint& constraint,
                              const std::array<double, 3>& point,
                              ConstraintLinearization* output)
{
    if (!output || constraint.pointMode != BALaserPointMode::Constrained)
    {
        return false;
    }
    *output = {};
    output->residualSize = 3;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            const double value = constraint.pointPriorSqrtInformation[
                static_cast<std::size_t>(row * 3 + column)];
            output->residual[static_cast<std::size_t>(row)] += value *
                (point[static_cast<std::size_t>(column)] -
                 constraint.pointPrior[static_cast<std::size_t>(column)]);
            output->pointJacobian[static_cast<std::size_t>(row * 3 + column)] = value;
        }
    }
    double squared_norm = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        squared_norm += output->residual[static_cast<std::size_t>(row)] *
                        output->residual[static_cast<std::size_t>(row)];
    }
    output->robustCost = 0.5 * squared_norm;
    return std::isfinite(output->robustCost);
}

} // namespace xjw::detail::plamatrix_ba
