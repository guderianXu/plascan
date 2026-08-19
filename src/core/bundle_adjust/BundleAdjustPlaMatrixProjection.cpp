#include "BundleAdjustPlaMatrixProjection.h"

#include "BundleAdjustValidation.h"

#include <plamatrix/optimization/robust_loss.h>

#include <algorithm>
#include <cmath>

namespace xjw::detail::plamatrix_ba
{
namespace
{

bool pixelByCameraPointJacobian(const FramePinholeCamera& camera,
                                const double camera_point[3],
                                double jacobian[6])
{
    if (!camera_point || !jacobian || !std::isfinite(camera_point[2]) ||
        std::abs(camera_point[2]) <= 1e-12)
    {
        return false;
    }
    const double x = camera_point[0] / camera_point[2];
    const double y = camera_point[1] / camera_point[2];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const auto distortion = camera.distortion();
    const double radial = 1.0 + distortion.radialK1 * r2 +
                          distortion.radialK2 * r4 +
                          distortion.radialK3 * r4 * r2;
    const double radial_slope = distortion.radialK1 +
                                2.0 * distortion.radialK2 * r2 +
                                3.0 * distortion.radialK3 * r4;
    const double radial_x = 2.0 * x * radial_slope;
    const double radial_y = 2.0 * y * radial_slope;
    const double distorted_x_by_x =
        radial + x * radial_x + 2.0 * distortion.tangentialP1 * y +
        6.0 * distortion.tangentialP2 * x;
    const double distorted_x_by_y =
        x * radial_y + 2.0 * distortion.tangentialP1 * x +
        2.0 * distortion.tangentialP2 * y;
    const double distorted_y_by_x =
        y * radial_x + 2.0 * distortion.tangentialP1 * x +
        2.0 * distortion.tangentialP2 * y;
    const double distorted_y_by_y =
        radial + y * radial_y + 6.0 * distortion.tangentialP1 * y +
        2.0 * distortion.tangentialP2 * x;

    const double u_scale = static_cast<double>(camera.uAxisSign()) * camera.focalX();
    const double v_scale = static_cast<double>(camera.vAxisSign()) * camera.focalY();
    const double inverse_depth = 1.0 / camera_point[2];
    jacobian[0] = u_scale * distorted_x_by_x * inverse_depth;
    jacobian[1] = u_scale * distorted_x_by_y * inverse_depth;
    jacobian[2] = -u_scale *
                  (distorted_x_by_x * x + distorted_x_by_y * y) * inverse_depth;
    jacobian[3] = v_scale * distorted_y_by_x * inverse_depth;
    jacobian[4] = v_scale * distorted_y_by_y * inverse_depth;
    jacobian[5] = -v_scale *
                  (distorted_y_by_x * x + distorted_y_by_y * y) * inverse_depth;
    return std::all_of(jacobian, jacobian + 6, [](double value)
    {
        return std::isfinite(value);
    });
}

FramePinholeCamera cameraWithSharedIntrinsics(
    const FramePinholeCamera& camera,
    const FramePinholeCamera& reference_camera,
    const std::array<double, 9>& parameters,
    const BAIntrinsicParameterMask& active)
{
    const auto enabled = [&](BAIntrinsicParameter parameter)
    {
        return active[static_cast<std::size_t>(parameter)];
    };
    FramePinholeCamera effective = camera;
    const auto source_intrinsics = camera.intrinsics();
    const auto reference_intrinsics = reference_camera.intrinsics();
    const double focal_x = enabled(BAIntrinsicParameter::FocalLength)
        ? std::exp(parameters[0])
        : source_intrinsics.focalX;
    const double source_aspect = source_intrinsics.focalX > 1e-12
        ? source_intrinsics.focalY / source_intrinsics.focalX
        : 1.0;
    const double aspect = enabled(BAIntrinsicParameter::FocalAspectRatio)
        ? std::exp(parameters[1])
        : source_aspect;
    const double principal_x = enabled(BAIntrinsicParameter::PrincipalPointX)
        ? reference_intrinsics.principalX + parameters[2]
        : source_intrinsics.principalX;
    const double principal_y = enabled(BAIntrinsicParameter::PrincipalPointY)
        ? reference_intrinsics.principalY + parameters[3]
        : source_intrinsics.principalY;
    effective.setIntrinsics(focal_x, focal_x * aspect, principal_x, principal_y);

    auto distortion = camera.distortion();
    if (enabled(BAIntrinsicParameter::RadialK1))
    {
        distortion.radialK1 = parameters[4];
    }
    if (enabled(BAIntrinsicParameter::RadialK2))
    {
        distortion.radialK2 = parameters[5];
    }
    if (enabled(BAIntrinsicParameter::RadialK3))
    {
        distortion.radialK3 = parameters[6];
    }
    if (enabled(BAIntrinsicParameter::TangentialP1))
    {
        distortion.tangentialP1 = parameters[7];
    }
    if (enabled(BAIntrinsicParameter::TangentialP2))
    {
        distortion.tangentialP2 = parameters[8];
    }
    effective.setDistortion(distortion);
    return effective;
}

} // namespace

bool linearizeObservation(const FramePinholeCamera& camera,
                          const std::array<double, 3>& point,
                          const BAObservation& observation,
                          double huber_delta,
                          ObservationLinearization* linearization)
{
    if (!linearization || !observationDataIsUsable(observation) ||
        !std::isfinite(huber_delta) || huber_delta < 0.0)
    {
        return false;
    }
    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    double positive_depth = 0.0;
    if (!camera.projectWorldPointWithDepth(world, pixel, positive_depth))
    {
        return false;
    }
    double camera_point[3] = {0.0, 0.0, 0.0};
    camera.worldToCamera(world, camera_point);
    double pixel_by_camera[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (!pixelByCameraPointJacobian(camera, camera_point, pixel_by_camera))
    {
        return false;
    }

    *linearization = ObservationLinearization{};
    linearization->residual = {{pixel[0] - observation.u, pixel[1] - observation.v}};
    const auto rotation = camera.cameraToWorldRotation();
    for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
    {
        for (int world_axis = 0; world_axis < 3; ++world_axis)
        {
            double derivative = 0.0;
            for (int camera_axis = 0; camera_axis < 3; ++camera_axis)
            {
                derivative += pixel_by_camera[pixel_axis * 3 + camera_axis] *
                              rotation[world_axis * 3 + camera_axis];
            }
            linearization->pointJacobian[pixel_axis * 3 + world_axis] = derivative;
            linearization->cameraJacobian[pixel_axis * 6 + 3 + world_axis] = -derivative;
        }

        const double* point_jacobian = linearization->pointJacobian.data() + pixel_axis * 3;
        const double dx = point[0] - camera.cameraCenter()[0];
        const double dy = point[1] - camera.cameraCenter()[1];
        const double dz = point[2] - camera.cameraCenter()[2];
        linearization->cameraJacobian[pixel_axis * 6 + 0] =
            point_jacobian[1] * dz - point_jacobian[2] * dy;
        linearization->cameraJacobian[pixel_axis * 6 + 1] =
            -point_jacobian[0] * dz + point_jacobian[2] * dx;
        linearization->cameraJacobian[pixel_axis * 6 + 2] =
            point_jacobian[0] * dy - point_jacobian[1] * dx;
    }

    const double observation_weight = sanitizedObservationWeight(observation);
    const double squared_norm = observation_weight *
        (linearization->residual[0] * linearization->residual[0] +
         linearization->residual[1] * linearization->residual[1]);
    const auto robust = plamatrix::evaluateHuberLoss(squared_norm, huber_delta);
    linearization->normalWeight = observation_weight * robust.weight;
    linearization->robustCost = robust.cost;
    return std::isfinite(linearization->normalWeight) &&
           std::isfinite(linearization->robustCost);
}

bool linearizeObservationWithSharedIntrinsics(
    const FramePinholeCamera& camera,
    const FramePinholeCamera& reference_camera,
    const std::array<double, 9>& shared_intrinsics,
    const BAIntrinsicParameterMask& active_parameters,
    const std::array<double, 3>& point,
    const BAObservation& observation,
    double huber_delta,
    ObservationLinearization* linearization)
{
    if (!linearization || !std::all_of(
            shared_intrinsics.begin(), shared_intrinsics.end(), [](double value)
            {
                return std::isfinite(value);
            }))
    {
        return false;
    }
    const FramePinholeCamera effective = cameraWithSharedIntrinsics(
        camera, reference_camera, shared_intrinsics, active_parameters);
    if (!linearizeObservation(
            effective, point, observation, huber_delta, linearization))
    {
        return false;
    }

    const double world[3] = {point[0], point[1], point[2]};
    double camera_point[3] = {0.0, 0.0, 0.0};
    effective.worldToCamera(world, camera_point);
    if (!std::isfinite(camera_point[2]) || std::abs(camera_point[2]) <= 1e-12)
    {
        return false;
    }
    const double x = camera_point[0] / camera_point[2];
    const double y = camera_point[1] / camera_point[2];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const auto distortion = effective.distortion();
    const double radial = 1.0 + distortion.radialK1 * r2 +
                          distortion.radialK2 * r4 + distortion.radialK3 * r6;
    const double distorted_x = x * radial +
        2.0 * distortion.tangentialP1 * x * y +
        distortion.tangentialP2 * (r2 + 2.0 * x * x);
    const double distorted_y = y * radial +
        distortion.tangentialP1 * (r2 + 2.0 * y * y) +
        2.0 * distortion.tangentialP2 * x * y;
    const double u_scale = static_cast<double>(effective.uAxisSign()) * effective.focalX();
    const double v_scale = static_cast<double>(effective.vAxisSign()) * effective.focalY();
    auto& jacobian = linearization->intrinsicJacobian;
    const auto set = [&](BAIntrinsicParameter parameter, double du, double dv)
    {
        const std::size_t index = static_cast<std::size_t>(parameter);
        if (active_parameters[index])
        {
            jacobian[index] = du;
            jacobian[9 + index] = dv;
        }
    };
    set(BAIntrinsicParameter::FocalLength,
        u_scale * distorted_x,
        v_scale * distorted_y);
    set(BAIntrinsicParameter::FocalAspectRatio, 0.0, v_scale * distorted_y);
    set(BAIntrinsicParameter::PrincipalPointX, 1.0, 0.0);
    set(BAIntrinsicParameter::PrincipalPointY, 0.0, 1.0);
    set(BAIntrinsicParameter::RadialK1, u_scale * x * r2, v_scale * y * r2);
    set(BAIntrinsicParameter::RadialK2, u_scale * x * r4, v_scale * y * r4);
    set(BAIntrinsicParameter::RadialK3, u_scale * x * r6, v_scale * y * r6);
    set(BAIntrinsicParameter::TangentialP1,
        u_scale * 2.0 * x * y,
        v_scale * (r2 + 2.0 * y * y));
    set(BAIntrinsicParameter::TangentialP2,
        u_scale * (r2 + 2.0 * x * x),
        v_scale * 2.0 * x * y);
    return std::all_of(jacobian.begin(), jacobian.end(), [](double value)
    {
        return std::isfinite(value);
    });
}

} // namespace xjw::detail::plamatrix_ba
