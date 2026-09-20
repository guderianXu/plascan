#include "FramePinholeOptimization.h"

#include "camera/core/types/CameraErrors.h"

#include <cmath>
#include <utility>

namespace xjw::camera_models::frame_pinhole
{
    namespace
    {

        std::array<double, 9> multiply(const std::array<double, 9>& first, const std::array<double, 9>& second)
        {
            std::array<double, 9> result{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int index = 0; index < 3; ++index)
                    {
                        result[static_cast<std::size_t>(row * 3 + column)] +=
                            first[static_cast<std::size_t>(row * 3 + index)] *
                            second[static_cast<std::size_t>(index * 3 + column)];
                    }
                }
            }
            return result;
        }

        std::array<double, 9> rotationFromDelta(const std::array<double, 6>& delta)
        {
            const double wx = delta[0];
            const double wy = delta[1];
            const double wz = delta[2];
            const double thetaSquared = wx * wx + wy * wy + wz * wz;
            std::array<double, 9> result{};
            if (thetaSquared < 1.0e-20)
            {
                result = {1.0, -wz, wy, wz, 1.0, -wx, -wy, wx, 1.0};
                return result;
            }

            const double theta = std::sqrt(thetaSquared);
            const double sineOverTheta = std::sin(theta) / theta;
            const double oneMinusCosineOverThetaSquared = (1.0 - std::cos(theta)) / thetaSquared;
            result = {1.0 - oneMinusCosineOverThetaSquared * (wy * wy + wz * wz),
                      oneMinusCosineOverThetaSquared * wx * wy - sineOverTheta * wz,
                      oneMinusCosineOverThetaSquared * wx * wz + sineOverTheta * wy,
                      oneMinusCosineOverThetaSquared * wx * wy + sineOverTheta * wz,
                      1.0 - oneMinusCosineOverThetaSquared * (wx * wx + wz * wz),
                      oneMinusCosineOverThetaSquared * wy * wz - sineOverTheta * wx,
                      oneMinusCosineOverThetaSquared * wx * wz - sineOverTheta * wy,
                      oneMinusCosineOverThetaSquared * wy * wz + sineOverTheta * wx,
                      1.0 - oneMinusCosineOverThetaSquared * (wx * wx + wy * wy)};
            return result;
        }

    } // namespace

    ParameterBlockSchema FramePinholeOptimization::schema(bool optimizeCalibration)
    {
        ParameterBlockSchema result{{"pose.rx", "pose.ry", "pose.rz", "pose.tx", "pose.ty", "pose.tz"}, {}};
        if (optimizeCalibration)
        {
            result.calibrationParameters = {"intrinsics.fx", "intrinsics.fy", "intrinsics.cx", "intrinsics.cy"};
        }
        return result;
    }

    FramePinholeInstance FramePinholeOptimization::applyPoseUpdate(const FramePinholeInstance& instance,
                                                                   camera_core::CameraInstanceId instanceId,
                                                                   const std::array<double, 6>& delta)
    {
        for (double value : delta)
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidRotation,
                                                         "pinhole pose update must contain finite values");
            }
        }
        const auto deltaRotation = rotationFromDelta(delta);
        const auto updatedRotation = multiply(deltaRotation, instance.pose().cameraToWorldRotation);
        const std::array<double, 3> updatedCenter{instance.pose().center[0] + delta[3],
                                                  instance.pose().center[1] + delta[4],
                                                  instance.pose().center[2] + delta[5]};
        return instance.withPose(std::move(instanceId),
                                 camera_core::Pose::create(instance.pose().frame, updatedCenter, updatedRotation));
    }

    std::shared_ptr<const FramePinholeDefinition>
    FramePinholeOptimization::applyCalibrationUpdate(const FramePinholeDefinition& definition,
                                                     camera_core::CameraDefinitionId definitionId,
                                                     const std::array<double, 4>& delta)
    {
        for (double value : delta)
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidIntrinsics,
                                                         "pinhole calibration update must contain finite values");
            }
        }
        Intrinsics intrinsics = definition.intrinsics();
        intrinsics.focalX += delta[0];
        intrinsics.focalY += delta[1];
        intrinsics.principalX += delta[2];
        intrinsics.principalY += delta[3];
        return FramePinholeDefinition::create(std::move(definitionId),
                                              intrinsics,
                                              definition.distortion(),
                                              definition.pixelConvention(),
                                              definition.worldFrame(),
                                              definition.depthAxisFlipped());
    }

} // namespace xjw::camera_models::frame_pinhole
