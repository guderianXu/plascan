#include "placamera/reference/MetashapeCameraReferenceAdapter.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace placamera::reference
{
    namespace
    {

        constexpr double kRadiansPerDegree = 0.01745329251994329576923690768489;

        RotationMatrix multiply(const RotationMatrix& first, const RotationMatrix& second) noexcept
        {
            RotationMatrix result{};
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    for (std::size_t inner = 0; inner < 3; ++inner)
                    {
                        result[row * 3 + column] += first[row * 3 + inner] * second[inner * 3 + column];
                    }
                }
            }
            return result;
        }

        RotationMatrix rotationX(double angle) noexcept
        {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            return {1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine};
        }

        RotationMatrix rotationY(double angle) noexcept
        {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            return {cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine};
        }

        RotationMatrix rotationZ(double angle) noexcept
        {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            return {cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0};
        }

        bool finite(const Vector3& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

        bool finite(const std::array<double, 9>& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

        std::array<double, 9> transformRotationCovariance(const Vector3& yprRadians,
                                                          const std::array<double, 9>& degreesSquared) noexcept
        {
            const RotationMatrix yaw_rotation = rotationZ(-yprRadians[0]);
            const RotationMatrix yaw_pitch_rotation = multiply(yaw_rotation, rotationX(yprRadians[1]));
            // Columns map [yaw,pitch,roll] increments in radians to a
            // left-multiplicative camera-to-reference rotation tangent.
            const std::array<double, 9> jacobian{{0.0,
                                                  yaw_rotation[0],
                                                  yaw_pitch_rotation[1],
                                                  0.0,
                                                  yaw_rotation[3],
                                                  yaw_pitch_rotation[4],
                                                  -1.0,
                                                  yaw_rotation[6],
                                                  yaw_pitch_rotation[7]}};
            std::array<double, 9> radians_squared{};
            const double unit_scale = kRadiansPerDegree * kRadiansPerDegree;
            for (std::size_t index = 0; index < degreesSquared.size(); ++index)
            {
                radians_squared[index] = degreesSquared[index] * unit_scale;
            }
            std::array<double, 9> intermediate{};
            std::array<double, 9> result{};
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    for (std::size_t inner = 0; inner < 3; ++inner)
                    {
                        intermediate[row * 3 + column] +=
                            jacobian[row * 3 + inner] * radians_squared[inner * 3 + column];
                    }
                }
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    for (std::size_t inner = 0; inner < 3; ++inner)
                    {
                        result[row * 3 + column] += intermediate[row * 3 + inner] * jacobian[column * 3 + inner];
                    }
                }
            }
            return result;
        }

    } // namespace

    Result<CameraReferenceObservation> adaptMetashapeCameraReference(ImageId image,
                                                                     ReferenceSourceId source,
                                                                     placoordinate::CoordinateFrameId frame,
                                                                     const MetashapeCameraReference& reference,
                                                                     CovarianceDefiniteness definiteness)
    {
        try
        {
            if (!reference.positionMeters && !reference.yawPitchRollDegrees)
            {
                return Result<CameraReferenceObservation>::failure(
                    CameraErrorCode::InvalidArgument,
                    "Metashape camera reference must contain a position, an orientation, or both");
            }
            if ((reference.positionCovarianceMetersSquared && !reference.positionMeters) ||
                (reference.yawPitchRollCovarianceDegreesSquared && !reference.yawPitchRollDegrees))
            {
                return Result<CameraReferenceObservation>::failure(
                    CameraErrorCode::InvalidArgument,
                    "Metashape covariance cannot be supplied without its corresponding reference value");
            }
            if ((reference.positionMeters && !finite(*reference.positionMeters)) ||
                (reference.yawPitchRollDegrees && !finite(*reference.yawPitchRollDegrees)) ||
                (reference.positionCovarianceMetersSquared && !finite(*reference.positionCovarianceMetersSquared)) ||
                (reference.yawPitchRollCovarianceDegreesSquared &&
                 !finite(*reference.yawPitchRollCovarianceDegreesSquared)))
            {
                return Result<CameraReferenceObservation>::failure(
                    CameraErrorCode::InvalidArgument, "Metashape camera reference must contain finite values");
            }

            CameraReferenceObservation observation{std::move(image), std::move(source), std::move(frame)};
            observation.enabled = reference.enabled;
            observation.position = reference.positionMeters;
            Vector3 ypr_radians{};
            if (reference.yawPitchRollDegrees)
            {
                for (std::size_t index = 0; index < ypr_radians.size(); ++index)
                {
                    ypr_radians[index] = (*reference.yawPitchRollDegrees)[index] * kRadiansPerDegree;
                }
                observation.orientation = multiply(multiply(rotationZ(-ypr_radians[0]), rotationX(ypr_radians[1])),
                                                   rotationY(ypr_radians[2]));
            }

            if (reference.positionCovarianceMetersSquared || reference.yawPitchRollCovarianceDegreesSquared)
            {
                std::array<double, 36> covariance{};
                if (reference.positionCovarianceMetersSquared)
                {
                    for (std::size_t row = 0; row < 3; ++row)
                    {
                        for (std::size_t column = 0; column < 3; ++column)
                        {
                            covariance[row * 6 + column] =
                                (*reference.positionCovarianceMetersSquared)[row * 3 + column];
                        }
                    }
                }
                if (reference.yawPitchRollCovarianceDegreesSquared)
                {
                    const std::array<double, 9> rotation_covariance =
                        transformRotationCovariance(ypr_radians, *reference.yawPitchRollCovarianceDegreesSquared);
                    for (std::size_t row = 0; row < 3; ++row)
                    {
                        for (std::size_t column = 0; column < 3; ++column)
                        {
                            covariance[(row + 3) * 6 + column + 3] = rotation_covariance[row * 3 + column];
                        }
                    }
                }
                const PoseCovarianceComponents components =
                    reference.positionCovarianceMetersSquared && reference.yawPitchRollCovarianceDegreesSquared
                        ? PoseCovarianceComponents::PositionAndRotation
                    : reference.positionCovarianceMetersSquared ? PoseCovarianceComponents::PositionOnly
                                                                : PoseCovarianceComponents::RotationOnly;
                observation.covariance = PoseCovariance::matrix(covariance, components, definiteness);
            }
            return Result<CameraReferenceObservation>::success(std::move(observation));
        }
        catch (const CameraValidationError& error)
        {
            return Result<CameraReferenceObservation>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<CameraReferenceObservation>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

} // namespace placamera::reference
