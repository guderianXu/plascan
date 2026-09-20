#include "FramePinholeProjection.h"

#include <cmath>
#include <limits>

namespace xjw::camera_models::frame_pinhole
{

    bool FramePinholeProjection::project(const FramePinholeInstance& instance,
                                         const std::array<double, 3>& world,
                                         ProjectionResult* result)
    {
        if (!result)
        {
            return false;
        }
        const auto camera = worldToCamera(instance.pose(), world);
        const double forwardDepth = instance.pinholeDefinition().depthAxisFlipped() ? -camera[2] : camera[2];
        if (!std::isfinite(forwardDepth) || !(forwardDepth > 1.0e-9) || !std::isfinite(camera[2]) ||
            std::fabs(camera[2]) < 1.0e-9)
        {
            return false;
        }

        const Intrinsics& intrinsics = instance.pinholeDefinition().intrinsics();
        const Distortion& distortion = instance.pinholeDefinition().distortion();
        const double normalizedX = camera[0] / camera[2];
        const double normalizedY = camera[1] / camera[2];
        double distortedX = 0.0;
        double distortedY = 0.0;
        distort(distortion, normalizedX, normalizedY, &distortedX, &distortedY);
        result->pixel = {
            static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * distortedX + intrinsics.principalX,
            static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * distortedY + intrinsics.principalY};
        result->positiveDepth = forwardDepth;
        return std::isfinite(result->pixel[0]) && std::isfinite(result->pixel[1]);
    }

    bool FramePinholeProjection::projectSigned(const FramePinholeInstance& instance,
                                               const std::array<double, 3>& world,
                                               std::array<double, 2>* pixel)
    {
        if (!pixel)
        {
            return false;
        }
        const auto camera = worldToCamera(instance.pose(), world);
        if (!std::isfinite(camera[2]) || std::fabs(camera[2]) < 1.0e-9)
        {
            return false;
        }

        const Intrinsics& intrinsics = instance.pinholeDefinition().intrinsics();
        double distortedX = 0.0;
        double distortedY = 0.0;
        distort(instance.pinholeDefinition().distortion(),
                camera[0] / camera[2],
                camera[1] / camera[2],
                &distortedX,
                &distortedY);
        *pixel = {static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * distortedX + intrinsics.principalX,
                  static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * distortedY + intrinsics.principalY};
        return std::isfinite((*pixel)[0]) && std::isfinite((*pixel)[1]);
    }

    bool FramePinholeProjection::unproject(const FramePinholeInstance& instance,
                                           const std::array<double, 2>& pixel,
                                           double positiveDepth,
                                           std::array<double, 3>* world)
    {
        if (!world || !std::isfinite(positiveDepth) || !(positiveDepth > 0.0) || !std::isfinite(pixel[0]) ||
            !std::isfinite(pixel[1]))
        {
            return false;
        }

        std::array<double, 2> normalized{};
        if (!undistort(instance.pinholeDefinition(), pixel, &normalized))
        {
            return false;
        }
        const double cameraZ = instance.pinholeDefinition().depthAxisFlipped() ? -positiveDepth : positiveDepth;
        const std::array<double, 3> camera{normalized[0] * cameraZ, normalized[1] * cameraZ, cameraZ};
        *world = cameraToWorld(instance.pose(), camera);
        return std::isfinite((*world)[0]) && std::isfinite((*world)[1]) && std::isfinite((*world)[2]);
    }

    bool FramePinholeProjection::undistort(const FramePinholeDefinition& definition,
                                           const std::array<double, 2>& pixel,
                                           std::array<double, 2>* normalized,
                                           int maxIterations,
                                           double tolerance)
    {
        if (!normalized || maxIterations < 0 || !std::isfinite(tolerance) || tolerance <= 0.0 ||
            !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]))
        {
            return false;
        }

        const Intrinsics& intrinsics = definition.intrinsics();
        const Distortion& distortion = definition.distortion();
        double x = static_cast<double>(intrinsics.uAxisSign) * (pixel[0] - intrinsics.principalX) / intrinsics.focalX;
        double y = static_cast<double>(intrinsics.vAxisSign) * (pixel[1] - intrinsics.principalY) / intrinsics.focalY;
        if (distortion.radialK1 == 0.0 && distortion.radialK2 == 0.0 && distortion.radialK3 == 0.0 &&
            distortion.tangentialP1 == 0.0 && distortion.tangentialP2 == 0.0)
        {
            *normalized = {x, y};
            return true;
        }

        for (int iteration = 0; iteration < maxIterations; ++iteration)
        {
            double distortedX = 0.0;
            double distortedY = 0.0;
            distort(distortion, x, y, &distortedX, &distortedY);
            const double currentU =
                static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * distortedX + intrinsics.principalX;
            const double currentV =
                static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * distortedY + intrinsics.principalY;
            const double residualU = currentU - pixel[0];
            const double residualV = currentV - pixel[1];
            if (std::fabs(residualU) < tolerance && std::fabs(residualV) < tolerance)
            {
                break;
            }

            constexpr double step = 1.0e-7;
            double shiftedX = 0.0;
            double shiftedY = 0.0;
            distort(distortion, x + step, y, &shiftedX, &shiftedY);
            const double j00 =
                static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * (shiftedX - distortedX) / step;
            const double j10 =
                static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * (shiftedY - distortedY) / step;
            distort(distortion, x, y + step, &shiftedX, &shiftedY);
            const double j01 =
                static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * (shiftedX - distortedX) / step;
            const double j11 =
                static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * (shiftedY - distortedY) / step;
            const double determinant = j00 * j11 - j01 * j10;
            if (!std::isfinite(determinant) || std::fabs(determinant) < 1.0e-15)
            {
                break;
            }
            x -= (j11 * residualU - j01 * residualV) / determinant;
            y -= (-j10 * residualU + j00 * residualV) / determinant;
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                return false;
            }
        }
        *normalized = {x, y};
        return true;
    }

    bool FramePinholeProjection::ray(const FramePinholeInstance& instance,
                                     const std::array<double, 2>& pixel,
                                     RayResult* result)
    {
        if (!result)
        {
            return false;
        }
        std::array<double, 3> pointAtUnitDepth{};
        if (!unproject(instance, pixel, 1.0, &pointAtUnitDepth))
        {
            return false;
        }
        result->origin = instance.pose().center;
        for (int axis = 0; axis < 3; ++axis)
        {
            result->direction[static_cast<std::size_t>(axis)] =
                pointAtUnitDepth[static_cast<std::size_t>(axis)] - result->origin[static_cast<std::size_t>(axis)];
        }
        double squaredNorm = 0.0;
        for (double value : result->direction)
        {
            squaredNorm += value * value;
        }
        if (!std::isfinite(squaredNorm) || !(squaredNorm > 0.0))
        {
            return false;
        }
        const double inverseNorm = 1.0 / std::sqrt(squaredNorm);
        for (double& value : result->direction)
        {
            value *= inverseNorm;
        }
        return true;
    }

    double FramePinholeProjection::positiveDepth(const FramePinholeInstance& instance,
                                                 const std::array<double, 3>& world)
    {
        const auto camera = worldToCamera(instance.pose(), world);
        const double depth = instance.pinholeDefinition().depthAxisFlipped() ? -camera[2] : camera[2];
        return std::isfinite(depth) ? depth : std::numeric_limits<double>::quiet_NaN();
    }

    void FramePinholeProjection::distort(
        const Distortion& distortion, double x, double y, double* distortedX, double* distortedY)
    {
        const double radiusSquared = x * x + y * y;
        const double radial = 1.0 + distortion.radialK1 * radiusSquared +
                              distortion.radialK2 * radiusSquared * radiusSquared +
                              distortion.radialK3 * radiusSquared * radiusSquared * radiusSquared;
        *distortedX = x * radial + 2.0 * distortion.tangentialP1 * x * y +
                      distortion.tangentialP2 * (radiusSquared + 2.0 * x * x);
        *distortedY = y * radial + distortion.tangentialP1 * (radiusSquared + 2.0 * y * y) +
                      2.0 * distortion.tangentialP2 * x * y;
    }

    std::array<double, 3> FramePinholeProjection::worldToCamera(const camera_core::Pose& pose,
                                                                const std::array<double, 3>& world)
    {
        const std::array<double, 3> offset{
            world[0] - pose.center[0], world[1] - pose.center[1], world[2] - pose.center[2]};
        const auto& rotation = pose.cameraToWorldRotation;
        return {rotation[0] * offset[0] + rotation[3] * offset[1] + rotation[6] * offset[2],
                rotation[1] * offset[0] + rotation[4] * offset[1] + rotation[7] * offset[2],
                rotation[2] * offset[0] + rotation[5] * offset[1] + rotation[8] * offset[2]};
    }

    std::array<double, 3> FramePinholeProjection::cameraToWorld(const camera_core::Pose& pose,
                                                                const std::array<double, 3>& camera)
    {
        const auto& rotation = pose.cameraToWorldRotation;
        return {pose.center[0] + rotation[0] * camera[0] + rotation[1] * camera[1] + rotation[2] * camera[2],
                pose.center[1] + rotation[3] * camera[0] + rotation[4] * camera[1] + rotation[5] * camera[2],
                pose.center[2] + rotation[6] * camera[0] + rotation[7] * camera[1] + rotation[8] * camera[2]};
    }

    std::array<double, 9> FramePinholeProjection::transpose(const std::array<double, 9>& rotation)
    {
        return {rotation[0],
                rotation[3],
                rotation[6],
                rotation[1],
                rotation[4],
                rotation[7],
                rotation[2],
                rotation[5],
                rotation[8]};
    }

} // namespace xjw::camera_models::frame_pinhole
