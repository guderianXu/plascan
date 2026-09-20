#pragma once

#include "FramePinholeInstance.h"

#include <array>

namespace xjw::camera_models::frame_pinhole
{

    struct ProjectionResult
    {
        std::array<double, 2> pixel{{0.0, 0.0}};
        double positiveDepth = 0.0;
    };

    struct RayResult
    {
        std::array<double, 3> origin{{0.0, 0.0, 0.0}};
        std::array<double, 3> direction{{0.0, 0.0, 1.0}};
    };

    class FramePinholeProjection
    {
    public:
        static bool
        project(const FramePinholeInstance& instance, const std::array<double, 3>& world, ProjectionResult* result);

        static bool projectSigned(const FramePinholeInstance& instance,
                                  const std::array<double, 3>& world,
                                  std::array<double, 2>* pixel);

        static bool unproject(const FramePinholeInstance& instance,
                              const std::array<double, 2>& pixel,
                              double positiveDepth,
                              std::array<double, 3>* world);

        static bool undistort(const FramePinholeDefinition& definition,
                              const std::array<double, 2>& pixel,
                              std::array<double, 2>* normalized,
                              int maxIterations = 20,
                              double tolerance = 1.0e-8);

        static bool ray(const FramePinholeInstance& instance, const std::array<double, 2>& pixel, RayResult* result);

        static double positiveDepth(const FramePinholeInstance& instance, const std::array<double, 3>& world);

    private:
        static void distort(const Distortion& distortion, double x, double y, double* distortedX, double* distortedY);
        static std::array<double, 3> worldToCamera(const camera_core::Pose& pose, const std::array<double, 3>& world);
        static std::array<double, 3> cameraToWorld(const camera_core::Pose& pose, const std::array<double, 3>& camera);
        static std::array<double, 9> transpose(const std::array<double, 9>& rotation);
    };

} // namespace xjw::camera_models::frame_pinhole
