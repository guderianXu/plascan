#pragma once

#include "camera/core/types/CameraPose.h"
#include "coordinate_system/types/TimeReference.h"

#include <array>
#include <optional>
#include <vector>

namespace xjw::camera_models::linescan
{

    struct TrajectorySample
    {
        xjw::coordinate_system::TimeReference time;
        std::array<double, 3> center{{0.0, 0.0, 0.0}};
        camera_core::Rotation cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    };

    struct TranslationalStateSample
    {
        xjw::coordinate_system::TimeReference time;
        std::array<double, 3> positionMeters{{0.0, 0.0, 0.0}};
        std::array<double, 3> velocityMetersPerSecond{{0.0, 0.0, 0.0}};
    };

    struct QuaternionTrajectorySample
    {
        xjw::coordinate_system::TimeReference time;
        std::array<double, 4> scalarFirst{{1.0, 0.0, 0.0, 0.0}};
    };

    struct FrameRotationTrajectory
    {
        camera_core::Rotation constantRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::vector<QuaternionTrajectorySample> samples;
    };

    struct FrameComposedTrajectory
    {
        std::vector<TranslationalStateSample> inertialStates;
        FrameRotationTrajectory inertialToWorld;
        FrameRotationTrajectory inertialToSensor;
    };

    class LineScanTrajectory
    {
    public:
        static LineScanTrajectory create(std::vector<TrajectorySample> samples);
        static LineScanTrajectory createFrameComposed(FrameComposedTrajectory trajectory);

        const std::vector<TrajectorySample>& samples() const noexcept;
        const FrameComposedTrajectory* frameComposed() const noexcept;
        xjw::coordinate_system::TimeScale timeScale() const noexcept;
        camera_core::Pose poseAt(xjw::coordinate_system::TimeReference time, const xjw::coordinate_system::CoordinateFrameId& frame) const;

    private:
        explicit LineScanTrajectory(std::vector<TrajectorySample> samples);
        explicit LineScanTrajectory(FrameComposedTrajectory trajectory);

        std::vector<TrajectorySample> _samples;
        std::optional<FrameComposedTrajectory> _frameComposed;
        xjw::coordinate_system::TimeScale _timeScale = xjw::coordinate_system::TimeScale::Tdb;
    };

} // namespace xjw::camera_models::linescan
