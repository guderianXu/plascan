#pragma once

#include "camera/core/types/CameraIds.h"
#include "camera/core/types/CameraPose.h"
#include "camera/core/types/CameraUncertainty.h"
#include "coordinate_system/types/CoordinateIds.h"
#include "coordinate_system/types/TimeReference.h"

#include <array>
#include <optional>
#include <string>

namespace xjw::camera_reference
{

    using xjw::camera_core::ImageId;
    using xjw::camera_core::Pose;
    using xjw::camera_core::PoseCovariance;
    using xjw::camera_core::ReferenceSourceId;
    using xjw::camera_core::Rotation;

    enum class LeverArmDirection
    {
        Unknown,
        SensorToCamera,
        CameraToSensor,
    };

    struct LeverArm
    {
        std::array<double, 3> vector{{0.0, 0.0, 0.0}};
        LeverArmDirection direction = LeverArmDirection::Unknown;
        std::optional<xjw::coordinate_system::CoordinateFrameId> vectorFrame;
    };

    struct CameraReferenceObservation
    {
        ImageId image;
        ReferenceSourceId source;
        xjw::coordinate_system::CoordinateFrameId frame;
        std::optional<xjw::coordinate_system::TimeReference> time;
        std::optional<std::array<double, 3>> position;
        std::optional<Rotation> orientation;
        std::optional<PoseCovariance> covariance;
        std::optional<LeverArm> leverArm;
        bool enabled = true;
    };

    enum class ReferenceResolutionStatus
    {
        Unresolved,
        Resolved,
    };

    struct ResolvedCameraReference
    {
        ReferenceResolutionStatus status = ReferenceResolutionStatus::Unresolved;
        std::optional<Pose> pose;
        std::optional<PoseCovariance> covariance;
        xjw::coordinate_system::CoordinateFrameId targetFrame = xjw::coordinate_system::CoordinateFrameId("unresolved");
        // Hash of the versioned frame/normalization policy and graph used to
        // produce `pose`. Unlike transformHash, this value intentionally
        // excludes observation values and can therefore be compared across
        // images in one numerical solve.
        std::string transformProvenanceHash;
        // Full observation-resolution fingerprint. It includes the source
        // values and is useful for stale-data/integrity diagnostics; it is
        // not a cross-image compatibility key.
        std::string transformHash;
        bool leverArmApplied = false;
        std::string reason;
    };

} // namespace xjw::camera_reference
