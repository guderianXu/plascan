#pragma once

#include "CameraReferenceSourceId.h"
#include "CameraReferenceUncertainty.h"
#include <placamera/types.h>
#include <placoordinate/types/CoordinateIds.h>
#include <placoordinate/types/TimeReference.h>

#include <array>
#include <optional>
#include <string>

namespace placamera::reference
{

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
        std::optional<placoordinate::CoordinateFrameId> vectorFrame;
    };

    struct CameraReferenceObservation
    {
        placamera::ImageId image;
        ReferenceSourceId source;
        placoordinate::CoordinateFrameId frame;
        std::optional<placoordinate::TimeReference> time;
        std::optional<std::array<double, 3>> position;
        std::optional<placamera::RotationMatrix> orientation;
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
        std::optional<std::array<double, 3>> position;
        std::optional<placamera::RotationMatrix> orientation;
        /** Present only when both position and orientation were resolved. */
        std::optional<placamera::Pose> pose;
        std::optional<PoseCovariance> covariance;
        placoordinate::CoordinateFrameId targetFrame = placoordinate::CoordinateFrameId("unresolved");
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

} // namespace placamera::reference
