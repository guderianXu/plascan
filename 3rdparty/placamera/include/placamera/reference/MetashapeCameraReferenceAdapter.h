#pragma once

#include "placamera/reference/CameraReferenceObservation.h"

#include <array>
#include <optional>

namespace placamera::reference
{

    /** Metashape public Camera.Reference fields before frame normalization. */
    struct MetashapeCameraReference
    {
        std::optional<placamera::Vector3> positionMeters;
        std::optional<std::array<double, 9>> positionCovarianceMetersSquared;
        std::optional<placamera::Vector3> yawPitchRollDegrees;
        std::optional<std::array<double, 9>> yawPitchRollCovarianceDegreesSquared;
        bool enabled = true;
    };

    /**
     * Adapt Metashape Rz(-yaw)*Rx(pitch)*Ry(roll) into a camera-to-reference
     * rotation. Degree-squared YPR covariance is mapped to the canonical
     * left-tangent rotation covariance in radians squared.
     */
    placamera::Result<CameraReferenceObservation>
    adaptMetashapeCameraReference(placamera::ImageId image,
                                  ReferenceSourceId source,
                                  placoordinate::CoordinateFrameId frame,
                                  const MetashapeCameraReference& reference,
                                  CovarianceDefiniteness definiteness = CovarianceDefiniteness::PositiveSemidefinite);

} // namespace placamera::reference
