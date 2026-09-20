#pragma once

#include "camera/reference/model/CameraReferenceObservation.h"

#include <optional>
#include <string>

namespace xjw::camera_reference
{

    struct CameraReferencePosePriorResult;

    /**
     * A resolved external pose that is safe to use as a solver prior.
     *
     * This type deliberately carries image identity and provenance together with
     * the canonical camera_core pose.  It contains no projection parameters and
     * therefore cannot be passed to a matching or MVS geometry API as a camera.
     */
    class ResolvedCameraPosePrior final
    {
    public:
        ResolvedCameraPosePrior(camera_core::ImageId image,
                                camera_core::ReferenceSourceId source,
                                camera_core::Pose pose,
                                std::optional<camera_core::PoseCovariance> covariance,
                                std::string transformProvenanceHash,
                                std::string transformHash,
                                bool leverArmApplied);

        const camera_core::ImageId& image() const noexcept;
        const camera_core::ReferenceSourceId& source() const noexcept;
        const camera_core::Pose& pose() const noexcept;
        const std::optional<camera_core::PoseCovariance>& covariance() const noexcept;
        const std::string& transformProvenanceHash() const noexcept;
        const std::string& transformHash() const noexcept;
        bool leverArmApplied() const noexcept;

    private:
        friend CameraReferencePosePriorResult makeResolvedCameraPosePrior(const CameraReferenceObservation& observation,
                                                                          const ResolvedCameraReference& reference);

        camera_core::ImageId _image;
        camera_core::ReferenceSourceId _source;
        camera_core::Pose _pose;
        std::optional<camera_core::PoseCovariance> _covariance;
        std::string _transformProvenanceHash;
        std::string _transformHash;
        bool _leverArmApplied = false;
    };

    struct CameraReferencePosePriorResult
    {
        std::optional<ResolvedCameraPosePrior> prior;
        std::string reason;

        bool ok() const noexcept
        {
            return prior.has_value();
        }
    };

    /**
     * Convert a resolver result into a solver-facing typed prior.
     *
     * The conversion is intentionally explicit so unresolved source values,
     * missing orientation conventions, and missing transform provenance cannot
     * enter a numerical solver through a partially populated object.
     */
    CameraReferencePosePriorResult makeResolvedCameraPosePrior(const CameraReferenceObservation& observation,
                                                               const ResolvedCameraReference& reference);

} // namespace xjw::camera_reference
