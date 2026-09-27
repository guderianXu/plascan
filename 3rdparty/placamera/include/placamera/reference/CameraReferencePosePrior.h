#pragma once

#include "placamera/reference/CameraReferenceObservation.h"

#include <placamera/result.h>

#include <optional>
#include <string>

namespace placamera::reference
{

    /**
     * A resolved external pose that is safe to use as a solver prior.
     *
     * This type deliberately carries image identity and provenance together with
     * the PlaCamera pose. It contains no projection parameters and
     * therefore cannot be passed to a matching or MVS geometry API as a camera.
     */
    class ResolvedCameraPosePrior final
    {
    public:
        ResolvedCameraPosePrior(placamera::ImageId image,
                                placamera::reference::ReferenceSourceId source,
                                placamera::Pose pose,
                                std::optional<placamera::reference::PoseCovariance> covariance,
                                std::string transformProvenanceHash,
                                std::string transformHash,
                                bool leverArmApplied);

        const placamera::ImageId& image() const noexcept;
        const placamera::reference::ReferenceSourceId& source() const noexcept;
        const placamera::Pose& pose() const noexcept;
        const std::optional<placamera::reference::PoseCovariance>& covariance() const noexcept;
        const std::string& transformProvenanceHash() const noexcept;
        const std::string& transformHash() const noexcept;
        bool leverArmApplied() const noexcept;

    private:
        placamera::ImageId _image;
        placamera::reference::ReferenceSourceId _source;
        placamera::Pose _pose;
        std::optional<placamera::reference::PoseCovariance> _covariance;
        std::string _transformProvenanceHash;
        std::string _transformHash;
        bool _leverArmApplied = false;
    };

    /**
     * Convert a resolver result into a solver-facing typed prior.
     *
     * The conversion is intentionally explicit so unresolved source values,
     * missing orientation conventions, and missing transform provenance cannot
     * enter a numerical solver through a partially populated object.
     */
    placamera::Result<ResolvedCameraPosePrior>
    makeResolvedCameraPosePrior(const CameraReferenceObservation& observation,
                                const ResolvedCameraReference& reference);

} // namespace placamera::reference
