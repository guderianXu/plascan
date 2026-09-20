#include "CameraReferencePosePrior.h"

#include <utility>

namespace xjw::camera_reference
{

    ResolvedCameraPosePrior::ResolvedCameraPosePrior(camera_core::ImageId image,
                                                     camera_core::ReferenceSourceId source,
                                                     camera_core::Pose pose,
                                                     std::optional<camera_core::PoseCovariance> covariance,
                                                     std::string transformProvenanceHash,
                                                     std::string transformHash,
                                                     bool leverArmApplied)
        : _image(std::move(image)), _source(std::move(source)), _pose(std::move(pose)),
          _covariance(std::move(covariance)), _transformProvenanceHash(std::move(transformProvenanceHash)),
          _transformHash(std::move(transformHash)), _leverArmApplied(leverArmApplied)
    {
    }

    const camera_core::ImageId& ResolvedCameraPosePrior::image() const noexcept
    {
        return _image;
    }

    const camera_core::ReferenceSourceId& ResolvedCameraPosePrior::source() const noexcept
    {
        return _source;
    }

    const camera_core::Pose& ResolvedCameraPosePrior::pose() const noexcept
    {
        return _pose;
    }

    const std::optional<camera_core::PoseCovariance>& ResolvedCameraPosePrior::covariance() const noexcept
    {
        return _covariance;
    }

    const std::string& ResolvedCameraPosePrior::transformProvenanceHash() const noexcept
    {
        return _transformProvenanceHash;
    }

    const std::string& ResolvedCameraPosePrior::transformHash() const noexcept
    {
        return _transformHash;
    }

    bool ResolvedCameraPosePrior::leverArmApplied() const noexcept
    {
        return _leverArmApplied;
    }

    CameraReferencePosePriorResult makeResolvedCameraPosePrior(const CameraReferenceObservation& observation,
                                                               const ResolvedCameraReference& reference)
    {
        CameraReferencePosePriorResult result;
        if (!observation.enabled)
        {
            result.reason = "reference observation is disabled";
            return result;
        }
        if (reference.status != ReferenceResolutionStatus::Resolved || !reference.pose)
        {
            result.reason = reference.reason.empty() ? "reference is unresolved" : reference.reason;
            return result;
        }
        if (reference.pose->frame != reference.targetFrame)
        {
            result.reason = "resolved reference pose frame does not match target frame";
            return result;
        }
        if (reference.transformHash.empty())
        {
            result.reason = "resolved reference is missing its observation resolution hash";
            return result;
        }
        if (reference.transformProvenanceHash.empty())
        {
            result.reason = "resolved reference is missing frame normalization provenance";
            return result;
        }

        result.prior.emplace(observation.image,
                             observation.source,
                             *reference.pose,
                             reference.covariance,
                             reference.transformProvenanceHash,
                             reference.transformHash,
                             reference.leverArmApplied);
        return result;
    }

} // namespace xjw::camera_reference
