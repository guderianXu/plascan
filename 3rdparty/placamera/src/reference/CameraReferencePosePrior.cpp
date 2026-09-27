#include "placamera/reference/CameraReferencePosePrior.h"

#include <utility>

namespace placamera::reference
{

    ResolvedCameraPosePrior::ResolvedCameraPosePrior(placamera::ImageId image,
                                                     placamera::reference::ReferenceSourceId source,
                                                     placamera::Pose pose,
                                                     std::optional<placamera::reference::PoseCovariance> covariance,
                                                     std::string transformProvenanceHash,
                                                     std::string transformHash,
                                                     bool leverArmApplied)
        : _image(std::move(image)), _source(std::move(source)), _pose(std::move(pose)),
          _covariance(std::move(covariance)), _transformProvenanceHash(std::move(transformProvenanceHash)),
          _transformHash(std::move(transformHash)), _leverArmApplied(leverArmApplied)
    {
    }

    const placamera::ImageId& ResolvedCameraPosePrior::image() const noexcept
    {
        return _image;
    }

    const placamera::reference::ReferenceSourceId& ResolvedCameraPosePrior::source() const noexcept
    {
        return _source;
    }

    const placamera::Pose& ResolvedCameraPosePrior::pose() const noexcept
    {
        return _pose;
    }

    const std::optional<placamera::reference::PoseCovariance>& ResolvedCameraPosePrior::covariance() const noexcept
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

    placamera::Result<ResolvedCameraPosePrior>
    makeResolvedCameraPosePrior(const CameraReferenceObservation& observation, const ResolvedCameraReference& reference)
    {
        if (!observation.enabled)
        {
            return placamera::Result<ResolvedCameraPosePrior>::failure(placamera::CameraErrorCode::InvalidArgument,
                                                                       "reference observation is disabled");
        }
        if (reference.status != ReferenceResolutionStatus::Resolved || !reference.pose)
        {
            return placamera::Result<ResolvedCameraPosePrior>::failure(
                placamera::CameraErrorCode::InvalidModelState,
                reference.reason.empty() ? "reference is unresolved" : reference.reason);
        }
        if (reference.pose->frame != reference.targetFrame)
        {
            return placamera::Result<ResolvedCameraPosePrior>::failure(
                placamera::CameraErrorCode::FrameMismatch, "resolved reference pose frame does not match target frame");
        }
        if (reference.transformHash.empty())
        {
            return placamera::Result<ResolvedCameraPosePrior>::failure(
                placamera::CameraErrorCode::InvalidModelState,
                "resolved reference is missing its observation resolution hash");
        }
        if (reference.transformProvenanceHash.empty())
        {
            return placamera::Result<ResolvedCameraPosePrior>::failure(
                placamera::CameraErrorCode::InvalidModelState,
                "resolved reference is missing frame normalization provenance");
        }

        return placamera::Result<ResolvedCameraPosePrior>::success(
            ResolvedCameraPosePrior(observation.image,
                                    observation.source,
                                    *reference.pose,
                                    reference.covariance,
                                    reference.transformProvenanceHash,
                                    reference.transformHash,
                                    reference.leverArmApplied));
    }

} // namespace placamera::reference
