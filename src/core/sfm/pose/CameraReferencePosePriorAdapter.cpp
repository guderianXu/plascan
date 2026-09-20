#include "CameraReferencePosePriorAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>

namespace xjw
{
    namespace
    {

        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

        bool validPositiveFinite(double value)
        {
            return std::isfinite(value) && value > 0.0;
        }

        std::array<std::size_t, 6> covarianceDiagonalIndices(camera_core::CovarianceLayout layout)
        {
            if (layout == camera_core::CovarianceLayout::Diagonal6)
            {
                return {{0, 1, 2, 3, 4, 5}};
            }
            return {{0, 6, 11, 15, 18, 20}};
        }

        bool covarianceSigmas(const camera_core::PoseCovariance& covariance,
                              double* positionSigmaMeters,
                              double* rotationSigmaDegrees)
        {
            if (!positionSigmaMeters || !rotationSigmaDegrees)
            {
                return false;
            }
            const auto& values = covariance.values();
            const auto indices = covarianceDiagonalIndices(covariance.layout());
            double maximumPositionVariance = 0.0;
            double maximumRotationVariance = 0.0;
            for (std::size_t index = 0; index < indices.size(); ++index)
            {
                if (indices[index] >= values.size() || !std::isfinite(values[indices[index]]) ||
                    values[indices[index]] < 0.0)
                {
                    return false;
                }
                if (index < 3)
                {
                    maximumPositionVariance = std::max(maximumPositionVariance, values[indices[index]]);
                }
                else
                {
                    maximumRotationVariance = std::max(maximumRotationVariance, values[indices[index]]);
                }
            }
            *positionSigmaMeters = std::sqrt(std::max(maximumPositionVariance, 1.0e-12));
            *rotationSigmaDegrees = std::sqrt(std::max(maximumRotationVariance, 1.0e-12)) * kRadiansToDegrees;
            return validPositiveFinite(*positionSigmaMeters) && validPositiveFinite(*rotationSigmaDegrees);
        }

    } // namespace

    CameraReferencePosePriorAdapterResult CameraReferencePosePriorAdapter::toBundleAdjustPriors(
        const std::vector<camera_models::frame_pinhole::FramePinholeNumericState>& cameras,
        const std::vector<camera_reference::ResolvedCameraPosePrior>& references,
        double defaultPositionSigmaMeters,
        double defaultRotationSigmaDegrees)
    {
        CameraReferencePosePriorAdapterResult result;
        if (!validPositiveFinite(defaultPositionSigmaMeters) || !validPositiveFinite(defaultRotationSigmaDegrees))
        {
            result.error = "default pose-prior sigmas must be finite and positive";
            return result;
        }

        result.priors.resize(cameras.size());
        if (cameras.empty())
        {
            result.valid = references.empty();
            if (!result.valid)
            {
                result.error = "cannot align pose references with an empty camera set";
            }
            return result;
        }

        const xjw::coordinate_system::CoordinateFrameId commonFrame = cameras.front().worldFrame();
        for (const auto& camera : cameras)
        {
            if (!camera.hasBoundIdentity())
            {
                result.error = "camera numeric state has no explicit image identity/world frame; bind a typed instance "
                               "before aligning external pose references";
                return result;
            }
            std::string validationError;
            if (!camera.validateNumericalState(&validationError))
            {
                result.error =
                    "camera " + camera.imageId().value() + " has invalid numerical state: " + validationError;
                return result;
            }
            if (camera.worldFrame() != commonFrame)
            {
                result.error = "numeric camera set mixes world frames at image " + camera.imageId().value() +
                               ": expected " + commonFrame.value() + ", observed " + camera.worldFrame().value();
                return result;
            }
        }

        std::unordered_map<camera_core::ImageId, std::size_t> cameraIndices;
        cameraIndices.reserve(cameras.size());
        for (std::size_t index = 0; index < cameras.size(); ++index)
        {
            if (!cameraIndices.emplace(cameras[index].imageId(), index).second)
            {
                result.error = "numeric camera set contains duplicate image " + cameras[index].imageId().value();
                return result;
            }
        }

        std::unordered_map<camera_core::ImageId, std::size_t> seenReferences;
        seenReferences.reserve(references.size());
        std::string commonTransformProvenanceHash;
        for (const auto& reference : references)
        {
            if (!seenReferences.emplace(reference.image(), seenReferences.size()).second)
            {
                result.error = "duplicate pose reference for image " + reference.image().value();
                return result;
            }
            const auto cameraIterator = cameraIndices.find(reference.image());
            if (cameraIterator == cameraIndices.end())
            {
                ++result.ignoredReferenceCount;
                result.ignoredReferenceImages.push_back(reference.image());
                continue;
            }
            const std::size_t cameraIndex = cameraIterator->second;
            if (reference.pose().frame != commonFrame)
            {
                result.error = "pose reference frame mismatch for image " + reference.image().value() +
                               ": camera uses " + commonFrame.value() + ", reference uses " +
                               reference.pose().frame.value();
                return result;
            }
            if (reference.transformProvenanceHash().empty())
            {
                result.error =
                    "pose reference is missing frame normalization provenance for image " + reference.image().value();
                return result;
            }
            if (commonTransformProvenanceHash.empty())
            {
                commonTransformProvenanceHash = reference.transformProvenanceHash();
                result.commonTransformProvenanceHash = commonTransformProvenanceHash;
            }
            else if (reference.transformProvenanceHash() != commonTransformProvenanceHash)
            {
                result.error = "pose references use different frame normalization provenance at image " +
                               reference.image().value() + " (source " + reference.source().value() + "): expected " +
                               commonTransformProvenanceHash + ", observed " + reference.transformProvenanceHash();
                return result;
            }

            BACameraPosePrior& prior = result.priors[cameraIndex];
            prior.enabled = true;
            prior.cameraToWorldRotation = reference.pose().cameraToWorldRotation;
            prior.cameraCenter = reference.pose().center;
            prior.positionSigmaMeters = defaultPositionSigmaMeters;
            prior.rotationSigmaDegrees = defaultRotationSigmaDegrees;
            if (reference.covariance())
            {
                if (!covarianceSigmas(*reference.covariance(), &prior.positionSigmaMeters, &prior.rotationSigmaDegrees))
                {
                    result.error = "pose reference covariance is invalid for image " + reference.image().value();
                    return result;
                }
            }
            ++result.matchedReferenceCount;
        }

        result.valid = true;
        return result;
    }

} // namespace xjw
