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

        bool covarianceSigmas(const placamera::reference::PoseCovariance& covariance,
                              double* positionSigmaMeters,
                              double* rotationSigmaDegrees)
        {
            if (!positionSigmaMeters || !rotationSigmaDegrees)
            {
                return false;
            }
            const auto& values = covariance.matrixValues();
            double maximumPositionVariance = 0.0;
            double maximumRotationVariance = 0.0;
            for (std::size_t index = 0; index < 6; ++index)
            {
                const double variance = values[index * 6 + index];
                if (!std::isfinite(variance) || variance < 0.0)
                {
                    return false;
                }
                if (index < 3)
                {
                    maximumPositionVariance = std::max(maximumPositionVariance, variance);
                }
                else
                {
                    maximumRotationVariance = std::max(maximumRotationVariance, variance);
                }
            }
            if (covariance.hasPosition())
            {
                *positionSigmaMeters = std::sqrt(std::max(maximumPositionVariance, 1.0e-12));
            }
            if (covariance.hasRotation())
            {
                *rotationSigmaDegrees = std::sqrt(std::max(maximumRotationVariance, 1.0e-12)) * kRadiansToDegrees;
            }
            return validPositiveFinite(*positionSigmaMeters) && validPositiveFinite(*rotationSigmaDegrees);
        }

    } // namespace

    CameraReferencePosePriorAdapterResult CameraReferencePosePriorAdapter::toBundleAdjustPriors(
        const std::vector<CameraReferenceTarget>& targets,
        const std::vector<placamera::reference::ResolvedCameraPosePrior>& references,
        double defaultPositionSigmaMeters,
        double defaultRotationSigmaDegrees)
    {
        CameraReferencePosePriorAdapterResult result;
        if (!validPositiveFinite(defaultPositionSigmaMeters) || !validPositiveFinite(defaultRotationSigmaDegrees))
        {
            result.error = "default pose-prior sigmas must be finite and positive";
            return result;
        }

        result.priors.resize(targets.size());
        if (targets.empty())
        {
            result.valid = references.empty();
            if (!result.valid)
            {
                result.error = "cannot align pose references with an empty camera set";
            }
            return result;
        }

        const placamera::FrameId commonFrame = targets.front().worldFrame;
        for (const auto& target : targets)
        {
            if (target.worldFrame != commonFrame)
            {
                result.error = "camera target set mixes world frames at image " + target.imageId.value() +
                               ": expected " + commonFrame.value() + ", observed " + target.worldFrame.value();
                return result;
            }
        }

        std::unordered_map<placamera::ImageId, std::size_t> cameraIndices;
        cameraIndices.reserve(targets.size());
        for (std::size_t index = 0; index < targets.size(); ++index)
        {
            if (!cameraIndices.emplace(targets[index].imageId, index).second)
            {
                result.error = "camera target set contains duplicate image " + targets[index].imageId.value();
                return result;
            }
        }

        std::unordered_map<placamera::ImageId, std::size_t> seenReferences;
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

            plabundle::CameraPosePrior prior;
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
            result.priors[cameraIndex] = prior;
            ++result.matchedReferenceCount;
        }

        result.valid = true;
        return result;
    }

} // namespace xjw
