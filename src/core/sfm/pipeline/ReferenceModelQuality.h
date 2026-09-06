#pragma once

#include "reconstruction/SfmReconstruction.h"

#include <cstdint>

namespace xjw
{

    struct ReferenceInitialPairScore
    {
        int alignedCameraCount = 0;
        int pointCount = 0;
        bool pointCountTier = false;
        bool accuracyTier = false;
        double maximumCameraReprojectionRms = 0.0;
        double geometryAccuracy = 0.0;
        ImageId firstImageId = kInvalidImageId;
        ImageId secondImageId = kInvalidImageId;
    };

    bool referenceInitialPairScoreBetter(const ReferenceInitialPairScore& candidate,
                                         const ReferenceInitialPairScore& incumbent);
    bool referenceInitialPairScoreStable(const ReferenceInitialPairScore& score, int totalImages);
    ReferenceInitialPairScore evaluateReferenceInitialPairScore(const SfmReconstruction& reconstruction,
                                                                ImageId firstImageId,
                                                                ImageId secondImageId);

    struct ReferenceStructureFilterResult
    {
        int farPoints = 0;
        int inaccuratePoints = 0;
        int weakPoints = 0;

        int totalRemoved() const
        {
            return farPoints + inaccuratePoints + weakPoints;
        }
    };

    /// Applies the reference far, inaccurate and weak-point filters in that order.
    ReferenceStructureFilterResult filterReferenceStructurePoints(SfmReconstruction& reconstruction,
                                                                  double requestedReprojectionThreshold,
                                                                  int threadCount = 1);

} // namespace xjw
