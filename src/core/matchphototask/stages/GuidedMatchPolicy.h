#pragma once

#include "FeatureSet.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <QHash>

#include <array>

namespace xjw::matchphotos
{

    struct GuidedMatchGeometryChoice
    {
        std::array<double, 9> fundamental{};
        QString geometrySource;
        QString skipReason;
        double epipolarThresholdPixels = 0.0;
        double estimatedMedianResidualPixels = -1.0;
        double referenceMedianResidualPixels = -1.0;
        bool eligible = false;
        bool autoSkippedHealthy = false;
        bool homographyDominant = false;
        bool referenceConflict = false;
    };

    struct GuidedMatchPolicyCache
    {
        QHash<QString, FramePinholeCamera> referenceCamerasByPath;
    };

    GuidedMatchPolicyCache buildGuidedMatchPolicyCache(const MatchPhotosContext& context);

    GuidedMatchGeometryChoice chooseGuidedMatchGeometry(const MatchPhotosContext& context,
                                                        const MatchPhotosOptions& options,
                                                        const GuidedMatchPolicyCache& cache,
                                                        const MatchPhotosMatchRecord& record,
                                                        const image_matching::FeatureSet& features0,
                                                        const image_matching::FeatureSet& features1);

} // namespace xjw::matchphotos
