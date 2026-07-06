#include "ReconstructionPrerequisiteReport.h"

#include <algorithm>

namespace xjw
{
namespace gui
{

QString reconstructionPrerequisiteActionToString(ReconstructionPrerequisiteRecommendedAction action)
{
    switch (action)
    {
    case ReconstructionPrerequisiteRecommendedAction::PrepareFeaturesAndMatches:
        return QStringLiteral("prepare_features_and_matches");
    case ReconstructionPrerequisiteRecommendedAction::FillMissingMatchesOnly:
        return QStringLiteral("fill_missing_matches_only");
    case ReconstructionPrerequisiteRecommendedAction::RunSfmWithExistingMatches:
        return QStringLiteral("run_sfm_with_existing_matches");
    case ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality:
        return QStringLiteral("inspect_match_quality");
    }
    return QStringLiteral("prepare_features_and_matches");
}

bool ReconstructionPrerequisiteReport::hasEnoughUpstreamData() const
{
    return imageCount >= 2 &&
           missingFeaturePairCount <= 0 &&
           validMatchPairCount > 0;
}

bool ReconstructionPrerequisiteReport::hasCompletedMatchingPass() const
{
    if (imageCount < 2 || missingFeaturePairCount > 0 || missingMatchPairCount > 0)
    {
        return false;
    }

    const int failedOrSettledPairCount = std::max(settledNoMatchPairCount, failedGeometryPairCount);
    const int processedOutcomeCount = validMatchPairCount + failedOrSettledPairCount;
    return plannedPairCount > 0 && processedOutcomeCount > 0;
}

bool ReconstructionPrerequisiteReport::shouldOfferGapFill() const
{
    return imageCount >= 2 &&
           missingFeaturePairCount <= 0 &&
           missingMatchPairCount > 0 &&
           validMatchPairCount > 0;
}

bool ReconstructionPrerequisiteReport::shouldRunFullRematch() const
{
    if (hasCompletedMatchingPass())
    {
        return false;
    }

    return imageCount >= 2 &&
           (missingFeaturePairCount > 0 || validMatchPairCount <= 0);
}

ReconstructionPrerequisiteRecommendedAction ReconstructionPrerequisiteReport::recommendedAction() const
{
    if (shouldOfferGapFill())
    {
        return ReconstructionPrerequisiteRecommendedAction::FillMissingMatchesOnly;
    }
    if (hasEnoughUpstreamData())
    {
        return ReconstructionPrerequisiteRecommendedAction::RunSfmWithExistingMatches;
    }
    if (hasCompletedMatchingPass())
    {
        return ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality;
    }
    return ReconstructionPrerequisiteRecommendedAction::PrepareFeaturesAndMatches;
}

QJsonObject ReconstructionPrerequisiteReport::toJson() const
{
    QJsonObject object;
    object[QStringLiteral("image_count")] = imageCount;
    object[QStringLiteral("planned_pair_count")] = plannedPairCount;
    object[QStringLiteral("valid_match_pair_count")] = validMatchPairCount;
    object[QStringLiteral("settled_no_match_pair_count")] = settledNoMatchPairCount;
    object[QStringLiteral("missing_feature_pair_count")] = missingFeaturePairCount;
    object[QStringLiteral("missing_match_pair_count")] = missingMatchPairCount;
    object[QStringLiteral("failed_geometry_pair_count")] = failedGeometryPairCount;
    object[QStringLiteral("recommended_action")] =
        reconstructionPrerequisiteActionToString(recommendedAction());
    return object;
}

} // namespace gui
} // namespace xjw
