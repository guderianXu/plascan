#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::aerial_triangulation
{

enum class ReconstructionPrerequisiteRecommendedAction
{
    PrepareFeaturesAndMatches,
    FillMissingMatchesOnly,
    RunSfmWithExistingMatches,
    InspectMatchQuality
};

QString reconstructionPrerequisiteActionToString(ReconstructionPrerequisiteRecommendedAction action);

struct ReconstructionPrerequisiteReport
{
    int imageCount = 0;
    int plannedPairCount = 0;
    int validMatchPairCount = 0;
    int settledNoMatchPairCount = 0;
    int missingFeaturePairCount = 0;
    int missingMatchPairCount = 0;
    int failedGeometryPairCount = 0;

    bool hasEnoughUpstreamData() const;
    bool hasCompletedMatchingPass() const;
    bool shouldOfferGapFill() const;
    bool shouldRunFullRematch() const;
    ReconstructionPrerequisiteRecommendedAction recommendedAction() const;
    QJsonObject toJson() const;
};

} // namespace xjw::aerial_triangulation
