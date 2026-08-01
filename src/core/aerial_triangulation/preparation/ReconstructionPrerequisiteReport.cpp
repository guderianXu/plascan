/**
 * @file ReconstructionPrerequisiteReport.cpp
 * @brief 根据特征、匹配和负缓存状态决定空三下一步动作。
 *
 * “已完成匹配”不要求每个候选对都有内点：几何失败或已确认无匹配也是稳定结果。
 * 只有尚未处理的缺口才需要补匹配，避免每次重置相机都重复运行全量匹配。
 */

#include "preparation/ReconstructionPrerequisiteReport.h"

#include <algorithm>

namespace xjw::aerial_triangulation
{

QString reconstructionPrerequisiteActionToString(ReconstructionPrerequisiteRecommendedAction action)
{
    switch (action)
    {
    case ReconstructionPrerequisiteRecommendedAction::PrepareImageMatches:
        return QStringLiteral("prepare_image_matches");
    case ReconstructionPrerequisiteRecommendedAction::FillMissingMatchesOnly:
        return QStringLiteral("fill_missing_matches_only");
    case ReconstructionPrerequisiteRecommendedAction::RunSfmWithExistingMatches:
        return QStringLiteral("run_sfm_with_existing_matches");
    case ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality:
        return QStringLiteral("inspect_match_quality");
    }
    return QStringLiteral("prepare_image_matches");
}

bool ReconstructionPrerequisiteReport::hasEnoughUpstreamData() const
{
    return imageCount >= 2 && validMatchPairCount > 0;
}

bool ReconstructionPrerequisiteReport::hasCompletedMatchingPass() const
{
    if (imageCount < 2 || missingMatchPairCount > 0)
    {
        return false;
    }

    // 两类计数可能描述同一批 pair，取最大值避免把重叠状态重复相加。
    const int failedOrSettledPairCount = std::max(settledNoMatchPairCount, failedGeometryPairCount);
    const int processedOutcomeCount = validMatchPairCount + failedOrSettledPairCount;
    return plannedPairCount > 0 && processedOutcomeCount > 0;
}

bool ReconstructionPrerequisiteReport::shouldOfferGapFill() const
{
    return imageCount >= 2 &&
           missingMatchPairCount > 0 &&
           validMatchPairCount > 0;
}

bool ReconstructionPrerequisiteReport::shouldRunFullRematch() const
{
    if (hasCompletedMatchingPass())
    {
        return false;
    }

    return imageCount >= 2 && validMatchPairCount <= 0;
}

ReconstructionPrerequisiteRecommendedAction ReconstructionPrerequisiteReport::recommendedAction() const
{
    // 优先补缺口；已有任意有效网络时直接 SfM；全部处理但无有效边时要求人工检查。
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
    return ReconstructionPrerequisiteRecommendedAction::PrepareImageMatches;
}

QJsonObject ReconstructionPrerequisiteReport::toJson() const
{
    QJsonObject object;
    object[QStringLiteral("image_count")] = imageCount;
    object[QStringLiteral("planned_pair_count")] = plannedPairCount;
    object[QStringLiteral("valid_match_pair_count")] = validMatchPairCount;
    object[QStringLiteral("settled_no_match_pair_count")] = settledNoMatchPairCount;
    object[QStringLiteral("missing_match_pair_count")] = missingMatchPairCount;
    object[QStringLiteral("failed_geometry_pair_count")] = failedGeometryPairCount;
    object[QStringLiteral("recommended_action")] =
        reconstructionPrerequisiteActionToString(recommendedAction());
    return object;
}

} // namespace xjw::aerial_triangulation
