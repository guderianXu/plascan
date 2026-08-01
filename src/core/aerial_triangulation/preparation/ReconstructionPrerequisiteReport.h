#pragma once

/**
 * @file ReconstructionPrerequisiteReport.h
 * @brief 根据逐影像匹配分片状态决定空三前的最小必要动作。
 *
 * 报告区分“缺失”“已确认无匹配”和“几何验证失败”。后两者代表一次匹配尝试已有
 * 确定结果，不能被当作待扫描文件反复重做。
 */

#include <QJsonObject>
#include <QString>

namespace xjw::aerial_triangulation
{

enum class ReconstructionPrerequisiteRecommendedAction
{
    PrepareImageMatches, ///< 没有可用匹配网络，执行完整影像匹配前端。
    FillMissingMatchesOnly, ///< 特征完整且已有部分有效 pair，仅补缺口。
    RunSfmWithExistingMatches, ///< 上游已足够，直接进入正式 SfM。
    InspectMatchQuality ///< 匹配轮次完成但没有可用几何，提示检查质量。
};

QString reconstructionPrerequisiteActionToString(ReconstructionPrerequisiteRecommendedAction action);

struct ReconstructionPrerequisiteReport
{
    int imageCount = 0; ///< 当前处理影像数。
    int plannedPairCount = 0; ///< 候选规划要求处理的 pair 数。
    int validMatchPairCount = 0; ///< 存在可用几何内点的 pair 数。
    int settledNoMatchPairCount = 0; ///< 缓存明确记录无匹配的 pair 数。
    int missingMatchPairCount = 0; ///< 候选 pair 尚未写入任何确定结果的数量。
    int failedGeometryPairCount = 0; ///< 已匹配但几何验证无内点的 pair 数。

    /// 至少两图且已有一个有效匹配对。
    bool hasEnoughUpstreamData() const;

    /// 所有计划 pair 均已有有效、无匹配或几何失败的确定结果。
    bool hasCompletedMatchingPass() const;

    /// 是否可在保留当前有效结果的前提下只补匹配缺口。
    bool shouldOfferGapFill() const;

    /// 是否必须重新运行完整的内存特征提取与影像匹配流程。
    bool shouldRunFullRematch() const;

    /// 按最少重算原则返回建议动作。
    ReconstructionPrerequisiteRecommendedAction recommendedAction() const;
    QJsonObject toJson() const;
};

} // namespace xjw::aerial_triangulation
