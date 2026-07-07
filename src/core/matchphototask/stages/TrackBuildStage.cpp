#include "TrackBuildStage.h"
#include "TiePointTrackManager.h"

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeTrackReport(MatchPhotosStageStatus status,
                                       const QString &message,
                                       int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("track_build");
    report.displayName = QStringLiteral("连接点轨迹");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

} // namespace

MatchPhotosStageReport TrackBuildStage::run(const MatchPhotosContext &context,
                                            const MatchPhotosOptions &options,
                                            const std::vector<MatchPhotosMatchRecord> &matchRecords,
                                            MatchPhotosResult *result) const
{
    if (options.planOnly)
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("plan-only 模式，跳过连接点轨迹构建"));
    }
    if (!options.enableTrackBuild)
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("连接点轨迹构建已禁用"));
    }
    if (matchRecords.empty())
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("没有可用于构建 track 的匹配结果"));
    }

    const TiePointTrackManager manager;
    const TiePointTrackBuildResult buildResult = manager.build(context, options, matchRecords);
    if (!buildResult.success)
    {
        return makeTrackReport(MatchPhotosStageStatus::Failed,
                               buildResult.errorMessage,
                               buildResult.consumedPairCount);
    }

    if (result)
    {
        result->trackCount = static_cast<int>(buildResult.tracks.size());
        result->acceptedTrackComponents = buildResult.acceptedComponents;
        result->rejectedTrackConflictComponents = buildResult.rejectedConflictComponents;
        result->trackSummary = buildResult.trackSummary;
        result->tiePointPath = buildResult.tiePointPath;
    }

    return makeTrackReport(MatchPhotosStageStatus::Completed,
                           QStringLiteral("连接点轨迹完成：track %1，消费匹配对 %2，跳过 %3")
                               .arg(static_cast<int>(buildResult.tracks.size()))
                               .arg(buildResult.consumedPairCount)
                               .arg(buildResult.skippedPairCount),
                           static_cast<int>(buildResult.tracks.size()));
}

} // namespace matchphotos
} // namespace xjw
