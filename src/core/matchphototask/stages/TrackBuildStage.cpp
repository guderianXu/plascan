#include "TrackBuildStage.h"

namespace xjw
{
namespace matchphotos
{

MatchPhotosStageReport TrackBuildStage::run(const MatchPhotosContext &context,
                                            const MatchPhotosOptions &options) const
{
    Q_UNUSED(context)
    Q_UNUSED(options)
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("track_build");
    report.displayName = QStringLiteral("轨迹构建");
    report.status = MatchPhotosStageStatus::Skipped;
    report.message = QStringLiteral("轨迹构建阶段尚未接入");
    return report;
}

} // namespace matchphotos
} // namespace xjw
