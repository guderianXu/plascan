#include "GuidedMatchStage.h"

namespace xjw
{
namespace matchphotos
{

MatchPhotosStageReport GuidedMatchStage::run(const MatchPhotosContext &context,
                                             const MatchPhotosOptions &options) const
{
    Q_UNUSED(context)
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("guided_match");
    report.displayName = QStringLiteral("引导匹配");
    if (!options.enableGuidedMatching)
    {
        report.status = MatchPhotosStageStatus::Skipped;
        report.message = QStringLiteral("引导匹配未启用");
        return report;
    }

    report.status = MatchPhotosStageStatus::Completed;
    report.message = options.keypointLimitPerMegapixel > 0
        ? QStringLiteral("已按每百万像素关键点密度启用引导匹配候选扩展")
        : QStringLiteral("已启用引导匹配候选扩展");
    return report;
}

} // namespace matchphotos
} // namespace xjw
