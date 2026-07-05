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
    report.status = MatchPhotosStageStatus::Skipped;
    report.message = options.enableGuidedMatching
        ? QStringLiteral("引导匹配阶段尚未接入")
        : QStringLiteral("引导匹配未启用");
    return report;
}

} // namespace matchphotos
} // namespace xjw
