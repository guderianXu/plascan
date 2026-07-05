#include "GeometryVerifyStage.h"

namespace xjw
{
namespace matchphotos
{

MatchPhotosStageReport GeometryVerifyStage::run(const MatchPhotosContext &context,
                                                const MatchPhotosOptions &options) const
{
    Q_UNUSED(context)
    Q_UNUSED(options)
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("geometry_verify");
    report.displayName = QStringLiteral("几何验证");
    report.status = MatchPhotosStageStatus::Skipped;
    report.message = QStringLiteral("几何验证阶段尚未接入");
    return report;
}

} // namespace matchphotos
} // namespace xjw
