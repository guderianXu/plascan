#include "MatchPhotosAlgorithmPlan.h"

namespace xjw
{
namespace matchphotos
{

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan)
{
    QString summary = QStringLiteral("%1 + %2")
                          .arg(plan.featureAlgorithm.isEmpty() ? QStringLiteral("end-to-end")
                                                               : plan.featureAlgorithm,
                               plan.matcherAlgorithm);
    if (plan.rotationRobust)
    {
        summary += QStringLiteral("，旋转鲁棒");
    }
    if (plan.preferCuda)
    {
        summary += QStringLiteral("，优先 CUDA");
    }
    if (plan.enableGuidedMatching)
    {
        summary += QStringLiteral("，启用引导匹配");
    }
    return summary;
}

} // namespace matchphotos
} // namespace xjw
