#include "MatchPhotosAlgorithmPlan.h"

namespace xjw
{
namespace matchphotos
{

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan)
{
    QString summary = plan.displayName.isEmpty()
        ? plan.algorithmId
        : plan.displayName;
    if (plan.algorithmVersion > 0)
    {
        summary += QStringLiteral(" v%1").arg(plan.algorithmVersion);
    }
    if (plan.rotationRobust)
    {
        summary += QStringLiteral("，旋转鲁棒");
    }
    if (plan.preferCuda)
    {
        summary += QStringLiteral("，优先 CUDA");
    }
    if (plan.executionBackend != image_matching::SiftComputeBackend::Automatic)
    {
        summary += QStringLiteral("，%1").arg(plan.backendReason);
    }
    if (plan.enableGuidedMatching)
    {
        summary += QStringLiteral("，启用引导匹配");
    }
    return summary;
}

} // namespace matchphotos
} // namespace xjw
