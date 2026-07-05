#include "MatchPhotosTask.h"

#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchingStage.h"
#include "TrackBuildStage.h"

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeAlgorithmSelectionReport(const MatchPhotosAlgorithmPlan &plan)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("algorithm_selection");
    report.displayName = QStringLiteral("算法选择");
    report.status = MatchPhotosStageStatus::Completed;
    report.message = QStringLiteral("%1：%2")
                         .arg(algorithmPlanSummary(plan), plan.reason);
    return report;
}

MatchPhotosStageReport makePairSelectionReport(const PairSelectionResult &selection)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("pair_selection");
    report.displayName = QStringLiteral("影像对选择");
    report.status = MatchPhotosStageStatus::Completed;
    report.itemCount = static_cast<int>(selection.candidates.size());
    report.message = selection.detail;
    return report;
}

bool appendStageAndStopOnFailure(MatchPhotosResult *result,
                                 const MatchPhotosStageReport &report)
{
    if (!result)
    {
        return true;
    }

    result->stages.push_back(report);
    if (report.status != MatchPhotosStageStatus::Failed)
    {
        return false;
    }

    result->success = false;
    result->errorMessage = report.message;
    return true;
}

} // namespace

MatchPhotosTask::MatchPhotosTask(const MatchPhotosOptions &options)
    : _options(options)
{
}

const MatchPhotosOptions &MatchPhotosTask::options() const
{
    return _options;
}

MatchPhotosResult MatchPhotosTask::run(const MatchPhotosContext &context) const
{
    MatchPhotosResult result;
    result.algorithmPlan = MatchPhotosAlgorithmSelector::select(_options);
    result.stages.push_back(makeAlgorithmSelectionReport(result.algorithmPlan));

    QString errorMessage;
    // 影像对选择是当前第一个真实阶段；
    // 后续所有步骤都必须共享同一份“允许匹配哪些影像对”的结论。
    result.pairSelection = PairSelector::select(context.pairInput, _options.pairPolicy, &errorMessage);
    if (!errorMessage.isEmpty())
    {
        result.errorMessage = errorMessage;
        result.success = false;
        return result;
    }

    result.stages.push_back(makePairSelectionReport(result.pairSelection));

    // 这些阶段对象当前刻意保持短生命周期、无状态。
    // 后续接入真实运行器后，取消和进度状态应放在上下文或运行器中维护。
    const FeatureStage featureStage;
    const MatchingStage matchingStage;
    const GeometryVerifyStage geometryVerifyStage;
    const TrackBuildStage trackBuildStage;
    const GuidedMatchStage guidedMatchStage;

    if (appendStageAndStopOnFailure(
            &result,
            featureStage.run(context, _options, result.algorithmPlan, &result.features)))
    {
        return result;
    }
    if (appendStageAndStopOnFailure(
            &result,
            matchingStage.run(context, _options, result.algorithmPlan, result.pairSelection, &result.matches)))
    {
        return result;
    }
    result.stages.push_back(geometryVerifyStage.run(context, _options));
    result.stages.push_back(trackBuildStage.run(context, _options));
    result.stages.push_back(guidedMatchStage.run(context, _options));

    result.success = true;
    return result;
}

} // namespace matchphotos
} // namespace xjw
