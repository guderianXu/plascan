#include "MatchPhotosTask.h"

#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "PlaMatchHctPairPreselector.h"
#include "TrackBuildStage.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftFeatureExtractor.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace xjw
{
    namespace matchphotos
    {

        QJsonObject MatchPhotosFunnelDiagnostics::toJson() const
        {
            QJsonObject object;
            object[QStringLiteral("all_pair_count")] = allPairCount;
            object[QStringLiteral("selected_pair_count")] = selectedPairCount;
            object[QStringLiteral("matched_pair_count")] = matchedPairCount;
            object[QStringLiteral("raw_match_count")] = static_cast<double>(rawMatchCount);
            object[QStringLiteral("geometry_passed_pair_count")] = geometryPassedPairCount;
            object[QStringLiteral("geometry_inlier_count")] = static_cast<double>(geometryInlierCount);
            object[QStringLiteral("guided_added_inlier_count")] = static_cast<double>(guidedAddedInlierCount);
            object[QStringLiteral("final_passed_pair_count")] = finalPassedPairCount;
            object[QStringLiteral("final_geometry_inlier_count")] = static_cast<double>(finalGeometryInlierCount);
            object[QStringLiteral("track_count")] = trackCount;
            object[QStringLiteral("pair_selection_ratio")] = pairSelectionRatio;
            object[QStringLiteral("matching_yield_ratio")] = matchingYieldRatio;
            object[QStringLiteral("geometry_pair_retention_ratio")] = geometryPairRetentionRatio;
            object[QStringLiteral("geometry_inlier_ratio")] = geometryInlierRatio;
            object[QStringLiteral("guided_gain_ratio")] = guidedGainRatio;
            return object;
        }

        MatchPhotosFunnelDiagnostics summarizeMatchPhotosFunnel(const PairSelectionResult& pairSelection,
                                                                const std::vector<MatchPhotosMatchRecord>& finalMatches,
                                                                int matchedPairCount,
                                                                std::int64_t rawMatchCount,
                                                                int geometryPassedPairCount,
                                                                std::int64_t geometryInlierCount,
                                                                std::int64_t guidedAddedInlierCount,
                                                                int trackCount)
        {
            MatchPhotosFunnelDiagnostics diagnostics;
            diagnostics.allPairCount = std::max(0, pairSelection.allPairCount);
            diagnostics.selectedPairCount = static_cast<int>(pairSelection.candidates.size());
            diagnostics.matchedPairCount = std::max(0, matchedPairCount);
            diagnostics.rawMatchCount = std::max<std::int64_t>(0, rawMatchCount);
            diagnostics.geometryPassedPairCount = std::max(0, geometryPassedPairCount);
            diagnostics.geometryInlierCount = std::max<std::int64_t>(0, geometryInlierCount);
            diagnostics.guidedAddedInlierCount = std::max<std::int64_t>(0, guidedAddedInlierCount);
            diagnostics.trackCount = std::max(0, trackCount);

            for (const MatchPhotosMatchRecord& record : finalMatches)
            {
                if (record.passedGeometry)
                {
                    ++diagnostics.finalPassedPairCount;
                    diagnostics.finalGeometryInlierCount += std::max(0, record.geometricInlierCount);
                }
            }

            const auto boundedRatio = [](std::int64_t numerator, std::int64_t denominator)
            {
                const double value =
                    denominator > 0 ? static_cast<double>(numerator) / static_cast<double>(denominator) : 0.0;
                return std::clamp(value, 0.0, 1.0);
            };
            diagnostics.pairSelectionRatio = boundedRatio(diagnostics.selectedPairCount, diagnostics.allPairCount);
            diagnostics.matchingYieldRatio = boundedRatio(diagnostics.matchedPairCount, diagnostics.selectedPairCount);
            diagnostics.geometryPairRetentionRatio =
                boundedRatio(diagnostics.geometryPassedPairCount, diagnostics.matchedPairCount);
            diagnostics.geometryInlierRatio = boundedRatio(diagnostics.geometryInlierCount, diagnostics.rawMatchCount);
            diagnostics.guidedGainRatio = diagnostics.geometryInlierCount > 0
                                              ? static_cast<double>(diagnostics.guidedAddedInlierCount) /
                                                    static_cast<double>(diagnostics.geometryInlierCount)
                                              : 0.0;
            return diagnostics;
        }

        namespace
        {

            MatchPhotosStageReport makeAlgorithmSelectionReport(const MatchPhotosAlgorithmPlan& plan)
            {
                MatchPhotosStageReport report;
                report.stageId = QStringLiteral("algorithm_selection");
                report.displayName = QStringLiteral("算法选择");
                report.status = plan.valid ? MatchPhotosStageStatus::Completed : MatchPhotosStageStatus::Failed;
                if (plan.valid)
                {
                    report.message = QStringLiteral("%1：%2").arg(algorithmPlanSummary(plan), plan.reason);
                    if (!plan.backendReason.isEmpty())
                    {
                        report.message += QStringLiteral("；%1").arg(plan.backendReason);
                    }
                }
                else
                {
                    report.message = plan.validationError;
                }
                return report;
            }

            MatchPhotosStageReport makePairSelectionReport(const PairSelectionResult& selection)
            {
                MatchPhotosStageReport report;
                report.stageId = QStringLiteral("pair_selection");
                report.displayName = QStringLiteral("影像对选择");
                report.status = MatchPhotosStageStatus::Completed;
                report.itemCount = static_cast<int>(selection.candidates.size());
                report.message = selection.detail;
                return report;
            }

            MatchPhotosStageReport makeMatchingFunnelReport(const MatchPhotosFunnelDiagnostics& diagnostics)
            {
                MatchPhotosStageReport report;
                report.stageId = QStringLiteral("matching_funnel");
                report.displayName = QStringLiteral("匹配漏斗诊断");
                report.status = MatchPhotosStageStatus::Completed;
                report.itemCount = diagnostics.trackCount;
                report.message = QStringLiteral("像对 %1/%2（%3%） → 有初始匹配 %4 对/%5 个 → "
                                                "几何通过 %6 对/%7 内点（%8%） → "
                                                "guided +%9（%10%） → 最终 %11 对/%12 内点 → %13 条轨迹")
                                     .arg(diagnostics.selectedPairCount)
                                     .arg(diagnostics.allPairCount)
                                     .arg(diagnostics.pairSelectionRatio * 100.0, 0, 'f', 1)
                                     .arg(diagnostics.matchedPairCount)
                                     .arg(static_cast<qlonglong>(diagnostics.rawMatchCount))
                                     .arg(diagnostics.geometryPassedPairCount)
                                     .arg(static_cast<qlonglong>(diagnostics.geometryInlierCount))
                                     .arg(diagnostics.geometryInlierRatio * 100.0, 0, 'f', 1)
                                     .arg(static_cast<qlonglong>(diagnostics.guidedAddedInlierCount))
                                     .arg(diagnostics.guidedGainRatio * 100.0, 0, 'f', 1)
                                     .arg(diagnostics.finalPassedPairCount)
                                     .arg(static_cast<qlonglong>(diagnostics.finalGeometryInlierCount))
                                     .arg(diagnostics.trackCount);
                return report;
            }

            MatchPhotosStageReport
            makeGenericPreselectionReport(MatchPhotosStageStatus status, const QString& message, int itemCount)
            {
                MatchPhotosStageReport report;
                report.stageId = QStringLiteral("generic_preselection");
                report.displayName = QStringLiteral("通用预选");
                report.status = status;
                report.message = message;
                report.itemCount = itemCount;
                return report;
            }

            image_matching::SiftComputeBackend requestedSiftBackend(ComputeDevice device)
            {
                switch (device)
                {
                case ComputeDevice::Auto:
                    return image_matching::SiftComputeBackend::Automatic;
                case ComputeDevice::Cpu:
                    return image_matching::SiftComputeBackend::Cpu;
                case ComputeDevice::Cuda:
                    return image_matching::SiftComputeBackend::Cuda;
                case ComputeDevice::OpenCl:
                    return image_matching::SiftComputeBackend::OpenCl;
                case ComputeDevice::Metal:
                    return image_matching::SiftComputeBackend::Metal;
                }
                return image_matching::SiftComputeBackend::Automatic;
            }

            MatchPhotosStageReport
            makeReferencePreselectionReport(MatchPhotosStageStatus status, const QString& message, int itemCount)
            {
                MatchPhotosStageReport report;
                report.stageId = QStringLiteral("reference_preselection");
                report.displayName = QStringLiteral("参考预选");
                report.status = status;
                report.message = message;
                report.itemCount = itemCount;
                return report;
            }

            bool appendStageAndStopOnFailure(MatchPhotosResult* result, const MatchPhotosStageReport& report)
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

            void clearTransientMatchPayloads(std::vector<MatchPhotosMatchRecord>* matchRecords)
            {
                if (!matchRecords)
                {
                    return;
                }

                for (MatchPhotosMatchRecord& record : *matchRecords)
                {
                    // PairMatchData 已经提交到每影像 `.pimatch` 分片，返回 GUI 后只需保留
                    // 路径和统计。及时释放坐标数组可显著降低大型项目的峰值内存。
                    record.pairData.reset();
                }
            }

        } // namespace

        MatchPhotosTask::MatchPhotosTask(const MatchPhotosOptions& options) : _options(options)
        {
        }

        const MatchPhotosOptions& MatchPhotosTask::options() const
        {
            return _options;
        }

        MatchPhotosResult MatchPhotosTask::run(const MatchPhotosContext& context) const
        {
            MatchPhotosResult result;
            MatchPhotosContext runtimeContext = context;
            if (!runtimeContext.featureCache)
            {
                runtimeContext.featureCache = std::make_shared<MatchPhotosFeatureCache>();
            }

            result.algorithmPlan = MatchPhotosAlgorithmSelector::select(_options);
            if (result.algorithmPlan.valid &&
                result.algorithmPlan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId))
            {
                try
                {
                    const image_matching::SiftComputeBackend backend =
                        image_matching::SiftFeatureExtractor::resolveBackend(requestedSiftBackend(_options.device),
                                                                             _options.cudaDevice);
                    result.algorithmPlan = MatchPhotosAlgorithmSelector::resolveExecutionBackend(
                        _options, std::move(result.algorithmPlan), backend, _options.cudaDevice);
                }
                catch (const std::exception& error)
                {
                    result.algorithmPlan.valid = false;
                    result.algorithmPlan.validationError = QString::fromUtf8(error.what());
                }
            }
            const QString algorithmName = result.algorithmPlan.displayName.isEmpty() ? result.algorithmPlan.algorithmId
                                                                                     : result.algorithmPlan.displayName;
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("algorithm_selection"),
                                      QStringLiteral("选择 %1 连接点流程").arg(algorithmName),
                                      0,
                                      1);
            result.stages.push_back(makeAlgorithmSelectionReport(result.algorithmPlan));
            if (!result.algorithmPlan.valid)
            {
                result.errorMessage = result.algorithmPlan.validationError;
                return result;
            }
            if (runtimeContext.computeDeviceCallback &&
                !result.algorithmPlan.computeDeviceDisplayName.trimmed().isEmpty())
            {
                runtimeContext.computeDeviceCallback(result.algorithmPlan.computeDeviceDisplayName);
            }
            reportMatchPhotosProgress(
                runtimeContext,
                QStringLiteral("algorithm_selection"),
                result.algorithmPlan.backendReason.isEmpty()
                    ? QStringLiteral("连接点算法已确定: %1").arg(algorithmName)
                    : QStringLiteral("连接点算法已确定: %1；%2").arg(algorithmName, result.algorithmPlan.backendReason),
                1,
                1);

            // 这些阶段对象当前刻意保持短生命周期、无状态。
            // 后续接入真实运行器后，取消和进度状态应放在上下文或运行器中维护。
            const FeatureStage featureStage;
            const MatchingStage matchingStage;
            const GeometryVerifyStage geometryVerifyStage;
            const TrackBuildStage trackBuildStage;
            const GuidedMatchStage guidedMatchStage;

            PairSelectionInput pairInput = runtimeContext.pairInput;
            PairSelectionPolicy pairPolicy = _options.pairPolicy;
            const bool manualOnly = pairPolicy.mode == PairSelectionMode::ManualOnly;
            MatchPhotosOptions effectiveOptions = _options;
            if (manualOnly)
            {
                effectiveOptions.useGenericPreselection = false;
                effectiveOptions.useReferencePreselection = false;
            }

            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("feature"),
                                      QStringLiteral("%1 特征提取：准备处理影像").arg(algorithmName),
                                      0,
                                      std::max(1, static_cast<int>(runtimeContext.pairInput.images.size())));
            const MatchPhotosStageReport featureReport =
                featureStage.run(runtimeContext, _options, result.algorithmPlan, &result.features);
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("feature"),
                                      featureReport.message,
                                      featureReport.itemCount,
                                      std::max(1, static_cast<int>(runtimeContext.pairInput.images.size())));
            if (appendStageAndStopOnFailure(&result, featureReport))
            {
                return result;
            }

            const bool automaticPreselectionRequested =
                effectiveOptions.useGenericPreselection || effectiveOptions.useReferencePreselection;
            const bool useUnifiedPreselection = pairPolicy.mode == PairSelectionMode::Auto &&
                                                automaticPreselectionRequested && !effectiveOptions.planOnly;
            MatchPhotosAlgorithmPlan preselectionPlan = result.algorithmPlan;
            if (useUnifiedPreselection && !result.algorithmPlan.suppliesCoarsePairPreselection)
            {
                MatchPhotosOptions preselectionOptions = effectiveOptions;
                preselectionOptions.algorithmId = QStringLiteral("plamatch_hct");
                preselectionPlan = MatchPhotosAlgorithmSelector::select(preselectionOptions);
                if (!preselectionPlan.valid)
                {
                    const MatchPhotosStageReport failedReport = makeGenericPreselectionReport(
                        MatchPhotosStageStatus::Failed,
                        QStringLiteral("无法准备统一 PlaMatch-HCT 预筛选：%1").arg(preselectionPlan.validationError),
                        0);
                    appendStageAndStopOnFailure(&result, failedReport);
                    return result;
                }
            }

            if (useUnifiedPreselection)
            {
                reportMatchPhotosProgress(runtimeContext,
                                          QStringLiteral("generic_preselection"),
                                          QStringLiteral("PlaMatch 通用预选：匹配 coarse 特征"),
                                          0,
                                          1);
                PlaMatchHctPairPreselectionStats stats;
                QString preselectionError;
                if (!runtimeContext.featureCache ||
                    !PlaMatchHctPairPreselector::selectWithPositions(pairInput.images,
                                                                     *runtimeContext.featureCache,
                                                                     effectiveOptions,
                                                                     runtimeContext.referenceCameras,
                                                                     runtimeContext.referencePositions,
                                                                     preselectionPlan.executionBackend,
                                                                     effectiveOptions.cudaDevice,
                                                                     &result.pairSelection,
                                                                     &stats,
                                                                     runtimeContext.cancelFlag,
                                                                     &preselectionError))
                {
                    const MatchPhotosStageReport failedReport = makeGenericPreselectionReport(
                        MatchPhotosStageStatus::Failed,
                        preselectionError.isEmpty() ? QStringLiteral("PlaMatch-HCT 预选缺少任务特征缓存")
                                                    : preselectionError,
                        0);
                    result.stages.push_back(failedReport);
                    result.success = false;
                    result.errorMessage = failedReport.message;
                    return result;
                }

                const MatchPhotosStageReport genericReport = makeGenericPreselectionReport(
                    effectiveOptions.useGenericPreselection ? MatchPhotosStageStatus::Completed
                                                            : MatchPhotosStageStatus::Skipped,
                    stats.detail,
                    stats.genericSelectedCount);
                result.stages.push_back(genericReport);
                reportMatchPhotosProgress(
                    runtimeContext, QStringLiteral("generic_preselection"), genericReport.message, 1, 1);

                const MatchPhotosStageReport referenceReport = makeReferencePreselectionReport(
                    effectiveOptions.useReferencePreselection ? MatchPhotosStageStatus::Completed
                                                              : MatchPhotosStageStatus::Skipped,
                    effectiveOptions.useReferencePreselection ? stats.detail : QStringLiteral("未启用参考预选"),
                    stats.referenceSelectedCount);
                result.stages.push_back(referenceReport);
                reportMatchPhotosProgress(
                    runtimeContext, QStringLiteral("reference_preselection"), referenceReport.message, 1, 1);
            }
            else
            {
                QString errorMessage;
                PairSelectionPolicy directPairPolicy = pairPolicy;
                if (directPairPolicy.mode == PairSelectionMode::Auto && !automaticPreselectionRequested)
                {
                    // 两种自动预选都关闭表示用户明确要求全量两两匹配。这里不能继续使用
                    // Auto 的大项目序列回退，否则超过 exhaustiveMaxImages 后只会生成局部窗口。
                    directPairPolicy.mode = PairSelectionMode::Exhaustive;
                }
                result.pairSelection = PairSelector::select(pairInput, directPairPolicy, &errorMessage);
                if (!errorMessage.isEmpty())
                {
                    result.errorMessage = errorMessage;
                    result.success = false;
                    return result;
                }
            }

            reportMatchPhotosProgress(
                runtimeContext, QStringLiteral("pair_selection"), QStringLiteral("影像对规划：生成候选匹配对"), 0, 1);
            result.stages.push_back(makePairSelectionReport(result.pairSelection));
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("pair_selection"),
                                      QStringLiteral("影像对规划完成：候选 %1 对")
                                          .arg(static_cast<int>(result.pairSelection.candidates.size())),
                                      1,
                                      1);

            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("matching"),
                                      QStringLiteral("两两匹配：准备处理候选影像对"),
                                      0,
                                      std::max(1, static_cast<int>(result.pairSelection.candidates.size())));
            const MatchPhotosStageReport matchingReport = matchingStage.run(
                runtimeContext, _options, result.algorithmPlan, result.pairSelection, &result.matches);
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("matching"),
                                      matchingReport.message,
                                      matchingReport.itemCount,
                                      std::max(1, static_cast<int>(result.pairSelection.candidates.size())));
            if (appendStageAndStopOnFailure(&result, matchingReport))
            {
                return result;
            }
            int matchedPairCount = 0;
            std::int64_t rawMatchCount = 0;
            for (const MatchPhotosMatchRecord& record : result.matches)
            {
                matchedPairCount += record.matchCount > 0 ? 1 : 0;
                rawMatchCount += std::max(0, record.matchCount);
            }
            const MatchPhotosStageReport geometryReport =
                geometryVerifyStage.run(runtimeContext, _options, &result.matches);
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("geometry"),
                                      geometryReport.message,
                                      geometryReport.itemCount,
                                      std::max(1, static_cast<int>(result.matches.size())));
            if (appendStageAndStopOnFailure(&result, geometryReport))
            {
                clearTransientMatchPayloads(&result.matches);
                return result;
            }
            int geometryPassedPairCount = 0;
            std::int64_t geometryInlierCount = 0;
            for (const MatchPhotosMatchRecord& record : result.matches)
            {
                geometryPassedPairCount += record.passedGeometry ? 1 : 0;
                geometryInlierCount += std::max(0, record.geometricInlierCount);
            }
            const MatchPhotosStageReport guidedReport =
                guidedMatchStage.run(runtimeContext, _options, result.algorithmPlan, &result.matches);
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("guided_match"),
                                      guidedReport.message,
                                      guidedReport.itemCount,
                                      std::max(1, guidedReport.itemCount));
            if (appendStageAndStopOnFailure(&result, guidedReport))
            {
                clearTransientMatchPayloads(&result.matches);
                return result;
            }
            const MatchPhotosStageReport trackReport =
                trackBuildStage.run(runtimeContext, _options, &result.matches, &result);
            reportMatchPhotosProgress(runtimeContext,
                                      QStringLiteral("track_build"),
                                      trackReport.message,
                                      trackReport.itemCount,
                                      std::max(1, trackReport.itemCount));
            const bool stopAfterTrackBuild = appendStageAndStopOnFailure(&result, trackReport);
            result.matchingFunnel = summarizeMatchPhotosFunnel(result.pairSelection,
                                                               result.matches,
                                                               matchedPairCount,
                                                               rawMatchCount,
                                                               geometryPassedPairCount,
                                                               geometryInlierCount,
                                                               guidedReport.itemCount,
                                                               result.trackCount);
            result.trackSummary[QStringLiteral("matching_funnel")] = result.matchingFunnel.toJson();
            if (stopAfterTrackBuild)
            {
                clearTransientMatchPayloads(&result.matches);
                return result;
            }
            const MatchPhotosStageReport funnelReport = makeMatchingFunnelReport(result.matchingFunnel);
            result.stages.push_back(funnelReport);
            reportMatchPhotosProgress(runtimeContext,
                                      funnelReport.stageId,
                                      funnelReport.message,
                                      funnelReport.itemCount,
                                      std::max(1, funnelReport.itemCount));
            clearTransientMatchPayloads(&result.matches);
            result.success = true;
            return result;
        }

    } // namespace matchphotos
} // namespace xjw
