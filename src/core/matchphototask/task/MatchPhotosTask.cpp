#include "MatchPhotosTask.h"

#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "OverlapAnalyzer.h"
#include "TrackBuildStage.h"
#include "VocabularyOverlapRetriever.h"
#include "io/PathIO.h"

#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <memory>

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
    report.status = plan.valid ? MatchPhotosStageStatus::Completed
                               : MatchPhotosStageStatus::Failed;
    report.message = plan.valid
        ? QStringLiteral("%1：%2").arg(algorithmPlanSummary(plan), plan.reason)
        : plan.validationError;
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

QString normalizedCameraLookupKey(const QString &path)
{
    const QFileInfo info(path);
    return info.exists() ? info.absoluteFilePath() : path;
}

bool loadVocabularyFeatures(const MatchPhotosContext &context,
                            const MatchPhotosAlgorithmPlan &plan,
                            std::vector<VocabularyImageFeatures> *features,
                            QString *errorMessage)
{
    if (!features)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：词汇预选特征输出为空");
        }
        return false;
    }

    features->clear();
    if (!context.featureCache)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("通用预选缺少任务级特征缓存");
        }
        return false;
    }
    features->reserve(static_cast<std::size_t>(context.pairInput.images.size()));
    for (const QString &imagePath : context.pairInput.images)
    {
        const std::shared_ptr<const image_matching::FeatureSet> cached =
            context.featureCache->find(imagePath);
        if (!cached)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("通用预选缺少内存特征：%1").arg(imagePath);
            }
            return false;
        }

        if (!cached->isConsistent())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("通用预选内存特征无效：%1").arg(imagePath);
            }
            return false;
        }

        VocabularyImageFeatures item;
        item.imagePath = common::io::toUtf8Path(imagePath);
        item.keypoints = cached->keypoints;
        // cv::Mat 采用引用计数；缓存贯穿整个任务，因此这里无需复制数百 MiB 描述子。
        item.descriptors = cached->descriptors;
        features->push_back(std::move(item));
    }
    return true;
}

VocabularyOverlapConfig makeVocabularyConfig(const MatchPhotosOptions &options,
                                             const MatchPhotosAlgorithmPlan &plan)
{
    VocabularyOverlapConfig config;
    config.topK = std::max(8, options.pairPolicy.sequenceWindow * 2);
    config.minPairsPerImage = std::max(4, options.pairPolicy.sequenceWindow);
    config.minSimilarity = 0.03;
    config.mutualTopK = true;
    config.keepOneWayTopK = true;
    config.connectComponents = true;
    config.useSequenceFallback = true;
    config.sequenceWindow = std::max(1, options.pairPolicy.sequenceWindow);
    config.closeSequenceLoop = true;
    config.geometryCheck = false;
    config.useCuda = options.device == ComputeDevice::Cuda ||
        (options.device == ComputeDevice::Auto && plan.preferCuda);
    return config;
}

MatchPhotosStageReport makeVocabularyPreselectionReport(MatchPhotosStageStatus status,
                                                        const QString &message,
                                                        int itemCount)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("generic_preselection");
    report.displayName = QStringLiteral("通用预选");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

bool buildVocabularyPreselection(const MatchPhotosContext &context,
                                 const MatchPhotosOptions &options,
                                 const MatchPhotosAlgorithmPlan &plan,
                                 VocabularyOverlapResult *result,
                                 MatchPhotosStageReport *report)
{
    if (!options.useGenericPreselection || options.planOnly)
    {
        if (report)
        {
            *report = makeVocabularyPreselectionReport(
                MatchPhotosStageStatus::Skipped,
                options.planOnly ? QStringLiteral("plan-only 模式，跳过通用预选")
                                 : QStringLiteral("未启用通用预选"),
                0);
        }
        return true;
    }

    std::vector<VocabularyImageFeatures> features;
    QString loadError;
    if (!loadVocabularyFeatures(context, plan, &features, &loadError))
    {
        if (report)
        {
            *report = makeVocabularyPreselectionReport(
                MatchPhotosStageStatus::Failed,
                loadError,
                0);
        }
        return false;
    }

    VocabularyOverlapConfig config = makeVocabularyConfig(options, plan);
    config.progressCallback = [&context, cancelFlag = context.cancelFlag](const std::string &stage, int percent)
    {
        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
        {
            return false;
        }

        reportMatchPhotosProgress(context,
                                  QStringLiteral("generic_preselection"),
                                  QStringLiteral("Generic 预选: %1").arg(QString::fromStdString(stage)),
                                  percent,
                                  100);
        return true;
    };

    std::string coreError;
    if (!VocabularyOverlapRetriever::retrieve(features, config, result, &coreError))
    {
        if (report)
        {
            *report = makeVocabularyPreselectionReport(
                MatchPhotosStageStatus::Failed,
                QString::fromStdString(coreError),
                0);
        }
        return false;
    }

    int acceptedCount = 0;
    if (result)
    {
        if (result->acceptedPairs.empty())
        {
            for (const VocabularyOverlapPairResult &candidate : result->candidates)
            {
                if (candidate.accepted)
                {
                    ++acceptedCount;
                }
            }
        }
        else
        {
            acceptedCount = static_cast<int>(result->acceptedPairs.size());
        }
    }
    if (report)
    {
        const QString detail = result && !result->detail.empty()
            ? QString::fromStdString(result->detail)
            : QStringLiteral("词汇树预选完成");
        *report = makeVocabularyPreselectionReport(
            MatchPhotosStageStatus::Completed,
            QStringLiteral("%1，接受 %2 对").arg(detail).arg(acceptedCount),
            acceptedCount);
    }
    return true;
}

MatchPhotosStageReport makeReferencePreselectionReport(MatchPhotosStageStatus status,
                                                       const QString &message,
                                                       int itemCount)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("reference_preselection");
    report.displayName = QStringLiteral("参考预选");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

bool cameraForImage(const MatchPhotosContext &context,
                    const QString &imagePath,
                    Camera *camera)
{
    if (!camera)
    {
        return false;
    }

    const QString directKey = imagePath;
    if (context.referenceCameras.contains(directKey))
    {
        *camera = context.referenceCameras.value(directKey);
        return camera->isValid();
    }

    const QString normalizedKey = normalizedCameraLookupKey(imagePath);
    if (context.referenceCameras.contains(normalizedKey))
    {
        *camera = context.referenceCameras.value(normalizedKey);
        return camera->isValid();
    }

    return false;
}

bool buildReferencePreselection(const MatchPhotosContext &context,
                                const MatchPhotosOptions &options,
                                OverlapAnalysisResult *result,
                                MatchPhotosStageReport *report)
{
    if (!options.useReferencePreselection || options.planOnly)
    {
        if (report)
        {
            *report = makeReferencePreselectionReport(
                MatchPhotosStageStatus::Skipped,
                options.planOnly ? QStringLiteral("plan-only 模式，跳过参考预选")
                                 : QStringLiteral("未启用参考预选"),
                0);
        }
        return true;
    }

    std::vector<OverlapImageInput> inputs;
    inputs.reserve(static_cast<std::size_t>(context.pairInput.images.size()));
    for (const QString &imagePath : context.pairInput.images)
    {
        Camera camera;
        if (!cameraForImage(context, imagePath, &camera))
        {
            if (report)
            {
                *report = makeReferencePreselectionReport(
                    MatchPhotosStageStatus::Failed,
                    QStringLiteral("参考预选需要每张影像都有可用相机：%1").arg(imagePath),
                    0);
            }
            return false;
        }

        cv::Mat image = common::io::readImage(imagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            if (report)
            {
                *report = makeReferencePreselectionReport(
                    MatchPhotosStageStatus::Failed,
                    QStringLiteral("参考预选无法读取影像尺寸：%1").arg(imagePath),
                    0);
            }
            return false;
        }

        OverlapImageInput input;
        input.imagePath = common::io::toUtf8Path(imagePath);
        input.camera = camera;
        input.width = image.cols;
        input.height = image.rows;
        inputs.push_back(std::move(input));
    }

    OverlapAnalysisOptions overlapOptions;
    overlapOptions.groundModel = OverlapGroundModel::ReferenceSphere;
    overlapOptions.neighborFactor = 2.0;
    overlapOptions.referenceSphere.body = ReferenceBody::Earth;
    overlapOptions.referenceSphere.radiusMeters = referenceBodyRadiusMeters(ReferenceBody::Earth);
    overlapOptions.referenceSphere.autoLocalTangentHeight = true;
    overlapOptions.referenceSphere.centerMode = ReferenceSphereCenterMode::Auto;

    std::string coreError;
    if (!OverlapAnalyzer::analyze(inputs, overlapOptions, result, &coreError))
    {
        if (report)
        {
            *report = makeReferencePreselectionReport(
                MatchPhotosStageStatus::Failed,
                QString::fromStdString(coreError),
                0);
        }
        return false;
    }

    if (report)
    {
        const int pairCount = result ? static_cast<int>(result->pairs.size()) : 0;
        const QString detail = result && !result->detail.empty()
            ? QString::fromStdString(result->detail)
            : QStringLiteral("相机参考预选完成");
        *report = makeReferencePreselectionReport(
            MatchPhotosStageStatus::Completed,
            QStringLiteral("%1，候选 %2 对").arg(detail).arg(pairCount),
            pairCount);
    }
    return true;
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

void clearTransientMatchPayloads(std::vector<MatchPhotosMatchRecord> *matchRecords)
{
    if (!matchRecords)
    {
        return;
    }

    for (MatchPhotosMatchRecord &record : *matchRecords)
    {
        // PairMatchData 已经提交到每影像 `.pimatch` 分片，返回 GUI 后只需保留
        // 路径和统计。及时释放坐标数组可显著降低大型项目的峰值内存。
        record.pairData.reset();
    }
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
    MatchPhotosContext runtimeContext = context;
    if (!runtimeContext.featureCache)
    {
        runtimeContext.featureCache = std::make_shared<MatchPhotosFeatureCache>();
    }

    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("algorithm_selection"),
                              QStringLiteral("选择 SIFT + LightGlue 连接点流程"),
                              0,
                              1);
    result.algorithmPlan = MatchPhotosAlgorithmSelector::select(_options);
    result.stages.push_back(makeAlgorithmSelectionReport(result.algorithmPlan));
    if (!result.algorithmPlan.valid)
    {
        result.errorMessage = result.algorithmPlan.validationError;
        return result;
    }
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("algorithm_selection"),
                              QStringLiteral("连接点算法已确定: SIFT + LightGlue"),
                              1,
                              1);

    // 这些阶段对象当前刻意保持短生命周期、无状态。
    // 后续接入真实运行器后，取消和进度状态应放在上下文或运行器中维护。
    const FeatureStage featureStage;
    const MatchingStage matchingStage;
    const GeometryVerifyStage geometryVerifyStage;
    const TrackBuildStage trackBuildStage;
    const GuidedMatchStage guidedMatchStage;

    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("feature"),
                              QStringLiteral("SIFT 特征提取：准备处理影像"),
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

    PairSelectionInput pairInput = runtimeContext.pairInput;
    PairSelectionPolicy pairPolicy = _options.pairPolicy;
    const bool manualOnly = pairPolicy.mode == PairSelectionMode::ManualOnly;
    MatchPhotosOptions effectiveOptions = _options;
    if (manualOnly)
    {
        effectiveOptions.useGenericPreselection = false;
        effectiveOptions.useReferencePreselection = false;
    }
    pairPolicy.includeVocabularyOverlap = effectiveOptions.useGenericPreselection;
    pairPolicy.includeCameraOverlap = effectiveOptions.useReferencePreselection;
    if (!effectiveOptions.useGenericPreselection && !effectiveOptions.useReferencePreselection &&
        pairPolicy.mode == PairSelectionMode::Auto)
    {
        pairPolicy.mode = PairSelectionMode::Exhaustive;
    }

    VocabularyOverlapResult vocabularyOverlap;
    MatchPhotosStageReport vocabularyReport;
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("generic_preselection"),
                              QStringLiteral("通用预选：构建候选影像对"),
                              0,
                              1);
    if (!buildVocabularyPreselection(runtimeContext,
                                     effectiveOptions,
                                     result.algorithmPlan,
                                     &vocabularyOverlap,
                                     &vocabularyReport))
    {
        result.stages.push_back(vocabularyReport);
        result.success = false;
        result.errorMessage = vocabularyReport.message;
        return result;
    }
    result.stages.push_back(vocabularyReport);
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("generic_preselection"),
                              vocabularyReport.message,
                              1,
                              1);
    if (effectiveOptions.useGenericPreselection)
    {
        pairInput.vocabularyOverlapResult = &vocabularyOverlap;
    }

    OverlapAnalysisResult cameraOverlap;
    MatchPhotosStageReport referenceReport;
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("reference_preselection"),
                              QStringLiteral("参考预选：检查相机参考"),
                              0,
                              1);
    if (!buildReferencePreselection(runtimeContext, effectiveOptions, &cameraOverlap, &referenceReport))
    {
        result.stages.push_back(referenceReport);
        result.success = false;
        result.errorMessage = referenceReport.message;
        return result;
    }
    result.stages.push_back(referenceReport);
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("reference_preselection"),
                              referenceReport.message,
                              1,
                              1);
    if (effectiveOptions.useReferencePreselection)
    {
        pairInput.cameraOverlapResult = &cameraOverlap;
    }

    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("pair_selection"),
                              QStringLiteral("影像对规划：生成候选匹配对"),
                              0,
                              1);
    QString errorMessage;
    result.pairSelection = PairSelector::select(pairInput, pairPolicy, &errorMessage);
    if (!errorMessage.isEmpty())
    {
        result.errorMessage = errorMessage;
        result.success = false;
        return result;
    }

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
    const MatchPhotosStageReport matchingReport =
        matchingStage.run(runtimeContext, _options, result.algorithmPlan, result.pairSelection, &result.matches);
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("matching"),
                              matchingReport.message,
                              matchingReport.itemCount,
                              std::max(1, static_cast<int>(result.pairSelection.candidates.size())));
    if (appendStageAndStopOnFailure(&result, matchingReport))
    {
        return result;
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
    const MatchPhotosStageReport trackReport =
        trackBuildStage.run(runtimeContext, _options, &result.matches, &result);
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("track_build"),
                              trackReport.message,
                              trackReport.itemCount,
                              std::max(1, trackReport.itemCount));
    const bool stopAfterTrackBuild = appendStageAndStopOnFailure(&result, trackReport);
    clearTransientMatchPayloads(&result.matches);
    if (stopAfterTrackBuild)
    {
        return result;
    }
    result.stages.push_back(guidedMatchStage.run(runtimeContext, _options));

    result.success = true;
    return result;
}

} // namespace matchphotos
} // namespace xjw
