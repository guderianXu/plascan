#include "MatchPhotosTask.h"

#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "OverlapAnalyzer.h"
#include "SparseSceneOverlapAnalyzer.h"
#include "TrackBuildStage.h"
#include "VocabularyOverlapRetriever.h"
#include "io/PathIO.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftFeatureExtractor.h"

#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>

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

MatchPhotosFunnelDiagnostics summarizeMatchPhotosFunnel(
    const PairSelectionResult &pairSelection,
    const std::vector<MatchPhotosMatchRecord> &finalMatches,
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

    for (const MatchPhotosMatchRecord &record : finalMatches)
    {
        if (record.passedGeometry)
        {
            ++diagnostics.finalPassedPairCount;
            diagnostics.finalGeometryInlierCount += std::max(0, record.geometricInlierCount);
        }
    }

    const auto boundedRatio = [](std::int64_t numerator, std::int64_t denominator)
    {
        const double value = denominator > 0
            ? static_cast<double>(numerator) / static_cast<double>(denominator)
            : 0.0;
        return std::clamp(value, 0.0, 1.0);
    };
    diagnostics.pairSelectionRatio = boundedRatio(
        diagnostics.selectedPairCount, diagnostics.allPairCount);
    diagnostics.matchingYieldRatio = boundedRatio(
        diagnostics.matchedPairCount, diagnostics.selectedPairCount);
    diagnostics.geometryPairRetentionRatio = boundedRatio(
        diagnostics.geometryPassedPairCount, diagnostics.matchedPairCount);
    diagnostics.geometryInlierRatio = boundedRatio(
        diagnostics.geometryInlierCount, diagnostics.rawMatchCount);
    diagnostics.guidedGainRatio = diagnostics.geometryInlierCount > 0
        ? static_cast<double>(diagnostics.guidedAddedInlierCount) /
              static_cast<double>(diagnostics.geometryInlierCount)
        : 0.0;
    return diagnostics;
}

namespace
{

MatchPhotosStageReport makeAlgorithmSelectionReport(const MatchPhotosAlgorithmPlan &plan)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("algorithm_selection");
    report.displayName = QStringLiteral("算法选择");
    report.status = plan.valid ? MatchPhotosStageStatus::Completed
                               : MatchPhotosStageStatus::Failed;
    if (plan.valid)
    {
        report.message = QStringLiteral("%1：%2")
                             .arg(algorithmPlanSummary(plan), plan.reason);
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

MatchPhotosStageReport makeMatchingFunnelReport(
    const MatchPhotosFunnelDiagnostics &diagnostics)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("matching_funnel");
    report.displayName = QStringLiteral("匹配漏斗诊断");
    report.status = MatchPhotosStageStatus::Completed;
    report.itemCount = diagnostics.trackCount;
    report.message =
        QStringLiteral("像对 %1/%2（%3%） → 有初始匹配 %4 对/%5 个 → "
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
        if (cached->descriptors.type() == CV_32F)
        {
            // cv::Mat 采用引用计数；缓存贯穿整个任务，因此浮点描述子无需复制。
            item.descriptors = cached->descriptors;
        }
        else if (cached->descriptors.type() == CV_8U)
        {
            // 词汇树当前以欧氏距离量化 float centroid。二进制通道在这里只做
            // 低成本候选预选，按位字节转换为 [0,1] float；最终匹配仍使用 Hamming。
            cached->descriptors.convertTo(item.descriptors, CV_32F, 1.0 / 255.0);
        }
        else
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("通用预选不支持描述子类型：%1")
                                    .arg(cached->descriptors.type());
            }
            return false;
        }
        features->push_back(std::move(item));
    }
    return true;
}

VocabularyOverlapConfig makeVocabularyConfig(const MatchPhotosOptions& options, const MatchPhotosAlgorithmPlan& plan)
{
    VocabularyOverlapConfig config;
    const bool expandedCoverage =
        options.profile == MatchPhotosProfile::HighAccuracy || options.profile == MatchPhotosProfile::DifficultTexture;
    const int topKMultiplier = expandedCoverage ? 3 : 2;
    config.topK = std::max(8, options.pairPolicy.sequenceWindow * topKMultiplier);
    config.minPairsPerImage = std::max(4, options.pairPolicy.sequenceWindow + (expandedCoverage ? 4 : 0));
    config.minSimilarity = 0.03;
    config.mutualTopK = true;
    config.keepOneWayTopK = true;
    config.cycleClosureMaxPairsPerImage = expandedCoverage ? std::max(2, options.pairPolicy.sequenceWindow / 2) : 0;
    config.connectComponents = true;
    config.useSequenceFallback = true;
    config.sequenceWindow = std::max(1, options.pairPolicy.sequenceWindow);
    // 通用预选的序列兜底必须沿用调用方对线性/闭环序列的声明。无条件闭环会
    // 把航带末端与开头加入词汇候选，形成外观自洽但空间错误的首尾分支。
    config.closeSequenceLoop = options.pairPolicy.closeSequenceLoop;
    config.geometryCheck = false;
    config.useCuda =
        options.device == ComputeDevice::Cuda || (options.device == ComputeDevice::Auto && plan.preferCuda);
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
                    FramePinholeCamera *camera)
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
                                MatchPhotosStageReport *report,
                                bool *usable)
{
    if (usable)
    {
        *usable = false;
    }
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
        FramePinholeCamera camera;
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

    if (options.referencePreselectionGeometry ==
        ReferencePreselectionGeometry::SparseScene)
    {
        SparseSceneOverlapStats stats;
        QString sparseError;
        const bool analyzed = !context.referenceSparsePointsPath.trimmed().isEmpty() &&
            QFileInfo::exists(context.referenceSparsePointsPath) &&
            SparseSceneOverlapAnalyzer::analyzeFile(
                context.referenceSparsePointsPath,
                inputs,
                SparseSceneOverlapOptions{},
                result,
                &stats,
                &sparseError);
        if (!analyzed)
        {
            if (report)
            {
                const QString reason = sparseError.isEmpty()
                    ? QStringLiteral("没有可用的已有 SfM 稀疏点场景")
                    : sparseError;
                *report = makeReferencePreselectionReport(
                    MatchPhotosStageStatus::Skipped,
                    QStringLiteral("已有 SfM 查漏不可用：%1；将使用通用/序列安全回退")
                        .arg(reason),
                    0);
            }
            return true;
        }

        if (usable)
        {
            *usable = true;
        }
        if (report)
        {
            *report = makeReferencePreselectionReport(
                MatchPhotosStageStatus::Completed,
                stats.detail,
                result ? static_cast<int>(result->pairs.size()) : 0);
        }
        return true;
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

    if (usable)
    {
        *usable = true;
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

    result.algorithmPlan = MatchPhotosAlgorithmSelector::select(_options);
    if (result.algorithmPlan.valid &&
        result.algorithmPlan.algorithmId ==
            QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        try
        {
            const image_matching::SiftComputeBackend backend =
                image_matching::SiftFeatureExtractor::resolveBackend(
                    requestedSiftBackend(_options.device), _options.cudaDevice);
            result.algorithmPlan = MatchPhotosAlgorithmSelector::resolveExecutionBackend(
                _options, std::move(result.algorithmPlan), backend, _options.cudaDevice);
        }
        catch (const std::exception &error)
        {
            result.algorithmPlan.valid = false;
            result.algorithmPlan.validationError = QString::fromUtf8(error.what());
        }
    }
    const QString algorithmName = result.algorithmPlan.displayName.isEmpty()
        ? result.algorithmPlan.algorithmId
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
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("algorithm_selection"),
                              result.algorithmPlan.backendReason.isEmpty()
                                  ? QStringLiteral("连接点算法已确定: %1")
                                        .arg(algorithmName)
                                  : QStringLiteral("连接点算法已确定: %1；%2")
                                        .arg(algorithmName,
                                             result.algorithmPlan.backendReason),
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
    bool referencePreselectionUsable = false;
    reportMatchPhotosProgress(runtimeContext,
                              QStringLiteral("reference_preselection"),
                              QStringLiteral("参考预选：检查相机参考"),
                              0,
                              1);
    if (!buildReferencePreselection(runtimeContext,
                                    effectiveOptions,
                                    &cameraOverlap,
                                    &referenceReport,
                                    &referencePreselectionUsable))
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
    const bool requestedReferenceUnavailable =
        effectiveOptions.useReferencePreselection && !referencePreselectionUsable;
    if (effectiveOptions.useReferencePreselection && referencePreselectionUsable)
    {
        pairInput.cameraOverlapResult = &cameraOverlap;
    }
    else
    {
        effectiveOptions.useReferencePreselection = false;
    }
    pairPolicy.includeCameraOverlap = effectiveOptions.useReferencePreselection;
    if (!effectiveOptions.useGenericPreselection && !effectiveOptions.useReferencePreselection &&
        pairPolicy.mode == PairSelectionMode::Auto && !requestedReferenceUnavailable)
    {
        pairPolicy.mode = PairSelectionMode::Exhaustive;
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
    int matchedPairCount = 0;
    std::int64_t rawMatchCount = 0;
    for (const MatchPhotosMatchRecord &record : result.matches)
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
    for (const MatchPhotosMatchRecord &record : result.matches)
    {
        geometryPassedPairCount += record.passedGeometry ? 1 : 0;
        geometryInlierCount += std::max(0, record.geometricInlierCount);
    }
    const MatchPhotosStageReport guidedReport = guidedMatchStage.run(
        runtimeContext, _options, result.algorithmPlan, &result.matches);
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
    result.matchingFunnel = summarizeMatchPhotosFunnel(
        result.pairSelection,
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
