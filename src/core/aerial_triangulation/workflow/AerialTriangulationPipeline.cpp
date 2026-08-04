/**
 * @file AerialTriangulationPipeline.cpp
 * @brief 连接点图到正式 SfM/BA 稀疏结果的候选搜索与提交实现。
 *
 * 当输入没有完整可信内参时，管线对多个焦距尺度运行相互隔离的 SfM 候选。
 * 候选按注册覆盖、摄影测量网络质量、闭环连续性和重投影误差排序；只有胜出模型
 * 可进入 AerialTriangulationResultWriter，避免失败候选污染工程相机和结果文件。
 */

#include "workflow/AerialTriangulationPipeline.h"

#include "ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "reporting/AerialTriangulationResultWriter.h"
#include "reporting/QualityReportWriter.h"
#include "reconstruction/CameraIntrinsicPriorSanitizer.h"
#include "reconstruction/SfmReconstruction.h"
#include "search/AdaptiveFocalSearch.h"
#include "search/SfmSearchPolicy.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <memory>
#include <thread>
#include <vector>

namespace xjw::aerial_triangulation
{
namespace
{

/**
 * @brief 判断每张影像是否都有可直接使用的可信内参。
 *
 * 独立相机文件优先；工程内参必须带可信来源且能反序列化为有效 Camera。
 * 由上一次 SfM 写回的 intrinsic_source=sfm_estimated 不作为新一轮初始化真值。
 */
bool hasCompleteCameraIntrinsicPrior(const PreparedAerialTriangulationInput &input)
{
    if (input.cameraPaths.size() == input.images.size() && !input.images.isEmpty() &&
        std::all_of(input.cameraPaths.cbegin(), input.cameraPaths.cend(), [](const QString &path)
        {
            return !path.trimmed().isEmpty() && QFileInfo::exists(path);
        }))
    {
        return true;
    }
    if (!input.useProjectCameraIntrinsics || input.projectMeta.isEmpty())
    {
        return false;
    }

    const QMap<QString, QJsonObject> imageMeta =
        xjw::common::project::projectImageMetaByPath(input.projectMeta, true);
    for (const QString &imagePath : input.images)
    {
        const auto metadata = imageMeta.constFind(
            xjw::common::project::normalizePath(imagePath));
        if (metadata == imageMeta.cend())
        {
            return false;
        }
        const QJsonObject cameraObject =
            metadata.value().value(QStringLiteral("camera")).toObject();
        Camera camera;
        if (!isTrustedProjectCameraIntrinsic(cameraObject) ||
            !xjw::common::project::cameraFromJson(cameraObject, &camera) ||
            !camera.isValid())
        {
            return false;
        }
    }
    return !input.images.isEmpty();
}

/// 从一次完整执行提取仅供焦距搜索排序的轻量指标。
AdaptiveFocalCandidate candidateFromExecution(
    double focalScale,
    const SfmAttemptExecutionResult &execution)
{
    return {
        focalScale,
        execution.result.success,
        execution.result.numRegisteredImages,
        execution.result.numPoints3D,
        execution.result.meanReprojError,
    };
}

/// 对候选内存模型即时计算网络质量，不写任何正式文件。
QJsonObject sparseQualityFromExecution(
    const PreparedAerialTriangulationInput &input,
    const SfmAttemptExecutionResult &execution)
{
    if (!execution.reconstruction)
    {
        return {};
    }
    const SparseQualityReport report = QualityReportWriter::build(
        input, *execution.reconstruction, execution.result);
    return report.diagnostics.value(QStringLiteral("sparse_quality")).toObject();
}

/**
 * @brief 将候选执行结果转换为确定性排序摘要。
 *
 * 闭环几何仅在明确的照片序列模式、全部影像已注册且影像数足够时计算。
 * 相邻中心距离的 MAD/最大比值用于发现局部跳跃，不用于强制圆形或等距轨迹。
 */
SfmCandidateSummary summaryFromExecution(
    int candidateIndex,
    double focalScale,
    const SfmAttemptExecutionResult &execution,
    bool sequenceLoopClosure)
{
    const QJsonArray pair = execution.result.sfmDiagnostics
        .value(QStringLiteral("selected_initial_pair")).toArray();
    const ImageId imageA = pair.size() >= 2
        ? static_cast<ImageId>(pair.at(0).toInt(-1))
        : kInvalidImageId;
    const ImageId imageB = pair.size() >= 2
        ? static_cast<ImageId>(pair.at(1).toInt(-1))
        : kInvalidImageId;
    SfmCandidateSummary summary{
        candidateIndex,
        focalScale,
        imageA,
        imageB,
        execution.result.numRegisteredImages,
        execution.result.numPoints3D,
        execution.result.meanReprojError,
        execution.result.success,
    };

    const QJsonObject sparseQuality = execution.result.sfmDiagnostics
        .value(QStringLiteral("sparse_quality")).toObject();
    const int pointCount = sparseQuality.value(QStringLiteral("point_count")).toInt();
    const int twoViewTrackCount = sparseQuality
        .value(QStringLiteral("two_view_track_count")).toInt();
    const QJsonObject angleSummary = sparseQuality
        .value(QStringLiteral("triangulation_angle")).toObject();
    summary.hasNetworkQuality = pointCount > 0 &&
        angleSummary.value(QStringLiteral("count")).toInt() > 0;
    summary.medianTriangulationAngleDeg = angleSummary
        .value(QStringLiteral("p50")).toDouble();
    summary.twoViewTrackRatio = pointCount > 0
        ? static_cast<double>(twoViewTrackCount) / static_cast<double>(pointCount)
        : 1.0;
    summary.observationGridCoverage = sparseQuality
        .value(QStringLiteral("observation_grid_coverage")).toObject()
        .value(QStringLiteral("mean")).toDouble();
    if (sequenceLoopClosure && execution.reconstruction)
    {
        std::vector<double> adjacent_distances;
        std::vector<ImageId> image_ids =
            execution.reconstruction->registeredImageIds();
        std::sort(image_ids.begin(), image_ids.end());
        if (image_ids.size() >= 6 &&
            image_ids.size() == execution.reconstruction->numImages())
        {
            adjacent_distances.reserve(image_ids.size());
            bool valid = true;
            for (std::size_t index = 0; index < image_ids.size(); ++index)
            {
                const ImageId current = image_ids[index];
                const ImageId next = image_ids[(index + 1) % image_ids.size()];
                if (!execution.reconstruction->hasCamera(current) ||
                    !execution.reconstruction->hasCamera(next))
                {
                    valid = false;
                    break;
                }
                const auto current_center =
                    execution.reconstruction->camera(current).cameraCenter();
                const auto next_center =
                    execution.reconstruction->camera(next).cameraCenter();
                const double dx = current_center[0] - next_center[0];
                const double dy = current_center[1] - next_center[1];
                const double dz = current_center[2] - next_center[2];
                const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (!std::isfinite(distance) || distance <= 1.0e-12)
                {
                    valid = false;
                    break;
                }
                adjacent_distances.push_back(distance);
            }
            if (valid)
            {
                std::vector<double> ordered = adjacent_distances;
                std::sort(ordered.begin(), ordered.end());
                const double median = 0.5 *
                    (ordered[(ordered.size() - 1) / 2] +
                     ordered[ordered.size() / 2]);
                std::vector<double> deviations;
                deviations.reserve(adjacent_distances.size());
                for (const double distance : adjacent_distances)
                {
                    deviations.push_back(std::fabs(distance - median));
                }
                std::sort(deviations.begin(), deviations.end());
                const double median_deviation = 0.5 *
                    (deviations[(deviations.size() - 1) / 2] +
                     deviations[deviations.size() / 2]);
                summary.hasClosedSequenceGeometry = median > 1.0e-12;
                summary.sequenceAdjacentDistanceMedian = median;
                summary.sequenceAdjacentDistanceMaximumRatio =
                    summary.hasClosedSequenceGeometry
                    ? ordered.back() / median : 0.0;
                summary.sequenceAdjacentDistanceMadRatio =
                    summary.hasClosedSequenceGeometry
                    ? median_deviation / median : 0.0;
            }
        }
    }
    return summary;
}

/// 将候选核心指标写入诊断 JSON，便于复现最终选择。
QJsonObject candidateToJson(const AdaptiveFocalCandidate &candidate,
                            const SfmCandidateSummary *summary)
{
    QJsonObject object{
        {QStringLiteral("focal_scale"), candidate.focalScale},
        {QStringLiteral("success"), candidate.success},
        {QStringLiteral("registered_images"), candidate.registeredImages},
        {QStringLiteral("points3d"), candidate.points3D},
        {QStringLiteral("mean_reprojection_error"), candidate.meanReprojectionError},
    };
    if (summary && summary->hasNetworkQuality)
    {
        object.insert(QStringLiteral("median_triangulation_angle_deg"),
                      summary->medianTriangulationAngleDeg);
        object.insert(QStringLiteral("two_view_track_ratio"),
                      summary->twoViewTrackRatio);
        object.insert(QStringLiteral("observation_grid_coverage"),
                      summary->observationGridCoverage);
    }
    if (summary && summary->hasClosedSequenceGeometry)
    {
        object.insert(QStringLiteral("sequence_adjacent_distance_median"),
                      summary->sequenceAdjacentDistanceMedian);
        object.insert(QStringLiteral("sequence_adjacent_distance_maximum_ratio"),
                      summary->sequenceAdjacentDistanceMaximumRatio);
        object.insert(QStringLiteral("sequence_adjacent_distance_mad_ratio"),
                      summary->sequenceAdjacentDistanceMadRatio);
    }
    return object;
}

} // namespace

AerialTriangulationPipeline::AerialTriangulationPipeline(
    AttemptRunner attemptRunner,
    ResultWriter resultWriter)
    : _attemptRunner(attemptRunner ? std::move(attemptRunner)
                                   : AttemptRunner([](const PreparedAerialTriangulationInput &input)
                                     {
                                         return SfmAttemptRunner().run(input);
                                     })),
      _resultWriter(resultWriter ? std::move(resultWriter)
                                 : ResultWriter([](const PreparedAerialTriangulationInput &input,
                                                   SfmAttemptExecutionResult *execution,
                                                   QString *errorMessage)
                                   {
                                       return AerialTriangulationResultWriter().write(
                                           input, execution, errorMessage);
                                   }))
{
}

AerialTriangulationReconstructionResult AerialTriangulationPipeline::run(
    const PreparedAerialTriangulationInput &input) const
{
    QElapsedTimer timer;
    timer.start();

    const auto originalProgress = input.progressFn;
    // 无完整相机先验时，初始焦距必须先从影像几何中估计；这与最终 BA 是否释放内参是两个独立职责。
    const bool needsFocalInitializationSearch =
        input.images.size() >= 3 && !hasCompleteCameraIntrinsicPrior(input);
    const int focalProbeLimit = needsFocalInitializationSearch
        ? focalProbeRegistrationLimit(input.images.size())
        : 0;

    PreparedAerialTriangulationInput attemptInput = input;
    attemptInput.threads = resolveSfmThreadBudget(input.threads);
    // 焦距粗搜索只比较固定内参下的几何结果。若把 BA 内参拟合混入候选，
    // 同一焦距尺度会因局部极值而产生不可比较的注册结果。
    if (needsFocalInitializationSearch)
    {
        attemptInput.adaptiveCameraModelFitting = false;
        attemptInput.coarseFocalEvaluation = true;
        attemptInput.maxRegisteredImages = focalProbeLimit;
    }
    attemptInput.progressFn = [originalProgress, needsFocalInitializationSearch](
                                  const QString &stage, int percent)
    {
        if (originalProgress)
        {
            const int span = needsFocalInitializationSearch ? 40 : 85;
            originalProgress(stage, 5 + std::clamp(percent, 0, 100) * span / 100);
        }
    };

    if (originalProgress)
    {
        originalProgress(QStringLiteral("读取连接点并构建观测图..."), 0);
    }

    // 先执行用户给定/默认焦距候选。若已有完整内参，这也是唯一一次试算。
    SfmAttemptExecutionResult execution = _attemptRunner(attemptInput);
    QVector<AdaptiveFocalCandidate> focalCandidates;
    std::vector<SfmCandidateSummary> candidateSummaries;
    QVector<SfmAttemptExecutionResult> candidateExecutions;
    double selectedFocalScale = input.estimatedFocalScale;
    bool adaptiveRefinementAccepted = false;

    if (needsFocalInitializationSearch)
    {
        // 每个候选保留独立重建对象和同一口径质量摘要，不能交叉复用相机状态。
        const auto appendCandidate = [&](double focalScale,
                                         SfmAttemptExecutionResult candidateExecution)
        {
            const int candidateIndex = focalCandidates.size();
            const QJsonObject sparseQuality =
                sparseQualityFromExecution(input, candidateExecution);
            if (!sparseQuality.isEmpty())
            {
                candidateExecution.result.sfmDiagnostics.insert(
                    QStringLiteral("sparse_quality"), sparseQuality);
            }
            focalCandidates.append(candidateFromExecution(focalScale, candidateExecution));
            candidateSummaries.push_back(summaryFromExecution(
                candidateIndex,
                focalScale,
                candidateExecution,
                input.sequenceLoopClosure));
            candidateExecutions.append(std::move(candidateExecution));
        };
        appendCandidate(input.estimatedFocalScale, std::move(execution));
        execution = {};

        std::vector<double> coarseScales;
        for (double focalScale : adaptiveFocalScaleCandidates())
        {
            if (std::abs(focalScale - input.estimatedFocalScale) > 1.0e-9)
            {
                coarseScales.push_back(focalScale);
            }
        }

        // 粗搜索按总线程预算分组并行。每个 worker 内部仍给 SfM/BA 保留多个线程，
        // 避免候选数较多时创建过量线程并争抢 GPU/内存。
        const int coarseCount = static_cast<int>(coarseScales.size());
        const SfmWorkerBudget workerBudget = allocateWorkers(coarseCount, input.threads);
        std::vector<SfmAttemptExecutionResult> coarseExecutions(
            static_cast<std::size_t>(coarseCount));
        std::vector<std::shared_ptr<std::atomic<int>>> coarseProgress;
        std::vector<std::shared_ptr<std::atomic<bool>>> coarseStarted;
        coarseProgress.reserve(static_cast<std::size_t>(coarseCount));
        coarseStarted.reserve(static_cast<std::size_t>(coarseCount));
        for (int index = 0; index < coarseCount; ++index)
        {
            coarseProgress.push_back(std::make_shared<std::atomic<int>>(0));
            coarseStarted.push_back(std::make_shared<std::atomic<bool>>(false));
        }

        std::atomic<int> nextCoarseIndex{0};
        std::atomic<int> completedCoarseCount{0};
        std::vector<std::future<void>> workers;
        workers.reserve(static_cast<std::size_t>(workerBudget.workerCount));
        for (int workerIndex = 0; workerIndex < workerBudget.workerCount; ++workerIndex)
        {
            workers.push_back(std::async(std::launch::async, [&, workerIndex]()
            {
                (void)workerIndex;
                while (true)
                {
                    const int scaleIndex = nextCoarseIndex.fetch_add(1);
                    if (scaleIndex >= coarseCount)
                    {
                        break;
                    }

                    PreparedAerialTriangulationInput coarseInput = input;
                    coarseInput.estimatedFocalScale =
                        coarseScales[static_cast<std::size_t>(scaleIndex)];
                    coarseInput.adaptiveCameraModelFitting = false;
                    coarseInput.coarseFocalEvaluation = true;
                    coarseInput.maxRegisteredImages = focalProbeLimit;
                    coarseInput.threads = workerBudget.threadsPerWorker;
                    const std::shared_ptr<std::atomic<int>> progress =
                        coarseProgress[static_cast<std::size_t>(scaleIndex)];
                    coarseStarted[static_cast<std::size_t>(scaleIndex)]->store(true);
                    coarseInput.progressFn = [progress](const QString &, int percent)
                    {
                        progress->store(std::clamp(percent, 0, 100));
                    };

                    try
                    {
                        coarseExecutions[static_cast<std::size_t>(scaleIndex)] =
                            _attemptRunner(coarseInput);
                    }
                    catch (const std::exception &error)
                    {
                        SfmAttemptExecutionResult failed;
                        failed.result.success = false;
                        failed.result.errorMessage =
                            QStringLiteral("焦距候选 %1 执行异常: %2")
                                .arg(coarseInput.estimatedFocalScale)
                                .arg(QString::fromUtf8(error.what()));
                        failed.result.summary = failed.result.errorMessage;
                        coarseExecutions[static_cast<std::size_t>(scaleIndex)] =
                            std::move(failed);
                    }
                    catch (...)
                    {
                        SfmAttemptExecutionResult failed;
                        failed.result.success = false;
                        failed.result.errorMessage =
                            QStringLiteral("焦距候选 %1 执行时发生未知异常")
                                .arg(coarseInput.estimatedFocalScale);
                        failed.result.summary = failed.result.errorMessage;
                        coarseExecutions[static_cast<std::size_t>(scaleIndex)] =
                            std::move(failed);
                    }

                    progress->store(100);
                    completedCoarseCount.fetch_add(1);
                }
            }));
        }

        // 汇总所有候选的真实内部进度；不把“当前候选”误显示成整体百分比。
        while (completedCoarseCount.load() < coarseCount)
        {
            if (originalProgress)
            {
                int accumulatedProgress = 0;
                int runningCount = 0;
                int highestCandidatePercent = 0;
                for (int index = 0; index < coarseCount; ++index)
                {
                    const int candidatePercent =
                        coarseProgress[static_cast<std::size_t>(index)]->load();
                    accumulatedProgress += candidatePercent;
                    highestCandidatePercent = std::max(highestCandidatePercent, candidatePercent);
                    if (coarseStarted[static_cast<std::size_t>(index)]->load() &&
                        candidatePercent < 100)
                    {
                        ++runningCount;
                    }
                }
                const int aggregatePercent = coarseCount > 0
                    ? accumulatedProgress / coarseCount
                    : 100;
                originalProgress(
                    QStringLiteral("无相机先验焦距粗搜索：已完成 %1/%2，运行中 %3，候选内部最高 %4%")
                        .arg(completedCoarseCount.load())
                        .arg(coarseCount)
                        .arg(runningCount)
                        .arg(highestCandidatePercent),
                    45 + aggregatePercent * 35 / 100);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        for (std::future<void> &worker : workers)
        {
            worker.get();
        }
        if (originalProgress)
        {
            originalProgress(
                QStringLiteral("无相机先验焦距粗搜索：已完成 %1/%1 个候选")
                    .arg(coarseCount),
                80);
        }

        if (input.cancelFlag && input.cancelFlag->load())
        {
            AerialTriangulationReconstructionResult cancelled;
            cancelled.success = false;
            cancelled.errorMessage = QStringLiteral("用户取消");
            cancelled.summary = cancelled.errorMessage;
            cancelled.durationSeconds = timer.elapsed() / 1000.0;
            return cancelled;
        }
        for (int scaleIndex = 0; scaleIndex < coarseCount; ++scaleIndex)
        {
            appendCandidate(
                coarseScales[static_cast<std::size_t>(scaleIndex)],
                std::move(coarseExecutions[static_cast<std::size_t>(scaleIndex)]));
        }

        // 排序首要目标是注册完整性，其次才是网络刚性、闭环连续性、RMS 和点数。
        const std::vector<SfmCandidateSummary> ranked = rankCandidates(candidateSummaries);
        const int selectedIndex = ranked.empty() ? 0 : ranked.front().candidateIndex;
        selectedFocalScale = focalCandidates.at(selectedIndex).focalScale;
        execution = std::move(candidateExecutions[selectedIndex]);

        // 大工程的候选只注册受限数量的连通影像。选出焦距后必须重新执行一次完整 SfM，
        // 探测模型只用于排序，绝不能作为最终稀疏云或相机位姿写入工程。
        if (focalProbeLimit > 0)
        {
            PreparedAerialTriangulationInput fullInput = input;
            fullInput.estimatedFocalScale = selectedFocalScale;
            fullInput.threads = resolveSfmThreadBudget(input.threads);
            fullInput.coarseFocalEvaluation = false;
            fullInput.maxRegisteredImages = 0;
            fullInput.progressFn = [originalProgress](const QString &stage, int percent)
            {
                if (originalProgress)
                {
                    originalProgress(
                        QStringLiteral("使用最佳焦距执行完整空三: %1").arg(stage),
                        80 + std::clamp(percent, 0, 100) * 10 / 100);
                }
            };
            execution = _attemptRunner(fullInput);
            const int probeCoverage =
                candidateSummaries[static_cast<std::size_t>(selectedIndex)].registeredImages;
            adaptiveRefinementAccepted = input.adaptiveCameraModelFitting &&
                execution.result.success &&
                execution.result.numRegisteredImages >= probeCoverage;
            if (input.adaptiveCameraModelFitting && !adaptiveRefinementAccepted)
            {
                // 联合自标定若连探测阶段已有的注册覆盖都无法保持，则用同一最佳焦距
                // 再跑一次固定内参完整解。该回退只在自标定退化时发生，不增加正常路径成本。
                fullInput.adaptiveCameraModelFitting = false;
                execution = _attemptRunner(fullInput);
            }
        }
        // 自适应内参只在粗搜索胜出焦距附近执行一次联合细化，并再次经过同一排序门控。
        else if (input.adaptiveCameraModelFitting)
        {
            PreparedAerialTriangulationInput refinementInput = input;
            refinementInput.estimatedFocalScale = selectedFocalScale;
            refinementInput.progressFn = [originalProgress](const QString &stage, int percent)
            {
                if (originalProgress)
                {
                    originalProgress(
                        QStringLiteral("使用最佳焦距细化内参: %1").arg(stage),
                        80 + std::clamp(percent, 0, 100) * 10 / 100);
                }
            };
            SfmAttemptExecutionResult refinedExecution = _attemptRunner(refinementInput);
            const QJsonObject sparseQuality = sparseQualityFromExecution(input, refinedExecution);
            if (!sparseQuality.isEmpty())
            {
                refinedExecution.result.sfmDiagnostics.insert(
                    QStringLiteral("sparse_quality"), sparseQuality);
            }

            SfmCandidateSummary baselineSummary = candidateSummaries[static_cast<std::size_t>(selectedIndex)];
            baselineSummary.candidateIndex = 0;
            SfmCandidateSummary refinementSummary = summaryFromExecution(
                1,
                selectedFocalScale,
                refinedExecution,
                input.sequenceLoopClosure);
            if (isBetterCandidate(refinementSummary, baselineSummary) ||
                isAcceptableCalibrationRefinement(refinementSummary, baselineSummary))
            {
                execution = std::move(refinedExecution);
                adaptiveRefinementAccepted = true;
            }
        }
    }
    else
    {
        focalCandidates.append(candidateFromExecution(input.estimatedFocalScale, execution));
    }

    // 在正式提交前固化所有焦距候选及选择结果；失败返回也保留诊断。
    QJsonArray focalCandidateJson;
    for (int index = 0; index < focalCandidates.size(); ++index)
    {
        const SfmCandidateSummary *summary = nullptr;
        if (index < static_cast<int>(candidateSummaries.size()))
        {
            summary = &candidateSummaries[static_cast<std::size_t>(index)];
        }
        focalCandidateJson.append(candidateToJson(focalCandidates.at(index), summary));
    }
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_focal_search"), focalCandidates.size() > 1);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_initialization_search"), focalCandidates.size() > 1);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_camera_model_fitting"), input.adaptiveCameraModelFitting);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_camera_model_refinement_accepted"), adaptiveRefinementAccepted);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_focal_scale"), selectedFocalScale);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_focal_seed_scale"), selectedFocalScale);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_focal_candidates"), focalCandidateJson);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_probe_registration_limit"), focalProbeLimit);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_probe_full_replay"), focalProbeLimit > 0);

    const bool focalInitializationSearch = focalCandidates.size() > 1;
    QString selfCalibrationStatus = QStringLiteral("trusted_prior");
    bool selfCalibrationRequiresReview = false;
    if (focalInitializationSearch && adaptiveRefinementAccepted)
    {
        selfCalibrationStatus = QStringLiteral("refined");
    }
    else if (focalInitializationSearch && input.adaptiveCameraModelFitting)
    {
        selfCalibrationStatus = QStringLiteral("coarse_seed_only");
        selfCalibrationRequiresReview = true;
    }
    else if (focalInitializationSearch)
    {
        selfCalibrationStatus = QStringLiteral("fixed_coarse_seed");
    }
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("camera_self_calibration_status"), selfCalibrationStatus);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("camera_self_calibration_requires_review"),
        selfCalibrationRequiresReview);

    if (!execution.result.success)
    {
        execution.result.durationSeconds = timer.elapsed() / 1000.0;
        return execution.result;
    }
    if (input.cancelFlag && input.cancelFlag->load())
    {
        execution.result.success = false;
        execution.result.errorMessage = QStringLiteral("用户取消");
        execution.result.summary = execution.result.errorMessage;
        execution.result.durationSeconds = timer.elapsed() / 1000.0;
        return execution.result;
    }

    if (originalProgress)
    {
        originalProgress(QStringLiteral("写出稀疏点云和质量报告..."), 92);
    }
    // 写出器是唯一正式提交点：PLY、质量 sidecar 和待回写相机在此同步生成。
    QString writeError;
    if (!_resultWriter(input, &execution, &writeError))
    {
        execution.result.success = false;
        execution.result.errorMessage = writeError;
        execution.result.summary = writeError;
        execution.result.durationSeconds = timer.elapsed() / 1000.0;
        return execution.result;
    }

    execution.result.durationSeconds = timer.elapsed() / 1000.0;
    if (originalProgress)
    {
        originalProgress(QStringLiteral("完成"), 100);
    }
    return execution.result;
}

} // namespace xjw::aerial_triangulation
