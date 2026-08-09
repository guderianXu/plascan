/**
 * @file AerialTriangulationPipeline.cpp
 * @brief 连接点图到正式 SfM/BA 稀疏结果的候选搜索与提交实现。
 *
 * 当输入没有完整可信内参时，管线对多个焦距尺度运行相互隔离的 SfM 候选。
 * 候选按注册覆盖、摄影测量网络质量、闭环连续性和重投影误差排序；只有胜出模型
 * 可进入 AerialTriangulationResultWriter，避免失败候选污染工程相机和结果文件。
 */

#include "workflow/AerialTriangulationPipeline.h"

#include "CameraIntrinsicPrior.h"
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
#include <QImageReader>
#include <QJsonArray>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace xjw::aerial_triangulation
{
namespace
{

struct AerialBlockGeometry
{
    bool valid = false;
    double opticalAxisConcentration = 0.0;
    double cameraCenterPlanarityRatio = 1.0;
    double cameraCenterNormalSpanRmsRatio = 1.0;
};

struct BatchFocalPrior
{
    bool valid = false;
    double focalScale = 0.0;
    QString source;
    QString make;
    QString model;
    int acceptedSamples = 0;
    int inspectedSamples = 0;
};

/**
 * @brief 从摄影块的代表影像建立一致焦距先验。
 *
 * 单张 EXIF 可能丢失或损坏，因此均匀抽样最多 32 张。至少 75% 样本必须可解析，
 * 且焦距尺度极差不超过 2%，才允许跳过无标定粗搜索。这样不会把混合相机批次
 * 或可换镜头机身误判为同一固定内参组。
 */
BatchFocalPrior resolveBatchFocalPrior(const QStringList &imagePaths)
{
    if (imagePaths.isEmpty())
    {
        return {};
    }

    const int inspectedCount = std::min(32, static_cast<int>(imagePaths.size()));
    std::vector<CameraIntrinsicPrior> priors;
    priors.reserve(static_cast<std::size_t>(inspectedCount));
    for (int sample = 0; sample < inspectedCount; ++sample)
    {
        const int imageIndex = inspectedCount == 1
            ? 0
            : sample * (imagePaths.size() - 1) / (inspectedCount - 1);
        const QString &path = imagePaths.at(imageIndex);
        QImageReader reader(path);
        const QSize size = reader.size();
        if (!size.isValid())
        {
            continue;
        }
        const auto metadata = xjw::common::io::readImageExifMetadata(path);
        if (!metadata.has_value())
        {
            continue;
        }
        const auto prior = estimateCameraIntrinsicPrior(
            *metadata, size.width(), size.height());
        if (prior.has_value() && prior->strong)
        {
            priors.push_back(*prior);
        }
    }

    const int minimumAccepted = std::max(1, (inspectedCount * 3 + 3) / 4);
    if (static_cast<int>(priors.size()) < minimumAccepted)
    {
        return {false, 0.0, {}, {}, {}, static_cast<int>(priors.size()), inspectedCount};
    }

    const QString make = priors.front().make.trimmed().toLower();
    const QString model = priors.front().model.trimmed().toLower();
    std::vector<double> scales;
    scales.reserve(priors.size());
    for (const CameraIntrinsicPrior &prior : priors)
    {
        if (prior.make.trimmed().toLower() != make ||
            prior.model.trimmed().toLower() != model)
        {
            return {false, 0.0, {}, {}, {}, static_cast<int>(priors.size()), inspectedCount};
        }
        scales.push_back(prior.focalScale);
    }
    std::sort(scales.begin(), scales.end());
    if (scales.front() <= 0.0 || scales.back() / scales.front() > 1.02)
    {
        return {false, 0.0, {}, {}, {}, static_cast<int>(priors.size()), inspectedCount};
    }
    const double medianScale = 0.5 *
        (scales[(scales.size() - 1) / 2] + scales[scales.size() / 2]);
    return {
        true,
        medianScale,
        priors.front().source,
        priors.front().make,
        priors.front().model,
        static_cast<int>(priors.size()),
        inspectedCount,
    };
}

/// 对 3x3 对称矩阵执行 Jacobi 对角化；这里只需要协方差特征值，避免引入额外线性代数依赖。
std::array<double, 3> symmetricEigenvalues(std::array<double, 9> matrix)
{
    for (int iteration = 0; iteration < 24; ++iteration)
    {
        int p = 0;
        int q = 1;
        double largest = std::fabs(matrix[1]);
        for (const auto [row, column] : {std::pair{0, 2}, std::pair{1, 2}})
        {
            const double value = std::fabs(matrix[row * 3 + column]);
            if (value > largest)
            {
                largest = value;
                p = row;
                q = column;
            }
        }
        if (largest <= 1.0e-12)
        {
            break;
        }

        const double app = matrix[p * 3 + p];
        const double aqq = matrix[q * 3 + q];
        const double apq = matrix[p * 3 + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        for (int index = 0; index < 3; ++index)
        {
            if (index == p || index == q)
            {
                continue;
            }
            const double aip = matrix[index * 3 + p];
            const double aiq = matrix[index * 3 + q];
            matrix[index * 3 + p] = matrix[p * 3 + index] = cosine * aip - sine * aiq;
            matrix[index * 3 + q] = matrix[q * 3 + index] = sine * aip + cosine * aiq;
        }
        matrix[p * 3 + p] = cosine * cosine * app - 2.0 * sine * cosine * apq +
                            sine * sine * aqq;
        matrix[q * 3 + q] = sine * sine * app + 2.0 * sine * cosine * apq +
                            cosine * cosine * aqq;
        matrix[p * 3 + q] = matrix[q * 3 + p] = 0.0;
    }

    std::array<double, 3> eigenvalues{{matrix[0], matrix[4], matrix[8]}};
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

/**
 * @brief 识别近垂直平行摄影块，并量化相机中心是否出现人工穹顶。
 *
 * 仅当物理前向光轴高度集中时才启用平面性指标。环拍/绕飞相机朝向变化明显，
 * 不会被航测平面先验影响。
 */
AerialBlockGeometry evaluateAerialBlockGeometry(const SfmReconstruction &reconstruction)
{
    const std::vector<ImageId> imageIds = reconstruction.registeredImageIds();
    if (imageIds.size() < 8)
    {
        return {};
    }

    std::array<double, 3> meanCenter{{0.0, 0.0, 0.0}};
    std::array<double, 3> meanAxis{{0.0, 0.0, 0.0}};
    for (const ImageId imageId : imageIds)
    {
        const Camera camera = reconstruction.camera(imageId).normalizedForPositiveDepth();
        const auto center = camera.cameraCenter();
        const auto rotation = camera.cameraToWorldRotation();
        for (int axis = 0; axis < 3; ++axis)
        {
            meanCenter[axis] += center[axis];
            meanAxis[axis] += rotation[axis * 3 + 2];
        }
    }
    const double count = static_cast<double>(imageIds.size());
    for (int axis = 0; axis < 3; ++axis)
    {
        meanCenter[axis] /= count;
        meanAxis[axis] /= count;
    }
    const double opticalAxisConcentration = std::sqrt(
        meanAxis[0] * meanAxis[0] + meanAxis[1] * meanAxis[1] + meanAxis[2] * meanAxis[2]);

    std::array<double, 9> covariance{};
    for (const ImageId imageId : imageIds)
    {
        const auto center = reconstruction.camera(imageId).cameraCenter();
        const std::array<double, 3> delta{{
            center[0] - meanCenter[0], center[1] - meanCenter[1], center[2] - meanCenter[2]}};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                covariance[row * 3 + column] += delta[row] * delta[column] / count;
            }
        }
    }
    const std::array<double, 3> eigenvalues = symmetricEigenvalues(covariance);
    const double trace = std::max(0.0, eigenvalues[0]) +
                         std::max(0.0, eigenvalues[1]) +
                         std::max(0.0, eigenvalues[2]);
    const double planarityRatio = trace > 1.0e-12
        ? std::max(0.0, eigenvalues[0]) / trace
        : 1.0;
    const double normalSpanRmsRatio = std::sqrt(std::max(0.0, planarityRatio));

    return {
        std::isfinite(opticalAxisConcentration) && opticalAxisConcentration >= 0.85 &&
            std::isfinite(planarityRatio) && std::isfinite(normalSpanRmsRatio),
        opticalAxisConcentration,
        planarityRatio,
        normalSpanRmsRatio,
    };
}

bool hasAbsoluteGeometryConstraint(
    const PreparedAerialTriangulationInput &input,
    const QJsonObject &diagnostics)
{
    return input.useProjectCameraPoses || input.lockInputCameraPoses ||
           diagnostics.value(QStringLiteral("control_network_applied")).toBool(false) ||
           diagnostics.value(QStringLiteral("control_point_constraints")).toInt(0) > 0;
}

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
    return QualityReportWriter::buildSparseQualitySummary(
        input, *execution.reconstruction, execution.result);
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
    if (execution.reconstruction)
    {
        const AerialBlockGeometry geometry =
            evaluateAerialBlockGeometry(*execution.reconstruction);
        summary.hasAerialBlockGeometry = geometry.valid;
        summary.opticalAxisConcentration = geometry.opticalAxisConcentration;
        summary.cameraCenterPlanarityRatio = geometry.cameraCenterPlanarityRatio;
    }
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
    if (summary && summary->hasAerialBlockGeometry)
    {
        object.insert(QStringLiteral("optical_axis_concentration"),
                      summary->opticalAxisConcentration);
        object.insert(QStringLiteral("camera_center_planarity_ratio"),
                      summary->cameraCenterPlanarityRatio);
    }
    return object;
}

struct FocalSearchBatchResult
{
    std::vector<SfmAttemptExecutionResult> executions;
    SfmWorkerBudget workerBudget;
};

/**
 * @brief 并行执行一个确定顺序的焦距候选批次。
 *
 * 结果槽位与 scales 一一对应，因此完成先后不会影响后续稳定排序。
 */
FocalSearchBatchResult runFocalSearchBatch(
    const AerialTriangulationPipeline::AttemptRunner &attemptRunner,
    const PreparedAerialTriangulationInput &baseInput,
    const std::vector<double> &scales,
    int focalProbeLimit,
    const std::function<void(const QString &, int)> &progressFn,
    const QString &stageName,
    int progressBase,
    int progressSpan)
{
    FocalSearchBatchResult batch;
    const int candidateCount = static_cast<int>(scales.size());
    if (candidateCount <= 0)
    {
        return batch;
    }

    batch.workerBudget = allocateWorkers(candidateCount, baseInput.threads);
    batch.executions.resize(static_cast<std::size_t>(candidateCount));
    std::vector<std::shared_ptr<std::atomic<int>>> candidateProgress;
    std::vector<std::shared_ptr<std::atomic<bool>>> candidateStarted;
    candidateProgress.reserve(static_cast<std::size_t>(candidateCount));
    candidateStarted.reserve(static_cast<std::size_t>(candidateCount));
    for (int index = 0; index < candidateCount; ++index)
    {
        candidateProgress.push_back(std::make_shared<std::atomic<int>>(0));
        candidateStarted.push_back(std::make_shared<std::atomic<bool>>(false));
    }

    std::atomic<int> nextCandidateIndex{0};
    std::atomic<int> completedCandidateCount{0};
    std::vector<std::jthread> workers;
    workers.reserve(static_cast<std::size_t>(batch.workerBudget.workerCount));
    for (int workerIndex = 0; workerIndex < batch.workerBudget.workerCount; ++workerIndex)
    {
        workers.emplace_back([&, workerIndex]()
        {
            while (true)
            {
                const int scaleIndex = nextCandidateIndex.fetch_add(1);
                if (scaleIndex >= candidateCount)
                {
                    break;
                }

                PreparedAerialTriangulationInput candidateInput = baseInput;
                candidateInput.estimatedFocalScale = scales[static_cast<std::size_t>(scaleIndex)];
                candidateInput.adaptiveCameraModelFitting = false;
                candidateInput.coarseFocalEvaluation = true;
                candidateInput.maxRegisteredImages = focalProbeLimit;
                candidateInput.threads = batch.workerBudget.threadsForWorker(workerIndex);
                const std::shared_ptr<std::atomic<int>> progress =
                    candidateProgress[static_cast<std::size_t>(scaleIndex)];
                candidateStarted[static_cast<std::size_t>(scaleIndex)]->store(true);
                candidateInput.progressFn = [progress](const QString &, int percent)
                {
                    progress->store(std::clamp(percent, 0, 100));
                };

                try
                {
                    batch.executions[static_cast<std::size_t>(scaleIndex)] =
                        attemptRunner(candidateInput);
                }
                catch (const std::exception &error)
                {
                    SfmAttemptExecutionResult failed;
                    failed.result.success = false;
                    failed.result.errorMessage =
                        QStringLiteral("焦距候选 %1 执行异常: %2")
                            .arg(candidateInput.estimatedFocalScale)
                            .arg(QString::fromUtf8(error.what()));
                    failed.result.summary = failed.result.errorMessage;
                    batch.executions[static_cast<std::size_t>(scaleIndex)] = std::move(failed);
                }
                catch (...)
                {
                    SfmAttemptExecutionResult failed;
                    failed.result.success = false;
                    failed.result.errorMessage =
                        QStringLiteral("焦距候选 %1 执行时发生未知异常")
                            .arg(candidateInput.estimatedFocalScale);
                    failed.result.summary = failed.result.errorMessage;
                    batch.executions[static_cast<std::size_t>(scaleIndex)] = std::move(failed);
                }

                progress->store(100);
                completedCandidateCount.fetch_add(1);
            }
        });
    }

    while (completedCandidateCount.load() < candidateCount)
    {
        if (progressFn)
        {
            int accumulatedProgress = 0;
            int runningCount = 0;
            int highestCandidatePercent = 0;
            for (int index = 0; index < candidateCount; ++index)
            {
                const int percent = candidateProgress[static_cast<std::size_t>(index)]->load();
                accumulatedProgress += percent;
                highestCandidatePercent = std::max(highestCandidatePercent, percent);
                if (candidateStarted[static_cast<std::size_t>(index)]->load() && percent < 100)
                {
                    ++runningCount;
                }
            }
            const int aggregatePercent = accumulatedProgress / candidateCount;
            progressFn(
                QStringLiteral("%1：已完成 %2/%3，运行中 %4/%5，CPU线程预算 %6，候选内部最高 %7%")
                    .arg(stageName)
                    .arg(completedCandidateCount.load())
                    .arg(candidateCount)
                    .arg(runningCount)
                    .arg(batch.workerBudget.workerCount)
                    .arg(batch.workerBudget.totalThreads())
                    .arg(highestCandidatePercent),
                progressBase + aggregatePercent * progressSpan / 100);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (std::jthread &worker : workers)
    {
        worker.join();
    }
    if (progressFn)
    {
        progressFn(QStringLiteral("%1：已完成 %2/%2 个候选")
                       .arg(stageName)
                       .arg(candidateCount),
                   progressBase + progressSpan);
    }
    return batch;
}

} // namespace

bool AerialTriangulationPipeline::shouldFlagAerialDomingRisk(
    bool geometryValid,
    double opticalAxisConcentration,
    double cameraCenterNormalSpanRmsRatio,
    bool hasAbsoluteGeometryConstraint)
{
    return geometryValid && !hasAbsoluteGeometryConstraint &&
           std::isfinite(opticalAxisConcentration) &&
           opticalAxisConcentration >= 0.90 &&
           std::isfinite(cameraCenterNormalSpanRmsRatio) &&
           cameraCenterNormalSpanRmsRatio >= 0.04;
}

AerialTriangulationPipeline::AerialTriangulationPipeline(
    AttemptRunner attemptRunner,
    ResultWriter resultWriter)
    : _usesProductionAttemptRunner(!attemptRunner),
      _attemptRunner(attemptRunner ? std::move(attemptRunner)
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
    const bool hasProjectOrExternalPrior = hasCompleteCameraIntrinsicPrior(input);
    const BatchFocalPrior metadataPrior = hasProjectOrExternalPrior
        ? BatchFocalPrior{}
        : resolveBatchFocalPrior(input.images);

    PreparedAerialTriangulationInput attemptInput = input;
    if (metadataPrior.valid)
    {
        attemptInput.estimatedFocalScale = metadataPrior.focalScale;
        attemptInput.hasTrustedFocalPrior = true;
        attemptInput.focalPriorSource = metadataPrior.source;
        attemptInput.focalPriorSampleCount = metadataPrior.acceptedSamples;
    }
    attemptInput.adaptiveCameraModelFitting = shouldRunAdaptiveCameraModelRefinement(
        input.adaptiveCameraModelFitting,
        hasProjectOrExternalPrior,
        metadataPrior.valid);

    // 无完整相机先验且元数据也无法建立一致先验时，才从影像几何搜索初始焦距。
    // 焦距初始化与最终 BA 是否释放少量内参是两个独立职责。
    const bool needsFocalInitializationSearch =
        input.images.size() >= 3 && !hasProjectOrExternalPrior && !metadataPrior.valid;
    const int focalProbeLimit = needsFocalInitializationSearch
        ? focalProbeRegistrationLimit(input.images.size())
        : 0;

    attemptInput.threads = resolveSfmThreadBudget(input.threads);
    double tiePointGraphPrepareSeconds = 0.0;
    int preparedTiePointTrackCount = 0;
    int preparedTiePointPairCount = 0;
    if (_usesProductionAttemptRunner)
    {
        if (originalProgress)
        {
            originalProgress(QStringLiteral("一次性读取并索引连接点图..."), 0);
        }
        auto sharedGraph = std::make_shared<PreparedTiePointGraph>();
        QString graphError;
        QElapsedTimer graphTimer;
        graphTimer.start();
        if (!SfmAttemptRunner::readTiePointGraph(
                input.tiePointPath, input.images, sharedGraph.get(), &graphError))
        {
            AerialTriangulationReconstructionResult failed;
            failed.success = false;
            failed.errorMessage = graphError;
            failed.summary = graphError;
            failed.durationSeconds = timer.elapsed() / 1000.0;
            return failed;
        }
        tiePointGraphPrepareSeconds = graphTimer.elapsed() / 1000.0;
        preparedTiePointTrackCount = sharedGraph->trackCount;
        preparedTiePointPairCount = static_cast<int>(sharedGraph->matchPairs.size());
        attemptInput.preparedTiePointGraph = std::move(sharedGraph);
    }
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

    // 有可信先验时只需一次正式试算；无先验搜索的默认焦距也进入统一并行队列，
    // 避免先串行跑完一个候选后才启动其它候选。
    SfmAttemptExecutionResult execution;
    bool adaptiveRefinementAccepted = false;
    if (!needsFocalInitializationSearch)
    {
        execution = _attemptRunner(attemptInput);
        adaptiveRefinementAccepted = attemptInput.adaptiveCameraModelFitting &&
            execution.result.success &&
            execution.result.sfmDiagnostics.value(
                QStringLiteral("ba_result_applied")).toBool() &&
            execution.result.sfmDiagnostics.value(
                QStringLiteral("ba_refined_intrinsic_count")).toInt() > 0;
    }
    QVector<AdaptiveFocalCandidate> focalCandidates;
    std::vector<SfmCandidateSummary> candidateSummaries;
    QVector<SfmAttemptExecutionResult> candidateExecutions;
    double selectedFocalScale = attemptInput.estimatedFocalScale;
    int focalSearchWorkerCount = 0;
    int focalSearchThreadBudget = 0;
    int focalSearchMinThreadsPerWorker = 0;
    int focalSearchMaxThreadsPerWorker = 0;
    int focalSearchCoarseCandidateCount = 0;
    int focalSearchRefinementCandidateCount = 0;
    int focalSearchFallbackCandidateCount = 0;
    bool focalSearchExhaustiveFallback = false;

    if (needsFocalInitializationSearch)
    {
        PreparedAerialTriangulationInput focalQualityInput = input;
        if (focalProbeLimit > 0 && focalQualityInput.images.size() > focalProbeLimit)
        {
            // probe 只承诺注册 focalProbeLimit 台相机；摘要门控也必须使用相同评估规模，
            // 否则会把 64/444 误判为注册覆盖不足并无条件触发全尺度回退。
            focalQualityInput.images.resize(focalProbeLimit);
        }
        // 每个候选保留独立重建对象和同一口径质量摘要，不能交叉复用相机状态。
        const auto appendCandidate = [&](double focalScale,
                                         SfmAttemptExecutionResult candidateExecution)
        {
            const int candidateIndex = focalCandidates.size();
            const QJsonObject sparseQuality =
                sparseQualityFromExecution(focalQualityInput, candidateExecution);
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
        constexpr double kFocalScaleTolerance = 1.0e-9;
        std::vector<double> evaluatedScales;
        evaluatedScales.push_back(input.estimatedFocalScale);
        for (double focalScale : adaptiveFocalCoarseScaleCandidates())
        {
            const bool alreadyQueued = std::any_of(
                evaluatedScales.cbegin(),
                evaluatedScales.cend(),
                [focalScale](double queuedScale)
                {
                    return std::abs(queuedScale - focalScale) <= kFocalScaleTolerance;
                });
            if (!alreadyQueued)
            {
                evaluatedScales.push_back(focalScale);
            }
        }
        focalSearchCoarseCandidateCount = static_cast<int>(evaluatedScales.size());
        const auto updateWorkerDiagnostics = [&](const SfmWorkerBudget &budget)
        {
            if (budget.workerCount <= 0)
            {
                return;
            }
            focalSearchWorkerCount = std::max(focalSearchWorkerCount, budget.workerCount);
            focalSearchThreadBudget = std::max(focalSearchThreadBudget, budget.totalThreads());
            focalSearchMinThreadsPerWorker = focalSearchMinThreadsPerWorker > 0
                ? std::min(focalSearchMinThreadsPerWorker, budget.threadsPerWorker)
                : budget.threadsPerWorker;
            focalSearchMaxThreadsPerWorker = std::max(
                focalSearchMaxThreadsPerWorker,
                budget.threadsPerWorker + (budget.workersWithExtraThread > 0 ? 1 : 0));
        };
        const auto appendBatch = [&](const std::vector<double> &scales,
                                     FocalSearchBatchResult batch)
        {
            updateWorkerDiagnostics(batch.workerBudget);
            for (int scaleIndex = 0; scaleIndex < static_cast<int>(scales.size()); ++scaleIndex)
            {
                appendCandidate(
                    scales[static_cast<std::size_t>(scaleIndex)],
                    std::move(batch.executions[static_cast<std::size_t>(scaleIndex)]));
            }
        };
        const auto cancelledResult = [&]()
        {
            AerialTriangulationReconstructionResult cancelled;
            cancelled.success = false;
            cancelled.errorMessage = QStringLiteral("用户取消");
            cancelled.summary = cancelled.errorMessage;
            cancelled.durationSeconds = timer.elapsed() / 1000.0;
            return cancelled;
        };

        appendBatch(
            evaluatedScales,
            runFocalSearchBatch(
                _attemptRunner,
                attemptInput,
                evaluatedScales,
                focalProbeLimit,
                originalProgress,
                QStringLiteral("无相机先验焦距粗搜索"),
                5,
                50));
        if (input.cancelFlag && input.cancelFlag->load())
        {
            return cancelledResult();
        }

        const std::vector<SfmCandidateSummary> rankedCoarse = rankCandidates(candidateSummaries);
        const int evaluationTarget = focalProbeLimit > 0
            ? focalProbeLimit
            : static_cast<int>(input.images.size());
        const int bestCoarseIndex = rankedCoarse.empty()
            ? -1
            : rankedCoarse.front().candidateIndex;
        const bool coarsePassesSparseQualityGate = bestCoarseIndex >= 0 &&
            bestCoarseIndex < static_cast<int>(candidateExecutions.size()) &&
            candidateExecutions[bestCoarseIndex].result.sfmDiagnostics
                .value(QStringLiteral("sparse_quality")).toObject()
                .value(QStringLiteral("quality_gate")).toObject()
                .value(QStringLiteral("acceptable_for_mvs")).toBool(false);
        const bool coarseHasProductionCloud = !rankedCoarse.empty() &&
            rankedCoarse.front().success &&
            shouldStopAdaptiveFocalReplay(
                evaluationTarget,
                rankedCoarse.front().registeredImages,
                rankedCoarse.front().points3D > 0) &&
            coarsePassesSparseQualityGate;

        std::vector<double> secondStageScales;
        if (coarseHasProductionCloud)
        {
            secondStageScales = adaptiveFocalRefinementScaleCandidates(
                rankedCoarse, evaluatedScales, 2);
            focalSearchRefinementCandidateCount = static_cast<int>(secondStageScales.size());
        }
        else
        {
            // 粗锚点无法建立完整生产模型时恢复旧的全尺度覆盖，避免弱纹理、极端
            // 视场或局部收敛使粗到细策略丢失唯一可用焦距。
            focalSearchExhaustiveFallback = true;
            for (const double scale : adaptiveFocalScaleCandidates())
            {
                const bool alreadyEvaluated = std::any_of(
                    evaluatedScales.cbegin(), evaluatedScales.cend(), [scale](double value)
                    {
                        return std::abs(value - scale) <= kFocalScaleTolerance;
                    });
                if (!alreadyEvaluated)
                {
                    secondStageScales.push_back(scale);
                }
            }
            focalSearchFallbackCandidateCount = static_cast<int>(secondStageScales.size());
        }

        if (!secondStageScales.empty())
        {
            appendBatch(
                secondStageScales,
                runFocalSearchBatch(
                    _attemptRunner,
                    attemptInput,
                    secondStageScales,
                    focalProbeLimit,
                    originalProgress,
                    focalSearchExhaustiveFallback
                        ? QStringLiteral("焦距全尺度兜底搜索")
                        : QStringLiteral("最佳焦距邻域细化"),
                    55,
                    25));
            evaluatedScales.insert(
                evaluatedScales.end(), secondStageScales.cbegin(), secondStageScales.cend());
        }
        if (input.cancelFlag && input.cancelFlag->load())
        {
            return cancelledResult();
        }

        // 排序首要目标是注册完整性，其次才是网络刚性、闭环连续性、RMS 和点数。
        const std::vector<SfmCandidateSummary> ranked = rankCandidates(candidateSummaries);
        const int selectedIndex = ranked.empty() ? 0 : ranked.front().candidateIndex;
        selectedFocalScale = focalCandidates.at(selectedIndex).focalScale;
        execution = std::move(candidateExecutions[selectedIndex]);
        // 排序完成后只保留胜出重建，及时释放其它候选的相机、点云和观测内存。
        candidateExecutions.clear();

        // 大工程的候选只注册受限数量的连通影像。选出焦距后必须重新执行一次完整 SfM，
        // 探测模型只用于排序，绝不能作为最终稀疏云或相机位姿写入工程。
        if (focalProbeLimit > 0)
        {
            PreparedAerialTriangulationInput fullInput = input;
            fullInput.preparedTiePointGraph = attemptInput.preparedTiePointGraph;
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
            refinementInput.preparedTiePointGraph = attemptInput.preparedTiePointGraph;
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
        focalCandidates.append(candidateFromExecution(attemptInput.estimatedFocalScale, execution));
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
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_worker_count"), focalSearchWorkerCount);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_thread_budget"), focalSearchThreadBudget);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_min_threads_per_worker"), focalSearchMinThreadsPerWorker);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_max_threads_per_worker"), focalSearchMaxThreadsPerWorker);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_coarse_candidate_count"),
        focalSearchCoarseCandidateCount);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_refinement_candidate_count"),
        focalSearchRefinementCandidateCount);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_fallback_candidate_count"),
        focalSearchFallbackCandidateCount);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_exhaustive_fallback"),
        focalSearchExhaustiveFallback);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_evaluated_candidate_count"),
        focalCandidates.size());
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("focal_search_shared_tie_point_graph"),
        static_cast<bool>(attemptInput.preparedTiePointGraph));
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("tie_point_graph_prepare_seconds"), tiePointGraphPrepareSeconds);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("tie_point_graph_track_count"), preparedTiePointTrackCount);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("tie_point_graph_pair_count"), preparedTiePointPairCount);
    QJsonObject metadataPriorJson{
        {QStringLiteral("used"), metadataPrior.valid},
        {QStringLiteral("source"), metadataPrior.source},
        {QStringLiteral("make"), metadataPrior.make},
        {QStringLiteral("model"), metadataPrior.model},
        {QStringLiteral("focal_scale"), metadataPrior.focalScale},
        {QStringLiteral("accepted_samples"), metadataPrior.acceptedSamples},
        {QStringLiteral("inspected_samples"), metadataPrior.inspectedSamples},
    };
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("image_metadata_focal_prior"), metadataPriorJson);

    // 正式完整解必须再次计算，不能沿用只注册 64 张影像的焦距探测指标。
    if (execution.reconstruction)
    {
        const AerialBlockGeometry geometry =
            evaluateAerialBlockGeometry(*execution.reconstruction);
        const bool absoluteGeometryConstraint =
            hasAbsoluteGeometryConstraint(input, execution.result.sfmDiagnostics);
        const bool domingRisk = shouldFlagAerialDomingRisk(
            geometry.valid,
            geometry.opticalAxisConcentration,
            geometry.cameraCenterNormalSpanRmsRatio,
            absoluteGeometryConstraint);
        QJsonObject aerialGeometry{
            {QStringLiteral("detected"), geometry.valid},
            {QStringLiteral("optical_axis_concentration"), geometry.opticalAxisConcentration},
            {QStringLiteral("camera_center_planarity_ratio"), geometry.cameraCenterPlanarityRatio},
            {QStringLiteral("camera_center_normal_span_rms_ratio"),
             geometry.cameraCenterNormalSpanRmsRatio},
        };
        // 这里只标记复核风险，不修改或拒绝几何：没有 GPS/GCP 时无法从重投影
        // 唯一区分真实航高变化与径向畸变/自由网漂移，强行压平会损伤真实地形。
        aerialGeometry.insert(QStringLiteral("doming_risk"), domingRisk);
        aerialGeometry.insert(
            QStringLiteral("doming_assessment"),
            domingRisk
                ? QStringLiteral("parallel_block_curvature_requires_absolute_control_review")
                : QStringLiteral("no_parallel_block_doming_risk_detected"));
        aerialGeometry.insert(
            QStringLiteral("absolute_geometry_constraint"),
            absoluteGeometryConstraint);
        execution.result.sfmDiagnostics.insert(
            QStringLiteral("aerial_block_geometry"), aerialGeometry);
    }

    const bool focalInitializationSearch = focalCandidates.size() > 1;
    QString selfCalibrationStatus = hasProjectOrExternalPrior
        ? QStringLiteral("trusted_calibration")
        : QStringLiteral("fixed_intrinsics");
    bool selfCalibrationRequiresReview = execution.result.sfmDiagnostics
        .value(QStringLiteral("aerial_block_geometry")).toObject()
        .value(QStringLiteral("doming_risk")).toBool(false);
    if (focalInitializationSearch && adaptiveRefinementAccepted)
    {
        selfCalibrationStatus = QStringLiteral("refined");
    }
    else if (metadataPrior.valid && adaptiveRefinementAccepted)
    {
        selfCalibrationStatus = QStringLiteral("refined_from_exif_focal_prior");
    }
    else if (metadataPrior.valid && input.adaptiveCameraModelFitting)
    {
        selfCalibrationStatus = QStringLiteral("exif_seed_refinement_rejected");
        selfCalibrationRequiresReview = true;
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
