#include "workflow/AerialTriangulationPipeline.h"

#include "project/ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "reporting/AerialTriangulationResultWriter.h"
#include "reporting/QualityReportWriter.h"
#include "search/AdaptiveFocalSearch.h"
#include "search/SfmSearchPolicy.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>

#include <algorithm>
#include <cmath>

namespace xjw::aerial_triangulation
{
namespace
{

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
        if (!xjw::common::project::cameraFromJson(cameraObject, &camera) ||
            !camera.isValid())
        {
            return false;
        }
    }
    return !input.images.isEmpty();
}

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

SfmCandidateSummary summaryFromExecution(
    int candidateIndex,
    double focalScale,
    const SfmAttemptExecutionResult &execution)
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
    return summary;
}

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

    PreparedAerialTriangulationInput attemptInput = input;
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

    SfmAttemptExecutionResult execution = _attemptRunner(attemptInput);
    QVector<AdaptiveFocalCandidate> focalCandidates;
    std::vector<SfmCandidateSummary> candidateSummaries;
    double selectedFocalScale = input.estimatedFocalScale;

    if (needsFocalInitializationSearch)
    {
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
                candidateIndex, focalScale, candidateExecution));
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

        for (int scaleIndex = 0; scaleIndex < static_cast<int>(coarseScales.size()); ++scaleIndex)
        {
            if (input.cancelFlag && input.cancelFlag->load())
            {
                AerialTriangulationReconstructionResult cancelled;
                cancelled.success = false;
                cancelled.errorMessage = QStringLiteral("用户取消");
                cancelled.summary = cancelled.errorMessage;
                cancelled.durationSeconds = timer.elapsed() / 1000.0;
                return cancelled;
            }

            PreparedAerialTriangulationInput coarseInput = input;
            coarseInput.estimatedFocalScale = coarseScales[static_cast<std::size_t>(scaleIndex)];
            coarseInput.adaptiveCameraModelFitting = false;
            coarseInput.progressFn = [originalProgress,
                                      scaleIndex,
                                      count = static_cast<int>(coarseScales.size())](
                                         const QString &stage, int percent)
            {
                if (originalProgress)
                {
                    const int start = 45 + scaleIndex * 35 / count;
                    const int end = 45 + (scaleIndex + 1) * 35 / count;
                    originalProgress(
                        QStringLiteral("无相机先验焦距粗搜索: %1").arg(stage),
                        start + std::clamp(percent, 0, 100) * (end - start) / 100);
                }
            };
            SfmAttemptExecutionResult coarseExecution = _attemptRunner(coarseInput);
            appendCandidate(coarseInput.estimatedFocalScale, std::move(coarseExecution));
        }

        const std::vector<SfmCandidateSummary> ranked = rankCandidates(candidateSummaries);
        const int selectedIndex = ranked.empty() ? 0 : ranked.front().candidateIndex;
        selectedFocalScale = focalCandidates.at(selectedIndex).focalScale;
        PreparedAerialTriangulationInput replayInput = input;
        replayInput.estimatedFocalScale = selectedFocalScale;
        replayInput.progressFn = [originalProgress](const QString &stage, int percent)
        {
            if (originalProgress)
            {
                originalProgress(
                    QStringLiteral("使用最佳焦距正式重建: %1").arg(stage),
                    80 + std::clamp(percent, 0, 100) * 10 / 100);
            }
        };
        execution = _attemptRunner(replayInput);
    }
    else
    {
        focalCandidates.append(candidateFromExecution(input.estimatedFocalScale, execution));
    }

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
        QStringLiteral("adaptive_focal_scale"), selectedFocalScale);
    execution.result.sfmDiagnostics.insert(
        QStringLiteral("adaptive_focal_candidates"), focalCandidateJson);

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
