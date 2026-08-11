/**
 * @file QualityReportWriter.cpp
 * @brief 从最终稀疏重建计算点级、相机级、网络级质量指标。
 *
 * 点级记录服务于匹配/三角化审查；相机级记录服务于异常影像定位；网络级指标
 * 服务于 MVS 入口门控和焦距候选排序。所有误差单位为像素，角度单位为度。
 */

#include "reporting/QualityReportWriter.h"

#include "project/SfmQualityJsonSerializer.h"
#include "project/SparseResultQuality.h"
#include "quality/SfmQualityMetrics.h"
#include "reconstruction/SfmReconstruction.h"

#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QSize>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <vector>

namespace xjw::aerial_triangulation
{
namespace
{

constexpr double kExportMaxReprojectionErrorPx = 4.0;

/**
 * @brief 计算一个三维点所有观测射线两两组合中的最大交会角。
 *
 * 最大角代表该轨迹最有利的基线组合；它用于网络诊断，不替代逐观测正深度和
 * 重投影验证。
 */
double maximumTriangulationAngleDegrees(const SfmReconstruction &reconstruction,
                                        const ScenePoint3D &point)
{
    double maximumAngle = 0.0;
    for (std::size_t first = 0; first + 1 < point.track.elements.size(); ++first)
    {
        const ImageId imageA = point.track.elements[first].imageId;
        if (!reconstruction.hasCamera(imageA))
        {
            continue;
        }
        const std::array<double, 3> centerA = reconstruction.camera(imageA).cameraCenter();
        for (std::size_t second = first + 1; second < point.track.elements.size(); ++second)
        {
            const ImageId imageB = point.track.elements[second].imageId;
            if (!reconstruction.hasCamera(imageB))
            {
                continue;
            }
            const std::array<double, 3> centerB = reconstruction.camera(imageB).cameraCenter();
            const std::array<double, 3> rayA{
                point.xyz[0] - centerA[0], point.xyz[1] - centerA[1], point.xyz[2] - centerA[2]};
            const std::array<double, 3> rayB{
                point.xyz[0] - centerB[0], point.xyz[1] - centerB[1], point.xyz[2] - centerB[2]};
            const double normA = std::hypot(rayA[0], rayA[1], rayA[2]);
            const double normB = std::hypot(rayB[0], rayB[1], rayB[2]);
            if (normA <= 1e-12 || normB <= 1e-12)
            {
                continue;
            }
            const double cosine = std::clamp((rayA[0] * rayB[0] + rayA[1] * rayB[1] +
                                              rayA[2] * rayB[2]) / (normA * normB),
                                             -1.0,
                                             1.0);
            maximumAngle = std::max(maximumAngle, std::acos(cosine) * 180.0 / 3.14159265358979323846);
        }
    }
    return maximumAngle;
}

/**
 * @brief 获取网格覆盖统计所需影像尺寸。
 *
 * 优先读取真实影像头；无头/测试输入回退到关键点包围范围，保证质量序列化仍可用。
 */
QSize inferImageSize(const PreparedAerialTriangulationInput &input,
                     const SfmReconstruction &reconstruction)
{
    for (const QString &path : input.images)
    {
        const QSize size = QImageReader(path).size();
        if (size.isValid())
        {
            return size;
        }
    }

    double maxX = 0.0;
    double maxY = 0.0;
    for (const ImageId imageId : reconstruction.allImageIds())
    {
        for (const FeatureKeypoint &keypoint : reconstruction.image(imageId).keypoints)
        {
            maxX = std::max(maxX, static_cast<double>(keypoint.x));
            maxY = std::max(maxY, static_cast<double>(keypoint.y));
        }
    }
    return QSize(std::max(1, static_cast<int>(std::ceil(maxX)) + 1),
                 std::max(1, static_cast<int>(std::ceil(maxY)) + 1));
}

struct CollectedSparseQuality
{
    std::vector<SfmQualityPoint> qualityPoints;
    std::unordered_map<ImageId, std::pair<double, int>> cameraErrors;
    std::optional<QJsonArray> serializedPoints;
};

/**
 * @brief 一次遍历收集网络质量；只有正式报告路径才序列化逐点明细。
 */
CollectedSparseQuality collectSparseQuality(const SfmReconstruction &reconstruction,
                                            bool serializeDetails)
{
    CollectedSparseQuality collected;
    if (serializeDetails)
    {
        collected.serializedPoints.emplace();
    }
    for (const Point3DId pointId : reconstruction.allPoint3DIds())
    {
        if (!reconstruction.hasPoint3D(pointId))
        {
            continue;
        }
        const ScenePoint3D &point = reconstruction.point3D(pointId);
        if (point.error > kExportMaxReprojectionErrorPx || point.track.length() < 2)
        {
            continue;
        }

        const double triangulationAngle = maximumTriangulationAngleDegrees(reconstruction, point);
        SfmQualityPoint qualityPoint;
        qualityPoint.trackLength = static_cast<int>(point.track.length());
        qualityPoint.reprojectionErrorPx = point.error;
        qualityPoint.triangulationAngleDeg = triangulationAngle;
        std::optional<QJsonArray> observations;
        if (serializeDetails)
        {
            observations.emplace();
        }

        for (const TrackElement &element : point.track.elements)
        {
            if (!reconstruction.hasImage(element.imageId))
            {
                continue;
            }
            const ImageData &image = reconstruction.image(element.imageId);
            if (element.featureIdx >= image.keypoints.size())
            {
                continue;
            }

            const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
            qualityPoint.observations.push_back(
                {static_cast<int>(element.imageId), keypoint.x, keypoint.y});
            if (!serializeDetails)
            {
                continue;
            }

            QJsonObject observation{
                {QStringLiteral("image_id"), static_cast<int>(element.imageId)},
                {QStringLiteral("image_path"), QString::fromStdString(image.imagePath)},
                {QStringLiteral("image_name"), QFileInfo(QString::fromStdString(image.imagePath)).fileName()},
                {QStringLiteral("feature_idx"), static_cast<int>(element.featureIdx)},
                {QStringLiteral("xy"), QJsonArray{keypoint.x, keypoint.y}},
            };
            collected.cameraErrors[element.imageId].first += point.error;
            ++collected.cameraErrors[element.imageId].second;

            if (reconstruction.hasCamera(element.imageId))
            {
                double projected[2]{};
                const FramePinholeCamera &camera = reconstruction.camera(element.imageId);
                const bool projectedOk = camera.projectWorldPoint(point.xyz.data(), projected) ||
                    camera.projectWorldPointSigned(point.xyz.data(), projected);
                if (projectedOk && std::isfinite(projected[0]) && std::isfinite(projected[1]))
                {
                    const double residualX = projected[0] - keypoint.x;
                    const double residualY = projected[1] - keypoint.y;
                    observation.insert(QStringLiteral("projected_xy"),
                                       QJsonArray{projected[0], projected[1]});
                    observation.insert(QStringLiteral("residual_xy"),
                                       QJsonArray{residualX, residualY});
                    observation.insert(QStringLiteral("residual_norm_px"),
                                       std::hypot(residualX, residualY));
                }
            }
            observations->append(observation);
        }

        if (serializeDetails)
        {
            collected.serializedPoints->append(QJsonObject{
                {QStringLiteral("track_len"), static_cast<int>(point.track.length())},
                {QStringLiteral("rms_reproj_px"), point.error},
                {QStringLiteral("triangulation_angle_deg"), triangulationAngle},
                {QStringLiteral("min_tri_angle_deg"), triangulationAngle},
                {QStringLiteral("point_xyz"), QJsonArray{point.xyz[0], point.xyz[1], point.xyz[2]}},
                {QStringLiteral("observations"), *observations},
            });
        }
        collected.qualityPoints.push_back(std::move(qualityPoint));
    }
    return collected;
}

QJsonObject buildSparseQualitySummaryObject(
    const PreparedAerialTriangulationInput &input,
    const SfmReconstruction &reconstruction,
    const AerialTriangulationReconstructionResult &result,
    const std::vector<SfmQualityPoint> &qualityPoints)
{
    const QSize imageSize = inferImageSize(input, reconstruction);
    SfmQualityMetricsOptions qualityOptions;
    qualityOptions.totalImageCount = input.images.size();
    qualityOptions.registeredImageCount = result.numRegisteredImages;
    qualityOptions.imageWidth = imageSize.width();
    qualityOptions.imageHeight = imageSize.height();
    qualityOptions.minTrackLength = 2;
    qualityOptions.minTriangulationAngleDeg = 2.0;
    qualityOptions.maxReprojectionErrorPx = input.quality >= 2 ? 1.5 : 3.0;
    qualityOptions.minObservationGridCoverageMeanForMvs = 0.08;
    QJsonObject sparseQuality = serializeSfmQualityMetrics(
        computeSfmQualityMetrics(qualityPoints, qualityOptions));
    sparseQuality.insert(QStringLiteral("mean_reprojection_error_px"), result.meanReprojError);
    return sparseQuality;
}

} // namespace

QJsonObject QualityReportWriter::buildSparseQualitySummary(
    const PreparedAerialTriangulationInput &input,
    const SfmReconstruction &reconstruction,
    const AerialTriangulationReconstructionResult &result)
{
    const CollectedSparseQuality collected = collectSparseQuality(reconstruction, false);
    return buildSparseQualitySummaryObject(
        input, reconstruction, result, collected.qualityPoints);
}

SparseQualityReport QualityReportWriter::build(
    const PreparedAerialTriangulationInput &input,
    const SfmReconstruction &reconstruction,
    const AerialTriangulationReconstructionResult &result)
{
    SparseQualityReport report;
    CollectedSparseQuality collected = collectSparseQuality(reconstruction, true);
    report.points = std::move(*collected.serializedPoints);

    // 第二阶段：为全部输入影像输出记录，包括未注册影像，便于 GUI 直接定位缺口。
    for (int index = 0; index < input.images.size(); ++index)
    {
        const ImageId imageId = static_cast<ImageId>(index);
        const auto error = collected.cameraErrors.find(imageId);
        const bool registered = reconstruction.isRegistered(imageId);
        const double residual = registered && error != collected.cameraErrors.end() &&
            error->second.second > 0
            ? error->second.first / error->second.second
            : (registered ? result.meanReprojError : 0.0);
        report.perCameraResiduals.append(QJsonObject{
            {QStringLiteral("path"), input.images.at(index)},
            {QStringLiteral("registered"), registered},
            {QStringLiteral("residual_px"), residual},
        });
    }

    // 第三阶段：汇总注册覆盖、轨迹长度、交会角、误差和影像网格覆盖。
    const QJsonObject sparseQuality = buildSparseQualitySummaryObject(
        input, reconstruction, result, collected.qualityPoints);

    // 第四阶段：保留 attempt 诊断，并追加最终模型重新计算的稳定质量字段。
    report.diagnostics = result.sfmDiagnostics;
    report.diagnostics.insert(QStringLiteral("sparse_quality"), sparseQuality);
    report.diagnostics.insert(QStringLiteral("ba_summary"), QJsonObject{
        {QStringLiteral("rms_before_px"), result.baRmsBefore},
        {QStringLiteral("rms_after_px"), result.baRmsAfter},
        {QStringLiteral("tracks_total"), result.baTracksTotal},
        {QStringLiteral("tracks_optimized"), result.baTracksOptimized},
        {QStringLiteral("tracks_filtered"), result.baTracksFiltered},
    });

    const bool baApplied = result.baTracksTotal > 0 || result.baTracksOptimized > 0;
    report.qualityMetadata = xjw::common::project::buildSparseQualityMetadata(
        report.points,
        result.numRegisteredImages,
        baApplied,
        xjw::common::project::kSparseResultKindSfmSparseReconstruction,
        QString(),
        QString(),
        input.images.size());
    return report;
}

} // namespace xjw::aerial_triangulation
