/**
 * @file QualityReportWriter.cpp
 * @brief 从最终稀疏重建计算点级、相机级、网络级质量指标。
 *
 * 点级记录服务于匹配/三角化审查；相机级记录服务于异常影像定位；网络级指标
 * 服务于 MVS 入口门控和焦距候选排序。所有误差单位为像素，角度单位为度。
 */

#include "reporting/QualityReportWriter.h"

#include "filtering/SparsePointCloudProcessor.h"
#include "geometry/TriangulationQuality.h"
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
#include <unordered_set>
#include <vector>

namespace xjw::aerial_triangulation
{
    namespace
    {

        constexpr double kExportMaxReprojectionErrorPx = 4.0;

        struct SparsePublishQualityPolicy
        {
            double maxReprojectionErrorPx = 3.0;
            double minTriangulationAngleDeg = 2.0;
            double maxReconstructionUncertainty = 100.0;
            int minTrackLength = 2;
            double twoViewMaxReprojectionErrorPx = 1.5;
            double twoViewMinTriangulationAngleDeg = 3.0;
            double twoViewMaxReconstructionUncertainty = 50.0;
            int statisticalNeighbors = 16;
            double statisticalStdDevMultiplier = 3.0;
            int spatialMaxTrackLength = 1'000'000;
            bool tightenTwoViewPoints = false;
            bool filterSpatialOutliers = false;
        };

        SparsePublishQualityPolicy publishPolicyFor(const PreparedAerialTriangulationInput& input,
                                                    int registeredImageCount)
        {
            SparsePublishQualityPolicy policy;
            if (input.quality >= 2)
            {
                policy.maxReprojectionErrorPx = 1.5;
                policy.maxReconstructionUncertainty = 100.0;
                policy.twoViewMaxReprojectionErrorPx = 1.0;
                policy.twoViewMaxReconstructionUncertainty = 50.0;
                if (registeredImageCount >= 8)
                {
                    // 成熟航测网络中的两视图点在真实建筑数据上贡献了绝大多数飞点。
                    // 先用多视图几何门控清理，再仅对最低保留轨长执行空间离群删除，
                    // 避免把屋檐和立面边界等合法稀疏结构误判为飞点。
                    policy.maxReprojectionErrorPx = 1.2;
                    policy.minTriangulationAngleDeg = 7.5;
                    policy.maxReconstructionUncertainty = 30.0;
                    policy.minTrackLength = 3;
                    policy.spatialMaxTrackLength = 3;
                }
            }
            policy.tightenTwoViewPoints = registeredImageCount >= 3;
            policy.filterSpatialOutliers = registeredImageCount >= 3;
            return policy;
        }

        /**
         * @brief 计算一个三维点所有观测射线两两组合中的最大交会角。
         *
         * 最大角代表该轨迹最有利的基线组合；它用于网络诊断，不替代逐观测正深度和
         * 重投影验证。
         */
        double maximumTriangulationAngleDegrees(const SfmReconstruction& reconstruction, const ScenePoint3D& point)
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
                    const double cosine = std::clamp(
                        (rayA[0] * rayB[0] + rayA[1] * rayB[1] + rayA[2] * rayB[2]) / (normA * normB), -1.0, 1.0);
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
        QSize inferImageSize(const PreparedAerialTriangulationInput& input, const SfmReconstruction& reconstruction)
        {
            for (const QString& path : input.images)
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
                for (const FeatureKeypoint& keypoint : reconstruction.image(imageId).keypoints)
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
            std::vector<Point3DId> publishedPointIds;
            QJsonObject cleanupDiagnostics;
        };

        struct SparsePublishCandidate
        {
            Point3DId pointId = kInvalidPoint3DId;
            TrackSource source = TrackSource::FeatureMatch;
            std::vector<ImageId> cameraIds;
            QJsonObject serializedPoint;
            SparsePointCloudPoint filterPoint;
        };

        struct SparseCleanupCounters
        {
            int reconstructionPoints = 0;
            int metricReadyPoints = 0;
            int removedIncompleteMetrics = 0;
            int removedByReprojection = 0;
            int removedByTrackLength = 0;
            int removedByTriangulationAngle = 0;
            int removedByReconstructionUncertainty = 0;
            int removedWeakTwoView = 0;
            int removedSpatialOutliers = 0;
        };

        QJsonObject cleanupDiagnostics(const SparsePublishQualityPolicy& policy,
                                       const SparseCleanupCounters& counters,
                                       int publishedPoints)
        {
            return QJsonObject{
                {QStringLiteral("reconstruction_points"), counters.reconstructionPoints},
                {QStringLiteral("metric_ready_points"), counters.metricReadyPoints},
                {QStringLiteral("published_points"), publishedPoints},
                {QStringLiteral("removed_incomplete_metrics"), counters.removedIncompleteMetrics},
                {QStringLiteral("removed_by_reprojection"), counters.removedByReprojection},
                {QStringLiteral("removed_by_track_length"), counters.removedByTrackLength},
                {QStringLiteral("removed_by_triangulation_angle"), counters.removedByTriangulationAngle},
                {QStringLiteral("removed_by_reconstruction_uncertainty"), counters.removedByReconstructionUncertainty},
                {QStringLiteral("removed_weak_two_view"), counters.removedWeakTwoView},
                {QStringLiteral("removed_spatial_outliers"), counters.removedSpatialOutliers},
                {QStringLiteral("policy"),
                 QJsonObject{
                     {QStringLiteral("max_reprojection_error_px"), policy.maxReprojectionErrorPx},
                     {QStringLiteral("min_track_length"), policy.minTrackLength},
                     {QStringLiteral("min_triangulation_angle_deg"), policy.minTriangulationAngleDeg},
                     {QStringLiteral("max_reconstruction_uncertainty"), policy.maxReconstructionUncertainty},
                     {QStringLiteral("tighten_two_view_points"), policy.tightenTwoViewPoints},
                     {QStringLiteral("two_view_max_reprojection_error_px"), policy.twoViewMaxReprojectionErrorPx},
                     {QStringLiteral("two_view_min_triangulation_angle_deg"), policy.twoViewMinTriangulationAngleDeg},
                     {QStringLiteral("two_view_max_reconstruction_uncertainty"),
                      policy.twoViewMaxReconstructionUncertainty},
                     {QStringLiteral("filter_spatial_outliers"), policy.filterSpatialOutliers},
                     {QStringLiteral("statistical_neighbors"), policy.statisticalNeighbors},
                     {QStringLiteral("statistical_stddev_multiplier"), policy.statisticalStdDevMultiplier},
                     {QStringLiteral("spatial_max_track_length"), policy.spatialMaxTrackLength},
                 }},
            };
        }

        bool passesPublishGeometryGate(const SparsePublishCandidate& candidate,
                                       const SparsePublishQualityPolicy& policy,
                                       SparseCleanupCounters* counters)
        {
            const SparsePointCloudPoint& point = candidate.filterPoint;
            if (point.rmsReprojPx > policy.maxReprojectionErrorPx)
            {
                ++counters->removedByReprojection;
                return false;
            }
            if (candidate.source == TrackSource::PriorMarker)
            {
                return true;
            }
            if (point.trackLen < policy.minTrackLength)
            {
                ++counters->removedByTrackLength;
                return false;
            }
            if (point.minTriAngleDeg < policy.minTriangulationAngleDeg)
            {
                ++counters->removedByTriangulationAngle;
                return false;
            }
            if (point.reconstructionUncertainty > policy.maxReconstructionUncertainty)
            {
                ++counters->removedByReconstructionUncertainty;
                return false;
            }
            if (policy.tightenTwoViewPoints && point.trackLen == 2 &&
                (point.rmsReprojPx > policy.twoViewMaxReprojectionErrorPx ||
                 point.minTriAngleDeg < policy.twoViewMinTriangulationAngleDeg ||
                 point.reconstructionUncertainty > policy.twoViewMaxReconstructionUncertainty))
            {
                ++counters->removedWeakTwoView;
                return false;
            }
            return true;
        }

        std::unordered_set<std::size_t>
        spatiallySupportedCandidateIndices(const std::vector<SparsePublishCandidate>& candidates,
                                           const SparsePublishQualityPolicy& policy,
                                           int* removed)
        {
            std::unordered_set<std::size_t> supported;
            if (removed)
            {
                *removed = 0;
            }

            std::vector<SparsePointCloudPoint> featurePoints;
            featurePoints.reserve(candidates.size());
            for (std::size_t index = 0; index < candidates.size(); ++index)
            {
                if (candidates[index].source == TrackSource::PriorMarker)
                {
                    supported.insert(index);
                    continue;
                }
                SparsePointCloudPoint point = candidates[index].filterPoint;
                point.sourceIndex = index;
                featurePoints.push_back(point);
            }

            constexpr std::size_t kMinimumStatisticalPointCount = 8;
            if (!policy.filterSpatialOutliers || featurePoints.size() < kMinimumStatisticalPointCount)
            {
                for (const SparsePointCloudPoint& point : featurePoints)
                {
                    supported.insert(point.sourceIndex);
                }
                return supported;
            }

            SparsePointCloudFilterOptions options;
            options.filterByReprojError = false;
            options.filterByTrackLen = false;
            options.filterByTriAngle = false;
            options.filterByReconstructionUncertainty = false;
            options.filterByProjectionAccuracy = false;
            options.filterByStatistical = true;
            options.statK = policy.statisticalNeighbors;
            options.statStdDevMul = policy.statisticalStdDevMultiplier;
            options.filterByNormalConsistency = false;
            options.filterByDensity = false;
            options.processingDevice = plapoint::ProcessingDevice::CPU;
            (void)SparsePointCloudProcessor::filter(&featurePoints, options);
            std::unordered_set<std::size_t> statisticallySupported;
            statisticallySupported.reserve(featurePoints.size());
            for (const SparsePointCloudPoint& point : featurePoints)
            {
                statisticallySupported.insert(point.sourceIndex);
            }
            for (std::size_t index = 0; index < candidates.size(); ++index)
            {
                if (candidates[index].source == TrackSource::PriorMarker || statisticallySupported.contains(index) ||
                    candidates[index].filterPoint.trackLen > policy.spatialMaxTrackLength)
                {
                    supported.insert(index);
                    continue;
                }
                if (removed)
                {
                    ++*removed;
                }
            }
            return supported;
        }

        /**
         * @brief 一次遍历收集网络质量；只有正式报告路径才序列化逐点明细。
         */
        CollectedSparseQuality collectSparseQuality(const SfmReconstruction& reconstruction,
                                                    bool serializeDetails,
                                                    const SparsePublishQualityPolicy& policy = {})
        {
            CollectedSparseQuality collected;
            SparseCleanupCounters cleanupCounters;
            cleanupCounters.reconstructionPoints = static_cast<int>(reconstruction.numPoints3D());
            std::vector<SparsePublishCandidate> publishCandidates;
            if (serializeDetails)
            {
                collected.serializedPoints.emplace();
                publishCandidates.reserve(reconstruction.numPoints3D());
            }
            for (const Point3DId pointId : reconstruction.allPoint3DIds())
            {
                if (!reconstruction.hasPoint3D(pointId))
                {
                    continue;
                }
                const ScenePoint3D& point = reconstruction.point3D(pointId);
                if (!std::isfinite(point.error) || point.error < 0.0 || point.error > kExportMaxReprojectionErrorPx ||
                    point.track.length() < 2 || !std::isfinite(point.xyz[0]) || !std::isfinite(point.xyz[1]) ||
                    !std::isfinite(point.xyz[2]))
                {
                    continue;
                }

                const double triangulationAngle = maximumTriangulationAngleDegrees(reconstruction, point);
                SfmQualityPoint qualityPoint;
                qualityPoint.trackLength = static_cast<int>(point.track.length());
                qualityPoint.reprojectionErrorPx = point.error;
                qualityPoint.triangulationAngleDeg = triangulationAngle;
                std::optional<QJsonArray> observations;
                std::vector<TiePointQualityObservation> qualityObservations;
                std::vector<ImageId> qualityCameraIds;
                bool hasCompleteQualityObservations = true;
                if (serializeDetails)
                {
                    observations.emplace();
                    qualityObservations.reserve(point.track.elements.size());
                    qualityCameraIds.reserve(point.track.elements.size());
                }

                for (const TrackElement& element : point.track.elements)
                {
                    if (!reconstruction.hasImage(element.imageId))
                    {
                        hasCompleteQualityObservations = false;
                        continue;
                    }
                    const ImageData& image = reconstruction.image(element.imageId);
                    if (element.featureIdx >= image.keypoints.size())
                    {
                        hasCompleteQualityObservations = false;
                        continue;
                    }

                    const FeatureKeypoint& keypoint = image.keypoints[element.featureIdx];
                    const double measurementScale = std::isfinite(keypoint.scale) && keypoint.scale > 0.0f
                                                        ? static_cast<double>(keypoint.scale)
                                                        : 1.0;
                    qualityPoint.observations.push_back({static_cast<int>(element.imageId), keypoint.x, keypoint.y});
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
                        {QStringLiteral("scale"), measurementScale},
                    };
                    qualityCameraIds.push_back(element.imageId);

                    if (reconstruction.hasCamera(element.imageId))
                    {
                        double projected[2]{};
                        const FramePinholeCamera& camera = reconstruction.camera(element.imageId);
                        qualityObservations.push_back({&camera, measurementScale});
                        const bool projectedOk = camera.projectWorldPoint(point.xyz.data(), projected) ||
                                                 camera.projectWorldPointSigned(point.xyz.data(), projected);
                        if (projectedOk && std::isfinite(projected[0]) && std::isfinite(projected[1]))
                        {
                            const double residualX = projected[0] - keypoint.x;
                            const double residualY = projected[1] - keypoint.y;
                            observation.insert(QStringLiteral("projected_xy"), QJsonArray{projected[0], projected[1]});
                            observation.insert(QStringLiteral("residual_xy"), QJsonArray{residualX, residualY});
                            observation.insert(QStringLiteral("residual_norm_px"), std::hypot(residualX, residualY));
                        }
                    }
                    else
                    {
                        hasCompleteQualityObservations = false;
                    }
                    observations->append(observation);
                }

                if (serializeDetails)
                {
                    const double uncertainty = reconstructionUncertainty(qualityObservations, point.xyz);
                    const double accuracy = projectionAccuracy(qualityObservations);
                    if (!hasCompleteQualityObservations || qualityObservations.size() != point.track.elements.size() ||
                        !std::isfinite(uncertainty) || uncertainty <= 0.0 || !std::isfinite(accuracy) ||
                        accuracy <= 0.0)
                    {
                        ++cleanupCounters.removedIncompleteMetrics;
                        collected.qualityPoints.push_back(std::move(qualityPoint));
                        continue;
                    }

                    QJsonObject serializedPoint{
                        {QStringLiteral("track_len"), static_cast<int>(point.track.length())},
                        {QStringLiteral("rms_reproj_px"), point.error},
                        {QStringLiteral("triangulation_angle_deg"), triangulationAngle},
                        {QStringLiteral("min_tri_angle_deg"), triangulationAngle},
                        {QStringLiteral("point_xyz"), QJsonArray{point.xyz[0], point.xyz[1], point.xyz[2]}},
                        {QStringLiteral("observations"), *observations},
                        {QStringLiteral("reconstruction_uncertainty"), uncertainty},
                        {QStringLiteral("projection_accuracy"), accuracy},
                        {QStringLiteral("track_source"),
                         point.track.source == TrackSource::PriorMarker ? QStringLiteral("prior_marker")
                                                                        : QStringLiteral("feature_match")},
                    };
                    if (!point.track.sourceId.empty())
                    {
                        serializedPoint.insert(QStringLiteral("track_source_id"),
                                               QString::fromStdString(point.track.sourceId));
                    }

                    SparsePublishCandidate candidate;
                    candidate.pointId = pointId;
                    candidate.source = point.track.source;
                    candidate.cameraIds = std::move(qualityCameraIds);
                    candidate.serializedPoint = std::move(serializedPoint);
                    candidate.filterPoint.x = point.xyz[0];
                    candidate.filterPoint.y = point.xyz[1];
                    candidate.filterPoint.z = point.xyz[2];
                    candidate.filterPoint.rmsReprojPx = point.error;
                    candidate.filterPoint.minTriAngleDeg = triangulationAngle;
                    candidate.filterPoint.reconstructionUncertainty = uncertainty;
                    candidate.filterPoint.projectionAccuracy = accuracy;
                    candidate.filterPoint.trackLen = static_cast<int>(point.track.length());
                    ++cleanupCounters.metricReadyPoints;
                    if (passesPublishGeometryGate(candidate, policy, &cleanupCounters))
                    {
                        publishCandidates.push_back(std::move(candidate));
                    }
                }
                collected.qualityPoints.push_back(std::move(qualityPoint));
            }

            if (serializeDetails)
            {
                const std::unordered_set<std::size_t> spatiallySupported = spatiallySupportedCandidateIndices(
                    publishCandidates, policy, &cleanupCounters.removedSpatialOutliers);
                for (std::size_t index = 0; index < publishCandidates.size(); ++index)
                {
                    if (!spatiallySupported.contains(index))
                    {
                        continue;
                    }
                    SparsePublishCandidate& candidate = publishCandidates[index];
                    collected.serializedPoints->append(candidate.serializedPoint);
                    collected.publishedPointIds.push_back(candidate.pointId);
                    for (const ImageId imageId : candidate.cameraIds)
                    {
                        collected.cameraErrors[imageId].first += candidate.filterPoint.rmsReprojPx;
                        ++collected.cameraErrors[imageId].second;
                    }
                }
                collected.cleanupDiagnostics =
                    cleanupDiagnostics(policy, cleanupCounters, static_cast<int>(collected.publishedPointIds.size()));
            }
            return collected;
        }

        QJsonObject buildSparseQualitySummaryObject(const PreparedAerialTriangulationInput& input,
                                                    const SfmReconstruction& reconstruction,
                                                    const AerialTriangulationReconstructionResult& result,
                                                    const std::vector<SfmQualityPoint>& qualityPoints)
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
            QJsonObject sparseQuality =
                serializeSfmQualityMetrics(computeSfmQualityMetrics(qualityPoints, qualityOptions));
            sparseQuality.insert(QStringLiteral("mean_reprojection_error_px"), result.meanReprojError);
            return sparseQuality;
        }

    } // namespace

    QJsonObject QualityReportWriter::buildSparseQualitySummary(const PreparedAerialTriangulationInput& input,
                                                               const SfmReconstruction& reconstruction,
                                                               const AerialTriangulationReconstructionResult& result)
    {
        const CollectedSparseQuality collected = collectSparseQuality(reconstruction, false);
        return buildSparseQualitySummaryObject(input, reconstruction, result, collected.qualityPoints);
    }

    SparseQualityReport QualityReportWriter::build(const PreparedAerialTriangulationInput& input,
                                                   const SfmReconstruction& reconstruction,
                                                   const AerialTriangulationReconstructionResult& result)
    {
        SparseQualityReport report;
        const SparsePublishQualityPolicy publishPolicy = publishPolicyFor(input, result.numRegisteredImages);
        CollectedSparseQuality collected = collectSparseQuality(reconstruction, true, publishPolicy);
        report.points = std::move(*collected.serializedPoints);
        report.publishedPointIds = std::move(collected.publishedPointIds);

        // 第二阶段：为全部输入影像输出记录，包括未注册影像，便于 GUI 直接定位缺口。
        for (int index = 0; index < input.images.size(); ++index)
        {
            const ImageId imageId = static_cast<ImageId>(index);
            const auto error = collected.cameraErrors.find(imageId);
            const bool registered = reconstruction.isRegistered(imageId);
            const double residual = registered && error != collected.cameraErrors.end() && error->second.second > 0
                                        ? error->second.first / error->second.second
                                        : (registered ? result.meanReprojError : 0.0);
            report.perCameraResiduals.append(QJsonObject{
                {QStringLiteral("path"), input.images.at(index)},
                {QStringLiteral("registered"), registered},
                {QStringLiteral("residual_px"), residual},
            });
        }

        // 第三阶段：汇总注册覆盖、轨迹长度、交会角、误差和影像网格覆盖。
        const QJsonObject sparseQuality =
            buildSparseQualitySummaryObject(input, reconstruction, result, collected.qualityPoints);

        // 第四阶段：保留 attempt 诊断，并追加最终模型重新计算的稳定质量字段。
        report.diagnostics = result.sfmDiagnostics;
        report.diagnostics.insert(QStringLiteral("sparse_quality"), sparseQuality);
        report.diagnostics.insert(QStringLiteral("sparse_point_cleanup"), collected.cleanupDiagnostics);
        report.diagnostics.insert(QStringLiteral("ba_summary"),
                                  QJsonObject{
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
