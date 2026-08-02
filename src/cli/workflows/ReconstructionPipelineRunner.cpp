// =============================================================================
// 文件: workflows/ReconstructionPipelineRunner.cpp
// 功能: PlaScan 一键重建 CLI
//       默认: .lis(image camera) -> SFM -> MVS -> 三维模型 -> DEM/DOM
//       PLASCAN_THREE_D_ONLY: 独立三维重建流程，仅 SFM -> MVS -> 三维模型
// =============================================================================
#include "ReconstructionPipelineRunner.h"

#include "ProjectCameraIO.h"
#include "ReconstructionCliOptions.h"
#include "ReconstructionCliProgress.h"
#include "ReconstructionCliReport.h"

#include "cli_common.h"
#include "cli_photogrammetry_common.h"
#include "CliConsole.h"

#include "Camera.h"
#include "DenseCloudQualityFilter.h"
#include "DepthFrameUtils.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "preparation/MatchResultCatalog.h"
#include "ModelWorkflowService.h"
#include "MvsSceneClassifier.h"
#include "PointCloudWorkflowConfig.h"
#include "PointCloudArtifactIO.h"
#include "workflow/AerialTriangulationWorkflow.h"
#include "SparseCloudPreprocessor.h"
#include "StreamingDepthFusionService.h"
#ifndef PLASCAN_THREE_D_ONLY
#include "TerrainPipeline.h"
#endif
#include "io/PathIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectSession.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/search/kdtree.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>
#include <QtGlobal>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace
{

std::vector<xjw::mvs::MvsSourcePairQuality> loadMvsSourcePairQualities(const QString &matchDir)
{
    std::vector<xjw::mvs::MvsSourcePairQuality> qualities;
    if (matchDir.trimmed().isEmpty())
    {
        return qualities;
    }

    // `.pimatch` 是匹配结果的唯一权威来源。旧流程先读 JSON 报告、再扫描匹配
    // 文件，会把同一像对重复加入 MVS 规划器，并且报告可能落后于当前缓存。
    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = matchDir;
    const xjw::aerial_triangulation::MatchResultCatalogSummary summary =
        xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    qualities.reserve(static_cast<size_t>(summary.pairGroups.size()));
    for (const xjw::aerial_triangulation::MatchPairGroup &group : summary.pairGroups)
    {
        if (group.bestVariantIndex < 0 || group.bestVariantIndex >= group.variants.size())
        {
            continue;
        }

        const xjw::aerial_triangulation::MatchVariant &variant = group.variants.at(group.bestVariantIndex);
        if (!variant.compatible)
        {
            continue;
        }

        xjw::mvs::MvsSourcePairQuality quality;
        quality.imageA = xjw::common::io::toUtf8Path(variant.imageA);
        quality.imageB = xjw::common::io::toUtf8Path(variant.imageB);
        quality.totalMatches = std::max(0, variant.totalMatches);
        quality.geometricInliers = std::max(0, variant.geometricVerifiedInliers);
        quality.hasVerificationStatistics = variant.hasInlierStats;
        quality.verified = variant.geometryPassed;
        quality.geometricCoverage = static_cast<float>(variant.geometricCoverage);
        quality.verificationReason = quality.verified
            ? "verified_from_pimatch"
            : "pimatch_geometry_gate_failed";
        qualities.push_back(std::move(quality));
    }
    return qualities;
}

using InputItem = xjw::cli::PhotogrammetryInputItem;

QStringList criticalOutputRelativePaths()
{
    QStringList paths = {
        QStringLiteral("report.json"),
        QStringLiteral("headless.plascan"),
        QStringLiteral("sparse"),
        QStringLiteral("mvs/dense_cloud.ply"),
        QStringLiteral("model")
    };
#ifndef PLASCAN_THREE_D_ONLY
    paths << QStringLiteral("terrain/products/dem.tif")
          << QStringLiteral("terrain/products/dom.png");
#endif
    return paths;
}

using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

PlaCloud fusedPointsToPointCloud(const std::vector<xjw::mvs::FusedPoint> &cloud,
                                 bool keepColor,
                                 bool keepNormals)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(cloud.size(), 3);
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        points(row, 0) = cloud[i].x;
        points(row, 1) = cloud[i].y;
        points(row, 2) = cloud[i].z;
    }

    PlaCloud pointCloud(std::move(points));
    if (keepColor)
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            colors(row, 0) = cloud[i].r;
            colors(row, 1) = cloud[i].g;
            colors(row, 2) = cloud[i].b;
        }
        pointCloud.setColors(std::move(colors));
    }

    if (keepNormals)
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(cloud.size(), 3);
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            normals(row, 0) = cloud[i].nx;
            normals(row, 1) = cloud[i].ny;
            normals(row, 2) = cloud[i].nz;
        }
        pointCloud.setNormals(std::move(normals));
    }

    return pointCloud;
}

std::vector<xjw::mvs::FusedPoint> pointCloudToFusedPoints(const PlaCloud &cloud)
{
    std::vector<xjw::mvs::FusedPoint> points;
    points.reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        xjw::mvs::FusedPoint point;
        point.x = cloud.points().getValue(row, 0);
        point.y = cloud.points().getValue(row, 1);
        point.z = cloud.points().getValue(row, 2);

        if (cloud.hasNormals())
        {
            float nx = cloud.normals()->getValue(row, 0);
            float ny = cloud.normals()->getValue(row, 1);
            float nz = cloud.normals()->getValue(row, 2);
            const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (std::isfinite(length) && length > 1.0e-12f)
            {
                point.nx = nx / length;
                point.ny = ny / length;
                point.nz = nz / length;
            }
            else
            {
                point.nz = 1.0f;
            }
        }
        else
        {
            point.nz = 1.0f;
        }

        if (cloud.hasColors())
        {
            point.r = cloud.colors()->getValue(row, 0);
            point.g = cloud.colors()->getValue(row, 1);
            point.b = cloud.colors()->getValue(row, 2);
        }
        points.push_back(point);
    }
    return points;
}

PlaCloud cloneCloudValue(const PlaCloud &cloud, bool includeNormals = true)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(cloud.size(), 3);
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        for (int d = 0; d < 3; ++d)
        {
            points(row, d) = cloud.points()(row, d);
        }
    }

    PlaCloud copy(std::move(points));
    if (cloud.hasColors()) copy.setColors(*cloud.colors());
    if (cloud.hasIntensities()) copy.setIntensities(*cloud.intensities());
    if (includeNormals && cloud.hasNormals()) copy.setNormals(*cloud.normals());
    if (cloud.hasScalarFields()) copy.setScalarFields(cloud.scalarFieldNames(), *cloud.scalarFields());
    if (cloud.hasFaces()) copy.setFaces(*cloud.faces());
    copy.setMaterialLibraryFile(cloud.materialLibraryFile());
    copy.setTextureImageFile(cloud.textureImageFile());
    return copy;
}

std::shared_ptr<PlaCloud> cloneCloud(const PlaCloud &cloud)
{
    return std::make_shared<PlaCloud>(cloneCloudValue(cloud));
}

QString processingDeviceLabel(plapoint::ProcessingDevice device)
{
    switch (device)
    {
    case plapoint::ProcessingDevice::CPU:
        return QStringLiteral("CPU");
    case plapoint::ProcessingDevice::GPU:
        return QStringLiteral("GPU");
    case plapoint::ProcessingDevice::Auto:
        return QStringLiteral("Auto");
    }
    return QStringLiteral("Unknown");
}

void reportPlaPointDevice(const std::function<void(const QString &, int)> &progress,
                          const QString &stage,
                          const plapoint::ProcessingReport &report,
                          std::size_t beforeCount,
                          std::size_t afterCount,
                          int percent)
{
    QString message = QStringLiteral("%1 [%2→%3, usedDevice=%4]")
        .arg(stage)
        .arg(beforeCount)
        .arg(afterCount)
        .arg(processingDeviceLabel(report.usedDevice));
    if (report.usedFallback)
    {
        message += QStringLiteral(" fallback=%1")
            .arg(QString::fromStdString(report.fallbackReason));
    }
    if (progress)
    {
        progress(message, percent);
    }
    std::fprintf(stdout, "%s\n", qUtf8Printable(message));
    std::fflush(stdout);
}

PlaCloud sorFilter(const PlaCloud &cloud,
                   int k,
                   float stdRatio,
                   plapoint::ProcessingDevice processingDevice,
                   plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() < static_cast<std::size_t>(k + 1))
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: point count is smaller than k + 1";
        }
        return cloneCloudValue(cloud);
    }
    return plapoint::statisticalOutlierRemoval(cloud, k, stdRatio, processingDevice, nullptr, report);
}

PlaCloud radiusFilter(const PlaCloud &cloud,
                      float radius,
                      int minNeighbors,
                      plapoint::ProcessingDevice processingDevice,
                      plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() == 0)
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: empty cloud";
        }
        return PlaCloud(0);
    }
    return plapoint::radiusOutlierRemoval(cloud, radius, minNeighbors, processingDevice, nullptr, report);
}

PlaCloud voxelDownsample(const PlaCloud &cloud,
                         float leafSize,
                         plapoint::ProcessingDevice processingDevice,
                         plapoint::ProcessingReport *report = nullptr)
{
    if (cloud.size() == 0 || leafSize <= 0.0f)
    {
        if (report)
        {
            report->requestedDevice = processingDevice;
            report->usedDevice = plapoint::ProcessingDevice::CPU;
            report->usedFallback = false;
            report->fallbackReason = "skipped: empty cloud or invalid leaf size";
        }
        return cloneCloudValue(cloud);
    }
    return plapoint::voxelDownsample(cloud, leafSize, processingDevice, report);
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> estimateNormals(
    const PlaCloud &cloud,
    int normalK,
    plapoint::ProcessingDevice processingDevice,
    plapoint::ProcessingReport *report = nullptr)
{
    return plapoint::estimateNormals(cloud, normalK, processingDevice, report);
}

float adaptivePreSorVoxelSize(const PlaCloud &cloud, float minimumLeafSize)
{
    if (cloud.size() == 0)
    {
        return minimumLeafSize;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const float x = cloud.points()(row, 0);
        const float y = cloud.points()(row, 1);
        const float z = cloud.points()(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }
        minX = std::min(minX, x);
        minY = std::min(minY, y);
        minZ = std::min(minZ, z);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
        maxZ = std::max(maxZ, z);
    }

    if (minX > maxX || minY > maxY || minZ > maxZ)
    {
        return minimumLeafSize;
    }
    const double dx = static_cast<double>(maxX - minX);
    const double dy = static_cast<double>(maxY - minY);
    const double dz = static_cast<double>(maxZ - minZ);
    const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!std::isfinite(diag) || diag <= 0.0)
    {
        return minimumLeafSize;
    }
    return std::max(minimumLeafSize, static_cast<float>(diag / 4096.0));
}

float adaptivePreSorVoxelSize(const std::vector<xjw::mvs::FusedPoint> &cloud,
                              float minimumLeafSize)
{
    if (cloud.empty())
    {
        return minimumLeafSize;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const auto &point : cloud)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        minZ = std::min(minZ, point.z);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
        maxZ = std::max(maxZ, point.z);
    }

    if (minX > maxX || minY > maxY || minZ > maxZ)
    {
        return minimumLeafSize;
    }

    const double dx = static_cast<double>(maxX - minX);
    const double dy = static_cast<double>(maxY - minY);
    const double dz = static_cast<double>(maxZ - minZ);
    const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!std::isfinite(diag) || diag <= 0.0)
    {
        return minimumLeafSize;
    }
    return std::max(minimumLeafSize, static_cast<float>(diag / 4096.0));
}

std::vector<xjw::mvs::FusedPoint> voxelDownsampleFusedPoints(
    const std::vector<xjw::mvs::FusedPoint> &cloud,
    float leafSize)
{
    if (cloud.empty() || leafSize <= 0.0f)
    {
        return cloud;
    }

    std::vector<xjw::mvs::FusedPoint> finiteCloud;
    finiteCloud.reserve(cloud.size());
    for (const auto &point : cloud)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }
        finiteCloud.push_back(point);
    }
    if (finiteCloud.empty())
    {
        return {};
    }

    PlaCloud input = fusedPointsToPointCloud(finiteCloud, true, true);
    PlaCloud output = plapoint::voxelDownsample(input, leafSize);
    return pointCloudToFusedPoints(output);
}

struct FusedVoxelDownsampleResult
{
    std::vector<xjw::mvs::FusedPoint> points;
    float leafSize = 0.0f;
    int passes = 0;
};

FusedVoxelDownsampleResult voxelDownsampleFusedPointsToTarget(
    const std::vector<xjw::mvs::FusedPoint> &cloud,
    float initialLeafSize,
    std::size_t targetPoints)
{
    FusedVoxelDownsampleResult result;
    result.leafSize = initialLeafSize;
    if (cloud.empty() || initialLeafSize <= 0.0f || targetPoints == 0)
    {
        result.points = cloud;
        return result;
    }

    result.points = voxelDownsampleFusedPoints(cloud, result.leafSize);
    result.passes = 1;

    constexpr int kMaxPasses = 6;
    while (result.points.size() > targetPoints && result.passes < kMaxPasses)
    {
        const double ratio = static_cast<double>(result.points.size())
            / static_cast<double>(targetPoints);
        float nextLeafSize = static_cast<float>(static_cast<double>(result.leafSize)
            * std::sqrt(std::max(1.0, ratio)) * 1.05);
        if (!(nextLeafSize > result.leafSize))
        {
            nextLeafSize = result.leafSize * 2.0f;
        }
        result.leafSize = nextLeafSize;
        result.points = voxelDownsampleFusedPoints(cloud, result.leafSize);
        ++result.passes;
    }

    return result;
}

xjw::mvs::TerrainHeightSpikeFilterOptions terrainSpikeOptionsFromRequest(
    const xjw::core::project::DenseRefineSettings &request)
{
    xjw::mvs::TerrainHeightSpikeFilterOptions options;
    options.enabled = request.terrainSpikeFilterEnabled;
    options.gridResolution = request.terrainSpikeGridResolution;
    options.minCellPoints = request.terrainSpikeMinCellPoints;
    options.minHeightThreshold = static_cast<float>(request.terrainSpikeMinHeightThreshold);
    options.madMultiplier = static_cast<float>(request.terrainSpikeMadMultiplier);
    return options;
}

QJsonObject terrainSpikeReportToJson(const xjw::mvs::TerrainHeightSpikeFilterReport &report)
{
    return QJsonObject{
        {QStringLiteral("input_points"), static_cast<double>(report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(report.outputPoints)},
        {QStringLiteral("removed_points"), static_cast<double>(report.removedPoints)},
        {QStringLiteral("median_cell_z_range_before"), report.medianCellZRangeBefore},
        {QStringLiteral("p95_cell_z_range_before"), report.p95CellZRangeBefore},
        {QStringLiteral("median_cell_z_range_after"), report.medianCellZRangeAfter},
        {QStringLiteral("p95_cell_z_range_after"), report.p95CellZRangeAfter}
    };
}

PlaCloud refineDenseCloud(PlaCloud cloud,
                          const xjw::core::project::DenseRefineSettings &request,
                          const std::function<void(const QString &, int)> &progress,
                          xjw::mvs::TerrainHeightSpikeFilterReport *terrainSpikeReport = nullptr)
{
    if (request.sorEnabled)
    {
        if (progress) progress(QStringLiteral("统计离群点移除 (SOR)..."), 20);
        const auto beforeSor = cloud.size();
        plapoint::ProcessingReport sorReport;
        cloud = sorFilter(cloud,
                          request.sorK,
                          static_cast<float>(request.sorStdDev),
                          request.processingDevice,
                          &sorReport);
        reportPlaPointDevice(progress, QStringLiteral("统计离群点移除 (SOR)"),
                             sorReport, beforeSor, cloud.size(), 22);

        if (cloud.size() > 64)
        {
            float minX = 1e30f;
            float minY = 1e30f;
            float minZ = 1e30f;
            float maxX = -1e30f;
            float maxY = -1e30f;
            float maxZ = -1e30f;
            for (std::size_t i = 0; i < cloud.size(); ++i)
            {
                const auto row = static_cast<plamatrix::Index>(i);
                const float x = cloud.points()(row, 0);
                const float y = cloud.points()(row, 1);
                const float z = cloud.points()(row, 2);
                minX = std::min(minX, x);
                minY = std::min(minY, y);
                minZ = std::min(minZ, z);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
                maxZ = std::max(maxZ, z);
            }

            const double dx = static_cast<double>(maxX - minX);
            const double dy = static_cast<double>(maxY - minY);
            const double dz = static_cast<double>(maxZ - minZ);
            const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double volume = std::max(dx * dy * dz, 1e-12);
            const double density = static_cast<double>(cloud.size()) / volume;
            const int radiusMinNeighbors = std::clamp(request.sorK / 2, 6, 32);
            double adaptiveRadius = std::cbrt(std::max(1.0, static_cast<double>(radiusMinNeighbors))
                                              / std::max(density, 1e-12));
            adaptiveRadius *= 1.2;
            if (diag > 1e-9)
            {
                const double radiusMin = std::max(diag * 0.001, 1e-4);
                const double radiusMax = std::max(radiusMin * 2.0, diag * 0.08);
                adaptiveRadius = std::clamp(adaptiveRadius, radiusMin, radiusMax);
            }

            if (progress) progress(QStringLiteral("半径离群点移除..."), 35);
            const auto beforeRadius = cloud.size();
            plapoint::ProcessingReport radiusReport;
            cloud = radiusFilter(cloud,
                                 static_cast<float>(adaptiveRadius),
                                 radiusMinNeighbors,
                                 request.processingDevice,
                                 &radiusReport);
            reportPlaPointDevice(progress, QStringLiteral("半径离群点移除"),
                                 radiusReport, beforeRadius, cloud.size(), 37);

        }
    }

    if (request.voxelEnabled && request.voxelSize > 0.0)
    {
        if (progress) progress(QStringLiteral("体素下采样..."), 50);
        const auto beforeVoxel = cloud.size();
        plapoint::ProcessingReport voxelReport;
        cloud = voxelDownsample(cloud,
                                static_cast<float>(request.voxelSize),
                                request.processingDevice,
                                &voxelReport);
        reportPlaPointDevice(progress, QStringLiteral("体素下采样"),
                             voxelReport, beforeVoxel, cloud.size(), 52);
    }

    if (request.terrainSpikeFilterEnabled)
    {
        if (progress) progress(QStringLiteral("局部高度突刺过滤..."), 60);
        const auto beforeTerrainFilter = cloud.size();
        xjw::mvs::TerrainHeightSpikeFilterReport localReport;
        cloud = xjw::mvs::filterTerrainHeightSpikes(
            cloud,
            terrainSpikeOptionsFromRequest(request),
            &localReport);
        if (terrainSpikeReport)
        {
            *terrainSpikeReport = localReport;
        }
        if (progress)
        {
            progress(QStringLiteral("局部高度突刺过滤: %1 -> %2 点，移除 %3 点")
                         .arg(beforeTerrainFilter)
                         .arg(cloud.size())
                         .arg(localReport.removedPoints),
                     62);
        }
    }

    if (request.normalsEnabled)
    {
        if (progress) progress(QStringLiteral("估计法向量..."), 70);
        plapoint::ProcessingReport normalReport;
        auto normals = estimateNormals(cloud, request.normalK, request.processingDevice, &normalReport);
        cloud.setNormals(std::move(normals));
        reportPlaPointDevice(progress, QStringLiteral("估计法向量"),
                             normalReport, cloud.size(), cloud.size(), 72);
    }

    return cloud;
}

bool loadCvMatStorage(const QString &path, cv::Mat *matrix, QString *error)
{
    const xjw::common::OperationResult result =
        xjw::core::project::loadDepthMatStorage(path, matrix);
    if (!result.ok && error)
    {
        *error = result.errorMessage;
    }
    return result.ok;
}

bool loadFusionFrameFromDepthMap(const QString &mvsDir,
                                 const std::vector<xjw::mvs::CameraView> &views,
                                 const xjw::mvs::FusionConfig &fusionConfig,
                                 int frameIndex,
                                 int fusionMaxImageDim,
                                 xjw::mvs::FusionFrameInput *frame,
                                 QString *error)
{
    if (!frame)
    {
        if (error) *error = QStringLiteral("内部错误：融合帧输出为空");
        return false;
    }
    if (frameIndex < 0 || frameIndex >= static_cast<int>(views.size()))
    {
        if (error) *error = QStringLiteral("融合帧索引越界: %1").arg(frameIndex);
        return false;
    }

    const QDir dir(mvsDir);
    const QString depthPngPath = dir.filePath(QStringLiteral("depth_%1.png").arg(frameIndex));
    const QString depthPath = xjw::core::project::rawDepthStoragePath(depthPngPath);

    cv::Mat depth;
    if (!loadCvMatStorage(depthPath, &depth, error))
    {
        return false;
    }

    cv::Mat confidence;
    const QString confPath = xjw::core::project::rawConfidenceStoragePath(depthPngPath);
    if (QFileInfo::exists(confPath))
    {
        QString confError;
        if (!loadCvMatStorage(confPath, &confidence, &confError))
        {
            std::fprintf(stderr, "  [MVS] 置信图读取失败，继续使用深度图: %s\n", qUtf8Printable(confError));
        }
    }

    const xjw::mvs::CameraView &view = views[static_cast<std::size_t>(frameIndex)];
    frame->sourceCamera = view.camera;
    frame->cameraModel = view.camera.normalizedForPositiveDepth();
    frame->cameraModel.setDistortion(xjw::Camera::Distortion{});
    frame->imgW = depth.cols;
    frame->imgH = depth.rows;
    frame->imagePath = view.imagePath;
    frame->depthMap = std::move(depth);
    frame->confidence = std::move(confidence);
    const bool downsampled = xjw::core::project::downsampleFusionFrameForMaxDimension(
        frame,
        fusionMaxImageDim);
    if (downsampled)
    {
        std::fprintf(stdout,
                     "  [MVS] 融合帧 %d 降采样到 %dx%d (maxDim=%d)\n",
                     frameIndex,
                     frame->imgW,
                     frame->imgH,
                     fusionMaxImageDim);
        std::fflush(stdout);
    }

    const xjw::mvs::DepthPostProcessStats postprocessStats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
            frame->depthMap,
            frame->confidence,
            fusionConfig,
            frameIndex,
            static_cast<int>(views.size()));

    frame->depthPostprocess = postprocessStats;
    return true;
}


bool fuseDepthMapsStreamingFromDisk(const QString &mvsDir,
                                    const std::vector<xjw::mvs::CameraView> &views,
                                    const xjw::core::project::DenseGenerationSettings &denseSettings,
                                    const xjw::mvs::DepthGenConfig &depthConfig,
                                    std::vector<xjw::mvs::FusedPoint> *fusedCloud,
                                    std::vector<xjw::mvs::DepthPostProcessStats> *depthPostprocessStats,
                                    QString *error)
{
    if (!fusedCloud || !depthPostprocessStats)
    {
        if (error)
        {
            *error = QStringLiteral("内部错误：MVS 流式融合输出为空");
        }
        return false;
    }

    xjw::mvs::StreamingDepthFusionConfig config;
    config.minConsistentViews = denseSettings.minConsistentViews;
    config.depthConsistency = denseSettings.depthConsistency;
    config.workerCount = denseSettings.threads;
    config.neighborCount = std::min(
        static_cast<int>(views.size()) - 1,
        std::clamp(std::max(4, denseSettings.minViews * 3), 2, 16));

    const xjw::mvs::FusionFrameLoader frameLoader =
        [&](int frameIndex, xjw::mvs::FusionFrameInput *frame, std::string *loaderError) {
            QString frameError;
            const bool ok = loadFusionFrameFromDepthMap(mvsDir,
                                                        views,
                                                        depthConfig.fusion,
                                                        frameIndex,
                                                        denseSettings.fusionMaxImageDim,
                                                        frame,
                                                        &frameError);
            if (!ok && loaderError)
            {
                *loaderError = frameError.toUtf8().toStdString();
            }
            return ok;
        };
    const xjw::mvs::FusedCloudReducer cloudReducer =
        [](std::vector<xjw::mvs::FusedPoint> *points) {
            if (!points || points->empty())
            {
                return;
            }
            constexpr std::size_t kTargetPoints = 1000000;
            const float leaf = adaptivePreSorVoxelSize(*points, 0.005f);
            const std::size_t before = points->size();
            FusedVoxelDownsampleResult downsample =
                voxelDownsampleFusedPointsToTarget(*points, leaf, kTargetPoints);
            if (!downsample.points.empty())
            {
                *points = std::move(downsample.points);
                std::fprintf(stdout,
                             "  [MVS  90%%] 流式融合预聚合 leaf=%.6f passes=%d points=%zu->%zu\n",
                             downsample.leafSize,
                             downsample.passes,
                             before,
                             points->size());
                std::fflush(stdout);
            }
        };
    const xjw::mvs::FusionProgress progress =
        [](const std::string &stage, int percent) {
            xjw::cli::printScopedProgress(
                QStringLiteral("MVS"), percent, xjw::cli::fromStdString(stage));
        };

    xjw::mvs::StreamingDepthFusionResult result;
    std::string fusionError;
    if (!xjw::mvs::fuseDepthMapsStreaming(static_cast<int>(views.size()),
                                          config,
                                          frameLoader,
                                          &result,
                                          &fusionError,
                                          progress,
                                          cloudReducer))
    {
        if (error)
        {
            *error = QString::fromUtf8(fusionError);
        }
        return false;
    }

    *fusedCloud = std::move(result.points);
    *depthPostprocessStats = std::move(result.depthPostprocessStats);
    return true;
}

QJsonObject depthPostprocessStatsToJson(const std::vector<xjw::mvs::FusionFrameInput> &frames)
{
    qint64 validBefore = 0;
    qint64 validAfterConfidence = 0;
    qint64 confidenceRemoved = 0;
    qint64 localDepthOutlierRemoved = 0;
    qint64 speckleRemoved = 0;
    qint64 edgeConfidenceRemoved = 0;
    qint64 geomConsistencyRemoved = 0;
    qint64 validAfter = 0;
    QJsonArray perFrame;

    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        const xjw::mvs::DepthPostProcessStats &stats = frames[index].depthPostprocess;
        validBefore += stats.validBeforePostprocess;
        validAfterConfidence += stats.validAfterConfidenceFilter;
        confidenceRemoved += stats.confidenceRemoved;
        localDepthOutlierRemoved += stats.localDepthOutlierRemoved;
        speckleRemoved += stats.speckleRemoved;
        edgeConfidenceRemoved += stats.edgeConfidenceRemoved;
        geomConsistencyRemoved += stats.geomConsistencyRemoved;
        validAfter += stats.validAfterPostprocess;

        perFrame.append(QJsonObject{
            {QStringLiteral("index"), static_cast<int>(index)},
            {QStringLiteral("valid_before"), stats.validBeforePostprocess},
            {QStringLiteral("valid_after_confidence"), stats.validAfterConfidenceFilter},
            {QStringLiteral("confidence_removed"), stats.confidenceRemoved},
            {QStringLiteral("local_depth_outlier_removed"), stats.localDepthOutlierRemoved},
            {QStringLiteral("speckle_removed"), stats.speckleRemoved},
            {QStringLiteral("edge_confidence_removed"), stats.edgeConfidenceRemoved},
            {QStringLiteral("geom_consistency_removed"), stats.geomConsistencyRemoved},
            {QStringLiteral("valid_after"), stats.validAfterPostprocess},
            {QStringLiteral("effective_confidence_threshold"), stats.effectiveConfidenceThreshold}
        });
    }

    return QJsonObject{
        {QStringLiteral("frames"), static_cast<int>(frames.size())},
        {QStringLiteral("valid_before"), static_cast<double>(validBefore)},
        {QStringLiteral("valid_after_confidence"), static_cast<double>(validAfterConfidence)},
        {QStringLiteral("confidence_removed"), static_cast<double>(confidenceRemoved)},
        {QStringLiteral("local_depth_outlier_removed"), static_cast<double>(localDepthOutlierRemoved)},
        {QStringLiteral("speckle_removed"), static_cast<double>(speckleRemoved)},
        {QStringLiteral("edge_confidence_removed"), static_cast<double>(edgeConfidenceRemoved)},
        {QStringLiteral("geom_consistency_removed"), static_cast<double>(geomConsistencyRemoved)},
        {QStringLiteral("valid_after"), static_cast<double>(validAfter)},
        {QStringLiteral("per_frame"), perFrame}
    };
}

QJsonObject depthPostprocessStatsToJson(const std::vector<xjw::mvs::DepthPostProcessStats> &statsByFrame)
{
    qint64 validBefore = 0;
    qint64 validAfterConfidence = 0;
    qint64 confidenceRemoved = 0;
    qint64 localDepthOutlierRemoved = 0;
    qint64 speckleRemoved = 0;
    qint64 edgeConfidenceRemoved = 0;
    qint64 geomConsistencyRemoved = 0;
    qint64 validAfter = 0;
    QJsonArray perFrame;

    for (std::size_t index = 0; index < statsByFrame.size(); ++index)
    {
        const xjw::mvs::DepthPostProcessStats &stats = statsByFrame[index];
        validBefore += stats.validBeforePostprocess;
        validAfterConfidence += stats.validAfterConfidenceFilter;
        confidenceRemoved += stats.confidenceRemoved;
        localDepthOutlierRemoved += stats.localDepthOutlierRemoved;
        speckleRemoved += stats.speckleRemoved;
        edgeConfidenceRemoved += stats.edgeConfidenceRemoved;
        geomConsistencyRemoved += stats.geomConsistencyRemoved;
        validAfter += stats.validAfterPostprocess;

        perFrame.append(QJsonObject{
            {QStringLiteral("index"), static_cast<int>(index)},
            {QStringLiteral("valid_before"), stats.validBeforePostprocess},
            {QStringLiteral("valid_after_confidence"), stats.validAfterConfidenceFilter},
            {QStringLiteral("confidence_removed"), stats.confidenceRemoved},
            {QStringLiteral("local_depth_outlier_removed"), stats.localDepthOutlierRemoved},
            {QStringLiteral("speckle_removed"), stats.speckleRemoved},
            {QStringLiteral("edge_confidence_removed"), stats.edgeConfidenceRemoved},
            {QStringLiteral("geom_consistency_removed"), stats.geomConsistencyRemoved},
            {QStringLiteral("valid_after"), stats.validAfterPostprocess},
            {QStringLiteral("effective_confidence_threshold"), stats.effectiveConfidenceThreshold}
        });
    }

    return QJsonObject{
        {QStringLiteral("frames"), static_cast<int>(statsByFrame.size())},
        {QStringLiteral("valid_before"), static_cast<double>(validBefore)},
        {QStringLiteral("valid_after_confidence"), static_cast<double>(validAfterConfidence)},
        {QStringLiteral("confidence_removed"), static_cast<double>(confidenceRemoved)},
        {QStringLiteral("local_depth_outlier_removed"), static_cast<double>(localDepthOutlierRemoved)},
        {QStringLiteral("speckle_removed"), static_cast<double>(speckleRemoved)},
        {QStringLiteral("edge_confidence_removed"), static_cast<double>(edgeConfidenceRemoved)},
        {QStringLiteral("geom_consistency_removed"), static_cast<double>(geomConsistencyRemoved)},
        {QStringLiteral("valid_after"), static_cast<double>(validAfter)},
        {QStringLiteral("per_frame"), perFrame}
    };
}

void limitMvsInputsForRegression(std::vector<xjw::mvs::CameraView> *views,
                                 QStringList *registeredImagePaths,
                                 QJsonArray *imageMetaArray,
                                 int maxFrames)
{
    if (!views || !registeredImagePaths || !imageMetaArray || maxFrames <= 0)
    {
        return;
    }

    const int limit = std::max(2, maxFrames);
    if (static_cast<int>(views->size()) <= limit)
    {
        return;
    }

    views->resize(static_cast<std::size_t>(limit));
    while (registeredImagePaths->size() > limit)
    {
        registeredImagePaths->removeLast();
    }

    QJsonArray limited;
    const int metaCount = std::min(limit, static_cast<int>(imageMetaArray->size()));
    for (int index = 0; index < metaCount; ++index)
    {
        limited.append(imageMetaArray->at(index));
    }
    *imageMetaArray = limited;
}

QJsonObject mvsSettingsToJson(const xjw::core::project::DenseGenerationSettings &denseSettings,
                              int requestedMaxFrames,
                              int mvsInputFrames,
                              int registeredImageCount)
{
    QJsonObject settings{
        {QStringLiteral("res_scale"), denseSettings.resScale},
        {QStringLiteral("iterations"), denseSettings.iterations},
        {QStringLiteral("threads"), denseSettings.threads},
        {QStringLiteral("gpu_frame_workers"), denseSettings.gpuFrameWorkers},
        {QStringLiteral("cpu_frame_workers"), denseSettings.cpuFrameWorkers},
        {QStringLiteral("patch_size"), denseSettings.patchSize},
        {QStringLiteral("min_views"), denseSettings.minViews},
        {QStringLiteral("patchmatch_confidence"), denseSettings.patchMatchConfidence},
        {QStringLiteral("fusion_min_confidence"), denseSettings.fusionMinConfidence},
        {QStringLiteral("min_consistent_views"), denseSettings.minConsistentViews},
        {QStringLiteral("depth_consistency"), denseSettings.depthConsistency},
        {QStringLiteral("max_reproj_error"), denseSettings.maxReprojError},
        {QStringLiteral("use_cuda"), denseSettings.useCuda},
        {QStringLiteral("requested_max_frames"), requestedMaxFrames},
        {QStringLiteral("mvs_input_frames"), mvsInputFrames},
        {QStringLiteral("registered_image_count"), registeredImageCount}
    };
    settings[QStringLiteral("fusion_max_image_dim")] = denseSettings.fusionMaxImageDim;
    return settings;
}

QJsonObject mvsDepthConfigToJson(const xjw::mvs::DepthGenConfig &config)
{
    return QJsonObject{
        {QStringLiteral("num_source_views"), config.numSourceViews},
        {QStringLiteral("gpu_frame_worker_count"), config.gpuFrameWorkerCount},
        {QStringLiteral("cpu_frame_worker_count"), config.cpuFrameWorkerCount},
        {QStringLiteral("cpu_worker_count"), config.cpuWorkerCount},
        {QStringLiteral("downsample_factor"), config.patchMatch.downsampleFactor},
        {QStringLiteral("patchmatch_iterations"), config.patchMatch.numIterations},
        {QStringLiteral("patch_half"), config.patchMatch.patchHalf},
        {QStringLiteral("patchmatch_confidence"), config.patchMatch.confidenceThresh},
        {QStringLiteral("fusion_confidence"), config.fusion.confidenceThresh},
        {QStringLiteral("min_consistent_views"), config.fusion.minConsistentViews},
        {QStringLiteral("adaptive_depth_cache_memory"), config.adaptiveDepthCacheMemory},
        {QStringLiteral("max_depth_cache_ram_fraction"), config.maxDepthCacheRamFraction},
        {QStringLiteral("min_free_ram_bytes"), static_cast<double>(config.minFreeRamBytes)}
    };
}

QString domOutputPath(const QJsonObject &dom)
{
    QString path = dom.value(QStringLiteral("dom_png")).toString();
    if (path.isEmpty())
    {
        path = dom.value(QStringLiteral("output_path")).toString();
    }
    return path;
}

} // namespace

int runReconstructionPipelineCliImpl(int argc, char *argv[])
{
    QCoreApplication qtApp(argc, argv);
    (void)xjw::cli::registerConsoleLogger();

#ifdef PLASCAN_THREE_D_ONLY
    CLI::App app{"PlaScan GUI-equivalent 3D reconstruction pipeline"};
#else
    CLI::App app{"PlaScan GUI-equivalent reconstruction pipeline"};
#endif
xjw::cli::ReconstructionCliOptions options;
    options.addTo(app);
    CLI11_PARSE(app, argc, argv);
    options.normalize();

    auto &listPathArg = options.listPathArg;
    auto &outputDirArg = options.outputDirArg;
    auto &device = options.device;
    auto &sfmMatchingAlgorithmId = options.sfmMatchingAlgorithmId;
    auto &sfmGuidedRematching = options.sfmGuidedRematching;
    auto &sfmLightGlueEnginePath = options.sfmLightGlueEnginePath;
    auto &sfmLoMaRTensorRtPackagePath = options.sfmLoMaRTensorRtPackagePath;
    auto &lockInputCameraPoses = options.lockInputCameraPoses;
    auto &quality = options.quality;
    auto &threads = options.threads;
    auto &cudaParallelPairs = options.cudaParallelPairs;
    auto &featureMaxImageDim = options.featureMaxImageDim;
    auto &mvs_quality = options.mvsQuality;
    auto &mvs_scene_profile = options.mvsSceneProfile;
    auto &mvs_depth_filter = options.mvsDepthFilter;
    auto &mvs_mask_dir_arg = options.mvsMaskDirArg;
    auto &mvs_save_levels = options.mvsSaveLevels;
    auto &mvsTwoSourceGrowth = options.mvsTwoSourceGrowth;
    auto &mvsTwoSourceGrowthDistance = options.mvsTwoSourceGrowthDistance;
    auto &mvsTwoSourceGrowthSpread = options.mvsTwoSourceGrowthSpread;
    auto &mvsTwoSourceGrowthNormalAngle = options.mvsTwoSourceGrowthNormalAngle;
    auto &mvsTwoSourceGrowthMaximumArea = options.mvsTwoSourceGrowthMaximumArea;
    auto &mvsResScale = options.mvsResScale;
    auto &mvsIterations = options.mvsIterations;
    auto &mvsConfidence = options.mvsConfidence;
    auto &mvsFusionConfidence = options.mvsFusionConfidence;
    auto &mvsGpuFrameWorkers = options.mvsGpuFrameWorkers;
    auto &mvsCpuFrameWorkers = options.mvsCpuFrameWorkers;
    auto &mvsMaxFrames = options.mvsMaxFrames;
    auto &mvsFusionMaxImageDim = options.mvsFusionMaxImageDim;
#ifndef PLASCAN_THREE_D_ONLY
    auto &demResolution = options.demResolution;
#endif
    auto &meshResolution = options.meshResolution;
    auto &skipModel = options.skipModel;
    auto &skipMesh = options.skipMesh;
    auto &stopAfterSfm = options.stopAfterSfm;
    auto &skipMvs = options.skipMvs;
    auto &mvsDepthOnly = options.mvsDepthOnly;
    auto &skipTerrain = options.skipTerrain;
    auto &exportObj = options.exportObj;
    auto &forceOutput = options.forceOutput;


    const QString listPath = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(listPathArg));
    const QString outputDir = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(outputDirArg));
    const QString mvs_mask_dir = mvs_mask_dir_arg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(mvs_mask_dir_arg));
    QString error;
    if (!mvs_mask_dir.isEmpty() && !QFileInfo(mvs_mask_dir).isDir())
    {
        std::fprintf(stderr,
                     "MVS 蒙版目录不存在或不是目录: %s\n",
                     qUtf8Printable(QDir::toNativeSeparators(mvs_mask_dir)));
        return cli::EXIT_ARG_ERR;
    }
    xjw::cli::OutputDirectoryPolicy outputPolicy;
    outputPolicy.allowNonEmpty = forceOutput;
    outputPolicy.protectedRelativePaths = criticalOutputRelativePaths();
    if (!xjw::cli::validateOutputDirectory(outputDir, outputPolicy, &error))
    {
        std::fprintf(stderr, "输出目录错误: %s\n", qUtf8Printable(error));
        return cli::EXIT_ARG_ERR;
    }
    if (!QDir().mkpath(outputDir))
    {
        std::fprintf(stderr, "输出目录创建失败: %s\n", qUtf8Printable(outputDir));
        return cli::EXIT_IO_ERR;
    }

    std::vector<InputItem> items;
    xjw::cli::PhotogrammetryListOptions listOptions;
    listOptions.allowImageOnlyRows = false;
    listOptions.loadCameras = true;
    listOptions.requireExistingImages = true;
    listOptions.requireExistingCameras = true;
    if (!xjw::cli::readPhotogrammetryImageList(listPath, listOptions, &items, &error))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(error));
        return cli::EXIT_ARG_ERR;
    }

    const QStringList images = xjw::cli::imagePaths(items);
    const QStringList cameraPaths = xjw::cli::cameraPathsForService(items);
    const QString projectPath =
        QDir(outputDir).filePath(QStringLiteral("headless.plascan"));
    xjw::common::project::ProjectSession projectSession;
    if (!projectSession.openOrCreate(
            projectPath,
            QFileInfo(projectPath).completeBaseName(),
            &error))
    {
        std::fprintf(stderr,
                     "工程打开/创建失败: %s\n",
                     qUtf8Printable(error));
        return cli::EXIT_IO_ERR;
    }
    if (!options.chunkIdArg.empty() && !options.chunkNameArg.empty())
    {
        std::fprintf(stderr,
                     "错误: --chunk-id 与 --chunk-name 不能同时使用\n");
        return cli::EXIT_ARG_ERR;
    }
    if (!projectSession.selectChunk(
            xjw::cli::fromStdString(options.chunkIdArg),
            xjw::cli::fromStdString(options.chunkNameArg),
            &error))
    {
        std::fprintf(stderr,
                     "Chunk 选择失败: %s\n",
                     qUtf8Printable(error));
        return cli::EXIT_IO_ERR;
    }
    if (!projectSession.mergeImages(
            xjw::cli::inputItemsToJson(items), &error)
        || !projectSession.save(&error))
    {
        std::fprintf(stderr,
                     "工程影像初始化失败: %s\n",
                     qUtf8Printable(error));
        return cli::EXIT_IO_ERR;
    }
    const QString pipelineRoot =
        QDir(projectSession.activeChunkRoot())
            .filePath(QStringLiteral("reconstruction"));
    const QString reportsRoot =
        QDir(projectSession.activeChunkRoot())
            .filePath(QStringLiteral("reports/reconstruction_pipeline"));
    if (!QDir().mkpath(pipelineRoot))
    {
        std::fprintf(stderr,
                     "Chunk 重建目录创建失败: %s\n",
                     qUtf8Printable(pipelineRoot));
        return cli::EXIT_IO_ERR;
    }
    QJsonObject projectMeta = projectSession.mergedMetadata();

    QJsonObject report;
    report[QStringLiteral("status")] = QStringLiteral("running");
    report[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    report[QStringLiteral("list_file")] = listPath;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("project_path")] = projectPath;
    report[QStringLiteral("chunk_directory")] =
        projectSession.activeChunk().directory;
    report[QStringLiteral("chunk_root")] =
        projectSession.activeChunkRoot();
    report[QStringLiteral("inputs")] = xjw::cli::inputPairsToJson(items);
    if (!mvs_mask_dir.isEmpty())
    {
        report[QStringLiteral("mvs_mask_dir")] = mvs_mask_dir;
    }

    auto writeFinalReport = [&](QJsonObject *finalReport) {
        QString reportError;
        if (!xjw::cli::writeReconstructionReport(
                reportsRoot, report, finalReport, &reportError))
        {
            std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(reportError));
            return false;
        }
        QJsonObject reportRecord = report;
        reportRecord[QStringLiteral("kind")] =
            QStringLiteral("reconstruction_pipeline_cli");
        reportRecord[QStringLiteral("path")] =
            finalReport->value(QStringLiteral("report_json")).toString();
        projectSession.upsertResultByPath(
            QStringLiteral("report_results"),
            QStringLiteral("path"),
            reportRecord);
        if (!projectSession.save(&reportError))
        {
            std::fprintf(stderr,
                         "报告已生成，但 Chunk 写回失败: %s\n",
                         qUtf8Printable(reportError));
            return false;
        }
        return true;
    };

#ifdef PLASCAN_THREE_D_ONLY
    constexpr int kTotalStages = 3;
#else
    constexpr int kTotalStages = 4;
#endif

    QJsonObject timings;
    const auto pipelineStart = std::chrono::steady_clock::now();
    auto recordTiming = [&timings](const QString &key,
                                   const std::chrono::steady_clock::time_point &start) {
        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        timings[key] = elapsedMs;
        return elapsedMs;
    };
    auto markSkippedStage = [&report](const QString &stage, const QString &reason) {
        QJsonObject skippedStages = report.value(QStringLiteral("skipped_stages")).toObject();
        skippedStages[stage] = reason;
        report[QStringLiteral("skipped_stages")] = skippedStages;
    };
    double sfmElapsedMs = 0.0;
    double sparsePreprocessElapsedMs = 0.0;
    double mvsElapsedMs = 0.0;
    double meshElapsedMs = 0.0;
#ifndef PLASCAN_THREE_D_ONLY
    double terrainElapsedMs = 0.0;
#endif

    xjw::cli::printPipelineStage(1, kTotalStages, QStringLiteral("SFM 稀疏重建..."));
    const auto sfmStart = std::chrono::steady_clock::now();
    xjw::aerial_triangulation::AerialTriangulationOptions sfmOptions;
    sfmOptions.images = images;
    sfmOptions.cameraPaths = cameraPaths;
    sfmOptions.projectMeta = projectMeta;
    sfmOptions.projectPath = projectPath;
    sfmOptions.assetsDir = QDir(projectSession.activeChunkRoot())
        .filePath(QStringLiteral("assets"));
    sfmOptions.matchDir = QDir(sfmOptions.assetsDir)
        .filePath(QStringLiteral("image_matches"));
    sfmOptions.outputDir = QDir(pipelineRoot)
        .filePath(QStringLiteral("sparse"));
    sfmOptions.device = QString::fromStdString(device);
    sfmOptions.matchingAlgorithmId =
        QString::fromStdString(sfmMatchingAlgorithmId);
    sfmOptions.lightGlueTensorRtEnginePath = sfmLightGlueEnginePath.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(QString::fromStdString(sfmLightGlueEnginePath));
    sfmOptions.lomaRTensorRtPackagePath = sfmLoMaRTensorRtPackagePath.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(QString::fromStdString(sfmLoMaRTensorRtPackagePath));
    sfmOptions.guidedImageMatching = sfmGuidedRematching;
    sfmOptions.lockInputCameraPoses = lockInputCameraPoses;
    const QStringList qualityNames = {
        QStringLiteral("low"),
        QStringLiteral("medium"),
        QStringLiteral("high"),
        QStringLiteral("highest"),
    };
    sfmOptions.quality = qualityNames.at(qBound(0, quality, qualityNames.size() - 1));
    sfmOptions.threads = std::max(1, threads);
    sfmOptions.cudaParallelPairs = std::max(0, cudaParallelPairs);
    sfmOptions.featureMaxImageDim = featureMaxImageDim;
    sfmOptions.resetAlignment = true;
    sfmOptions.autoGenerateMissingMatches = true;
    sfmOptions.progressFn = [](const QString &stage, int percent) {
        xjw::cli::printScopedProgress(QStringLiteral("SFM"), percent, stage);
    };

    const xjw::aerial_triangulation::AerialTriangulationResult sfmWorkflowResult =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::run(sfmOptions);
    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &sfmResult =
        sfmWorkflowResult.reconstructionResult;
    sfmElapsedMs = recordTiming(QStringLiteral("sfm_elapsed_ms"), sfmStart);
    QJsonObject sfmJson;
    sfmJson[QStringLiteral("success")] = sfmResult.success;
    sfmJson[QStringLiteral("summary")] = sfmResult.summary;
    sfmJson[QStringLiteral("sparse_cloud")] = sfmResult.sparseCloudPath;
    sfmJson[QStringLiteral("registered_images")] = sfmResult.numRegisteredImages;
    sfmJson[QStringLiteral("points")] = sfmResult.numPoints3D;
    sfmJson[QStringLiteral("mean_reprojection_error")] = sfmResult.meanReprojError;
    report[QStringLiteral("sfm")] = sfmJson;

    const QString resultCreatedAt =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    // 与 GUI 和独立空三 CLI 共用逐影像持久化契约。这里不保存 SIFT 描述子，
    // 也不再创建成对 `.match` 或 JSON sidecar 工程记录。
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &image :
         sfmWorkflowResult.tiePointResult.imageMatchFiles)
    {
        projectSession.upsertResultByPath(
            QStringLiteral("image_match_results"),
            QStringLiteral("output"),
            QJsonObject{
                {QStringLiteral("image"), image.imagePath},
                {QStringLiteral("output"), image.matchFilePath},
                {QStringLiteral("neighbors"),
                 QJsonArray::fromStringList(image.neighborImagePaths)},
                {QStringLiteral("neighbor_variant_count"),
                 image.neighborVariantCount},
                {QStringLiteral("settings"), image.settings},
                {QStringLiteral("created_at"), resultCreatedAt}
            });
    }
    if (!sfmResult.sparseCloudPath.isEmpty())
    {
        QJsonObject sparseFiles{
            {QStringLiteral("sparse_cloud_xyz"),
             sfmResult.sparseCloudPath}
        };
        const QJsonObject extraFiles =
            sfmResult.resultRecordExtra
                .value(QStringLiteral("files")).toObject();
        for (auto it = extraFiles.constBegin();
             it != extraFiles.constEnd();
             ++it)
        {
            sparseFiles.insert(it.key(), it.value());
        }
        QJsonObject sparseRecord = sfmResult.resultRecordExtra;
        sparseRecord.remove(QStringLiteral("files"));
        sparseRecord[QStringLiteral("created_at")] = resultCreatedAt;
        sparseRecord[QStringLiteral("output_dir")] =
            sfmOptions.outputDir;
        sparseRecord[QStringLiteral("sparse_point_count")] =
            sfmResult.numPoints3D;
        sparseRecord[QStringLiteral("selected_images")] =
            QJsonArray::fromStringList(images);
        sparseRecord[QStringLiteral("files")] = sparseFiles;
        sparseRecord[QStringLiteral("quality_metadata")] =
            sfmResult.qualityMetadata;
        projectSession.appendResult(
            QStringLiteral("aerial_triangulation_results"),
            sparseRecord);
    }
    if (sfmResult.success
        && !projectSession.updateImageCameras(
            sfmResult.pendingCamUpdates, nullptr, &error))
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] =
            QStringLiteral("SFM 相机写回失败: %1").arg(error);
        QJsonObject finalReport;
        writeFinalReport(&finalReport);
        return cli::EXIT_IO_ERR;
    }
    if (!sfmResult.success || sfmResult.sparseCloudPath.isEmpty())
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] = sfmResult.errorMessage;
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "SFM 失败: %s\n", qUtf8Printable(sfmResult.errorMessage));
        std::fprintf(stderr,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    if (stopAfterSfm || skipMvs)
    {
        const QString reason = stopAfterSfm
            ? QStringLiteral("用户请求在 SFM 后停止")
            : QStringLiteral("用户请求跳过 MVS");
        report[QStringLiteral("status")] = QStringLiteral("ok");
        report[QStringLiteral("stop_stage")] = QStringLiteral("sfm");
        report[QStringLiteral("mvs")] = QJsonObject{
            {QStringLiteral("status"), QStringLiteral("skipped")},
            {QStringLiteral("reason"), reason}
        };
        report[QStringLiteral("model")] = QJsonObject{
            {QStringLiteral("status"), QStringLiteral("skipped")},
            {QStringLiteral("reason"), reason}
        };
        markSkippedStage(QStringLiteral("mvs"), reason);
        markSkippedStage(QStringLiteral("mesh"), reason);
#ifndef PLASCAN_THREE_D_ONLY
        markSkippedStage(QStringLiteral("terrain"), reason);
#endif
        timings[QStringLiteral("sparse_preprocess_elapsed_ms")] = 0.0;
        timings[QStringLiteral("mvs_elapsed_ms")] = 0.0;
        timings[QStringLiteral("mesh_elapsed_ms")] = 0.0;
#ifndef PLASCAN_THREE_D_ONLY
        timings[QStringLiteral("terrain_elapsed_ms")] = 0.0;
#endif
        const double totalElapsedMs = recordTiming(QStringLiteral("total_elapsed_ms"), pipelineStart);
        report[QStringLiteral("timings")] = timings;

        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stdout, "status=ok\n");
        std::fprintf(stdout, "output_dir=%s\n", qUtf8Printable(outputDir));
        std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(sfmResult.sparseCloudPath));
        std::fprintf(stdout, "skipped_mvs=%s\n", qUtf8Printable(reason));
        std::fprintf(stdout, "elapsed_total=%.3fs\n", totalElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_sfm=%.3fs\n", sfmElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_sparse_preprocess=0.000s\n");
        std::fprintf(stdout, "elapsed_mvs=0.000s\n");
        std::fprintf(stdout, "elapsed_mesh=0.000s\n");
#ifndef PLASCAN_THREE_D_ONLY
        std::fprintf(stdout, "elapsed_terrain=0.000s\n");
#endif
        std::fprintf(stdout,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_OK;
    }

    constexpr int kMinimumRegisteredImagesForDenseWorkflow = 2;
    constexpr int kMinimumSparsePointsForDenseWorkflow = 20;
    if (sfmResult.numPoints3D < kMinimumSparsePointsForDenseWorkflow)
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] =
            QStringLiteral("SFM 稀疏点云点数过少(%1 < %2)，停止执行 MVS 和模型生成")
                .arg(sfmResult.numPoints3D)
                .arg(kMinimumSparsePointsForDenseWorkflow);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "SFM 失败: %s\n", qUtf8Printable(report.value(QStringLiteral("reason")).toString()));
        std::fprintf(stderr,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    const auto sparsePreprocessStart = std::chrono::steady_clock::now();
    QMap<QString, xjw::Camera> cameraByImage;
    for (auto it = sfmResult.pendingCamUpdates.constBegin(); it != sfmResult.pendingCamUpdates.constEnd(); ++it)
    {
        xjw::Camera camera;
        const QString imagePath = xjw::cli::cleanAbsolutePath(it.key());
        if (xjw::common::project::cameraFromJson(it.value(), &camera) && camera.isValid())
        {
            cameraByImage.insert(imagePath, camera);
        }
    }

    QStringList registeredImagePaths;
    for (const InputItem &item : items)
    {
        const QString imagePath = xjw::cli::cleanAbsolutePath(item.imagePath);
        if (cameraByImage.contains(imagePath))
        {
            registeredImagePaths.append(imagePath);
        }
    }
    sfmJson[QStringLiteral("registered_image_paths")] = QJsonArray::fromStringList(registeredImagePaths);
    report[QStringLiteral("sfm")] = sfmJson;
    const int originalRegisteredImageCount = registeredImagePaths.size();

    QJsonArray imageMetaArray;
    std::vector<xjw::mvs::CameraView> views;
    views.reserve(static_cast<size_t>(registeredImagePaths.size()));
    int project_mask_count = 0;
    for (const QString &imagePath : registeredImagePaths)
    {
        const xjw::Camera camera = cameraByImage.value(imagePath);
        if (!camera.isValid())
        {
            continue;
        }

        xjw::mvs::CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(imagePath);
        view.camera = camera;
        if (!mvs_mask_dir.isEmpty())
        {
            const QString mask_path = QDir(mvs_mask_dir).filePath(
                QFileInfo(imagePath).completeBaseName() + QStringLiteral("_mask.png"));
            if (QFileInfo::exists(mask_path))
            {
                view.validRegionMaskPath = xjw::common::io::toUtf8Path(mask_path);
                ++project_mask_count;
            }
        }
        cv::Mat image = xjw::common::io::readImage(view.imagePath, cv::IMREAD_GRAYSCALE);
        if (!image.empty())
        {
            view.imageWidth = image.cols;
            view.imageHeight = image.rows;
        }
        views.push_back(std::move(view));

        imageMetaArray.append(QJsonObject{
            {QStringLiteral("path"), imagePath},
            {QStringLiteral("name"), QFileInfo(imagePath).fileName()},
            {QStringLiteral("camera"), xjw::cli::cameraToJson(camera)}
        });
    }
    limitMvsInputsForRegression(&views, &registeredImagePaths, &imageMetaArray, mvsMaxFrames);
    project_mask_count = static_cast<int>(std::count_if(
        views.cbegin(),
        views.cend(),
        [](const xjw::mvs::CameraView &view) {
            return !view.validRegionMaskPath.empty();
        }));
    const int mvs_input_image_count = static_cast<int>(registeredImagePaths.size());
    sfmJson[QStringLiteral("mvs_image_paths")] = QJsonArray::fromStringList(registeredImagePaths);
    sfmJson[QStringLiteral("mvs_input_images")] = mvs_input_image_count;
    sfmJson[QStringLiteral("mvs_project_mask_count")] = project_mask_count;
    sfmJson[QStringLiteral("mvs_project_mask_missing_count")] =
        std::max(0, mvs_input_image_count - project_mask_count);
    if (!mvs_mask_dir.isEmpty())
    {
        sfmJson[QStringLiteral("mvs_mask_dir")] = mvs_mask_dir;
        std::fprintf(stdout,
                     "mvs_project_masks=%d/%d dir=%s\n",
                     project_mask_count,
                     mvs_input_image_count,
                     qUtf8Printable(QDir::toNativeSeparators(mvs_mask_dir)));
    }
    if (mvsMaxFrames > 0)
    {
        sfmJson[QStringLiteral("mvs_max_frames")] = mvsMaxFrames;
    }
    report[QStringLiteral("sfm")] = sfmJson;
    projectMeta[QStringLiteral("images")] = imageMetaArray;

    if (views.size() < static_cast<size_t>(kMinimumRegisteredImagesForDenseWorkflow))
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] =
            QStringLiteral("SFM 后可用于 MVS 的相机不足(%1 < %2)，停止执行 MVS 和模型生成")
                .arg(views.size())
                .arg(kMinimumRegisteredImagesForDenseWorkflow);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr,
                     "MVS 输入不足: report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    xjw::mvs::SparseCloud sparse;
    {
        xjw::mvs::SparseCloudPreprocessor preprocessor;
        xjw::mvs::PreprocessResult preprocessResult;
        std::string preprocessError;
        if (preprocessor.run(xjw::common::io::toUtf8Path(sfmResult.sparseCloudPath),
                             views,
                             preprocessResult,
                             &preprocessError))
        {
            sparse = preprocessResult.cloud;
        }
        else
        {
            std::fprintf(stderr, "稀疏点云预处理失败，继续尝试 MVS: %s\n", preprocessError.c_str());
        }
    }

    sfmJson[QStringLiteral("filtered_sparse_points")] = static_cast<int>(sparse.points.size());
    report[QStringLiteral("sfm")] = sfmJson;
    if (sparse.points.size() < static_cast<size_t>(kMinimumSparsePointsForDenseWorkflow))
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] =
            QStringLiteral("预处理后的 SFM 稀疏点云点数过少(%1 < %2)，停止执行 MVS 和模型生成")
                .arg(sparse.points.size())
                .arg(kMinimumSparsePointsForDenseWorkflow);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "MVS 输入不足: %s\n", qUtf8Printable(report.value(QStringLiteral("reason")).toString()));
        std::fprintf(stderr,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }
    sparsePreprocessElapsedMs = recordTiming(
        QStringLiteral("sparse_preprocess_elapsed_ms"),
        sparsePreprocessStart);

    xjw::cli::printPipelineStage(2, kTotalStages, QStringLiteral("MVS 稠密点云..."));
    const auto mvsStart = std::chrono::steady_clock::now();
    const QString mvsDir =
        QDir(pipelineRoot).filePath(QStringLiteral("mvs"));
    QDir().mkpath(mvsDir);
    xjw::core::project::DenseGenerationSettings denseSettings;
    denseSettings.threads = std::max(1, threads);
    denseSettings.useCuda = (device == "cuda" || device == "auto");
    denseSettings.pipelineMode = true;
    denseSettings.qualityProfile = QString::fromStdString(mvs_quality);
    denseSettings.resScale = mvsResScale;
    denseSettings.iterations = mvsIterations;
    denseSettings.patchMatchConfidence = mvsConfidence;
    denseSettings.fusionMinConfidence = mvsFusionConfidence;
    denseSettings.gpuFrameWorkers = mvsGpuFrameWorkers;
    denseSettings.cpuFrameWorkers = mvsCpuFrameWorkers;
    denseSettings.fusionMaxImageDim = mvsFusionMaxImageDim;
    denseSettings.enableTwoSourceCrossViewGrowth = mvsTwoSourceGrowth;
    denseSettings.twoSourceGrowthDistancePixels = mvsTwoSourceGrowthDistance;
    denseSettings.twoSourceGrowthInverseDepthSpread = static_cast<float>(
        mvsTwoSourceGrowthSpread);
    denseSettings.twoSourceGrowthNormalAngleDegrees = static_cast<float>(
        mvsTwoSourceGrowthNormalAngle);
    denseSettings.twoSourceGrowthMaximumComponentArea =
        mvsTwoSourceGrowthMaximumArea;
    const int denseMinViewCount = std::clamp(static_cast<int>(views.size()),
                                             kMinimumRegisteredImagesForDenseWorkflow,
                                             3);
    denseSettings.minViews = denseMinViewCount;
    denseSettings.minConsistentViews = denseMinViewCount;
    denseSettings.depthConsistency = 1.0f;
    xjw::mvs::DepthGenConfig depthConfig =
        xjw::core::project::buildDepthGenConfig(denseSettings, static_cast<int>(views.size()));
    if (mvs_scene_profile == "aerial_terrain")
    {
        depthConfig.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    }
    else if (mvs_scene_profile == "orbital_object")
    {
        depthConfig.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    }
    else
    {
        depthConfig.sceneProfile = xjw::mvs::MvsSceneProfile::Auto;
    }
    const xjw::mvs::MvsSceneClassification sceneClassification =
        xjw::mvs::classifyMvsScene(views, sparse);
    const xjw::mvs::MvsSceneProfile effectiveSceneProfile =
        depthConfig.sceneProfile == xjw::mvs::MvsSceneProfile::Auto
            ? sceneClassification.profile
            : depthConfig.sceneProfile;
    depthConfig.numSourceViews = xjw::mvs::recommendedMvsSourceViewCount(
        effectiveSceneProfile,
        depthConfig.patchMatch.downsampleFactor,
        depthConfig.numSourceViews,
        static_cast<int>(views.size()));
    depthConfig.patchMatch.numSourceViews = depthConfig.numSourceViews;
    depthConfig.adaptiveDepthFilterMode = mvs_depth_filter == "auto";
    if (mvs_depth_filter == "mild")
    {
        depthConfig.depthFilterMode = xjw::mvs::DepthFilterMode::Mild;
    }
    else if (mvs_depth_filter == "aggressive")
    {
        depthConfig.depthFilterMode = xjw::mvs::DepthFilterMode::Aggressive;
    }
    else
    {
        depthConfig.depthFilterMode = xjw::mvs::DepthFilterMode::Moderate;
    }
    depthConfig.saveIntermediatePyramidLevels = mvs_save_levels;
    depthConfig.runFusion = false;
    depthConfig.saveIntermediateDepthMaps = true;
    depthConfig.intermediateDir = xjw::common::io::toUtf8Path(mvsDir);
    depthConfig.sourcePairQualities =
        loadMvsSourcePairQualities(sfmOptions.matchDir);
    const std::size_t verified_pair_count = static_cast<std::size_t>(std::count_if(
        depthConfig.sourcePairQualities.cbegin(),
        depthConfig.sourcePairQualities.cend(),
        [](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            return quality.verified && quality.geometricInliers > 0;
        }));
    const std::size_t missing_statistics_count =
        static_cast<std::size_t>(std::count_if(
            depthConfig.sourcePairQualities.cbegin(),
            depthConfig.sourcePairQualities.cend(),
            [](const xjw::mvs::MvsSourcePairQuality &quality)
            {
                return !quality.hasVerificationStatistics;
            }));
    depthConfig.requireVerifiedSourcePairs = verified_pair_count > 0;
    if (!depthConfig.sourcePairQualities.empty())
    {
        std::fprintf(stdout,
                     "  [MVS] pair audit verified=%zu missing_stats=%zu failed=%zu\n",
                     verified_pair_count,
                     missing_statistics_count,
                     depthConfig.sourcePairQualities.size() -
                         verified_pair_count - missing_statistics_count);
        std::fflush(stdout);
    }

    xjw::mvs::DepthMapGenerator generator;
    generator.setViews(views);
    generator.setSparseCloud(sparse);
    generator.setConfig(depthConfig);
    generator.setOutputDir(xjw::common::io::toUtf8Path(mvsDir));

    QEventLoop loop;
    bool depthOk = false;
    QString mvsError;
    QJsonArray depthArtifacts;
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::progressChanged, &loop,
                     [](const QString &stage, float ratio) {
        xjw::cli::printScopedProgress(
            QStringLiteral("MVS"), static_cast<int>(ratio * 100.0f), stage);
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::errorOccurred, &loop,
                     [&mvsError](const QString &message) {
        mvsError = message;
        std::fprintf(stderr, "  [MVS] %s\n", qUtf8Printable(message));
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::depthMapArtifactSaved, &loop,
                     [&depthArtifacts](const QJsonObject &artifact) {
        depthArtifacts.append(artifact);
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::finished, &loop,
                     [&loop, &depthOk](bool success) {
        depthOk = success;
        loop.quit();
    });
    QTimer::singleShot(0, &generator, &xjw::mvs::DepthMapGenerator::start);
    loop.exec();

    std::vector<xjw::mvs::DepthPostProcessStats> depthPostprocessStats;
    std::vector<xjw::mvs::FusedPoint> fusedCloud;
    bool mvsOk = depthOk;
    QString denseCloudPathForReport;
    QString refinedCloudPathForModel;
    int densePointCount = 0;
    int refinedPointCount = 0;
    xjw::mvs::TerrainHeightSpikeFilterReport terrainSpikeReport;
    if (mvsOk && mvsDepthOnly)
    {
        const QString depthOnlyReason = QStringLiteral("用户请求只生成 MVS 深度图");
        QJsonObject denseReport;
        denseReport[QStringLiteral("status")] = QStringLiteral("depth_only");
        denseReport[QStringLiteral("depth_maps")] = depthArtifacts;
        denseReport[QStringLiteral("depth_postprocess")] = depthPostprocessStatsToJson(depthPostprocessStats);
        denseReport[QStringLiteral("points")] = 0;
        denseReport[QStringLiteral("refined_points")] = 0;
        denseReport[QStringLiteral("mvs_settings")] = mvsSettingsToJson(denseSettings,
                                                                        mvsMaxFrames,
                                                                        static_cast<int>(views.size()),
                                                                        originalRegisteredImageCount);
        denseReport[QStringLiteral("mvs_depth_config")] = mvsDepthConfigToJson(depthConfig);
        report[QStringLiteral("dense")] = denseReport;
        report[QStringLiteral("status")] = QStringLiteral("ok");
        report[QStringLiteral("stop_stage")] = QStringLiteral("mvs_depth");
        report[QStringLiteral("model")] = QJsonObject{
            {QStringLiteral("status"), QStringLiteral("skipped")},
            {QStringLiteral("reason"), depthOnlyReason}
        };
#ifndef PLASCAN_THREE_D_ONLY
        report[QStringLiteral("terrain")] = QJsonObject{
            {QStringLiteral("status"), QStringLiteral("skipped")},
            {QStringLiteral("reason"), depthOnlyReason}
        };
#endif
        markSkippedStage(QStringLiteral("mvs_fusion"), depthOnlyReason);
        markSkippedStage(QStringLiteral("mesh"), depthOnlyReason);
#ifndef PLASCAN_THREE_D_ONLY
        markSkippedStage(QStringLiteral("terrain"), depthOnlyReason);
#endif
        mvsElapsedMs = recordTiming(QStringLiteral("mvs_elapsed_ms"), mvsStart);
        timings[QStringLiteral("mesh_elapsed_ms")] = 0.0;
#ifndef PLASCAN_THREE_D_ONLY
        timings[QStringLiteral("terrain_elapsed_ms")] = 0.0;
#endif
        const double totalElapsedMs = recordTiming(QStringLiteral("total_elapsed_ms"), pipelineStart);
        report[QStringLiteral("timings")] = timings;

        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        const int depthMapCount = static_cast<int>(depthArtifacts.size());
        std::fprintf(stdout, "status=ok\n");
        std::fprintf(stdout, "output_dir=%s\n", qUtf8Printable(outputDir));
        std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(sfmResult.sparseCloudPath));
        std::fprintf(stdout, "depth_maps=%d\n", depthMapCount);
        std::fprintf(stdout, "skipped_mvs_fusion=%s\n", qUtf8Printable(depthOnlyReason));
        std::fprintf(stdout, "elapsed_total=%.3fs\n", totalElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_sfm=%.3fs\n", sfmElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_sparse_preprocess=%.3fs\n", sparsePreprocessElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_mvs=%.3fs\n", mvsElapsedMs / 1000.0);
        std::fprintf(stdout, "elapsed_mesh=0.000s\n");
#ifndef PLASCAN_THREE_D_ONLY
        std::fprintf(stdout, "elapsed_terrain=0.000s\n");
#endif
        std::fprintf(stdout,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_OK;
    }
    if (mvsOk)
    {
        if (!fuseDepthMapsStreamingFromDisk(mvsDir,
                                            views,
                                            denseSettings,
                                            depthConfig,
                                            &fusedCloud,
                                            &depthPostprocessStats,
                                            &error))
        {
            mvsOk = false;
            mvsError = error;
        }
    }
    if (mvsOk)
    {
            constexpr std::size_t kLargeCloudPreVoxelThreshold = 2000000;
            constexpr std::size_t kMaxRefineInputPoints = 250000;
            std::vector<xjw::mvs::FusedPoint> preAggregatedFusedCloud;
            const std::vector<xjw::mvs::FusedPoint> *refineFusedCloud = &fusedCloud;
            bool preAggregatedBeforePlaPoint = false;

            xjw::core::project::DenseRefineSettings refineSettings;
            refineSettings.sorEnabled = true;
            refineSettings.sorK = 30;
            refineSettings.sorStdDev = 2.0;
            refineSettings.voxelEnabled = false;
            refineSettings.voxelSize = 0.005;
            refineSettings.normalsEnabled = true;
            refineSettings.normalK = 30;
            refineSettings.smoothNormals = false;
            refineSettings.threads = std::max(1, threads);
            refineSettings.processingDevice = denseSettings.useCuda
                ? plapoint::ProcessingDevice::GPU
                : plapoint::ProcessingDevice::CPU;

            if (fusedCloud.size() > kLargeCloudPreVoxelThreshold)
            {
                const float preVoxelSize = adaptivePreSorVoxelSize(
                    fusedCloud,
                    static_cast<float>(refineSettings.voxelSize));
                const auto beforePreVoxel = fusedCloud.size();
                std::fprintf(stdout,
                             "  [MVS  18%%] 开始大点云预降采样 leaf=%.6f points=%zu targetPoints=%zu\n",
                             preVoxelSize,
                             beforePreVoxel,
                             kMaxRefineInputPoints);
                std::fflush(stdout);
                const auto preVoxelStart = std::chrono::steady_clock::now();
                FusedVoxelDownsampleResult preVoxelResult = voxelDownsampleFusedPointsToTarget(
                    fusedCloud,
                    preVoxelSize,
                    kMaxRefineInputPoints);
                preAggregatedFusedCloud = std::move(preVoxelResult.points);
                const double preVoxelMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - preVoxelStart).count();
                if (!preAggregatedFusedCloud.empty())
                {
                    refineFusedCloud = &preAggregatedFusedCloud;
                    preAggregatedBeforePlaPoint = true;
                }
                std::fprintf(stdout,
                             "  [MVS  18%%] 完成大点云预降采样 leaf=%.6f passes=%d points=%zu->%zu elapsed=%.1f ms\n",
                             preVoxelResult.leafSize,
                             preVoxelResult.passes,
                             beforePreVoxel,
                             refineFusedCloud->size(),
                             preVoxelMs);
                std::fflush(stdout);
            }

            std::fprintf(stdout,
                         "  [MVS  16%%] 写出原始稠密点云 points=%zu...\n",
                         fusedCloud.size());
            std::fflush(stdout);
            const auto rawWriteStart = std::chrono::steady_clock::now();
            PlaCloud rawPointCloud = fusedPointsToPointCloud(fusedCloud, true, true);
            const QString denseCloudPath = QDir(mvsDir).filePath(QStringLiteral("dense_cloud.ply"));
            if (!xjw::mvs::writeDensePointCloudPly(denseCloudPath, rawPointCloud, true, &error))
            {
                mvsOk = false;
            }
            else
            {
                const double rawWriteMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - rawWriteStart).count();
                std::fprintf(stdout,
                             "  [MVS  16%%] 完成原始稠密点云写出 points=%zu elapsed=%.1f ms\n",
                             fusedCloud.size(),
                             rawWriteMs);
                std::fflush(stdout);

                PlaCloud refineInput = fusedPointsToPointCloud(*refineFusedCloud, true, true);
                if (!preAggregatedBeforePlaPoint && refineInput.size() > kLargeCloudPreVoxelThreshold)
                {
                    const float preVoxelSize = adaptivePreSorVoxelSize(
                        refineInput,
                        static_cast<float>(refineSettings.voxelSize));
                    const auto beforePreVoxel = refineInput.size();
                    plapoint::ProcessingReport preVoxelReport;
                    std::fprintf(stdout,
                                 "  [MVS  18%%] 开始大点云预降采样 leaf=%.6f points=%zu\n",
                                 preVoxelSize,
                                 beforePreVoxel);
                    std::fflush(stdout);
                    const auto preVoxelStart = std::chrono::steady_clock::now();
                    refineInput = voxelDownsample(refineInput,
                                                  preVoxelSize,
                                                  refineSettings.processingDevice,
                                                  &preVoxelReport);
                    const double preVoxelMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - preVoxelStart).count();
                    reportPlaPointDevice(nullptr,
                                         QStringLiteral("大点云预降采样"),
                                         preVoxelReport,
                                         beforePreVoxel,
                                         refineInput.size(),
                                         18);
                    std::fprintf(stdout,
                                 "  [MVS  18%%] 完成大点云预降采样 leaf=%.6f points=%zu->%zu elapsed=%.1f ms\n",
                                 preVoxelSize,
                                 beforePreVoxel,
                                 refineInput.size(),
                                 preVoxelMs);
                    std::fflush(stdout);
                }

                PlaCloud refinedCloud = refineDenseCloud(std::move(refineInput),
                                                         refineSettings,
                                                         [](const QString &stage, int percent) {
                                                             std::fprintf(stdout,
                                                                          "  [MVS %3d%%] %s\n",
                                                                          percent,
                                                                          qUtf8Printable(stage));
                                                             std::fflush(stdout);
                                                         },
                                                         &terrainSpikeReport);
                const QString refinedCloudPath = QDir(mvsDir).filePath(QStringLiteral("dense_cloud_refined.ply"));
                if (!xjw::mvs::writeDensePointCloudPly(
                        refinedCloudPath,
                        refinedCloud,
                        refineSettings.normalsEnabled && refinedCloud.hasNormals(),
                        &error))
                {
                    mvsOk = false;
                }
                else
                {
                    denseCloudPathForReport = denseCloudPath;
                    refinedCloudPathForModel = refinedCloudPath;
                    densePointCount = static_cast<int>(fusedCloud.size());
                    refinedPointCount = static_cast<int>(refinedCloud.size());
                }
            }
    }

    if (!mvsOk
        || denseCloudPathForReport.isEmpty()
        || refinedCloudPathForModel.isEmpty()
        || !QFileInfo::exists(refinedCloudPathForModel))
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] = !error.isEmpty()
            ? error
            : (mvsError.isEmpty() ? QStringLiteral("MVS 未生成有效稠密点云") : mvsError);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "MVS 失败: %s\n", qUtf8Printable(report.value(QStringLiteral("reason")).toString()));
        std::fprintf(stderr,
                     "report=%s\n",
                     qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }
    QJsonObject denseReport{
        {QStringLiteral("point_cloud"), denseCloudPathForReport},
        {QStringLiteral("refined_point_cloud"), refinedCloudPathForModel},
        {QStringLiteral("points"), densePointCount},
        {QStringLiteral("refined_points"), refinedPointCount},
        {QStringLiteral("terrain_spike_filter"), terrainSpikeReportToJson(terrainSpikeReport)},
        {QStringLiteral("has_rgb"), true},
        {QStringLiteral("has_normals"), true}
    };
    denseReport[QStringLiteral("depth_postprocess")] = depthPostprocessStatsToJson(depthPostprocessStats);
    denseReport[QStringLiteral("depth_maps")] = depthArtifacts;
    denseReport[QStringLiteral("mvs_settings")] = mvsSettingsToJson(denseSettings,
                                                                    mvsMaxFrames,
                                                                    static_cast<int>(views.size()),
                                                                    originalRegisteredImageCount);
    denseReport[QStringLiteral("mvs_depth_config")] = mvsDepthConfigToJson(depthConfig);
    report[QStringLiteral("dense")] = denseReport;
    mvsElapsedMs = recordTiming(QStringLiteral("mvs_elapsed_ms"), mvsStart);

    if (!skipModel)
    {
        xjw::cli::printPipelineStage(3, kTotalStages, QStringLiteral("三维网格模型..."));
        const auto meshStart = std::chrono::steady_clock::now();
        xjw::mesh::workflow::MeshBuildRequest meshRequest;
        meshRequest.pointCloudPath = refinedCloudPathForModel;
        meshRequest.outputRoot =
            QDir(pipelineRoot).filePath(QStringLiteral("model"));
        meshRequest.exportObj = exportObj;
        const bool aerialTerrain =
            effectiveSceneProfile == xjw::mvs::MvsSceneProfile::AerialTerrain;
        const bool preserveMeshDetail = mvs_quality == "highest" || mvs_quality == "high";
        meshRequest.reconstruction = xjw::mesh::workflow::reconstructionConfigForDenseScene(
            meshResolution,
            aerialTerrain,
            preserveMeshDetail);
        meshRequest.progress = [lastMeshProgressPercent = -1,
                                lastMeshProgressStage = QString()](const QString &stage, int percent) mutable {
            if (percent == lastMeshProgressPercent && stage == lastMeshProgressStage)
            {
                return;
            }
            lastMeshProgressPercent = percent;
            lastMeshProgressStage = stage;
            std::fprintf(stdout, "  [Mesh %3d%%] %s\n", percent, qUtf8Printable(stage));
            std::fflush(stdout);
        };
        xjw::mesh::workflow::WorkflowResult meshResult;
        try
        {
            meshResult = xjw::mesh::workflow::buildMeshAndOptionalTexture(meshRequest);
        }
        catch (const std::exception &ex)
        {
            meshResult.ok = false;
            meshResult.errorMessage = QStringLiteral("模型生成异常: %1").arg(QString::fromUtf8(ex.what()));
        }
        report[QStringLiteral("model")] = meshResult.payload;
        if (!meshResult.ok)
        {
            report[QStringLiteral("model_error")] = meshResult.errorMessage;
            std::fprintf(stderr, "模型生成失败: %s\n", qUtf8Printable(meshResult.errorMessage));
        }
        meshElapsedMs = recordTiming(QStringLiteral("mesh_elapsed_ms"), meshStart);
    }
    else
    {
        markSkippedStage(QStringLiteral("mesh"), QStringLiteral("用户请求跳过网格模型"));
        timings[QStringLiteral("mesh_elapsed_ms")] = 0.0;
    }

#ifndef PLASCAN_THREE_D_ONLY
    if (!skipTerrain)
    {
        xjw::cli::printPipelineStage(4, 4, QStringLiteral("DEM/DOM 产品..."));
        const auto terrainStart = std::chrono::steady_clock::now();
        const QString terrainDir =
            QDir(pipelineRoot).filePath(QStringLiteral("terrain"));
        QJsonObject demResult;
        if (!xjw::TerrainPipeline::generateDemProducts(refinedCloudPathForModel,
                                                       terrainDir,
                                                       demResolution,
                                                       QStringLiteral("float32"),
                                                       true,
                                                       &demResult,
                                                       &error))
        {
            report[QStringLiteral("terrain_error")] = error;
            std::fprintf(stderr, "DEM 生成失败: %s\n", qUtf8Printable(error));
        }
        else
        {
            QJsonObject domResult;
            const QString domPath = QDir(terrainDir).filePath(QStringLiteral("products/dom.png"));
            if (!xjw::TerrainPipeline::generateOrthoProduct(registeredImagePaths,
                                                            demResult.value(QStringLiteral("dem_tif")).toString(),
                                                            domPath,
                                                            demResolution,
                                                            projectMeta,
                                                            &domResult,
                                                            &error))
            {
                report[QStringLiteral("terrain_error")] = error;
                std::fprintf(stderr, "DOM 生成失败: %s\n", qUtf8Printable(error));
            }
            report[QStringLiteral("terrain")] = QJsonObject{
                {QStringLiteral("dem"), demResult},
                {QStringLiteral("dom"), domResult}
            };
        }
        terrainElapsedMs = recordTiming(QStringLiteral("terrain_elapsed_ms"), terrainStart);
    }
    else
    {
        timings[QStringLiteral("terrain_elapsed_ms")] = 0.0;
    }
#endif

#ifndef PLASCAN_THREE_D_ONLY
    const QJsonObject terrain = report.value(QStringLiteral("terrain")).toObject();
    const QString demPath =
        terrain.value(QStringLiteral("dem")).toObject()
            .value(QStringLiteral("dem_tif")).toString();
    const QString domPath = domOutputPath(terrain.value(QStringLiteral("dom")).toObject());
#endif
    const QJsonObject model = report.value(QStringLiteral("model")).toObject();
    const QString modelPath = model.value(QStringLiteral("final_model_path")).toString(
        model.value(QStringLiteral("model_ply")).toString());

    const bool modelOk = skipModel || (!modelPath.isEmpty() && QFileInfo::exists(modelPath));
#ifdef PLASCAN_THREE_D_ONLY
    const bool terrainOk = true;
#else
    const bool terrainOk = skipTerrain || ((!demPath.isEmpty() && QFileInfo::exists(demPath))
                                           && (!domPath.isEmpty() && QFileInfo::exists(domPath)));
#endif
    report[QStringLiteral("status")] = (modelOk && terrainOk) ? QStringLiteral("ok") : QStringLiteral("partial");
    const double totalElapsedMs = recordTiming(QStringLiteral("total_elapsed_ms"), pipelineStart);
    report[QStringLiteral("timings")] = timings;

    if (!denseCloudPathForReport.isEmpty())
    {
        projectSession.upsertResultByPath(
            QStringLiteral("dense_cloud_results"),
            QStringLiteral("dense_cloud_xyz"),
            QJsonObject{
                {QStringLiteral("created_at"), resultCreatedAt},
                {QStringLiteral("kind"), QStringLiteral("dense_cloud")},
                {QStringLiteral("result_type"),
                 QStringLiteral("dense_cloud")},
                {QStringLiteral("source_sparse_cloud"),
                 sfmResult.sparseCloudPath},
                {QStringLiteral("dense_cloud_xyz"),
                 denseCloudPathForReport},
                {QStringLiteral("refined_dense_cloud"),
                 refinedCloudPathForModel},
                {QStringLiteral("point_count"), densePointCount},
                {QStringLiteral("refined_point_count"),
                 refinedPointCount}
            });
    }
    if (!modelPath.isEmpty())
    {
        QJsonObject modelRecord = model;
        modelRecord[QStringLiteral("created_at")] = resultCreatedAt;
        modelRecord[QStringLiteral("kind")] = QStringLiteral("mesh");
        modelRecord[QStringLiteral("result_type")] =
            QStringLiteral("mesh");
        modelRecord[QStringLiteral("model_ply")] = modelPath;
        modelRecord[QStringLiteral("source_dense_cloud")] =
            refinedCloudPathForModel;
        projectSession.upsertResultByPath(
            QStringLiteral("model_results"),
            QStringLiteral("model_ply"),
            modelRecord);
    }
#ifndef PLASCAN_THREE_D_ONLY
    if (!demPath.isEmpty())
    {
        QJsonObject demRecord =
            terrain.value(QStringLiteral("dem")).toObject();
        demRecord[QStringLiteral("created_at")] = resultCreatedAt;
        demRecord[QStringLiteral("dem_tif")] = demPath;
        demRecord[QStringLiteral("source_sparse_cloud")] =
            sfmResult.sparseCloudPath;
        projectSession.upsertResultByPath(
            QStringLiteral("dem_results"),
            QStringLiteral("dem_tif"),
            demRecord);
    }
    if (!domPath.isEmpty())
    {
        QJsonObject orthoRecord =
            terrain.value(QStringLiteral("dom")).toObject();
        orthoRecord[QStringLiteral("created_at")] = resultCreatedAt;
        orthoRecord[QStringLiteral("output_path")] = domPath;
        orthoRecord[QStringLiteral("dem_path")] = demPath;
        projectSession.upsertResultByPath(
            QStringLiteral("ortho_results"),
            QStringLiteral("output_path"),
            orthoRecord);
    }
#endif
    QJsonObject finalReport;
    if (!writeFinalReport(&finalReport))
    {
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout, "status=%s\n", qUtf8Printable(report.value(QStringLiteral("status")).toString()));
    std::fprintf(stdout, "output_dir=%s\n", qUtf8Printable(outputDir));
    std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(sfmResult.sparseCloudPath));
    std::fprintf(stdout, "dense_cloud=%s points=%d\n", qUtf8Printable(denseCloudPathForReport), densePointCount);
    std::fprintf(stdout,
                 "refined_dense_cloud=%s points=%d\n",
                 qUtf8Printable(refinedCloudPathForModel),
                 refinedPointCount);
    if (!modelPath.isEmpty()) std::fprintf(stdout, "model=%s\n", qUtf8Printable(modelPath));
#ifndef PLASCAN_THREE_D_ONLY
    if (!demPath.isEmpty()) std::fprintf(stdout, "dem=%s\n", qUtf8Printable(demPath));
    if (!domPath.isEmpty()) std::fprintf(stdout, "dom=%s\n", qUtf8Printable(domPath));
#endif
    std::fprintf(stdout, "elapsed_total=%.3fs\n", totalElapsedMs / 1000.0);
    std::fprintf(stdout, "elapsed_sfm=%.3fs\n", sfmElapsedMs / 1000.0);
    std::fprintf(stdout, "elapsed_sparse_preprocess=%.3fs\n", sparsePreprocessElapsedMs / 1000.0);
    std::fprintf(stdout, "elapsed_mvs=%.3fs\n", mvsElapsedMs / 1000.0);
    std::fprintf(stdout, "elapsed_mesh=%.3fs\n", meshElapsedMs / 1000.0);
#ifndef PLASCAN_THREE_D_ONLY
    std::fprintf(stdout, "elapsed_terrain=%.3fs\n", terrainElapsedMs / 1000.0);
#endif
    std::fprintf(stdout, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));

    return (modelOk && terrainOk) ? cli::EXIT_OK : cli::EXIT_ALGO_ERR;
}

int xjw::cli::runReconstructionPipelineCli(int argc, char *argv[])
{
    return runReconstructionPipelineCliImpl(argc, argv);
}
