#include "TerrainPipeline.h"

#include "DemDomIO.h"
#include "DemDomTypes.h"
#include "DemGenerator.h"
#include "DomGenerator.h"
#include "ObjMtlLoader.h"
#include "OrthoGenerationOptions.h"
#include "OrthoProjector.h"
#include "PointCloudDomGenerator.h"
#include "SmallBodyGlobalProductGenerator.h"
#include "projection/AsteroidProjection.h"
#include "Camera.h"
#include "io/PathIO.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/obj_io.h>
#include <plapoint/io/xyz_io.h>
#include <plamatrix/dense/dense_matrix.h>

#include <QDir>
#include <QDateTime>
#include <QFileInfo>
#include <QJsonArray>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace
{

xjw::DemRasterFormat parseDemRasterFormat(const QString &demType)
{
    if (demType.compare(QStringLiteral("uint16"), Qt::CaseInsensitive) == 0)
    {
        return xjw::DemRasterFormat::UInt16Tiff;
    }

    return xjw::DemRasterFormat::Float32Tiff;
}

void appendQualityArtifacts(QJsonObject *output, const xjw::DemQualityArtifacts &artifacts)
{
    if (!output)
    {
        return;
    }
    output->insert(QStringLiteral("error_path"), artifacts.errorPath);
    output->insert(QStringLiteral("count_path"), artifacts.countPath);
    output->insert(QStringLiteral("confidence_path"), artifacts.confidencePath);
    output->insert(QStringLiteral("coverage_path"), artifacts.coveragePath);
}

QString normalizePathForOrtho(const QString &path)
{
    QString normalized = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

bool applyRelativeDemOffset(const QJsonObject &record, double *zOffset)
{
    if (record.value(QStringLiteral("dem_reference")).toString()
        != QStringLiteral("relative"))
    {
        return false;
    }
    if (zOffset)
    {
        *zOffset =
            record.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
    }
    return true;
}

bool resolveDemVerticalOffsetForOrtho(const QJsonObject &projectMeta,
                                      const QString &demPath,
                                      double *zOffset)
{
    if (zOffset)
    {
        *zOffset = 0.0;
    }

    if (projectMeta.isEmpty() || demPath.isEmpty())
    {
        return false;
    }

    const QString normalizedDemPath = normalizePathForOrtho(demPath);
    const QJsonArray demResults = projectMeta.value(QStringLiteral("dem_results")).toArray();
    QJsonObject uniqueFileNameMatch;
    int fileNameMatchCount = 0;

    for (int index = demResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = demResults.at(index).toObject();
        const QString recordDemPath = record.value(QStringLiteral("dem_tif")).toString();
        if (recordDemPath.isEmpty())
        {
            continue;
        }

        const QString normalizedRecordPath = normalizePathForOrtho(recordDemPath);
        if (normalizedRecordPath == normalizedDemPath)
        {
            return applyRelativeDemOffset(record, zOffset);
        }
        if (QFileInfo(recordDemPath).fileName().compare(
                QFileInfo(demPath).fileName(),
#ifdef Q_OS_WIN
                Qt::CaseInsensitive
#else
                Qt::CaseSensitive
#endif
                ) == 0)
        {
            uniqueFileNameMatch = record;
            ++fileNameMatchCount;
        }
    }

    return fileNameMatchCount == 1
        && applyRelativeDemOffset(uniqueFileNameMatch, zOffset);
}

std::shared_ptr<xjw::PlaPointCloud> readTerrainPointCloud(const QString &pointCloudPath,
                                                          QString *errorMsg)
{
    const std::string path = xjw::common::io::toNativeNarrowPath(pointCloudPath);
    const QString suffix = QFileInfo(pointCloudPath).suffix().toLower();
    try
    {
        if (suffix == QStringLiteral("ply"))
        {
            return plapoint::io::readPly<float>(path);
        }
        if (suffix == QStringLiteral("obj"))
        {
            return plapoint::io::readObj<float>(path);
        }
        if (suffix == QStringLiteral("xyz"))
        {
            return plapoint::io::readXyz<float>(path);
        }
    }
    catch (const std::exception &exception)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("读取点云失败: %1")
                .arg(QString::fromUtf8(exception.what()));
        }
        return {};
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的点云格式: %1；请选择 PLY、OBJ 或 XYZ")
            .arg(pointCloudPath);
    }
    return {};
}

bool generatePointCloudOrtho(const QString &pointCloudPath,
                             const QString &outputPath,
                             const xjw::OrthoGenerationOptions &options,
                             QJsonObject *result,
                             QString *errorMsg,
                             const std::atomic_bool *cancelFlag,
                             const xjw::TerrainPipeline::OrthoProgressCallback &progressCallback)
{
    if (progressCallback)
    {
        progressCallback(QStringLiteral("读取彩色点云"), 0);
    }
    const std::shared_ptr<xjw::PlaPointCloud> cloud =
        readTerrainPointCloud(pointCloudPath, errorMsg);
    if (!cloud || cloud->size() == 0)
    {
        if (errorMsg && errorMsg->isEmpty())
        {
            *errorMsg = QStringLiteral("点云文件为空或无法读取");
        }
        return false;
    }

    xjw::PointCloudDomResult projected;
    if (!xjw::PointCloudDomGenerator::generate(
            *cloud, options, &projected, errorMsg, cancelFlag, progressCallback))
    {
        return false;
    }
    if (progressCallback)
    {
        progressCallback(QStringLiteral("写出点云正射影像"), 95);
    }
    const bool outputGeoTiff = outputPath.endsWith(QStringLiteral(".tif"), Qt::CaseInsensitive)
        || outputPath.endsWith(QStringLiteral(".tiff"), Qt::CaseInsensitive);
    if (outputGeoTiff)
    {
        if (!xjw::DemDomIO::writeDomGeoTiff(projected.imageBgr,
                                            projected.validMask,
                                            projected.reference,
                                            outputPath,
                                            errorMsg))
        {
            return false;
        }
    }
    else
    {
        cv::Mat northUp;
        cv::flip(projected.imageBgr, northUp, 0);
        cv::Mat alpha;
        cv::flip(projected.validMask, alpha, 0);
        std::vector<cv::Mat> channels;
        cv::split(northUp, channels);
        channels.push_back(alpha);
        cv::merge(channels, northUp);
        if (!xjw::DemDomIO::writeDomImage(
                northUp, outputPath, xjw::DomImageFormat::Png, errorMsg))
        {
            return false;
        }
    }
    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")] =
            QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("output_path")] = outputPath;
        output[QStringLiteral("point_cloud_path")] = pointCloudPath;
        output[QStringLiteral("resolved_settings")] = projected.resolvedOptions.toResolvedJson();
        output[QStringLiteral("projection_wkt")] = projected.reference.projection.projectionWkt;
        output[QStringLiteral("projection_wkt_present")] =
            !projected.reference.projection.projectionWkt.isEmpty();
        output[QStringLiteral("dom_georeferenced")] = outputGeoTiff;
        output[QStringLiteral("has_coverage_alpha")] = true;
        output[QStringLiteral("input_point_count")] =
            static_cast<double>(projected.inputPointCount);
        output[QStringLiteral("projected_point_count")] =
            static_cast<double>(projected.projectedPointCount);
        output[QStringLiteral("valid_pixel_count")] =
            static_cast<double>(projected.validPixelCount);
        output[QStringLiteral("coverage_ratio")] = projected.coverageRatio;
        output[QStringLiteral("width")] = projected.reference.width;
        output[QStringLiteral("height")] = projected.reference.height;
        output[QStringLiteral("pixel_size_x")] = projected.reference.stepX;
        output[QStringLiteral("pixel_size_y")] = projected.reference.stepY;
        output[QStringLiteral("algorithm_version")] = QStringLiteral("point_cloud_dom_v1");
        *result = output;
    }
    if (progressCallback)
    {
        progressCallback(QStringLiteral("点云正射影像生成完成"), 100);
    }
    return true;
}

} // namespace

namespace xjw {

bool TerrainPipeline::generateDemProducts(const QString &pointCloudPath,
                                          const QString &outputDir,
                                          double demResolution,
                                          const QString &demType,
                                          bool generateDenseCloud,
                                          QJsonObject *result,
                                          QString *errorMsg)
{
    // Auto-detect file format and read
    std::shared_ptr<PlaPointCloud> cloudPtr;
    const std::string path = xjw::common::io::toNativeNarrowPath(pointCloudPath);
    const QString suffix = QFileInfo(pointCloudPath).suffix().toLower();
    try
    {
        if (suffix == QStringLiteral("ply"))
            cloudPtr = plapoint::io::readPly<float>(path);
        else if (suffix == QStringLiteral("obj"))
            cloudPtr = plapoint::io::readObj<float>(path);
        else if (suffix == QStringLiteral("xyz"))
            cloudPtr = plapoint::io::readXyz<float>(path);
        else
        {
            // Try all formats
            try { cloudPtr = plapoint::io::readPly<float>(path); } catch (...) {}
            if (!cloudPtr) try { cloudPtr = plapoint::io::readObj<float>(path); } catch (...) {}
            if (!cloudPtr) try { cloudPtr = plapoint::io::readXyz<float>(path); } catch (...) {}
        }
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("读取点云失败: %1").arg(QString::fromStdString(e.what()));
        return false;
    }

    if (!cloudPtr || cloudPtr->size() == 0)
    {
        if (errorMsg) *errorMsg = QStringLiteral("读取点云失败: 文件为空或格式不支持");
        return false;
    }

    DemGenerationOptions options;
    options.gridResolution = demResolution;
    options.generateDenseCloud = generateDenseCloud;
    options.generateMesh = generateDenseCloud;
    options.rasterFormat = parseDemRasterFormat(demType);
    options.holeFillIterations = 20;
    options.holeFillMinNeighbors = 3;
    options.holeFillSearchRadius = 5;
    options.useSubPixelBilinearSplat = true;

    DemGridData demGrid;
    PlaPointCloud denseCloud;
    if (!DemGenerator::generateFromPointCloud(*cloudPtr, options, &demGrid, &denseCloud, errorMsg))
    {
        return false;
    }

    double relativeZOffset = 0.0;
    bool relativeDem = !generateDenseCloud;
    if (relativeDem)
    {
        double minElevation = 0.0;
        cv::minMaxLoc(demGrid.elevation, &minElevation, nullptr, nullptr, nullptr, demGrid.validMask);
        relativeZOffset = minElevation;

        for (int row = 0; row < demGrid.height; ++row)
        {
            for (int col = 0; col < demGrid.width; ++col)
            {
                if (demGrid.validMask.at<uchar>(row, col) != 0)
                {
                    demGrid.elevation.at<float>(row, col) -= static_cast<float>(relativeZOffset);
                }
            }
        }
    }

    const QString productsDir = QDir(outputDir).filePath(QStringLiteral("products"));
    const QString depthPng = QDir(productsDir).filePath(QStringLiteral("depth_map.png"));
    const QString denseXyz = QDir(productsDir).filePath(QStringLiteral("dense_cloud.xyz"));
    const QString demTif = QDir(productsDir).filePath(QStringLiteral("dem.tif"));
    const QString meshPly = QDir(productsDir).filePath(QStringLiteral("model_from_dense.ply"));

    if (!DemDomIO::writeDemPreviewPng(demGrid, depthPng, errorMsg)) {
        return false;
    }

    if (!DemDomIO::writeDemRaster(demGrid, demTif, options.rasterFormat, errorMsg)) {
        return false;
    }

    DemQualityArtifacts qualityArtifacts;
    if (!DemDomIO::writeDemQualityRasters(demGrid, productsDir, &qualityArtifacts, errorMsg))
    {
        return false;
    }

    int densePointCount = 0;
    if (generateDenseCloud && denseCloud.size() > 0)
    {
        if (!DemDomIO::writeDenseCloudXyz(denseCloud, denseXyz, errorMsg))
        {
            return false;
        }
        densePointCount = static_cast<int>(denseCloud.size());
    }

    int vertexCount = 0;
    int faceCount = 0;
    if (options.generateMesh
        && !DemDomIO::writeMeshPlyFromDemGrid(demGrid, meshPly, &vertexCount, &faceCount, errorMsg))
    {
        if (errorMsg && errorMsg->isEmpty())
        {
            *errorMsg = QStringLiteral("DEM 网格模型写出失败");
        }
        if (errorMsg)
        {
            errorMsg->clear();
        }
        vertexCount = 0;
        faceCount = 0;
    }

    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("source_sparse_cloud")] = pointCloudPath;
        output[QStringLiteral("source_point_cloud")] = pointCloudPath;
        output[QStringLiteral("dem_reference")] = relativeDem ? QStringLiteral("relative") : QStringLiteral("absolute");
        output[QStringLiteral("relative_z_offset")] = relativeZOffset;
        output[QStringLiteral("grid_width")] = demGrid.width;
        output[QStringLiteral("grid_height")] = demGrid.height;
        output[QStringLiteral("cell_size_x")] = demGrid.stepX;
        output[QStringLiteral("cell_size_y")] = demGrid.stepY;
        output[QStringLiteral("hole_fill_iterations")] = options.holeFillIterations;
        output[QStringLiteral("hole_fill_min_neighbors")] = options.holeFillMinNeighbors;
        output[QStringLiteral("hole_fill_search_radius")] = options.holeFillSearchRadius;
        output[QStringLiteral("subpixel_bilinear_splat")] = options.useSubPixelBilinearSplat;
        output[QStringLiteral("grid_min_x")] = demGrid.minX;
        output[QStringLiteral("grid_min_y")] = demGrid.minY;
        output[QStringLiteral("depth_png")] = depthPng;
        output[QStringLiteral("dem_tif")] = demTif;
        appendQualityArtifacts(&output, qualityArtifacts);
        output[QStringLiteral("dense_cloud_xyz")] = generateDenseCloud ? denseXyz : QString();
        output[QStringLiteral("dense_point_count")] = densePointCount;
        output[QStringLiteral("mesh_ply")] = faceCount > 0 ? meshPly : QString();
        output[QStringLiteral("vertex_count")] = vertexCount;
        output[QStringLiteral("face_count")] = faceCount;
        *result = output;
    }

    return true;
}

bool TerrainPipeline::estimateOrthoProduct(const QString &demPath,
                                           const QJsonObject &settings,
                                           QJsonObject *result,
                                           QString *errorMsg)
{
    OrthoGenerationOptions options;
    if (!OrthoGenerationOptions::fromJson(settings, &options, errorMsg))
    {
        return false;
    }

    if (options.surfaceType == OrthoSurfaceType::PointCloud)
    {
        const std::shared_ptr<PlaPointCloud> cloud = readTerrainPointCloud(demPath, errorMsg);
        if (!cloud || cloud->size() == 0)
        {
            return false;
        }
        return PointCloudDomGenerator::estimate(*cloud, options, result, errorMsg);
    }

    DemGridData metadata;
    if (!DemDomIO::readDemMetadata(demPath, &metadata, errorMsg))
    {
        return false;
    }

    OrthoOutputGrid planned;
    if (!OrthoProjector::planOutputGrid(metadata, options, &planned, errorMsg))
    {
        return false;
    }

    if (result)
    {
        const double dem_min_x = metadata.minX - 0.5 * metadata.stepX;
        const double dem_min_y = metadata.minY - 0.5 * metadata.stepY;
        const double dem_max_x =
            dem_min_x + static_cast<double>(metadata.width) * metadata.stepX;
        const double dem_max_y =
            dem_min_y + static_cast<double>(metadata.height) * metadata.stepY;
        QJsonObject output;
        output[QStringLiteral("resolved_settings")] =
            planned.resolvedOptions.toResolvedJson();
        output[QStringLiteral("coordinate_system")] =
            metadata.projection.coordinateSystem.isEmpty()
                ? QStringLiteral("Local Coordinates")
                : metadata.projection.coordinateSystem;
        output[QStringLiteral("projection_wkt_present")] =
            !metadata.projection.projectionWkt.isEmpty();
        output[QStringLiteral("dem_min_x")] = dem_min_x;
        output[QStringLiteral("dem_min_y")] = dem_min_y;
        output[QStringLiteral("dem_max_x")] = dem_max_x;
        output[QStringLiteral("dem_max_y")] = dem_max_y;
        output[QStringLiteral("dem_pixel_size_x")] = metadata.stepX;
        output[QStringLiteral("dem_pixel_size_y")] = metadata.stepY;
        output[QStringLiteral("min_x")] = planned.minEdgeX;
        output[QStringLiteral("min_y")] = planned.minEdgeY;
        output[QStringLiteral("max_x")] = planned.maxEdgeX;
        output[QStringLiteral("max_y")] = planned.maxEdgeY;
        output[QStringLiteral("pixel_size_x")] = planned.reference.stepX;
        output[QStringLiteral("pixel_size_y")] = planned.reference.stepY;
        output[QStringLiteral("width")] = planned.reference.width;
        output[QStringLiteral("height")] = planned.reference.height;
        output[QStringLiteral("estimated_memory_bytes")] =
            static_cast<double>(planned.estimatedMemoryBytes);
        *result = output;
    }
    return true;
}

bool TerrainPipeline::generateOrthoProduct(
    const QStringList &images,
    const QString &demPath,
    const QString &outputPath,
    const QJsonObject &settings,
    const QJsonObject &projectMeta,
    QJsonObject *result,
    QString *errorMsg,
    const std::atomic_bool *cancelFlag,
    const OrthoProgressCallback &progressCallback)
{
    if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("正射影像生成已取消");
        }
        return false;
    }

    OrthoGenerationOptions options;
    if (!OrthoGenerationOptions::fromJson(settings, &options, errorMsg))
    {
        return false;
    }

    if (options.surfaceType == OrthoSurfaceType::PointCloud)
    {
        return generatePointCloudOrtho(demPath,
                                       outputPath,
                                       options,
                                       result,
                                       errorMsg,
                                       cancelFlag,
                                       progressCallback);
    }
    if (images.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有可用于 DOM 生成的影像");
        }
        return false;
    }

    if (progressCallback)
    {
        progressCallback(QStringLiteral("读取 DEM"), 0);
    }
    DemGridData demMetadata;
    if (!DemDomIO::readDemMetadata(demPath, &demMetadata, errorMsg))
    {
        return false;
    }
    DemGridData demGrid;
    if (!DemDomIO::readDemRaster(demPath, &demGrid, errorMsg))
    {
        return false;
    }

    cv::Mat dom_image;
    cv::Mat output_valid_mask;
    DemGridData output_reference;
    OrthoGenerationOptions resolved_options = options;
    int selected_camera_count = 0;
    int loaded_camera_count = 0;
    int contributing_camera_count = 0;
    qint64 filled_pixel_count = 0;
    qint64 hole_filled_pixel_count = 0;
    double coverage_ratio = 0.0;
    double min_x = demGrid.minX - 0.5 * demGrid.stepX;
    double min_y = demGrid.minY - 0.5 * demGrid.stepY;
    double max_x = min_x + static_cast<double>(demGrid.width) * demGrid.stepX;
    double max_y = min_y + static_cast<double>(demGrid.height) * demGrid.stepY;
    const bool camera_projected = !projectMeta.isEmpty();

    if (camera_projected)
    {
        double dem_z_offset = 0.0;
        resolveDemVerticalOffsetForOrtho(projectMeta, demPath, &dem_z_offset);
        std::vector<OrthoImageInput> inputs;
        if (!OrthoProjector::buildImageInputs(images, projectMeta, &inputs, errorMsg))
        {
            return false;
        }

        OrthoProjectionResult projection;
        const auto projector_progress =
            [&progressCallback](const QString &stage, int percent)
            {
                if (progressCallback)
                {
                    progressCallback(stage, std::clamp(percent * 9 / 10, 0, 90));
                }
            };
        if (!OrthoProjector::project(demGrid,
                                     inputs,
                                     options,
                                     dem_z_offset,
                                     &projection,
                                     errorMsg,
                                     cancelFlag,
                                     projector_progress))
        {
            return false;
        }

        dom_image = std::move(projection.imageBgr);
        cv::bitwise_or(
            projection.coverageMask,
            projection.holeFilledMask,
            output_valid_mask);
        output_reference = projection.outputGrid.reference;
        resolved_options = projection.outputGrid.resolvedOptions;
        selected_camera_count = projection.selectedCameraCount;
        loaded_camera_count = projection.loadedCameraCount;
        contributing_camera_count = projection.contributingCameraCount;
        filled_pixel_count = projection.filledPixelCount;
        hole_filled_pixel_count = projection.holeFilledPixelCount;
        coverage_ratio = projection.coverageRatio;
        min_x = projection.outputGrid.minEdgeX;
        min_y = projection.outputGrid.minEdgeY;
        max_x = projection.outputGrid.maxEdgeX;
        max_y = projection.outputGrid.maxEdgeY;
    }
    else
    {
        if (options.bounds.enabled
            || options.sizingMode == OrthoSizingMode::MaximumDimension
            || (options.pixelSizeX > 0.0 && options.pixelSizeY > 0.0
                && std::abs(options.pixelSizeX - options.pixelSizeY) > 1e-12))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral(
                    "无相机参数的兼容 DOM 模式不支持自定义区域、最大尺寸或非方形像元");
            }
            return false;
        }
        DomGenerationOptions fallback_options;
        fallback_options.outputResolution =
            options.pixelSizeX > 0.0 ? options.pixelSizeX : options.pixelSizeY;
        fallback_options.imageFormat = DomImageFormat::Png;
        fallback_options.enableSharpnessWeighting = options.sharpnessWeighting;
        fallback_options.enableExposureCompensation = options.colorCorrection;
        fallback_options.minBlendWeight = 0.05;
        if (!DomGenerator::generateFromImages(
                demGrid, images, fallback_options, &dom_image, errorMsg))
        {
            return false;
        }
        output_reference = demGrid;
        output_reference.width = dom_image.cols;
        output_reference.height = dom_image.rows;
        output_reference.stepX =
            demGrid.stepX * static_cast<double>(demGrid.width) / dom_image.cols;
        output_reference.stepY =
            demGrid.stepY * static_cast<double>(demGrid.height) / dom_image.rows;
        output_reference.minX =
            min_x + 0.5 * output_reference.stepX;
        output_reference.minY =
            min_y + 0.5 * output_reference.stepY;
        if (!demGrid.validMask.empty())
        {
            cv::resize(
                demGrid.validMask,
                output_valid_mask,
                dom_image.size(),
                0.0,
                0.0,
                cv::INTER_NEAREST);
        }
        resolved_options.pixelSizeX = output_reference.stepX;
        resolved_options.pixelSizeY = output_reference.stepY;
        filled_pixel_count = cv::countNonZero(demGrid.validMask);
        coverage_ratio = static_cast<double>(filled_pixel_count)
            / static_cast<double>(demGrid.width * demGrid.height);
    }
    output_reference.validMask = output_valid_mask;

    if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("正射影像生成已取消");
        }
        return false;
    }
    if (progressCallback)
    {
        progressCallback(QStringLiteral("写出正射影像"), 95);
    }

    const DomImageFormat output_format =
        (outputPath.endsWith(QStringLiteral(".tif"), Qt::CaseInsensitive)
         || outputPath.endsWith(QStringLiteral(".tiff"), Qt::CaseInsensitive))
        ? DomImageFormat::Tiff
        : DomImageFormat::Png;
    const bool output_geotiff = output_format == DomImageFormat::Tiff;
    if (output_geotiff)
    {
        if (!DemDomIO::writeDomGeoTiff(
                dom_image,
                output_valid_mask,
                output_reference,
                outputPath,
                errorMsg))
        {
            return false;
        }
    }
    else
    {
        cv::Mat north_up_image;
        cv::flip(dom_image, north_up_image, 0);
        if (!output_valid_mask.empty())
        {
            cv::Mat north_up_alpha;
            cv::flip(output_valid_mask, north_up_alpha, 0);
            std::vector<cv::Mat> channels;
            cv::split(north_up_image, channels);
            channels.push_back(north_up_alpha);
            cv::merge(channels, north_up_image);
        }
        if (!DemDomIO::writeDomImage(
                north_up_image, outputPath, output_format, errorMsg))
        {
            return false;
        }
    }

    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")] =
            QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("output_path")] = outputPath;
        output[QStringLiteral("source_image_count")] = images.size();
        output[QStringLiteral("dem_path")] = demPath;
        output[QStringLiteral("resolved_settings")] =
            resolved_options.toResolvedJson();
        output[QStringLiteral("output_resolution")] = output_reference.stepX;
        output[QStringLiteral("pixel_size_x")] = output_reference.stepX;
        output[QStringLiteral("pixel_size_y")] = output_reference.stepY;
        output[QStringLiteral("min_x")] = min_x;
        output[QStringLiteral("min_y")] = min_y;
        output[QStringLiteral("max_x")] = max_x;
        output[QStringLiteral("max_y")] = max_y;
        output[QStringLiteral("dom_georeferenced")] = output_geotiff;
        output[QStringLiteral("has_coverage_alpha")] =
            !output_valid_mask.empty();
        output[QStringLiteral("valid_pixel_count")] =
            output_valid_mask.empty()
                ? 0.0
                : static_cast<double>(cv::countNonZero(output_valid_mask));
        output[QStringLiteral("projection_wkt_present")] =
            !output_reference.projection.projectionWkt.isEmpty();
        output[QStringLiteral("camera_projected")] = camera_projected;
        output[QStringLiteral("selected_camera_count")] = selected_camera_count;
        output[QStringLiteral("loaded_camera_count")] = loaded_camera_count;
        output[QStringLiteral("contributing_camera_count")] =
            contributing_camera_count;
        output[QStringLiteral("filled_pixel_count")] =
            static_cast<double>(filled_pixel_count);
        output[QStringLiteral("hole_filled_pixel_count")] =
            static_cast<double>(hole_filled_pixel_count);
        output[QStringLiteral("coverage_ratio")] = coverage_ratio;
        output[QStringLiteral("width")] = dom_image.cols;
        output[QStringLiteral("height")] = dom_image.rows;
        output[QStringLiteral("algorithm_version")] = camera_projected
            ? QStringLiteral("ortho_projector_v1")
            : QStringLiteral("aligned_composite_legacy_v1");
        *result = output;
    }

    if (progressCallback)
    {
        progressCallback(QStringLiteral("正射影像生成完成"), 100);
    }
    return true;
}

bool TerrainPipeline::generateOrthoProduct(const QStringList &images,
                                           const QString &demPath,
                                           const QString &outputPath,
                                           double resolution,
                                           const QJsonObject &projectMeta,
                                           QJsonObject *result,
                                           QString *errorMsg)
{
    QJsonObject settings;
    if (resolution > 0.0)
    {
        settings[QStringLiteral("pixel_size_x")] = resolution;
        settings[QStringLiteral("pixel_size_y")] = resolution;
    }
    return generateOrthoProduct(images,
                                demPath,
                                outputPath,
                                settings,
                                projectMeta,
                                result,
                                errorMsg,
                                nullptr,
                                {});
}

bool TerrainPipeline::generateOrthoProduct(const QStringList &images,
                                           const QString &demPath,
                                           const QString &outputPath,
                                           double resolution,
                                           QJsonObject *result,
                                           QString *errorMsg)
{
    return generateOrthoProduct(images,
                                demPath,
                                outputPath,
                                resolution,
                                QJsonObject(),
                                result,
                                errorMsg);
}

bool TerrainPipeline::generateFromObjMtl(const QString &objPath,
                                          const QString &outputDir,
                                          double demResolution,
                                          QJsonObject *result,
                                          QString *errorMsg)
{
    // 1. 加载 OBJ + 纹理（plapoint native）
    std::shared_ptr<PlaPointCloud> cloudPtr;
    try
    {
        cloudPtr = plapoint::io::readObj<float>(xjw::common::io::toNativeNarrowPath(objPath));
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("OBJ 读取失败: %1").arg(QString::fromStdString(e.what()));
        return false;
    }

    TerrainMeshInput meshInput;
    meshInput.mesh = std::move(*cloudPtr);

    // Load texture image if referenced
    const std::string &texFile = meshInput.mesh.textureImageFile();
    if (!texFile.empty())
    {
        QString texPath = QFileInfo(objPath).dir().filePath(xjw::common::io::fromUtf8Path(texFile));
        meshInput.texture = xjw::common::io::readImage(texPath, cv::IMREAD_COLOR);
        if (meshInput.texture.empty())
        {
            texPath = QDir::cleanPath(QFileInfo(objPath).absolutePath() + QDir::separator() +
                                      xjw::common::io::fromUtf8Path(texFile));
            meshInput.texture = xjw::common::io::readImage(texPath, cv::IMREAD_COLOR);
        }
    }

    // 2. 生成 DEM（直接用顶点 Z 光栅化）
    DemGenerationOptions demOptions;
    demOptions.gridResolution = demResolution;
    demOptions.holeFillIterations = 6;
    demOptions.holeFillMinNeighbors = 5;
    demOptions.holeFillSearchRadius = 2;
    demOptions.useSubPixelBilinearSplat = true;

    DemGridData demGrid;
    PlaPointCloud denseCloud;
    if (!DemGenerator::generateFromPointCloud(meshInput.mesh, demOptions, &demGrid, &denseCloud, errorMsg))
    {
        return false;
    }

    // 3. 准备输出目录
    const QString productsDir = QDir(outputDir).filePath(QStringLiteral("products"));
    if (!QDir().mkpath(productsDir))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建输出目录: %1").arg(productsDir);
        }
        return false;
    }

    const QString depthPng = QDir(productsDir).filePath(QStringLiteral("depth_map.png"));
    const QString demTif   = QDir(productsDir).filePath(QStringLiteral("dem.tif"));
    const QString domPng   = QDir(productsDir).filePath(QStringLiteral("dom.png"));

    // 4. 写出 DEM（灰度）
    if (!DemDomIO::writeDemPreviewPng(demGrid, depthPng, errorMsg))
    {
        return false;
    }
    if (!DemDomIO::writeDemRaster(demGrid, demTif, DemRasterFormat::Float32Tiff, errorMsg))
    {
        return false;
    }

    // 5. 生成并写出 DOM（彩色）
    bool hasTexture = !meshInput.texture.empty() && meshInput.mesh.hasFaces();
    if (hasTexture)
    {
        cv::Mat domImage;
        if (!DomGenerator::generateFromTexturedMesh(meshInput, demGrid, &domImage, errorMsg))
        {
            return false;
        }
        if (!DemDomIO::writeDomImage(domImage, domPng, DomImageFormat::Png, errorMsg))
        {
            return false;
        }
    }

    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")]  = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("source_obj")]  = objPath;
        output[QStringLiteral("grid_width")]  = demGrid.width;
        output[QStringLiteral("grid_height")] = demGrid.height;
        output[QStringLiteral("cell_size_x")] = demGrid.stepX;
        output[QStringLiteral("cell_size_y")] = demGrid.stepY;
        output[QStringLiteral("dem_tif")]     = demTif;
        output[QStringLiteral("depth_png")]   = depthPng;
        output[QStringLiteral("dom_png")]     = hasTexture ? domPng : QString();
        output[QStringLiteral("has_texture")] = hasTexture;
        *result = output;
    }

    return true;
}

bool TerrainPipeline::generateFromObjMtlDir(const QString &dirPath,
                                             const QString &outputDir,
                                             double demResolution,
                                             QJsonObject *result,
                                             QString *errorMsg)
{
    // 1. 扫描目录下所有 OBJ 文件
    QDir dir(dirPath);
    const QStringList objNames = dir.entryList(QStringList{QStringLiteral("*.obj")}, QDir::Files);
    if (objNames.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("目录中未找到 OBJ 文件: %1").arg(dirPath);
        }
        return false;
    }

    // 2. 加载所有瓦片（同时收集顶点，用于 DEM 整体边界）
    std::vector<TerrainMeshInput> tileInputs;
    tileInputs.reserve(static_cast<std::size_t>(objNames.size()));

    // Accumulate all points for combined cloud
    std::vector<float> allXs, allYs, allZs;

    for (const QString &objName : objNames)
    {
        const QString objPath = dir.filePath(objName);

        TerrainMeshInput tile;
        QString tileLoadError;
        if (!ObjMtlLoader::load(objPath, &tile, &tileLoadError) || tile.mesh.size() == 0)
        {
            continue;
        }

        // Accumulate points for combined cloud
        for (size_t i = 0; i < tile.mesh.size(); ++i)
        {
            auto pt = tile.mesh[i];
            allXs.push_back(pt.x());
            allYs.push_back(pt.y());
            allZs.push_back(pt.z());
        }

        tileInputs.push_back(std::move(tile));
    }

    if (allXs.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("所有 OBJ 瓦片加载均失败: %1").arg(dirPath);
        }
        return false;
    }

    // Build combined point cloud
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> combinedPts(allXs.size(), 3);
    for (size_t i = 0; i < allXs.size(); ++i)
    {
        combinedPts(static_cast<plamatrix::Index>(i), 0) = allXs[i];
        combinedPts(static_cast<plamatrix::Index>(i), 1) = allYs[i];
        combinedPts(static_cast<plamatrix::Index>(i), 2) = allZs[i];
    }
    PlaPointCloud combinedCloud(std::move(combinedPts));

    // 3. 用合并后的顶点云生成全局 DEM
    DemGenerationOptions demOptions;
    demOptions.gridResolution = demResolution;
    demOptions.holeFillIterations = 6;
    demOptions.holeFillMinNeighbors = 5;
    demOptions.holeFillSearchRadius = 2;
    demOptions.useSubPixelBilinearSplat = true;

    DemGridData demGrid;
    PlaPointCloud denseCloud;
    if (!DemGenerator::generateFromPointCloud(combinedCloud, demOptions, &demGrid, &denseCloud, errorMsg))
    {
        return false;
    }

    // 4. 准备输出目录
    const QString productsDir = QDir(outputDir).filePath(QStringLiteral("products"));
    if (!QDir().mkpath(productsDir))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建输出目录: %1").arg(productsDir);
        }
        return false;
    }

    const QString depthPng = QDir(productsDir).filePath(QStringLiteral("depth_map.png"));
    const QString demTif   = QDir(productsDir).filePath(QStringLiteral("dem.tif"));
    const QString domPng   = QDir(productsDir).filePath(QStringLiteral("dom.png"));

    // 5. 写出 DEM
    if (!DemDomIO::writeDemPreviewPng(demGrid, depthPng, errorMsg))
    {
        return false;
    }
    if (!DemDomIO::writeDemRaster(demGrid, demTif, DemRasterFormat::Float32Tiff, errorMsg))
    {
        return false;
    }

    // 6. 逐瓦片光栅化 DOM（in-place 拼接到同一图像）
    cv::Mat domImage; // 第一个瓦片时 generateFromTexturedMesh 会自动分配
    bool hasAnyTexture = false;

    for (const TerrainMeshInput &tile : tileInputs)
    {
        if (tile.texture.empty() || !tile.mesh.hasFaces())
        {
            continue;
        }

        hasAnyTexture = true;
        QString tileError;
        DomGenerator::generateFromTexturedMesh(tile, demGrid, &domImage, &tileError);
    }

    if (hasAnyTexture)
    {
        if (!DemDomIO::writeDomImage(domImage, domPng, DomImageFormat::Png, errorMsg))
        {
            return false;
        }
    }

    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")]  = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("source_dir")]  = dirPath;
        output[QStringLiteral("tile_count")]  = static_cast<int>(tileInputs.size());
        output[QStringLiteral("grid_width")]  = demGrid.width;
        output[QStringLiteral("grid_height")] = demGrid.height;
        output[QStringLiteral("cell_size_x")] = demGrid.stepX;
        output[QStringLiteral("cell_size_y")] = demGrid.stepY;
        output[QStringLiteral("dem_tif")]     = demTif;
        output[QStringLiteral("depth_png")]   = depthPng;
        output[QStringLiteral("dom_png")]     = hasAnyTexture ? domPng : QString();
        output[QStringLiteral("has_texture")] = hasAnyTexture;
        *result = output;
    }

    return true;
}

bool TerrainPipeline::generateSmallBodyGlobalProducts(
    const QString &surfacePath,
    const QString &outputDir,
    const SmallBodyGlobalOptions &options,
    QJsonObject *result,
    QString *errorMsg,
    const std::atomic_bool *cancelFlag,
    const OrthoProgressCallback &progressCallback)
{
    SmallBodyGlobalProducts products;
    if (!SmallBodyGlobalProductGenerator::generate(
            surfacePath, outputDir, options, &products, errorMsg,
            cancelFlag, progressCallback))
    {
        return false;
    }
    if (result)
    {
        QJsonObject output = products.report;
        output[QStringLiteral("report_json")] = products.reportPath;
        output[QStringLiteral("preview_png")] = products.previewPath;
        output[QStringLiteral("radial_dem_tif")] = products.radialDemPath;
        output[QStringLiteral("elevation_dem_tif")] = products.elevationDemPath;
        output[QStringLiteral("dom_tif")] = products.domPath;
        output[QStringLiteral("reliability_tif")] = products.reliabilityPath;
        output[QStringLiteral("coverage_tif")] = products.coveragePath;
        output[QStringLiteral("ambiguity_tif")] = products.ambiguityPath;
        *result = output;
    }
    return true;
}

// =============================================================================
// generateFromObjMtlWithAsteroidProjections
// =============================================================================

bool TerrainPipeline::generateFromObjMtlWithAsteroidProjections(
    const QString &objPath,
    const QString &outputDir,
    double demResolution,
    QJsonObject *result,
    QString *errorMsg)
{
    // 1. 加载 OBJ + 纹理 (plapoint native)
    std::shared_ptr<PlaPointCloud> cloudPtr;
    try
    {
        cloudPtr = plapoint::io::readObj<float>(xjw::common::io::toNativeNarrowPath(objPath));
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("OBJ 读取失败: %1").arg(QString::fromStdString(e.what()));
        return false;
    }

    TerrainMeshInput meshInput;
    meshInput.mesh = std::move(*cloudPtr);

    // Load texture
    const std::string &texFile = meshInput.mesh.textureImageFile();
    if (!texFile.empty())
    {
        QString texPath = QFileInfo(objPath).dir().filePath(xjw::common::io::fromUtf8Path(texFile));
        meshInput.texture = xjw::common::io::readImage(texPath, cv::IMREAD_COLOR);
        if (meshInput.texture.empty())
        {
            texPath = QDir::cleanPath(QFileInfo(objPath).absolutePath() + QDir::separator() +
                                      xjw::common::io::fromUtf8Path(texFile));
            meshInput.texture = xjw::common::io::readImage(texPath, cv::IMREAD_COLOR);
        }
    }

    if (meshInput.mesh.size() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("OBJ 网格不含顶点，无法生成小行星地形产品");
        }
        return false;
    }

    // 2. 准备输出目录
    const QString productsDir = QDir(outputDir).filePath(QStringLiteral("products"));
    if (!QDir().mkpath(productsDir))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建输出目录: %1").arg(productsDir);
        }
        return false;
    }

    const bool hasTexture = !meshInput.texture.empty() && meshInput.mesh.hasFaces();

    // 3. 计算质心（全部投影共用）
    const AsteroidBodyCenter center = AsteroidProjection::computeCenter(meshInput.mesh);

    // 4. 为三轴椭球投影拟合椭球（提前计算，避免重复 PCA）
    std::array<double, 9> triaxRotMatrix = AsteroidProjection::identityRotation();
    const TriaxialEllipsoidParams triaxEllipsoid =
        AsteroidProjection::fitEllipsoid(meshInput.mesh, center, &triaxRotMatrix);

    // 5. 写出正射 DEM 预览（不含投影，仅供质检）
    {
        DemGenerationOptions baseOptions;
        baseOptions.gridResolution = demResolution;
        baseOptions.holeFillIterations = 6;
        baseOptions.holeFillMinNeighbors = 5;
        baseOptions.holeFillSearchRadius = 2;
        baseOptions.useSubPixelBilinearSplat = true;

        DemGridData baseGrid;
        PlaPointCloud denseCloud;
        if (DemGenerator::generateFromPointCloud(meshInput.mesh, baseOptions, &baseGrid, &denseCloud, nullptr))
        {
            const QString depthPng = QDir(productsDir).filePath(QStringLiteral("depth_map.png"));
            DemDomIO::writeDemPreviewPng(baseGrid, depthPng, nullptr);
        }
    }

    // 6. 三种投影的元数据配置
    struct ProjectionJob
    {
        AsteroidProjectionType type;
        std::array<double, 9> rotMatrix;
        TriaxialEllipsoidParams ellipsoid;
        const char *suffix;           // 文件名后缀，如 "stere"
    };

    const ProjectionJob jobs[] = {
        {AsteroidProjectionType::Stereographic,
         AsteroidProjection::identityRotation(),
         {center.referenceRadius, center.referenceRadius, center.referenceRadius},
         "stere"},
        {AsteroidProjectionType::AzimuthalEquidistant,
         AsteroidProjection::identityRotation(),
         {center.referenceRadius, center.referenceRadius, center.referenceRadius},
         "aeqd"},
        {AsteroidProjectionType::TriaxialEllipsoid,
         triaxRotMatrix,
         triaxEllipsoid,
         "triaxial"},
    };

    QJsonObject outputJson;
    outputJson[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    outputJson[QStringLiteral("source_obj")]  = objPath;
    outputJson[QStringLiteral("has_texture")] = hasTexture;
    outputJson[QStringLiteral("centroid_x")]  = center.cx;
    outputJson[QStringLiteral("centroid_y")]  = center.cy;
    outputJson[QStringLiteral("centroid_z")]  = center.cz;
    outputJson[QStringLiteral("reference_radius")] = center.referenceRadius;
    outputJson[QStringLiteral("ellipsoid_a")] = triaxEllipsoid.a;
    outputJson[QStringLiteral("ellipsoid_b")] = triaxEllipsoid.b;
    outputJson[QStringLiteral("ellipsoid_c")] = triaxEllipsoid.c;

    const std::size_t nPts = meshInput.mesh.size();

    for (const ProjectionJob &job : jobs)
    {
        AsteroidProjectionParams params;
        params.type = job.type;
        params.referenceRadius = center.referenceRadius;
        params.autoFitEllipsoid = false;
        params.ellipsoid = job.ellipsoid;

        // 6a. 构建投影后的点云（用于 DEM 生成）
        std::vector<float> projXs, projYs, projZs;
        projXs.reserve(nPts);
        projYs.reserve(nPts);
        projZs.reserve(nPts);
        for (std::size_t i = 0; i < nPts; ++i)
        {
            auto pt = meshInput.mesh[i];
            double u = 0, v = 0, elev = 0;
            AsteroidProjection::projectPoint(
                static_cast<double>(pt.x()),
                static_cast<double>(pt.y()),
                static_cast<double>(pt.z()),
                center, params, job.rotMatrix,
                &u, &v, &elev);
            projXs.push_back(static_cast<float>(u));
            projYs.push_back(static_cast<float>(v));
            projZs.push_back(static_cast<float>(elev));
        }

        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> projPts(projXs.size(), 3);
        for (size_t i = 0; i < projXs.size(); ++i)
        {
            projPts(static_cast<plamatrix::Index>(i), 0) = projXs[i];
            projPts(static_cast<plamatrix::Index>(i), 1) = projYs[i];
            projPts(static_cast<plamatrix::Index>(i), 2) = projZs[i];
        }
        PlaPointCloud projCloud(std::move(projPts));

        // 6b. 生成投影 DEM
        DemGenerationOptions demOptions;
        demOptions.gridResolution = demResolution;
        demOptions.holeFillIterations = 6;
        demOptions.holeFillMinNeighbors = 5;
        demOptions.holeFillSearchRadius = 2;
        demOptions.useSubPixelBilinearSplat = true;
        demOptions.projection.projectionWkt =
            AsteroidProjection::buildProjectionWkt(job.type, center, job.ellipsoid);

        DemGridData projDemGrid;
        PlaPointCloud denseCloud;
        QString demError;
        if (!DemGenerator::generateFromPointCloud(projCloud, demOptions, &projDemGrid, &denseCloud, &demError))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("投影 %1 DEM 生成失败: %2")
                                .arg(QString::fromLatin1(job.suffix))
                                .arg(demError);
            }
            return false;
        }

        // 6c. 写出投影 DEM GeoTIFF
        const QString demTif = QDir(productsDir).filePath(
            QStringLiteral("dem_%1.tif").arg(QString::fromLatin1(job.suffix)));
        QString demMsgBuf;
        if (!DemDomIO::writeDemRaster(projDemGrid, demTif, DemRasterFormat::Float32Tiff, &demMsgBuf))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("写出 %1 DEM 失败: %2")
                                .arg(QString::fromLatin1(job.suffix))
                                .arg(demMsgBuf);
            }
            return false;
        }
        outputJson[QStringLiteral("dem_%1").arg(QString::fromLatin1(job.suffix))] = demTif;
        outputJson[QStringLiteral("grid_width_%1").arg(QString::fromLatin1(job.suffix))] = projDemGrid.width;
        outputJson[QStringLiteral("grid_height_%1").arg(QString::fromLatin1(job.suffix))] = projDemGrid.height;

        // 6d. 若有纹理则构建投影网格并生成 DOM
        if (hasTexture)
        {
            // 用投影后顶点坐标覆盖位置，但保留 UV 纹理坐标和三角面（深拷贝 mesh）
            TerrainMeshInput projMesh;
            projMesh.texture = meshInput.texture;
            // Manually deep-copy PointCloud (DenseMatrix is move-only)
            {
                const auto &srcPts = meshInput.mesh.points();
                plamatrix::DenseMatrix<float, plamatrix::Device::CPU> ptsCopy(srcPts.rows(), srcPts.cols());
                for (plamatrix::Index r = 0; r < srcPts.rows(); ++r)
                    for (int c = 0; c < 3; ++c)
                        ptsCopy(r, c) = srcPts(r, c);
                projMesh.mesh = PlaPointCloud(std::move(ptsCopy));
            }
            if (meshInput.mesh.hasNormals()) projMesh.mesh.setNormals(*meshInput.mesh.normals());
            if (meshInput.mesh.hasColors()) projMesh.mesh.setColors(*meshInput.mesh.colors());
            if (meshInput.mesh.hasTextureCoords()) projMesh.mesh.setTextureCoords(*meshInput.mesh.textureCoords());
            if (meshInput.mesh.hasFaces()) projMesh.mesh.setFaces(*meshInput.mesh.faces());
            if (meshInput.mesh.hasFaceTextureIndices()) projMesh.mesh.setFaceTextureIndices(*meshInput.mesh.faceTextureIndices());
            projMesh.mesh.setMaterialLibraryFile(meshInput.mesh.materialLibraryFile());
            projMesh.mesh.setTextureImageFile(meshInput.mesh.textureImageFile());
            for (std::size_t i = 0; i < nPts; ++i)
            {
                auto pt = meshInput.mesh[i];
                double u = 0, v = 0, elev = 0;
                AsteroidProjection::projectPoint(
                    static_cast<double>(pt.x()),
                    static_cast<double>(pt.y()),
                    static_cast<double>(pt.z()),
                    center, params, job.rotMatrix,
                    &u, &v, &elev);
                projMesh.mesh.points()(static_cast<plamatrix::Index>(i), 0) = static_cast<float>(u);
                projMesh.mesh.points()(static_cast<plamatrix::Index>(i), 1) = static_cast<float>(v);
                projMesh.mesh.points()(static_cast<plamatrix::Index>(i), 2) = static_cast<float>(elev);
            }

            cv::Mat domImage;
            QString domError;
            if (DomGenerator::generateFromTexturedMesh(projMesh, projDemGrid, &domImage, &domError))
            {
                const QString domTif = QDir(productsDir).filePath(
                    QStringLiteral("dom_%1.tif").arg(QString::fromLatin1(job.suffix)));
                QString domMsgBuf;
                if (!DemDomIO::writeDomGeoTiff(domImage, projDemGrid, domTif, &domMsgBuf))
                {
                    if (errorMsg)
                    {
                        *errorMsg = QStringLiteral("写出 %1 DOM GeoTIFF 失败: %2")
                                        .arg(QString::fromLatin1(job.suffix))
                                        .arg(domMsgBuf);
                    }
                    return false;
                }
                outputJson[QStringLiteral("dom_%1").arg(QString::fromLatin1(job.suffix))] = domTif;
            }
            else
            {
                outputJson[QStringLiteral("dom_%1").arg(QString::fromLatin1(job.suffix))] = QString();
            }
        }
        else
        {
            outputJson[QStringLiteral("dom_%1").arg(QString::fromLatin1(job.suffix))] = QString();
        }
    }

    if (result)
    {
        *result = outputJson;
    }

    return true;
}

bool TerrainPipeline::generateDemFromDepthMaps(const std::vector<cv::Mat> &depthMaps,
                                               const std::vector<Camera> &cameras,
                                               const QString &outputDir,
                                               QJsonObject *result,
                                               QString *errorMsg)
{
    if (depthMaps.empty() || cameras.empty())
    {
        if (errorMsg)
            *errorMsg = QStringLiteral("深度图或相机参数为空");
        return false;
    }

    DemGenerationOptions options;
    options.gridResolution = 0.0;
    options.holeFillIterations = 20;
    options.holeFillMinNeighbors = 3;
    options.holeFillSearchRadius = 5;
    options.rasterFormat = DemRasterFormat::Float32Tiff;

    DemGridData demGrid;
    QString genError;
    if (!DemGenerator::generateFromDepthMaps(depthMaps, cameras, options, &demGrid, &genError))
    {
        if (errorMsg)
            *errorMsg = QStringLiteral("从深度图生成 DEM 失败: %1").arg(genError);
        return false;
    }

    // When outputting 3-band XYZ, keep absolute coordinates.
    // Only apply relative offset for single-band Z-only DEM.
    double relativeZOffset = 0.0;
    if (!demGrid.hasWorldXY())
    {
        double minElev = 0.0;
        cv::minMaxLoc(demGrid.elevation, &minElev, nullptr, nullptr, nullptr, demGrid.validMask);
        relativeZOffset = minElev;
        for (int r = 0; r < demGrid.height; ++r)
            for (int c = 0; c < demGrid.width; ++c)
                if (demGrid.validMask.at<uchar>(r, c) != 0)
                    demGrid.elevation.at<float>(r, c) -= static_cast<float>(relativeZOffset);
    }

    const QString productsDir = QDir(outputDir).filePath(QStringLiteral("products"));
    QDir().mkpath(productsDir);
    const QString demTif = QDir(productsDir).filePath(QStringLiteral("dem.tif"));
    const QString depthPng = QDir(productsDir).filePath(QStringLiteral("depth_map.png"));

    if (!DemDomIO::writeDemRaster(demGrid, demTif, options.rasterFormat, errorMsg))
        return false;
    if (!DemDomIO::writeDemPreviewPng(demGrid, depthPng, errorMsg))
        return false;

    DemQualityArtifacts qualityArtifacts;
    if (!DemDomIO::writeDemQualityRasters(demGrid, productsDir, &qualityArtifacts, errorMsg))
        return false;

    // Write dense point cloud PLY if XYZ data is available
    QString plyPath;
    int plyPointCount = 0;
    if (demGrid.hasWorldXY())
    {
        plyPath = QDir(productsDir).filePath(QStringLiteral("dense_cloud.ply"));
        DemDomIO::writeDenseCloudPly(demGrid, plyPath, &plyPointCount);
    }

    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("dem_tif")] = demTif;
        output[QStringLiteral("depth_png")] = depthPng;
        appendQualityArtifacts(&output, qualityArtifacts);
        if (!plyPath.isEmpty())
        {
            output[QStringLiteral("dense_cloud_ply")] = plyPath;
            output[QStringLiteral("dense_cloud_points")] = plyPointCount;
        }
        output[QStringLiteral("grid_width")] = demGrid.width;
        output[QStringLiteral("grid_height")] = demGrid.height;
        output[QStringLiteral("relative_z_offset")] = relativeZOffset;
        output[QStringLiteral("dem_reference")] = QStringLiteral("relative");
        output[QStringLiteral("method")] = QStringLiteral("depth_map_direct");
        *result = output;
    }

    return true;
}

} // namespace xjw
