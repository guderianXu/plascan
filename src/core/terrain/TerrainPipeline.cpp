#include "TerrainPipeline.h"

#include "DemDomIO.h"
#include "DemDomTypes.h"
#include "DemGenerator.h"
#include "DomGenerator.h"
#include "io/ObjMtlLoader.h"
#include "projection/AsteroidProjection.h"
#include "Camera.h"

#include "io/PointCloudIO.h"
#include "data/PointCloud.h"

#include <QDir>
#include <QDateTime>
#include <QFileInfo>
#include <QJsonArray>
#include <QSet>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
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

QString normalizePathForOrtho(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
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

    for (const QJsonValue &value : demResults)
    {
        const QJsonObject record = value.toObject();
        const QString recordDemPath = record.value(QStringLiteral("dem_tif")).toString();
        if (recordDemPath.isEmpty())
        {
            continue;
        }

        const QString normalizedRecordPath = normalizePathForOrtho(recordDemPath);
        if (normalizedRecordPath != normalizedDemPath
            && QFileInfo(recordDemPath).fileName() != QFileInfo(demPath).fileName())
        {
            continue;
        }

        if (record.value(QStringLiteral("dem_reference")).toString() == QStringLiteral("relative"))
        {
            if (zOffset)
            {
                *zOffset = record.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
            }
            return true;
        }

        return false;
    }

    return false;
}

bool pathTokenMatchesImageForOrtho(const QString &token, const QString &imagePath)
{
    if (token.isEmpty() || imagePath.isEmpty())
    {
        return false;
    }

    const QString normalizedToken = normalizePathForOrtho(token);
    const QString normalizedImage = normalizePathForOrtho(imagePath);
    if (normalizedToken == normalizedImage)
    {
        return true;
    }

    if (QFileInfo(token).fileName() == QFileInfo(imagePath).fileName())
    {
        return true;
    }

    return QFileInfo(token).completeBaseName() == QFileInfo(imagePath).completeBaseName();
}

bool cameraFromJsonForOrtho(const QJsonObject &cameraObject, xjw::Camera *camera)
{
    if (!camera || cameraObject.isEmpty())
    {
        return false;
    }

    const QJsonArray centerArray = cameraObject.value(QStringLiteral("C")).toArray();
    const QJsonArray rotationArray = cameraObject.value(QStringLiteral("R")).toArray();
    if (centerArray.size() < 3 || rotationArray.size() < 9)
    {
        return false;
    }

    std::array<double, 3> center{{centerArray.at(0).toDouble(),
                                  centerArray.at(1).toDouble(),
                                  centerArray.at(2).toDouble()}};
    if (cameraObject.value(QStringLiteral("camera_center_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive)
        == 0)
    {
        center[0] /= 1000.0;
        center[1] /= 1000.0;
        center[2] /= 1000.0;
    }

    std::array<double, 9> rotation{};
    for (int index = 0; index < 9; ++index)
    {
        rotation[index] = rotationArray.at(index).toDouble();
    }

    const double pitch = cameraObject.value(QStringLiteral("pitch")).toDouble(1.0);
    const bool intrinsicsInMillimeters =
        cameraObject.value(QStringLiteral("intrinsics_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive)
        == 0;
    const double fu = cameraObject.value(QStringLiteral("fu")).toDouble();
    const double fv = cameraObject.value(QStringLiteral("fv")).toDouble();
    const double cu = cameraObject.value(QStringLiteral("cu")).toDouble();
    const double cv = cameraObject.value(QStringLiteral("cv")).toDouble();

    if (intrinsicsInMillimeters)
    {
        camera->setIntrinsicsMillimeters(fu, fv, cu, cv, pitch);
    }
    else
    {
        camera->setPixelPitch(pitch);
        camera->setIntrinsics(fu, fv, cu, cv);
    }

    camera->setAxisDirections(cameraObject.value(QStringLiteral("u_direction")).toInt(1),
                              cameraObject.value(QStringLiteral("v_direction")).toInt(1));
    camera->setDepthAxisFlipped(cameraObject.value(QStringLiteral("depth_axis_flipped")).toBool(false));
    camera->setDistortion(cameraObject.value(QStringLiteral("k1")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("k2")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("k3")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("p1")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("p2")).toDouble(0.0));
    camera->setPose(rotation, center);
    return true;
}

struct OrthoCameraFrame
{
    QString imagePath;
    xjw::Camera camera;
    cv::Mat imageBgr;
    double gain = 1.0;
};

cv::Size resolveOrthoOutputSize(const xjw::DemGridData &demGrid, double outputResolution)
{
    if (outputResolution > 0.0)
    {
        const double scaleX = demGrid.stepX / outputResolution;
        const double scaleY = demGrid.stepY / outputResolution;
        return cv::Size(std::max(1, static_cast<int>(std::round(demGrid.width * scaleX))),
                        std::max(1, static_cast<int>(std::round(demGrid.height * scaleY))));
    }

    return cv::Size(demGrid.width, demGrid.height);
}

double computeLumaMean(const cv::Mat &imageBgr)
{
    cv::Mat gray;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}

cv::Vec3f sampleBilinearBgr(const cv::Mat &imageBgr, double u, double v)
{
    const double x = std::clamp(u, 0.0, std::max(0.0, static_cast<double>(imageBgr.cols - 1)));
    const double y = std::clamp(v, 0.0, std::max(0.0, static_cast<double>(imageBgr.rows - 1)));

    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, imageBgr.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, imageBgr.rows - 1);
    const int x1 = std::min(x0 + 1, imageBgr.cols - 1);
    const int y1 = std::min(y0 + 1, imageBgr.rows - 1);

    const float fx = static_cast<float>(x - x0);
    const float fy = static_cast<float>(y - y0);

    const cv::Vec3f c00 = imageBgr.at<cv::Vec3b>(y0, x0);
    const cv::Vec3f c10 = imageBgr.at<cv::Vec3b>(y0, x1);
    const cv::Vec3f c01 = imageBgr.at<cv::Vec3b>(y1, x0);
    const cv::Vec3f c11 = imageBgr.at<cv::Vec3b>(y1, x1);

    return (1.0f - fx) * (1.0f - fy) * c00
           + fx * (1.0f - fy) * c10
           + (1.0f - fx) * fy * c01
           + fx * fy * c11;
}

bool buildOrthoFrames(const QJsonObject &projectMeta,
                      const QStringList &selectedImages,
                      std::vector<OrthoCameraFrame> *frames,
                      QString *errorMsg)
{
    if (!frames)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("内部错误：DOM 相机帧输出参数无效");
        }
        return false;
    }

    frames->clear();
    QSet<QString> selectedNormalized;
    for (const QString &path : selectedImages)
    {
        selectedNormalized.insert(normalizePathForOrtho(path));
    }

    const QJsonArray imageArray = projectMeta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : imageArray)
    {
        const QJsonObject imageObject = value.toObject();
        const QString imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (imagePath.isEmpty())
        {
            continue;
        }

        bool selected = selectedNormalized.isEmpty();
        if (!selected)
        {
            const QString normalized = normalizePathForOrtho(imagePath);
            if (selectedNormalized.contains(normalized))
            {
                selected = true;
            }
            else
            {
                for (const QString &candidate : selectedImages)
                {
                    if (pathTokenMatchesImageForOrtho(imagePath, candidate))
                    {
                        selected = true;
                        break;
                    }
                }
            }
        }
        if (!selected)
        {
            continue;
        }

        xjw::Camera camera;
        if (!cameraFromJsonForOrtho(imageObject.value(QStringLiteral("camera")).toObject(), &camera)
            || !camera.isValid())
        {
            continue;
        }

        const QString normalizedPath = normalizePathForOrtho(imagePath);
        cv::Mat imageBgr = cv::imread(normalizedPath.toStdString(), cv::IMREAD_COLOR);
        if (imageBgr.empty())
        {
            continue;
        }

        OrthoCameraFrame frame;
        frame.imagePath = normalizedPath;
        frame.camera = camera;
        frame.imageBgr = std::move(imageBgr);
        frames->push_back(std::move(frame));
    }

    if (frames->empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("未找到可用于正射投影的有效影像+相机参数，请先完成相机初始化/空三。\n"
                                       "并确认所选影像路径与项目元数据一致。");
        }
        return false;
    }

    const double targetLuma = computeLumaMean(frames->front().imageBgr);
    for (OrthoCameraFrame &frame : *frames)
    {
        const double luma = computeLumaMean(frame.imageBgr);
        frame.gain = std::clamp(targetLuma / std::max(1e-6, luma), 0.7, 1.3);
    }

    return true;
}

bool renderOrthoFromDemAndCameras(const xjw::DemGridData &demGrid,
                                  const std::vector<OrthoCameraFrame> &frames,
                                  double outputResolution,
                                  double demZOffset,
                                  cv::Mat *domImage,
                                  int *contributingPixelCount)
{
    if (!domImage)
    {
        return false;
    }

    const cv::Size outputSize = resolveOrthoOutputSize(demGrid, outputResolution);
    if (outputSize.width <= 0 || outputSize.height <= 0)
    {
        return false;
    }

    *domImage = cv::Mat(outputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    const double scaleX = static_cast<double>(outputSize.width) / static_cast<double>(demGrid.width);
    const double scaleY = static_cast<double>(outputSize.height) / static_cast<double>(demGrid.height);

    int filledPixels = 0;
    for (int row = 0; row < outputSize.height; ++row)
    {
        const double demRow = ((static_cast<double>(row) + 0.5) / scaleY) - 0.5;
        const int demRowIdx = std::clamp(static_cast<int>(std::lround(demRow)), 0, demGrid.height - 1);

        for (int col = 0; col < outputSize.width; ++col)
        {
            const double demCol = ((static_cast<double>(col) + 0.5) / scaleX) - 0.5;
            const int demColIdx = std::clamp(static_cast<int>(std::lround(demCol)), 0, demGrid.width - 1);
            if (demGrid.validMask.at<uchar>(demRowIdx, demColIdx) == 0)
            {
                continue;
            }

            const double world[3] = {
                demGrid.minX + demGrid.stepX * demCol,
                demGrid.minY + demGrid.stepY * demRow,
                static_cast<double>(demGrid.elevation.at<float>(demRowIdx, demColIdx)) + demZOffset
            };

            cv::Vec3f bestColor(0.0f, 0.0f, 0.0f);
            double bestScore = 0.0;
            for (const OrthoCameraFrame &frame : frames)
            {
                double pixel[2] = {0.0, 0.0};
                if (!frame.camera.projectWorldPoint(world, pixel))
                {
                    continue;
                }

                const double u = pixel[0];
                const double v = pixel[1];
                if (u < 0.0 || v < 0.0
                    || u >= static_cast<double>(frame.imageBgr.cols - 1)
                    || v >= static_cast<double>(frame.imageBgr.rows - 1))
                {
                    continue;
                }

                const cv::Vec3f sampled = sampleBilinearBgr(frame.imageBgr, u, v) * static_cast<float>(frame.gain);
                const double du = (u - frame.camera.principalX()) / std::max(1.0, frame.camera.focalX());
                const double dv = (v - frame.camera.principalY()) / std::max(1.0, frame.camera.focalY());
                const double viewWeight = 1.0 / (1.0 + du * du + dv * dv);
                const double edgeDistance = std::min(
                    std::min(u, static_cast<double>(frame.imageBgr.cols - 1) - u),
                    std::min(v, static_cast<double>(frame.imageBgr.rows - 1) - v));
                const double edgeWeight = std::clamp(edgeDistance / 20.0, 0.0, 1.0);
                const double score = viewWeight * edgeWeight;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestColor = sampled;
                }
            }

            if (bestScore <= 0.0)
            {
                continue;
            }

            (*domImage).at<cv::Vec3b>(row, col) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(bestColor[0], 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(bestColor[1], 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(bestColor[2], 0.0f, 255.0f)));
            ++filledPixels;
        }
    }

    if (contributingPixelCount)
    {
        *contributingPixelCount = filledPixels;
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
    pointcloud::PointCloud pointCloud;
    pointcloud::PointCloudIOResult ioResult;
    if (!pointcloud::readPointCloud(pointCloudPath.toStdString(), &pointCloud, {}, &ioResult)) 
    {
        if (errorMsg) *errorMsg = QStringLiteral("读取点云失败: %1").arg(QString::fromStdString(ioResult.errorMessage));
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
    pointcloud::PointCloud denseCloud;
    if (!DemGenerator::generateFromPointCloud(pointCloud, options, &demGrid, &denseCloud, errorMsg)) 
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

    int densePointCount = 0;
    if (generateDenseCloud && !denseCloud.empty()) 
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
        output[QStringLiteral("dense_cloud_xyz")] = generateDenseCloud ? denseXyz : QString();
        output[QStringLiteral("dense_point_count")] = densePointCount;
        output[QStringLiteral("mesh_ply")] = faceCount > 0 ? meshPly : QString();
        output[QStringLiteral("vertex_count")] = vertexCount;
        output[QStringLiteral("face_count")] = faceCount;
        *result = output;
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
    if (images.isEmpty()) 
    {
        if (errorMsg) *errorMsg = QStringLiteral("没有可用于 DOM 生成的影像");
        return false;
    }

    DemGridData demGrid;
    if (!DemDomIO::readDemRaster(demPath, &demGrid, errorMsg)) {
        return false;
    }

    cv::Mat domImage;
    int contributingPixels = 0;
    if (!projectMeta.isEmpty())
    {
        double demZOffset = 0.0;
        resolveDemVerticalOffsetForOrtho(projectMeta, demPath, &demZOffset);

        std::vector<OrthoCameraFrame> frames;
        if (!buildOrthoFrames(projectMeta, images, &frames, errorMsg))
        {
            return false;
        }

        if (!renderOrthoFromDemAndCameras(demGrid,
                                          frames,
                                          resolution,
                                          demZOffset,
                                          &domImage,
                                          &contributingPixels))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("DOM 正射投影失败：无法基于 DEM 与相机参数生成结果");
            }
            return false;
        }
    }
    else
    {
        DomGenerationOptions fallbackOptions;
        fallbackOptions.outputResolution = resolution;
        fallbackOptions.imageFormat = DomImageFormat::Png;
        fallbackOptions.enableSharpnessWeighting = true;
        fallbackOptions.enableExposureCompensation = true;
        fallbackOptions.minBlendWeight = 0.05;
        if (!DomGenerator::generateFromImages(demGrid, images, fallbackOptions, &domImage, errorMsg))
        {
            return false;
        }
    }

    const DomImageFormat outputFormat = (outputPath.endsWith(QStringLiteral(".tif"), Qt::CaseInsensitive)
                                         || outputPath.endsWith(QStringLiteral(".tiff"), Qt::CaseInsensitive))
        ? DomImageFormat::Tiff
        : DomImageFormat::Png;

    const bool outputGeoTiff = outputFormat == DomImageFormat::Tiff;
    if (outputGeoTiff)
    {
        if (!DemDomIO::writeDomGeoTiff(domImage, demGrid, outputPath, errorMsg))
        {
            return false;
        }
    }
    else if (!DemDomIO::writeDomImage(domImage, outputPath, outputFormat, errorMsg))
    {
        return false;
    }

    const double effectiveResolution = resolution > 0.0 ? resolution : demGrid.stepX;

    if (result) {
        QJsonObject output;
        output[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        output[QStringLiteral("output_path")] = outputPath;
        output[QStringLiteral("source_image_count")] = images.size();
        output[QStringLiteral("dem_path")] = demPath;
        output[QStringLiteral("output_resolution")] = effectiveResolution;
        output[QStringLiteral("dom_georeferenced")] = outputGeoTiff;
        output[QStringLiteral("projection_wkt_present")] = !demGrid.projection.projectionWkt.isEmpty();
        output[QStringLiteral("camera_projected")] = !projectMeta.isEmpty();
        output[QStringLiteral("filled_pixel_count")] = contributingPixels;
        output[QStringLiteral("width")] = domImage.cols;
        output[QStringLiteral("height")] = domImage.rows;
        *result = output;
    }

    return true;
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
    // 1. 加载 OBJ + MTL + 纹理
    TerrainMeshInput meshInput;
    if (!pointcloud::ObjMtlLoader::load(objPath, &meshInput, errorMsg))
    {
        return false;
    }

    // 2. 生成 DEM（直接用顶点 Z 光栅化）
    DemGenerationOptions demOptions;
    demOptions.gridResolution = demResolution;
    demOptions.holeFillIterations = 6;
    demOptions.holeFillMinNeighbors = 5;
    demOptions.holeFillSearchRadius = 2;
    demOptions.useSubPixelBilinearSplat = true;

    DemGridData demGrid;
    pointcloud::PointCloud denseCloud;
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
    bool hasTexture = !meshInput.texture.empty() && !meshInput.mesh.faces().empty();
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

    pointcloud::PointCloud combinedCloud;

    for (const QString &objName : objNames)
    {
        const QString objPath = dir.filePath(objName);
        TerrainMeshInput tile;
        QString tileError;
        if (!pointcloud::ObjMtlLoader::load(objPath, &tile, &tileError))
        {
            // 跳过损坏的瓦片，不中断整体
            continue;
        }

        if (!tile.mesh.empty())
        {
            combinedCloud.append(tile.mesh);
            tileInputs.push_back(std::move(tile));
        }
    }

    if (combinedCloud.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("所有 OBJ 瓦片加载均失败: %1").arg(dirPath);
        }
        return false;
    }

    // 3. 用合并后的顶点云生成全局 DEM
    DemGenerationOptions demOptions;
    demOptions.gridResolution = demResolution;
    demOptions.holeFillIterations = 6;
    demOptions.holeFillMinNeighbors = 5;
    demOptions.holeFillSearchRadius = 2;
    demOptions.useSubPixelBilinearSplat = true;

    DemGridData demGrid;
    pointcloud::PointCloud denseCloud;
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
        if (tile.texture.empty() || tile.mesh.faces().empty())
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
    // 1. 加载 OBJ + MTL + 纹理
    TerrainMeshInput meshInput;
    if (!pointcloud::ObjMtlLoader::load(objPath, &meshInput, errorMsg))
    {
        return false;
    }

    if (meshInput.mesh.empty())
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

    const bool hasTexture = !meshInput.texture.empty() && !meshInput.mesh.faces().empty();

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
        pointcloud::PointCloud denseCloud;
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

    const auto &origPositions = meshInput.mesh.positions();
    const std::size_t nPts = origPositions.size();

    for (const ProjectionJob &job : jobs)
    {
        AsteroidProjectionParams params;
        params.type = job.type;
        params.referenceRadius = center.referenceRadius;
        params.autoFitEllipsoid = false;
        params.ellipsoid = job.ellipsoid;

        // 6a. 构建投影后的点云（用于 DEM 生成）
        pointcloud::PointCloud projCloud;
        projCloud.reserve(nPts);
        for (std::size_t i = 0; i < nPts; ++i)
        {
            const auto &p = origPositions[i];
            double u = 0, v = 0, elev = 0;
            AsteroidProjection::projectPoint(
                static_cast<double>(p.x),
                static_cast<double>(p.y),
                static_cast<double>(p.z),
                center, params, job.rotMatrix,
                &u, &v, &elev);
            projCloud.addPoint(pointcloud::Point3f{
                static_cast<float>(u),
                static_cast<float>(v),
                static_cast<float>(elev)});
        }

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
        pointcloud::PointCloud denseCloud;
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
            projMesh.mesh = meshInput.mesh;  // 深拷贝（包含 UV + faces）
            for (std::size_t i = 0; i < nPts; ++i)
            {
                const auto &p = origPositions[i];
                double u = 0, v = 0, elev = 0;
                AsteroidProjection::projectPoint(
                    static_cast<double>(p.x),
                    static_cast<double>(p.y),
                    static_cast<double>(p.z),
                    center, params, job.rotMatrix,
                    &u, &v, &elev);
                projMesh.mesh.setPosition(i, pointcloud::Point3f{
                    static_cast<float>(u),
                    static_cast<float>(v),
                    static_cast<float>(elev)});
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
