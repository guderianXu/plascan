#include "ProjectWorkflowOperations.h"

#include "filtering/SparsePointCloudProcessor.h"
#include "project/SparseResultQuality.h"
#include "TerrainPipeline.h"
#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>

#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace xjw::core::project
{

using xjw::common::project::isProductionSparseResult;
using xjw::common::project::sparseResultBlockingReason;

namespace {

std::uint8_t colorByteFromJson(const QJsonValue &value)
{
    const int channel = value.toInt(255);
    if (channel < 0)
    {
        return 0;
    }
    if (channel > 255)
    {
        return 255;
    }
    return static_cast<std::uint8_t>(channel);
}

bool applyColorArray(const QJsonArray &array,
                     xjw::SparsePointCloudPoint *point)
{
    if (!point || array.size() < 3)
    {
        return false;
    }

    point->hasColor = true;
    point->red = colorByteFromJson(array.at(0));
    point->green = colorByteFromJson(array.at(1));
    point->blue = colorByteFromJson(array.at(2));
    return true;
}

bool hasAnyPointColor(const std::vector<xjw::SparsePointCloudPoint> &points)
{
    for (const xjw::SparsePointCloudPoint &point : points)
    {
        if (point.hasColor)
        {
            return true;
        }
    }
    return false;
}

void copyColorsFromPlyCloud(const plapoint::PointCloud<float, plamatrix::Device::CPU> &cloud,
                            std::vector<xjw::SparsePointCloudPoint> *points)
{
    if (!points || !cloud.hasColors() || cloud.size() != points->size())
    {
        return;
    }

    const auto *colors = cloud.colors();
    if (!colors)
    {
        return;
    }

    for (std::size_t i = 0; i < points->size(); ++i)
    {
        xjw::SparsePointCloudPoint &point = (*points)[i];
        if (point.hasColor)
        {
            continue;
        }

        const auto row = static_cast<plamatrix::Index>(i);
        point.hasColor = true;
        point.red = colors->getValue(row, 0);
        point.green = colors->getValue(row, 1);
        point.blue = colors->getValue(row, 2);
    }
}

void copyColorsFromSourcePly(const QString &path,
                             std::vector<xjw::SparsePointCloudPoint> *points)
{
    if (!points || points->empty() || path.isEmpty() || !QFileInfo::exists(path))
    {
        return;
    }

    try
    {
        const auto cloud = plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(path));
        if (cloud)
        {
            copyColorsFromPlyCloud(*cloud, points);
        }
    }
    catch (const std::exception &)
    {
        // 颜色补全是向后兼容路径；sidecar 本身仍可独立驱动后处理。
    }
}

bool loadJsonObjectFile(const QString &path,
                        QJsonObject *object,
                        QString *errorMessage = nullptr)
{
    if (!object)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少 JSON 输出对象");
        }
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取点级结果文件: %1").arg(path);
        }
        return false;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    if (!doc.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("点级结果文件格式无效: %1").arg(path);
        }
        return false;
    }

    *object = doc.object();
    return true;
}

std::vector<xjw::SparsePointCloudPoint> sparsePointsFromJson(const QJsonArray &pointsArray)
{
    std::vector<xjw::SparsePointCloudPoint> points;
    points.reserve(pointsArray.size());
    for (qsizetype source_index = 0; source_index < pointsArray.size(); ++source_index)
    {
        const QJsonValue value = pointsArray.at(source_index);
        const QJsonObject pointObj = value.toObject();
        const QJsonArray xyz = pointObj.value(QStringLiteral("point_xyz")).toArray();
        if (xyz.size() < 3)
        {
            continue;
        }

        xjw::SparsePointCloudPoint point;
        point.x = xyz.at(0).toDouble();
        point.y = xyz.at(1).toDouble();
        point.z = xyz.at(2).toDouble();
        point.rmsReprojPx = pointObj.value(QStringLiteral("rms_reproj_px")).toDouble(
            pointObj.value(QStringLiteral("rms_after")).toDouble());
        point.minTriAngleDeg = pointObj.value(QStringLiteral("min_tri_angle_deg")).toDouble();
        point.reconstructionUncertainty = pointObj
            .value(QStringLiteral("reconstruction_uncertainty"))
            .toDouble(std::numeric_limits<double>::quiet_NaN());
        point.projectionAccuracy = pointObj.value(QStringLiteral("projection_accuracy"))
            .toDouble(std::numeric_limits<double>::quiet_NaN());
        point.trackLen = pointObj.value(QStringLiteral("track_len")).toInt(0);
        const QJsonValue cleanValue = pointObj.value(QStringLiteral("clean_tie_points"));
        const QJsonObject clean = cleanValue.toObject();
        const QJsonValue hasGeometryValue = clean.value(QStringLiteral("has_projection_geometry"));
        const QJsonValue imageCountValue = clean.value(QStringLiteral("image_count"));
        const double cleanReprojectionError =
            clean.value(QStringLiteral("reprojection_error")).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double cleanProjectionAccuracy =
            clean.value(QStringLiteral("projection_accuracy")).toDouble(std::numeric_limits<double>::quiet_NaN());
        const double cleanImageCountValue = imageCountValue.toDouble(std::numeric_limits<double>::quiet_NaN());
        const int cleanImageCount = std::isfinite(cleanImageCountValue) &&
                                            cleanImageCountValue <= static_cast<double>(std::numeric_limits<int>::max())
                                        ? static_cast<int>(cleanImageCountValue)
                                        : -1;
        double cleanReconstructionUncertainty = std::numeric_limits<double>::quiet_NaN();
        bool validCleanReconstructionUncertainty = hasGeometryValue.isBool();
        if (validCleanReconstructionUncertainty && hasGeometryValue.toBool())
        {
            if (clean.value(QStringLiteral("reconstruction_uncertainty_infinite")).toBool(false))
            {
                cleanReconstructionUncertainty = std::numeric_limits<double>::infinity();
            }
            else
            {
                cleanReconstructionUncertainty = clean.value(QStringLiteral("reconstruction_uncertainty"))
                                                     .toDouble(std::numeric_limits<double>::quiet_NaN());
                validCleanReconstructionUncertainty =
                    std::isfinite(cleanReconstructionUncertainty) && cleanReconstructionUncertainty > 0.0;
            }
        }
        point.hasCleanTiePointReprojectionError =
            cleanValue.isObject() && std::isfinite(cleanReprojectionError) && cleanReprojectionError >= 0.0;
        point.hasCleanTiePointReconstructionUncertainty = cleanValue.isObject() && validCleanReconstructionUncertainty;
        point.hasCleanTiePointImageCount = cleanValue.isObject() && imageCountValue.isDouble() &&
                                           cleanImageCount >= 0 &&
                                           std::trunc(cleanImageCountValue) == cleanImageCountValue;
        point.hasCleanTiePointProjectionAccuracy =
            cleanValue.isObject() && std::isfinite(cleanProjectionAccuracy) && cleanProjectionAccuracy >= 0.0;
        if (point.hasCleanTiePointReprojectionError)
        {
            point.cleanTiePointReprojectionError = cleanReprojectionError;
        }
        if (point.hasCleanTiePointReconstructionUncertainty)
        {
            point.cleanTiePointReconstructionUncertainty = cleanReconstructionUncertainty;
        }
        if (point.hasCleanTiePointImageCount)
        {
            point.cleanTiePointImageCount = cleanImageCount;
        }
        if (point.hasCleanTiePointProjectionAccuracy)
        {
            point.cleanTiePointProjectionAccuracy = cleanProjectionAccuracy;
        }
        point.sourceIndex = static_cast<std::size_t>(source_index);
        if (!applyColorArray(pointObj.value(QStringLiteral("color_rgb")).toArray(), &point))
        {
            if (!applyColorArray(pointObj.value(QStringLiteral("rgb")).toArray(), &point))
            {
                applyColorArray(pointObj.value(QStringLiteral("color")).toArray(), &point);
            }
        }
        points.push_back(point);
    }
    return points;
}

bool isExternalPlySource(const SparsePointContext &context,
                         const QJsonObject &settings)
{
    const QString sourceKind = settings.value(QStringLiteral("sourceKind")).toString();
    if (sourceKind == QLatin1String("external_ply"))
    {
        return true;
    }
    if (sourceKind == QLatin1String("project_result"))
    {
        return false;
    }

    const QString externalPath =
        settings.value(QStringLiteral("externalSparseCloudPath")).toString().trimmed();
    return !externalPath.isEmpty() ||
           (context.sidecarPath.isEmpty() && !context.sparseCloudPath.isEmpty());
}

QString externalPlySourcePath(const SparsePointContext &context,
                              const QJsonObject &settings)
{
    QString path = settings.value(QStringLiteral("externalSparseCloudPath")).toString().trimmed();
    if (path.isEmpty())
    {
        path = context.sparseCloudPath;
    }
    return QDir::cleanPath(path);
}

bool loadSparsePointsFromPly(const QString &path,
                             std::vector<xjw::SparsePointCloudPoint> *points,
                             QString *errorMessage)
{
    if (!points)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少点云输出对象");
        }
        return false;
    }
    if (path.isEmpty() || !QFileInfo::exists(path))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("外部 PLY 点云不存在: %1").arg(path);
        }
        return false;
    }

    try
    {
        const auto cloud = plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(path));
        if (!cloud || cloud->size() == 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("外部 PLY 点云为空: %1").arg(path);
            }
            return false;
        }

        points->clear();
        points->reserve(cloud->size());
        const auto &matrix = cloud->points();
        const bool hasColors = cloud->hasColors() && cloud->colors();
        for (std::size_t i = 0; i < cloud->size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            xjw::SparsePointCloudPoint point;
            point.x = matrix.getValue(row, 0);
            point.y = matrix.getValue(row, 1);
            point.z = matrix.getValue(row, 2);
            point.rmsReprojPx = 0.0;
            point.minTriAngleDeg = 0.0;
            point.reconstructionUncertainty = std::numeric_limits<double>::quiet_NaN();
            point.projectionAccuracy = std::numeric_limits<double>::quiet_NaN();
            point.trackLen = 0;
            point.sourceIndex = i;
            if (hasColors)
            {
                const auto *colors = cloud->colors();
                point.hasColor = true;
                point.red = colors->getValue(row, 0);
                point.green = colors->getValue(row, 1);
                point.blue = colors->getValue(row, 2);
            }
            points->push_back(point);
        }
        return true;
    }
    catch (const std::exception &ex)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取外部 PLY 点云失败: %1 (%2)")
                                .arg(path, QString::fromStdString(ex.what()));
        }
        return false;
    }
}

bool loadSparsePointSource(const SparsePointContext &context,
                           const QJsonObject &settings,
                           QJsonObject *sourceRoot,
                           std::vector<xjw::SparsePointCloudPoint> *points,
                           QString *errorMessage)
{
    if (!sourceRoot || !points)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少稀疏点输入对象");
        }
        return false;
    }

    if (isExternalPlySource(context, settings))
    {
        const QString path = externalPlySourcePath(context, settings);
        if (!loadSparsePointsFromPly(path, points, errorMessage))
        {
            return false;
        }

        QJsonObject root;
        root[QStringLiteral("source_kind")] = QStringLiteral("external_ply");
        root[QStringLiteral("source_ply")] = path;
        root[QStringLiteral("quality_metrics_available")] = false;
        root[QStringLiteral("point_count")] = static_cast<int>(points->size());
        *sourceRoot = root;
        return true;
    }

    if (!loadJsonObjectFile(context.sidecarPath, sourceRoot, errorMessage))
    {
        return false;
    }

    *points = sparsePointsFromJson(sourceRoot->value(QStringLiteral("points")).toArray());
    copyColorsFromSourcePly(context.sparseCloudPath, points);
    if (!sourceRoot->contains(QStringLiteral("quality_metrics_available")))
    {
        (*sourceRoot)[QStringLiteral("quality_metrics_available")] = true;
    }
    return true;
}

QJsonObject sparsePointToJson(const xjw::SparsePointCloudPoint &point)
{
    QJsonObject object;
    QJsonArray xyz;
    xyz.append(point.x);
    xyz.append(point.y);
    xyz.append(point.z);
    object[QStringLiteral("point_xyz")] = xyz;
    object[QStringLiteral("rms_reproj_px")] = point.rmsReprojPx;
    object[QStringLiteral("min_tri_angle_deg")] = point.minTriAngleDeg;
    if (std::isfinite(point.reconstructionUncertainty))
    {
        object[QStringLiteral("reconstruction_uncertainty")] =
            point.reconstructionUncertainty;
    }
    if (std::isfinite(point.projectionAccuracy))
    {
        object[QStringLiteral("projection_accuracy")] = point.projectionAccuracy;
    }
    object[QStringLiteral("track_len")] = point.trackLen;
    if (point.hasColor)
    {
        object[QStringLiteral("color_rgb")] = QJsonArray{
            static_cast<int>(point.red),
            static_cast<int>(point.green),
            static_cast<int>(point.blue)
        };
    }
    return object;
}

QJsonArray sparsePointsToJson(const std::vector<xjw::SparsePointCloudPoint> &points,
                              const QJsonArray &sourcePoints)
{
    QJsonArray array;
    for (const xjw::SparsePointCloudPoint &point : points)
    {
        const bool has_source_point =
            point.sourceIndex != xjw::SparsePointCloudPoint::kInvalidSourceIndex &&
            point.sourceIndex < static_cast<std::size_t>(sourcePoints.size()) &&
            sourcePoints.at(static_cast<qsizetype>(point.sourceIndex)).isObject();
        array.append(has_source_point
                         ? sourcePoints.at(static_cast<qsizetype>(point.sourceIndex)).toObject()
                         : sparsePointToJson(point));
    }
    return array;
}

QJsonObject sourceQualityMetadata(const QJsonObject &sourceRoot)
{
    const QJsonObject nested = sourceRoot.value(QStringLiteral("quality")).toObject();
    return nested.isEmpty() ? sourceRoot : nested;
}

QJsonObject refreshSparsePointQuality(const QJsonObject &sourceRoot,
                                      const QJsonArray &filteredPoints,
                                      const QStringList &selectedImages,
                                      const QString &parentSidecar)
{
    if (!sourceRoot.value(QStringLiteral("quality_metrics_available")).toBool(true))
    {
        return {};
    }

    const QJsonObject nested_source_quality =
        sourceRoot.value(QStringLiteral("quality")).toObject();
    const QJsonObject source_quality = sourceQualityMetadata(sourceRoot);
    const int registered_image_count = source_quality.value(
        QStringLiteral("registered_image_count")).toInt(
            source_quality.value(QStringLiteral("camera_count")).toInt(selectedImages.size()));
    const int input_image_count = source_quality.value(QStringLiteral("input_image_count")).toInt(
        source_quality.value(QStringLiteral("total_image_count")).toInt(selectedImages.size()));
    const bool ba_applied = source_quality.value(QStringLiteral("ba_applied")).toBool(false);

    QString source_result_kind =
        xjw::common::project::sparseResultKind(sourceRoot);
    if (source_result_kind == xjw::common::project::kSparseResultKindSparsePostprocess)
    {
        source_result_kind = source_quality.value(QStringLiteral("source_result_kind")).toString();
    }

    // 只保留原有 quality 扩展字段。旧 sidecar 可能把少量质量统计放在根级，
    // 根对象还包含完整 points 数组，不能把整棵根对象复制进 quality。
    QJsonObject refreshed = nested_source_quality;
    const QJsonObject calculated = xjw::common::project::buildSparseQualityMetadata(
        filteredPoints,
        registered_image_count,
        ba_applied,
        xjw::common::project::kSparseResultKindSparsePostprocess,
        source_result_kind,
        parentSidecar,
        input_image_count);
    for (auto it = calculated.begin(); it != calculated.end(); ++it)
    {
        refreshed[it.key()] = it.value();
    }
    refreshed[QStringLiteral("point_count")] = filteredPoints.size();
    return refreshed;
}

bool writeSparsePointCloudPly(const QString &path,
                              const std::vector<xjw::SparsePointCloudPoint> &points,
                              QString *errorMessage = nullptr)
{
    try
    {
        using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> matrix(points.size(), 3);
        const bool writeColors = hasAnyPointColor(points);
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(points.size(), 3);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            const xjw::SparsePointCloudPoint &point = points[i];
            matrix(row, 0) = static_cast<float>(point.x);
            matrix(row, 1) = static_cast<float>(point.y);
            matrix(row, 2) = static_cast<float>(point.z);
            if (writeColors)
            {
                colors(row, 0) = point.hasColor ? point.red : 255;
                colors(row, 1) = point.hasColor ? point.green : 255;
                colors(row, 2) = point.hasColor ? point.blue : 255;
            }
        }

        PlaCloud cloud(std::move(matrix));
        if (writeColors)
        {
            cloud.setColors(std::move(colors));
        }
        plapoint::io::writePly<float>(
            xjw::common::io::toNativeNarrowPath(path), cloud, plapoint::io::PlyFormat::ASCII);
        return true;
    }
    catch (const std::exception &ex)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入稀疏点云文件: %1 (%2)")
                                .arg(path, QString::fromStdString(ex.what()));
        }
        return false;
    }
}

QJsonObject makeSparsePointSidecar(QJsonObject sourceRoot,
                                   const std::vector<xjw::SparsePointCloudPoint> &points,
                                   const QString &operation,
                                   const QJsonObject &settings,
                                   const QJsonObject &summary,
                                   const QStringList &selectedImages,
                                   const QString &parentSidecar)
{
    const QJsonArray sourcePoints = sourceRoot.value(QStringLiteral("points")).toArray();
    const QJsonArray filteredPoints = sparsePointsToJson(points, sourcePoints);
    sourceRoot[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    sourceRoot[QStringLiteral("operation")] = operation;
    sourceRoot[QStringLiteral("settings")] = settings;
    sourceRoot[QStringLiteral("summary")] = summary;
    sourceRoot[QStringLiteral("selected_images")] = QJsonArray::fromStringList(selectedImages);
    sourceRoot[QStringLiteral("parent_sidecar")] = parentSidecar;
    sourceRoot[QStringLiteral("point_count")] = static_cast<int>(points.size());
    sourceRoot[QStringLiteral("points")] = filteredPoints;

    const QJsonObject quality = refreshSparsePointQuality(
        sourceRoot, filteredPoints, selectedImages, parentSidecar);
    if (!quality.isEmpty())
    {
        sourceRoot = xjw::common::project::mergeSparseQualityIntoRecord(sourceRoot, quality);
        sourceRoot[QStringLiteral("point_count")] = filteredPoints.size();
    }
    return sourceRoot;
}

} // namespace

QString sparseOperationDisplayName(const QString &operation)
{
    if (operation == QLatin1String("triangulation"))
    {
        return QStringLiteral("两视预览云");
    }
    if (operation == QLatin1String("outlier_removal"))
    {
        return QStringLiteral("离群点剔除");
    }
    if (operation == QLatin1String("sparse_refine"))
    {
        return QStringLiteral("稀疏点云精修");
    }
    if (operation == QLatin1String("bundle_adjust"))
    {
        return QStringLiteral("平差稀疏点云");
    }
    if (operation == QLatin1String("spatial_cleanup"))
    {
        return QStringLiteral("空间清理点云");
    }
    return QStringLiteral("稀疏点云");
}

int findLatestAtResultIndex(const QJsonObject &meta,
                            const QString &operation)
{
    const QJsonArray results = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = results.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = results.at(index).toObject();
        if (!operation.isEmpty())
        {
            const QString recordOperation = record.value(QStringLiteral("operation")).toString();
            if (!recordOperation.isEmpty() && recordOperation != operation)
            {
                continue;
            }
        }
        return index;
    }
    return -1;
}

int findLatestProductionAtResultIndex(const QJsonObject &meta)
{
    const QJsonArray results = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = results.size() - 1; index >= 0; --index)
    {
        if (isProductionSparseResult(results.at(index).toObject()))
        {
            return index;
        }
    }
    return -1;
}

bool writeJsonObjectFile(const QString &path,
                         const QJsonObject &object,
                         QString *errorMessage)
{
    const OperationResult result = writeJsonObjectFileResult(path, object);
    if (!result.ok && errorMessage)
    {
        *errorMessage = result.errorMessage;
    }
    return result.ok;
}

OperationResult writeJsonObjectFileResult(const QString &path,
                                          const QJsonObject &object)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return {false, QStringLiteral("无法写入文件: %1").arg(path)};
    }

    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    file.close();
    return {true, QString()};
}

TerrainPipelineResult runDemProducts(const QString &sparsePath,
                                     const QString &outputDir,
                                     double demResolution,
                                     const QString &demType,
                                     bool genPointCloud)
{
    TerrainPipelineResult result;
    result.ok = xjw::TerrainPipeline::generateDemProducts(sparsePath,
                                                          outputDir,
                                                          demResolution,
                                                          demType,
                                                          genPointCloud,
                                                          &result.payload,
                                                          &result.error);
    return result;
}

TerrainPipelineResult runOrthoProduct(const QStringList &images,
                                      const QString &demPath,
                                      const QString &outputPath,
                                      const QJsonObject &settings,
                                      const QJsonObject &projectMeta,
                                      const std::atomic_bool *cancelFlag,
                                      const std::function<void(const QString &, int)> &progressCallback)
{
    TerrainPipelineResult result;
    result.ok = xjw::TerrainPipeline::generateOrthoProduct(images,
                                                           demPath,
                                                           outputPath,
                                                           settings,
                                                           projectMeta,
                                                           &result.payload,
                                                           &result.error,
                                                           cancelFlag,
                                                           progressCallback);
    return result;
}

bool resolveSparsePointContext(const QJsonObject &meta,
                               int requestedIndex,
                               SparsePointContext *context,
                               QString *errorMessage)
{
    if (!context)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少稀疏点上下文");
        }
        return false;
    }

    const QJsonArray results = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (results.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("当前项目中还没有可用于处理的稀疏点云结果");
        }
        return false;
    }

    if (requestedIndex >= 0 && requestedIndex < results.size())
    {
        const QJsonObject record = results.at(requestedIndex).toObject();
        if (!isProductionSparseResult(record))
        {
            if (errorMessage)
            {
                const QString reason = sparseResultBlockingReason(record);
                *errorMessage = reason.isEmpty()
                    ? QStringLiteral("所选稀疏点云不是正式 SfM/BA 结果，无法继续执行该操作")
                    : reason;
            }
            return false;
        }

        const QJsonObject files = record.value(QStringLiteral("files")).toObject();
        QString sidecarPath = files.value(QStringLiteral("sparse_cloud_points_json")).toString();
        const QString outputDir = record.value(QStringLiteral("output_dir")).toString();
        if (sidecarPath.isEmpty() && !outputDir.isEmpty())
        {
            const QString candidate = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
            if (QFileInfo::exists(candidate))
            {
                sidecarPath = candidate;
            }
        }

        if (!sidecarPath.isEmpty() && QFileInfo::exists(sidecarPath))
        {
            context->sourceResultIndex = requestedIndex;
            context->outputDir = outputDir;
            context->sparseCloudPath = files.value(QStringLiteral("sparse_cloud_xyz")).toString();
            context->sidecarPath = sidecarPath;
            context->selectedImages.clear();
            for (const QJsonValue &imageValue : record.value(QStringLiteral("selected_images")).toArray())
            {
                context->selectedImages.append(imageValue.toString());
            }
            return true;
        }
    }

    for (int index = results.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = results.at(index).toObject();
        if (!isProductionSparseResult(record))
        {
            continue;
        }

        const QJsonObject files = record.value(QStringLiteral("files")).toObject();
        QString sidecarPath = files.value(QStringLiteral("sparse_cloud_points_json")).toString();
        const QString outputDir = record.value(QStringLiteral("output_dir")).toString();
        if (sidecarPath.isEmpty() && !outputDir.isEmpty())
        {
            const QString candidate = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
            if (QFileInfo::exists(candidate))
            {
                sidecarPath = candidate;
            }
        }

        if (sidecarPath.isEmpty() || !QFileInfo::exists(sidecarPath))
        {
            continue;
        }

        context->sourceResultIndex = index;
        context->outputDir = outputDir;
        context->sparseCloudPath = files.value(QStringLiteral("sparse_cloud_xyz")).toString();
        context->sidecarPath = sidecarPath;
        context->selectedImages.clear();
        for (const QJsonValue &imageValue : record.value(QStringLiteral("selected_images")).toArray())
        {
            context->selectedImages.append(imageValue.toString());
        }
        return true;
    }

    if (errorMessage)
    {
        *errorMessage = QStringLiteral("未找到可用的正式 SfM/BA 稀疏点云结果。请先运行空中三角测量，而不是两视预览三角化。");
    }
    return false;
}

SparsePointContextResult resolveSparsePointContextResult(const QJsonObject &meta,
                                                         int requestedIndex)
{
    SparsePointContextResult result;
    QString errorMessage;
    result.status.ok = resolveSparsePointContext(meta,
                                                 requestedIndex,
                                                 &result.context,
                                                 &errorMessage);
    result.status.errorMessage = errorMessage;
    return result;
}

bool runSparsePointOutlierRemoval(const SparsePointContext &context,
                                  const QJsonObject &settings,
                                  const QString &outputDir,
                                  SparsePointOperationResult *result,
                                  QString *errorMessage)
{
    if (!result)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少输出对象");
        }
        return false;
    }

    QJsonObject sourceRoot;
    std::vector<xjw::SparsePointCloudPoint> points;
    if (!loadSparsePointSource(context, settings, &sourceRoot, &points, errorMessage))
    {
        return false;
    }
    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("点级 sidecar 中没有可处理的稀疏点");
        }
        return false;
    }

    const int inputCount = static_cast<int>(points.size());
    const bool qualityMetricsAvailable =
        sourceRoot.value(QStringLiteral("quality_metrics_available")).toBool(true);
    QJsonObject summary;
    summary[QStringLiteral("input_points")] = inputCount;

    xjw::SparsePointCloudFilterOptions options;
    options.filterByReprojError = qualityMetricsAvailable &&
                                  settings.value(QStringLiteral("filterByReprojError")).toBool();
    options.maxReprojError = settings.value(QStringLiteral("maxReprojError")).toDouble(2.5);
    options.filterByTrackLen = qualityMetricsAvailable &&
                               settings.value(QStringLiteral("filterByTrackLen")).toBool();
    options.minTrackLen = settings.value(QStringLiteral("minTrackLen")).toInt(3);
    options.filterByTriAngle = qualityMetricsAvailable &&
                               settings.value(QStringLiteral("filterByTriAngle")).toBool();
    options.minTriAngleDeg = settings.value(QStringLiteral("minTriAngleDeg")).toDouble(2.0);
    options.filterByReconstructionUncertainty = qualityMetricsAvailable
        && settings.value(QStringLiteral("filterByReconstructionUncertainty")).toBool();
    options.maxReconstructionUncertainty = settings
        .value(QStringLiteral("maxReconstructionUncertainty"))
        .toDouble(10.0);
    options.filterByProjectionAccuracy = qualityMetricsAvailable
        && settings.value(QStringLiteral("filterByProjectionAccuracy")).toBool();
    options.maxProjectionAccuracy = settings
        .value(QStringLiteral("maxProjectionAccuracy"))
        .toDouble(2.0);
    options.filterByStatistical = settings.value(QStringLiteral("filterByStatistical")).toBool();
    options.statK = settings.value(QStringLiteral("statK")).toInt(16);
    options.statStdDevMul = settings.value(QStringLiteral("statStdDevMul")).toDouble(2.5);
    options.filterByDensity = settings.value(QStringLiteral("filterByDensity")).toBool(false);
    options.densityRadius = settings.value(QStringLiteral("densityRadius")).toDouble(0.5);
    options.densityMinNeighbors = settings.value(QStringLiteral("densityMinNeighbors")).toInt(5);

    const bool referenceCleanTiePoints =
        settings.value(QStringLiteral("cleanTiePointsReferenceSemantics")).toBool(false);
    const bool usesReferenceCleanTiePointMetric = options.filterByReprojError || options.filterByTrackLen ||
                                                  options.filterByReconstructionUncertainty ||
                                                  options.filterByProjectionAccuracy;
    if (referenceCleanTiePoints && usesReferenceCleanTiePointMetric)
    {
        if (!qualityMetricsAvailable)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("当前成果没有 Clean Tie Points 参考算法所需的逐点质量数据");
            }
            return false;
        }
        if (options.filterByTrackLen)
        {
            const int imageCountLevel =
                settings.value(QStringLiteral("imageCountLevel")).toInt(std::max(0, options.minTrackLen - 1));
            if (imageCountLevel < 0 || imageCountLevel == std::numeric_limits<int>::max())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("Clean Tie Points 图像计数阈值无效");
                }
                return false;
            }
            options.minTrackLen = imageCountLevel + 1;
        }
        for (xjw::SparsePointCloudPoint& point : points)
        {
            if ((options.filterByReprojError && !point.hasCleanTiePointReprojectionError) ||
                (options.filterByReconstructionUncertainty && !point.hasCleanTiePointReconstructionUncertainty) ||
                (options.filterByTrackLen && !point.hasCleanTiePointImageCount) ||
                (options.filterByProjectionAccuracy && !point.hasCleanTiePointProjectionAccuracy))
            {
                if (errorMessage)
                {
                    *errorMessage =
                        QStringLiteral("当前成果缺少 Clean Tie Points 参考算法指标，请重新运行空中三角测量");
                }
                return false;
            }
            if (options.filterByReprojError)
            {
                point.rmsReprojPx = point.cleanTiePointReprojectionError;
            }
            if (options.filterByReconstructionUncertainty)
            {
                point.reconstructionUncertainty = point.cleanTiePointReconstructionUncertainty;
            }
            if (options.filterByTrackLen)
            {
                point.trackLen = point.cleanTiePointImageCount;
            }
            if (options.filterByProjectionAccuracy)
            {
                point.projectionAccuracy = point.cleanTiePointProjectionAccuracy;
            }
        }
    }

    xjw::SparsePointCloudOptimizeOptions optimizeOptions;
    optimizeOptions.filterOptions = options;
    const xjw::SparsePointCloudOptimizeResult optimizeResult =
        xjw::SparsePointCloudProcessor::optimize(points, optimizeOptions);
    points = optimizeResult.points;
    const xjw::SparsePointCloudFilterStats stats = optimizeResult.rounds.empty()
        ? xjw::SparsePointCloudFilterStats{}
        : optimizeResult.rounds.front().filterStats;
    summary[QStringLiteral("removed_by_reproj")] = stats.removedByReprojError;
    summary[QStringLiteral("removed_by_track_len")] = stats.removedByTrackLen;
    summary[QStringLiteral("removed_by_tri_angle")] = stats.removedByTriAngle;
    summary[QStringLiteral("removed_by_reconstruction_uncertainty")] =
        stats.removedByReconstructionUncertainty;
    summary[QStringLiteral("removed_by_projection_accuracy")] =
        stats.removedByProjectionAccuracy;
    summary[QStringLiteral("removed_by_statistical")] = stats.removedByStatistical;
    summary[QStringLiteral("removed_by_density")] = stats.removedByDensity;
    summary[QStringLiteral("output_points")] = static_cast<int>(points.size());
    summary[QStringLiteral("removed_total")] = inputCount - static_cast<int>(points.size());
    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("过滤结果为空，请放宽阈值后重试");
        }
        return false;
    }

    QDir().mkpath(outputDir);
    const QString sparseCloudPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_filtered.ply"));
    const QString sidecarPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
    if (!writeSparsePointCloudPly(sparseCloudPath, points, errorMessage))
    {
        return false;
    }

    const QJsonObject sidecar = makeSparsePointSidecar(sourceRoot,
                                                       points,
                                                       QStringLiteral("outlier_removal"),
                                                       settings,
                                                       summary,
                                                       context.selectedImages,
                                                       context.sidecarPath);
    if (!writeJsonObjectFile(sidecarPath, sidecar, errorMessage))
    {
        return false;
    }

    result->outputDir = outputDir;
    result->sparseCloudPath = sparseCloudPath;
    result->sidecarPath = sidecarPath;
    result->inputCount = inputCount;
    result->outputCount = static_cast<int>(points.size());
    QJsonObject files;
    files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
    result->extraRecord[QStringLiteral("files")] = files;
    result->extraRecord[QStringLiteral("source")] = QStringLiteral("sparse_outlier_removal");
    result->extraRecord[QStringLiteral("operation")] = QStringLiteral("outlier_removal");
    result->extraRecord[QStringLiteral("source_result_index")] = context.sourceResultIndex;
    if (sourceRoot.value(QStringLiteral("source_kind")).toString() == QLatin1String("external_ply"))
    {
        result->extraRecord[QStringLiteral("source_ply")] =
            sourceRoot.value(QStringLiteral("source_ply")).toString();
    }
    result->extraRecord[QStringLiteral("operation_settings")] = settings;
    result->extraRecord[QStringLiteral("operation_summary")] = summary;
    const QJsonObject quality = sidecar.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        result->extraRecord =
            xjw::common::project::mergeSparseQualityIntoRecord(result->extraRecord, quality);
    }
    return true;
}

bool runSparsePointLocalOptim(const SparsePointContext &context,
                              const QJsonObject &settings,
                              const QString &outputDir,
                              SparsePointOperationResult *result,
                              QString *errorMessage)
{
    if (!result)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少输出对象");
        }
        return false;
    }

    QJsonObject sourceRoot;
    std::vector<xjw::SparsePointCloudPoint> points;
    if (!loadSparsePointSource(context, settings, &sourceRoot, &points, errorMessage))
    {
        return false;
    }
    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("点级 sidecar 中没有可处理的稀疏点");
        }
        return false;
    }

    const int inputCount = static_cast<int>(points.size());
    const bool qualityMetricsAvailable =
        sourceRoot.value(QStringLiteral("quality_metrics_available")).toBool(true);

    xjw::SparsePointCloudSpatialCleanupOptions options;
    options.voxelSize = settings.value(QStringLiteral("voxelSize")).toDouble(0.0);
    options.minVoxelPoints = settings.value(QStringLiteral("minVoxelPoints")).toInt(2);
    options.localReprojFilter = qualityMetricsAvailable &&
                                settings.value(QStringLiteral("localReprojFilter")).toBool(true);
    options.localReprojStdMul = settings.value(QStringLiteral("localReprojStdMul")).toDouble(2.5);
    options.deduplicationRadius = settings.value(QStringLiteral("deduplicationRadius")).toDouble(-1.0);

    const xjw::SparsePointCloudSpatialCleanupResult optResult =
        xjw::SparsePointCloudProcessor::spatialCleanup(&points, options);

    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("空间清理后没有剩余稀疏点，请降低过滤强度后重试");
        }
        return false;
    }

    QDir().mkpath(outputDir);
    const QString sparseCloudPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_spatial_cleanup.ply"));
    const QString sidecarPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
    if (!writeSparsePointCloudPly(sparseCloudPath, points, errorMessage))
    {
        return false;
    }

    QJsonObject summary;
    summary[QStringLiteral("input_points")] = optResult.inputPoints;
    summary[QStringLiteral("output_points")] = optResult.outputPoints;
    summary[QStringLiteral("removed_by_voxel_isolation")] = optResult.removedByVoxelIsolation;
    summary[QStringLiteral("removed_by_local_reproj")] = optResult.removedByLocalReproj;
    summary[QStringLiteral("removed_by_deduplication")] = optResult.removedByDeduplication;
    summary[QStringLiteral("removed_total")] =
        optResult.removedByVoxelIsolation + optResult.removedByLocalReproj + optResult.removedByDeduplication;

    const QJsonObject sidecar = makeSparsePointSidecar(sourceRoot,
                                                       points,
                                                       QStringLiteral("spatial_cleanup"),
                                                       settings,
                                                       summary,
                                                       context.selectedImages,
                                                       context.sidecarPath);
    if (!writeJsonObjectFile(sidecarPath, sidecar, errorMessage))
    {
        return false;
    }

    result->outputDir = outputDir;
    result->sparseCloudPath = sparseCloudPath;
    result->sidecarPath = sidecarPath;
    result->inputCount = inputCount;
    result->outputCount = static_cast<int>(points.size());
    QJsonObject files;
    files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
    result->extraRecord[QStringLiteral("files")] = files;
    result->extraRecord[QStringLiteral("source")] = QStringLiteral("sparse_spatial_cleanup");
    result->extraRecord[QStringLiteral("operation")] = QStringLiteral("spatial_cleanup");
    result->extraRecord[QStringLiteral("source_result_index")] = context.sourceResultIndex;
    if (sourceRoot.value(QStringLiteral("source_kind")).toString() == QLatin1String("external_ply"))
    {
        result->extraRecord[QStringLiteral("source_ply")] =
            sourceRoot.value(QStringLiteral("source_ply")).toString();
    }
    result->extraRecord[QStringLiteral("operation_settings")] = settings;
    result->extraRecord[QStringLiteral("operation_summary")] = summary;
    const QJsonObject quality = sidecar.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        result->extraRecord =
            xjw::common::project::mergeSparseQualityIntoRecord(result->extraRecord, quality);
    }
    return true;
}

bool runSparsePointRefine(const SparsePointContext &context,
                          const QJsonObject &settings,
                          const QString &outputDir,
                          SparsePointOperationResult *result,
                          QString *errorMessage)
{
    if (!result)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少输出对象");
        }
        return false;
    }

    QJsonObject sourceRoot;
    std::vector<xjw::SparsePointCloudPoint> basePoints;
    if (!loadSparsePointSource(context, settings, &sourceRoot, &basePoints, errorMessage))
    {
        return false;
    }
    if (basePoints.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("点级 sidecar 中没有可精修的稀疏点");
        }
        return false;
    }
    const bool qualityMetricsAvailable =
        sourceRoot.value(QStringLiteral("quality_metrics_available")).toBool(true);

    xjw::SparsePointCloudOptimizeOptions options;
    options.filterOptions.filterByReprojError =
        qualityMetricsAvailable && settings.value(QStringLiteral("filterByReprojError")).toBool(true);
    options.filterOptions.maxReprojError = settings.value(QStringLiteral("maxReprojError")).toDouble(4.0);
    options.filterOptions.filterByTrackLen =
        qualityMetricsAvailable && settings.value(QStringLiteral("filterByTrackLen")).toBool(true);
    options.filterOptions.minTrackLen = settings.value(QStringLiteral("minTrackLen")).toInt(3);
    options.filterOptions.filterByTriAngle =
        qualityMetricsAvailable && settings.value(QStringLiteral("filterByTriAngle")).toBool(true);
    options.filterOptions.minTriAngleDeg = settings.value(QStringLiteral("minTriAngleDeg")).toDouble(
        settings.value(QStringLiteral("minAngle")).toDouble(2.0));
    options.filterOptions.filterByStatistical = settings.value(QStringLiteral("filterByStatistical")).toBool(true);
    options.filterOptions.statK = settings.value(QStringLiteral("statK")).toInt(
        settings.value(QStringLiteral("knnNeighbors")).toInt(20));
    options.filterOptions.statStdDevMul = settings.value(QStringLiteral("statStdDevMul")).toDouble(
        settings.value(QStringLiteral("stdDevMultiplier")).toDouble(2.0));
    options.filterOptions.filterByDensity = settings.value(QStringLiteral("filterByDensity")).toBool(false);
    options.filterOptions.densityRadius = settings.value(QStringLiteral("densityRadius")).toDouble(0.5);
    options.filterOptions.densityMinNeighbors = settings.value(QStringLiteral("densityMinNeighbors")).toInt(5);
    options.filterOptions.filterByNormalConsistency = settings.value(QStringLiteral("normalConsistency")).toBool(false);
    options.iterative = true;
    options.iterRounds = settings.value(QStringLiteral("iterRounds")).toInt(3);
    options.restartFromInputEachRound = settings.value(QStringLiteral("retriangulate")).toBool(false);
    options.tightenThresholds = false;
    options.enableSpatialCleanup = settings.value(QStringLiteral("enableSpatialCleanup")).toBool(false);
    options.spatialCleanupOptions.voxelSize = settings.value(QStringLiteral("voxelSize")).toDouble(0.0);
    options.spatialCleanupOptions.minVoxelPoints = settings.value(QStringLiteral("minVoxelPoints")).toInt(2);
    options.spatialCleanupOptions.localReprojFilter =
        qualityMetricsAvailable && settings.value(QStringLiteral("localReprojFilter")).toBool(true);
    options.spatialCleanupOptions.localReprojStdMul = settings.value(QStringLiteral("localReprojStdMul")).toDouble(2.5);
    options.spatialCleanupOptions.deduplicationRadius = settings.value(QStringLiteral("deduplicationRadius")).toDouble(-1.0);

    const xjw::SparsePointCloudOptimizeResult refineResult =
        xjw::SparsePointCloudProcessor::optimize(basePoints, options);

    if (refineResult.points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("精修后没有剩余稀疏点，请降低过滤强度后重试");
        }
        return false;
    }

    QDir().mkpath(outputDir);
    const QString sparseCloudPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_refined.ply"));
    const QString sidecarPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
    if (!writeSparsePointCloudPly(sparseCloudPath, refineResult.points, errorMessage))
    {
        return false;
    }

    QJsonObject summary;
    QJsonArray roundSummaries;
    for (const xjw::SparsePointCloudOptimizeRound &round : refineResult.rounds)
    {
        QJsonObject roundSummary;
        roundSummary[QStringLiteral("round")] = round.round;
        roundSummary[QStringLiteral("input_points")] = round.inputPoints;
        roundSummary[QStringLiteral("max_reproj_error")] = round.filterOptions.maxReprojError;
        roundSummary[QStringLiteral("min_tri_angle_deg")] = round.filterOptions.minTriAngleDeg;
        roundSummary[QStringLiteral("min_track_len")] = round.filterOptions.minTrackLen;
        roundSummary[QStringLiteral("stat_stddev_mul")] = round.filterOptions.statStdDevMul;
        roundSummary[QStringLiteral("removed_by_reproj")] = round.filterStats.removedByReprojError;
        roundSummary[QStringLiteral("removed_by_track_len")] = round.filterStats.removedByTrackLen;
        roundSummary[QStringLiteral("removed_by_tri_angle")] = round.filterStats.removedByTriAngle;
        roundSummary[QStringLiteral("removed_by_statistical")] = round.filterStats.removedByStatistical;
        roundSummary[QStringLiteral("removed_by_normal_consistency")] = round.filterStats.removedByNormalConsistency;
        roundSummary[QStringLiteral("removed_by_density")] = round.filterStats.removedByDensity;
        roundSummary[QStringLiteral("output_points")] = round.outputPoints;
        roundSummaries.append(roundSummary);
    }
    summary[QStringLiteral("input_points")] = refineResult.inputPoints;
    summary[QStringLiteral("output_points")] = refineResult.outputPoints;
    summary[QStringLiteral("removed_total")] = refineResult.removedTotal;
    summary[QStringLiteral("iter_rounds")] = options.iterRounds;
    summary[QStringLiteral("retriangulate")] = options.restartFromInputEachRound;
    summary[QStringLiteral("normal_consistency")] = options.filterOptions.filterByNormalConsistency;
    summary[QStringLiteral("spatial_cleanup")] = options.enableSpatialCleanup;
    summary[QStringLiteral("rounds")] = roundSummaries;

    const QJsonObject sidecar = makeSparsePointSidecar(sourceRoot,
                                                       refineResult.points,
                                                       QStringLiteral("sparse_refine"),
                                                       settings,
                                                       summary,
                                                       context.selectedImages,
                                                       context.sidecarPath);
    if (!writeJsonObjectFile(sidecarPath, sidecar, errorMessage))
    {
        return false;
    }

    result->outputDir = outputDir;
    result->sparseCloudPath = sparseCloudPath;
    result->sidecarPath = sidecarPath;
    result->inputCount = refineResult.inputPoints;
    result->outputCount = refineResult.outputPoints;
    QJsonObject files;
    files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
    result->extraRecord[QStringLiteral("files")] = files;
    result->extraRecord[QStringLiteral("source")] = QStringLiteral("sparse_cloud_refine");
    result->extraRecord[QStringLiteral("operation")] = QStringLiteral("sparse_refine");
    result->extraRecord[QStringLiteral("source_result_index")] = context.sourceResultIndex;
    if (sourceRoot.value(QStringLiteral("source_kind")).toString() == QLatin1String("external_ply"))
    {
        result->extraRecord[QStringLiteral("source_ply")] =
            sourceRoot.value(QStringLiteral("source_ply")).toString();
    }
    result->extraRecord[QStringLiteral("operation_settings")] = settings;
    result->extraRecord[QStringLiteral("operation_summary")] = summary;
    const QJsonObject quality = sidecar.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        result->extraRecord =
            xjw::common::project::mergeSparseQualityIntoRecord(result->extraRecord, quality);
    }
    return true;
}

QJsonArray summarizeAtResults(const QJsonObject &meta)
{
    const QJsonArray atArray = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    QJsonArray summary;
    for (int i = 0; i < atArray.size(); ++i)
    {
        const QJsonObject at = atArray.at(i).toObject();
        const QJsonObject files = at.value(QStringLiteral("files")).toObject();
        const QJsonObject quality = at.value(QStringLiteral("quality")).toObject();
        int sparsePointCount = at.value(QStringLiteral("sparse_point_count")).toInt(0);
        if (sparsePointCount <= 0)
        {
            sparsePointCount = at.value(QStringLiteral("point_count")).toInt(
                quality.value(QStringLiteral("point_count")).toInt(0));
        }

        QJsonObject item;
        item[QStringLiteral("index")] = i;
        item[QStringLiteral("created_at")] = at.value(QStringLiteral("created_at")).toString();
        item[QStringLiteral("output_dir")] = at.value(QStringLiteral("output_dir")).toString();
        item[QStringLiteral("sparse_point_count")] = sparsePointCount;
        item[QStringLiteral("point_count")] = sparsePointCount;
        const QJsonArray selImgs = at.value(QStringLiteral("selected_images")).toArray();
        item[QStringLiteral("image_count")] = selImgs.size();
        item[QStringLiteral("image0")] = selImgs.isEmpty() ? QString() : selImgs.first().toString();
        item[QStringLiteral("image1")] = selImgs.size() > 1 ? selImgs.last().toString() : QString();
        item[QStringLiteral("sparse_cloud_xyz")] = files.value(QStringLiteral("sparse_cloud_xyz")).toString();
        item[QStringLiteral("sparse_cloud_points_json")] =
            files.value(QStringLiteral("sparse_cloud_points_json")).toString();
        item[QStringLiteral("operation")] = at.value(QStringLiteral("operation")).toString(QStringLiteral("triangulation"));
        item[QStringLiteral("operation_display_name")] = at.value(QStringLiteral("operation_display_name")).toString(
            sparseOperationDisplayName(item.value(QStringLiteral("operation")).toString()));
        item[QStringLiteral("source_result_index")] = at.value(QStringLiteral("source_result_index")).toInt(-1);
        item[QStringLiteral("is_latest")] = (i == atArray.size() - 1);
        if (!quality.isEmpty())
        {
            item[QStringLiteral("quality")] = quality;
            for (auto it = quality.begin(); it != quality.end(); ++it)
            {
                if (!item.contains(it.key()))
                {
                    item[it.key()] = it.value();
                }
            }
        }
        if (at.contains(QStringLiteral("operation_summary")))
        {
            item[QStringLiteral("operation_summary")] = at.value(QStringLiteral("operation_summary"));
        }

        const QString opLabel = sparseOperationDisplayName(item.value(QStringLiteral("operation")).toString());
        const QString dirName = QFileInfo(item.value(QStringLiteral("output_dir")).toString()).fileName();
        QString displayName = QStringLiteral("#%1 %2").arg(i).arg(opLabel);
        if (sparsePointCount > 0)
        {
            displayName += QStringLiteral("  [%1 点]").arg(sparsePointCount);
        }
        if (!dirName.isEmpty())
        {
            displayName += QStringLiteral("  (%1)").arg(dirName);
        }
        if (item.value(QStringLiteral("source_result_index")).toInt(-1) >= 0)
        {
            displayName += QStringLiteral("  [源 #%1]").arg(item.value(QStringLiteral("source_result_index")).toInt());
        }
        if (item.value(QStringLiteral("is_latest")).toBool())
        {
            displayName += QStringLiteral("  [当前]");
        }
        item[QStringLiteral("display_name")] = displayName;
        summary.append(item);
    }
    return summary;
}

} // namespace xjw::core::project
