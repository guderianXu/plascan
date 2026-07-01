// =============================================================================
// 文件: cli_reconstruct_pipeline.cpp
// 功能: PlaScan 一键重建 CLI
//       默认: .lis(image camera) -> SFM -> MVS -> 三维模型 -> DEM/DOM
//       PLASCAN_THREE_D_ONLY: GUI 三维重建等价流程，仅 SFM -> MVS -> 三维模型
// =============================================================================
#include "cli_common.h"

#include "Camera.h"
#include "DenseCloudQualityFilter.h"
#include "DepthFrameUtils.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "Logger.h"
#include "ModelWorkflowService.h"
#include "ProjectDenseWorkflowConfig.h"
#include "SFMService.h"
#include "SparseCloudPreprocessor.h"
#ifndef PLASCAN_THREE_D_ONLY
#include "TerrainPipeline.h"
#endif
#include "project/ProjectCommonUtils.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>
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
#include <QSaveFile>
#include <QTextStream>
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
#include <iterator>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace
{

int registerCliConsoleLogger()
{
    static int sinkId = 0;
    if (sinkId != 0)
    {
        return sinkId;
    }

    sinkId = Logger::instance()->registerSink([](const Logger::Entry &entry) {
        FILE *stream = entry.level >= Logger::Warn ? stderr : stdout;
        std::fwrite(entry.formatted.data(), 1, entry.formatted.size(), stream);
        std::fflush(stream);
    });
    return sinkId;
}

struct InputItem
{
    QString imagePath;
    QString cameraPath;
    xjw::Camera camera;
};

QString cleanAbsolutePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString resolveListToken(const QString &token, const QDir &baseDir)
{
    QString trimmed = token.trimmed();
    if (trimmed.startsWith(QStringLiteral("~/")))
    {
        trimmed = QDir::home().filePath(trimmed.mid(2));
    }

    const QFileInfo info(trimmed);
    if (info.isAbsolute())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(QFileInfo(baseDir.filePath(trimmed)).absoluteFilePath());
}

bool hasUnquotedComma(const QString &line)
{
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (const QChar ch : line)
    {
        if (escaped)
        {
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                inQuote = false;
            }
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            continue;
        }
        if (ch == QLatin1Char(','))
        {
            return true;
        }
    }

    return false;
}

bool appendParsedToken(QStringList *parts, QString *token, bool *hasToken)
{
    if (!parts || !token || !hasToken)
    {
        return false;
    }
    if (*hasToken || !token->isEmpty())
    {
        parts->append(token->trimmed());
        token->clear();
        *hasToken = false;
    }
    return true;
}

bool parseShellTokens(const QString &line, QStringList *parts, QString *error)
{
    if (!parts)
    {
        if (error) *error = QStringLiteral("内部错误：列表行输出对象为空");
        return false;
    }

    parts->clear();
    QString token;
    bool hasToken = false;
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (const QChar ch : line)
    {
        if (escaped)
        {
            token.append(ch);
            hasToken = true;
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            hasToken = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                inQuote = false;
            }
            else
            {
                token.append(ch);
            }
            hasToken = true;
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            hasToken = true;
            continue;
        }
        if (ch.isSpace())
        {
            appendParsedToken(parts, &token, &hasToken);
            continue;
        }

        token.append(ch);
        hasToken = true;
    }

    if (escaped)
    {
        if (error) *error = QStringLiteral("行尾转义字符缺少目标字符");
        return false;
    }
    if (inQuote)
    {
        if (error) *error = QStringLiteral("引号未闭合");
        return false;
    }

    appendParsedToken(parts, &token, &hasToken);
    return true;
}

bool parseCsvTokens(const QString &line, QStringList *parts, QString *error)
{
    if (!parts)
    {
        if (error) *error = QStringLiteral("内部错误：列表行输出对象为空");
        return false;
    }

    parts->clear();
    QString token;
    bool hasToken = false;
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (int index = 0; index < line.size(); ++index)
    {
        const QChar ch = line.at(index);
        if (escaped)
        {
            token.append(ch);
            hasToken = true;
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            hasToken = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                if (quoteChar == QLatin1Char('"')
                    && index + 1 < line.size()
                    && line.at(index + 1) == QLatin1Char('"'))
                {
                    token.append(ch);
                    hasToken = true;
                    ++index;
                }
                else
                {
                    inQuote = false;
                    hasToken = true;
                }
            }
            else
            {
                token.append(ch);
                hasToken = true;
            }
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            hasToken = true;
            continue;
        }
        if (ch == QLatin1Char(','))
        {
            parts->append(token.trimmed());
            token.clear();
            hasToken = false;
            continue;
        }

        token.append(ch);
        hasToken = true;
    }

    if (escaped)
    {
        if (error) *error = QStringLiteral("行尾转义字符缺少目标字符");
        return false;
    }
    if (inQuote)
    {
        if (error) *error = QStringLiteral("引号未闭合");
        return false;
    }

    if (hasToken || !token.isEmpty() || line.endsWith(QLatin1Char(',')))
    {
        parts->append(token.trimmed());
    }
    return true;
}

bool parseListLine(const QString &line, QStringList *parts, QString *error)
{
    if (hasUnquotedComma(line))
    {
        return parseCsvTokens(line, parts, error);
    }
    return parseShellTokens(line, parts, error);
}

QStringList criticalOutputPaths(const QString &outputDir)
{
    const QDir dir(outputDir);
    QStringList paths = {
        dir.filePath(QStringLiteral("report.json")),
        dir.filePath(QStringLiteral("headless.plascan")),
        dir.filePath(QStringLiteral("sparse")),
        dir.filePath(QStringLiteral("mvs/dense_cloud.ply")),
        dir.filePath(QStringLiteral("model"))
    };
#ifndef PLASCAN_THREE_D_ONLY
    paths << dir.filePath(QStringLiteral("terrain/products/dem.tif"))
          << dir.filePath(QStringLiteral("terrain/products/dom.png"));
#endif
    return paths;
}

bool validateOutputDirectory(const QString &outputDir, bool force, QString *error)
{
    const QFileInfo outputInfo(outputDir);
    if (outputInfo.exists() && !outputInfo.isDir())
    {
        if (error) *error = QStringLiteral("输出路径已存在但不是目录: %1").arg(outputDir);
        return false;
    }

    if (force)
    {
        return true;
    }

    if (outputInfo.exists())
    {
        const QDir dir(outputDir);
        const QFileInfoList entries = dir.entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries);
        if (!entries.isEmpty())
        {
            if (error) *error = QStringLiteral("输出目录非空，拒绝覆盖已有结果: %1；如需复用/覆盖请添加 --force").arg(outputDir);
            return false;
        }
    }

    for (const QString &path : criticalOutputPaths(outputDir))
    {
        if (QFileInfo::exists(path))
        {
            if (error) *error = QStringLiteral("输出目录已有关键输出文件，拒绝覆盖: %1；如需复用/覆盖请添加 --force").arg(path);
            return false;
        }
    }

    return true;
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto distortion = camera.distortion();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject object;
    object[QStringLiteral("model")] = QStringLiteral("tsai");
    object[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    object[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    object[QStringLiteral("pitch")] = camera.pixelPitch();
    object[QStringLiteral("fu")] = camera.focalXMillimeters();
    object[QStringLiteral("fv")] = camera.focalYMillimeters();
    object[QStringLiteral("cu")] = camera.principalXMillimeters();
    object[QStringLiteral("cv")] = camera.principalYMillimeters();
    object[QStringLiteral("k1")] = distortion.radialK1;
    object[QStringLiteral("k2")] = distortion.radialK2;
    object[QStringLiteral("k3")] = distortion.radialK3;
    object[QStringLiteral("p1")] = distortion.tangentialP1;
    object[QStringLiteral("p2")] = distortion.tangentialP2;
    object[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    object[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    object[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray centerArray;
    for (const double value : center)
    {
        centerArray.append(value);
    }
    object[QStringLiteral("C")] = centerArray;

    QJsonArray rotationArray;
    for (const double value : rotation)
    {
        rotationArray.append(value);
    }
    object[QStringLiteral("R")] = rotationArray;
    return object;
}

bool readImageCameraList(const QString &listPath,
                         std::vector<InputItem> *items,
                         QJsonObject *projectMeta,
                         QString *error)
{
    if (!items || !projectMeta)
    {
        if (error) *error = QStringLiteral("内部错误：列表输出对象为空");
        return false;
    }

    QFile file(listPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (error) *error = QStringLiteral("无法打开列表文件: %1").arg(listPath);
        return false;
    }

    items->clear();
    QJsonArray imageArray;
    const QDir listDir(QFileInfo(listPath).absolutePath());
    QTextStream stream(&file);
    int lineNumber = 0;
    while (!stream.atEnd())
    {
        ++lineNumber;
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
        {
            continue;
        }

        QStringList parts;
        QString parseError;
        if (!parseListLine(line, &parts, &parseError))
        {
            if (error)
            {
                *error = QStringLiteral("%1:%2 %3").arg(listPath).arg(lineNumber).arg(parseError);
            }
            return false;
        }

        if (parts.size() != 2)
        {
            if (error)
            {
                *error = QStringLiteral("%1:%2 需要 '<image> <camera.tsai>'")
                             .arg(listPath)
                             .arg(lineNumber);
            }
            return false;
        }

        InputItem item;
        item.imagePath = resolveListToken(parts.at(0), listDir);
        item.cameraPath = resolveListToken(parts.at(1), listDir);
        if (!QFileInfo::exists(item.imagePath))
        {
            if (error) *error = QStringLiteral("%1:%2 影像不存在: %3").arg(listPath).arg(lineNumber).arg(item.imagePath);
            return false;
        }
        if (!item.camera.loadFromFile(item.cameraPath.toStdString()) || !item.camera.isValid())
        {
            if (error) *error = QStringLiteral("%1:%2 相机读取失败: %3").arg(listPath).arg(lineNumber).arg(item.cameraPath);
            return false;
        }

        QJsonObject imageObject;
        imageObject[QStringLiteral("path")] = item.imagePath;
        imageObject[QStringLiteral("name")] = QFileInfo(item.imagePath).fileName();
        imageObject[QStringLiteral("camera")] = cameraToJson(item.camera);
        imageArray.append(imageObject);
        items->push_back(std::move(item));
    }

    if (items->size() < 2)
    {
        if (error) *error = QStringLiteral("至少需要 2 组 image/camera 输入");
        return false;
    }

    (*projectMeta)[QStringLiteral("images")] = imageArray;
    return true;
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

struct FusedVoxelKey
{
    std::int64_t ix = 0;
    std::int64_t iy = 0;
    std::int64_t iz = 0;

    bool operator==(const FusedVoxelKey &other) const
    {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

struct FusedVoxelKeyHash
{
    std::size_t operator()(const FusedVoxelKey &key) const
    {
        std::size_t seed = 0;
        const auto mix = [&seed](std::int64_t value) {
            const auto hashed = std::hash<std::int64_t>{}(value);
            seed ^= hashed + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        };
        mix(key.ix);
        mix(key.iy);
        mix(key.iz);
        return seed;
    }
};

struct FusedVoxelAccumulator
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    std::uint32_t count = 0;
};

std::vector<xjw::mvs::FusedPoint> voxelDownsampleFusedPoints(
    const std::vector<xjw::mvs::FusedPoint> &cloud,
    float leafSize)
{
    if (cloud.empty() || leafSize <= 0.0f)
    {
        return cloud;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    for (const auto &point : cloud)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        minZ = std::min(minZ, point.z);
    }
    if (minX == std::numeric_limits<float>::max())
    {
        return {};
    }

    std::unordered_map<FusedVoxelKey, FusedVoxelAccumulator, FusedVoxelKeyHash> voxels;
    voxels.reserve(std::min<std::size_t>(cloud.size(), 1000000));
    const double invLeaf = 1.0 / static_cast<double>(leafSize);
    for (const auto &point : cloud)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }

        const FusedVoxelKey key{
            static_cast<std::int64_t>(std::floor((static_cast<double>(point.x) - minX) * invLeaf)),
            static_cast<std::int64_t>(std::floor((static_cast<double>(point.y) - minY) * invLeaf)),
            static_cast<std::int64_t>(std::floor((static_cast<double>(point.z) - minZ) * invLeaf))
        };

        auto &acc = voxels[key];
        acc.x += point.x;
        acc.y += point.y;
        acc.z += point.z;
        if (std::isfinite(point.nx) && std::isfinite(point.ny) && std::isfinite(point.nz))
        {
            acc.nx += point.nx;
            acc.ny += point.ny;
            acc.nz += point.nz;
        }
        acc.r += point.r;
        acc.g += point.g;
        acc.b += point.b;
        ++acc.count;
    }

    std::vector<xjw::mvs::FusedPoint> output;
    output.reserve(voxels.size());
    for (const auto &[key, acc] : voxels)
    {
        Q_UNUSED(key);
        if (acc.count == 0)
        {
            continue;
        }
        const double invCount = 1.0 / static_cast<double>(acc.count);
        xjw::mvs::FusedPoint point;
        point.x = static_cast<float>(acc.x * invCount);
        point.y = static_cast<float>(acc.y * invCount);
        point.z = static_cast<float>(acc.z * invCount);

        const double normalLength = std::sqrt(acc.nx * acc.nx + acc.ny * acc.ny + acc.nz * acc.nz);
        if (std::isfinite(normalLength) && normalLength > 1e-12)
        {
            point.nx = static_cast<float>(acc.nx / normalLength);
            point.ny = static_cast<float>(acc.ny / normalLength);
            point.nz = static_cast<float>(acc.nz / normalLength);
        }
        else
        {
            point.nz = 1.0f;
        }

        const auto toByte = [invCount](double value) {
            return static_cast<std::uint8_t>(
                std::clamp(static_cast<int>(std::lround(value * invCount)), 0, 255));
        };
        point.r = toByte(acc.r);
        point.g = toByte(acc.g);
        point.b = toByte(acc.b);
        output.push_back(point);
    }

    return output;
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

bool writePointCloudPly(const QString &path,
                        const PlaCloud &pointCloud,
                        bool writeNormals,
                        QString *error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    try
    {
        if (writeNormals || !pointCloud.hasNormals())
        {
            plapoint::io::writePly(path.toStdString(), pointCloud, plapoint::io::PlyFormat::BinaryLE);
        }
        else
        {
            const PlaCloud withoutNormals = cloneCloudValue(pointCloud, false);
            plapoint::io::writePly(path.toStdString(), withoutNormals, plapoint::io::PlyFormat::BinaryLE);
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (error) *error = QString::fromStdString(e.what());
        return false;
    }
}

xjw::mvs::TerrainHeightSpikeFilterOptions terrainSpikeOptionsFromRequest(
    const xjw::gui::project::DenseRefineSettings &request)
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
                          const xjw::gui::project::DenseRefineSettings &request,
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

    frame->cameraModel = views[static_cast<std::size_t>(frameIndex)].positiveDepthModel();
    frame->imgW = depth.cols;
    frame->imgH = depth.rows;
    frame->imagePath = views[static_cast<std::size_t>(frameIndex)].imagePath;
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

bool loadFusionFramesFromDepthMaps(const QString &mvsDir,
                                   const std::vector<xjw::mvs::CameraView> &views,
                                   const xjw::mvs::FusionConfig &fusionConfig,
                                   int fusionMaxImageDim,
                                   std::vector<xjw::mvs::FusionFrameInput> *frames,
                                   QString *error)
{
    if (!frames)
    {
        if (error) *error = QStringLiteral("内部错误：融合帧输出为空");
        return false;
    }

    frames->clear();
    frames->reserve(views.size());

    for (std::size_t index = 0; index < views.size(); ++index)
    {
        xjw::mvs::FusionFrameInput frame;
        if (!loadFusionFrameFromDepthMap(mvsDir,
                                         views,
                                         fusionConfig,
                                         static_cast<int>(index),
                                         fusionMaxImageDim,
                                         &frame,
                                         error))
        {
            return false;
        }
        frames->push_back(std::move(frame));
    }

    return true;
}

std::vector<int> streamingFusionWindowIndices(int refIndex, int frameCount, int neighborCount)
{
    std::vector<int> indices;
    if (refIndex < 0 || refIndex >= frameCount || frameCount <= 0)
    {
        return indices;
    }

    indices.push_back(refIndex);
    const int maxNeighbors = std::min(std::max(1, neighborCount), frameCount - 1);
    for (int offset = 1; static_cast<int>(indices.size()) < maxNeighbors + 1; ++offset)
    {
        const int left = refIndex - offset;
        if (left >= 0)
        {
            indices.push_back(left);
            if (static_cast<int>(indices.size()) >= maxNeighbors + 1)
            {
                break;
            }
        }

        const int right = refIndex + offset;
        if (right < frameCount)
        {
            indices.push_back(right);
        }

        if (left < 0 && right >= frameCount)
        {
            break;
        }
    }
    return indices;
}

bool fuseDepthMapsStreamingFromDisk(const QString &mvsDir,
                                    const std::vector<xjw::mvs::CameraView> &views,
                                    const xjw::gui::project::DenseGenerationSettings &denseSettings,
                                    const xjw::mvs::DepthGenConfig &depthConfig,
                                    std::vector<xjw::mvs::FusedPoint> *fusedCloud,
                                    std::vector<xjw::mvs::DepthPostProcessStats> *depthPostprocessStats,
                                    QString *error)
{
    if (!fusedCloud || !depthPostprocessStats)
    {
        if (error) *error = QStringLiteral("内部错误：MVS 流式融合输出为空");
        return false;
    }

    fusedCloud->clear();
    depthPostprocessStats->clear();
    const int frameCount = static_cast<int>(views.size());
    if (frameCount < 2)
    {
        if (error) *error = QStringLiteral("MVS 深度图融合至少需要 2 帧");
        return false;
    }

    constexpr std::size_t kStreamingFusionPreVoxelThreshold = 2000000;
    constexpr std::size_t kStreamingFusionTargetPoints = 1000000;
    constexpr int kStreamingFusionCacheFrameLimit = 32;
    const int neighborCount = std::min(frameCount - 1, std::clamp(std::max(4, denseSettings.minViews * 3), 2, 16));
    fusedCloud->reserve(std::min<std::size_t>(kStreamingFusionTargetPoints, 200000));

    std::fprintf(stdout,
                 "  [MVS  70%%] 流式深度图融合: frames=%d neighborWindow=%d\n",
                 frameCount,
                 neighborCount);
    std::fflush(stdout);

    const bool useCachedFrames = frameCount <= kStreamingFusionCacheFrameLimit;
    std::vector<xjw::mvs::FusionFrameInput> cachedFrames;
    if (useCachedFrames)
    {
        std::fprintf(stdout,
                     "  [MVS  70%%] 小规模融合缓存: frames=%d limit=%d\n",
                     frameCount,
                     kStreamingFusionCacheFrameLimit);
        std::fflush(stdout);
        if (!loadFusionFramesFromDepthMaps(mvsDir,
                                           views,
                                           depthConfig.fusion,
                                           denseSettings.fusionMaxImageDim,
                                           &cachedFrames,
                                           error))
        {
            return false;
        }
        depthPostprocessStats->reserve(cachedFrames.size());
        for (const auto &frame : cachedFrames)
        {
            depthPostprocessStats->push_back(frame.depthPostprocess);
        }
    }

    for (int refIndex = 0; refIndex < frameCount; ++refIndex)
    {
        const auto windowStart = std::chrono::steady_clock::now();
        const std::vector<int> windowIndices = streamingFusionWindowIndices(refIndex, frameCount, neighborCount);
        std::vector<xjw::mvs::FusionFrameInput> frames;
        frames.reserve(windowIndices.size());
        for (const int frameIndex : windowIndices)
        {
            if (useCachedFrames)
            {
                frames.push_back(cachedFrames[static_cast<std::size_t>(frameIndex)]);
                continue;
            }

            xjw::mvs::FusionFrameInput frame;
            if (!loadFusionFrameFromDepthMap(mvsDir,
                                             views,
                                             depthConfig.fusion,
                                             frameIndex,
                                             denseSettings.fusionMaxImageDim,
                                             &frame,
                                             error))
            {
                return false;
            }
            frames.push_back(std::move(frame));
        }

        if (frames.size() < 2)
        {
            continue;
        }

        if (!useCachedFrames)
        {
            depthPostprocessStats->push_back(frames.front().depthPostprocess);
        }

        xjw::mvs::StereoFusionConfig fusionCfg;
        fusionCfg.minNumPixels = std::max(1, denseSettings.minConsistentViews);
        fusionCfg.maxReprojError = denseSettings.depthConsistency;
        fusionCfg.maxDepthError = 0.05f;
        fusionCfg.checkNumImages = std::min(neighborCount, static_cast<int>(frames.size()) - 1);
        fusionCfg.workerCount = std::max(1, denseSettings.threads);
        fusionCfg.useColor = true;
        fusionCfg.colorCacheCapacity = 2;
        fusionCfg.fuseOnlyFirstFrame = true;
        if (frames.size() <= 2)
        {
            fusionCfg.minNumPixels = 1;
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.08f);
            fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f);
        }

        xjw::mvs::DepthMapFusion fusion(fusionCfg);
        std::vector<xjw::mvs::FusedPoint> batchPoints;
        std::string fuseErr;
        const bool ok = fusion.fuse(frames,
                                    batchPoints,
                                    [refIndex, frameCount](const std::string &stage, float ratio) {
                                        std::fprintf(stdout,
                                                     "  [MVS %3d%%] 流式深度图融合 %d/%d: %s\n",
                                                     70 + static_cast<int>(
                                                              ((static_cast<float>(refIndex) + ratio) /
                                                               static_cast<float>(std::max(1, frameCount))) * 20.0f),
                                                     refIndex + 1,
                                                     frameCount,
                                                     stage.c_str());
                                        std::fflush(stdout);
                                    },
                                    &fuseErr);
        if (!ok)
        {
            if (error) *error = QString::fromStdString(fuseErr);
            return false;
        }

        fusedCloud->insert(fusedCloud->end(),
                           std::make_move_iterator(batchPoints.begin()),
                           std::make_move_iterator(batchPoints.end()));

        if (fusedCloud->size() > kStreamingFusionPreVoxelThreshold)
        {
            const float leaf = adaptivePreSorVoxelSize(*fusedCloud, 0.005f);
            const auto before = fusedCloud->size();
            FusedVoxelDownsampleResult downsample =
                voxelDownsampleFusedPointsToTarget(*fusedCloud, leaf, kStreamingFusionTargetPoints);
            if (!downsample.points.empty())
            {
                *fusedCloud = std::move(downsample.points);
                std::fprintf(stdout,
                             "  [MVS  90%%] 流式融合预聚合 leaf=%.6f passes=%d points=%zu->%zu\n",
                             downsample.leafSize,
                             downsample.passes,
                             before,
                             fusedCloud->size());
                std::fflush(stdout);
            }
        }

        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - windowStart).count();
        std::fprintf(stdout,
                     "  [MVS %3d%%] 流式深度图融合批次 %d/%d window=%zu points=%zu elapsed=%.1f ms\n",
                     70 + ((refIndex + 1) * 20) / std::max(1, frameCount),
                     refIndex + 1,
                     frameCount,
                     frames.size(),
                     fusedCloud->size(),
                     elapsedMs);
        std::fflush(stdout);
    }

    if (fusedCloud->empty())
    {
        if (error) *error = QStringLiteral("MVS 流式融合没有生成有效稠密点");
        return false;
    }
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

QJsonObject mvsSettingsToJson(const xjw::gui::project::DenseGenerationSettings &denseSettings,
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

bool writeReport(const QString &outputDir, const QJsonObject &report, QJsonObject *writtenReport, QString *error)
{
    if (!QDir().mkpath(outputDir))
    {
        if (error) *error = QStringLiteral("无法创建报告目录: %1").arg(outputDir);
        return false;
    }

    const QString reportPath = QDir(outputDir).filePath(QStringLiteral("report.json"));
    QJsonObject out = report;
    out[QStringLiteral("report_json")] = reportPath;
    const QByteArray payload = QJsonDocument(out).toJson(QJsonDocument::Indented);

    QSaveFile file(reportPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (error) *error = QStringLiteral("无法打开报告文件: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }
    if (file.write(payload) != payload.size())
    {
        if (error) *error = QStringLiteral("报告写入失败: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }
    if (!file.commit())
    {
        if (error) *error = QStringLiteral("报告提交失败: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }

    if (writtenReport)
    {
        *writtenReport = out;
    }
    return true;
}

QJsonArray inputsToJson(const std::vector<InputItem> &items)
{
    QJsonArray array;
    for (const InputItem &item : items)
    {
        array.append(QJsonObject{
            {QStringLiteral("image"), item.imagePath},
            {QStringLiteral("camera"), item.cameraPath}
        });
    }
    return array;
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

int main(int argc, char *argv[])
{
    QCoreApplication qtApp(argc, argv);
    (void)registerCliConsoleLogger();

#ifdef PLASCAN_THREE_D_ONLY
    CLI::App app{"PlaScan GUI-equivalent 3D reconstruction pipeline"};
#else
    CLI::App app{"PlaScan GUI-equivalent reconstruction pipeline"};
#endif
    std::string listPathArg;
#ifdef PLASCAN_THREE_D_ONLY
    std::string outputDirArg = "three_d_reconstruction_output";
#else
    std::string outputDirArg = "full_pipeline_output";
#endif
    std::string device = "auto";
    std::string sfmFeatureAlgorithm = "disk";
    std::string sfmMatchAlgorithm = "lightglue";
    bool sfmGuidedRematching = false;
    int quality = 3;
    int threads = 8;
    int cudaParallelPairs = 1;
    int featureMaxImageDim = 0;
    double mvsResScale = 0.5;
    int mvsIterations = 6;
    double mvsConfidence = 0.20;
    double mvsFusionConfidence = 0.50;
    int mvsGpuFrameWorkers = 0;
    int mvsCpuFrameWorkers = 0;
    int mvsMaxFrames = 0;
    int mvsFusionMaxImageDim = 2048;
#ifndef PLASCAN_THREE_D_ONLY
    double demResolution = 0.0;
#endif
    int meshResolution = 224;
    bool skipModel = false;
    bool skipMesh = false;
    bool stopAfterSfm = false;
    bool skipMvs = false;
    bool mvsDepthOnly = false;
#ifdef PLASCAN_THREE_D_ONLY
    bool skipTerrain = true;
#else
    bool skipTerrain = false;
#endif
    bool exportObj = true;
    bool skipTexture = false;
    bool forceOutput = false;

    app.add_option("list_file", listPathArg, "image/camera .lis file")->required();
    app.add_option("-o,--output-dir", outputDirArg, "output directory");
    app.add_option("--device", device, "auto, cpu, cuda")->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    app.add_option("--sfm-feature-algorithm", sfmFeatureAlgorithm,
                   "SFM feature algorithm: disk, aliked, sift")
        ->check(CLI::IsMember({"disk", "aliked", "sift"}));
    app.add_option("--sfm-match-algorithm", sfmMatchAlgorithm,
                   "SFM match algorithm: lightglue, sift_flann, sift_bf_l2")
        ->check(CLI::IsMember({"lightglue", "sift_flann", "sift_bf_l2"}));
    app.add_flag("--sfm-guided-rematching",
                 sfmGuidedRematching,
                 "enable guided rematching after initial SfM");
    app.add_option("--quality", quality, "SFM quality level 0..3");
    app.add_option("--threads", threads, "CPU thread count");
    app.add_option("--cuda-parallel-pairs", cudaParallelPairs, "LightGlue CUDA parallel pair count");
    app.add_option("--feature-max-image-dim", featureMaxImageDim,
                   "deep feature max image side; 0 uses auto/adaptive quality preset, negative starts unbounded");
    app.add_option("--mvs-res-scale", mvsResScale,
                   "MVS depth resolution scale, e.g. 0.5 for half resolution");
    app.add_option("--mvs-iterations", mvsIterations, "MVS PatchMatch iterations");
    app.add_option("--mvs-confidence", mvsConfidence, "MVS PatchMatch confidence threshold");
    app.add_option("--mvs-fusion-confidence", mvsFusionConfidence, "MVS fusion confidence threshold");
    app.add_option("--mvs-gpu-frame-workers", mvsGpuFrameWorkers,
                   "MVS CUDA frame workers; 0 chooses automatically");
    app.add_option("--mvs-cpu-frame-workers", mvsCpuFrameWorkers,
                   "MVS CPU frame workers; 0 chooses automatically");
    app.add_option("--mvs-max-frames", mvsMaxFrames,
                   "limit MVS to first N registered frames for regression/debug runs; 0 uses all");
    app.add_option("--mvs-fusion-max-image-dim", mvsFusionMaxImageDim,
                   "max image side used during depth-map fusion; 0 keeps full resolution");
#ifndef PLASCAN_THREE_D_ONLY
    app.add_option("--dem-resolution", demResolution, "DEM/DOM resolution; 0 lets TerrainPipeline choose");
#endif
    app.add_option("--mesh-resolution", meshResolution, "mesh reconstruction grid resolution");
    app.add_flag("--stop-after-sfm", stopAfterSfm, "run SFM only, write report, then stop before MVS");
    app.add_flag("--skip-mvs", skipMvs, "skip MVS and downstream mesh/terrain stages after SFM");
    app.add_flag("--mvs-depth-only", mvsDepthOnly,
                 "run MVS depth-map estimation only, then skip fusion, mesh, and terrain");
    app.add_flag("--skip-mesh", skipMesh, "skip mesh reconstruction after MVS dense cloud generation");
#ifndef PLASCAN_THREE_D_ONLY
    app.add_flag("--skip-model", skipModel, "skip mesh reconstruction");
    app.add_flag("--skip-terrain", skipTerrain, "skip DEM/DOM generation");
#endif
    app.add_flag("--export-obj", exportObj, "also export OBJ/MTL/texture where supported");
    app.add_flag("--skip-texture", skipTexture, "skip OBJ/MTL texture export and keep only PLY mesh");
    app.add_flag("--force", forceOutput, "allow reusing or overwriting a non-empty output directory");

    CLI11_PARSE(app, argc, argv);
    if (skipTexture)
    {
        exportObj = false;
    }
    if (skipMesh)
    {
        skipModel = true;
    }
    mvsResScale = std::clamp(mvsResScale, 0.05, 1.0);
    mvsIterations = std::max(1, mvsIterations);
    mvsConfidence = std::clamp(mvsConfidence, 0.0, 1.0);
    mvsFusionConfidence = std::clamp(mvsFusionConfidence, 0.0, 1.0);
    mvsGpuFrameWorkers = std::max(0, mvsGpuFrameWorkers);
    mvsCpuFrameWorkers = std::max(0, mvsCpuFrameWorkers);
    mvsMaxFrames = std::max(0, mvsMaxFrames);
    mvsFusionMaxImageDim = std::max(0, mvsFusionMaxImageDim);

    const QString listPath = cleanAbsolutePath(QString::fromStdString(listPathArg));
    const QString outputDir = cleanAbsolutePath(QString::fromStdString(outputDirArg));
    QString error;
    if (!validateOutputDirectory(outputDir, forceOutput, &error))
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
    QJsonObject projectMeta;
    if (!readImageCameraList(listPath, &items, &projectMeta, &error))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(error));
        return cli::EXIT_ARG_ERR;
    }

    QStringList images;
    QStringList cameraPaths;
    for (const InputItem &item : items)
    {
        images.append(item.imagePath);
        cameraPaths.append(item.cameraPath);
    }

    QJsonObject report;
    report[QStringLiteral("status")] = QStringLiteral("running");
    report[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    report[QStringLiteral("list_file")] = listPath;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("inputs")] = inputsToJson(items);

    auto writeFinalReport = [&](QJsonObject *finalReport) {
        QString reportError;
        if (!writeReport(outputDir, report, finalReport, &reportError))
        {
            std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(reportError));
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

    std::fprintf(stdout, "[1/%d] SFM 稀疏重建...\n", kTotalStages);
    std::fflush(stdout);
    const auto sfmStart = std::chrono::steady_clock::now();
    xjw::gui::SFMServiceOptions sfmOptions;
    sfmOptions.images = images;
    sfmOptions.cameraPaths = cameraPaths;
    sfmOptions.projectMeta = projectMeta;
    sfmOptions.plascanPath = QDir(outputDir).filePath(QStringLiteral("headless.plascan"));
    sfmOptions.outputDir = QDir(outputDir).filePath(QStringLiteral("sparse"));
    sfmOptions.device = QString::fromStdString(device);
    sfmOptions.featureAlgorithm = QString::fromStdString(sfmFeatureAlgorithm);
    sfmOptions.matchAlgorithm = QString::fromStdString(sfmMatchAlgorithm);
    sfmOptions.enableGuidedRematching = sfmGuidedRematching;
    sfmOptions.quality = qBound(0, quality, 3);
    sfmOptions.threads = std::max(1, threads);
    sfmOptions.cudaParallelPairs = std::max(1, cudaParallelPairs);
    sfmOptions.featureMaxImageDim = featureMaxImageDim;
    sfmOptions.progressFn = [](const QString &stage, int percent) {
        std::fprintf(stdout, "  [SFM %3d%%] %s\n", percent, qUtf8Printable(stage));
        std::fflush(stdout);
    };

    const xjw::gui::SFMServiceResult sfmResult = xjw::gui::SFMService::run(sfmOptions);
    sfmElapsedMs = recordTiming(QStringLiteral("sfm_elapsed_ms"), sfmStart);
    QJsonObject sfmJson;
    sfmJson[QStringLiteral("success")] = sfmResult.success;
    sfmJson[QStringLiteral("summary")] = sfmResult.summary;
    sfmJson[QStringLiteral("sparse_cloud")] = sfmResult.sparseCloudPath;
    sfmJson[QStringLiteral("registered_images")] = sfmResult.numRegisteredImages;
    sfmJson[QStringLiteral("points")] = sfmResult.numPoints3D;
    sfmJson[QStringLiteral("mean_reprojection_error")] = sfmResult.meanReprojError;
    report[QStringLiteral("sfm")] = sfmJson;
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
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
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
        std::fprintf(stdout, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
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
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    const auto sparsePreprocessStart = std::chrono::steady_clock::now();
    QMap<QString, xjw::Camera> cameraByImage;
    for (auto it = sfmResult.pendingCamUpdates.constBegin(); it != sfmResult.pendingCamUpdates.constEnd(); ++it)
    {
        xjw::Camera camera;
        const QString imagePath = cleanAbsolutePath(it.key());
        if (xjw::common::project::cameraFromJson(it.value(), &camera) && camera.isValid())
        {
            cameraByImage.insert(imagePath, camera);
        }
    }

    QStringList registeredImagePaths;
    for (const InputItem &item : items)
    {
        const QString imagePath = cleanAbsolutePath(item.imagePath);
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
    for (const QString &imagePath : registeredImagePaths)
    {
        const xjw::Camera camera = cameraByImage.value(imagePath);
        if (!camera.isValid())
        {
            continue;
        }

        xjw::mvs::CameraView view;
        view.imagePath = imagePath.toStdString();
        view.camera = camera;
        cv::Mat image = cv::imread(view.imagePath, cv::IMREAD_GRAYSCALE);
        if (!image.empty())
        {
            view.imageWidth = image.cols;
            view.imageHeight = image.rows;
        }
        views.push_back(std::move(view));

        imageMetaArray.append(QJsonObject{
            {QStringLiteral("path"), imagePath},
            {QStringLiteral("name"), QFileInfo(imagePath).fileName()},
            {QStringLiteral("camera"), cameraToJson(camera)}
        });
    }
    limitMvsInputsForRegression(&views, &registeredImagePaths, &imageMetaArray, mvsMaxFrames);
    sfmJson[QStringLiteral("mvs_image_paths")] = QJsonArray::fromStringList(registeredImagePaths);
    sfmJson[QStringLiteral("mvs_input_images")] = registeredImagePaths.size();
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
        std::fprintf(stderr, "MVS 输入不足: report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    xjw::mvs::SparseCloud sparse;
    {
        xjw::mvs::SparseCloudPreprocessor preprocessor;
        xjw::mvs::PreprocessResult preprocessResult;
        std::string preprocessError;
        if (preprocessor.run(sfmResult.sparseCloudPath.toStdString(), views, preprocessResult, &preprocessError))
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
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }
    sparsePreprocessElapsedMs = recordTiming(
        QStringLiteral("sparse_preprocess_elapsed_ms"),
        sparsePreprocessStart);

    std::fprintf(stdout, "[2/%d] MVS 稠密点云...\n", kTotalStages);
    std::fflush(stdout);
    const auto mvsStart = std::chrono::steady_clock::now();
    const QString mvsDir = QDir(outputDir).filePath(QStringLiteral("mvs"));
    QDir().mkpath(mvsDir);
    xjw::gui::project::DenseGenerationSettings denseSettings;
    denseSettings.threads = std::max(1, threads);
    denseSettings.useCuda = (device == "cuda" || device == "auto");
    denseSettings.pipelineMode = true;
    denseSettings.resScale = mvsResScale;
    denseSettings.iterations = mvsIterations;
    denseSettings.patchMatchConfidence = mvsConfidence;
    denseSettings.fusionMinConfidence = mvsFusionConfidence;
    denseSettings.gpuFrameWorkers = mvsGpuFrameWorkers;
    denseSettings.cpuFrameWorkers = mvsCpuFrameWorkers;
    denseSettings.fusionMaxImageDim = mvsFusionMaxImageDim;
    const int denseMinViewCount = std::clamp(static_cast<int>(views.size()),
                                             kMinimumRegisteredImagesForDenseWorkflow,
                                             3);
    denseSettings.minViews = denseMinViewCount;
    denseSettings.minConsistentViews = denseMinViewCount;
    denseSettings.depthConsistency = 1.0f;
    xjw::mvs::DepthGenConfig depthConfig =
        xjw::gui::project::buildDepthGenConfig(denseSettings, static_cast<int>(views.size()));
    depthConfig.runFusion = false;
    depthConfig.saveIntermediateDepthMaps = true;
    depthConfig.intermediateDir = mvsDir.toStdString();

    xjw::mvs::DepthMapGenerator generator;
    generator.setViews(views);
    generator.setSparseCloud(sparse);
    generator.setConfig(depthConfig);
    generator.setOutputDir(mvsDir.toStdString());

    QEventLoop loop;
    bool depthOk = false;
    QString mvsError;
    QJsonArray depthArtifacts;
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::progressChanged, &loop,
                     [](const QString &stage, float ratio) {
        std::fprintf(stdout, "  [MVS %3d%%] %s\n", static_cast<int>(ratio * 100.0f), qUtf8Printable(stage));
        std::fflush(stdout);
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
        std::fprintf(stdout, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
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

            xjw::gui::project::DenseRefineSettings refineSettings;
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
            if (!writePointCloudPly(denseCloudPath, rawPointCloud, true, &error))
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
                if (!writePointCloudPly(refinedCloudPath,
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
        report[QStringLiteral("reason")] = !error.isEmpty() ? error : (mvsError.isEmpty() ? QStringLiteral("MVS 未生成有效稠密点云") : mvsError);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "MVS 失败: %s\n", qUtf8Printable(report.value(QStringLiteral("reason")).toString()));
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
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
        std::fprintf(stdout, "[3/%d] 三维网格模型...\n", kTotalStages);
        std::fflush(stdout);
        const auto meshStart = std::chrono::steady_clock::now();
        xjw::mesh::workflow::MeshBuildRequest meshRequest;
        meshRequest.pointCloudPath = refinedCloudPathForModel;
        meshRequest.outputRoot = QDir(outputDir).filePath(QStringLiteral("model"));
        meshRequest.exportObj = exportObj;
        meshRequest.reconstruction.resolution = qBound(64, meshResolution, 1024);
        meshRequest.reconstruction.poissonDepth = 9;
        meshRequest.reconstruction.simplifyTargetFaces = 28000;
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
        std::fprintf(stdout, "[4/4] DEM/DOM 产品...\n");
        std::fflush(stdout);
        const auto terrainStart = std::chrono::steady_clock::now();
        const QString terrainDir = QDir(outputDir).filePath(QStringLiteral("terrain"));
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
    const QString demPath = terrain.value(QStringLiteral("dem")).toObject().value(QStringLiteral("dem_tif")).toString();
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
    QJsonObject finalReport;
    if (!writeFinalReport(&finalReport))
    {
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout, "status=%s\n", qUtf8Printable(report.value(QStringLiteral("status")).toString()));
    std::fprintf(stdout, "output_dir=%s\n", qUtf8Printable(outputDir));
    std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(sfmResult.sparseCloudPath));
    std::fprintf(stdout, "dense_cloud=%s points=%d\n", qUtf8Printable(denseCloudPathForReport), densePointCount);
    std::fprintf(stdout, "refined_dense_cloud=%s points=%d\n", qUtf8Printable(refinedCloudPathForModel), refinedPointCount);
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
