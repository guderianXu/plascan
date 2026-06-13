// =============================================================================
// 文件: cli_reconstruct_pipeline.cpp
// 功能: PlaScan 一键重建 CLI
//       默认: .lis(image camera) -> SFM -> MVS -> 三维模型 -> DEM/DOM
//       PLASCAN_THREE_D_ONLY: GUI 三维重建等价流程，仅 SFM -> MVS -> 三维模型
// =============================================================================
#include "cli_common.h"

#include "Camera.h"
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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <functional>
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
    std::fprintf(stdout, "%s\n", message.toLocal8Bit().constData());
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

    constexpr int kMaxPasses = 4;
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

PlaCloud refineDenseCloud(PlaCloud cloud,
                          const xjw::gui::project::DenseRefineSettings &request,
                          const std::function<void(const QString &, int)> &progress)
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

            const auto afterRadius = cloud.size();
            const bool largeCloud = beforeSor > 200000;
            const bool weakRemoval = (beforeSor > 0)
                && (static_cast<double>(beforeSor - afterRadius) / static_cast<double>(beforeSor) < 0.02);
            if (largeCloud && weakRemoval)
            {
                const double stricterStdDev = std::clamp(request.sorStdDev - 0.3, 0.8, request.sorStdDev);
                const int stricterK = std::clamp(request.sorK + 6, request.sorK, 96);
                if (progress) progress(QStringLiteral("离群点二次清理..."), 42);
                const auto beforeStrictSor = cloud.size();
                plapoint::ProcessingReport strictSorReport;
                cloud = sorFilter(cloud,
                                  stricterK,
                                  static_cast<float>(stricterStdDev),
                                  request.processingDevice,
                                  &strictSorReport);
                reportPlaPointDevice(progress, QStringLiteral("离群点二次清理"),
                                     strictSorReport, beforeStrictSor, cloud.size(), 44);
            }
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
    if (!matrix)
    {
        if (error) *error = QStringLiteral("内部错误：矩阵输出为空");
        return false;
    }

    cv::FileStorage storage(path.toStdString(), cv::FileStorage::READ);
    if (!storage.isOpened())
    {
        if (error) *error = QStringLiteral("无法读取矩阵文件: %1").arg(path);
        return false;
    }

    storage[QStringLiteral("mat").toStdString()] >> *matrix;
    storage.release();
    if (matrix->empty())
    {
        if (error) *error = QStringLiteral("矩阵文件内容为空: %1").arg(path);
        return false;
    }
    return true;
}

bool loadFusionFramesFromDepthMaps(const QString &mvsDir,
                                   const std::vector<xjw::mvs::CameraView> &views,
                                   float confidenceThreshold,
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
    const float effectiveConfidence = views.size() <= 2 ? 0.0f : confidenceThreshold;
    const QDir dir(mvsDir);

    for (std::size_t index = 0; index < views.size(); ++index)
    {
        const QString depthPath = dir.filePath(QStringLiteral("depth_%1.yml.gz").arg(index));
        cv::Mat depth;
        if (!loadCvMatStorage(depthPath, &depth, error))
        {
            return false;
        }

        cv::Mat confidence;
        const QString confPath = dir.filePath(QStringLiteral("depth_%1_conf.yml.gz").arg(index));
        if (QFileInfo::exists(confPath))
        {
            QString confError;
            if (!loadCvMatStorage(confPath, &confidence, &confError))
            {
                std::fprintf(stderr, "  [MVS] 置信图读取失败，继续使用深度图: %s\n", qUtf8Printable(confError));
            }
        }

        if (!confidence.empty() && effectiveConfidence > 0.0f)
        {
            cv::Mat filteredDepth = depth.clone();
            for (int row = 0; row < filteredDepth.rows; ++row)
            {
                float *depthRow = filteredDepth.ptr<float>(row);
                const float *confRow = confidence.ptr<float>(row);
                for (int col = 0; col < filteredDepth.cols; ++col)
                {
                    if (confRow[col] < effectiveConfidence)
                    {
                        depthRow[col] = 0.0f;
                    }
                }
            }
            depth = std::move(filteredDepth);
        }

        xjw::mvs::FusionFrameInput frame;
        frame.cameraModel = views[index].positiveDepthModel();
        frame.imgW = depth.cols;
        frame.imgH = depth.rows;
        frame.imagePath = views[index].imagePath;
        frame.depthMap = std::move(depth);
        frame.confidence = std::move(confidence);
        frames->push_back(std::move(frame));
    }

    return true;
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
    int quality = 3;
    int threads = 8;
    int cudaParallelPairs = 1;
    int featureMaxImageDim = 0;
#ifndef PLASCAN_THREE_D_ONLY
    double demResolution = 0.0;
#endif
    int meshResolution = 224;
    bool skipModel = false;
    bool skipMesh = false;
    bool stopAfterSfm = false;
    bool skipMvs = false;
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
    app.add_option("--quality", quality, "SFM quality level 0..3");
    app.add_option("--threads", threads, "CPU thread count");
    app.add_option("--cuda-parallel-pairs", cudaParallelPairs, "LightGlue CUDA parallel pair count");
    app.add_option("--feature-max-image-dim", featureMaxImageDim,
                   "deep feature max image side; 0 uses auto/adaptive quality preset, negative starts unbounded");
#ifndef PLASCAN_THREE_D_ONLY
    app.add_option("--dem-resolution", demResolution, "DEM/DOM resolution; 0 lets TerrainPipeline choose");
#endif
    app.add_option("--mesh-resolution", meshResolution, "mesh reconstruction grid resolution");
    app.add_flag("--stop-after-sfm", stopAfterSfm, "run SFM only, write report, then stop before MVS");
    app.add_flag("--skip-mvs", skipMvs, "skip MVS and downstream mesh/terrain stages after SFM");
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
    const int denseMinViewCount = std::clamp(static_cast<int>(views.size()),
                                             kMinimumRegisteredImagesForDenseWorkflow,
                                             3);
    denseSettings.minViews = denseMinViewCount;
    denseSettings.minConsistentViews = denseMinViewCount;
    denseSettings.fusionMinConfidence = 0.50f;
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
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::finished, &loop,
                     [&loop, &depthOk](bool success) {
        depthOk = success;
        loop.quit();
    });
    QTimer::singleShot(0, &generator, &xjw::mvs::DepthMapGenerator::start);
    loop.exec();

    std::vector<xjw::mvs::FusionFrameInput> fusionFrames;
    bool mvsOk = depthOk;
    QString denseCloudPathForReport;
    QString refinedCloudPathForModel;
    int densePointCount = 0;
    int refinedPointCount = 0;
    if (mvsOk)
    {
        if (!loadFusionFramesFromDepthMaps(mvsDir,
                                           views,
                                           denseSettings.fusionMinConfidence,
                                           &fusionFrames,
                                           &error))
        {
            mvsOk = false;
            mvsError = error;
        }
    }
    if (mvsOk)
    {
        xjw::mvs::StereoFusionConfig fusionCfg;
        fusionCfg.minNumPixels = std::max(1, denseSettings.minConsistentViews);
        fusionCfg.maxReprojError = denseSettings.depthConsistency;
        fusionCfg.maxDepthError = 0.05f;
        fusionCfg.checkNumImages = std::min(50, static_cast<int>(fusionFrames.size()));
        fusionCfg.workerCount = std::max(1, threads);
        if (fusionFrames.size() <= 2)
        {
            fusionCfg.minNumPixels = 1;
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.08f);
            fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f);
        }

        xjw::mvs::DepthMapFusion fusion(fusionCfg);
        std::string fuseError;
        std::vector<xjw::mvs::FusedPoint> fusedCloud;
        mvsOk = fusion.fuse(fusionFrames,
                            fusedCloud,
                            [](const std::string &stage, float ratio) {
                                std::fprintf(stdout,
                                             "  [MVS %3d%%] %s\n",
                                             70 + static_cast<int>(ratio * 25.0f),
                                             stage.c_str());
                                std::fflush(stdout);
                            },
                            &fuseError);
        if (!mvsOk)
        {
            mvsError = QString::fromStdString(fuseError);
        }
        else
        {
            constexpr std::size_t kLargeCloudPreVoxelThreshold = 2000000;
            constexpr std::size_t kMaxRefineInputPoints = 600000;
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
                                                         });
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
    report[QStringLiteral("dense")] = QJsonObject{
        {QStringLiteral("point_cloud"), denseCloudPathForReport},
        {QStringLiteral("refined_point_cloud"), refinedCloudPathForModel},
        {QStringLiteral("points"), densePointCount},
        {QStringLiteral("refined_points"), refinedPointCount},
        {QStringLiteral("has_rgb"), true},
        {QStringLiteral("has_normals"), true}
    };
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
