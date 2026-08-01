#include "DepthFrameUtils.h"

#include "DepthMapGenerator.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw::core::project
{

namespace
{

constexpr std::array<char, 16> kFastDepthMatMagic{
    'P', 'L', 'A', 'S', 'D', 'E', 'P', 'T',
    'H', 'M', 'A', 'T', '0', '1', '\0', '\0'
};

struct FastDepthMatHeader
{
    char magic[16] = {};
    qint32 rows = 0;
    qint32 cols = 0;
    qint32 type = 0;
    quint64 dataBytes = 0;
};

QString firstExistingPath(const QStringList &paths)
{
    for (const QString &path : paths)
    {
        if (!path.trimmed().isEmpty() && QFileInfo::exists(path))
        {
            return path;
        }
    }
    return QString();
}

QString resolveExistingRawDepthPath(const QString &pngPath, const QString &preferredPath = QString())
{
    return firstExistingPath({
        preferredPath,
        rawDepthStoragePath(pngPath)
    });
}

QString resolveExistingRawConfidencePath(const QString &pngPath, const QString &preferredPath = QString())
{
    return firstExistingPath({
        preferredPath,
        rawConfidenceStoragePath(pngPath)
    });
}

xjw::common::OperationResult loadFastDepthMatStorage(const QString &path, cv::Mat *matrix)
{
    if (!matrix)
    {
        return {false, QStringLiteral("内部错误：矩阵输出参数无效")};
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return {false, QStringLiteral("无法读取二进制深度文件：%1").arg(path)};
    }

    FastDepthMatHeader header;
    if (file.read(reinterpret_cast<char *>(&header), sizeof(header)) != static_cast<qint64>(sizeof(header)))
    {
        return {false, QStringLiteral("二进制深度文件头不完整：%1").arg(path)};
    }

    if (std::memcmp(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size()) != 0)
    {
        return {false, QStringLiteral("二进制深度文件标识无效：%1").arg(path)};
    }
    if (header.rows <= 0 || header.cols <= 0 || header.dataBytes == 0)
    {
        return {false, QStringLiteral("二进制深度文件尺寸无效：%1").arg(path)};
    }

    const size_t elemSize = CV_ELEM_SIZE(header.type);
    if (elemSize == 0)
    {
        return {false, QStringLiteral("二进制深度文件类型无效：%1").arg(path)};
    }

    const quint64 expectedBytes = static_cast<quint64>(header.rows)
        * static_cast<quint64>(header.cols)
        * static_cast<quint64>(elemSize);
    if (header.dataBytes != expectedBytes)
    {
        return {false, QStringLiteral("二进制深度文件大小不匹配：%1").arg(path)};
    }

    cv::Mat loaded(header.rows, header.cols, header.type);
    if (file.read(reinterpret_cast<char *>(loaded.data), static_cast<qint64>(header.dataBytes)) !=
        static_cast<qint64>(header.dataBytes))
    {
        return {false, QStringLiteral("二进制深度文件数据不完整：%1").arg(path)};
    }

    *matrix = std::move(loaded);
    return {true, QString()};
}

int frameIndexFromPath(const QString &path)
{
    const QRegularExpression re(QStringLiteral("(\\d+)(?!.*\\d)"));
    const QRegularExpressionMatch match = re.match(QFileInfo(path).completeBaseName());
    return match.hasMatch() ? match.captured(1).toInt() : std::numeric_limits<int>::max();
}

QString normalizedDirectoryPath(const QString &path)
{
    const QFileInfo path_info(path);
    const QString directory_path = path_info.isDir()
        ? path_info.absoluteFilePath()
        : path_info.absolutePath();
    return QDir::cleanPath(directory_path);
}

StoredDepthFramesResult collectStoredDepthFramesInDirectory(const QJsonArray &depth_results,
                                                            const QString &batch_directory)
{
    StoredDepthFramesResult result;
    const QString normalized_batch_directory = normalizedDirectoryPath(batch_directory);
    if (normalized_batch_directory.isEmpty())
    {
        result.status = {false, QStringLiteral("深度图批次目录为空")};
        return result;
    }

    for (const QJsonValue &value : depth_results)
    {
        const QJsonObject record = value.toObject();
        const QString depth_png = record.value(QStringLiteral("depth_png")).toString();
        const QString raw_depth_path = resolveExistingRawDepthPath(
            depth_png,
            record.value(QStringLiteral("raw_depth_path")).toString());
        if (raw_depth_path.isEmpty() ||
            normalizedDirectoryPath(raw_depth_path).compare(normalized_batch_directory,
                                                              Qt::CaseInsensitive) != 0)
        {
            continue;
        }

        StoredDepthFrameRecord frame;
        frame.refImage = record.value(QStringLiteral("ref_image")).toString();
        frame.depthPng = depth_png;
        frame.rawDepthPath = raw_depth_path;
        frame.rawConfidencePath = resolveExistingRawConfidencePath(
            frame.depthPng,
            record.value(QStringLiteral("raw_confidence_path")).toString());
        const QJsonArray source_images = record.value(QStringLiteral("source_images")).toArray();
        for (const QJsonValue &source_image : source_images)
        {
            const QString path = source_image.toString().trimmed();
            if (!path.isEmpty())
            {
                frame.sourceImages.push_back(QDir::cleanPath(path));
            }
        }
        frame.configHash = record.value(QStringLiteral("config_hash")).toString();
        frame.algorithmRevision = record.value(
            QStringLiteral("algorithm_revision")).toInt(0);
        frame.projectInputSignature =
            record.value(QStringLiteral("project_input_signature")).toString();
        frame.reconstructionGenerationId =
            record.value(QStringLiteral("reconstruction_generation_id")).toString();
        frame.gridWidth = record.value(QStringLiteral("grid_width")).toInt();
        frame.gridHeight = record.value(QStringLiteral("grid_height")).toInt();
        if (!frame.refImage.isEmpty() && depthFrameArtifactsExist(frame))
        {
            result.frames.push_back(std::move(frame));
        }
    }

    std::sort(result.frames.begin(), result.frames.end(), [](const StoredDepthFrameRecord &lhs,
                                                              const StoredDepthFrameRecord &rhs) {
        return frameIndexFromPath(lhs.rawDepthPath) < frameIndexFromPath(rhs.rawDepthPath);
    });
    result.batchDir = normalized_batch_directory;
    if (result.frames.empty())
    {
        result.status = {false,
                         QStringLiteral("所选目录不包含可复用的原始深度图：%1")
                             .arg(normalized_batch_directory)};
        return result;
    }

    result.status = {true, QString()};
    return result;
}

} // namespace

std::uint64_t estimateFusionFrameWorkingSetBytes(int width,
                                                 int height,
                                                 int fusionMaxImageDim)
{
    if (width <= 0 || height <= 0)
    {
        return 64ULL * 1024ULL * 1024ULL;
    }

    const std::uint64_t source_pixels = static_cast<std::uint64_t>(width) *
                                        static_cast<std::uint64_t>(height);
    std::uint64_t target_pixels = source_pixels;
    if (fusionMaxImageDim > 0 && std::max(width, height) > fusionMaxImageDim)
    {
        const double scale = static_cast<double>(fusionMaxImageDim) /
                             static_cast<double>(std::max(width, height));
        const int target_width = std::max(1, static_cast<int>(std::lround(width * scale)));
        const int target_height = std::max(1, static_cast<int>(std::lround(height * scale)));
        target_pixels = static_cast<std::uint64_t>(target_width) *
                        static_cast<std::uint64_t>(target_height);
    }

    const std::uint64_t load_peak = source_pixels * 8ULL + target_pixels * 8ULL;
    const std::uint64_t postprocess_peak = target_pixels * 24ULL;
    return std::max(load_peak, postprocess_peak) + 16ULL * 1024ULL * 1024ULL;
}

int recommendedDepthFrameLoadWorkers(int requestedWorkers,
                                     std::uint64_t availableMemoryBytes,
                                     std::uint64_t frameWorkingSetBytes)
{
    const int requested = std::clamp(requestedWorkers > 0 ? requestedWorkers : 4, 1, 4);
    if (availableMemoryBytes == 0 || frameWorkingSetBytes == 0)
    {
        return std::clamp(requested, 2, 4);
    }

    const std::uint64_t loading_budget = availableMemoryBytes / 2ULL;
    const std::uint64_t memory_workers = loading_budget / frameWorkingSetBytes;
    return static_cast<int>(std::clamp<std::uint64_t>(
        std::min<std::uint64_t>(requested, memory_workers), 1ULL, 4ULL));
}

QString rawDepthStoragePath(const QString &pngPath)
{
    const QFileInfo info(pngPath);
    return info.dir().filePath(info.completeBaseName() + QStringLiteral(".bin"));
}

QString rawConfidenceStoragePath(const QString &pngPath)
{
    const QFileInfo info(pngPath);
    return info.dir().filePath(info.completeBaseName() + QStringLiteral("_conf.bin"));
}

xjw::common::OperationResult loadDepthMatStorage(const QString &path, cv::Mat *matrix)
{
    if (!path.endsWith(QStringLiteral(".bin"), Qt::CaseInsensitive))
    {
        return {false, QStringLiteral("不支持的深度矩阵格式：%1").arg(path)};
    }
    return loadFastDepthMatStorage(path, matrix);
}

xjw::common::OperationResult writeDepthMatStorage(const QString &path, const cv::Mat &matrix)
{
    if (matrix.empty())
    {
        return {false, QStringLiteral("矩阵为空，无法写入二进制深度文件：%1").arg(path)};
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return {false, QStringLiteral("无法写入二进制深度文件：%1").arg(path)};
    }

    const cv::Mat contiguous = matrix.isContinuous() ? matrix : matrix.clone();
    FastDepthMatHeader header;
    std::memcpy(header.magic, kFastDepthMatMagic.data(), kFastDepthMatMagic.size());
    header.rows = contiguous.rows;
    header.cols = contiguous.cols;
    header.type = contiguous.type();
    header.dataBytes = static_cast<quint64>(contiguous.total() * contiguous.elemSize());

    if (file.write(reinterpret_cast<const char *>(&header), sizeof(header)) != static_cast<qint64>(sizeof(header)) ||
        file.write(reinterpret_cast<const char *>(contiguous.data), static_cast<qint64>(header.dataBytes)) !=
            static_cast<qint64>(header.dataBytes))
    {
        return {false, QStringLiteral("写入二进制深度文件失败：%1").arg(path)};
    }

    return {true, QString()};
}

bool depthFrameArtifactsExist(const QString &pngPath, bool requireConfidence)
{
    if (pngPath.trimmed().isEmpty() || !QFileInfo::exists(pngPath))
    {
        return false;
    }

    const QString rawDepthPath = resolveExistingRawDepthPath(pngPath);
    if (rawDepthPath.isEmpty())
    {
        return false;
    }

    if (requireConfidence && resolveExistingRawConfidencePath(pngPath).isEmpty())
    {
        return false;
    }

    return true;
}

bool depthFrameArtifactsExist(const StoredDepthFrameRecord &frame, bool requireConfidence)
{
    if (frame.depthPng.trimmed().isEmpty() || !QFileInfo::exists(frame.depthPng))
    {
        return false;
    }

    if (frame.rawDepthPath.trimmed().isEmpty() || !QFileInfo::exists(frame.rawDepthPath))
    {
        const QString fallbackRawDepthPath = resolveExistingRawDepthPath(frame.depthPng, frame.rawDepthPath);
        if (fallbackRawDepthPath.isEmpty())
        {
            return false;
        }
    }

    if (requireConfidence &&
        (frame.rawConfidencePath.trimmed().isEmpty() || !QFileInfo::exists(frame.rawConfidencePath)))
    {
        const QString fallbackConfidencePath =
            resolveExistingRawConfidencePath(frame.depthPng, frame.rawConfidencePath);
        if (fallbackConfidencePath.isEmpty())
        {
            return false;
        }
    }

    return true;
}

StoredDepthFramesResult collectLatestStoredDepthFrames(const QJsonObject &projectMeta)
{
    StoredDepthFramesResult result;

    const QJsonArray depthResults = projectMeta.value(QStringLiteral("depth_map_results")).toArray();
    QString latestDir;
    for (int index = depthResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = depthResults.at(index).toObject();
        const QString depthPng = record.value(QStringLiteral("depth_png")).toString();
        const QString rawDepthPath = resolveExistingRawDepthPath(
            depthPng,
            record.value(QStringLiteral("raw_depth_path")).toString());
        if (!depthPng.isEmpty() &&
            !rawDepthPath.isEmpty() &&
            QFileInfo::exists(depthPng) &&
            QFileInfo::exists(rawDepthPath))
        {
            latestDir = QFileInfo(rawDepthPath).absolutePath();
            break;
        }
    }

    if (latestDir.isEmpty())
    {
        result.status = {false, QStringLiteral("未找到可复用的原始深度图，请先执行深度图估计")};
        return result;
    }

    return collectStoredDepthFramesInDirectory(depthResults, latestDir);
}

StoredDepthFramesResult collectStoredDepthFramesForDirectory(const QJsonObject &projectMeta,
                                                             const QString &batchDirectory)
{
    return collectStoredDepthFramesInDirectory(
        projectMeta.value(QStringLiteral("depth_map_results")).toArray(),
        batchDirectory);
}

std::vector<int> storedFusionSourceIndices(const std::vector<StoredDepthFrameRecord> &frames,
                                           int referenceIndex)
{
    std::vector<int> indices;
    if (referenceIndex < 0 || referenceIndex >= static_cast<int>(frames.size()))
    {
        return indices;
    }

    const QStringList &source_images = frames[static_cast<std::size_t>(referenceIndex)].sourceImages;
    indices.reserve(static_cast<std::size_t>(source_images.size()));
    for (const QString &source_image : source_images)
    {
        const QString normalized_source = QDir::cleanPath(source_image);
        for (int index = 0; index < static_cast<int>(frames.size()); ++index)
        {
            if (index == referenceIndex)
            {
                continue;
            }
            if (QDir::cleanPath(frames[static_cast<std::size_t>(index)].refImage)
                    .compare(normalized_source, Qt::CaseInsensitive) == 0)
            {
                indices.push_back(index);
                break;
            }
        }
    }
    return indices;
}

bool downsampleFusionFrameForMaxDimension(xjw::mvs::FusionFrameInput *frame,
                                          int fusionMaxImageDim)
{
    if (!frame || frame->depthMap.empty() || fusionMaxImageDim <= 0)
    {
        return false;
    }

    const int oldWidth = frame->depthMap.cols;
    const int oldHeight = frame->depthMap.rows;
    const int oldMaxSide = std::max(oldWidth, oldHeight);
    if (oldWidth <= 0 || oldHeight <= 0 || oldMaxSide <= fusionMaxImageDim)
    {
        frame->imgW = oldWidth;
        frame->imgH = oldHeight;
        return false;
    }

    const double scale = static_cast<double>(fusionMaxImageDim) / static_cast<double>(oldMaxSide);
    const cv::Size targetSize(std::max(1, static_cast<int>(std::round(oldWidth * scale))),
                              std::max(1, static_cast<int>(std::round(oldHeight * scale))));
    if (targetSize.width == oldWidth && targetSize.height == oldHeight)
    {
        frame->imgW = oldWidth;
        frame->imgH = oldHeight;
        return false;
    }

    cv::Mat resizedDepth;
    cv::resize(frame->depthMap, resizedDepth, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
    frame->depthMap = std::move(resizedDepth);

    if (!frame->confidence.empty())
    {
        cv::Mat resizedConfidence;
        cv::resize(frame->confidence, resizedConfidence, targetSize, 0.0, 0.0, cv::INTER_AREA);
        frame->confidence = std::move(resizedConfidence);
    }

    const double scaleX = static_cast<double>(targetSize.width) / static_cast<double>(oldWidth);
    const double scaleY = static_cast<double>(targetSize.height) / static_cast<double>(oldHeight);
    frame->cameraModel = frame->cameraModel.scaledIntrinsics(scaleX, scaleY);
    frame->imgW = targetSize.width;
    frame->imgH = targetSize.height;
    return true;
}

FusionFrameBuildResult buildStoredFusionFrame(const StoredDepthFrameRecord &stored,
                                              const xjw::Camera &camera,
                                              const xjw::mvs::FusionConfig &fusionConfig,
                                              int viewCount,
                                              int fusionMaxImageDim)
{
    FusionFrameBuildResult result;
    const auto total_start = std::chrono::steady_clock::now();
    result.frame.sourceCamera = camera;
    result.frame.cameraModel = camera.normalizedForPositiveDepth();
    result.frame.cameraModel.setDistortion(xjw::Camera::Distortion{});
    result.frame.imgW = stored.gridWidth;
    result.frame.imgH = stored.gridHeight;
    result.frame.imagePath = xjw::common::io::toUtf8Path(stored.refImage);

    const QString rawDepthPath = resolveExistingRawDepthPath(stored.depthPng, stored.rawDepthPath);
    const auto read_start = std::chrono::steady_clock::now();
    result.status = loadDepthMatStorage(rawDepthPath, &result.frame.depthMap);
    if (!result.status.ok)
    {
        return result;
    }

    const QString rawConfidencePath = resolveExistingRawConfidencePath(stored.depthPng,
                                                                       stored.rawConfidencePath);
    if (!rawConfidencePath.isEmpty())
    {
        (void)loadDepthMatStorage(rawConfidencePath, &result.frame.confidence);
    }
    const auto read_done = std::chrono::steady_clock::now();

    const auto resize_start = read_done;
    downsampleFusionFrameForMaxDimension(&result.frame, fusionMaxImageDim);
    const auto resize_done = std::chrono::steady_clock::now();

    const auto postprocess_start = resize_done;
    result.frame.depthPostprocess = xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(
        result.frame.depthMap,
        result.frame.confidence,
        fusionConfig,
        frameIndexFromPath(stored.rawDepthPath),
        viewCount);
    const auto postprocess_done = std::chrono::steady_clock::now();
    result.frame.confidence.release();
    result.frame.imgW = result.frame.depthMap.cols;
    result.frame.imgH = result.frame.depthMap.rows;
    result.status = {true, QString()};
    result.readMs = std::chrono::duration<double, std::milli>(read_done - read_start).count();
    result.resizeMs = std::chrono::duration<double, std::milli>(resize_done - resize_start).count();
    result.postprocessMs =
        std::chrono::duration<double, std::milli>(postprocess_done - postprocess_start).count();
    result.totalMs =
        std::chrono::duration<double, std::milli>(postprocess_done - total_start).count();
    return result;
}

} // namespace xjw::core::project
