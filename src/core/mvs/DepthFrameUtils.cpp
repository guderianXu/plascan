#include "DepthFrameUtils.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <array>
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

} // namespace

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

    for (const QJsonValue &value : depthResults)
    {
        const QJsonObject record = value.toObject();
        const QString depthPng = record.value(QStringLiteral("depth_png")).toString();
        const QString rawDepthPath = resolveExistingRawDepthPath(
            depthPng,
            record.value(QStringLiteral("raw_depth_path")).toString());
        if (rawDepthPath.isEmpty())
        {
            continue;
        }
        if (QFileInfo(rawDepthPath).absolutePath() != latestDir)
        {
            continue;
        }

        StoredDepthFrameRecord frame;
        frame.refImage = record.value(QStringLiteral("ref_image")).toString();
        frame.depthPng = depthPng;
        frame.rawDepthPath = rawDepthPath;
        frame.rawConfidencePath = resolveExistingRawConfidencePath(
            frame.depthPng,
            record.value(QStringLiteral("raw_confidence_path")).toString());
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

    result.batchDir = latestDir;
    if (result.frames.empty())
    {
        result.status = {false, QStringLiteral("最近一次深度图批次不包含可用帧")};
        return result;
    }

    result.status = {true, QString()};
    return result;
}

xjw::mvs::PositiveDepthCameraModel scalePositiveDepthCameraModel(
    const xjw::mvs::PositiveDepthCameraModel &camera,
    double scaleX,
    double scaleY)
{
    xjw::mvs::PositiveDepthCameraModel scaled = camera;
    scaled.fx *= static_cast<float>(scaleX);
    scaled.fy *= static_cast<float>(scaleY);
    scaled.cx *= static_cast<float>(scaleX);
    scaled.cy *= static_cast<float>(scaleY);
    return scaled;
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
    frame->cameraModel = scalePositiveDepthCameraModel(frame->cameraModel, scaleX, scaleY);
    frame->imgW = targetSize.width;
    frame->imgH = targetSize.height;
    return true;
}

FusionFrameBuildResult buildStoredFusionFrame(const StoredDepthFrameRecord &stored,
                                              const xjw::Camera &camera,
                                              float confidenceThreshold,
                                              int viewCount,
                                              int fusionMaxImageDim)
{
    FusionFrameBuildResult result;
    result.frame.cameraModel = camera.toPositiveDepthModel();
    result.frame.imgW = stored.gridWidth;
    result.frame.imgH = stored.gridHeight;
    result.frame.imagePath = stored.refImage.toStdString();

    const QString rawDepthPath = resolveExistingRawDepthPath(stored.depthPng, stored.rawDepthPath);
    result.status = loadDepthMatStorage(rawDepthPath, &result.frame.depthMap);
    if (!result.status.ok)
    {
        return result;
    }

    const float effectiveThreshold = (viewCount <= 2) ? 0.0f : confidenceThreshold;
    const bool shouldLoadConfidence = effectiveThreshold > 0.0f;
    if (shouldLoadConfidence)
    {
        const QString rawConfidencePath = resolveExistingRawConfidencePath(stored.depthPng, stored.rawConfidencePath);
        if (!rawConfidencePath.isEmpty())
        {
            (void)loadDepthMatStorage(rawConfidencePath, &result.frame.confidence);
        }
    }

    if (shouldLoadConfidence &&
        !result.frame.confidence.empty() &&
        result.frame.confidence.size() == result.frame.depthMap.size() &&
        result.frame.confidence.type() == CV_32F &&
        result.frame.depthMap.type() == CV_32F)
    {
        for (int row = 0; row < result.frame.depthMap.rows; ++row)
        {
            float *depthRow = result.frame.depthMap.ptr<float>(row);
            const float *confRow = result.frame.confidence.ptr<float>(row);
            for (int col = 0; col < result.frame.depthMap.cols; ++col)
            {
                if (confRow[col] < effectiveThreshold)
                {
                    depthRow[col] = 0.0f;
                }
            }
        }
    }

    downsampleFusionFrameForMaxDimension(&result.frame, fusionMaxImageDim);
    result.frame.confidence.release();
    result.frame.imgW = result.frame.depthMap.cols;
    result.frame.imgH = result.frame.depthMap.rows;
    result.status = {true, QString()};
    return result;
}

} // namespace xjw::core::project
