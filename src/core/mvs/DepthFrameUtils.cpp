#include "DepthFrameUtils.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <limits>

#include <opencv2/core.hpp>

namespace xjw::core::project
{

namespace
{

xjw::common::OperationResult loadCvMatStorage(const QString &path, cv::Mat *matrix)
{
    if (!matrix)
    {
        return {false, QStringLiteral("内部错误：矩阵输出参数无效")};
    }

    cv::FileStorage storage(path.toStdString(), cv::FileStorage::READ);
    if (!storage.isOpened())
    {
        return {false, QStringLiteral("无法读取矩阵文件：%1").arg(path)};
    }

    storage["mat"] >> *matrix;
    storage.release();
    if (matrix->empty())
    {
        return {false, QStringLiteral("矩阵文件内容为空：%1").arg(path)};
    }

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
    return info.dir().filePath(info.completeBaseName() + QStringLiteral(".yml.gz"));
}

QString rawConfidenceStoragePath(const QString &pngPath)
{
    const QFileInfo info(pngPath);
    return info.dir().filePath(info.completeBaseName() + QStringLiteral("_conf.yml.gz"));
}

bool depthFrameArtifactsExist(const QString &pngPath, bool requireConfidence)
{
    if (pngPath.trimmed().isEmpty() || !QFileInfo::exists(pngPath))
    {
        return false;
    }

    const QString rawDepthPath = rawDepthStoragePath(pngPath);
    if (!QFileInfo::exists(rawDepthPath))
    {
        return false;
    }

    if (requireConfidence && !QFileInfo::exists(rawConfidenceStoragePath(pngPath)))
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
        return false;
    }

    if (requireConfidence &&
        (frame.rawConfidencePath.trimmed().isEmpty() || !QFileInfo::exists(frame.rawConfidencePath)))
    {
        return false;
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
        const QString rawDepthPath = record.value(QStringLiteral("raw_depth_path")).toString();
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
        const QString rawDepthPath = record.value(QStringLiteral("raw_depth_path")).toString();
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
        frame.depthPng = record.value(QStringLiteral("depth_png")).toString();
        frame.rawDepthPath = rawDepthPath;
        frame.rawConfidencePath = record.value(QStringLiteral("raw_confidence_path")).toString();
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

FusionFrameBuildResult buildStoredFusionFrame(const StoredDepthFrameRecord &stored,
                                              const xjw::Camera &camera,
                                              float confidenceThreshold,
                                              int viewCount)
{
    FusionFrameBuildResult result;
    result.frame.cameraModel = camera.toPositiveDepthModel();
    result.frame.imgW = stored.gridWidth;
    result.frame.imgH = stored.gridHeight;
    result.frame.imagePath = stored.refImage.toStdString();

    result.status = loadCvMatStorage(stored.rawDepthPath, &result.frame.depthMap);
    if (!result.status.ok)
    {
        return result;
    }

    if (!stored.rawConfidencePath.isEmpty() && QFileInfo::exists(stored.rawConfidencePath))
    {
        (void)loadCvMatStorage(stored.rawConfidencePath, &result.frame.confidence);
    }

    if (!result.frame.confidence.empty())
    {
        float effectiveThreshold = confidenceThreshold;
        if (viewCount <= 2)
        {
            effectiveThreshold = 0.0f;
        }

        if (effectiveThreshold > 0.0f)
        {
            cv::Mat filteredDepth = result.frame.depthMap.clone();
            for (int row = 0; row < filteredDepth.rows; ++row)
            {
                float *depthRow = filteredDepth.ptr<float>(row);
                const float *confRow = result.frame.confidence.ptr<float>(row);
                for (int col = 0; col < filteredDepth.cols; ++col)
                {
                    if (confRow[col] < effectiveThreshold)
                    {
                        depthRow[col] = 0.0f;
                    }
                }
            }
            result.frame.depthMap = std::move(filteredDepth);
        }
    }

    result.frame.imgW = result.frame.depthMap.cols;
    result.frame.imgH = result.frame.depthMap.rows;
    result.status = {true, QString()};
    return result;
}

} // namespace xjw::core::project
