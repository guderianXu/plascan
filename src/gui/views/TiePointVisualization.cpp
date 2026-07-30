#include "TiePointVisualization.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <algorithm>
#include <cmath>

namespace xjw::gui::tie_points
{

bool ScalarRange::isValid() const
{
    return std::isfinite(minimum) && std::isfinite(maximum) && maximum >= minimum;
}

double ScalarRange::normalize(double value) const
{
    if (!isValid() || !std::isfinite(value) || maximum <= minimum)
    {
        return 0.5;
    }
    return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
}

bool ImageCountMetadata::isValidFor(qsizetype pointCount) const
{
    return pointCount > 0 && counts.size() == pointCount;
}

QColor elevationColor(double elevation, const ScalarRange &range)
{
    return scalarRampColor(range.normalize(elevation));
}

QColor imageCountColor(int imageCount, const ScalarRange &range)
{
    return scalarRampColor(1.0 - range.normalize(static_cast<double>(imageCount)));
}

QColor scalarRampColor(double normalizedValue)
{
    // Metashape 风格的高对比色带。固定满饱和度并适度压低亮度，
    // 在白色三维背景和点精灵抗锯齿下仍能保持鲜明、清晰。
    const double value = std::clamp(normalizedValue, 0.0, 1.0);
    return QColor::fromHsvF((1.0 - value) * (2.0 / 3.0), 1.0, 0.92);
}

float pointSizeForMode(ColorMode mode)
{
    return mode == ColorMode::Color ? 1.8f : 3.0f;
}

QString inferSidecarPath(const QString &pointCloudPath)
{
    const QFileInfo pointCloudInfo(pointCloudPath);
    if (pointCloudInfo.filePath().trimmed().isEmpty())
    {
        return QString();
    }

    const QString baseName = pointCloudInfo.completeBaseName();
    QStringList candidates;
    if (baseName == QLatin1String("sfm_sparse"))
    {
        candidates.append(QStringLiteral("sfm_sparse_points.json"));
    }
    candidates.append(QStringLiteral("sparse_cloud_points.json"));
    candidates.append(baseName + QStringLiteral("_points.json"));

    const QDir directory = pointCloudInfo.absoluteDir();
    for (const QString &candidate : candidates)
    {
        const QString path = directory.filePath(candidate);
        if (QFileInfo::exists(path))
        {
            return QDir::cleanPath(path);
        }
    }
    return QString();
}

ImageCountMetadata loadImageCountMetadata(const QString &sidecarPath)
{
    ImageCountMetadata result;
    if (sidecarPath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("未找到连接点观测数据文件");
        return result;
    }

    QFile file(sidecarPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        result.errorMessage = QStringLiteral("无法读取连接点观测数据: %1").arg(sidecarPath);
        return result;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        result.errorMessage = QStringLiteral("连接点观测数据格式无效: %1")
                                  .arg(parseError.errorString());
        return result;
    }

    const QJsonArray points = document.object().value(QStringLiteral("points")).toArray();
    if (points.isEmpty())
    {
        result.errorMessage = QStringLiteral("连接点观测数据中没有点记录");
        return result;
    }

    result.counts.reserve(points.size());
    for (const QJsonValue &pointValue : points)
    {
        const QJsonObject point = pointValue.toObject();
        if (!point.contains(QStringLiteral("track_len")))
        {
            result.counts.clear();
            result.errorMessage = QStringLiteral("连接点观测数据缺少 track_len 字段");
            return result;
        }
        result.counts.push_back(std::max(0, point.value(QStringLiteral("track_len")).toInt()));
    }
    return result;
}

} // namespace xjw::gui::tie_points
