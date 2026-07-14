#include "FeatureResidualLoader.h"

#include "ProjectData.h"
#include "ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <cmath>

namespace
{

QString normalizedPath(const QString &path)
{
    QFileInfo info(path);
    QString normalized = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
    normalized = QDir::cleanPath(normalized);
#ifdef Q_OS_WIN
    normalized = normalized.toLower();
#endif
    return normalized;
}

QString latestSidecarPath(const QJsonObject &metadata, const QString &projectPath)
{
    const QJsonArray records = metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject files = records.at(index).toObject().value(QStringLiteral("files")).toObject();
        const QString path = ProjectIO::resolveProjectResourcePath(
            projectPath,
            files.value(QStringLiteral("sparse_cloud_points_json")).toString());
        if (QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

} // namespace

namespace xjw::gui::views
{

QVector<FeatureResidualVector> loadFeatureResidualsForImage(const QString &projectPath,
                                                            const QString &imagePath,
                                                            QString *errorMessage)
{
    QVector<FeatureResidualVector> residuals;
    if (projectPath.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
    {
        return residuals;
    }

    const ProjectResultsSnapshot snapshot = ProjectData::loadProjectResultsSnapshot(projectPath);
    if (!snapshot.success)
    {
        if (errorMessage)
        {
            *errorMessage = snapshot.errorMessage;
        }
        return residuals;
    }

    const QString sidecarPath = latestSidecarPath(snapshot.resultsMeta, projectPath);
    if (sidecarPath.isEmpty())
    {
        return residuals;
    }

    QFile file(sidecarPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取连接点残差文件: %1").arg(sidecarPath);
        }
        return residuals;
    }

    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("连接点残差文件格式无效: %1").arg(sidecarPath);
        }
        return residuals;
    }

    const QString targetPath = normalizedPath(imagePath);
    const QJsonArray points = document.object().value(QStringLiteral("points")).toArray();
    for (const QJsonValue &pointValue : points)
    {
        const QJsonArray observations = pointValue.toObject().value(QStringLiteral("observations")).toArray();
        for (const QJsonValue &observationValue : observations)
        {
            const QJsonObject observation = observationValue.toObject();
            if (normalizedPath(observation.value(QStringLiteral("image_path")).toString()) != targetPath)
            {
                continue;
            }

            const QJsonArray observed = observation.value(QStringLiteral("xy")).toArray();
            const QJsonArray projected = observation.value(QStringLiteral("projected_xy")).toArray();
            if (observed.size() < 2 || projected.size() < 2)
            {
                continue;
            }

            FeatureResidualVector vector;
            vector.observed = QPointF(observed.at(0).toDouble(), observed.at(1).toDouble());
            vector.projected = QPointF(projected.at(0).toDouble(), projected.at(1).toDouble());
            vector.magnitudePx = std::hypot(vector.projected.x() - vector.observed.x(),
                                             vector.projected.y() - vector.observed.y());
            if (std::isfinite(vector.magnitudePx))
            {
                residuals.append(vector);
            }
        }
    }
    return residuals;
}

} // namespace xjw::gui::views
