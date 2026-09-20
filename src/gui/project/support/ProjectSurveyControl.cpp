#include "ProjectSurveyControl.h"

#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"
#include "io/MarkerCsv.h"
#include "io/MarkerSetStore.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSaveFile>
#include <QSet>

#include <cmath>

namespace xjw::gui::project
{

namespace
{

QHash<QString, QString> imageIdentityMap(const ProjectData &projectData)
{
    QHash<QString, QString> identities;
    for (const QJsonValue &value : projectData.coreFilesMeta().value(QStringLiteral("images")).toArray())
    {
        const QJsonObject image = value.toObject();
        const QString path = image.value(QStringLiteral("path")).toString().trimmed();
        const QString image_id = image.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (!path.isEmpty() && !image_id.isEmpty())
        {
            identities.insert(path, image_id);
        }
    }
    return identities;
}

bool restoreSidecar(const QString &path, bool existed, const QByteArray &bytes)
{
    if (!existed)
    {
        return !QFile::exists(path) || QFile::remove(path);
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size())
    {
        file.cancelWriting();
        return false;
    }
    return file.commit();
}

void applyAgisoftWgs84Defaults(const QString &path,
                               control_points::MarkerCsvImportOptions *options)
{
    if (!options)
    {
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return;
    }
    QString header;
    while (!file.atEnd())
    {
        QString candidate = QString::fromUtf8(file.readLine()).trimmed();
        if (!candidate.isEmpty() && candidate.front() == QChar::ByteOrderMark)
        {
            candidate.remove(0, 1);
        }
        if (!candidate.trimmed().isEmpty())
        {
            header = candidate.trimmed();
            break;
        }
    }
    QSet<QString> columns;
    const QStringList fields = header.split(QLatin1Char('\t'), Qt::SkipEmptyParts);
    for (QString field : fields)
    {
        field = field.trimmed().toCaseFolded();
        if (field.startsWith(QLatin1Char('#')))
        {
            field.remove(0, 1);
        }
        columns.insert(field);
    }
    const bool agisoftWgs84 = columns.contains(QStringLiteral("name"))
        && columns.contains(QStringLiteral("lat"))
        && columns.contains(QStringLiteral("lon"))
        && columns.contains(QStringLiteral("ell.h(m)"));
    if (!agisoftWgs84)
    {
        return;
    }
    if (options->defaultRole.trimmed().isEmpty())
    {
        options->defaultRole = QStringLiteral("control");
    }
    options->sourceCrs = QStringLiteral("EPSG:4979");
    options->axisOrder = QStringLiteral("longitude_latitude");
    options->verticalDatum = QStringLiteral("ellipsoidal");
    options->verticalUnit = QStringLiteral("m");
}

SurveyControlProjectImportResult publishMarkerSet(ProjectData *projectData,
                                                  const control_points::MarkerSet &markerSet)
{
    const QString sidecarPath =
        xjw::common::project::ProjectIO::markerSetPath(projectData->currentProjectPath());
    SurveyControlSidecarSnapshot previous;
    QString snapshotError;
    if (!captureSurveyControlSidecar(sidecarPath, &previous, &snapshotError))
    {
        return {false, snapshotError, 0, 0, 0};
    }
    const auto written = writeSurveyControlMarkerSet(sidecarPath, markerSet);
    if (!written.imported)
    {
        return written;
    }
    auto committed = commitSurveyControlMarkerSet(projectData, markerSet);
    if (!committed.imported)
    {
        QString restoreError;
        if (!restoreSurveyControlSidecar(previous, &restoreError))
        {
            committed.errorMessage += QStringLiteral("；且旧 sidecar 恢复失败: %1").arg(restoreError);
        }
    }
    return committed;
}

} // namespace

PreparedSurveyControlImport prepareSurveyControlCsv(
    const QHash<QString, QString>& imageIdentityByPath,
    const QString& csvPath,
    const QString& defaultRole)
{
    PreparedSurveyControlImport result;
    control_points::MarkerCsvImportOptions options;
    options.defaultRole = defaultRole;
    options.imageIdentityByPath = imageIdentityByPath;
    applyAgisoftWgs84Defaults(csvPath, &options);
    const auto imported = control_points::readMarkerCsvFile(csvPath, options);
    if (!imported.ok)
    {
        result.errorMessage = imported.error;
        return result;
    }
    result.prepared = true;
    result.markerSet = imported.markerSet;
    return result;
}

bool captureSurveyControlSidecar(const QString& sidecarPath,
                                 SurveyControlSidecarSnapshot* snapshot,
                                 QString* errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!snapshot || sidecarPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("标记 sidecar 路径为空");
        }
        return false;
    }

    snapshot->path = sidecarPath;
    snapshot->existed = QFile::exists(sidecarPath);
    snapshot->bytes.clear();
    if (!snapshot->existed)
    {
        return true;
    }

    QFile previous(sidecarPath);
    if (!previous.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取现有标记 sidecar: %1").arg(previous.errorString());
        }
        return false;
    }
    snapshot->bytes = previous.readAll();
    if (previous.error() != QFileDevice::NoError)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取现有标记 sidecar 失败: %1").arg(previous.errorString());
        }
        snapshot->bytes.clear();
        return false;
    }
    return true;
}

bool restoreSurveyControlSidecar(const SurveyControlSidecarSnapshot& snapshot,
                                 QString* errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (snapshot.path.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("标记 sidecar 恢复路径为空");
        }
        return false;
    }
    if (restoreSidecar(snapshot.path, snapshot.existed, snapshot.bytes))
    {
        return true;
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("无法恢复旧标记 sidecar: %1").arg(snapshot.path);
    }
    return false;
}

SurveyControlProjectImportResult writeSurveyControlMarkerSet(
    const QString& sidecarPath,
    const control_points::MarkerSet& markerSet)
{
    SurveyControlProjectImportResult result;
    SurveyControlSidecarSnapshot previous;
    QString snapshotError;
    if (!captureSurveyControlSidecar(sidecarPath, &previous, &snapshotError))
    {
        result.errorMessage = QStringLiteral("无法读取现有标记 sidecar，已取消导入: %1")
                                  .arg(snapshotError);
        return result;
    }

    const control_points::MarkerSetStore store(sidecarPath);
    const auto saved = store.save(markerSet);
    if (!saved.ok)
    {
        result.errorMessage = saved.error;
        return result;
    }
    const auto verified = store.load();
    if (!verified.ok || !(verified.markerSet == markerSet))
    {
        QString restoreError;
        const bool restored = restoreSurveyControlSidecar(previous, &restoreError);
        result.errorMessage = QStringLiteral("标记 sidecar 写后校验失败，metadata 未修改%1: %2")
                                  .arg(restored ? QString() : QStringLiteral("，且旧 sidecar 恢复失败: %1").arg(restoreError),
                                       verified.error);
        return result;
    }
    result.imported = true;
    return result;
}

SurveyControlProjectImportResult commitSurveyControlMarkerSet(
    ProjectData* projectData,
    const control_points::MarkerSet& markerSet)
{
    SurveyControlProjectImportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法写入测绘控制 metadata");
        return result;
    }

    QJsonObject metadata = projectData->coreFilesMeta();
    int controlCount = 0;
    int checkCount = 0;
    for (const control_points::Marker& marker : markerSet.markers())
    {
        if (marker.role == control_points::MarkerRole::ControlPoint)
        {
            ++controlCount;
        }
        else if (marker.role == control_points::MarkerRole::CheckPoint)
        {
            ++checkCount;
        }
    }
    metadata[QStringLiteral("marker_set")] = QJsonObject{
        {QStringLiteral("path"), QStringLiteral("assets/control_points/marker_set.json")},
        {QStringLiteral("schema_version"), markerSet.schemaVersion()},
        {QStringLiteral("marker_count"), markerSet.markers().size()},
        {QStringLiteral("control_point_count"), controlCount},
        {QStringLiteral("check_point_count"), checkCount},
        {QStringLiteral("scale_bar_count"), markerSet.scaleBars().size()},
        {QStringLiteral("updated_at"), markerSet.updatedAt().toString(Qt::ISODateWithMs)}};
    projectData->updateMetadata(metadata, true);
    if (projectData->coreFilesMeta().value(QStringLiteral("marker_set")) !=
        metadata.value(QStringLiteral("marker_set")))
    {
        result.errorMessage = QStringLiteral("测绘控制 metadata 写回校验失败");
        return result;
    }
    result.controlPointCount = controlCount;
    result.checkPointCount = checkCount;
    result.scaleBarCount = markerSet.scaleBars().size();
    result.imported = true;
    return result;
}

SurveyControlProjectImportResult importSurveyControlCsv(ProjectData *projectData,
                                                        const QString &csvPath,
                                                        const QString &defaultRole)
{
    SurveyControlProjectImportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法导入控制点数据");
        return result;
    }

    const auto prepared = prepareSurveyControlCsv(imageIdentityMap(*projectData), csvPath, defaultRole);
    if (!prepared.prepared)
    {
        result.errorMessage = prepared.errorMessage;
        return result;
    }
    return publishMarkerSet(projectData, prepared.markerSet);
}

QJsonObject surveyControlDialogMetadata(ProjectData *projectData, QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!projectData || !projectData->hasProject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("项目未打开，无法读取标记参考");
        }
        return {};
    }

    const QString sidecarPath =
        xjw::common::project::ProjectIO::markerSetPath(projectData->currentProjectPath());
    const control_points::MarkerSetIoResult loaded =
        control_points::MarkerSetStore(sidecarPath).load();
    if (!loaded.ok)
    {
        if (errorMessage)
        {
            *errorMessage = loaded.error;
        }
        return {};
    }

    QJsonArray controlPoints;
    QJsonArray checkPoints;
    QHash<QString, QString> labels;
    for (const control_points::Marker &marker : loaded.markerSet.markers())
    {
        labels.insert(marker.id, marker.label);
        if (marker.role == control_points::MarkerRole::TieMarker)
        {
            continue;
        }
        QJsonObject point{
            {QStringLiteral("id"), marker.label},
            {QStringLiteral("enabled"), marker.enabled}
        };
        if (marker.referenceCoordinate)
        {
            const control_points::ReferenceCoordinate &reference =
                *marker.referenceCoordinate;
            point[QStringLiteral("x")] = reference.x;
            point[QStringLiteral("y")] = reference.y;
            point[QStringLiteral("z")] = reference.z;
            point[QStringLiteral("sigma_m")] =
                (reference.sigmaX + reference.sigmaY + reference.sigmaZ) / 3.0;
        }
        if (marker.role == control_points::MarkerRole::ControlPoint)
        {
            controlPoints.append(point);
        }
        else
        {
            checkPoints.append(point);
        }
    }

    QJsonArray scaleBars;
    for (const control_points::ScaleBar &scaleBar : loaded.markerSet.scaleBars())
    {
        QJsonObject value{
            {QStringLiteral("id"), scaleBar.label},
            {QStringLiteral("from_id"), labels.value(scaleBar.firstMarkerId,
                                                       scaleBar.firstMarkerId)},
            {QStringLiteral("to_id"), labels.value(scaleBar.secondMarkerId,
                                                     scaleBar.secondMarkerId)},
            {QStringLiteral("measured_m"), scaleBar.measuredDistance},
            {QStringLiteral("sigma_m"), scaleBar.sigma},
            {QStringLiteral("enabled"), scaleBar.enabled}
        };
        if (std::isfinite(scaleBar.estimatedDistance))
        {
            value[QStringLiteral("estimated_m")] = scaleBar.estimatedDistance;
        }
        if (std::isfinite(scaleBar.residual))
        {
            value[QStringLiteral("residual_m")] = scaleBar.residual;
        }
        scaleBars.append(value);
    }

    return QJsonObject{
        {QStringLiteral("source_path"), sidecarPath},
        {QStringLiteral("control_points"), controlPoints},
        {QStringLiteral("check_points"), checkPoints},
        {QStringLiteral("scale_bars"), scaleBars}
    };
}

} // namespace xjw::gui::project
