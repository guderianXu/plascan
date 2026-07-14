#include "ProjectSurveyControl.h"

#include "ProjectData.h"
#include "ProjectIO.h"
#include "io/MarkerCsv.h"
#include "io/MarkerSetStore.h"
#include "io/SurveyControlMigration.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSaveFile>

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

SurveyControlProjectImportResult publishMarkerSet(ProjectData *projectData,
                                                  const control_points::MarkerSet &markerSet)
{
    SurveyControlProjectImportResult result;
    const QString sidecar_path = ProjectIO::markerSetPath(projectData->currentProjectPath());
    const bool sidecar_existed = QFile::exists(sidecar_path);
    QByteArray previous_bytes;
    if (sidecar_existed)
    {
        QFile previous(sidecar_path);
        if (!previous.open(QIODevice::ReadOnly))
        {
            result.errorMessage = QStringLiteral("无法读取现有标记 sidecar，已取消导入: %1")
                                      .arg(previous.errorString());
            return result;
        }
        previous_bytes = previous.readAll();
    }

    const control_points::MarkerSetStore store(sidecar_path);
    const auto saved = store.save(markerSet);
    if (!saved.ok)
    {
        result.errorMessage = saved.error;
        return result;
    }

    const auto verified = store.load();
    if (!verified.ok || !(verified.markerSet == markerSet))
    {
        const bool restored = restoreSidecar(sidecar_path, sidecar_existed, previous_bytes);
        result.errorMessage = QStringLiteral("标记 sidecar 写后校验失败，metadata 未修改%1: %2")
                                  .arg(restored ? QString() : QStringLiteral("，且旧 sidecar 恢复失败"),
                                       verified.error);
        return result;
    }

    QJsonObject metadata = projectData->coreFilesMeta();
    metadata.remove(QStringLiteral("survey_control"));
    int control_count = 0;
    int check_count = 0;
    for (const control_points::Marker &marker : markerSet.markers())
    {
        if (marker.role == control_points::MarkerRole::ControlPoint) ++control_count;
        else if (marker.role == control_points::MarkerRole::CheckPoint) ++check_count;
    }
    metadata[QStringLiteral("marker_set")] = QJsonObject{
        {QStringLiteral("path"), QStringLiteral("assets/control_points/marker_set.json")},
        {QStringLiteral("schema_version"), markerSet.schemaVersion()},
        {QStringLiteral("marker_count"), markerSet.markers().size()},
        {QStringLiteral("control_point_count"), control_count},
        {QStringLiteral("check_point_count"), check_count},
        {QStringLiteral("scale_bar_count"), markerSet.scaleBars().size()},
        {QStringLiteral("updated_at"), markerSet.updatedAt().toString(Qt::ISODateWithMs)}
    };
    projectData->updateMetadata(metadata, true);

    result.controlPointCount = control_count;
    result.checkPointCount = check_count;
    result.scaleBarCount = markerSet.scaleBars().size();
    result.imported = true;
    return result;
}

} // namespace

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

    control_points::MarkerCsvImportOptions options;
    options.defaultRole = defaultRole;
    options.imageIdentityByPath = imageIdentityMap(*projectData);
    const auto imported = control_points::readMarkerCsvFile(csvPath, options);
    if (!imported.ok)
    {
        result.errorMessage = imported.error;
        return result;
    }
    return publishMarkerSet(projectData, imported.markerSet);
}

SurveyControlProjectImportResult migrateLegacySurveyControl(ProjectData *projectData)
{
    SurveyControlProjectImportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法迁移旧测绘控制数据");
        return result;
    }

    const QJsonObject legacy = projectData->coreFilesMeta()
                                   .value(QStringLiteral("survey_control")).toObject();
    if (legacy.isEmpty())
    {
        result.imported = true;
        return result;
    }

    const auto migrated = control_points::migrateSurveyControl(legacy, imageIdentityMap(*projectData));
    if (!migrated.ok)
    {
        result.errorMessage = migrated.error;
        return result;
    }
    return publishMarkerSet(projectData, migrated.markerSet);
}

} // namespace xjw::gui::project
