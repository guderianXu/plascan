#include "MetashapeCameraReferenceSetBuilder.h"

#include "CameraReferenceProjectIdentity.h"
#include "project/services/MetashapeCameraReferenceImporter.h"
#include "project/ProjectMetadata.h"

#include <QHash>
#include <QJsonArray>

namespace xjw::gui::reference
{
namespace
{

struct ImageIdentity
{
    QString uuid;
    QString path;
};

QString portableFileName(QString path)
{
    path.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return path.section(QLatin1Char('/'), -1).trimmed();
}

QHash<QString, QVector<ImageIdentity>> projectImagesByName(const QJsonObject &metadata)
{
    QHash<QString, QVector<ImageIdentity>> result;
    const QJsonObject files = xjw::common::project::projectFilesRootObject(metadata);
    for (const QJsonValue &value : files.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject image = value.toObject();
        const QString uuid = image.value(QStringLiteral("image_uuid")).toString().trimmed();
        const QString path = image.value(QStringLiteral("path")).toString().trimmed();
        const QString key = portableFileName(path).toCaseFolded();
        if (!uuid.isEmpty() && !path.isEmpty() && !key.isEmpty())
        {
            result[key].append({uuid, path});
        }
    }
    return result;
}

camera_reference::RawCameraReference rawReference(
    const reference_import::RawCameraReferenceRecord &source)
{
    camera_reference::RawCameraReference result;
    result.position = camera_reference::Vector3d{{
        source.wgs84LongitudeDegrees,
        source.wgs84LatitudeDegrees,
        source.wgs84EllipsoidalHeightMeters}};
    if (source.stdDevEastMeters && source.stdDevNorthMeters && source.stdDevUpMeters)
    {
        result.positionSigma = camera_reference::Vector3d{{
            *source.stdDevEastMeters,
            *source.stdDevNorthMeters,
            *source.stdDevUpMeters}};
        result.positionSigmaFrame = QStringLiteral("local_enu");
        result.positionSigmaUnit = QStringLiteral("m");
    }
    result.horizontalSigmaMeters = source.stdDevHorizontalMeters;
    result.orientationYprDegrees = camera_reference::Vector3d{{
        source.yawDegrees, source.pitchDegrees, source.rollDegrees}};
    result.timestamp = source.timeText;
    return result;
}

} // namespace

camera_reference::CameraReferenceSet buildMetashapeCameraReferenceSet(
    const reference_import::MetashapeCameraReferenceImportResult &imported,
    const QJsonObject &metadata,
    const QString &cameraPath,
    const QString &offsetPath,
    const QString &contentHash)
{
    camera_reference::CameraReferenceSet result;
    camera_reference::CameraReferenceSource source;
    source.kind = QStringLiteral("metashape_camera_reference_txt");
    source.displayName = portableFileName(cameraPath);
    source.contentSha256 = contentHash;
    source.sourceCrs = QStringLiteral("EPSG:4979");
    source.axisOrder = QStringLiteral("longitude_latitude");
    source.verticalDatum = QStringLiteral("ellipsoidal");
    source.verticalUnit = QStringLiteral("m");
    source.orientationConvention =
        QStringLiteral("metashape_aerial_yaw_z_pitch_x_roll_y_unresolved");
    source.angleUnit = QStringLiteral("deg");
    result.replaceSource(source);
    result.setImageSetFingerprint(cameraReferenceImageSetFingerprint(metadata));

    QString leverArmId;
    if (imported.leverArm)
    {
        leverArmId = QStringLiteral("default_gnss_offset");
        camera_reference::CameraReferenceLeverArm leverArm;
        leverArm.id = leverArmId;
        leverArm.vectorMeters = {{imported.leverArm->xMeters,
                                  imported.leverArm->yMeters,
                                  imported.leverArm->zMeters}};
        leverArm.vectorFrame = QStringLiteral("metashape_axes_unresolved");
        leverArm.vectorDirection = QStringLiteral("source_definition_unresolved");
        leverArm.source = offsetPath;
        result.addLeverArm(leverArm);
    }

    const QHash<QString, QVector<ImageIdentity>> projectImages = projectImagesByName(metadata);
    for (const reference_import::RawCameraReferenceRecord &sourceRecord : imported.records)
    {
        const camera_reference::RawCameraReference raw = rawReference(sourceRecord);
        const QVector<ImageIdentity> matches = projectImages.value(
            portableFileName(sourceRecord.fileName).toCaseFolded());
        if (matches.size() != 1)
        {
            camera_reference::UnmatchedCameraReferenceRecord unmatched;
            unmatched.sourceLabel = sourceRecord.fileName;
            unmatched.leverArmId = leverArmId;
            unmatched.raw = raw;
            unmatched.reason = matches.isEmpty()
                ? QStringLiteral("项目中没有同名影像")
                : QStringLiteral("项目中存在多个同名影像，无法唯一绑定");
            result.addUnmatchedRecord(unmatched);
            continue;
        }

        camera_reference::CameraReferenceRecord record;
        record.imageUuid = matches.front().uuid;
        record.imagePathSnapshot = matches.front().path;
        record.sourceLabel = sourceRecord.fileName;
        record.leverArmId = leverArmId;
        record.raw = raw;
        record.resolved.error = QStringLiteral(
            "待配置求解坐标系、姿态约定与 GNSS 杆臂方向；当前仅保留源观测");
        result.addRecord(record);
    }
    return result;
}

} // namespace xjw::gui::reference
