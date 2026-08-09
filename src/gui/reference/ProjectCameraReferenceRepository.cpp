#include "ProjectCameraReferenceRepository.h"

#include "CameraReferenceProjectIdentity.h"
#include "ProjectData.h"
#include "io/CameraReferenceSetStore.h"
#include "project/ProjectIO.h"

#include <QFile>
#include <QJsonObject>
#include <QSaveFile>

#include <algorithm>

namespace xjw::gui::reference
{
namespace
{

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

} // namespace

ProjectCameraReferenceRepository::ProjectCameraReferenceRepository(ProjectData *projectData,
                                                                   QObject *parent)
    : QObject(parent)
    , _projectData(projectData)
{
}

bool ProjectCameraReferenceRepository::open(QString *error)
{
    if (!_projectData || !_projectData->hasProject())
    {
        reset();
        if (error)
        {
            *error = QStringLiteral("没有打开的工程，无法加载相机参考");
        }
        return false;
    }

    const camera_reference::CameraReferenceSetIoResult result =
        camera_reference::CameraReferenceSetStore(sidecarPath()).load();
    if (!result.ok)
    {
        reset();
        if (error)
        {
            *error = result.error;
        }
        return false;
    }
    const QString storedFingerprint = result.referenceSet.imageSetFingerprint();
    const QString currentFingerprint = cameraReferenceImageSetFingerprint(
        _projectData->metadata());
    if (!storedFingerprint.isEmpty() && storedFingerprint != currentFingerprint)
    {
        reset();
        if (error)
        {
            *error = QStringLiteral(
                "项目影像集合已变化，相机参考绑定已停用；请重新导入相机参考文件");
        }
        return false;
    }
    _referenceSet = result.referenceSet;
    ++_revision;
    emit referenceSetChanged(_revision);
    return true;
}

void ProjectCameraReferenceRepository::reset()
{
    _referenceSet = camera_reference::CameraReferenceSet();
    ++_revision;
    emit referenceSetChanged(_revision);
}

const camera_reference::CameraReferenceSet &
ProjectCameraReferenceRepository::referenceSet() const noexcept
{
    return _referenceSet;
}

QString ProjectCameraReferenceRepository::sidecarPath() const
{
    return _projectData
        ? xjw::common::project::ProjectIO::cameraReferenceSetPath(
              _projectData->currentProjectPath())
        : QString();
}

quint64 ProjectCameraReferenceRepository::revision() const noexcept
{
    return _revision;
}

bool ProjectCameraReferenceRepository::replaceReferenceSet(
    const camera_reference::CameraReferenceSet &referenceSet,
    QString *error)
{
    QString validationError;
    if (!referenceSet.validate(&validationError))
    {
        if (error)
        {
            *error = validationError;
        }
        return false;
    }

    const camera_reference::CameraReferenceSet previous = _referenceSet;
    _referenceSet = referenceSet;
    if (!save(error))
    {
        _referenceSet = previous;
        return false;
    }
    ++_revision;
    emit referenceSetChanged(_revision);
    return true;
}

bool ProjectCameraReferenceRepository::setRecordEnabled(const QString &imageUuid,
                                                        bool enabled,
                                                        QString *error)
{
    const auto iterator = std::find_if(
        _referenceSet.records().cbegin(),
        _referenceSet.records().cend(),
        [&imageUuid](const camera_reference::CameraReferenceRecord &record)
    {
        return record.imageUuid == imageUuid;
    });
    if (iterator == _referenceSet.records().cend())
    {
        if (error)
        {
            *error = QStringLiteral("未找到相机参考记录: %1").arg(imageUuid);
        }
        return false;
    }

    camera_reference::CameraReferenceSet updated = _referenceSet;
    camera_reference::CameraReferenceRecord record = *iterator;
    record.enabled = enabled;
    try
    {
        updated.replaceRecord(record);
    }
    catch (const std::exception &exception)
    {
        if (error)
        {
            *error = QString::fromUtf8(exception.what());
        }
        return false;
    }
    return replaceReferenceSet(updated, error);
}

bool ProjectCameraReferenceRepository::save(QString *error)
{
    if (!_projectData || !_projectData->hasProject())
    {
        if (error)
        {
            *error = QStringLiteral("没有打开的工程，无法保存相机参考");
        }
        return false;
    }

    const camera_reference::CameraReferenceSetStore store(sidecarPath());
    const QString path = sidecarPath();
    const bool sidecarExisted = QFile::exists(path);
    QByteArray previousBytes;
    if (sidecarExisted)
    {
        QFile previous(path);
        if (!previous.open(QIODevice::ReadOnly))
        {
            if (error)
            {
                *error = QStringLiteral("无法读取现有相机参考 sidecar，已取消保存: %1")
                    .arg(previous.errorString());
            }
            return false;
        }
        previousBytes = previous.readAll();
    }
    const camera_reference::CameraReferenceSetIoResult saved = store.save(_referenceSet);
    if (!saved.ok)
    {
        if (error)
        {
            *error = saved.error;
        }
        return false;
    }
    const camera_reference::CameraReferenceSetIoResult verified = store.load();
    if (!verified.ok || !(verified.referenceSet == _referenceSet))
    {
        const bool restored = restoreSidecar(path, sidecarExisted, previousBytes);
        if (error)
        {
            *error = QStringLiteral("相机参考 sidecar 写后校验失败: %1；%2")
                .arg(verified.error,
                     restored ? QStringLiteral("已恢复旧文件")
                              : QStringLiteral("旧文件恢复失败"));
        }
        return false;
    }

    int rawPositionCount = 0;
    int rawOrientationCount = 0;
    int usablePositionCount = 0;
    int usableOrientationCount = 0;
    for (const camera_reference::CameraReferenceRecord &record : _referenceSet.records())
    {
        rawPositionCount += record.raw.position.has_value() ? 1 : 0;
        rawOrientationCount += record.raw.orientationYprDegrees.has_value() ? 1 : 0;
        usablePositionCount += record.resolved.positionUsable ? 1 : 0;
        usableOrientationCount += record.resolved.orientationUsable ? 1 : 0;
    }
    for (const camera_reference::UnmatchedCameraReferenceRecord &record
         : _referenceSet.unmatchedRecords())
    {
        rawPositionCount += record.raw.position.has_value() ? 1 : 0;
        rawOrientationCount += record.raw.orientationYprDegrees.has_value() ? 1 : 0;
    }

    QJsonObject metadata = _projectData->coreFilesMeta();
    metadata[QStringLiteral("camera_reference_set")] = QJsonObject{
        {QStringLiteral("path"),
         QStringLiteral("assets/camera_references/camera_reference_set.json")},
        {QStringLiteral("schema_version"), _referenceSet.schemaVersion()},
        {QStringLiteral("record_count"), _referenceSet.records().size()},
        {QStringLiteral("unmatched_record_count"),
         _referenceSet.unmatchedRecords().size()},
        {QStringLiteral("raw_position_count"), rawPositionCount},
        {QStringLiteral("raw_orientation_count"), rawOrientationCount},
        {QStringLiteral("position_usable_count"), usablePositionCount},
        {QStringLiteral("orientation_usable_count"), usableOrientationCount},
        {QStringLiteral("source_kind"), _referenceSet.source().kind},
        {QStringLiteral("updated_at"),
         _referenceSet.updatedAt().toString(Qt::ISODateWithMs)}
    };
    _projectData->updateMetadata(metadata, true);
    return true;
}

} // namespace xjw::gui::reference
