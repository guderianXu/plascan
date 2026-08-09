#pragma once

#include "model/CameraReferenceSet.h"

#include <QObject>

class ProjectData;

namespace xjw::gui::reference
{

class ProjectCameraReferenceRepository final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectCameraReferenceRepository(ProjectData *projectData,
                                              QObject *parent = nullptr);

    bool open(QString *error = nullptr);
    void reset();

    const camera_reference::CameraReferenceSet &referenceSet() const noexcept;
    QString sidecarPath() const;
    quint64 revision() const noexcept;

    bool replaceReferenceSet(const camera_reference::CameraReferenceSet &referenceSet,
                             QString *error = nullptr);
    bool setRecordEnabled(const QString &imageUuid,
                          bool enabled,
                          QString *error = nullptr);

signals:
    void referenceSetChanged(quint64 revision);

private:
    bool save(QString *error);

    ProjectData *_projectData = nullptr;
    camera_reference::CameraReferenceSet _referenceSet;
    quint64 _revision = 0;
};

} // namespace xjw::gui::reference
