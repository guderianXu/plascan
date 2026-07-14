#pragma once

#include "io/MarkerSetStore.h"
#include "commands/MarkerChangeSet.h"

#include <QObject>
#include <QString>

class ProjectData;

namespace xjw::gui::markers
{

class ProjectMarkerRepository final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectMarkerRepository(ProjectData *projectData, QObject *parent = nullptr);

    bool open(QString *error = nullptr);
    bool save(QString *error = nullptr);
    void reset();

    const control_points::MarkerSet &markerSet() const noexcept;
    control_points::MarkerSet &markerSet() noexcept;
    QString sidecarPath() const;
    quint64 revision() const noexcept;

    void applyChangeSet(const control_points::MarkerChangeSet &changeSet);
    void revertChangeSet(const control_points::MarkerChangeSet &changeSet);

signals:
    void markerSetChanged(quint64 revision, const QVector<QString> &affectedMarkerIds);

private:
    ProjectData *_projectData = nullptr;
    control_points::MarkerSet _markerSet;
    quint64 _revision = 0;
};

} // namespace xjw::gui::markers
