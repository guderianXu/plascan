#include "ProjectMarkerRepository.h"

#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"
#include "ProjectSurveyControl.h"

#include <QJsonObject>

namespace xjw::gui::markers
{

ProjectMarkerRepository::ProjectMarkerRepository(ProjectData *projectData, QObject *parent)
    : QObject(parent)
    , _projectData(projectData)
{
}

bool ProjectMarkerRepository::open(QString *error)
{
    if (!_projectData || !_projectData->hasProject())
    {
        if (error) *error = QStringLiteral("没有打开的工程，无法加载标记点");
        return false;
    }

    const control_points::MarkerSetIoResult result =
        control_points::MarkerSetStore(sidecarPath()).load();
    if (!result.ok)
    {
        if (error) *error = result.error;
        return false;
    }
    _markerSet = result.markerSet;
    ++_revision;
    emit markerSetChanged(_revision, {});
    return true;
}

bool ProjectMarkerRepository::save(QString *error)
{
    if (!_projectData || !_projectData->hasProject())
    {
        if (error) *error = QStringLiteral("没有打开的工程，无法保存标记点");
        return false;
    }

    const control_points::MarkerSetIoResult result =
        control_points::MarkerSetStore(sidecarPath()).save(_markerSet);
    if (!result.ok)
    {
        if (error) *error = result.error;
        return false;
    }

    QJsonObject metadata = _projectData->coreFilesMeta();
    int control_count = 0;
    int check_count = 0;
    for (const control_points::Marker &marker : _markerSet.markers())
    {
        if (marker.role == control_points::MarkerRole::ControlPoint) ++control_count;
        else if (marker.role == control_points::MarkerRole::CheckPoint) ++check_count;
    }
    metadata[QStringLiteral("marker_set")] = QJsonObject{
        {QStringLiteral("path"), QStringLiteral("assets/control_points/marker_set.json")},
        {QStringLiteral("schema_version"), _markerSet.schemaVersion()},
        {QStringLiteral("marker_count"), _markerSet.markers().size()},
        {QStringLiteral("control_point_count"), control_count},
        {QStringLiteral("check_point_count"), check_count},
        {QStringLiteral("scale_bar_count"), _markerSet.scaleBars().size()},
        {QStringLiteral("updated_at"), _markerSet.updatedAt().toString(Qt::ISODateWithMs)}
    };
    _projectData->updateMetadata(metadata, true);
    emit markerSetChanged(_revision, {});
    return true;
}

void ProjectMarkerRepository::reset()
{
    _markerSet = control_points::MarkerSet();
    ++_revision;
    emit markerSetChanged(_revision, {});
}

const control_points::MarkerSet &ProjectMarkerRepository::markerSet() const noexcept
{
    return _markerSet;
}

control_points::MarkerSet &ProjectMarkerRepository::markerSet() noexcept
{
    return _markerSet;
}

QString ProjectMarkerRepository::sidecarPath() const
{
    return _projectData ? xjw::common::project::ProjectIO::markerSetPath(_projectData->currentProjectPath()) : QString();
}

quint64 ProjectMarkerRepository::revision() const noexcept
{
    return _revision;
}

void ProjectMarkerRepository::applyChangeSet(const control_points::MarkerChangeSet &changeSet)
{
    changeSet.apply(&_markerSet);
    ++_revision;
    emit markerSetChanged(_revision, changeSet.affectedMarkerIds());
}

void ProjectMarkerRepository::revertChangeSet(const control_points::MarkerChangeSet &changeSet)
{
    changeSet.revert(&_markerSet);
    ++_revision;
    emit markerSetChanged(_revision, changeSet.affectedMarkerIds());
}

} // namespace xjw::gui::markers
