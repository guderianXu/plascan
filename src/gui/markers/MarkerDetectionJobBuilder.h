#pragma once

#include "MarkerTaskRunner.h"

#include <QStringList>

class ProjectData;

namespace xjw::gui::markers
{

struct MarkerDetectionJobBuildOptions
{
    quint64 baseRevision = 0;
    QVector<control_points::MarkerTargetFamily> targetFamilies;
    control_points::MarkerDetectionOptions detectorOptions;
    int maxConcurrentImages = 0;
};

struct MarkerDetectionJobBuildResult
{
    bool ok = false;
    MarkerDetectionJob job;
    QStringList errors;
};

class MarkerDetectionJobBuilder final
{
public:
    static MarkerDetectionJobBuildResult build(
        const ProjectData &projectData,
        const MarkerDetectionJobBuildOptions &options);
};

} // namespace xjw::gui::markers
