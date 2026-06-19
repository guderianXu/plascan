#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QVector>

namespace xjw::gui::project {

enum class ProjectDashboardStepState
{
    Missing,
    Ready,
    Complete,
    Warning
};

struct ProjectDashboardStep
{
    QString id;
    QString title;
    QString detail;
    ProjectDashboardStepState state = ProjectDashboardStepState::Missing;
};

struct ProjectDashboardSummary
{
    int imageCount = 0;
    int cameraCount = 0;
    int featureResultCount = 0;
    int matchResultCount = 0;
    int sparseResultCount = 0;
    int bundleAdjustResultCount = 0;
    int depthMapResultCount = 0;
    int denseCloudResultCount = 0;
    int modelResultCount = 0;
    int demResultCount = 0;
    int orthoResultCount = 0;
    int reportResultCount = 0;
    int qualityReportCount = 0;
    int referenceDatasetCount = 0;
    int lidarReferenceCount = 0;
    int pointCloudReferenceCount = 0;
    int demReferenceCount = 0;
    int baPriorReferenceCount = 0;
    int validationReferenceCount = 0;
    QVector<ProjectDashboardStep> workflowSteps;
    QJsonArray referenceDatasets;
    QJsonArray qualityReports;
};

QString projectDashboardStepStateName(ProjectDashboardStepState state);

bool projectDashboardStepById(const ProjectDashboardSummary &summary,
                              const QString &id,
                              ProjectDashboardStep *step);

ProjectDashboardSummary buildProjectDashboardSummary(const QJsonObject &metadata);

} // namespace xjw::gui::project
