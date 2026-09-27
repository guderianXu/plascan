#pragma once

#include "ProjectWorkflowReports.h"

#include <placamera/instance_set.h>

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project {

struct BundleAdjustCommitResult
{
    bool success = false;
    int updatedCameraCount = 0;
    QString errorMessage;
    QString warningMessage;
};

struct BundleAdjustArtifactsResult
{
    bool reportSaved = false;
    QString reportWarning;
    BundleAdjustSparseCloudExport sparseCloudExport;
};

struct BundleAdjustPreviewPresentation
{
    QString summaryText;
    QString detailedText;
    bool qualityWarning = false;
};

BundleAdjustPreviewPresentation buildBundleAdjustPreviewPresentation(
    const QJsonObject &ba_result,
    int pending_camera_count);

BundleAdjustCommitResult commitBundleAdjustPreview(
    ProjectData* projectData,
    const placamera::CameraInstanceSet& cameraInstances,
    const QJsonObject& baResult,
    const QMap<QString, QJsonObject>& annotationsByImageId = {});

BundleAdjustArtifactsResult finalizeBundleAdjustArtifacts(const QString &assetsDir,
                                                          const QJsonObject &baResult,
                                                          const QStringList &images,
                                                          const QString &reportOutputDir,
                                                          const QString &reportSource,
                                                          const QMap<QString, QJsonObject> &beforeCameras,
                                                          const QMap<QString, QJsonObject> &afterCameras,
                                                          const QString &sparseCloudOutputDir,
                                                          bool useDedicatedFileName);

} // namespace xjw::gui::project
