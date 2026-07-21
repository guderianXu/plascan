#pragma once

#include <QMap>
#include <QJsonObject>
#include <QSet>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::aerial_triangulation {
struct AerialTriangulationReconstructionResult;
}

namespace xjw::gui::project {

struct InitPoseFinalizeResult
{
    bool success = false;
    int updatedCameraCount = 0;
    QString errorMessage;
};

InitPoseFinalizeResult finalizeInitializedCameraPoses(ProjectData *projectData,
                                                     const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result,
                                                     const QSet<QString> &targetImages,
                                                     const QSet<QString> &existingImages,
                                                     bool overwriteExisting,
                                                     const QStringList &allImages,
                                                     const QString &outputDir);

} // namespace xjw::gui::project
