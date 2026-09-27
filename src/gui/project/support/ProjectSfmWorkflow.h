#pragma once

#include <placamera/instance_set.h>

#include <QMap>
#include <QJsonObject>
#include <QSet>
#include <QString>
#include <QStringList>

namespace xjw::aerial_triangulation
{
    struct AerialTriangulationReconstructionResult;
}

namespace xjw::gui::project
{

    struct InitPoseFinalizeResult
    {
        placamera::CameraInstanceSet cameraInstances;
        QMap<QString, QJsonObject> cameraAnnotationsByImageId;
        QString errorMessage;
        QString sparseCloudPath;
        int sparsePointCount = 0;
        QStringList selectedImages;
        QString outputDir;
        QJsonObject resultRecordExtra;
    };

    InitPoseFinalizeResult
    finalizeInitializedCameraPoses(const xjw::aerial_triangulation::AerialTriangulationReconstructionResult& result,
                                   const QSet<QString>& targetImages,
                                   const QSet<QString>& existingImages,
                                   bool overwriteExisting,
                                   const QJsonObject& projectMetadata,
                                   const QStringList& allImages,
                                   const QString& outputDir);

} // namespace xjw::gui::project
