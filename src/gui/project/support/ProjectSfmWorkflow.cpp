#include "ProjectSfmWorkflow.h"

#include "project/ProjectMetadata.h"
#include "model/AerialTriangulationResult.h"
#include "camera/project/CameraInstanceUpdate.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMap>

namespace xjw::gui::project
{

    namespace
    {

        xjw::camera_project::CameraInstanceUpdates
        filterSfmCameraUpdates(const xjw::camera_project::CameraInstanceUpdates& updates,
                               const QSet<QString>& targetImages,
                               const QSet<QString>& existingImages,
                               bool overwriteExisting,
                               const QJsonObject& projectMetadata)
        {
            QSet<QString> targetImageIds;
            QSet<QString> existingImageIds;
            for (const QJsonValue& value : xjw::common::project::projectImageEntries(projectMetadata))
            {
                const QJsonObject image = value.toObject();
                const QString imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
                const QString normalizedPath =
                    xjw::common::project::normalizePath(image.value(QStringLiteral("path")).toString());
                if (imageId.isEmpty() || normalizedPath.isEmpty())
                {
                    continue;
                }
                if (targetImages.contains(normalizedPath))
                {
                    targetImageIds.insert(imageId);
                }
                if (existingImages.contains(normalizedPath))
                {
                    existingImageIds.insert(imageId);
                }
            }

            xjw::camera_project::CameraInstanceUpdates filteredUpdates;
            for (const xjw::camera_project::CameraInstanceUpdate& update : updates)
            {
                const QString imageId = QString::fromStdString(update.imageId.value()).trimmed();
                if (!targetImageIds.contains(imageId))
                {
                    continue;
                }
                if (!overwriteExisting && existingImageIds.contains(imageId))
                {
                    continue;
                }
                filteredUpdates.push_back(update);
            }
            return filteredUpdates;
        }

    } // namespace

    InitPoseFinalizeResult
    finalizeInitializedCameraPoses(const xjw::aerial_triangulation::AerialTriangulationReconstructionResult& result,
                                   const QSet<QString>& targetImages,
                                   const QSet<QString>& existingImages,
                                   bool overwriteExisting,
                                   const QJsonObject& projectMetadata,
                                   const QStringList& allImages,
                                   const QString& outputDir)
    {
        InitPoseFinalizeResult finalizeResult;
        finalizeResult.cameraUpdates = filterSfmCameraUpdates(
            result.cameraInstanceUpdates, targetImages, existingImages, overwriteExisting, projectMetadata);
        finalizeResult.sparseCloudPath = result.sparseCloudPath;
        finalizeResult.sparsePointCount = result.numPoints3D;
        finalizeResult.selectedImages = allImages;
        finalizeResult.outputDir = outputDir;
        finalizeResult.resultRecordExtra = result.resultRecordExtra;
        return finalizeResult;
    }

} // namespace xjw::gui::project
