#include "ProjectSfmWorkflow.h"

#include "project/ProjectMetadata.h"
#include "model/AerialTriangulationResult.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMap>

namespace xjw::gui::project
{

    namespace
    {

        placamera::CameraInstanceSet filterSfmCameraInstances(const placamera::CameraInstanceSet& instances,
                                                              const QSet<QString>& targetImages,
                                                              const QSet<QString>& existingImages,
                                                              bool overwriteExisting,
                                                              const QJsonObject& projectMetadata,
                                                              QString* errorMessage)
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

            placamera::CameraInstanceSet filtered;
            for (const auto& camera : instances.values())
            {
                const QString imageId = QString::fromStdString(camera->imageId().value()).trimmed();
                if (!targetImageIds.contains(imageId))
                {
                    continue;
                }
                if (!overwriteExisting && existingImageIds.contains(imageId))
                {
                    continue;
                }
                const auto added = filtered.add(camera);
                if (!added.ok())
                {
                    if (errorMessage)
                    {
                        *errorMessage =
                            QStringLiteral("SFM 相机筛选失败: %1").arg(QString::fromStdString(added.message()));
                    }
                    return {};
                }
            }
            return filtered;
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
        finalizeResult.cameraInstances = filterSfmCameraInstances(result.cameraInstances,
                                                                  targetImages,
                                                                  existingImages,
                                                                  overwriteExisting,
                                                                  projectMetadata,
                                                                  &finalizeResult.errorMessage);
        for (const auto& camera : finalizeResult.cameraInstances.values())
        {
            const QString image_id = QString::fromStdString(camera->imageId().value());
            const auto annotation = result.cameraAnnotationsByImageId.constFind(image_id);
            if (annotation != result.cameraAnnotationsByImageId.constEnd())
            {
                finalizeResult.cameraAnnotationsByImageId.insert(image_id, annotation.value());
            }
        }
        finalizeResult.sparseCloudPath = result.sparseCloudPath;
        finalizeResult.sparsePointCount = result.numPoints3D;
        finalizeResult.selectedImages = allImages;
        finalizeResult.outputDir = outputDir;
        finalizeResult.resultRecordExtra = result.resultRecordExtra;
        return finalizeResult;
    }

} // namespace xjw::gui::project
