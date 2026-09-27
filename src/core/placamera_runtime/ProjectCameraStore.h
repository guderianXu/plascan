#pragma once

#include <placamera/instance_set.h>

#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <vector>

namespace xjw::placamera_runtime
{

    struct ProjectCameraLoadResult
    {
        placamera::CameraInstanceSet instances;
        QStringList errors;

        bool ok() const noexcept
        {
            return errors.isEmpty();
        }
    };

    struct ProjectCameraWriteResult
    {
        int insertedCount = 0;
        int updatedCount = 0;
        int clearedCount = 0;
        QStringList errors;

        bool ok() const noexcept
        {
            return errors.isEmpty();
        }
    };

    ProjectCameraLoadResult loadProjectCameras(const QJsonObject& projectFiles);

    /** Atomically insert native frame, RPC or line-scan cameras for images without a camera binding. */
    ProjectCameraWriteResult insertProjectCameras(QJsonObject* projectFiles,
                                                  const placamera::CameraInstanceSet& instances);

    /** Insert new native cameras and update bound ones as one project transaction. */
    ProjectCameraWriteResult upsertProjectCameras(QJsonObject* projectFiles,
                                                  const placamera::CameraInstanceSet& instances,
                                                  const QMap<QString, QJsonObject>& annotationsByImageId = {});

    /** Replace bindings for selected images, clearing those without a new native instance. */
    ProjectCameraWriteResult replaceProjectCameras(QJsonObject* projectFiles,
                                                   const std::vector<placamera::ImageId>& targetImageIds,
                                                   const placamera::CameraInstanceSet& instances,
                                                   const QMap<QString, QJsonObject>& annotationsByImageId = {});

    /** Atomically write selected native instances without changing image size. Refined calibration needs a new ID. */
    ProjectCameraWriteResult writeProjectCameras(QJsonObject* projectFiles,
                                                 const placamera::CameraInstanceSet& instances);

} // namespace xjw::placamera_runtime
