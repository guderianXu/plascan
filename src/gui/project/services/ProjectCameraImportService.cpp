#include "ProjectCameraImportService.h"

#include "io/ImageIO.h"
#include "io/PathIO.h"
#include "placamera_runtime/ProjectCameraStore.h"
#include "project/ProjectIO.h"

#include <placamera/tsai.h>

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QUuid>

#include <exception>
#include <memory>
#include <utility>

namespace xjw::gui::project
{
    namespace
    {
        bool isCancelled(const std::atomic<bool>* cancelFlag)
        {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        }

        QString absolutePath(const QString& path)
        {
            return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        }

        bool readImageSize(const QString& imagePath, placamera::ImageSize* size, QString* error)
        {
            QString image_error;
            const QSize image_size = xjw::common::io::readImageSize(imagePath, &image_error);
            if (!image_size.isValid())
            {
                if (error)
                {
                    *error = image_error;
                }
                return false;
            }
            *size = {image_size.width(), image_size.height()};
            return true;
        }

        bool parseTsaiImport(const QString& imagePath, const QString& tsaiPath, PreparedFrameCameraImport* output)
        {
            output->imageAbsPath = absolutePath(imagePath);
            output->sourceFile = absolutePath(tsaiPath);
            output->sourceFormat = QStringLiteral("tsai");
            output->sourceKind = QStringLiteral("tsai_import");
            output->camera.reset();
            output->error.clear();
            if (!readImageSize(output->imageAbsPath, &output->imageSize, &output->error))
            {
                return false;
            }

            const auto camera = placamera::loadTsaiFramePinhole(
                xjw::common::io::toUtf8Path(output->sourceFile),
                placamera::CameraDefinitionId("tsai-import-" +
                                              QUuid::createUuid().toString(QUuid::WithoutBraces).toStdString()),
                placamera::FrameId("project-world"));
            if (!camera)
            {
                output->error = QStringLiteral("无法解析相机文件: %1 (%2)")
                                    .arg(output->sourceFile, QString::fromStdString(camera.message()));
                return false;
            }
            output->camera = PreparedFrameCamera{camera.value().definition, camera.value().pose};
            return true;
        }
    } // namespace

    SingleCameraImportStatus buildSingleCameraImport(const QString& imagePath,
                                                     const QString& tsaiPath,
                                                     PreparedFrameCameraImport* out,
                                                     const std::atomic<bool>* cancelFlag,
                                                     const CameraImportProgress& progress)
    {
        if (!out)
        {
            return SingleCameraImportStatus::ParseFailed;
        }
        *out = {};
        if (progress)
        {
            progress(0, 1);
        }
        if (isCancelled(cancelFlag))
        {
            return SingleCameraImportStatus::Cancelled;
        }
        if (imagePath.trimmed().isEmpty())
        {
            out->error = QStringLiteral("请选择有效的影像");
            return SingleCameraImportStatus::EmptyImagePath;
        }
        if (!parseTsaiImport(imagePath, tsaiPath, out))
        {
            return SingleCameraImportStatus::ParseFailed;
        }
        if (isCancelled(cancelFlag))
        {
            return SingleCameraImportStatus::Cancelled;
        }
        if (progress)
        {
            progress(1, 1);
        }
        return SingleCameraImportStatus::Ok;
    }

    bool bindImportedFrameCameras(const QJsonObject& projectCore,
                                  const QString& projectPath,
                                  const std::vector<PreparedFrameCameraImport>& imports,
                                  placamera::CameraInstanceSet* cameras,
                                  QMap<QString, QJsonObject>* annotationsByImageId,
                                  QString* error)
    {
        if (!cameras || !annotationsByImageId || imports.empty())
        {
            if (error)
            {
                *error = QStringLiteral("没有可绑定的相机结果");
            }
            return false;
        }
        const auto current = xjw::placamera_runtime::loadProjectCameras(projectCore);
        if (!current.ok())
        {
            if (error)
            {
                *error = current.errors.join(QStringLiteral("; "));
            }
            return false;
        }

        QMap<QString, QString> image_ids_by_path;
        for (const QJsonValue& value : projectCore.value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = value.toObject();
            const QString stored_path = image.value(QStringLiteral("path")).toString().trimmed();
            const QString path = stored_path.isEmpty()
                                     ? QString()
                                     : QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(
                                           projectPath, stored_path));
            const QString image_id = image.value(QStringLiteral("image_uuid")).toString().trimmed();
            if (path.isEmpty() || image_id.isEmpty() || image_ids_by_path.contains(path))
            {
                if (error)
                {
                    *error = QStringLiteral("工程影像路径或身份无效: %1").arg(path);
                }
                return false;
            }
            image_ids_by_path.insert(path, image_id);
        }

        placamera::CameraInstanceSet pending;
        QMap<QString, QJsonObject> annotations;
        for (const PreparedFrameCameraImport& imported : imports)
        {
            const QString path = QDir::cleanPath(imported.imageAbsPath);
            const QString image_id = image_ids_by_path.value(path);
            if (image_id.isEmpty() || !imported.camera || !imported.imageSize.isValid())
            {
                if (error)
                {
                    *error = QStringLiteral("相机影像未在当前工程注册或解析结果无效: %1").arg(path);
                }
                return false;
            }
            try
            {
                const placamera::ImageId native_image_id(image_id.toStdString());
                const auto previous = current.instances.forImage(native_image_id);
                if (previous.ok() && (previous.value()->modelType() != imported.camera->definition->modelType() ||
                                      previous.value()->groundFrame() != imported.camera->definition->groundFrame()))
                {
                    if (error)
                    {
                        *error = QStringLiteral("导入相机与现有模型或地面坐标系不一致: %1").arg(path);
                    }
                    return false;
                }
                const placamera::CameraInstanceId instance_id(previous.ok() ? previous.value()->instanceId().value()
                                                                            : "caminst-" + image_id.toStdString());
                auto model =
                    placamera::bindCentralCamera(*imported.camera,
                                                 {instance_id,
                                                  native_image_id,
                                                  imported.imageSize,
                                                  previous.ok() ? previous.value()->captureTime() : std::nullopt,
                                                  imported.acquisition});
                if (!model)
                {
                    if (error)
                    {
                        *error = QStringLiteral("相机身份绑定失败: %1 (%2)")
                                     .arg(path, QString::fromStdString(model.message()));
                    }
                    return false;
                }
                const auto added = pending.add(model.takeValue());
                if (!added.ok())
                {
                    if (error)
                    {
                        *error = QStringLiteral("重复或冲突的相机影像: %1").arg(path);
                    }
                    return false;
                }
                annotations.insert(
                    image_id,
                    QJsonObject{{QStringLiteral("source"), imported.sourceKind},
                                {QStringLiteral("metadata"),
                                 QJsonObject{{QStringLiteral("source_file"), imported.sourceFile},
                                             {QStringLiteral("source_format"), imported.sourceFormat}}}});
            }
            catch (const std::exception& exception)
            {
                if (error)
                {
                    *error = QStringLiteral("相机身份绑定失败: %1 (%2)").arg(path, QString::fromUtf8(exception.what()));
                }
                return false;
            }
        }
        *cameras = std::move(pending);
        *annotationsByImageId = std::move(annotations);
        if (error)
        {
            error->clear();
        }
        return true;
    }

} // namespace xjw::gui::project
