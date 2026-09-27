#include "ProjectCameraInitialization.h"

#include "placamera_runtime/ProjectCameraStore.h"
#include "project/ProjectMetadata.h"

#include <placamera/frame_camera.h>

#include <QImageReader>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QUuid>

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>

namespace xjw::gui::project
{

    namespace
    {

        std::optional<double>
        readerExifValue(QImageReader& reader, const QStringList& preferredTokens, bool exclude35mm = false)
        {
            const QStringList keys = reader.textKeys();
            for (const QString& key : keys)
            {
                const QString lowerKey = key.toLower();
                if (exclude35mm && lowerKey.contains(QStringLiteral("35mm")))
                {
                    continue;
                }

                bool matched = false;
                for (const QString& token : preferredTokens)
                {
                    if (lowerKey.contains(token.toLower()))
                    {
                        matched = true;
                        break;
                    }
                }
                if (!matched)
                {
                    continue;
                }

                if (const auto value = parsePossiblyFractionalNumber(reader.text(key));
                    value.has_value() && *value > 0.0)
                {
                    return value;
                }
            }
            return std::nullopt;
        }

    } // namespace

    std::optional<double> parsePossiblyFractionalNumber(const QString& text)
    {
        const QString trimmed = text.trimmed();
        if (trimmed.isEmpty())
        {
            return std::nullopt;
        }

        if (trimmed.contains('/'))
        {
            const QStringList parts = trimmed.split('/', Qt::SkipEmptyParts);
            if (parts.size() == 2)
            {
                bool ok0 = false;
                bool ok1 = false;
                const double num = parts[0].trimmed().toDouble(&ok0);
                const double den = parts[1].trimmed().toDouble(&ok1);
                if (ok0 && ok1 && std::abs(den) > 1e-12)
                {
                    return num / den;
                }
            }
        }

        static const QRegularExpression re(QStringLiteral(R"(([+-]?\d+(?:\.\d+)?))"));
        const QRegularExpressionMatch match = re.match(trimmed);
        if (!match.hasMatch())
        {
            return std::nullopt;
        }

        bool ok = false;
        const double value = match.captured(1).toDouble(&ok);
        if (!ok)
        {
            return std::nullopt;
        }
        return value;
    }

    std::optional<double>
    focalPixelsFromExif(const QString& imagePath, const QSize& size, double sensorWidthMm, QString* sourceTag)
    {
        QImageReader reader(imagePath);
        if (const auto focal35 =
                readerExifValue(reader, {QStringLiteral("focallengthin35mmfilm"), QStringLiteral("35mm")});
            focal35.has_value())
        {
            const double diagPx = std::hypot((double)size.width(), (double)size.height());
            if (sourceTag)
            {
                *sourceTag = QStringLiteral("exif_35mm");
            }
            return *focal35 / 43.27 * diagPx;
        }

        if (const auto focalMm = readerExifValue(reader, {QStringLiteral("focallength")}, true);
            focalMm.has_value() && sensorWidthMm > 1e-9)
        {
            if (sourceTag)
            {
                *sourceTag = QStringLiteral("exif_mm");
            }
            return *focalMm / sensorWidthMm * size.width();
        }

        return std::nullopt;
    }

    QStringList resolveInitTargets(const QStringList& allImages, const QJsonObject& settings, QString* errorMsg)
    {
        if (allImages.isEmpty())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("当前项目没有影像");
            }
            return {};
        }

        const int applyScope = settings.value(QStringLiteral("applyScope")).toInt(0);
        if (applyScope == 0)
        {
            return allImages;
        }

        const QString targetImagePath = settings.value(QStringLiteral("applyTargetImagePath")).toString();
        if (targetImagePath.isEmpty())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("请先选择目标影像");
            }
            return {};
        }
        return {targetImagePath};
    }

    bool withPreparedCameras(const QJsonObject& baseMeta,
                             const CameraInitializationResult& prepared,
                             bool overwriteExisting,
                             QJsonObject* output,
                             QString* error)
    {
        if (!output)
        {
            if (error)
            {
                *error = QStringLiteral("相机初始化输出文档为空");
            }
            return false;
        }
        const QJsonObject project_files = xjw::common::project::projectFilesRootObject(baseMeta);
        const auto current = xjw::placamera_runtime::loadProjectCameras(project_files);
        if (!current.ok())
        {
            if (error)
            {
                *error = current.errors.join(QStringLiteral("; "));
            }
            return false;
        }
        placamera::CameraInstanceSet selected;
        QMap<QString, QJsonObject> annotations;
        for (const auto& model : prepared.cameras.values())
        {
            if (!overwriteExisting && current.instances.forImage(model->imageId()).ok())
            {
                continue;
            }
            const auto added = selected.add(model);
            if (!added)
            {
                if (error)
                {
                    *error = QString::fromStdString(added.message());
                }
                return false;
            }
            const QString image_id = QString::fromStdString(model->imageId().value());
            annotations.insert(image_id, prepared.annotationsByImageId.value(image_id));
        }
        QJsonObject candidate = project_files;
        if (!selected.empty())
        {
            const auto written = xjw::placamera_runtime::upsertProjectCameras(&candidate, selected, annotations);
            if (!written.ok())
            {
                if (error)
                {
                    *error = written.errors.join(QStringLiteral("; "));
                }
                return false;
            }
        }
        if (baseMeta.value(QStringLiteral("project_files")).isObject())
        {
            QJsonObject wrapped = baseMeta;
            wrapped.insert(QStringLiteral("project_files"), candidate);
            *output = std::move(wrapped);
        }
        else
        {
            *output = std::move(candidate);
        }
        return true;
    }

    CameraInitializationResult prepareCameraInitializations(const CameraInitializationRequest& request,
                                                            const std::atomic<bool>* cancelFlag,
                                                            const CameraInitializationProgress& progress)
    {
        CameraInitializationResult result;

        const auto is_cancelled = [cancelFlag]() { return cancelFlag && cancelFlag->load(std::memory_order_relaxed); };
        if (is_cancelled())
        {
            result.cancelled = true;
            return result;
        }
        const auto current = xjw::placamera_runtime::loadProjectCameras(request.projectMetadata);
        if (!current.ok())
        {
            result.error = current.errors.join(QStringLiteral("; "));
            return result;
        }
        const auto images_by_path = xjw::common::project::projectImageMetaByPath(request.projectMetadata, true);
        for (auto it = images_by_path.constBegin(); it != images_by_path.constEnd(); ++it)
        {
            if (is_cancelled())
            {
                result.cancelled = true;
                return result;
            }
            const QString image_id = it.value().value(QStringLiteral("image_uuid")).toString();
            if (!image_id.isEmpty() && current.instances.forImage(placamera::ImageId(image_id.toStdString())).ok())
            {
                result.existingImages.insert(it.key());
            }
        }
        if (is_cancelled())
        {
            result.cancelled = true;
            return result;
        }
        const int total = request.images.size();
        if (progress)
        {
            progress(0, total);
        }

        const bool overwrite_existing = request.settings.value(QStringLiteral("overwriteExisting")).toBool(false);
        const bool exif_auto = request.settings.value(QStringLiteral("exifAuto")).toBool(true);
        const double default_focal_mm = request.settings.value(QStringLiteral("defaultFocal")).toDouble(50.0);
        const double sensor_width_mm = request.settings.value(QStringLiteral("sensorWidth")).toDouble(23.5);
        const double fx = request.settings.value(QStringLiteral("fx")).toDouble(0.0);
        const double fy = request.settings.value(QStringLiteral("fy")).toDouble(0.0);
        const double cx_input = request.settings.value(QStringLiteral("cx")).toDouble(-1.0);
        const double cy_input = request.settings.value(QStringLiteral("cy")).toDouble(-1.0);
        const QString distortion_model = request.settings.value(QStringLiteral("distortionModel"))
                                             .toString(QStringLiteral("Brown (k1, k2, p1, p2)"));

        if (request.mode == CameraInitializationMode::Intrinsics && (fx <= 0.0 || fy <= 0.0))
        {
            result.error = QStringLiteral("fx/fy 必须大于 0。");
            return result;
        }

        double k1 = 0.0;
        double k2 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
        if (distortion_model.contains(QStringLiteral("径向")))
        {
            k1 = request.settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = request.settings.value(QStringLiteral("k2")).toDouble(0.0);
        }
        else if (distortion_model.contains(QStringLiteral("Brown")))
        {
            k1 = request.settings.value(QStringLiteral("k1")).toDouble(0.0);
            k2 = request.settings.value(QStringLiteral("k2")).toDouble(0.0);
            p1 = request.settings.value(QStringLiteral("p1")).toDouble(0.0);
            p2 = request.settings.value(QStringLiteral("p2")).toDouble(0.0);
        }

        int completed = 0;
        for (const QString& raw_path : request.images)
        {
            if (is_cancelled())
            {
                result.cancelled = true;
                return result;
            }

            const QString image_path = QDir::cleanPath(QFileInfo(raw_path).absoluteFilePath());
            const QString image_id = images_by_path.value(image_path).value(QStringLiteral("image_uuid")).toString();
            if (image_id.isEmpty())
            {
                result.error = QStringLiteral("影像不在当前项目中: %1").arg(image_path);
                return result;
            }
            const auto previous = current.instances.forImage(placamera::ImageId(image_id.toStdString()));
            if (!overwrite_existing && result.existingImages.contains(image_path))
            {
                ++result.skippedExisting;
                if (progress)
                {
                    progress(++completed, total);
                }
                continue;
            }
            if (previous.ok() && previous.value()->modelType() != "frame_pinhole")
            {
                result.error = QStringLiteral("影像 %1 已绑定非帧式相机，不能用帧式内参覆盖").arg(image_path);
                return result;
            }

            QImageReader reader(image_path);
            const QSize size = reader.size();
            if (is_cancelled())
            {
                result.cancelled = true;
                return result;
            }
            if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
            {
                ++result.invalidSizeCount;
                if (progress)
                {
                    progress(++completed, total);
                }
                continue;
            }

            double focal_x = fx;
            double focal_y = fy;
            double principal_x = cx_input;
            double principal_y = cy_input;
            QJsonObject metadata{{QStringLiteral("imported_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                                 {QStringLiteral("yaw_deg"), 0.0},
                                 {QStringLiteral("pitch_deg"), 0.0},
                                 {QStringLiteral("roll_deg"), 0.0},
                                 {QStringLiteral("distortion_model"), distortion_model},
                                 {QStringLiteral("pose_initialized_as_identity"), true},
                                 {QStringLiteral("pose_note"), QStringLiteral("R=I, C=[0,0,0]")}};
            if (request.mode == CameraInitializationMode::ExifOrDefault)
            {
                QString focal_source = QStringLiteral("default_mm");
                double focal_px = default_focal_mm / std::max(1e-9, sensor_width_mm) * size.width();
                if (exif_auto)
                {
                    if (const auto exif_px = focalPixelsFromExif(image_path, size, sensor_width_mm, &focal_source);
                        exif_px.has_value())
                    {
                        focal_px = *exif_px;
                        ++result.exifCount;
                    }
                    else
                    {
                        ++result.fallbackCount;
                    }
                }
                else
                {
                    ++result.fallbackCount;
                }
                focal_x = focal_px;
                focal_y = focal_px;
                principal_x = size.width() * 0.5;
                principal_y = size.height() * 0.5;
                metadata[QStringLiteral("distortion_model")] = QStringLiteral("none");
                metadata[QStringLiteral("focal_source")] = focal_source;
                metadata[QStringLiteral("focal_px")] = focal_px;
                metadata[QStringLiteral("default_focal_mm")] = default_focal_mm;
                metadata[QStringLiteral("sensor_width_mm")] = sensor_width_mm;
            }
            else
            {
                principal_x = cx_input <= 0.0 ? size.width() * 0.5 : cx_input;
                principal_y = cy_input <= 0.0 ? size.height() * 0.5 : cy_input;
                if (cx_input <= 0.0 || cy_input <= 0.0)
                {
                    ++result.autoPrincipalPointCount;
                }
                metadata[QStringLiteral("focal_px")] = fx;
                metadata[QStringLiteral("focal_px_y")] = fy;
            }
            try
            {
                const placamera::FrameId frame =
                    previous.ok() ? previous.value()->groundFrame() : placamera::FrameId("project-world");
                const QString instance_id = previous.ok()
                                                ? QString::fromStdString(previous.value()->instanceId().value())
                                                : QStringLiteral("caminst-") + image_id;
                const QString definition_id =
                    QStringLiteral("camdef-init-") + QUuid::createUuid().toString(QUuid::WithoutBraces);
                const auto definition = placamera::FramePinholeDefinition::create(
                    placamera::CameraDefinitionId(definition_id.toStdString()),
                    {focal_x, focal_y, principal_x, principal_y, 1.0, 1, 1},
                    {request.mode == CameraInitializationMode::ExifOrDefault ? 0.0 : k1,
                     request.mode == CameraInitializationMode::ExifOrDefault ? 0.0 : k2,
                     0.0,
                     request.mode == CameraInitializationMode::ExifOrDefault ? 0.0 : p1,
                     request.mode == CameraInitializationMode::ExifOrDefault ? 0.0 : p2},
                    placamera::PixelConvention::PixelCenter,
                    frame);
                auto camera = std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                    placamera::CameraInstanceId(instance_id.toStdString()),
                    placamera::ImageId(image_id.toStdString()),
                    definition,
                    {size.width(), size.height()},
                    placamera::Pose::create(frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
                const auto added = result.cameras.add(std::move(camera));
                if (!added)
                {
                    result.error = QStringLiteral("影像 %1 的相机初值无效: %2")
                                       .arg(image_path, QString::fromStdString(added.message()));
                    return result;
                }
            }
            catch (const std::exception& exception)
            {
                result.error =
                    QStringLiteral("影像 %1 的相机初值无效: %2").arg(image_path, QString::fromUtf8(exception.what()));
                return result;
            }
            result.annotationsByImageId.insert(
                image_id,
                QJsonObject{{QStringLiteral("source"), request.source}, {QStringLiteral("metadata"), metadata}});
            if (progress)
            {
                progress(++completed, total);
            }
        }

        result.cancelled = is_cancelled();
        return result;
    }

} // namespace xjw::gui::project
