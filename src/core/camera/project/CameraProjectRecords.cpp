#include "CameraProjectRecords.h"

#include "CameraProjectValidation.h"
#include "camera/models/CameraModelFactories.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>

#include <cmath>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace xjw::camera_project
{
    namespace
    {

        QString normalizedPath(const QString& path)
        {
            const QString trimmed = path.trimmed();
            if (trimmed.isEmpty())
            {
                return {};
            }
            return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
        }

        std::optional<QString> modelType(const QJsonObject& metadata)
        {
            const QString model = metadata.value(QStringLiteral("model")).toString();
            if (model == QStringLiteral("frame_pinhole") || model == QStringLiteral("rpc00b") ||
                model == QStringLiteral("planetary_linescan"))
            {
                return model;
            }
            return std::nullopt;
        }

        QString frameForModel(const QJsonObject& metadata, const QString&)
        {
            return metadata.value(QStringLiteral("world_frame")).toString();
        }

        QJsonArray finiteArray(const QJsonValue& value, int expected)
        {
            const QJsonArray values = value.toArray();
            if (values.size() != expected)
            {
                return {};
            }
            for (const QJsonValue& item : values)
            {
                if (!item.isDouble() || !std::isfinite(item.toDouble()))
                {
                    return {};
                }
            }
            return values;
        }

        QJsonValue scaledValue(const QJsonValue& value, double scale)
        {
            if (!value.isDouble() || !std::isfinite(value.toDouble()) || !std::isfinite(scale))
            {
                return value;
            }
            return QJsonValue(value.toDouble() * scale);
        }

        bool validateInputContainers(const QJsonObject& metadata, const QString& model, QString* error)
        {
            const auto requireExactString = [&metadata, error](const QString& key, const QSet<QString>& allowed)
            {
                if (!metadata.contains(key))
                {
                    return true;
                }
                const QJsonValue value = metadata.value(key);
                if (!value.isString() || !allowed.contains(value.toString()))
                {
                    if (error)
                    {
                        *error = QStringLiteral("%1 must use the canonical value").arg(key);
                    }
                    return false;
                }
                return true;
            };
            const auto rejectFields = [&metadata, error](std::initializer_list<QString> keys)
            {
                for (const QString& key : keys)
                {
                    if (metadata.contains(key))
                    {
                        if (error)
                        {
                            *error = QStringLiteral("legacy or alternate camera field is not supported: %1").arg(key);
                        }
                        return false;
                    }
                }
                return true;
            };

            if (!metadata.value(QStringLiteral("world_frame")).isString() ||
                metadata.value(QStringLiteral("world_frame")).toString().isEmpty())
            {
                if (error)
                {
                    *error = QStringLiteral("world_frame is required and must be a non-empty string");
                }
                return false;
            }

            if (model == QStringLiteral("frame_pinhole") &&
                (!metadata.contains(QStringLiteral("intrinsics_unit")) ||
                 !requireExactString(QStringLiteral("intrinsics_unit"), QSet<QString>{QStringLiteral("mm")}) ||
                 !metadata.contains(QStringLiteral("camera_center_unit")) ||
                 !requireExactString(QStringLiteral("camera_center_unit"), QSet<QString>{QStringLiteral("m")}) ||
                 !metadata.contains(QStringLiteral("pixel_convention")) ||
                 !requireExactString(QStringLiteral("pixel_convention"), QSet<QString>{QStringLiteral("center")})))
            {
                if (error && error->isEmpty())
                {
                    *error = QStringLiteral(
                        "frame_pinhole requires intrinsics_unit=mm, camera_center_unit=m and pixel_convention=center");
                }
                return false;
            }
            if (model == QStringLiteral("frame_pinhole") && (!rejectFields({QStringLiteral("intrinsics"),
                                                                            QStringLiteral("distortion"),
                                                                            QStringLiteral("pixel_pitch"),
                                                                            QStringLiteral("pixel_pitch_mm"),
                                                                            QStringLiteral("fx"),
                                                                            QStringLiteral("fy"),
                                                                            QStringLiteral("cx"),
                                                                            QStringLiteral("cy"),
                                                                            QStringLiteral("fx_px"),
                                                                            QStringLiteral("fy_px"),
                                                                            QStringLiteral("cx_px"),
                                                                            QStringLiteral("cy_px"),
                                                                            QStringLiteral("u_axis_sign"),
                                                                            QStringLiteral("v_axis_sign"),
                                                                            QStringLiteral("radial_k1"),
                                                                            QStringLiteral("radial_k2"),
                                                                            QStringLiteral("radial_k3"),
                                                                            QStringLiteral("tangential_p1"),
                                                                            QStringLiteral("tangential_p2"),
                                                                            QStringLiteral("image_samples"),
                                                                            QStringLiteral("image_lines")})))
            {
                return false;
            }
            if (model == QStringLiteral("rpc00b"))
            {
                if (metadata.value(QStringLiteral("world_frame")).toString() != QStringLiteral("EPSG:4978"))
                {
                    if (error)
                    {
                        *error = QStringLiteral("rpc00b world_frame must be EPSG:4978");
                    }
                    return false;
                }
                if (!rejectFields({QStringLiteral("line_offset"),
                                   QStringLiteral("sample_offset"),
                                   QStringLiteral("latitude_offset"),
                                   QStringLiteral("longitude_offset"),
                                   QStringLiteral("height_offset"),
                                   QStringLiteral("sample_scale"),
                                   QStringLiteral("latitude_scale"),
                                   QStringLiteral("longitude_scale"),
                                   QStringLiteral("line_numerator"),
                                   QStringLiteral("line_denominator"),
                                   QStringLiteral("sample_numerator"),
                                   QStringLiteral("sample_denominator"),
                                   QStringLiteral("error_bias_m"),
                                   QStringLiteral("error_random_m"),
                                   QStringLiteral("image_width"),
                                   QStringLiteral("image_height")}) ||
                    !requireExactString(QStringLiteral("rpc_spec"), QSet<QString>{QStringLiteral("RPC00B")}) ||
                    !metadata.contains(QStringLiteral("rpc_spec")) ||
                    !requireExactString(QStringLiteral("ground_crs"), QSet<QString>{QStringLiteral("EPSG:4979")}) ||
                    !metadata.contains(QStringLiteral("ground_crs")) ||
                    !requireExactString(QStringLiteral("height_datum"),
                                        QSet<QString>{QStringLiteral("WGS84_ellipsoidal")}) ||
                    !metadata.contains(QStringLiteral("height_datum")) ||
                    !requireExactString(QStringLiteral("pixel_convention"),
                                        QSet<QString>{QStringLiteral("opencv_zero_based_center")}) ||
                    !metadata.contains(QStringLiteral("pixel_convention")))
                {
                    if (error && error->isEmpty())
                    {
                        *error = QStringLiteral("rpc00b requires canonical coordinate and pixel conventions");
                    }
                    return false;
                }
            }
            if (model == QStringLiteral("planetary_linescan"))
            {
                if (!rejectFields({QStringLiteral("focal_length_mm"),
                                   QStringLiteral("sample_pitch_mm"),
                                   QStringLiteral("principal_sample"),
                                   QStringLiteral("distortion_k1"),
                                   QStringLiteral("trajectory_samples"),
                                   QStringLiteral("line_rate"),
                                   QStringLiteral("start_time"),
                                   QStringLiteral("image_width"),
                                   QStringLiteral("image_height")}))
                {
                    return false;
                }
                for (const QString& key :
                     {QStringLiteral("optics"), QStringLiteral("trajectory"), QStringLiteral("line_timing")})
                {
                    if (!metadata.value(key).isObject())
                    {
                        if (error)
                        {
                            *error = QStringLiteral("%1 must be a required object").arg(key);
                        }
                        return false;
                    }
                }
                if (!requireExactString(QStringLiteral("pixel_convention"),
                                        QSet<QString>{QStringLiteral("pixel_center"), QStringLiteral("zero_based")}) ||
                    !metadata.contains(QStringLiteral("pixel_convention")))
                {
                    if (error && error->isEmpty())
                    {
                        *error = QStringLiteral("pixel_convention is required");
                    }
                    return false;
                }
            }
            return true;
        }

        QJsonValue canonicalFramePixelConvention(const QJsonObject& metadata)
        {
            return metadata.value(QStringLiteral("pixel_convention"));
        }

        QJsonValue canonicalLinePixelConvention(const QJsonObject& metadata)
        {
            return metadata.value(QStringLiteral("pixel_convention"));
        }

        QJsonObject instanceMetadata(const QJsonObject& metadata, const QString& model)
        {
            // Definition parameters are encoded below.  Keep importer- and
            // solver-specific values that describe this image in the instance
            // state instead of silently dropping them or making them part of a
            // shared definition fingerprint.
            QJsonObject extras = metadata.value(QStringLiteral("metadata")).toObject();
            for (auto it = metadata.constBegin(); it != metadata.constEnd(); ++it)
            {
                if (it.key() != QStringLiteral("metadata"))
                {
                    extras.insert(it.key(), it.value());
                }
            }

            const QSet<QString> commonKeys{QStringLiteral("model"),
                                           QStringLiteral("world_frame"),
                                           QStringLiteral("ground_crs"),
                                           QStringLiteral("height_datum"),
                                           QStringLiteral("C"),
                                           QStringLiteral("R"),
                                           QStringLiteral("camera_center_unit"),
                                           QStringLiteral("image_width"),
                                           QStringLiteral("image_height"),
                                           QStringLiteral("image_samples"),
                                           QStringLiteral("image_lines"),
                                           QStringLiteral("source"),
                                           QStringLiteral("source_file"),
                                           QStringLiteral("image_correction"),
                                           QStringLiteral("trajectory"),
                                           QStringLiteral("line_timing"),
                                           QStringLiteral("capture_time"),
                                           QStringLiteral("metadata")};
            for (const QString& key : commonKeys)
            {
                extras.remove(key);
            }

            const QSet<QString> framePinholeKeys{QStringLiteral("intrinsics_unit"),
                                                 QStringLiteral("pitch"),
                                                 QStringLiteral("fu"),
                                                 QStringLiteral("fv"),
                                                 QStringLiteral("cu"),
                                                 QStringLiteral("cv"),
                                                 QStringLiteral("u_direction"),
                                                 QStringLiteral("v_direction"),
                                                 QStringLiteral("k1"),
                                                 QStringLiteral("k2"),
                                                 QStringLiteral("k3"),
                                                 QStringLiteral("p1"),
                                                 QStringLiteral("p2"),
                                                 QStringLiteral("pixel_convention"),
                                                 QStringLiteral("depth_axis_flipped")};
            const QSet<QString> rpcKeys{QStringLiteral("rpc_spec"),
                                        QStringLiteral("pixel_convention"),
                                        QStringLiteral("line_off"),
                                        QStringLiteral("samp_off"),
                                        QStringLiteral("lat_off"),
                                        QStringLiteral("long_off"),
                                        QStringLiteral("height_off"),
                                        QStringLiteral("line_scale"),
                                        QStringLiteral("samp_scale"),
                                        QStringLiteral("lat_scale"),
                                        QStringLiteral("long_scale"),
                                        QStringLiteral("height_scale"),
                                        QStringLiteral("line_num_coeff"),
                                        QStringLiteral("line_den_coeff"),
                                        QStringLiteral("samp_num_coeff"),
                                        QStringLiteral("samp_den_coeff"),
                                        QStringLiteral("err_bias_m"),
                                        QStringLiteral("err_rand_m")};
            const QSet<QString> lineScanKeys{QStringLiteral("optics"), QStringLiteral("pixel_convention")};
            const QSet<QString>* modelKeys = &framePinholeKeys;
            if (model == QStringLiteral("rpc00b"))
            {
                modelKeys = &rpcKeys;
            }
            else if (model == QStringLiteral("planetary_linescan"))
            {
                modelKeys = &lineScanKeys;
            }
            for (const QString& key : *modelKeys)
            {
                extras.remove(key);
            }
            return extras;
        }

        QJsonObject definitionParameters(const QJsonObject& metadata, const QString& model)
        {
            if (model == QStringLiteral("frame_pinhole"))
            {
                const QJsonValue pixelPitchValue = metadata.value(QStringLiteral("pitch"));
                const double pixelPitch = pixelPitchValue.isDouble() ? pixelPitchValue.toDouble() : 0.0;
                const double unitScale = pixelPitch > 0.0 ? 1.0 / pixelPitch : 0.0;
                return QJsonObject{
                    {QStringLiteral("intrinsics"),
                     QJsonObject{
                         {QStringLiteral("fx_px"), scaledValue(metadata.value(QStringLiteral("fu")), unitScale)},
                         {QStringLiteral("fy_px"), scaledValue(metadata.value(QStringLiteral("fv")), unitScale)},
                         {QStringLiteral("cx_px"), scaledValue(metadata.value(QStringLiteral("cu")), unitScale)},
                         {QStringLiteral("cy_px"), scaledValue(metadata.value(QStringLiteral("cv")), unitScale)},
                         {QStringLiteral("pixel_pitch_mm"), pixelPitchValue},
                         {QStringLiteral("u_axis_sign"), metadata.value(QStringLiteral("u_direction"))},
                         {QStringLiteral("v_axis_sign"), metadata.value(QStringLiteral("v_direction"))}}},
                    {QStringLiteral("distortion"),
                     QJsonObject{{QStringLiteral("k1"), metadata.value(QStringLiteral("k1"))},
                                 {QStringLiteral("k2"), metadata.value(QStringLiteral("k2"))},
                                 {QStringLiteral("k3"), metadata.value(QStringLiteral("k3"))},
                                 {QStringLiteral("p1"), metadata.value(QStringLiteral("p1"))},
                                 {QStringLiteral("p2"), metadata.value(QStringLiteral("p2"))}}},
                    {QStringLiteral("pixel_convention"), canonicalFramePixelConvention(metadata)},
                    {QStringLiteral("depth_axis_flipped"), metadata.value(QStringLiteral("depth_axis_flipped"))}};
            }

            if (model == QStringLiteral("rpc00b"))
            {
                QJsonObject parameters{
                    {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                    {QStringLiteral("line_offset"), metadata.value(QStringLiteral("line_off"))},
                    {QStringLiteral("sample_offset"), metadata.value(QStringLiteral("samp_off"))},
                    {QStringLiteral("latitude_offset"), metadata.value(QStringLiteral("lat_off"))},
                    {QStringLiteral("longitude_offset"), metadata.value(QStringLiteral("long_off"))},
                    {QStringLiteral("height_offset"), metadata.value(QStringLiteral("height_off"))},
                    {QStringLiteral("line_scale"), metadata.value(QStringLiteral("line_scale"))},
                    {QStringLiteral("sample_scale"), metadata.value(QStringLiteral("samp_scale"))},
                    {QStringLiteral("latitude_scale"), metadata.value(QStringLiteral("lat_scale"))},
                    {QStringLiteral("longitude_scale"), metadata.value(QStringLiteral("long_scale"))},
                    {QStringLiteral("height_scale"), metadata.value(QStringLiteral("height_scale"))},
                    {QStringLiteral("line_numerator"), metadata.value(QStringLiteral("line_num_coeff"))},
                    {QStringLiteral("line_denominator"), metadata.value(QStringLiteral("line_den_coeff"))},
                    {QStringLiteral("sample_numerator"), metadata.value(QStringLiteral("samp_num_coeff"))},
                    {QStringLiteral("sample_denominator"), metadata.value(QStringLiteral("samp_den_coeff"))}};
                if (metadata.value(QStringLiteral("ground_crs")).isString())
                {
                    parameters.insert(QStringLiteral("ground_crs"), metadata.value(QStringLiteral("ground_crs")));
                }
                if (metadata.value(QStringLiteral("height_datum")).isString())
                {
                    parameters.insert(QStringLiteral("height_datum"), metadata.value(QStringLiteral("height_datum")));
                }
                if (const QJsonValue value = metadata.value(QStringLiteral("err_bias_m")); !value.isUndefined())
                {
                    parameters.insert(QStringLiteral("error_bias_m"), value);
                }
                if (const QJsonValue value = metadata.value(QStringLiteral("err_rand_m")); !value.isUndefined())
                {
                    parameters.insert(QStringLiteral("error_random_m"), value);
                }
                return parameters;
            }

            return QJsonObject{{QStringLiteral("optics"), metadata.value(QStringLiteral("optics"))},
                               {QStringLiteral("pixel_convention"), canonicalLinePixelConvention(metadata)}};
        }

        QString
        definitionIdFor(const QString& model, const QString& frame, int schemaVersion, const QJsonObject& parameters)
        {
            const QJsonObject fingerprint{{QStringLiteral("model_type"), model},
                                          {QStringLiteral("schema_version"), schemaVersion},
                                          {QStringLiteral("frame"), frame},
                                          {QStringLiteral("parameters"), parameters}};
            const QByteArray digest = QCryptographicHash::hash(
                QJsonDocument(fingerprint).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256);
            return QStringLiteral("camdef-") + QString::fromLatin1(digest.toHex().left(24));
        }

        QJsonObject imageByPath(const QJsonArray& images, const QString& path)
        {
            const QString normalized = normalizedPath(path);
            for (const QJsonValue& value : images)
            {
                const QJsonObject image = value.toObject();
                if (normalizedPath(image.value(QStringLiteral("path")).toString()) == normalized)
                {
                    return image;
                }
            }
            return {};
        }

        QJsonObject imageById(const QJsonArray& images, const QString& imageId)
        {
            for (const QJsonValue& value : images)
            {
                const QJsonObject image = value.toObject();
                if (image.value(QStringLiteral("image_uuid")).toString().trimmed() == imageId)
                {
                    return image;
                }
            }
            return {};
        }

        QJsonObject instanceByImageId(const QJsonArray& instances, const QString& imageId)
        {
            for (const QJsonValue& value : instances)
            {
                const QJsonObject instance = value.toObject();
                if (instance.value(QStringLiteral("image_uuid")).toString().trimmed() == imageId)
                {
                    return instance;
                }
            }
            return {};
        }

        QJsonObject definitionById(const QJsonArray& definitions, const QString& definitionId)
        {
            for (const QJsonValue& value : definitions)
            {
                const QJsonObject definition = value.toObject();
                if (definition.value(QStringLiteral("id")).toString().trimmed() == definitionId)
                {
                    return definition;
                }
            }
            return {};
        }

        bool validateTypedUpdateBinding(const CameraInstanceUpdate& update,
                                        const QJsonArray& images,
                                        const QJsonArray& instances,
                                        const QJsonArray& definitions,
                                        QString* error)
        {
            const QString imageId = QString::fromStdString(update.imageId.value()).trimmed();
            const QJsonObject image = imageById(images, imageId);
            if (image.isEmpty())
            {
                if (error)
                {
                    *error = QStringLiteral("camera instance update references unknown ImageId: %1").arg(imageId);
                }
                return false;
            }

            const QJsonObject instance = instanceByImageId(instances, imageId);
            if (instance.isEmpty())
            {
                if (error)
                {
                    *error =
                        QStringLiteral("camera instance update has no canonical instance for ImageId: %1").arg(imageId);
                }
                return false;
            }
            const QString expectedInstanceId = instance.value(QStringLiteral("id")).toString().trimmed();
            if (expectedInstanceId != QString::fromStdString(update.instanceId.value()).trimmed())
            {
                if (error)
                {
                    *error = QStringLiteral(
                                 "camera instance update binding is stale for ImageId %1: expected instance %2, got %3")
                                 .arg(imageId,
                                      expectedInstanceId,
                                      QString::fromStdString(update.instanceId.value()).trimmed());
                }
                return false;
            }

            const QString definitionId = instance.value(QStringLiteral("definition_id")).toString().trimmed();
            const QJsonObject definition = definitionById(definitions, definitionId);
            if (definition.isEmpty())
            {
                if (error)
                {
                    *error = QStringLiteral("camera instance update references missing definition %1 for ImageId %2")
                                 .arg(definitionId, imageId);
                }
                return false;
            }
            const QString expectedModel = definition.value(QStringLiteral("model_type")).toString().trimmed();
            const auto updateModel = modelType(update.modelMetadata);
            if (!updateModel)
            {
                if (error)
                {
                    *error =
                        QStringLiteral("camera instance update has an unsupported model for ImageId %1").arg(imageId);
                }
                return false;
            }
            if (expectedModel != *updateModel)
            {
                if (error)
                {
                    *error =
                        QStringLiteral("camera instance update changes model type for ImageId %1: expected %2, got %3")
                            .arg(imageId, expectedModel, *updateModel);
                }
                return false;
            }
            const QString expectedFrame = definition.value(QStringLiteral("frame")).toString().trimmed();
            if (expectedFrame != QString::fromStdString(update.worldFrame.value()).trimmed())
            {
                if (error)
                {
                    *error =
                        QStringLiteral("camera instance update binding uses world frame %1 for ImageId %2, expected %3")
                            .arg(QString::fromStdString(update.worldFrame.value()).trimmed(), imageId, expectedFrame);
                }
                return false;
            }
            return true;
        }

        QJsonObject makeInstance(const QString& imageId,
                                 const QString& definitionId,
                                 const QJsonObject& metadata,
                                 const QJsonObject& image,
                                 const QString& model,
                                 const QString& frame,
                                 const QString& existingInstanceId = QString())
        {
            QJsonObject instance{
                {QStringLiteral("id"),
                 existingInstanceId.isEmpty() ? QStringLiteral("caminst-") + imageId : existingInstanceId},
                {QStringLiteral("image_uuid"), imageId},
                {QStringLiteral("definition_id"), definitionId},
                {QStringLiteral("schema_version"), CameraProjectValidation::CurrentInstanceSchemaVersion}};

            int samples = model == QStringLiteral("frame_pinhole")
                              ? metadata.value(QStringLiteral("image_width")).toInt()
                              : metadata.value(QStringLiteral("image_samples")).toInt();
            int lines = model == QStringLiteral("frame_pinhole")
                            ? metadata.value(QStringLiteral("image_height")).toInt()
                            : metadata.value(QStringLiteral("image_lines")).toInt();
            if (samples <= 0)
            {
                samples = image.value(QStringLiteral("samples")).toInt();
            }
            if (lines <= 0)
            {
                lines = image.value(QStringLiteral("lines")).toInt();
            }
            if (samples > 0 && lines > 0)
            {
                instance.insert(QStringLiteral("image_size"),
                                QJsonObject{{QStringLiteral("samples"), samples}, {QStringLiteral("lines"), lines}});
            }

            const QJsonArray center = finiteArray(metadata.value(QStringLiteral("C")), 3);
            const QJsonArray rotation = finiteArray(metadata.value(QStringLiteral("R")), 9);
            if ((metadata.contains(QStringLiteral("C")) && center.isEmpty()) ||
                (metadata.contains(QStringLiteral("R")) && rotation.isEmpty()))
            {
                return {};
            }
            if (model == QStringLiteral("frame_pinhole") && (center.isEmpty() || rotation.isEmpty()))
            {
                return {};
            }
            if (model == QStringLiteral("frame_pinhole"))
            {
                instance.insert(QStringLiteral("pose"),
                                QJsonObject{{QStringLiteral("frame"), frame},
                                            {QStringLiteral("center_m"), center},
                                            {QStringLiteral("camera_to_world_rotation"), rotation}});
            }

            QJsonObject state;
            const QString source = metadata.value(QStringLiteral("source")).toString();
            const QString sourceFile = metadata.value(QStringLiteral("source_file")).toString();
            if (!source.isEmpty())
            {
                state.insert(QStringLiteral("source"), source);
            }
            if (!sourceFile.isEmpty())
            {
                state.insert(QStringLiteral("source_file"), sourceFile);
            }
            if (metadata.contains(QStringLiteral("image_correction")) &&
                !metadata.value(QStringLiteral("image_correction")).isObject())
            {
                return {};
            }
            if (metadata.value(QStringLiteral("image_correction")).isObject())
            {
                state.insert(QStringLiteral("image_correction"), metadata.value(QStringLiteral("image_correction")));
            }
            if (metadata.contains(QStringLiteral("trajectory")))
            {
                if (!metadata.value(QStringLiteral("trajectory")).isObject())
                {
                    return {};
                }
                state.insert(QStringLiteral("trajectory"), metadata.value(QStringLiteral("trajectory")));
            }
            if (metadata.contains(QStringLiteral("line_timing")))
            {
                if (!metadata.value(QStringLiteral("line_timing")).isObject())
                {
                    return {};
                }
                state.insert(QStringLiteral("line_timing"), metadata.value(QStringLiteral("line_timing")));
            }
            if (metadata.contains(QStringLiteral("capture_time")))
            {
                if (!metadata.value(QStringLiteral("capture_time")).isObject())
                {
                    return {};
                }
                state.insert(QStringLiteral("capture_time"), metadata.value(QStringLiteral("capture_time")));
            }
            const QJsonObject extras = instanceMetadata(metadata, model);
            if (!extras.isEmpty())
            {
                state.insert(QStringLiteral("metadata"), extras);
            }
            if (!state.isEmpty())
            {
                instance.insert(QStringLiteral("state"), state);
            }
            return instance;
        }

        bool validateCanonicalRecord(const camera_core::CameraModelRegistry& registry,
                                     const QString& model,
                                     const QString& definitionId,
                                     const QString& frame,
                                     int parameterSchemaVersion,
                                     const QJsonObject& parameters,
                                     const QJsonObject& instance,
                                     QString* error)
        {
            try
            {
                std::unique_ptr<camera_core::CameraDefinition> definition =
                    registry.createDefinition(model.toStdString(),
                                              camera_core::CameraDefinitionId(definitionId.toStdString()),
                                              xjw::coordinate_system::CoordinateFrameId(frame.toStdString()),
                                              parameterSchemaVersion,
                                              QJsonDocument(parameters).toJson(QJsonDocument::Compact).toStdString());
                std::shared_ptr<const camera_core::CameraDefinition> sharedDefinition(std::move(definition));

                QJsonObject state = instance.value(QStringLiteral("state")).toObject();
                state.insert(QStringLiteral("image_size"), instance.value(QStringLiteral("image_size")));
                if (instance.value(QStringLiteral("pose")).isObject())
                {
                    state.insert(QStringLiteral("pose"), instance.value(QStringLiteral("pose")));
                }
                const QString imageId = instance.value(QStringLiteral("image_uuid")).toString();
                const QString instanceId = instance.value(QStringLiteral("id")).toString();
                registry.createInstance(model.toStdString(),
                                        camera_core::CameraInstanceId(instanceId.toStdString()),
                                        camera_core::ImageId(imageId.toStdString()),
                                        std::move(sharedDefinition),
                                        QJsonDocument(state).toJson(QJsonDocument::Compact).toStdString());
                return true;
            }
            catch (const std::exception& exception)
            {
                if (error)
                {
                    *error = QString::fromUtf8(exception.what());
                }
                return false;
            }
        }

    } // namespace

    CameraProjectUpdateResult
    CameraProjectRecords::upsertByImagePath(QJsonObject* projectFiles,
                                            const QMap<QString, QJsonObject>& metadataByImagePath)
    {
        QMap<QString, QJsonObject> metadataByImageId;
        if (projectFiles)
        {
            const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
            for (auto it = metadataByImagePath.constBegin(); it != metadataByImagePath.constEnd(); ++it)
            {
                const QString imageId = imageByPath(images, it.key()).value(QStringLiteral("image_uuid")).toString();
                if (!imageId.isEmpty())
                {
                    metadataByImageId.insert(imageId, it.value());
                }
            }
        }
        return updateByImageId(projectFiles, {}, metadataByImageId, false);
    }

    CameraProjectUpdateResult CameraProjectRecords::upsertByImageId(QJsonObject* projectFiles,
                                                                    const CameraInstanceUpdates& updates)
    {
        CameraProjectUpdateResult result;
        if (!projectFiles)
        {
            result.errors.append(QStringLiteral("camera project document is null"));
            return result;
        }
        if (updates.empty())
        {
            result.errors.append(QStringLiteral("camera instance update list is empty"));
            return result;
        }

        const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
        const QJsonArray definitions = projectFiles->value(QStringLiteral("camera_definitions")).toArray();
        const QJsonArray instances = projectFiles->value(QStringLiteral("camera_instances")).toArray();
        QSet<QString> seen;
        QMap<QString, QJsonObject> metadataByImageId;
        for (const CameraInstanceUpdate& update : updates)
        {
            const QString imageId = QString::fromStdString(update.imageId.value()).trimmed();
            if (imageId.isEmpty())
            {
                result.errors.append(QStringLiteral("camera instance update contains an empty ImageId"));
                continue;
            }
            if (seen.contains(imageId))
            {
                result.errors.append(
                    QStringLiteral("camera instance update contains duplicate ImageId: %1").arg(imageId));
                continue;
            }
            seen.insert(imageId);
            QString bindingError;
            if (!validateTypedUpdateBinding(update, images, instances, definitions, &bindingError))
            {
                result.errors.append(bindingError);
                continue;
            }
            if (update.modelMetadata.isEmpty())
            {
                result.errors.append(
                    QStringLiteral("camera instance update metadata is empty for ImageId: %1").arg(imageId));
                continue;
            }
            const auto updateModel = modelType(update.modelMetadata);
            if (!updateModel)
            {
                result.errors.append(QStringLiteral("unsupported camera model for image: %1").arg(imageId));
                continue;
            }
            const QString metadataFrame = frameForModel(update.modelMetadata, *updateModel);
            if (metadataFrame != QString::fromStdString(update.worldFrame.value()).trimmed())
            {
                result.errors.append(
                    QStringLiteral(
                        "camera instance update metadata frame %1 does not match binding frame %2 for ImageId %3")
                        .arg(metadataFrame, QString::fromStdString(update.worldFrame.value()).trimmed(), imageId));
                continue;
            }
            metadataByImageId.insert(imageId, update.modelMetadata);
        }
        if (!result.errors.isEmpty())
        {
            return result;
        }

        result = updateByImageId(projectFiles, {}, metadataByImageId, false);
        if (result.ok() && result.updatedCount != static_cast<int>(updates.size()))
        {
            result.errors.append(QStringLiteral("camera instance update count mismatch: %1/%2")
                                     .arg(result.updatedCount)
                                     .arg(static_cast<int>(updates.size())));
        }
        return result;
    }

    CameraProjectUpdateResult
    CameraProjectRecords::replaceByImagePath(QJsonObject* projectFiles,
                                             const QStringList& targetImagePaths,
                                             const QMap<QString, QJsonObject>& metadataByImagePath)
    {
        QSet<QString> targetIds;
        if (projectFiles)
        {
            const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
            for (const QString& path : targetImagePaths)
            {
                const QString imageId = imageByPath(images, path).value(QStringLiteral("image_uuid")).toString();
                if (!imageId.isEmpty())
                {
                    targetIds.insert(imageId);
                }
            }
        }
        QMap<QString, QJsonObject> metadataByImageId;
        if (projectFiles)
        {
            const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
            for (auto it = metadataByImagePath.constBegin(); it != metadataByImagePath.constEnd(); ++it)
            {
                const QString imageId = imageByPath(images, it.key()).value(QStringLiteral("image_uuid")).toString();
                if (!imageId.isEmpty())
                {
                    metadataByImageId.insert(imageId, it.value());
                }
            }
        }
        return updateByImageId(projectFiles, targetIds, metadataByImageId, true);
    }

    CameraProjectUpdateResult CameraProjectRecords::replaceByImageId(QJsonObject* projectFiles,
                                                                     const CameraImageIds& targetImageIds,
                                                                     const CameraInstanceUpdates& updates)
    {
        CameraProjectUpdateResult result;
        if (!projectFiles)
        {
            result.errors.append(QStringLiteral("camera project document is null"));
            return result;
        }
        if (targetImageIds.empty())
        {
            result.errors.append(QStringLiteral("target ImageId list is empty"));
            return result;
        }

        const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
        const QJsonArray definitions = projectFiles->value(QStringLiteral("camera_definitions")).toArray();
        const QJsonArray instances = projectFiles->value(QStringLiteral("camera_instances")).toArray();
        QSet<QString> targetIds;
        for (const camera_core::ImageId& imageId : targetImageIds)
        {
            const QString value = QString::fromStdString(imageId.value()).trimmed();
            if (value.isEmpty())
            {
                result.errors.append(QStringLiteral("target ImageId is empty"));
                continue;
            }
            if (targetIds.contains(value))
            {
                result.errors.append(QStringLiteral("duplicate target ImageId: %1").arg(value));
                continue;
            }
            if (imageById(images, value).isEmpty())
            {
                result.errors.append(QStringLiteral("target ImageId is not registered in the project: %1").arg(value));
                continue;
            }
            targetIds.insert(value);
        }

        QSet<QString> seenUpdates;
        QMap<QString, QJsonObject> metadataByImageId;
        for (const CameraInstanceUpdate& update : updates)
        {
            const QString imageId = QString::fromStdString(update.imageId.value()).trimmed();
            if (imageId.isEmpty())
            {
                result.errors.append(QStringLiteral("camera instance update contains an empty ImageId"));
                continue;
            }
            if (seenUpdates.contains(imageId))
            {
                result.errors.append(
                    QStringLiteral("camera instance update contains duplicate ImageId: %1").arg(imageId));
                continue;
            }
            seenUpdates.insert(imageId);
            if (!targetIds.contains(imageId))
            {
                result.errors.append(
                    QStringLiteral("camera instance update is outside the target ImageId set: %1").arg(imageId));
                continue;
            }
            QString bindingError;
            if (!validateTypedUpdateBinding(update, images, instances, definitions, &bindingError))
            {
                result.errors.append(bindingError);
                continue;
            }
            if (update.modelMetadata.isEmpty())
            {
                result.errors.append(
                    QStringLiteral("camera instance update metadata is empty for ImageId: %1").arg(imageId));
                continue;
            }
            const auto updateModel = modelType(update.modelMetadata);
            if (!updateModel)
            {
                result.errors.append(QStringLiteral("unsupported camera model for image: %1").arg(imageId));
                continue;
            }
            const QString metadataFrame = frameForModel(update.modelMetadata, *updateModel);
            if (metadataFrame != QString::fromStdString(update.worldFrame.value()).trimmed())
            {
                result.errors.append(
                    QStringLiteral(
                        "camera instance update metadata frame %1 does not match binding frame %2 for ImageId %3")
                        .arg(metadataFrame, QString::fromStdString(update.worldFrame.value()).trimmed(), imageId));
                continue;
            }
            metadataByImageId.insert(imageId, update.modelMetadata);
        }
        if (!result.errors.isEmpty())
        {
            return result;
        }
        return updateByImageId(projectFiles, targetIds, metadataByImageId, true);
    }

    CameraProjectUpdateResult CameraProjectRecords::clearByImagePath(QJsonObject* projectFiles,
                                                                     const QStringList& imagePaths)
    {
        QSet<QString> targetIds;
        if (projectFiles)
        {
            const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
            for (const QString& path : imagePaths)
            {
                const QString imageId = imageByPath(images, path).value(QStringLiteral("image_uuid")).toString();
                if (!imageId.isEmpty())
                {
                    targetIds.insert(imageId);
                }
            }
        }
        return updateByImageId(projectFiles, targetIds, {}, true);
    }

    CameraProjectUpdateResult CameraProjectRecords::updateByImageId(QJsonObject* projectFiles,
                                                                    const QSet<QString>& targetImageIds,
                                                                    const QMap<QString, QJsonObject>& metadataByImageId,
                                                                    bool clearMissing)
    {
        CameraProjectUpdateResult result;
        if (!projectFiles)
        {
            result.errors.append(QStringLiteral("camera project document is null"));
            return result;
        }

        const QJsonArray images = projectFiles->value(QStringLiteral("images")).toArray();
        QJsonArray definitions = projectFiles->value(QStringLiteral("camera_definitions")).toArray();
        QJsonArray instances = projectFiles->value(QStringLiteral("camera_instances")).toArray();
        const camera_core::CameraModelRegistry registry = camera_models::makeBuiltinCameraModelRegistry();
        QMap<QString, int> definitionIndex;
        QMap<QString, int> instanceIndex;
        for (int index = 0; index < definitions.size(); ++index)
        {
            definitionIndex.insert(definitions.at(index).toObject().value(QStringLiteral("id")).toString(), index);
        }
        for (int index = 0; index < instances.size(); ++index)
        {
            instanceIndex.insert(instances.at(index).toObject().value(QStringLiteral("image_uuid")).toString(), index);
        }

        QSet<QString> touched;
        for (auto it = metadataByImageId.constBegin(); it != metadataByImageId.constEnd(); ++it)
        {
            const QJsonObject image = imageById(images, it.key());
            const QString imageId = image.value(QStringLiteral("image_uuid")).toString();
            if (imageId.isEmpty())
            {
                continue;
            }

            const auto modelValue = modelType(it.value());
            if (!modelValue)
            {
                result.errors.append(QStringLiteral("unsupported camera model for image: %1").arg(it.key()));
                continue;
            }
            const QString model = *modelValue;
            const QString frame = frameForModel(it.value(), model);
            const std::optional<int> parameterSchemaVersion =
                camera_models::builtinCameraParameterSchemaVersion(model.toStdString());
            if (!parameterSchemaVersion)
            {
                result.errors.append(QStringLiteral("camera model has no registered parameter schema: %1").arg(model));
                continue;
            }
            QString inputError;
            if (!validateInputContainers(it.value(), model, &inputError))
            {
                result.errors.append(QStringLiteral("camera definition metadata for image is invalid: %1 (%2)")
                                         .arg(it.key(), inputError));
                continue;
            }
            const QJsonObject parameters = definitionParameters(it.value(), model);
            const QString definitionId = definitionIdFor(model, frame, *parameterSchemaVersion, parameters);
            const QJsonObject definition{{QStringLiteral("id"), definitionId},
                                         {QStringLiteral("model_type"), model},
                                         {QStringLiteral("schema_version"), *parameterSchemaVersion},
                                         {QStringLiteral("frame"), frame},
                                         {QStringLiteral("parameters"), parameters}};
            const auto definitionIt = definitionIndex.constFind(definitionId);
            if (definitionIt == definitionIndex.constEnd())
            {
                definitionIndex.insert(definitionId, definitions.size());
                definitions.append(definition);
            }
            else
            {
                definitions[definitionIt.value()] = definition;
            }

            const auto existingInstanceIt = instanceIndex.constFind(imageId);
            QString existingInstanceId;
            if (existingInstanceIt != instanceIndex.constEnd())
            {
                existingInstanceId = instances.at(existingInstanceIt.value())
                                         .toObject()
                                         .value(QStringLiteral("id"))
                                         .toString()
                                         .trimmed();
            }
            QJsonObject instance =
                makeInstance(imageId, definitionId, it.value(), image, model, frame, existingInstanceId);
            if (instance.isEmpty())
            {
                result.errors.append(QStringLiteral("camera pose for image is incomplete: %1").arg(it.key()));
                continue;
            }
            if (!instance.value(QStringLiteral("image_size")).isObject() &&
                existingInstanceIt != instanceIndex.constEnd())
            {
                const QJsonObject previous = instances.at(existingInstanceIt.value()).toObject();
                if (previous.value(QStringLiteral("image_size")).isObject())
                {
                    instance.insert(QStringLiteral("image_size"), previous.value(QStringLiteral("image_size")));
                }
            }
            if (!instance.value(QStringLiteral("image_size")).isObject())
            {
                result.errors.append(QStringLiteral("image has no valid size for camera instance: %1").arg(it.key()));
                continue;
            }
            QString factoryError;
            if (!validateCanonicalRecord(
                    registry, model, definitionId, frame, *parameterSchemaVersion, parameters, instance, &factoryError))
            {
                result.errors.append(
                    QStringLiteral("camera metadata for image is invalid: %1 (%2)").arg(it.key(), factoryError));
                continue;
            }
            const auto instanceIt = instanceIndex.constFind(imageId);
            if (instanceIt == instanceIndex.constEnd())
            {
                instanceIndex.insert(imageId, instances.size());
                instances.append(instance);
            }
            else
            {
                instances[instanceIt.value()] = instance;
            }
            touched.insert(imageId);
            ++result.updatedCount;
        }

        if (clearMissing)
        {
            QJsonArray kept;
            for (const QJsonValue& value : instances)
            {
                const QString imageId = value.toObject().value(QStringLiteral("image_uuid")).toString();
                if (targetImageIds.contains(imageId) && !touched.contains(imageId))
                {
                    ++result.clearedCount;
                    continue;
                }
                kept.append(value);
            }
            instances = kept;
        }

        if (!result.errors.isEmpty())
        {
            return result;
        }

        QJsonObject candidate = *projectFiles;
        // Keep the image collection untouched.  CameraProjectValidation
        // rejects legacy per-image camera fields instead of silently
        // rewriting them into the canonical definition/instance graph.
        candidate.insert(QStringLiteral("images"), images);
        candidate.insert(QStringLiteral("camera_definitions"), definitions);
        candidate.insert(QStringLiteral("camera_instances"), instances);
        const CameraProjectValidationResult validation =
            CameraProjectStore::validate(candidate, CameraProjectData{definitions, instances});
        if (!validation.ok())
        {
            result.errors = validation.errors;
            return result;
        }
        *projectFiles = std::move(candidate);
        return result;
    }

    QJsonObject CameraProjectRecords::instanceForImage(const QJsonObject& projectFiles, const QString& imageUuid)
    {
        for (const QJsonValue& value : projectFiles.value(QStringLiteral("camera_instances")).toArray())
        {
            const QJsonObject instance = value.toObject();
            if (instance.value(QStringLiteral("image_uuid")).toString() == imageUuid)
            {
                return instance;
            }
        }
        return {};
    }

    QJsonObject CameraProjectRecords::definitionForInstance(const QJsonObject& projectFiles,
                                                            const QJsonObject& instance)
    {
        const QString definitionId = instance.value(QStringLiteral("definition_id")).toString();
        for (const QJsonValue& value : projectFiles.value(QStringLiteral("camera_definitions")).toArray())
        {
            const QJsonObject definition = value.toObject();
            if (definition.value(QStringLiteral("id")).toString() == definitionId)
            {
                return definition;
            }
        }
        return {};
    }

    QJsonObject CameraProjectRecords::modelParametersForInstance(const QJsonObject& projectFiles,
                                                                 const QJsonObject& instance)
    {
        if (instance.isEmpty())
        {
            return {};
        }
        const QJsonObject definition = definitionForInstance(projectFiles, instance);
        if (definition.isEmpty())
        {
            return {};
        }
        const QString model = definition.value(QStringLiteral("model_type")).toString();
        if (model != QStringLiteral("frame_pinhole") && model != QStringLiteral("rpc00b") &&
            model != QStringLiteral("planetary_linescan"))
        {
            return {};
        }
        const QJsonObject parameters = definition.value(QStringLiteral("parameters")).toObject();
        const QJsonObject state = instance.value(QStringLiteral("state")).toObject();
        const QJsonObject instanceExtras = state.value(QStringLiteral("metadata")).toObject();
        QJsonObject metadata = instanceExtras;
        for (auto it = state.constBegin(); it != state.constEnd(); ++it)
        {
            if (it.key() != QStringLiteral("metadata"))
            {
                metadata.insert(it.key(), it.value());
            }
        }

        if (model == QStringLiteral("frame_pinhole"))
        {
            const QJsonObject intrinsics = parameters.value(QStringLiteral("intrinsics")).toObject();
            const QJsonObject distortion = parameters.value(QStringLiteral("distortion")).toObject();
            const QJsonValue pixelPitch = intrinsics.value(QStringLiteral("pixel_pitch_mm"));
            const double pitch = pixelPitch.toDouble();
            metadata.insert(QStringLiteral("model"), QStringLiteral("frame_pinhole"));
            metadata.insert(QStringLiteral("intrinsics_unit"), QStringLiteral("mm"));
            metadata.insert(QStringLiteral("camera_center_unit"), QStringLiteral("m"));
            metadata.insert(QStringLiteral("pitch"), pixelPitch);
            metadata.insert(QStringLiteral("fu"), scaledValue(intrinsics.value(QStringLiteral("fx_px")), pitch));
            metadata.insert(QStringLiteral("fv"), scaledValue(intrinsics.value(QStringLiteral("fy_px")), pitch));
            metadata.insert(QStringLiteral("cu"), scaledValue(intrinsics.value(QStringLiteral("cx_px")), pitch));
            metadata.insert(QStringLiteral("cv"), scaledValue(intrinsics.value(QStringLiteral("cy_px")), pitch));
            metadata.insert(QStringLiteral("u_direction"), intrinsics.value(QStringLiteral("u_axis_sign")));
            metadata.insert(QStringLiteral("v_direction"), intrinsics.value(QStringLiteral("v_axis_sign")));
            metadata.insert(QStringLiteral("pixel_convention"), parameters.value(QStringLiteral("pixel_convention")));
            metadata.insert(QStringLiteral("k1"), distortion.value(QStringLiteral("k1")));
            metadata.insert(QStringLiteral("k2"), distortion.value(QStringLiteral("k2")));
            metadata.insert(QStringLiteral("k3"), distortion.value(QStringLiteral("k3")));
            metadata.insert(QStringLiteral("p1"), distortion.value(QStringLiteral("p1")));
            metadata.insert(QStringLiteral("p2"), distortion.value(QStringLiteral("p2")));
            metadata.insert(QStringLiteral("depth_axis_flipped"),
                            parameters.value(QStringLiteral("depth_axis_flipped")).toBool(false));
        }
        else if (model == QStringLiteral("rpc00b"))
        {
            metadata.insert(QStringLiteral("model"), QStringLiteral("rpc00b"));
            metadata.insert(QStringLiteral("rpc_spec"), QStringLiteral("RPC00B"));
            metadata.insert(QStringLiteral("pixel_convention"), QStringLiteral("opencv_zero_based_center"));
            const auto copy = [&parameters, &metadata](const QString& source, const QString& target)
            {
                if (parameters.contains(source))
                {
                    metadata.insert(target, parameters.value(source));
                }
            };
            copy(QStringLiteral("line_offset"), QStringLiteral("line_off"));
            copy(QStringLiteral("sample_offset"), QStringLiteral("samp_off"));
            copy(QStringLiteral("latitude_offset"), QStringLiteral("lat_off"));
            copy(QStringLiteral("longitude_offset"), QStringLiteral("long_off"));
            copy(QStringLiteral("height_offset"), QStringLiteral("height_off"));
            copy(QStringLiteral("line_scale"), QStringLiteral("line_scale"));
            copy(QStringLiteral("sample_scale"), QStringLiteral("samp_scale"));
            copy(QStringLiteral("latitude_scale"), QStringLiteral("lat_scale"));
            copy(QStringLiteral("longitude_scale"), QStringLiteral("long_scale"));
            copy(QStringLiteral("height_scale"), QStringLiteral("height_scale"));
            copy(QStringLiteral("line_numerator"), QStringLiteral("line_num_coeff"));
            copy(QStringLiteral("line_denominator"), QStringLiteral("line_den_coeff"));
            copy(QStringLiteral("sample_numerator"), QStringLiteral("samp_num_coeff"));
            copy(QStringLiteral("sample_denominator"), QStringLiteral("samp_den_coeff"));
            copy(QStringLiteral("error_bias_m"), QStringLiteral("err_bias_m"));
            copy(QStringLiteral("error_random_m"), QStringLiteral("err_rand_m"));
            if (parameters.contains(QStringLiteral("ground_crs")))
            {
                metadata.insert(QStringLiteral("ground_crs"), parameters.value(QStringLiteral("ground_crs")));
            }
            if (parameters.contains(QStringLiteral("height_datum")))
            {
                metadata.insert(QStringLiteral("height_datum"), parameters.value(QStringLiteral("height_datum")));
            }
        }
        else
        {
            metadata.insert(QStringLiteral("model"), QStringLiteral("planetary_linescan"));
            metadata.insert(QStringLiteral("optics"), parameters.value(QStringLiteral("optics")));
            metadata.insert(QStringLiteral("pixel_convention"), parameters.value(QStringLiteral("pixel_convention")));
        }
        metadata.insert(QStringLiteral("world_frame"), definition.value(QStringLiteral("frame")));

        const QJsonObject pose = instance.value(QStringLiteral("pose")).toObject();
        if (!pose.isEmpty())
        {
            metadata.insert(QStringLiteral("C"), pose.value(QStringLiteral("center_m")));
            metadata.insert(QStringLiteral("R"), pose.value(QStringLiteral("camera_to_world_rotation")));
            metadata.insert(QStringLiteral("camera_center_unit"), QStringLiteral("m"));
        }
        const QJsonObject size = instance.value(QStringLiteral("image_size")).toObject();
        if (!size.isEmpty())
        {
            if (model == QStringLiteral("frame_pinhole"))
            {
                metadata.insert(QStringLiteral("image_width"), size.value(QStringLiteral("samples")));
                metadata.insert(QStringLiteral("image_height"), size.value(QStringLiteral("lines")));
            }
            else
            {
                metadata.insert(QStringLiteral("image_samples"), size.value(QStringLiteral("samples")));
                metadata.insert(QStringLiteral("image_lines"), size.value(QStringLiteral("lines")));
            }
        }
        return metadata;
    }

    QJsonObject CameraProjectRecords::modelParametersForImage(const QJsonObject& projectFiles, const QJsonObject& image)
    {
        const QJsonObject instance =
            instanceForImage(projectFiles, image.value(QStringLiteral("image_uuid")).toString());
        return modelParametersForInstance(projectFiles, instance);
    }

} // namespace xjw::camera_project
