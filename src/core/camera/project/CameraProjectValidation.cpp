#include "CameraProjectValidation.h"

#include "camera/models/CameraModelFactories.h"

#include <QJsonObject>
#include <QSet>

#include <cmath>
#include <limits>

namespace xjw::camera_project
{
    namespace
    {

        void appendError(QStringList* errors, const QString& message)
        {
            errors->append(message);
        }

        bool validIdentifier(const QJsonValue& value, QString* identifier)
        {
            if (!value.isString())
            {
                return false;
            }
            const QString candidate = value.toString();
            if (candidate.trimmed().isEmpty() || candidate != candidate.trimmed())
            {
                return false;
            }
            if (identifier)
            {
                *identifier = candidate;
            }
            return true;
        }

        bool validPositiveInt(const QJsonValue& value)
        {
            if (!value.isDouble())
            {
                return false;
            }
            const double number = value.toDouble();
            return std::isfinite(number) && number > 0.0 &&
                   number <= static_cast<double>(std::numeric_limits<int>::max()) && std::floor(number) == number;
        }

        bool validFiniteArray(const QJsonValue& value, int expected)
        {
            if (!value.isArray())
            {
                return false;
            }
            const QJsonArray values = value.toArray();
            if (values.size() != expected)
            {
                return false;
            }
            for (const QJsonValue& item : values)
            {
                if (!item.isDouble() || !std::isfinite(item.toDouble()))
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    CameraProjectValidationResult CameraProjectValidation::validate(const QJsonArray& definitions,
                                                                    const QJsonArray& instances,
                                                                    const QJsonArray& images)
    {
        CameraProjectValidationResult result;
        QSet<QString> imageIds;
        for (int index = 0; index < images.size(); ++index)
        {
            const QJsonValue value = images.at(index);
            if (!value.isObject())
            {
                appendError(&result.errors, QStringLiteral("images[%1] must be an object").arg(index));
                continue;
            }
            const QJsonObject image = value.toObject();
            if (image.contains(QStringLiteral("camera")))
            {
                appendError(&result.errors,
                            QStringLiteral("images[%1].camera is not supported; use camera_instances").arg(index));
            }
            if (image.contains(QStringLiteral("camera_file")))
            {
                appendError(&result.errors,
                            QStringLiteral("images[%1].camera_file is not supported; use camera_instances").arg(index));
            }
            QString imageId;
            if (!validIdentifier(image.value(QStringLiteral("image_uuid")), &imageId))
            {
                appendError(&result.errors,
                            QStringLiteral("images[%1].image_uuid must be a trimmed non-empty string").arg(index));
            }
            else if (imageIds.contains(imageId))
            {
                appendError(&result.errors, QStringLiteral("duplicate image_uuid: %1").arg(imageId));
            }
            else
            {
                imageIds.insert(imageId);
            }
        }

        QSet<QString> definitionIds;
        for (int index = 0; index < definitions.size(); ++index)
        {
            const QJsonValue value = definitions.at(index);
            if (!value.isObject())
            {
                appendError(&result.errors, QStringLiteral("camera_definitions[%1] must be an object").arg(index));
                continue;
            }
            const QJsonObject definition = value.toObject();
            QString definitionId;
            if (!validIdentifier(definition.value(QStringLiteral("id")), &definitionId))
            {
                appendError(&result.errors,
                            QStringLiteral("camera_definitions[%1].id must be a trimmed non-empty string").arg(index));
            }
            else if (definitionIds.contains(definitionId))
            {
                appendError(&result.errors, QStringLiteral("duplicate camera definition id: %1").arg(definitionId));
            }
            else
            {
                definitionIds.insert(definitionId);
            }

            const QString modelType = definition.value(QStringLiteral("model_type")).toString().trimmed();
            if (!definition.value(QStringLiteral("model_type")).isString() || modelType.isEmpty())
            {
                appendError(&result.errors, QStringLiteral("camera_definitions[%1].model_type is required").arg(index));
            }
            const QJsonValue schema = definition.value(QStringLiteral("schema_version"));
            const int schemaVersion = schema.toInt(-1);
            const bool positiveInteger =
                schema.isDouble() && schemaVersion > 0 && schema.toDouble() == static_cast<double>(schemaVersion);
            const std::optional<int> expectedVersion =
                camera_models::builtinCameraParameterSchemaVersion(modelType.toStdString());
            if (!positiveInteger)
            {
                appendError(
                    &result.errors,
                    QStringLiteral("camera_definitions[%1].schema_version must be a positive integer").arg(index));
            }
            else if (expectedVersion && schemaVersion != *expectedVersion)
            {
                appendError(&result.errors,
                            QStringLiteral("camera_definitions[%1].schema_version must be %2 for model %3")
                                .arg(index)
                                .arg(*expectedVersion)
                                .arg(modelType));
            }

            QString frame;
            if (!validIdentifier(definition.value(QStringLiteral("frame")), &frame))
            {
                appendError(
                    &result.errors,
                    QStringLiteral("camera_definitions[%1].frame must be a trimmed non-empty string").arg(index));
            }
            if (!definition.value(QStringLiteral("parameters")).isObject())
            {
                appendError(&result.errors,
                            QStringLiteral("camera_definitions[%1].parameters must be an object").arg(index));
            }
        }

        QSet<QString> instanceIds;
        QSet<QString> boundImages;
        for (int index = 0; index < instances.size(); ++index)
        {
            const QJsonValue value = instances.at(index);
            if (!value.isObject())
            {
                appendError(&result.errors, QStringLiteral("camera_instances[%1] must be an object").arg(index));
                continue;
            }
            const QJsonObject instance = value.toObject();
            QString imageId;
            if (!validIdentifier(instance.value(QStringLiteral("image_uuid")), &imageId))
            {
                appendError(
                    &result.errors,
                    QStringLiteral("camera_instances[%1].image_uuid must be a trimmed non-empty string").arg(index));
            }
            else
            {
                if (!imageIds.contains(imageId))
                {
                    appendError(
                        &result.errors,
                        QStringLiteral("camera_instances[%1] references unknown image: %2").arg(index).arg(imageId));
                }
                if (boundImages.contains(imageId))
                {
                    appendError(&result.errors,
                                QStringLiteral("image has more than one camera instance: %1").arg(imageId));
                }
                boundImages.insert(imageId);
            }

            QString instanceId;
            if (!validIdentifier(instance.value(QStringLiteral("id")), &instanceId))
            {
                appendError(
                    &result.errors,
                    QStringLiteral("camera_instances[%1].id must be a required trimmed non-empty string").arg(index));
            }
            else
            {
                if (instanceIds.contains(instanceId))
                {
                    appendError(&result.errors, QStringLiteral("duplicate camera instance id: %1").arg(instanceId));
                }
                instanceIds.insert(instanceId);
            }

            QString definitionId;
            if (!validIdentifier(instance.value(QStringLiteral("definition_id")), &definitionId))
            {
                appendError(&result.errors,
                            QStringLiteral("camera_instances[%1].definition_id is required").arg(index));
            }
            else if (!definitionIds.contains(definitionId))
            {
                appendError(&result.errors,
                            QStringLiteral("camera_instances[%1] references unknown definition: %2")
                                .arg(index)
                                .arg(definitionId));
            }

            const QJsonValue schema = instance.value(QStringLiteral("schema_version"));
            if (!schema.isDouble() || schema.toInt(-1) != CurrentInstanceSchemaVersion ||
                schema.toDouble() != static_cast<double>(CurrentInstanceSchemaVersion))
            {
                appendError(&result.errors,
                            QStringLiteral("camera_instances[%1].schema_version must be %2")
                                .arg(index)
                                .arg(CurrentInstanceSchemaVersion));
            }

            const QJsonObject imageSize = instance.value(QStringLiteral("image_size")).toObject();
            if (!instance.value(QStringLiteral("image_size")).isObject() ||
                !validPositiveInt(imageSize.value(QStringLiteral("samples"))) ||
                !validPositiveInt(imageSize.value(QStringLiteral("lines"))))
            {
                appendError(&result.errors,
                            QStringLiteral("camera_instances[%1].image_size must contain positive samples and lines")
                                .arg(index));
            }

            if (instance.contains(QStringLiteral("state")) && !instance.value(QStringLiteral("state")).isObject())
            {
                appendError(&result.errors, QStringLiteral("camera_instances[%1].state must be an object").arg(index));
            }
            if (instance.contains(QStringLiteral("pose")) && !instance.value(QStringLiteral("pose")).isObject())
            {
                appendError(&result.errors, QStringLiteral("camera_instances[%1].pose must be an object").arg(index));
            }
            if (instance.value(QStringLiteral("pose")).isObject())
            {
                const QJsonObject pose = instance.value(QStringLiteral("pose")).toObject();
                QString poseFrame;
                if (!validIdentifier(pose.value(QStringLiteral("frame")), &poseFrame))
                {
                    appendError(&result.errors,
                                QStringLiteral("camera_instances[%1].pose.frame must be a trimmed non-empty string")
                                    .arg(index));
                }
                if (!validFiniteArray(pose.value(QStringLiteral("center_m")), 3))
                {
                    appendError(
                        &result.errors,
                        QStringLiteral("camera_instances[%1].pose.center_m must contain 3 finite values").arg(index));
                }
                if (!validFiniteArray(pose.value(QStringLiteral("camera_to_world_rotation")), 9))
                {
                    appendError(&result.errors,
                                QStringLiteral(
                                    "camera_instances[%1].pose.camera_to_world_rotation must contain 9 finite values")
                                    .arg(index));
                }
            }
        }

        return result;
    }

} // namespace xjw::camera_project
