#include "CameraProjectRuntime.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xjw::camera_project
{
    namespace
    {

        QString requiredString(const QJsonObject& object, const QString& key)
        {
            return object.value(key).toString().trimmed();
        }

        QByteArray compactJson(const QJsonObject& object)
        {
            return QJsonDocument(object).toJson(QJsonDocument::Compact);
        }

        void appendError(QStringList* errors, const QString& prefix, const std::exception& exception)
        {
            errors->append(prefix + QStringLiteral(": ") + QString::fromUtf8(exception.what()));
        }

    } // namespace

    camera_core::CameraOperationPlan
    CameraProjectRuntimeResult::planOperationForImages(const std::vector<camera_core::ImageId>& images,
                                                       camera_core::CameraOperation operation) const
    {
        camera_core::CameraOperationPlan plan;
        plan.operation = operation;
        plan.requirements = camera_core::requirementsFor(operation);
        if (!errors.isEmpty())
        {
            plan.inputError = "canonical camera runtime is invalid: " + errors.join(QStringLiteral("; ")).toStdString();
            return plan;
        }

        std::string selectionError;
        const camera_core::CameraInstanceSet selected = instances.select(images, &selectionError);
        if (!selectionError.empty())
        {
            plan.inputError = std::move(selectionError);
            return plan;
        }
        return camera_core::planCameraOperation(selected, operation);
    }

    bool
    CameraProjectRuntimeResult::framePinholeStateForImage(const camera_core::ImageId& image,
                                                          camera_models::frame_pinhole::FramePinholeNumericState* state,
                                                          std::string* error) const
    {
        if (error)
        {
            error->clear();
        }
        if (!state)
        {
            if (error)
            {
                *error = "frame-pinhole numeric state output is null";
            }
            return false;
        }
        *state = camera_models::frame_pinhole::FramePinholeNumericState{};
        if (!errors.isEmpty())
        {
            if (error)
            {
                *error = "canonical camera runtime is invalid: " + errors.join(QStringLiteral("; ")).toStdString();
            }
            return false;
        }

        const camera_core::CameraInstanceLookupResult lookup = instances.forImage(image);
        if (!lookup.ok())
        {
            if (error)
            {
                *error =
                    "cannot resolve frame-pinhole numeric state for image '" + image.value() + "': " + lookup.error;
            }
            return false;
        }

        const auto pinhole =
            std::dynamic_pointer_cast<const camera_models::frame_pinhole::FramePinholeInstance>(lookup.instance);
        if (!pinhole)
        {
            if (error)
            {
                *error = "camera instance for image '" + image.value() + "' is not a frame-pinhole instance (model '" +
                         std::string(lookup.instance->definition().modelType()) + "')";
            }
            return false;
        }

        camera_models::frame_pinhole::FramePinholeNumericState resolved;
        std::string conversionError;
        bool converted = false;
        try
        {
            converted = camera_models::frame_pinhole::FramePinholeNumericState::fromInstance(
                *pinhole, &resolved, &conversionError);
        }
        catch (const std::exception& exception)
        {
            conversionError = exception.what();
        }
        catch (...)
        {
            conversionError = "unknown exception while decoding the frame-pinhole numeric state";
        }
        if (!converted)
        {
            if (error)
            {
                *error = "invalid frame-pinhole instance for image '" + image.value() + "': " + conversionError;
            }
            return false;
        }
        if (!resolved.hasBoundIdentity())
        {
            if (error)
            {
                *error = "frame-pinhole numeric state for image '" + image.value() +
                         "' has no explicit instance identity or world frame";
            }
            return false;
        }
        std::string validationError;
        if (!resolved.validateNumericalState(&validationError))
        {
            if (error)
            {
                *error = "invalid frame-pinhole numeric state for image '" + image.value() + "': " + validationError;
            }
            return false;
        }
        if (resolved.imageId() != image)
        {
            if (error)
            {
                *error =
                    "frame-pinhole numeric state image identity does not match requested image '" + image.value() + "'";
            }
            return false;
        }

        *state = std::move(resolved);
        return true;
    }

    bool CameraProjectRuntimeResult::framePinholeStatesForImages(
        const std::vector<camera_core::ImageId>& images,
        std::vector<camera_models::frame_pinhole::FramePinholeNumericState>* states,
        std::string* error) const
    {
        if (states)
        {
            states->clear();
        }
        if (error)
        {
            error->clear();
        }
        if (!states)
        {
            if (error)
            {
                *error = "frame-pinhole numeric state output is null";
            }
            return false;
        }
        if (images.empty())
        {
            if (error)
            {
                *error = "frame-pinhole numeric state selection cannot be empty";
            }
            return false;
        }

        std::unordered_set<camera_core::ImageId> seen;
        seen.reserve(images.size());
        std::vector<camera_models::frame_pinhole::FramePinholeNumericState> resolvedStates;
        resolvedStates.reserve(images.size());
        for (const camera_core::ImageId& image : images)
        {
            if (!seen.insert(image).second)
            {
                if (error)
                {
                    *error = "duplicate frame-pinhole numeric state image: " + image.value();
                }
                return false;
            }

            camera_models::frame_pinhole::FramePinholeNumericState state;
            std::string stateError;
            if (!framePinholeStateForImage(image, &state, &stateError))
            {
                if (error)
                {
                    *error = stateError;
                }
                return false;
            }
            resolvedStates.push_back(std::move(state));
        }

        *states = std::move(resolvedStates);
        return true;
    }

    CameraProjectRuntimeResult CameraProjectRuntime::load(const QJsonObject& projectFiles,
                                                          const camera_core::CameraModelRegistry& registry)
    {
        CameraProjectRuntimeResult result;
        CameraProjectData data;
        if (!CameraProjectStore::load(projectFiles, &data, &result.errors))
        {
            return result;
        }

        QMap<QString, std::shared_ptr<const camera_core::CameraDefinition>> definitions;
        const QJsonArray definitionValues = data.definitions;
        for (int index = 0; index < definitionValues.size(); ++index)
        {
            const QJsonObject definition = definitionValues.at(index).toObject();
            const QString id = requiredString(definition, QStringLiteral("id"));
            const QString model = requiredString(definition, QStringLiteral("model_type"));
            const QString frame = requiredString(definition, QStringLiteral("frame"));
            const int schemaVersion = definition.value(QStringLiteral("schema_version")).toInt(-1);
            try
            {
                std::unique_ptr<camera_core::CameraDefinition> decoded = registry.createDefinition(
                    model.toStdString(),
                    camera_core::CameraDefinitionId(id.toStdString()),
                    xjw::coordinate_system::CoordinateFrameId(frame.toStdString()),
                    schemaVersion,
                    compactJson(definition.value(QStringLiteral("parameters")).toObject()).toStdString());
                definitions.insert(id, std::shared_ptr<const camera_core::CameraDefinition>(std::move(decoded)));
            }
            catch (const std::exception& exception)
            {
                appendError(
                    &result.errors, QStringLiteral("camera_definitions[%1] (%2)").arg(index).arg(id), exception);
            }
        }

        const QJsonArray instanceValues = data.instances;
        for (int index = 0; index < instanceValues.size(); ++index)
        {
            const QJsonObject instance = instanceValues.at(index).toObject();
            const QString id = requiredString(instance, QStringLiteral("id"));
            const QString image = requiredString(instance, QStringLiteral("image_uuid"));
            const QString definitionId = requiredString(instance, QStringLiteral("definition_id"));
            const auto definitionIt = definitions.constFind(definitionId);
            if (definitionIt == definitions.constEnd())
            {
                result.errors.append(
                    QStringLiteral("camera_instances[%1] (%2) references a definition that could not be decoded: %3")
                        .arg(index)
                        .arg(id)
                        .arg(definitionId));
                continue;
            }

            QJsonObject state = instance.value(QStringLiteral("state")).toObject();
            // Model factories receive one canonical instance-state object.  Keep
            // image size and pose in that object even though they are top-level
            // persistence fields, so factories do not need a second JSON API.
            state.insert(QStringLiteral("image_size"), instance.value(QStringLiteral("image_size")));
            if (instance.value(QStringLiteral("pose")).isObject())
            {
                state.insert(QStringLiteral("pose"), instance.value(QStringLiteral("pose")));
            }

            try
            {
                std::unique_ptr<camera_core::CameraInstance> decoded =
                    registry.createInstance(std::string(definitionIt.value()->modelType()),
                                            camera_core::CameraInstanceId(id.toStdString()),
                                            camera_core::ImageId(image.toStdString()),
                                            definitionIt.value(),
                                            compactJson(state).toStdString());
                std::shared_ptr<const camera_core::CameraInstance> shared(std::move(decoded));
                std::string addError;
                if (!result.instances.add(std::move(shared), &addError))
                {
                    result.errors.append(QStringLiteral("camera_instances[%1] (%2): %3")
                                             .arg(index)
                                             .arg(id)
                                             .arg(QString::fromStdString(addError)));
                }
            }
            catch (const std::exception& exception)
            {
                appendError(&result.errors, QStringLiteral("camera_instances[%1] (%2)").arg(index).arg(id), exception);
            }
        }

        if (!result.errors.isEmpty())
        {
            result.instances = camera_core::CameraInstanceSet{};
        }
        return result;
    }

} // namespace xjw::camera_project
