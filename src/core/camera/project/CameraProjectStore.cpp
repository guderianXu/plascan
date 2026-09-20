#include "CameraProjectStore.h"

#include <QString>

#include <utility>

namespace xjw::camera_project
{

    CameraProjectError::CameraProjectError(const QStringList& errors)
        : std::runtime_error(errors.join(QStringLiteral("; ")).toStdString())
    {
    }

    CameraProjectValidationResult CameraProjectStore::validate(const QJsonObject& projectFiles,
                                                               const CameraProjectData& data)
    {
        return CameraProjectValidation::validate(
            data.definitions, data.instances, projectFiles.value(QStringLiteral("images")).toArray());
    }

    bool CameraProjectStore::load(const QJsonObject& projectFiles, CameraProjectData* data, QStringList* errors)
    {
        QStringList localErrors;
        QStringList* reportedErrors = errors ? errors : &localErrors;
        reportedErrors->clear();
        if (!data)
        {
            reportedErrors->append(QStringLiteral("camera project output is null"));
            return false;
        }

        if (!projectFiles.value(QStringLiteral("images")).isArray())
        {
            reportedErrors->append(QStringLiteral("project_files.images must be an array"));
        }

        CameraProjectData candidate;
        const QJsonValue definitions = projectFiles.value(QStringLiteral("camera_definitions"));
        const QJsonValue instances = projectFiles.value(QStringLiteral("camera_instances"));
        if (!definitions.isArray())
        {
            reportedErrors->append(QStringLiteral("project_files.camera_definitions must be an array"));
        }
        else
        {
            candidate.definitions = definitions.toArray();
        }
        if (!instances.isArray())
        {
            reportedErrors->append(QStringLiteral("project_files.camera_instances must be an array"));
        }
        else
        {
            candidate.instances = instances.toArray();
        }
        if (!reportedErrors->isEmpty())
        {
            return false;
        }

        const CameraProjectValidationResult validation = validate(projectFiles, candidate);
        if (!validation.ok())
        {
            *reportedErrors = validation.errors;
            return false;
        }
        *data = std::move(candidate);
        return true;
    }

    bool CameraProjectStore::save(QJsonObject* projectFiles, const CameraProjectData& data, QStringList* errors)
    {
        QStringList localErrors;
        QStringList* reportedErrors = errors ? errors : &localErrors;
        reportedErrors->clear();
        if (!projectFiles)
        {
            reportedErrors->append(QStringLiteral("camera project document is null"));
            return false;
        }

        const CameraProjectValidationResult validation = validate(*projectFiles, data);
        if (!validation.ok())
        {
            *reportedErrors = validation.errors;
            return false;
        }

        QJsonObject candidate = *projectFiles;
        candidate.insert(QStringLiteral("camera_definitions"), data.definitions);
        candidate.insert(QStringLiteral("camera_instances"), data.instances);
        *projectFiles = std::move(candidate);
        return true;
    }

} // namespace xjw::camera_project
