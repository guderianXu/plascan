#pragma once

#include "CameraProjectValidation.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>

#include <stdexcept>

namespace xjw::camera_project
{

    struct CameraProjectData
    {
        QJsonArray definitions;
        QJsonArray instances;
    };

    class CameraProjectError : public std::runtime_error
    {
    public:
        explicit CameraProjectError(const QStringList& errors);
    };

    class CameraProjectStore
    {
    public:
        static bool load(const QJsonObject& projectFiles, CameraProjectData* data, QStringList* errors = nullptr);

        static bool save(QJsonObject* projectFiles, const CameraProjectData& data, QStringList* errors = nullptr);

        static CameraProjectValidationResult validate(const QJsonObject& projectFiles, const CameraProjectData& data);
    };

} // namespace xjw::camera_project
