#pragma once

#include <QJsonArray>
#include <QStringList>

namespace xjw::camera_project
{

    struct CameraProjectValidationResult
    {
        QStringList errors;

        bool ok() const noexcept
        {
            return errors.isEmpty();
        }
    };

    class CameraProjectValidation
    {
    public:
        static constexpr int CurrentInstanceSchemaVersion = 1;

        static CameraProjectValidationResult
        validate(const QJsonArray& definitions, const QJsonArray& instances, const QJsonArray& images);
    };

} // namespace xjw::camera_project
