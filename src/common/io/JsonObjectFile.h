#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::common::io
{

struct JsonObjectReadResult
{
    bool exists = false;
    bool success = false;
    QJsonObject object;
    QString errorMessage;
};

/**
 * @brief Reads a JSON object file without treating a missing file as an error.
 */
JsonObjectReadResult readJsonObjectFile(const QString &path);

/**
 * @brief Atomically writes a JSON object using indented UTF-8 JSON.
 */
bool writeJsonObjectFileAtomic(const QString &path,
                               const QJsonObject &object,
                               QString *errorMessage = nullptr);

} // namespace xjw::common::io
