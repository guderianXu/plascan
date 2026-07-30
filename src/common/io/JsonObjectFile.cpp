#include "io/JsonObjectFile.h"

#include "io/PathIO.h"

#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>

namespace xjw::common::io
{

JsonObjectReadResult readJsonObjectFile(const QString &path)
{
    JsonObjectReadResult result;
    const QString normalized_path = path.trimmed();
    if (normalized_path.isEmpty())
    {
        result.errorMessage = QStringLiteral("JSON 文件路径为空");
        return result;
    }

    const QFileInfo file_info(normalized_path);
    result.exists = file_info.exists();
    if (!result.exists)
    {
        result.success = true;
        return result;
    }

    QString read_error;
    const QByteArray bytes = readFileBytes(file_info.absoluteFilePath(), &read_error);
    if (!read_error.isEmpty())
    {
        result.errorMessage = read_error;
        return result;
    }

    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parse_error);
    if (parse_error.error != QJsonParseError::NoError)
    {
        result.errorMessage = QStringLiteral("无法解析 JSON 文件: %1 (%2)")
                                  .arg(file_info.absoluteFilePath(), parse_error.errorString());
        return result;
    }
    if (!document.isObject())
    {
        result.errorMessage = QStringLiteral("JSON 根节点必须是对象: %1").arg(file_info.absoluteFilePath());
        return result;
    }

    result.success = true;
    result.object = document.object();
    return result;
}

bool writeJsonObjectFileAtomic(const QString &path, const QJsonObject &object, QString *errorMessage)
{
    const QString normalized_path = path.trimmed();
    if (normalized_path.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("JSON 文件路径为空");
        }
        return false;
    }

    return writeFileBytesAtomic(normalized_path,
                                QJsonDocument(object).toJson(QJsonDocument::Indented),
                                errorMessage);
}

} // namespace xjw::common::io
