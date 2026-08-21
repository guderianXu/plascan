#include "CliJsonIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonParseError>
#include <QMap>
#include <QSaveFile>

namespace xjw::cli
{

bool readJsonFile(const QString &path, QJsonObject *object, QString *errorMessage)
{
    if (!object)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：JSON 输出对象为空");
        }
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 JSON 文件: %1").arg(path);
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("JSON 文件不是有效对象: %1 (%2)")
                                .arg(path, parseError.errorString());
        }
        return false;
    }
    *object = document.object();
    return true;
}

bool writeJsonFile(const QString &path, const QJsonObject &object, QString *errorMessage)
{
    const QFileInfo info(path);
    if (!QDir().mkpath(info.absolutePath()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建 JSON 输出目录: %1").arg(info.absolutePath());
        }
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入 JSON 文件: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    if (!file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("提交 JSON 文件失败: %1").arg(path);
        }
        return false;
    }
    return true;
}

QJsonArray latestJsonObjectsByNonNegativeIntegerKey(const QJsonArray &objects,
                                                    const QString &key)
{
    QMap<int, QJsonObject> latest_by_key;
    QJsonArray unkeyed_objects;
    for (const QJsonValue &value : objects)
    {
        if (!value.isObject())
        {
            continue;
        }

        const QJsonObject object = value.toObject();
        const QJsonValue key_value = object.value(key);
        const int integer_key = key_value.isDouble() ? key_value.toInt(-1) : -1;
        if (integer_key < 0)
        {
            unkeyed_objects.append(object);
            continue;
        }
        latest_by_key.insert(integer_key, object);
    }

    QJsonArray result;
    for (auto iterator = latest_by_key.cbegin();
         iterator != latest_by_key.cend();
         ++iterator)
    {
        result.append(iterator.value());
    }
    for (const QJsonValue &value : unkeyed_objects)
    {
        result.append(value);
    }
    return result;
}

void writeJson(FILE *stream, const QJsonObject &object, QJsonDocument::JsonFormat format)
{
    if (!stream)
    {
        return;
    }
    const QByteArray bytes = QJsonDocument(object).toJson(format);
    std::fwrite(bytes.constData(), 1, static_cast<size_t>(bytes.size()), stream);
    if (!bytes.endsWith('\n'))
    {
        std::fputc('\n', stream);
    }
    std::fflush(stream);
}

} // namespace xjw::cli
