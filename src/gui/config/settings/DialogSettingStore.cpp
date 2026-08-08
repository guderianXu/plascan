#include "DialogSettingStore.h"

#include "io/JsonObjectFile.h"
#include "json/JsonObjectMerge.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <utility>

DialogSettingStore::DialogSettingStore(const QString &dialogKey, QObject *parent)
    : QObject(parent)
    , _dialogKey(dialogKey.trimmed())
{
}

void DialogSettingStore::setProjectPath(const QString &plascanPath)
{
    _plascanPath = plascanPath;
}

QString DialogSettingStore::dialogFilePath() const
{
    if (_plascanPath.trimmed().isEmpty())
    {
        return QString();
    }
    const QString root = xjw::common::project::ProjectIO::projectRootFromPlascan(_plascanPath);
    return root.isEmpty()
        ? QString()
        : QDir(root).filePath(QStringLiteral("project_dialog.json"));
}

QJsonObject DialogSettingStore::loadByKey(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (_dialogKey.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("对话框设置键不能为空");
        }
        return {};
    }

    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        return {};
    }
    const xjw::common::io::JsonObjectReadResult result =
        xjw::common::io::readJsonObjectFile(path);
    if (!result.success)
    {
        if (errorMessage)
        {
            *errorMessage = result.errorMessage;
        }
        return {};
    }
    return result.object.value(_dialogKey).toObject();
}

bool DialogSettingStore::saveByKey(const QJsonObject &value, QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (_dialogKey.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("对话框设置键不能为空");
        }
        return false;
    }

    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("未设置有效的项目路径");
        }
        return false;
    }

    const xjw::common::io::JsonObjectReadResult result =
        xjw::common::io::readJsonObjectFile(path);
    if (!result.success)
    {
        if (errorMessage)
        {
            *errorMessage = result.errorMessage;
        }
        return false;
    }

    QJsonObject root = result.object;
    root.insert(_dialogKey, value);
    return xjw::common::io::writeJsonObjectFileAtomic(path, root, errorMessage);
}

QJsonObject DialogSettingStore::load(QString *errorMessage) const
{
    return loadByKey(errorMessage);
}

bool DialogSettingStore::save(const QJsonObject &settings, QString *errorMessage) const
{
    const bool saved = saveByKey(settings, errorMessage);
    if (saved && _changeCallback)
    {
        _changeCallback();
    }
    return saved;
}

bool DialogSettingStore::merge(const QJsonObject &partial, QString *errorMessage) const
{
    QString load_error;
    const QJsonObject current = load(&load_error);
    if (!load_error.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = load_error;
        }
        return false;
    }

    return save(xjw::common::json::deepMergeObjects(current, partial), errorMessage);
}

void DialogSettingStore::setChangeCallback(std::function<void()> callback)
{
    _changeCallback = std::move(callback);
}
