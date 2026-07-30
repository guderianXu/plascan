#include "DialogSettingStore.h"

#include "json/JsonObjectMerge.h"

#include <utility>

DialogSettingStore::DialogSettingStore(const QString &dialogKey, QObject *parent)
    : QObject(parent)
    , _dialogKey(dialogKey.trimmed())
{
}

void DialogSettingStore::setProjectPath(const QString &plascanPath)
{
    ProjectDialogJsonSettingBase::setProjectPath(plascanPath);
}

QJsonObject DialogSettingStore::load(QString *errorMessage) const
{
    if (_dialogKey.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("对话框设置键不能为空");
        }
        return QJsonObject();
    }

    return loadByKey(_dialogKey, errorMessage);
}

bool DialogSettingStore::save(const QJsonObject &settings, QString *errorMessage) const
{
    if (_dialogKey.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("对话框设置键不能为空");
        }
        return false;
    }

    const bool saved = saveByKey(_dialogKey, settings, errorMessage);
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
