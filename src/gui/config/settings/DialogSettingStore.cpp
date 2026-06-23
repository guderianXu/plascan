#include "DialogSettingStore.h"

#include "../JsonMergeUtil.h"

DialogSettingStore::DialogSettingStore(const QString &dialogKey, QObject *parent)
    : QObject(parent)
    , _dialogKey(dialogKey.trimmed())
{
}

void DialogSettingStore::setProjectPath(const QString &plascanPath)
{
    ProjectDialogJsonSettingBase::setProjectPath(plascanPath);
}

QJsonObject DialogSettingStore::load() const
{
    if (_dialogKey.isEmpty())
    {
        return QJsonObject();
    }

    return loadByKey(_dialogKey);
}

bool DialogSettingStore::save(const QJsonObject &settings) const
{
    if (_dialogKey.isEmpty())
    {
        return false;
    }

    return saveByKey(_dialogKey, settings);
}

bool DialogSettingStore::merge(const QJsonObject &partial) const
{
    const QJsonObject current = load();
    return save(JsonMergeUtil::deepMerge(current, partial));
}
