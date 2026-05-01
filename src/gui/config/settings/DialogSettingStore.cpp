#include "DialogSettingStore.h"

#include "../JsonMergeUtil.h"

DialogSettingStore::DialogSettingStore(const QString &dialogKey, QObject *parent)
    : QObject(parent)
    , m_dialogKey(dialogKey.trimmed())
{
}

void DialogSettingStore::setProjectPath(const QString &plascanPath)
{
    ProjectDialogJsonSettingBase::setProjectPath(plascanPath);
}

QJsonObject DialogSettingStore::load() const
{
    if (m_dialogKey.isEmpty()) return QJsonObject();
    return loadByKey(m_dialogKey);
}

bool DialogSettingStore::save(const QJsonObject &settings) const
{
    if (m_dialogKey.isEmpty()) return false;
    return saveByKey(m_dialogKey, settings);
}

bool DialogSettingStore::merge(const QJsonObject &partial) const
{
    const QJsonObject current = load();
    return save(JsonMergeUtil::deepMerge(current, partial));
}
