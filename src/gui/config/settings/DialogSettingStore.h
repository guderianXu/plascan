#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>

#include "ProjectDialogJsonSettingBase.h"

class DialogSettingStore : public QObject, public ProjectDialogJsonSettingBase
{
    Q_OBJECT
public:
    explicit DialogSettingStore(const QString &dialogKey, QObject *parent = nullptr);

    QString key() const { return m_dialogKey; }

    void setProjectPath(const QString &plascanPath);
    QJsonObject load() const;
    bool save(const QJsonObject &settings) const;
    bool merge(const QJsonObject &partial) const;

private:
    QString m_dialogKey;
};
