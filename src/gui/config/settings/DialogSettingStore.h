#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>

#include <functional>

#include "ProjectDialogJsonSettingBase.h"

class DialogSettingStore : public QObject, public ProjectDialogJsonSettingBase
{
    Q_OBJECT
public:
    explicit DialogSettingStore(const QString &dialogKey, QObject *parent = nullptr);

    QString key() const { return _dialogKey; }

    void setProjectPath(const QString &plascanPath);
    QJsonObject load(QString *errorMessage = nullptr) const;
    bool save(const QJsonObject &settings, QString *errorMessage = nullptr) const;
    bool merge(const QJsonObject &partial, QString *errorMessage = nullptr) const;
    void setChangeCallback(std::function<void()> callback);

private:
    QString _dialogKey;
    std::function<void()> _changeCallback;
};
