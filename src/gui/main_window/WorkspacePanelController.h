#pragma once

#include <QAction>
#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QVector>
#include <QWidget>

class QDockWidget;
class QToolBar;

class WorkspacePanelController : public QObject
{
    Q_OBJECT
public:
    explicit WorkspacePanelController(QObject *parent = nullptr);

    void registerDock(const QString &settingKey,
                      QAction *action,
                      QDockWidget *dock,
                      bool defaultVisible);
    void registerToolBar(const QString &settingKey,
                         QAction *action,
                         QToolBar *toolBar,
                         bool defaultVisible);

    QJsonObject visibilitySnapshot() const;
    void applyVisibility(const QJsonObject &settings);
    void syncActions();

signals:
    void visibilitySettingChanged(const QString &settingKey, bool visible);

private:
    struct Entry
    {
        QString settingKey;
        QPointer<QAction> action;
        QPointer<QWidget> widget;
        bool defaultVisible{};
        bool explicitlyVisible{};
    };

    void registerWidget(const QString &settingKey,
                        QAction *action,
                        QWidget *widget,
                        bool defaultVisible);
    void handleVisibilityChanged(QWidget *widget);

    QVector<Entry> _entries;
    bool _applying{};
};
