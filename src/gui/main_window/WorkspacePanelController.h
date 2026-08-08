#pragma once

#include <QAction>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QPointer>
#include <QVector>
#include <QWidget>

#include "WorkspacePanelDescriptor.h"

class QDockWidget;
class QToolBar;

class WorkspacePanelController : public QObject
{
    Q_OBJECT
public:
    explicit WorkspacePanelController(QObject *parent = nullptr);

    bool registerDock(WorkspacePanelId id,
                      QAction *action,
                      QDockWidget *dock);
    bool registerToolBar(WorkspacePanelId id,
                         QAction *action,
                         QToolBar *toolBar);

    QJsonObject visibilitySnapshot() const;
    QList<QAction *> actions(WorkspacePanelKind kind) const;
    void applyVisibility(const QJsonObject &settings);
    void syncActions();
    void restoreDefaultVisibility();
    void ensureRequiredProjectPanelsVisible();
    void setPanelVisible(WorkspacePanelId id, bool visible, bool raise = true);

signals:
    void visibilitySettingChanged(WorkspacePanelId id, bool visible);

private:
    struct Entry
    {
        WorkspacePanelDescriptor descriptor;
        QPointer<QAction> action;
        QPointer<QWidget> widget;
        bool explicitlyVisible{};
    };

    bool registerWidget(WorkspacePanelId id,
                        QAction *action,
                        QWidget *widget);
    Entry *findEntry(WorkspacePanelId id);
    const Entry *findEntry(WorkspacePanelId id) const;
    void handleVisibilityChanged(WorkspacePanelId id);

    QVector<Entry> _entries;
    bool _applying{};
};
