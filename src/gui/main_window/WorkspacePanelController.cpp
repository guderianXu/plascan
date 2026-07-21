#include "WorkspacePanelController.h"

#include <QDockWidget>
#include <QSignalBlocker>
#include <QToolBar>

WorkspacePanelController::WorkspacePanelController(QObject *parent)
    : QObject(parent)
{
}

void WorkspacePanelController::registerDock(const QString &settingKey,
                                            QAction *action,
                                            QDockWidget *dock,
                                            bool defaultVisible)
{
    registerWidget(settingKey, action, dock, defaultVisible);
    if (!dock)
    {
        return;
    }

    connect(dock, &QDockWidget::visibilityChanged, this, [this, dock](bool visible)
    {
        Q_UNUSED(visible)
        handleVisibilityChanged(dock);
    });
}

void WorkspacePanelController::registerToolBar(const QString &settingKey,
                                               QAction *action,
                                               QToolBar *toolBar,
                                               bool defaultVisible)
{
    registerWidget(settingKey, action, toolBar, defaultVisible);
    if (!toolBar)
    {
        return;
    }

    connect(toolBar, &QToolBar::visibilityChanged, this, [this, toolBar](bool visible)
    {
        Q_UNUSED(visible)
        handleVisibilityChanged(toolBar);
    });
}

QJsonObject WorkspacePanelController::visibilitySnapshot() const
{
    QJsonObject settings;
    for (const Entry &entry : _entries)
    {
        if (!entry.widget || entry.settingKey.isEmpty())
        {
            continue;
        }
        settings[entry.settingKey] = !entry.widget->isHidden();
    }
    return settings;
}

void WorkspacePanelController::applyVisibility(const QJsonObject &settings)
{
    _applying = true;
    for (Entry &entry : _entries)
    {
        if (!entry.widget)
        {
            continue;
        }

        const bool visible = settings.contains(entry.settingKey)
            ? settings.value(entry.settingKey).toBool(entry.defaultVisible)
            : entry.defaultVisible;
        if (entry.action)
        {
            const QSignalBlocker blocker(entry.action);
            entry.action->setChecked(visible);
        }
        entry.explicitlyVisible = visible;
        entry.widget->setVisible(visible);
    }
    _applying = false;
}

void WorkspacePanelController::syncActions()
{
    for (Entry &entry : _entries)
    {
        if (!entry.action || !entry.widget)
        {
            continue;
        }
        const QSignalBlocker blocker(entry.action);
        entry.explicitlyVisible = !entry.widget->isHidden();
        entry.action->setChecked(entry.explicitlyVisible);
    }
}

void WorkspacePanelController::registerWidget(const QString &settingKey,
                                              QAction *action,
                                              QWidget *widget,
                                              bool defaultVisible)
{
    if (settingKey.isEmpty() || !action || !widget)
    {
        return;
    }

    action->setCheckable(true);
    {
        const QSignalBlocker blocker(action);
        action->setChecked(!widget->isHidden());
    }
    connect(action, &QAction::toggled, widget, [widget](bool visible)
    {
        widget->setVisible(visible);
        if (visible)
        {
            widget->raise();
        }
    });

    _entries.push_back(Entry{settingKey,
                             action,
                             widget,
                             defaultVisible,
                             !widget->isHidden()});
}

void WorkspacePanelController::handleVisibilityChanged(QWidget *widget)
{
    for (Entry &entry : _entries)
    {
        if (entry.widget != widget)
        {
            continue;
        }
        const bool explicitlyVisible = !widget->isHidden();
        if (entry.action)
        {
            const QSignalBlocker blocker(entry.action);
            entry.action->setChecked(explicitlyVisible);
        }
        if (entry.explicitlyVisible == explicitlyVisible)
        {
            return;
        }
        entry.explicitlyVisible = explicitlyVisible;
        if (!_applying)
        {
            emit visibilitySettingChanged(entry.settingKey, explicitlyVisible);
        }
        return;
    }
}
