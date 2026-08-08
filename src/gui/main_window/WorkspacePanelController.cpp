#include "WorkspacePanelController.h"

#include <QDockWidget>
#include <QSignalBlocker>
#include <QToolBar>

WorkspacePanelController::WorkspacePanelController(QObject *parent)
    : QObject(parent)
{
}

bool WorkspacePanelController::registerDock(WorkspacePanelId id,
                                            QAction *action,
                                            QDockWidget *dock)
{
    if (!registerWidget(id, action, dock))
    {
        return false;
    }

    connect(dock, &QDockWidget::visibilityChanged, this, [this, id](bool visible)
    {
        Q_UNUSED(visible)
        handleVisibilityChanged(id);
    });
    return true;
}

bool WorkspacePanelController::registerToolBar(WorkspacePanelId id,
                                               QAction *action,
                                               QToolBar *toolBar)
{
    if (!registerWidget(id, action, toolBar))
    {
        return false;
    }

    connect(toolBar, &QToolBar::visibilityChanged, this, [this, id](bool visible)
    {
        Q_UNUSED(visible)
        handleVisibilityChanged(id);
    });
    return true;
}

QJsonObject WorkspacePanelController::visibilitySnapshot() const
{
    QJsonObject settings;
    for (const Entry &entry : _entries)
    {
        if (!entry.widget || entry.descriptor.settingKey.isEmpty())
        {
            continue;
        }
        settings[entry.descriptor.settingKey] = !entry.widget->isHidden();
    }
    return settings;
}

QList<QAction *> WorkspacePanelController::actions(
    WorkspacePanelKind kind) const
{
    QList<QAction *> result;
    for (const Entry &entry : _entries)
    {
        if (entry.descriptor.kind == kind && entry.action)
        {
            result.push_back(entry.action);
        }
    }
    return result;
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

        const QString &settingKey = entry.descriptor.settingKey;
        const bool visible = settings.contains(settingKey)
            ? settings.value(settingKey).toBool(entry.descriptor.defaultVisible)
            : entry.descriptor.defaultVisible;
        setPanelVisible(entry.descriptor.id, visible, false);
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

void WorkspacePanelController::restoreDefaultVisibility()
{
    applyVisibility(QJsonObject());
}

void WorkspacePanelController::ensureRequiredProjectPanelsVisible()
{
    for (const Entry &entry : _entries)
    {
        if (entry.descriptor.requiredForProject)
        {
            setPanelVisible(entry.descriptor.id, true);
        }
    }
}

void WorkspacePanelController::setPanelVisible(WorkspacePanelId id,
                                               bool visible,
                                               bool raise)
{
    Entry *entry = findEntry(id);
    if (!entry || !entry->widget)
    {
        return;
    }

    const bool changed = entry->explicitlyVisible != visible;
    entry->explicitlyVisible = visible;
    if (entry->action)
    {
        const QSignalBlocker blocker(entry->action);
        entry->action->setChecked(visible);
    }
    entry->widget->setVisible(visible);
    if (visible && raise)
    {
        entry->widget->raise();
    }
    if (changed && !_applying)
    {
        emit visibilitySettingChanged(id, visible);
    }
}

bool WorkspacePanelController::registerWidget(WorkspacePanelId id,
                                              QAction *action,
                                              QWidget *widget)
{
    const WorkspacePanelDescriptor descriptor = workspacePanelDescriptor(id);
    if (descriptor.settingKey.isEmpty() || !action || !widget || findEntry(id))
    {
        return false;
    }

    action->setCheckable(true);
    {
        const QSignalBlocker blocker(action);
        action->setChecked(!widget->isHidden());
    }
    connect(action, &QAction::toggled, this, [this, id](bool visible)
    {
        setPanelVisible(id, visible);
    });

    _entries.push_back(Entry{descriptor,
                             action,
                             widget,
                             !widget->isHidden()});
    return true;
}

WorkspacePanelController::Entry *WorkspacePanelController::findEntry(
    WorkspacePanelId id)
{
    for (Entry &entry : _entries)
    {
        if (entry.descriptor.id == id)
        {
            return &entry;
        }
    }
    return nullptr;
}

const WorkspacePanelController::Entry *WorkspacePanelController::findEntry(
    WorkspacePanelId id) const
{
    for (const Entry &entry : _entries)
    {
        if (entry.descriptor.id == id)
        {
            return &entry;
        }
    }
    return nullptr;
}

void WorkspacePanelController::handleVisibilityChanged(WorkspacePanelId id)
{
    Entry *entry = findEntry(id);
    if (!entry || !entry->widget)
    {
        return;
    }

    const bool explicitlyVisible = !entry->widget->isHidden();
    if (entry->action)
    {
        const QSignalBlocker blocker(entry->action);
        entry->action->setChecked(explicitlyVisible);
    }
    if (entry->explicitlyVisible == explicitlyVisible)
    {
        return;
    }
    entry->explicitlyVisible = explicitlyVisible;
    if (!_applying)
    {
        emit visibilitySettingChanged(id, explicitlyVisible);
    }
}
