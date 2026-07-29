#pragma once

#include <QMetaType>
#include <QString>

enum class WorkspacePanelId
{
    Workspace,
    Properties,
    Photos,
    Log,
    MainToolbar
};

enum class WorkspacePanelKind
{
    Dock,
    ToolBar
};

Q_DECLARE_METATYPE(WorkspacePanelId)

struct WorkspacePanelDescriptor
{
    WorkspacePanelId id;
    WorkspacePanelKind kind;
    QString settingKey;
    bool defaultVisible;
    bool requiredForProject;
};

inline WorkspacePanelDescriptor workspacePanelDescriptor(WorkspacePanelId id)
{
    switch (id)
    {
    case WorkspacePanelId::Workspace:
        return {id,
                WorkspacePanelKind::Dock,
                QStringLiteral("workspace_visible"),
                true,
                true};
    case WorkspacePanelId::Properties:
        return {id,
                WorkspacePanelKind::Dock,
                QStringLiteral("properties_visible"),
                true,
                true};
    case WorkspacePanelId::Photos:
        return {id,
                WorkspacePanelKind::Dock,
                QStringLiteral("photos_visible"),
                true,
                true};
    case WorkspacePanelId::Log:
        return {id,
                WorkspacePanelKind::Dock,
                QStringLiteral("log_visible"),
                false,
                false};
    case WorkspacePanelId::MainToolbar:
        return {id,
                WorkspacePanelKind::ToolBar,
                QStringLiteral("main_toolbar_visible"),
                true,
                false};
    }
    return {id, WorkspacePanelKind::Dock, QString(), false, false};
}
