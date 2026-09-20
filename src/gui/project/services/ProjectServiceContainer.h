#pragma once

#include "project/tasks/ProjectTaskContext.h"

#include <QObject>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <functional>
#include <memory>

class ProjectData;
class ProjectUiMessageAdapter;
class QWidget;
class FileDialogStateManager;
class ProjectLifecycleService;

namespace xjw::gui::project
{

    class ProjectResourceCleanupCoordinator;
    class ProjectResourceService;
    class ProjectSession;
    class ProjectTaskOrchestrator;

    using TiePointResultWriter = std::function<bool(const ProjectTaskContext& expected,
                                                    const QString& sparseCloudPath,
                                                    int sparsePointCount,
                                                    const QStringList& selectedImages,
                                                    const QString& outputDir,
                                                    const QJsonObject& extraRecord)>;

    class ProjectServiceContainer final : public QObject
    {
        Q_OBJECT

    public:
        ProjectServiceContainer(ProjectData* projectData,
                                QWidget* parentWidget,
                                TiePointResultWriter tiePointResultWriter = {},
                                QObject* parent = nullptr);
        ~ProjectServiceContainer() override;

        ProjectSession& session() const;
        ProjectUiMessageAdapter& messages() const;
        ProjectLifecycleService& lifecycle() const;
        ProjectResourceService& resources() const;
        ProjectResourceCleanupCoordinator& cleanup() const;
        ProjectTaskOrchestrator& tasks() const;

        void setDirectoryAccessors(std::function<QString(const QString& key)> getLastDir,
                                   std::function<void(const QString& key, const QString& dir)> saveLastDir);

    private:
        std::unique_ptr<ProjectSession> _session;
        std::unique_ptr<ProjectUiMessageAdapter> _messages;
        std::unique_ptr<ProjectResourceCleanupCoordinator> _cleanup;
        std::unique_ptr<ProjectLifecycleService> _lifecycle;
        std::unique_ptr<ProjectResourceService> _resources;
        std::unique_ptr<ProjectTaskOrchestrator> _tasks;
    };

} // namespace xjw::gui::project
