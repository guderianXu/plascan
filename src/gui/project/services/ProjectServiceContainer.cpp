#include "ProjectServiceContainer.h"

#include "ProjectLifecycleService.h"
#include "ProjectResourceCleanupCoordinator.h"
#include "ProjectResourceRecoveryBinding.h"
#include "ProjectResourceService.h"
#include "ProjectSession.h"
#include "image/GenerateMaskDialog.h"
#include "project/tasks/ProjectTaskOrchestrator.h"
#include "ProjectUiMessageAdapter.h"
#include "Logger.h"

#include <QDialog>
#include <QDir>
#include <QPointer>

namespace xjw::gui::project
{

    ProjectServiceContainer::ProjectServiceContainer(ProjectData* projectData,
                                                     QWidget* parentWidget,
                                                     TiePointResultWriter tiePointResultWriter,
                                                     QObject* parent)
        : QObject(parent), _session(std::make_unique<ProjectSession>(projectData)),
          _messages(std::make_unique<QtProjectUiMessageAdapter>(parentWidget)),
          _cleanup(std::make_unique<ProjectResourceCleanupCoordinator>(_session.get(), _messages.get())),
          _lifecycle(std::make_unique<ProjectLifecycleService>(
              projectData, _session.get(), _messages.get(), _cleanup.get(), parentWidget)),
          _resources(
              std::make_unique<ProjectResourceService>(_session.get(), _messages.get(), _cleanup.get(), parentWidget)),
          _tasks(std::make_unique<ProjectTaskOrchestrator>(
              _session.get(),
              _messages.get(),
              [guarded_parent = QPointer<QWidget>(parentWidget)](
                  const QStringList& selectedImages, const QString& currentImage) -> std::optional<QJsonObject>
              {
                  GenerateMaskDialog dialog(selectedImages, currentImage, guarded_parent.data());
                  if (dialog.exec() != QDialog::Accepted)
                  {
                      return std::nullopt;
                  }
                  return dialog.collectSettings();
              },
              nullptr))
    {
        xjw::app::project::ProjectResourceRecoveryBinding::install(projectData);

        if (!tiePointResultWriter)
        {
            tiePointResultWriter = [this](const ProjectTaskContext& expected,
                                          const QString& sparseCloudPath,
                                          int sparsePointCount,
                                          const QStringList& selectedImages,
                                          const QString& outputDir,
                                          const QJsonObject& extraRecord)
            {
                if (!expected.cancelFlag || expected.cancelFlag->load(std::memory_order_relaxed) ||
                    !_session->isCurrent(expected.session))
                {
                    return false;
                }
                const TiePointMutationResult result = _session->replaceTiePointResult(
                    expected.session, sparseCloudPath, sparsePointCount, selectedImages, outputDir, extraRecord);
                const bool still_current = expected.cancelFlag &&
                                           !expected.cancelFlag->load(std::memory_order_relaxed) &&
                                           _session->isCurrent(expected.session);
                if (!result.success)
                {
                    if (still_current)
                    {
                        LOG_ERROR(QStringLiteral("替换当前连接点失败: %1").arg(result.errorMessage));
                        if (expected.taskId != QLatin1String("bundle_adjust"))
                        {
                            _messages->warning(nullptr, QStringLiteral("连接点写入失败"), result.errorMessage);
                        }
                    }
                    return false;
                }
                if (still_current && !result.cleanupWarnings.isEmpty())
                {
                    LOG_WARN(QStringLiteral("当前连接点已更新，但旧文件清理失败: %1")
                                 .arg(result.cleanupWarnings.join(QStringLiteral("；"))));
                }
                if (still_current)
                {
                    LOG_INFO(QStringLiteral("空三代次已更新为 %1；旧深度图、稠密点云、模型、DEM 和正射结果已失效")
                                 .arg(result.reconstructionGenerationId));
                }
                return true;
            };
        }
        if (tiePointResultWriter)
        {
            // Both task controllers receive the same narrow write-back boundary.
            _tasks->setTiePointResultWriter(std::move(tiePointResultWriter));
        }

        _resources->setPortableExportLauncher([this](const QString& outputPath, QString* errorMessage)
                                              { return _lifecycle->startPortableExport(outputPath, errorMessage); });
        _resources->setBundleAdjustLauncher(
            [this](const QStringList& images,
                   const QString& outputDir,
                   int threads,
                   bool dryRun,
                   const QJsonObject& extraSettings)
            { return _tasks->startBundleAdjustAsync(images, outputDir, threads, dryRun, extraSettings); });
        const QPointer<ProjectTaskOrchestrator> task_orchestrator(_tasks.get());
        _lifecycle->setSessionDrainHandler(
            [task_orchestrator](std::function<void()> continuation) {
                return task_orchestrator && task_orchestrator->cancelAndDrainForSessionChange(std::move(continuation));
            });
        _lifecycle->setSessionFutureTracker(
            [task_orchestrator](QFuture<void> future)
            {
                if (task_orchestrator)
                {
                    task_orchestrator->trackSessionFuture(std::move(future));
                }
            });
        _tasks->setBundleAdjustPostExternalCommitObserver(
            [this](const ProjectTaskContext& expected)
            {
                if (expected.cancelFlag && !expected.cancelFlag->load(std::memory_order_relaxed) &&
                    _session->isCurrent(expected.session))
                {
                    _resources->refreshReconstructionQualityReport();
                }
            });
        connect(_tasks.get(),
                &ProjectTaskOrchestrator::reconstructionQualityRefreshRequested,
                _resources.get(),
                &ProjectResourceService::refreshReconstructionQualityReport);
        connect(_tasks.get(),
                &ProjectTaskOrchestrator::sparseMatchPairReady,
                _session.get(),
                &ProjectSession::matchPairReady);
        connect(_tasks.get(),
                &ProjectTaskOrchestrator::cameraMatchPairReady,
                _session.get(),
                &ProjectSession::matchPairReady);
        connect(_cleanup.get(),
                &ProjectResourceCleanupCoordinator::tiePointDeletionFinished,
                _resources.get(),
                &ProjectResourceService::refreshReconstructionQualityReport);

        setDirectoryAccessors([](const QString&) { return QDir::homePath(); }, [](const QString&, const QString&) {});
    }

    ProjectServiceContainer::~ProjectServiceContainer() = default;

    ProjectSession& ProjectServiceContainer::session() const
    {
        return *_session;
    }

    ProjectUiMessageAdapter& ProjectServiceContainer::messages() const
    {
        return *_messages;
    }

    ProjectLifecycleService& ProjectServiceContainer::lifecycle() const
    {
        return *_lifecycle;
    }

    ProjectResourceService& ProjectServiceContainer::resources() const
    {
        return *_resources;
    }

    ProjectResourceCleanupCoordinator& ProjectServiceContainer::cleanup() const
    {
        return *_cleanup;
    }

    ProjectTaskOrchestrator& ProjectServiceContainer::tasks() const
    {
        return *_tasks;
    }

    void ProjectServiceContainer::setDirectoryAccessors(
        std::function<QString(const QString& key)> getLastDir,
        std::function<void(const QString& key, const QString& dir)> saveLastDir)
    {
        _tasks->setDirectoryAccessors(getLastDir, saveLastDir);
        _lifecycle->setDirectoryAccessors(getLastDir, saveLastDir);
        _resources->setDirectoryAccessors(getLastDir, saveLastDir);
    }

} // namespace xjw::gui::project
