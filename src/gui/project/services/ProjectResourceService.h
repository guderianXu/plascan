#pragma once

#include "project/support/ProjectSessionContext.h"
#include "tasks/GuiTaskRunner.h"

#include <QFuture>
#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QStringList>

#include <functional>

class ProjectData;
class ProjectUiMessageAdapter;
class QWidget;

namespace xjw::gui::project
{

    class ProjectResourceCleanupCoordinator;
    class ProjectSession;

    enum class OperationStatus
    {
        Success,
        Cancelled,
        Failed
    };

    struct OperationResult
    {
        OperationStatus status = OperationStatus::Failed;
        QString message;

        bool succeeded() const
        {
            return status == OperationStatus::Success;
        }
    };

    class ProjectResourceService final : public QObject
    {
        Q_OBJECT

    public:
        ProjectResourceService(ProjectSession* session,
                               ProjectUiMessageAdapter* messages,
                               ProjectResourceCleanupCoordinator* cleanup,
                               QWidget* parentWidget,
                               QObject* parent = nullptr);
        ~ProjectResourceService() override;

        void setDirectoryAccessors(std::function<QString(const QString& key)> getLastDir,
                                   std::function<void(const QString& key, const QString& dir)> saveLastDir);
        void setPortableExportLauncher(std::function<bool(const QString& outputPath, QString* errorMessage)> launcher);
        void setBundleAdjustLauncher(std::function<bool(const QStringList& images,
                                                        const QString& outputDir,
                                                        int threads,
                                                        bool dryRun,
                                                        const QJsonObject& extraSettings)> launcher);

        OperationResult addPhoto();
        OperationResult addPhotos(const QStringList& imagePaths, const QString& sourceLabel = QString());
        OperationResult addFolder();
        OperationResult addFolder(const QString& folderPath);
        OperationResult importPointCloud();
        OperationResult importModel();
        OperationResult importProjectAsset(const QString& selectedPath, bool modelAsset);
        OperationResult removeResources(const QStringList& resourcePaths);
        OperationResult
        deleteGeneratedData(const QString& section, const QStringList& resourcePaths, QWidget* requestWidget = nullptr);
        OperationResult importReferenceDataset();
        OperationResult registerReferenceDataset(const QString& path,
                                                 const QString& type = QString(),
                                                 const QString& role = QStringLiteral("validation"));
        OperationResult importSurveyControlCsv();
        OperationResult importSurveyControlCsv(const QString& csvPath);
        void openSurveyControlDialog();
        void runReferenceQualityCheck();
        void prepareReferenceTerrainBundleAdjust();
        void refreshReconstructionQualityReport();
        OperationResult packResource(const QString& resourcePath);
        OperationResult exportPortableProject(const QString& outputPath);

        bool isImageImportActive() const;
        bool isPortableExportInProgress() const;
        bool isBusy() const;
        ProjectSession* session() const noexcept
        {
            return _session;
        }

    signals:
        void imageImportProgressChanged(const QString& stage, int done, int total);
        void imageImportFinished(bool success, const QString& message);
        void portableExportFinished(bool success, const QString& outputPath, const QString& message);
        void surveyControlChanged();
        void resourceChanged();

    private:
        QString readLastDir(const QString& key) const;
        void writeLastDir(const QString& key, const QString& directory) const;
        OperationResult failed(const QString& message) const;
        OperationResult cancelled(const QString& message = QString()) const;
        OperationResult success(const QString& message = QString()) const;
        bool requireProject(const QString& operation);
        void presentFailure(const QString& operation, const QString& message) const;
        QString normalizedPath(const QString& path) const;
        QStringList normalizedImagePaths(const QStringList& imagePaths) const;
        bool isCurrent(const xjw::gui::project::ProjectSessionContext& context) const;
        OperationResult
        startImageImportTask(const QStringList& imagePaths, const QString& sourceLabel, bool scanFolder);
        bool beginTask(const QString& operation);
        void finishTask();
        void cancelTask();

        ProjectSession* _session = nullptr;
        ProjectUiMessageAdapter* _messages = nullptr;
        ProjectResourceCleanupCoordinator* _cleanup = nullptr;
        QWidget* _parentWidget = nullptr;
        std::function<QString(const QString& key)> _getLastDir;
        std::function<void(const QString& key, const QString& dir)> _saveLastDir;
        bool _imageImportActive = false;
        std::function<bool(const QString& outputPath, QString* errorMessage)> _portableExportLauncher;
        std::function<bool(const QStringList& images,
                           const QString& outputDir,
                           int threads,
                           bool dryRun,
                           const QJsonObject& extraSettings)>
            _bundleAdjustLauncher;
        bool _taskActive = false;
        QString _taskOperation;
        xjw::gui::project::ProjectSessionContext _taskContext;
        xjw::gui::tasks::TaskCancellationSource _cancellationSource;
        xjw::gui::tasks::TaskCancellationToken _taskToken = _cancellationSource.reset();
        QFuture<void> _taskFuture;
    };

} // namespace xjw::gui::project

using OperationStatus = xjw::gui::project::OperationStatus;
using OperationResult = xjw::gui::project::OperationResult;
