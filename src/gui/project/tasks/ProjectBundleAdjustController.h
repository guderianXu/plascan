#pragma once

#include "ProjectTaskContext.h"
#include "project/ProjectSessionModel.h"
#include "project/support/ProjectBundleAdjustExecution.h"

#include <QFuture>
#include <QJsonObject>
#include <QMap>
#include <QObject>
#include <QStringList>
#include <QVector>

#include <functional>

class ProjectUiMessageAdapter;

namespace xjw::gui::project
{

    class ProjectSession;
    class ProjectTaskOrchestrator;

    class ProjectBundleAdjustController final : public QObject
    {
        Q_OBJECT

    public:
        using ExecutionRunner = std::function<BundleAdjustExecutionResult(const QJsonObject& coreData,
                                                                          const QString& projectPath,
                                                                          const QStringList& selectedImages,
                                                                          int minMatches,
                                                                          xjw::gui::BaServiceOptions options)>;
        using TiePointResultWriter = std::function<bool(const ProjectTaskContext& expected,
                                                        const QString& sparseCloudPath,
                                                        int sparsePointCount,
                                                        const QStringList& selectedImages,
                                                        const QString& outputDir,
                                                        const QJsonObject& extraRecord)>;
        using PostExternalCommitObserver = std::function<void(const ProjectTaskContext& expected)>;

        ProjectBundleAdjustController(ProjectSession* session,
                                      ProjectUiMessageAdapter* messages,
                                      QObject* parent = nullptr);
        ~ProjectBundleAdjustController() override;

        bool startAsync(const QStringList& images,
                        const QString& outputDir,
                        int threads,
                        bool dryRun,
                        const QJsonObject& extraSettings);
        bool acceptPreview(QString* errorMessage = nullptr);
        void discardPreview();
        bool cancel();
        bool isRunning() const noexcept;
        bool hasPendingPreview() const noexcept;
        ProjectTaskContext taskContext() const;
        void waitForFinished();

        void setExecutionRunnerForTesting(ExecutionRunner runner);
        void setTiePointResultWriter(TiePointResultWriter writer);
        void setPostExternalCommitObserver(PostExternalCommitObserver observer);

    signals:
        void progressChanged(const QString& stage, int percent);
        void previewReady(const QJsonObject& preview);
        void finished(bool success, const QString& message);
        void error(const QString& message);

    private:
        friend class ProjectTaskOrchestrator;

        using AdmissionRelease = std::function<void()>;
        using StartAdmissionProvider = std::function<AdmissionRelease()>;
        using PreviewAdmissionPredicate = std::function<bool(QString* errorMessage)>;

        ProjectBundleAdjustController(ProjectSession* session,
                                      ProjectUiMessageAdapter* messages,
                                      StartAdmissionProvider startAdmissionProvider,
                                      PreviewAdmissionPredicate previewAdmissionPredicate,
                                      QObject* parent);
        bool buildOptions(const QStringList& images,
                          const QString& outputDir,
                          int threads,
                          bool dryRun,
                          const QJsonObject& extraSettings,
                          xjw::gui::BaServiceOptions* options,
                          QString* errorMessage) const;
        bool isCurrentTask(const ProjectTaskContext& context, bool allowCancelled = false) const;
        bool isCurrentPreview(const ProjectTaskContext& context) const;
        void handleExecutionFinished(const ProjectTaskContext& context,
                                     const BundleAdjustExecutionResult& executionResult,
                                     bool dryRun);
        void handleExecutionError(const ProjectTaskContext& context, const QString& message);
        bool commitTerminal(const ProjectTaskContext& context, bool success, const QString& message);
        void completeOnce(bool success, const QString& message);
        void presentPreview(const ProjectTaskContext& context);
        void handleSessionChanged();
        bool hasPendingWork() const noexcept;
        void settleAfterDrain();
        void reapFinishedFutures();
        void clearTaskAfterWorkerExit(const ProjectTaskContext& context);
        void clearPreviewState(bool cancelContext);
        ProjectBundleAdjustMetadataStageResolveResult rollbackPreviewMetadataStage();

        ProjectSession* _session = nullptr;
        ProjectUiMessageAdapter* _messages = nullptr;
        ExecutionRunner _executionRunner;
        TiePointResultWriter _tiePointResultWriter;
        PostExternalCommitObserver _postExternalCommitObserver;
        const StartAdmissionProvider _startAdmissionProvider;
        const PreviewAdmissionPredicate _previewAdmissionPredicate;
        ProjectTaskContext _taskContext;
        QVector<QFuture<void>> _activeFutures;
        xjw::camera_project::CameraInstanceUpdates _pendingCameraUpdates;
        QMap<QString, QJsonObject> _pendingBeforeCameraMeta;
        QJsonObject _pendingResult;
        ProjectTaskContext _previewContext;
        ProjectBundleAdjustMetadataStageToken _previewMetadataStage;
        QString _pendingTiePointSparseCloudPath;
        QStringList _pendingTiePointSelectedImages;
        QString _pendingTiePointOutputDir;
        QJsonObject _pendingTiePointExtraRecord;
        int _pendingTiePointCount = 0;
        bool _isRunning = false;
        bool _terminalEmitted = false;
        bool _hasPendingPreview = false;
        bool _acceptanceInProgress = false;
        bool _previewArtifactsFinalized = false;
        bool _pendingTiePointWrite = false;
        bool _destroying = false;
    };

} // namespace xjw::gui::project
