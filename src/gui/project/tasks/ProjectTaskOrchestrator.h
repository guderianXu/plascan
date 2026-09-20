#pragma once

#include "project/manager/ProjectMaskWorkflowController.h"
#include "project/support/ProjectTerrainRequests.h"
#include "ProjectTaskContext.h"

#include <QFuture>
#include <QImage>
#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>
#include <memory>

class ProjectSparseReconstructionManager;
class ProjectCameraSetupManager;
class ProjectModelManager;
class ProjectPointCloudWorkflowController;
class ProjectTerrainProductsManager;
class ProjectUiMessageAdapter;
class ProjectTaskOrchestratorMaskTestPeer;
class ProjectTaskOrchestratorPointModelTestPeer;
class ProjectTaskOrchestratorTerrainTestPeer;
class ProjectTaskOrchestratorCameraTestPeer;

namespace xjw::gui::project
{

    class ProjectSession;
    class ProjectBundleAdjustController;

    class ProjectTaskOrchestrator final : public QObject
    {
        Q_OBJECT

    public:
        using SparseTiePointResultWriter = std::function<bool(const ProjectTaskContext& expected,
                                                              const QString& sparseCloudPath,
                                                              int sparsePointCount,
                                                              const QStringList& selectedImages,
                                                              const QString& outputDir,
                                                              const QJsonObject& extraRecord)>;

        ProjectTaskOrchestrator(ProjectSession* session, ProjectUiMessageAdapter* messages, QObject* parent = nullptr);
        ProjectTaskOrchestrator(ProjectSession* session,
                                ProjectUiMessageAdapter* messages,
                                MaskSettingsProvider maskSettingsProvider,
                                QObject* parent);
        ~ProjectTaskOrchestrator() override;

        ProjectSession* session() const noexcept;
        ProjectTaskContext context(const QString& taskId) const;
        bool hasActiveTask() const noexcept;
        void cancelActiveTask();
        void waitForActiveTask();
        bool cancelAndDrainForSessionChange(std::function<void()> continuation);
        bool isSessionDrainInProgress() const noexcept;
        bool trackExternalSparseFuture(const ProjectTaskContext& context, QFuture<void> future);
        void trackSessionFuture(QFuture<void> future);
        void invalidateForSessionChange();
        void setTiePointResultWriter(SparseTiePointResultWriter writer);
        void setDirectoryAccessors(std::function<QString(const QString& key)> getLastDir,
                                   std::function<void(const QString& key, const QString& dir)> saveLastDir);
        void
        setBundleAdjustPostExternalCommitObserver(std::function<void(const ProjectTaskContext& expected)> observer);
        void trackSparseFutureForTesting(QFuture<void> future);
        bool startBundleAdjustAsync(const QStringList& images,
                                    const QString& outputDir,
                                    int threads,
                                    bool dryRun,
                                    const QJsonObject& extraSettings);
        bool acceptBundleAdjustPreview(QString* errorMessage = nullptr);
        void discardBundleAdjustPreview();
        bool hasPendingBundleAdjustPreview() const noexcept;
        bool hasRunningBundleAdjustTask() const noexcept;
        void startGenerateModelAsync(const QJsonObject& settings);
        void startCreatePointCloudAsync(const QJsonObject& settings);
        void startMeshReconstructionAsync(const QJsonObject& settings);
        void startTextureMappingAsync(const QJsonObject& settings);
        void cancelModelGeneration();
        void cancelPointCloudGeneration();
        bool isModelGenerationRunning() const noexcept;
        void startDemFromPointCloudAsync(const DemGenerationRequest& request);
        void cancelDemGeneration();
        void startMapProjectAsync(const OrthoGenerationRequest& request);
        void cancelMapProject();
        bool importCameraForImage(const QString& imagePath);
        bool importCamerasByFilenameBatch();
        bool initializeCamerasFromExifOrDefault(const QJsonObject& settings);
        bool initializeCamerasFromIntrinsics(const QJsonObject& settings);
        bool initializeCameraPosesWithSFM(const QJsonObject& settings);
        bool hasRunningCameraTask() const noexcept;

        // Controlled task seam used by workflow adapters and deterministic tests.
        bool beginTask(const QString& taskId);
        bool isTaskActive(const ProjectTaskContext& context) const;
        bool reportSparseProgress(const ProjectTaskContext& context, const QString& stage, int percent);
        bool reportSparseComputeDevice(const ProjectTaskContext& context, const QString& displayName);
        bool reportSparseMatchPair(const ProjectTaskContext& context,
                                   const QString& firstImage,
                                   const QString& secondImage,
                                   const QString& matchFilePath,
                                   int matchCount);
        bool reportSparseTiePointResult(const ProjectTaskContext& context,
                                        const QString& sparseCloudPath,
                                        const QString& sidecarPath);
        bool finishTask(const ProjectTaskContext& context, bool success, const QString& errorMessage = QString());

        void startTriangulationAsync(const QJsonObject& settings);
        void startSparseCloudOutlierRemovalAsync(const QJsonObject& settings);
        void startSparseCloudLocalOptimAsync(const QJsonObject& settings);
        void startSparseCloudRefineAsync(const QJsonObject& settings);

        void setActiveImagePath(const QString& imagePath);
        void openGenerateMaskDialog();
        void openGenerateMaskDialogForImages(const QStringList& selectedImages);
        void clearMasksForImages(const QStringList& selectedImages);
        void saveInteractiveMask(const QString& imagePath, const QImage& mask, const QString& method, quint64 revision);
        void cancelMaskGeneration();
        bool hasRunningMaskTask() const noexcept;

    signals:
        void progressChanged(const QString& stage, int percent);
        void sparseProgressChanged(const QString& stage, int percent);
        void sparseComputeDeviceChanged(const QString& displayName);
        void sparseMatchPairReady(const QString& firstImage,
                                  const QString& secondImage,
                                  const QString& matchFilePath,
                                  int matchCount);
        void sparseFinished(bool success);
        void sparseTiePointResultReady(const QString& sparseCloudPath, const QString& sidecarPath);
        void bundleAdjustProgressChanged(const QString& stage, int percent);
        void bundleAdjustFinished(bool success);
        void bundleAdjustPreviewReady(const QJsonObject& preview);
        void taskStarted(const QString& taskId);
        void taskFinished(const QString& taskId, bool success);
        void taskError(const QString& taskId, const QString& message);
        void maskGenerationProgressChanged(const QString& stage, int done, int total);
        void maskGenerationFinished(bool success);
        void masksGenerated(const QStringList& imagePaths);
        void interactiveMaskSaved(const QString& imagePath, quint64 revision);
        void interactiveMaskSaveFailed(const QString& imagePath, quint64 revision, const QString& message);
        void projectMetadataUpdated(const QString& projectPath);
        void meshProgressChanged(const QString& stage, int percent);
        void meshProgressFinished(bool success);
        void pointCloudProgressChanged(const QString& stage, int percent);
        void pointCloudProgressFinished(bool success);
        void pointCloudResultReady(const QString& path, int pointCount);
        void backgroundTaskProgressChanged(const QString& taskId, int value, int maximum);
        void backgroundTaskFinished(const QString& taskId);
        void demPipelineProgressChanged(const QString& stage, int percent);
        void demPipelineFinished(bool success, const QString& message);
        void orthoPipelineStarted();
        void orthoPipelineProgressChanged(const QString& stage, int percent);
        void orthoPipelineFinished(bool success, const QString& message, const QJsonObject& result);
        void reconstructionQualityRefreshRequested();
        void cameraProgressChanged(const QString& stage, int percent);
        void cameraFinished(bool success);
        void
        cameraMatchPairReady(const QString& img0, const QString& img1, const QString& matchFilePath, int numMatches);
        void imageMatchResultAppended(const QString& imagePath);

    private:
        friend class ::ProjectMaskWorkflowController;
        friend class ::ProjectTaskOrchestratorMaskTestPeer;
        friend class ::ProjectTaskOrchestratorPointModelTestPeer;
        friend class ::ProjectTaskOrchestratorTerrainTestPeer;
        friend class ::ProjectTaskOrchestratorCameraTestPeer;

        enum class MaskLaneMode
        {
            Idle,
            Modal,
            Clear,
            Batch,
            Interactive
        };

        enum class MaskCancelReason
        {
            None,
            User,
            SessionChanged,
            Destroying
        };

        struct PendingInteractiveMask
        {
            QString imagePath;
            QImage mask;
            QString method;
            quint64 revision = 0;
            bool valid = false;
        };

        void cancelActiveTaskInternal(bool emitFinishedSignal);
        void requestSessionDrainCancellation();
        bool hasPendingSessionDrainWork() const;
        void scheduleSessionDrainPoll(int delayMs);
        void pollSessionDrain();
        void settleSessionDrainState();
        void pruneSessionFutures();
        void waitForSessionFutures();
        bool taskAdmissionBlocked() const noexcept;
        bool isSparseTask(const QString& taskId) const;
        void emitSparseFinishedIfNeeded(const QString& taskId, bool success);
        void handleSessionChanged(const ProjectSessionContext& context);
        std::function<void()> reserveBundleAdjustStart();
        bool admitBundleAdjustPreview(QString* errorMessage) const;
        bool reserveMaskLane(MaskLaneMode mode, const QString& taskId);
        bool maskContextIsLive(const ProjectTaskContext& context, bool allowUserBatchCancel = false) const;
        bool maskTailIsLive(const ProjectTaskContext& context) const;
        MaskCancelReason maskCancelReason(const ProjectTaskContext& context) const;
        void setMaskLaneMode(const ProjectTaskContext& context, MaskLaneMode mode);
        void settleMaskOperation(const ProjectTaskContext& context);
        void invalidateMaskLane(MaskCancelReason reason);
        void startPendingInteractiveMask();
        void showMaskBusyWarning(const QString& title, const QString& text);
        void setMaskArtifactStagedObserverForTesting(std::function<void(int)> observer);
        void setMaskClearRemovalFailureIndexForTesting(int index);
        void setMaskClearRecoveryCleanupFailureIndexForTesting(int index);
        void setMaskPublicationObserverForTesting(
            std::function<void(const QString&, const StagedMaskArtifact&, const QString&)> observer);
        void trackMaskFutureForTesting(QFuture<void> future);
        void trackPointFutureForTesting(QFuture<void> future);
        void trackModelFutureForTesting(QFuture<void> future);
        void trackTerrainFutureForTesting(QFuture<void> future);
        void trackCameraFutureForTesting(QFuture<void> future);
        int maskOwnedArtifactCountForTesting() const;
        int maskFutureCountForTesting() const;
        ProjectTaskContext createIndependentContext(const QString& taskId) const;
        bool pointContextMatches(const ProjectTaskContext& context, bool requireCurrent = true) const;
        bool modelContextMatches(const ProjectTaskContext& context, bool requireCurrent = true) const;
        void releasePointLane();
        void releaseModelLane();
        bool releasePointLaneIfMatches(const ProjectTaskContext& context);
        bool releaseModelLaneIfMatches(const ProjectTaskContext& context);
        bool demContextMatches(const ProjectTaskContext& context, bool requireCurrent = true) const;
        bool orthoContextMatches(const ProjectTaskContext& context, bool requireCurrent = true) const;
        void releaseDemLane();
        void releaseOrthoLane();
        bool releaseDemLaneIfMatches(const ProjectTaskContext& context);
        bool releaseOrthoLaneIfMatches(const ProjectTaskContext& context);
        void invalidateTerrainLanes();
        bool cameraContextMatches(const ProjectTaskContext& context, bool requireCurrent = true) const;
        void releaseCameraLane();
        bool releaseCameraLaneIfMatches(const ProjectTaskContext& context);
        void cancelCameraTask();
        void finishAutomaticModel(bool success);
        bool startAutomaticModelDepth(const QJsonObject& settings);
        void startAutomaticModelBuild(const QString& outputDirectory);

        QPointer<ProjectSession> _session;
        ProjectUiMessageAdapter* _messages = nullptr;
        std::unique_ptr<ProjectSparseReconstructionManager> _sparseReconstruction;
        std::unique_ptr<ProjectBundleAdjustController> _bundleAdjust;
        std::unique_ptr<::ProjectMaskWorkflowController> _maskWorkflow;
        std::unique_ptr<::ProjectPointCloudWorkflowController> _pointCloudWorkflow;
        std::unique_ptr<::ProjectModelManager> _modelManager;
        std::unique_ptr<::ProjectTerrainProductsManager> _terrainProducts;
        std::unique_ptr<::ProjectCameraSetupManager> _cameraSetup;
        ProjectTaskContext _activeContext;
        bool _hasActiveTask = false;
        bool _bundleAdjustStartReserved = false;
        bool _destroying = false;
        int _sparseLaunchDepth = 0;
        ProjectTaskContext _maskContext;
        MaskLaneMode _maskLaneMode = MaskLaneMode::Idle;
        MaskCancelReason _maskCancelReason = MaskCancelReason::None;
        PendingInteractiveMask _pendingInteractiveMask;
        ProjectTaskContext _pointContext;
        ProjectTaskContext _modelContext;
        ProjectTaskContext _demContext;
        ProjectTaskContext _orthoContext;
        ProjectTaskContext _cameraContext;
        QJsonObject _automaticModelSettings;
        QString _automaticDepthOutputDirectory;
        bool _pointLaneActive = false;
        bool _modelLaneActive = false;
        bool _demLaneActive = false;
        bool _orthoLaneActive = false;
        bool _cameraLaneActive = false;
        bool _automaticModelActive = false;
        QVector<QFuture<void>> _sessionFutures;
        std::function<void()> _sessionDrainContinuation;
        ProjectTaskContext _sessionDrainSparseContext;
        bool _sessionDrainInProgress = false;
        bool _sessionDrainSettlePassPending = false;
    };

} // namespace xjw::gui::project
