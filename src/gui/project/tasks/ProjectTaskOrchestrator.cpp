#include "ProjectTaskOrchestrator.h"

#include "ProjectBundleAdjustController.h"
#include "ProjectCameraSetupManager.h"
#include "ProjectModelManager.h"
#include "ProjectPointCloudWorkflowController.h"
#include "ProjectSparseReconstructionManager.h"
#include "ProjectTerrainProductsManager.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ModelOutputPolicy.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QPointer>
#include <QThread>
#include <QTimer>
#include <QUuid>

#include <algorithm>
#include <utility>

namespace xjw::gui::project
{

    namespace
    {

        class SparseLaunchGuard final
        {
        public:
            explicit SparseLaunchGuard(int& depth) : _depth(depth)
            {
                ++_depth;
            }

            ~SparseLaunchGuard()
            {
                --_depth;
            }

        private:
            int& _depth;
        };

        QString normalizedDepthBatchDirectory(const QString& path)
        {
            const QString clean_path = QDir::cleanPath(path.trimmed());
            if (clean_path.isEmpty())
            {
                return QString();
            }
            const QFileInfo info(clean_path);
            return info.isDir() ? clean_path : QDir::cleanPath(info.absolutePath());
        }

        QString storedDepthBatchQualityProfile(const QJsonObject& metadata, const QString& sourcePath)
        {
            const QString requested_directory = normalizedDepthBatchDirectory(sourcePath);
            if (requested_directory.isEmpty())
            {
                return QString();
            }
            for (const QJsonValue& value : metadata.value(QStringLiteral("depth_map_results")).toArray())
            {
                const QJsonObject record = value.toObject();
                const QString record_path = record.value(QStringLiteral("mvs_output_dir"))
                                                .toString(record.value(QStringLiteral("depth_png")).toString());
                if (normalizedDepthBatchDirectory(record_path).compare(requested_directory, Qt::CaseInsensitive) != 0)
                {
                    continue;
                }
                const QString quality = record.value(QStringLiteral("quality_profile")).toString().trimmed().toLower();
                if (!quality.isEmpty())
                {
                    return quality;
                }
            }
            return QString();
        }

    } // namespace

    ProjectTaskOrchestrator::ProjectTaskOrchestrator(ProjectSession* session,
                                                     ProjectUiMessageAdapter* messages,
                                                     QObject* parent)
        : ProjectTaskOrchestrator(session, messages, MaskSettingsProvider{}, parent)
    {
    }

    ProjectTaskOrchestrator::ProjectTaskOrchestrator(ProjectSession* session,
                                                     ProjectUiMessageAdapter* messages,
                                                     MaskSettingsProvider maskSettingsProvider,
                                                     QObject* parent)
        : QObject(parent), _session(session), _messages(messages)
    {
        _sparseReconstruction.reset(new ProjectSparseReconstructionManager(
            _session.get(),
            _messages,
            nullptr,
            [this](const ProjectTaskContext& expected,
                   const QString& sparseCloudPath,
                   int sparsePointCount,
                   const QStringList& selectedImages,
                   const QString& outputDir,
                   const QJsonObject& extraRecord)
            {
                Q_UNUSED(expected);
                Q_UNUSED(sparseCloudPath);
                Q_UNUSED(sparsePointCount);
                Q_UNUSED(selectedImages);
                Q_UNUSED(outputDir);
                Q_UNUSED(extraRecord);
                return false;
            },
            this));
        _bundleAdjust.reset(new ProjectBundleAdjustController(
            _session.get(),
            _messages,
            [this]() { return reserveBundleAdjustStart(); },
            [this](QString* errorMessage) { return admitBundleAdjustPreview(errorMessage); },
            this));
        _maskWorkflow.reset(
            new ProjectMaskWorkflowController(_session.get(), _messages, std::move(maskSettingsProvider), this));
        _pointCloudWorkflow.reset(new ProjectPointCloudWorkflowController(_session.get(), _messages, this));
        _modelManager.reset(new ProjectModelManager(_session.get(), _messages, this));
        _terrainProducts.reset(new ProjectTerrainProductsManager(_session.get(), _messages, this));
        _cameraSetup.reset(new ProjectCameraSetupManager(_session.get(), _messages, this));

        if (_session)
        {
            connect(
                _session.get(), &ProjectSession::sessionChanged, this, &ProjectTaskOrchestrator::handleSessionChanged);
            connect(_session.get(),
                    &QObject::destroyed,
                    this,
                    [this]()
                    {
                        invalidateMaskLane(MaskCancelReason::SessionChanged);
                        invalidateForSessionChange();
                    });
        }
        connect(_sparseReconstruction.get(),
                &ProjectSparseReconstructionManager::atProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext progress_context = _activeContext;
                    const auto is_current = [this, &progress_context]()
                    {
                        return _hasActiveTask && isSparseTask(progress_context.taskId) && _session &&
                               _session->isCurrent(progress_context.session) && progress_context.cancelFlag &&
                               !progress_context.cancelFlag->load(std::memory_order_relaxed) &&
                               _activeContext.taskId == progress_context.taskId &&
                               _activeContext.cancelFlag == progress_context.cancelFlag;
                    };
                    if (!is_current())
                    {
                        return;
                    }
                    emit progressChanged(stage, percent);
                    if (!is_current())
                    {
                        return;
                    }
                    emit sparseProgressChanged(stage, percent);
                });
        connect(_sparseReconstruction.get(),
                &ProjectSparseReconstructionManager::atProgressFinished,
                this,
                [this](bool success)
                {
                    if (_hasActiveTask && isSparseTask(_activeContext.taskId))
                    {
                        finishTask(_activeContext, success);
                    }
                });
        connect(_sparseReconstruction.get(),
                &ProjectSparseReconstructionManager::tiePointResultReady,
                this,
                &ProjectTaskOrchestrator::sparseTiePointResultReady);
        connect(_bundleAdjust.get(),
                &ProjectBundleAdjustController::progressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext progress_context = _bundleAdjust->taskContext();
                    const auto is_current = [this, &progress_context]()
                    {
                        if (!_bundleAdjust || !_bundleAdjust->isRunning() || !_session ||
                            !_session->isCurrent(progress_context.session) || !progress_context.cancelFlag ||
                            progress_context.cancelFlag->load(std::memory_order_relaxed))
                        {
                            return false;
                        }
                        const ProjectTaskContext current_context = _bundleAdjust->taskContext();
                        return current_context.taskId == progress_context.taskId &&
                               current_context.cancelFlag == progress_context.cancelFlag;
                    };
                    if (!is_current())
                    {
                        return;
                    }
                    emit progressChanged(stage, percent);
                    if (!is_current())
                    {
                        return;
                    }
                    emit bundleAdjustProgressChanged(stage, percent);
                });
        connect(_bundleAdjust.get(),
                &ProjectBundleAdjustController::previewReady,
                this,
                &ProjectTaskOrchestrator::bundleAdjustPreviewReady);
        connect(_bundleAdjust.get(),
                &ProjectBundleAdjustController::finished,
                this,
                [this](bool success, const QString& message)
                {
                    emit bundleAdjustFinished(success);
                    if (_hasActiveTask && _activeContext.taskId == QLatin1String("bundle_adjust"))
                    {
                        finishTask(_activeContext, success, message);
                    }
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::progressChanged,
                this,
                [this](const QString& stage, int done, int total)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (!maskTailIsLive(signal_context))
                    {
                        return;
                    }
                    emit maskGenerationProgressChanged(stage, done, total);
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::finished,
                this,
                [this](bool success)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (maskTailIsLive(signal_context))
                    {
                        emit maskGenerationFinished(success);
                    }
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::masksGenerated,
                this,
                [this](const QStringList& images)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (maskTailIsLive(signal_context))
                    {
                        emit masksGenerated(images);
                    }
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::interactiveMaskSaved,
                this,
                [this](const QString& image, quint64 revision)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (maskTailIsLive(signal_context))
                    {
                        emit interactiveMaskSaved(image, revision);
                    }
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::interactiveMaskSaveFailed,
                this,
                [this](const QString& image, quint64 revision, const QString& message)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (maskTailIsLive(signal_context))
                    {
                        emit interactiveMaskSaveFailed(image, revision, message);
                    }
                });
        connect(_maskWorkflow.get(),
                &ProjectMaskWorkflowController::projectMetadataUpdated,
                this,
                [this](const QString& projectPath)
                {
                    const ProjectTaskContext signal_context = _maskContext;
                    if (maskTailIsLive(signal_context))
                    {
                        emit projectMetadataUpdated(projectPath);
                    }
                });
        connect(_pointCloudWorkflow.get(),
                &ProjectPointCloudWorkflowController::pointCloudProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext signal_context = _pointContext;
                    if (!pointContextMatches(signal_context) || !signal_context.cancelFlag ||
                        signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    if (_automaticModelActive && modelContextMatches(signal_context))
                    {
                        emit meshProgressChanged(QStringLiteral("准备深度图：%1").arg(stage),
                                                 std::clamp(percent * 3 / 5, 0, 59));
                        return;
                    }
                    emit pointCloudProgressChanged(stage, percent);
                });
        connect(_pointCloudWorkflow.get(),
                &ProjectPointCloudWorkflowController::pointCloudResultReady,
                this,
                [this](const QString& path, int pointCount)
                {
                    const ProjectTaskContext signal_context = _pointContext;
                    if (pointContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit pointCloudResultReady(path, pointCount);
                    }
                });
        connect(_pointCloudWorkflow.get(),
                &ProjectPointCloudWorkflowController::depthMapBatchReady,
                this,
                [this](const QString& outputDirectory, int)
                {
                    const ProjectTaskContext signal_context = _pointContext;
                    if (_automaticModelActive && pointContextMatches(signal_context) &&
                        modelContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        _automaticDepthOutputDirectory = outputDirectory;
                    }
                });
        connect(_pointCloudWorkflow.get(),
                &ProjectPointCloudWorkflowController::pointCloudProgressFinished,
                this,
                [this](bool success)
                {
                    const ProjectTaskContext signal_context = _pointContext;
                    if (!pointContextMatches(signal_context))
                    {
                        return;
                    }
                    const bool automatic = _automaticModelActive && modelContextMatches(signal_context);
                    const QString depth_output = _automaticDepthOutputDirectory;
                    if (!automatic)
                    {
                        if (success)
                        {
                            emit reconstructionQualityRefreshRequested();
                            if (!pointContextMatches(signal_context))
                            {
                                return;
                            }
                        }
                        const bool terminal_success = success && signal_context.cancelFlag &&
                                                      !signal_context.cancelFlag->load(std::memory_order_relaxed);
                        if (releasePointLaneIfMatches(signal_context))
                        {
                            emit pointCloudProgressFinished(terminal_success);
                        }
                        return;
                    }
                    releasePointLaneIfMatches(signal_context);
                    if (!success || depth_output.isEmpty())
                    {
                        finishAutomaticModel(false);
                        return;
                    }
                    startAutomaticModelBuild(depth_output);
                });
        connect(_modelManager.get(),
                &ProjectModelManager::meshProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext signal_context = _modelContext;
                    if (!modelContextMatches(signal_context) || !signal_context.cancelFlag ||
                        signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    emit meshProgressChanged(
                        stage, _automaticModelActive ? 60 + std::clamp(percent, 0, 100) * 40 / 100 : percent);
                });
        connect(_modelManager.get(),
                &ProjectModelManager::meshProgressFinished,
                this,
                [this](bool success)
                {
                    const ProjectTaskContext signal_context = _modelContext;
                    if (!modelContextMatches(signal_context))
                    {
                        return;
                    }
                    if (success)
                    {
                        emit reconstructionQualityRefreshRequested();
                        if (!modelContextMatches(signal_context))
                        {
                            return;
                        }
                    }
                    const bool terminal_success = success && signal_context.cancelFlag &&
                                                  !signal_context.cancelFlag->load(std::memory_order_relaxed);
                    if (!releaseModelLaneIfMatches(signal_context))
                    {
                        return;
                    }
                    _automaticModelActive = false;
                    _automaticModelSettings = {};
                    _automaticDepthOutputDirectory.clear();
                    emit meshProgressFinished(terminal_success);
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::backgroundTaskProgressChanged,
                this,
                [this](const QString& taskId, int value, int maximum)
                {
                    const bool dem_live = _demLaneActive && _demContext.taskId == taskId &&
                                          demContextMatches(_demContext) && _demContext.cancelFlag &&
                                          !_demContext.cancelFlag->load(std::memory_order_relaxed);
                    const bool ortho_live = _orthoLaneActive && _orthoContext.taskId == taskId &&
                                            orthoContextMatches(_orthoContext) && _orthoContext.cancelFlag &&
                                            !_orthoContext.cancelFlag->load(std::memory_order_relaxed);
                    if (dem_live || ortho_live)
                    {
                        emit backgroundTaskProgressChanged(taskId, value, maximum);
                    }
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::backgroundTaskFinished,
                this,
                [this](const QString& taskId)
                {
                    const bool dem_current =
                        _demLaneActive && _demContext.taskId == taskId && demContextMatches(_demContext);
                    const bool ortho_current =
                        _orthoLaneActive && _orthoContext.taskId == taskId && orthoContextMatches(_orthoContext);
                    if (dem_current || ortho_current)
                    {
                        emit backgroundTaskFinished(taskId);
                    }
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::demPipelineProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext signal_context = _demContext;
                    if (demContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit demPipelineProgressChanged(stage, percent);
                    }
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::demPipelineFinished,
                this,
                [this](bool success, const QString& message)
                {
                    const ProjectTaskContext signal_context = _demContext;
                    if (!demContextMatches(signal_context, false))
                    {
                        return;
                    }
                    const bool current = _session && _session->isCurrent(signal_context.session);
                    const bool terminal_success = success && signal_context.cancelFlag &&
                                                  !signal_context.cancelFlag->load(std::memory_order_relaxed);
                    releaseDemLaneIfMatches(signal_context);
                    if (!current)
                    {
                        return;
                    }
                    if (terminal_success)
                    {
                        emit reconstructionQualityRefreshRequested();
                        if (!_session || !_session->isCurrent(signal_context.session))
                        {
                            return;
                        }
                    }
                    emit demPipelineFinished(terminal_success, message);
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::orthoPipelineStarted,
                this,
                [this]()
                {
                    const ProjectTaskContext signal_context = _orthoContext;
                    if (orthoContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit orthoPipelineStarted();
                    }
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::orthoPipelineProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext signal_context = _orthoContext;
                    if (orthoContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit orthoPipelineProgressChanged(stage, percent);
                    }
                });
        connect(_terrainProducts.get(),
                &ProjectTerrainProductsManager::orthoPipelineFinished,
                this,
                [this](bool success, const QString& message, const QJsonObject& result)
                {
                    const ProjectTaskContext signal_context = _orthoContext;
                    if (!orthoContextMatches(signal_context, false))
                    {
                        return;
                    }
                    const bool current = _session && _session->isCurrent(signal_context.session);
                    const bool terminal_success = success && signal_context.cancelFlag &&
                                                  !signal_context.cancelFlag->load(std::memory_order_relaxed);
                    releaseOrthoLaneIfMatches(signal_context);
                    if (!current)
                    {
                        return;
                    }
                    if (terminal_success)
                    {
                        emit reconstructionQualityRefreshRequested();
                        if (!_session || !_session->isCurrent(signal_context.session))
                        {
                            return;
                        }
                    }
                    emit orthoPipelineFinished(terminal_success, message, result);
                });
        connect(_cameraSetup.get(),
                &ProjectCameraSetupManager::atProgressChanged,
                this,
                [this](const QString& stage, int percent)
                {
                    const ProjectTaskContext signal_context = _cameraContext;
                    if (cameraContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit cameraProgressChanged(stage, percent);
                    }
                });
        connect(_cameraSetup.get(),
                &ProjectCameraSetupManager::matchPairReady,
                this,
                [this](const QString& img0, const QString& img1, const QString& matchFilePath, int numMatches)
                {
                    const ProjectTaskContext signal_context = _cameraContext;
                    if (cameraContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit cameraMatchPairReady(img0, img1, matchFilePath, numMatches);
                    }
                });
        connect(_cameraSetup.get(),
                &ProjectCameraSetupManager::imageMatchResultAppended,
                this,
                [this](const QString& imagePath)
                {
                    const ProjectTaskContext signal_context = _cameraContext;
                    if (cameraContextMatches(signal_context) && signal_context.cancelFlag &&
                        !signal_context.cancelFlag->load(std::memory_order_relaxed))
                    {
                        emit imageMatchResultAppended(imagePath);
                    }
                });
        connect(_cameraSetup.get(),
                &ProjectCameraSetupManager::atProgressFinished,
                this,
                [this](bool success)
                {
                    const ProjectTaskContext signal_context = _cameraContext;
                    if (!cameraContextMatches(signal_context, false))
                    {
                        return;
                    }
                    const bool current = _session && _session->isCurrent(signal_context.session);
                    const bool terminal_success = success && signal_context.cancelFlag &&
                                                  !signal_context.cancelFlag->load(std::memory_order_relaxed);
                    releaseCameraLaneIfMatches(signal_context);
                    if (current)
                    {
                        emit cameraFinished(terminal_success);
                    }
                });
    }

    ProjectTaskOrchestrator::~ProjectTaskOrchestrator()
    {
        _destroying = true;
        _sessionDrainContinuation = {};
        _sessionDrainInProgress = false;
        _sessionDrainSettlePassPending = false;
        _sessionDrainSparseContext = {};
        invalidateMaskLane(MaskCancelReason::Destroying);
        cancelActiveTaskInternal(false);
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->cancelActiveTask();
        }
        if (_modelManager)
        {
            _modelManager->cancelActiveTask();
        }
        invalidateTerrainLanes();
        cancelCameraTask();
        releasePointLane();
        releaseModelLane();
        _automaticModelActive = false;
        if (_maskWorkflow)
        {
            _maskWorkflow->waitForActiveTask();
            _maskWorkflow.reset();
        }
        if (_sparseReconstruction)
        {
            _sparseReconstruction->waitForActiveTask();
        }
        if (_bundleAdjust)
        {
            _bundleAdjust->waitForFinished();
            _bundleAdjust.reset();
        }
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->waitForActiveTask();
            _pointCloudWorkflow.reset();
        }
        if (_modelManager)
        {
            _modelManager->waitForActiveTask();
            _modelManager.reset();
        }
        if (_terrainProducts)
        {
            _terrainProducts->waitForActiveTask();
            _terrainProducts.reset();
        }
        if (_cameraSetup)
        {
            _cameraSetup->waitForActiveTask();
            _cameraSetup.reset();
        }
        waitForSessionFutures();
    }

    ProjectSession* ProjectTaskOrchestrator::session() const noexcept
    {
        return _session.get();
    }

    ProjectTaskContext ProjectTaskOrchestrator::context(const QString& taskId) const
    {
        if (_hasActiveTask && _activeContext.taskId == taskId)
        {
            return _activeContext;
        }

        ProjectTaskContext result;
        result.taskId = taskId;
        result.session = _session ? _session->context() : ProjectSessionContext{};
        result.cancelFlag = std::make_shared<std::atomic<bool>>(false);
        return result;
    }

    bool ProjectTaskOrchestrator::hasActiveTask() const noexcept
    {
        return _hasActiveTask;
    }

    void ProjectTaskOrchestrator::cancelActiveTask()
    {
        cancelCameraTask();
        cancelActiveTaskInternal(true);
    }

    void ProjectTaskOrchestrator::waitForActiveTask()
    {
        if (_sparseReconstruction)
        {
            _sparseReconstruction->waitForActiveTask();
        }
        if (_bundleAdjust)
        {
            _bundleAdjust->waitForFinished();
        }
        if (_maskWorkflow)
        {
            _maskWorkflow->waitForActiveTask();
        }
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->waitForActiveTask();
        }
        if (_modelManager)
        {
            _modelManager->waitForActiveTask();
        }
        if (_terrainProducts)
        {
            _terrainProducts->waitForActiveTask();
        }
        if (_cameraSetup)
        {
            _cameraSetup->waitForActiveTask();
        }
        waitForSessionFutures();
    }

    bool ProjectTaskOrchestrator::cancelAndDrainForSessionChange(std::function<void()> continuation)
    {
        if (_destroying || _sessionDrainInProgress || !continuation || QThread::currentThread() != thread())
        {
            return false;
        }

        _sessionDrainInProgress = true;
        _sessionDrainSettlePassPending = false;
        _sessionDrainContinuation = std::move(continuation);
        if (_hasActiveTask && isSparseTask(_activeContext.taskId))
        {
            _sessionDrainSparseContext = _activeContext;
        }
        else
        {
            _sessionDrainSparseContext = {};
        }

        requestSessionDrainCancellation();
        scheduleSessionDrainPoll(0);
        return true;
    }

    bool ProjectTaskOrchestrator::isSessionDrainInProgress() const noexcept
    {
        return _sessionDrainInProgress;
    }

    bool ProjectTaskOrchestrator::trackExternalSparseFuture(const ProjectTaskContext& context, QFuture<void> future)
    {
        if (!future.isValid() || !_sparseReconstruction || !isSparseTask(context.taskId))
        {
            return false;
        }

        const bool current_context = isTaskActive(context);
        const bool draining_context = _sessionDrainInProgress && context.cancelFlag &&
                                      _sessionDrainSparseContext.cancelFlag &&
                                      context.taskId == _sessionDrainSparseContext.taskId &&
                                      context.cancelFlag == _sessionDrainSparseContext.cancelFlag &&
                                      context.session.matches(_sessionDrainSparseContext.session);
        if (!current_context && !draining_context)
        {
            return false;
        }

        _sessionDrainSettlePassPending = false;
        _sparseReconstruction->trackFuture(std::move(future));
        return true;
    }

    void ProjectTaskOrchestrator::trackSessionFuture(QFuture<void> future)
    {
        if (!future.isValid())
        {
            return;
        }
        pruneSessionFutures();
        _sessionDrainSettlePassPending = false;
        _sessionFutures.push_back(std::move(future));
    }

    void ProjectTaskOrchestrator::setTiePointResultWriter(SparseTiePointResultWriter writer)
    {
        if (_sparseReconstruction)
        {
            _sparseReconstruction->setTiePointResultWriter(writer);
        }
        if (_bundleAdjust)
        {
            if (!writer)
            {
                _bundleAdjust->setTiePointResultWriter({});
                return;
            }
            _bundleAdjust->setTiePointResultWriter(
                [writer = std::move(writer)](const ProjectTaskContext& expected,
                                             const QString& sparseCloudPath,
                                             int sparsePointCount,
                                             const QStringList& selectedImages,
                                             const QString& outputDir,
                                             const QJsonObject& extraRecord) {
                    return writer(expected, sparseCloudPath, sparsePointCount, selectedImages, outputDir, extraRecord);
                });
        }
    }

    void ProjectTaskOrchestrator::setDirectoryAccessors(
        std::function<QString(const QString& key)> getLastDir,
        std::function<void(const QString& key, const QString& dir)> saveLastDir)
    {
        if (_cameraSetup)
        {
            _cameraSetup->setDirectoryAccessors(std::move(getLastDir), std::move(saveLastDir));
        }
    }

    void ProjectTaskOrchestrator::setBundleAdjustPostExternalCommitObserver(
        std::function<void(const ProjectTaskContext& expected)> observer)
    {
        if (_bundleAdjust)
        {
            _bundleAdjust->setPostExternalCommitObserver(std::move(observer));
        }
    }

    bool ProjectTaskOrchestrator::startBundleAdjustAsync(
        const QStringList& images, const QString& outputDir, int threads, bool dryRun, const QJsonObject& extraSettings)
    {
        if (taskAdmissionBlocked() || _hasActiveTask ||
            (_sparseReconstruction && _sparseReconstruction->hasRunningTask()) || !_bundleAdjust ||
            _bundleAdjust->isRunning() || !_bundleAdjust->startAsync(images, outputDir, threads, dryRun, extraSettings))
        {
            return false;
        }

        const ProjectTaskContext task_context = _bundleAdjust->taskContext();
        if (!task_context.cancelFlag || task_context.cancelFlag->load(std::memory_order_relaxed))
        {
            return false;
        }
        _activeContext = task_context;
        _hasActiveTask = true;
        emit taskStarted(_activeContext.taskId);
        return true;
    }

    bool ProjectTaskOrchestrator::acceptBundleAdjustPreview(QString* errorMessage)
    {
        if (taskAdmissionBlocked() || !_bundleAdjust)
        {
            return false;
        }
        return _bundleAdjust->acceptPreview(errorMessage);
    }

    void ProjectTaskOrchestrator::discardBundleAdjustPreview()
    {
        if (_bundleAdjust)
        {
            _bundleAdjust->discardPreview();
        }
    }

    bool ProjectTaskOrchestrator::hasPendingBundleAdjustPreview() const noexcept
    {
        return _bundleAdjust && _bundleAdjust->hasPendingPreview();
    }

    bool ProjectTaskOrchestrator::hasRunningBundleAdjustTask() const noexcept
    {
        return _bundleAdjust && _bundleAdjust->isRunning();
    }

    void ProjectTaskOrchestrator::startGenerateModelAsync(const QJsonObject& settings)
    {
        if (taskAdmissionBlocked() || !_session || !_session->hasProject())
        {
            return;
        }
        const QString source_data = settings.value(QStringLiteral("source_data")).toString();
        const QString depth_source = settings.value(QStringLiteral("depthMapSourcePath"))
                                         .toString(settings.value(QStringLiteral("source_path")).toString())
                                         .trimmed();
        const bool reuse_depth_maps = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
        const bool force_depth_recompute = settings.value(QStringLiteral("force_depth_recompute")).toBool(false);
        const QString requested_depth_quality =
            settings.value(QStringLiteral("depthQualityProfile")).toString(QStringLiteral("medium"));
        const QJsonObject metadata = _session->metadata();
        const QString stored_depth_quality = storedDepthBatchQualityProfile(metadata, depth_source);
        const bool stored_depth_quality_mismatch =
            !stored_depth_quality.isEmpty() && stored_depth_quality != requested_depth_quality;
        const auto sparse_scaffold = resolveSparseScaffoldSource(metadata, depth_source);
        const bool allow_sparse_scaffold_fallback =
            settings.value(QStringLiteral("tsdfOrbitalSparseScaffoldCompletion")).toBool(true) &&
            !sparse_scaffold.pointCloudPath.isEmpty() && !sparse_scaffold.pointsJsonPath.isEmpty();
        const auto compatibility =
            source_data == QStringLiteral("depth_maps") && !depth_source.isEmpty()
                ? assessStoredDepthBatchCompatibility(metadata,
                                                      depth_source,
                                                      settings.value(QStringLiteral("at_index")).toInt(-1),
                                                      settings.value(QStringLiteral("sceneProfile")).toString(),
                                                      allow_sparse_scaffold_fallback,
                                                      depthBatchRequirementsForModelSettings(settings))
                : StoredDepthBatchCompatibility{};
        const bool incompatible = !depth_source.isEmpty() && !compatibility.compatible;
        const bool prepare_depth_maps =
            source_data == QStringLiteral("depth_maps") &&
            (settings.value(QStringLiteral("automatic_depth_maps")).toBool(false) || force_depth_recompute ||
             incompatible || stored_depth_quality_mismatch || !reuse_depth_maps || depth_source.isEmpty());
        if (!prepare_depth_maps)
        {
            startMeshReconstructionAsync(settings);
            return;
        }
        startAutomaticModelDepth(settings);
    }

    void ProjectTaskOrchestrator::startCreatePointCloudAsync(const QJsonObject& settings)
    {
        if (_destroying || taskAdmissionBlocked() || _pointLaneActive || !_pointCloudWorkflow ||
            _pointCloudWorkflow->isRunning())
        {
            if (_messages)
            {
                _messages->information(nullptr,
                                       QStringLiteral("创建点云"),
                                       QStringLiteral("已有点云任务正在运行，请等待或先取消当前任务。"));
            }
            return;
        }
        _pointContext = createIndependentContext(xjw::mesh::workflow::createModelRunId());
        const ProjectTaskContext start_context = _pointContext;
        _pointLaneActive = true;
        if (!_pointCloudWorkflow->startCreatePointCloudAsync(settings, start_context))
        {
            releasePointLaneIfMatches(start_context);
        }
    }

    void ProjectTaskOrchestrator::startMeshReconstructionAsync(const QJsonObject& settings)
    {
        if (_destroying || taskAdmissionBlocked() || _modelLaneActive || !_modelManager || _modelManager->isRunning())
        {
            if (_messages)
            {
                _messages->information(
                    nullptr, QStringLiteral("生成模型"), QStringLiteral("已有模型或纹理任务正在运行，请等待其完成。"));
            }
            return;
        }
        _modelContext = createIndependentContext(xjw::mesh::workflow::createModelRunId());
        const ProjectTaskContext start_context = _modelContext;
        _modelLaneActive = true;
        if (!_modelManager->startMeshReconstructionAsync(settings, start_context))
        {
            releaseModelLaneIfMatches(start_context);
        }
    }

    void ProjectTaskOrchestrator::startTextureMappingAsync(const QJsonObject& settings)
    {
        if (_destroying || taskAdmissionBlocked() || _modelLaneActive || !_modelManager || _modelManager->isRunning())
        {
            if (_messages)
            {
                _messages->information(
                    nullptr, QStringLiteral("纹理映射"), QStringLiteral("已有模型或纹理任务正在运行，请等待其完成。"));
            }
            return;
        }
        _modelContext = createIndependentContext(xjw::mesh::workflow::createModelRunId());
        const ProjectTaskContext start_context = _modelContext;
        _modelLaneActive = true;
        if (!_modelManager->startTextureMappingAsync(settings, start_context))
        {
            releaseModelLaneIfMatches(start_context);
        }
    }

    void ProjectTaskOrchestrator::cancelModelGeneration()
    {
        if (_modelContext.cancelFlag)
        {
            _modelContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_automaticModelActive && _pointLaneActive && _pointCloudWorkflow)
        {
            _pointCloudWorkflow->cancelActiveTask();
            return;
        }
        if (_modelManager)
        {
            _modelManager->cancelActiveTask();
        }
    }

    void ProjectTaskOrchestrator::cancelPointCloudGeneration()
    {
        if (_pointContext.cancelFlag)
        {
            _pointContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->cancelActiveTask();
        }
    }

    bool ProjectTaskOrchestrator::isModelGenerationRunning() const noexcept
    {
        return _modelLaneActive || (_modelManager && _modelManager->isRunning());
    }

    void ProjectTaskOrchestrator::startDemFromPointCloudAsync(const DemGenerationRequest& request)
    {
        if (_destroying || taskAdmissionBlocked() || !_terrainProducts)
        {
            return;
        }
        if (_demLaneActive)
        {
            const QString message = QStringLiteral("已有 DEM/DOM 任务正在运行，请等待其完成后再启动新任务。");
            if (_messages)
            {
                _messages->warning(nullptr, QStringLiteral("创建 DEM/DOM"), message);
            }
            emit demPipelineFinished(false, message);
            return;
        }

        const QString prefix = request.isSmallBodyGlobal()
                                   ? QStringLiteral("dem-global")
                                   : (request.isImageStereo() ? QStringLiteral("dem-rpc") : QStringLiteral("dem"));
        _demContext = createIndependentContext(
            QStringLiteral("%1:%2").arg(prefix, QUuid::createUuid().toString(QUuid::WithoutBraces)));
        _demLaneActive = true;
        _terrainProducts->startDemFromPointCloudAsync(request, _demContext);
    }

    void ProjectTaskOrchestrator::cancelDemGeneration()
    {
        if (!_demLaneActive)
        {
            return;
        }
        const ProjectTaskContext context = _demContext;
        if (context.cancelFlag)
        {
            context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_terrainProducts)
        {
            _terrainProducts->cancelDemGeneration(context);
        }
    }

    void ProjectTaskOrchestrator::startMapProjectAsync(const OrthoGenerationRequest& request)
    {
        if (_destroying || taskAdmissionBlocked() || !_terrainProducts)
        {
            return;
        }
        if (_orthoLaneActive)
        {
            const QString message = QStringLiteral("已有正射影像生成任务正在运行，请等待其完成或取消");
            emit orthoPipelineFinished(false, message, QJsonObject());
            return;
        }

        const QString prefix = request.isRpc() ? QStringLiteral("ortho-rpc") : QStringLiteral("ortho");
        _orthoContext = createIndependentContext(
            QStringLiteral("%1:%2").arg(prefix, QUuid::createUuid().toString(QUuid::WithoutBraces)));
        _orthoLaneActive = true;
        _terrainProducts->startMapProjectAsync(request, _orthoContext);
    }

    void ProjectTaskOrchestrator::cancelMapProject()
    {
        if (!_orthoLaneActive)
        {
            return;
        }
        const ProjectTaskContext context = _orthoContext;
        if (context.cancelFlag)
        {
            context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_terrainProducts)
        {
            _terrainProducts->cancelMapProject(context);
        }
    }

    bool ProjectTaskOrchestrator::importCameraForImage(const QString& imagePath)
    {
        return startCameraOperation(QStringLiteral("camera-import-single"),
                                    [this, imagePath](const ProjectTaskContext& context)
                                    { return _cameraSetup->importCameraForImage(imagePath, context); });
    }

    bool ProjectTaskOrchestrator::importCameraProject()
    {
        return startCameraOperation(QStringLiteral("camera-import-project"),
                                    [this](const ProjectTaskContext& context)
                                    { return _cameraSetup->importCameraProject(context); });
    }

    bool ProjectTaskOrchestrator::initializeCamerasFromExifOrDefault(const QJsonObject& settings)
    {
        return startCameraOperation(QStringLiteral("camera-init-exif"),
                                    [this, settings](const ProjectTaskContext& context)
                                    { return _cameraSetup->initializeCamerasFromExifOrDefault(settings, context); });
    }

    bool ProjectTaskOrchestrator::initializeCamerasFromIntrinsics(const QJsonObject& settings)
    {
        return startCameraOperation(QStringLiteral("camera-init-intrinsics"),
                                    [this, settings](const ProjectTaskContext& context)
                                    { return _cameraSetup->initializeCamerasFromIntrinsics(settings, context); });
    }

    bool ProjectTaskOrchestrator::initializeCameraPosesWithSFM(const QJsonObject& settings)
    {
        return startCameraOperation(QStringLiteral("camera-sfm"),
                                    [this, settings](const ProjectTaskContext& context)
                                    { return _cameraSetup->initializeCameraPosesWithSFM(settings, context); });
    }

    bool ProjectTaskOrchestrator::hasRunningCameraTask() const noexcept
    {
        return _cameraLaneActive;
    }

    void ProjectTaskOrchestrator::trackSparseFutureForTesting(QFuture<void> future)
    {
        if (_sparseReconstruction)
        {
            _sparseReconstruction->trackFutureForTesting(std::move(future));
        }
    }

    void ProjectTaskOrchestrator::trackPointFutureForTesting(QFuture<void> future)
    {
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->trackFutureForTesting(std::move(future));
        }
    }

    void ProjectTaskOrchestrator::trackModelFutureForTesting(QFuture<void> future)
    {
        if (_modelManager)
        {
            _modelManager->trackFutureForTesting(std::move(future));
        }
    }

    void ProjectTaskOrchestrator::trackTerrainFutureForTesting(QFuture<void> future)
    {
        if (_terrainProducts)
        {
            _terrainProducts->trackFutureForTesting(std::move(future));
        }
    }

    void ProjectTaskOrchestrator::trackCameraFutureForTesting(QFuture<void> future)
    {
        if (_cameraSetup)
        {
            _cameraSetup->trackFutureForTesting(std::move(future));
        }
    }

    std::function<void()> ProjectTaskOrchestrator::reserveBundleAdjustStart()
    {
        if (_destroying || taskAdmissionBlocked() || _bundleAdjustStartReserved || _hasActiveTask ||
            (_sparseReconstruction && _sparseReconstruction->hasRunningTask()))
        {
            return {};
        }

        _bundleAdjustStartReserved = true;
        const QPointer<ProjectTaskOrchestrator> guarded_this(this);
        return [guarded_this]()
        {
            if (guarded_this)
            {
                guarded_this->_bundleAdjustStartReserved = false;
            }
        };
    }

    bool ProjectTaskOrchestrator::admitBundleAdjustPreview(QString* errorMessage) const
    {
        const bool active_sparse_task = _hasActiveTask && isSparseTask(_activeContext.taskId);
        if (_destroying || _sessionDrainInProgress || active_sparse_task ||
            (_sparseReconstruction && _sparseReconstruction->hasRunningTask()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("稀疏重建任务仍占用处理通道，暂不能接受平差预览");
            }
            return false;
        }
        return true;
    }

    void ProjectTaskOrchestrator::cancelActiveTaskInternal(bool emitFinishedSignal)
    {
        if (!_hasActiveTask)
        {
            if (_bundleAdjust && (_bundleAdjust->isRunning() || _bundleAdjust->hasPendingPreview()))
            {
                if (emitFinishedSignal)
                {
                    _bundleAdjust->cancel();
                }
                else
                {
                    const ProjectTaskContext context = _bundleAdjust->taskContext();
                    if (context.cancelFlag)
                    {
                        context.cancelFlag->store(true, std::memory_order_relaxed);
                    }
                    _bundleAdjust->discardPreview();
                }
            }
            return;
        }

        const QString taskId = _activeContext.taskId;
        if (taskId == QLatin1String("bundle_adjust") && _bundleAdjust)
        {
            if (emitFinishedSignal)
            {
                _bundleAdjust->cancel();
            }
            else if (_activeContext.cancelFlag)
            {
                _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
                _bundleAdjust->discardPreview();
                _hasActiveTask = false;
                _activeContext = {};
            }
            return;
        }
        if (_activeContext.cancelFlag)
        {
            _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (isSparseTask(taskId) && _sparseLaunchDepth == 0 && _sparseReconstruction)
        {
            _sparseReconstruction->releaseTaskReservation();
        }
        _hasActiveTask = false;
        _activeContext = {};
        if (emitFinishedSignal)
        {
            emitSparseFinishedIfNeeded(taskId, false);
            emit taskFinished(taskId, false);
        }
    }

    void ProjectTaskOrchestrator::requestSessionDrainCancellation()
    {
        invalidateMaskLane(MaskCancelReason::SessionChanged);
        cancelActiveTaskInternal(false);

        if (_pointContext.cancelFlag)
        {
            _pointContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_modelContext.cancelFlag)
        {
            _modelContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->cancelActiveTask();
        }
        if (_modelManager)
        {
            _modelManager->cancelActiveTask();
        }

        const ProjectTaskContext dem_context = _demContext;
        const ProjectTaskContext ortho_context = _orthoContext;
        if (dem_context.cancelFlag)
        {
            dem_context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (ortho_context.cancelFlag)
        {
            ortho_context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_terrainProducts)
        {
            _terrainProducts->cancelDemGeneration(dem_context);
            _terrainProducts->cancelMapProject(ortho_context);
        }
        cancelCameraTask();
    }

    bool ProjectTaskOrchestrator::hasPendingSessionDrainWork() const
    {
        const auto has_unfinished_future = [](const QVector<QFuture<void>>& futures)
        {
            return std::any_of(futures.cbegin(),
                               futures.cend(),
                               [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
        };

        return (_sparseReconstruction && _sparseReconstruction->hasRunningTask()) ||
               (_bundleAdjust && _bundleAdjust->hasPendingWork()) ||
               (_maskWorkflow && _maskWorkflow->hasPendingWork()) ||
               (_pointCloudWorkflow && _pointCloudWorkflow->hasPendingWork()) ||
               (_modelManager && _modelManager->hasPendingWork()) ||
               (_terrainProducts && _terrainProducts->hasPendingWork()) ||
               (_cameraSetup && _cameraSetup->hasPendingWork()) || has_unfinished_future(_sessionFutures);
    }

    void ProjectTaskOrchestrator::scheduleSessionDrainPoll(int delayMs)
    {
        QTimer::singleShot(std::max(0, delayMs), this, [this]() { pollSessionDrain(); });
    }

    void ProjectTaskOrchestrator::pollSessionDrain()
    {
        if (_destroying || !_sessionDrainInProgress)
        {
            return;
        }

        pruneSessionFutures();
        if (hasPendingSessionDrainWork())
        {
            _sessionDrainSettlePassPending = false;
            scheduleSessionDrainPoll(15);
            return;
        }

        if (!_sessionDrainSettlePassPending)
        {
            _sessionDrainSettlePassPending = true;
            scheduleSessionDrainPoll(0);
            return;
        }

        settleSessionDrainState();
        std::function<void()> continuation = std::move(_sessionDrainContinuation);
        _sessionDrainContinuation = {};
        const QPointer<ProjectTaskOrchestrator> self(this);
        if (continuation)
        {
            continuation();
        }
        if (!self || self->_destroying)
        {
            return;
        }
        self->_sessionDrainInProgress = false;
        self->_sessionDrainSettlePassPending = false;
        self->_sessionDrainSparseContext = {};
    }

    void ProjectTaskOrchestrator::settleSessionDrainState()
    {
        if (_activeContext.cancelFlag)
        {
            _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_sparseReconstruction)
        {
            _sparseReconstruction->releaseTaskReservation();
        }
        _hasActiveTask = false;
        _activeContext = {};
        _bundleAdjustStartReserved = false;
        if (_bundleAdjust)
        {
            _bundleAdjust->settleAfterDrain();
        }

        _maskContext = {};
        _maskLaneMode = MaskLaneMode::Idle;
        _maskCancelReason = MaskCancelReason::None;
        _pendingInteractiveMask = {};
        releasePointLane();
        releaseModelLane();
        releaseDemLane();
        releaseOrthoLane();
        releaseCameraLane();
        _automaticModelActive = false;
        _automaticModelSettings = {};
        _automaticDepthOutputDirectory.clear();
    }

    void ProjectTaskOrchestrator::pruneSessionFutures()
    {
        _sessionFutures.erase(std::remove_if(_sessionFutures.begin(),
                                             _sessionFutures.end(),
                                             [](const QFuture<void>& future)
                                             { return !future.isValid() || future.isFinished(); }),
                              _sessionFutures.end());
    }

    void ProjectTaskOrchestrator::waitForSessionFutures()
    {
        for (QFuture<void>& future : _sessionFutures)
        {
            if (future.isValid() && !future.isFinished())
            {
                future.waitForFinished();
            }
        }
        _sessionFutures.clear();
    }

    bool ProjectTaskOrchestrator::taskAdmissionBlocked() const noexcept
    {
        return _destroying || _sessionDrainInProgress || (_session && _session->isBusy());
    }

    void ProjectTaskOrchestrator::invalidateForSessionChange()
    {
        invalidateTerrainLanes();
        cancelCameraTask();
        if (_pointContext.cancelFlag)
        {
            _pointContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_modelContext.cancelFlag)
        {
            _modelContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_pointCloudWorkflow)
        {
            _pointCloudWorkflow->cancelActiveTask();
        }
        if (_modelManager)
        {
            _modelManager->cancelActiveTask();
        }
        releasePointLane();
        releaseModelLane();
        _automaticModelActive = false;
        _automaticModelSettings = {};
        _automaticDepthOutputDirectory.clear();

        if (!_hasActiveTask)
        {
            return;
        }

        if (_activeContext.cancelFlag)
        {
            _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (isSparseTask(_activeContext.taskId) && _sparseLaunchDepth == 0 && _sparseReconstruction)
        {
            _sparseReconstruction->releaseTaskReservation();
        }
        _hasActiveTask = false;
        _activeContext = {};
    }

    bool ProjectTaskOrchestrator::beginTask(const QString& taskId)
    {
        const bool sparse_task = isSparseTask(taskId);
        if (_destroying || taskAdmissionBlocked() || _bundleAdjustStartReserved || _hasActiveTask ||
            (_bundleAdjust && _bundleAdjust->isRunning()) ||
            (sparse_task && _bundleAdjust && _bundleAdjust->hasPendingPreview()) || taskId.trimmed().isEmpty() ||
            !_session)
        {
            return false;
        }
        if (sparse_task && (!_sparseReconstruction || !_sparseReconstruction->reserveTask()))
        {
            return false;
        }

        _activeContext.taskId = taskId;
        _activeContext.session = _session->context();
        _activeContext.cancelFlag = std::make_shared<std::atomic<bool>>(false);
        _hasActiveTask = true;
        emit taskStarted(taskId);
        return true;
    }

    bool ProjectTaskOrchestrator::isTaskActive(const ProjectTaskContext& context) const
    {
        return _hasActiveTask && _session && _session->isCurrent(context.session) && context.cancelFlag &&
               !context.cancelFlag->load(std::memory_order_relaxed) && context.taskId == _activeContext.taskId &&
               context.cancelFlag == _activeContext.cancelFlag;
    }

    bool
    ProjectTaskOrchestrator::reportSparseProgress(const ProjectTaskContext& context, const QString& stage, int percent)
    {
        if (!isSparseTask(context.taskId) || !isTaskActive(context))
        {
            return false;
        }
        emit progressChanged(stage, percent);
        if (!isTaskActive(context))
        {
            return false;
        }
        emit sparseProgressChanged(stage, percent);
        return isTaskActive(context);
    }

    bool ProjectTaskOrchestrator::reportSparseComputeDevice(const ProjectTaskContext& context,
                                                            const QString& displayName)
    {
        if (!isSparseTask(context.taskId) || !isTaskActive(context))
        {
            return false;
        }
        emit sparseComputeDeviceChanged(displayName);
        return isTaskActive(context);
    }

    bool ProjectTaskOrchestrator::reportSparseMatchPair(const ProjectTaskContext& context,
                                                        const QString& firstImage,
                                                        const QString& secondImage,
                                                        const QString& matchFilePath,
                                                        int matchCount)
    {
        if (!isSparseTask(context.taskId) || !isTaskActive(context))
        {
            return false;
        }
        emit sparseMatchPairReady(firstImage, secondImage, matchFilePath, matchCount);
        return isTaskActive(context);
    }

    bool ProjectTaskOrchestrator::reportSparseTiePointResult(const ProjectTaskContext& context,
                                                             const QString& sparseCloudPath,
                                                             const QString& sidecarPath)
    {
        if (!isSparseTask(context.taskId) || !isTaskActive(context))
        {
            return false;
        }
        emit sparseTiePointResultReady(sparseCloudPath, sidecarPath);
        return isTaskActive(context);
    }

    bool
    ProjectTaskOrchestrator::finishTask(const ProjectTaskContext& context, bool success, const QString& errorMessage)
    {
        if (!_hasActiveTask || context.taskId != _activeContext.taskId ||
            context.cancelFlag != _activeContext.cancelFlag || !_session || !_session->isCurrent(context.session))
        {
            if (_hasActiveTask && context.taskId == _activeContext.taskId &&
                context.cancelFlag == _activeContext.cancelFlag)
            {
                if (_activeContext.cancelFlag)
                {
                    _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
                }
                if (isSparseTask(_activeContext.taskId) && _sparseLaunchDepth == 0 && _sparseReconstruction)
                {
                    _sparseReconstruction->releaseTaskReservation();
                }
                _hasActiveTask = false;
                _activeContext = {};
            }
            return false;
        }

        const QString taskId = context.taskId;
        if (isSparseTask(taskId) && _sparseLaunchDepth == 0 && _sparseReconstruction)
        {
            _sparseReconstruction->releaseTaskReservation();
        }
        _hasActiveTask = false;
        _activeContext = {};
        if (!success && !errorMessage.isEmpty())
        {
            emit taskError(taskId, errorMessage);
        }
        emitSparseFinishedIfNeeded(taskId, success);
        emit taskFinished(taskId, success);
        return true;
    }

    bool ProjectTaskOrchestrator::isSparseTask(const QString& taskId) const
    {
        return taskId == QLatin1String("aerial_triangulation") || taskId == QLatin1String("triangulation") ||
               taskId == QLatin1String("sparse_outlier_removal") || taskId == QLatin1String("sparse_local_optim") ||
               taskId == QLatin1String("sparse_refine");
    }

    void ProjectTaskOrchestrator::emitSparseFinishedIfNeeded(const QString& taskId, bool success)
    {
        if (isSparseTask(taskId))
        {
            emit sparseFinished(success);
        }
    }

    void ProjectTaskOrchestrator::startTriangulationAsync(const QJsonObject& settings)
    {
        SparseLaunchGuard launch_guard(_sparseLaunchDepth);
        if (beginTask(QStringLiteral("triangulation")))
        {
            const ProjectTaskContext task_context = _activeContext;
            if (_hasActiveTask && _activeContext.cancelFlag == task_context.cancelFlag)
            {
                _sparseReconstruction->startTriangulationAsync(settings, task_context);
            }
            _sparseReconstruction->releaseTaskReservation();
        }
    }

    void ProjectTaskOrchestrator::startSparseCloudOutlierRemovalAsync(const QJsonObject& settings)
    {
        SparseLaunchGuard launch_guard(_sparseLaunchDepth);
        if (beginTask(QStringLiteral("sparse_outlier_removal")))
        {
            const ProjectTaskContext task_context = _activeContext;
            if (_hasActiveTask && _activeContext.cancelFlag == task_context.cancelFlag)
            {
                _sparseReconstruction->startSparseCloudOutlierRemovalAsync(settings, task_context);
            }
            _sparseReconstruction->releaseTaskReservation();
        }
    }

    void ProjectTaskOrchestrator::startSparseCloudLocalOptimAsync(const QJsonObject& settings)
    {
        SparseLaunchGuard launch_guard(_sparseLaunchDepth);
        if (beginTask(QStringLiteral("sparse_local_optim")))
        {
            const ProjectTaskContext task_context = _activeContext;
            if (_hasActiveTask && _activeContext.cancelFlag == task_context.cancelFlag)
            {
                _sparseReconstruction->startSparseCloudLocalOptimAsync(settings, task_context);
            }
            _sparseReconstruction->releaseTaskReservation();
        }
    }

    void ProjectTaskOrchestrator::startSparseCloudRefineAsync(const QJsonObject& settings)
    {
        SparseLaunchGuard launch_guard(_sparseLaunchDepth);
        if (beginTask(QStringLiteral("sparse_refine")))
        {
            const ProjectTaskContext task_context = _activeContext;
            if (_hasActiveTask && _activeContext.cancelFlag == task_context.cancelFlag)
            {
                _sparseReconstruction->startSparseCloudRefineAsync(settings, task_context);
            }
            _sparseReconstruction->releaseTaskReservation();
        }
    }

    void ProjectTaskOrchestrator::setActiveImagePath(const QString& imagePath)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->setActiveImagePath(imagePath);
        }
    }

    void ProjectTaskOrchestrator::openGenerateMaskDialog()
    {
        if (!reserveMaskLane(MaskLaneMode::Modal, QStringLiteral("mask_generation")))
        {
            showMaskBusyWarning(QStringLiteral("生成蒙版"),
                                QStringLiteral("已有照片蒙版生成任务正在运行，请等待完成或取消后再试。"));
            return;
        }
        _maskWorkflow->openDialog(_maskContext);
    }

    void ProjectTaskOrchestrator::openGenerateMaskDialogForImages(const QStringList& selectedImages)
    {
        if (!reserveMaskLane(MaskLaneMode::Modal, QStringLiteral("mask_generation")))
        {
            showMaskBusyWarning(QStringLiteral("生成蒙版"),
                                QStringLiteral("已有照片蒙版生成任务正在运行，请等待完成或取消后再试。"));
            return;
        }
        _maskWorkflow->openDialogForImages(_maskContext, selectedImages);
    }

    void ProjectTaskOrchestrator::clearMasksForImages(const QStringList& selectedImages)
    {
        if (!reserveMaskLane(MaskLaneMode::Clear, QStringLiteral("mask_clear")))
        {
            showMaskBusyWarning(QStringLiteral("清除蒙版"),
                                QStringLiteral("照片蒙版生成任务正在运行，请等待完成或取消后再试。"));
            return;
        }
        _maskWorkflow->clearMasksForImages(_maskContext, selectedImages);
    }

    void ProjectTaskOrchestrator::saveInteractiveMask(const QString& imagePath,
                                                      const QImage& mask,
                                                      const QString& method,
                                                      quint64 revision)
    {
        if (_destroying || taskAdmissionBlocked())
        {
            return;
        }
        if (_maskLaneMode != MaskLaneMode::Idle)
        {
            _pendingInteractiveMask = {imagePath, mask, method, revision, true};
            return;
        }
        if (!reserveMaskLane(MaskLaneMode::Interactive, QStringLiteral("mask_interactive_save")))
        {
            return;
        }
        _maskWorkflow->saveInteractiveMask(_maskContext, imagePath, mask, method, revision);
    }

    void ProjectTaskOrchestrator::cancelMaskGeneration()
    {
        _pendingInteractiveMask = {};
        if (_maskLaneMode == MaskLaneMode::Idle)
        {
            return;
        }
        _maskCancelReason = MaskCancelReason::User;
        if (_maskContext.cancelFlag)
        {
            _maskContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_maskWorkflow)
        {
            _maskWorkflow->cancelActiveTask();
        }
    }

    bool ProjectTaskOrchestrator::hasRunningMaskTask() const noexcept
    {
        return _maskLaneMode != MaskLaneMode::Idle || (_maskWorkflow && _maskWorkflow->hasRunningTask());
    }

    bool ProjectTaskOrchestrator::reserveMaskLane(MaskLaneMode mode, const QString& taskId)
    {
        if (_destroying || taskAdmissionBlocked() || !_session || !_maskWorkflow || _maskLaneMode != MaskLaneMode::Idle)
        {
            return false;
        }
        _maskContext.taskId = taskId;
        _maskContext.session = _session->context();
        _maskContext.cancelFlag = std::make_shared<std::atomic<bool>>(false);
        _maskLaneMode = mode;
        _maskCancelReason = MaskCancelReason::None;
        return true;
    }

    bool ProjectTaskOrchestrator::maskContextIsLive(const ProjectTaskContext& context, bool allowUserBatchCancel) const
    {
        if (_destroying || _maskLaneMode == MaskLaneMode::Idle || !_session || context.taskId != _maskContext.taskId ||
            context.cancelFlag != _maskContext.cancelFlag || !_session->isCurrent(context.session))
        {
            return false;
        }
        if (_maskCancelReason == MaskCancelReason::None)
        {
            return context.cancelFlag && !context.cancelFlag->load(std::memory_order_relaxed);
        }
        return allowUserBatchCancel && _maskLaneMode == MaskLaneMode::Batch &&
               _maskCancelReason == MaskCancelReason::User;
    }

    bool ProjectTaskOrchestrator::maskTailIsLive(const ProjectTaskContext& context) const
    {
        return maskContextIsLive(context, true);
    }

    ProjectTaskOrchestrator::MaskCancelReason
    ProjectTaskOrchestrator::maskCancelReason(const ProjectTaskContext& context) const
    {
        if (_maskLaneMode == MaskLaneMode::Idle || context.taskId != _maskContext.taskId ||
            context.cancelFlag != _maskContext.cancelFlag)
        {
            return MaskCancelReason::SessionChanged;
        }
        return _maskCancelReason;
    }

    void ProjectTaskOrchestrator::setMaskLaneMode(const ProjectTaskContext& context, MaskLaneMode mode)
    {
        if (context.taskId == _maskContext.taskId && context.cancelFlag == _maskContext.cancelFlag &&
            _maskLaneMode != MaskLaneMode::Idle)
        {
            _maskLaneMode = mode;
        }
    }

    void ProjectTaskOrchestrator::settleMaskOperation(const ProjectTaskContext& context)
    {
        if (_maskLaneMode == MaskLaneMode::Idle || context.taskId != _maskContext.taskId ||
            context.cancelFlag != _maskContext.cancelFlag)
        {
            return;
        }
        _maskContext = {};
        _maskLaneMode = MaskLaneMode::Idle;
        _maskCancelReason = MaskCancelReason::None;
        if (!_destroying)
        {
            startPendingInteractiveMask();
        }
    }

    void ProjectTaskOrchestrator::invalidateMaskLane(MaskCancelReason reason)
    {
        _pendingInteractiveMask = {};
        if (_maskLaneMode == MaskLaneMode::Idle)
        {
            return;
        }
        _maskCancelReason = reason;
        if (_maskContext.cancelFlag)
        {
            _maskContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_maskWorkflow)
        {
            _maskWorkflow->cancelActiveTask();
        }
    }

    void ProjectTaskOrchestrator::startPendingInteractiveMask()
    {
        if (!_pendingInteractiveMask.valid || _destroying || _sessionDrainInProgress ||
            _maskLaneMode != MaskLaneMode::Idle)
        {
            return;
        }
        PendingInteractiveMask pending = std::move(_pendingInteractiveMask);
        _pendingInteractiveMask = {};
        saveInteractiveMask(pending.imagePath, pending.mask, pending.method, pending.revision);
    }

    void ProjectTaskOrchestrator::showMaskBusyWarning(const QString& title, const QString& text)
    {
        if (_messages)
        {
            _messages->warning(nullptr, title, text);
        }
    }

    void ProjectTaskOrchestrator::setMaskArtifactStagedObserverForTesting(std::function<void(int)> observer)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->setArtifactStagedObserverForTesting(std::move(observer));
        }
    }

    void ProjectTaskOrchestrator::setMaskClearRemovalFailureIndexForTesting(int index)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->setClearRemovalFailureIndexForTesting(index);
        }
    }

    void ProjectTaskOrchestrator::setMaskClearRecoveryCleanupFailureIndexForTesting(int index)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->setClearRecoveryCleanupFailureIndexForTesting(index);
        }
    }

    void ProjectTaskOrchestrator::setMaskPublicationObserverForTesting(
        std::function<void(const QString&, const StagedMaskArtifact&, const QString&)> observer)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->setPublicationObserverForTesting(std::move(observer));
        }
    }

    void ProjectTaskOrchestrator::trackMaskFutureForTesting(QFuture<void> future)
    {
        if (_maskWorkflow)
        {
            _maskWorkflow->trackFutureForTesting(std::move(future));
        }
    }

    int ProjectTaskOrchestrator::maskOwnedArtifactCountForTesting() const
    {
        return _maskWorkflow ? _maskWorkflow->ownedArtifactCountForTesting() : 0;
    }

    int ProjectTaskOrchestrator::maskFutureCountForTesting() const
    {
        return _maskWorkflow ? _maskWorkflow->futureCountForTesting() : 0;
    }

    ProjectTaskContext ProjectTaskOrchestrator::createIndependentContext(const QString& taskId) const
    {
        ProjectTaskContext result;
        result.taskId = taskId;
        result.session = _session ? _session->context() : ProjectSessionContext{};
        result.cancelFlag = std::make_shared<std::atomic<bool>>(false);
        return result;
    }

    bool ProjectTaskOrchestrator::pointContextMatches(const ProjectTaskContext& context, bool requireCurrent) const
    {
        return _pointLaneActive && context.cancelFlag && _pointContext.taskId == context.taskId &&
               _pointContext.cancelFlag == context.cancelFlag &&
               (!requireCurrent || (_session && _session->isCurrent(context.session)));
    }

    bool ProjectTaskOrchestrator::modelContextMatches(const ProjectTaskContext& context, bool requireCurrent) const
    {
        return _modelLaneActive && context.cancelFlag && _modelContext.taskId == context.taskId &&
               _modelContext.cancelFlag == context.cancelFlag &&
               (!requireCurrent || (_session && _session->isCurrent(context.session)));
    }

    bool ProjectTaskOrchestrator::demContextMatches(const ProjectTaskContext& context, bool requireCurrent) const
    {
        return _demLaneActive && context.cancelFlag && _demContext.taskId == context.taskId &&
               _demContext.cancelFlag == context.cancelFlag &&
               (!requireCurrent || (_session && _session->isCurrent(context.session)));
    }

    bool ProjectTaskOrchestrator::orthoContextMatches(const ProjectTaskContext& context, bool requireCurrent) const
    {
        return _orthoLaneActive && context.cancelFlag && _orthoContext.taskId == context.taskId &&
               _orthoContext.cancelFlag == context.cancelFlag &&
               (!requireCurrent || (_session && _session->isCurrent(context.session)));
    }

    bool ProjectTaskOrchestrator::cameraContextMatches(const ProjectTaskContext& context, bool requireCurrent) const
    {
        return _cameraLaneActive && context.cancelFlag && _cameraContext.taskId == context.taskId &&
               _cameraContext.cancelFlag == context.cancelFlag &&
               (!requireCurrent || (_session && _session->isCurrent(context.session)));
    }

    bool ProjectTaskOrchestrator::startCameraOperation(const QString& taskIdPrefix,
                                                       std::function<bool(const ProjectTaskContext&)> starter)
    {
        if (_destroying || taskAdmissionBlocked() || !_cameraSetup || !starter)
        {
            return false;
        }
        if (_cameraLaneActive)
        {
            if (_messages)
            {
                _messages->information(nullptr,
                                       QStringLiteral("相机设置"),
                                       QStringLiteral("已有相机导入或初始化任务正在运行，请等待或先取消当前任务。"));
            }
            return false;
        }

        _cameraContext = createIndependentContext(
            QStringLiteral("%1:%2").arg(taskIdPrefix, QUuid::createUuid().toString(QUuid::WithoutBraces)));
        const ProjectTaskContext start_context = _cameraContext;
        _cameraLaneActive = true;
        if (!starter(start_context))
        {
            releaseCameraLaneIfMatches(start_context);
            return false;
        }
        return true;
    }

    void ProjectTaskOrchestrator::releasePointLane()
    {
        _pointLaneActive = false;
        _pointContext = {};
    }

    void ProjectTaskOrchestrator::releaseModelLane()
    {
        _modelLaneActive = false;
        _modelContext = {};
    }

    void ProjectTaskOrchestrator::releaseDemLane()
    {
        _demLaneActive = false;
        _demContext = {};
    }

    void ProjectTaskOrchestrator::releaseOrthoLane()
    {
        _orthoLaneActive = false;
        _orthoContext = {};
    }

    void ProjectTaskOrchestrator::releaseCameraLane()
    {
        _cameraLaneActive = false;
        _cameraContext = {};
    }

    bool ProjectTaskOrchestrator::releasePointLaneIfMatches(const ProjectTaskContext& context)
    {
        if (!pointContextMatches(context, false))
        {
            return false;
        }
        releasePointLane();
        return true;
    }

    bool ProjectTaskOrchestrator::releaseModelLaneIfMatches(const ProjectTaskContext& context)
    {
        if (!modelContextMatches(context, false))
        {
            return false;
        }
        releaseModelLane();
        return true;
    }

    bool ProjectTaskOrchestrator::releaseDemLaneIfMatches(const ProjectTaskContext& context)
    {
        if (!demContextMatches(context, false))
        {
            return false;
        }
        releaseDemLane();
        return true;
    }

    bool ProjectTaskOrchestrator::releaseOrthoLaneIfMatches(const ProjectTaskContext& context)
    {
        if (!orthoContextMatches(context, false))
        {
            return false;
        }
        releaseOrthoLane();
        return true;
    }

    bool ProjectTaskOrchestrator::releaseCameraLaneIfMatches(const ProjectTaskContext& context)
    {
        if (!cameraContextMatches(context, false))
        {
            return false;
        }
        releaseCameraLane();
        return true;
    }

    void ProjectTaskOrchestrator::cancelCameraTask()
    {
        if (!_cameraLaneActive)
        {
            return;
        }
        const ProjectTaskContext context = _cameraContext;
        if (context.cancelFlag)
        {
            context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_cameraSetup)
        {
            _cameraSetup->cancelActiveTask(context);
        }
    }

    void ProjectTaskOrchestrator::invalidateTerrainLanes()
    {
        const ProjectTaskContext dem_context = _demContext;
        const ProjectTaskContext ortho_context = _orthoContext;
        if (dem_context.cancelFlag)
        {
            dem_context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (ortho_context.cancelFlag)
        {
            ortho_context.cancelFlag->store(true, std::memory_order_relaxed);
        }
        if (_terrainProducts)
        {
            _terrainProducts->cancelDemGeneration(dem_context);
            _terrainProducts->cancelMapProject(ortho_context);
        }
        releaseDemLane();
        releaseOrthoLane();
    }

    void ProjectTaskOrchestrator::finishAutomaticModel(bool success)
    {
        releasePointLane();
        releaseModelLane();
        _automaticModelActive = false;
        _automaticModelSettings = {};
        _automaticDepthOutputDirectory.clear();
        emit meshProgressFinished(success);
    }

    bool ProjectTaskOrchestrator::startAutomaticModelDepth(const QJsonObject& settings)
    {
        if (_destroying || _pointLaneActive || _modelLaneActive || !_pointCloudWorkflow || !_modelManager ||
            _pointCloudWorkflow->isRunning() || _modelManager->isRunning())
        {
            return false;
        }

        const ProjectTaskContext shared_context = createIndependentContext(xjw::mesh::workflow::createModelRunId());
        _pointContext = shared_context;
        _modelContext = shared_context;
        _pointLaneActive = true;
        _modelLaneActive = true;
        _automaticModelActive = true;
        _automaticModelSettings = settings;
        _automaticDepthOutputDirectory.clear();

        QJsonObject depth_settings = settings;
        depth_settings[QStringLiteral("depthQualityProfile")] =
            settings.value(QStringLiteral("depthQualityProfile")).toString(QStringLiteral("medium"));
        depth_settings[QStringLiteral("reuseDepthMaps")] = false;
        depth_settings[QStringLiteral("force_depth_recompute")] = true;
        depth_settings[QStringLiteral("depthFilterMode")] =
            settings.value(QStringLiteral("depthFiltering")).toString(QStringLiteral("mild"));
        depth_settings[QStringLiteral("calculatePointColors")] = false;
        depth_settings[QStringLiteral("replaceDefaultPointCloud")] = false;
        if (!_pointCloudWorkflow->startDepthMapsOnlyAsync(depth_settings, shared_context))
        {
            if (_automaticModelActive && pointContextMatches(shared_context, false) &&
                modelContextMatches(shared_context, false))
            {
                finishAutomaticModel(false);
            }
            return false;
        }
        return true;
    }

    void ProjectTaskOrchestrator::startAutomaticModelBuild(const QString& outputDirectory)
    {
        const ProjectTaskContext transition_context = _modelContext;
        if (!_automaticModelActive || !_modelLaneActive || !_modelManager || !_session ||
            !_session->isCurrent(transition_context.session) || !transition_context.cancelFlag ||
            transition_context.cancelFlag->load(std::memory_order_relaxed))
        {
            finishAutomaticModel(false);
            return;
        }

        QJsonObject model_settings = _automaticModelSettings;
        model_settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
        model_settings[QStringLiteral("source_path")] = outputDirectory;
        model_settings[QStringLiteral("depthMapSourcePath")] = outputDirectory;
        model_settings[QStringLiteral("reuseDepthMaps")] = true;
        model_settings[QStringLiteral("automatic_depth_maps")] = false;
        model_settings[QStringLiteral("force_depth_recompute")] = false;
        model_settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("recovered_ooc");
        emit meshProgressChanged(QStringLiteral("深度图估计完成，正在生成三维模型..."), 60);
        if (!_automaticModelActive || !modelContextMatches(transition_context) || !transition_context.cancelFlag ||
            transition_context.cancelFlag->load(std::memory_order_relaxed))
        {
            if (_automaticModelActive && modelContextMatches(transition_context, false))
            {
                finishAutomaticModel(false);
            }
            return;
        }
        if (!_modelManager->startMeshReconstructionAsync(model_settings, transition_context))
        {
            if (_automaticModelActive && modelContextMatches(transition_context, false))
            {
                finishAutomaticModel(false);
            }
        }
    }

    void ProjectTaskOrchestrator::handleSessionChanged(const ProjectSessionContext&)
    {
        invalidateMaskLane(MaskCancelReason::SessionChanged);
        invalidateForSessionChange();
    }

} // namespace xjw::gui::project
