#include "project/services/ProjectServiceContainer.h"
#include "project/services/ProjectLifecycleService.h"
#include "project/services/ProjectResourceCleanupCoordinator.h"
#include "project/services/ProjectResourceService.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/tasks/ProjectBundleAdjustController.h"
#include "project/tasks/ProjectTaskOrchestrator.h"

#include "ProjectCameraIO.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/project/CameraProjectRecords.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "project/ProjectSessionModel.h"
#include "project/manager/ProjectMaskWorkflowController.h"
#include "project/manager/ProjectCameraSetupManager.h"
#include "project/manager/ProjectModelManager.h"
#include "project/manager/ProjectPointCloudWorkflowController.h"
#include "project/manager/ProjectSparseReconstructionManager.h"
#include "project/manager/ProjectTerrainProductsManager.h"
#include "io/PathIO.h"
#include "log/Logger.h"

#include <QApplication>
#include <QAbstractButton>
#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QMap>
#include <QMetaMethod>
#include <QMessageBox>
#include <QPointer>
#include <QPromise>
#include <QSemaphore>
#include <QThread>
#include <QThreadPool>
#include <QTemporaryDir>
#include <QTimer>
#include <QWidget>
#include <QtConcurrent/QtConcurrent>
#include <QtTest/QSignalSpy>
#include <QtTest/QTest>

#include <gtest/gtest.h>

#include <atomic>
#include <concepts>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

class ProjectTaskOrchestratorMaskTestPeer
{
public:
    using Orchestrator = xjw::gui::project::ProjectTaskOrchestrator;
    using TaskContext = xjw::gui::project::ProjectTaskContext;
    using PublicationObserver =
        std::function<void(const QString& phase, const StagedMaskArtifact& artifact, const QString& backupPath)>;

    static TaskContext installLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_maskContext = context;
        orchestrator->_maskLaneMode = Orchestrator::MaskLaneMode::Batch;
        orchestrator->_maskCancelReason = Orchestrator::MaskCancelReason::None;
        return context;
    }

    static bool laneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_maskLaneMode != Orchestrator::MaskLaneMode::Idle;
    }

    static void setArtifactStagedObserver(xjw::gui::project::ProjectTaskOrchestrator* orchestrator,
                                          std::function<void(int)> observer)
    {
        orchestrator->setMaskArtifactStagedObserverForTesting(std::move(observer));
    }

    static int ownedArtifactCount(const xjw::gui::project::ProjectTaskOrchestrator* orchestrator)
    {
        return orchestrator->maskOwnedArtifactCountForTesting();
    }

    static void setClearRemovalFailureIndex(xjw::gui::project::ProjectTaskOrchestrator* orchestrator, int index)
    {
        orchestrator->setMaskClearRemovalFailureIndexForTesting(index);
    }

    template <typename Orchestrator>
    static bool setClearRecoveryCleanupFailureIndex(Orchestrator* orchestrator, int index)
    {
        if constexpr (requires(Orchestrator * candidate, int value) {
                          candidate->setMaskClearRecoveryCleanupFailureIndexForTesting(value);
                      })
        {
            orchestrator->setMaskClearRecoveryCleanupFailureIndexForTesting(index);
            return true;
        }
        Q_UNUSED(orchestrator);
        Q_UNUSED(index);
        return false;
    }

    template <typename Orchestrator>
    static bool setPublicationObserver(Orchestrator* orchestrator, PublicationObserver observer)
    {
        if constexpr (requires(Orchestrator * candidate, PublicationObserver callback) {
                          candidate->setMaskPublicationObserverForTesting(std::move(callback));
                      })
        {
            orchestrator->setMaskPublicationObserverForTesting(std::move(observer));
            return true;
        }
        Q_UNUSED(orchestrator);
        Q_UNUSED(observer);
        return false;
    }

    template <typename Orchestrator> static std::optional<int> futureCount(const Orchestrator* orchestrator)
    {
        if constexpr (requires(const Orchestrator* candidate) { candidate->maskFutureCountForTesting(); })
        {
            return orchestrator->maskFutureCountForTesting();
        }
        Q_UNUSED(orchestrator);
        return std::nullopt;
    }

    template <typename Orchestrator> static bool trackFuture(Orchestrator* orchestrator, QFuture<void> future)
    {
        if constexpr (requires(Orchestrator * candidate, QFuture<void> value) {
                          candidate->trackMaskFutureForTesting(std::move(value));
                      })
        {
            orchestrator->trackMaskFutureForTesting(std::move(future));
            return true;
        }
        Q_UNUSED(orchestrator);
        Q_UNUSED(future);
        return false;
    }
};

class ProjectTaskOrchestratorPointModelTestPeer
{
public:
    using Orchestrator = xjw::gui::project::ProjectTaskOrchestrator;
    using TaskContext = xjw::gui::project::ProjectTaskContext;

    static TaskContext installPointLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_pointContext = context;
        orchestrator->_pointLaneActive = true;
        return context;
    }

    static TaskContext installModelLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_modelContext = context;
        orchestrator->_modelLaneActive = true;
        return context;
    }

    static TaskContext installAutomaticModelTransition(Orchestrator* orchestrator)
    {
        TaskContext context = installModelLane(orchestrator, QStringLiteral("automatic-transition"));
        orchestrator->_pointContext = {};
        orchestrator->_pointLaneActive = false;
        orchestrator->_automaticModelActive = true;
        orchestrator->_automaticModelSettings = {};
        orchestrator->_automaticDepthOutputDirectory = QStringLiteral("unused-depth-output");
        return context;
    }

    static bool releasePointLane(Orchestrator* orchestrator, const TaskContext& expected)
    {
        return orchestrator->releasePointLaneIfMatches(expected);
    }

    static bool releaseModelLane(Orchestrator* orchestrator, const TaskContext& expected)
    {
        return orchestrator->releaseModelLaneIfMatches(expected);
    }

    static TaskContext pointContext(const Orchestrator* orchestrator)
    {
        return orchestrator->_pointContext;
    }

    static TaskContext modelContext(const Orchestrator* orchestrator)
    {
        return orchestrator->_modelContext;
    }

    static bool pointLaneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_pointLaneActive;
    }

    static bool modelLaneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_modelLaneActive;
    }

    static void emitPointFinished(Orchestrator* orchestrator, bool success)
    {
        orchestrator->_pointCloudWorkflow->pointCloudProgressFinished(success);
    }

    static void emitModelFinished(Orchestrator* orchestrator, bool success)
    {
        orchestrator->_modelManager->meshProgressFinished(success);
    }

    static void startAutomaticModelBuild(Orchestrator* orchestrator)
    {
        orchestrator->startAutomaticModelBuild(QStringLiteral("unused-depth-output"));
    }

    static void trackPointFuture(Orchestrator* orchestrator, QFuture<void> future)
    {
        orchestrator->trackPointFutureForTesting(std::move(future));
    }

    static void trackModelFuture(Orchestrator* orchestrator, QFuture<void> future)
    {
        orchestrator->trackModelFutureForTesting(std::move(future));
    }
};

class ProjectTaskOrchestratorTerrainTestPeer
{
public:
    using Orchestrator = xjw::gui::project::ProjectTaskOrchestrator;
    using TaskContext = xjw::gui::project::ProjectTaskContext;

    static TaskContext installDemLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_demContext = context;
        orchestrator->_demLaneActive = true;
        orchestrator->_terrainProducts->_demContext = context;
        return context;
    }

    static TaskContext installOrthoLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_orthoContext = context;
        orchestrator->_orthoLaneActive = true;
        orchestrator->_terrainProducts->_orthoContext = context;
        return context;
    }

    static bool demLaneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_demLaneActive;
    }

    static bool orthoLaneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_orthoLaneActive;
    }

    static TaskContext demContext(const Orchestrator* orchestrator)
    {
        return orchestrator->_demContext;
    }

    static TaskContext orthoContext(const Orchestrator* orchestrator)
    {
        return orchestrator->_orthoContext;
    }

    static bool settleDemIfControllerCurrent(Orchestrator* orchestrator, const TaskContext& context, bool success)
    {
        if (!orchestrator->_terrainProducts->demContextMatches(context, false))
        {
            return false;
        }
        orchestrator->_terrainProducts->clearDemContextIfMatches(context);
        orchestrator->_terrainProducts->demPipelineFinished(success, QStringLiteral("terrain-dem-terminal"));
        return true;
    }

    static bool settleOrthoIfControllerCurrent(Orchestrator* orchestrator, const TaskContext& context, bool success)
    {
        if (!orchestrator->_terrainProducts->orthoContextMatches(context, false))
        {
            return false;
        }
        orchestrator->_terrainProducts->clearOrthoContextIfMatches(context);
        orchestrator->_terrainProducts->orthoPipelineFinished(
            success, QStringLiteral("terrain-ortho-terminal"), QJsonObject());
        return true;
    }

    static void trackFuture(Orchestrator* orchestrator, QFuture<void> future)
    {
        orchestrator->trackTerrainFutureForTesting(std::move(future));
    }
};

class ProjectTaskOrchestratorCameraTestPeer
{
public:
    using Orchestrator = xjw::gui::project::ProjectTaskOrchestrator;
    using TaskContext = xjw::gui::project::ProjectTaskContext;
    using SfmRunner = ProjectCameraSetupManager::SfmRunner;

    static ProjectCameraSetupManager* controller(Orchestrator* orchestrator)
    {
        return orchestrator->_cameraSetup.get();
    }

    static TaskContext installLane(Orchestrator* orchestrator, const QString& taskId)
    {
        TaskContext context = orchestrator->createIndependentContext(taskId);
        orchestrator->_cameraContext = context;
        orchestrator->_cameraLaneActive = true;
        orchestrator->_cameraSetup->_sfmContext = context;
        return context;
    }

    static bool laneActive(const Orchestrator* orchestrator)
    {
        return orchestrator->_cameraLaneActive;
    }

    static TaskContext context(const Orchestrator* orchestrator)
    {
        return orchestrator->_cameraContext;
    }

    static void complete(Orchestrator* orchestrator, const TaskContext& context, bool success)
    {
        orchestrator->_cameraSetup->completeTask(context, success);
    }

    static void setRunner(Orchestrator* orchestrator, SfmRunner runner)
    {
        orchestrator->_cameraSetup->setSfmRunnerForTesting(std::move(runner));
    }

    static void trackFuture(Orchestrator* orchestrator, QFuture<void> future)
    {
        orchestrator->trackCameraFutureForTesting(std::move(future));
    }
};

namespace
{

    QJsonObject bundleAdjustCameraMetadata()
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(1200.0, 1200.0, 512.0, 384.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0});
        camera.setImageSize(xjw::camera_core::ImageSize{1024, 768});
        QJsonObject metadata = xjw::common::project::serializeFramePinholeNumericState(camera);
        metadata.insert(QStringLiteral("aligned"), true);
        return metadata;
    }

    using MaskSettingsProviderProbe =
        std::function<std::optional<QJsonObject>(const QStringList& selectedImages, const QString& currentImage)>;

    template <typename T> concept PublicMaskSetActiveImage = requires(T & controller, const QString& image)
    {
        controller.setActiveImagePath(image);
    };

    template <typename T>
    concept PublicMaskOpenDialog = requires(T & controller, const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.openDialog(context);
    };

    template <typename T>
    concept PublicMaskOpenDialogForImages =
        requires(T & controller, const xjw::gui::project::ProjectTaskContext& context, const QStringList& images)
    {
        controller.openDialogForImages(context, images);
    };

    template <typename T>
    concept PublicMaskClear =
        requires(T & controller, const xjw::gui::project::ProjectTaskContext& context, const QStringList& images)
    {
        controller.clearMasksForImages(context, images);
    };

    template <typename T>
    concept PublicMaskInteractiveSave = requires(T & controller,
                                                 const xjw::gui::project::ProjectTaskContext& context,
                                                 const QString& image,
                                                 const QImage& mask,
                                                 const QString& method,
                                                 quint64 revision)
    {
        controller.saveInteractiveMask(context, image, mask, method, revision);
    };

    template <typename T> concept PublicMaskCancel = requires(T & controller)
    {
        controller.cancelActiveTask();
    };

    template <typename T> concept PublicMaskWait = requires(T & controller)
    {
        controller.waitForActiveTask();
    };

    template <typename T> concept PublicMaskRunning = requires(const T& controller)
    {
        controller.hasRunningTask();
    };

    template <typename T> concept PublicMaskFutureInjection = requires(T & controller, QFuture<void> future)
    {
        controller.trackFutureForTesting(future);
    };

    template <typename T> concept PublicMaskStagedObserver = requires(T & controller, std::function<void(int)> observer)
    {
        controller.setArtifactStagedObserverForTesting(std::move(observer));
    };

    template <typename T> concept PublicMaskOwnedArtifactCount = requires(const T& controller)
    {
        controller.ownedArtifactCountForTesting();
    };

    template <typename T> concept PublicMaskClearRemovalFailure = requires(T & controller)
    {
        controller.setClearRemovalFailureIndexForTesting(0);
    };

    template <typename T> concept PublicMaskClearRecoveryCleanupFailure = requires(T & controller)
    {
        controller.setClearRecoveryCleanupFailureIndexForTesting(0);
    };

    template <typename T> concept PublicMaskClearRecoveryCleanupFailureForwarder = requires(T & orchestrator)
    {
        orchestrator.setMaskClearRecoveryCleanupFailureIndexForTesting(0);
    };

    template <typename T>
    concept PublicMaskPublicationObserver =
        requires(T & controller, ProjectTaskOrchestratorMaskTestPeer::PublicationObserver observer)
    {
        controller.setPublicationObserverForTesting(std::move(observer));
    };

    template <typename T> concept PublicMaskFutureCount = requires(const T& controller)
    {
        controller.futureCountForTesting();
    };

    template <typename T> concept PublicMaskDestructor = requires(T * controller)
    {
        delete controller;
    };

    template <typename T> concept PublicMaskProgressSignal = requires
    {
        &T::progressChanged;
    };

    template <typename T> concept PublicMaskFinishedSignal = requires
    {
        &T::finished;
    };

    template <typename T> concept PublicMasksGeneratedSignal = requires
    {
        &T::masksGenerated;
    };

    template <typename T> concept PublicInteractiveMaskSavedSignal = requires
    {
        &T::interactiveMaskSaved;
    };

    template <typename T> concept PublicInteractiveMaskFailedSignal = requires
    {
        &T::interactiveMaskSaveFailed;
    };

    template <typename T> concept PublicMaskMetadataUpdatedSignal = requires
    {
        &T::projectMetadataUpdated;
    };

    using MaskControllerProbe = ProjectMaskWorkflowController;
    static_assert(!PublicMaskSetActiveImage<MaskControllerProbe>);
    static_assert(!PublicMaskOpenDialog<MaskControllerProbe>);
    static_assert(!PublicMaskOpenDialogForImages<MaskControllerProbe>);
    static_assert(!PublicMaskClear<MaskControllerProbe>);
    static_assert(!PublicMaskInteractiveSave<MaskControllerProbe>);
    static_assert(!PublicMaskCancel<MaskControllerProbe>);
    static_assert(!PublicMaskWait<MaskControllerProbe>);
    static_assert(!PublicMaskRunning<MaskControllerProbe>);
    static_assert(!PublicMaskFutureInjection<MaskControllerProbe>);
    static_assert(!PublicMaskStagedObserver<MaskControllerProbe>);
    static_assert(!PublicMaskOwnedArtifactCount<MaskControllerProbe>);
    static_assert(!PublicMaskClearRemovalFailure<MaskControllerProbe>);
    static_assert(!PublicMaskClearRecoveryCleanupFailure<MaskControllerProbe>);
    static_assert(!PublicMaskPublicationObserver<MaskControllerProbe>);
    static_assert(!PublicMaskFutureCount<MaskControllerProbe>);
    static_assert(!PublicMaskClearRecoveryCleanupFailureForwarder<xjw::gui::project::ProjectTaskOrchestrator>);
    static_assert(!std::constructible_from<MaskControllerProbe,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           MaskSettingsProviderProbe,
                                           QObject*>);
    static_assert(PublicMaskDestructor<MaskControllerProbe>);
    static_assert(PublicMaskProgressSignal<MaskControllerProbe>);
    static_assert(PublicMaskFinishedSignal<MaskControllerProbe>);
    static_assert(PublicMasksGeneratedSignal<MaskControllerProbe>);
    static_assert(PublicInteractiveMaskSavedSignal<MaskControllerProbe>);
    static_assert(PublicInteractiveMaskFailedSignal<MaskControllerProbe>);
    static_assert(PublicMaskMetadataUpdatedSignal<MaskControllerProbe>);

    template <typename T>
    concept PublicPointStart =
        requires(T & controller, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.startCreatePointCloudAsync(settings, context);
    };

    template <typename T> concept PublicPointCancel = requires(T & controller)
    {
        controller.cancelActiveTask();
    };

    template <typename T> concept PublicPointWait = requires(T & controller)
    {
        controller.waitForActiveTask();
    };

    template <typename T>
    concept PublicModelStart =
        requires(T & controller, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.startMeshReconstructionAsync(settings, context);
    };

    template <typename T> concept PublicModelCancel = requires(T & controller)
    {
        controller.cancelActiveTask();
    };

    template <typename T> concept PublicModelWait = requires(T & controller)
    {
        controller.waitForActiveTask();
    };

    static_assert(!PublicPointStart<ProjectPointCloudWorkflowController>);
    static_assert(!PublicPointCancel<ProjectPointCloudWorkflowController>);
    static_assert(!PublicPointWait<ProjectPointCloudWorkflowController>);
    static_assert(!PublicModelStart<ProjectModelManager>);
    static_assert(!PublicModelCancel<ProjectModelManager>);
    static_assert(!PublicModelWait<ProjectModelManager>);
    static_assert(!std::constructible_from<ProjectPointCloudWorkflowController,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           QObject*>);
    static_assert(!std::constructible_from<ProjectModelManager,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           QObject*>);

    template <typename T>
    concept PublicTerrainDemStart = requires(T & controller,
                                             const xjw::gui::project::DemGenerationRequest& request,
                                             const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.startDemFromPointCloudAsync(request, context);
    };

    template <typename T>
    concept PublicTerrainOrthoStart = requires(T & controller,
                                               const xjw::gui::project::OrthoGenerationRequest& request,
                                               const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.startMapProjectAsync(request, context);
    };

    template <typename T>
    concept PublicTerrainCancel = requires(T & controller, const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.cancelDemGeneration(context);
        controller.cancelMapProject(context);
    };

    template <typename T> concept PublicTerrainWait = requires(T & controller)
    {
        controller.waitForActiveTask();
    };

    template <typename T> concept PublicTerrainFutureInjection = requires(T & controller, QFuture<void> future)
    {
        controller.trackFutureForTesting(future);
    };

    template <typename T> concept PublicTerrainDestructor = requires(T * controller)
    {
        delete controller;
    };

    template <typename T> concept PublicTerrainSignals = requires
    {
        &T::backgroundTaskProgressChanged;
        &T::backgroundTaskFinished;
        &T::demPipelineProgressChanged;
        &T::demPipelineFinished;
        &T::orthoPipelineStarted;
        &T::orthoPipelineProgressChanged;
        &T::orthoPipelineFinished;
    };

    static_assert(!PublicTerrainDemStart<ProjectTerrainProductsManager>);
    static_assert(!PublicTerrainOrthoStart<ProjectTerrainProductsManager>);
    static_assert(!PublicTerrainCancel<ProjectTerrainProductsManager>);
    static_assert(!PublicTerrainWait<ProjectTerrainProductsManager>);
    static_assert(!PublicTerrainFutureInjection<ProjectTerrainProductsManager>);
    static_assert(!std::constructible_from<ProjectTerrainProductsManager,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           QObject*>);
    static_assert(PublicTerrainDestructor<ProjectTerrainProductsManager>);
    static_assert(PublicTerrainSignals<ProjectTerrainProductsManager>);

    template <typename T> concept PublicCameraImport = requires(T & controller, const QString& imagePath)
    {
        controller.importCameraForImage(imagePath);
        controller.importCamerasByFilenameBatch();
    };

    template <typename T>
    concept PublicCameraInitialize =
        requires(T & controller, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        controller.initializeCamerasFromExifOrDefault(settings);
        controller.initializeCamerasFromIntrinsics(settings);
        controller.initializeCameraPosesWithSFM(settings, context);
    };

    template <typename T>
    concept PublicCameraCancellation =
        requires(T & controller, const xjw::gui::project::ProjectTaskContext& context, QFuture<void> future)
    {
        controller.cancelActiveTask(context);
        controller.waitForActiveTask();
        controller.trackFutureForTesting(future);
    };

    template <typename T> concept PublicCameraDestructor = requires(T * controller)
    {
        delete controller;
    };

    template <typename T> concept PublicCameraSignals = requires
    {
        &T::atProgressChanged;
        &T::atProgressFinished;
        &T::matchPairReady;
    };

    static_assert(!PublicCameraImport<ProjectCameraSetupManager>);
    static_assert(!PublicCameraInitialize<ProjectCameraSetupManager>);
    static_assert(!PublicCameraCancellation<ProjectCameraSetupManager>);
    static_assert(!std::constructible_from<ProjectCameraSetupManager,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           QObject*>);
    static_assert(PublicCameraDestructor<ProjectCameraSetupManager>);
    static_assert(PublicCameraSignals<ProjectCameraSetupManager>);

    using SparseWriterProbe = std::function<bool(const xjw::gui::project::ProjectTaskContext& expected,
                                                 const QString& sparseCloudPath,
                                                 int sparsePointCount,
                                                 const QStringList& selectedImages,
                                                 const QString& outputDir,
                                                 const QJsonObject& extraRecord)>;

    template <typename T>
    concept PublicSparseTriangulationStart =
        requires(T & manager, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        manager.startTriangulationAsync(settings, context);
    };

    template <typename T>
    concept PublicSparseOutlierRemovalStart =
        requires(T & manager, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        manager.startSparseCloudOutlierRemovalAsync(settings, context);
    };

    template <typename T>
    concept PublicSparseLocalOptimStart =
        requires(T & manager, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        manager.startSparseCloudLocalOptimAsync(settings, context);
    };

    template <typename T>
    concept PublicSparseRefineStart =
        requires(T & manager, const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& context)
    {
        manager.startSparseCloudRefineAsync(settings, context);
    };

    template <typename T> concept PublicSparseWriterSetter = requires(T & manager, SparseWriterProbe writer)
    {
        manager.setTiePointResultWriter(writer);
    };

    template <typename T> concept PublicSparseWait = requires(T & manager)
    {
        manager.waitForActiveTask();
    };

    template <typename T> concept PublicSparseReserve = requires(T & manager)
    {
        manager.reserveTask();
    };

    template <typename T> concept PublicSparseRelease = requires(T & manager)
    {
        manager.releaseTaskReservation();
    };

    template <typename T> concept PublicSparseRunningQuery = requires(const T& manager)
    {
        manager.hasRunningTask();
    };

    template <typename T> concept PublicSparseFutureInjection = requires(T & manager, QFuture<void> future)
    {
        manager.trackFutureForTesting(future);
    };

    template <typename T> concept PublicSparseDestructor = requires(T * manager)
    {
        delete manager;
    };

    template <typename T> concept PublicSparseProgressSignal = requires
    {
        &T::atProgressChanged;
    };

    template <typename T> concept PublicSparseFinishedSignal = requires
    {
        &T::atProgressFinished;
    };

    template <typename T> concept PublicSparseTiePointSignal = requires
    {
        &T::tiePointResultReady;
    };

    using SparseManagerProbe = ProjectSparseReconstructionManager;
    static_assert(!PublicSparseTriangulationStart<SparseManagerProbe>);
    static_assert(!PublicSparseOutlierRemovalStart<SparseManagerProbe>);
    static_assert(!PublicSparseLocalOptimStart<SparseManagerProbe>);
    static_assert(!PublicSparseRefineStart<SparseManagerProbe>);
    static_assert(!PublicSparseWriterSetter<SparseManagerProbe>);
    static_assert(!PublicSparseWait<SparseManagerProbe>);
    static_assert(!PublicSparseReserve<SparseManagerProbe>);
    static_assert(!PublicSparseRelease<SparseManagerProbe>);
    static_assert(!PublicSparseRunningQuery<SparseManagerProbe>);
    static_assert(!PublicSparseFutureInjection<SparseManagerProbe>);
    static_assert(!std::constructible_from<SparseManagerProbe,
                                           xjw::gui::project::ProjectSession*,
                                           ProjectUiMessageAdapter*,
                                           QWidget*,
                                           SparseWriterProbe,
                                           QObject*>);
    static_assert(PublicSparseDestructor<SparseManagerProbe>);
    static_assert(PublicSparseProgressSignal<SparseManagerProbe>);
    static_assert(PublicSparseFinishedSignal<SparseManagerProbe>);
    static_assert(PublicSparseTiePointSignal<SparseManagerProbe>);

    QApplication& qtApplication()
    {
        static QApplication* application = nullptr;
        if (!application)
        {
            qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
            int argc = 1;
            auto* argv = new char*[2];
            argv[0] = const_cast<char*>("test_project_task_orchestrator");
            argv[1] = nullptr;
            application = new QApplication(argc, argv);
        }
        return *application;
    }

    bool prepareCameraProject(ProjectData* projectData,
                              const QString& directory,
                              QStringList* images,
                              QString* errorMessage = nullptr)
    {
        if (!projectData || !images ||
            !projectData->createProject(QDir(directory).filePath(QStringLiteral("camera_setup.plascan")),
                                        QStringLiteral("camera_setup")))
        {
            return false;
        }
        for (const QString& name : {QStringLiteral("camera-a.png"), QStringLiteral("camera-b.png")})
        {
            const QString path = QDir(directory).filePath(name);
            QImage image(32, 24, QImage::Format_RGB32);
            image.fill(Qt::white);
            if (!image.save(path))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法创建测试影像: %1").arg(path);
                }
                return false;
            }
            images->push_back(path);
        }
        return projectData->addImages(*images, errorMessage);
    }

    class ProjectTaskOrchestratorTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            qtApplication();
        }

        ProjectData projectData;
        xjw::gui::project::ProjectSession session{&projectData};
        xjw::gui::project::ProjectTaskOrchestrator orchestrator{&session, nullptr};
    };

    bool prepareBundleAdjustPreviewFixture(ProjectData* projectData,
                                           const QString& directory,
                                           QStringList* images,
                                           xjw::gui::project::BundleAdjustExecutionResult* executionResult,
                                           QString* errorMessage)
    {
        if (!projectData || !images || !executionResult)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("测试夹具参数无效");
            }
            return false;
        }
        for (const QString& name : {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")})
        {
            const QString path = QDir(directory).filePath(name);
            QFile file(path);
            if (!file.open(QIODevice::WriteOnly) || file.write("image") != 5)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法创建测试影像: %1").arg(path);
                }
                return false;
            }
            images->append(path);
        }
        if (!projectData->addImages(*images, errorMessage))
        {
            return false;
        }
        const QJsonObject camera = bundleAdjustCameraMetadata();
        int updatedCount = 0;
        if (!projectData->setCameraInstances(
                {{images->at(0), camera}, {images->at(1), camera}}, &updatedCount, errorMessage))
        {
            return false;
        }

        executionResult->serviceResult.success = true;
        executionResult->serviceResult.resultJson =
            QJsonObject{{QStringLiteral("output_dir"), directory},
                        {QStringLiteral("track_count"), 1},
                        {QStringLiteral("optimized_count"), 1},
                        {QStringLiteral("mean_rms_before"), 1.0},
                        {QStringLiteral("mean_rms_after"), 0.5},
                        {QStringLiteral("points"),
                         QJsonArray{QJsonObject{{QStringLiteral("valid"), true},
                                                {QStringLiteral("converged"), true},
                                                {QStringLiteral("rms_after"), 0.5},
                                                {QStringLiteral("track_len"), 2},
                                                {QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}}}}}};
        const QJsonObject core = projectData->coreFilesMeta();
        const QJsonArray definitions = core.value(QStringLiteral("camera_definitions")).toArray();
        if (definitions.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("测试相机定义为空");
            }
            return false;
        }
        const QString worldFrame = definitions.at(0).toObject().value(QStringLiteral("frame")).toString();
        for (const QJsonValue& imageValue : core.value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = imageValue.toObject();
            const QString imageId = image.value(QStringLiteral("image_uuid")).toString();
            const QJsonObject instance = xjw::camera_project::CameraProjectRecords::instanceForImage(core, imageId);
            QJsonObject update = xjw::camera_project::CameraProjectRecords::modelParametersForImage(core, image);
            update.insert(QStringLiteral("world_frame"), worldFrame);
            update.insert(QStringLiteral("solution"), QStringLiteral("bundle-adjust-preview"));
            executionResult->serviceResult.cameraInstanceUpdates.push_back(
                {xjw::camera_core::ImageId(imageId.toStdString()),
                 xjw::camera_core::CameraInstanceId(instance.value(QStringLiteral("id")).toString().toStdString()),
                 xjw::coordinate_system::CoordinateFrameId(worldFrame.toStdString()),
                 update});
        }
        return true;
    }

    xjw::gui::project::ProjectBundleAdjustController*
    ownedBaController(xjw::gui::project::ProjectTaskOrchestrator* orchestrator)
    {
        return orchestrator ? orchestrator->findChild<xjw::gui::project::ProjectBundleAdjustController*>(
                                  QString(), Qt::FindDirectChildrenOnly)
                            : nullptr;
    }

    class AcceptingReviewMessages final : public ProjectUiMessageAdapter
    {
    public:
        void information(QWidget*, const QString& title, const QString& text) override
        {
            ++informationCount;
            informationTitles.push_back(title);
            informationTexts.push_back(text);
            if (informationAction)
            {
                informationAction();
            }
        }
        void warning(QWidget*, const QString& title, const QString& text) override
        {
            ++warningCount;
            warningTitles.push_back(title);
            warningTexts.push_back(text);
            if (warningAction)
            {
                warningAction();
            }
        }
        void critical(QWidget*, const QString& title, const QString& text) override
        {
            ++criticalCount;
            criticalTitles.push_back(title);
            criticalTexts.push_back(text);
            if (criticalAction)
            {
                criticalAction();
            }
        }
        UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
        {
            ++questionCount;
            if (questionAction)
            {
                questionAction();
            }
            return questionAnswer;
        }
        UiReviewDecision review(QWidget*, const UiReviewDialogRequest&) override
        {
            ++reviewCount;
            if (maximumReviewCount >= 0 && reviewCount > maximumReviewCount)
            {
                reviewLimitExceeded = true;
                return UiReviewDecision::Discard;
            }
            if (reviewAction)
            {
                reviewAction();
            }
            return reviewDecision;
        }
        UiDialogResult getText(QWidget*, const QString&, const QString&, const QString&) override
        {
            return {};
        }
        UiDialogResult getDouble(QWidget*, const QString&, const QString&, double, double, double, int) override
        {
            return {};
        }
        UiDialogResult getItem(QWidget*, const QString&, const QString&, const QStringList&, int) override
        {
            return {};
        }
        UiDialogResult selectOpenFile(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
        {
            return openFileResult;
        }
        UiDialogResult selectOpenFiles(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
        {
            return {};
        }
        UiDialogResult selectDirectory(QWidget*, const QString&, const QString&, QDir::Filters) override
        {
            return directoryResult;
        }
        UiDialogResult
        selectSaveFile(QWidget*, const QString&, const QString&, const QString&, QDir::Filters, const QString&) override
        {
            return {};
        }

        int informationCount = 0;
        int warningCount = 0;
        int criticalCount = 0;
        int questionCount = 0;
        int reviewCount = 0;
        int maximumReviewCount = -1;
        bool reviewLimitExceeded = false;
        UiReviewDecision reviewDecision = UiReviewDecision::Accept;
        UiAnswer questionAnswer = UiAnswer::Cancel;
        std::function<void()> informationAction;
        std::function<void()> warningAction;
        std::function<void()> criticalAction;
        std::function<void()> questionAction;
        std::function<void()> reviewAction;
        QStringList informationTitles;
        QStringList informationTexts;
        QStringList warningTitles;
        QStringList warningTexts;
        QStringList criticalTitles;
        QStringList criticalTexts;
        UiDialogResult openFileResult;
        UiDialogResult directoryResult;
    };

    class ScopedLoggerCapture final
    {
    public:
        ScopedLoggerCapture()
            : _sinkId(Logger::instance()->registerSink(
                  [this](const Logger::Entry& entry)
                  {
                      std::lock_guard<std::mutex> lock(_mutex);
                      _entries.push_back(entry);
                  }))
        {
        }

        ~ScopedLoggerCapture()
        {
            Logger::instance()->unregisterSink(_sinkId);
        }

        ScopedLoggerCapture(const ScopedLoggerCapture&) = delete;
        ScopedLoggerCapture& operator=(const ScopedLoggerCapture&) = delete;

        std::vector<Logger::Entry> entries() const
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return _entries;
        }

    private:
        mutable std::mutex _mutex;
        std::vector<Logger::Entry> _entries;
        int _sinkId = 0;
    };

    int occurrenceCount(const QString& text, const QString& needle)
    {
        if (needle.isEmpty())
        {
            return 0;
        }
        int count = 0;
        qsizetype offset = 0;
        while ((offset = text.indexOf(needle, offset)) >= 0)
        {
            ++count;
            offset += needle.size();
        }
        return count;
    }

    class ScopedPendingFuture final
    {
    public:
        ScopedPendingFuture()
        {
            _promise.start();
        }

        ~ScopedPendingFuture()
        {
            finish();
        }

        QFuture<void> future()
        {
            return _promise.future();
        }

        void finish()
        {
            if (!_finished)
            {
                _promise.finish();
                _finished = true;
            }
        }

    private:
        QPromise<void> _promise;
        bool _finished = false;
    };

    class ScopedGlobalThreadPoolGate final
    {
    public:
        ScopedGlobalThreadPoolGate()
            : _pool(QThreadPool::globalInstance()), _previousMaxThreadCount(_pool->maxThreadCount())
        {
            _pool->waitForDone();
            _pool->setMaxThreadCount(1);
            _blocker = QtConcurrent::run(_pool,
                                         [this]()
                                         {
                                             _entered.release();
                                             _release.acquire();
                                         });
            _ready = _entered.tryAcquire(1, 5000);
        }

        ~ScopedGlobalThreadPoolGate()
        {
            release();
            if (_blocker.isValid())
            {
                _blocker.waitForFinished();
            }
            _pool->waitForDone();
            _pool->setMaxThreadCount(_previousMaxThreadCount);
        }

        bool isReady() const noexcept
        {
            return _ready;
        }

        void release()
        {
            if (!_released.exchange(true))
            {
                _release.release();
            }
        }

    private:
        QThreadPool* _pool = nullptr;
        int _previousMaxThreadCount = 1;
        QSemaphore _entered;
        QSemaphore _release;
        QFuture<void> _blocker;
        std::atomic<bool> _released{false};
        bool _ready = false;
    };

    class ScopedThreadJoiner final
    {
    public:
        explicit ScopedThreadJoiner(std::function<void()> function) : _thread(std::move(function))
        {
        }

        ~ScopedThreadJoiner()
        {
            join();
        }

        void join()
        {
            if (_thread.joinable())
            {
                _thread.join();
            }
        }

    private:
        std::thread _thread;
    };

    class ScopedArtifactStageGate final
    {
    public:
        ~ScopedArtifactStageGate()
        {
            release();
        }

        std::function<void(int)> observer()
        {
            return [this](int stagedCount)
            {
                if (stagedCount == 1 && !_entered.exchange(true))
                {
                    _firstStaged.release();
                    _resume.acquire();
                }
            };
        }

        bool waitForFirstStaged()
        {
            return _firstStaged.tryAcquire(1, 5000);
        }

        void release()
        {
            if (!_released.exchange(true))
            {
                _resume.release();
            }
        }

    private:
        QSemaphore _firstStaged;
        QSemaphore _resume;
        std::atomic<bool> _entered{false};
        std::atomic<bool> _released{false};
    };

    class ScopedWorkerGate final
    {
    public:
        ~ScopedWorkerGate()
        {
            release();
        }

        void enterAndWait()
        {
            _entered.release();
            _resume.acquire();
        }

        bool waitUntilEntered()
        {
            return _entered.tryAcquire(1, 5000);
        }

        void release()
        {
            if (!_released.exchange(true))
            {
                _resume.release();
            }
        }

    private:
        QSemaphore _entered;
        QSemaphore _resume;
        std::atomic<bool> _released{false};
    };

    bool prepareMaskProject(ProjectData* projectData,
                            const QString& directory,
                            const QVector<QPair<QString, QSize>>& imageDefinitions,
                            QStringList* images,
                            QString* errorMessage = nullptr)
    {
        if (!projectData || !images)
        {
            return false;
        }
        const QString project_path = QDir(directory).filePath(QStringLiteral("mask_workflow.plascan"));
        if (!projectData->createProject(project_path, QStringLiteral("mask_workflow")))
        {
            return false;
        }
        for (const auto& definition : imageDefinitions)
        {
            const QString path = QDir(directory).filePath(definition.first);
            QImage image(definition.second, QImage::Format_RGB32);
            image.fill(QColor(16, 24, 32));
            if (!image.save(path, "PNG"))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法创建蒙版测试影像：%1").arg(path);
                }
                return false;
            }
            images->push_back(path);
        }
        return projectData->addImages(*images, errorMessage);
    }

    QJsonObject selectedMaskSettings(const QStringList& images)
    {
        QJsonArray selected;
        for (const QString& image : images)
        {
            selected.push_back(image);
        }
        return QJsonObject{{QStringLiteral("method"), QStringLiteral("threshold")},
                           {QStringLiteral("scope"), QStringLiteral("selected_images")},
                           {QStringLiteral("selected_images"), selected},
                           {QStringLiteral("auto_threshold"), true},
                           {QStringLiteral("morphology_radius"), 0},
                           {QStringLiteral("min_component_area"), 0},
                           {QStringLiteral("operation"), QStringLiteral("replace")}};
    }

    QJsonObject projectImageRecord(const ProjectData& projectData, const QString& imagePath)
    {
        const QString project_path = projectData.currentProjectPath();
        const QString expected_key = xjw::common::project::normalizePath(imagePath);
        for (const QJsonValue& value : projectData.coreFilesMeta().value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = value.toObject();
            const QString resolved = xjw::common::project::ProjectIO::resolveProjectResourcePath(
                project_path, image.value(QStringLiteral("path")).toString());
            if (xjw::common::project::normalizePath(resolved) == expected_key)
            {
                return image;
            }
        }
        return {};
    }

    QString maskPathForImage(const ProjectData& projectData, const QString& imagePath)
    {
        return xjw::common::project::ProjectIO::maskOutputPathForImage(projectData.currentProjectPath(), imagePath);
    }

    QStringList maskTaskArtifacts(const QString& finalPath)
    {
        const QFileInfo final_info(finalPath);
        return QDir(final_info.absolutePath())
            .entryList({final_info.fileName() + QStringLiteral(".plascan-mask-*")}, QDir::Files | QDir::Hidden);
    }

    void expectSingleImageClearSuccess(const ProjectData& projectData,
                                       const QString& selectedImage,
                                       const xjw::gui::project::ProjectTaskOrchestrator& orchestrator,
                                       const AcceptingReviewMessages& messages,
                                       const QSignalSpy& generatedSpy,
                                       const QSignalSpy& updatedSpy)
    {
        EXPECT_FALSE(projectImageRecord(projectData, selectedImage).contains(QStringLiteral("mask_path")));
        EXPECT_EQ(generatedSpy.count(), 1);
        if (generatedSpy.count() == 1)
        {
            EXPECT_EQ(generatedSpy.constFirst().at(0).toStringList(), QStringList{QDir::cleanPath(selectedImage)});
        }
        EXPECT_EQ(updatedSpy.count(), 1);
        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_EQ(messages.informationCount, 1);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
    }

    TEST_F(ProjectTaskOrchestratorTest, SparseManagerSensitiveCapabilitiesAreNotPublicOrInvokable)
    {
        using Manager = ProjectSparseReconstructionManager;

        EXPECT_FALSE((PublicSparseTriangulationStart<Manager>));
        EXPECT_FALSE((PublicSparseOutlierRemovalStart<Manager>));
        EXPECT_FALSE((PublicSparseLocalOptimStart<Manager>));
        EXPECT_FALSE((PublicSparseRefineStart<Manager>));
        EXPECT_FALSE((PublicSparseWriterSetter<Manager>));
        EXPECT_FALSE((PublicSparseWait<Manager>));
        EXPECT_FALSE((PublicSparseReserve<Manager>));
        EXPECT_FALSE((PublicSparseRelease<Manager>));
        EXPECT_FALSE((PublicSparseRunningQuery<Manager>));
        EXPECT_FALSE((PublicSparseFutureInjection<Manager>));
        EXPECT_FALSE((std::constructible_from<Manager,
                                              xjw::gui::project::ProjectSession*,
                                              ProjectUiMessageAdapter*,
                                              QWidget*,
                                              SparseWriterProbe,
                                              QObject*>));

        EXPECT_TRUE((PublicSparseDestructor<Manager>));
        EXPECT_TRUE((PublicSparseProgressSignal<Manager>));
        EXPECT_TRUE((PublicSparseFinishedSignal<Manager>));
        EXPECT_TRUE((PublicSparseTiePointSignal<Manager>));

        auto* manager = orchestrator.findChild<Manager*>(QString(), Qt::FindDirectChildrenOnly);
        ASSERT_NE(manager, nullptr);
        const QStringList sensitive_method_names{QStringLiteral("startTriangulationAsync"),
                                                 QStringLiteral("startSparseCloudOutlierRemovalAsync"),
                                                 QStringLiteral("startSparseCloudLocalOptimAsync"),
                                                 QStringLiteral("startSparseCloudRefineAsync"),
                                                 QStringLiteral("setTiePointResultWriter"),
                                                 QStringLiteral("waitForActiveTask"),
                                                 QStringLiteral("reserveTask"),
                                                 QStringLiteral("releaseTaskReservation"),
                                                 QStringLiteral("hasRunningTask"),
                                                 QStringLiteral("trackFutureForTesting")};
        const QMetaObject* meta_object = manager->metaObject();
        ASSERT_NE(meta_object, nullptr);
        for (int index = 0; index < meta_object->methodCount(); ++index)
        {
            const QString method_name = QString::fromLatin1(meta_object->method(index).name());
            EXPECT_FALSE(sensitive_method_names.contains(method_name)) << qPrintable(method_name);
        }
    }

    TEST_F(ProjectTaskOrchestratorTest, MaskControllerSensitiveCapabilitiesAreNotPublicOrInvokable)
    {
        using Controller = ProjectMaskWorkflowController;

        EXPECT_FALSE((PublicMaskSetActiveImage<Controller>));
        EXPECT_FALSE((PublicMaskOpenDialog<Controller>));
        EXPECT_FALSE((PublicMaskOpenDialogForImages<Controller>));
        EXPECT_FALSE((PublicMaskClear<Controller>));
        EXPECT_FALSE((PublicMaskInteractiveSave<Controller>));
        EXPECT_FALSE((PublicMaskCancel<Controller>));
        EXPECT_FALSE((PublicMaskWait<Controller>));
        EXPECT_FALSE((PublicMaskRunning<Controller>));
        EXPECT_FALSE((PublicMaskFutureInjection<Controller>));
        EXPECT_FALSE((PublicMaskStagedObserver<Controller>));
        EXPECT_FALSE((PublicMaskOwnedArtifactCount<Controller>));
        EXPECT_FALSE((PublicMaskClearRemovalFailure<Controller>));
        EXPECT_FALSE((PublicMaskClearRecoveryCleanupFailure<Controller>));
        EXPECT_FALSE((PublicMaskPublicationObserver<Controller>));
        EXPECT_FALSE((PublicMaskFutureCount<Controller>));
        EXPECT_FALSE((PublicMaskClearRecoveryCleanupFailureForwarder<xjw::gui::project::ProjectTaskOrchestrator>));
        EXPECT_FALSE((std::constructible_from<Controller,
                                              xjw::gui::project::ProjectSession*,
                                              ProjectUiMessageAdapter*,
                                              MaskSettingsProviderProbe,
                                              QObject*>));

        EXPECT_TRUE((PublicMaskDestructor<Controller>));
        EXPECT_TRUE((PublicMaskProgressSignal<Controller>));
        EXPECT_TRUE((PublicMaskFinishedSignal<Controller>));
        EXPECT_TRUE((PublicMasksGeneratedSignal<Controller>));
        EXPECT_TRUE((PublicInteractiveMaskSavedSignal<Controller>));
        EXPECT_TRUE((PublicInteractiveMaskFailedSignal<Controller>));
        EXPECT_TRUE((PublicMaskMetadataUpdatedSignal<Controller>));

        auto* controller = orchestrator.findChild<Controller*>(QString(), Qt::FindDirectChildrenOnly);
        ASSERT_NE(controller, nullptr);
        EXPECT_EQ(orchestrator.findChildren<Controller*>(QString(), Qt::FindDirectChildrenOnly).size(), 1);
        const QStringList sensitive_method_names{QStringLiteral("setActiveImagePath"),
                                                 QStringLiteral("openDialog"),
                                                 QStringLiteral("openDialogForImages"),
                                                 QStringLiteral("clearMasksForImages"),
                                                 QStringLiteral("saveInteractiveMask"),
                                                 QStringLiteral("cancelActiveTask"),
                                                 QStringLiteral("waitForActiveTask"),
                                                 QStringLiteral("hasRunningTask"),
                                                 QStringLiteral("trackFutureForTesting"),
                                                 QStringLiteral("setArtifactStagedObserverForTesting"),
                                                 QStringLiteral("setClearRemovalFailureIndexForTesting"),
                                                 QStringLiteral("setClearRecoveryCleanupFailureIndexForTesting"),
                                                 QStringLiteral("ownedArtifactCountForTesting"),
                                                 QStringLiteral("setPublicationObserverForTesting"),
                                                 QStringLiteral("futureCountForTesting")};
        const QMetaObject* meta_object = controller->metaObject();
        ASSERT_NE(meta_object, nullptr);
        for (int index = 0; index < meta_object->methodCount(); ++index)
        {
            const QString method_name = QString::fromLatin1(meta_object->method(index).name());
            EXPECT_FALSE(sensitive_method_names.contains(method_name)) << qPrintable(method_name);
        }

        const QString orchestrator_sensitive_method_name =
            QStringLiteral("setMaskClearRecoveryCleanupFailureIndexForTesting");
        const QMetaObject* orchestrator_meta_object = orchestrator.metaObject();
        ASSERT_NE(orchestrator_meta_object, nullptr);
        for (int index = 0; index < orchestrator_meta_object->methodCount(); ++index)
        {
            const QString method_name = QString::fromLatin1(orchestrator_meta_object->method(index).name());
            EXPECT_NE(method_name, orchestrator_sensitive_method_name) << qPrintable(method_name);
        }
    }

    TEST_F(ProjectTaskOrchestratorTest, PointAndModelLaneReleaseRequiresTheOriginalContext)
    {
        const auto old_point =
            ProjectTaskOrchestratorPointModelTestPeer::installPointLane(&orchestrator, QStringLiteral("point-old"));
        const auto current_point =
            ProjectTaskOrchestratorPointModelTestPeer::installPointLane(&orchestrator, QStringLiteral("point-current"));
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::releasePointLane(&orchestrator, old_point));
        EXPECT_TRUE(ProjectTaskOrchestratorPointModelTestPeer::pointLaneActive(&orchestrator));
        EXPECT_EQ(ProjectTaskOrchestratorPointModelTestPeer::pointContext(&orchestrator).cancelFlag,
                  current_point.cancelFlag);
        EXPECT_TRUE(ProjectTaskOrchestratorPointModelTestPeer::releasePointLane(&orchestrator, current_point));

        const auto old_model =
            ProjectTaskOrchestratorPointModelTestPeer::installModelLane(&orchestrator, QStringLiteral("model-old"));
        const auto current_model =
            ProjectTaskOrchestratorPointModelTestPeer::installModelLane(&orchestrator, QStringLiteral("model-current"));
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::releaseModelLane(&orchestrator, old_model));
        EXPECT_TRUE(ProjectTaskOrchestratorPointModelTestPeer::modelLaneActive(&orchestrator));
        EXPECT_EQ(ProjectTaskOrchestratorPointModelTestPeer::modelContext(&orchestrator).cancelFlag,
                  current_model.cancelFlag);
        EXPECT_TRUE(ProjectTaskOrchestratorPointModelTestPeer::releaseModelLane(&orchestrator, current_model));
    }

    TEST_F(ProjectTaskOrchestratorTest, AutomaticTransitionCancellationStopsBeforeModelStart)
    {
        const auto transition_context =
            ProjectTaskOrchestratorPointModelTestPeer::installAutomaticModelTransition(&orchestrator);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::meshProgressFinished);
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::meshProgressChanged,
                         &orchestrator,
                         [this](const QString&, int percent)
                         {
                             if (percent == 60)
                             {
                                 orchestrator.cancelModelGeneration();
                             }
                         });

        ProjectTaskOrchestratorPointModelTestPeer::startAutomaticModelBuild(&orchestrator);

        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_FALSE(finished_spy.constFirst().at(0).toBool());
        EXPECT_TRUE(transition_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::modelLaneActive(&orchestrator));
        EXPECT_FALSE(orchestrator.isModelGenerationRunning());
    }

    TEST_F(ProjectTaskOrchestratorTest, QualityRefreshReentryCannotEmitStalePointOrModelSuccess)
    {
        QSignalSpy quality_spy(&orchestrator,
                               &xjw::gui::project::ProjectTaskOrchestrator::reconstructionQualityRefreshRequested);
        QSignalSpy point_finished_spy(&orchestrator,
                                      &xjw::gui::project::ProjectTaskOrchestrator::pointCloudProgressFinished);
        QSignalSpy model_finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::meshProgressFinished);
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::reconstructionQualityRefreshRequested,
                         &session,
                         [this]() { session.advanceGeneration(); });

        ProjectTaskOrchestratorPointModelTestPeer::installPointLane(&orchestrator, QStringLiteral("point-quality"));
        ProjectTaskOrchestratorPointModelTestPeer::emitPointFinished(&orchestrator, true);
        EXPECT_EQ(quality_spy.count(), 1);
        EXPECT_EQ(point_finished_spy.count(), 0);
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::pointLaneActive(&orchestrator));

        ProjectTaskOrchestratorPointModelTestPeer::installModelLane(&orchestrator, QStringLiteral("model-quality"));
        ProjectTaskOrchestratorPointModelTestPeer::emitModelFinished(&orchestrator, true);
        EXPECT_EQ(quality_spy.count(), 2);
        EXPECT_EQ(model_finished_spy.count(), 0);
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::modelLaneActive(&orchestrator));
    }

    TEST_F(ProjectTaskOrchestratorTest, DestructionJoinsPendingPointAndModelFutures)
    {
        auto owned_orchestrator = std::make_unique<xjw::gui::project::ProjectTaskOrchestrator>(&session, nullptr);
        QPromise<void> point_promise;
        QPromise<void> model_promise;
        point_promise.start();
        model_promise.start();
        ProjectTaskOrchestratorPointModelTestPeer::trackPointFuture(owned_orchestrator.get(), point_promise.future());
        ProjectTaskOrchestratorPointModelTestPeer::trackModelFuture(owned_orchestrator.get(), model_promise.future());

        QSemaphore destruction_entered;
        std::atomic<bool> destruction_returned{false};
        std::atomic<bool> returned_before_settle{true};
        ScopedThreadJoiner settler(
            [&]()
            {
                destruction_entered.acquire();
                QThread::msleep(100);
                returned_before_settle.store(destruction_returned.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
                point_promise.finish();
                model_promise.finish();
            });

        destruction_entered.release();
        owned_orchestrator.reset();
        destruction_returned.store(true, std::memory_order_relaxed);
        settler.join();
        EXPECT_FALSE(returned_before_settle.load(std::memory_order_relaxed));
    }

    TEST_F(ProjectTaskOrchestratorTest, TerrainControllerCapabilitiesArePrivateAndSolelyOwned)
    {
        using Controller = ProjectTerrainProductsManager;

        EXPECT_FALSE((PublicTerrainDemStart<Controller>));
        EXPECT_FALSE((PublicTerrainOrthoStart<Controller>));
        EXPECT_FALSE((PublicTerrainCancel<Controller>));
        EXPECT_FALSE((PublicTerrainWait<Controller>));
        EXPECT_FALSE((PublicTerrainFutureInjection<Controller>));
        EXPECT_FALSE((std::constructible_from<Controller,
                                              xjw::gui::project::ProjectSession*,
                                              ProjectUiMessageAdapter*,
                                              QObject*>));
        EXPECT_TRUE((PublicTerrainDestructor<Controller>));
        EXPECT_TRUE((PublicTerrainSignals<Controller>));
        EXPECT_EQ(orchestrator.findChildren<Controller*>(QString(), Qt::FindDirectChildrenOnly).size(), 1);
    }

    TEST_F(ProjectTaskOrchestratorTest, TerrainLanesRunIndependentlyAndCancelIndependently)
    {
        const auto dem_context =
            ProjectTaskOrchestratorTerrainTestPeer::installDemLane(&orchestrator, QStringLiteral("dem:test"));
        const auto ortho_context =
            ProjectTaskOrchestratorTerrainTestPeer::installOrthoLane(&orchestrator, QStringLiteral("ortho:test"));
        QSignalSpy dem_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished);
        QSignalSpy ortho_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineFinished);

        orchestrator.cancelDemGeneration();

        EXPECT_TRUE(dem_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_FALSE(ortho_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::orthoLaneActive(&orchestrator));
        EXPECT_TRUE(
            ProjectTaskOrchestratorTerrainTestPeer::settleDemIfControllerCurrent(&orchestrator, dem_context, false));
        EXPECT_EQ(dem_spy.count(), 1);
        EXPECT_FALSE(dem_spy.constFirst().at(0).toBool());
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::orthoLaneActive(&orchestrator));

        orchestrator.cancelMapProject();
        EXPECT_TRUE(ortho_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::settleOrthoIfControllerCurrent(
            &orchestrator, ortho_context, false));
        EXPECT_EQ(ortho_spy.count(), 1);
        EXPECT_FALSE(ortho_spy.constFirst().at(0).toBool());
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::orthoLaneActive(&orchestrator));
    }

    TEST_F(ProjectTaskOrchestratorTest, TerrainCancellationRetainsLaneUntilSettlementAndAllowsRestart)
    {
        const auto context =
            ProjectTaskOrchestratorTerrainTestPeer::installDemLane(&orchestrator, QStringLiteral("dem:cancel"));
        QSignalSpy terminal_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished);

        orchestrator.cancelDemGeneration();
        EXPECT_TRUE(context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));

        orchestrator.startDemFromPointCloudAsync(xjw::gui::project::DemGenerationRequest{});
        ASSERT_EQ(terminal_spy.count(), 1);
        EXPECT_FALSE(terminal_spy.constFirst().at(0).toBool());
        EXPECT_TRUE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));

        ASSERT_TRUE(
            ProjectTaskOrchestratorTerrainTestPeer::settleDemIfControllerCurrent(&orchestrator, context, false));
        ASSERT_EQ(terminal_spy.count(), 2);
        EXPECT_FALSE(terminal_spy.at(1).at(0).toBool());
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));

        orchestrator.startDemFromPointCloudAsync(xjw::gui::project::DemGenerationRequest{});
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_EQ(terminal_spy.count(), 3);
    }

    TEST_F(ProjectTaskOrchestratorTest, TerrainSessionReplacementCancelsBothAndOldSettlementCannotReleaseNewLanes)
    {
        const auto old_dem =
            ProjectTaskOrchestratorTerrainTestPeer::installDemLane(&orchestrator, QStringLiteral("dem:old"));
        const auto old_ortho =
            ProjectTaskOrchestratorTerrainTestPeer::installOrthoLane(&orchestrator, QStringLiteral("ortho:old"));
        QSignalSpy dem_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished);
        QSignalSpy ortho_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineFinished);
        QSignalSpy quality_spy(&orchestrator,
                               &xjw::gui::project::ProjectTaskOrchestrator::reconstructionQualityRefreshRequested);

        session.advanceGeneration();

        EXPECT_TRUE(old_dem.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_TRUE(old_ortho.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::orthoLaneActive(&orchestrator));

        const auto current_dem =
            ProjectTaskOrchestratorTerrainTestPeer::installDemLane(&orchestrator, QStringLiteral("dem:new"));
        const auto current_ortho =
            ProjectTaskOrchestratorTerrainTestPeer::installOrthoLane(&orchestrator, QStringLiteral("ortho:new"));
        EXPECT_FALSE(
            ProjectTaskOrchestratorTerrainTestPeer::settleDemIfControllerCurrent(&orchestrator, old_dem, true));
        EXPECT_FALSE(
            ProjectTaskOrchestratorTerrainTestPeer::settleOrthoIfControllerCurrent(&orchestrator, old_ortho, true));
        EXPECT_EQ(dem_spy.count(), 0);
        EXPECT_EQ(ortho_spy.count(), 0);
        EXPECT_EQ(quality_spy.count(), 0);
        EXPECT_EQ(ProjectTaskOrchestratorTerrainTestPeer::demContext(&orchestrator).cancelFlag, current_dem.cancelFlag);
        EXPECT_EQ(ProjectTaskOrchestratorTerrainTestPeer::orthoContext(&orchestrator).cancelFlag,
                  current_ortho.cancelFlag);

        EXPECT_TRUE(
            ProjectTaskOrchestratorTerrainTestPeer::settleDemIfControllerCurrent(&orchestrator, current_dem, true));
        EXPECT_TRUE(
            ProjectTaskOrchestratorTerrainTestPeer::settleOrthoIfControllerCurrent(&orchestrator, current_ortho, true));
        EXPECT_EQ(dem_spy.count(), 1);
        EXPECT_EQ(ortho_spy.count(), 1);
        EXPECT_EQ(quality_spy.count(), 2);
    }

    TEST_F(ProjectTaskOrchestratorTest, DestructionJoinsPendingTerrainFutures)
    {
        auto owned_orchestrator = std::make_unique<xjw::gui::project::ProjectTaskOrchestrator>(&session, nullptr);
        QPromise<void> dem_promise;
        QPromise<void> ortho_promise;
        dem_promise.start();
        ortho_promise.start();
        ProjectTaskOrchestratorTerrainTestPeer::trackFuture(owned_orchestrator.get(), dem_promise.future());
        ProjectTaskOrchestratorTerrainTestPeer::trackFuture(owned_orchestrator.get(), ortho_promise.future());

        QSemaphore destruction_entered;
        std::atomic<bool> destruction_returned{false};
        std::atomic<bool> returned_before_settle{true};
        ScopedThreadJoiner settler(
            [&]()
            {
                destruction_entered.acquire();
                QThread::msleep(100);
                returned_before_settle.store(destruction_returned.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
                dem_promise.finish();
                ortho_promise.finish();
            });

        destruction_entered.release();
        owned_orchestrator.reset();
        destruction_returned.store(true, std::memory_order_relaxed);
        settler.join();
        EXPECT_FALSE(returned_before_settle.load(std::memory_order_relaxed));
    }

    TEST_F(ProjectTaskOrchestratorTest, CameraControllerCapabilitiesArePrivateAndSolelyOwned)
    {
        using Controller = ProjectCameraSetupManager;

        EXPECT_FALSE((PublicCameraImport<Controller>));
        EXPECT_FALSE((PublicCameraInitialize<Controller>));
        EXPECT_FALSE((PublicCameraCancellation<Controller>));
        EXPECT_FALSE((std::constructible_from<Controller,
                                              xjw::gui::project::ProjectSession*,
                                              ProjectUiMessageAdapter*,
                                              QObject*>));
        EXPECT_TRUE((PublicCameraDestructor<Controller>));
        EXPECT_TRUE((PublicCameraSignals<Controller>));
        EXPECT_EQ(orchestrator.findChildren<Controller*>(QString(), Qt::FindDirectChildrenOnly).size(), 1);
    }

    TEST(ProjectCameraSetupTest, SynchronousIntrinsicsInitializationUsesSessionWritePort)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        QStringList images;
        QString error;
        ASSERT_TRUE(prepareCameraProject(&project_data, temp_dir.path(), &images, &error)) << qPrintable(error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);

        const QJsonObject settings{{QStringLiteral("fx"), 800.0},
                                   {QStringLiteral("fy"), 810.0},
                                   {QStringLiteral("distortionModel"), QStringLiteral("none")}};
        ASSERT_TRUE(orchestrator.initializeCamerasFromIntrinsics(settings));
        EXPECT_EQ(messages.informationCount, 1);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);

        const QJsonObject metadata = session.coreMetadata();
        const QMap<QString, QJsonObject> image_by_path = xjw::common::project::projectImageMetaByPath(metadata, true);
        for (const QString& image : images)
        {
            const QJsonObject image_record = image_by_path.value(QDir::cleanPath(QFileInfo(image).absoluteFilePath()));
            const QString image_id = image_record.value(QStringLiteral("image_uuid")).toString();
            EXPECT_FALSE(xjw::camera_project::CameraProjectRecords::instanceForImage(metadata, image_id).isEmpty());
        }
    }

    TEST(ProjectCameraSetupTest, CameraLaneIsIndependentAndDirectTerminalRestartIsSafe)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        QStringList images;
        QString error;
        ASSERT_TRUE(prepareCameraProject(&project_data, temp_dir.path(), &images, &error)) << qPrintable(error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        ProjectTaskOrchestratorCameraTestPeer::setRunner(
            &orchestrator,
            [](xjw::aerial_triangulation::AerialTriangulationOptions)
            {
                xjw::aerial_triangulation::AerialTriangulationResult result;
                result.reconstructionResult.success = true;
                return result;
            });

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished);
        bool restarted = false;
        bool restart_accepted = false;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished,
                         &orchestrator,
                         [&](bool success)
                         {
                             if (success && !restarted)
                             {
                                 restarted = true;
                                 restart_accepted = orchestrator.initializeCameraPosesWithSFM(QJsonObject());
                             }
                         });

        ASSERT_TRUE(orchestrator.initializeCameraPosesWithSFM(QJsonObject()));
        EXPECT_TRUE(orchestrator.hasActiveTask());
        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 2, 5000);
        EXPECT_TRUE(restart_accepted);
        EXPECT_FALSE(orchestrator.hasRunningCameraTask());
        EXPECT_TRUE(orchestrator.finishTask(orchestrator.context(QStringLiteral("triangulation")), true));
    }

    TEST(ProjectCameraSetupTest, CancellationRetainsLaneUntilWorkerSettlesAndEmitsOneFalseTerminal)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        QStringList images;
        QString error;
        ASSERT_TRUE(prepareCameraProject(&project_data, temp_dir.path(), &images, &error)) << qPrintable(error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSemaphore worker_entered;
        QSemaphore worker_release;
        ProjectTaskOrchestratorCameraTestPeer::setRunner(
            &orchestrator,
            [&](xjw::aerial_triangulation::AerialTriangulationOptions)
            {
                worker_entered.release();
                worker_release.acquire();
                xjw::aerial_triangulation::AerialTriangulationResult result;
                result.reconstructionResult.success = true;
                return result;
            });
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished);

        ASSERT_TRUE(orchestrator.initializeCameraPosesWithSFM(QJsonObject()));
        ASSERT_TRUE(worker_entered.tryAcquire(1, 5000));
        orchestrator.cancelActiveTask();
        EXPECT_TRUE(orchestrator.hasRunningCameraTask());
        EXPECT_FALSE(orchestrator.initializeCameraPosesWithSFM(QJsonObject()));
        worker_release.release();

        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 1, 5000);
        EXPECT_FALSE(finished_spy.constFirst().at(0).toBool());
        EXPECT_FALSE(orchestrator.hasRunningCameraTask());
    }

    TEST(ProjectCameraSetupTest, SessionReplacementSilencesStaleCameraCompletion)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        QStringList images;
        QString error;
        ASSERT_TRUE(prepareCameraProject(&project_data, temp_dir.path(), &images, &error)) << qPrintable(error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSemaphore worker_entered;
        QSemaphore worker_release;
        ProjectTaskOrchestratorCameraTestPeer::setRunner(
            &orchestrator,
            [&](xjw::aerial_triangulation::AerialTriangulationOptions)
            {
                worker_entered.release();
                worker_release.acquire();
                xjw::aerial_triangulation::AerialTriangulationResult result;
                result.reconstructionResult.success = true;
                return result;
            });
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished);

        ASSERT_TRUE(orchestrator.initializeCameraPosesWithSFM(QJsonObject()));
        ASSERT_TRUE(worker_entered.tryAcquire(1, 5000));
        session.advanceGeneration();
        EXPECT_TRUE(orchestrator.hasRunningCameraTask());
        worker_release.release();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningCameraTask(), 5000);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
    }

    TEST_F(ProjectTaskOrchestratorTest, DestructionJoinsPendingCameraFuture)
    {
        auto owned_orchestrator = std::make_unique<xjw::gui::project::ProjectTaskOrchestrator>(&session, nullptr);
        QPromise<void> promise;
        promise.start();
        ProjectTaskOrchestratorCameraTestPeer::trackFuture(owned_orchestrator.get(), promise.future());

        QSemaphore destruction_entered;
        std::atomic<bool> destruction_returned{false};
        std::atomic<bool> returned_before_settle{true};
        ScopedThreadJoiner settler(
            [&]()
            {
                destruction_entered.acquire();
                QThread::msleep(100);
                returned_before_settle.store(destruction_returned.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
                promise.finish();
            });

        destruction_entered.release();
        owned_orchestrator.reset();
        destruction_returned.store(true, std::memory_order_relaxed);
        settler.join();
        EXPECT_FALSE(returned_before_settle.load(std::memory_order_relaxed));
    }

    TEST(ProjectMaskWorkflowTest, SettingsProviderReceivesResolvedSelectionAndCurrentAndCancelIsSideEffectFree)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        QString fixture_error;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("first.png"), QSize(16, 16)}, {QStringLiteral("second.png"), QSize(16, 16)}},
            &images,
            &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        QStringList provider_selection;
        QString provider_current;
        int provider_calls = 0;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            nullptr,
            [&](const QStringList& selected, const QString& current) -> std::optional<QJsonObject>
            {
                ++provider_calls;
                provider_selection = selected;
                provider_current = current;
                return std::nullopt;
            },
            nullptr);
        QSignalSpy progress_spy(&orchestrator,
                                &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        orchestrator.setActiveImagePath(images.at(1));
        orchestrator.openGenerateMaskDialogForImages({images.at(0)});

        EXPECT_EQ(provider_calls, 1);
        EXPECT_EQ(provider_selection, QStringList{QDir::cleanPath(images.at(0))});
        EXPECT_EQ(provider_current, QDir::cleanPath(images.at(1)));
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_EQ(progress_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_FALSE(QFileInfo::exists(maskPathForImage(project_data, images.at(0))));
        EXPECT_FALSE(projectImageRecord(project_data, images.at(0)).contains(QStringLiteral("mask_path")));
    }

    TEST(ProjectMaskWorkflowTest, SettingsProviderDeleteLaterDestroysOwnerWithoutLateAccess)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        QPointer<xjw::gui::project::ProjectTaskOrchestrator> orchestrator_guard;
        bool deleted_inside_provider = false;
        orchestrator_guard = new xjw::gui::project::ProjectTaskOrchestrator(
            &session,
            nullptr,
            [&](const QStringList&, const QString&) -> std::optional<QJsonObject>
            {
                auto* doomed = orchestrator_guard.data();
                if (doomed)
                {
                    doomed->deleteLater();
                    QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
                }
                deleted_inside_provider = orchestrator_guard.isNull();
                return selectedMaskSettings(images);
            },
            nullptr);

        orchestrator_guard->openGenerateMaskDialogForImages(images);

        EXPECT_TRUE(deleted_inside_provider);
        EXPECT_TRUE(orchestrator_guard.isNull());
        EXPECT_FALSE(QFileInfo::exists(maskPathForImage(project_data, images.constFirst())));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        if (orchestrator_guard)
        {
            delete orchestrator_guard.data();
        }
    }

    TEST(ProjectMaskWorkflowTest, IndependentSessionDeletionInsideProviderSettlesModalLane)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        QPointer<xjw::gui::project::ProjectSession> session_guard =
            new xjw::gui::project::ProjectSession(&project_data);
        int provider_calls = 0;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            session_guard.data(),
            nullptr,
            [&](const QStringList&, const QString&) -> std::optional<QJsonObject>
            {
                ++provider_calls;
                auto* doomed = session_guard.data();
                if (doomed)
                {
                    doomed->deleteLater();
                    QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
                }
                return selectedMaskSettings(images);
            },
            nullptr);
        QSignalSpy progress_spy(&orchestrator,
                                &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        orchestrator.openGenerateMaskDialogForImages(images);

        EXPECT_EQ(provider_calls, 1);
        EXPECT_TRUE(session_guard.isNull());
        EXPECT_EQ(orchestrator.session(), nullptr);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_EQ(progress_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_FALSE(QFileInfo::exists(maskPathForImage(project_data, images.constFirst())));
    }

    TEST(ProjectMaskWorkflowTest, ServiceContainerOwnsOrchestratorAndExactlyOneMaskController)
    {
        qtApplication();
        ProjectData project_data;
        xjw::gui::project::ProjectServiceContainer container(&project_data, nullptr);

        EXPECT_EQ(container.findChildren<ProjectMaskWorkflowController*>(QString(), Qt::FindDirectChildrenOnly).size(),
                  0);
        EXPECT_EQ(container.tasks()
                      .findChildren<ProjectMaskWorkflowController*>(QString(), Qt::FindDirectChildrenOnly)
                      .size(),
                  1);
    }

    TEST(ProjectMaskWorkflowTest, ProviderSessionChangeAndReentrantCommandCannotLaunchWorker)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator* orchestrator_pointer = nullptr;
        int provider_calls = 0;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [&](const QStringList&, const QString&) -> std::optional<QJsonObject>
            {
                ++provider_calls;
                orchestrator_pointer->openGenerateMaskDialogForImages(images);
                session.advanceGeneration();
                return selectedMaskSettings(images);
            },
            nullptr);
        orchestrator_pointer = &orchestrator;
        QSignalSpy progress_spy(&orchestrator,
                                &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        orchestrator.openGenerateMaskDialogForImages(images);

        EXPECT_EQ(provider_calls, 1);
        EXPECT_EQ(messages.warningCount, 1);
        EXPECT_EQ(progress_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_FALSE(QFileInfo::exists(maskPathForImage(project_data, images.constFirst())));
    }

    TEST(ProjectMaskWorkflowTest, SessionChangeInsideClearConfirmationPreservesFileMetadataAndSignals)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write("original-mask"), 13);
        final_file.close();
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), final_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);
        const QJsonObject metadata_before = project_data.coreFilesMeta();

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        messages.questionAction = [&session]() { session.advanceGeneration(); };
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_EQ(project_data.coreFilesMeta(), metadata_before);
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearQuestionDeleteLaterDestroysOwnerWithoutFileOrMetadataMutation)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write("original-mask"), 13);
        final_file.close();
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), final_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);
        const QJsonObject metadata_before = project_data.coreFilesMeta();

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        QPointer<xjw::gui::project::ProjectTaskOrchestrator> orchestrator_guard;
        bool deleted_inside_question = false;
        messages.questionAction = [&]()
        {
            auto* doomed = orchestrator_guard.data();
            if (doomed)
            {
                doomed->deleteLater();
                QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
            }
            deleted_inside_question = orchestrator_guard.isNull();
        };
        orchestrator_guard = new xjw::gui::project::ProjectTaskOrchestrator(&session, &messages);

        orchestrator_guard->clearMasksForImages(images);

        EXPECT_TRUE(deleted_inside_question);
        EXPECT_TRUE(orchestrator_guard.isNull());
        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_EQ(project_data.coreFilesMeta(), metadata_before);
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
        if (orchestrator_guard)
        {
            delete orchestrator_guard.data();
        }
    }

    TEST(ProjectMaskWorkflowTest, IndependentSessionDeletionInsideQuestionSettlesClearLane)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        QPointer<xjw::gui::project::ProjectSession> session_guard =
            new xjw::gui::project::ProjectSession(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write("original-mask"), 13);
        final_file.close();
        QString port_error;
        ASSERT_TRUE(session_guard->publishImageMaskRecords(
            session_guard->context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), final_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);
        const QJsonObject metadata_before = project_data.coreFilesMeta();

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        messages.questionAction = [&]()
        {
            auto* doomed = session_guard.data();
            if (doomed)
            {
                doomed->deleteLater();
                QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
            }
        };
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(session_guard.data(), &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_TRUE(session_guard.isNull());
        EXPECT_EQ(orchestrator.session(), nullptr);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_EQ(project_data.coreFilesMeta(), metadata_before);
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearPortSessionDestructionCommitsPrimitiveAndSuppressesTail)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        QPointer<xjw::gui::project::ProjectSession> session_guard =
            new xjw::gui::project::ProjectSession(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write("original-mask"), 13);
        final_file.close();
        QString port_error;
        ASSERT_TRUE(session_guard->publishImageMaskRecords(
            session_guard->context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), final_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(session_guard.data(), &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        bool deleted_inside_port = false;
        QObject::connect(
            &project_data,
            &ProjectData::metadataChanged,
            &orchestrator,
            [&](const QJsonObject&)
            {
                auto* doomed = session_guard.data();
                if (!doomed)
                {
                    return;
                }
                doomed->deleteLater();
                QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
                deleted_inside_port = session_guard.isNull();
            },
            Qt::DirectConnection);

        orchestrator.clearMasksForImages(images);

        EXPECT_TRUE(deleted_inside_port);
        EXPECT_TRUE(session_guard.isNull());
        EXPECT_EQ(orchestrator.session(), nullptr);
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearPublishesOnlyAfterPerImageFilesAreDeletedAndSurvivesGenerationReentry)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString standard_path = maskPathForImage(project_data, images.constFirst());
        const QString masks_directory = QFileInfo(standard_path).absolutePath();
        const QString custom_path = QDir(masks_directory).filePath(QStringLiteral("custom/custom-mask.png"));
        ASSERT_TRUE(QDir().mkpath(QFileInfo(custom_path).absolutePath()));
        for (const auto& entry : QVector<QPair<QString, QByteArray>>{
                 {custom_path, QByteArrayLiteral("custom-mask")}, {standard_path, QByteArrayLiteral("standard-mask")}})
        {
            QFile file(entry.first);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            ASSERT_EQ(file.write(entry.second), entry.second.size());
        }
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), custom_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        bool generation_advanced = false;
        bool tombstone_exposed_during_commit = false;
        QObject::connect(
            &project_data,
            &ProjectData::metadataChanged,
            &session,
            [&](const QJsonObject&)
            {
                if (generation_advanced)
                {
                    return;
                }
                QDirIterator iterator(masks_directory,
                                      {QStringLiteral("*.plascan-mask-*.tombstone")},
                                      QDir::Files,
                                      QDirIterator::Subdirectories);
                while (iterator.hasNext())
                {
                    const QString tombstone = iterator.next();
                    if (tombstone.startsWith(standard_path + QStringLiteral(".plascan-mask-")))
                    {
                        tombstone_exposed_during_commit = true;
                        QFile::remove(tombstone);
                        QDir().mkpath(tombstone);
                        break;
                    }
                }
                generation_advanced = true;
                session.advanceGeneration();
            },
            Qt::DirectConnection);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_TRUE(generation_advanced);
        EXPECT_FALSE(tombstone_exposed_during_commit);
        EXPECT_FALSE(QFileInfo::exists(custom_path));
        EXPECT_FALSE(QFileInfo::exists(standard_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(maskTaskArtifacts(standard_path).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(custom_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearPerImageRemovalFailureRestoresEveryFileBeforeMetadataMutation)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString standard_path = maskPathForImage(project_data, images.constFirst());
        const QString custom_path =
            QDir(QFileInfo(standard_path).absolutePath()).filePath(QStringLiteral("custom/custom-mask.png"));
        const QByteArray custom_bytes = QByteArrayLiteral("custom-mask");
        const QByteArray standard_bytes = QByteArrayLiteral("standard-mask");
        ASSERT_TRUE(QDir().mkpath(QFileInfo(custom_path).absolutePath()));
        for (const auto& entry :
             QVector<QPair<QString, QByteArray>>{{custom_path, custom_bytes}, {standard_path, standard_bytes}})
        {
            QFile file(entry.first);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            ASSERT_EQ(file.write(entry.second), entry.second.size());
        }
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), custom_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);
        const QJsonObject metadata_before = project_data.coreFilesMeta();
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        ProjectTaskOrchestratorMaskTestPeer::setClearRemovalFailureIndex(&orchestrator, 1);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_EQ(metadata_spy.count(), 0);
        EXPECT_EQ(project_data.coreFilesMeta(), metadata_before);
        for (const auto& entry :
             QVector<QPair<QString, QByteArray>>{{custom_path, custom_bytes}, {standard_path, standard_bytes}})
        {
            QFile restored(entry.first);
            ASSERT_TRUE(restored.open(QIODevice::ReadOnly));
            EXPECT_EQ(restored.readAll(), entry.second);
        }
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        ASSERT_EQ(messages.warningCount, 1);
        EXPECT_TRUE(messages.warningTexts.constFirst().contains(QStringLiteral("项目记录已保留")));
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(maskTaskArtifacts(standard_path).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(custom_path).isEmpty());
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
    }

    TEST(ProjectMaskWorkflowTest, ClearRecoveryCleanupFailureReportsExactTargetAndRecoveryMapping)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        const QByteArray original_bytes("clear-recovery-old-bytes\0with-tail", 34);
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write(original_bytes), original_bytes.size());
        final_file.close();
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), final_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        const bool injection_supported =
            ProjectTaskOrchestratorMaskTestPeer::setClearRecoveryCleanupFailureIndex(&orchestrator, 0);
        ASSERT_TRUE(injection_supported) << "missing private clear recovery-cleanup failure capability";
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        ScopedLoggerCapture log_capture;

        orchestrator.clearMasksForImages(images);

        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_EQ(metadata_spy.count(), 1);
        ASSERT_EQ(generated_spy.count(), 1);
        EXPECT_EQ(generated_spy.constFirst().at(0).toStringList(), QStringList{QDir::cleanPath(images.constFirst())});
        ASSERT_EQ(updated_spy.count(), 1);
        EXPECT_EQ(updated_spy.constFirst().at(0).toString(), project_data.currentProjectPath());

        QStringList recovery_paths;
        const QFileInfo final_info(final_path);
        for (const QString& artifact_name : maskTaskArtifacts(final_path))
        {
            if (artifact_name.endsWith(QStringLiteral(".recovery")))
            {
                recovery_paths.push_back(QDir(final_info.absolutePath()).filePath(artifact_name));
            }
        }
        ASSERT_EQ(recovery_paths.size(), 1);
        const QString recovery_path = recovery_paths.constFirst();
        QFile recovery(recovery_path);
        ASSERT_TRUE(recovery.open(QIODevice::ReadOnly));
        EXPECT_EQ(recovery.readAll(), original_bytes);
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::ownedArtifactCount(&orchestrator), 0);

        const QString expected_mapping = QStringLiteral("目标=%1，恢复副本=%2").arg(final_path, recovery_path);
        ASSERT_EQ(messages.criticalCount, 1);
        ASSERT_EQ(messages.criticalTexts.size(), 1);
        EXPECT_EQ(occurrenceCount(messages.criticalTexts.constFirst(), expected_mapping), 1);
        QStringList matching_log_messages;
        for (const Logger::Entry& entry : log_capture.entries())
        {
            const QString message = QString::fromUtf8(entry.message);
            if (entry.level == Logger::Error && message.contains(QStringLiteral("蒙版事务回滚失败")))
            {
                matching_log_messages.push_back(message);
            }
        }
        ASSERT_EQ(matching_log_messages.size(), 1);
        EXPECT_EQ(occurrenceCount(matching_log_messages.constFirst(), expected_mapping), 1);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
    }

    TEST(ProjectMaskWorkflowTest, ClearPreservesManagedPathReferencedByUnselectedImage)
    {
        qtApplication();
        struct Scenario
        {
            const char* name;
            bool unselectedHasExplicitRecord;
        };
        const Scenario scenarios[]{{"explicit path shared by both image records", true},
                                   {"selected explicit path aliases unselected standard path", false}};

        for (const Scenario& scenario : scenarios)
        {
            SCOPED_TRACE(scenario.name);
            QTemporaryDir temporary_directory;
            ASSERT_TRUE(temporary_directory.isValid());
            ProjectData project_data;
            QStringList images;
            ASSERT_TRUE(prepareMaskProject(
                &project_data,
                temporary_directory.path(),
                {{QStringLiteral("selected.png"), QSize(16, 16)}, {QStringLiteral("unselected.png"), QSize(16, 16)}},
                &images));
            xjw::gui::project::ProjectSession session(&project_data);
            const QString mask_directory =
                QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
            const QString shared_path = scenario.unselectedHasExplicitRecord
                                            ? QDir(mask_directory).filePath(QStringLiteral("shared/managed-mask.png"))
                                            : maskPathForImage(project_data, images.at(1));
            const QByteArray shared_bytes = scenario.unselectedHasExplicitRecord
                                                ? QByteArrayLiteral("explicit-shared-mask-bytes")
                                                : QByteArrayLiteral("implicit-standard-mask-bytes");
            ASSERT_TRUE(QDir().mkpath(QFileInfo(shared_path).absolutePath()));
            QFile shared_file(shared_path);
            ASSERT_TRUE(shared_file.open(QIODevice::WriteOnly));
            ASSERT_EQ(shared_file.write(shared_bytes), shared_bytes.size());
            shared_file.close();
            QString port_error;
            const QJsonObject shared_record{{QStringLiteral("mask_path"), shared_path},
                                            {QStringLiteral("mask_method"), QStringLiteral("threshold")}};
            QMap<QString, QJsonObject> records{{images.at(0), shared_record}};
            if (scenario.unselectedHasExplicitRecord)
            {
                records.insert(images.at(1), shared_record);
            }
            ASSERT_TRUE(session.publishImageMaskRecords(session.context(), records, nullptr, &port_error))
                << qPrintable(port_error);

            AcceptingReviewMessages messages;
            messages.questionAnswer = UiAnswer::Yes;
            xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
            QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
            QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

            orchestrator.clearMasksForImages({images.at(0)});

            QFile preserved(shared_path);
            const bool preserved_opened = preserved.open(QIODevice::ReadOnly);
            EXPECT_TRUE(preserved_opened);
            if (preserved_opened)
            {
                EXPECT_EQ(preserved.readAll(), shared_bytes);
            }
            EXPECT_FALSE(projectImageRecord(project_data, images.at(0)).contains(QStringLiteral("mask_path")));
            const QJsonObject unselected_record = projectImageRecord(project_data, images.at(1));
            if (scenario.unselectedHasExplicitRecord)
            {
                EXPECT_EQ(unselected_record.value(QStringLiteral("mask_path")).toString(),
                          QDir::cleanPath(shared_path));
            }
            else
            {
                EXPECT_FALSE(unselected_record.contains(QStringLiteral("mask_path")));
            }
            EXPECT_EQ(generated_spy.count(), 1);
            if (!generated_spy.isEmpty())
            {
                EXPECT_EQ(generated_spy.constFirst().at(0).toStringList(), QStringList{QDir::cleanPath(images.at(0))});
            }
            EXPECT_EQ(updated_spy.count(), 1);
            EXPECT_EQ(messages.informationCount, 1);
            EXPECT_EQ(messages.warningCount, 0);
            EXPECT_TRUE(maskTaskArtifacts(shared_path).isEmpty());
            EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        }
    }

    TEST(ProjectMaskWorkflowTest, ClearPhysicalParentEscapePreservesOutsideFile)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("selected.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        const QString outside_directory = QDir(temporary_directory.path()).filePath(QStringLiteral("outside"));
        ASSERT_TRUE(QDir().mkpath(mask_directory));
        ASSERT_TRUE(QDir().mkpath(outside_directory));
        const QString outside_victim = QDir(outside_directory).filePath(QStringLiteral("victim.png"));
        const QByteArray victim_bytes = QByteArrayLiteral("outside-victim-must-survive");
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(outside_victim, victim_bytes, &io_error))
            << qPrintable(io_error);
        const QString escape_directory = QDir(mask_directory).filePath(QStringLiteral("escape"));
        std::error_code link_error;
        std::filesystem::create_directory_symlink(xjw::common::io::toFilesystemPath(outside_directory),
                                                  xjw::common::io::toFilesystemPath(escape_directory),
                                                  link_error);
        if (link_error)
        {
            GTEST_SKIP() << "directory symlink unavailable: " << link_error.message();
        }
        const QFileInfo escape_info(escape_directory);
        ASSERT_TRUE(escape_info.isSymLink() || escape_info.isJunction());
        const QString selected_alias = QDir(escape_directory).filePath(QStringLiteral("victim.png"));
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), selected_alias},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_TRUE(QFileInfo::exists(outside_victim));
        QString read_error;
        EXPECT_EQ(xjw::common::io::readFileBytes(outside_victim, &read_error), victim_bytes) << qPrintable(read_error);
        const QFileInfo retained_escape_info(escape_directory);
        EXPECT_TRUE(retained_escape_info.isSymLink() || retained_escape_info.isJunction());
        expectSingleImageClearSuccess(
            project_data, images.constFirst(), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(outside_victim).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(selected_alias).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearSelectedParentAliasHonorsUnselectedDirectProtection)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("selected.png"), QSize(16, 16)}, {QStringLiteral("unselected.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        const QString real_directory = QDir(mask_directory).filePath(QStringLiteral("real"));
        const QString alias_directory = QDir(mask_directory).filePath(QStringLiteral("alias"));
        ASSERT_TRUE(QDir().mkpath(real_directory));
        const QString direct_path = QDir(real_directory).filePath(QStringLiteral("shared.png"));
        const QByteArray shared_bytes = QByteArrayLiteral("parent-alias-shared-bytes");
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(direct_path, shared_bytes, &io_error))
            << qPrintable(io_error);
        std::error_code link_error;
        std::filesystem::create_directory_symlink(xjw::common::io::toFilesystemPath(real_directory),
                                                  xjw::common::io::toFilesystemPath(alias_directory),
                                                  link_error);
        if (link_error)
        {
            GTEST_SKIP() << "directory symlink unavailable: " << link_error.message();
        }
        const QFileInfo alias_info(alias_directory);
        ASSERT_TRUE(alias_info.isSymLink() || alias_info.isJunction());
        const QString selected_alias = QDir(alias_directory).filePath(QStringLiteral("shared.png"));
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.at(0),
              QJsonObject{{QStringLiteral("mask_path"), selected_alias},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(1),
              QJsonObject{{QStringLiteral("mask_path"), direct_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages({images.at(0)});

        EXPECT_TRUE(QFileInfo::exists(direct_path));
        QString read_error;
        EXPECT_EQ(xjw::common::io::readFileBytes(direct_path, &read_error), shared_bytes) << qPrintable(read_error);
        const QJsonObject unselected_record = projectImageRecord(project_data, images.at(1));
        EXPECT_EQ(unselected_record.value(QStringLiteral("mask_path")).toString(), QDir::cleanPath(direct_path));
        const QFileInfo retained_alias_info(alias_directory);
        EXPECT_TRUE(retained_alias_info.isSymLink() || retained_alias_info.isJunction());
        expectSingleImageClearSuccess(project_data, images.at(0), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(direct_path).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(selected_alias).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearLeafSymlinkPreservesLinkAndOutsideTarget)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("selected.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        const QString outside_directory = QDir(temporary_directory.path()).filePath(QStringLiteral("outside"));
        ASSERT_TRUE(QDir().mkpath(mask_directory));
        ASSERT_TRUE(QDir().mkpath(outside_directory));
        const QString outside_target = QDir(outside_directory).filePath(QStringLiteral("target.png"));
        const QByteArray target_bytes = QByteArrayLiteral("outside-leaf-target-bytes");
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(outside_target, target_bytes, &io_error))
            << qPrintable(io_error);
        const QString leaf_link = QDir(mask_directory).filePath(QStringLiteral("selected-link.png"));
        std::error_code link_error;
        std::filesystem::create_symlink(xjw::common::io::toFilesystemPath(outside_target),
                                        xjw::common::io::toFilesystemPath(leaf_link),
                                        link_error);
        if (link_error)
        {
            GTEST_SKIP() << "file symlink unavailable: " << link_error.message();
        }
        const QFileInfo initial_link_info(leaf_link);
        ASSERT_TRUE(initial_link_info.isSymLink() || initial_link_info.isJunction());
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), leaf_link},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        const QFileInfo retained_link_info(leaf_link);
        EXPECT_TRUE(retained_link_info.isSymLink() || retained_link_info.isJunction());
        EXPECT_TRUE(QFileInfo::exists(outside_target));
        QString read_error;
        EXPECT_EQ(xjw::common::io::readFileBytes(outside_target, &read_error), target_bytes) << qPrintable(read_error);
        expectSingleImageClearSuccess(
            project_data, images.constFirst(), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(leaf_link).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(outside_target).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearUnselectedLeafSymlinkProtectsSelectedDirectTarget)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("selected.png"), QSize(16, 16)}, {QStringLiteral("unselected.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        const QString real_directory = QDir(mask_directory).filePath(QStringLiteral("real"));
        ASSERT_TRUE(QDir().mkpath(real_directory));
        const QString direct_path = QDir(real_directory).filePath(QStringLiteral("shared.png"));
        const QByteArray shared_bytes = QByteArrayLiteral("leaf-link-protected-bytes");
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(direct_path, shared_bytes, &io_error))
            << qPrintable(io_error);
        const QString unselected_link = QDir(mask_directory).filePath(QStringLiteral("unselected-link.png"));
        std::error_code link_error;
        std::filesystem::create_symlink(xjw::common::io::toFilesystemPath(direct_path),
                                        xjw::common::io::toFilesystemPath(unselected_link),
                                        link_error);
        if (link_error)
        {
            GTEST_SKIP() << "file symlink unavailable: " << link_error.message();
        }
        const QFileInfo initial_link_info(unselected_link);
        ASSERT_TRUE(initial_link_info.isSymLink() || initial_link_info.isJunction());
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.at(0),
              QJsonObject{{QStringLiteral("mask_path"), direct_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(1),
              QJsonObject{{QStringLiteral("mask_path"), unselected_link},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages({images.at(0)});

        EXPECT_TRUE(QFileInfo::exists(direct_path));
        QString read_error;
        EXPECT_EQ(xjw::common::io::readFileBytes(direct_path, &read_error), shared_bytes) << qPrintable(read_error);
        const QFileInfo retained_link_info(unselected_link);
        EXPECT_TRUE(retained_link_info.isSymLink() || retained_link_info.isJunction());
        const QJsonObject unselected_record = projectImageRecord(project_data, images.at(1));
        EXPECT_EQ(unselected_record.value(QStringLiteral("mask_path")).toString(), QDir::cleanPath(unselected_link));
        expectSingleImageClearSuccess(project_data, images.at(0), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(direct_path).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(unselected_link).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearSelectedHardlinkLeavesUnselectedHardlinkEntry)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("selected.png"), QSize(16, 16)}, {QStringLiteral("unselected.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        ASSERT_TRUE(QDir().mkpath(mask_directory));
        const QString selected_entry = QDir(mask_directory).filePath(QStringLiteral("selected-hardlink.png"));
        const QString unselected_entry = QDir(mask_directory).filePath(QStringLiteral("unselected-hardlink.png"));
        const QByteArray shared_bytes = QByteArrayLiteral("hardlink-shared-bytes");
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(selected_entry, shared_bytes, &io_error))
            << qPrintable(io_error);
        std::error_code link_error;
        std::filesystem::create_hard_link(xjw::common::io::toFilesystemPath(selected_entry),
                                          xjw::common::io::toFilesystemPath(unselected_entry),
                                          link_error);
        if (link_error)
        {
            GTEST_SKIP() << "hardlink unavailable: " << link_error.message();
        }
        std::error_code equivalent_error;
        ASSERT_TRUE(std::filesystem::equivalent(xjw::common::io::toFilesystemPath(selected_entry),
                                                xjw::common::io::toFilesystemPath(unselected_entry),
                                                equivalent_error));
        ASSERT_FALSE(equivalent_error) << equivalent_error.message();
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.at(0),
              QJsonObject{{QStringLiteral("mask_path"), selected_entry},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(1),
              QJsonObject{{QStringLiteral("mask_path"), unselected_entry},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages({images.at(0)});

        EXPECT_FALSE(QFileInfo::exists(selected_entry));
        EXPECT_TRUE(QFileInfo::exists(unselected_entry));
        QString read_error;
        EXPECT_EQ(xjw::common::io::readFileBytes(unselected_entry, &read_error), shared_bytes)
            << qPrintable(read_error);
        const QJsonObject unselected_record = projectImageRecord(project_data, images.at(1));
        EXPECT_EQ(unselected_record.value(QStringLiteral("mask_path")).toString(), QDir::cleanPath(unselected_entry));
        expectSingleImageClearSuccess(project_data, images.at(0), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(selected_entry).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(unselected_entry).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearTwoSelectedHardlinkEntriesDeletesBothNames)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("first.png"), QSize(16, 16)}, {QStringLiteral("second.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString mask_directory = QFileInfo(maskPathForImage(project_data, images.constFirst())).absolutePath();
        ASSERT_TRUE(QDir().mkpath(mask_directory));
        const QString first_entry = QDir(mask_directory).filePath(QStringLiteral("first-hardlink.png"));
        const QString second_entry = QDir(mask_directory).filePath(QStringLiteral("second-hardlink.png"));
        QString io_error;
        ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(
            first_entry, QByteArrayLiteral("two-selected-hardlink-bytes"), &io_error))
            << qPrintable(io_error);
        std::error_code link_error;
        std::filesystem::create_hard_link(xjw::common::io::toFilesystemPath(first_entry),
                                          xjw::common::io::toFilesystemPath(second_entry),
                                          link_error);
        if (link_error)
        {
            GTEST_SKIP() << "hardlink unavailable: " << link_error.message();
        }
        std::error_code equivalent_error;
        ASSERT_TRUE(std::filesystem::equivalent(xjw::common::io::toFilesystemPath(first_entry),
                                                xjw::common::io::toFilesystemPath(second_entry),
                                                equivalent_error));
        ASSERT_FALSE(equivalent_error) << equivalent_error.message();
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.at(0),
              QJsonObject{{QStringLiteral("mask_path"), first_entry},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(1),
              QJsonObject{{QStringLiteral("mask_path"), second_entry},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_FALSE(QFileInfo::exists(first_entry));
        EXPECT_FALSE(QFileInfo::exists(second_entry));
        EXPECT_FALSE(projectImageRecord(project_data, images.at(0)).contains(QStringLiteral("mask_path")));
        EXPECT_FALSE(projectImageRecord(project_data, images.at(1)).contains(QStringLiteral("mask_path")));
        ASSERT_EQ(generated_spy.count(), 1);
        QStringList reported_images = generated_spy.constFirst().at(0).toStringList();
        QStringList expected_images{QDir::cleanPath(images.at(0)), QDir::cleanPath(images.at(1))};
        reported_images.sort();
        expected_images.sort();
        EXPECT_EQ(reported_images, expected_images);
        EXPECT_EQ(updated_spy.count(), 1);
        EXPECT_EQ(messages.questionCount, 1);
        EXPECT_EQ(messages.informationCount, 1);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_TRUE(maskTaskArtifacts(first_entry).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(second_entry).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearMissingExplicitAndStandardPathsKeepsZeroFileSuccess)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("selected.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString standard_path = maskPathForImage(project_data, images.constFirst());
        const QString mask_directory = QFileInfo(standard_path).absolutePath();
        const QString missing_directory = QDir(mask_directory).filePath(QStringLiteral("missing"));
        ASSERT_TRUE(QDir().mkpath(missing_directory));
        const QString explicit_path = QDir(missing_directory).filePath(QStringLiteral("explicit.png"));
        ASSERT_FALSE(QFileInfo::exists(explicit_path));
        ASSERT_FALSE(QFileInfo::exists(standard_path));
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.constFirst(),
              QJsonObject{{QStringLiteral("mask_path"), explicit_path},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_FALSE(QFileInfo::exists(explicit_path));
        EXPECT_FALSE(QFileInfo::exists(standard_path));
        expectSingleImageClearSuccess(
            project_data, images.constFirst(), orchestrator, messages, generated_spy, updated_spy);
        EXPECT_TRUE(maskTaskArtifacts(explicit_path).isEmpty());
        EXPECT_TRUE(maskTaskArtifacts(standard_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ClearSharedSelectedPathsFormOneTransitiveFailureComponent)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(&project_data,
                                       temporary_directory.path(),
                                       {{QStringLiteral("a.png"), QSize(16, 16)},
                                        {QStringLiteral("b.png"), QSize(16, 16)},
                                        {QStringLiteral("c.png"), QSize(16, 16)}},
                                       &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString standard_b = maskPathForImage(project_data, images.at(1));
        const QString standard_c = maskPathForImage(project_data, images.at(2));
        const QString shared_ab =
            QDir(QFileInfo(standard_b).absolutePath()).filePath(QStringLiteral("shared/transitive-ab.png"));
        const QVector<QPair<QString, QByteArray>> files{{shared_ab, QByteArrayLiteral("shared-a-b")},
                                                        {standard_b, QByteArrayLiteral("shared-b-c")},
                                                        {standard_c, QByteArrayLiteral("later-c-file")}};
        for (const auto& entry : files)
        {
            ASSERT_TRUE(QDir().mkpath(QFileInfo(entry.first).absolutePath()));
            QFile file(entry.first);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            ASSERT_EQ(file.write(entry.second), entry.second.size());
        }
        QString port_error;
        ASSERT_TRUE(session.publishImageMaskRecords(
            session.context(),
            {{images.at(0),
              QJsonObject{{QStringLiteral("mask_path"), shared_ab},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(1),
              QJsonObject{{QStringLiteral("mask_path"), shared_ab},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {images.at(2),
              QJsonObject{{QStringLiteral("mask_path"), standard_b},
                          {QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            nullptr,
            &port_error))
            << qPrintable(port_error);
        const QJsonObject metadata_before = project_data.coreFilesMeta();
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        ProjectTaskOrchestratorMaskTestPeer::setClearRemovalFailureIndex(&orchestrator, 1);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_EQ(metadata_spy.count(), 0);
        EXPECT_EQ(project_data.coreFilesMeta(), metadata_before);
        for (const auto& entry : files)
        {
            QFile restored(entry.first);
            ASSERT_TRUE(restored.open(QIODevice::ReadOnly)) << qPrintable(entry.first);
            EXPECT_EQ(restored.readAll(), entry.second) << qPrintable(entry.first);
            EXPECT_TRUE(maskTaskArtifacts(entry.first).isEmpty()) << qPrintable(entry.first);
        }
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        ASSERT_EQ(messages.warningCount, 1);
        EXPECT_TRUE(messages.warningTexts.constFirst().contains(QStringLiteral("项目记录已保留")));
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
    }

    TEST(ProjectMaskWorkflowTest, ClearFileOnlySuccessUsesFilesystemSet)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile final_file(final_path);
        ASSERT_TRUE(final_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(final_file.write("file-only-mask"), 14);
        final_file.close();
        ASSERT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);

        AcceptingReviewMessages messages;
        messages.questionAnswer = UiAnswer::Yes;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.clearMasksForImages(images);

        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_EQ(metadata_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        ASSERT_EQ(generated_spy.count(), 1);
        EXPECT_EQ(generated_spy.constFirst().at(0).toStringList(), QStringList{QDir::cleanPath(images.constFirst())});
        ASSERT_EQ(messages.informationCount, 1);
        EXPECT_TRUE(messages.informationTexts.constFirst().contains(QStringLiteral("已清除 1 张照片的蒙版。")));
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
    }

    TEST(ProjectMaskWorkflowTest, GenerationAdvanceDropsQueuedInteractiveSaveWithoutAnyTail)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(255);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 1);
        EXPECT_TRUE(orchestrator.hasRunningMaskTask());
        session.advanceGeneration();
        gate.release();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(saved_spy.count(), 0);
        EXPECT_EQ(failed_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, ExplicitInteractiveCancelDropsQueuedPublicationAndPendingTail)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(127);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 1);
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("pending"), 2);
        orchestrator.cancelMaskGeneration();
        gate.release();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(saved_spy.count(), 0);
        EXPECT_EQ(failed_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, BatchMetadataNoOpStillReportsPublishedImageAndKeepsOldFinalReadable)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("first.png"), QSize(32, 32)}, {QStringLiteral("second.png"), QSize(32, 32)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        const QStringList final_paths{maskPathForImage(project_data, images.at(0)),
                                      maskPathForImage(project_data, images.at(1))};
        const QVector<QByteArray> original_bytes{QByteArrayLiteral("readable-old-final-first"),
                                                 QByteArrayLiteral("readable-old-final-second")};
        for (int index = 0; index < final_paths.size(); ++index)
        {
            ASSERT_TRUE(QDir().mkpath(QFileInfo(final_paths.at(index)).absolutePath()));
            QFile original(final_paths.at(index));
            ASSERT_TRUE(original.open(QIODevice::WriteOnly));
            ASSERT_EQ(original.write(original_bytes.at(index)), original_bytes.at(index).size());
        }
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        QStringList observed_backup_images;
        bool all_old_finals_visible = true;
        bool every_backup_matches = true;
        bool duplicate_backup_callback = false;
        bool unknown_backup_artifact = false;
        bool seeded_every_exact_record = true;
        QStringList seed_errors;
        int metadata_count_after_seed = -1;
        const bool observer_supported = ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(
            &orchestrator,
            [&](const QString& phase, const StagedMaskArtifact& artifact, const QString& backup_path)
            {
                if (phase != QLatin1String("backups_prepared"))
                {
                    return;
                }
                for (int index = 0; index < final_paths.size(); ++index)
                {
                    QFile current_final(final_paths.at(index));
                    all_old_finals_visible = all_old_finals_visible && current_final.open(QIODevice::ReadOnly) &&
                                             current_final.readAll() == original_bytes.at(index);
                }

                const QString normalized_image = QDir::cleanPath(artifact.imagePath);
                const int artifact_index = images.indexOf(normalized_image);
                if (artifact_index < 0)
                {
                    unknown_backup_artifact = true;
                    return;
                }
                if (observed_backup_images.contains(normalized_image))
                {
                    duplicate_backup_callback = true;
                    return;
                }
                observed_backup_images.push_back(normalized_image);
                QFile backup(backup_path);
                every_backup_matches = every_backup_matches && backup.open(QIODevice::ReadOnly) &&
                                       backup.readAll() == original_bytes.at(artifact_index);
                QString seed_error;
                const bool seeded = session.publishImageMaskRecords(
                    session.context(), {{artifact.imagePath, artifact.record}}, nullptr, &seed_error);
                seeded_every_exact_record = seeded_every_exact_record && seeded;
                if (!seeded)
                {
                    seed_errors.push_back(seed_error);
                }
                if (observed_backup_images.size() == images.size())
                {
                    metadata_count_after_seed = metadata_spy.count();
                }
            });
        ASSERT_TRUE(observer_supported) << "missing private publication observer capability";

        orchestrator.openGenerateMaskDialogForImages(images);
        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_TRUE(ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(&orchestrator, {}));

        EXPECT_EQ(observed_backup_images.size(), images.size());
        EXPECT_FALSE(duplicate_backup_callback);
        EXPECT_FALSE(unknown_backup_artifact);
        EXPECT_TRUE(all_old_finals_visible);
        EXPECT_TRUE(every_backup_matches);
        EXPECT_TRUE(seeded_every_exact_record) << qPrintable(seed_errors.join(QStringLiteral("; ")));
        EXPECT_GE(metadata_count_after_seed, 0);
        EXPECT_EQ(metadata_spy.count(), metadata_count_after_seed);
        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_TRUE(finished_spy.constFirst().at(0).toBool());
        ASSERT_EQ(generated_spy.count(), 1);
        QStringList reported_images = generated_spy.constFirst().at(0).toStringList();
        QStringList expected_images{QDir::cleanPath(images.at(0)), QDir::cleanPath(images.at(1))};
        reported_images.sort();
        expected_images.sort();
        EXPECT_EQ(reported_images, expected_images);
        EXPECT_EQ(updated_spy.count(), 0);
        ASSERT_EQ(messages.informationCount, 1);
        EXPECT_TRUE(messages.informationTexts.constFirst().contains(QStringLiteral("已生成 2 张照片的蒙版。")));
        EXPECT_EQ(messages.warningCount, 0);
        for (int index = 0; index < final_paths.size(); ++index)
        {
            QFile published(final_paths.at(index));
            ASSERT_TRUE(published.open(QIODevice::ReadOnly));
            EXPECT_NE(published.readAll(), original_bytes.at(index));
            EXPECT_FALSE(QImage(final_paths.at(index)).isNull());
            EXPECT_TRUE(maskTaskArtifacts(final_paths.at(index)).isEmpty());
        }
    }

    TEST(ProjectMaskWorkflowTest, InteractiveMetadataNoOpEmitsSavedWithoutMetadataUpdated)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(32, 32)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        const QByteArray original_bytes = QByteArrayLiteral("interactive-old-final");
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        QFile original(final_path);
        ASSERT_TRUE(original.open(QIODevice::WriteOnly));
        ASSERT_EQ(original.write(original_bytes), original_bytes.size());
        original.close();
        QSignalSpy metadata_spy(&project_data, &ProjectData::metadataChanged);
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        bool backup_phase_seen = false;
        bool seeded_exact_record = false;
        QString seed_error;
        int metadata_count_after_seed = -1;
        const bool observer_supported = ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(
            &orchestrator,
            [&](const QString& phase, const StagedMaskArtifact& artifact, const QString&)
            {
                if (phase != QLatin1String("backups_prepared") || backup_phase_seen)
                {
                    return;
                }
                backup_phase_seen = true;
                seeded_exact_record = session.publishImageMaskRecords(
                    session.context(), {{artifact.imagePath, artifact.record}}, nullptr, &seed_error);
                metadata_count_after_seed = metadata_spy.count();
            });
        ASSERT_TRUE(observer_supported) << "missing private publication observer capability";
        QImage mask(QSize(32, 32), QImage::Format_Grayscale8);
        mask.fill(191);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 11);
        QTRY_COMPARE_WITH_TIMEOUT(saved_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_TRUE(ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(&orchestrator, {}));

        EXPECT_TRUE(backup_phase_seen);
        EXPECT_TRUE(seeded_exact_record) << qPrintable(seed_error);
        EXPECT_GE(metadata_count_after_seed, 0);
        EXPECT_EQ(metadata_spy.count(), metadata_count_after_seed);
        EXPECT_EQ(failed_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(generated_spy.count(), 0);
        ASSERT_EQ(saved_spy.count(), 1);
        EXPECT_EQ(saved_spy.constFirst().at(0).toString(), QDir::cleanPath(images.constFirst()));
        EXPECT_EQ(saved_spy.constFirst().at(1).toULongLong(), 11u);
        QFile published(final_path);
        ASSERT_TRUE(published.open(QIODevice::ReadOnly));
        EXPECT_NE(published.readAll(), original_bytes);
        EXPECT_FALSE(QImage(final_path).isNull());
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, SameGenerationPortRejectionRestoresExistingFinalByteForByte)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(QFileInfo(final_path).absolutePath()));
        const QByteArray original_bytes("existing-final-mask-bytes\0with-tail", 35);
        QFile original(final_path);
        ASSERT_TRUE(original.open(QIODevice::WriteOnly));
        ASSERT_EQ(original.write(original_bytes), original_bytes.size());
        original.close();
        bool backups_prepared_seen = false;
        bool old_final_visible_before_publish = false;
        bool old_backup_complete_before_publish = false;
        bool rollback_restore_seen = false;
        bool new_final_complete_before_restore = false;
        bool old_backup_complete_before_restore = false;
        const bool observer_supported = ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(
            &orchestrator,
            [&](const QString& phase, const StagedMaskArtifact& artifact, const QString& backup_path)
            {
                if (phase == QLatin1String("backups_prepared"))
                {
                    backups_prepared_seen = true;
                    QFile current_final(artifact.finalPath);
                    old_final_visible_before_publish =
                        current_final.open(QIODevice::ReadOnly) && current_final.readAll() == original_bytes;
                    QFile backup(backup_path);
                    old_backup_complete_before_publish =
                        backup.open(QIODevice::ReadOnly) && backup.readAll() == original_bytes;
                }
                else if (phase == QLatin1String("before_rollback_restore"))
                {
                    rollback_restore_seen = true;
                    QFile current_final(artifact.finalPath);
                    const QByteArray current_bytes =
                        current_final.open(QIODevice::ReadOnly) ? current_final.readAll() : QByteArray{};
                    new_final_complete_before_restore = !current_bytes.isEmpty() && current_bytes != original_bytes &&
                                                        !QImage(artifact.finalPath).isNull();
                    QFile backup(backup_path);
                    old_backup_complete_before_restore =
                        backup.open(QIODevice::ReadOnly) && backup.readAll() == original_bytes;
                }
            });
        ASSERT_TRUE(observer_supported) << "missing private publication observer capability";
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(255);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 7);
        QJsonObject current_metadata = project_data.coreFilesMeta();
        current_metadata.insert(QStringLiteral("images"), QJsonArray());
        project_data.persistMetadata(current_metadata, false);
        gate.release();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(saved_spy.count(), 0);
        ASSERT_EQ(failed_spy.count(), 1);
        EXPECT_EQ(failed_spy.at(0).at(1).toULongLong(), 7u);
        EXPECT_EQ(updated_spy.count(), 0);
        QFile restored(final_path);
        ASSERT_TRUE(restored.open(QIODevice::ReadOnly));
        EXPECT_EQ(restored.readAll(), original_bytes);
        EXPECT_TRUE(backups_prepared_seen);
        EXPECT_TRUE(old_final_visible_before_publish);
        EXPECT_TRUE(old_backup_complete_before_publish);
        EXPECT_TRUE(rollback_restore_seen);
        EXPECT_TRUE(new_final_complete_before_restore);
        EXPECT_TRUE(old_backup_complete_before_restore);
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, BeforeRollbackRestoreOwnerDestructionLogsEveryRecoveryMappingOnce)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("first.png"), QSize(32, 32)}, {QStringLiteral("second.png"), QSize(32, 32)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        const QStringList final_paths{QDir::cleanPath(maskPathForImage(project_data, images.at(0))),
                                      QDir::cleanPath(maskPathForImage(project_data, images.at(1)))};
        const QVector<QByteArray> original_bytes{QByteArray("rollback-old-first\0tail", 23),
                                                 QByteArray("rollback-old-second\0tail", 24)};
        QMap<QString, QByteArray> original_by_final;
        for (int index = 0; index < final_paths.size(); ++index)
        {
            ASSERT_TRUE(QDir().mkpath(QFileInfo(final_paths.at(index)).absolutePath()));
            QFile original(final_paths.at(index));
            ASSERT_TRUE(original.open(QIODevice::WriteOnly));
            ASSERT_EQ(original.write(original_bytes.at(index)), original_bytes.at(index).size());
            original_by_final.insert(final_paths.at(index), original_bytes.at(index));
        }

        using Orchestrator = xjw::gui::project::ProjectTaskOrchestrator;
        QPointer<Orchestrator> orchestrator_guard = new Orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        QMap<QString, QString> backup_by_final;
        QMap<QString, QByteArray> published_bytes_by_final;
        int backup_callbacks = 0;
        int rollback_callbacks = 0;
        bool callback_files_complete = true;
        bool port_rejection_armed = false;
        bool owner_destroyed_inside_observer = false;
        const bool observer_supported = ProjectTaskOrchestratorMaskTestPeer::setPublicationObserver(
            orchestrator_guard.data(),
            [&](const QString& phase, const StagedMaskArtifact& artifact, const QString& backup_path)
            {
                const QString final_path = QDir::cleanPath(artifact.finalPath);
                if (phase == QLatin1String("backups_prepared"))
                {
                    ++backup_callbacks;
                    backup_by_final.insert(final_path, backup_path);
                    QFile staged(artifact.stagingPath);
                    const bool staged_opened = staged.open(QIODevice::ReadOnly);
                    const QByteArray staged_bytes = staged_opened ? staged.readAll() : QByteArray{};
                    published_bytes_by_final.insert(final_path, staged_bytes);
                    QFile backup(backup_path);
                    const bool backup_opened = backup.open(QIODevice::ReadOnly);
                    callback_files_complete = callback_files_complete && staged_opened && !staged_bytes.isEmpty() &&
                                              backup_opened && backup.readAll() == original_by_final.value(final_path);
                    if (backup_callbacks == images.size())
                    {
                        QJsonObject metadata = project_data.coreFilesMeta();
                        metadata.insert(QStringLiteral("images"), QJsonArray());
                        project_data.persistMetadata(metadata, false);
                        port_rejection_armed =
                            project_data.coreFilesMeta().value(QStringLiteral("images")).toArray().isEmpty();
                    }
                    return;
                }
                if (phase != QLatin1String("before_rollback_restore"))
                {
                    return;
                }
                ++rollback_callbacks;
                if (rollback_callbacks != 1)
                {
                    return;
                }
                auto* doomed = orchestrator_guard.data();
                if (doomed)
                {
                    doomed->deleteLater();
                    QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
                }
                owner_destroyed_inside_observer = orchestrator_guard.isNull();
            });
        ASSERT_TRUE(observer_supported) << "missing private publication observer capability";
        ScopedLoggerCapture log_capture;

        orchestrator_guard->openGenerateMaskDialogForImages(images);
        QTRY_VERIFY_WITH_TIMEOUT(orchestrator_guard.isNull(), 5000);

        EXPECT_TRUE(owner_destroyed_inside_observer);
        EXPECT_TRUE(port_rejection_armed);
        EXPECT_EQ(backup_callbacks, images.size());
        EXPECT_EQ(rollback_callbacks, 1);
        EXPECT_TRUE(callback_files_complete);
        EXPECT_EQ(backup_by_final.size(), final_paths.size());
        EXPECT_EQ(published_bytes_by_final.size(), final_paths.size());
        QStringList expected_mappings;
        for (const QString& final_path : final_paths)
        {
            ASSERT_TRUE(backup_by_final.contains(final_path)) << qPrintable(final_path);
            const QString backup_path = backup_by_final.value(final_path);
            QFile published(final_path);
            ASSERT_TRUE(published.open(QIODevice::ReadOnly)) << qPrintable(final_path);
            EXPECT_EQ(published.readAll(), published_bytes_by_final.value(final_path)) << qPrintable(final_path);
            EXPECT_FALSE(QImage(final_path).isNull()) << qPrintable(final_path);
            QFile backup(backup_path);
            ASSERT_TRUE(backup.open(QIODevice::ReadOnly)) << qPrintable(backup_path);
            EXPECT_EQ(backup.readAll(), original_by_final.value(final_path)) << qPrintable(backup_path);
            expected_mappings.push_back(QStringLiteral("目标=%1，恢复副本=%2").arg(final_path, backup_path));
        }
        EXPECT_EQ(messages.criticalCount, 0);

        QStringList matching_log_messages;
        for (const Logger::Entry& entry : log_capture.entries())
        {
            const QString message = QString::fromUtf8(entry.message);
            if (entry.level == Logger::Error && message.contains(QStringLiteral("蒙版事务回滚失败")))
            {
                matching_log_messages.push_back(message);
            }
        }
        ASSERT_EQ(matching_log_messages.size(), 1);
        for (const QString& expected_mapping : expected_mappings)
        {
            EXPECT_EQ(occurrenceCount(matching_log_messages.constFirst(), expected_mapping), 1)
                << qPrintable(expected_mapping);
        }
        if (orchestrator_guard)
        {
            delete orchestrator_guard.data();
        }
    }

    TEST(ProjectMaskWorkflowTest, GenerationAdvanceDropsQueuedBatchArtifactAndAllLateTails)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(32, 32)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        QSignalSpy progress_spy(&orchestrator,
                                &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());

        orchestrator.openGenerateMaskDialogForImages(images);
        const int initial_progress_count = progress_spy.count();
        EXPECT_EQ(initial_progress_count, 1);
        session.advanceGeneration();
        gate.release();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(progress_spy.count(), initial_progress_count);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, MetadataChangedReentryKeepsCommitAndSuppressesEveryMaskTail)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        QObject::connect(
            &project_data,
            &ProjectData::metadataChanged,
            &session,
            [&session](const QJsonObject&) { session.advanceGeneration(); },
            Qt::DirectConnection);
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(255);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 9);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        const QString final_path = maskPathForImage(project_data, images.constFirst());
        EXPECT_TRUE(QFileInfo::exists(final_path));
        const QJsonObject image = projectImageRecord(project_data, images.constFirst());
        EXPECT_EQ(image.value(QStringLiteral("mask_path")).toString(), QDir::cleanPath(final_path));
        EXPECT_EQ(image.value(QStringLiteral("mask_method")).toString(), QStringLiteral("interactive"));
        EXPECT_EQ(saved_spy.count(), 0);
        EXPECT_EQ(failed_spy.count(), 0);
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, InteractiveRevisionsOneTwoThreePublishOnlyOneAndThree)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage first(QSize(16, 16), QImage::Format_Grayscale8);
        QImage second(QSize(16, 16), QImage::Format_Grayscale8);
        QImage third(QSize(16, 16), QImage::Format_Grayscale8);
        first.fill(32);
        second.fill(96);
        third.fill(224);

        orchestrator.saveInteractiveMask(images.constFirst(), first, QStringLiteral("revision-one"), 1);
        orchestrator.saveInteractiveMask(images.constFirst(), second, QStringLiteral("revision-two"), 2);
        orchestrator.saveInteractiveMask(images.constFirst(), third, QStringLiteral("revision-three"), 3);
        gate.release();
        QTRY_COMPARE_WITH_TIMEOUT(saved_spy.count(), 2, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(failed_spy.count(), 0);
        EXPECT_EQ(saved_spy.at(0).at(1).toULongLong(), 1u);
        EXPECT_EQ(saved_spy.at(1).at(1).toULongLong(), 3u);
        EXPECT_EQ(projectImageRecord(project_data, images.constFirst()).value(QStringLiteral("mask_method")).toString(),
                  QStringLiteral("revision-three"));
    }

    TEST(ProjectMaskWorkflowTest, InteractiveFailureStillAdvancesToLatestPendingRevision)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ASSERT_TRUE(QDir().mkpath(final_path));
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QObject::connect(
            &orchestrator,
            &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed,
            &orchestrator,
            [final_path](const QString&, quint64 revision, const QString&)
            {
                if (revision == 1)
                {
                    QDir(final_path).removeRecursively();
                }
            },
            Qt::DirectConnection);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(255);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("revision-one"), 1);
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("revision-two"), 2);
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("revision-three"), 3);
        gate.release();
        QTRY_COMPARE_WITH_TIMEOUT(failed_spy.count(), 1, 5000);
        QTRY_COMPARE_WITH_TIMEOUT(saved_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_EQ(failed_spy.at(0).at(1).toULongLong(), 1u);
        EXPECT_EQ(saved_spy.at(0).at(1).toULongLong(), 3u);
        EXPECT_TRUE(QFileInfo(final_path).isFile());
        EXPECT_EQ(projectImageRecord(project_data, images.constFirst()).value(QStringLiteral("mask_method")).toString(),
                  QStringLiteral("revision-three"));
    }

    TEST(ProjectMaskWorkflowTest, MaskLaneRunsBesideSparseAndRejectsOrCoalescesConflictingMaskCommands)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        int provider_calls = 0;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [&](const QStringList&, const QString&) -> std::optional<QJsonObject>
            {
                ++provider_calls;
                return selectedMaskSettings(images);
            },
            nullptr);
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        const auto sparse_context = orchestrator.context(QStringLiteral("triangulation"));
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        ScopedGlobalThreadPoolGate gate;
        ASSERT_TRUE(gate.isReady());
        QImage mask(QSize(16, 16), QImage::Format_Grayscale8);
        mask.fill(255);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("first"), 1);
        orchestrator.openGenerateMaskDialogForImages(images);
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("second"), 2);
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("latest"), 3);
        EXPECT_TRUE(orchestrator.hasActiveTask());
        EXPECT_TRUE(orchestrator.hasRunningMaskTask());
        EXPECT_EQ(provider_calls, 0);
        EXPECT_EQ(messages.warningCount, 1);
        gate.release();
        QTRY_COMPARE_WITH_TIMEOUT(saved_spy.count(), 2, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_TRUE(orchestrator.hasActiveTask());
        EXPECT_EQ(saved_spy.at(0).at(1).toULongLong(), 1u);
        EXPECT_EQ(saved_spy.at(1).at(1).toULongLong(), 3u);
        EXPECT_TRUE(orchestrator.finishTask(sparse_context, true));
    }

    TEST(ProjectMaskWorkflowTest, MaskLaneRunsBesideBundleAdjustWithoutSharingItsActiveContext)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("first.png"), QSize(16, 16)}, {QStringLiteral("second.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            nullptr,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings({images.constFirst()}); },
            nullptr);
        auto* bundle_adjust = ownedBaController(&orchestrator);
        ASSERT_NE(bundle_adjust, nullptr);
        ScopedWorkerGate bundle_adjust_gate;
        bundle_adjust->setExecutionRunnerForTesting(
            [&bundle_adjust_gate](
                const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            {
                bundle_adjust_gate.enterAndWait();
                return xjw::gui::project::BundleAdjustExecutionResult{};
            });
        ScopedArtifactStageGate mask_gate;
        ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, mask_gate.observer());
        QSignalSpy mask_finished(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temporary_directory.path(), 1, true, {}));
        ASSERT_TRUE(bundle_adjust_gate.waitUntilEntered());
        orchestrator.openGenerateMaskDialogForImages({images.constFirst()});
        ASSERT_TRUE(mask_gate.waitForFirstStaged());

        EXPECT_TRUE(orchestrator.hasActiveTask());
        EXPECT_TRUE(orchestrator.hasRunningBundleAdjustTask());
        EXPECT_TRUE(orchestrator.hasRunningMaskTask());

        mask_gate.release();
        QTRY_COMPARE_WITH_TIMEOUT(mask_finished.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_TRUE(mask_finished.at(0).at(0).toBool());
        EXPECT_TRUE(orchestrator.hasActiveTask());
        EXPECT_TRUE(orchestrator.hasRunningBundleAdjustTask());

        orchestrator.cancelActiveTask();
        bundle_adjust_gate.release();
        orchestrator.waitForActiveTask();
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningBundleAdjustTask(), 5000);
    }

    TEST(ProjectMaskWorkflowTest, ExplicitBatchCancelCommitsOnlyFullyStagedPrefixAndOneFalseTerminal)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("small.png"), QSize(16, 16)}, {QStringLiteral("second.png"), QSize(16, 16)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        ScopedArtifactStageGate stage_gate;
        ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, stage_gate.observer());

        orchestrator.openGenerateMaskDialogForImages(images);
        ASSERT_TRUE(stage_gate.waitForFirstStaged());
        orchestrator.cancelMaskGeneration();
        stage_gate.release();
        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 1, 15000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 15000);

        EXPECT_FALSE(finished_spy.at(0).at(0).toBool());
        ASSERT_EQ(generated_spy.count(), 1);
        EXPECT_EQ(generated_spy.at(0).at(0).toStringList(), QStringList{QDir::cleanPath(images.at(0))});
        EXPECT_TRUE(QFileInfo::exists(maskPathForImage(project_data, images.at(0))));
        EXPECT_FALSE(QFileInfo::exists(maskPathForImage(project_data, images.at(1))));
        EXPECT_TRUE(projectImageRecord(project_data, images.at(0)).contains(QStringLiteral("mask_path")));
        EXPECT_FALSE(projectImageRecord(project_data, images.at(1)).contains(QStringLiteral("mask_path")));
        ASSERT_EQ(messages.informationCount, 1);
        EXPECT_TRUE(messages.informationTexts.constFirst().contains(QStringLiteral("已取消，已保留 1 张照片的蒙版。")));
    }

    TEST(ProjectMaskWorkflowTest, InitialProgressDirectCancelReportsOneFalseTerminalWithoutLaunchingWorker)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        int initial_progress_count = 0;
        QObject::connect(
            &orchestrator,
            &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged,
            &orchestrator,
            [&](const QString&, int done, int)
            {
                if (done == 0)
                {
                    ++initial_progress_count;
                    orchestrator.cancelMaskGeneration();
                }
            },
            Qt::DirectConnection);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);

        orchestrator.openGenerateMaskDialogForImages(images);

        EXPECT_EQ(initial_progress_count, 1);
        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_FALSE(finished_spy.constFirst().at(0).toBool());
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        ASSERT_EQ(messages.warningCount, 1);
        EXPECT_TRUE(messages.warningTexts.constFirst().contains(QStringLiteral("蒙版生成已取消。")));
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, PostCommitMetadataSignalDirectCancelKeepsCommitAndUsesCancelTerminal)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        QObject::connect(
            &orchestrator,
            &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated,
            &orchestrator,
            [&]() { orchestrator.cancelMaskGeneration(); },
            Qt::DirectConnection);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        orchestrator.openGenerateMaskDialogForImages(images);
        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        ASSERT_EQ(updated_spy.count(), 1);
        ASSERT_EQ(generated_spy.count(), 1);
        EXPECT_FALSE(finished_spy.constFirst().at(0).toBool());
        ASSERT_EQ(messages.informationCount, 1);
        EXPECT_TRUE(messages.informationTexts.constFirst().contains(QStringLiteral("已取消，已保留 1 张照片的蒙版。")));
        EXPECT_EQ(messages.warningCount, 0);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_TRUE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, PostCommitInformationDirectCancelDowngradesTerminal)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        messages.informationAction = [&]() { orchestrator.cancelMaskGeneration(); };
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        orchestrator.openGenerateMaskDialogForImages(images);
        QTRY_COMPARE_WITH_TIMEOUT(finished_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_FALSE(finished_spy.constFirst().at(0).toBool());
        EXPECT_EQ(messages.informationCount, 1);
        EXPECT_EQ(messages.warningCount, 0);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_TRUE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, PublishPortSessionDestructionCommitsPrimitiveAndSuppressesTail)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(16, 16)}}, &images));
        QPointer<xjw::gui::project::ProjectSession> session_guard =
            new xjw::gui::project::ProjectSession(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            session_guard.data(),
            &messages,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy generated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy updated_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);
        bool deleted_inside_port = false;
        QObject::connect(
            &project_data,
            &ProjectData::metadataChanged,
            &orchestrator,
            [&](const QJsonObject&)
            {
                auto* doomed = session_guard.data();
                if (!doomed)
                {
                    return;
                }
                doomed->deleteLater();
                QCoreApplication::sendPostedEvents(doomed, QEvent::DeferredDelete);
                deleted_inside_port = session_guard.isNull();
            },
            Qt::DirectConnection);

        orchestrator.openGenerateMaskDialogForImages(images);
        QTRY_VERIFY_WITH_TIMEOUT(deleted_inside_port, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_TRUE(session_guard.isNull());
        EXPECT_EQ(orchestrator.session(), nullptr);
        EXPECT_EQ(updated_spy.count(), 0);
        EXPECT_EQ(generated_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.criticalCount, 0);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        EXPECT_TRUE(QFileInfo::exists(final_path));
        EXPECT_TRUE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
    }

    TEST(ProjectMaskWorkflowTest, WaitForActiveTaskJoinsInteractiveAndBatchFutures)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data,
            temporary_directory.path(),
            {{QStringLiteral("interactive.png"), QSize(32, 32)}, {QStringLiteral("batch.png"), QSize(32, 32)}},
            &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            nullptr,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings({images.at(1)}); },
            nullptr);
        QImage mask(QSize(32, 32), QImage::Format_Grayscale8);
        mask.fill(255);
        QSignalSpy interactive_saved(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy batch_finished(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);
        QSignalSpy masks_generated(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated);
        QSignalSpy metadata_updated(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated);

        {
            ScopedArtifactStageGate stage_gate;
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, stage_gate.observer());
            orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 1);
            ASSERT_TRUE(stage_gate.waitForFirstStaged());
            QSemaphore wait_call_entered;
            std::atomic<bool> wait_returned{false};
            std::atomic<bool> returned_before_release{true};
            ScopedThreadJoiner releaser(
                [&]()
                {
                    wait_call_entered.acquire();
                    QThread::msleep(100);
                    returned_before_release.store(wait_returned.load());
                    stage_gate.release();
                });
            EXPECT_TRUE(orchestrator.hasRunningMaskTask());
            wait_call_entered.release();
            orchestrator.waitForActiveTask();
            wait_returned.store(true);
            releaser.join();
            EXPECT_FALSE(returned_before_release.load());
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, {});
        }
        QTRY_COMPARE_WITH_TIMEOUT(interactive_saved.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_EQ(interactive_saved.at(0).at(0).toString(), QDir::cleanPath(images.at(0)));
        EXPECT_EQ(interactive_saved.at(0).at(1).toULongLong(), 1U);
        EXPECT_TRUE(QFileInfo::exists(maskPathForImage(project_data, images.at(0))));
        EXPECT_TRUE(projectImageRecord(project_data, images.at(0)).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(maskPathForImage(project_data, images.at(0))).isEmpty());
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::ownedArtifactCount(&orchestrator), 0);
        EXPECT_EQ(metadata_updated.count(), 1);

        {
            ScopedArtifactStageGate stage_gate;
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, stage_gate.observer());
            orchestrator.openGenerateMaskDialogForImages(images);
            ASSERT_TRUE(stage_gate.waitForFirstStaged());
            QSemaphore wait_call_entered;
            std::atomic<bool> wait_returned{false};
            std::atomic<bool> returned_before_release{true};
            ScopedThreadJoiner releaser(
                [&]()
                {
                    wait_call_entered.acquire();
                    QThread::msleep(100);
                    returned_before_release.store(wait_returned.load());
                    stage_gate.release();
                });
            EXPECT_TRUE(orchestrator.hasRunningMaskTask());
            wait_call_entered.release();
            orchestrator.waitForActiveTask();
            wait_returned.store(true);
            releaser.join();
            EXPECT_FALSE(returned_before_release.load());
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, {});
        }
        QTRY_COMPARE_WITH_TIMEOUT(batch_finished.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_TRUE(batch_finished.at(0).at(0).toBool());
        ASSERT_EQ(masks_generated.count(), 1);
        EXPECT_EQ(masks_generated.at(0).at(0).toStringList(), QStringList{QDir::cleanPath(images.at(1))});
        EXPECT_TRUE(QFileInfo::exists(maskPathForImage(project_data, images.at(1))));
        EXPECT_TRUE(projectImageRecord(project_data, images.at(1)).contains(QStringLiteral("mask_path")));
        EXPECT_TRUE(maskTaskArtifacts(maskPathForImage(project_data, images.at(1))).isEmpty());
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::ownedArtifactCount(&orchestrator), 0);
        EXPECT_EQ(metadata_updated.count(), 2);
    }

    TEST(ProjectMaskWorkflowTest, InteractiveWorkerExceptionCleansStagingBeforeReportingFailure)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(32, 32)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        QImage mask(QSize(32, 32), QImage::Format_Grayscale8);
        mask.fill(255);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(
            &orchestrator, [](int) { throw std::runtime_error("controlled stage observer failure"); });

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 7);
        QTRY_COMPARE_WITH_TIMEOUT(failed_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);

        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::ownedArtifactCount(&orchestrator), 0);
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
    }

    TEST(ProjectMaskWorkflowTest, FinishedMaskFuturesArePrunedDuringNormalUse)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(32, 32)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(
            &session,
            nullptr,
            [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
            { return selectedMaskSettings(images); },
            nullptr);
        const std::optional<int> initial_count = ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator);
        ASSERT_TRUE(initial_count.has_value()) << "missing private mask-future count capability";
        EXPECT_EQ(*initial_count, 0);

        QPromise<void> completed_promise;
        completed_promise.start();
        QFuture<void> completed_future = completed_promise.future();
        completed_promise.finish();
        completed_future.waitForFinished();
        ASSERT_TRUE(ProjectTaskOrchestratorMaskTestPeer::trackFuture(&orchestrator, completed_future))
            << "missing private mask-future registration capability";
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        QImage mask(QSize(32, 32), QImage::Format_Grayscale8);
        mask.fill(223);
        QSignalSpy saved_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved);
        QSignalSpy failed_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed);
        QSignalSpy batch_finished(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished);

        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("success"), 1);
        QTRY_COMPARE_WITH_TIMEOUT(saved_spy.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        orchestrator.openGenerateMaskDialogForImages(images);
        QTRY_COMPARE_WITH_TIMEOUT(batch_finished.count(), 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        {
            ScopedGlobalThreadPoolGate gate;
            ASSERT_TRUE(gate.isReady());
            orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("stale"), 2);
            session.advanceGeneration();
            gate.release();
            QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        }
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(
            &orchestrator, [](int) { throw std::runtime_error("controlled future-pruning failure"); });
        const int failed_before_exception = failed_spy.count();
        orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("exception"), 3);
        QTRY_COMPARE_WITH_TIMEOUT(failed_spy.count(), failed_before_exception + 1, 5000);
        QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(&orchestrator, {});
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        {
            ScopedGlobalThreadPoolGate gate;
            ASSERT_TRUE(gate.isReady());
            orchestrator.saveInteractiveMask(images.constFirst(), mask, QStringLiteral("cancelled"), 4);
            orchestrator.cancelMaskGeneration();
            gate.release();
            QTRY_VERIFY_WITH_TIMEOUT(!orchestrator.hasRunningMaskTask(), 5000);
        }
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);

        const int failed_before_early_return = failed_spy.count();
        orchestrator.saveInteractiveMask(
            QDir(temporary_directory.path()).filePath(QStringLiteral("not-a-project-image.png")),
            mask,
            QStringLiteral("early-return"),
            5);
        QTRY_COMPARE_WITH_TIMEOUT(failed_spy.count(), failed_before_early_return + 1, 5000);
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_EQ(ProjectTaskOrchestratorMaskTestPeer::futureCount(&orchestrator).value_or(-1), 0);
    }

    TEST(ProjectMaskWorkflowTest, OrdinaryDestructionJoinsInteractiveAndBatchFuturesAndCleansArtifacts)
    {
        qtApplication();
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        ProjectData project_data;
        QStringList images;
        ASSERT_TRUE(prepareMaskProject(
            &project_data, temporary_directory.path(), {{QStringLiteral("image.png"), QSize(32, 32)}}, &images));
        xjw::gui::project::ProjectSession session(&project_data);
        const QString final_path = maskPathForImage(project_data, images.constFirst());
        QImage mask(QSize(32, 32), QImage::Format_Grayscale8);
        mask.fill(255);

        {
            ScopedArtifactStageGate stage_gate;
            auto orchestrator = std::make_unique<xjw::gui::project::ProjectTaskOrchestrator>(&session, nullptr);
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(orchestrator.get(), stage_gate.observer());
            orchestrator->saveInteractiveMask(images.constFirst(), mask, QStringLiteral("interactive"), 1);
            ASSERT_TRUE(stage_gate.waitForFirstStaged());
            QSemaphore destruction_call_entered;
            std::atomic<bool> destruction_returned{false};
            std::atomic<bool> returned_before_release{true};
            ScopedThreadJoiner releaser(
                [&]()
                {
                    destruction_call_entered.acquire();
                    QThread::msleep(100);
                    returned_before_release.store(destruction_returned.load());
                    stage_gate.release();
                });
            EXPECT_TRUE(orchestrator->hasRunningMaskTask());
            destruction_call_entered.release();
            orchestrator.reset();
            destruction_returned.store(true);
            releaser.join();
            EXPECT_FALSE(returned_before_release.load());
        }
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());

        {
            ScopedArtifactStageGate stage_gate;
            auto orchestrator = std::make_unique<xjw::gui::project::ProjectTaskOrchestrator>(
                &session,
                nullptr,
                [images](const QStringList&, const QString&) -> std::optional<QJsonObject>
                { return selectedMaskSettings(images); },
                nullptr);
            ProjectTaskOrchestratorMaskTestPeer::setArtifactStagedObserver(orchestrator.get(), stage_gate.observer());
            orchestrator->openGenerateMaskDialogForImages(images);
            ASSERT_TRUE(stage_gate.waitForFirstStaged());
            QSemaphore destruction_call_entered;
            std::atomic<bool> destruction_returned{false};
            std::atomic<bool> returned_before_release{true};
            ScopedThreadJoiner releaser(
                [&]()
                {
                    destruction_call_entered.acquire();
                    QThread::msleep(100);
                    returned_before_release.store(destruction_returned.load());
                    stage_gate.release();
                });
            EXPECT_TRUE(orchestrator->hasRunningMaskTask());
            destruction_call_entered.release();
            orchestrator.reset();
            destruction_returned.store(true);
            releaser.join();
            EXPECT_FALSE(returned_before_release.load());
        }
        EXPECT_FALSE(QFileInfo::exists(final_path));
        EXPECT_TRUE(maskTaskArtifacts(final_path).isEmpty());
        EXPECT_FALSE(projectImageRecord(project_data, images.constFirst()).contains(QStringLiteral("mask_path")));
    }

    TEST_F(ProjectTaskOrchestratorTest, ContextCarriesSessionAndSharedCancelFlag)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sfm")));

        const auto context = orchestrator.context(QStringLiteral("sfm"));

        EXPECT_EQ(context.taskId, QStringLiteral("sfm"));
        EXPECT_TRUE(session.isCurrent(context.session));
        ASSERT_NE(context.cancelFlag, nullptr);
        EXPECT_FALSE(context.cancelFlag->load());
        EXPECT_TRUE(orchestrator.hasActiveTask());
    }

    TEST_F(ProjectTaskOrchestratorTest, BusySessionRejectsGenericAndIndependentTaskAdmissionUntilOperationEnds)
    {
        QSignalSpy dem_finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished);
        ASSERT_TRUE(session.tryBeginOperation(QStringLiteral("打开项目")));

        EXPECT_FALSE(orchestrator.beginTask(QStringLiteral("aerial_triangulation")));
        orchestrator.startDemFromPointCloudAsync(xjw::gui::project::DemGenerationRequest{});
        EXPECT_FALSE(orchestrator.hasActiveTask());
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_EQ(dem_finished_spy.count(), 0);

        session.endOperation(QStringLiteral("打开项目"));

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("aerial_triangulation")));
        orchestrator.cancelActiveTask();
        orchestrator.startDemFromPointCloudAsync(xjw::gui::project::DemGenerationRequest{});
        EXPECT_EQ(dem_finished_spy.count(), 1);
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
    }

    TEST_F(ProjectTaskOrchestratorTest, CancelActiveTaskSetsSharedFlagAndFinishesTask)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sfm")));
        const auto context = orchestrator.context(QStringLiteral("sfm"));
        QSignalSpy finishedSpy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskFinished);

        orchestrator.cancelActiveTask();

        ASSERT_NE(context.cancelFlag, nullptr);
        EXPECT_TRUE(context.cancelFlag->load());
        EXPECT_FALSE(orchestrator.hasActiveTask());
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_EQ(finishedSpy.at(0).at(0).toString(), QStringLiteral("sfm"));
        EXPECT_FALSE(finishedSpy.at(0).at(1).toBool());
    }

    TEST_F(ProjectTaskOrchestratorTest, SparseCancelBridgesOneTypedFailureAndDropsLateCompletion)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        const auto context = orchestrator.context(QStringLiteral("triangulation"));
        QSignalSpy sparseFinishedSpy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseFinished);
        QSignalSpy taskFinishedSpy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskFinished);

        orchestrator.cancelActiveTask();
        ASSERT_EQ(sparseFinishedSpy.count(), 1);
        EXPECT_FALSE(sparseFinishedSpy.at(0).at(0).toBool());
        ASSERT_EQ(taskFinishedSpy.count(), 1);
        EXPECT_FALSE(taskFinishedSpy.at(0).at(1).toBool());

        EXPECT_FALSE(orchestrator.finishTask(context, false));
        EXPECT_EQ(sparseFinishedSpy.count(), 1);
        EXPECT_EQ(taskFinishedSpy.count(), 1);
    }

    TEST_F(ProjectTaskOrchestratorTest, CancelThenStartIsGatedUntilOldSparseFutureFinishes)
    {
        QFuture<void> firstFuture = QtConcurrent::run([]() { QThread::msleep(100); });
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        orchestrator.trackSparseFutureForTesting(firstFuture);
        const auto cancelledContext = orchestrator.context(QStringLiteral("triangulation"));
        orchestrator.cancelActiveTask();

        EXPECT_FALSE(orchestrator.beginTask(QStringLiteral("sparse_refine")));
        orchestrator.waitForActiveTask();
        ASSERT_TRUE(firstFuture.isFinished());

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sparse_refine")));
        EXPECT_FALSE(orchestrator.finishTask(cancelledContext, false));
        EXPECT_TRUE(orchestrator.hasActiveTask());
        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST_F(ProjectTaskOrchestratorTest, SynchronousProgressCancelCannotRestartReservedSparseTask)
    {
        bool restartAccepted = true;
        bool emittedProgress = false;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::taskStarted,
                         &orchestrator,
                         [&](const QString& taskId)
                         {
                             if (taskId == QStringLiteral("triangulation") && !emittedProgress)
                             {
                                 emittedProgress = true;
                                 emit orchestrator.sparseProgressChanged(QStringLiteral("同步测试进度"), 10);
                             }
                         });
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged,
                         &orchestrator,
                         [&]()
                         {
                             orchestrator.cancelActiveTask();
                             restartAccepted = orchestrator.beginTask(QStringLiteral("sparse_refine"));
                         });

        orchestrator.startTriangulationAsync(QJsonObject());

        EXPECT_FALSE(restartAccepted);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        orchestrator.waitForActiveTask();
    }

    TEST_F(ProjectTaskOrchestratorTest, CompletionFromOldContextIsDropped)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sfm")));
        const auto oldContext = orchestrator.context(QStringLiteral("sfm"));
        QSignalSpy finishedSpy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskFinished);

        session.advanceGeneration();
        ASSERT_FALSE(orchestrator.finishTask(oldContext, true));

        EXPECT_EQ(finishedSpy.count(), 0);
        EXPECT_FALSE(orchestrator.hasActiveTask());
    }

    TEST_F(ProjectTaskOrchestratorTest, SessionChangeCancelsAndInvalidatesActiveTask)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sfm")));
        const auto context = orchestrator.context(QStringLiteral("sfm"));
        QSignalSpy finishedSpy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskFinished);

        session.advanceGeneration();

        ASSERT_NE(context.cancelFlag, nullptr);
        EXPECT_TRUE(context.cancelFlag->load());
        EXPECT_FALSE(orchestrator.hasActiveTask());
        EXPECT_EQ(finishedSpy.count(), 0);
    }

    TEST_F(ProjectTaskOrchestratorTest, SessionDrainDefersProjectSwitchUntilTrackedWorkerFinishes)
    {
        QTemporaryDir temporary;
        ASSERT_TRUE(temporary.isValid());
        ASSERT_TRUE(projectData.createProject(temporary.filePath(QStringLiteral("session-drain.plascan")),
                                              QStringLiteral("session-drain")));
        const QString first_chunk = projectData.activeChunkId();
        QString second_chunk;
        QString error_message;
        ASSERT_TRUE(projectData.createChunk(QStringLiteral("second"), &second_chunk, &error_message))
            << qPrintable(error_message);
        ASSERT_TRUE(projectData.switchChunk(first_chunk, &error_message)) << qPrintable(error_message);

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("aerial_triangulation")));
        const auto active_task = orchestrator.context(QStringLiteral("aerial_triangulation"));
        ScopedPendingFuture worker;
        orchestrator.trackSessionFuture(worker.future());
        const xjw::gui::project::ProjectSessionContext before = session.context();
        const QString project_path_before = projectData.currentProjectPath();
        int continuation_count = 0;
        int active_chunk_change_count = 0;
        QObject::connect(&projectData,
                         &ProjectData::activeChunkChanged,
                         &orchestrator,
                         [&active_chunk_change_count]() { ++active_chunk_change_count; });

        ASSERT_TRUE(orchestrator.cancelAndDrainForSessionChange(
            [&]()
            {
                ++continuation_count;
                EXPECT_TRUE(projectData.switchChunk(second_chunk, &error_message)) << qPrintable(error_message);
            }));

        EXPECT_TRUE(orchestrator.isSessionDrainInProgress());
        ASSERT_NE(active_task.cancelFlag, nullptr);
        EXPECT_TRUE(active_task.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_EQ(continuation_count, 0);
        EXPECT_EQ(active_chunk_change_count, 0);
        EXPECT_EQ(projectData.currentProjectPath(), project_path_before);
        EXPECT_EQ(projectData.activeChunkId(), first_chunk);
        EXPECT_TRUE(session.isCurrent(before));

        QCoreApplication::processEvents(QEventLoop::AllEvents);
        EXPECT_EQ(continuation_count, 0);
        EXPECT_EQ(projectData.activeChunkId(), first_chunk);
        EXPECT_TRUE(session.isCurrent(before));

        worker.finish();
        QTRY_COMPARE_WITH_TIMEOUT(continuation_count, 1, 5000);
        EXPECT_FALSE(orchestrator.isSessionDrainInProgress());
        EXPECT_EQ(projectData.currentProjectPath(), project_path_before);
        EXPECT_EQ(projectData.activeChunkId(), second_chunk);
        EXPECT_FALSE(session.isCurrent(before));
        EXPECT_EQ(active_chunk_change_count, 1);

        QCoreApplication::processEvents(QEventLoop::AllEvents);
        EXPECT_EQ(continuation_count, 1);
        EXPECT_EQ(active_chunk_change_count, 1);
    }

    TEST_F(ProjectTaskOrchestratorTest, SessionDrainRejectsReentryAndRunsOnlyAcceptedContinuation)
    {
        ScopedPendingFuture worker;
        orchestrator.trackSessionFuture(worker.future());
        int accepted_continuation_count = 0;
        int rejected_continuation_count = 0;

        ASSERT_TRUE(orchestrator.cancelAndDrainForSessionChange([&accepted_continuation_count]()
                                                                { ++accepted_continuation_count; }));
        ASSERT_TRUE(orchestrator.isSessionDrainInProgress());

        EXPECT_FALSE(orchestrator.cancelAndDrainForSessionChange([&rejected_continuation_count]()
                                                                 { ++rejected_continuation_count; }));
        EXPECT_TRUE(orchestrator.isSessionDrainInProgress());
        EXPECT_EQ(accepted_continuation_count, 0);
        EXPECT_EQ(rejected_continuation_count, 0);

        worker.finish();
        QTRY_COMPARE_WITH_TIMEOUT(accepted_continuation_count, 1, 5000);
        EXPECT_EQ(rejected_continuation_count, 0);
        EXPECT_FALSE(orchestrator.isSessionDrainInProgress());

        QCoreApplication::processEvents(QEventLoop::AllEvents);
        EXPECT_EQ(accepted_continuation_count, 1);
        EXPECT_EQ(rejected_continuation_count, 0);
    }

    TEST_F(ProjectTaskOrchestratorTest, FailedSessionMutationStillReleasesEveryLaneAfterDrain)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("aerial_triangulation")));
        const auto generic_context = orchestrator.context(QStringLiteral("aerial_triangulation"));
        const auto point_context =
            ProjectTaskOrchestratorPointModelTestPeer::installPointLane(&orchestrator, QStringLiteral("point:drain"));
        const auto model_context =
            ProjectTaskOrchestratorPointModelTestPeer::installModelLane(&orchestrator, QStringLiteral("model:drain"));
        const auto dem_context =
            ProjectTaskOrchestratorTerrainTestPeer::installDemLane(&orchestrator, QStringLiteral("dem:drain"));
        const auto ortho_context =
            ProjectTaskOrchestratorTerrainTestPeer::installOrthoLane(&orchestrator, QStringLiteral("ortho:drain"));
        const auto camera_context =
            ProjectTaskOrchestratorCameraTestPeer::installLane(&orchestrator, QStringLiteral("camera:drain"));
        const auto mask_context =
            ProjectTaskOrchestratorMaskTestPeer::installLane(&orchestrator, QStringLiteral("mask:drain"));
        ScopedPendingFuture worker;
        orchestrator.trackSessionFuture(worker.future());
        const auto before = session.context();
        int continuation_count = 0;

        ASSERT_TRUE(orchestrator.cancelAndDrainForSessionChange([&continuation_count]() { ++continuation_count; }));

        const auto expect_cancelled = [](const xjw::gui::project::ProjectTaskContext& context)
        {
            ASSERT_NE(context.cancelFlag, nullptr);
            EXPECT_TRUE(context.cancelFlag->load(std::memory_order_relaxed));
        };
        expect_cancelled(generic_context);
        expect_cancelled(point_context);
        expect_cancelled(model_context);
        expect_cancelled(dem_context);
        expect_cancelled(ortho_context);
        expect_cancelled(camera_context);
        expect_cancelled(mask_context);
        EXPECT_TRUE(session.isCurrent(before));
        EXPECT_EQ(continuation_count, 0);

        worker.finish();
        QTRY_COMPARE_WITH_TIMEOUT(continuation_count, 1, 5000);

        EXPECT_FALSE(orchestrator.isSessionDrainInProgress());
        EXPECT_TRUE(session.isCurrent(before));
        EXPECT_FALSE(orchestrator.hasActiveTask());
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::pointLaneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorPointModelTestPeer::modelLaneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::demLaneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorTerrainTestPeer::orthoLaneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorCameraTestPeer::laneActive(&orchestrator));
        EXPECT_FALSE(ProjectTaskOrchestratorMaskTestPeer::laneActive(&orchestrator));
        EXPECT_FALSE(orchestrator.hasRunningMaskTask());
        EXPECT_TRUE(orchestrator.beginTask(QStringLiteral("post-drain")));
        orchestrator.cancelActiveTask();
    }

    TEST_F(ProjectTaskOrchestratorTest, WaitForActiveTaskIsSafeAfterCancellation)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
        EXPECT_FALSE(orchestrator.hasActiveTask());
    }

    TEST_F(ProjectTaskOrchestratorTest, ManualSparseCompletionReleasesReservation)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        const auto context = orchestrator.context(QStringLiteral("triangulation"));

        ASSERT_TRUE(orchestrator.finishTask(context, true));
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("sparse_refine")));
        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST_F(ProjectTaskOrchestratorTest, ActiveSparseTaskRejectsBundleAdjustStart)
    {
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));

        EXPECT_FALSE(orchestrator.startBundleAdjustAsync(
            {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, QStringLiteral("output"), 1, true, {}));

        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, RunningBundleAdjustRejectsSparseStart)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("orchestrator.plascan")),
                                               QStringLiteral("orchestrator")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions options)
                {
                    while (!options.baOpt.cancelFlag->load(std::memory_order_relaxed))
                    {
                        QThread::msleep(1);
                    }
                    xjw::gui::project::BundleAdjustExecutionResult result;
                    result.serviceResult.success = false;
                    return result;
                });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(
            {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));
        EXPECT_FALSE(orchestrator.beginTask(QStringLiteral("triangulation")));

        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SynchronousBundleAdjustProgressCancelIsNotLost)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("orchestrator_cancel.plascan")),
                                               QStringLiteral("orchestrator_cancel")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions options)
                {
                    while (!options.baOpt.cancelFlag->load(std::memory_order_relaxed))
                    {
                        QThread::msleep(1);
                    }
                    return xjw::gui::project::BundleAdjustExecutionResult{};
                });
        QSignalSpy finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustFinished);
        bool cancellation_requested = false;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustProgressChanged,
                         &orchestrator,
                         [&]()
                         {
                             if (!cancellation_requested)
                             {
                                 cancellation_requested = true;
                                 orchestrator.cancelActiveTask();
                             }
                         });

        EXPECT_FALSE(orchestrator.startBundleAdjustAsync(
            {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));
        EXPECT_TRUE(cancellation_requested);
        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_FALSE(finished_spy.at(0).at(0).toBool());
        EXPECT_FALSE(orchestrator.hasActiveTask());

        orchestrator.waitForActiveTask();
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SparseProgressBridgeSuppressesTypedSignalAfterGenericCancel)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("sparse_progress_gate.plascan")),
                                               QStringLiteral("sparse_progress_gate")));
        QStringList images;
        for (const QString& name : {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")})
        {
            const QString path = temp_dir.filePath(name);
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            file.write("image");
            file.close();
            images.append(path);
        }
        ASSERT_TRUE(project_data.addImages(images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        QSignalSpy typed_progress(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged);
        int generic_progress_count = 0;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::progressChanged,
                         &orchestrator,
                         [&]()
                         {
                             ++generic_progress_count;
                             orchestrator.cancelActiveTask();
                         });

        orchestrator.startTriangulationAsync({});

        EXPECT_EQ(generic_progress_count, 1);
        EXPECT_EQ(typed_progress.count(), 0);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, BundleAdjustProgressBridgeSuppressesTypedSignalAfterGenericCancel)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_progress_gate.plascan")),
                                               QStringLiteral("ba_progress_gate")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return xjw::gui::project::BundleAdjustExecutionResult{}; });
        QSignalSpy typed_progress(&orchestrator,
                                  &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustProgressChanged);
        int generic_progress_count = 0;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::progressChanged,
                         &orchestrator,
                         [&]()
                         {
                             ++generic_progress_count;
                             orchestrator.cancelActiveTask();
                         });

        EXPECT_FALSE(orchestrator.startBundleAdjustAsync(
            {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));

        EXPECT_EQ(generic_progress_count, 1);
        EXPECT_EQ(typed_progress.count(), 0);
        orchestrator.waitForActiveTask();
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SparseProgressBridgeSuppressesTypedSignalAfterSessionChange)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("sparse_progress_session.plascan")),
                                               QStringLiteral("sparse_progress_session")));
        QStringList images;
        for (const QString& name : {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")})
        {
            const QString path = temp_dir.filePath(name);
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            file.write("image");
            file.close();
            images.append(path);
        }
        ASSERT_TRUE(project_data.addImages(images));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        QSignalSpy typed_progress(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged);
        int generic_progress_count = 0;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::progressChanged,
                         &orchestrator,
                         [&]()
                         {
                             ++generic_progress_count;
                             session.advanceGeneration();
                         });

        orchestrator.startTriangulationAsync({});

        EXPECT_EQ(generic_progress_count, 1);
        EXPECT_EQ(typed_progress.count(), 0);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, BundleAdjustProgressBridgeSuppressesTypedSignalAfterSessionChange)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_progress_session.plascan")),
                                               QStringLiteral("ba_progress_session")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return xjw::gui::project::BundleAdjustExecutionResult{}; });
        QSignalSpy typed_progress(&orchestrator,
                                  &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustProgressChanged);
        int generic_progress_count = 0;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::progressChanged,
                         &orchestrator,
                         [&]()
                         {
                             ++generic_progress_count;
                             session.advanceGeneration();
                         });

        EXPECT_FALSE(orchestrator.startBundleAdjustAsync(
            {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));

        EXPECT_EQ(generic_progress_count, 1);
        EXPECT_EQ(typed_progress.count(), 0);
        orchestrator.waitForActiveTask();
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, CancelledSparseWorkerBlocksBundleAdjustUntilFutureExits)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("sparse_then_ba.plascan")),
                                               QStringLiteral("sparse_then_ba")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        ScopedPendingFuture sparse_future;
        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        orchestrator.trackSparseFutureForTesting(sparse_future.future());
        orchestrator.cancelActiveTask();
        int runner_calls = 0;
        controller->setExecutionRunnerForTesting(
            [&](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            {
                ++runner_calls;
                xjw::gui::project::BundleAdjustExecutionResult result;
                result.serviceResult.success = true;
                return result;
            });

        EXPECT_FALSE(
            controller->startAsync({QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));
        QTRY_VERIFY(!controller->isRunning());
        EXPECT_EQ(runner_calls, 0);

        sparse_future.finish();
        EXPECT_TRUE(
            controller->startAsync({QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {}));
        QTRY_VERIFY(!controller->isRunning());
        EXPECT_EQ(runner_calls, 1);
        EXPECT_FALSE(orchestrator.hasActiveTask());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SparseSynchronousProgressCancelCannotReenterBundleAdjust)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("sparse_reentrant_ba.plascan")),
                                               QStringLiteral("sparse_reentrant_ba")));
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        int runner_calls = 0;
        controller->setExecutionRunnerForTesting(
            [&](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            {
                ++runner_calls;
                return xjw::gui::project::BundleAdjustExecutionResult{};
            });
        bool progress_emitted = false;
        bool ba_start_accepted = true;
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::taskStarted,
                         &orchestrator,
                         [&](const QString& task_id)
                         {
                             if (task_id == QStringLiteral("triangulation") && !progress_emitted)
                             {
                                 progress_emitted = true;
                                 emit orchestrator.sparseProgressChanged(QStringLiteral("同步 sparse 进度"), 10);
                             }
                         });
        QObject::connect(&orchestrator,
                         &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged,
                         &orchestrator,
                         [&]()
                         {
                             orchestrator.cancelActiveTask();
                             ba_start_accepted = controller->startAsync(
                                 {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")}, temp_dir.path(), 1, true, {});
                         });

        orchestrator.startTriangulationAsync(QJsonObject());

        EXPECT_TRUE(progress_emitted);
        EXPECT_FALSE(ba_start_accepted);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        QTRY_VERIFY(!controller->isRunning());
        EXPECT_EQ(runner_calls, 0);
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         PendingBundleAdjustPreviewRejectsSynchronousAndOrdinarySparseAdmissionUntilDiscard)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("pending_ba_sparse.plascan")),
                                               QStringLiteral("pending_ba_sparse")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);

        const QString old_sparse_path = temp_dir.filePath(QStringLiteral("pending_preview_existing_sparse.ply"));
        QFile old_sparse_file(old_sparse_path);
        ASSERT_TRUE(old_sparse_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(old_sparse_file.write("existing sparse artifact"), 24);
        old_sparse_file.close();
        const auto old_replace =
            session.replaceTiePointResult(session.context(), old_sparse_path, 9, images, temp_dir.path(), {});
        ASSERT_TRUE(old_replace.success) << qPrintable(old_replace.errorMessage);
        ASSERT_TRUE(QFileInfo::exists(old_sparse_path));

        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        controller->setExecutionRunnerForTesting(
            [execution_result](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            { return execution_result; });
        int writer_calls = 0;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString& sparse_cloud_path,
                int sparse_point_count,
                const QStringList& selected_images,
                const QString& output_dir,
                const QJsonObject& extra_record)
            {
                ++writer_calls;
                return session
                    .replaceTiePointResult(expected.session,
                                           sparse_cloud_path,
                                           sparse_point_count,
                                           selected_images,
                                           output_dir,
                                           extra_record)
                    .success;
            });

        bool synchronous_admission_attempted = false;
        bool synchronous_admission_accepted = false;
        QObject::connect(
            &orchestrator,
            &xjw::gui::project::ProjectTaskOrchestrator::taskFinished,
            &orchestrator,
            [&](const QString& task_id, bool)
            {
                if (task_id != QStringLiteral("bundle_adjust"))
                {
                    return;
                }
                synchronous_admission_attempted = true;
                synchronous_admission_accepted = orchestrator.beginTask(QStringLiteral("triangulation"));
                if (synchronous_admission_accepted)
                {
                    orchestrator.cancelActiveTask();
                }
            },
            Qt::DirectConnection);

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(synchronous_admission_attempted);
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        EXPECT_FALSE(synchronous_admission_accepted);

        const bool ordinary_admission_accepted = orchestrator.beginTask(QStringLiteral("triangulation"));
        EXPECT_FALSE(ordinary_admission_accepted);
        if (ordinary_admission_accepted)
        {
            orchestrator.cancelActiveTask();
        }

        const QJsonObject metadata_before_sparse_attempts = project_data.metadataIncludingResults();
        const QJsonArray old_records =
            metadata_before_sparse_attempts.value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(old_records.size(), 1);
        const QString old_record_path = old_records.at(0)
                                            .toObject()
                                            .value(QStringLiteral("files"))
                                            .toObject()
                                            .value(QStringLiteral("sparse_cloud_xyz"))
                                            .toString();
        ASSERT_FALSE(old_record_path.isEmpty());
        QSignalSpy task_started_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskStarted);
        QSignalSpy task_finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::taskFinished);
        QSignalSpy progress_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::progressChanged);
        QSignalSpy sparse_progress_spy(&orchestrator,
                                       &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged);
        QSignalSpy sparse_finished_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseFinished);
        QSignalSpy sparse_result_spy(&orchestrator,
                                     &xjw::gui::project::ProjectTaskOrchestrator::sparseTiePointResultReady);

        orchestrator.startTriangulationAsync({});
        orchestrator.startSparseCloudOutlierRemovalAsync({});
        orchestrator.startSparseCloudLocalOptimAsync({});
        orchestrator.startSparseCloudRefineAsync({});

        EXPECT_EQ(task_started_spy.count(), 0);
        EXPECT_EQ(task_finished_spy.count(), 0);
        EXPECT_EQ(progress_spy.count(), 0);
        EXPECT_EQ(sparse_progress_spy.count(), 0);
        EXPECT_EQ(sparse_finished_spy.count(), 0);
        EXPECT_EQ(sparse_result_spy.count(), 0);
        EXPECT_EQ(writer_calls, 0);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        EXPECT_TRUE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(project_data.metadataIncludingResults(), metadata_before_sparse_attempts);
        const QJsonArray retained_records =
            project_data.metadataIncludingResults().value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(retained_records.size(), 1);
        EXPECT_EQ(retained_records.at(0)
                      .toObject()
                      .value(QStringLiteral("files"))
                      .toObject()
                      .value(QStringLiteral("sparse_cloud_xyz"))
                      .toString(),
                  old_record_path);
        EXPECT_TRUE(QFileInfo::exists(old_sparse_path));
        QFile retained_sparse_file(old_sparse_path);
        ASSERT_TRUE(retained_sparse_file.open(QIODevice::ReadOnly));
        EXPECT_EQ(retained_sparse_file.readAll(), QByteArrayLiteral("existing sparse artifact"));
        retained_sparse_file.close();

        orchestrator.discardBundleAdjustPreview();
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         DirectBundleAdjustStartRejectsActiveSparseReservationWithoutSideEffects)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_accept_sparse_active.plascan")),
                                               QStringLiteral("ba_accept_sparse_active")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        messages.maximumReviewCount = 1;
        messages.reviewDecision = UiReviewDecision::Discard;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        int runner_calls = 0;
        controller->setExecutionRunnerForTesting(
            [&](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            {
                ++runner_calls;
                return execution_result;
            });
        int writer_calls = 0;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext&,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                ++writer_calls;
                return true;
            });

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        const QJsonObject core_before_start = project_data.coreFilesMeta();
        const QJsonArray bundle_adjust_before_start = project_data.getBundleAdjustResults();
        const bool accepted = controller->startAsync(images, temp_dir.path(), 1, false, {});
        EXPECT_FALSE(accepted);
        if (accepted)
        {
            QTRY_VERIFY(!controller->isRunning());
            QTRY_VERIFY(!controller->hasPendingPreview());
        }

        EXPECT_FALSE(controller->isRunning());
        EXPECT_FALSE(controller->hasPendingPreview());
        EXPECT_EQ(runner_calls, 0);
        EXPECT_EQ(messages.reviewCount, 0);
        EXPECT_FALSE(messages.reviewLimitExceeded);
        EXPECT_EQ(writer_calls, 0);
        EXPECT_EQ(project_data.coreFilesMeta(), core_before_start);
        EXPECT_EQ(project_data.getBundleAdjustResults(), bundle_adjust_before_start);

        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         BundleAdjustStartLeaseRejectsSynchronousSparseAdmissionBeforeRunningState)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_start_lease.plascan")),
                                               QStringLiteral("ba_start_lease")));
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);

        bool sparse_admission_attempted = false;
        bool sparse_admission_accepted = true;
        messages.warningAction = [&]()
        {
            sparse_admission_attempted = true;
            sparse_admission_accepted = orchestrator.beginTask(QStringLiteral("triangulation"));
            if (sparse_admission_accepted)
            {
                orchestrator.cancelActiveTask();
            }
        };

        EXPECT_FALSE(controller->startAsync({QStringLiteral("only-one-image.jpg")}, temp_dir.path(), 1, true, {}));

        EXPECT_TRUE(sparse_admission_attempted);
        EXPECT_FALSE(sparse_admission_accepted);
        EXPECT_EQ(messages.warningCount, 1);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        EXPECT_FALSE(controller->isRunning());
        EXPECT_FALSE(controller->hasPendingPreview());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         DirectBundleAdjustPreviewRejectsSparseFutureBeforeRealWriterAndPreservesOldArtifact)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_accept_sparse_future.plascan")),
                                               QStringLiteral("ba_accept_sparse_future")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        controller->setExecutionRunnerForTesting(
            [execution_result](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            { return execution_result; });

        const QString old_sparse_path = temp_dir.filePath(QStringLiteral("existing_sparse.ply"));
        QFile old_sparse_file(old_sparse_path);
        ASSERT_TRUE(old_sparse_file.open(QIODevice::WriteOnly));
        ASSERT_EQ(old_sparse_file.write("old sparse artifact"), 19);
        old_sparse_file.close();
        const auto old_replace =
            session.replaceTiePointResult(session.context(), old_sparse_path, 7, images, temp_dir.path(), {});
        ASSERT_TRUE(old_replace.success) << qPrintable(old_replace.errorMessage);
        ASSERT_TRUE(QFileInfo::exists(old_sparse_path));

        int writer_calls = 0;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString& sparse_cloud_path,
                int sparse_point_count,
                const QStringList& selected_images,
                const QString& output_dir,
                const QJsonObject& extra_record)
            {
                ++writer_calls;
                return session
                    .replaceTiePointResult(expected.session,
                                           sparse_cloud_path,
                                           sparse_point_count,
                                           selected_images,
                                           output_dir,
                                           extra_record)
                    .success;
            });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        ScopedPendingFuture pending_future;
        orchestrator.trackSparseFutureForTesting(pending_future.future());
        const QJsonObject metadata_before_accept = project_data.metadataIncludingResults();
        const QJsonArray old_records =
            metadata_before_accept.value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(old_records.size(), 1);
        const QString old_record_path = old_records.at(0)
                                            .toObject()
                                            .value(QStringLiteral("files"))
                                            .toObject()
                                            .value(QStringLiteral("sparse_cloud_xyz"))
                                            .toString();
        ASSERT_FALSE(old_record_path.isEmpty());

        QString error_message;
        EXPECT_FALSE(controller->acceptPreview(&error_message));
        EXPECT_EQ(error_message, QStringLiteral("稀疏重建任务仍占用处理通道，暂不能接受平差预览"));
        EXPECT_TRUE(controller->hasPendingPreview());
        EXPECT_EQ(writer_calls, 0);
        EXPECT_EQ(project_data.metadataIncludingResults(), metadata_before_accept);
        const QJsonArray retained_records =
            project_data.metadataIncludingResults().value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(retained_records.size(), 1);
        EXPECT_EQ(retained_records.at(0)
                      .toObject()
                      .value(QStringLiteral("files"))
                      .toObject()
                      .value(QStringLiteral("sparse_cloud_xyz"))
                      .toString(),
                  old_record_path);
        EXPECT_TRUE(QFileInfo::exists(old_sparse_path));

        pending_future.finish();
        error_message.clear();
        EXPECT_TRUE(controller->acceptPreview(&error_message)) << qPrintable(error_message);
        EXPECT_EQ(writer_calls, 1);
        EXPECT_FALSE(controller->hasPendingPreview());
        EXPECT_FALSE(QFileInfo::exists(old_sparse_path));
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         AutomaticBundleAdjustPreviewRetriesAfterSparseFutureReleaseAndWritesOnce)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_automatic_sparse_gate.plascan")),
                                               QStringLiteral("ba_automatic_sparse_gate")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);

        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        messages.maximumReviewCount = 2;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        auto* controller = ownedBaController(&orchestrator);
        ASSERT_NE(controller, nullptr);
        controller->setExecutionRunnerForTesting(
            [execution_result](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            { return execution_result; });

        ScopedPendingFuture pending_future;
        bool sparse_future_installed = false;
        QObject::connect(
            &orchestrator,
            &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustPreviewReady,
            &orchestrator,
            [&](const QJsonObject& preview)
            {
                if (!preview.isEmpty() && !sparse_future_installed)
                {
                    sparse_future_installed = true;
                    orchestrator.trackSparseFutureForTesting(pending_future.future());
                }
            },
            Qt::DirectConnection);

        int writer_calls = 0;
        QVector<int> writer_calls_at_review;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext&,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                ++writer_calls;
                return true;
            });
        messages.reviewAction = [&]() { writer_calls_at_review.append(writer_calls); };
        messages.warningAction = [&]()
        {
            EXPECT_EQ(writer_calls, 0);
            pending_future.finish();
        };

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        QTRY_VERIFY(!orchestrator.hasPendingBundleAdjustPreview());

        EXPECT_TRUE(sparse_future_installed);
        EXPECT_EQ(messages.reviewCount, 2);
        EXPECT_EQ(messages.warningCount, 1);
        EXPECT_EQ(messages.informationCount, 1);
        EXPECT_FALSE(messages.reviewLimitExceeded);
        EXPECT_EQ(writer_calls_at_review, QVector<int>({0, 0}));
        EXPECT_EQ(writer_calls, 1);
        EXPECT_FALSE(orchestrator.hasActiveTask());
        pending_future.finish();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, DelayedBundleAdjustPreviewUsesCapturedWriterContext)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("delayed_ba_preview.plascan")),
                                               QStringLiteral("delayed_ba_preview")));

        QStringList images;
        for (const QString& name : {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")})
        {
            const QString path = temp_dir.filePath(name);
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            file.write("image");
            file.close();
            images.append(path);
        }
        ASSERT_TRUE(project_data.addImages(images));
        const QJsonObject camera = bundleAdjustCameraMetadata();
        int updated_count = 0;
        QString camera_error;
        ASSERT_TRUE(project_data.setCameraInstances(
            {{images.at(0), camera}, {images.at(1), camera}}, &updated_count, &camera_error))
            << qPrintable(camera_error);

        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        execution_result.serviceResult.success = true;
        execution_result.serviceResult.resultJson =
            QJsonObject{{QStringLiteral("output_dir"), temp_dir.path()},
                        {QStringLiteral("track_count"), 1},
                        {QStringLiteral("optimized_count"), 1},
                        {QStringLiteral("mean_rms_before"), 1.0},
                        {QStringLiteral("mean_rms_after"), 0.5},
                        {QStringLiteral("points"),
                         QJsonArray{QJsonObject{{QStringLiteral("valid"), true},
                                                {QStringLiteral("converged"), true},
                                                {QStringLiteral("rms_after"), 0.5},
                                                {QStringLiteral("track_len"), 2},
                                                {QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}}}}}};
        const QJsonObject core = project_data.coreFilesMeta();
        const QString world_frame = core.value(QStringLiteral("camera_definitions"))
                                        .toArray()
                                        .at(0)
                                        .toObject()
                                        .value(QStringLiteral("frame"))
                                        .toString();
        for (const QJsonValue& image_value : core.value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = image_value.toObject();
            const QString image_id = image.value(QStringLiteral("image_uuid")).toString();
            const QJsonObject instance = xjw::camera_project::CameraProjectRecords::instanceForImage(core, image_id);
            QJsonObject update = xjw::camera_project::CameraProjectRecords::modelParametersForImage(core, image);
            update.insert(QStringLiteral("world_frame"), world_frame);
            execution_result.serviceResult.cameraInstanceUpdates.push_back(
                {xjw::camera_core::ImageId(image_id.toStdString()),
                 xjw::camera_core::CameraInstanceId(instance.value(QStringLiteral("id")).toString().toStdString()),
                 xjw::coordinate_system::CoordinateFrameId(world_frame.toStdString()),
                 update});
        }

        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        xjw::gui::project::ProjectTaskContext writer_context;
        bool writer_called = false;
        int post_external_commit_calls = 0;
        orchestrator.setBundleAdjustPostExternalCommitObserver([&](const xjw::gui::project::ProjectTaskContext&)
                                                               { ++post_external_commit_calls; });
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                writer_context = expected;
                writer_called = true;
                return expected.cancelFlag && !expected.cancelFlag->load(std::memory_order_relaxed) &&
                       session.isCurrent(expected.session);
            });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        QTRY_VERIFY(!orchestrator.hasActiveTask());

        QString accept_error;
        EXPECT_TRUE(orchestrator.acceptBundleAdjustPreview(&accept_error)) << qPrintable(accept_error);
        EXPECT_TRUE(writer_called);
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_FALSE(writer_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_TRUE(session.isCurrent(writer_context.session));
        EXPECT_EQ(post_external_commit_calls, 0);
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_TRUE(orchestrator.beginTask(QStringLiteral("triangulation")));
        orchestrator.cancelActiveTask();
        orchestrator.waitForActiveTask();
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         BundleAdjustPostExternalObserverSessionAdvanceKeepsCommitButRejectsTail)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_observer_reentry.plascan")),
                                               QStringLiteral("ba_observer_reentry")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        const QJsonObject core_before_accept = project_data.coreFilesMeta();

        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        xjw::gui::project::ProjectTaskContext writer_context;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString& sparse_cloud_path,
                int sparse_point_count,
                const QStringList& selected_images,
                const QString& output_dir,
                const QJsonObject& extra_record)
            {
                writer_context = expected;
                return session
                    .replaceTiePointResult(expected.session,
                                           sparse_cloud_path,
                                           sparse_point_count,
                                           selected_images,
                                           output_dir,
                                           extra_record)
                    .success;
            });
        int observer_calls = 0;
        orchestrator.setBundleAdjustPostExternalCommitObserver(
            [&](const xjw::gui::project::ProjectTaskContext& expected)
            {
                ++observer_calls;
                EXPECT_TRUE(session.isCurrent(expected.session));
                QJsonObject metadata = project_data.metadataIncludingResults();
                metadata[QStringLiteral("report_results")] =
                    QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("post_external_test")}}};
                project_data.updateMetadata(metadata, true);
                session.advanceGeneration();
            });

        bool acceptance_invoked = false;
        bool acceptance_result = true;
        QString accept_error;
        messages.reviewAction = [&]()
        {
            acceptance_invoked = true;
            acceptance_result = orchestrator.acceptBundleAdjustPreview(&accept_error);
        };

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(acceptance_invoked);
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        EXPECT_FALSE(acceptance_result);
        EXPECT_FALSE(accept_error.isEmpty());
        EXPECT_EQ(observer_calls, 1);
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(messages.informationCount, 0);
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_TRUE(writer_context.cancelFlag->load(std::memory_order_relaxed));

        const QJsonObject metadata = project_data.metadataIncludingResults();
        EXPECT_EQ(metadata.value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
        EXPECT_EQ(metadata.value(QStringLiteral("aerial_triangulation_results")).toArray().size(), 1);
        EXPECT_EQ(metadata.value(QStringLiteral("report_results")).toArray().size(), 1);
        EXPECT_NE(project_data.coreFilesMeta(), core_before_accept);
    }

    TEST(ProjectTaskOrchestratorIntegrationTest,
         RecursiveAcceptDuringPostExternalReportIsRejectedWithoutDuplicateCommit)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_recursive_accept.plascan")),
                                               QStringLiteral("ba_recursive_accept")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);

        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });

        int writer_calls = 0;
        int observer_calls = 0;
        xjw::gui::project::ProjectTaskContext preview_context;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString& sparse_cloud_path,
                int sparse_point_count,
                const QStringList& selected_images,
                const QString& output_dir,
                const QJsonObject& extra_record)
            {
                ++writer_calls;
                preview_context = expected;
                return session
                    .replaceTiePointResult(expected.session,
                                           sparse_cloud_path,
                                           sparse_point_count,
                                           selected_images,
                                           output_dir,
                                           extra_record)
                    .success;
            });
        orchestrator.setBundleAdjustPostExternalCommitObserver(
            [&](const xjw::gui::project::ProjectTaskContext& expected)
            {
                ++observer_calls;
                EXPECT_TRUE(session.isCurrent(expected.session));
                QJsonObject metadata = project_data.metadataIncludingResults();
                metadata[QStringLiteral("report_results")] =
                    QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("recursive_accept_report")}}};
                project_data.updateMetadata(metadata, true);
            });

        QVector<int> metadata_ba_counts;
        int recursive_accept_calls = 0;
        bool recursive_accept_result = true;
        QString recursive_accept_error;
        QObject::connect(
            &project_data,
            &ProjectData::metadataChanged,
            &orchestrator,
            [&](const QJsonObject& metadata)
            {
                metadata_ba_counts.append(metadata.value(QStringLiteral("bundle_adjust_results")).toArray().size());
                if (recursive_accept_calls == 0 &&
                    !metadata.value(QStringLiteral("report_results")).toArray().isEmpty())
                {
                    ++recursive_accept_calls;
                    recursive_accept_result = orchestrator.acceptBundleAdjustPreview(&recursive_accept_error);
                }
            },
            Qt::DirectConnection);

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());

        QString outer_error;
        EXPECT_TRUE(orchestrator.acceptBundleAdjustPreview(&outer_error)) << qPrintable(outer_error);
        EXPECT_EQ(recursive_accept_calls, 1);
        EXPECT_FALSE(recursive_accept_result);
        EXPECT_FALSE(recursive_accept_error.isEmpty());
        EXPECT_EQ(writer_calls, 1);
        EXPECT_EQ(observer_calls, 1);
        EXPECT_EQ(metadata_ba_counts, QVector<int>({1, 1}));

        const QJsonObject metadata = project_data.metadataIncludingResults();
        EXPECT_EQ(metadata.value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
        EXPECT_EQ(metadata.value(QStringLiteral("aerial_triangulation_results")).toArray().size(), 1);
        EXPECT_EQ(metadata.value(QStringLiteral("report_results")).toArray().size(), 1);
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        ASSERT_NE(preview_context.cancelFlag, nullptr);
        EXPECT_FALSE(preview_context.cancelFlag->load(std::memory_order_relaxed));
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, BundleAdjustWriterFailureCanRetryWithoutDuplicateProjectCommit)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_writer_retry.plascan")),
                                               QStringLiteral("ba_writer_retry")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);

        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        int writer_calls = 0;
        xjw::gui::project::ProjectTaskContext writer_context;
        QString first_sparse_cloud_path;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString& sparse_cloud_path,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                ++writer_calls;
                writer_context = expected;
                if (first_sparse_cloud_path.isEmpty())
                {
                    first_sparse_cloud_path = sparse_cloud_path;
                }
                else
                {
                    EXPECT_EQ(sparse_cloud_path, first_sparse_cloud_path);
                }
                return writer_calls > 1;
            });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        int metadata_changed_count = 0;
        QObject::connect(&project_data,
                         &ProjectData::metadataChanged,
                         [&metadata_changed_count](const QJsonObject&) { ++metadata_changed_count; });
        const QJsonObject before_accept = project_data.metadataIncludingResults();

        QString first_error;
        EXPECT_FALSE(orchestrator.acceptBundleAdjustPreview(&first_error));
        EXPECT_FALSE(first_error.isEmpty());
        EXPECT_TRUE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(writer_calls, 1);
        EXPECT_EQ(metadata_changed_count, 0);
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);

        QString retry_error;
        EXPECT_TRUE(orchestrator.acceptBundleAdjustPreview(&retry_error)) << qPrintable(retry_error);
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(writer_calls, 2);
        EXPECT_EQ(metadata_changed_count, 1);
        EXPECT_EQ(
            project_data.metadataIncludingResults().value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_FALSE(writer_context.cancelFlag->load(std::memory_order_relaxed));
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, BundleAdjustWriterFailureThenDiscardRestoresOriginalMetadata)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_writer_discard.plascan")),
                                               QStringLiteral("ba_writer_discard")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        orchestrator.setTiePointResultWriter([](const xjw::gui::project::ProjectTaskContext&,
                                                const QString&,
                                                int,
                                                const QStringList&,
                                                const QString&,
                                                const QJsonObject&) { return false; });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        const QJsonObject before_accept = project_data.metadataIncludingResults();
        QString error_message;
        ASSERT_FALSE(orchestrator.acceptBundleAdjustPreview(&error_message));
        EXPECT_TRUE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);

        orchestrator.discardBundleAdjustPreview();

        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);
        EXPECT_TRUE(
            project_data.metadataIncludingResults().value(QStringLiteral("bundle_adjust_results")).toArray().isEmpty());
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SynchronousDiscardFromWriterCancelsContextAndRollsBackStage)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_writer_reentrant_discard.plascan")),
                                               QStringLiteral("ba_writer_reentrant_discard")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        xjw::gui::project::ProjectTaskContext writer_context;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                writer_context = expected;
                orchestrator.discardBundleAdjustPreview();
                return true;
            });
        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        const QJsonObject before_accept = project_data.metadataIncludingResults();

        QString error_message;
        EXPECT_FALSE(orchestrator.acceptBundleAdjustPreview(&error_message));
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_TRUE(writer_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, SynchronousCancelFromWriterCancelsContextAndRollsBackStage)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_writer_reentrant_cancel.plascan")),
                                               QStringLiteral("ba_writer_reentrant_cancel")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        xjw::gui::project::ProjectTaskContext writer_context;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                writer_context = expected;
                orchestrator.cancelActiveTask();
                return true;
            });
        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        const QJsonObject before_accept = project_data.metadataIncludingResults();

        QString error_message;
        EXPECT_FALSE(orchestrator.acceptBundleAdjustPreview(&error_message));
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_TRUE(writer_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, AutomaticAcceptSuppressesSuccessMessageAfterWriterDiscard)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_writer_no_success_tail.plascan")),
                                               QStringLiteral("ba_writer_no_success_tail")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        xjw::gui::project::ProjectSession session(&project_data);
        AcceptingReviewMessages messages;
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, &messages);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        xjw::gui::project::ProjectTaskContext writer_context;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext& expected,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                writer_context = expected;
                orchestrator.discardBundleAdjustPreview();
                return true;
            });
        const QJsonObject before_accept = project_data.metadataIncludingResults();

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        QTRY_VERIFY(!orchestrator.hasPendingBundleAdjustPreview());

        EXPECT_EQ(messages.reviewCount, 1);
        EXPECT_EQ(messages.informationCount, 0);
        EXPECT_EQ(messages.warningCount, 0);
        ASSERT_NE(writer_context.cancelFlag, nullptr);
        EXPECT_TRUE(writer_context.cancelFlag->load(std::memory_order_relaxed));
        EXPECT_EQ(project_data.metadataIncludingResults(), before_accept);
    }

    TEST(ProjectTaskOrchestratorIntegrationTest, DiscardCancelAndSessionChangeInvalidateSavedBundleAdjustWriterContext)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("ba_context_invalidation.plascan")),
                                               QStringLiteral("ba_context_invalidation")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);

        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        ownedBaController(&orchestrator)
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        int writer_calls = 0;
        orchestrator.setTiePointResultWriter(
            [&](const xjw::gui::project::ProjectTaskContext&,
                const QString&,
                int,
                const QStringList&,
                const QString&,
                const QJsonObject&)
            {
                ++writer_calls;
                return true;
            });

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        orchestrator.discardBundleAdjustPreview();
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        orchestrator.cancelActiveTask();
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());

        ASSERT_TRUE(orchestrator.startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(orchestrator.hasPendingBundleAdjustPreview());
        QTRY_VERIFY(!orchestrator.hasRunningBundleAdjustTask());
        session.advanceGeneration();
        EXPECT_FALSE(orchestrator.hasPendingBundleAdjustPreview());
        EXPECT_EQ(writer_calls, 0);
    }

    TEST_F(ProjectTaskOrchestratorTest, OwnsBundleAdjustControllerOnSharedSessionBoundary)
    {
        ASSERT_NE(ownedBaController(&orchestrator), nullptr);
        EXPECT_EQ(ownedBaController(&orchestrator)->parent(), &orchestrator);
        EXPECT_EQ(orchestrator.session(), &session);
    }

    TEST_F(ProjectTaskOrchestratorTest, TiePointWriterRequiresExpectedTaskContext)
    {
        orchestrator.setTiePointResultWriter([this](const xjw::gui::project::ProjectTaskContext& expected,
                                                    const QString&,
                                                    int,
                                                    const QStringList&,
                                                    const QString&,
                                                    const QJsonObject&)
                                             { return expected.cancelFlag && session.isCurrent(expected.session); });
    }

    TEST(ProjectTaskOrchestratorExternalAtTest, CancelEmitsOneSparseFailureAndDropsLateCompletion)
    {
        qtApplication();
        ProjectData projectData;
        xjw::gui::project::ProjectServiceContainer container(&projectData, nullptr);

        ASSERT_TRUE(container.tasks().beginTask(QStringLiteral("aerial_triangulation")));
        const auto context = container.tasks().context(QStringLiteral("aerial_triangulation"));
        QSignalSpy finishedSpy(&container.tasks(), &xjw::gui::project::ProjectTaskOrchestrator::sparseFinished);

        container.tasks().cancelActiveTask();

        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_FALSE(finishedSpy.at(0).at(0).toBool());
        EXPECT_FALSE(container.tasks().finishTask(context, false));
        EXPECT_EQ(finishedSpy.count(), 1);
    }

    TEST(ProjectManagerTaskBridgeTest, BundleAdjustPublishesExternalCommitBeforePostCasQualityReport)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("pm_post_cas_report.plascan")),
                                               QStringLiteral("pm_post_cas_report")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        const QJsonObject core_before_accept = project_data.coreFilesMeta();

        QWidget window;
        xjw::gui::project::ProjectServiceContainer container(&project_data, &window);
        ownedBaController(&container.tasks())
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });

        QVector<QJsonObject> transaction_events;
        bool capture_transaction_events = true;
        QObject::connect(&project_data,
                         &ProjectData::metadataChanged,
                         &container,
                         [&transaction_events, &capture_transaction_events](const QJsonObject& metadata)
                         {
                             if (capture_transaction_events &&
                                 (!metadata.value(QStringLiteral("bundle_adjust_results")).toArray().isEmpty() ||
                                  !metadata.value(QStringLiteral("aerial_triangulation_results")).toArray().isEmpty() ||
                                  !metadata.value(QStringLiteral("report_results")).toArray().isEmpty()))
                             {
                                 transaction_events.push_back(metadata);
                             }
                         });
        int preview_accept_count = 0;
        int success_information_count = 0;
        int unexpected_dialog_count = 0;
        QTimer dialog_driver;
        QObject::connect(&dialog_driver,
                         &QTimer::timeout,
                         [&]()
                         {
                             auto* message_box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
                             if (!message_box)
                             {
                                 return;
                             }
                             if (message_box->objectName() == QStringLiteral("bundleAdjustPreviewMessageBox"))
                             {
                                 if (QAbstractButton* accept = message_box->findChild<QAbstractButton*>(
                                         QStringLiteral("keepBundleAdjustPreviewButton")))
                                 {
                                     ++preview_accept_count;
                                     accept->click();
                                 }
                                 return;
                             }
                             if (message_box->text().contains(QStringLiteral("已保留本次平差结果")))
                             {
                                 ++success_information_count;
                                 capture_transaction_events = false;
                             }
                             else
                             {
                                 ++unexpected_dialog_count;
                             }
                             if (QAbstractButton* ok = message_box->button(QMessageBox::Ok))
                             {
                                 ok->click();
                             }
                             else
                             {
                                 message_box->reject();
                             }
                         });
        dialog_driver.start(5);

        ASSERT_TRUE(container.tasks().startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_COMPARE(preview_accept_count, 1);
        QTRY_COMPARE(success_information_count, 1);
        QTRY_VERIFY(!container.tasks().hasRunningBundleAdjustTask());
        QTRY_VERIFY(!container.tasks().hasPendingBundleAdjustPreview());
        dialog_driver.stop();

        ASSERT_GE(transaction_events.size(), 2);
        const QJsonObject first_external = transaction_events.front();
        EXPECT_EQ(first_external.value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
        EXPECT_EQ(first_external.value(QStringLiteral("aerial_triangulation_results")).toArray().size(), 1);
        EXPECT_TRUE(first_external.value(QStringLiteral("report_results")).toArray().isEmpty());
        EXPECT_NE(project_data.coreFilesMeta(), core_before_accept);

        int events_with_report = 0;
        for (const QJsonObject& event : transaction_events)
        {
            EXPECT_EQ(event.value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
            EXPECT_EQ(event.value(QStringLiteral("aerial_triangulation_results")).toArray().size(), 1);
            if (!event.value(QStringLiteral("report_results")).toArray().isEmpty())
            {
                ++events_with_report;
            }
        }
        EXPECT_EQ(events_with_report, 1);
        EXPECT_EQ(project_data.metadataIncludingResults().value(QStringLiteral("report_results")).toArray().size(), 1);
        EXPECT_EQ(unexpected_dialog_count, 0);
    }

    TEST(ProjectManagerTaskBridgeTest, WriterReentryRejectsStaleCompletionWithoutRefreshingQualityReport)
    {
        qtApplication();
        QTemporaryDir temp_dir;
        ASSERT_TRUE(temp_dir.isValid());
        ProjectData project_data;
        ASSERT_TRUE(project_data.createProject(temp_dir.filePath(QStringLiteral("pm_writer_reentry.plascan")),
                                               QStringLiteral("pm_writer_reentry")));
        QStringList images;
        xjw::gui::project::BundleAdjustExecutionResult execution_result;
        QString fixture_error;
        ASSERT_TRUE(prepareBundleAdjustPreviewFixture(
            &project_data, temp_dir.path(), &images, &execution_result, &fixture_error))
            << qPrintable(fixture_error);
        const QJsonObject core_before_accept = project_data.coreFilesMeta();

        QWidget window;
        xjw::gui::project::ProjectServiceContainer container(&project_data, &window);
        ownedBaController(&container.tasks())
            ->setExecutionRunnerForTesting(
                [execution_result](
                    const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
                { return execution_result; });
        bool session_advanced_during_writer = false;
        QObject::connect(&project_data,
                         &ProjectData::metadataChanged,
                         &container,
                         [&](const QJsonObject& metadata)
                         {
                             if (!session_advanced_during_writer &&
                                 !metadata.value(QStringLiteral("aerial_triangulation_results")).toArray().isEmpty())
                             {
                                 session_advanced_during_writer = true;
                                 container.session().advanceGeneration();
                             }
                         });
        QTimer preview_acceptor;
        QObject::connect(
            &preview_acceptor,
            &QTimer::timeout,
            []()
            {
                auto* message_box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
                if (!message_box || message_box->objectName() != QStringLiteral("bundleAdjustPreviewMessageBox"))
                {
                    return;
                }
                if (QAbstractButton* accept =
                        message_box->findChild<QAbstractButton*>(QStringLiteral("keepBundleAdjustPreviewButton")))
                {
                    accept->click();
                }
            });
        preview_acceptor.start(5);

        ASSERT_TRUE(container.tasks().startBundleAdjustAsync(images, temp_dir.path(), 1, false, {}));
        QTRY_VERIFY(session_advanced_during_writer);
        QTRY_VERIFY(!container.tasks().hasRunningBundleAdjustTask());
        preview_acceptor.stop();

        EXPECT_TRUE(session_advanced_during_writer);
        const QJsonObject metadata = project_data.metadataIncludingResults();
        EXPECT_EQ(metadata.value(QStringLiteral("aerial_triangulation_results")).toArray().size(), 1);
        EXPECT_EQ(metadata.value(QStringLiteral("bundle_adjust_results")).toArray().size(), 1);
        EXPECT_NE(project_data.coreFilesMeta(), core_before_accept);
        EXPECT_TRUE(metadata.value(QStringLiteral("report_results")).toArray().isEmpty());
        EXPECT_FALSE(container.tasks().hasPendingBundleAdjustPreview());
    }

    TEST(ProjectTaskOrchestratorExternalAtTest, ContextChecksProgressDevicePairAndSessionInvalidation)
    {
        qtApplication();
        ProjectData project_data;
        xjw::gui::project::ProjectSession session(&project_data);
        xjw::gui::project::ProjectTaskOrchestrator orchestrator(&session, nullptr);
        QSignalSpy progress_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged);
        QSignalSpy device_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseComputeDeviceChanged);
        QSignalSpy pair_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseMatchPairReady);
        QSignalSpy tie_point_spy(&orchestrator, &xjw::gui::project::ProjectTaskOrchestrator::sparseTiePointResultReady);

        ASSERT_TRUE(orchestrator.beginTask(QStringLiteral("aerial_triangulation")));
        const auto context = orchestrator.context(QStringLiteral("aerial_triangulation"));
        ASSERT_TRUE(orchestrator.isTaskActive(context));
        EXPECT_FALSE(orchestrator.startBundleAdjustAsync({}, {}, 1, false, {}));
        EXPECT_TRUE(orchestrator.reportSparseProgress(context, QStringLiteral("stage"), 25));
        EXPECT_TRUE(orchestrator.reportSparseComputeDevice(context, QStringLiteral("CPU")));
        EXPECT_TRUE(orchestrator.reportSparseMatchPair(
            context, QStringLiteral("a"), QStringLiteral("b"), QStringLiteral("pair.pimatch"), 7));
        EXPECT_TRUE(orchestrator.reportSparseTiePointResult(
            context, QStringLiteral("sparse.ply"), QStringLiteral("sparse.json")));
        EXPECT_EQ(progress_spy.count(), 1);
        EXPECT_EQ(device_spy.count(), 1);
        EXPECT_EQ(pair_spy.count(), 1);
        EXPECT_EQ(tie_point_spy.count(), 1);

        session.advanceGeneration();
        EXPECT_FALSE(orchestrator.isTaskActive(context));
        EXPECT_FALSE(orchestrator.reportSparseProgress(context, QStringLiteral("late"), 90));
        EXPECT_FALSE(orchestrator.reportSparseComputeDevice(context, QStringLiteral("late")));
        EXPECT_FALSE(orchestrator.reportSparseMatchPair(context, {}, {}, {}, 0));
        EXPECT_FALSE(orchestrator.reportSparseTiePointResult(context, {}, {}));
        EXPECT_EQ(progress_spy.count(), 1);
        EXPECT_EQ(device_spy.count(), 1);
        EXPECT_EQ(pair_spy.count(), 1);
        EXPECT_EQ(tie_point_spy.count(), 1);
    }

    TEST(ProjectServiceContainerTest, OwnsOneSharedSessionAndRoutesServices)
    {
        qtApplication();
        ProjectData projectData;
        xjw::gui::project::ProjectServiceContainer container(&projectData, nullptr);

        EXPECT_EQ(container.session().data(), &projectData);
        EXPECT_EQ(&container.session(), container.tasks().session());
        EXPECT_EQ(&container.session(), container.resources().session());
        EXPECT_EQ(&container.session(), container.lifecycle().session());
        EXPECT_EQ(&container.session(), container.cleanup().session());
    }

} // namespace
