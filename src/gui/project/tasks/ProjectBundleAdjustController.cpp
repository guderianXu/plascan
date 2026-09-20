#include "ProjectBundleAdjustController.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "project/ProjectConfigManager.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/support/ProjectBundleAdjustWorkflow.h"

#include <QDateTime>
#include <QDir>
#include <QJsonArray>
#include <QPointer>
#include <QScopeGuard>

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::gui::project
{

    namespace
    {

        constexpr auto kTaskId = "bundle_adjust";

        QMap<QString, QJsonObject>
        cameraUpdatesForPresentation(const QJsonObject& projectMeta,
                                     const xjw::camera_project::CameraInstanceUpdates& updates)
        {
            QMap<QString, QJsonObject> result;
            const QJsonArray images = xjw::common::project::projectImageEntries(projectMeta);
            for (const auto& update : updates)
            {
                const QString image_id = QString::fromStdString(update.imageId.value()).trimmed();
                for (const QJsonValue& value : images)
                {
                    const QJsonObject image = value.toObject();
                    if (image.value(QStringLiteral("image_uuid")).toString().trimmed() != image_id)
                    {
                        continue;
                    }
                    const QString path =
                        xjw::common::project::normalizePath(image.value(QStringLiteral("path")).toString());
                    if (!path.isEmpty())
                    {
                        result.insert(path, update.modelMetadata);
                    }
                    break;
                }
            }
            return result;
        }

        QString buildInputError(xjw::core::project::BaInputBuildStatus status, const QString& detail)
        {
            if (!detail.trimmed().isEmpty())
            {
                return detail;
            }
            if (status == xjw::core::project::BaInputBuildStatus::NotEnoughCameras)
            {
                return QStringLiteral("所选影像中可用相机参数不足（至少需要两台相机）");
            }
            if (status == xjw::core::project::BaInputBuildStatus::InvalidInput)
            {
                return QStringLiteral("控制点或标记的影像身份无效");
            }
            return QStringLiteral("未找到可用于光束法平差的匹配点（请检查选中影像是否已有匹配结果）");
        }

    } // namespace

    ProjectBundleAdjustController::ProjectBundleAdjustController(ProjectSession* session,
                                                                 ProjectUiMessageAdapter* messages,
                                                                 QObject* parent)
        : ProjectBundleAdjustController(session, messages, {}, {}, parent)
    {
    }

    ProjectBundleAdjustController::ProjectBundleAdjustController(ProjectSession* session,
                                                                 ProjectUiMessageAdapter* messages,
                                                                 StartAdmissionProvider startAdmissionProvider,
                                                                 PreviewAdmissionPredicate previewAdmissionPredicate,
                                                                 QObject* parent)
        : QObject(parent), _session(session), _messages(messages), _executionRunner(runBundleAdjustExecution),
          _startAdmissionProvider(std::move(startAdmissionProvider)),
          _previewAdmissionPredicate(std::move(previewAdmissionPredicate))
    {
        if (_session)
        {
            connect(_session,
                    &ProjectSession::sessionChanged,
                    this,
                    [this](const ProjectSessionContext&) { handleSessionChanged(); });
        }
    }

    ProjectBundleAdjustController::~ProjectBundleAdjustController()
    {
        _destroying = true;
        if (_taskContext.cancelFlag)
        {
            _taskContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        discardPreview();
        waitForFinished();
    }

    bool ProjectBundleAdjustController::startAsync(
        const QStringList& images, const QString& outputDir, int threads, bool dryRun, const QJsonObject& extraSettings)
    {
        AdmissionRelease admission_release;
        if (_startAdmissionProvider)
        {
            admission_release = _startAdmissionProvider();
            if (!admission_release)
            {
                return false;
            }
        }
        const auto admission_guard = qScopeGuard(
            [&admission_release]()
            {
                if (admission_release)
                {
                    admission_release();
                }
            });

        reapFinishedFutures();
        if (_isRunning || !_session || !_session->hasProject())
        {
            if (_messages)
            {
                _messages->information(
                    nullptr,
                    QStringLiteral("光束法平差"),
                    _isRunning ? QStringLiteral("已有光束法平差任务正在运行，请等待其结束或先取消当前任务。")
                               : QStringLiteral("请先打开项目"));
            }
            return false;
        }
        if (images.size() < 2)
        {
            if (_messages)
            {
                _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("至少需要选择两张影像"));
            }
            return false;
        }
        if (outputDir.trimmed().isEmpty())
        {
            if (_messages)
            {
                _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("请指定输出目录"));
            }
            return false;
        }

        xjw::gui::BaServiceOptions options;
        QString options_error;
        if (!buildOptions(images, outputDir, threads, dryRun, extraSettings, &options, &options_error))
        {
            if (_messages)
            {
                _messages->warning(nullptr, QStringLiteral("光束法平差"), options_error);
            }
            return false;
        }

        discardPreview();
        _taskContext.taskId = QString::fromLatin1(kTaskId);
        _taskContext.session = _session->context();
        _taskContext.cancelFlag = std::make_shared<std::atomic<bool>>(false);
        options.baOpt.cancelFlag = _taskContext.cancelFlag;
        _isRunning = true;
        _terminalEmitted = false;

        const ProjectTaskContext task_context = _taskContext;
        QPointer<ProjectBundleAdjustController> self(this);
        options.baOpt.progressCallback =
            [self, task_context](int currentIteration, int maxIterations, double avgRms, int validPoints)
        {
            if (!self || !task_context.cancelFlag || task_context.cancelFlag->load(std::memory_order_relaxed))
            {
                return false;
            }
            const int safe_max_iterations = std::max(1, maxIterations);
            const int percent =
                qBound(10, 10 + static_cast<int>(std::lround(80.0 * currentIteration / safe_max_iterations)), 90);
            const QString stage = QStringLiteral("光束法平差优化中... %1/%2 RMS=%3 有效点=%4")
                                      .arg(currentIteration)
                                      .arg(safe_max_iterations)
                                      .arg(avgRms, 0, 'f', 4)
                                      .arg(validPoints);
            QMetaObject::invokeMethod(
                self.data(),
                [self, task_context, stage, percent]()
                {
                    if (self && self->isCurrentTask(task_context))
                    {
                        emit self->progressChanged(stage, percent);
                    }
                },
                Qt::QueuedConnection);
            return true;
        };

        const QJsonObject core_data = _session->coreMetadata();
        const QString project_path = _session->projectPath();
        const int min_matches = qMax(0, extraSettings.value(QStringLiteral("min_matches")).toInt(0));
        const ExecutionRunner runner = _executionRunner;

        emit progressChanged(QStringLiteral("光束法平差准备中..."), 1);
        LOG_INFO(QStringLiteral("BA: 特征/匹配准备完毕，启动光束法平差"));

        QFuture<void> future = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [self,
             runner,
             core_data,
             project_path,
             images,
             min_matches,
             options = std::move(options),
             task_context]() mutable
            {
                if (!self || !task_context.cancelFlag || task_context.cancelFlag->load(std::memory_order_relaxed))
                {
                    return BundleAdjustExecutionResult{};
                }
                QMetaObject::invokeMethod(
                    self.data(),
                    [self, task_context]()
                    {
                        if (self && self->isCurrentTask(task_context))
                        {
                            emit self->progressChanged(QStringLiteral("光束法平差构建输入..."), 5);
                        }
                    },
                    Qt::QueuedConnection);
                return runner(core_data, project_path, images, min_matches, std::move(options));
            },
            [task_context, dryRun](ProjectBundleAdjustController* controller,
                                   xjw::gui::tasks::TaskOutcome<BundleAdjustExecutionResult> outcome)
            {
                if (!outcome.succeeded())
                {
                    controller->handleExecutionError(task_context, outcome.errorMessage);
                    controller->clearTaskAfterWorkerExit(task_context);
                    return;
                }
                controller->handleExecutionFinished(task_context, *outcome.value, dryRun);
                controller->clearTaskAfterWorkerExit(task_context);
            });
        _activeFutures.push_back(std::move(future));
        return true;
    }

    bool ProjectBundleAdjustController::buildOptions(const QStringList& images,
                                                     const QString& outputDir,
                                                     int threads,
                                                     bool dryRun,
                                                     const QJsonObject& extraSettings,
                                                     xjw::gui::BaServiceOptions* options,
                                                     QString* errorMessage) const
    {
        if (!options || !ProjectConfigManager::validateBundleAdjustSettings(extraSettings, errorMessage))
        {
            return false;
        }

        options->selectedImages = images;
        options->outputDir = QDir::cleanPath(outputDir);
        options->dryRun = dryRun;
        options->threads = threads;
        options->baOpt.maxIterations = qBound(3, extraSettings.value(QStringLiteral("max_iterations")).toInt(20), 200);
        options->baOpt.refineCameraPose = extraSettings.value(QStringLiteral("refine_camera_pose")).toBool(true);
        options->baOpt.numThreads = threads;
        const QString backend =
            extraSettings.value(QStringLiteral("ba_backend")).toString(QStringLiteral("auto")).trimmed().toLower();
        if (backend == QLatin1String("auto"))
        {
            options->baOpt.backend = xjw::BABackend::Auto;
        }
        else if (backend == QLatin1String("plamatrix_cpu"))
        {
            options->baOpt.backend = xjw::BABackend::PlaMatrixCpu;
        }
        else if (backend == QLatin1String("plamatrix_cuda"))
        {
            options->baOpt.backend = xjw::BABackend::PlaMatrixCuda;
        }
        else if (backend == QLatin1String("plamatrix_opencl"))
        {
            options->baOpt.backend = xjw::BABackend::PlaMatrixOpenCl;
        }
        else
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("不支持 BA 后端：%1").arg(backend);
            }
            return false;
        }

        options->baOpt.plaMatrixDevice = qMax(0, extraSettings.value(QStringLiteral("ba_plamatrix_device")).toInt(0));
        options->baOpt.minPlaMatrixCudaCameras = qMax(
            1,
            extraSettings.value(QStringLiteral("ba_min_cuda_cameras")).toInt(options->baOpt.minPlaMatrixCudaCameras));
        options->baOpt.minPlaMatrixCudaObservations =
            qMax(1,
                 extraSettings.value(QStringLiteral("ba_min_cuda_observations"))
                     .toInt(options->baOpt.minPlaMatrixCudaObservations));
        options->baOpt.minPlaMatrixOpenClCameras = qMax(1,
                                                        extraSettings.value(QStringLiteral("ba_min_opencl_cameras"))
                                                            .toInt(options->baOpt.minPlaMatrixOpenClCameras));
        options->baOpt.minPlaMatrixOpenClObservations =
            qMax(1,
                 extraSettings.value(QStringLiteral("ba_min_opencl_observations"))
                     .toInt(options->baOpt.minPlaMatrixOpenClObservations));
        options->baOpt.minPlaMatrixDenseCameras = qMax(
            1,
            extraSettings.value(QStringLiteral("ba_min_dense_cameras")).toInt(options->baOpt.minPlaMatrixDenseCameras));
        options->baOpt.minPlaMatrixCudaDenseObservations =
            qMax(1,
                 extraSettings.value(QStringLiteral("ba_min_cuda_dense_observations"))
                     .toInt(options->baOpt.minPlaMatrixCudaDenseObservations));
        options->baOpt.minPlaMatrixOpenClDenseObservations =
            qMax(1,
                 extraSettings.value(QStringLiteral("ba_min_opencl_dense_observations"))
                     .toInt(options->baOpt.minPlaMatrixOpenClDenseObservations));
        options->baOpt.maxInitialTrackRms = qMax(0.0,
                                                 extraSettings.value(QStringLiteral("ba_max_initial_track_rms"))
                                                     .toDouble(options->baOpt.maxInitialTrackRms));
        options->baOpt.allowBackendFallback =
            extraSettings.value(QStringLiteral("ba_allow_backend_fallback")).toBool(true);
        options->baOpt.maxAcceptedConstraintRmsGrowth =
            qMax(1.0,
                 extraSettings.value(QStringLiteral("ba_max_accepted_constraint_rms_growth"))
                     .toDouble(options->baOpt.maxAcceptedConstraintRmsGrowth));
        options->baOpt.enableBackendQualityGate =
            extraSettings.value(QStringLiteral("ba_enable_backend_quality_gate")).toBool(true);
        options->baOpt.maxAcceptedRmsGrowth = qMax(0.0,
                                                   extraSettings.value(QStringLiteral("ba_max_accepted_rms_growth"))
                                                       .toDouble(options->baOpt.maxAcceptedRmsGrowth));
        options->baOpt.minAcceptedValidTrackRatio =
            qMax(0.0,
                 extraSettings.value(QStringLiteral("ba_min_accepted_valid_track_ratio"))
                     .toDouble(options->baOpt.minAcceptedValidTrackRatio));
        options->baOpt.enablePointFilter = true;
        options->baOpt.filterMaxReprojError =
            extraSettings.value(QStringLiteral("filter_max_reproj_error")).toDouble(2.5);
        options->baOpt.filterSigmaFactor = extraSettings.value(QStringLiteral("filter_sigma_factor")).toDouble(3.0);
        options->exportTsai = extraSettings.value(QStringLiteral("export_tsai")).toBool(true);
        options->exportSummaryTxt = extraSettings.value(QStringLiteral("export_summary_txt")).toBool(true);
        options->exportPointsCsv = extraSettings.value(QStringLiteral("export_points_csv")).toBool(true);
        options->exportCameraCsv = extraSettings.value(QStringLiteral("export_camera_csv")).toBool(true);
        options->exportRunJson = extraSettings.value(QStringLiteral("export_run_json")).toBool(true);
        options->exportEvalPlot = extraSettings.value(QStringLiteral("export_eval_plot")).toBool(true);
        options->exportObservationDetails =
            extraSettings.value(QStringLiteral("export_observation_details")).toBool(true);
        options->enableLaserConstraints = extraSettings.value(QStringLiteral("enable_laser_constraints")).toBool(false);
        options->laserConstraintCloudPath =
            extraSettings.value(QStringLiteral("laser_constraint_cloud_path")).toString().trimmed();
        options->laserAssociationMaxDistanceMeters =
            qMax(0.0, extraSettings.value(QStringLiteral("laser_association_max_distance_m")).toDouble(0.05));
        options->laserVoxelSizeMeters =
            qMax(0.0, extraSettings.value(QStringLiteral("laser_voxel_size_m")).toDouble(0.0));
        options->laserMaxCurvature =
            qBound(0.0, extraSettings.value(QStringLiteral("laser_max_curvature")).toDouble(0.2), 1.0);
        options->laserMaxSamples =
            qBound(1, extraSettings.value(QStringLiteral("laser_max_samples")).toInt(500000), 10000000);
        options->laserUseMissingNormalsAsHeightPlanes =
            extraSettings.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool(false);
        options->laserWeight = qMax(0.0, extraSettings.value(QStringLiteral("laser_weight")).toDouble(0.0));
        options->laserSigmaMeters = qMax(1e-9, extraSettings.value(QStringLiteral("laser_sigma_m")).toDouble(0.0025));
        options->laserHuberDeltaMeters =
            qMax(1e-9, extraSettings.value(QStringLiteral("laser_huber_delta_m")).toDouble(0.05));
        options->enablePlanetaryLaserRangeConstraints =
            extraSettings.value(QStringLiteral("enable_planetary_laser_range_constraints")).toBool(false);
        options->planetaryLaserDataPath =
            extraSettings.value(QStringLiteral("planetary_laser_data_path")).toString().trimmed();
        options->planetaryLaserCameraCoordinateFrame =
            extraSettings.value(QStringLiteral("planetary_laser_camera_coordinate_frame")).toString().trimmed();
        options->planetaryLaserCameraSensorFrame =
            extraSettings.value(QStringLiteral("planetary_laser_camera_sensor_frame")).toString().trimmed();
        options->planetaryLaserConfirmUnknownSensorIsFrame =
            extraSettings.value(QStringLiteral("planetary_laser_confirm_unknown_sensor_is_frame")).toBool(false);
        options->planetaryLaserConfirmUnknownRangeIsOneWay =
            extraSettings.value(QStringLiteral("planetary_laser_confirm_unknown_range_is_one_way")).toBool(false);
        options->planetaryLaserAllowUnmappedShots =
            extraSettings.value(QStringLiteral("planetary_laser_allow_unmapped_shots")).toBool(false);
        options->planetaryLaserAllowUnmappedMeasuredImages =
            extraSettings.value(QStringLiteral("planetary_laser_allow_unmapped_measured_images")).toBool(false);
        options->planetaryLaserRangeWeight =
            qMax(1.0e-12, extraSettings.value(QStringLiteral("planetary_laser_range_weight")).toDouble(1.0));
        options->planetaryLaserRangeHuberDeltaSigma =
            qMax(0.0, extraSettings.value(QStringLiteral("planetary_laser_range_huber_delta_sigma")).toDouble(3.0));
        options->enableReferenceTerrainPrior =
            extraSettings.value(QStringLiteral("enable_reference_terrain_prior")).toBool(false);
        options->referenceTerrainDemPath =
            extraSettings.value(QStringLiteral("reference_terrain_dem_path")).toString().trimmed();
        options->referenceTerrainSigmaMeters =
            qMax(1e-9, extraSettings.value(QStringLiteral("reference_terrain_sigma_m")).toDouble(1.0));
        options->referenceTerrainMaxAssociationDistanceMeters = qMax(
            0.0, extraSettings.value(QStringLiteral("reference_terrain_max_association_distance_m")).toDouble(2.0));
        options->referenceTerrainHuberDeltaMeters =
            qMax(1e-9, extraSettings.value(QStringLiteral("reference_terrain_huber_delta_m")).toDouble(0.5));
        return true;
    }

    bool ProjectBundleAdjustController::acceptPreview(QString* errorMessage)
    {
        if (_acceptanceInProgress)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("平差预览正在应用，拒绝重复接受");
            }
            return false;
        }
        if (_previewAdmissionPredicate && !_previewAdmissionPredicate(errorMessage))
        {
            return false;
        }
        _acceptanceInProgress = true;
        const auto acceptance_guard = qScopeGuard([this]() { _acceptanceInProgress = false; });

        const ProjectTaskContext preview_context = _previewContext;
        if (!isCurrentPreview(preview_context))
        {
            if (errorMessage)
            {
                *errorMessage = _hasPendingPreview ? QStringLiteral("项目会话已变化，平差预览已失效")
                                                   : QStringLiteral("当前没有可应用的平差预览结果");
            }
            if (_hasPendingPreview)
            {
                discardPreview();
            }
            return false;
        }

        const QStringList all_images = _session->allImages();
        if (!_previewArtifactsFinalized)
        {
            const QString assets_dir = xjw::common::project::ProjectIO::projectAssetsDir(_session->projectPath());
            const QString ba_output_dir =
                assets_dir.isEmpty()
                    ? QString()
                    : QDir(assets_dir)
                          .filePath(QStringLiteral("aerial_triangulation/ba_refined_%1")
                                        .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));
            const QMap<QString, QJsonObject> after_camera_meta =
                cameraUpdatesForPresentation(_session->coreMetadata(), _pendingCameraUpdates);
            const BundleAdjustArtifactsResult artifacts =
                finalizeBundleAdjustArtifacts(assets_dir,
                                              _pendingResult,
                                              all_images,
                                              _pendingResult.value(QStringLiteral("output_dir")).toString(),
                                              QStringLiteral("reconstruction_bundle_adjust"),
                                              _pendingBeforeCameraMeta,
                                              after_camera_meta,
                                              ba_output_dir,
                                              true);
            if (!artifacts.reportWarning.isEmpty())
            {
                LOG_WARN(QStringLiteral("BA: %1").arg(artifacts.reportWarning));
            }
            if (artifacts.sparseCloudExport.exported)
            {
                _pendingTiePointSparseCloudPath = artifacts.sparseCloudExport.sparseCloudPath;
                _pendingTiePointCount = artifacts.sparseCloudExport.pointCount;
                _pendingTiePointSelectedImages = all_images;
                _pendingTiePointOutputDir = artifacts.sparseCloudExport.outputDir;
                _pendingTiePointExtraRecord = artifacts.sparseCloudExport.extraRecord;
                _pendingTiePointWrite = true;
            }
            else if (!artifacts.sparseCloudExport.errorMessage.isEmpty())
            {
                LOG_WARN(QStringLiteral("写入 BA 点云 sidecar 失败: %1").arg(artifacts.sparseCloudExport.errorMessage));
            }
            _previewArtifactsFinalized = true;
        }

        if (!isCurrentPreview(preview_context))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("项目会话已变化，平差预览已失效");
            }
            return false;
        }

        ProjectBundleAdjustMetadataStageToken stage_token;
        QString stage_error;
        if (!_session->stageBundleAdjustMetadata(
                preview_context.session, _pendingCameraUpdates, _pendingResult, &stage_token, &stage_error))
        {
            if (errorMessage)
            {
                *errorMessage = stage_error;
            }
            return false;
        }
        _previewMetadataStage = stage_token;

        if (_pendingTiePointWrite)
        {
            const bool writer_succeeded =
                _tiePointResultWriter &&
                _tiePointResultWriter(preview_context,
                                      _pendingTiePointSparseCloudPath,
                                      _pendingTiePointCount,
                                      _pendingTiePointSelectedImages,
                                      _pendingTiePointOutputDir,
                                      _pendingTiePointExtraRecord);
            if (!writer_succeeded)
            {
                const auto rolled_back = _session->resolveBundleAdjustMetadataStage(
                    preview_context.session, stage_token, ProjectBundleAdjustMetadataStageDecision::Rollback);
                _previewMetadataStage = {};
                if (rolled_back.status != ProjectBundleAdjustMetadataStageStatus::RolledBack)
                {
                    clearPreviewState(true);
                }
                if (errorMessage)
                {
                    *errorMessage = !_tiePointResultWriter
                                        ? QStringLiteral("BA 连接点成果写入器未配置")
                                        : (rolled_back.status == ProjectBundleAdjustMetadataStageStatus::RolledBack
                                               ? QStringLiteral("写入 BA 连接点成果失败")
                                               : rolled_back.errorMessage);
                }
                return false;
            }
            _pendingTiePointWrite = false;
        }

        if (!isCurrentPreview(preview_context))
        {
            _session->resolveBundleAdjustMetadataStage(
                preview_context.session, stage_token, ProjectBundleAdjustMetadataStageDecision::Rollback);
            _previewMetadataStage = {};
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("平差预览在写入期间已失效");
            }
            return false;
        }

        const auto committed = _session->resolveBundleAdjustMetadataStage(
            preview_context.session, stage_token, ProjectBundleAdjustMetadataStageDecision::Commit);
        _previewMetadataStage = {};
        if (committed.status != ProjectBundleAdjustMetadataStageStatus::Committed &&
            committed.status != ProjectBundleAdjustMetadataStageStatus::ExternalCommit)
        {
            if (errorMessage)
            {
                *errorMessage = committed.errorMessage;
            }
            clearPreviewState(true);
            return false;
        }
        if (committed.status == ProjectBundleAdjustMetadataStageStatus::ExternalCommit &&
            isCurrentPreview(preview_context) && _postExternalCommitObserver)
        {
            _postExternalCommitObserver(preview_context);
        }
        if (!isCurrentPreview(preview_context))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("平差预览在提交期间已失效");
            }
            return false;
        }

        const int updated_camera_count = stage_token.updatedCameraCount();
        clearPreviewState(false);
        if (_messages)
        {
            _messages->information(nullptr,
                                   QStringLiteral("光束法平差"),
                                   QStringLiteral("已保留本次平差结果，并更新 %1 台相机参数。\n"
                                                  "详细指标可在“工具 > 查看工作流程报告”中查看。")
                                       .arg(updated_camera_count));
        }
        return true;
    }

    void ProjectBundleAdjustController::setPostExternalCommitObserver(PostExternalCommitObserver observer)
    {
        _postExternalCommitObserver = std::move(observer);
    }

    void ProjectBundleAdjustController::discardPreview()
    {
        clearPreviewState(true);
    }

    void ProjectBundleAdjustController::clearPreviewState(bool cancelContext)
    {
        if (cancelContext && _previewContext.cancelFlag)
        {
            _previewContext.cancelFlag->store(true, std::memory_order_relaxed);
        }
        rollbackPreviewMetadataStage();
        _pendingCameraUpdates.clear();
        _pendingBeforeCameraMeta.clear();
        _pendingResult = {};
        _previewContext = {};
        _pendingTiePointSparseCloudPath.clear();
        _pendingTiePointSelectedImages.clear();
        _pendingTiePointOutputDir.clear();
        _pendingTiePointExtraRecord = {};
        _pendingTiePointCount = 0;
        _hasPendingPreview = false;
        _previewArtifactsFinalized = false;
        _pendingTiePointWrite = false;
    }

    ProjectBundleAdjustMetadataStageResolveResult ProjectBundleAdjustController::rollbackPreviewMetadataStage()
    {
        if (!_previewMetadataStage.isValid() || !_session)
        {
            return {};
        }
        const ProjectBundleAdjustMetadataStageToken stage_token = _previewMetadataStage;
        _previewMetadataStage = {};
        return _session->resolveBundleAdjustMetadataStage(
            _previewContext.session, stage_token, ProjectBundleAdjustMetadataStageDecision::Rollback);
    }

    bool ProjectBundleAdjustController::cancel()
    {
        if (!_isRunning || !_taskContext.cancelFlag)
        {
            if (_hasPendingPreview)
            {
                discardPreview();
                emit previewReady(QJsonObject());
                return true;
            }
            return false;
        }
        _taskContext.cancelFlag->store(true, std::memory_order_relaxed);
        discardPreview();
        emit previewReady(QJsonObject());
        completeOnce(false, QString());
        return true;
    }

    bool ProjectBundleAdjustController::isRunning() const noexcept
    {
        return _isRunning;
    }

    bool ProjectBundleAdjustController::hasPendingPreview() const noexcept
    {
        return _hasPendingPreview;
    }

    ProjectTaskContext ProjectBundleAdjustController::taskContext() const
    {
        return _taskContext;
    }

    void ProjectBundleAdjustController::waitForFinished()
    {
        for (QFuture<void>& future : _activeFutures)
        {
            future.waitForFinished();
        }
        _activeFutures.clear();
    }

    void ProjectBundleAdjustController::setExecutionRunnerForTesting(ExecutionRunner runner)
    {
        if (!_isRunning && runner)
        {
            _executionRunner = std::move(runner);
        }
    }

    void ProjectBundleAdjustController::setTiePointResultWriter(TiePointResultWriter writer)
    {
        _tiePointResultWriter = std::move(writer);
    }

    bool ProjectBundleAdjustController::isCurrentTask(const ProjectTaskContext& context, bool allowCancelled) const
    {
        return !_destroying && _isRunning && _session && _session->isCurrent(context.session) &&
               _taskContext.taskId == context.taskId && _taskContext.cancelFlag == context.cancelFlag &&
               context.cancelFlag && (allowCancelled || !context.cancelFlag->load(std::memory_order_relaxed));
    }

    bool ProjectBundleAdjustController::isCurrentPreview(const ProjectTaskContext& context) const
    {
        return !_destroying && _hasPendingPreview && !_pendingCameraUpdates.empty() && _session &&
               _session->isCurrent(_previewContext.session) && _previewContext.session.matches(context.session) &&
               _previewContext.taskId == context.taskId && _previewContext.cancelFlag == context.cancelFlag &&
               context.cancelFlag && !context.cancelFlag->load(std::memory_order_relaxed);
    }

    void ProjectBundleAdjustController::handleExecutionFinished(const ProjectTaskContext& context,
                                                                const BundleAdjustExecutionResult& executionResult,
                                                                bool dryRun)
    {
        if (!isCurrentTask(context))
        {
            return;
        }
        if (executionResult.buildStatus != xjw::core::project::BaInputBuildStatus::Ok)
        {
            const QString message =
                buildInputError(executionResult.buildStatus, executionResult.serviceResult.errorMessage);
            discardPreview();
            if (_messages)
            {
                _messages->warning(nullptr, QStringLiteral("提示"), message);
            }
            if (!isCurrentTask(context))
            {
                return;
            }
            emit previewReady(QJsonObject());
            if (!isCurrentTask(context))
            {
                return;
            }
            commitTerminal(context, false, message);
            return;
        }

        emit progressChanged(QStringLiteral("光束法平差整理结果..."), 95);
        if (!isCurrentTask(context))
        {
            return;
        }
        const xjw::gui::BaServiceResult& result = executionResult.serviceResult;
        if (!result.success && !dryRun && _messages)
        {
            _messages->warning(nullptr,
                               QStringLiteral("平差提示"),
                               QStringLiteral("光束法平差执行出现问题：%1").arg(result.errorMessage));
        }
        if (!isCurrentTask(context))
        {
            return;
        }
        const bool has_preview = result.success && !dryRun && !result.cameraInstanceUpdates.empty();
        if (has_preview)
        {
            _pendingBeforeCameraMeta = executionResult.beforeCamMeta;
            _pendingCameraUpdates = result.cameraInstanceUpdates;
            _pendingResult = result.resultJson;
            _previewContext = context;
            _hasPendingPreview = true;
        }
        else
        {
            discardPreview();
        }

        emit previewReady(result.resultJson);
        if (!isCurrentTask(context))
        {
            return;
        }
        if (!commitTerminal(context, result.success, result.errorMessage))
        {
            return;
        }
        if (has_preview)
        {
            presentPreview(context);
        }
    }

    void ProjectBundleAdjustController::handleExecutionError(const ProjectTaskContext& context, const QString& message)
    {
        if (!isCurrentTask(context))
        {
            return;
        }
        discardPreview();
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("光束法平差"), message);
        }
        if (!isCurrentTask(context))
        {
            return;
        }
        emit previewReady(QJsonObject());
        if (!isCurrentTask(context))
        {
            return;
        }
        commitTerminal(context, false, message);
    }

    bool ProjectBundleAdjustController::commitTerminal(const ProjectTaskContext& context,
                                                       bool success,
                                                       const QString& message)
    {
        if (!isCurrentTask(context))
        {
            return false;
        }
        _isRunning = false;
        completeOnce(success, message);
        return true;
    }

    void ProjectBundleAdjustController::completeOnce(bool success, const QString& message)
    {
        if (_terminalEmitted)
        {
            return;
        }
        _terminalEmitted = true;
        if (!success && !message.isEmpty())
        {
            emit error(message);
        }
        emit finished(success, message);
    }

    void ProjectBundleAdjustController::presentPreview(const ProjectTaskContext& context)
    {
        while (_messages && isCurrentPreview(context))
        {
            const BundleAdjustPreviewPresentation presentation =
                buildBundleAdjustPreviewPresentation(_pendingResult, static_cast<int>(_pendingCameraUpdates.size()));
            UiReviewDialogRequest request;
            request.objectName = QStringLiteral("bundleAdjustPreviewMessageBox");
            request.title = QStringLiteral("参考地形约束重新平差");
            request.text = presentation.summaryText;
            request.informativeText =
                presentation.qualityWarning
                    ? QStringLiteral("关键质量指标存在警告。请展开详细信息核对后，再决定是否写回项目。")
                    : QStringLiteral("请核对指标后选择“保留结果”写回项目，或选择“丢弃结果”保持原相机参数。");
            request.detailedText = presentation.detailedText;
            request.acceptText = QStringLiteral("保留结果");
            request.discardText = QStringLiteral("丢弃结果");
            request.acceptObjectName = QStringLiteral("keepBundleAdjustPreviewButton");
            request.discardObjectName = QStringLiteral("discardBundleAdjustPreviewButton");
            request.warningIcon = presentation.qualityWarning;

            const UiReviewDecision decision = _messages->review(nullptr, request);
            if (!isCurrentPreview(context))
            {
                return;
            }
            if (decision != UiReviewDecision::Accept)
            {
                discardPreview();
                LOG_INFO(QStringLiteral("BA: 用户丢弃了待提交的平差结果"));
                return;
            }

            QString error_message;
            if (acceptPreview(&error_message))
            {
                return;
            }
            if (!isCurrentPreview(context))
            {
                return;
            }
            _messages->warning(
                nullptr,
                QStringLiteral("光束法平差"),
                QStringLiteral("应用平差结果失败：%1\n\n结果仍保留在内存中，可重试或选择丢弃。").arg(error_message));
        }
    }

    void ProjectBundleAdjustController::handleSessionChanged()
    {
        discardPreview();
        if (_isRunning && _taskContext.cancelFlag)
        {
            _taskContext.cancelFlag->store(true, std::memory_order_relaxed);
            _terminalEmitted = true;
        }
    }

    bool ProjectBundleAdjustController::hasPendingWork() const noexcept
    {
        return std::any_of(_activeFutures.cbegin(),
                           _activeFutures.cend(),
                           [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
    }

    void ProjectBundleAdjustController::settleAfterDrain()
    {
        if (hasPendingWork())
        {
            return;
        }
        discardPreview();
        _taskContext = {};
        _isRunning = false;
        _terminalEmitted = true;
        reapFinishedFutures();
    }

    void ProjectBundleAdjustController::reapFinishedFutures()
    {
        for (qsizetype index = _activeFutures.size() - 1; index >= 0; --index)
        {
            if (_activeFutures.at(index).isFinished())
            {
                _activeFutures.removeAt(index);
            }
        }
    }

    void ProjectBundleAdjustController::clearTaskAfterWorkerExit(const ProjectTaskContext& context)
    {
        if (_taskContext.taskId == context.taskId && _taskContext.cancelFlag == context.cancelFlag)
        {
            _isRunning = false;
            _taskContext = {};
        }
    }

} // namespace xjw::gui::project
