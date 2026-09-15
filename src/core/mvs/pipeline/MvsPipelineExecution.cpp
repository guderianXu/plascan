#include "MvsPipelineInternals.h"
#include "DepthArtifactSaveQueue.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void MvsPipelineService::runInBackgroundImpl()
    {
        const int NV = static_cast<int>(_views.size());
        const int runCpuThreadBudget = resolvedTotalCpuThreadBudget(_config);
#ifdef _OPENMP
        // This background thread owns the serial preparation, consistency, and
        // fusion stages. Keep their implicit OpenMP teams inside the same budget
        // used by frame workers instead of falling back to all logical CPUs.
        omp_set_num_threads(runCpuThreadBudget);
#endif

        if (!_config.runDepthEstimation)
        {
            errorOccurred(QStringLiteral("当前生成器配置未启用深度估计阶段"));
            emitFinishedOnce(false);
            return;
        }

        _stageSnapshotRecorder.reset();
        if (!_config.stageSnapshotReferenceIndices.empty())
        {
            _stageSnapshotRecorder = std::make_unique<MvsStageSnapshotRecorder>(_config, NV);
            if (!_stageSnapshotRecorder->enabled())
            {
                LOG_WARN(QStringLiteral("[MVS][阶段快照] 诊断通道未启用：%1；主流程继续")
                             .arg(_stageSnapshotRecorder->initializationError()));
            }
            else
            {
                LOG_INFO(QStringLiteral("[MVS][阶段快照] 已启用：manifest=%1 refs=%2 "
                                        "max_long_edge=%3 budget=%4 MiB")
                             .arg(_stageSnapshotRecorder->manifestPath())
                             .arg(_config.stageSnapshotReferenceIndices.size())
                             .arg(_config.stageSnapshotMaximumLongEdge)
                             .arg(_config.stageSnapshotBudgetBytes / (1024 * 1024)));
            }
        }

        _sceneClassification = classifyMvsScene(_views, _sparse);
        _effectiveSceneProfile =
            _config.sceneProfile == MvsSceneProfile::Auto ? _sceneClassification.profile : _config.sceneProfile;
        // The recovered scene algorithm owns source selection and filtering.
        // Keep these values aligned with its fixed Mild/16-neighbor contract so
        // legacy PlaScan tuning cannot alter the production depth result.
        _effectiveDepthFilterMode = DepthFilterMode::Mild;
        _configuredSourceViewCount = 16;
        _config.numSourceViews = 16;
        _config.patchMatch.numSourceViews = _config.numSourceViews;
        LOG_INFO(QStringLiteral("[MVS] recovered scene 输入分类: profile=%1 plane_thickness=%2 "
                                "down_looking=%3 axis_convergence=%4 filter=mild "
                                "source_pool<=%5 reason=%6")
                     .arg(sceneProfileId(_effectiveSceneProfile))
                     .arg(_sceneClassification.planeThicknessRatio, 0, 'f', 3)
                     .arg(_sceneClassification.downLookingConsistency, 0, 'f', 3)
                     .arg(_sceneClassification.opticalAxisConvergence, 0, 'f', 3)
                     .arg(_config.numSourceViews)
                     .arg(QString::fromStdString(_sceneClassification.reason)));

        const PatchMatchBackend configuredBackend = _config.patchMatch.backend;
        if (configuredBackend == PatchMatchBackend::Cpu || configuredBackend == PatchMatchBackend::OpenCl)
        {
            const QString error =
                QStringLiteral("当前 recovered 多视深度生产算法仅支持 CUDA；CPU/OpenCL 不会回退到旧 PatchMatch");
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(error));
            errorOccurred(error);
            emitFinishedOnce(false);
            return;
        }
        const bool automaticAcceleration = configuredBackend == PatchMatchBackend::Auto;
        const bool probeCuda = configuredBackend == PatchMatchBackend::Cuda || automaticAcceleration;
        const int cudaDeviceCount = probeCuda ? PatchMatchDepthEstimator::cudaDeviceCount() : 0;

        std::vector<DepthComputeWorker> physicalAcceleratorWorkers;
        std::vector<std::unique_ptr<GpuDeviceLeaseSet>> acceleratorDeviceLeases;
        std::unordered_set<std::string> selectedPhysicalDeviceIdentities;
        QStringList acceleratorPreparationFailures;
        auto acquire_device_lease = [&acceleratorPreparationFailures](
                                        const GpuDeviceDescriptor& descriptor) -> std::unique_ptr<GpuDeviceLeaseSet>
        {
            auto lease = std::make_unique<GpuDeviceLeaseSet>();
            QString lease_error;
            if (!lease->acquire({descriptor}, &lease_error))
            {
                acceleratorPreparationFailures.push_back(lease_error);
                LOG_WARN(QStringLiteral("[MVS] %1").arg(lease_error));
                return nullptr;
            }
            return lease;
        };

        auto try_add_cuda_device = [&](int device_index)
        {
            const std::string name = PatchMatchDepthEstimator::cudaDeviceName(device_index);
            std::string identity = PatchMatchDepthEstimator::cudaDeviceIdentity(device_index);
            if (identity.empty())
            {
                identity = fallbackGpuPhysicalIdentity("NVIDIA", name, device_index);
            }
            const GpuDeviceDescriptor descriptor{identity,
                                                 name.empty() ? "CUDA:" + std::to_string(device_index) : name};
            std::unique_ptr<GpuDeviceLeaseSet> lease = acquire_device_lease(descriptor);
            if (!lease)
            {
                return;
            }
            physicalAcceleratorWorkers.push_back({DepthComputeBackend::Cuda, device_index});
            selectedPhysicalDeviceIdentities.insert(descriptor.physicalIdentity);
            acceleratorDeviceLeases.push_back(std::move(lease));
        };

        if (_config.patchMatch.cudaDeviceIndex >= 0 && _config.patchMatch.cudaDeviceIndex < cudaDeviceCount)
        {
            try_add_cuda_device(_config.patchMatch.cudaDeviceIndex);
        }
        else if (_config.patchMatch.cudaDeviceIndex < 0)
        {
            for (int device_index = 0; device_index < cudaDeviceCount; ++device_index)
            {
                try_add_cuda_device(device_index);
            }
        }
        const bool cudaAvailable = !physicalAcceleratorWorkers.empty();
        // recovered depth is CUDA-only. Do not enumerate, lease, or initialize an
        // OpenCL alias during Auto selection.
        const bool probeOpenCl = false;
        const std::vector<OpenClDeviceInfo> detectedOpenClDevices =
            probeOpenCl ? PatchMatchDepthEstimator::openClDevices() : std::vector<OpenClDeviceInfo>{};
        std::vector<OpenClDeviceInfo> selectedOpenClDevices;
        for (const OpenClDeviceInfo& device : detectedOpenClDevices)
        {
            if (_config.patchMatch.openClDeviceIndex >= 0 && device.index != _config.patchMatch.openClDeviceIndex)
            {
                continue;
            }
            const GpuDeviceDescriptor descriptor{device.physicalDeviceIdentity, device.vendor + " " + device.name};
            if (selectedPhysicalDeviceIdentities.contains(descriptor.physicalIdentity))
            {
                LOG_DEBUG(QStringLiteral("[MVS] 跳过与已选 CUDA 设备重复的 OpenCL 接口: index=%1 device=%2 identity=%3")
                              .arg(device.index)
                              .arg(QString::fromStdString(device.name))
                              .arg(QString::fromStdString(descriptor.physicalIdentity)));
                continue;
            }
            if (shouldSkipUnstableOpenClCudaAlias(device.vendor, descriptor.physicalIdentity, cudaAvailable))
            {
                LOG_WARN(QStringLiteral("[MVS] 跳过无法取得稳定 PCI 身份的 NVIDIA OpenCL 接口："
                                        "index=%1 device=%2；CUDA 已接管该厂商设备，避免重复执行通道")
                             .arg(device.index)
                             .arg(QString::fromStdString(device.name)));
                continue;
            }
            std::unique_ptr<GpuDeviceLeaseSet> lease = acquire_device_lease(descriptor);
            if (!lease)
            {
                continue;
            }

            std::string preparation_error;
            if (!PatchMatchDepthEstimator::prepareOpenClDevice(device.index, &preparation_error))
            {
                QString detail = QStringLiteral("OpenCL GPU %1 (%2) 预检失败：%3")
                                     .arg(device.index)
                                     .arg(QString::fromStdString(device.name))
                                     .arg(QString::fromStdString(preparation_error));
                if (detail.size() > 1024)
                {
                    detail = detail.left(1021) + QStringLiteral("...");
                }
                acceleratorPreparationFailures.push_back(detail);
                LOG_WARN(QStringLiteral("[MVS] %1").arg(detail));
                continue;
            }
            selectedOpenClDevices.push_back(device);
            physicalAcceleratorWorkers.push_back({DepthComputeBackend::OpenCl, device.index});
            selectedPhysicalDeviceIdentities.insert(descriptor.physicalIdentity);
            acceleratorDeviceLeases.push_back(std::move(lease));
        }
        const std::optional<DepthComputeBackend> requestedBackend =
            configuredBackend == PatchMatchBackend::Auto
                ? std::nullopt
                : std::make_optional(configuredBackend == PatchMatchBackend::Cuda     ? DepthComputeBackend::Cuda
                                     : configuredBackend == PatchMatchBackend::OpenCl ? DepthComputeBackend::OpenCl
                                                                                      : DepthComputeBackend::Cpu);
        const DepthComputeBackend effectiveBackend =
            resolveDepthComputeBackend(requestedBackend, cudaAvailable, !selectedOpenClDevices.empty());
        const bool openClAvailable = !selectedOpenClDevices.empty();
        const bool heterogeneousAuto =
            configuredBackend == PatchMatchBackend::Auto && automaticAcceleration && cudaAvailable && openClAvailable;
        const bool requestedBackendUnavailable = (effectiveBackend == DepthComputeBackend::Cuda && !cudaAvailable) ||
                                                 (effectiveBackend == DepthComputeBackend::OpenCl && !openClAvailable);
        if (requestedBackendUnavailable)
        {
            QString message =
                QStringLiteral("请求的 %1 深度估计后端不可用或设备编号无效；显式后端不会自动切换到其他设备")
                    .arg(QString::fromLatin1(depthComputeBackendName(effectiveBackend)));
            if (!acceleratorPreparationFailures.isEmpty())
            {
                message += QStringLiteral("。%1").arg(acceleratorPreparationFailures.join(QStringLiteral("；")));
            }
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
            errorOccurred(message);
            emitFinishedOnce(false);
            return;
        }
        // A heterogeneous Auto batch retains the Auto token in the workspace hash;
        // single-family and explicit batches keep the resolved strict backend. This
        // prevents a CUDA-only resume from silently reusing a CUDA+OpenCL workset.
        _config.patchMatch.backend = heterogeneousAuto                                 ? PatchMatchBackend::Auto
                                     : effectiveBackend == DepthComputeBackend::Cuda   ? PatchMatchBackend::Cuda
                                     : effectiveBackend == DepthComputeBackend::OpenCl ? PatchMatchBackend::OpenCl
                                                                                       : PatchMatchBackend::Cpu;
        _config.patchMatch.cudaFallbackToCpu = false;
        _config.patchMatch.openClFallbackToCpu = false;

        const QString effective_backend_name = QString::fromLatin1(depthComputeBackendName(effectiveBackend));
        QString backend_message;
        if (configuredBackend == PatchMatchBackend::Auto)
        {
            backend_message = heterogeneousAuto
                                  ? QStringLiteral("深度估计后端：Auto 异构调度已启用 CUDA + OpenCL（逐帧收益调度）")
                                  : QStringLiteral("深度估计后端：Auto 已选择 %1").arg(effective_backend_name);
        }
        else
        {
            backend_message = QStringLiteral("深度估计后端：请求并使用 %1").arg(effective_backend_name);
        }
        if (configuredBackend == PatchMatchBackend::Auto && effectiveBackend == DepthComputeBackend::Cpu &&
            !acceleratorPreparationFailures.isEmpty())
        {
            backend_message += QStringLiteral("；CUDA 租约或 OpenCL 运行时预检不可用，已继续使用 CPU");
        }
        LOG_INFO(QStringLiteral("[MVS] %1").arg(backend_message));
        progressChanged(backend_message, 0.0f);

        progressChanged(QStringLiteral("读取 %1 张影像头部并规划内存...").arg(NV), 0.0f);
        QString imagePlanningError;
        if (!probeImageMetadata(&imagePlanningError))
        {
            if (!_cancelled.load())
            {
                LOG_ERROR(QStringLiteral("[MVS] %1").arg(imagePlanningError));
                errorOccurred(imagePlanningError);
            }
            emitFinishedOnce(false);
            return;
        }

        // The recovered producer is a closed scene operation: all d4 PatchMatch
        // pyramids are completed before its three-level voting stage starts.
        // Keep this boundary ahead of the former per-frame scheduler so an
        // unsupported request cannot silently execute the old depth algorithm.
        if (!cudaAvailable)
        {
            const QString error = QStringLiteral("recovered 多视深度需要可用 CUDA 设备；未执行旧算法回退");
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(error));
            errorOccurred(error);
            emitFinishedOnce(false);
            return;
        }
        if (_config.runFusion)
        {
            const QString error =
                QStringLiteral("scene-wide recovered 深度与融合必须分阶段执行；请先保存深度图，再调用现有流式融合阶段");
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(error));
            errorOccurred(error);
            emitFinishedOnce(false);
            return;
        }

        initializeWorkspaceManifest();
        std::fill(_skipFrameMask.begin(), _skipFrameMask.end(), 0);
        _depthFrames.assign(static_cast<std::size_t>(NV), DepthFrameResult{});
        const auto cuda_worker =
            std::find_if(physicalAcceleratorWorkers.begin(),
                         physicalAcceleratorWorkers.end(),
                         [](const DepthComputeWorker& worker) { return worker.backend == DepthComputeBackend::Cuda; });
        const int recovered_device_index =
            cuda_worker != physicalAcceleratorWorkers.end() ? cuda_worker->deviceIndex : 0;
        progressChanged(QStringLiteral("执行 recovered scene-wide CUDA PatchMatch 与三层投票..."), 0.05f);
        RecoveredDepthSceneResult recovered_result;
        std::string recovered_error;
        const std::string recovered_workspace_root =
            !_config.intermediateDir.empty() ? _config.intermediateDir : _outputDir;
        if (!runRecoveredDepthScene(_views,
                                    _sparse,
                                    recovered_device_index,
                                    recovered_workspace_root,
                                    _config.qualityProfile,
                                    &recovered_result,
                                    &recovered_error))
        {
            const QString error =
                QStringLiteral("recovered 多视深度失败：%1").arg(QString::fromStdString(recovered_error));
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(error));
            errorOccurred(error);
            emitFinishedOnce(false);
            return;
        }

        bool recovered_save_ok = true;
        std::vector<std::uint8_t> recovered_frame_seen(static_cast<std::size_t>(NV), 0);
        for (RecoveredDepthFrame& recovered_frame : recovered_result.frames)
        {
            if (_cancelled.load(std::memory_order_relaxed))
            {
                emitFinishedOnce(false);
                return;
            }
            const int frame_index = recovered_frame.viewIndex;
            if (frame_index < 0 || frame_index >= NV ||
                recovered_frame_seen[static_cast<std::size_t>(frame_index)] != 0)
            {
                recovered_save_ok = false;
                break;
            }
            recovered_frame_seen[static_cast<std::size_t>(frame_index)] = 1;
            DepthFrameResult frame;
            frame.refViewIdx = frame_index;
            frame.preparedRasterSize = cv::Size(_views[static_cast<std::size_t>(frame_index)].imageWidth,
                                                _views[static_cast<std::size_t>(frame_index)].imageHeight);
            frame.effectiveNativeFinalDepthGrid = true;
            frame.cameraModel = recovered_frame.camera;
            frame.sourceViewIndices = std::move(recovered_frame.sourceViewIndices);
            frame.requestedSourceViewCount = 16;
            frame.sourceViewShortfall =
                std::max(0, frame.requestedSourceViewCount - static_cast<int>(frame.sourceViewIndices.size()));
            frame.depthMap = QSharedPointer<cv::Mat>::create(std::move(recovered_frame.depth));
            frame.confidence = QSharedPointer<cv::Mat>::create(std::move(recovered_frame.confidence));
            frame.validMask = QSharedPointer<cv::Mat>::create(std::move(recovered_frame.validMask));
            frame.photometricSourceMask =
                QSharedPointer<cv::Mat>::create(std::move(recovered_frame.photometricSourceMask));
            frame.geometrySupportCount =
                QSharedPointer<cv::Mat>::create(std::move(recovered_frame.geometrySupportCount));
            frame.inverseDepthRelativeSpread =
                QSharedPointer<cv::Mat>::create(std::move(recovered_frame.inverseDepthRelativeSpread));
            frame.supportRegionMask = QSharedPointer<cv::Mat>::create(std::move(recovered_frame.supportRegionMask));
            frame.qualityMetrics = analyzeDepthMapQuality(
                *frame.depthMap, *frame.confidence, static_cast<int>(frame.sourceViewIndices.size()));
            frame.depthCompleteness.finalMetrics = analyzeDepthCompleteness(*frame.depthMap, *frame.supportRegionMask);
            const int valid_count = cv::countNonZero(*frame.validMask);
            frame.depthCompleteness.pyramidValidCount = valid_count;
            frame.depthCompleteness.afterMaskValidCount = valid_count;
            frame.depthCompleteness.afterSparseSupportValidCount = valid_count;
            frame.depthCompleteness.preOutputFilterValidCount = valid_count;
            frame.depthCompleteness.postOutputFilterValidCount = valid_count;
            frame.depthCompleteness.outputFilterRetentionRatio = 1.0f;
            frame.depthPostprocess.validBeforePostprocess = valid_count;
            frame.depthPostprocess.validAfterPostprocess = valid_count;
            frame.pixelDomainDiagnostics = makePixelDomainDiagnostics(
                frame.preparedRasterSize, frame.depthMap->size(), _config.fusion, true, true);
            frame.pixelDomainDiagnostics.insert(QStringLiteral("producer"), QStringLiteral("recovered_scene_d4"));
            frame.pixelDomainDiagnostics.insert(QStringLiteral("confidence_semantics"),
                                                QStringLiteral("binary_valid_after_three_level_voting"));
            frame.pixelDomainDiagnostics.insert(QStringLiteral("source_selection"),
                                                QStringLiteral("sfm_track_ranked_1_to_16"));
            // The reference implementation's three-level voting chain is the
            // complete recovered quality filter.  Do not opt this frame into
            // PlaScan's legacy consistency/quality gate, which can rewrite the
            // voted depth and downgrade an otherwise valid reference camera.
            frame.qualityDecision.acceptance = DepthFrameAcceptance::Accepted;
            frame.qualityDecision.reasons.emplace_back("adaptive_geometry_fallback_to_discrete_core");
            frame.qualityDecision.reasons.emplace_back("recovered_voting_geometry_proxy");
            frame.initialQualityAcceptanceAvailable = false;
            frame.initialQualityAcceptance = DepthFrameAcceptance::Accepted;
            frame.maskSource = std::move(recovered_frame.maskSource);
            frame.maskCoverage = recovered_frame.maskCoverage;
            frame.selectedLevel = 1;
            frame.pyramidRequestedLevelCount = 3;
            frame.pyramidActiveLevelCount = 3;
            frame.effectivePatchMatchConfidenceThreshold = 0.0f;
            frame.depthPostprocessApplied = true;
            frame.success = true;
            frame.elapsedMs =
                (recovered_result.patchMatchSeconds + recovered_result.votingSeconds) * 1000.0 / std::max(1, NV);
            frame.device = "CUDA:" + std::to_string(recovered_device_index) + ":recovered_scene_d4";
            _depthFrames[static_cast<std::size_t>(frame_index)] = std::move(frame);
            markManifestFrameRunning(frame_index);
            if (!saveDepthFrameArtifacts(frame_index,
                                         _depthFrames[static_cast<std::size_t>(frame_index)],
                                         QStringLiteral("recovered三层投票")))
            {
                recovered_save_ok = false;
                break;
            }
            depthMapReady(_depthFrames[static_cast<std::size_t>(frame_index)]);
            progressChanged(QStringLiteral("recovered 深度图：帧 %1/%2").arg(frame_index + 1).arg(NV),
                            0.1f + 0.85f * static_cast<float>(frame_index + 1) / static_cast<float>(std::max(1, NV)));
        }
        recovered_save_ok = recovered_save_ok && std::all_of(recovered_frame_seen.cbegin(),
                                                             recovered_frame_seen.cend(),
                                                             [](std::uint8_t seen) { return seen != 0; });
        if (!recovered_save_ok)
        {
            errorOccurred(QStringLiteral("recovered 深度图工件写入失败"));
            emitFinishedOnce(false);
            return;
        }
        progressChanged(QStringLiteral("完成"), 1.0f);
        emitFinishedOnce(true);
        return;
    }
} // namespace xjw::mvs
