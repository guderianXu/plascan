import re
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MvsSchedulerConfigTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_depth_generation_exposes_frame_worker_controls(self):
        types = self.read("src/core/mvs/MvsTypes.h")
        self.assertIn("gpuFrameWorkerCount", types)
        self.assertIn("cpuFrameWorkerCount", types)
        self.assertIn("cudaUseParallelSweep", types)

    def test_gui_dense_settings_forward_frame_worker_controls(self):
        config_h = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.h")
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        self.assertIn("gpuFrameWorkers", config_h)
        self.assertIn("cpuFrameWorkers", config_h)
        self.assertIn("gpu_frame_workers", config_cpp)
        self.assertIn("cpu_frame_workers", config_cpp)
        self.assertIn("config.gpuFrameWorkerCount", config_cpp)
        self.assertIn("config.cpuFrameWorkerCount", config_cpp)

    def test_depth_scheduler_no_longer_hardcodes_single_cuda_worker(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        self.assertNotIn("const int cpuWorkers = cudaAvailable ? 0 : 1;", scheduler)
        self.assertIn("gpuFrameWorkers", scheduler)
        self.assertIn("workerIndex < gpuFrameWorkers", scheduler)
        self.assertIn("cpuFrameWorkers", scheduler)

    def test_depth_scheduler_saves_depth_artifacts_asynchronously(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("DepthFrameArtifactSaveQueue", scheduler)
        self.assertIn("saveQueue.enqueue(i, res, QStringLiteral(\"初始\"))", scheduler)
        self.assertIn("saveQueue.waitUntilIdle()", scheduler)
        self.assertIn("saveQueue.stop()", scheduler)
        self.assertNotIn("if (!saveDepthFrameArtifacts(i, res, QStringLiteral(\"初始\")))", scheduler)

    def test_depth_only_scheduler_releases_saved_full_resolution_frames(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("void releasePixelStorage()", header)
        self.assertIn("shouldRetainAllDepthFramesInMemory", scheduler)
        self.assertIn("std::atomic<bool> keepDepthFramesInMemory", scheduler)
        self.assertIn("DepthFrameResult storedResult = res;", scheduler)
        self.assertIn("if (!keepDepthFramesInMemory.load())", scheduler)
        self.assertIn("storedResult.releasePixelStorage();", scheduler)
        self.assertIn("_depthFrames[i] = storedResult;", scheduler)
        self.assertNotIn("_depthFrames[i] = res;", scheduler)
        self.assertIn("if (keepDepthFramesInMemory.load() && NV >= 2)", scheduler)

    def test_depth_cache_retention_is_budgeted_from_system_memory(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("SystemMemorySnapshot", scheduler)
        self.assertIn("querySystemMemorySnapshot", scheduler)
        self.assertIn("estimateDepthFrameCacheBytes", scheduler)
        self.assertIn("retainedDepthMemoryBudgetBytes", scheduler)
        self.assertIn("_config.maxDepthCacheRamFraction", scheduler)
        self.assertIn("_config.minFreeRamBytes", scheduler)
        self.assertIn("深度图内存策略", scheduler)

    def test_depth_cache_unknown_dimensions_use_streaming_not_cache(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("refreshViewImageDimensionsFromCache", scheduler)
        self.assertIn("refreshViewImageDimensionsFromCache();", scheduler)
        self.assertIn("无有效影像尺寸，采用保守流式模式", scheduler)
        self.assertNotIn('QStringLiteral("无有效影像尺寸");\n        }\n        return true;', scheduler)

    def test_gui_mvs_views_populate_image_dimensions_before_memory_policy(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("applyImageSizeToMvsView", manager)
        self.assertGreaterEqual(manager.count("applyImageSizeToMvsView(imgPath, &view)"), 2)
        self.assertNotIn("cv::imread(imagePath.toStdString()", manager)
        self.assertNotIn("view.imageWidth = 0;", manager)
        self.assertNotIn("view.imageHeight = 0;", manager)

    def test_stored_depth_batch_load_filters_confidence_in_place_and_drops_confidence(self):
        helper = self.read("src/core/mvs/DepthFrameUtils.cpp")

        self.assertIn("const bool shouldLoadConfidence", helper)
        self.assertIn("result.frame.confidence.release();", helper)
        self.assertNotIn("cv::Mat filteredDepth = result.frame.depthMap.clone();", helper)

    def test_fusion_frame_builder_drops_confidence_after_postprocess(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index("FusionFrameInput DepthMapGenerator::buildFusionFrame")
        end = scheduler.index("void DepthMapGenerator::crossCheckDepthConsistency")
        body = scheduler[start:end]

        self.assertIn("frame.confidence.release();", body)

    def test_gui_depth_batch_loading_reports_incremental_progress(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        fuse_start = manager.index("void ProjectDenseReconstructionManager::startFuseDepthMapsAsync")
        refine_start = manager.index("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync")
        fuse_body = manager[fuse_start:refine_start]

        self.assertIn("正在加载深度图 %1/%2", fuse_body)
        self.assertIn("流式深度图融合 %1/%2", fuse_body)
        self.assertIn("mvsProgressChanged", fuse_body)

    def test_depth_fusion_uses_lazy_color_lru_cache(self):
        header = self.read("src/core/mvs/DepthMapFusion.h")
        fusion = self.read("src/core/mvs/DepthMapFusion.cpp")

        self.assertIn("bool  useColor", header)
        self.assertIn("int   colorCacheCapacity", header)
        self.assertIn("class ColorImageCache", fusion)
        self.assertIn("ColorImageCache colorCache", fusion)
        self.assertIn("colorCache.get", fusion)
        self.assertNotIn("std::vector<cv::Mat> colorImages(NF)", fusion)
        self.assertNotIn("colorImages[fi] = cv::imread", fusion)

    def test_sfm_sparse_export_batches_color_sampling_by_image(self):
        service = self.read("src/core/aerial_triangulation/AerialTriangulationService.cpp")

        self.assertIn("struct SparseExportColorRequest", service)
        self.assertIn("sampleSparseExportColorsByImage", service)
        self.assertIn("colorRequestsByImage", service)
        self.assertIn("std::vector<unsigned char> colorFilled", service)
        self.assertNotIn("SparseExportColorCache", service)
        self.assertNotIn("QMap<ImageId, cv::Mat> imgColorCache", service)
        self.assertNotIn("colorCache.get", service)

    def test_depth_fusion_can_limit_fusion_to_streaming_reference_frame(self):
        header = self.read("src/core/mvs/DepthMapFusion.h")
        fusion = self.read("src/core/mvs/DepthMapFusion.cpp")

        self.assertIn("bool  fuseOnlyFirstFrame", header)
        self.assertIn("const int fusionStartFrame", fusion)
        self.assertIn("const int fusionEndFrame", fusion)

    def test_gui_depth_fusion_streams_neighbor_windows_with_lru_depth_cache(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("class DepthFrameLruCache", manager)
        self.assertIn("nearestFusionWindowIndices", manager)
        self.assertIn("streamFusionWindowSize", manager)
        self.assertIn("depthFrameCache.get", manager)
        self.assertIn("fusionCfg.fuseOnlyFirstFrame = true", manager)
        self.assertIn("流式深度图融合", manager)
        self.assertNotIn("frames.reserve(storedFrames.size());\n        for (const auto &stored : storedFrames)", manager)

    def test_depth_scheduler_switches_to_streaming_when_memory_pressure_rises(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("memoryPressureRequiresStreaming", scheduler)
        self.assertIn("keepDepthFramesInMemory.compare_exchange_strong", scheduler)
        self.assertIn("releaseStoredDepthFramePixelStorage", scheduler)
        self.assertIn("内存压力升高，切换为流式保存", scheduler)
        self.assertIn("无法继续本次内存融合", scheduler)

    def test_depth_artifact_save_queue_bounds_full_resolution_backlog(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        queue_start = scheduler.index("class DepthFrameArtifactSaveQueue")
        queue_end = scheduler.index("// =============================================================================", queue_start)
        queue_block = scheduler[queue_start:queue_end]

        self.assertIn("maxBufferedTasks", queue_block)
        self.assertIn("m_capacityCv.wait", queue_block)
        self.assertIn("m_tasks.size() < m_maxBufferedTasks", queue_block)
        self.assertIn("m_capacityCv.notify_one()", queue_block)

    def test_depth_cancel_drains_pending_save_queue_before_waiting_for_idle(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        queue_start = scheduler.index("class DepthFrameArtifactSaveQueue")
        queue_end = scheduler.index("// =============================================================================", queue_start)
        queue_block = scheduler[queue_start:queue_end]

        self.assertIn("void cancel()", queue_block)
        self.assertIn("m_dropPendingTasks", queue_block)
        self.assertIn("m_tasks.clear()", queue_block)

        join_pos = scheduler.index("for (std::thread &worker : workers)")
        cleanup_pos = scheduler.index("// 释放图像缓存", join_pos)
        post_worker_block = scheduler[join_pos:cleanup_pos]
        self.assertIn("if (_cancelled.load())", post_worker_block)
        self.assertLess(post_worker_block.index("if (_cancelled.load())"),
                        post_worker_block.index("saveQueue.waitUntilIdle()"))
        self.assertIn("saveQueue.cancel()", post_worker_block)

    def test_depth_postprocess_stages_poll_cancel_before_more_work(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        cross_start = scheduler.index("void DepthMapGenerator::crossCheckDepthConsistency()")
        cross_end = scheduler.index("bool DepthMapGenerator::saveDepthFrameArtifacts", cross_start)
        cross_block = scheduler[cross_start:cross_end]

        self.assertIn("if (_cancelled.load())", cross_block)

        filtered_save_pos = scheduler.index("saveQueue.enqueue(i, res, QStringLiteral(\"过滤后\"))")
        filtered_block = scheduler[filtered_save_pos - 400:filtered_save_pos + 200]
        self.assertIn("if (_cancelled.load())", filtered_block)

    def test_dense_sparse_preload_respects_cancel_before_starting_mvs(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertGreaterEqual(manager.count("QPointer<DepthMapGenerator> genSelf(gen)"), 2)
        self.assertGreaterEqual(
            manager.count("const QString projectPath = _owner ? _owner->currentProjectPath() : QString()"),
            2,
        )
        self.assertGreaterEqual(
            manager.count("QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]()"),
            2,
        )
        self.assertGreaterEqual(manager.count("self->_owner->currentProjectPath() != projectPath"), 2)
        self.assertGreaterEqual(manager.count("if (genSelf->isCancelled())"), 2)
        self.assertGreaterEqual(manager.count("return;"), 2)
        self.assertGreaterEqual(manager.count('QMetaObject::invokeMethod(genSelf.data(), "finished"'), 2)
        self.assertGreaterEqual(
            manager.count("QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]()"),
            2,
        )
        self.assertIn("Q_ARG(bool, false)", manager)
        self.assertNotIn("QtConcurrent::run([gen, sparseXyz, views, request]()", manager)
        self.assertNotIn("QtConcurrent::run([genSelf, sparseXyz, views, request]()", manager)
        self.assertNotIn("gen->setSparseCloud(sparse)", manager)
        self.assertNotIn('QMetaObject::invokeMethod(gen, "start"', manager)

    def test_depth_artifact_saving_uses_fast_binary_and_timing_logs(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("writeFastDepthMatStorage", scheduler)
        self.assertIn("saveDepthPreviewPng", scheduler)
        self.assertIn("maxPreviewDimension", scheduler)
        self.assertIn("保存%1深度产物耗时", scheduler)
        self.assertNotIn("FileStorage storage(path, cv::FileStorage::WRITE)", scheduler)
        self.assertNotIn(".yml.gz", scheduler)

    def test_manual_depth_estimation_can_auto_pipeline_two_cuda_frames(self):
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")

        self.assertIn("autoGpuFrameWorkers", config_cpp)
        self.assertIn("settings.gpuFrameWorkers", config_cpp)
        self.assertNotIn("return 1;", config_cpp)
        self.assertIn("return std::clamp", config_cpp)

    def test_preload_images_runs_with_bounded_parallel_workers(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("preloadImagesWorkerCount", scheduler)
        self.assertIn("std::atomic<int> nextImage", scheduler)
        self.assertIn("preloadWorkers.emplace_back", scheduler)
        self.assertIn("preloadImages(): workers=", scheduler)

    def test_depth_scheduler_precomputes_mvs_visibility_and_source_views(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("FrameMvsCache", header)
        self.assertIn("prepareFrameCaches", header)
        self.assertIn("_visibilityBits", header)
        self.assertIn("_pairCommonCounts", header)
        self.assertIn("prepareFrameCaches();", scheduler)
        self.assertIn("sourceViewIndicesForFrame", scheduler)
        self.assertIn("visibleSparsePointIndicesForFrame", scheduler)
        self.assertNotIn("selectMvsSourceViewIndices(_views, _sparse, refIdx, numSrc)", scheduler)

    def test_depth_scheduler_caches_selected_source_shared_sparse_points(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        visible_start = scheduler.index("std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame")
        visible_end = scheduler.index("// =============================================================================", visible_start)
        visible_block = scheduler[visible_start:visible_end]

        self.assertIn("sourceSharedPointIndices", header)
        self.assertIn("sourceSharedPointIndices.reserve", scheduler)
        self.assertIn("sourceSharedPointIndices.push_back", scheduler)
        self.assertIn("sourceIndicesMatchCachedPrefix", scheduler)
        self.assertIn("return cache.sourceSharedPointIndices;", visible_block)
        self.assertLess(visible_block.index("return cache.sourceSharedPointIndices;"),
                        visible_block.index("std::vector<size_t> filtered;"))

    def test_frame_cache_visibility_scan_uses_parallel_shards(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index("void DepthMapGenerator::prepareFrameCaches()")
        end = scheduler.index("std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame", start)
        block = scheduler[start:end]

        self.assertIn("VisibilityCacheShard", block)
        self.assertIn("visibilityWorkerCount", block)
        self.assertIn("#pragma omp parallel", block)
        self.assertIn("#pragma omp for", block)
        self.assertIn("shard.visiblePointIndicesByView", block)
        self.assertIn("shard.pairCommonCounts", block)
        self.assertIn("mergeVisibilityCacheShards", block)
        self.assertIn("buildVisibilityBitsFromFrameCaches", block)
        self.assertNotIn("_frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices.push_back(pointIndex);",
                         block)

    def test_source_view_scoring_short_circuits_angle_sampling_when_top_sources_are_proven(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index("void DepthMapGenerator::prepareFrameCaches()")
        end = scheduler.index("std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame", start)
        block = scheduler[start:end]

        self.assertIn("rankedSourceCandidates", block)
        self.assertIn("desiredSourceCount", block)
        self.assertIn("currentSourceScoreCutoff", block)
        self.assertIn("remaining candidates are sorted by common count", block)
        self.assertIn("candidate.commonVisiblePoints <= currentSourceScoreCutoff", block)
        self.assertLess(block.index("candidate.commonVisiblePoints <= currentSourceScoreCutoff"),
                        block.index("sampledMedianAngle(refIdx, candidate.viewIndex)"))

    def test_depth_frame_reuses_visible_sparse_points_for_range_hint_and_support(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("estimateDepthRangeFromVisiblePoints", scheduler)
        self.assertIn("buildHintDepthFromVisiblePoints", scheduler)
        self.assertIn("buildSparseSupportMaskFromVisiblePoints", scheduler)
        self.assertIn("const std::vector<size_t> visibleSparsePointIndices", scheduler)
        self.assertIn("visibleSparsePointIndices)", scheduler)

    def test_depth_frame_builds_sparse_hints_at_patchmatch_work_resolution(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("patchMatchWorkSize", scheduler)
        self.assertIn("collectProjectedSparseDepthSamples", scheduler)
        self.assertIn("buildHintDepthFromProjectedSamples", scheduler)
        self.assertIn("const cv::Size coarseHintSize = patchMatchWorkSize(workRefImg, coarseCfg);", scheduler)
        self.assertIn("const cv::Size fineHintSize = patchMatchWorkSize(workRefImg, fineCfg);", scheduler)
        self.assertIn("coarseHintSize.width", scheduler)
        self.assertIn("coarseHintSize.height", scheduler)
        self.assertIn("fineHintSize.width", scheduler)
        self.assertIn("fineHintSize.height", scheduler)
        self.assertNotIn("cv::Mat hintDepth = buildHintDepthFromVisiblePoints(refIdx, W, H, visibleSparsePointIndices);",
                         scheduler)

    def test_depth_frame_reuses_projected_sparse_samples_for_hint_and_support(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("ProjectedSparseDepthSample", header)
        self.assertIn("collectProjectedSparseDepthSamples", scheduler)
        self.assertIn("workRefSparseSamples", scheduler)
        self.assertIn("buildHintDepthFromProjectedSamples(refIdx", scheduler)
        self.assertIn("buildSparseSupportMaskFromProjectedSamples(refIdx", scheduler)
        self.assertNotIn("buildHintDepthForCamera(refIdx,\n                                                 coarseHintCam", scheduler)
        self.assertNotIn("buildHintDepthForCamera(refIdx,\n                                                         fineHintCam", scheduler)

    def test_sparse_support_mask_uses_fine_patchmatch_work_size(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index("const std::vector<ProjectedSparseDepthSample> workRefSparseSamples")
        end = scheduler.index("timing.hintMs = elapsedMs", start)
        block = scheduler[start:end]

        self.assertIn("supportMaskCfg", block)
        self.assertIn("makeFinePatchMatchConfig(pmCfg, useRectified, 0.0f)", block)
        self.assertIn("const cv::Size supportMaskSize = patchMatchWorkSize(refImg, supportMaskCfg);", block)
        self.assertNotIn("patchMatchWorkSize(refImg, pmCfg)", block)

    def test_projected_sparse_sample_depth_iqr_uses_bounded_quantile_sampling(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index(
            "std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples")
        end = scheduler.index("cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples", start)
        block = scheduler[start:end]

        self.assertIn("kMaxProjectedDepthQuantileSamples", scheduler)
        self.assertIn("depthQuantileSamples", block)
        self.assertIn("std::nth_element", block)
        self.assertNotIn("std::sort(allZc", block)
        self.assertNotIn("allZc.reserve(visiblePointIndices.size())", block)

    def test_projected_sparse_samples_are_collected_in_single_visible_point_pass(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        start = scheduler.index(
            "std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples")
        end = scheduler.index("cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples", start)
        block = scheduler[start:end]

        self.assertEqual(block.count("for (size_t pointIndex : visiblePointIndices)"), 1)
        self.assertIn("projectedCandidates", block)
        self.assertIn("cam.projectWithDepth", block)
        self.assertIn("candidate.depth", block)
        self.assertNotIn("float Zc = cam.R_cw[6]*pt[0]", block)
        self.assertNotIn("cam.project(pt[0]", block)

    def test_positive_depth_camera_can_project_and_return_depth_once(self):
        header = self.read("src/core/camera/PositiveDepthCameraModel.h")
        source = self.read("src/core/camera/PositiveDepthCameraModel.cpp")

        self.assertIn("projectWithDepth", header)
        self.assertIn("PositiveDepthCameraModel::projectWithDepth", source)
        self.assertIn("return projectWithDepth", source)

    def test_fine_sparse_hint_uses_seed_overlay_without_full_propagation(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("buildSparseSeedDepthFromProjectedSamples", header)
        self.assertIn("buildSparseSeedDepthFromProjectedSamples", scheduler)

        fine_start = scheduler.index("cv::resize(coarseDepth, fineHint")
        fine_end = scheduler.index("const int hintValid", fine_start)
        fine_block = scheduler[fine_start:fine_end]

        self.assertIn("fineSparseSeedHint", fine_block)
        self.assertIn("fineSparseSeedHint.copyTo(fineHint", fine_block)
        self.assertNotIn("buildHintDepthFromProjectedSamples(refIdx", fine_block)

    def test_coarse_sparse_hint_uses_distance_transform_propagation(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        hint_start = scheduler.index("cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples")
        hint_end = scheduler.index("cv::Mat DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples")
        hint_body = scheduler[hint_start:hint_end]

        self.assertIn("cv::distanceTransform", hint_body)
        self.assertIn("DIST_LABEL_PIXEL", hint_body)
        self.assertIn("maxHintRadius", hint_body)
        self.assertNotIn("cv::Mat distMap", hint_body)
        self.assertNotIn("INT_MAX / 2", hint_body)

    def test_patchmatch_accepts_prescaled_hint_without_extra_resize(self):
        cuda = self.read("src/core/mvs/PatchMatchCUDA.cu")

        self.assertIn("hintDepth->cols == sW && hintDepth->rows == sH", cuda)
        self.assertIn("hintDepth->cols == W && hintDepth->rows == H", cuda)
        self.assertIn("hintScaled = *hintDepth", cuda)

    def test_patchmatch_gpu_avoids_duplicate_reference_resize_before_upload(self):
        cuda = self.read("src/core/mvs/PatchMatchCUDA.cu")
        gpu_start = cuda.index("bool PatchMatchDepthEstimator::estimateGPU")
        cpu_start = cuda.index("bool PatchMatchDepthEstimator::estimateCPU")
        gpu_body = cuda[gpu_start:cpu_start]

        self.assertIn("const int sW = std::max(1, refW / ds);", gpu_body)
        self.assertIn("const int sH = std::max(1, refH / ds);", gpu_body)
        self.assertIn("getOrUploadGrayImageGpu(refGray, sW, sH, ds", gpu_body)
        self.assertNotIn("cv::resize(refGray, refScaled", gpu_body)

    def test_sparse_hint_skips_propagation_when_no_seed_pixels(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("if (seedHintCnt <= 0)", scheduler)
        self.assertIn("没有可用 hint seed", scheduler)

    def test_depth_frame_logs_stage_timings(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("FrameTiming", scheduler)
        self.assertIn("耗时统计", scheduler)
        self.assertIn("source=", scheduler)
        self.assertIn("patchmatch=", scheduler)
        self.assertIn("filter=", scheduler)

    def test_sparse_support_is_soft_prior_not_hard_depth_clip(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("applySparseSupportPrior", header)
        self.assertIn("applySparseSupportPrior(depthMap, confMap, supportMask, refIdx)", scheduler)
        self.assertNotIn("depthMap.setTo(0, supportMask == 0)", scheduler)
        self.assertNotIn("confMap.setTo(0, supportMask == 0)", scheduler)
        self.assertIn("稀疏支撑软约束", scheduler)

    def test_fusion_frame_applies_local_depth_outlier_filter(self):
        mvs_types = self.read("src/core/mvs/MvsTypes.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("removeLocalDepthOutliers", scheduler)
        self.assertIn("postprocessFusionDepthMap(filteredDepth", scheduler)
        self.assertIn("DepthPostProcessStats", mvs_types)
        self.assertIn("enableLocalDepthOutlierFilter", mvs_types)
        self.assertIn("localDepthOutlierRelThresh", mvs_types)
        self.assertIn("maxLocalDepthOutlierRemovalRatio", mvs_types)

    def test_cli_depth_reuse_applies_same_postprocess_and_reports_stats(self):
        mvs_types = self.read("src/core/mvs/MvsTypes.h")
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")

        self.assertIn("DepthPostProcessStats", mvs_types)
        self.assertIn("postprocessFusionDepthMap", scheduler)
        self.assertIn("postprocessFusionDepthMap", cli)
        self.assertIn("depth_postprocess", cli)
        self.assertIn("local_depth_outlier_removed", cli)
        self.assertIn("speckle_removed", cli)
        self.assertIn("edge_confidence_removed", cli)
        self.assertIn("geom_consistency_removed", cli)
        self.assertIn("rawDepthStoragePath(depthPngPath)", cli)
        self.assertIn("loadDepthMatStorage", cli)

    def test_depth_generator_skips_internal_fusion_postprocess_when_fusion_disabled(self):
        scheduler = self.read("src/core/mvs/DepthMapGenerator.cpp")

        postprocess_guard = re.search(
            r"if\s*\(\s*_config\.runFusion\s*&&\s*keepDepthFramesInMemory\.load\(\)\s*&&"
            r"\s*\(\s*savePreviewPng\s*\|\|\s*saveRawDepth\s*\)\s*\)",
            scheduler,
            re.S,
        )
        self.assertIsNotNone(postprocess_guard)

    def test_reconstruct_pipeline_cli_exposes_mvs_regression_controls(self):
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")

        self.assertIn("--mvs-res-scale", cli)
        self.assertIn("--mvs-iterations", cli)
        self.assertIn("--mvs-confidence", cli)
        self.assertIn("--mvs-gpu-frame-workers", cli)
        self.assertIn("--mvs-cpu-frame-workers", cli)
        self.assertIn("--mvs-max-frames", cli)
        self.assertIn("denseSettings.resScale = mvsResScale;", cli)
        self.assertIn("denseSettings.iterations = mvsIterations;", cli)
        self.assertIn("denseSettings.patchMatchConfidence = mvsConfidence;", cli)
        self.assertIn("denseSettings.fusionMinConfidence = mvsFusionConfidence;", cli)
        self.assertIn("denseSettings.gpuFrameWorkers = mvsGpuFrameWorkers;", cli)
        self.assertIn("denseSettings.cpuFrameWorkers = mvsCpuFrameWorkers;", cli)
        self.assertIn("limitMvsInputsForRegression", cli)

    def test_reconstruct_pipeline_cli_records_depth_artifacts_and_mvs_settings(self):
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")

        self.assertIn("QJsonArray depthArtifacts;", cli)
        self.assertIn("&xjw::mvs::DepthMapGenerator::depthMapArtifactSaved", cli)
        self.assertIn("depthArtifacts.append(artifact)", cli)
        self.assertIn('denseReport[QStringLiteral("depth_maps")] = depthArtifacts;', cli)
        self.assertIn('denseReport[QStringLiteral("mvs_settings")]', cli)
        self.assertIn('denseReport[QStringLiteral("mvs_depth_config")]', cli)

    def test_reconstruct_pipeline_cli_can_stop_after_mvs_depth_maps(self):
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")

        self.assertIn("bool mvsDepthOnly = false;", cli)
        self.assertIn("--mvs-depth-only", cli)
        self.assertIn('denseReport[QStringLiteral("status")] = QStringLiteral("depth_only");', cli)
        self.assertIn('report[QStringLiteral("stop_stage")] = QStringLiteral("mvs_depth");', cli)
        self.assertIn('markSkippedStage(QStringLiteral("mvs_fusion"), depthOnlyReason);', cli)
        self.assertIn('markSkippedStage(QStringLiteral("mesh"), depthOnlyReason);', cli)
        self.assertIn('markSkippedStage(QStringLiteral("terrain"), depthOnlyReason);', cli)
        self.assertIn("const int depthMapCount = static_cast<int>(depthArtifacts.size());", cli)
        self.assertIn('std::fprintf(stdout, "depth_maps=%d\\n", depthMapCount);', cli)

    def test_reconstruct_pipeline_cli_streams_depth_fusion_windows(self):
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")

        self.assertIn("streamingFusionWindowIndices", cli)
        self.assertIn("fuseDepthMapsStreamingFromDisk", cli)
        self.assertIn("fusionCfg.fuseOnlyFirstFrame = true", cli)
        self.assertIn("流式深度图融合", cli)
        self.assertIn("voxelDownsampleFusedPointsToTarget", cli)
        self.assertIn("kStreamingFusionCacheFrameLimit", cli)
        self.assertIn("const bool useCachedFrames", cli)
        self.assertIn("if (useCachedFrames)", cli)

    def test_reconstruct_pipeline_cli_downsamples_fusion_frames(self):
        cli = self.read("src/cli/cli_reconstruct_pipeline.cpp")
        helper = self.read("src/core/mvs/DepthFrameUtils.cpp")

        self.assertIn("--mvs-fusion-max-image-dim", cli)
        self.assertIn("mvsFusionMaxImageDim", cli)
        self.assertIn("scalePositiveDepthCameraModel", helper)
        self.assertIn("downsampleFusionFrameForMaxDimension", helper)
        self.assertIn("loadFusionFrameFromDepthMap(", cli)
        self.assertIn("fusionMaxImageDim", cli)
        self.assertIn('settings[QStringLiteral("fusion_max_image_dim")]', cli)
        self.assertIn("cv::resize(frame->depthMap", helper)
        self.assertIn("cv::resize(frame->confidence", helper)

    def test_depth_fusion_resizes_color_cache_to_frame_grid(self):
        fusion = self.read("src/core/mvs/DepthMapFusion.cpp")

        self.assertIn("m_frames[frameIdx].imgW", fusion)
        self.assertIn("m_frames[frameIdx].imgH", fusion)
        self.assertIn("cv::resize(image, image", fusion)

    def test_streaming_depth_fusion_uses_prefiltered_fast_path(self):
        header = self.read("src/core/mvs/DepthMapFusion.h")
        fusion = self.read("src/core/mvs/DepthMapFusion.cpp")

        self.assertIn("fuseFirstFrameObservationsFast", header)
        self.assertIn("fuseFirstFrameObservationsFast", fusion)
        self.assertIn("使用已过滤深度图快速反投影", fusion)
        self.assertIn("_config.fuseOnlyFirstFrame", fusion)
        self.assertIn("resolveFusionWorkerCount", fusion)

    def test_gui_streaming_depth_fusion_downsamples_stored_frames(self):
        header = self.read("src/core/mvs/DepthFrameUtils.h")
        helper = self.read("src/core/mvs/DepthFrameUtils.cpp")
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("int fusionMaxImageDim", header)
        self.assertIn("scalePositiveDepthCameraModel", helper)
        self.assertIn("downsampleFusionFrameForMaxDimension", helper)
        self.assertIn("fusionMaxImageDim", manager)
        self.assertIn("_fusionMaxImageDim", manager)
        self.assertIn("request.fusionMaxImageDim", manager)
        self.assertIn("_fusionMaxImageDim)", manager)

    def test_cuda_scheduler_defaults_can_pipeline_two_frame_workers(self):
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        self.assertIn("autoGpuFrameWorkers", config_cpp)
        self.assertIn("const int desired = threads >= 8 ? 2 : 1;", config_cpp)
        self.assertIn("return std::clamp(desired, 1, maxGpuWorkers);", config_cpp)
        self.assertNotIn("Q_UNUSED(threads)", config_cpp)

    def test_two_view_dense_config_does_not_force_full_resolution_or_32_iterations(self):
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        self.assertNotIn("config.patchMatch.downsampleFactor = 1;", config_cpp)
        self.assertNotIn("std::max(config.patchMatch.numIterations, 32)", config_cpp)
        self.assertIn("两视图立体对：允许融合但不强制全分辨率", config_cpp)

    def test_multiscale_patchmatch_uses_requested_iterations(self):
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")
        self.assertNotIn("coarseCfg.numIterations    = 12", generator)
        self.assertNotIn("fineCfg.numIterations = useRectified ? 6 : 8", generator)
        self.assertIn("makeCoarsePatchMatchConfig", generator)
        self.assertIn("makeFinePatchMatchConfig", generator)

    def test_cuda_parallel_sweep_path_is_available(self):
        cuda = self.read("src/core/mvs/PatchMatchCUDA.cu")
        self.assertIn("kernelCheckerboardSweep", cuda)
        self.assertIn("cudaUseParallelSweep", cuda)
        self.assertIn("kernelSweepTB", cuda)

    def test_depth_consistency_uses_patchmatch_source_views(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("sourceViewIndices", header)
        self.assertIn("std::vector<int> sourceIndices", generator)
        self.assertIn("sourceIndices.push_back(si)", generator)
        self.assertIn("result.sourceViewIndices = sourceIndices", generator)
        self.assertIn("consistencySourceIndicesForFrame", generator)
        self.assertIn("const std::vector<int> consistencySources", generator)

    def test_depth_preview_saved_after_consistency_and_checks_write_result(self):
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("saveDepthPreviewPng", generator)
        self.assertIn("if (!cv::imwrite(path, colorVis))", generator)
        consistency_pos = generator.index("crossCheckDepthConsistency();")
        save_pos = generator.index("saveQueue.enqueue(i, res, QStringLiteral(\"过滤后\"))")
        signal_pos = generator.index("emit depthMapSaved")
        self.assertGreater(save_pos, consistency_pos)
        self.assertIn("saveDepthFrameArtifacts", generator)
        self.assertGreater(signal_pos, 0)

    def test_depth_artifact_metadata_records_sources_device_and_mask(self):
        header = self.read("src/core/mvs/DepthMapGenerator.h")
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        tree = self.read("src/gui/widgets/DataTreeWidget.cpp")

        self.assertIn("void depthMapArtifactSaved(QJsonObject artifact)", header)
        self.assertIn('artifact[QStringLiteral("source_images")]', generator)
        self.assertIn('artifact[QStringLiteral("source_indices")]', generator)
        self.assertIn('artifact[QStringLiteral("valid_mask_path")]', generator)
        self.assertIn('artifact[QStringLiteral("device")]', generator)
        self.assertIn('artifact[QStringLiteral("elapsed_ms")]', generator)
        self.assertIn("emit depthMapArtifactSaved(artifact)", generator)

        self.assertIn("makeProjectDepthRecordFromArtifact", manager)
        self.assertIn("&DepthMapGenerator::depthMapArtifactSaved", manager)
        self.assertIn('depthResult[QStringLiteral("mvs_output_dir")]', manager)

        self.assertIn('obj.value(QStringLiteral("device")).toString()', tree)

    def test_depth_estimation_checks_cancel_between_expensive_stages(self):
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn('cancelled("深度范围估计后")', generator)
        self.assertIn('cancelled("极线校正后")', generator)
        self.assertIn('cancelled("构建深度 hint 后")', generator)
        self.assertIn('cancelled("粗层 PatchMatch 后")', generator)
        self.assertIn('cancelled("精细层 PatchMatch 后")', generator)
        self.assertIn('cancelled("深度后处理后")', generator)

    def test_depth_reuse_requires_raw_artifacts_and_normalized_camera_lookup(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        depth_utils = self.read("src/core/mvs/DepthFrameUtils.cpp")

        self.assertIn("depthFrameArtifactsExist(pngPath)", manager)
        self.assertIn("cameraForImagePath(camMap, imgPath", manager)
        self.assertIn("cameraForImagePath(_cameraMap, _records[index].refImage", manager)
        self.assertNotIn("camMap.value(imgPath)", manager)
        self.assertNotIn("camMap.value(stored.refImage)", manager)

        self.assertIn('record.value(QStringLiteral("depth_png"))', depth_utils)
        self.assertIn("depthFrameArtifactsExist(frame)", depth_utils)
        self.assertIn("QFileInfo::exists(frame.depthPng)", depth_utils)
        self.assertIn("QFileInfo::exists(frame.rawDepthPath)", depth_utils)

    def test_dem_previews_do_not_pollute_mvs_depth_results(self):
        terrain = self.read("src/gui/project/manager/ProjectTerrainProductsManager.cpp")
        model = self.read("src/gui/project/manager/ProjectModelManager.cpp")
        tree = self.read("src/gui/widgets/DataTreeWidget.cpp")

        self.assertNotIn(
            'upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral("depth_map_results")',
            terrain,
        )
        self.assertNotIn("depth_map_results", model)
        self.assertIn("collectLatestStoredDepthFrames", terrain)
        self.assertIn('depthResultKind(obj) == QStringLiteral("mvs_depth")', tree)
        self.assertIn("depth_preview_png", tree)

    def test_disparity_heatmap_masks_invalid_pixels_with_alpha(self):
        header = self.read("src/gui/widgets/DisparityHeatmapOverlay.h")
        source = self.read("src/gui/widgets/DisparityHeatmapOverlay.cpp")

        self.assertIn("QImage heatmapImage() const", header)
        self.assertIn("QImage::Format_RGBA8888", source)
        self.assertIn("alphaRow[col] = validRow[col] ? 255 : 0", source)
        self.assertIn("if (_showInvalid)", source)

    def test_pipeline_dense_stage_reuses_depth_fusion_entrypoint(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("if (request.pipelineMode)", manager)
        self.assertIn("genCfg.runFusion = false", manager)
        self.assertIn("shouldStartFusion = success && (continueMissingMode || pipelineMode)", manager)
        self.assertIn("startFuseDepthMapsAsync(settings)", manager)

    def test_one_click_three_d_fusion_defaults_match_depth_fusion_dialog(self):
        workflow = self.read("src/gui/main_window/MenuWorkflowController.cpp")
        dialog_ui = self.read("src/gui/dialogs/DepthFusionDialog.ui")

        self.assertIn('denseSettings[QStringLiteral("keepNormals")] = true;', workflow)
        self.assertIn('registered_image_count', workflow)
        self.assertIn('denseMinViewCount', workflow)
        self.assertIn('denseSettings[QStringLiteral("minConsistentViews")] = denseMinViewCount;', workflow)
        self.assertIn('denseSettings[QStringLiteral("minViews")] = denseMinViewCount;', workflow)
        self.assertIn('denseSettings[QStringLiteral("minConfidence")] = 0.50;', workflow)
        self.assertIn('denseSettings[QStringLiteral("depthConsistency")] = 1.0;', workflow)

        self.assertIn('<string>保留法向量</string>', dialog_ui)
        self.assertIn('<number>3</number>', dialog_ui)
        self.assertIn('<double>0.500000000000000</double>', dialog_ui)
        self.assertIn('<double>1.000000000000000</double>', dialog_ui)

    def test_one_click_three_d_runs_dense_refine_before_mesh_and_defaults_to_texture(self):
        workflow = self.read("src/gui/main_window/MenuWorkflowController.cpp")
        dialog_ui = self.read("src/gui/dialogs/ThreeDReconstructionDialog.ui")

        self.assertIn("startThreeDReconstructionDenseRefineStage(settings)", workflow)
        self.assertIn("startDenseCloudRefineAsync(refineSettings)", workflow)
        self.assertIn("startThreeDReconstructionMeshStage(settings)", workflow)
        self.assertIn('refineSettings[QStringLiteral("pipeline_mode")] = true;', workflow)
        self.assertIn('refineSettings[QStringLiteral("normalsEnabled")] = true;', workflow)

        self.assertIn('<widget class="QCheckBox" name="m_exportObjCheck">', dialog_ui)
        self.assertIn('<bool>true</bool>', dialog_ui)

    def test_sparse_cloud_preprocessing_uses_requested_processing_device(self):
        config_h = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.h")
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        dense_settings = config_h[
            config_h.index("struct DenseGenerationSettings"):
            config_h.index("DenseGenerationSettings denseGenerationSettingsFromJson")
        ]
        dense_parse = config_cpp[
            config_cpp.index("DenseGenerationSettings denseGenerationSettingsFromJson"):
            config_cpp.index("xjw::mvs::DepthGenConfig buildDepthGenConfig")
        ]

        self.assertIn("plapoint::ProcessingDevice processingDevice", dense_settings)
        self.assertIn("parsed.processingDevice = processingDeviceFromString", dense_parse)
        self.assertEqual(manager.count("SparseCloudPreprocessor pp(request.processingDevice);"), 2)
        self.assertNotIn("SparseCloudPreprocessor pp;", manager)

    def test_dense_cloud_dialog_advanced_mvs_settings_reach_depth_config(self):
        config_h = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.h")
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")

        self.assertIn("bool geomConsistency", config_h)
        self.assertIn("int speckleMinArea", config_h)
        self.assertIn("QString qualityProfile", config_h)
        self.assertIn('settings.value(QStringLiteral("geomConsistency")).toBool(true)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("speckleMinArea")).toInt(16)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("qualityProfile"))', config_cpp)
        self.assertIn('applyDenseQualityProfile', config_cpp)
        self.assertIn('QStringLiteral("fast_preview")', config_cpp)
        self.assertIn('QStringLiteral("standard")', config_cpp)
        self.assertIn('QStringLiteral("high_quality")', config_cpp)
        self.assertIn("config.patchMatch.geomConsistency = settings.geomConsistency", config_cpp)
        self.assertIn("config.fusion.minSpeckleComponentArea", config_cpp)
        self.assertIn("config.fusion.enableSpeckleFilter", config_cpp)
        self.assertIn("config.fusion.enableAdaptiveConfidenceFilter", config_cpp)

    def test_dense_cloud_production_profile_controls_fusion_depth_threshold(self):
        config_h = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.h")
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("float fusionRelDepthThreshold = 0.03f", config_h)
        self.assertIn("parsed->fusionRelDepthThreshold", config_cpp)
        self.assertIn("config.fusion.relDepthThresh = settings.fusionRelDepthThreshold", config_cpp)
        self.assertIn("fusionCfg.maxDepthError = request.fusionRelDepthThreshold", manager)
        self.assertNotIn("fusionCfg.maxDepthError = 0.05f;", manager)

    def test_sparse_cloud_preprocessor_samples_spacing_and_parallelizes_linear_passes(self):
        preprocessor = self.read("src/core/mvs/SparseCloudPreprocessor.cpp")

        self.assertIn("kMaxMedianSpacingSamples", preprocessor)
        self.assertIn("sampleCount", preprocessor)
        self.assertIn("#pragma omp parallel for", preprocessor)
        self.assertNotIn("distances.reserve(cloud.size());", preprocessor)

    def test_depth_generator_skips_inline_dense_filters_for_large_clouds(self):
        generator = self.read("src/core/mvs/DepthMapGenerator.cpp")

        self.assertIn("kMaxInlineDenseFilterPoints", generator)
        self.assertIn("initialCount <= kMaxInlineDenseFilterPoints", generator)
        self.assertIn("跳过内联稠密点云过滤", generator)
        self.assertNotIn("\"SOR-2\"", generator)

    def test_dense_cloud_refine_avoids_second_strict_sor_pass(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("统计离群点移除 (SOR)", manager)
        self.assertIn("半径离群点移除", manager)
        self.assertNotIn("离群点二次清理", manager)
        self.assertNotIn("strictSorReport", manager)

    def test_gui_mvs_cancel_reaches_fusion_and_refine_workers(self):
        header = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.h")
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        self.assertIn("std::shared_ptr<std::atomic_bool> _activeMvsCancelFlag", header)
        self.assertIn("createActiveMvsCancelFlag", header)
        self.assertIn("clearActiveMvsCancelFlag", header)

        cancel_start = manager.index("void ProjectDenseReconstructionManager::cancelMvs()")
        cancel_body = manager[cancel_start:]
        self.assertIn("_activeMvsCancelFlag->store(true", cancel_body)
        self.assertIn("gen->requestCancel()", cancel_body)

        fuse_start = manager.index("void ProjectDenseReconstructionManager::startFuseDepthMapsAsync")
        refine_start = manager.index("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync")
        fuse_body = manager[fuse_start:refine_start]
        refine_body = manager[refine_start:]

        self.assertIn("const auto cancelFlag = createActiveMvsCancelFlag();", fuse_body)
        self.assertIn("cancelFlag", fuse_body)
        self.assertIn("cancelFlag->load", fuse_body)
        self.assertIn("深度图融合已取消", fuse_body)
        self.assertIn("clearActiveMvsCancelFlag(cancelFlag)", fuse_body)

        self.assertIn("const auto cancelFlag = createActiveMvsCancelFlag();", refine_body)
        self.assertIn("cancelFlag", refine_body)
        self.assertIn("cancelFlag->load", refine_body)
        self.assertIn("密集点云后处理已取消", refine_body)
        self.assertIn("clearActiveMvsCancelFlag(cancelFlag)", refine_body)

    def test_gui_mvs_cancel_finishes_status_when_no_worker_accepts_request(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")

        cancel_start = manager.index("void ProjectDenseReconstructionManager::cancelMvs()")
        cancel_body = manager[cancel_start:]

        self.assertIn("if (!requestedCancel)", cancel_body)
        self.assertIn("emit mvsProgressFinished(false);", cancel_body)
        self.assertLess(cancel_body.index("if (!requestedCancel)"),
                        cancel_body.index("qDebug() << \"[MVS] 已请求取消\""))

    def test_gui_dense_cloud_refine_preconditions_large_clouds_before_expensive_filters(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        refine_start = manager.index("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync")
        refine_body = manager[refine_start:]

        self.assertIn("kMaxDenseRefineFilterInputPoints", manager)
        self.assertIn("preconditionDenseRefineCloudForFilters", manager)
        self.assertIn("点云过大，先进行预降采样", manager)
        self.assertLess(refine_body.index("preconditionDenseRefineCloudForFilters"),
                        refine_body.index("sorFilter(cloud"))
        self.assertLess(refine_body.index("preconditionDenseRefineCloudForFilters"),
                        refine_body.index("estimateNormals(cloud"))
        self.assertIn("!precondition.consumedRequestedVoxel", refine_body)

    def test_gui_dense_cloud_refine_uses_streaming_cli_before_loading_large_ply(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        refine_start = manager.index("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync")
        refine_body = manager[refine_start:]

        self.assertIn("kStreamingDenseRefineMinPoints", manager)
        self.assertIn("shouldUseStreamingDenseRefine", manager)
        self.assertIn("runStreamingDenseCloudRefineCli", manager)
        self.assertIn("dense_cloud_refine_cli", manager)
        self.assertIn("parseBinaryPlyVertexStreamHeader", manager)
        self.assertIn("QProcess process", manager)
        self.assertIn("--terrain-filter-passes", manager)
        self.assertLess(refine_body.index("runStreamingDenseCloudRefineCli"),
                        refine_body.index("readPointCloudPly(inputPly"))

    def test_dense_refine_defaults_use_metashape_quality_tuned_terrain_filter(self):
        config_h = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.h")
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")

        self.assertIn("int terrainSpikeGridResolution = 260;", config_h)
        self.assertIn("int terrainSpikeMinCellPoints = 32;", config_h)
        self.assertIn("double terrainSpikeMinHeightThreshold = 0.25;", config_h)
        self.assertIn("double terrainSpikeMadMultiplier = 3.0;", config_h)
        self.assertIn("bool terrainLocalPlaneFilterEnabled = true;", config_h)
        self.assertIn("int terrainLocalPlaneMinPoints = 12;", config_h)
        self.assertIn("double terrainLocalPlaneMinResidualThreshold = 0.12;", config_h)
        self.assertIn("double terrainLocalPlaneMadMultiplier = 4.0;", config_h)
        self.assertIn("int terrainFilterPasses = 2;", config_h)

        self.assertIn('settings.value(QStringLiteral("terrainSpikeGridResolution")).toInt(260)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainSpikeMinCellPoints")).toInt(32)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainSpikeMinHeightThreshold")).toDouble(0.25)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainSpikeMadMultiplier")).toDouble(3.0)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainLocalPlaneFilterEnabled")).toBool(true)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainLocalPlaneMinPoints")).toInt(12)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainLocalPlaneMinResidualThreshold")).toDouble(0.12)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainLocalPlaneMadMultiplier")).toDouble(4.0)', config_cpp)
        self.assertIn('settings.value(QStringLiteral("terrainFilterPasses")).toInt(2)', config_cpp)

        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        self.assertIn("options.localPlaneFilterEnabled = request.terrainLocalPlaneFilterEnabled", manager)
        self.assertIn("options.localPlaneMinPoints = request.terrainLocalPlaneMinPoints", manager)
        self.assertIn("options.localPlaneMinResidualThreshold", manager)
        self.assertIn("options.localPlaneMadMultiplier", manager)
        self.assertIn("local_plane_removed_points", manager)


if __name__ == "__main__":
    unittest.main()
