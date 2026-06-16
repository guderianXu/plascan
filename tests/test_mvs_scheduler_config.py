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
        self.assertIn("m_visibilityBits", header)
        self.assertIn("m_pairCommonCounts", header)
        self.assertIn("prepareFrameCaches();", scheduler)
        self.assertIn("sourceViewIndicesForFrame", scheduler)
        self.assertIn("visibleSparsePointIndicesForFrame", scheduler)
        self.assertNotIn("selectMvsSourceViewIndices(m_views, m_sparse, refIdx, numSrc)", scheduler)

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

    def test_cuda_scheduler_defaults_to_one_frame_worker(self):
        config_cpp = self.read("src/gui/project/support/ProjectDenseWorkflowConfig.cpp")
        self.assertIn("autoGpuFrameWorkers", config_cpp)
        self.assertIn("return 1;", config_cpp)
        self.assertNotIn("threads / 4), 1, maxGpuWorkers", config_cpp)

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

    def test_depth_reuse_requires_raw_artifacts_and_normalized_camera_lookup(self):
        manager = self.read("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        depth_utils = self.read("src/core/mvs/DepthFrameUtils.cpp")

        self.assertIn("depthFrameArtifactsExist(pngPath)", manager)
        self.assertIn("cameraForImagePath(camMap, imgPath", manager)
        self.assertIn("cameraForImagePath(camMap, stored.refImage", manager)
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

        self.assertNotIn("depth_map_results", terrain)
        self.assertNotIn("depth_map_results", model)
        self.assertIn('depthResultKind(obj) == QStringLiteral("mvs_depth")', tree)
        self.assertIn("depth_preview_png", tree)

    def test_disparity_heatmap_masks_invalid_pixels_with_alpha(self):
        header = self.read("src/gui/widgets/DisparityHeatmapOverlay.h")
        source = self.read("src/gui/widgets/DisparityHeatmapOverlay.cpp")

        self.assertIn("QImage heatmapImage() const", header)
        self.assertIn("QImage::Format_RGBA8888", source)
        self.assertIn("alphaRow[col] = validRow[col] ? 255 : 0", source)
        self.assertIn("if (m_showInvalid)", source)

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


if __name__ == "__main__":
    unittest.main()
