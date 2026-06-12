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


if __name__ == "__main__":
    unittest.main()
