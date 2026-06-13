import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ThreeDReconstructionCliTest(unittest.TestCase):
    def test_cli_target_exists_and_reuses_gui_reconstruction_source(self):
        cmake = (ROOT / "src/cli/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("three_d_reconstruction_cli", cmake)
        self.assertIn("cli_reconstruct_pipeline.cpp", cmake)
        self.assertIn("PLASCAN_THREE_D_ONLY", cmake)

    def test_three_d_only_mode_runs_sfm_mvs_mesh_without_dem_dom(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn("PLASCAN_THREE_D_ONLY", source)
        self.assertIn("SFMServiceOptions", source)
        self.assertIn("buildDepthGenConfig", source)
        self.assertIn("buildMeshAndOptionalTexture", source)

        terrain_guard = re.search(
            r"#ifndef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)#endif",
            source,
            re.S,
        )
        self.assertIsNotNone(terrain_guard)
        self.assertIn("TerrainPipeline", terrain_guard.group("body"))

        self.assertIsNotNone(re.search(
            r"#ifdef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)three_d_reconstruction_output",
            source,
            re.S,
        ))

    def test_cli_stops_before_mvs_when_sfm_sparse_cloud_is_too_small(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        sfm_guard = re.search(
            r"kMinimumSparsePointsForDenseWorkflow\s*=\s*20.*?"
            r"if\s*\(\s*sfmResult\.numPoints3D\s*<\s*kMinimumSparsePointsForDenseWorkflow\s*\)(?P<body>.*?)"
            r"QMap<QString,\s*xjw::Camera>",
            source,
            re.S,
        )
        self.assertIsNotNone(sfm_guard)
        self.assertIn("SFM 稀疏点云点数过少", sfm_guard.group("body"))
        self.assertIn("return cli::EXIT_ALGO_ERR", sfm_guard.group("body"))

    def test_cli_stops_before_mvs_when_registered_views_are_too_few(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn("kMinimumRegisteredImagesForDenseWorkflow = 2", source)
        view_guard = re.search(
            r"if\s*\(\s*views\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumRegisteredImagesForDenseWorkflow\)\s*\)"
            r"(?P<body>.*?)"
            r"xjw::mvs::SparseCloud sparse",
            source,
            re.S,
        )
        self.assertIsNotNone(view_guard)
        self.assertIn("SFM 后可用于 MVS 的相机不足", view_guard.group("body"))
        self.assertIn("return cli::EXIT_ALGO_ERR", view_guard.group("body"))

    def test_cli_rechecks_sparse_cloud_after_preprocessing(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        preprocess_guard = re.search(
            r"filtered_sparse_points.*?"
            r"if\s*\(\s*sparse\.points\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumSparsePointsForDenseWorkflow\)\s*\)"
            r"(?P<body>.*?)"
            r"std::fprintf\(stdout,\s*\"\[2/%d\] MVS",
            source,
            re.S,
        )
        self.assertIsNotNone(preprocess_guard)
        self.assertIn("预处理后的 SFM 稀疏点云点数过少", preprocess_guard.group("body"))
        self.assertIn("return cli::EXIT_ALGO_ERR", preprocess_guard.group("body"))

    def test_cli_defaults_match_gui_three_d_reconstruction_workflow(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn('std::string device = "auto"', source)
        self.assertIn("registerCliConsoleLogger", source)
        self.assertIn("Logger::instance()->registerSink", source)
        self.assertIn("--feature-max-image-dim", source)
        self.assertIn("sfmOptions.featureMaxImageDim = featureMaxImageDim", source)
        self.assertIn("const int denseMinViewCount", source)
        self.assertIn("denseSettings.minViews = denseMinViewCount", source)
        self.assertIn("denseSettings.minConsistentViews = denseMinViewCount", source)
        self.assertIn("denseSettings.fusionMinConfidence = 0.50f", source)
        self.assertIn("denseSettings.depthConsistency = 1.0f", source)
        self.assertIn("depthConfig.runFusion = false", source)
        self.assertIn("loadFusionFramesFromDepthMaps", source)
        self.assertIn("DepthMapFusion fusion", source)
        self.assertIn("DenseRefineSettings refineSettings", source)
        self.assertIn("refineDenseCloud", source)
        self.assertIn("dense_cloud_refined.ply", source)
        self.assertIn("meshRequest.pointCloudPath = refinedCloudPathForModel", source)
        self.assertIn("bool exportObj = true", source)
        self.assertIn("--skip-texture", source)
        self.assertIn("int meshResolution = 224", source)
        self.assertIn("meshRequest.reconstruction.poissonDepth = 9", source)
        self.assertIn("meshRequest.reconstruction.simplifyTargetFaces = 28000", source)

    def test_cli_supports_stage_control_for_benchmark_runs(self):
        source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn("--stop-after-sfm", source)
        self.assertIn("--skip-mvs", source)
        self.assertIn("--skip-mesh", source)

        stop_guard = re.search(
            r"if\s*\(\s*stopAfterSfm\s*\|\|\s*skipMvs\s*\)(?P<body>.*?)"
            r"kMinimumRegisteredImagesForDenseWorkflow",
            source,
            re.S,
        )
        self.assertIsNotNone(stop_guard)
        self.assertIn('report[QStringLiteral("status")] = QStringLiteral("ok")', stop_guard.group("body"))
        self.assertIn("mvs", stop_guard.group("body"))
        self.assertIn("mesh", stop_guard.group("body"))
        self.assertIn("return cli::EXIT_OK", stop_guard.group("body"))

        self.assertLess(source.index("stopAfterSfm || skipMvs"),
                        source.index("kMinimumSparsePointsForDenseWorkflow"))

    def test_mvs_uses_only_sfm_registered_cameras(self):
        cli_source = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")
        gui_source = (ROOT / "src/gui/main_window/MenuWorkflowController.cpp").read_text(encoding="utf-8")

        self.assertIn("registeredImagePaths", cli_source)
        self.assertIn("sfmResult.pendingCamUpdates", cli_source)
        self.assertNotIn("cameraByImage.insert(item.imagePath, item.camera);", cli_source)

        self.assertIn("registeredImages", gui_source)
        self.assertIn("result.pendingCamUpdates.keys()", gui_source)
        self.assertIn("appendAtResult(result.sparseCloudPath", gui_source)


if __name__ == "__main__":
    unittest.main()
