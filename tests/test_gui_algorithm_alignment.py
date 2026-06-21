from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        return ""
    brace = source.find("{", start)
    if brace < 0:
        return ""

    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return source[start:]


class GuiAlgorithmAlignmentTest(unittest.TestCase):
    def test_workflow_quality_mapping_covers_fast_standard_quality(self):
        source = read_source("src/gui/main_window/MenuWorkflowController.cpp")

        helper = function_body(source, "int sfmQualityLevelFromWorkflowQuality")
        self.assertIn('quality == QStringLiteral("fast")', helper)
        self.assertRegex(helper, r"return\s+0\s*;")
        self.assertIn('quality == QStringLiteral("quality")', helper)
        self.assertRegex(helper, r"return\s+2\s*;")
        self.assertRegex(helper, r"return\s+1\s*;")
        self.assertGreaterEqual(source.count("opts.quality = sfmQualityLevelFromWorkflowQuality(quality);"), 2)

    def test_dem_pipeline_uses_canonical_feature_and_match_config_keys(self):
        manager = read_source("src/gui/project/manager/ProjectTerrainProductsManager.cpp")
        runner = read_source("src/gui/tasks/FeatureExtractionRunner.cpp")

        self.assertIn("canonicalFeatureAlgorithmFromMatcher", manager)
        self.assertIn("canonicalMatchAlgorithmFromMatcher", manager)
        self.assertIn('featureConfig[QStringLiteral("device")] = QStringLiteral("CUDA");', manager)
        self.assertIn('featureConfig[QStringLiteral("max_num_keypoints")]', manager)
        self.assertNotIn('featureConfig[QStringLiteral("max_keypoints")]', manager)
        self.assertIn('matchConfig[QStringLiteral("use_cuda")] = true;', manager)
        self.assertIn('matchConfig[QStringLiteral("outlier_method")]', manager)
        self.assertNotIn('matchConfig[QStringLiteral("device")]', manager)
        self.assertNotIn('matchConfig[QStringLiteral("outlier_filter")]', manager)

        self.assertIn("maxKeypointsFromConfig", runner)
        self.assertIn("deviceString.toLower()", runner)

    def test_bundle_adjust_defaults_match_core_and_cli(self):
        manager = read_source("src/gui/project/manager/ProjectManager.cpp")
        cli = read_source("src/cli/cli_bundle_adjust.cpp")

        self.assertIn('extraSettings.value(QStringLiteral("max_point_iterations")).toInt(12)', manager)
        self.assertIn('extraSettings.value(QStringLiteral("max_camera_iterations")).toInt(10)', manager)
        self.assertIn('extraSettings.value(QStringLiteral("finite_diff_eps")).toDouble(1e-6)', manager)
        self.assertIn('extraSettings.value(QStringLiteral("damping")).toDouble(1e-3)', manager)
        self.assertIn('extraSettings.value(QStringLiteral("step_tolerance")).toDouble(1e-8)', manager)
        self.assertIn('extraSettings.value(QStringLiteral("filter_max_reproj_error")).toDouble(2.5)', manager)

        self.assertIn("int maxPointIterations = 12;", cli)
        self.assertIn("int maxCameraIterations = 10;", cli)
        self.assertIn("double stepTolerance = 1e-8;", cli)

    def test_mvs_depth_metadata_and_cleanup_include_valid_mask(self):
        dense_manager = read_source("src/gui/project/manager/ProjectDenseReconstructionManager.cpp")
        cleanup = read_source("src/gui/project/services/ProjectResourceCleanupService.cpp")

        self.assertIn("existingDepthRecordForPath", dense_manager)
        self.assertIn("validMaskStoragePath", dense_manager)
        self.assertIn('depthResult[QStringLiteral("valid_mask_path")]', dense_manager)
        self.assertIn("validMaskStoragePath", function_body(dense_manager, "void removeDepthArtifactsForIndices"))
        self.assertIn('record.value(QStringLiteral("valid_mask_path"))', cleanup)

    def test_dense_match_dialog_filters_reach_core_config(self):
        manager = read_source("src/gui/project/manager/ProjectManager.cpp")
        service = read_source("src/core/dense_match/DenseMatchService.cpp")

        self.assertIn('settings.value(QStringLiteral("lr_threshold"))', manager)
        self.assertIn("cfg.lrCheckThreshold", manager)
        self.assertIn("cfg.enableLRCheck", manager)
        self.assertIn('settings.value(QStringLiteral("median_filter"))', manager)
        self.assertIn("cfg.medianFilterSize", manager)
        self.assertIn("enableLRCheck", service)
        self.assertIn("checkLRConsistency", service)

    def test_dialogs_do_not_emit_unsupported_core_settings(self):
        dense_cloud = function_body(read_source("src/gui/dialogs/DenseCloudDialog.cpp"),
                                    "QJsonObject DenseCloudDialog::collectSettings")
        depth_fusion = function_body(read_source("src/gui/dialogs/DepthFusionDialog.cpp"),
                                     "QJsonObject DepthFusionDialog::collectSettings")
        triangulation = function_body(read_source("src/gui/dialogs/TriangulationDialog.cpp"),
                                      "QJsonObject TriangulationDialog::collectSettings")
        texture = function_body(read_source("src/gui/dialogs/TextureMappingDialog.cpp"),
                                "QJsonObject TextureMappingDialog::collectSettings")

        for key in ["num_disparities", "block_size", "uniqueness_ratio", "speckle_window_size",
                    "use_full_dp", "use_wls_filter"]:
            self.assertNotIn(key, dense_cloud)
        self.assertIn('s["resScale"]', dense_cloud)
        self.assertIn('s["iterations"]', dense_cloud)
        self.assertIn('s["patchSize"]', dense_cloud)
        self.assertIn('s["minViews"]', dense_cloud)

        self.assertNotIn('o["cuda"]', depth_fusion)

        for key in ["depthStability", "filterMode", "maxReprojError", "minAngleFilter"]:
            self.assertNotIn(key, triangulation)

        for key in ["colorCorrection", "ghostFilter", "seamsMargin", "threads"]:
            self.assertNotIn(key, texture)

    def test_mesh_decimation_reaches_reconstruction_config(self):
        manager = read_source("src/gui/project/manager/ProjectModelManager.cpp")

        self.assertIn('settings.value(QStringLiteral("decimate")).toBool(false)', manager)
        self.assertIn('settings.value(QStringLiteral("decimateRatio"))', manager)
        self.assertRegex(manager, r"cfg\.simplifyTargetFaces\s*=.*decimate")


if __name__ == "__main__":
    unittest.main()
