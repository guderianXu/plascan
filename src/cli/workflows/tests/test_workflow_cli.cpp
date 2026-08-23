#include "CliTestSupport.h"
#include "PointCloudWorkflowConfig.h"

// 这些测试只覆盖 GUI“工作流程”菜单对应的 CLI。

#ifndef PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH
#define PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH ""
#endif

#ifndef PLASCAN_TEXTURE_MAP_CLI_PATH
#define PLASCAN_TEXTURE_MAP_CLI_PATH ""
#endif

namespace
{

// CLI 报告属于活动 Chunk，入口程序会以 `键=绝对路径` 的形式回报真实位置。
// 测试从标准输出取路径，避免重新假设项目内部目录布局。
QString reportedPath(const CliResult &result, const QString &key)
{
    const QRegularExpression expression(
        QStringLiteral("(?:^|[\\r\\n])%1=([^\\r\\n]+)")
            .arg(QRegularExpression::escape(key)));
    const QRegularExpressionMatch match = expression.match(combinedOutput(result));
    return match.hasMatch() ? match.captured(1).trimmed() : QString();
}

QJsonObject replayCameraModel(double centerX)
{
    return QJsonObject{
        {QStringLiteral("fx"), 100.0},
        {QStringLiteral("fy"), 100.0},
        {QStringLiteral("cx"), 1.0},
        {QStringLiteral("cy"), 1.0},
        {QStringLiteral("rotation_world_to_camera"),
         QJsonArray{1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0}},
        {QStringLiteral("camera_center"), QJsonArray{centerX, 0.0, 0.0}}
    };
}

QString writeReplayManifest(const QString &root, bool includeVerifiedSourcePlan)
{
    const QString image0 = QDir(root).filePath(QStringLiteral("image_0.pgm"));
    const QString image1 = QDir(root).filePath(QStringLiteral("image_1.pgm"));
    const QByteArray pgm("P2\n2 2\n255\n10 20\n30 40\n");
    writeBytesFile(image0, pgm);
    writeBytesFile(image1, pgm);

    QJsonArray sourcePlan;
    if (includeVerifiedSourcePlan)
    {
        sourcePlan.append(QJsonObject{
            {QStringLiteral("view_index"), 1},
            {QStringLiteral("source_image"), image1},
            {QStringLiteral("verification_status"), QStringLiteral("verified")},
            {QStringLiteral("verified_pair_geometry"), true},
            {QStringLiteral("pair_total_matches"), 80},
            {QStringLiteral("geometric_inliers"), 70},
            {QStringLiteral("pair_coverage_score"), 0.75},
            {QStringLiteral("verification_reason"), QStringLiteral("test_verified_pair")}
        });
    }

    const QJsonArray frames{
        QJsonObject{
            {QStringLiteral("ref_index"), 0},
            {QStringLiteral("ref_image"), image0},
            {QStringLiteral("source_plan"), sourcePlan},
            {QStringLiteral("camera_model"), replayCameraModel(0.0)},
            {QStringLiteral("status"), QStringLiteral("completed")}
        },
        QJsonObject{
            {QStringLiteral("ref_index"), 1},
            {QStringLiteral("ref_image"), image1},
            {QStringLiteral("source_plan"), QJsonArray{}},
            {QStringLiteral("camera_model"), replayCameraModel(0.1)},
            {QStringLiteral("status"), QStringLiteral("completed")}
        }
    };
    const QString manifestPath =
        QDir(root).filePath(QStringLiteral("mvs_manifest.json"));
    writeTextFile(
        manifestPath,
        QString::fromUtf8(QJsonDocument(QJsonObject{
            {QStringLiteral("config_hash"), QStringLiteral("test")},
            {QStringLiteral("frames"), frames}
        }).toJson(QJsonDocument::Indented)));
    return manifestPath;
}

QString writeRelativeReplayManifest(const QString &root)
{
    const QString imageDirectory =
        QDir(root).filePath(QStringLiteral("影像 目录"));
    QDir().mkpath(imageDirectory);
    const QString image0 =
        QDir(imageDirectory).filePath(QStringLiteral("影像_0.pgm"));
    const QString image1 =
        QDir(imageDirectory).filePath(QStringLiteral("影像_1.pgm"));
    const QByteArray pgm("P2\n2 2\n255\n10 20\n30 40\n");
    writeBytesFile(image0, pgm);
    writeBytesFile(image1, pgm);

    const QString storedImage0 = QDir(root).relativeFilePath(image0);
    const QString storedImage1 = QDir(root).relativeFilePath(image1);
    const QJsonArray frames{
        QJsonObject{
            {QStringLiteral("ref_index"), 0},
            {QStringLiteral("ref_image"), storedImage0},
            {QStringLiteral("source_plan"),
             QJsonArray{QJsonObject{
                 {QStringLiteral("view_index"), 1},
                 {QStringLiteral("source_image"), storedImage1},
                 {QStringLiteral("verification_status"), QStringLiteral("verified")},
                 {QStringLiteral("pair_total_matches"), 80},
                 {QStringLiteral("geometric_inliers"), 70},
                 {QStringLiteral("pair_coverage_score"), 0.75}}}},
            {QStringLiteral("camera_model"), replayCameraModel(0.0)},
            {QStringLiteral("status"), QStringLiteral("completed")}},
        QJsonObject{
            {QStringLiteral("ref_index"), 1},
            {QStringLiteral("ref_image"), storedImage1},
            {QStringLiteral("source_plan"), QJsonArray{}},
            {QStringLiteral("camera_model"), replayCameraModel(0.1)},
            {QStringLiteral("status"), QStringLiteral("completed")}}
    };
    const QString manifestPath =
        QDir(root).filePath(QStringLiteral("mvs_manifest.json"));
    writeTextFile(
        manifestPath,
        QString::fromUtf8(QJsonDocument(QJsonObject{
            {QStringLiteral("config_hash"), QStringLiteral("test-relative")},
            {QStringLiteral("frames"), frames}
        }).toJson(QJsonDocument::Indented)));
    return manifestPath;
}

} // namespace

TEST(MvsDepthReprocessCliContractTest, PairAuditIsOptionalWithManifestSourcePlanFallback)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "manifest_source_plan",
        "pair_evidence_provenance",
        "isVerifiedSourcePlanEntry",
        "loadManifestSourcePlanPairQualities",
    });
    expectMatches(
        source,
        R"(add_option\("--pair-audit"(?P<body>.*?)check\(CLI::ExistingFile\))");
    expectNotContainsAll(
        sectionBetween(source,
                       "add_option(\"--pair-audit\"",
                       "app.add_option(\"--sparse-cloud\""),
        {"->required()"});
}

TEST(MvsDepthReprocessCliContractTest, PoseRefinementIsExplicitCandidateOnlyOptIn)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--depth-pose-candidates",
        "config.depthPoseRefinement.enabled = depthPoseCandidates",
        "不覆盖项目相机或重算深度",
    });
}

TEST(MvsDepthReprocessCliContractTest, PatchMatchUpgradesHaveReproducibleDisableControls)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--disable-per-pixel-source-selection",
        "--disable-asymmetric-propagation",
        "--disable-geometric-guidance-pass",
        "config.patchMatch.enablePerPixelSourceSelection =",
        "config.patchMatch.enableAsymmetricPropagation =",
        "config.patchMatch.enableGeometricGuidancePass =",
    });
}

TEST(MvsDepthReprocessCliContractTest, NativeFinalDepthGridIsExplicitAndAudited)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--native-depth-grid",
        "config.preserveNativeFinalDepthGrid = nativeDepthGrid",
        "preserve_native_final_depth_grid",
        "仅通用、未极线校正帧",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     SourceMaximumAngleCapIsExplicitTighteningOnlyAndAudited)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--source-max-angle-deg",
        "CLI::Range(0.0, 90.0)",
        "config.sourceMaximumAngleDegCap",
        "source_maximum_angle_deg_cap",
        "source_maximum_angle_scope",
        "patchmatch_source_plan",
        "min_scene_maximum_and_configured_cap",
        "0=禁用，只收紧场景推导值",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     SourceMaximumAngleCapCannotBeBypassedBySequenceFallback)
{
    const QString generator = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString planner = readSourceFile(
        QStringLiteral("src/core/mvs/MvsSourcePlanner.cpp"));

    expectContainsAll(generator, {
        "if (angle_cap_enabled)",
        "plannerOptions.allowSequenceFallback = false",
        "mvsSourceAngleDiagnosticsToJson",
        "safe_baseline_source_shortfall",
        "applySourceAngleCapShortfallSafety",
        "source_angle_cap_source_shortfall",
    });
    expectContainsAll(planner, {
        "explicit_source_angle_cap",
        "sequence_fallback_allowed",
        "selected_source_count",
        "selected_maximum_degrees",
        "angle_rejected_candidate_count",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     CompleteVisibilityPoolAndSoftRankingAreExplicitAndAudited)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));
    const QString generator = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString planner = readSourceFile(
        QStringLiteral("src/core/mvs/MvsSourcePlanner.cpp"));

    expectContainsAll(source, {
        "--source-complete-visibility-pool",
        "--source-angle-soft-ranking-strength",
        "默认关闭的专家实验",
        "内部诊断",
        "不作为普通用户质量策略",
        "CLI::Range(0.0, 4.0)",
        "config.evaluateCompleteVisibilityCandidatePool",
        "config.sourceAngleSoftRankingStrength",
        "validateMvsSourceRankingConfiguration",
        "evaluate_complete_visibility_candidate_pool",
        "source_candidate_pool_policy",
        "complete_evaluated_visibility_candidate_pool",
        "source_angle_soft_ranking_strength",
        "legacy_score*exp(-strength*t)",
    });
    expectContainsAll(generator, {
        "legacyCandidatePoolClosed",
        "evaluateCompleteVisibilityCandidatePool",
        "plannerOptions.auditSourceRanking",
        "legacyCandidateAngles.push_back(medianAngle)",
        "legacyCandidateAngles)",
        "if (legacy_evaluated_candidate && hasRequiredPairQuality",
        "if (_config.evaluateCompleteVisibilityCandidatePool)",
        "plannerOptions.softMaxTriangulationAngleDeg",
        "plannerOptions.maxTriangulationAngleDeg",
        "mvsSourceRankingDiagnosticsToJson",
    });
    expectContainsAll(planner, {
        "complete_evaluated_visibility_candidate_pool",
        "all_co_visible_or_required_pairs",
        "deterministic_bounded_graph",
        "legacy_score*exp(-strength*t)",
        "control_selected_count",
        "treatment_selected_count",
        "soft_maximum_degrees",
        "effective_maximum_degrees",
        "count_invariant",
        "selection_changed",
        "selected_view_set_changed",
        "selected_order_changed",
        "control_candidate_ranking",
        "treatment_candidate_ranking",
        "control_qualified_candidate_count",
        "control_qualified_tier_entry_count",
        "treatment_qualified_candidate_count",
        "treatment_qualified_tier_entry_count",
        "selected_by_plan",
        "ranking_soft_maximum_degrees",
        "ranking_effective_maximum_degrees",
    });

    const QString default_report_initializer = sectionBetween(
        source,
        "QJsonObject report{",
        "if (completeVisibilityCandidatePool)");
    expectNotContainsAll(default_report_initializer, {
        "evaluate_complete_visibility_candidate_pool",
        "source_candidate_pool_policy",
        "source_angle_soft_ranking_strength",
        "source_angle_soft_ranking_policy",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     RejectsSoftRankingWithoutCompletePoolOrWithHardCap)
{
    const QString exe = executablePath(PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString manifestPath = writeReplayManifest(tempDir.path(), true);
    const QString sparsePath =
        QDir(tempDir.path()).filePath(QStringLiteral("empty_sparse.ply"));
    writeBinaryPly(sparsePath, {});

    const QString outputWithoutPool =
        QDir(tempDir.path()).filePath(QStringLiteral("without_pool"));
    const CliResult withoutPool = runCli(exe, {
        QStringLiteral("--input-manifest"), manifestPath,
        QStringLiteral("--sparse-cloud"), sparsePath,
        QStringLiteral("--output-dir"), outputWithoutPool,
        QStringLiteral("--source-angle-soft-ranking-strength"),
        QStringLiteral("1"),
    });
    EXPECT_EQ(withoutPool.exitCode, 1);
    expectContainsAll(withoutPool.stderrText, {
        "soft ranking",
        "complete visibility candidate pool",
    });
    EXPECT_FALSE(QFileInfo::exists(outputWithoutPool));

    const QString outputWithCap =
        QDir(tempDir.path()).filePath(QStringLiteral("with_cap"));
    const CliResult withCap = runCli(exe, {
        QStringLiteral("--input-manifest"), manifestPath,
        QStringLiteral("--sparse-cloud"), sparsePath,
        QStringLiteral("--output-dir"), outputWithCap,
        QStringLiteral("--source-complete-visibility-pool"),
        QStringLiteral("--source-angle-soft-ranking-strength"),
        QStringLiteral("1"),
        QStringLiteral("--source-max-angle-deg"), QStringLiteral("25"),
    });
    EXPECT_EQ(withCap.exitCode, 1);
    expectContainsAll(withCap.stderrText, {
        "soft ranking",
        "hard source angle cap",
    });
    EXPECT_FALSE(QFileInfo::exists(outputWithCap));
}

TEST(MvsDepthReprocessCliContractTest, TargetedGapRecoveryHasExplicitDiagnosticOptOut)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--disable-targeted-gap-recovery",
        "config.enableTargetedGapRecovery = !disableTargetedGapRecovery",
        "用于同输入 A/B 对比",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     DepthLayerReliabilityAnchorGateIsExplicitAndDefaultOff)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));
    const QString generator = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(source, {
        "--depth-layer-reliability-anchor-gate",
        "config.enableDepthLayerReliabilityAnchorGate =",
        "depth_layer_reliability_anchor_gate",
        "不直接删除深度，默认关闭",
    });
    expectContainsAll(generator, {
        "DepthLayerReliabilityClass::Reliable",
        "native_interpolation_anchor_eligibility",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     DepthLayerReliabilityCorrectionIsIndependentAndDefaultOff)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));
    const QString generator = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(source, {
        "--depth-layer-reliability-guided-correction",
        "config.enableDepthLayerReliabilityGuidedCorrection =",
        "depth_layer_reliability_guided_correction",
        "至少三个独立投影来源",
        "默认关闭",
    });
    expectContainsAll(generator, {
        "enableReliabilityGuidedCorrection =",
        "native_reliability_classes",
        "depth_layer_reliability.result.classMap",
    });
}

TEST(MvsDepthReprocessCliContractTest,
     StageSnapshotsAreSelectedBoundedAndConditionallyReported)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));
    const QString generator = readSourceFile(
        QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(source, {
        "--stage-snapshot-refs",
        "--stage-snapshot-max-long-edge",
        "--stage-snapshot-budget-mib",
        "config.stageSnapshotReferenceIndices = stageSnapshotRefs",
        "stage_snapshots/manifest.json",
        "stage_snapshot_status",
    });
    expectContainsAll(generator, {
        "MvsStageSnapshotStage::PatchMatchOutput",
        "MvsStageSnapshotStage::CrossViewConsistency",
        "MvsStageSnapshotStage::ConfidencePostprocess",
        "MvsStageSnapshotStage::FinalAdmission",
        "single_frame_output_after_sparse_prior_and_local_filter",
        "after_cross_view_filter_and_repair_before_confidence_postprocess",
        "after_confidence_postprocess_and_anchored_repair",
        "after_final_quality_evaluation_before_artifact_publication",
    });

    const QString default_report_initializer = sectionBetween(
        source,
        "QJsonObject report{",
        "if (completeVisibilityCandidatePool)");
    expectNotContainsAll(default_report_initializer, {
        "stage_snapshot_ref_indices",
        "stage_snapshot_manifest",
        "stage_snapshot_status",
    });
}

TEST(MvsDepthReprocessCliContractTest, CanPinOneOpenClGpuForReproducibleReplay)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "--opencl-device-index",
        "config.patchMatch.openClDeviceIndex = openClDeviceIndex",
        "显式指定可避免多 GPU 混跑",
    });
}

TEST(MvsDepthReprocessCliContractTest, AutoEnablesCudaAndOpenClReplayScheduling)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    expectContainsAll(source, {
        "设备：auto/cuda/opencl/cpu",
        "CLI::IsMember({\"auto\", \"cuda\", \"opencl\", \"cpu\"})",
        "std::string device = \"auto\"",
        "device == \"cuda\" || device == \"auto\"",
        "{QStringLiteral(\"patchMatchBackend\"), QString::fromStdString(device)}",
        "int threads = 0",
        "CPU 线程预算；0=逻辑线程数减 2",
    });
}

TEST(MvsDepthReprocessCliContractTest, UsesVerifiedManifestSourcePlanWithoutPairAudit)
{
    const QString exe = executablePath(PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString manifestPath = writeReplayManifest(tempDir.path(), true);
    const QString sparsePath =
        QDir(tempDir.path()).filePath(QStringLiteral("empty_sparse.ply"));
    writeBinaryPly(sparsePath, {});

    const CliResult result = runCli(exe, {
        QStringLiteral("--input-manifest"), manifestPath,
        QStringLiteral("--sparse-cloud"), sparsePath,
        QStringLiteral("--output-dir"),
        QDir(tempDir.path()).filePath(QStringLiteral("output")),
    });

    EXPECT_NE(result.exitCode, 0);
    expectContainsAll(result.stdoutText, {
        "pair_evidence_provenance=manifest_source_plan",
        "verified_pairs=1",
    });
    expectContainsAll(result.stderrText, {"稀疏点云预处理失败"});
    expectNotContainsAll(combinedOutput(result), {
        "不含当前影像集合的已验证 MVS 源像对",
    });
}

TEST(MvsDepthReprocessCliContractTest, RejectsMissingAuditAndManifestEvidence)
{
    const QString exe = executablePath(PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString manifestPath = writeReplayManifest(tempDir.path(), false);
    const QString sparsePath =
        QDir(tempDir.path()).filePath(QStringLiteral("empty_sparse.ply"));
    const QString outputPath =
        QDir(tempDir.path()).filePath(QStringLiteral("output"));
    writeBinaryPly(sparsePath, {});

    const CliResult result = runCli(exe, {
        QStringLiteral("--input-manifest"), manifestPath,
        QStringLiteral("--sparse-cloud"), sparsePath,
        QStringLiteral("--output-dir"), outputPath,
    });

    EXPECT_NE(result.exitCode, 0);
    expectContainsAll(result.stderrText, {
        "未提供 --pair-audit",
        "source_plan",
        "不含当前影像集合的已验证 MVS 源像对",
    });
    expectNotContainsAll(result.stderrText, {"稀疏点云预处理失败"});
    EXPECT_FALSE(QFileInfo::exists(outputPath));
}

TEST(MvsDepthReprocessCliContractTest, ResolvesUnicodeRelativeManifestPaths)
{
    const QString exe = executablePath(PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString manifestPath = writeRelativeReplayManifest(tempDir.path());
    const QString sparsePath =
        QDir(tempDir.path()).filePath(QStringLiteral("empty_sparse.ply"));
    const QString outputPath =
        QDir(tempDir.path()).filePath(QStringLiteral("output"));
    writeBinaryPly(sparsePath, {});

    const CliResult result = runCli(exe, {
        QStringLiteral("--input-manifest"), manifestPath,
        QStringLiteral("--sparse-cloud"), sparsePath,
        QStringLiteral("--output-dir"), outputPath,
    });

    EXPECT_NE(result.exitCode, 0);
    expectContainsAll(result.stdoutText, {
        "pair_evidence_provenance=manifest_source_plan",
        "verified_pairs=1",
    });
    expectContainsAll(result.stderrText, {"稀疏点云预处理失败"});
    expectNotContainsAll(result.stderrText, {
        "MVS replay 影像不存在",
        "不含当前影像集合的已验证 MVS 源像对",
    });
    EXPECT_FALSE(QFileInfo::exists(outputPath));
}

TEST(WorkflowCliModuleTest, MirrorsGuiWorkflowMenuEntryPoints)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/workflows/CMakeLists.txt"));
    const QString menu = readSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString aerial = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_aerial_triangulation.cpp"));
    const QString mesh = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mesh_reconstruct.cpp"));
    const QString texture = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_texture_map.cpp"));

    expectContainsAll(cmake, {
        "aerial_triangulation_cli",
        "mesh_reconstruct_cli",
        "texture_map_cli",
        "three_d_reconstruction_cli",
        "reconstruct_pipeline_cli",
    });
    expectContainsAll(menu, {
        "空中三角测量...",
        "生成模型...",
        "创建 DEM",
        "生成 正射影像",
        "设置...",
    });
    expectContainsAll(aerial, {
        "AerialTriangulationWorkflow::run",
        "--dry-run-config",
        "--reference-mode",
        "--auto-generate-missing-matches",
        "--no-reset-alignment",
        "--mask-dir",
        "--assets-dir",
        "--match-dir",
        "--export-camera-dir",
        "exportFinalBaCameras",
        "options.assetsDir",
        "options.matchDir",
        "options.outputDir = reconstructionDir",
        "options.maskPaths = xjw::cli::maskPathsFromDirectory",
        "tie_point_preparation_executed",
        "--no-adaptive-camera-model-fitting",
        "--initial-image-id-1",
        "--initial-image-id-2",
        "options.useInitialPairHint",
        "aerial_triangulation_cli_report.json",
    });
    expectMatches(
        aerial,
        R"(if\s*\(\s*result\.reconstructionResult\.success\s*&&\s*)"
        R"(!requestedCameraExportDir\.isEmpty\(\)\s*\).*?exportFinalBaCameras)");
    expectContainsAll(mesh, {
        "GUI 等价模型生成工具",
        "xjw::mesh::workflow::buildModel",
    });
    expectContainsAll(texture, {
        "GUI 等价多视图纹理生成工具",
        "--depth-map-dir",
        "--allow-vertex-color-fallback",
        "textureConfigFromSettings",
        "xjw::mesh::workflow::buildTextureOnly",
    });
}

TEST(ReconstructPipelineCliGTest, SourceUsesUtf8ProgressAndCapsLargeDenseRefineInput)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));

    expectNotContainsAll(source, {
        "message.toLocal8Bit()",
        "离群点二次清理",
        "strictSorReport",
    });
    expectContainsAll(source, {
        "qUtf8Printable(message)",
        "constexpr std::size_t kMaxRefineInputPoints = 250000;",
        "constexpr int kMaxPasses = 6;",
        "targetPoints=%zu",
    });
}

TEST(AerialTriangulationEntryPointContractTest, FrontendsOnlyInvokeWorkflow)
{
    const QString reconstructCli =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const QString menuController =
        readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString cameraSetup =
        readSourceFile(QStringLiteral("src/gui/project/manager/ProjectCameraSetupManager.cpp"));

    expectContainsAll(reconstructCli, {"AerialTriangulationWorkflow::run"});
    expectContainsAll(menuController, {"AerialTriangulationWorkflow::run"});
    expectContainsAll(cameraSetup, {"AerialTriangulationWorkflow::run"});

    for (const QString *source : {&reconstructCli, &menuController, &cameraSetup})
    {
        expectNotContainsAll(*source, {
            "AerialTriangulationPipeline::run",
            "SfmAttemptRunner",
            "TiePointPreparation",
        });
    }
}

TEST(ReconstructPipelineCliGTest, FusedPreAggregationUsesPlaPointVoxelGrid)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const int start = source.indexOf(
        QStringLiteral("std::vector<xjw::mvs::FusedPoint> voxelDownsampleFusedPoints("));
    const int end = source.indexOf(QStringLiteral("struct FusedVoxelDownsampleResult"), start);

    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString body = source.mid(start, end - start);

    expectContainsAll(body, {
        "fusedPointsToPointCloud",
        "plapoint::voxelDownsample",
        "pointCloudToFusedPoints",
    });
    expectNotContainsAll(body, {
        "std::unordered_map<FusedVoxelKey",
    });
}

TEST(ReconstructPipelineCliGTest, SmallRingBatchesUseAdaptiveFusionConsensus)
{
    const QString source = readSourceFile(
        QStringLiteral("src/core/mvs/StreamingDepthFusionService.cpp"));

    expectContainsAll(source, {
        "frameCount <= 32",
        "std::min(fusionConfig.minNumPixels, 2)",
    });
}

TEST(ReconstructPipelineCliGTest, MeshPolicyFollowsEffectiveSceneProfile)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));

    expectContainsAll(source, {
        "effectiveSceneProfile == xjw::mvs::MvsSceneProfile::AerialTerrain",
        "const bool preserveMeshDetail",
        "reconstructionConfigForDenseScene(",
    });
}

TEST(ReconstructPipelineCliGTest, ExposesAdaptiveDepthPyramidOptions)
{
    const QString source =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.cpp"));

    expectContainsAll(source,
                      {"--mvs-quality",
                       "--mvs-backend",
                       "opencl",
                       "--point-cloud-backend",
                       "--mvs-scene-profile",
                       "general/custom",
                       "--mvs-depth-filter",
                       "--mvs-save-levels",
                       "--mvs-native-depth-grid",
                       "--mvs-mask-dir",
                       "denseSettings.qualityProfile",
                       "depthConfig.sceneProfile",
                       "depthConfig.depthFilterMode",
                       "depthConfig.saveIntermediatePyramidLevels",
                       "depthConfig.preserveNativeFinalDepthGrid"});
}

TEST(ReconstructPipelineCliGTest, ReportsCudaAndOpenClDepthFramesAsHybrid)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const QString classification = sectionBetween(
        source, "QString mvsBackendFromArtifacts", "void reportPlaPointDevice");

    expectContainsAll(classification, {
        "has_cuda",
        "has_opencl",
        "has_cpu",
        "return QStringLiteral(\"hybrid\")",
        "return QStringLiteral(\"mixed\")",
    });
}

TEST(ReconstructPipelineCliGTest, ReportsOnlyLatestDepthArtifactForEachFrame)
{
    const QString pipeline = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const QString replay = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mvs_depth_reprocess.cpp"));

    for (const QString *source : {&pipeline, &replay})
    {
        expectContainsAll(*source, {
            "depthArtifactEvents",
            "depthArtifactEvents.append(artifact)",
            "latestJsonObjectsByNonNegativeIntegerKey",
            "QStringLiteral(\"ref_index\")",
        });
    }
}

TEST(ReconstructPipelineCliGTest, ReusesQualifiedStoredDepthFramesForStreamingFusion)
{
    const QString pipeline = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));

    expectContainsAll(pipeline, {
        "collectStoredDepthFramesForDirectory(",
        "selectFusionEligibleStoredDepthFrames(",
        "buildStoredFusionFrame(",
        "storedFusionSourceIndices(",
        "stored.refIndex",
        "fuseDepthMapsStreaming(frame_count",
    });
    EXPECT_FALSE(pipeline.contains(QStringLiteral("loadFusionFrameFromDepthMap")));
    EXPECT_FALSE(pipeline.contains(QStringLiteral(
        "QStringLiteral(\"depth_%1.png\").arg(frameIndex)")));
}

TEST(ReconstructPipelineCliGTest, RoutesPlaPointBackendIndependentlyFromMvs)
{
    const QString options =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.h"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.cpp"));
    const QString workflow =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const QString config =
        readSourceFile(QStringLiteral("src/core/project_workflows/PointCloudWorkflowConfig.cpp"));
    const QString mvs =
        readSourceFile(QStringLiteral("src/core/mvs/MvsTypes.h"))
        + readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString gui_workflow =
        readSourceFile(QStringLiteral(
            "src/core/project_workflows/PointCloudInputPreparation.cpp"))
        + readSourceFile(QStringLiteral(
            "src/gui/project/manager/ProjectPointCloudWorkflowController.cpp"));
    const QString height_grid_workflow = readSourceFile(QStringLiteral(
        "src/core/mesh/SurfaceReconstructorHeightGrid.cpp"));
    const QString mesh_workflow =
        readSourceFile(QStringLiteral("src/core/mesh/SurfaceReconstructor.cpp"))
        + height_grid_workflow;

    expectContainsAll(options, {
        R"(std::string pointCloudBackend = "auto")",
        "--point-cloud-backend",
        "auto prefers CUDA, then OpenCL, then CPU",
    });
    expectContainsAll(config, {
        R"(normalized == QStringLiteral("opencl"))",
        "ProcessingDevice::CUDA",
        "ProcessingDevice::OpenCL",
        "config.pointCloudProcessingDevice = settings.processingDevice",
    });
    expectContainsAll(workflow, {
        "SparseCloudPreprocessor preprocessor(point_cloud_processing_device)",
        "denseSettings.processingDevice = point_cloud_processing_device",
        "input, leafSize, processingDevice, processingReport",
        "refineSettings.processingDevice = denseSettings.processingDevice",
        "meshRequest.reconstruction.preprocessingDevice = denseSettings.processingDevice",
        "requestedDevice=%4, actualDevice=%5",
        "processingReportWasSkipped",
        R"(QStringLiteral("skipped"))",
        "processingDeviceUnavailableReason",
        "dense_point_cloud_backend_actual_by_stage",
        "mvs_backend_actual",
        "patchmatch_backend_id",
    });
    expectNotContainsAll(workflow, {
        R"(mvs_backend == "auto" ? device : mvs_backend)",
    });
    expectContainsAll(mvs, {
        "pointCloudProcessingDevice",
        "_config.pointCloudProcessingDevice",
        "workerConfig.patchMatch.cudaFallbackToCpu = false",
        "_config.patchMatch.cudaFallbackToCpu = false",
        "prepareOpenClDevice",
        "acceleratorDeviceLeases",
        "acquire_device_lease",
        "CUDA 重试失败，配置禁止回退 CPU",
    });
    expectContainsAll(gui_workflow, {
        "SparseCloudPreprocessor preprocessor(processingDevice)",
        "context->request.processingDevice",
        "processingDeviceUnavailableReason",
        "DenseCloudBuilder::statisticalOutlierRemoval",
        "stored_backend_matches_request",
        R"(QStringLiteral("mvs_backend_request_applied"))",
        R"(QStringLiteral("depth_maps_reused"))",
        R"(QStringLiteral("point_cloud_processing"))",
        R"(QStringLiteral("mvs_backend_actual"))",
    });
    expectContainsAll(mesh_workflow, {
        "poisson.setProcessingDevice(config.poissonSolverDevice)",
        "plapoint::opencl::hasUsableOpenClDevice()",
        "plapoint::opencl::buildHeightGrid(cloud, options)",
    });
    expectNotContainsAll(mesh_workflow, {
        "poisson.setProcessingDevice(config.preprocessingDevice)",
    });
    const qsizetype cuda_route = height_grid_workflow.indexOf(
        QStringLiteral("processingDevice == plapoint::ProcessingDevice::CUDA"));
    const qsizetype opencl_route = height_grid_workflow.indexOf(
        QStringLiteral("processingDevice == plapoint::ProcessingDevice::OpenCL"));
    ASSERT_GE(cuda_route, 0);
    ASSERT_GE(opencl_route, 0);
    EXPECT_LT(cuda_route, opencl_route);
}

TEST(PointCloudWorkflowConfigGTest, AcceptsStableBackendAliases)
{
    const auto snake_case = xjw::core::project::denseGenerationSettingsFromJson(
        QJsonObject{
            {QStringLiteral("mvs_backend"), QStringLiteral("opencl")},
            {QStringLiteral("point_cloud_backend"), QStringLiteral("cuda")},
        });
    EXPECT_EQ(snake_case.patchMatchBackend, xjw::mvs::PatchMatchBackend::OpenCl);
    EXPECT_EQ(snake_case.processingDevice, plapoint::ProcessingDevice::CUDA);

    const auto camel_case = xjw::core::project::denseGenerationSettingsFromJson(
        QJsonObject{
            {QStringLiteral("mvsBackend"), QStringLiteral("cuda")},
            {QStringLiteral("pointCloudBackend"), QStringLiteral("opencl")},
        });
    EXPECT_EQ(camel_case.patchMatchBackend, xjw::mvs::PatchMatchBackend::Cuda);
    EXPECT_EQ(camel_case.processingDevice, plapoint::ProcessingDevice::OpenCL);

    const auto refine = xjw::core::project::denseRefineSettingsFromJson(
        QJsonObject{
            {QStringLiteral("point_cloud_backend"), QStringLiteral("cpu")},
        });
    EXPECT_EQ(refine.processingDevice, plapoint::ProcessingDevice::CPU);

    const auto removed_cuda_flag = xjw::core::project::denseGenerationSettingsFromJson(
        QJsonObject{{QStringLiteral("cuda"), false}});
    EXPECT_EQ(removed_cuda_flag.patchMatchBackend, xjw::mvs::PatchMatchBackend::Auto);

    const auto explicit_auto = xjw::core::project::denseGenerationSettingsFromJson(
        QJsonObject{
            {QStringLiteral("cuda"), false},
            {QStringLiteral("patchMatchBackend"), QStringLiteral("auto")},
        });
    EXPECT_EQ(explicit_auto.patchMatchBackend, xjw::mvs::PatchMatchBackend::Auto);

    const xjw::mvs::DepthGenConfig auto_config =
        xjw::core::project::buildDepthGenConfig(removed_cuda_flag, 8);
    EXPECT_GT(auto_config.gpuFrameWorkerCount, 0);
    EXPECT_GT(auto_config.cpuFrameWorkerCount, 0);
}

TEST(ReconstructPipelineCliGTest, DepthPyramidRegressionScriptCoversDinoAndUav9)
{
    const QString script = readSourceFile(
        QStringLiteral("scripts/validation/run_depth_pyramid_regression.ps1"));

    expectContainsAll(script,
                      {"middlebury_dino_sparse_ring",
                       "agisoft_aerial_gcps_small",
                       "--mvs-scene-profile",
                       "--mvs-depth-filter",
                       "--mvs-save-levels",
                       "model_quality_cli.exe"});
}

TEST(ReconstructPipelineCliGTest, DepthPyramidRegressionScriptUsesPlyQualityInput)
{
    const QString script = readSourceFile(
        QStringLiteral("scripts/validation/run_depth_pyramid_regression.ps1"));

    expectContainsAll(script, {"$report.model.model_ply", "$report.model.mesh_ply"});
    EXPECT_EQ(script.indexOf(QStringLiteral("$report.model.final_model_path")), -1);
}

TEST(ReconstructPipelineCliGTest, DepthOverlayRegressionScriptEmitsMachineReadableQualityReport)
{
    const QString script = readSourceFile(
        QStringLiteral("scripts/validation/run_depth_overlay_regression.ps1"));

    expectContainsAll(script,
                      {"[string]$Project",
                       "[string]$ListFile",
                       "[string]$OutputDirectory",
                       "[string]$SceneProfile",
                       "[int]$MaximumDimension",
                       "comparison_report.json",
                       "selected_level",
                       "mask_source",
                       "Test-ImageCameraListFile",
                       "<image> <camera.tsai>",
                       "--mvs-workspace",
                       "mean_mask_normalized_coverage",
                       "minimum_consistency_retention",
                       "small_internal_hole_pixel_count",
                       "component_face_counts",
                       "rejection_counts",
                       "finite_normals",
                       "peak_memory_bytes"});
}

TEST(ReconstructPipelineCliGTest, DepthOverlayRegressionScriptUsesStableExitCodeAndPlyQualityInput)
{
    const QString script = readSourceFile(
        QStringLiteral("scripts/validation/run_depth_overlay_regression.ps1"));

    expectContainsAll(script,
                      {"$processHandle = $process.Handle",
                       "model_ply",
                       "vertex_count",
                       "face_count",
                       "actual_mesh_algorithm",
                       "Group-Object",
                       "depth_postprocess",
                       "$qualityRun.stderr"});

    EXPECT_GE(script.indexOf(QStringLiteral("model_ply")), 0);
    EXPECT_GE(script.indexOf(QStringLiteral("mesh_ply")), 0);
    EXPECT_EQ(script.indexOf(QStringLiteral("final_model_path")), -1);
}

TEST(ReconstructPipelineCliGTest, DepthOverlayRegressionBuildsMeshDirectlyFromDepthMaps)
{
    const QString script = readSourceFile(
        QStringLiteral("scripts/validation/run_depth_overlay_regression.ps1"));

    expectContainsAll(script,
                      {"--mvs-depth-only",
                       "mesh_reconstruct_cli.exe",
                       "--source-data", "depth_maps",
                       "--depth-map-dir",
                       "reconstruction_mode",
                       "depth_tsdf",
                       "actual_mesh_algorithm"});
}

TEST(ReconstructPipelineCliGTest, NonEmptyOutputDirRequiresForce)
{
    const QString exe = executablePath(PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    QDir().mkpath(outputDir);
    writeTextFile(QDir(outputDir).filePath(QStringLiteral("existing.txt")), QStringLiteral("keep"));
    const QString missingList = QDir(tempDir.path()).filePath(QStringLiteral("missing.lis"));

    const CliResult result = runCli(exe, {missingList, QStringLiteral("--output-dir"), outputDir});
    EXPECT_NE(result.exitCode, 0);
    expectContainsAll(combinedOutput(result), {"输出目录", "非空"});

    const CliResult forced = runCli(exe, {missingList, QStringLiteral("--output-dir"), outputDir, QStringLiteral("--force")});
    EXPECT_NE(forced.exitCode, 0);
    expectContainsAll(combinedOutput(forced), {"列表读取失败"});
    expectNotContainsAll(combinedOutput(forced), {"非空"});
}

TEST(ReconstructPipelineCliGTest, QuotedLisPathsSupportSpacesCommasAndUnicode)
{
    const QString exe = executablePath(PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString root = tempDir.path();

    const QString imageWithSpace = QDir(root).filePath(QStringLiteral("影像 一.png"));
    const QString cameraWithSpace = QDir(root).filePath(QStringLiteral("相机 一.tsai"));
    writeBytesFile(imageWithSpace, QByteArray("placeholder"));
    writeTsaiCamera(cameraWithSpace);
    const QString shellList = QDir(root).filePath(QStringLiteral("input_shell.lis"));
    writeTextFile(shellList, QStringLiteral("'影像 一.png' '相机 一.tsai'\n"));

    const CliResult shellResult = runCli(exe, {shellList, QStringLiteral("--output-dir"), QDir(root).filePath(QStringLiteral("out_shell"))});
    EXPECT_NE(shellResult.exitCode, 0);
    expectContainsAll(combinedOutput(shellResult), {"至少需要 2 组"});
    expectNotContainsAll(combinedOutput(shellResult), {"需要 '<image> <camera.tsai>'", "影像不存在"});

    const QString imageWithComma = QDir(root).filePath(QStringLiteral("影像, 一.png"));
    const QString cameraCsv = QDir(root).filePath(QStringLiteral("相机, 一.tsai"));
    writeBytesFile(imageWithComma, QByteArray("placeholder"));
    writeTsaiCamera(cameraCsv);
    const QString csvList = QDir(root).filePath(QStringLiteral("input_csv.lis"));
    writeTextFile(csvList, QStringLiteral("\"影像, 一.png\",\"相机, 一.tsai\"\n"));

    const CliResult csvResult = runCli(exe, {csvList, QStringLiteral("--output-dir"), QDir(root).filePath(QStringLiteral("out_csv"))});
    EXPECT_NE(csvResult.exitCode, 0);
    expectContainsAll(combinedOutput(csvResult), {"至少需要 2 组"});
    expectNotContainsAll(combinedOutput(csvResult), {"影像不存在"});
}

TEST(PhotogrammetryWorkflowCliGTest, AerialTriangulationCliAcceptsImageOnlyListForDryRun)
{
    const QString exe = executablePath(PLASCAN_AERIAL_TRIANGULATION_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString root = tempDir.path();
    const QString image0 = QDir(root).filePath(QStringLiteral("templeSR0001.png"));
    const QString image1 = QDir(root).filePath(QStringLiteral("templeSR0002.png"));
    writeBytesFile(image0, QByteArray("placeholder"));
    writeBytesFile(image1, QByteArray("placeholder"));
    const QString maskDir = QDir(root).filePath(QStringLiteral("masks"));
    ASSERT_TRUE(QDir().mkpath(maskDir));
    writeBytesFile(QDir(maskDir).filePath(QStringLiteral("templeSR0001_mask.png")), QByteArray("mask"));
    writeBytesFile(QDir(maskDir).filePath(QStringLiteral("templeSR0002_mask.png")), QByteArray("mask"));

    const QString list = QDir(root).filePath(QStringLiteral("images_only.lis"));
    writeTextFile(list, QStringLiteral("templeSR0001.png\ntempleSR0002.png\n"));

    const QString outputDir = QDir(root).filePath(QStringLiteral("at_out"));
    const QString cameraExportDir =
        QDir(root).filePath(QStringLiteral("final ba cameras"));
    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), list,
        QStringLiteral("--output-dir"), outputDir,
        QStringLiteral("--project"), QDir(root).filePath(QStringLiteral("headless.plascan")),
        QStringLiteral("--dry-run-config"),
        QStringLiteral("--quality"), QStringLiteral("highest"),
        QStringLiteral("--keypoint-limit"), QStringLiteral("40000"),
        QStringLiteral("--tiepoint-limit"), QStringLiteral("4000"),
        QStringLiteral("--mask-dir"), maskDir,
        QStringLiteral("--mask-apply-mode"), QStringLiteral("keypoints"),
        QStringLiteral("--export-camera-dir"), cameraExportDir,
        QStringLiteral("--no-reset-alignment"),
        QStringLiteral("--no-auto-generate-missing-matches"),
        QStringLiteral("--force"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectContainsAll(combinedOutput(result), {"aerial_triangulation_cli_report.json", "dry_run"});
    expectNotContainsAll(combinedOutput(result), {"需要 '<image> <camera.tsai>'"});
    const QString reportPath = reportedPath(result, QStringLiteral("aerial_triangulation_cli_report.json"));
    ASSERT_FALSE(reportPath.isEmpty()) << qPrintable(combinedOutput(result));
    const QJsonObject report = readJsonObject(reportPath);
    QDir chunkRoot = QFileInfo(reportPath).absoluteDir();
    ASSERT_TRUE(chunkRoot.cdUp());
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    EXPECT_TRUE(report.value(QStringLiteral("dry_run")).toBool());
    EXPECT_TRUE(report.value(QStringLiteral("camera_export_requested")).toBool());
    EXPECT_FALSE(report.value(QStringLiteral("camera_export_performed")).toBool());
    EXPECT_EQ(QDir::cleanPath(report.value(QStringLiteral("camera_export_dir")).toString()),
              QDir::cleanPath(cameraExportDir));
    EXPECT_FALSE(QFileInfo::exists(cameraExportDir));
    EXPECT_EQ(report.value(QStringLiteral("image_count")).toInt(), 2);
    const QJsonObject resolved = report.value(QStringLiteral("resolved_settings")).toObject();
    EXPECT_EQ(resolved.value(QStringLiteral("quality")).toString(), QStringLiteral("highest"));
    EXPECT_EQ(resolved.value(QStringLiteral("keypoint_limit")).toInt(), 40000);
    EXPECT_EQ(resolved.value(QStringLiteral("tiepoint_limit")).toInt(), 4000);
    EXPECT_TRUE(resolved.value(QStringLiteral("reuse_existing_matches")).toBool());
    EXPECT_EQ(resolved.value(QStringLiteral("tie_point_preparation")).toString(),
              QStringLiteral("fill_missing"));
    const QJsonObject tiePointContext = report.value(QStringLiteral("tie_point_context")).toObject();
    EXPECT_EQ(tiePointContext.value(QStringLiteral("mask_count")).toInt(), 6);
    EXPECT_FALSE(tiePointContext.contains(QStringLiteral("feature_dir")));
    EXPECT_EQ(QDir::cleanPath(tiePointContext.value(QStringLiteral("match_dir")).toString()),
              QDir::cleanPath(chunkRoot.filePath(QStringLiteral("assets/image_matches"))));
    const QJsonObject tiePointOptions = report.value(QStringLiteral("tie_point_options")).toObject();
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("max_keypoints")).toInt(), 40000);
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("keypoint_limit_per_megapixel")).toInt(), 0);
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("max_tie_points_per_image")).toInt(), 4000);
    const QJsonObject pipelineInput = report.value(QStringLiteral("pipeline_input")).toObject();
    EXPECT_TRUE(pipelineInput.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_EQ(QDir::cleanPath(pipelineInput.value(QStringLiteral("output_dir")).toString()),
              QDir::cleanPath(chunkRoot.filePath(QStringLiteral("reconstruction/sparse/sfm_sparse"))));
    const QString sharedImagesDir = QDir(root).filePath(QStringLiteral("headless.files/shared/images"));
    EXPECT_TRUE(!QFileInfo::exists(sharedImagesDir) ||
                QDir(sharedImagesDir).entryList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot).isEmpty())
        << "--dry-run-config must not import or copy source images";
}

TEST(PhotogrammetryWorkflowCliGTest, AerialTriangulationCliAllowsSequenceReferenceWithoutCameraFiles)
{
    const QString exe = executablePath(PLASCAN_AERIAL_TRIANGULATION_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString root = tempDir.path();
    const QString image0 = QDir(root).filePath(QStringLiteral("dinoSR0001.png"));
    const QString image1 = QDir(root).filePath(QStringLiteral("dinoSR0002.png"));
    const QString image2 = QDir(root).filePath(QStringLiteral("dinoSR0003.png"));
    writeBytesFile(image0, QByteArray("placeholder"));
    writeBytesFile(image1, QByteArray("placeholder"));
    writeBytesFile(image2, QByteArray("placeholder"));

    const QString list = QDir(root).filePath(QStringLiteral("sequence_images.lis"));
    writeTextFile(list, QStringLiteral("dinoSR0001.png\ndinoSR0002.png\ndinoSR0003.png\n"));

    const QString outputDir = QDir(root).filePath(QStringLiteral("at_sequence_out"));
    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), list,
        QStringLiteral("--output-dir"), outputDir,
        QStringLiteral("--project"), QDir(root).filePath(QStringLiteral("headless.plascan")),
        QStringLiteral("--dry-run-config"),
        QStringLiteral("--reference-preselection"),
        QStringLiteral("--reference-mode"), QStringLiteral("sequence"),
        QStringLiteral("--force"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectNotContainsAll(combinedOutput(result), {"参考预选需要完整相机文件"});
    const QString reportPath = reportedPath(result, QStringLiteral("aerial_triangulation_cli_report.json"));
    ASSERT_FALSE(reportPath.isEmpty()) << qPrintable(combinedOutput(result));
    const QJsonObject report = readJsonObject(reportPath);
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    const QJsonObject pipelineInput = report.value(QStringLiteral("pipeline_input")).toObject();
    EXPECT_TRUE(pipelineInput.value(QStringLiteral("use_sequence_pose_recovery")).toBool());
    EXPECT_FALSE(pipelineInput.value(QStringLiteral("sequence_loop_closure")).toBool());
}

TEST(MeshReconstructCliGTest, UsesSharedModelWorkflowEntry)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/workflows/CMakeLists.txt"));
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mesh_reconstruct.cpp"));

    expectContainsAll(cmake, {
        "mesh_reconstruct_cli",
        "cli_mesh_reconstruct.cpp",
        "meshing",
    });
    expectContainsAll(source, {
        "--source-data",
        "--point-cloud",
        "--depth-map-dir",
        "--dense-cloud",
        "--sparse-scaffold",
        "--sparse-points-json",
        "--output-dir",
        "--settings-json",
        "--settings-key",
        "xjw::mesh::workflow::ModelBuildRequest",
        "xjw::mesh::workflow::buildModel",
        "reconstruction_mode",
        "depth_tsdf",
        "sparseScaffoldPointCloudPath",
        "sparseScaffoldPointsPath",
    });
}

TEST(MeshReconstructCliGTest, BuildsModelFromGuiStyleSettingsJson)
{
    const QString exe = executablePath(PLASCAN_MESH_RECONSTRUCT_CLI_PATH);
    ASSERT_FALSE(exe.isEmpty()) << "mesh_reconstruct_cli target is unavailable";
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString input_ply = QDir(temp_dir.path()).filePath(QStringLiteral("grid.ply"));
    const QString output_dir = QDir(temp_dir.path()).filePath(QStringLiteral("model"));
    const QString settings_path = QDir(temp_dir.path()).filePath(QStringLiteral("settings.json"));

    QVector<Point3f> points;
    for (int y = 0; y < 20; ++y)
    {
        for (int x = 0; x < 20; ++x)
        {
            points.push_back(Point3f{static_cast<float>(x),
                                     static_cast<float>(y),
                                     0.05f * static_cast<float>(x + y)});
        }
    }
    writeBinaryPly(input_ply, points);

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("point_cloud");
    settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    settings[QStringLiteral("method")] = QStringLiteral("Height Grid");
    settings[QStringLiteral("meshResolution")] = 64;
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("disabled");
    QJsonObject root;
    root[QStringLiteral("generate_model")] = settings;
    writeTextFile(settings_path,
                  QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented)));

    const CliResult result = runCli(exe, {
        QStringLiteral("--source-data"), QStringLiteral("point_cloud"),
        QStringLiteral("--point-cloud"), input_ply,
        QStringLiteral("--output-dir"), output_dir,
        QStringLiteral("--settings-json"), settings_path,
        QStringLiteral("--settings-key"), QStringLiteral("generate_model"),
    }, 120000);

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectContainsAll(result.stdoutText, {
        R"("ok": true)",
        R"("mesh_algorithm": "height_grid")",
        R"("model_output_policy": "create_versioned_result")",
        R"("model_run_id": )",
    });
    const QDir runs_dir(QDir(output_dir).filePath(QStringLiteral("model_runs")));
    const QStringList run_directories = runs_dir.entryList(
        QDir::Dirs | QDir::NoDotAndDotDot);
    ASSERT_EQ(run_directories.size(), 1);
    const QString run_root = runs_dir.filePath(run_directories.front());
    EXPECT_TRUE(QFileInfo::exists(
        QDir(run_root).filePath(QStringLiteral("products/model_from_mesh.ply"))));
    EXPECT_TRUE(QFileInfo::exists(
        QDir(run_root).filePath(QStringLiteral("model_result.json"))));
}

TEST(ThreeDReconstructionCliContractTest, TargetExistsAndThreeDOnlyModeSkipsTerrain)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/workflows/CMakeLists.txt"));
    const QString source =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.h"));

    expectContainsAll(cmake, {
        "three_d_reconstruction_cli",
        "cli_reconstruct_pipeline.cpp",
        "PLASCAN_THREE_D_ONLY",
    });
    expectContainsAll(source, {
        "PLASCAN_THREE_D_ONLY",
        "AerialTriangulationOptions",
        "AerialTriangulationWorkflow::run",
        "buildDepthGenConfig",
        "buildMeshAndOptionalTexture",
    });
    expectMatches(source, R"(#ifndef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)#endif)");
    expectMatches(source, R"(#ifdef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)three_d_reconstruction_output)");
}

TEST(ThreeDReconstructionCliContractTest, StopsBeforeMvsWhenSfmOutputsAreInsufficient)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));

    expectMatches(source,
                  R"(kMinimumSparsePointsForDenseWorkflow\s*=\s*20.*?if\s*\(\s*sfmResult\.numPoints3D\s*<\s*kMinimumSparsePointsForDenseWorkflow\s*\).*?SFM 稀疏点云点数过少.*?return cli::EXIT_ALGO_ERR)");
    expectContainsAll(source, {"kMinimumRegisteredImagesForDenseWorkflow = 2"});
    expectMatches(source,
                  R"(if\s*\(\s*views\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumRegisteredImagesForDenseWorkflow\)\s*\).*?SFM 后可用于 MVS 的相机不足.*?return cli::EXIT_ALGO_ERR)");
    expectMatches(source,
                  R"(filtered_sparse_points.*?if\s*\(\s*sparse\.points\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumSparsePointsForDenseWorkflow\)\s*\).*?预处理后的 SFM 稀疏点云点数过少.*?return cli::EXIT_ALGO_ERR)");
}

TEST(ThreeDReconstructionCliContractTest, DefaultsMatchGuiWorkflowAndSupportsStageControl)
{
    const QString source =
        readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.h"))
        + readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionCliOptions.cpp"))
        + readSourceFile(QStringLiteral("src/cli/common/CliConsole.cpp"))
        + readSourceFile(QStringLiteral("src/core/mvs/StreamingDepthFusionService.cpp"));

    expectContainsAll(source, {
        R"(std::string device = "auto")",
        R"(std::string mvsBackend = "auto")",
        "registerConsoleLogger",
        "Logger::instance()->registerSink",
        "--feature-max-image-dim",
        "sfmOptions.featureMaxImageDim = featureMaxImageDim",
        "const int denseMinViewCount",
        "denseSettings.minViews = denseMinViewCount",
        "denseSettings.minConsistentViews = denseMinViewCount",
        "--mvs-fusion-confidence",
        "double mvsFusionConfidence = 0.50",
        "denseSettings.fusionMinConfidence = mvsFusionConfidence",
        "denseSettings.depthConsistency = 1.0f",
        "depthConfig.runFusion = false",
        "fuseDepthMapsStreaming",
        "DepthMapFusion fusion",
        "DenseRefineSettings refineSettings",
        "refineDenseCloud",
        "dense_cloud_refined.ply",
        "meshRequest.pointCloudPath = refinedCloudPathForModel",
        "bool exportObj = true",
        "--skip-texture",
        "int meshResolution = 224",
        "reconstructionConfigForDenseScene(",
        "effectiveSceneProfile",
        "preserveMeshDetail",
        "--stop-after-sfm",
        "--skip-mvs",
        "--skip-mesh",
    });

    const QString stopGuard = sectionBetween(source, "if (stopAfterSfm || skipMvs)", "kMinimumRegisteredImagesForDenseWorkflow");
    expectContainsAll(stopGuard, {
        R"(report[QStringLiteral("status")] = QStringLiteral("ok"))",
        "mvs",
        "mesh",
        "return cli::EXIT_OK",
    });
    EXPECT_LT(indexOfOrFail(source, "stopAfterSfm || skipMvs"),
              indexOfOrFail(source, "kMinimumSparsePointsForDenseWorkflow"));
}

TEST(ThreeDReconstructionCliContractTest, StreamsFusionAndUsesRegisteredCameras)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    const QString fusionService =
        readSourceFile(QStringLiteral("src/core/mvs/StreamingDepthFusionService.cpp"));
    const QString gui = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));

    expectContainsAll(fusionService, {
        "cacheFrameLimit",
        "cachedFrames",
        "cacheFrames",
        "frameLoader",
    });
    expectContainsAll(source, {
        "registeredImagePaths",
        "sfmResult.pendingCamUpdates",
    });
    expectNotContainsAll(source, {
        "cameraByImage.insert(item.imagePath, item.camera);",
    });
    expectContainsAll(gui, {
        "registeredImages",
        "result.pendingCamUpdates.keys()",
        "replaceTiePointResult(result.sparseCloudPath",
    });
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudIsVoxelThinnedBeforeSor)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int preVoxelIndex = source.indexOf(QStringLiteral("大点云预降采样"));
    const int refineIndex = source.indexOf(QStringLiteral("refineDenseCloud(std::move(refineInput)"));
    ASSERT_GE(preVoxelIndex, 0);
    ASSERT_GE(refineIndex, 0);
    EXPECT_LT(preVoxelIndex, refineIndex);
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudIsPreAggregatedBeforePlaPointRefinement)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("voxelDownsampleFusedPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("开始大点云预降采样")));
    EXPECT_TRUE(source.contains(QStringLiteral("完成大点云预降采样")));

    const int preAggregateIndex = source.indexOf(QStringLiteral("voxelDownsampleFusedPointsToTarget"));
    const int preAggregateArgIndex = source.indexOf(QStringLiteral("fusedCloud"), preAggregateIndex);
    const int plaCloudIndex = source.indexOf(QStringLiteral("fusedPointsToPointCloud(fusedCloud"));
    ASSERT_GE(preAggregateIndex, 0);
    ASSERT_GE(preAggregateArgIndex, 0);
    ASSERT_GE(plaCloudIndex, 0);
    EXPECT_LT(preAggregateIndex, plaCloudIndex);
    EXPECT_LT(preAggregateArgIndex, plaCloudIndex);
}

TEST(ReconstructPipelineCliTest, LargeDenseCloudPreAggregationCapsPlaPointRefineInput)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("kMaxRefineInputPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("voxelDownsampleFusedPointsToTarget")));
    EXPECT_TRUE(source.contains(QStringLiteral("targetPoints=%zu")));
}

TEST(ReconstructPipelineCliTest, LongRunningCliProgressIsFlushedAndThrottled)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("lastMeshProgressPercent")));
    EXPECT_TRUE(source.contains(QStringLiteral("lastMeshProgressStage")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::fflush(stdout);")));
}

TEST(ReconstructPipelineCliTest, FinalSummaryReportsElapsedTimings)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/workflows/ReconstructionPipelineRunner.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("timings")));
    EXPECT_TRUE(source.contains(QStringLiteral("total_elapsed_ms")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_total=%.3fs")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_sfm=%.3fs")));
    EXPECT_TRUE(source.contains(QStringLiteral("elapsed_mvs=%.3fs")));
}
