#include "CliTestSupport.h"

// 这些测试只覆盖 GUI“工作流程”菜单对应的 CLI。

TEST(WorkflowCliModuleTest, MirrorsGuiWorkflowMenuEntryPoints)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/workflows/CMakeLists.txt"));
    const QString menu = readSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString aerial = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_aerial_triangulation.cpp"));
    const QString mesh = readSourceFile(
        QStringLiteral("src/cli/workflows/cli_mesh_reconstruct.cpp"));

    expectContainsAll(cmake, {
        "aerial_triangulation_cli",
        "mesh_reconstruct_cli",
        "three_d_reconstruction_cli",
        "reconstruct_pipeline_cli",
    });
    expectContainsAll(menu, {
        "空中三角测量...",
        "三维重建",
        "生成模型...",
        "创建 DEM",
        "生成 正射影像",
    });
    expectContainsAll(aerial, {
        "AerialTriangulationWorkflow::run",
        "--dry-run-config",
        "--reference-mode",
        "--auto-generate-missing-matches",
        "--no-reset-alignment",
        "--mask-dir",
        "--assets-dir",
        "--feature-dir",
        "--match-dir",
        "options.assetsDir",
        "options.featureDir",
        "options.matchDir",
        "options.outputDir = outputDir",
        "options.maskPaths = xjw::cli::maskPathsFromDirectory",
        "tie_point_preparation_executed",
        "--no-adaptive-camera-model-fitting",
        "--initial-image-id-1",
        "--initial-image-id-2",
        "options.useInitialPairHint",
        "aerial_triangulation_cli_report.json",
    });
    expectContainsAll(mesh, {
        "GUI 等价模型生成工具",
        "xjw::mesh::workflow::buildModel",
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
                       "--mvs-scene-profile",
                       "--mvs-depth-filter",
                       "--mvs-save-levels",
                       "--mvs-mask-dir",
                       "denseSettings.qualityProfile",
                       "depthConfig.sceneProfile",
                       "depthConfig.depthFilterMode",
                       "depthConfig.saveIntermediatePyramidLevels"});
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
        QStringLiteral("--no-reset-alignment"),
        QStringLiteral("--no-auto-generate-missing-matches"),
        QStringLiteral("--force"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectContainsAll(combinedOutput(result), {"aerial_triangulation_cli_report.json", "dry_run"});
    expectNotContainsAll(combinedOutput(result), {"需要 '<image> <camera.tsai>'"});
    const QJsonObject report =
        readJsonObject(QDir(outputDir).filePath(QStringLiteral("aerial_triangulation_cli_report.json")));
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    EXPECT_TRUE(report.value(QStringLiteral("dry_run")).toBool());
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
    EXPECT_TRUE(tiePointContext.value(QStringLiteral("feature_dir")).toString()
                    .endsWith(QStringLiteral("assets/ip")));
    EXPECT_TRUE(tiePointContext.value(QStringLiteral("match_dir")).toString()
                    .endsWith(QStringLiteral("assets/matches")));
    const QJsonObject tiePointOptions = report.value(QStringLiteral("tie_point_options")).toObject();
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("max_keypoints")).toInt(), 40000);
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("keypoint_limit_per_megapixel")).toInt(), 0);
    EXPECT_EQ(tiePointOptions.value(QStringLiteral("max_tie_points_per_image")).toInt(), 4000);
    const QJsonObject pipelineInput = report.value(QStringLiteral("pipeline_input")).toObject();
    EXPECT_TRUE(pipelineInput.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_EQ(QDir::cleanPath(pipelineInput.value(QStringLiteral("output_dir")).toString()),
              QDir::cleanPath(QDir(outputDir).filePath(QStringLiteral("sfm_sparse"))));
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
    const QJsonObject report =
        readJsonObject(QDir(outputDir).filePath(QStringLiteral("aerial_triangulation_cli_report.json")));
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    const QJsonObject pipelineInput = report.value(QStringLiteral("pipeline_input")).toObject();
    EXPECT_TRUE(pipelineInput.value(QStringLiteral("sequence_loop_closure")).toBool());
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
        "--output-dir",
        "--settings-json",
        "--settings-key",
        "xjw::mesh::workflow::ModelBuildRequest",
        "xjw::mesh::workflow::buildModel",
        "reconstruction_mode",
        "depth_tsdf",
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
    });
    EXPECT_TRUE(QFileInfo::exists(QDir(output_dir).filePath(QStringLiteral("products/model_from_mesh.ply"))));
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
