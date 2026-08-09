#include "CliTestSupport.h"

TEST(BundleAdjustCliGTest, SourceExposesLidarCompareOptionsAndHeadlessDefaults)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/reconstruction/CMakeLists.txt"));
    const QString source = readSourceFile(
        QStringLiteral("src/cli/reconstruction/cli_bundle_adjust.cpp"));

    expectContainsAll(cmake, {
        "bundle_adjust_cli",
        "cli_bundle_adjust.cpp",
    });
    expectContainsAll(source, {
        "--laser-cloud",
        "--laser-missing-normals-as-height-planes",
        "options.laserUseMissingNormalsAsHeightPlanes",
        "--ab-compare",
        "ba_ab_compare.json",
        "quality_gate",
        "reprojection_rms_regressed",
        "--fail-on-quality-gate",
        "failOnQualityGate",
        "cli::EXIT_ALGO_ERR",
        "bool exportEvalPlot = false;",
        "--export-eval-plot",
        "options.exportEvalPlot = exportEvalPlot;",
        "baInput.surveyControlTrackCount > 0",
        "baOptions.enableControlPointConstraints = true",
        "baInput.scaleBarConstraints",
        "baOptions.enableScaleBarConstraints = true",
        "baOptions.scaleBarConstraints = baInput.scaleBarConstraints",
        "survey_control_tracks",
        "scale_bars",
    });
}

TEST(BundleAdjustCliGTest, MissingProjectFailsBeforeCreatingOutput)
{
    const QString exe = executablePath(PLASCAN_BUNDLE_ADJUST_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString outDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    const QString missingProject = QDir(tempDir.path()).filePath(QStringLiteral("missing.plascan"));

    const CliResult result = runCli(exe, {missingProject, QStringLiteral("--output-dir"), outDir});

    EXPECT_NE(result.exitCode, 0);
    EXPECT_FALSE(QFileInfo::exists(outDir));
    expectContainsAll(combinedOutput(result), {"项目文件不存在"});
}

TEST(BundleAdjustCliGTest, FailedStrictAbGateDoesNotWriteLaserCameras)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/reconstruction/cli_bundle_adjust.cpp"));
    const qsizetype gateGuard = source.indexOf(
        QStringLiteral("if (failOnQualityGate && !quality_gate_passed)"));
    ASSERT_GE(gateGuard, 0);
    const qsizetype cameraWriteback = source.indexOf(
        QStringLiteral("projectSession.updateImageCameras"), gateGuard);
    ASSERT_GE(cameraWriteback, 0);
    EXPECT_LT(gateGuard, cameraWriteback);
    EXPECT_NE(source.indexOf(QStringLiteral("质量门禁失败，未写回相机"), gateGuard), -1);
}

TEST(CodeStyleTest, CliBundleAdjustSourceKeepsLinesWithinStyleLimit)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/reconstruction/cli_bundle_adjust.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const QStringList lines = source.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i)
    {
        EXPECT_LE(lines.at(i).size(), 120)
            << "cli_bundle_adjust.cpp:" << (i + 1)
            << " has " << lines.at(i).size() << " characters";
    }
}
