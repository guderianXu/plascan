#include "CliTestSupport.h"

TEST(MatchPhotosCliGTest, DedicatedTargetExposesCurrentMatchingWorkflowOptions)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/features/CMakeLists.txt"));
    const QString common = readSourceFile(QStringLiteral("src/cli/common/cli_photogrammetry_common.h"));
    const QString matchSource = readSourceFile(QStringLiteral("src/cli/features/cli_match_photos.cpp"));

    expectContainsAll(cmake, {
        "match_photos_cli",
        "cli_match_photos.cpp",
        "matchphototask",
    });
    expectContainsAll(common, {
        "readPhotogrammetryImageList",
        "allowImageOnlyRows",
        "cameraPathsForService",
    });
    expectContainsAll(matchSource, {
        "MatchPhotosTask task",
        "--keypoint-limit",
        "--keypoint-limit-per-mpx",
        "--tiepoint-limit",
        "--mask-apply-mode",
        "--guided-image-matching",
        "match_photos_report.json",
    });
}

TEST(MatchPhotosCliGTest, AcceptsImageOnlyListInPlanMode)
{
    const QString exe = executablePath(PLASCAN_MATCH_PHOTOS_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString root = tempDir.path();
    const QString image0 = QDir(root).filePath(QStringLiteral("temple 0001.png"));
    const QString image1 = QDir(root).filePath(QStringLiteral("temple 0002.png"));
    writeBytesFile(image0, QByteArray("placeholder"));
    writeBytesFile(image1, QByteArray("placeholder"));
    const QString list = QDir(root).filePath(QStringLiteral("images_only.lis"));
    writeTextFile(list, QStringLiteral("'temple 0001.png'\n\"temple 0002.png\"\n"));

    const QString outputDir = QDir(root).filePath(QStringLiteral("match_out"));
    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), list,
        QStringLiteral("--output-dir"), outputDir,
        QStringLiteral("--project"), QDir(root).filePath(QStringLiteral("headless.plascan")),
        QStringLiteral("--plan-only"),
        QStringLiteral("--force"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    const QString output = combinedOutput(result);
    expectContainsAll(output, {"match_photos_report.json"});
    expectNotContainsAll(output, {"需要 '<image> <camera.tsai>'"});

    // 报告属于活动 Chunk，而不是用户传入的临时输出目录。直接读取 CLI 回报的
    // 最终路径，保证测试与 GUI/CLI 共用的项目布局一致。
    const QRegularExpression reportPattern(
        QStringLiteral("match_photos_report\\.json=([^\\r\\n]+)"));
    const QRegularExpressionMatch reportMatch = reportPattern.match(output);
    ASSERT_TRUE(reportMatch.hasMatch()) << qPrintable(output);
    const QJsonObject report = readJsonObject(reportMatch.captured(1).trimmed());
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("image_count")).toInt(), 2);
}

TEST(FeatureMatchCliGTest, ExposesOnlyRawImagePimatchContract)
{
    const QString source = readSourceFile(
        QStringLiteral("src/cli/features/cli_feature_match.cpp"));
    expectContainsAll(source, {
        "--left",
        "--right",
        "--output-dir",
        "--device",
        "auto_sift",
        "sift_lightglue",
        "MatchPhotosTask",
        ".pimatch",
    });
    expectNotContainsAll(source, {
        "--sp1",
        "--sp2",
        "QStringLiteral(\"bf\")",
        ".match.json",
        "FeatureFileIO",
    });

    const QString exe = executablePath(PLASCAN_FEATURE_MATCH_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    const CliResult result = runCli(exe, {QStringLiteral("--help")});
    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectContainsAll(combinedOutput(result), {
        "--left",
        "--right",
        "--output-dir",
        "--device",
        "auto_sift",
        "sift_lightglue",
    });
    expectNotContainsAll(combinedOutput(result), {"--sp1", "--sp2", " bf"});
}
