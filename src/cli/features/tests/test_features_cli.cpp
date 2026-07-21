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
    expectContainsAll(combinedOutput(result), {"match_photos_report.json"});
    expectNotContainsAll(combinedOutput(result), {"需要 '<image> <camera.tsai>'"});
    const QJsonObject report = readJsonObject(QDir(outputDir).filePath(QStringLiteral("match_photos_report.json")));
    EXPECT_TRUE(report.value(QStringLiteral("success")).toBool());
    EXPECT_EQ(report.value(QStringLiteral("image_count")).toInt(), 2);
}

TEST(FeatureMatchCliGTest, BfSiftWritesBaV2SidecarIndicesAndPoints)
{
    const QString exe = executablePath(PLASCAN_FEATURE_MATCH_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString left = QDir(tempDir.path()).filePath(QStringLiteral("left.sift"));
    const QString right = QDir(tempDir.path()).filePath(QStringLiteral("right.sift"));
    const QString out = QDir(tempDir.path()).filePath(QStringLiteral("left__right.match"));

    writeSiftFeature(left,
                     QByteArray("left.jpg"),
                     {{10.0f, 20.0f, 0.9f}, {30.0f, 40.0f, 0.8f}},
                     {{1.0f, 0.0f}, {0.0f, 1.0f}});
    writeSiftFeature(right,
                     QByteArray("right.jpg"),
                     {{11.0f, 21.0f, 0.95f}, {31.0f, 41.0f, 0.85f}},
                     {{1.0f, 0.0f}, {0.0f, 1.0f}});

    const CliResult result = runCli(exe, {
        QStringLiteral("--algorithm"), QStringLiteral("bf"),
        QStringLiteral("--sp1"), left,
        QStringLiteral("--sp2"), right,
        QStringLiteral("--output"), out,
        QStringLiteral("--match-threshold"), QStringLiteral("0.0"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    const QString sidecarPath = out + QStringLiteral(".json");
    ASSERT_TRUE(QFileInfo::exists(sidecarPath)) << qPrintable(combinedOutput(result));
    const QJsonObject sidecar = readJsonObject(sidecarPath);

    EXPECT_EQ(sidecar.value(QStringLiteral("feature_format_version")).toInt(), 2);
    EXPECT_EQ(sidecar.value(QStringLiteral("image0_name")).toString(), QStringLiteral("left.jpg"));
    EXPECT_EQ(sidecar.value(QStringLiteral("image1_name")).toString(), QStringLiteral("right.jpg"));
    EXPECT_EQ(sidecar.value(QStringLiteral("feature0_path")).toString(), left);
    EXPECT_EQ(sidecar.value(QStringLiteral("feature1_path")).toString(), right);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices0")).toArray().size(), 2);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices1")).toArray().size(), 2);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices0")).toArray().at(0).toInt(), 0);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices0")).toArray().at(1).toInt(), 1);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices1")).toArray().at(0).toInt(), 0);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_indices1")).toArray().at(1).toInt(), 1);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_points0")).toArray().at(0).toArray().at(0).toDouble(), 10.0);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_points1")).toArray().at(1).toArray().at(1).toDouble(), 41.0);
    EXPECT_EQ(sidecar.value(QStringLiteral("num_matches")).toInt(), 2);
    EXPECT_EQ(sidecar.value(QStringLiteral("matched_scores")).toArray().size(), 2);
}
