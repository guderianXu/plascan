#include "CliTestSupport.h"

TEST(DenseCloudRefineCliGTest, SourceExposesQualityFilterOptions)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/dense/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/dense/cli_dense_cloud_refine.cpp"));
    const QString service = readSourceFile(
        QStringLiteral("src/core/mvs/DenseCloudRefinementService.cpp"));

    expectContainsAll(cmake, {
        "dense_cloud_refine_cli",
        "cli_dense_cloud_refine.cpp",
    });
    expectContainsAll(source, {
        "--input",
        "--output",
        "--report-json",
        "--terrain-grid-cells",
        "--terrain-min-cell-points",
        "--terrain-min-height-threshold",
        "--terrain-mad-multiplier",
        "--terrain-local-plane-filter",
        "--terrain-local-plane-min-points",
        "--terrain-local-plane-min-residual-threshold",
        "--terrain-local-plane-mad-multiplier",
        "--terrain-filter-passes",
        "--streaming-chunk-mb",
        "int terrainGridCells = 260;",
        "int terrainMinCellPoints = 32;",
        "float terrainMinHeightThreshold = 0.25f;",
        "float terrainMadMultiplier = 3.0f;",
        "bool terrainLocalPlaneFilter = true;",
        "int terrainLocalPlaneMinPoints = 12;",
        "float terrainLocalPlaneMinResidualThreshold = 0.12f;",
        "float terrainLocalPlaneMadMultiplier = 4.0f;",
        "int terrainFilterPasses = 2;",
        "terrain_spike_filter",
        "terrain_filter_passes",
        "pass_reports",
        "local_plane_removed_points",
    });
    expectContainsAll(service, {
        "parseBinaryPlyVertexStreamHeader",
        "readPlyVertexChunk",
        "filterTerrainHeightSpikes",
    });
    expectNotContainsAll(source, {"streamingOptions.localPlaneFilterEnabled = false"});
}

TEST(DenseCloudRefineCliGTest, StreamingCliAppliesLocalPlaneFilter)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("sloped_spike_binary.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("refined.ply"));
    const QString reportJson = QDir(tempDir.path()).filePath(QStringLiteral("report.json"));

    QVector<Point3f> points;
    for (int y = 0; y < 8; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            points.push_back(Point3f{static_cast<float>(x), static_cast<float>(y), 0.15f * x + 0.08f * y});
        }
    }
    points.push_back(Point3f{3.25f, 4.25f, 0.15f * 3.25f + 0.08f * 4.25f + 0.42f});
    writeBinaryPly(inputPly, points);

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"), reportJson,
        QStringLiteral("--terrain-grid-cells"), QStringLiteral("1"),
        QStringLiteral("--terrain-min-cell-points"), QStringLiteral("12"),
        QStringLiteral("--terrain-min-height-threshold"), QStringLiteral("2.0"),
        QStringLiteral("--terrain-mad-multiplier"), QStringLiteral("20.0"),
        QStringLiteral("--terrain-local-plane-filter"),
        QStringLiteral("--terrain-local-plane-min-points"), QStringLiteral("12"),
        QStringLiteral("--terrain-local-plane-min-residual-threshold"), QStringLiteral("0.10"),
        QStringLiteral("--terrain-local-plane-mad-multiplier"), QStringLiteral("4.0"),
    }, 120000);

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    const QJsonObject report = readJsonObject(reportJson);
    EXPECT_EQ(report.value(QStringLiteral("mode")).toString(), QStringLiteral("streaming"));
    EXPECT_EQ(report.value(QStringLiteral("terrain_filter_passes")).toInt(), 2);
    EXPECT_EQ(report.value(QStringLiteral("pass_reports")).toArray().size(), 2);
    EXPECT_EQ(report.value(QStringLiteral("input_points")).toInt(), 65);
    EXPECT_EQ(report.value(QStringLiteral("output_points")).toInt(), 64);
    EXPECT_EQ(report.value(QStringLiteral("terrain_spike_filter")).toObject()
                  .value(QStringLiteral("local_plane_removed_points")).toInt(),
              1);
    EXPECT_TRUE(QFileInfo::exists(outputPly));
}

TEST(DenseCloudRefineCliGTest, StreamingCliPreservesPlyScalarProperties)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("attributed_binary.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("refined_attributed.ply"));
    const QString reportJson = QDir(tempDir.path()).filePath(QStringLiteral("report_attributed.json"));

    writeBinaryPlyWithScalarProperties(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    });

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"), reportJson,
        QStringLiteral("--disable-terrain-spike-filter"),
        QStringLiteral("--terrain-filter-passes"), QStringLiteral("1"),
    }, 120000);

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    const QString outputHeader = readPlyHeader(outputPly);
    expectContainsAll(outputHeader, {
        "property ushort intensity",
        "property float confidence",
    });

    const QJsonObject report = readJsonObject(reportJson);
    EXPECT_EQ(report.value(QStringLiteral("mode")).toString(), QStringLiteral("streaming"));
    EXPECT_EQ(report.value(QStringLiteral("input_points")).toInt(), 4);
    EXPECT_EQ(report.value(QStringLiteral("output_points")).toInt(), 4);
}
