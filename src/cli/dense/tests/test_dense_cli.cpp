#include "CliTestSupport.h"
#include "io/PathIO.h"

#include <QLockFile>

#include <filesystem>

#ifndef PLASCAN_DENSE_MATCH_CLI_PATH
#define PLASCAN_DENSE_MATCH_CLI_PATH ""
#endif

namespace
{

constexpr int kArgumentErrorExitCode = 1;
constexpr int kAlgorithmErrorExitCode = 3;

} // namespace

TEST(DenseMatchCliContractGTest, SourceExposesUnifiedDeviceSelection)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/dense/cli_dense_match.cpp"));

    expectContainsAll(source, {
        "\"--device\"",
        "auto, cpu, cuda, opencl",
        "\"--opencl-device\"",
        "DenseMatchComputeBackend::Cuda",
        "DenseMatchComputeBackend::Cpu",
        "--device 不能与兼容选项 --cuda/--no-cuda 同时使用",
    });
    expectMatches(source, R"(std::string\s+deviceStr\s*=\s*"auto"\s*;)");
    expectMatches(source, R"(parseDenseMatchComputeBackend\s*\(\s*deviceStr\s*\))");
    expectMatches(source, R"(cfg\s*\.\s*computeBackend\s*=\s*computeBackend\s*;)");
    expectMatches(source, R"(cfg\s*\.\s*openClDevice\s*=\s*openClDevice\s*;)");
    expectMatches(source, R"(denseMatchComputeBackendName\s*\(\s*computeBackend\s*\))");
}
TEST(DenseMatchCliGTest, HelpListsUnifiedAndLegacyDeviceOptions)
{
    const QString exe = executablePath(PLASCAN_DENSE_MATCH_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    const CliResult result = runCli(exe, {QStringLiteral("--help")});

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    expectContainsAll(combinedOutput(result), {
        "--device",
        "--opencl-device",
        "--cuda",
        "--no-cuda",
        "auto, cpu, cuda, opencl",
    });
}

TEST(DenseMatchCliGTest, RejectsConflictingUnifiedAndLegacyDeviceOptions)
{
    const QString exe = executablePath(PLASCAN_DENSE_MATCH_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    const CliResult mixedResult = runCli(exe, {
        QStringLiteral("--left"), QStringLiteral("left.tif"),
        QStringLiteral("--right"), QStringLiteral("right.tif"),
        QStringLiteral("--output"), QStringLiteral("output.tif"),
        QStringLiteral("--device"), QStringLiteral("opencl"),
        QStringLiteral("--cuda"),
    });
    EXPECT_EQ(mixedResult.exitCode, kArgumentErrorExitCode)
        << qPrintable(combinedOutput(mixedResult));
    expectContainsAll(combinedOutput(mixedResult), {
        "--device",
        "--cuda/--no-cuda",
    });

    const CliResult legacyResult = runCli(exe, {
        QStringLiteral("--left"), QStringLiteral("left.tif"),
        QStringLiteral("--right"), QStringLiteral("right.tif"),
        QStringLiteral("--output"), QStringLiteral("output.tif"),
        QStringLiteral("--cuda"),
        QStringLiteral("--no-cuda"),
    });
    EXPECT_EQ(legacyResult.exitCode, kArgumentErrorExitCode)
        << qPrintable(combinedOutput(legacyResult));
    expectContainsAll(combinedOutput(legacyResult), {
        "--cuda",
        "--no-cuda",
    });
}

TEST(DenseMatchCliGTest, RejectsExtremeKernelSizesBeforeAllocation)
{
    const QString exe = executablePath(PLASCAN_DENSE_MATCH_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString input = QDir(tempDir.path()).filePath(QStringLiteral("pixel.pgm"));
    QByteArray pgm("P5\n1 1\n255\n");
    pgm.append(static_cast<char>(1));
    writeBytesFile(input, pgm);

    const QString correlationOutput = QDir(tempDir.path()).filePath(
        QStringLiteral("correlation-overflow.tif"));
    const CliResult correlationResult = runCli(exe, {
        QStringLiteral("--left"), input,
        QStringLiteral("--right"), input,
        QStringLiteral("--output"), correlationOutput,
        QStringLiteral("--kernel-w"), QStringLiteral("2147483647"),
        QStringLiteral("--kernel-h"), QStringLiteral("1"),
        QStringLiteral("--no-cuda"),
    });
    EXPECT_EQ(correlationResult.exitCode, kAlgorithmErrorExitCode)
        << qPrintable(combinedOutput(correlationResult));
    expectContainsAll(combinedOutput(correlationResult), {
        "size=1x1",
        "disparity=[0,256)",
        "correlation kernels",
    });
    EXPECT_FALSE(QFileInfo::exists(correlationOutput));

    const QString medianOutput = QDir(tempDir.path()).filePath(
        QStringLiteral("median-overflow.tif"));
    const CliResult medianResult = runCli(exe, {
        QStringLiteral("--left"), input,
        QStringLiteral("--right"), input,
        QStringLiteral("--output"), medianOutput,
        QStringLiteral("--median-filter"), QStringLiteral("2147483647"),
        QStringLiteral("--no-cuda"),
    });
    EXPECT_EQ(medianResult.exitCode, kAlgorithmErrorExitCode)
        << qPrintable(combinedOutput(medianResult));
    expectContainsAll(combinedOutput(medianResult), {
        "size=1x1",
        "disparity=[0,256)",
        "median filter size",
    });
    EXPECT_FALSE(QFileInfo::exists(medianOutput));
}

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

TEST(DenseCloudRefineCliGTest, RejectsReportPathThatAliasesInputWithoutChangingSource)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });
    QFile input(inputPly);
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = input.readAll();
    input.close();

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"),
        QDir(tempDir.path()).filePath(QStringLiteral("./input.ply")),
    });

    EXPECT_EQ(result.exitCode, kArgumentErrorExitCode) << qPrintable(combinedOutput(result));
    EXPECT_FALSE(QFileInfo::exists(outputPly));
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    EXPECT_EQ(input.readAll(), originalBytes);
}

TEST(DenseCloudRefineCliGTest, RejectsReportPathThatAliasesOutput)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"),
        QDir(tempDir.path()).filePath(QStringLiteral("./output.ply")),
    });

    EXPECT_EQ(result.exitCode, kArgumentErrorExitCode) << qPrintable(combinedOutput(result));
    EXPECT_FALSE(QFileInfo::exists(outputPly));
}

TEST(DenseCloudRefineCliGTest, ReportFailureKeepsExistingOutputAndReport)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    const QString reportDirectory = QDir(tempDir.path()).filePath(QStringLiteral("report.json"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });
    QFile oldOutput(outputPly);
    ASSERT_TRUE(oldOutput.open(QIODevice::WriteOnly));
    ASSERT_EQ(oldOutput.write("old-output"), 10);
    oldOutput.close();
    ASSERT_TRUE(QDir().mkpath(reportDirectory));
    QFile reportSentinel(QDir(reportDirectory).filePath(QStringLiteral("keep.txt")));
    ASSERT_TRUE(reportSentinel.open(QIODevice::WriteOnly));
    ASSERT_EQ(reportSentinel.write("old-report"), 10);
    reportSentinel.close();

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"), reportDirectory,
        QStringLiteral("--disable-terrain-spike-filter"),
        QStringLiteral("--terrain-filter-passes"), QStringLiteral("1"),
    });

    EXPECT_EQ(result.exitCode, 2) << qPrintable(combinedOutput(result));
    ASSERT_TRUE(oldOutput.open(QIODevice::ReadOnly));
    EXPECT_EQ(oldOutput.readAll(), QByteArray("old-output"));
    ASSERT_TRUE(reportSentinel.open(QIODevice::ReadOnly));
    EXPECT_EQ(reportSentinel.readAll(), QByteArray("old-report"));
}

TEST(DenseCloudRefineCliGTest, NormalizesMissingIntermediateOutputComponents)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString normalizedOutput = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    const QString normalizedReport = QDir(tempDir.path()).filePath(QStringLiteral("report.json"));
    const QString aliasedOutput = QDir(tempDir.path()).filePath(
        QStringLiteral("missing/../output.ply"));
    const QString aliasedReport = QDir(tempDir.path()).filePath(
        QStringLiteral("other-missing/../report.json"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), aliasedOutput,
        QStringLiteral("--report-json"), aliasedReport,
        QStringLiteral("--disable-terrain-spike-filter"),
        QStringLiteral("--terrain-filter-passes"), QStringLiteral("1"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    EXPECT_TRUE(QFileInfo::exists(normalizedOutput));
    ASSERT_TRUE(QFileInfo::exists(normalizedReport));
    const QString canonicalOutput = QFileInfo(normalizedOutput).canonicalFilePath();
    EXPECT_EQ(readJsonObject(normalizedReport).value(QStringLiteral("output")).toString(),
              QDir::toNativeSeparators(canonicalOutput));
}

TEST(DenseCloudRefineCliGTest, ReportLockKeepsBothExistingArtifactsUnchanged)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPly = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    const QString reportJson = QDir(tempDir.path()).filePath(QStringLiteral("report.json"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });
    QFile output(outputPly);
    ASSERT_TRUE(output.open(QIODevice::WriteOnly));
    ASSERT_EQ(output.write("old-output"), 10);
    output.close();
    QFile report(reportJson);
    ASSERT_TRUE(report.open(QIODevice::WriteOnly));
    ASSERT_EQ(report.write("old-report"), 10);
    report.close();

    QLockFile reportLock(QDir(tempDir.path()).filePath(
        QStringLiteral(".report.json.refine.lock")));
    ASSERT_TRUE(reportLock.tryLock(0));
    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), outputPly,
        QStringLiteral("--report-json"), reportJson,
        QStringLiteral("--disable-terrain-spike-filter"),
    });

    EXPECT_EQ(result.exitCode, 2) << qPrintable(combinedOutput(result));
    ASSERT_TRUE(output.open(QIODevice::ReadOnly));
    EXPECT_EQ(output.readAll(), QByteArray("old-output"));
    ASSERT_TRUE(report.open(QIODevice::ReadOnly));
    EXPECT_EQ(report.readAll(), QByteArray("old-report"));
}

TEST(DenseCloudRefineCliGTest, RejectsLinkedOutputWithoutTouchingTarget)
{
    const QString exe = executablePath(PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString inputPly = QDir(tempDir.path()).filePath(QStringLiteral("input.ply"));
    const QString targetPly = QDir(tempDir.path()).filePath(QStringLiteral("target.ply"));
    const QString linkedOutput = QDir(tempDir.path()).filePath(QStringLiteral("output.ply"));
    const QString reportJson = QDir(tempDir.path()).filePath(QStringLiteral("report.json"));
    writeBinaryPly(inputPly, {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    });
    QFile target(targetPly);
    ASSERT_TRUE(target.open(QIODevice::WriteOnly));
    ASSERT_EQ(target.write("preserve-target"), 15);
    target.close();

    std::error_code linkError;
    std::filesystem::create_symlink(
        xjw::common::io::toFilesystemPath(targetPly),
        xjw::common::io::toFilesystemPath(linkedOutput),
        linkError);
    if (linkError)
    {
        return;
    }

    const CliResult result = runCli(exe, {
        QStringLiteral("--input"), inputPly,
        QStringLiteral("--output"), linkedOutput,
        QStringLiteral("--report-json"), reportJson,
    });

    EXPECT_EQ(result.exitCode, kArgumentErrorExitCode)
        << qPrintable(combinedOutput(result));
    ASSERT_TRUE(target.open(QIODevice::ReadOnly));
    EXPECT_EQ(target.readAll(), QByteArray("preserve-target"));
    EXPECT_FALSE(QFileInfo::exists(reportJson));
}
