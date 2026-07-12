#include <gtest/gtest.h>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QIODevice>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QVector>

#include <zip.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>

#ifndef PLASCAN_MESH_RECONSTRUCT_CLI_PATH
#define PLASCAN_MESH_RECONSTRUCT_CLI_PATH ""
#endif

namespace
{

struct CliResult
{
    int exitCode = -1;
    QString stdoutText;
    QString stderrText;
};

struct Point3f
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

QString repoRoot()
{
    return QStringLiteral(PLASCAN_SOURCE_DIR);
}

QString readSourceFile(const QString &relativePath)
{
    QFile file(QDir(repoRoot()).filePath(relativePath));
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(file.fileName());
    if (!file.isOpen())
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QString readTextFile(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QString utf8(const char *text)
{
    return QString::fromUtf8(text);
}

void expectContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_TRUE(text.contains(utf8(needle))) << needle;
    }
}

void expectNotContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_FALSE(text.contains(utf8(needle))) << needle;
    }
}

QString executablePath(const char *path)
{
    return utf8(path).trimmed();
}

#define SKIP_IF_MISSING_EXECUTABLE(path)                                                               \
    do                                                                                                 \
    {                                                                                                  \
        if ((path).isEmpty() || !QFileInfo::exists(path))                                              \
        {                                                                                              \
            GTEST_SKIP() << "CLI executable not available: " << qPrintable(path);                      \
        }                                                                                              \
    } while (false)

CliResult runCli(const QString &program, const QStringList &arguments, int timeoutMs = 60000)
{
    QProcess process;
    process.setProgram(program);
    process.setArguments(arguments);
    process.setWorkingDirectory(repoRoot());
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start();

    if (!process.waitForStarted(timeoutMs))
    {
        return CliResult{-1, QString(), process.errorString()};
    }
    if (!process.waitForFinished(timeoutMs))
    {
        process.kill();
        process.waitForFinished(5000);
        return CliResult{-1, QString::fromUtf8(process.readAllStandardOutput()),
                         QStringLiteral("process timeout: %1").arg(program)};
    }

    return CliResult{process.exitCode(),
                     QString::fromUtf8(process.readAllStandardOutput()),
                     QString::fromUtf8(process.readAllStandardError())};
}

QString combinedOutput(const CliResult &result)
{
    return result.stdoutText + result.stderrText;
}

void writeTextFile(const QString &path, const QString &text)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text)) << qPrintable(path);
    file.write(text.toUtf8());
}

void writeBytesFile(const QString &path, const QByteArray &bytes)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    file.write(bytes);
}

QJsonObject readJsonObject(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QJsonObject();
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    EXPECT_TRUE(doc.isObject()) << qPrintable(path);
    return doc.object();
}

void writeTsaiCamera(const QString &path)
{
    writeTextFile(path,
                  QStringLiteral("VERSION_3\n"
                                 "PINHOLE\n"
                                 "TSAI\n"
                                 "fu = 100\n"
                                 "fv = 100\n"
                                 "cu = 50\n"
                                 "cv = 50\n"
                                 "u_direction = 1 0 0\n"
                                 "v_direction = 0 1 0\n"
                                 "w_direction = 0 0 1\n"
                                 "C = 0 0 0\n"
                                 "R = 1 0 0 0 1 0 0 0 1\n"
                                 "pitch = 1\n"));
}

void writeUInt32LE(QFile *file, quint32 value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeUInt16LE(QFile *file, quint16 value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeFloatLE(QFile *file, float value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeSiftFeature(const QString &path,
                      const QByteArray &imageName,
                      const QVector<QVector<float>> &keypoints,
                      const QVector<QVector<float>> &descriptors)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    file.write("SFTB", 4);
    writeUInt32LE(&file, 1);
    writeUInt32LE(&file, static_cast<quint32>(imageName.size()));
    file.write(imageName);
    writeUInt32LE(&file, static_cast<quint32>(keypoints.size()));
    for (const QVector<float> &keypoint : keypoints)
    {
        ASSERT_EQ(keypoint.size(), 3);
        writeFloatLE(&file, keypoint[0]);
        writeFloatLE(&file, keypoint[1]);
        writeFloatLE(&file, keypoint[2]);
    }
    const quint32 descriptorDim = descriptors.isEmpty() ? 0u : static_cast<quint32>(descriptors.front().size());
    writeUInt32LE(&file, descriptorDim);
    for (const QVector<float> &row : descriptors)
    {
        ASSERT_EQ(static_cast<quint32>(row.size()), descriptorDim);
        for (float value : row)
        {
            writeFloatLE(&file, value);
        }
    }
}

void writeBinaryPly(const QString &path, const QVector<Point3f> &points)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    const QByteArray header = QStringLiteral("ply\n"
                                            "format binary_little_endian 1.0\n"
                                            "element vertex %1\n"
                                            "property float x\n"
                                            "property float y\n"
                                            "property float z\n"
                                            "end_header\n")
                                  .arg(points.size())
                                  .toUtf8();
    file.write(header);
    for (const Point3f &point : points)
    {
        writeFloatLE(&file, point.x);
        writeFloatLE(&file, point.y);
        writeFloatLE(&file, point.z);
    }
}

void writeBinaryPlyWithScalarProperties(const QString &path, const QVector<Point3f> &points)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    const QByteArray header = QStringLiteral("ply\n"
                                            "format binary_little_endian 1.0\n"
                                            "element vertex %1\n"
                                            "property float x\n"
                                            "property float y\n"
                                            "property float z\n"
                                            "property ushort intensity\n"
                                            "property float confidence\n"
                                            "end_header\n")
                                  .arg(points.size())
                                  .toUtf8();
    file.write(header);
    for (int i = 0; i < points.size(); ++i)
    {
        const Point3f &point = points.at(i);
        writeFloatLE(&file, point.x);
        writeFloatLE(&file, point.y);
        writeFloatLE(&file, point.z);
        writeUInt16LE(&file, static_cast<quint16>(1000 + i));
        writeFloatLE(&file, 0.5f + 0.1f * static_cast<float>(i));
    }
}

QString readPlyHeader(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QString();
    }
    const QByteArray prefix = file.read(4096);
    const int endHeader = prefix.indexOf("end_header\n");
    EXPECT_GE(endHeader, 0) << qPrintable(path);
    if (endHeader < 0)
    {
        return QString::fromUtf8(prefix);
    }
    return QString::fromUtf8(prefix.left(endHeader + static_cast<int>(QByteArray("end_header\n").size())));
}

void writeZipEntry(const QString &zipPath, const QString &entryName, const QByteArray &contents)
{
    int errorCode = 0;
    const QByteArray nativeZipPath = QFile::encodeName(zipPath);
    zip_t *archive = zip_open(nativeZipPath.constData(), ZIP_CREATE | ZIP_TRUNCATE, &errorCode);
    ASSERT_NE(archive, nullptr) << qPrintable(zipPath) << " zip_error=" << errorCode;

    zip_source_t *source = zip_source_buffer(archive, contents.constData(), contents.size(), 0);
    ASSERT_NE(source, nullptr);
    const QByteArray nativeEntryName = entryName.toUtf8();
    ASSERT_GE(zip_file_add(archive, nativeEntryName.constData(), source, ZIP_FL_OVERWRITE), 0);
    ASSERT_EQ(zip_close(archive), 0);
}

#ifndef PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH
#define PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH ""
#endif

#ifndef PLASCAN_CAMERA_CONVERT_CLI_PATH
#define PLASCAN_CAMERA_CONVERT_CLI_PATH ""
#endif

#ifndef PLASCAN_BUNDLE_ADJUST_CLI_PATH
#define PLASCAN_BUNDLE_ADJUST_CLI_PATH ""
#endif

#ifndef PLASCAN_FEATURE_MATCH_CLI_PATH
#define PLASCAN_FEATURE_MATCH_CLI_PATH ""
#endif

#ifndef PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH
#define PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH ""
#endif

#ifndef PLASCAN_MATCH_PHOTOS_CLI_PATH
#define PLASCAN_MATCH_PHOTOS_CLI_PATH ""
#endif

#ifndef PLASCAN_AERIAL_TRIANGULATION_CLI_PATH
#define PLASCAN_AERIAL_TRIANGULATION_CLI_PATH ""
#endif

} // namespace

TEST(ReconstructPipelineCliGTest, SourceUsesUtf8ProgressAndCapsLargeDenseRefineInput)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

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

TEST(ReconstructPipelineCliGTest, FusedPreAggregationUsesPlaPointVoxelGrid)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
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
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(source, {
        "frameCount <= 32",
        "std::min(fusionCfg.minNumPixels, 2)",
    });
}

TEST(ReconstructPipelineCliGTest, Arbitrary3dMeshNeverFallsBackToHeightGrid)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(source, {
        "meshRequest.reconstruction.allowHeightGridFallback = false;",
        "meshRequest.reconstruction.orientNormalsForClosedSurface = true;",
    });
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

TEST(PhotogrammetryWorkflowCliGTest, DedicatedCliTargetsExposeCurrentWorkflowOptions)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString common = readSourceFile(QStringLiteral("src/cli/cli_photogrammetry_common.h"));
    const QString matchSource = readSourceFile(QStringLiteral("src/cli/cli_match_photos.cpp"));
    const QString atSource = readSourceFile(QStringLiteral("src/cli/cli_aerial_triangulation.cpp"));

    expectContainsAll(cmake, {
        "match_photos_cli",
        "cli_match_photos.cpp",
        "aerial_triangulation_cli",
        "cli_aerial_triangulation.cpp",
        "matchphototask",
        "AerialTriangulationWorkflow.cpp",
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
    expectContainsAll(atSource, {
        "AerialTriangulationWorkflow::run",
        "--dry-run-config",
        "--reference-mode",
        "--auto-generate-missing-matches",
        "--no-reset-alignment",
        "--mask-dir",
        "options.assetsDir",
        "options.featureDir",
        "options.matchDir",
        "options.maskPaths = xjw::cli::maskPathsFromDirectory",
        "tie_point_preparation_executed",
        "--no-adaptive-camera-model-fitting",
        "aerial_triangulation_cli_report.json",
    });
}

TEST(PhotogrammetryWorkflowCliGTest, MatchPhotosCliAcceptsImageOnlyListInPlanMode)
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
    EXPECT_EQ(resolved.value(QStringLiteral("resolved_keypoint_budget")).toInt(), 40000);
    EXPECT_EQ(resolved.value(QStringLiteral("resolved_tiepoint_limit")).toInt(), 4000);
    EXPECT_EQ(resolved.value(QStringLiteral("tie_point_preparation")).toString(),
              QStringLiteral("skipped_reuse_only"));
    const QJsonObject tiePointContext = report.value(QStringLiteral("tie_point_context")).toObject();
    EXPECT_EQ(tiePointContext.value(QStringLiteral("mask_count")).toInt(), 6);
    EXPECT_TRUE(tiePointContext.value(QStringLiteral("feature_dir")).toString()
                    .endsWith(QStringLiteral("assets/ip")));
    EXPECT_TRUE(tiePointContext.value(QStringLiteral("match_dir")).toString()
                    .endsWith(QStringLiteral("assets/matches")));
    const QJsonObject serviceOptions = report.value(QStringLiteral("service_options")).toObject();
    EXPECT_TRUE(serviceOptions.contains(QStringLiteral("tie_point_feature_max_keypoints")));
    EXPECT_TRUE(serviceOptions.contains(QStringLiteral("tie_point_keypoint_limit_per_megapixel")));
    EXPECT_TRUE(serviceOptions.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
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
    const QJsonObject serviceOptions = report.value(QStringLiteral("service_options")).toObject();
    EXPECT_TRUE(serviceOptions.value(QStringLiteral("known_camera_sequence_loop_closure")).toBool());
}

TEST(BundleAdjustCliGTest, SourceExposesLidarCompareOptionsAndHeadlessDefaults)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_bundle_adjust.cpp"));

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

TEST(DenseCloudRefineCliGTest, SourceExposesQualityFilterOptions)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_dense_cloud_refine.cpp"));

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
        "int terrain_grid_cells = 260;",
        "int terrain_min_cell_points = 32;",
        "float terrain_min_height_threshold = 0.25f;",
        "float terrain_mad_multiplier = 3.0f;",
        "bool terrain_local_plane_filter = true;",
        "int terrain_local_plane_min_points = 12;",
        "float terrain_local_plane_min_residual_threshold = 0.12f;",
        "float terrain_local_plane_mad_multiplier = 4.0f;",
        "int terrain_filter_passes = 2;",
        "parseBinaryPlyVertexStreamHeader",
        "readPlyVertexChunk",
        "filterTerrainHeightSpikes",
        "terrain_spike_filter",
        "terrain_filter_passes",
        "pass_reports",
        "local_plane_removed_points",
    });
    expectNotContainsAll(source, {"streamingOptions.localPlaneFilterEnabled = false"});
}

TEST(MeshReconstructCliGTest, UsesSharedModelWorkflowEntry)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_mesh_reconstruct.cpp"));

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

TEST(CameraConvertCliGTest, ListsFormatsAndRejectsExistingOutputWithoutOverwrite)
{
    const QString exe = executablePath(PLASCAN_CAMERA_CONVERT_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    const CliResult list = runCli(exe, {QStringLiteral("--list-formats")});
    EXPECT_EQ(list.exitCode, 0) << qPrintable(combinedOutput(list));
    expectContainsAll(list.stdoutText, {
        "middlebury-par",
        "epfl-camera",
        "colmap-text",
        "metashape-xml",
    });

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString source = QDir(tempDir.path()).filePath(QStringLiteral("epfl"));
    QDir().mkpath(source);
    writeBytesFile(QDir(source).filePath(QStringLiteral("rdimage.000.ppm")), QByteArray("fake"));
    writeTextFile(QDir(source).filePath(QStringLiteral("rdimage.000.ppm.camera")),
                  QStringLiteral("100 0 50\n0 100 50\n0 0 1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n0 0 0\n"));
    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("out"));
    QDir().mkpath(outputDir);
    writeTextFile(QDir(outputDir).filePath(QStringLiteral("keep.txt")), QStringLiteral("keep"));

    const CliResult existing = runCli(exe, {
        QStringLiteral("--format"), QStringLiteral("epfl-camera"),
        QStringLiteral("--input"), source,
        QStringLiteral("--output-dir"), outputDir,
    });
    EXPECT_NE(existing.exitCode, 0);
    expectContainsAll(combinedOutput(existing), {"非空"});
}

TEST(CameraConvertCliGTest, MiddleburyAndColmapConversionsWritePlaScanInputs)
{
    const QString exe = executablePath(PLASCAN_CAMERA_CONVERT_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString root = tempDir.path();

    const QString middlebury = QDir(root).filePath(QStringLiteral("dinoSparseRing"));
    QDir().mkpath(middlebury);
    writeBytesFile(QDir(middlebury).filePath(QStringLiteral("dinoSR0001.png")), QByteArray("fake"));
    writeBytesFile(QDir(middlebury).filePath(QStringLiteral("dinoSR0002.png")), QByteArray("fake"));
    writeTextFile(QDir(middlebury).filePath(QStringLiteral("dinoSR_par.txt")),
                  QStringLiteral("2\n"
                                 "dinoSR0001.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 1 2 3\n"
                                 "dinoSR0002.png 121 0 41 0 131 51 0 0 1 1 0 0 0 1 0 0 0 1 4 5 6\n"));
    const QString middleburyOut = QDir(root).filePath(QStringLiteral("plascan_middlebury"));
    const CliResult middleburyResult = runCli(exe, {
        QStringLiteral("--format"), QStringLiteral("middlebury-par"),
        QStringLiteral("--input"), middlebury,
        QStringLiteral("--output-dir"), middleburyOut,
        QStringLiteral("--overwrite"),
    });
    EXPECT_EQ(middleburyResult.exitCode, 0) << qPrintable(combinedOutput(middleburyResult));
    EXPECT_TRUE(QFileInfo::exists(QDir(middleburyOut).filePath(QStringLiteral("image_camera.lis"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(middleburyOut).filePath(QStringLiteral("cameras/dinoSR0001.tsai"))));
    const QJsonObject middleburySummary =
        readJsonObject(QDir(middleburyOut).filePath(QStringLiteral("summary.json")));
    EXPECT_EQ(middleburySummary.value(QStringLiteral("input_format")).toString(), QStringLiteral("middlebury-par"));
    EXPECT_EQ(middleburySummary.value(QStringLiteral("camera_count")).toInt(), 2);

    const QString dataset = QDir(root).filePath(QStringLiteral("south-building"));
    const QString colmap = QDir(dataset).filePath(QStringLiteral("sparse"));
    const QString images = QDir(dataset).filePath(QStringLiteral("images"));
    QDir().mkpath(colmap);
    QDir().mkpath(images);
    writeBytesFile(QDir(images).filePath(QStringLiteral("P1180141.JPG")), QByteArray("fake"));
    writeTextFile(QDir(colmap).filePath(QStringLiteral("cameras.txt")),
                  QStringLiteral("1 SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997\n"));
    writeTextFile(QDir(colmap).filePath(QStringLiteral("images.txt")),
                  QStringLiteral("1 1 0 0 0 10 20 30 1 P1180141.JPG\n0 0 -1\n"));
    writeTextFile(QDir(colmap).filePath(QStringLiteral("points3D.txt")), QStringLiteral("# unused\n"));

    const QString colmapOut = QDir(root).filePath(QStringLiteral("plascan_colmap"));
    const CliResult colmapResult = runCli(exe, {
        QStringLiteral("--format"), QStringLiteral("colmap-text"),
        QStringLiteral("--input"), colmap,
        QStringLiteral("--output-dir"), colmapOut,
        QStringLiteral("--overwrite"),
    });
    EXPECT_EQ(colmapResult.exitCode, 0) << qPrintable(combinedOutput(colmapResult));
    EXPECT_TRUE(QFileInfo::exists(QDir(colmapOut).filePath(QStringLiteral("image_camera.lis"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(colmapOut).filePath(QStringLiteral("cameras/P1180141.tsai"))));
    const QJsonObject colmapSummary = readJsonObject(QDir(colmapOut).filePath(QStringLiteral("summary.json")));
    EXPECT_EQ(colmapSummary.value(QStringLiteral("input_format")).toString(), QStringLiteral("colmap-text"));
    EXPECT_EQ(colmapSummary.value(QStringLiteral("camera_count")).toInt(), 1);
}

TEST(CameraConvertCliGTest, MetashapeChunkZipConversionExportsSupportedDistortion)
{
    const QString exe = executablePath(PLASCAN_CAMERA_CONVERT_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(exe);

    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString dataset = QDir(tempDir.path()).filePath(QStringLiteral("depth_images"));
    const QString images = QDir(dataset).filePath(QStringLiteral("Depthimages"));
    const QString project = QDir(dataset).filePath(QStringLiteral("Metashape/Project_depthimages.files/0"));
    QDir().mkpath(images);
    QDir().mkpath(project);
    writeBytesFile(QDir(images).filePath(QStringLiteral("IMG_0262.JPG")), QByteArray("fake"));
    writeBytesFile(QDir(images).filePath(QStringLiteral("IMG_0263.JPG")), QByteArray("fake"));

    const QString docXml = QStringLiteral(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<document>\n"
        "  <chunk>\n"
        "    <sensors>\n"
        "      <sensor id=\"0\" label=\"RGB\" type=\"frame\">\n"
        "        <resolution width=\"1000\" height=\"800\"/>\n"
        "        <calibration type=\"frame\" class=\"adjusted\">\n"
        "          <f>500</f><cx>10</cx><cy>-20</cy>\n"
        "          <k1>0.01</k1><k2>-0.02</k2><k3>0.03</k3>\n"
        "          <p1>0.0004</p1><p2>-0.0005</p2>\n"
        "        </calibration>\n"
        "      </sensor>\n"
        "    </sensors>\n"
        "    <cameras>\n"
        "      <camera id=\"0\" sensor_id=\"0\" component_id=\"0\" label=\"IMG_0262_0\">\n"
        "        <transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform>\n"
        "      </camera>\n"
        "      <camera id=\"1\" sensor_id=\"0\" component_id=\"0\" label=\"IMG_0263_0\">\n"
        "        <transform>1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1</transform>\n"
        "      </camera>\n"
        "    </cameras>\n"
        "  </chunk>\n"
        "</document>\n");
    writeZipEntry(QDir(project).filePath(QStringLiteral("chunk.zip")),
                  QStringLiteral("doc.xml"),
                  docXml.toUtf8());

    const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("plascan"));
    const CliResult result = runCli(exe, {
        QStringLiteral("--format"), QStringLiteral("auto"),
        QStringLiteral("--input"), dataset,
        QStringLiteral("--output-dir"), outputDir,
        QStringLiteral("--overwrite"),
    });

    EXPECT_EQ(result.exitCode, 0) << qPrintable(combinedOutput(result));
    EXPECT_TRUE(QFileInfo::exists(QDir(outputDir).filePath(QStringLiteral("image_camera.lis"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(outputDir).filePath(QStringLiteral("cameras/IMG_0262.tsai"))));
    expectNotContainsAll(result.stderrText, {"distortion terms are not exported"});
    const QJsonObject summary = readJsonObject(QDir(outputDir).filePath(QStringLiteral("summary.json")));
    EXPECT_EQ(summary.value(QStringLiteral("input_format")).toString(), QStringLiteral("metashape-xml"));
    EXPECT_EQ(summary.value(QStringLiteral("camera_count")).toInt(), 2);
    EXPECT_EQ(summary.value(QStringLiteral("warnings")).toArray().size(), 0);
    const QString tsai = readTextFile(QDir(outputDir).filePath(QStringLiteral("cameras/IMG_0262.tsai")));
    expectContainsAll(tsai, {
        "k1 = 0.01",
        "k2 = -0.02",
        "k3 = 0.03",
        "p1 = 0.0004",
        "p2 = -0.0005",
    });
}
