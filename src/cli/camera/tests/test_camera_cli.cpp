#include "CliTestSupport.h"

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
