#include "CliTestSupport.h"

#ifndef PLASCAN_MODEL_QUALITY_CLI_PATH
#define PLASCAN_MODEL_QUALITY_CLI_PATH ""
#endif

TEST(ModelQualityCliGTest, UsesSharedQualityEvaluator)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/quality/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/quality/cli_model_quality.cpp"));

    expectContainsAll(cmake, {
        "model_quality_cli",
        "cli_model_quality.cpp",
        "qc",
    });
    expectContainsAll(source, {
        "--mesh",
        "--image-camera-list",
        "--world-frame",
        "--mvs-workspace",
        "--scene-type",
        "--validation-split",
        "--reference-cloud",
        "--reference-camera-list",
        "--output-dir",
        "ModelImageQualityEvaluator",
    });
}

TEST(ModelQualityCliGTest, RequiresWorldFrameAndRendersExternalTsaiWithPlaCamera)
{
    const QString executable = executablePath(PLASCAN_MODEL_QUALITY_CLI_PATH);
    SKIP_IF_MISSING_EXECUTABLE(executable);

    const QString temporary_root = QDir(repoRoot()).filePath(QStringLiteral("build/tmp"));
    ASSERT_TRUE(QDir().mkpath(temporary_root));
    QTemporaryDir directory(QDir(temporary_root).filePath(QStringLiteral("model-quality-cli-XXXXXX")));
    ASSERT_TRUE(directory.isValid());

    const QString mesh = QDir(directory.path()).filePath(QStringLiteral("triangle.ply"));
    writeTextFile(mesh,
                  QStringLiteral("ply\nformat ascii 1.0\nelement vertex 3\n"
                                 "property float x\nproperty float y\nproperty float z\n"
                                 "element face 1\nproperty list uchar int vertex_indices\n"
                                 "end_header\n-0.5 -0.5 2\n0.5 -0.5 2\n0 0.5 2\n3 0 1 2\n"));
    const QString first_image = QDir(directory.path()).filePath(QStringLiteral("first.ppm"));
    const QString second_image = QDir(directory.path()).filePath(QStringLiteral("second.ppm"));
    QByteArray ppm("P6\n128 128\n255\n");
    ppm.append(QByteArray(128 * 128 * 3, char(128)));
    writeBytesFile(first_image, ppm);
    writeBytesFile(second_image, ppm);
    const QString camera = QDir(directory.path()).filePath(QStringLiteral("camera.tsai"));
    writeTsaiCamera(camera);
    const QString list = QDir(directory.path()).filePath(QStringLiteral("images.lis"));
    writeTextFile(list, first_image + QLatin1Char(' ') + camera + QLatin1Char('\n') +
                            second_image + QLatin1Char(' ') + camera + QLatin1Char('\n'));
    const QString output = QDir(directory.path()).filePath(QStringLiteral("quality"));
    const QStringList arguments{QStringLiteral("--mesh"), mesh,
                                QStringLiteral("--image-camera-list"), list,
                                QStringLiteral("--scene-type"), QStringLiteral("aerial"),
                                QStringLiteral("--validation-split"), QStringLiteral("all"),
                                QStringLiteral("--output-dir"), output};

    const CliResult missing_frame = runCli(executable, arguments);
    EXPECT_EQ(missing_frame.exitCode, 1);
    EXPECT_TRUE(combinedOutput(missing_frame).contains(QStringLiteral("--world-frame")));

    QStringList explicit_frame_arguments = arguments;
    explicit_frame_arguments << QStringLiteral("--world-frame") << QStringLiteral("quality-test-world");
    const CliResult rendered = runCli(executable, explicit_frame_arguments);
    EXPECT_TRUE(QFileInfo::exists(QDir(output).filePath(QStringLiteral("model_quality_report.json"))))
        << qPrintable(combinedOutput(rendered));
    EXPECT_FALSE(combinedOutput(rendered).contains(QStringLiteral("验收相机读取失败")));
}
