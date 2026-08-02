#include "cli_photogrammetry_common.h"
#include "FinalBaCameraExporter.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QTemporaryDir>

namespace
{

using xjw::cli::parsePhotogrammetryListLine;

xjw::Camera makeFinalCamera(double centerX)
{
    xjw::Camera camera;
    camera.setIntrinsics(900.0, 905.0, 320.0, 240.0);
    camera.setPixelPitch(0.01);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {centerX, 0.0, 2.0});
    return camera;
}

void writePlaceholder(const QString &path)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write("image"), 5);
}

TEST(CliPhotogrammetryCommonTest, ParsesShellAndCsvRows)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"image one.tif\" camera.tsai"), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image one.tif"),
                                  QStringLiteral("camera.tsai")}));

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"image,one.tif\",\"camera\"\"one.tsai\""), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image,one.tif"),
                                  QStringLiteral("camera\"one.tsai")}));
}

TEST(CliPhotogrammetryCommonTest, PreservesWindowsPathSeparators)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("E:\\code\\images\\image.tif E:\\code\\cameras\\camera.tsai"),
        &parts,
        &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("E:\\code\\images\\image.tif"),
                                  QStringLiteral("E:\\code\\cameras\\camera.tsai")}));

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"E:\\data set\\image one.tif\" \"\\\\server\\camera set\\camera.tsai\""),
        &parts,
        &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("E:\\data set\\image one.tif"),
                                  QStringLiteral("\\\\server\\camera set\\camera.tsai")}));
}

TEST(CliPhotogrammetryCommonTest, PreservesTrailingCsvCellAndReportsMalformedRows)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(QStringLiteral("image.tif,"), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image.tif"), QString()}));

    EXPECT_FALSE(parsePhotogrammetryListLine(QStringLiteral("\"image.tif"), &parts, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("引号未闭合")));

    EXPECT_FALSE(parsePhotogrammetryListLine(QStringLiteral("image.tif\\"), &parts, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("行尾转义")));
}

TEST(CliPhotogrammetryCommonTest, SerializesProjectItemsAndReportPairsSeparately)
{
    xjw::cli::PhotogrammetryInputItem item;
    item.imagePath = QStringLiteral("image.tif");
    item.cameraPath = QStringLiteral("camera.tsai");
    item.hasCameraPath = true;
    const std::vector<xjw::cli::PhotogrammetryInputItem> items{item};

    const QJsonObject projectItem = xjw::cli::inputItemsToJson(items).first().toObject();
    EXPECT_EQ(projectItem.value(QStringLiteral("path")).toString(), item.imagePath);
    EXPECT_EQ(projectItem.value(QStringLiteral("camera_path")).toString(), item.cameraPath);

    const QJsonObject reportPair = xjw::cli::inputPairsToJson(items).first().toObject();
    EXPECT_EQ(reportPair.value(QStringLiteral("image")).toString(), item.imagePath);
    EXPECT_EQ(reportPair.value(QStringLiteral("camera")).toString(), item.cameraPath);
}

TEST(CliPhotogrammetryCommonTest, ExportsCompleteFinalBaCameraSetForDirectReuse)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imageA =
        QDir(tempDir.path()).filePath(QStringLiteral("first folder/shared image.png"));
    const QString imageB =
        QDir(tempDir.path()).filePath(QStringLiteral("second folder/shared image.png"));
    writePlaceholder(imageA);
    writePlaceholder(imageB);

    QMap<QString, QJsonObject> metadata;
    metadata.insert(QDir::cleanPath(QFileInfo(imageA).absoluteFilePath()),
                    xjw::cli::cameraToJson(makeFinalCamera(0.0)));
    metadata.insert(QDir::cleanPath(QFileInfo(imageB).absoluteFilePath()),
                    xjw::cli::cameraToJson(makeFinalCamera(1.0)));
    const QString outputDir =
        QDir(tempDir.path()).filePath(QStringLiteral("final camera export"));
    xjw::cli::FinalBaCameraExportResult exportResult;
    QString error;

    ASSERT_TRUE(xjw::cli::exportFinalBaCameras(
        {imageA, imageB}, metadata, outputDir, &exportResult, &error))
        << qPrintable(error);
    EXPECT_EQ(exportResult.cameraPaths.size(), 2);
    EXPECT_TRUE(QFileInfo::exists(exportResult.imageCameraList));
    EXPECT_NE(QFileInfo(exportResult.cameraPaths.at(0)).fileName(),
              QFileInfo(exportResult.cameraPaths.at(1)).fileName());

    xjw::cli::PhotogrammetryListOptions options;
    options.allowImageOnlyRows = false;
    options.loadCameras = true;
    options.requireExistingCameras = true;
    std::vector<xjw::cli::PhotogrammetryInputItem> items;
    ASSERT_TRUE(xjw::cli::readPhotogrammetryImageList(
        exportResult.imageCameraList, options, &items, &error))
        << qPrintable(error);
    ASSERT_EQ(items.size(), 2u);
    EXPECT_EQ(QDir::cleanPath(items.at(0).imagePath), QDir::cleanPath(imageA));
    EXPECT_EQ(QDir::cleanPath(items.at(1).imagePath), QDir::cleanPath(imageB));
    EXPECT_TRUE(items.at(0).hasLoadedCamera);
    EXPECT_TRUE(items.at(1).hasLoadedCamera);
    EXPECT_NEAR(items.at(0).camera.cameraCenter().at(0), 0.0, 1e-12);
    EXPECT_NEAR(items.at(1).camera.cameraCenter().at(0), 1.0, 1e-12);
}

TEST(CliPhotogrammetryCommonTest, RejectsIncompleteOrExistingFinalCameraDestination)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    writePlaceholder(imageA);
    writePlaceholder(imageB);

    QMap<QString, QJsonObject> incomplete;
    incomplete.insert(QDir::cleanPath(QFileInfo(imageA).absoluteFilePath()),
                      xjw::cli::cameraToJson(makeFinalCamera(0.0)));
    const QString missingOutput = QDir(tempDir.path()).filePath(QStringLiteral("missing_output"));
    QString error;
    EXPECT_FALSE(xjw::cli::exportFinalBaCameras(
        {imageA, imageB}, incomplete, missingOutput, nullptr, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("没有影像对应的相机")));
    EXPECT_FALSE(QFileInfo::exists(missingOutput));

    const QString existingOutput = QDir(tempDir.path()).filePath(QStringLiteral("existing"));
    ASSERT_TRUE(QDir().mkpath(existingOutput));
    const QString sentinel = QDir(existingOutput).filePath(QStringLiteral("keep.txt"));
    writePlaceholder(sentinel);
    EXPECT_FALSE(xjw::cli::exportFinalBaCameras(
        {imageA}, incomplete, existingOutput, nullptr, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("拒绝覆盖")));
    EXPECT_TRUE(QFileInfo::exists(sentinel));
}

} // namespace
