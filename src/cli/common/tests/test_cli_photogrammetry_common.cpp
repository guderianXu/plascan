#include "cli_photogrammetry_common.h"
#include "FinalBaCameraExporter.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

    using xjw::cli::parsePhotogrammetryListLine;

    xjw::camera_models::frame_pinhole::FramePinholeNumericState makeFinalCamera(double centerX)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(900.0, 905.0, 320.0, 240.0);
        camera.setPixelPitch(0.01);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {centerX, 0.0, 2.0});
        return camera;
    }

    void writePlaceholder(const QString& path)
    {
        ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_EQ(file.write("image"), 5);
    }

    QJsonObject canonicalPinholeDefinition()
    {
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("cli-test-definition")},
                           {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("frame"), QStringLiteral("local")},
                           {QStringLiteral("parameters"),
                            QJsonObject{{QStringLiteral("intrinsics"),
                                         QJsonObject{{QStringLiteral("fx_px"), 900.0},
                                                     {QStringLiteral("fy_px"), 905.0},
                                                     {QStringLiteral("cx_px"), 320.0},
                                                     {QStringLiteral("cy_px"), 240.0},
                                                     {QStringLiteral("pixel_pitch_mm"), 0.01},
                                                     {QStringLiteral("u_axis_sign"), 1},
                                                     {QStringLiteral("v_axis_sign"), 1}}},
                                        {QStringLiteral("distortion"),
                                         QJsonObject{{QStringLiteral("k1"), 0.0},
                                                     {QStringLiteral("k2"), 0.0},
                                                     {QStringLiteral("k3"), 0.0},
                                                     {QStringLiteral("p1"), 0.0},
                                                     {QStringLiteral("p2"), 0.0}}},
                                        {QStringLiteral("pixel_convention"), QStringLiteral("center")},
                                        {QStringLiteral("depth_axis_flipped"), false}}}};
    }

    QJsonObject canonicalPinholeInstance(const QString& imageId)
    {
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("cli-test-instance-") + imageId},
                           {QStringLiteral("image_uuid"), imageId},
                           {QStringLiteral("definition_id"), QStringLiteral("cli-test-definition")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), 640}, {QStringLiteral("lines"), 480}}},
                           {QStringLiteral("pose"),
                            QJsonObject{{QStringLiteral("frame"), QStringLiteral("local")},
                                        {QStringLiteral("center_m"), QJsonArray{0.0, 0.0, 2.0}},
                                        {QStringLiteral("camera_to_world_rotation"),
                                         QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}}}};
    }

    QJsonObject canonicalProjectFiles(const QStringList& images)
    {
        QJsonArray imageEntries;
        QJsonArray instances;
        for (int index = 0; index < images.size(); ++index)
        {
            const QString imageId = QStringLiteral("cli-test-image-%1").arg(index + 1);
            imageEntries.append(
                QJsonObject{{QStringLiteral("image_uuid"), imageId}, {QStringLiteral("path"), images.at(index)}});
            instances.append(canonicalPinholeInstance(imageId));
        }
        return QJsonObject{{QStringLiteral("images"), imageEntries},
                           {QStringLiteral("camera_definitions"), QJsonArray{canonicalPinholeDefinition()}},
                           {QStringLiteral("camera_instances"), instances}};
    }

    TEST(CliPhotogrammetryCommonTest, ParsesShellAndCsvRows)
    {
        QStringList parts;
        QString error;

        EXPECT_TRUE(parsePhotogrammetryListLine(QStringLiteral("\"image one.tif\" camera.tsai"), &parts, &error));
        EXPECT_EQ(parts, (QStringList{QStringLiteral("image one.tif"), QStringLiteral("camera.tsai")}));

        EXPECT_TRUE(
            parsePhotogrammetryListLine(QStringLiteral("\"image,one.tif\",\"camera\"\"one.tsai\""), &parts, &error));
        EXPECT_EQ(parts, (QStringList{QStringLiteral("image,one.tif"), QStringLiteral("camera\"one.tsai")}));
    }

    TEST(CliPhotogrammetryCommonTest, PreservesWindowsPathSeparators)
    {
        QStringList parts;
        QString error;

        EXPECT_TRUE(parsePhotogrammetryListLine(
            QStringLiteral("E:\\code\\images\\image.tif E:\\code\\cameras\\camera.tsai"), &parts, &error));
        EXPECT_EQ(parts,
                  (QStringList{QStringLiteral("E:\\code\\images\\image.tif"),
                               QStringLiteral("E:\\code\\cameras\\camera.tsai")}));

        EXPECT_TRUE(parsePhotogrammetryListLine(
            QStringLiteral("\"E:\\data set\\image one.tif\" \"\\\\server\\camera set\\camera.tsai\""), &parts, &error));
        EXPECT_EQ(parts,
                  (QStringList{QStringLiteral("E:\\data set\\image one.tif"),
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
        EXPECT_FALSE(projectItem.contains(QStringLiteral("camera")));

        const QJsonObject reportPair = xjw::cli::inputPairsToJson(items).first().toObject();
        EXPECT_EQ(reportPair.value(QStringLiteral("image")).toString(), item.imagePath);
        EXPECT_EQ(reportPair.value(QStringLiteral("camera")).toString(), item.cameraPath);
    }

    TEST(CliPhotogrammetryCommonTest, ReadsPositionOnlyReferenceCsv)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString csvPath = QDir(tempDir.path()).filePath(QStringLiteral("reference.csv"));
        QFile file(csvPath);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
        ASSERT_GT(file.write("name,x,y,z\nIMG_001.tif,1.5,2.5,3.5\nIMG_002;4;5;6\n"), 0);
        file.close();

        const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("IMG_001.tif"));
        const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("IMG_002.tif"));
        writePlaceholder(imageA);
        writePlaceholder(imageB);
        const QStringList images{imageA, imageB};
        const QJsonObject projectFiles = canonicalProjectFiles(images);
        std::vector<xjw::camera_core::ImageId> imageIds;
        ASSERT_TRUE(xjw::cli::resolveProjectImageIds(projectFiles, images, &imageIds, nullptr));

        xjw::camera_reference::ReferenceCameraPositionMap positions;
        QString error;
        ASSERT_TRUE(xjw::cli::readReferencePositionCsv(csvPath, projectFiles, images, imageIds, &positions, &error))
            << qPrintable(error);
        ASSERT_EQ(positions.size(), 2U);
        const auto first = positions.find(imageIds.at(0));
        ASSERT_NE(first, positions.end());
        EXPECT_DOUBLE_EQ(first->second.center()[0], 1.5);
        const auto second = positions.find(imageIds.at(1));
        ASSERT_NE(second, positions.end());
        EXPECT_DOUBLE_EQ(second->second.center()[2], 6.0);
        EXPECT_EQ(first->second.worldFrame().value(), "local");
    }

    TEST(CliPhotogrammetryCommonTest, ExportsCompleteFinalBaCameraSetForDirectReuse)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("first folder/shared image.png"));
        const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("second folder/shared image.png"));
        writePlaceholder(imageA);
        writePlaceholder(imageB);

        QMap<QString, QJsonObject> metadata;
        metadata.insert(QDir::cleanPath(QFileInfo(imageA).absoluteFilePath()),
                        xjw::cli::cameraToJson(makeFinalCamera(0.0)));
        metadata.insert(QDir::cleanPath(QFileInfo(imageB).absoluteFilePath()),
                        xjw::cli::cameraToJson(makeFinalCamera(1.0)));
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("final camera export"));
        xjw::cli::FinalBaCameraExportResult exportResult;
        QString error;

        ASSERT_TRUE(xjw::cli::exportFinalBaCameras({imageA, imageB}, metadata, outputDir, &exportResult, &error))
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
        ASSERT_TRUE(xjw::cli::readPhotogrammetryImageList(exportResult.imageCameraList, options, &items, &error))
            << qPrintable(error);
        ASSERT_EQ(items.size(), 2u);
        EXPECT_EQ(QDir::cleanPath(items.at(0).imagePath), QDir::cleanPath(imageA));
        EXPECT_EQ(QDir::cleanPath(items.at(1).imagePath), QDir::cleanPath(imageB));
        EXPECT_TRUE(items.at(0).hasLoadedCamera);
        EXPECT_TRUE(items.at(1).hasLoadedCamera);
        EXPECT_NEAR(items.at(0).camera.cameraCenter().at(0), 0.0, 1e-12);
        EXPECT_NEAR(items.at(1).camera.cameraCenter().at(0), 1.0, 1e-12);
    }

    TEST(CliPhotogrammetryCommonTest, AcceptsExistingEmptyFinalCameraDestination)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString image = QDir(tempDir.path()).filePath(QStringLiteral("image.png"));
        writePlaceholder(image);

        QMap<QString, QJsonObject> metadata;
        metadata.insert(QDir::cleanPath(QFileInfo(image).absoluteFilePath()),
                        xjw::cli::cameraToJson(makeFinalCamera(0.0)));
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("precreated_empty"));
        ASSERT_TRUE(QDir().mkpath(outputDir));

        xjw::cli::FinalBaCameraExportResult exportResult;
        QString error;
        ASSERT_TRUE(xjw::cli::exportFinalBaCameras({image}, metadata, outputDir, &exportResult, &error))
            << qPrintable(error);
        EXPECT_TRUE(QFileInfo::exists(exportResult.imageCameraList));
        EXPECT_EQ(exportResult.cameraPaths.size(), 1);
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
        EXPECT_FALSE(xjw::cli::exportFinalBaCameras({imageA, imageB}, incomplete, missingOutput, nullptr, &error));
        EXPECT_TRUE(error.contains(QStringLiteral("没有影像对应的相机")));
        EXPECT_FALSE(QFileInfo::exists(missingOutput));

        const QString existingOutput = QDir(tempDir.path()).filePath(QStringLiteral("existing"));
        ASSERT_TRUE(QDir().mkpath(existingOutput));
        const QString sentinel = QDir(existingOutput).filePath(QStringLiteral("keep.txt"));
        writePlaceholder(sentinel);
        EXPECT_FALSE(xjw::cli::exportFinalBaCameras({imageA}, incomplete, existingOutput, nullptr, &error));
        EXPECT_TRUE(error.contains(QStringLiteral("拒绝覆盖")));
        EXPECT_TRUE(QFileInfo::exists(sentinel));
    }

} // namespace
