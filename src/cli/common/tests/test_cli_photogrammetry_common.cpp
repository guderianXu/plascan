#include "cli_photogrammetry_common.h"
#include "FinalBaCameraExporter.h"
#include "project/ProjectFramePinholeMetadataIO.h"
#include "project/ProjectCameraIO.h"
#include <placamera/tsai.h>
#include "io/PathIO.h"

#include <gtest/gtest.h>
#include <placamera/frame_camera.h>

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

    using xjw::cli::parsePhotogrammetryListLine;

    std::shared_ptr<const placamera::FramePinholeModel>
    makeNativeFinalCamera(const std::string& imageId,
                          double centerX,
                          placamera::BrownConradyDistortion distortion = {},
                          int uAxisSign = 1,
                          bool depthAxisFlipped = false)
    {
        const placamera::FrameId frame("project-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("final-definition-" + imageId),
                                                      {900.0, 905.0, 320.0, 240.0, 0.01, uAxisSign, 1},
                                                      distortion,
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame,
                                                      depthAxisFlipped);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("final-instance-" + imageId),
            placamera::ImageId(imageId),
            definition,
            {640, 480},
            placamera::Pose::create(frame, {centerX, 0.0, 2.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
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
        std::vector<placamera::ImageId> imageIds;
        ASSERT_TRUE(xjw::cli::resolveProjectImageIds(projectFiles, images, &imageIds, nullptr));

        placamera::reference::ReferenceCameraPositionMap positions;
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

    TEST(CliPhotogrammetryCommonTest, BuildsExternalTsaiReferenceWithoutLegacyNumericCamera)
    {
        const QStringList images{QStringLiteral("left.tif"), QStringLiteral("right.tif")};
        const QJsonObject project_files = canonicalProjectFiles(images);
        const std::vector<placamera::ImageId> image_ids{placamera::ImageId("cli-test-image-1"),
                                                        placamera::ImageId("cli-test-image-2")};
        std::vector<xjw::cli::PhotogrammetryInputItem> items(2);
        items[0].imagePath = images[0];
        items[0].cameraPath = QString::fromUtf8(TEST_DATA_DIR "/tsai/1.tsai");
        items[0].hasCameraPath = true;
        items[1].imagePath = images[1];

        placamera::reference::ReferenceCameraGeometryMap geometries;
        QString error;
        ASSERT_TRUE(xjw::cli::buildReferenceCameraGeometries(
            project_files, items, images, image_ids, true, &geometries, &error))
            << error.toStdString();
        ASSERT_EQ(geometries.size(), 2U);
        const auto first = geometries.find(image_ids[0]);
        ASSERT_NE(first, geometries.end());
        EXPECT_EQ(first->second.model().imageId().value(), "cli-test-image-1");
        EXPECT_EQ(first->second.model().instanceId().value(), "cli-test-instance-cli-test-image-1");
        EXPECT_EQ(first->second.model().groundFrame().value(), "local");
        EXPECT_NEAR(first->second.model().pose().center[0], -3206.42347383, 1.0e-8);
        const auto second = geometries.find(image_ids[1]);
        ASSERT_NE(second, geometries.end());
        EXPECT_DOUBLE_EQ(second->second.model().pose().center[2], 2.0);

        geometries.clear();
        ASSERT_TRUE(xjw::cli::buildReferenceCameraGeometries(
            project_files, items, images, image_ids, false, &geometries, &error))
            << error.toStdString();
        const auto canonical_first = geometries.find(image_ids[0]);
        ASSERT_NE(canonical_first, geometries.end());
        EXPECT_DOUBLE_EQ(canonical_first->second.model().pose().center[0], 0.0);
    }

    TEST(CliPhotogrammetryCommonTest, ExportsCompleteFinalBaCameraSetForDirectReuse)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("first folder/shared image.png"));
        const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("second folder/shared image.png"));
        writePlaceholder(imageA);
        writePlaceholder(imageB);

        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(makeNativeFinalCamera("first-image", 0.0)).ok());
        ASSERT_TRUE(cameras.add(makeNativeFinalCamera("second-image", 1.0)).ok());
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("final camera export"));
        xjw::cli::FinalBaCameraExportResult exportResult;
        QString error;

        ASSERT_TRUE(
            xjw::cli::exportFinalBaCameras({imageA, imageB},
                                           {placamera::ImageId("first-image"), placamera::ImageId("second-image")},
                                           cameras,
                                           outputDir,
                                           &exportResult,
                                           &error))
            << qPrintable(error);
        EXPECT_EQ(exportResult.cameraPaths.size(), 2);
        EXPECT_TRUE(QFileInfo::exists(exportResult.imageCameraList));
        EXPECT_NE(QFileInfo(exportResult.cameraPaths.at(0)).fileName(),
                  QFileInfo(exportResult.cameraPaths.at(1)).fileName());

        xjw::cli::PhotogrammetryListOptions options;
        options.allowImageOnlyRows = false;
        options.requireExistingCameras = true;
        std::vector<xjw::cli::PhotogrammetryInputItem> items;
        ASSERT_TRUE(xjw::cli::readPhotogrammetryImageList(exportResult.imageCameraList, options, &items, &error))
            << qPrintable(error);
        ASSERT_EQ(items.size(), 2u);
        EXPECT_EQ(QDir::cleanPath(items.at(0).imagePath), QDir::cleanPath(imageA));
        EXPECT_EQ(QDir::cleanPath(items.at(1).imagePath), QDir::cleanPath(imageB));
        const auto first_camera =
            placamera::loadTsaiFramePinhole(xjw::common::io::toUtf8Path(items.at(0).cameraPath),
                                            placamera::CameraDefinitionId("first-exported-definition"),
                                            placamera::FrameId("project-world"));
        const auto second_camera =
            placamera::loadTsaiFramePinhole(xjw::common::io::toUtf8Path(items.at(1).cameraPath),
                                            placamera::CameraDefinitionId("second-exported-definition"),
                                            placamera::FrameId("project-world"));
        ASSERT_TRUE(first_camera) << first_camera.message();
        ASSERT_TRUE(second_camera) << second_camera.message();
        EXPECT_NEAR(first_camera.value().pose.center.at(0), 0.0, 1e-12);
        EXPECT_NEAR(second_camera.value().pose.center.at(0), 1.0, 1e-12);
    }

    TEST(CliPhotogrammetryCommonTest, AcceptsExistingEmptyFinalCameraDestination)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString image = QDir(tempDir.path()).filePath(QStringLiteral("image.png"));
        writePlaceholder(image);

        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(makeNativeFinalCamera("image", 0.0)).ok());
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("precreated_empty"));
        ASSERT_TRUE(QDir().mkpath(outputDir));

        xjw::cli::FinalBaCameraExportResult exportResult;
        QString error;
        ASSERT_TRUE(xjw::cli::exportFinalBaCameras(
            {image}, {placamera::ImageId("image")}, cameras, outputDir, &exportResult, &error))
            << qPrintable(error);
        EXPECT_TRUE(QFileInfo::exists(exportResult.imageCameraList));
        EXPECT_EQ(exportResult.cameraPaths.size(), 1);
    }

    TEST(CliPhotogrammetryCommonTest, FinalBaExportPreservesPlaCameraTsaiGeometry)
    {
        QTemporaryDir tempDir(QStringLiteral(PLASCAN_TEST_TMP_ROOT "/placamera-final-ba-XXXXXX"));
        ASSERT_TRUE(tempDir.isValid());
        const QString image = QDir(tempDir.path()).filePath(QStringLiteral("image.png"));
        writePlaceholder(image);

        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(
            cameras.add(makeNativeFinalCamera("image", 3.0, {0.01, -0.002, 0.0003, 0.0004, -0.0005}, -1, true)).ok());
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("native_export"));
        xjw::cli::FinalBaCameraExportResult exported;
        QString error;
        ASSERT_TRUE(xjw::cli::exportFinalBaCameras(
            {image}, {placamera::ImageId("image")}, cameras, outputDir, &exported, &error))
            << qPrintable(error);
        ASSERT_EQ(exported.cameraPaths.size(), 1);

        const auto restored = placamera::loadTsaiFramePinhole(exported.cameraPaths.front().toStdString(),
                                                              placamera::CameraDefinitionId("roundtrip-definition"),
                                                              placamera::FrameId("project-world"));
        ASSERT_TRUE(restored) << restored.message();
        const auto& intrinsics = restored.value().definition->intrinsics();
        const auto& distortion = restored.value().definition->distortion();
        EXPECT_DOUBLE_EQ(intrinsics.focalX, 900.0);
        EXPECT_DOUBLE_EQ(intrinsics.focalY, 905.0);
        EXPECT_EQ(intrinsics.uAxisSign, -1);
        EXPECT_EQ(intrinsics.vAxisSign, 1);
        EXPECT_TRUE(restored->definition->depthAxisFlipped());
        EXPECT_DOUBLE_EQ(distortion.radialK1, 0.01);
        EXPECT_DOUBLE_EQ(distortion.radialK2, -0.002);
        EXPECT_DOUBLE_EQ(distortion.radialK3, 0.0003);
        EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.0004);
        EXPECT_DOUBLE_EQ(distortion.tangentialP2, -0.0005);
        EXPECT_DOUBLE_EQ(restored->pose.center[0], 3.0);
        EXPECT_DOUBLE_EQ(restored->pose.center[2], 2.0);
    }

    TEST(CliPhotogrammetryCommonTest, FinalBaExportRejectsMismatchedImageIdBeforeWriting)
    {
        QTemporaryDir tempDir(QStringLiteral(PLASCAN_TEST_TMP_ROOT "/placamera-final-ba-XXXXXX"));
        ASSERT_TRUE(tempDir.isValid());
        const QString image = QDir(tempDir.path()).filePath(QStringLiteral("image.png"));
        writePlaceholder(image);
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(makeNativeFinalCamera("other-image", 0.0)).ok());
        const QString outputDir = QDir(tempDir.path()).filePath(QStringLiteral("invalid_export"));
        QString error;

        EXPECT_FALSE(xjw::cli::exportFinalBaCameras(
            {image}, {placamera::ImageId("image")}, cameras, outputDir, nullptr, &error));
        EXPECT_TRUE(error.contains(QStringLiteral("没有影像对应"))) << qPrintable(error);
        EXPECT_FALSE(QFileInfo::exists(outputDir));
    }

    TEST(CliPhotogrammetryCommonTest, PlaCameraMetadataDecoderRejectsInvalidImageSize)
    {
        QJsonObject metadata =
            xjw::common::project::serializeFramePinholeModel(*makeNativeFinalCamera("metadata-test-image", 0.0));
        metadata.remove(QStringLiteral("image_height"));
        metadata.insert(QStringLiteral("image_width"), 640);
        QString error;
        EXPECT_FALSE(xjw::common::project::decodeFramePinholeMetadata(
            metadata, placamera::CameraDefinitionId("invalid-image-size"), &error));
        EXPECT_TRUE(error.contains(QStringLiteral("宽高"))) << qPrintable(error);

        metadata.insert(QStringLiteral("image_height"), 480.5);
        EXPECT_FALSE(xjw::common::project::decodeFramePinholeMetadata(
            metadata, placamera::CameraDefinitionId("fractional-image-size"), &error));
        EXPECT_TRUE(error.contains(QStringLiteral("正整数"))) << qPrintable(error);

        metadata.insert(QStringLiteral("image_height"), 480);
        EXPECT_TRUE(xjw::common::project::decodeFramePinholeMetadata(
            metadata, placamera::CameraDefinitionId("valid-image-size"), &error))
            << qPrintable(error);
    }

    TEST(CliPhotogrammetryCommonTest, RejectsIncompleteOrExistingFinalCameraDestination)
    {
        QTemporaryDir tempDir;
        ASSERT_TRUE(tempDir.isValid());
        const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
        const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
        writePlaceholder(imageA);
        writePlaceholder(imageB);

        placamera::CameraInstanceSet incomplete;
        ASSERT_TRUE(incomplete.add(makeNativeFinalCamera("first-image", 0.0)).ok());
        const QString missingOutput = QDir(tempDir.path()).filePath(QStringLiteral("missing_output"));
        QString error;
        EXPECT_FALSE(
            xjw::cli::exportFinalBaCameras({imageA, imageB},
                                           {placamera::ImageId("first-image"), placamera::ImageId("second-image")},
                                           incomplete,
                                           missingOutput,
                                           nullptr,
                                           &error));
        EXPECT_TRUE(error.contains(QStringLiteral("没有影像对应的帧相机")));
        EXPECT_FALSE(QFileInfo::exists(missingOutput));

        const QString existingOutput = QDir(tempDir.path()).filePath(QStringLiteral("existing"));
        ASSERT_TRUE(QDir().mkpath(existingOutput));
        const QString sentinel = QDir(existingOutput).filePath(QStringLiteral("keep.txt"));
        writePlaceholder(sentinel);
        EXPECT_FALSE(xjw::cli::exportFinalBaCameras(
            {imageA}, {placamera::ImageId("first-image")}, incomplete, existingOutput, nullptr, &error));
        EXPECT_TRUE(error.contains(QStringLiteral("拒绝覆盖")));
        EXPECT_TRUE(QFileInfo::exists(sentinel));
    }

} // namespace
