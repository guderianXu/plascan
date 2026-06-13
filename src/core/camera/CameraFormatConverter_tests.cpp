#include <gtest/gtest.h>

#include "Camera.h"
#include "CameraFormatConverter.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <optional>
#include <sstream>

namespace
{

void writeText(const std::filesystem::path &path, const std::string &text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    ASSERT_TRUE(out.good()) << path;
    out << text;
}

std::string readText(const std::filesystem::path &path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

TEST(CameraFormatConverterTest, KnownFormatsExposeCliNames)
{
    const auto formats = xjw::camera::supportedFormatNames();

    EXPECT_NE(std::find(formats.begin(), formats.end(), "auto"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "middlebury-par"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "epfl-camera"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "colmap-text"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "metashape-xml"), formats.end());

    EXPECT_EQ(xjw::camera::parseCameraFormat("middlebury_par"), xjw::camera::CameraFormat::MiddleburyPar);
    EXPECT_EQ(xjw::camera::parseCameraFormat("epfl-camera"), xjw::camera::CameraFormat::EpflCamera);
    EXPECT_EQ(xjw::camera::parseCameraFormat("colmap-text"), xjw::camera::CameraFormat::ColmapText);
    const auto metashape = xjw::camera::parseCameraFormat("metashape");
    ASSERT_TRUE(metashape.has_value());
    EXPECT_EQ(xjw::camera::cameraFormatName(*metashape), "metashape-xml");
    EXPECT_EQ(xjw::camera::parseCameraFormat("unknown"), std::nullopt);
}

TEST(CameraFormatConverterTest, MiddleburyParConvertsToTsaiAndImageCameraList)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_middlebury_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path source = root / "extracted" / "dinoSparseRing";
    writeText(source / "dinoSR0001.png", "fake image");
    writeText(source / "dinoSR0002.png", "fake image");
    writeText(source / "dinoSR_par.txt",
              "2\n"
              "dinoSR0001.png 120 0 40 0 130 50 0 0 1 0 -1 0 1 0 0 0 0 1 2 -3 4\n"
              "dinoSR0002.png 121 0 41 0 131 51 0 0 1 1 0 0 0 1 0 0 0 1 4 5 6\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::MiddleburyPar;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.datasetId = "dino";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.cameraCount, 2);
    EXPECT_EQ(result.inputFormat, xjw::camera::CameraFormat::MiddleburyPar);
    EXPECT_TRUE(std::filesystem::exists(result.imageCameraList));
    EXPECT_TRUE(std::filesystem::exists(result.summaryPath));

    const std::string lis = readText(result.imageCameraList);
    EXPECT_NE(lis.find("dinoSR0001.png cameras/dinoSR0001.tsai"), std::string::npos);
    EXPECT_NE(lis.find("dinoSR0002.png cameras/dinoSR0002.tsai"), std::string::npos);

    xjw::Camera camera;
    ASSERT_TRUE(camera.loadFromFile((options.outputDir / "cameras" / "dinoSR0001.tsai").string()));
    EXPECT_DOUBLE_EQ(camera.focalX(), 120.0);
    EXPECT_DOUBLE_EQ(camera.focalY(), 130.0);
    EXPECT_DOUBLE_EQ(camera.principalX(), 40.0);
    EXPECT_DOUBLE_EQ(camera.principalY(), 50.0);

    const auto center = camera.cameraCenter();
    EXPECT_DOUBLE_EQ(center[0], 3.0);
    EXPECT_DOUBLE_EQ(center[1], 2.0);
    EXPECT_DOUBLE_EQ(center[2], -4.0);

    const auto rotation = camera.cameraToWorldRotation();
    EXPECT_DOUBLE_EQ(rotation[0], 0.0);
    EXPECT_DOUBLE_EQ(rotation[1], 1.0);
    EXPECT_DOUBLE_EQ(rotation[2], 0.0);
    EXPECT_DOUBLE_EQ(rotation[3], -1.0);
    EXPECT_DOUBLE_EQ(rotation[4], 0.0);
    EXPECT_DOUBLE_EQ(rotation[5], 0.0);
    EXPECT_DOUBLE_EQ(rotation[6], 0.0);
    EXPECT_DOUBLE_EQ(rotation[7], 0.0);
    EXPECT_DOUBLE_EQ(rotation[8], 1.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"middlebury-par\""), std::string::npos);
    EXPECT_NE(summary.find("\"camera_count\": 2"), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(CameraFormatConverterTest, EpflCameraConvertsWithSkewWarning)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_epfl_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path source = root / "epfl";
    writeText(source / "rdimage.000.ppm", "fake image");
    writeText(source / "rdimage.000.ppm.camera",
              "3954.75 -8.5 1619.9\n"
              "0 3948.0 1151.4\n"
              "0 0 1\n"
              "0 0 0\n"
              "0 -1 0\n"
              "1 0 0\n"
              "0 0 1\n"
              "60 -11 -35\n"
              "3072 2048\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::EpflCamera;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(result.cameraCount, 1);
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings.front().find("skew"), std::string::npos);

    xjw::Camera camera;
    ASSERT_TRUE(camera.loadFromFile((options.outputDir / "cameras" / "rdimage.000.ppm.tsai").string()));
    EXPECT_DOUBLE_EQ(camera.focalX(), 3954.75);
    EXPECT_DOUBLE_EQ(camera.focalY(), 3948.0);
    EXPECT_DOUBLE_EQ(camera.principalX(), 1619.9);
    EXPECT_DOUBLE_EQ(camera.principalY(), 1151.4);

    const auto center = camera.cameraCenter();
    EXPECT_DOUBLE_EQ(center[0], 60.0);
    EXPECT_DOUBLE_EQ(center[1], -11.0);
    EXPECT_DOUBLE_EQ(center[2], -35.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"epfl-camera\""), std::string::npos);
    EXPECT_NE(summary.find("skew"), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(CameraFormatConverterTest, ColmapTextConvertsSiblingImagesToTsaiAndImageCameraList)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_colmap_text_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path dataset = root / "south-building";
    const std::filesystem::path source = dataset / "sparse";
    const std::filesystem::path images = dataset / "images";
    writeText(images / "P1180141.JPG", "fake image");
    writeText(images / "P1180142.JPG", "fake image");
    writeText(source / "cameras.txt",
              "# Camera list\n"
              "1 SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997\n"
              "2 PINHOLE 3072 2304 2600 2610 1530 1150\n");
    writeText(source / "images.txt",
              "# Image list\n"
              "1 1 0 0 0 10 20 30 1 P1180141.JPG\n"
              "\n"
              "2 0 0 1 0 1 2 3 2 P1180142.JPG\n"
              "\n");
    writeText(source / "points3D.txt", "# unused by converter\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::Auto;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.datasetId = "south-building";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.inputFormat, xjw::camera::CameraFormat::ColmapText);
    EXPECT_EQ(result.cameraCount, 2);
    EXPECT_TRUE(std::filesystem::exists(result.imageCameraList));

    const std::string lis = readText(result.imageCameraList);
    EXPECT_NE(lis.find("../south-building/images/P1180141.JPG cameras/P1180141.tsai"), std::string::npos);
    EXPECT_NE(lis.find("../south-building/images/P1180142.JPG cameras/P1180142.tsai"), std::string::npos);

    xjw::Camera first;
    ASSERT_TRUE(first.loadFromFile((options.outputDir / "cameras" / "P1180141.tsai").string()));
    EXPECT_DOUBLE_EQ(first.focalX(), 2559.68);
    EXPECT_DOUBLE_EQ(first.focalY(), 2559.68);
    EXPECT_DOUBLE_EQ(first.principalX(), 1536.0);
    EXPECT_DOUBLE_EQ(first.principalY(), 1152.0);

    const auto firstCenter = first.cameraCenter();
    EXPECT_DOUBLE_EQ(firstCenter[0], -10.0);
    EXPECT_DOUBLE_EQ(firstCenter[1], -20.0);
    EXPECT_DOUBLE_EQ(firstCenter[2], -30.0);

    xjw::Camera second;
    ASSERT_TRUE(second.loadFromFile((options.outputDir / "cameras" / "P1180142.tsai").string()));
    EXPECT_DOUBLE_EQ(second.focalX(), 2600.0);
    EXPECT_DOUBLE_EQ(second.focalY(), 2610.0);
    EXPECT_DOUBLE_EQ(second.principalX(), 1530.0);
    EXPECT_DOUBLE_EQ(second.principalY(), 1150.0);

    const auto secondCenter = second.cameraCenter();
    EXPECT_DOUBLE_EQ(secondCenter[0], 1.0);
    EXPECT_DOUBLE_EQ(secondCenter[1], -2.0);
    EXPECT_DOUBLE_EQ(secondCenter[2], 3.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"colmap-text\""), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(CameraFormatConverterTest, ColmapTextWritesImageCameraListInImageNameOrder)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_colmap_order_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path dataset = root / "south-building";
    const std::filesystem::path source = dataset / "sparse";
    const std::filesystem::path images = dataset / "images";
    writeText(images / "P1180141.JPG", "fake image");
    writeText(images / "P1180142.JPG", "fake image");
    writeText(images / "P1180200.JPG", "fake image");
    writeText(source / "cameras.txt",
              "# Camera list\n"
              "1 SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997\n");
    writeText(source / "images.txt",
              "# Image list\n"
              "10 1 0 0 0 10 20 30 1 P1180200.JPG\n"
              "\n"
              "11 1 0 0 0 11 21 31 1 P1180141.JPG\n"
              "\n"
              "12 1 0 0 0 12 22 32 1 P1180142.JPG\n"
              "\n");
    writeText(source / "points3D.txt", "# unused by converter\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::ColmapText;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.datasetId = "south-building";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;

    const std::string lis = readText(result.imageCameraList);
    const auto first = lis.find("P1180141.JPG");
    const auto second = lis.find("P1180142.JPG");
    const auto third = lis.find("P1180200.JPG");
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    ASSERT_NE(third, std::string::npos);
    EXPECT_LT(first, second);
    EXPECT_LT(second, third);

    std::filesystem::remove_all(root);
}

TEST(CameraFormatConverterTest, MetashapeXmlConvertsDepthImagesProject)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_metashape_xml_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path dataset = root / "depth_images";
    const std::filesystem::path images = dataset / "Depthimages";
    const std::filesystem::path metashape = dataset / "Metashape" / "Project_depthimages.files" / "0";
    writeText(images / "IMG_0001.JPG", "fake image");
    writeText(images / "IMG_0002.JPG", "fake image");
    writeText(images / "AERIAL_f001_002.JPG", "fake image");
    writeText(metashape / "doc.xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
              "<document>\n"
              "  <chunk>\n"
              "    <sensors>\n"
              "      <sensor id=\"0\" label=\"RGB\" type=\"frame\">\n"
              "        <resolution width=\"1000\" height=\"800\"/>\n"
              "        <calibration type=\"frame\" class=\"adjusted\">\n"
              "          <f>500</f><cx>10</cx><cy>-20</cy><k1>0.01</k1>\n"
              "        </calibration>\n"
              "      </sensor>\n"
              "    </sensors>\n"
              "    <cameras>\n"
              "      <camera id=\"0\" sensor_id=\"0\" component_id=\"0\" label=\"IMG_0001_0\">\n"
              "        <transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform>\n"
              "      </camera>\n"
              "      <camera id=\"1\" sensor_id=\"0\" component_id=\"0\" label=\"IMG_0002_1\">\n"
              "        <transform>0 -1 0 4 1 0 0 5 0 0 1 6 0 0 0 1</transform>\n"
              "      </camera>\n"
              "      <camera id=\"2\" sensor_id=\"0\" component_id=\"0\" label=\"AERIAL_f001_002\">\n"
              "        <transform>1 0 0 7 0 1 0 8 0 0 1 9 0 0 0 1</transform>\n"
              "      </camera>\n"
              "    </cameras>\n"
              "  </chunk>\n"
              "</document>\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::Auto;
    options.inputPath = dataset;
    options.outputDir = root / "out";
    options.datasetId = "depth_images";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(xjw::camera::cameraFormatName(result.inputFormat), "metashape-xml");
    ASSERT_EQ(result.cameraCount, 3);
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings.front().find("distortion"), std::string::npos);

    const std::string lis = readText(result.imageCameraList);
    EXPECT_NE(lis.find("../depth_images/Depthimages/IMG_0001.JPG cameras/IMG_0001.tsai"), std::string::npos);
    EXPECT_NE(lis.find("../depth_images/Depthimages/IMG_0002.JPG cameras/IMG_0002.tsai"), std::string::npos);
    EXPECT_NE(lis.find("../depth_images/Depthimages/AERIAL_f001_002.JPG cameras/AERIAL_f001_002.tsai"),
              std::string::npos);

    xjw::Camera first;
    ASSERT_TRUE(first.loadFromFile((options.outputDir / "cameras" / "IMG_0001.tsai").string()));
    EXPECT_DOUBLE_EQ(first.focalX(), 500.0);
    EXPECT_DOUBLE_EQ(first.focalY(), 500.0);
    EXPECT_DOUBLE_EQ(first.principalX(), 510.0);
    EXPECT_DOUBLE_EQ(first.principalY(), 380.0);

    const auto firstCenter = first.cameraCenter();
    EXPECT_DOUBLE_EQ(firstCenter[0], 1.0);
    EXPECT_DOUBLE_EQ(firstCenter[1], 2.0);
    EXPECT_DOUBLE_EQ(firstCenter[2], 3.0);

    xjw::Camera second;
    ASSERT_TRUE(second.loadFromFile((options.outputDir / "cameras" / "IMG_0002.tsai").string()));
    const auto secondCenter = second.cameraCenter();
    EXPECT_DOUBLE_EQ(secondCenter[0], 4.0);
    EXPECT_DOUBLE_EQ(secondCenter[1], 5.0);
    EXPECT_DOUBLE_EQ(secondCenter[2], 6.0);
    const auto secondRotation = second.cameraToWorldRotation();
    EXPECT_DOUBLE_EQ(secondRotation[0], 0.0);
    EXPECT_DOUBLE_EQ(secondRotation[1], -1.0);
    EXPECT_DOUBLE_EQ(secondRotation[3], 1.0);
    EXPECT_DOUBLE_EQ(secondRotation[4], 0.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"metashape-xml\""), std::string::npos);

    std::filesystem::remove_all(root);
}
