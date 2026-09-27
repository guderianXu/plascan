#include <placamera/project_import.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{

    struct TestFiles
    {
        explicit TestFiles(std::string name) : path(std::filesystem::path(PLACAMERA_TEST_TMP_ROOT) / name)
        {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }

        ~TestFiles()
        {
            std::filesystem::remove_all(path);
        }

        void write(const std::string& name, const std::string& content) const
        {
            std::ofstream output(path / name, std::ios::binary);
            output << content;
            if (!output)
            {
                throw std::runtime_error("test file write failed");
            }
        }

        std::filesystem::path path;
    };

    TEST(PlaCameraProjectImport, ColmapRetainsExactFisheyeModelAndPose)
    {
        TestFiles files("project_import_colmap");
        files.write("cameras.txt", "# cameras\n3 OPENCV_FISHEYE 800 600 500 501 400 300 0.1 0.02 0 0\n");
        files.write("images.txt", "# images\n7 1 0 0 0 1 2 3 3 image.jpg\n\n");

        const auto imported_result = placamera::importCameraProject(files.path);
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        EXPECT_EQ(imported.format, placamera::CameraProjectFormat::ColmapText);
        ASSERT_EQ(imported.cameras.size(), 1u);
        EXPECT_TRUE(imported.references.empty());
        EXPECT_EQ(imported.cameras[0].imageName, "image.jpg");
        EXPECT_DOUBLE_EQ(imported.cameras[0].center[0], -1.0);
        EXPECT_DOUBLE_EQ(imported.cameras[0].calibration.intrinsicMatrix[2], 399.5);
        ASSERT_TRUE(imported.cameras[0].sourceColmapCamera.has_value());
        EXPECT_EQ(imported.cameras[0].sourceColmapCamera->model, "OPENCV_FISHEYE");
        EXPECT_EQ(imported.cameras[0].sourceColmapCamera->width, 800);
        EXPECT_EQ(imported.cameras[0].sourceImageId, 7);
        EXPECT_FALSE(imported.cameras[0].compatibility.isExactlyRepresentable());
        ASSERT_TRUE(imported.cameras[0].compatibility.unsupportedReason.has_value());
    }

    TEST(PlaCameraProjectImport, ColmapPublishesExactBrownDistortionWhenRepresentable)
    {
        TestFiles files("project_import_colmap_brown");
        files.write("cameras.txt", "3 OPENCV 800 600 500 501 400 300 0.1 -0.02 0.003 -0.004\n");
        files.write("images.txt", "7 1 0 0 0 1 2 3 3 image.jpg\n\n");

        const auto imported_result = placamera::importCameraProject(files.path);
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        ASSERT_EQ(imported.cameras.size(), 1u);
        EXPECT_TRUE(imported.cameras[0].compatibility.isExactlyRepresentable());
        const auto& distortion = imported.cameras[0].calibration.distortion;
        EXPECT_DOUBLE_EQ(distortion.radialK1, 0.1);
        EXPECT_DOUBLE_EQ(distortion.radialK2, -0.02);
        EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.003);
        EXPECT_DOUBLE_EQ(distortion.tangentialP2, -0.004);
    }

    TEST(PlaCameraProjectImport, MetashapeReferenceRemainsUnresolved)
    {
        TestFiles files("project_import_reference");
        files.write("Cameras.txt",
                    "\xEF\xBB\xBF# file\tWGS84_lat\tWGS84_lon\tWGS84_H\troll\tpitch\tyaw\t"
                    "time\tStd Dev n (m)\tStd Dev e (m)\tStd Dev u (m)\tStd Dev Hz (m)\n"
                    "image.jpg\t59\t31\t225\t1\t2\t3\ttime\t0.1\t0.2\t0.3\t0.4\n");
        files.write("GNSS_offset.txt", "Z=-0.03\nX=0.36\nY=-0.18\n");

        const auto imported_result =
            placamera::importCameraProject(files.path / "Cameras.txt",
                                           placamera::CameraProjectFormat::MetashapeReferenceText,
                                           files.path / "GNSS_offset.txt");
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        EXPECT_TRUE(imported.cameras.empty());
        ASSERT_EQ(imported.references.size(), 1u);
        EXPECT_DOUBLE_EQ(imported.references[0].latitudeDegrees, 59.0);
        EXPECT_DOUBLE_EQ(imported.references[0].yawDegrees, 3.0);
        ASSERT_TRUE(imported.leverArm.has_value());
        EXPECT_DOUBLE_EQ(imported.leverArm->xMeters, 0.36);
    }

    TEST(PlaCameraProjectImport, MiddleburyFileReturnsGeometry)
    {
        TestFiles files("project_import_middlebury");
        files.write("scene_par.txt", "1\nimage.png 120 0 40 0 130 50 0 0 1 0 -1 0 1 0 0 0 0 1 2 -3 4\n");
        const auto imported_result = placamera::importCameraProject(files.path);
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        ASSERT_EQ(imported.cameras.size(), 1u);
        EXPECT_EQ(imported.cameras[0].imageName, "image.png");
        EXPECT_FALSE(imported.cameras[0].sourceColmapCamera.has_value());
        EXPECT_FALSE(imported.cameras[0].sourceImageId.has_value());
    }

    TEST(PlaCameraProjectImport, EpflFileUsesImageNameFromFilename)
    {
        TestFiles files("project_import_epfl");
        files.write("frame.jpg.camera",
                    "100 0 50\n0 101 60\n0 0 1\n0 0 0\n"
                    "1 0 0\n0 1 0\n0 0 1\n3 4 5\n");
        const auto imported_result = placamera::importCameraProject(files.path);
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        EXPECT_EQ(imported.format, placamera::CameraProjectFormat::EpflCamera);
        ASSERT_EQ(imported.cameras.size(), 1u);
        EXPECT_EQ(imported.cameras[0].imageName, "frame.jpg");
        EXPECT_DOUBLE_EQ(imported.cameras[0].center[2], 5.0);
    }

    TEST(PlaCameraProjectImport, MetashapeDocumentReturnsCalibratedGeometry)
    {
        TestFiles files("project_import_metashape");
        files.write("doc.xml",
                    "<sensor id='3'><resolution width='1000' height='800'/>"
                    "<calibration><f>400</f><cx>10</cx><cy>-5</cy><b1>2</b1><k4>0.0001</k4>"
                    "</calibration></sensor>"
                    "<camera sensor_id='3' label='frame.jpg'>"
                    "<transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform></camera>");
        const auto imported_result = placamera::importCameraProject(files.path / "doc.xml");
        ASSERT_TRUE(imported_result) << imported_result.message();
        const auto& imported = imported_result.value();
        EXPECT_EQ(imported.format, placamera::CameraProjectFormat::MetashapeXml);
        ASSERT_EQ(imported.cameras.size(), 1u);
        EXPECT_EQ(imported.cameras[0].imageName, "frame.jpg");
        EXPECT_DOUBLE_EQ(imported.cameras[0].center[1], 2.0);
        ASSERT_TRUE(imported.cameras[0].calibration.metashapeCalibration);
        const auto& calibration = *imported.cameras[0].calibration.metashapeCalibration;
        EXPECT_DOUBLE_EQ(calibration.f, 400.0);
        EXPECT_DOUBLE_EQ(calibration.cx, 510.0);
        EXPECT_DOUBLE_EQ(calibration.cy, 395.0);
        EXPECT_DOUBLE_EQ(calibration.b1, 2.0);
        EXPECT_DOUBLE_EQ(calibration.k4, 0.0001);
        ASSERT_TRUE(calibration.principalPointDecomposition);
        EXPECT_EQ(*calibration.principalPointDecomposition,
                  (placamera::PrincipalPointDecomposition{500.0, 400.0, 10.0, -5.0}));

        const auto geometry = placamera::makeCentralCameraGeometry(
            imported.cameras[0], placamera::CameraDefinitionId("metashape-import"), placamera::FrameId("world"));
        ASSERT_TRUE(geometry) << geometry.message();
        EXPECT_EQ(geometry.value().definition->principalPointDecomposition(), calibration.principalPointDecomposition);
    }

} // namespace
