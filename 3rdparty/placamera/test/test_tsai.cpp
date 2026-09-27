#include <placamera/tsai.h>

#include <gtest/gtest.h>

#include <sstream>

TEST(PlaCameraTsai, RoundTripsDefinitionAndPoseWithoutFileSystem)
{
    const auto definition = placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("input"),
                                                                      {1000.0, 900.0, 500.0, 400.0, 0.01},
                                                                      {0.01, 0.02, 0.03, 0.001, -0.002},
                                                                      placamera::PixelConvention::PixelCenter,
                                                                      placamera::FrameId("world"));
    const auto pose = placamera::Pose::create(
        placamera::FrameId("world"), {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    std::stringstream stream;
    ASSERT_TRUE(placamera::writeTsaiFramePinhole(stream, {definition, pose}));
    const auto imported =
        placamera::readTsaiFramePinhole(stream, placamera::CameraDefinitionId("output"), placamera::FrameId("world"));
    ASSERT_TRUE(imported) << imported.message();
    EXPECT_DOUBLE_EQ(imported.value().definition->intrinsics().focalX, 1000.0);
    EXPECT_DOUBLE_EQ(imported.value().definition->distortion().radialK1, 0.01);
    EXPECT_EQ(imported.value().pose.center, pose.center);
}

TEST(PlaCameraTsai, PixelWriterKeepsVersionedInterchangeLayout)
{
    placamera::TsaiPixelCamera camera;
    camera.focalX = 120.0;
    camera.focalY = 130.0;
    camera.principalX = 40.0;
    camera.principalY = 50.0;
    camera.center = {-1.0, 2.0, 3.0};
    camera.cameraToWorldRotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::stringstream stream;
    ASSERT_TRUE(placamera::writeTsaiPixelCamera(stream, camera));
    EXPECT_EQ(stream.str(),
              "VERSION_3\nPINHOLE\nTSAI\nfu = 120\nfv = 130\ncu = 40\ncv = 50\n"
              "u_direction = 1 0 0\nv_direction = 0 1 0\nw_direction = 0 0 1\npitch = 1\n"
              "k1 = 0\nk2 = 0\nk3 = 0\np1 = 0\np2 = 0\nC = -1 2 3\n"
              "R = 1 0 0 0 1 0 0 0 1\n");
}
