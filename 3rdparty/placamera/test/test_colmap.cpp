#include <placamera/colmap.h>

#include <gtest/gtest.h>

TEST(PlaCameraColmap, ConvertsCameraIntrinsicsAndPose)
{
    const auto parsed_camera = placamera::parseColmapCameraLine("7 SIMPLE_RADIAL 100 80 50 49.5 39.5 0.1");
    ASSERT_TRUE(parsed_camera) << parsed_camera.message();
    const auto& camera = parsed_camera.value();
    EXPECT_EQ(camera.cameraId, 7);

    const auto has_distortion = placamera::colmapHasDistortion(camera);
    ASSERT_TRUE(has_distortion) << has_distortion.message();
    EXPECT_TRUE(has_distortion.value());

    const auto intrinsics = placamera::colmapRasterIntrinsics(camera);
    ASSERT_TRUE(intrinsics) << intrinsics.message();
    EXPECT_DOUBLE_EQ(intrinsics.value().focalX, 50.0);
    EXPECT_DOUBLE_EQ(intrinsics.value().principalX, 49.0);
    EXPECT_DOUBLE_EQ(intrinsics.value().principalY, 39.0);

    const auto pixel = placamera::projectColmapNormalized(camera, 0.2, 0.0);
    ASSERT_TRUE(pixel) << pixel.message();
    EXPECT_NEAR(pixel.value()[0], 59.04, 1.0e-12);

    const auto image =
        placamera::parseColmapImageLine("3 1 0 0 0 1 2 3 7 sample.jpg", placamera::FrameId("test-world"));
    ASSERT_TRUE(image) << image.message();
    EXPECT_EQ(image.value().cameraId, 7);
    EXPECT_EQ(image.value().imageName, "sample.jpg");
    EXPECT_DOUBLE_EQ(image.value().pose.center[0], -1.0);
    EXPECT_DOUBLE_EQ(image.value().pose.center[1], -2.0);
    EXPECT_DOUBLE_EQ(image.value().pose.center[2], -3.0);
}

TEST(PlaCameraColmap, RejectsUnsupportedAndMalformedModels)
{
    const auto unknown = placamera::parseColmapCameraLine("1 UNKNOWN 100 80 1 2 3");
    EXPECT_FALSE(unknown);
    EXPECT_EQ(unknown.errorCode(), placamera::CameraErrorCode::UnsupportedModel);

    const auto malformed = placamera::parseColmapCameraLine("1 PINHOLE 100 80 1 2 3");
    EXPECT_FALSE(malformed);
    EXPECT_EQ(malformed.errorCode(), placamera::CameraErrorCode::UnsupportedModel);

    const auto invalid_pose = placamera::parseColmapImageLine("1 0 0 0 0 0 0 0 1 x.jpg", placamera::FrameId("world"));
    EXPECT_FALSE(invalid_pose);
    EXPECT_EQ(invalid_pose.errorCode(), placamera::CameraErrorCode::ParseFailure);
}

TEST(PlaCameraColmap, MapsOnlyExactBrownConradyModels)
{
    const auto radial = placamera::parseColmapCameraLine("1 RADIAL 100 80 50 49.5 39.5 0.1 -0.02");
    ASSERT_TRUE(radial);
    const auto radial_distortion = placamera::colmapBrownConradyDistortion(radial.value());
    ASSERT_TRUE(radial_distortion) << radial_distortion.message();
    EXPECT_DOUBLE_EQ(radial_distortion.value().radialK1, 0.1);
    EXPECT_DOUBLE_EQ(radial_distortion.value().radialK2, -0.02);

    const auto full =
        placamera::parseColmapCameraLine("2 FULL_OPENCV 100 80 50 51 49.5 39.5 0.1 -0.02 0.003 -0.004 0.005 0 0 0");
    ASSERT_TRUE(full);
    const auto full_distortion = placamera::colmapBrownConradyDistortion(full.value());
    ASSERT_TRUE(full_distortion) << full_distortion.message();
    EXPECT_DOUBLE_EQ(full_distortion.value().radialK3, 0.005);
    EXPECT_DOUBLE_EQ(full_distortion.value().tangentialP1, 0.003);

    const auto rational =
        placamera::parseColmapCameraLine("3 FULL_OPENCV 100 80 50 51 49.5 39.5 0.1 -0.02 0.003 -0.004 0.005 0.006 0 0");
    ASSERT_TRUE(rational);
    const auto rational_mapping = placamera::colmapBrownConradyDistortion(rational.value());
    EXPECT_FALSE(rational_mapping);
    EXPECT_EQ(rational_mapping.errorCode(), placamera::CameraErrorCode::UnsupportedModel);

    const auto fisheye = placamera::parseColmapCameraLine("4 OPENCV_FISHEYE 100 80 50 51 49.5 39.5 0 0 0 0");
    ASSERT_TRUE(fisheye);
    EXPECT_FALSE(placamera::colmapBrownConradyDistortion(fisheye.value()));
}
