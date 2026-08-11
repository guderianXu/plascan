#include "CameraModel.h"
#include "FramePinholeCamera.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

namespace
{

xjw::Camera makeFrameCamera(bool flipDepthAxis = false)
{
    xjw::Camera camera;
    camera.setIntrinsics(800.0, 850.0, 320.0, 240.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {1.0, 2.0, 3.0});
    camera.setDistortion(1.0e-3, -2.0e-5, 1.0e-7, 2.0e-4, -1.0e-4);
    camera.setAxisDirections(-1, 1);
    camera.setDepthAxisFlipped(flipDepthAxis);
    return camera;
}

} // namespace

TEST(CameraModel, DefaultFrameModelIsInvalid)
{
    xjw::FramePinholeCamera camera;
    const xjw::CameraModel &model = camera;

    EXPECT_EQ(model.modelType(), xjw::CameraModelType::FramePinhole);
    EXPECT_FALSE(model.isValid());
    EXPECT_FALSE(model.imageSize().has_value());
    EXPECT_TRUE(model.worldFrameName().empty());

    xjw::CameraImagingRay ray;
    xjw::CameraGroundProjection projection;
    EXPECT_FALSE(model.rayForPixel({0.0, 0.0}, &ray));
    EXPECT_FALSE(model.groundToImage({0.0, 0.0, 1.0}, &projection));
}

TEST(CameraModel, FrameModelExposesOwnedCameraGeometryPolymorphically)
{
    std::unique_ptr<xjw::CameraModel> model = std::make_unique<xjw::FramePinholeCamera>(
        makeFrameCamera(), xjw::CameraImageSize{640, 480}, "LOCAL_TEST_FRAME");

    ASSERT_TRUE(model->isValid());
    ASSERT_TRUE(model->imageSize().has_value());
    EXPECT_EQ(model->imageSize()->samples, 640);
    EXPECT_EQ(model->imageSize()->lines, 480);
    EXPECT_EQ(model->worldFrameName(), "LOCAL_TEST_FRAME");

    xjw::CameraImagingRay ray;
    ASSERT_TRUE(model->rayForPixel({320.0, 240.0}, &ray));
    EXPECT_DOUBLE_EQ(ray.originMeters[0], 1.0);
    EXPECT_DOUBLE_EQ(ray.originMeters[1], 2.0);
    EXPECT_DOUBLE_EQ(ray.originMeters[2], 3.0);
    EXPECT_NEAR(ray.direction[0], 0.0, 1.0e-14);
    EXPECT_NEAR(ray.direction[1], 0.0, 1.0e-14);
    EXPECT_NEAR(ray.direction[2], 1.0, 1.0e-14);
    EXPECT_FALSE(ray.ephemerisTimeSeconds.has_value());

    xjw::CameraGroundProjection projection;
    ASSERT_TRUE(model->groundToImage({1.0, 2.0, 13.0}, &projection));
    EXPECT_NEAR(projection.image.sample, 320.0, 1.0e-12);
    EXPECT_NEAR(projection.image.line, 240.0, 1.0e-12);
    EXPECT_NEAR(projection.positiveDepthMeters, 10.0, 1.0e-12);
    EXPECT_FALSE(projection.ephemerisTimeSeconds.has_value());
}

TEST(CameraModel, FrameRayRoundTripsWithDistortionAndFlippedDepth)
{
    xjw::FramePinholeCamera camera(makeFrameCamera(true));
    const xjw::CameraModel &model = camera;
    const xjw::CameraImageCoordinate input{401.25, 173.5};

    xjw::CameraImagingRay ray;
    ASSERT_TRUE(model.rayForPixel(input, &ray));
    const double norm = std::sqrt(ray.direction[0] * ray.direction[0]
                                  + ray.direction[1] * ray.direction[1]
                                  + ray.direction[2] * ray.direction[2]);
    EXPECT_NEAR(norm, 1.0, 1.0e-14);
    EXPECT_LT(ray.direction[2], 0.0);

    std::array<double, 3> ground = ray.originMeters;
    for (int axis = 0; axis < 3; ++axis)
    {
        ground[axis] += 250.0 * ray.direction[axis];
    }

    xjw::CameraGroundProjection projection;
    ASSERT_TRUE(model.groundToImage(ground, &projection));
    EXPECT_NEAR(projection.image.sample, input.sample, 1.0e-7);
    EXPECT_NEAR(projection.image.line, input.line, 1.0e-7);
    EXPECT_GT(projection.positiveDepthMeters, 0.0);
}
