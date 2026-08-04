#include "CameraIntrinsicPrior.h"

#include <gtest/gtest.h>

TEST(CameraIntrinsicPriorTest, UsesExif35MillimeterEquivalentFocalLength)
{
    xjw::common::io::ImageExifMetadata metadata;
    metadata.make = QStringLiteral("Example");
    metadata.model = QStringLiteral("Camera");
    metadata.focalLength35Mm = 35.0;

    const auto prior = estimateCameraIntrinsicPrior(metadata, 6000, 4000);

    ASSERT_TRUE(prior.has_value());
    EXPECT_NEAR(prior->focalPixels, 5833.333, 0.01);
    EXPECT_NEAR(prior->focalScale, 35.0 / 36.0, 1.0e-6);
    EXPECT_EQ(prior->source, QStringLiteral("exif_focal_length_35mm"));
    EXPECT_TRUE(prior->strong);
}
TEST(CameraIntrinsicPriorTest, ResolvesSonyRx1rFixedLensWithoutFocalTag)
{
    xjw::common::io::ImageExifMetadata metadata;
    metadata.make = QStringLiteral("SONY");
    metadata.model = QStringLiteral("DSC-RX1R");

    const auto prior = estimateCameraIntrinsicPrior(metadata, 6000, 4000);

    ASSERT_TRUE(prior.has_value());
    EXPECT_NEAR(prior->focalPixels, 35.0 / 35.8 * 6000.0, 1.0e-6);
    EXPECT_NEAR(prior->focalScale, 0.9776536313, 1.0e-9);
    EXPECT_EQ(prior->source, QStringLiteral("fixed_lens_camera_catalog"));
}

TEST(CameraIntrinsicPriorTest, RejectsInterchangeableLensCameraWithoutSensorMetadata)
{
    xjw::common::io::ImageExifMetadata metadata;
    metadata.make = QStringLiteral("SONY");
    metadata.model = QStringLiteral("ILCE-7RM5");
    metadata.focalLengthMm = 35.0;

    EXPECT_FALSE(estimateCameraIntrinsicPrior(metadata, 6000, 4000).has_value());
}
