#include "detection/MarkerDetectorFactory.h"

#include <gtest/gtest.h>

using xjw::control_points::MarkerDetectorFactory;
using xjw::control_points::MarkerTargetFamily;

TEST(MarkerDetectorFactoryTest, CreatesEveryImplementedDetectorFamily)
{
    const QVector<MarkerTargetFamily> families = {
        MarkerTargetFamily::AprilTag16h5,
        MarkerTargetFamily::AprilTag25h9,
        MarkerTargetFamily::AprilTag36h10,
        MarkerTargetFamily::AprilTag36h11,
        MarkerTargetFamily::AprilTagCircle21h7,
        MarkerTargetFamily::AprilTagStandard41h12,
        MarkerTargetFamily::AprilTagStandard52h13,
        MarkerTargetFamily::NonCodedCircle,
        MarkerTargetFamily::NonCodedFourQuadrant,
    };
    for (const MarkerTargetFamily family : families)
    {
        EXPECT_NE(MarkerDetectorFactory::create(family), nullptr);
    }
}

TEST(MarkerDetectorFactoryTest, RejectsCircularFamiliesUntilCompatibilityCorpusIsInstalled)
{
    EXPECT_THROW(MarkerDetectorFactory::create(MarkerTargetFamily::Circular12Bit),
                 std::runtime_error);
}

TEST(MarkerDetectorFactoryTest, ParsesStableCliFamilyNames)
{
    EXPECT_EQ(MarkerDetectorFactory::parseFamily(QStringLiteral("tag36h11")),
              MarkerTargetFamily::AprilTag36h11);
    EXPECT_EQ(MarkerDetectorFactory::parseFamily(QStringLiteral("noncoded-circle")),
              MarkerTargetFamily::NonCodedCircle);
    EXPECT_FALSE(MarkerDetectorFactory::parseFamily(QStringLiteral("unknown-family")).has_value());
}
