#include "coordinate_system/transform/CoordinateTransformService.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{

    using namespace xjw::coordinate_system;

    CoordinateFrame rootFrame(const char* id, LinearUnit unit = LinearUnit::Metre)
    {
        return CoordinateFrame::create(CoordinateFrameId(id),
                                       CoordinateFrameKind::Ecef,
                                       unit,
                                       AngleUnit::Degree,
                                       std::nullopt,
                                       RigidTransform::identity());
    }

    TEST(CoordinateTransformServiceTest, TransformsPointVectorAndRotationThroughRigidFrame)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("root"));
        transforms.registerFrame(CoordinateFrame::create(
            CoordinateFrameId("local"),
            CoordinateFrameKind::LocalCartesian,
            LinearUnit::Metre,
            AngleUnit::Radian,
            CoordinateFrameId("root"),
            RigidTransform{{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, {10.0, 20.0, 30.0}}));

        EXPECT_EQ(transforms.transformPoint(CoordinateFrameId("local"), CoordinateFrameId("root"), {1.0, 2.0, 3.0}),
                  (std::array<double, 3>{8.0, 21.0, 33.0}));
        EXPECT_EQ(transforms.transformVector(CoordinateFrameId("local"), CoordinateFrameId("root"), {1.0, 0.0, 0.0}),
                  (std::array<double, 3>{0.0, 1.0, 0.0}));
        const RotationMatrix3d identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        EXPECT_EQ(transforms.transformRotation(CoordinateFrameId("local"), CoordinateFrameId("root"), identity),
                  (RotationMatrix3d{{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}}));
    }

    TEST(CoordinateTransformServiceTest, ConvertsPhysicalUnitsAndRoundTrips)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("metres"));
        transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("kilometres"),
                                                         CoordinateFrameKind::LocalCartesian,
                                                         LinearUnit::Kilometre,
                                                         AngleUnit::Radian,
                                                         CoordinateFrameId("metres"),
                                                         RigidTransform::identity()));

        const auto metres =
            transforms.transformPoint(CoordinateFrameId("kilometres"), CoordinateFrameId("metres"), {1.0, 2.0, 3.0});
        EXPECT_EQ(metres, (std::array<double, 3>{1000.0, 2000.0, 3000.0}));
        EXPECT_EQ(transforms.transformPoint(CoordinateFrameId("metres"), CoordinateFrameId("kilometres"), metres),
                  (std::array<double, 3>{1.0, 2.0, 3.0}));
    }

    TEST(CoordinateTransformServiceTest, RejectsProjectUnitToPhysicalUnitConversion)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("metres"));
        transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("relative"),
                                                         CoordinateFrameKind::LocalCartesian,
                                                         LinearUnit::ProjectUnit,
                                                         AngleUnit::Radian,
                                                         CoordinateFrameId("metres"),
                                                         RigidTransform::identity()));

        EXPECT_THROW(
            transforms.transformPoint(CoordinateFrameId("relative"), CoordinateFrameId("metres"), {1.0, 2.0, 3.0}),
            CoordinateTransformError);
    }

    TEST(CoordinateTransformServiceTest, RejectsUnknownOrDisconnectedFrames)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("first"));
        transforms.registerFrame(rootFrame("second"));

        EXPECT_THROW(transforms.resolve(CoordinateFrameId("missing")), CoordinateTransformError);
        EXPECT_THROW(
            transforms.transformPoint(CoordinateFrameId("first"), CoordinateFrameId("second"), {0.0, 0.0, 0.0}),
            CoordinateTransformError);
    }

    TEST(CoordinateTransformServiceTest, ProducesStablePlanHash)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("root"));
        transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("child"),
                                                         CoordinateFrameKind::LocalCartesian,
                                                         LinearUnit::Metre,
                                                         AngleUnit::Radian,
                                                         CoordinateFrameId("root"),
                                                         RigidTransform::identity()));

        const std::string first = transforms.transformChainHash(CoordinateFrameId("child"), CoordinateFrameId("root"));
        const std::string second = transforms.transformChainHash(CoordinateFrameId("child"), CoordinateFrameId("root"));
        EXPECT_FALSE(first.empty());
        EXPECT_EQ(first, second);
    }

} // namespace
