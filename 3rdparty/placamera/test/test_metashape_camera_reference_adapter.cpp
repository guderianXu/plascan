#include <placamera/reference/MetashapeCameraReferenceAdapter.h>

#include <gtest/gtest.h>

#include <cmath>

namespace
{

    using namespace placamera;
    using namespace placamera::reference;

    TEST(MetashapeCameraReferenceAdapterTest, ConvertsYprDegreesAndAnisotropicCovariance)
    {
        MetashapeCameraReference source;
        source.positionMeters = Vector3{{1.0, 2.0, 3.0}};
        source.positionCovarianceMetersSquared = std::array<double, 9>{{4.0, 0.5, 0.0, 0.5, 9.0, 0.0, 0.0, 0.0, 16.0}};
        source.yawPitchRollDegrees = Vector3{{10.0, 2.0, -3.0}};
        source.yawPitchRollCovarianceDegreesSquared =
            std::array<double, 9>{{1.0, 0.1, 0.0, 0.1, 4.0, 0.2, 0.0, 0.2, 9.0}};

        const auto adapted = adaptMetashapeCameraReference(
            ImageId("image"), ReferenceSourceId("metashape"), FrameId("reference"), source);
        ASSERT_TRUE(adapted) << adapted.message();
        ASSERT_TRUE(adapted.value().position);
        ASSERT_TRUE(adapted.value().orientation);
        ASSERT_TRUE(adapted.value().covariance);
        EXPECT_EQ(adapted.value().covariance->components(), PoseCovarianceComponents::PositionAndRotation);

        constexpr double radians = 0.017453292519943295769;
        const double yaw_cosine = std::cos(10.0 * radians);
        const double yaw_sine = std::sin(10.0 * radians);
        EXPECT_NEAR((*adapted.value().orientation)[0],
                    yaw_cosine * std::cos(-3.0 * radians) +
                        yaw_sine * std::sin(2.0 * radians) * std::sin(-3.0 * radians),
                    1.0e-14);
        EXPECT_NEAR(adapted.value().covariance->matrixValues()[0], 4.0, 1.0e-14);
        EXPECT_NEAR(adapted.value().covariance->matrixValues()[1], 0.5, 1.0e-14);
        EXPECT_GT(adapted.value().covariance->matrixValues()[21], 0.0);
        EXPECT_GT(adapted.value().covariance->matrixValues()[28], 0.0);
        EXPECT_GT(adapted.value().covariance->matrixValues()[35], 0.0);
    }

    TEST(MetashapeCameraReferenceAdapterTest, PreservesIndependentPositionAndRotationStates)
    {
        MetashapeCameraReference position;
        position.positionMeters = Vector3{{1.0, 2.0, 3.0}};
        position.positionCovarianceMetersSquared = std::array<double, 9>{{1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0}};
        const auto position_only = adaptMetashapeCameraReference(
            ImageId("position"), ReferenceSourceId("metashape"), FrameId("reference"), position);
        ASSERT_TRUE(position_only) << position_only.message();
        EXPECT_TRUE(position_only.value().position);
        EXPECT_FALSE(position_only.value().orientation);
        ASSERT_TRUE(position_only.value().covariance);
        EXPECT_EQ(position_only.value().covariance->components(), PoseCovarianceComponents::PositionOnly);

        MetashapeCameraReference rotation;
        rotation.yawPitchRollDegrees = Vector3{{0.0, 0.0, 0.0}};
        rotation.yawPitchRollCovarianceDegreesSquared =
            std::array<double, 9>{{1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0}};
        const auto rotation_only = adaptMetashapeCameraReference(
            ImageId("rotation"), ReferenceSourceId("metashape"), FrameId("reference"), rotation);
        ASSERT_TRUE(rotation_only) << rotation_only.message();
        EXPECT_FALSE(rotation_only.value().position);
        EXPECT_TRUE(rotation_only.value().orientation);
        ASSERT_TRUE(rotation_only.value().covariance);
        EXPECT_EQ(rotation_only.value().covariance->components(), PoseCovarianceComponents::RotationOnly);
    }

} // namespace
