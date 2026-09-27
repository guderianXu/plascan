#include <placamera/types.h>
#include <placamera/result.h>

#include <gtest/gtest.h>
#include <placoordinate/types/CoordinateErrors.h>

#include <limits>
#include <type_traits>

namespace
{

    using namespace placamera;

    TEST(PlaCameraTypesTest, StrongIdentifiersRejectBlankValues)
    {
        EXPECT_THROW(FrameId(""), placoordinate::CoordinateValidationError);
        EXPECT_THROW(CameraInstanceId("   \t"), CameraValidationError);
        EXPECT_EQ(FrameId("body-fixed").value(), "body-fixed");
    }

    TEST(PlaCameraTypesTest, ReusesPlaCoordinateFrameAndTimeTypes)
    {
        static_assert(std::is_same_v<FrameId, placoordinate::CoordinateFrameId>);
        static_assert(std::is_same_v<TimeScale, placoordinate::TimeScale>);
        static_assert(std::is_same_v<TimeReference, placoordinate::TimeReference>);
    }

    TEST(PlaCameraTypesTest, PoseAcceptsOnlyProperFiniteRotations)
    {
        const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        const Pose pose = Pose::create(FrameId("world"), {1.0, 2.0, 3.0}, identity);
        EXPECT_EQ(pose.center, (Vector3{1.0, 2.0, 3.0}));

        RotationMatrix reflection = identity;
        reflection[0] = -1.0;
        EXPECT_THROW(Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, reflection), CameraValidationError);
    }

    TEST(PlaCameraTypesTest, TimeReferenceRejectsNonFiniteSeconds)
    {
        EXPECT_THROW(TimeReference::create(TimeScale::Tdb, std::numeric_limits<double>::infinity()),
                     placoordinate::CoordinateValidationError);
        EXPECT_DOUBLE_EQ(TimeReference::create(TimeScale::Relative, 12.5).seconds, 12.5);
    }

    TEST(PlaCameraTypesTest, SuccessfulEvaluationHasNoErrorCode)
    {
        const auto result = EvaluationResult<int>::success(42);
        ASSERT_TRUE(result);
        EXPECT_EQ(result.errorCode(), CameraErrorCode::None);
        EXPECT_EQ(result.value(), 42);
    }

} // namespace
