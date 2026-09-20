#include "camera/core/types/CameraIds.h"
#include "camera/core/types/CameraPose.h"
#include "camera/core/types/CameraUncertainty.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace
{

    using namespace xjw::camera_core;

    TEST(CameraCoreTypesTest, RejectsEmptyStrongIds)
    {
        EXPECT_THROW(CameraDefinitionId(std::string()), CameraValidationError);
        EXPECT_THROW(ImageId(std::string(" ")), CameraValidationError);
    }

    TEST(CameraCoreTypesTest, RejectsNonRotationPose)
    {
        EXPECT_THROW(Pose::create(xjw::coordinate_system::CoordinateFrameId(std::string("local")),
                                  {0.0, 0.0, 0.0},
                                  {2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
                     CameraValidationError);
    }

    TEST(CameraCoreTypesTest, RejectsInvalidCovarianceShape)
    {
        EXPECT_THROW(PoseCovariance::diagonal({1.0, 2.0, 3.0}), CameraValidationError);
        EXPECT_THROW(PoseCovariance::symmetric({1.0, 2.0}), CameraValidationError);
    }

} // namespace
