#include <initializer_list>
#include <string>

#include <gtest/gtest.h>

#include <placamera/tsai.h>

namespace
{

    TEST(FramePinholeTsaiPlaCameraTest, LoadsDefinitionAndPoseWithoutLegacyCameraState)
    {
        const std::string path = std::string(PLACAMERA_TSAI_FIXTURE_DIR) + "/1.tsai";
        const auto parsed = placamera::loadTsaiFramePinhole(
            path, placamera::CameraDefinitionId("tsai-definition"), placamera::FrameId("project-world"));
        ASSERT_TRUE(parsed) << parsed.message();
        EXPECT_EQ(parsed.value().definition->definitionId().value(), "tsai-definition");
        EXPECT_EQ(parsed.value().definition->groundFrame().value(), "project-world");
        EXPECT_EQ(parsed.value().pose.frame.value(), "project-world");
        EXPECT_NEAR(parsed.value().definition->intrinsics().focalX * parsed.value().definition->intrinsics().pixelPitch,
                    16.573351657,
                    1.0e-10);
        EXPECT_NEAR(parsed.value().definition->intrinsics().principalX *
                        parsed.value().definition->intrinsics().pixelPitch,
                    3.811246229,
                    1.0e-10);

        const auto bound = placamera::bindFramePinhole(parsed.value(),
                                                       {placamera::CameraInstanceId("tsai-instance"),
                                                        placamera::ImageId("tsai-image"),
                                                        placamera::ImageSize{4000, 3000}});
        ASSERT_TRUE(bound) << bound.message();
        const auto& model = *bound.value();
        const auto& rotation = parsed.value().pose.cameraToWorldRotation;
        const placamera::GroundCoordinate on_axis{placamera::FrameId("project-world"),
                                                  {parsed.value().pose.center[0] + 10.0 * rotation[2],
                                                   parsed.value().pose.center[1] + 10.0 * rotation[5],
                                                   parsed.value().pose.center[2] + 10.0 * rotation[8]}};
        const auto projected = model.groundToImage(on_axis);
        ASSERT_TRUE(projected.ok());
        EXPECT_NEAR(projected.value().image.sample, parsed.value().definition->intrinsics().principalX, 1.0e-6);
        EXPECT_NEAR(projected.value().image.line, parsed.value().definition->intrinsics().principalY, 1.0e-6);
    }

    TEST(FramePinholeTsaiPlaCameraTest, ReportsSourcePathForMissingFile)
    {
        const std::string path = std::string(PLACAMERA_TSAI_FIXTURE_DIR) + "/missing-camera.tsai";
        const auto parsed = placamera::loadTsaiFramePinhole(
            path, placamera::CameraDefinitionId("tsai-definition"), placamera::FrameId("project-world"));
        EXPECT_FALSE(parsed);
        EXPECT_EQ(parsed.errorCode(), placamera::CameraErrorCode::IoFailure);
        EXPECT_EQ(parsed.error().source, path);
    }

    TEST(FramePinholeTsaiPlaCameraTest, ImportsAllTrackedTsaiRotations)
    {
        for (const int index : {1, 2, 3, 4, 5})
        {
            const std::string path = std::string(PLACAMERA_TSAI_FIXTURE_DIR) + "/" + std::to_string(index) + ".tsai";
            const auto parsed = placamera::loadTsaiFramePinhole(
                path, placamera::CameraDefinitionId("tsai-definition"), placamera::FrameId("project-world"));
            ASSERT_TRUE(parsed) << path << ": " << parsed.message();
            EXPECT_EQ(parsed.value().pose.frame, parsed.value().definition->groundFrame());
        }
    }

    TEST(FramePinholeTsaiPlaCameraTest, RejectsReflectedRotation)
    {
        const std::string path = std::string(PLACAMERA_TSAI_FIXTURE_DIR) + "/invalid_rotation.tsai";
        const auto parsed = placamera::loadTsaiFramePinhole(
            path, placamera::CameraDefinitionId("tsai-definition"), placamera::FrameId("project-world"));
        EXPECT_FALSE(parsed);
        EXPECT_EQ(parsed.error().source, path);
        EXPECT_NE(parsed.message().find("rotation"), std::string::npos);
    }

} // namespace
