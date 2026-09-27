#include <placamera/frame_camera.h>
#include <placamera/registry.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using namespace placamera;

    RasterModelPtr makeFrameModel()
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 50.0;
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("definition"), intrinsics, {}, PixelConvention::PixelCenter, FrameId("world"));
        auto model = FramePinholeModel::create(
            CameraInstanceId("instance"),
            ImageId("image"),
            definition,
            ImageSize{100, 100},
            Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}));
        return std::make_shared<const FramePinholeModel>(std::move(model));
    }

    TEST(ModelStateTest, PreservesOpaqueTextPayload)
    {
        const std::string payload("json\0payload", 12);
        const ModelState state = ModelState::fromText("frame_pinhole", 1, "application/json", payload);
        EXPECT_EQ(state.modelType, "frame_pinhole");
        EXPECT_EQ(state.schemaVersion, 1);
        EXPECT_EQ(state.mediaType, "application/json");
        EXPECT_EQ(state.payloadAsText(), payload);
    }

    TEST(ModelRegistryTest, RegistersAndConstructsOpaqueModelFactories)
    {
        ModelRegistry registry;
        ASSERT_TRUE(registry.registerFactory("frame_pinhole",
                                             [](const ModelState& state) -> Result<ModelRegistry::ModelPointer>
                                             {
                                                 if (state.schemaVersion != 1 || state.mediaType != "application/json")
                                                 {
                                                     return Result<ModelRegistry::ModelPointer>::failure(
                                                         CameraErrorCode::InvalidModelState, "unsupported test state");
                                                 }
                                                 return Result<ModelRegistry::ModelPointer>::success(makeFrameModel());
                                             }));
        const auto duplicate =
            registry.registerFactory("frame_pinhole",
                                     [](const ModelState&) -> Result<ModelRegistry::ModelPointer>
                                     { return Result<ModelRegistry::ModelPointer>::success(makeFrameModel()); });
        EXPECT_FALSE(duplicate);
        EXPECT_EQ(duplicate.errorCode(), CameraErrorCode::InvalidModelState);
        EXPECT_TRUE(registry.contains("frame_pinhole"));
        EXPECT_EQ(registry.registeredModelTypes(), std::vector<std::string>{"frame_pinhole"});

        const ModelState state = ModelState::fromText("frame_pinhole", 1, "application/json", "{}");
        auto result = registry.create(state);
        ASSERT_TRUE(result) << result.message();
        ASSERT_NE(result.value(), nullptr);
        EXPECT_EQ(result.value()->modelType(), "frame_pinhole");
    }

    TEST(ModelRegistryTest, ReportsUnknownAndMalformedStates)
    {
        ModelRegistry registry;
        const ModelState unknown = ModelState::fromText("unknown", 1, "application/octet-stream", "");
        const auto result = registry.create(unknown);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.errorCode(), CameraErrorCode::UnsupportedModel);

        ModelState malformed = unknown;
        malformed.schemaVersion = 0;
        const auto invalid = registry.create(malformed);
        EXPECT_FALSE(invalid);
        EXPECT_EQ(invalid.errorCode(), CameraErrorCode::InvalidModelState);
    }

} // namespace
