#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/core/model/CameraDefinition.h"
#include "camera/core/model/CameraInstance.h"
#include "camera/core/model/CameraInstanceSet.h"
#include "camera/core/model/CameraModelRegistry.h"

#include <gtest/gtest.h>

#include <memory>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;

    CameraModelFactory makeFactory(CapabilitySet definitionCapabilities, CapabilitySet instanceCapabilities)
    {
        CameraModelFactory factory;
        factory.createDefinition =
            [definitionCapabilities](
                const CameraDefinitionId& id, const CoordinateFrameId& frame, int schemaVersion, std::string_view)
        { return std::make_unique<CameraDefinition>(id, "test_model", frame, schemaVersion, definitionCapabilities); };
        factory.createInstance = [instanceCapabilities](const CameraInstanceId& id,
                                                        const ImageId& image,
                                                        std::shared_ptr<const CameraDefinition> definition,
                                                        std::string_view)
        {
            return std::make_unique<CameraInstance>(
                id, image, std::move(definition), ImageSize{640, 480}, std::nullopt, instanceCapabilities);
        };
        return factory;
    }

    TEST(CameraCoreRegistryTest, RejectsDuplicateModelTypes)
    {
        CameraModelRegistry registry;
        const CameraModelFactory factory =
            makeFactory(CapabilitySet{CapabilityKind::Projection}, CapabilitySet{CapabilityKind::Projection});
        registry.registerFactory("test_model", factory);
        EXPECT_THROW(registry.registerFactory("test_model", factory), CameraRegistryError);
    }

    TEST(CameraCoreRegistryTest, CreatesRegisteredDefinitionAndInstance)
    {
        CameraModelRegistry registry;
        registry.registerFactory(
            "test_model",
            makeFactory(CapabilitySet{CapabilityKind::Projection}, CapabilitySet{CapabilityKind::Projection}));

        auto definition = registry.createDefinition(
            "test_model", CameraDefinitionId("definition-1"), CoordinateFrameId("local"), 1, "{}");
        ASSERT_TRUE(definition);

        auto instance = registry.createInstance("test_model",
                                                CameraInstanceId("instance-1"),
                                                ImageId("image-1"),
                                                std::shared_ptr<const CameraDefinition>(std::move(definition)),
                                                "{}");
        ASSERT_TRUE(instance);
        EXPECT_EQ(instance->imageId().value(), "image-1");
        EXPECT_EQ(instance->definition().modelType(), "test_model");
    }

    TEST(CameraCoreRegistryTest, ReportsMissingCapabilitiesWithContext)
    {
        CameraModelRegistry registry;
        registry.registerFactory(
            "test_model",
            makeFactory(CapabilitySet{CapabilityKind::Projection}, CapabilitySet{CapabilityKind::Projection}));
        auto definition = registry.createDefinition(
            "test_model", CameraDefinitionId("definition-1"), CoordinateFrameId("local"), 1, "{}");
        auto instance = registry.createInstance("test_model",
                                                CameraInstanceId("instance-1"),
                                                ImageId("image-1"),
                                                std::shared_ptr<const CameraDefinition>(std::move(definition)),
                                                "{}");

        const CapabilityCheckResult result =
            requireCapabilities(*instance, CapabilitySet{CapabilityKind::Projection, CapabilityKind::StaticPose});
        EXPECT_FALSE(result.ok());
        ASSERT_EQ(result.missing().size(), 1U);
        EXPECT_EQ(result.missing().front(), CapabilityKind::StaticPose);
        EXPECT_NE(result.message().find("image-1"), std::string::npos);
        EXPECT_NE(result.message().find("test_model"), std::string::npos);
    }

    std::shared_ptr<const CameraInstance> makeInstance(const char* image, const char* frame)
    {
        auto definition = std::make_shared<CameraDefinition>(CameraDefinitionId(std::string("definition-") + image),
                                                             "test_model",
                                                             CoordinateFrameId(frame),
                                                             1,
                                                             CapabilitySet{});
        return std::make_shared<CameraInstance>(CameraInstanceId(std::string("instance-") + image),
                                                ImageId(image),
                                                std::move(definition),
                                                ImageSize{640, 480},
                                                std::nullopt,
                                                CapabilitySet{});
    }

    TEST(CameraCoreInstanceSetTest, EmptySetHasNoCommonFrameFailure)
    {
        const CameraInstanceSet set;
        const CameraInstanceSetFrameResult result = set.requireCommonWorldFrame();

        EXPECT_TRUE(result.ok());
        EXPECT_FALSE(result.commonFrame.has_value());
    }

    TEST(CameraCoreInstanceSetTest, ReportsCommonFrameForSingleFrameSet)
    {
        CameraInstanceSet set;
        ASSERT_TRUE(set.add(makeInstance("image-1", "world")));
        ASSERT_TRUE(set.add(makeInstance("image-2", "world")));

        const CameraInstanceSetFrameResult result = set.requireCommonWorldFrame();

        ASSERT_TRUE(result.ok());
        ASSERT_TRUE(result.commonFrame.has_value());
        EXPECT_EQ(result.commonFrame->value(), "world");
    }

    TEST(CameraCoreInstanceSetTest, ReportsEveryImageWithConflictingFrame)
    {
        CameraInstanceSet set;
        ASSERT_TRUE(set.add(makeInstance("image-1", "world-a")));
        ASSERT_TRUE(set.add(makeInstance("image-2", "world-b")));
        ASSERT_TRUE(set.add(makeInstance("image-3", "world-c")));

        const CameraInstanceSetFrameResult result = set.requireCommonWorldFrame();

        ASSERT_FALSE(result.ok());
        ASSERT_EQ(result.failures.size(), 2U);
        EXPECT_EQ(result.failures[0].image.value(), "image-2");
        EXPECT_EQ(result.failures[0].frame.value(), "world-b");
        EXPECT_EQ(result.failures[1].image.value(), "image-3");
        EXPECT_EQ(result.failures[1].frame.value(), "world-c");
    }

} // namespace
