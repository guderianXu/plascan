#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/models/rpc/RpcDefinition.h"
#include "camera/models/rpc/RpcBiasAdjustment.h"
#include "camera/models/rpc/RpcInstance.h"
#include "camera/models/rpc/RpcIntersectionService.h"
#include "camera/models/rpc/RpcProjection.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models::rpc;

    std::shared_ptr<const RpcDefinition> makeDefinition(double heightCoefficient)
    {
        RpcDefinition::Parameters parameters;
        parameters.lineOffset = 500.0;
        parameters.sampleOffset = 500.0;
        parameters.latitudeOffset = 20.0;
        parameters.longitudeOffset = 110.0;
        parameters.heightOffset = 1000.0;
        parameters.lineScale = 1000.0;
        parameters.sampleScale = 1000.0;
        parameters.latitudeScale = 0.1;
        parameters.longitudeScale = 0.1;
        parameters.heightScale = 1000.0;
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleNumerator[3] = heightCoefficient;
        parameters.sampleDenominator[0] = 1.0;
        return RpcDefinition::create(
            CameraDefinitionId("rpc-definition"), CoordinateFrameId("wgs84-geodetic"), parameters);
    }

    RpcInstance makeInstance(std::shared_ptr<const RpcDefinition> definition, const char* instanceId = "rpc-instance")
    {
        return RpcInstance::create(
            CameraInstanceId(instanceId), ImageId("image-rpc"), std::move(definition), ImageSize{1000, 1000});
    }

    TEST(RpcTypedModelTest, ProjectsAndInvertsWithoutStaticPose)
    {
        const auto instance = makeInstance(makeDefinition(0.25));
        const RpcDefinition::GeodeticCoordinate ground{{110.01, 19.98, 1300.0}};
        ImagePoint image;
        ASSERT_TRUE(RpcProjection::groundToImage(instance, ground, &image));
        EXPECT_NEAR(image.sample, 675.0, 1.0e-9);
        EXPECT_NEAR(image.line, 300.0, 1.0e-9);

        RpcDefinition::GeodeticCoordinate restored;
        ASSERT_TRUE(RpcProjection::imageToGroundAtHeight(instance, image, ground[2], &restored));
        EXPECT_NEAR(restored[0], ground[0], 1.0e-12);
        EXPECT_NEAR(restored[1], ground[1], 1.0e-12);
        EXPECT_DOUBLE_EQ(restored[2], ground[2]);

        const CapabilityCheckResult check = requireCapabilities(instance, CapabilitySet{CapabilityKind::StaticPose});
        EXPECT_FALSE(check.ok());
        ASSERT_EQ(check.missing().size(), 1U);
        EXPECT_EQ(check.missing().front(), CapabilityKind::StaticPose);
    }

    TEST(RpcTypedModelTest, AppliesInstanceCorrectionAndBuildsApproximateRay)
    {
        ImageCorrection correction;
        correction.sampleOffsetPixels = 2.0;
        correction.sampleSamplePixels = 3.0;
        correction.sampleLinePixels = -4.0;
        correction.lineOffsetPixels = -1.0;
        correction.lineSamplePixels = 5.0;
        correction.lineLinePixels = 2.0;
        const auto instance = RpcInstance::create(CameraInstanceId("rpc-instance"),
                                                  ImageId("image-rpc"),
                                                  makeDefinition(0.25),
                                                  ImageSize{1000, 1000},
                                                  correction);

        const RpcDefinition::GeodeticCoordinate ground{{110.01, 19.98, 1300.0}};
        ImagePoint image;
        ASSERT_TRUE(RpcProjection::groundToImage(instance, ground, &image));
        EXPECT_NEAR(image.sample, 678.325, 1.0e-9);
        EXPECT_NEAR(image.line, 299.475, 1.0e-9);

        RpcRay ray;
        ASSERT_TRUE(RpcProjection::ray(instance, image, &ray));
        const double norm = std::hypot(ray.direction[0], std::hypot(ray.direction[1], ray.direction[2]));
        EXPECT_NEAR(norm, 1.0, 1.0e-12);
    }

    TEST(RpcTypedModelTest, RejectsInvalidScales)
    {
        RpcDefinition::Parameters parameters;
        EXPECT_THROW(
            RpcDefinition::create(CameraDefinitionId("invalid"), CoordinateFrameId("wgs84-geodetic"), parameters),
            CameraValidationError);
    }

    TEST(RpcTypedModelTest, IntersectsOpposingRpcLookDirectionsWithoutStaticCenters)
    {
        const auto first = makeInstance(makeDefinition(0.25), "rpc-first");
        const auto second = RpcInstance::create(
            CameraInstanceId("rpc-second"), ImageId("image-rpc-second"), makeDefinition(-0.25), ImageSize{1000, 1000});
        const RpcDefinition::GeodeticCoordinate expected{{110.01, 19.98, 1300.0}};
        ImagePoint firstImage;
        ImagePoint secondImage;
        ASSERT_TRUE(RpcProjection::groundToImage(first, expected, &firstImage));
        ASSERT_TRUE(RpcProjection::groundToImage(second, expected, &secondImage));

        RpcIntersectionResult result;
        ASSERT_TRUE(RpcIntersectionService::intersect(first, firstImage, second, secondImage, &result));
        EXPECT_NEAR(result.geodetic[0], expected[0], 1.0e-8);
        EXPECT_NEAR(result.geodetic[1], expected[1], 1.0e-8);
        EXPECT_NEAR(result.geodetic[2], expected[2], 0.1);
        EXPECT_LT(result.reprojectionRmsPixels, 1.0e-4);
    }

    TEST(RpcTypedModelTest, EstimatesAffineCorrectionOnTypedInstance)
    {
        const auto instance = makeInstance(makeDefinition(0.25));
        ImageCorrection expected;
        expected.sampleOffsetPixels = 1.5;
        expected.sampleSamplePixels = 0.75;
        expected.sampleLinePixels = -0.25;
        expected.lineOffsetPixels = -2.0;
        expected.lineSamplePixels = 0.4;
        expected.lineLinePixels = 0.6;
        const auto adjusted = instance.withImageCorrection(CameraInstanceId("rpc-adjusted"), expected);

        std::vector<RpcControlPointObservation> observations;
        for (const RpcDefinition::GeodeticCoordinate ground :
             {RpcDefinition::GeodeticCoordinate{109.98, 19.98, 800.0},
              RpcDefinition::GeodeticCoordinate{110.02, 19.98, 1000.0},
              RpcDefinition::GeodeticCoordinate{109.98, 20.02, 1200.0},
              RpcDefinition::GeodeticCoordinate{110.02, 20.02, 1400.0},
              RpcDefinition::GeodeticCoordinate{110.00, 20.00, 1100.0}})
        {
            ImagePoint observed;
            ASSERT_TRUE(RpcProjection::groundToImage(adjusted, ground, &observed));
            observations.push_back({ground, observed, 1.0});
        }

        RpcBiasAdjustmentResult result;
        std::string error;
        ASSERT_TRUE(estimateRpcImageCorrection(instance, observations, &result, &error)) << error;
        EXPECT_NEAR(result.correction.sampleOffsetPixels, expected.sampleOffsetPixels, 1.0e-9);
        EXPECT_NEAR(result.correction.sampleSamplePixels, expected.sampleSamplePixels, 1.0e-9);
        EXPECT_NEAR(result.correction.sampleLinePixels, expected.sampleLinePixels, 1.0e-9);
        EXPECT_NEAR(result.correction.lineOffsetPixels, expected.lineOffsetPixels, 1.0e-9);
        EXPECT_NEAR(result.correction.lineSamplePixels, expected.lineSamplePixels, 1.0e-9);
        EXPECT_NEAR(result.correction.lineLinePixels, expected.lineLinePixels, 1.0e-9);
        EXPECT_LT(result.rmsAfterPixels, 1.0e-9);
    }

} // namespace
