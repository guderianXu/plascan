#include <placamera/rpc_adjustment.h>

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

namespace
{

    using namespace placamera;

    std::shared_ptr<const RpcDefinition> makeDefinition(std::string id, double heightCoefficient)
    {
        RpcParameters parameters;
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
        return RpcDefinition::create(CameraDefinitionId(std::move(id)), FrameId("wgs84-ecef"), parameters);
    }

    RpcModel makeModel(std::string instance,
                       std::string image,
                       std::shared_ptr<const RpcDefinition> definition,
                       RpcImageCorrection correction = {})
    {
        return RpcModel::create(CameraInstanceId(std::move(instance)),
                                ImageId(std::move(image)),
                                std::move(definition),
                                ImageSize{2000, 2000},
                                correction);
    }

    TEST(RpcBiasAdjustmentTest, RecoversAffineImageCorrectionFromControlPoints)
    {
        RpcImageCorrection expected;
        expected.sampleOffsetPixels = 2.0;
        expected.sampleSamplePixels = 3.0;
        expected.sampleLinePixels = -4.0;
        expected.lineOffsetPixels = -1.0;
        expected.lineSamplePixels = 5.0;
        expected.lineLinePixels = 2.0;
        const auto definition = makeDefinition("rpc-definition", 0.25);
        const RpcModel adjusted = makeModel("adjusted", "image", definition, expected);

        const std::vector<GeodeticCoordinate> points{{110.00, 20.00, 1000.0},
                                                     {110.01, 20.00, 1000.0},
                                                     {110.00, 20.01, 1000.0},
                                                     {110.01, 20.01, 1200.0},
                                                     {109.99, 19.99, 800.0}};
        std::vector<RpcControlPointObservation> observations;
        for (const GeodeticCoordinate& point : points)
        {
            const auto observed = adjusted.groundToImageGeodetic(point);
            ASSERT_TRUE(observed) << observed.message();
            observations.push_back({point, observed.value().image, 1.0});
        }

        const auto result = estimateRpcImageCorrection(adjusted, observations);
        ASSERT_TRUE(result) << result.message();
        EXPECT_NEAR(result.value().correction.sampleOffsetPixels, expected.sampleOffsetPixels, 1.0e-9);
        EXPECT_NEAR(result.value().correction.sampleSamplePixels, expected.sampleSamplePixels, 1.0e-9);
        EXPECT_NEAR(result.value().correction.sampleLinePixels, expected.sampleLinePixels, 1.0e-9);
        EXPECT_NEAR(result.value().correction.lineOffsetPixels, expected.lineOffsetPixels, 1.0e-9);
        EXPECT_NEAR(result.value().correction.lineSamplePixels, expected.lineSamplePixels, 1.0e-9);
        EXPECT_NEAR(result.value().correction.lineLinePixels, expected.lineLinePixels, 1.0e-9);
        EXPECT_LT(result.value().rmsAfterPixels, 1.0e-9);
        EXPECT_GT(result.value().rmsBeforePixels, result.value().rmsAfterPixels);
    }

    TEST(RpcIntersectionTest, RecoversGeodeticPointFromConvergingRpcRays)
    {
        const RpcModel first = makeModel("first", "first-image", makeDefinition("first-definition", 0.25));
        const RpcModel second = makeModel("second", "second-image", makeDefinition("second-definition", -0.25));
        const GeodeticCoordinate expected{110.012, 19.987, 1350.0};
        const auto first_image = first.groundToImageGeodetic(expected);
        const auto second_image = second.groundToImageGeodetic(expected);
        ASSERT_TRUE(first_image);
        ASSERT_TRUE(second_image);

        RpcIntersectionOptions options;
        options.pixelTolerance = 1.0e-6;
        options.positionToleranceMeters = 1.0e-4;
        const auto intersection =
            intersectRpc(first, first_image.value().image, second, second_image.value().image, options);
        ASSERT_TRUE(intersection) << intersection.message();
        EXPECT_NEAR(intersection.value().geodetic.longitudeDegrees, expected.longitudeDegrees, 1.0e-8);
        EXPECT_NEAR(intersection.value().geodetic.latitudeDegrees, expected.latitudeDegrees, 1.0e-8);
        EXPECT_NEAR(intersection.value().geodetic.heightMeters, expected.heightMeters, 1.0e-3);
        EXPECT_LT(intersection.value().reprojectionRmsPixels, options.pixelTolerance);
    }

    TEST(RpcIntersectionTest, RejectsMismatchedFramesAndDegenerateControlSets)
    {
        const RpcModel first = makeModel("first", "first-image", makeDefinition("first-definition", 0.25));
        RpcParameters parameters = first.rpcDefinition().parameters();
        const auto other_definition =
            RpcDefinition::create(CameraDefinitionId("other-definition"), FrameId("other-ecef"), parameters);
        const RpcModel other = makeModel("other", "other-image", other_definition);
        EXPECT_FALSE(intersectRpc(first, {500.0, 500.0}, other, {500.0, 500.0}));

        const auto adjustment = estimateRpcImageCorrection(
            first, {{GeodeticCoordinate{110.0, 20.0, 1000.0}, ImageCoordinate{500.0, 500.0}, 1.0}});
        EXPECT_FALSE(adjustment);
    }

} // namespace
