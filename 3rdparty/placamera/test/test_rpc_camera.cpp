#include <placamera/rpc_camera.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <utility>

namespace
{

    using namespace placamera;

    std::shared_ptr<const RpcDefinition> makeDefinition(double heightCoefficient = 0.25)
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
        return RpcDefinition::create(CameraDefinitionId("rpc-definition"), FrameId("wgs84-ecef"), parameters);
    }

    RpcModel makeModel(std::shared_ptr<const RpcDefinition> definition, RpcImageCorrection correction = {})
    {
        return RpcModel::create(CameraInstanceId("rpc-instance"),
                                ImageId("rpc-image"),
                                std::move(definition),
                                ImageSize{1000, 1000},
                                correction,
                                TimeReference::create(TimeScale::Utc, 100.0));
    }

    RpcModel makeGroundCorrectedModel(RpcGroundCorrection correction)
    {
        return RpcModel::createWithCorrection(CameraInstanceId("rpc-ground-instance"),
                                              ImageId("rpc-image"),
                                              makeDefinition(),
                                              ImageSize{1000, 1000},
                                              RpcCorrection::groundCoordinates(correction),
                                              TimeReference::create(TimeScale::Utc, 100.0));
    }

    TEST(RpcModelTest, ProjectsAndInvertsRpc00BGeodeticCoordinates)
    {
        const RpcModel model = makeModel(makeDefinition());
        const GeodeticCoordinate ground{110.01, 19.98, 1300.0};
        const auto projected = model.groundToImageGeodetic(ground);
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 675.0, 1.0e-9);
        EXPECT_NEAR(projected.value().image.line, 300.0, 1.0e-9);
        EXPECT_FALSE(projected.value().positiveDepth.has_value());
        ASSERT_TRUE(projected.value().acquisitionTime.has_value());

        const auto restored = model.imageToGroundAtHeight(projected.value().image, ground.heightMeters);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_LT(restored.achievedPrecisionPixels(), 1.0e-7);
        EXPECT_NEAR(restored.value().longitudeDegrees, ground.longitudeDegrees, 1.0e-12);
        EXPECT_NEAR(restored.value().latitudeDegrees, ground.latitudeDegrees, 1.0e-12);
        EXPECT_DOUBLE_EQ(restored.value().heightMeters, ground.heightMeters);
    }

    TEST(RpcModelTest, UsesCartesianGroundFrameThroughRasterModelContract)
    {
        const RpcModel rpc_model = makeModel(makeDefinition());
        const GeodeticCoordinate geodetic{110.01, 19.98, 1300.0};
        const auto cartesian = geodeticToCartesian(geodetic, rpc_model.rpcDefinition().ellipsoid());
        ASSERT_TRUE(cartesian) << cartesian.message();

        const RasterModel& model = rpc_model;
        const auto projected = model.groundToImage(GroundCoordinate{FrameId("wgs84-ecef"), cartesian.value()});
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 675.0, 1.0e-6);
        EXPECT_NEAR(projected.value().image.line, 300.0, 1.0e-6);
        EXPECT_FALSE(model.capabilities().contains(CapabilityKind::StaticPose));

        const auto locus = model.imageToImagingLocus(projected.value().image);
        ASSERT_TRUE(locus) << locus.message();
        EXPECT_EQ(locus.value().origin.frame, FrameId("wgs84-ecef"));
        const double norm =
            std::hypot(locus.value().direction[0], std::hypot(locus.value().direction[1], locus.value().direction[2]));
        EXPECT_NEAR(norm, 1.0, 1.0e-12);
    }

    TEST(RpcModelTest, AppliesPerImageAffineCorrection)
    {
        RpcImageCorrection correction;
        correction.sampleOffsetPixels = 2.0;
        correction.sampleSamplePixels = 3.0;
        correction.sampleLinePixels = -4.0;
        correction.lineOffsetPixels = -1.0;
        correction.lineSamplePixels = 5.0;
        correction.lineLinePixels = 2.0;
        const RpcModel model = makeModel(makeDefinition(), correction);

        const auto projected = model.groundToImageGeodetic({110.01, 19.98, 1300.0});
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 678.325, 1.0e-9);
        EXPECT_NEAR(projected.value().image.line, 299.475, 1.0e-9);
    }

    TEST(RpcModelTest, AppliesPhysicalGroundDomainAffineCorrection)
    {
        RpcGroundCorrection correction;
        correction.sampleOffsetPixels = 1.0;
        correction.lineOffsetPixels = -2.0;
        correction.sampleLongitudePixelsPerDegree = 10.0;
        correction.sampleLatitudePixelsPerDegree = 20.0;
        correction.sampleHeightPixelsPerMeter = 0.001;
        correction.lineLongitudePixelsPerDegree = -5.0;
        correction.lineLatitudePixelsPerDegree = 4.0;
        correction.lineHeightPixelsPerMeter = 0.002;
        const RpcModel model = makeGroundCorrectedModel(correction);

        const GeodeticCoordinate ground{110.01, 19.98, 1300.0};
        const auto projected = model.groundToImageGeodetic(ground);
        ASSERT_TRUE(projected) << projected.message();
        EXPECT_NEAR(projected.value().image.sample, 676.0, 1.0e-9);
        EXPECT_NEAR(projected.value().image.line, 298.47, 1.0e-9);

        const auto restored = model.imageToGroundAtHeight(projected.value().image, ground.heightMeters);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().longitudeDegrees, ground.longitudeDegrees, 3.0e-12);
        EXPECT_NEAR(restored.value().latitudeDegrees, ground.latitudeDegrees, 3.0e-12);
    }

    TEST(RpcModelTest, KeepsCorrectionDomainsDistinctDuringAccessAndOptimization)
    {
        RpcGroundCorrection correction;
        correction.sampleLongitudePixelsPerDegree = 10.0;
        const RpcModel model = makeGroundCorrectedModel(correction);
        EXPECT_EQ(model.correctionDomain(), RpcCorrectionDomain::GroundCoordinates);
        EXPECT_EQ(model.normalizedImageCorrection(), nullptr);
        ASSERT_NE(model.groundCorrection(), nullptr);
        EXPECT_THROW(static_cast<void>(model.imageCorrection()), CameraValidationError);

        const OptimizationLayout layout = model.optimizationLayout();
        ASSERT_TRUE(layout.isValid());
        EXPECT_EQ(layout.parameterCount(), 8U);
        OptimizationUpdate update{CameraInstanceId("updated-ground"), std::nullopt, std::vector<double>(8, 0.0)};
        update.delta[2] = 2.5;
        const auto updated_base = model.withOptimizationUpdate(update);
        ASSERT_TRUE(updated_base) << updated_base.message();
        const auto* updated = dynamic_cast<const RpcModel*>(updated_base.value().get());
        ASSERT_NE(updated, nullptr);
        ASSERT_NE(updated->groundCorrection(), nullptr);
        EXPECT_DOUBLE_EQ(updated->groundCorrection()->sampleLongitudePixelsPerDegree, 12.5);
        EXPECT_EQ(updated->correctionDomain(), RpcCorrectionDomain::GroundCoordinates);
    }

    TEST(RpcModelTest, SupportsCustomSphericalReferenceBodies)
    {
        const ReferenceEllipsoid moon{1737400.0, 0.0};
        const GeodeticCoordinate expected{35.0, -22.0, 1250.0};
        const auto cartesian = geodeticToCartesian(expected, moon);
        ASSERT_TRUE(cartesian) << cartesian.message();
        const auto restored = cartesianToGeodetic(cartesian.value(), moon);
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().longitudeDegrees, expected.longitudeDegrees, 1.0e-12);
        EXPECT_NEAR(restored.value().latitudeDegrees, expected.latitudeDegrees, 1.0e-12);
        EXPECT_NEAR(restored.value().heightMeters, expected.heightMeters, 1.0e-8);
    }

    TEST(RpcModelTest, ReportsFrameMismatchAndRejectsInvalidDefinition)
    {
        const RpcModel model = makeModel(makeDefinition());
        const auto mismatch = model.groundToImage(GroundCoordinate{FrameId("other"), {6378137.0, 0.0, 0.0}});
        EXPECT_FALSE(mismatch);
        EXPECT_EQ(mismatch.errorCode(), CameraErrorCode::FrameMismatch);

        RpcParameters invalid;
        EXPECT_THROW(RpcDefinition::create(CameraDefinitionId("invalid"), FrameId("wgs84-ecef"), invalid),
                     CameraValidationError);
    }

} // namespace
