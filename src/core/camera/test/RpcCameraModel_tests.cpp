#include "CameraModel.h"
#include "RpcBiasAdjustment.h"
#include "RpcCameraIO.h"
#include "RpcCameraModel.h"
#include "RpcStereoIntersection.h"

#include <cpl_vsi.h>
#include <gdal_priv.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    xjw::RpcCameraModel makeLinearRpc(double heightCoefficient)
    {
        xjw::RpcCameraModel::Parameters parameters;
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

        xjw::RpcCameraModel camera;
        EXPECT_TRUE(camera.setParameters(parameters));
        camera.setImageSize(xjw::CameraImageSize{1000, 1000});
        return camera;
    }

    std::string coefficients(const xjw::RpcCameraModel::Coefficients& values)
    {
        std::ostringstream stream;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index > 0)
            {
                stream << ' ';
            }
            stream << values[index];
        }
        return stream.str();
    }

    xjw::RpcMetadata metadataFor(const xjw::RpcCameraModel& camera)
    {
        const auto& parameters = camera.parameters();
        return {{"LINE_OFF", std::to_string(parameters.lineOffset)},
                {"SAMP_OFF", std::to_string(parameters.sampleOffset)},
                {"LAT_OFF", std::to_string(parameters.latitudeOffset)},
                {"LONG_OFF", std::to_string(parameters.longitudeOffset)},
                {"HEIGHT_OFF", std::to_string(parameters.heightOffset)},
                {"LINE_SCALE", std::to_string(parameters.lineScale)},
                {"SAMP_SCALE", std::to_string(parameters.sampleScale)},
                {"LAT_SCALE", std::to_string(parameters.latitudeScale)},
                {"LONG_SCALE", std::to_string(parameters.longitudeScale)},
                {"HEIGHT_SCALE", std::to_string(parameters.heightScale)},
                {"LINE_NUM_COEFF", coefficients(parameters.lineNumerator)},
                {"LINE_DEN_COEFF", coefficients(parameters.lineDenominator)},
                {"SAMP_NUM_COEFF", coefficients(parameters.sampleNumerator)},
                {"SAMP_DEN_COEFF", coefficients(parameters.sampleDenominator)}};
    }

} // namespace

TEST(RpcCameraModel, ValidatesRequiredScalesAndDenominators)
{
    xjw::RpcCameraModel camera;
    xjw::RpcCameraModel::Parameters parameters;
    std::string error;
    EXPECT_FALSE(camera.setParameters(parameters, &error));
    EXPECT_FALSE(camera.isValid());
    EXPECT_FALSE(error.empty());
}

TEST(RpcCameraModel, ProjectsAndInvertsAtKnownHeight)
{
    const xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    const xjw::RpcCameraModel::GeodeticCoordinate ground{{110.01, 19.98, 1300.0}};
    xjw::CameraImageCoordinate image;
    ASSERT_TRUE(camera.groundToImageGeodetic(ground, &image));
    EXPECT_NEAR(image.sample, 675.0, 1.0e-9);
    EXPECT_NEAR(image.line, 300.0, 1.0e-9);

    xjw::RpcCameraModel::GeodeticCoordinate restored;
    ASSERT_TRUE(camera.imageToGroundAtHeight(image, ground[2], &restored));
    EXPECT_NEAR(restored[0], ground[0], 1.0e-12);
    EXPECT_NEAR(restored[1], ground[1], 1.0e-12);
    EXPECT_DOUBLE_EQ(restored[2], ground[2]);
}

TEST(RpcCameraModel, AppliesAndInvertsAffineImageCorrection)
{
    xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    xjw::RpcCameraModel::ImageCorrection correction;
    correction.sampleOffsetPixels = 2.0;
    correction.sampleSamplePixels = 3.0;
    correction.sampleLinePixels = -4.0;
    correction.lineOffsetPixels = -1.0;
    correction.lineSamplePixels = 5.0;
    correction.lineLinePixels = 2.0;
    ASSERT_TRUE(camera.setImageCorrection(correction));

    const xjw::RpcCameraModel::GeodeticCoordinate ground{{110.01, 19.98, 1300.0}};
    xjw::CameraImageCoordinate image;
    ASSERT_TRUE(camera.groundToImageGeodetic(ground, &image));
    EXPECT_NEAR(image.sample, 678.325, 1.0e-9);
    EXPECT_NEAR(image.line, 299.475, 1.0e-9);

    xjw::RpcCameraModel::GeodeticCoordinate restored;
    ASSERT_TRUE(camera.imageToGroundAtHeight(image, ground[2], &restored));
    EXPECT_NEAR(restored[0], ground[0], 1.0e-12);
    EXPECT_NEAR(restored[1], ground[1], 1.0e-12);
}

TEST(RpcCameraModel, RejectsNonFiniteImageCorrection)
{
    xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    xjw::RpcCameraModel::ImageCorrection correction;
    correction.sampleOffsetPixels = std::numeric_limits<double>::quiet_NaN();
    std::string error;
    EXPECT_FALSE(camera.setImageCorrection(correction, &error));
    EXPECT_FALSE(error.empty());
}

TEST(RpcBiasAdjustment, EstimatesAffineCorrectionFromControlPoints)
{
    xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    xjw::RpcCameraModel::ImageCorrection expected;
    expected.sampleOffsetPixels = 1.5;
    expected.sampleSamplePixels = 2.0;
    expected.sampleLinePixels = -3.0;
    expected.lineOffsetPixels = -2.5;
    expected.lineSamplePixels = 4.0;
    expected.lineLinePixels = 0.75;

    const std::array<xjw::RpcCameraModel::GeodeticCoordinate, 5> grounds{{{110.00, 20.00, 900.0},
                                                                          {110.03, 20.00, 1100.0},
                                                                          {110.00, 20.04, 1300.0},
                                                                          {109.97, 19.98, 1500.0},
                                                                          {110.02, 19.96, 700.0}}};
    std::vector<xjw::RpcControlPointObservation> observations;
    for (const auto& ground : grounds)
    {
        xjw::CameraImageCoordinate raw;
        ASSERT_TRUE(camera.groundToImageGeodeticUncorrected(ground, &raw));
        const double normalized_sample =
            (raw.sample - camera.parameters().sampleOffset) / camera.parameters().sampleScale;
        const double normalized_line = (raw.line - camera.parameters().lineOffset) / camera.parameters().lineScale;
        xjw::CameraImageCoordinate observed;
        observed.sample = raw.sample + expected.sampleOffsetPixels + expected.sampleSamplePixels * normalized_sample +
                          expected.sampleLinePixels * normalized_line;
        observed.line = raw.line + expected.lineOffsetPixels + expected.lineSamplePixels * normalized_sample +
                        expected.lineLinePixels * normalized_line;
        observations.push_back({ground, observed, 1.0});
    }

    xjw::RpcBiasAdjustmentResult result;
    std::string error;
    ASSERT_TRUE(xjw::estimateRpcImageCorrection(camera, observations, &result, &error)) << error;
    EXPECT_NEAR(result.correction.sampleOffsetPixels, expected.sampleOffsetPixels, 1.0e-10);
    EXPECT_NEAR(result.correction.sampleSamplePixels, expected.sampleSamplePixels, 1.0e-10);
    EXPECT_NEAR(result.correction.sampleLinePixels, expected.sampleLinePixels, 1.0e-10);
    EXPECT_NEAR(result.correction.lineOffsetPixels, expected.lineOffsetPixels, 1.0e-10);
    EXPECT_NEAR(result.correction.lineSamplePixels, expected.lineSamplePixels, 1.0e-10);
    EXPECT_NEAR(result.correction.lineLinePixels, expected.lineLinePixels, 1.0e-10);
    EXPECT_LT(result.rmsAfterPixels, 1.0e-10);
    EXPECT_GT(result.rmsBeforePixels, 1.0);
}

TEST(RpcBiasAdjustment, RobustTranslationLimitsOneGrossOutlier)
{
    const xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    const std::array<xjw::RpcCameraModel::GeodeticCoordinate, 6> grounds{{{110.00, 20.00, 900.0},
                                                                          {110.03, 20.00, 1100.0},
                                                                          {110.00, 20.04, 1300.0},
                                                                          {109.97, 19.98, 1500.0},
                                                                          {110.02, 19.96, 700.0},
                                                                          {109.99, 20.02, 1200.0}}};
    std::vector<xjw::RpcControlPointObservation> observations;
    for (const auto& ground : grounds)
    {
        xjw::CameraImageCoordinate raw;
        ASSERT_TRUE(camera.groundToImageGeodeticUncorrected(ground, &raw));
        observations.push_back({ground, {raw.sample + 3.0, raw.line - 2.0}, 1.0});
    }
    observations.back().observedImage.sample += 50.0;
    observations.back().observedImage.line -= 40.0;

    xjw::RpcBiasAdjustmentOptions options;
    options.model = xjw::RpcImageCorrectionModel::Translation;
    options.huberThresholdPixels = 1.0;
    options.maximumIterations = 20;
    xjw::RpcBiasAdjustmentResult result;
    std::string error;
    ASSERT_TRUE(xjw::estimateRpcImageCorrection(camera, observations, &result, options, &error)) << error;
    EXPECT_NEAR(result.correction.sampleOffsetPixels, 3.0, 0.25);
    EXPECT_NEAR(result.correction.lineOffsetPixels, -2.0, 0.25);
    EXPECT_LT(result.iterations, options.maximumIterations + 1);
}

TEST(RpcCameraModel, CommonInterfaceUsesWgs84EcefMeters)
{
    const xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    const xjw::RpcCameraModel::GeodeticCoordinate geodetic{{110.01, 19.98, 1300.0}};
    xjw::RpcCameraModel::EcefCoordinate ecef;
    ASSERT_TRUE(xjw::RpcCameraModel::geodeticToEcef(geodetic, &ecef));

    const xjw::CameraModel& model = camera;
    EXPECT_EQ(model.modelType(), xjw::CameraModelType::RationalPolynomial);
    EXPECT_EQ(model.worldFrameName(), "EPSG:4978");
    xjw::CameraGroundProjection projection;
    ASSERT_TRUE(model.groundToImage(ecef, &projection));
    EXPECT_NEAR(projection.image.sample, 675.0, 1.0e-6);
    EXPECT_NEAR(projection.image.line, 300.0, 1.0e-6);
    EXPECT_TRUE(std::isfinite(projection.positiveDepthMeters));
}

TEST(RpcCameraModel, ApproximateRayPassesThroughHeightConstrainedSolutions)
{
    const xjw::RpcCameraModel camera = makeLinearRpc(0.25);
    const xjw::CameraImageCoordinate image{675.0, 300.0};
    xjw::CameraImagingRay ray;
    ASSERT_TRUE(camera.rayForPixel(image, &ray));
    const double norm = std::hypot(ray.direction[0], std::hypot(ray.direction[1], ray.direction[2]));
    EXPECT_NEAR(norm, 1.0, 1.0e-12);

    xjw::RpcCameraModel::EcefCoordinate ground;
    ASSERT_TRUE(camera.imageToEcefAtHeight(image, 1300.0, &ground));
    const std::array<double, 3> delta{
        {ground[0] - ray.originMeters[0], ground[1] - ray.originMeters[1], ground[2] - ray.originMeters[2]}};
    const std::array<double, 3> cross{{delta[1] * ray.direction[2] - delta[2] * ray.direction[1],
                                       delta[2] * ray.direction[0] - delta[0] * ray.direction[2],
                                       delta[0] * ray.direction[1] - delta[1] * ray.direction[0]}};
    // The RPC ray is the chord between two ellipsoidal-height solutions, not a
    // physical central-projection ray. Earth curvature keeps this synthetic
    // intermediate solution within a metre of that initialization chord.
    EXPECT_LT(std::hypot(cross[0], std::hypot(cross[1], cross[2])), 1.0);
}

TEST(RpcCameraIO, ParsesStandardGdalMetadata)
{
    const xjw::RpcCameraModel source = makeLinearRpc(-0.25);
    xjw::RpcMetadata metadata = metadataFor(source);
    metadata["ERR_BIAS"] = "2.5";
    metadata["err_rand"] = "1.25";

    xjw::RpcCameraModel restored;
    std::string error;
    ASSERT_TRUE(xjw::rpcCameraFromMetadata(metadata, &restored, &error)) << error;
    EXPECT_TRUE(restored.isValid());
    ASSERT_TRUE(restored.parameters().errorBiasMeters.has_value());
    ASSERT_TRUE(restored.parameters().errorRandomMeters.has_value());
    EXPECT_DOUBLE_EQ(*restored.parameters().errorBiasMeters, 2.5);
    EXPECT_DOUBLE_EQ(*restored.parameters().errorRandomMeters, 1.25);
    EXPECT_EQ(restored.parameters().sampleNumerator, source.parameters().sampleNumerator);
}

TEST(RpcCameraIO, RejectsIncompleteMetadata)
{
    xjw::RpcMetadata metadata = metadataFor(makeLinearRpc(0.25));
    metadata.erase("LINE_NUM_COEFF");
    xjw::RpcCameraModel camera;
    std::string error;
    EXPECT_FALSE(xjw::rpcCameraFromMetadata(metadata, &camera, &error));
    EXPECT_NE(error.find("LINE_NUM_COEFF"), std::string::npos);
}

TEST(RpcCameraIO, LoadsRpcMetadataFromGdalRaster)
{
    GDALAllRegister();
    constexpr const char* raster_path = "/vsimem/plascan_rpc_camera_test.tif";
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    ASSERT_NE(driver, nullptr);
    GDALDataset* dataset = driver->Create(raster_path, 16, 12, 1, GDT_Byte, nullptr);
    ASSERT_NE(dataset, nullptr);
    const xjw::RpcMetadata metadata = metadataFor(makeLinearRpc(0.25));
    for (const auto& [key, value] : metadata)
    {
        ASSERT_EQ(dataset->SetMetadataItem(key.c_str(), value.c_str(), "RPC"), CE_None);
    }
    GDALClose(dataset);

    xjw::RpcCameraModel camera;
    std::string error;
    EXPECT_TRUE(xjw::loadRpcCameraFromRaster(raster_path, &camera, &error)) << error;
    ASSERT_TRUE(camera.imageSize().has_value());
    EXPECT_EQ(camera.imageSize()->samples, 16);
    EXPECT_EQ(camera.imageSize()->lines, 12);
    EXPECT_DOUBLE_EQ(camera.parameters().longitudeOffset, 110.0);
    EXPECT_EQ(VSIUnlink(raster_path), 0);
}

TEST(RpcStereoIntersection, RecoversHeightFromOpposingRpcLookDirections)
{
    const xjw::RpcCameraModel first = makeLinearRpc(0.25);
    const xjw::RpcCameraModel second = makeLinearRpc(-0.25);
    const xjw::RpcCameraModel::GeodeticCoordinate expected{{110.01, 19.98, 1300.0}};
    xjw::CameraImageCoordinate first_observation;
    xjw::CameraImageCoordinate second_observation;
    ASSERT_TRUE(first.groundToImageGeodetic(expected, &first_observation));
    ASSERT_TRUE(second.groundToImageGeodetic(expected, &second_observation));

    xjw::RpcStereoIntersectionResult result;
    std::string error;
    ASSERT_TRUE(xjw::intersectRpcObservations(first, first_observation, second, second_observation, &result, &error))
        << error;
    EXPECT_NEAR(result.geodetic[0], expected[0], 1.0e-9);
    EXPECT_NEAR(result.geodetic[1], expected[1], 1.0e-9);
    EXPECT_NEAR(result.geodetic[2], expected[2], 0.02);
    EXPECT_LT(result.reprojectionRmsPixels, 1.0e-5);
}
