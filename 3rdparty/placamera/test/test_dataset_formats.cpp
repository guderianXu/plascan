#include <placamera/dataset_formats.h>

#include <gtest/gtest.h>

#include <array>
#include <sstream>
#include <string>
#include <utility>

TEST(PlaCameraDatasetFormats, MiddleburyWorldToCameraPoseBecomesCameraToWorld)
{
    std::istringstream input("1\n"
                             "image.png 120 0 40 0 130 50 0 0 1 0 -1 0 1 0 0 0 0 1 2 -3 4\n");
    const auto parsed = placamera::readMiddleburyPar(input);
    ASSERT_TRUE(parsed) << parsed.message();
    const auto& cameras = parsed.value();
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0].imageName, "image.png");
    EXPECT_DOUBLE_EQ(cameras[0].calibration.intrinsicMatrix[0], 120.0);
    EXPECT_DOUBLE_EQ(cameras[0].center[0], 3.0);
    EXPECT_DOUBLE_EQ(cameras[0].center[1], 2.0);
    EXPECT_DOUBLE_EQ(cameras[0].center[2], -4.0);
    EXPECT_DOUBLE_EQ(cameras[0].cameraToWorldRotation[1], 1.0);
    EXPECT_TRUE(cameras[0].compatibility.isExactlyRepresentable());
}

TEST(PlaCameraDatasetFormats, EpflRetainsSourceOnlyRadialTerms)
{
    std::istringstream input("100 -2 50\n0 101 60\n0 0 1\n0.1 0 0\n"
                             "1 0 0\n0 1 0\n0 0 1\n3 4 5\n");
    const auto parsed = placamera::readEpflCamera(input);
    ASSERT_TRUE(parsed) << parsed.message();
    const auto& camera = parsed.value();
    EXPECT_FALSE(camera.compatibility.isExactlyRepresentable());
    ASSERT_EQ(camera.compatibility.sourceOnlyTerms.size(), 3u);
    EXPECT_EQ(camera.compatibility.sourceOnlyTerms[0].name, "epfl_radial_1");
    EXPECT_DOUBLE_EQ(camera.compatibility.sourceOnlyTerms[0].value, 0.1);
    EXPECT_DOUBLE_EQ(camera.calibration.intrinsicMatrix[1], -2.0);
    EXPECT_EQ(camera.center, (std::array<double, 3>{3.0, 4.0, 5.0}));
}

TEST(PlaCameraDatasetFormats, MetashapeUsesAdjustedSensorAndRetainsExtendedCalibration)
{
    const std::string xml = "<sensor id='3'><resolution width='1000' height='800'/>"
                            "<calibration><f>400</f></calibration>"
                            "<calibration class='adjusted'><f>500</f><cx>10</cx><b1>2</b1><b2>-0.5</b2>"
                            "<k4>0.1</k4><p1>0.002</p1><p3>0.0004</p3><p4>-0.00005</p4>"
                            "</calibration></sensor>"
                            "<camera sensor_id='3' label='images/frame.jpg'>"
                            "<transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform></camera>";
    const auto parsed = placamera::parseMetashapeDocument(xml);
    ASSERT_TRUE(parsed) << parsed.message();
    const auto& cameras = parsed.value();
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0].imageName, "images/frame.jpg");
    EXPECT_DOUBLE_EQ(cameras[0].calibration.intrinsicMatrix[0], 502.0);
    EXPECT_DOUBLE_EQ(cameras[0].calibration.intrinsicMatrix[1], -0.5);
    EXPECT_DOUBLE_EQ(cameras[0].calibration.intrinsicMatrix[2], 510.0);
    EXPECT_DOUBLE_EQ(cameras[0].center[2], 3.0);
    EXPECT_TRUE(cameras[0].compatibility.isExactlyRepresentable());
    const auto& distortion = cameras[0].calibration.distortion;
    EXPECT_EQ(distortion.tangentialConvention, placamera::BrownTangentialConvention::Metashape);
    EXPECT_DOUBLE_EQ(distortion.radialK4, 0.1);
    EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.002);
    EXPECT_DOUBLE_EQ(distortion.tangentialP3, 0.0004);
    EXPECT_DOUBLE_EQ(distortion.tangentialP4, -0.00005);
    ASSERT_TRUE(cameras[0].calibration.metashapeCalibration);
    const auto& exact = *cameras[0].calibration.metashapeCalibration;
    EXPECT_DOUBLE_EQ(exact.f, 500.0);
    EXPECT_DOUBLE_EQ(exact.cx, 510.0);
    EXPECT_DOUBLE_EQ(exact.cy, 400.0);
    EXPECT_DOUBLE_EQ(exact.b1, 2.0);
    EXPECT_DOUBLE_EQ(exact.b2, -0.5);
    ASSERT_TRUE(exact.principalPointDecomposition);
    EXPECT_EQ(*exact.principalPointDecomposition, (placamera::PrincipalPointDecomposition{500.0, 400.0, 10.0, 0.0}));

    const auto geometry = placamera::makeCentralCameraGeometry(
        cameras[0], placamera::CameraDefinitionId("metashape-camera"), placamera::FrameId("world"));
    ASSERT_TRUE(geometry) << geometry.message();
    const auto calibration = geometry.value().definition->calibration();
    EXPECT_DOUBLE_EQ(calibration.f, 500.0);
    EXPECT_DOUBLE_EQ(calibration.b1, 2.0);
    EXPECT_DOUBLE_EQ(calibration.b2, -0.5);
    EXPECT_DOUBLE_EQ(calibration.k4, 0.1);
    EXPECT_DOUBLE_EQ(calibration.p3, 0.0004);
    ASSERT_TRUE(calibration.principalPointDecomposition);
    EXPECT_EQ(calibration.principalPointDecomposition, exact.principalPointDecomposition);

    auto conflicting = cameras[0];
    conflicting.calibration.metashapeCalibration->cx += 1.0;
    const auto rejected = placamera::makeCentralCameraGeometry(
        conflicting, placamera::CameraDefinitionId("conflicting-camera"), placamera::FrameId("world"));
    EXPECT_FALSE(rejected);
    EXPECT_EQ(rejected.errorCode(), placamera::CameraErrorCode::InvalidModelState);
}

TEST(PlaCameraDatasetFormats, MetashapeReportsStructuredParseFailure)
{
    const std::string xml = "<sensor sensor_id='3'><resolution width='1000' height='800'/>"
                            "<calibration><f>500</f></calibration></sensor>"
                            "<camera sensor_id='3' label='frame.jpg'>"
                            "<transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform></camera>";
    const auto parsed = placamera::parseMetashapeDocument(xml);
    EXPECT_FALSE(parsed);
    EXPECT_EQ(parsed.errorCode(), placamera::CameraErrorCode::ParseFailure);
    EXPECT_EQ(parsed.error().source, "Metashape XML");
}

TEST(PlaCameraDatasetFormats, MetashapeRetainsEveryNativeProjectionFamily)
{
    const std::array<std::pair<const char*, placamera::FrameProjectionModel>, 6> projections{{
        {"frame", placamera::FrameProjectionModel::Perspective},
        {"fisheye", placamera::FrameProjectionModel::Fisheye},
        {"equidistant_fisheye", placamera::FrameProjectionModel::EquidistantFisheye},
        {"equisolid_fisheye", placamera::FrameProjectionModel::EquisolidFisheye},
        {"spherical", placamera::FrameProjectionModel::Spherical},
        {"cylindrical", placamera::FrameProjectionModel::Cylindrical},
    }};
    std::string xml;
    for (std::size_t index = 0; index < projections.size(); ++index)
    {
        xml += "<sensor id='" + std::to_string(index) + "' type='" + projections[index].first +
               "'><resolution width='1000' height='800'/><calibration><f>400</f></calibration></sensor>";
        xml += "<camera sensor_id='" + std::to_string(index) + "' label='frame-" + std::to_string(index) +
               ".jpg'><transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>";
    }

    const auto parsed = placamera::parseMetashapeDocument(xml);
    ASSERT_TRUE(parsed) << parsed.message();
    ASSERT_EQ(parsed.value().size(), projections.size());
    for (std::size_t index = 0; index < projections.size(); ++index)
    {
        EXPECT_EQ(parsed.value()[index].calibration.projectionModel, projections[index].second);
        const auto geometry =
            placamera::makeCentralCameraGeometry(parsed.value()[index],
                                                 placamera::CameraDefinitionId("projection-" + std::to_string(index)),
                                                 placamera::FrameId("world"));
        ASSERT_TRUE(geometry) << geometry.message();
        EXPECT_EQ(geometry.value().definition->projectionModel(), projections[index].second);
    }
}

TEST(PlaCameraDatasetFormats, MetashapeRejectsUnknownProjectionFamily)
{
    const std::string xml = "<sensor id='3' type='unknown'><resolution width='1000' height='800'/>"
                            "<calibration><f>400</f></calibration></sensor>"
                            "<camera sensor_id='3' label='frame.jpg'>"
                            "<transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>";
    const auto parsed = placamera::parseMetashapeDocument(xml);
    EXPECT_FALSE(parsed);
    EXPECT_EQ(parsed.errorCode(), placamera::CameraErrorCode::ParseFailure);
    EXPECT_NE(parsed.message().find("unsupported projection type"), std::string::npos);
}

TEST(PlaCameraDatasetFormats, MetashapeRetainsRollingShutterOptimizationMode)
{
    const std::string xml =
        "<sensor id='1' type='frame'><resolution width='1000' height='800'/>"
        "<property name='rolling_shutter_flags' value='3'/><calibration><f>400</f></calibration></sensor>"
        "<sensor id='2' type='frame'><resolution width='1000' height='800'/>"
        "<property name='rolling_shutter' value='true'/><calibration><f>400</f></calibration></sensor>"
        "<camera sensor_id='1' label='regularized.jpg'>"
        "<transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>"
        "<camera sensor_id='2' label='full.jpg'>"
        "<transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>";
    const auto parsed = placamera::parseMetashapeDocument(xml);
    ASSERT_TRUE(parsed) << parsed.message();
    ASSERT_EQ(parsed.value().size(), 2U);
    EXPECT_EQ(parsed.value()[0].acquisition.rollingShutterMode, placamera::RollingShutterMode::Regularized);
    EXPECT_EQ(parsed.value()[1].acquisition.rollingShutterMode, placamera::RollingShutterMode::Full);
}

TEST(PlaCameraDatasetFormats, BuildsTypedFrameGeometryOnlyForExactCalibration)
{
    std::istringstream input("1\n"
                             "image.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 2 -3 4\n");
    const auto parsed = placamera::readMiddleburyPar(input);
    ASSERT_TRUE(parsed) << parsed.message();
    ASSERT_EQ(parsed.value().size(), 1u);

    const auto geometry = placamera::makeCentralCameraGeometry(
        parsed.value().front(), placamera::CameraDefinitionId("dataset-camera"), placamera::FrameId("dataset-world"));
    ASSERT_TRUE(geometry) << geometry.message();
    EXPECT_DOUBLE_EQ(geometry.value().definition->intrinsics().focalX, 120.0);
    EXPECT_EQ(geometry.value().definition->groundFrame(), placamera::FrameId("dataset-world"));
    EXPECT_EQ(geometry.value().pose.frame, placamera::FrameId("dataset-world"));

    auto unsupported = parsed.value().front();
    unsupported.compatibility.unsupportedReason = "source model is fisheye";
    const auto unsupported_model = placamera::makeDatasetFramePinhole(
        unsupported, placamera::CameraDefinitionId("unsupported-camera"), placamera::FrameId("dataset-world"));
    EXPECT_FALSE(unsupported_model);
    EXPECT_EQ(unsupported_model.errorCode(), placamera::CameraErrorCode::UnsupportedModel);

    unsupported = parsed.value().front();
    unsupported.compatibility.sourceOnlyTerms.push_back({"k4", 0.01});
    EXPECT_FALSE(placamera::makeDatasetFramePinhole(
        unsupported, placamera::CameraDefinitionId("extra-term-camera"), placamera::FrameId("dataset-world")));

    unsupported = parsed.value().front();
    unsupported.calibration.intrinsicMatrix[1] = 0.01;
    const auto skew = placamera::makeDatasetFramePinhole(
        unsupported, placamera::CameraDefinitionId("skew-camera"), placamera::FrameId("dataset-world"));
    ASSERT_TRUE(skew) << skew.message();
    EXPECT_DOUBLE_EQ(skew.value().definition->intrinsics().skew, 0.01);
}
