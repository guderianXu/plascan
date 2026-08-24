#include "ProjectCameraIO.h"

#include "FramePinholeCamera.h"
#include "RpcCameraModel.h"

#include <QJsonObject>

#include <gtest/gtest.h>

#include <memory>

namespace
{

using namespace xjw::common::project;

xjw::FramePinholeCamera makeCamera()
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsicsMillimeters(35.0, 35.5, 0.2, -0.1, 0.005);
    camera.setAxisDirections(-1, 1);
    camera.setDepthAxisFlipped(true);
    camera.setDistortion(0.01, -0.001, 0.0001, 0.0002, -0.0003);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{1.0, 2.0, 3.0}});
    return camera;
}

xjw::RpcCameraModel makeRpcCamera()
{
    xjw::RpcCameraModel::Parameters parameters;
    parameters.lineOffset = 3000.0;
    parameters.sampleOffset = 4000.0;
    parameters.latitudeOffset = 34.0;
    parameters.longitudeOffset = 108.0;
    parameters.heightOffset = 1200.0;
    parameters.lineScale = 3000.0;
    parameters.sampleScale = 4000.0;
    parameters.latitudeScale = 0.1;
    parameters.longitudeScale = 0.1;
    parameters.heightScale = 1000.0;
    parameters.lineNumerator[2] = 1.0;
    parameters.lineDenominator[0] = 1.0;
    parameters.sampleNumerator[1] = 1.0;
    parameters.sampleDenominator[0] = 1.0;
    parameters.errorBiasMeters = 3.5;

    xjw::RpcCameraModel camera;
    EXPECT_TRUE(camera.setParameters(parameters));
    xjw::RpcCameraModel::ImageCorrection correction;
    correction.sampleOffsetPixels = 1.25;
    correction.sampleLinePixels = -0.5;
    correction.lineOffsetPixels = -2.0;
    correction.lineSamplePixels = 0.75;
    EXPECT_TRUE(camera.setImageCorrection(correction));
    camera.setImageSize(xjw::CameraImageSize{8000, 6000});
    return camera;
}

TEST(ProjectCameraIOTest, RoundTripsCameraJson)
{
    const xjw::FramePinholeCamera source = makeCamera();
    const QJsonObject json = cameraToJson(source);

    xjw::FramePinholeCamera restored;
    ASSERT_TRUE(cameraFromJson(json, &restored));
    EXPECT_TRUE(restored.isValid());
    EXPECT_DOUBLE_EQ(restored.focalXMillimeters(), source.focalXMillimeters());
    EXPECT_DOUBLE_EQ(restored.focalYMillimeters(), source.focalYMillimeters());
    EXPECT_EQ(restored.uAxisSign(), source.uAxisSign());
    EXPECT_EQ(restored.depthAxisFlipped(), source.depthAxisFlipped());
    EXPECT_EQ(restored.cameraCenter(), source.cameraCenter());
}

TEST(ProjectCameraIOTest, ReadsCameraFromImageEntry)
{
    const QJsonObject image{
        {QStringLiteral("path"), QStringLiteral("image.tif")},
        {QStringLiteral("camera"), cameraToJson(makeCamera())}};

    xjw::FramePinholeCamera camera;
    EXPECT_TRUE(imageCameraFromEntry(image, &camera));
    EXPECT_TRUE(camera.isValid());
}

TEST(ProjectCameraIOTest, RoundTripsRpcCameraJsonAndFactory)
{
    const xjw::RpcCameraModel source = makeRpcCamera();
    const QJsonObject json = cameraToJson(source);
    EXPECT_EQ(json.value(QStringLiteral("model")).toString(), QStringLiteral("rpc"));

    xjw::RpcCameraModel restored;
    ASSERT_TRUE(cameraFromJson(json, &restored));
    ASSERT_TRUE(restored.imageSize().has_value());
    EXPECT_EQ(restored.imageSize()->samples, 8000);
    EXPECT_EQ(restored.parameters().lineNumerator, source.parameters().lineNumerator);
    ASSERT_TRUE(restored.parameters().errorBiasMeters.has_value());
    EXPECT_DOUBLE_EQ(*restored.parameters().errorBiasMeters, 3.5);
    EXPECT_DOUBLE_EQ(restored.imageCorrection().sampleOffsetPixels, 1.25);
    EXPECT_DOUBLE_EQ(restored.imageCorrection().sampleLinePixels, -0.5);
    EXPECT_DOUBLE_EQ(restored.imageCorrection().lineOffsetPixels, -2.0);
    EXPECT_DOUBLE_EQ(restored.imageCorrection().lineSamplePixels, 0.75);

    std::unique_ptr<xjw::CameraModel> generic = cameraModelFromJson(json);
    ASSERT_NE(generic, nullptr);
    EXPECT_EQ(generic->modelType(), xjw::CameraModelType::RationalPolynomial);
}

TEST(ProjectCameraIOTest, ReadsRpcCameraFromImageEntryPolymorphically)
{
    const QJsonObject image{
        {QStringLiteral("path"), QStringLiteral("image.tif")},
        {QStringLiteral("camera"), cameraToJson(makeRpcCamera())}};
    std::unique_ptr<xjw::CameraModel> camera = imageCameraModelFromEntry(image);
    ASSERT_NE(camera, nullptr);
    EXPECT_TRUE(camera->isValid());
    EXPECT_EQ(camera->modelType(), xjw::CameraModelType::RationalPolynomial);
}

TEST(ProjectCameraIOTest, ReportsInvalidTsaiPath)
{
    QJsonObject metadata;
    QString error;

    EXPECT_FALSE(parseTsaiCamera(QStringLiteral("missing-camera.tsai"), &metadata, &error));
    EXPECT_TRUE(metadata.isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("missing-camera.tsai")));
}

} // namespace
