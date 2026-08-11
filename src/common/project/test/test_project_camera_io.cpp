#include "ProjectCameraIO.h"

#include "FramePinholeCamera.h"

#include <QJsonObject>

#include <gtest/gtest.h>

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

TEST(ProjectCameraIOTest, ReportsInvalidTsaiPath)
{
    QJsonObject metadata;
    QString error;

    EXPECT_FALSE(parseTsaiCamera(QStringLiteral("missing-camera.tsai"), &metadata, &error));
    EXPECT_TRUE(metadata.isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("missing-camera.tsai")));
}

} // namespace
