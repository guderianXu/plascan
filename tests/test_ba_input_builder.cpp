#include <gtest/gtest.h>

#include "BaInputBuilder.h"
#include "Camera.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>

namespace
{

xjw::Camera makeCamera(double cx)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, 0.0, 0.0}});
    return camera;
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonArray centerArray;
    centerArray.append(center[0]);
    centerArray.append(center[1]);
    centerArray.append(center[2]);

    QJsonArray rotationArray;
    for (double value : rotation)
    {
        rotationArray.append(value);
    }

    QJsonObject object;
    object[QStringLiteral("intrinsics_unit")] = QStringLiteral("px");
    object[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    object[QStringLiteral("pitch")] = camera.pixelPitch();
    object[QStringLiteral("fu")] = intrinsics.focalX;
    object[QStringLiteral("fv")] = intrinsics.focalY;
    object[QStringLiteral("cu")] = intrinsics.principalX;
    object[QStringLiteral("cv")] = intrinsics.principalY;
    object[QStringLiteral("C")] = centerArray;
    object[QStringLiteral("R")] = rotationArray;
    return object;
}

QJsonObject imageEntry(const QString &path, const xjw::Camera &camera)
{
    QJsonObject object;
    object[QStringLiteral("path")] = path;
    object[QStringLiteral("camera")] = cameraToJson(camera);
    return object;
}

} // namespace

TEST(BaInputBuilderSurveyControl, BuildsControlPointTracksFromSurveyObservations)
{
    const QString image0 = QStringLiteral("E:/images/img_001.jpg");
    const QString image1 = QStringLiteral("E:/images/img_002.jpg");

    QJsonObject controlPoint;
    controlPoint[QStringLiteral("id")] = QStringLiteral("GCP001");
    controlPoint[QStringLiteral("enabled")] = true;
    controlPoint[QStringLiteral("x")] = 0.0;
    controlPoint[QStringLiteral("y")] = 0.0;
    controlPoint[QStringLiteral("z")] = 10.0;
    controlPoint[QStringLiteral("sigma_m")] = 0.05;
    controlPoint[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{controlPoint};

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        imageEntry(image0, makeCamera(0.0)),
        imageEntry(image1, makeCamera(1.0)),
    };
    meta[QStringLiteral("survey_control")] = surveyControl;

    xjw::core::project::BaInputBuildResult result;
    const xjw::core::project::BaInputBuildStatus status =
        xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result);

    EXPECT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_EQ(result.surveyControlTrackCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().observations.size(), 2u);
    ASSERT_EQ(result.tracks.front().controlPointConstraints.size(), 1u);
    EXPECT_NEAR(result.tracks.front().initialPoint[2], 10.0, 1e-12);
    EXPECT_NEAR(result.tracks.front().controlPointConstraints.front().point[2], 10.0, 1e-12);
    EXPECT_NEAR(result.tracks.front().controlPointConstraints.front().sigmaMeters, 0.05, 1e-12);
}

TEST(BaInputBuilderSurveyControl, BuildsScaleBarConstraintsBetweenSurveyControlTracks)
{
    const QString image0 = QStringLiteral("E:/images/img_001.jpg");
    const QString image1 = QStringLiteral("E:/images/img_002.jpg");

    QJsonObject controlPoint0;
    controlPoint0[QStringLiteral("id")] = QStringLiteral("GCP001");
    controlPoint0[QStringLiteral("enabled")] = true;
    controlPoint0[QStringLiteral("x")] = 0.0;
    controlPoint0[QStringLiteral("y")] = 0.0;
    controlPoint0[QStringLiteral("z")] = 10.0;
    controlPoint0[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject controlPoint1 = controlPoint0;
    controlPoint1[QStringLiteral("id")] = QStringLiteral("GCP002");
    controlPoint1[QStringLiteral("x")] = 10.0;
    controlPoint1[QStringLiteral("u")] = 0.0;
    controlPoint1[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 1512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 1512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject scaleBar;
    scaleBar[QStringLiteral("id")] = QStringLiteral("SB001");
    scaleBar[QStringLiteral("enabled")] = true;
    scaleBar[QStringLiteral("from_id")] = QStringLiteral("GCP001");
    scaleBar[QStringLiteral("to_id")] = QStringLiteral("GCP002");
    scaleBar[QStringLiteral("measured_m")] = 9.5;
    scaleBar[QStringLiteral("sigma_m")] = 0.02;

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{controlPoint0, controlPoint1};
    surveyControl[QStringLiteral("scale_bars")] = QJsonArray{scaleBar};

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        imageEntry(image0, makeCamera(0.0)),
        imageEntry(image1, makeCamera(1.0)),
    };
    meta[QStringLiteral("survey_control")] = surveyControl;

    xjw::core::project::BaInputBuildResult result;
    const xjw::core::project::BaInputBuildStatus status =
        xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result);

    EXPECT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_EQ(result.surveyControlTrackCount, 2);
    EXPECT_EQ(result.surveyScaleBarConstraintCount, 1);
    ASSERT_EQ(result.scaleBarConstraints.size(), 1u);
    EXPECT_EQ(result.scaleBarConstraints.front().trackIndexA, 0);
    EXPECT_EQ(result.scaleBarConstraints.front().trackIndexB, 1);
    EXPECT_NEAR(result.scaleBarConstraints.front().measuredDistanceMeters, 9.5, 1e-12);
    EXPECT_NEAR(result.scaleBarConstraints.front().sigmaMeters, 0.02, 1e-12);
}
