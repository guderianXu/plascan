#include "reconstruction/CameraIntrinsicPriorSanitizer.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <gtest/gtest.h>

#include <QJsonArray>

#include <string>
#include <vector>

namespace
{

QJsonObject makeCamera(double focal)
{
    return QJsonObject{
        {QStringLiteral("model"), QStringLiteral("frame_pinhole")},
        {QStringLiteral("fu"), focal},
        {QStringLiteral("fv"), focal},
        {QStringLiteral("cu"), 320.0},
        {QStringLiteral("cv"), 240.0},
        {QStringLiteral("pitch"), 1.0},
        {QStringLiteral("intrinsics_unit"), QStringLiteral("mm")},
        {QStringLiteral("camera_center_unit"), QStringLiteral("m")},
        {QStringLiteral("pixel_convention"), QStringLiteral("center")},
        {QStringLiteral("k1"), 0.0},
        {QStringLiteral("k2"), 0.0},
        {QStringLiteral("k3"), 0.0},
        {QStringLiteral("p1"), 0.0},
        {QStringLiteral("p2"), 0.0},
        {QStringLiteral("u_direction"), 1.0},
        {QStringLiteral("v_direction"), 1.0},
        {QStringLiteral("depth_axis_flipped"), false},
        {QStringLiteral("world_frame"), QStringLiteral("project-world")},
        {QStringLiteral("C"), QJsonArray{0.0, 0.0, 0.0}},
        {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}},
    };
}

} // namespace

TEST(CameraIntrinsicPriorSanitizerTest, NormalizesOnlyExtremeFocalOutlierInDominantCameraGroup)
{
    std::vector<xjw::camera_core::ImageId> imageIds;
    xjw::aerial_triangulation::CameraIntrinsicsByImageId cameras;
    for (int index = 0; index < 15; ++index)
    {
        const auto imageId = xjw::camera_core::ImageId("image-" + std::to_string(index));
        imageIds.push_back(imageId);
        cameras.emplace(imageId, makeCamera(1536.0 + (index % 3)));
    }
    const auto badImageId = xjw::camera_core::ImageId("image-bad");
    imageIds.push_back(badImageId);
    cameras.emplace(badImageId, makeCamera(352.0));

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.inspectedCameraCount, 16);
    EXPECT_EQ(result.dominantGroupCount, 15);
    EXPECT_EQ(result.normalizedCameraCount, 1);
    ASSERT_EQ(result.normalizedImageIds.size(), 1U);
    EXPECT_EQ(result.normalizedImageIds.front(), badImageId);
    EXPECT_NEAR(cameras.at(badImageId).value(QStringLiteral("fu")).toDouble(),
                1537.0,
                1.0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.front()).value(QStringLiteral("fu")).toDouble(),
                     1536.0);
}

TEST(CameraIntrinsicPriorSanitizerTest, KeepsModerateFocalDifferencesAndMixedCameraGroups)
{
    std::vector<xjw::camera_core::ImageId> imageIds;
    xjw::aerial_triangulation::CameraIntrinsicsByImageId cameras;
    const QList<double> focals{900.0, 920.0, 1100.0, 1120.0, 1700.0, 1720.0};
    for (int index = 0; index < focals.size(); ++index)
    {
        const auto imageId = xjw::camera_core::ImageId("camera-" + std::to_string(index));
        imageIds.push_back(imageId);
        cameras.emplace(imageId, makeCamera(focals.at(index)));
    }

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.normalizedCameraCount, 0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.front()).value(QStringLiteral("fu")).toDouble(),
                     900.0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.back()).value(QStringLiteral("fu")).toDouble(),
                     1720.0);
}

TEST(CameraIntrinsicPriorSanitizerTest, NormalizesTypedNumericStatesWithoutGoingThroughProjectJson)
{
    std::vector<xjw::camera_core::ImageId> imageIds;
    xjw::aerial_triangulation::FramePinholeStatesByImageId cameras;
    for (int index = 0; index < 15; ++index)
    {
        const auto imageId = xjw::camera_core::ImageId("typed-image-" + std::to_string(index));
        imageIds.push_back(imageId);
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(1536.0 + (index % 3), 1536.0 + (index % 3), 320.0, 240.0);
        cameras.emplace(imageId, camera);
    }
    const auto badImageId = xjw::camera_core::ImageId("typed-image-bad");
    imageIds.push_back(badImageId);
    xjw::camera_models::frame_pinhole::FramePinholeNumericState badCamera;
    badCamera.setIntrinsics(352.0, 352.0, 320.0, 240.0);
    cameras.emplace(badImageId, badCamera);

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.normalizedCameraCount, 1);
    ASSERT_EQ(result.normalizedImageIds.size(), 1U);
    EXPECT_EQ(result.normalizedImageIds.front(), badImageId);
    EXPECT_NEAR(cameras.at(badImageId).focalX(), 1537.0, 1.0);
    EXPECT_NEAR(cameras.at(badImageId).focalY(), 1537.0, 1.0);
}

TEST(CameraIntrinsicPriorSanitizerTest, AcceptsOnlyExplicitlyTrustedIntrinsicSources)
{
    QJsonObject imported = makeCamera(9000.0);
    imported.insert(QStringLiteral("source_file"), QStringLiteral("onc_t.tsai"));
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(imported));

    QJsonObject manual = makeCamera(9000.0);
    manual.insert(QStringLiteral("source"), QStringLiteral("init_from_intrinsics"));
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(manual));

    QJsonObject exif = makeCamera(9000.0);
    exif.insert(QStringLiteral("source"), QStringLiteral("init_from_exif_or_default"));
    exif.insert(QStringLiteral("focal_source"), QStringLiteral("exif_focal_length"));
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(exif));

    QJsonObject fallback = makeCamera(1200.0);
    fallback.insert(QStringLiteral("source"), QStringLiteral("init_from_exif_or_default"));
    fallback.insert(QStringLiteral("focal_source"), QStringLiteral("default_mm"));
    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(fallback));

    QJsonObject sfmEstimated = makeCamera(430.0);
    sfmEstimated.insert(QStringLiteral("intrinsic_source"), QStringLiteral("sfm_estimated"));
    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(sfmEstimated));

    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(makeCamera(430.0)));
}
