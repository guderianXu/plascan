#include "reconstruction/CameraIntrinsicPriorSanitizer.h"
#include <gtest/gtest.h>

#include <QJsonArray>

#include <memory>
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
            {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
        };
    }

    std::shared_ptr<const placamera::FramePinholeModel>
    makeModel(const std::string& image_id, double focal, double center_x = 0.0)
    {
        const placamera::FrameId frame("project-world");
        const placamera::FrameIntrinsics intrinsics{focal, focal, 320.0, 240.0, 1.0, 1, 1};
        const placamera::BrownConradyDistortion distortion{0.01, 0.0, 0.0, 0.0, 0.0};
        auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("definition-" + image_id),
                                                      intrinsics,
                                                      distortion,
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        const auto pose =
            placamera::Pose::create(frame, {center_x, 0.0, 2.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        auto model = placamera::FramePinholeModel::create(placamera::CameraInstanceId("instance-" + image_id),
                                                          placamera::ImageId(image_id),
                                                          std::move(definition),
                                                          placamera::ImageSize{640, 480},
                                                          pose);
        return std::make_shared<const placamera::FramePinholeModel>(std::move(model));
    }

} // namespace

TEST(CameraIntrinsicPriorSanitizerTest, NormalizesOnlyExtremeFocalOutlierInDominantCameraGroup)
{
    std::vector<placamera::ImageId> imageIds;
    xjw::aerial_triangulation::FramePinholeModelsByImageId cameras;
    for (int index = 0; index < 15; ++index)
    {
        const auto imageId = placamera::ImageId("image-" + std::to_string(index));
        imageIds.push_back(imageId);
        cameras.emplace(imageId, makeModel(std::string(imageId.value()), 1536.0 + (index % 3)));
    }
    const auto badImageId = placamera::ImageId("image-bad");
    imageIds.push_back(badImageId);
    const auto original_bad = makeModel(std::string(badImageId.value()), 352.0, 3.0);
    cameras.emplace(badImageId, original_bad);

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.inspectedCameraCount, 16);
    EXPECT_EQ(result.dominantGroupCount, 15);
    EXPECT_EQ(result.normalizedCameraCount, 1);
    ASSERT_EQ(result.normalizedImageIds.size(), 1U);
    EXPECT_EQ(result.normalizedImageIds.front(), badImageId);
    EXPECT_NEAR(cameras.at(badImageId)->pinholeDefinition().intrinsics().focalX, 1537.0, 1.0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.front())->pinholeDefinition().intrinsics().focalX, 1536.0);
    EXPECT_DOUBLE_EQ(original_bad->pinholeDefinition().intrinsics().focalX, 352.0);
    EXPECT_DOUBLE_EQ(cameras.at(badImageId)->pose().center[0], 3.0);
    EXPECT_DOUBLE_EQ(cameras.at(badImageId)->pinholeDefinition().distortion().radialK1, 0.01);
    EXPECT_EQ(cameras.at(badImageId)->imageSize().samples, 640);
}

TEST(CameraIntrinsicPriorSanitizerTest, KeepsModerateFocalDifferencesAndMixedCameraGroups)
{
    std::vector<placamera::ImageId> imageIds;
    xjw::aerial_triangulation::FramePinholeModelsByImageId cameras;
    const QList<double> focals{900.0, 920.0, 1100.0, 1120.0, 1700.0, 1720.0};
    for (int index = 0; index < focals.size(); ++index)
    {
        const auto imageId = placamera::ImageId("camera-" + std::to_string(index));
        imageIds.push_back(imageId);
        cameras.emplace(imageId, makeModel(std::string(imageId.value()), focals.at(index)));
    }

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.normalizedCameraCount, 0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.front())->pinholeDefinition().intrinsics().focalX, 900.0);
    EXPECT_DOUBLE_EQ(cameras.at(imageIds.back())->pinholeDefinition().intrinsics().focalX, 1720.0);
}

TEST(CameraIntrinsicPriorSanitizerTest, KeepsPlaCameraInstanceIdentityAfterFocalSanitization)
{
    std::vector<placamera::ImageId> imageIds;
    xjw::aerial_triangulation::FramePinholeModelsByImageId cameras;
    for (int index = 0; index < 15; ++index)
    {
        const auto imageId = placamera::ImageId("typed-image-" + std::to_string(index));
        imageIds.push_back(imageId);
        cameras.emplace(imageId, makeModel(std::string(imageId.value()), 1536.0 + (index % 3)));
    }
    const auto badImageId = placamera::ImageId("typed-image-bad");
    imageIds.push_back(badImageId);
    cameras.emplace(badImageId, makeModel(std::string(badImageId.value()), 352.0));

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imageIds, &cameras);

    EXPECT_EQ(result.normalizedCameraCount, 1);
    ASSERT_EQ(result.normalizedImageIds.size(), 1U);
    EXPECT_EQ(result.normalizedImageIds.front(), badImageId);
    EXPECT_NEAR(cameras.at(badImageId)->pinholeDefinition().intrinsics().focalX, 1537.0, 1.0);
    EXPECT_NEAR(cameras.at(badImageId)->pinholeDefinition().intrinsics().focalY, 1537.0, 1.0);
    EXPECT_EQ(cameras.at(badImageId)->imageId().value(), badImageId.value());
    EXPECT_EQ(cameras.at(badImageId)->instanceId().value(), "instance-typed-image-bad");
}

TEST(CameraIntrinsicPriorSanitizerTest, AcceptsOnlyExplicitlyTrustedIntrinsicSources)
{
    const QJsonObject imported{
        {QStringLiteral("metadata"), QJsonObject{{QStringLiteral("source_file"), QStringLiteral("onc_t.tsai")}}}};
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(imported));

    const QJsonObject manual{{QStringLiteral("source"), QStringLiteral("init_from_intrinsics")}};
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(manual));

    const QJsonObject exif{{QStringLiteral("source"), QStringLiteral("init_from_exif_or_default")},
                           {QStringLiteral("metadata"),
                            QJsonObject{{QStringLiteral("focal_source"), QStringLiteral("exif_focal_length")}}}};
    EXPECT_TRUE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(exif));

    const QJsonObject fallback{
        {QStringLiteral("source"), QStringLiteral("init_from_exif_or_default")},
        {QStringLiteral("metadata"), QJsonObject{{QStringLiteral("focal_source"), QStringLiteral("default_mm")}}}};
    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(fallback));

    const QJsonObject sfmEstimated{
        {QStringLiteral("metadata"),
         QJsonObject{{QStringLiteral("intrinsic_source"), QStringLiteral("sfm_estimated")}}}};
    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(sfmEstimated));

    EXPECT_FALSE(xjw::aerial_triangulation::isTrustedProjectCameraIntrinsic(QJsonObject{}));
}
