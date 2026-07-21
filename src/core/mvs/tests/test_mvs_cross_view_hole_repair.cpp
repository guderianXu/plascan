#include "DepthCrossViewHoleRepair.h"

#include <gtest/gtest.h>

namespace
{

xjw::Camera cameraAt(double x)
{
    xjw::Camera camera;
    camera.setIntrinsics(80.0, 80.0, 32.0, 32.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {x, 0.0, 0.0});
    return camera;
}

TEST(DepthCrossViewHoleRepairTest, RepairsHoleConfirmedByTwoDistinctSources)
{
    const xjw::Camera reference_camera = cameraAt(0.0);
    cv::Mat reference(64, 64, CV_32FC1, cv::Scalar(2.0f));
    reference(cv::Rect(29, 29, 7, 7)).setTo(0.0f);
    const cv::Mat support(64, 64, CV_8UC1, cv::Scalar(255));
    const cv::Mat source_depth(64, 64, CV_32FC1, cv::Scalar(2.0f));

    std::vector<cv::Mat> projected;
    projected.push_back(xjw::mvs::projectSourceDepthToReference(
        source_depth, cameraAt(-0.08), reference_camera, reference.size(), 1.0f));
    projected.push_back(xjw::mvs::projectSourceDepthToReference(
        source_depth, cameraAt(0.08), reference_camera, reference.size(), 1.0f));
    cv::Mat confidence(64, 64, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat votes(64, 64, CV_16UC1, cv::Scalar(0));
    cv::Mat repaired_mask;
    cv::Mat source_mask(64, 64, CV_16UC1, cv::Scalar(0));
    cv::Mat inverse_sum(64, 64, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat inverse_squared_sum(64, 64, CV_32FC1, cv::Scalar(0.0f));

    const xjw::mvs::CrossViewHoleRepairStats stats =
        xjw::mvs::repairDepthHolesFromProjectedSources(
            reference,
            support,
            projected,
            {},
            &confidence,
            &votes,
            &repaired_mask,
            &source_mask,
            &inverse_sum,
            &inverse_squared_sum);

    EXPECT_GT(stats.repairedPixelCount, 30U);
    EXPECT_NEAR(reference.at<float>(32, 32), 2.0f, 1.0e-4f);
    EXPECT_GE(confidence.at<float>(32, 32), 0.65f);
    EXPECT_EQ(votes.at<std::uint16_t>(32, 32), 1);
    EXPECT_EQ(repaired_mask.at<std::uint8_t>(32, 32), 255);
    EXPECT_EQ(source_mask.at<std::uint16_t>(32, 32), 0x0003);
    EXPECT_NEAR(inverse_sum.at<float>(32, 32), 0.5f, 1.0e-4f);
    EXPECT_NEAR(inverse_squared_sum.at<float>(32, 32), 0.25f, 1.0e-4f);
}

TEST(DepthCrossViewHoleRepairTest, RejectsDisagreeingDepthModes)
{
    cv::Mat reference(32, 32, CV_32FC1, cv::Scalar(0.0f));
    const cv::Mat support(32, 32, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(32, 32, CV_32FC1, cv::Scalar(2.0f)),
        cv::Mat(32, 32, CV_32FC1, cv::Scalar(5.0f))};

    const xjw::mvs::CrossViewHoleRepairStats stats =
        xjw::mvs::repairDepthHolesFromProjectedSources(
            reference, support, projected);

    EXPECT_EQ(stats.repairedPixelCount, 0U);
    EXPECT_EQ(stats.rejectedDepthSpreadCount, reference.total());
    EXPECT_EQ(cv::countNonZero(reference > 0.0f), 0);
}

TEST(DepthCrossViewHoleRepairTest, DoesNotRepairOutsideSupportMask)
{
    cv::Mat reference(24, 24, CV_32FC1, cv::Scalar(0.0f));
    const cv::Mat support(24, 24, CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Mat> projected = {
        cv::Mat(24, 24, CV_32FC1, cv::Scalar(2.0f)),
        cv::Mat(24, 24, CV_32FC1, cv::Scalar(2.0f))};

    const xjw::mvs::CrossViewHoleRepairStats stats =
        xjw::mvs::repairDepthHolesFromProjectedSources(
            reference, support, projected);

    EXPECT_EQ(stats.consideredHolePixelCount, 0U);
    EXPECT_EQ(stats.repairedPixelCount, 0U);
}

TEST(DepthCrossViewHoleRepairTest, GrowsStableTwoSourceComponentFromStrongCore)
{
    const xjw::Camera camera = cameraAt(0.0);
    cv::Mat reference(64, 64, CV_32FC1, cv::Scalar(2.0f));
    reference(cv::Rect(30, 30, 5, 5)).setTo(0.0f);
    const cv::Mat support(64, 64, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f)),
        cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f))};
    cv::Mat confidence(64, 64, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat votes(64, 64, CV_16UC1, cv::Scalar(3));
    votes(cv::Rect(30, 30, 5, 5)).setTo(0);
    cv::Mat repaired_mask;
    cv::Mat source_mask(64, 64, CV_16UC1, cv::Scalar(0x0007));
    source_mask(cv::Rect(30, 30, 5, 5)).setTo(0);
    cv::Mat inverse_sum(64, 64, CV_32FC1, cv::Scalar(1.5f));
    cv::Mat inverse_squared_sum(64, 64, CV_32FC1, cv::Scalar(0.75f));
    const cv::Mat guide(64, 64, CV_8UC1, cv::Scalar(128));
    xjw::mvs::CrossViewHoleRepairOptions options;
    options.minimumDistinctSourceCount = 3;
    options.enableTwoSourceGrowth = true;
    options.maximumGrowthDistancePixels = 3;
    options.maximumGrowthComponentArea = 32;

    const xjw::mvs::CrossViewHoleRepairStats stats =
        xjw::mvs::repairDepthHolesFromProjectedSources(
            reference,
            support,
            projected,
            options,
            &confidence,
            &votes,
            &repaired_mask,
            &source_mask,
            &inverse_sum,
            &inverse_squared_sum,
            &camera,
            &guide);

    EXPECT_EQ(stats.twoSourceCandidatePixelCount, 25U);
    EXPECT_EQ(stats.twoSourceGrownPixelCount, 25U);
    EXPECT_EQ(cv::countNonZero(reference(cv::Rect(30, 30, 5, 5)) > 0.0f), 25);
    EXPECT_EQ(source_mask.at<std::uint16_t>(32, 32), 0x0003);
    EXPECT_EQ(votes.at<std::uint16_t>(32, 32), 1);
    EXPECT_EQ(repaired_mask.at<std::uint8_t>(32, 32), 255);
}

TEST(DepthCrossViewHoleRepairTest, RejectsOversizedTwoSourceComponent)
{
    const xjw::Camera camera = cameraAt(0.0);
    cv::Mat reference(64, 64, CV_32FC1, cv::Scalar(2.0f));
    reference(cv::Rect(28, 28, 7, 7)).setTo(0.0f);
    const cv::Mat support(64, 64, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f)),
        cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f))};
    cv::Mat votes(64, 64, CV_16UC1, cv::Scalar(3));
    votes(cv::Rect(28, 28, 7, 7)).setTo(0);
    cv::Mat source_mask(64, 64, CV_16UC1, cv::Scalar(0x0007));
    source_mask(cv::Rect(28, 28, 7, 7)).setTo(0);
    cv::Mat inverse_sum(64, 64, CV_32FC1, cv::Scalar(1.5f));
    cv::Mat inverse_squared_sum(64, 64, CV_32FC1, cv::Scalar(0.75f));
    const cv::Mat guide(64, 64, CV_8UC1, cv::Scalar(128));
    xjw::mvs::CrossViewHoleRepairOptions options;
    options.minimumDistinctSourceCount = 3;
    options.enableTwoSourceGrowth = true;
    options.maximumGrowthComponentArea = 32;

    const xjw::mvs::CrossViewHoleRepairStats stats =
        xjw::mvs::repairDepthHolesFromProjectedSources(
            reference,
            support,
            projected,
            options,
            nullptr,
            &votes,
            nullptr,
            &source_mask,
            &inverse_sum,
            &inverse_squared_sum,
            &camera,
            &guide);

    EXPECT_EQ(stats.twoSourceCandidatePixelCount, 49U);
    EXPECT_EQ(stats.twoSourceGrownPixelCount, 0U);
    EXPECT_EQ(stats.growthRejectedComponentAreaCount, 1U);
    EXPECT_EQ(cv::countNonZero(reference(cv::Rect(28, 28, 7, 7)) > 0.0f), 0);
}

} // namespace
