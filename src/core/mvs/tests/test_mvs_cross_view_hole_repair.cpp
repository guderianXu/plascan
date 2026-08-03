#include "DepthCrossViewHoleRepair.h"
#include "DepthAnchoredHoleInterpolator.h"

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

TEST(DepthCrossViewHoleRepairTest, RefinesNativeDepthTowardStableProjectedLayer)
{
    cv::Mat reference(1, 1, CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat support(1, 1, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.01f)),
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.02f)),
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.01f))};
    cv::Mat confidence(1, 1, CV_32FC1, cv::Scalar(0.9f));
    cv::Mat consistent(1, 1, CV_16UC1, cv::Scalar(3));
    cv::Mat contradicted(1, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat selected_mask;

    const auto stats = xjw::mvs::selectDominantProjectedDepthLayer(
        reference,
        support,
        projected,
        consistent,
        contradicted,
        {},
        &confidence,
        &selected_mask);

    EXPECT_EQ(stats.stableLayerPixelCount, 1U);
    EXPECT_EQ(stats.refinedNativePixelCount, 1U);
    EXPECT_GT(reference.at<float>(0, 0), 2.0f);
    EXPECT_LT(reference.at<float>(0, 0), 2.01f);
    EXPECT_EQ(selected_mask.at<std::uint8_t>(0, 0), 0);
    EXPECT_FLOAT_EQ(confidence.at<float>(0, 0), 0.9f);
}

TEST(DepthCrossViewHoleRepairTest, SwitchesContradictedNativeDepthToDominantLayer)
{
    cv::Mat reference(1, 1, CV_32FC1, cv::Scalar(4.0f));
    const cv::Mat support(1, 1, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.0f)),
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.01f)),
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(2.02f))};
    cv::Mat confidence(1, 1, CV_32FC1, cv::Scalar(0.4f));
    cv::Mat consistent(1, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat contradicted(1, 1, CV_16UC1, cv::Scalar(3));
    cv::Mat selected_mask;
    cv::Mat source_mask(1, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat inverse_sum(1, 1, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat inverse_squared_sum(1, 1, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat selected_votes(1, 1, CV_16UC1, cv::Scalar(0));

    const auto stats = xjw::mvs::selectDominantProjectedDepthLayer(
        reference,
        support,
        projected,
        consistent,
        contradicted,
        {},
        &confidence,
        &selected_mask,
        &source_mask,
        &inverse_sum,
        &inverse_squared_sum,
        &selected_votes);

    EXPECT_EQ(stats.switchedNativePixelCount, 1U);
    EXPECT_NEAR(reference.at<float>(0, 0), 2.01f, 1.0e-6f);
    EXPECT_EQ(selected_mask.at<std::uint8_t>(0, 0), 255);
    EXPECT_EQ(source_mask.at<std::uint16_t>(0, 0), 0x0007);
    EXPECT_EQ(selected_votes.at<std::uint16_t>(0, 0), 3);
    EXPECT_GT(inverse_sum.at<float>(0, 0), 1.0f);
    EXPECT_GT(inverse_squared_sum.at<float>(0, 0), 0.0f);
}

TEST(DepthCrossViewHoleRepairTest, TransfersStableObservedLayerIntoMissingPixel)
{
    cv::Mat reference(1, 2, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat support(1, 2, CV_8UC1, cv::Scalar(255));
    support.at<std::uint8_t>(0, 1) = 0;
    const std::vector<cv::Mat> projected = {
        cv::Mat(1, 2, CV_32FC1, cv::Scalar(3.0f)),
        cv::Mat(1, 2, CV_32FC1, cv::Scalar(3.01f))};
    cv::Mat confidence(1, 2, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat consistent(1, 2, CV_16UC1, cv::Scalar(2));
    cv::Mat contradicted(1, 2, CV_16UC1, cv::Scalar(0));
    cv::Mat selected_mask;

    const auto stats = xjw::mvs::selectDominantProjectedDepthLayer(
        reference,
        support,
        projected,
        consistent,
        contradicted,
        {},
        &confidence,
        &selected_mask);

    EXPECT_EQ(stats.transferredMissingPixelCount, 1U);
    EXPECT_NEAR(reference.at<float>(0, 0), 3.0f, 1.0e-6f);
    EXPECT_EQ(selected_mask.at<std::uint8_t>(0, 0), 255);
    EXPECT_GT(confidence.at<float>(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(reference.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(0, 1), 0.0f);
}

TEST(DepthCrossViewHoleRepairTest, KeepsAmbiguousNativeDepthButReducesConfidence)
{
    cv::Mat reference(1, 1, CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat support(1, 1, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(3.0f)),
        cv::Mat(1, 1, CV_32FC1, cv::Scalar(5.0f))};
    cv::Mat confidence(1, 1, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat consistent(1, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat contradicted(1, 1, CV_16UC1, cv::Scalar(2));

    const auto stats = xjw::mvs::selectDominantProjectedDepthLayer(
        reference,
        support,
        projected,
        consistent,
        contradicted,
        {},
        &confidence);

    EXPECT_EQ(stats.ambiguousNativePixelCount, 1U);
    EXPECT_FLOAT_EQ(reference.at<float>(0, 0), 2.0f);
    EXPECT_NEAR(confidence.at<float>(0, 0), 0.36f, 1.0e-6f);
}

TEST(DepthCrossViewHoleRepairTest,
     UsesValidNativeBoundaryAsInterpolationAnchor)
{
    cv::Mat reference(48, 48, CV_32FC1, cv::Scalar(2.0f));
    const cv::Rect hole(16, 16, 16, 16);
    reference(hole).setTo(0.0f);
    const cv::Mat support(48, 48, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = {
        cv::Mat(48, 48, CV_32FC1, cv::Scalar(0.0f)),
        cv::Mat(48, 48, CV_32FC1, cv::Scalar(0.0f))};
    cv::Mat confidence(48, 48, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat votes(48, 48, CV_16UC1, cv::Scalar(1));
    votes(hole).setTo(0);
    cv::Mat repaired_mask;
    xjw::mvs::CrossViewHoleRepairOptions options;
    options.includeValidNativeInterpolationAnchors = true;
    options.anchoredInterpolation.enabled = true;

    const auto stats = xjw::mvs::repairDepthHolesFromProjectedSources(
        reference,
        support,
        projected,
        options,
        &confidence,
        &votes,
        &repaired_mask);

    EXPECT_GT(stats.anchoredInterpolation.anchorPixelCount, 0U);
    EXPECT_EQ(stats.anchoredInterpolation.acceptedComponentCount, 1U);
    EXPECT_EQ(cv::countNonZero(reference(hole) <= 0.0f), 0);
    EXPECT_NEAR(reference.at<float>(24, 24), 2.0f, 1.0e-3f);
    EXPECT_EQ(repaired_mask.at<std::uint8_t>(24, 24), 255);
}

TEST(DepthAnchoredHoleInterpolatorTest, FillsInternalComponentBetweenStrongAnchors)
{
    cv::Mat depth(48, 48, CV_32FC1, cv::Scalar(2.0f));
    const cv::Rect hole(16, 16, 16, 16);
    depth(hole).setTo(0.0f);
    cv::Mat anchors(48, 48, CV_8UC1, cv::Scalar(0));
    for (const cv::Point &point : {
             cv::Point(18, 18), cv::Point(24, 18), cv::Point(29, 18),
             cv::Point(18, 24), cv::Point(29, 24),
             cv::Point(18, 29), cv::Point(24, 29), cv::Point(29, 29)})
    {
        depth.at<float>(point.y, point.x) = 2.0f;
        anchors.at<std::uint8_t>(point.y, point.x) = 255;
    }
    const cv::Mat support(48, 48, CV_8UC1, cv::Scalar(255));
    cv::Mat confidence(48, 48, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat repaired(48, 48, CV_8UC1, cv::Scalar(0));
    const cv::Mat guide(48, 48, CV_8UC1, cv::Scalar(128));
    xjw::mvs::DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;

    const auto stats = xjw::mvs::interpolateAnchoredInternalDepthHoles(
        depth, support, anchors, &guide, options, &confidence, &repaired);

    EXPECT_GT(stats.acceptedComponentCount, 0U);
    EXPECT_EQ(cv::countNonZero(depth(hole) <= 0.0f), 0);
    EXPECT_NEAR(depth.at<float>(24, 24), 2.0f, 1.0e-3f);
    EXPECT_NEAR(confidence.at<float>(24, 24), 0.45f, 1.0e-4f);
    EXPECT_EQ(repaired.at<std::uint8_t>(24, 24), 255);
}

TEST(DepthAnchoredHoleInterpolatorTest, PreservesOpeningInSupportMask)
{
    cv::Mat depth(40, 40, CV_32FC1, cv::Scalar(2.0f));
    const cv::Rect opening(14, 14, 12, 12);
    depth(opening).setTo(0.0f);
    cv::Mat support(40, 40, CV_8UC1, cv::Scalar(255));
    support(opening).setTo(0);
    cv::Mat anchors(40, 40, CV_8UC1, cv::Scalar(255));
    xjw::mvs::DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;

    const auto stats = xjw::mvs::interpolateAnchoredInternalDepthHoles(
        depth, support, anchors, nullptr, options);

    EXPECT_EQ(stats.interpolatedPixelCount, 0U);
    EXPECT_EQ(cv::countNonZero(depth(opening) > 0.0f), 0);
}

TEST(DepthAnchoredHoleInterpolatorTest,
     RecoversProtectedInteriorOfSilhouetteConnectedHole)
{
    cv::Mat depth(72, 72, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat support(72, 72, CV_8UC1, cv::Scalar(0));
    support(cv::Rect(6, 6, 60, 60)).setTo(255);
    depth.setTo(2.0f, support);

    const cv::Rect interior_hole(18, 20, 30, 28);
    depth(interior_hole).setTo(0.0f);
    depth(cv::Rect(6, 32, 12, 2)).setTo(0.0f);
    cv::Mat anchors = depth > 0.0f;
    cv::Mat repaired(72, 72, CV_8UC1, cv::Scalar(0));

    xjw::mvs::DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;
    options.maximumComponentArea = 2000;
    options.maximumComponentAreaRatio = 0.5f;
    options.allowSilhouetteConnectedInterior = true;
    options.silhouetteProtectionRadiusPixels = 3;

    const auto stats = xjw::mvs::interpolateAnchoredInternalDepthHoles(
        depth, support, anchors, nullptr, options, nullptr, &repaired);

    EXPECT_EQ(stats.protectedSilhouetteComponentCount, 1U);
    EXPECT_GT(stats.protectedSilhouettePixelCount, 0U);
    EXPECT_EQ(stats.acceptedComponentCount, 1U);
    EXPECT_GT(stats.interpolatedPixelCount, 800U);
    EXPECT_NEAR(depth.at<float>(34, 32), 2.0f, 1.0e-3f);
    EXPECT_EQ(depth.at<float>(32, 6), 0.0f);
    EXPECT_EQ(repaired.at<std::uint8_t>(34, 32), 255);
}

TEST(DepthAnchoredHoleInterpolatorTest,
     RejectsSilhouetteConnectedHoleWhenProtectionIsDisabled)
{
    cv::Mat depth(48, 48, CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat support(48, 48, CV_8UC1, cv::Scalar(255));
    depth(cv::Rect(0, 16, 24, 16)).setTo(0.0f);
    cv::Mat anchors = depth > 0.0f;
    xjw::mvs::DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;
    options.maximumComponentArea = 1000;
    options.maximumComponentAreaRatio = 0.5f;

    const auto stats = xjw::mvs::interpolateAnchoredInternalDepthHoles(
        depth, support, anchors, nullptr, options);

    EXPECT_EQ(stats.acceptedComponentCount, 0U);
    EXPECT_EQ(stats.rejectedSilhouetteComponentCount, 1U);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(0, 16, 24, 16)) > 0.0f), 0);
}

TEST(DepthAnchoredHoleInterpolatorTest, RejectsDepthDiscontinuity)
{
    cv::Mat depth(48, 48, CV_32FC1, cv::Scalar(2.0f));
    depth.colRange(24, 48).setTo(4.0f);
    const cv::Rect hole(20, 16, 8, 16);
    depth(hole).setTo(0.0f);
    cv::Mat anchors(48, 48, CV_8UC1, cv::Scalar(0));
    for (const cv::Point &point : {
             cv::Point(21, 18), cv::Point(26, 18), cv::Point(21, 24),
             cv::Point(26, 24), cv::Point(21, 29), cv::Point(26, 29)})
    {
        depth.at<float>(point.y, point.x) = point.x < 24 ? 2.0f : 4.0f;
        anchors.at<std::uint8_t>(point.y, point.x) = 255;
    }
    const cv::Mat support(48, 48, CV_8UC1, cv::Scalar(255));
    xjw::mvs::DepthAnchoredHoleInterpolationOptions options;
    options.enabled = true;

    const auto stats = xjw::mvs::interpolateAnchoredInternalDepthHoles(
        depth, support, anchors, nullptr, options);

    EXPECT_GT(stats.rejectedBoundarySpreadComponentCount, 0U);
    EXPECT_GT(cv::countNonZero(depth(hole) <= 0.0f), 0);
}

} // namespace
