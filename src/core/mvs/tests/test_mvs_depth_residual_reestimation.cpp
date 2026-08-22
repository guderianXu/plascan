#include "DepthResidualReestimation.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <set>
#include <vector>

namespace
{

xjw::mvs::DepthResidualReestimationOptions testOptions()
{
    xjw::mvs::DepthResidualReestimationOptions options;
    options.minimumResidualPixelCount = 1;
    options.minimumResidualRatio = 0.0f;
    return options;
}

TEST(DepthResidualReestimationTest,
     LocalPatchMatchGroupsKeepThreeViewsAndOppositeSectorCoverage)
{
    const auto groups =
        xjw::mvs::buildDepthResidualPatchMatchSourceGroups({0, 1, 3, 4});
    ASSERT_EQ(groups.size(), 2U);
    EXPECT_EQ(groups[0].size(), 3U);
    EXPECT_EQ(groups[1].size(), 3U);
    EXPECT_NE(groups[0], groups[1]);
    for (const auto &group : groups)
    {
        EXPECT_EQ(std::set<int>(group.begin(), group.end()).size(), 3U);
    }
}

std::vector<cv::Mat> projectedDepths(
    const cv::Size &size,
    const std::vector<float> &depths)
{
    std::vector<cv::Mat> projected;
    projected.reserve(depths.size());
    for (const float depth : depths)
    {
        projected.emplace_back(size, CV_32FC1, cv::Scalar(depth));
    }
    return projected;
}

std::vector<xjw::mvs::ProjectedDepthEvidence> projectedEvidence(
    const cv::Size &size,
    const std::vector<float> &depths,
    const std::vector<int> &sectors)
{
    std::vector<xjw::mvs::ProjectedDepthEvidence> evidence;
    for (std::size_t index = 0; index < depths.size(); ++index)
    {
        xjw::mvs::ProjectedDepthEvidence item;
        item.depth = cv::Mat(size, CV_32FC1, cv::Scalar(depths[index]));
        item.confidence = cv::Mat(size, CV_32FC1, cv::Scalar(0.90f));
        item.reprojectionErrorPixels = cv::Mat(
            size, CV_32FC1, cv::Scalar(0.10f));
        item.baselineSector = sectors[index];
        evidence.push_back(std::move(item));
    }
    return evidence;
}

TEST(DepthResidualReestimationTest,
     PreflightCountsOnlyMissingPixelsInsideSupport)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat support(16, 16, CV_8UC1, cv::Scalar(0));
    support(cv::Rect(4, 4, 4, 4)).setTo(cv::Scalar(255));
    depth(cv::Rect(4, 4, 4, 4)).setTo(cv::Scalar(5.0f));
    depth.at<float>(6, 6) = 0.0f;
    auto options = testOptions();
    options.minimumResidualPixelCount = 2;

    const auto preflight =
        xjw::mvs::inspectDepthResidualReestimationNeed(
            depth, support, options);

    EXPECT_EQ(preflight.supportPixelCount, 16);
    EXPECT_EQ(preflight.requestedResidualPixelCount, 1);
    EXPECT_FLOAT_EQ(preflight.requestedResidualRatio, 1.0f / 16.0f);
    EXPECT_FALSE(preflight.shouldProjectSources);
    EXPECT_EQ(preflight.skippedReason,
              QStringLiteral("residual_below_threshold"));
}

TEST(DepthResidualReestimationTest,
     PreflightAllowsExistingValidResidualTarget)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));

    const auto preflight =
        xjw::mvs::inspectDepthResidualReestimationNeed(
            depth, support, testOptions());

    EXPECT_TRUE(preflight.shouldProjectSources);
    EXPECT_EQ(preflight.requestedResidualPixelCount, 1);
    EXPECT_TRUE(preflight.skippedReason.isEmpty());
}

TEST(DepthResidualReestimationTest,
     ExplicitMaskKeepsOnlyLargeConnectedWeakRegions)
{
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    cv::Mat requested(16, 16, CV_8UC1, cv::Scalar(0));
    requested.at<std::uint8_t>(2, 2) = 255;
    requested(cv::Rect(8, 8, 2, 2)).setTo(cv::Scalar(255));
    auto options = testOptions();
    options.minimumResidualPixelCount = 4;

    const auto preflight = xjw::mvs::inspectDepthReestimationMask(
        support, requested, options);

    ASSERT_TRUE(preflight.shouldProjectSources);
    EXPECT_EQ(preflight.requestedResidualPixelCount, 4);
    EXPECT_EQ(preflight.residualMask.at<std::uint8_t>(2, 2), 0);
    EXPECT_NE(preflight.residualMask.at<std::uint8_t>(8, 8), 0);
}

TEST(DepthResidualReestimationTest, BuildsLayerFromIndependentSectors)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.02f});

    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth, support, projected, {0, 1}, testOptions());

    ASSERT_TRUE(target.valid);
    EXPECT_EQ(target.layerCoveredPixelCount, 1);
    EXPECT_NE(target.residualMask.at<std::uint8_t>(8, 8), 0);
    EXPECT_NEAR(target.hintDepth.at<float>(8, 8), 5.01f, 0.01f);
    EXPECT_EQ(target.layerSourceCount.at<std::uint8_t>(8, 8), 2);
    EXPECT_EQ(target.layerSectorCount.at<std::uint8_t>(8, 8), 2);
}

TEST(DepthResidualReestimationTest, RejectsRepeatedSourcesFromOneSector)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.01f, 4.99f});

    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth, support, projected, {2, 2, 2}, testOptions());

    EXPECT_FALSE(target.valid);
    EXPECT_EQ(target.layerCoveredPixelCount, 0);
    EXPECT_EQ(target.insufficientSectorPixelCount, 1);
}

TEST(DepthResidualReestimationTest, SelectsDominantVisibleDepthLayer)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {4.98f, 5.0f, 5.02f, 8.0f, 8.02f});

    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth, support, projected, {0, 1, 2, 3, 4}, testOptions());

    ASSERT_TRUE(target.valid);
    EXPECT_NEAR(target.hintDepth.at<float>(8, 8), 5.0f, 0.02f);
    EXPECT_EQ(target.layerSourceCount.at<std::uint8_t>(8, 8), 3);
}

TEST(DepthResidualReestimationTest, RecoversOnlyDoubleHypothesisGeometryConsensus)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.02f});
    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth, support, projected, {0, 1}, testOptions());
    ASSERT_TRUE(target.valid);
    std::vector<cv::Mat> candidates = projectedDepths(
        depth.size(), {0.0f, 0.0f});
    candidates[0].at<float>(8, 8) = 5.0f;
    candidates[1].at<float>(8, 8) = 5.01f;
    std::vector<cv::Mat> confidences = projectedDepths(
        depth.size(), {0.8f, 0.75f});
    cv::Mat confidence;
    cv::Mat recovered;

    const auto stats = xjw::mvs::mergeDepthResidualReestimationCandidates(
        depth,
        confidence,
        candidates,
        confidences,
        target,
        projected,
        {0, 1},
        &recovered,
        testOptions());

    EXPECT_EQ(stats.consensusCandidatePixelCount, 1);
    EXPECT_EQ(stats.recoveredPixelCount, 1);
    EXPECT_NE(recovered.at<std::uint8_t>(8, 8), 0);
    EXPECT_NEAR(depth.at<float>(8, 8), 5.005f, 0.01f);
}

TEST(DepthResidualReestimationTest, RejectsCandidateInObservedFreeSpace)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(5.0f));
    depth.at<float>(8, 8) = 0.0f;
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.01f});
    auto options = testOptions();
    options.maximumCandidatePriorRelativeDifference = 0.20f;
    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth, support, projected, {0, 1}, options);
    ASSERT_TRUE(target.valid);
    std::vector<cv::Mat> candidates = projectedDepths(
        depth.size(), {0.0f, 0.0f});
    candidates[0].at<float>(8, 8) = 4.5f;
    candidates[1].at<float>(8, 8) = 4.51f;
    std::vector<cv::Mat> confidences = projectedDepths(
        depth.size(), {0.8f, 0.75f});
    cv::Mat confidence;

    const auto stats = xjw::mvs::mergeDepthResidualReestimationCandidates(
        depth,
        confidence,
        candidates,
        confidences,
        target,
        projected,
        {0, 1},
        nullptr,
        options);

    EXPECT_EQ(stats.rejectedFreeSpacePixelCount, 1);
    EXPECT_EQ(stats.recoveredPixelCount, 0);
}

TEST(DepthResidualReestimationTest,
     RejectedNativeLayerUsesCostGatedLocalPatchMatchCandidate)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(8.0f));
    cv::Mat confidence(16, 16, CV_32FC1, cv::Scalar(0.70f));
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    cv::Mat requested(16, 16, CV_8UC1, cv::Scalar(0));
    requested.at<std::uint8_t>(8, 8) = 255;
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.01f, 4.99f});
    const auto evidence = projectedEvidence(
        depth.size(), {5.0f, 5.01f, 4.99f}, {0, 1, 2});
    auto options = testOptions();
    options.minimumLayerSourceCount = 3;
    options.minimumGeometryConfirmationCount = 3;
    options.allowValidDepthReplacement = true;
    options.minimumReplacementCostAdvantage = 0.08f;
    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth,
        support,
        projected,
        {0, 1, 2},
        options,
        xjw::mvs::inspectDepthReestimationMask(
            support, requested, options));
    ASSERT_TRUE(target.valid);
    std::vector<cv::Mat> candidates = projectedDepths(
        depth.size(), {0.0f, 0.0f});
    candidates[0].at<float>(8, 8) = 5.0f;
    candidates[1].at<float>(8, 8) = 5.01f;
    std::vector<cv::Mat> candidate_confidences = projectedDepths(
        depth.size(), {0.85f, 0.80f});
    cv::Mat reliability(
        depth.size(), CV_8UC1,
        cv::Scalar(static_cast<std::uint8_t>(
            xjw::mvs::DepthLayerReliabilityClass::Reliable)));
    reliability.at<std::uint8_t>(8, 8) = static_cast<std::uint8_t>(
        xjw::mvs::DepthLayerReliabilityClass::RejectedLayer);
    cv::Mat source_mask(depth.size(), CV_16UC1, cv::Scalar(0));
    cv::Mat inverse_sum(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat inverse_squared_sum(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat source_count(depth.size(), CV_16UC1, cv::Scalar(0));
    xjw::mvs::DepthGeometryHypothesisRerankMaps rerank_maps;
    xjw::mvs::DepthResidualReestimationEvidenceOutputs evidence_outputs;
    evidence_outputs.geometrySourceMask = &source_mask;
    evidence_outputs.sourceInverseDepthSum = &inverse_sum;
    evidence_outputs.sourceInverseDepthSquaredSum = &inverse_squared_sum;
    evidence_outputs.confirmedSourceCount = &source_count;
    evidence_outputs.rerankMaps = &rerank_maps;

    const auto stats = xjw::mvs::mergeDepthResidualReestimationCandidates(
        depth,
        confidence,
        candidates,
        candidate_confidences,
        target,
        projected,
        {0, 1, 2},
        nullptr,
        options,
        &reliability,
        &evidence,
        &evidence_outputs);

    EXPECT_EQ(stats.replacedPixelCount, 1);
    EXPECT_EQ(stats.recoveredPixelCount, 1);
    EXPECT_NEAR(depth.at<float>(8, 8), 5.005f, 0.01f);
    EXPECT_LT(confidence.at<float>(8, 8), 0.85f);
    EXPECT_EQ(source_mask.at<std::uint16_t>(8, 8), 0x7);
    EXPECT_EQ(source_count.at<std::uint16_t>(8, 8), 3);
    EXPECT_NEAR(inverse_sum.at<float>(8, 8), 0.60f, 0.01f);
    EXPECT_EQ(
        rerank_maps.decisionAction.at<std::uint8_t>(8, 8),
        static_cast<std::uint8_t>(
            xjw::mvs::DepthGeometryHypothesisAction::SwitchLayer));
}

TEST(DepthResidualReestimationTest,
     AmbiguousNativeLayerCannotJumpToDistantLocalCandidate)
{
    cv::Mat depth(16, 16, CV_32FC1, cv::Scalar(8.0f));
    cv::Mat confidence(16, 16, CV_32FC1, cv::Scalar(0.70f));
    const cv::Mat support(16, 16, CV_8UC1, cv::Scalar(255));
    cv::Mat requested(16, 16, CV_8UC1, cv::Scalar(0));
    requested.at<std::uint8_t>(8, 8) = 255;
    const std::vector<cv::Mat> projected = projectedDepths(
        depth.size(), {5.0f, 5.01f, 4.99f});
    const auto evidence = projectedEvidence(
        depth.size(), {5.0f, 5.01f, 4.99f}, {0, 1, 2});
    auto options = testOptions();
    options.minimumLayerSourceCount = 3;
    options.minimumGeometryConfirmationCount = 3;
    options.allowValidDepthReplacement = true;
    const auto target = xjw::mvs::buildDepthResidualReestimationTarget(
        depth,
        support,
        projected,
        {0, 1, 2},
        options,
        xjw::mvs::inspectDepthReestimationMask(
            support, requested, options));
    ASSERT_TRUE(target.valid);
    std::vector<cv::Mat> candidates = projectedDepths(
        depth.size(), {0.0f, 0.0f});
    candidates[0].at<float>(8, 8) = 5.0f;
    candidates[1].at<float>(8, 8) = 5.01f;
    std::vector<cv::Mat> candidate_confidences = projectedDepths(
        depth.size(), {0.85f, 0.80f});
    cv::Mat reliability(
        depth.size(), CV_8UC1,
        cv::Scalar(static_cast<std::uint8_t>(
            xjw::mvs::DepthLayerReliabilityClass::Reliable)));
    reliability.at<std::uint8_t>(8, 8) = static_cast<std::uint8_t>(
        xjw::mvs::DepthLayerReliabilityClass::AmbiguousLowTexture);

    const auto stats = xjw::mvs::mergeDepthResidualReestimationCandidates(
        depth,
        confidence,
        candidates,
        candidate_confidences,
        target,
        projected,
        {0, 1, 2},
        nullptr,
        options,
        &reliability,
        &evidence);

    EXPECT_EQ(stats.replacedPixelCount, 0);
    EXPECT_EQ(stats.rejectedReplacementCorrectionPixelCount, 1);
    EXPECT_FLOAT_EQ(depth.at<float>(8, 8), 8.0f);
}

} // namespace
