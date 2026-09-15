#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "DepthFrameResult.h"

using namespace xjw::mvs;

TEST(DepthFrameContract, DefaultFrameCannotEnterFusionOrConsistency)
{
    const DepthFrameResult frame;
    EXPECT_FALSE(frame.eligibleForFusion());
    EXPECT_FALSE(frame.eligibleForConsistencyCheck());
    EXPECT_FALSE(frame.eligibleAsConsistencySource());
}

TEST(DepthFrameContract, AdmissionRolesRemainDistinct)
{
    DepthFrameResult frame;
    frame.success = true;
    frame.qualityDecision.acceptance = DepthFrameAcceptance::Accepted;
    EXPECT_TRUE(frame.eligibleForFusion());
    EXPECT_TRUE(frame.eligibleAsConsistencySource());
    frame.qualityDecision.acceptance = DepthFrameAcceptance::ValidationOnly;
    EXPECT_FALSE(frame.eligibleForFusion());
    EXPECT_TRUE(frame.eligibleForConsistencyCheck());
    frame.qualityDecision.acceptance = DepthFrameAcceptance::Rejected;
    EXPECT_FALSE(frame.eligibleForConsistencyCheck());
    frame.qualityDecision.acceptance = DepthFrameAcceptance::Accepted;
    frame.success = false;
    EXPECT_FALSE(frame.eligibleForFusion());
}

TEST(DepthFrameContract, StreamingReleaseKeepsSupportDomainAndAuditMetadata)
{
    DepthFrameResult frame;
    frame.refViewIdx = 3;
    frame.success = true;
    frame.preparedRasterSize = cv::Size(640, 480);
    frame.geometrySourceViewIndices = {0, 2};
    frame.sourceViewShortfall = 1;
    frame.targetedGapRecoveredMaskExpected = true;
    frame.depthCompleteness.preConsistencyValidCount = 10;
    frame.depthCompleteness.postConsistencyValidCount = 8;
    frame.depthCompleteness.publishedPostConsistencyValidCount = 8;
    frame.depthMap = QSharedPointer<cv::Mat>::create(2, 2, CV_32FC1, cv::Scalar(10));
    frame.confidence = QSharedPointer<cv::Mat>::create(2, 2, CV_32FC1, cv::Scalar(1));
    frame.geometrySourceMask = QSharedPointer<cv::Mat>::create(2, 2, CV_16UC1, cv::Scalar(3));
    frame.supportRegionMask = QSharedPointer<cv::Mat>::create(2, 2, CV_8UC1, cv::Scalar(255));
    frame.validMask = QSharedPointer<cv::Mat>::create(2, 2, CV_8UC1, cv::Scalar(255));
    frame.intermediatePyramidLevels.emplace_back();
    frame.pyramidLevels.emplace_back();

    // Shared readers remain valid when the result drops its pixel references.
    const auto reader = frame.depthMap;
    frame.releaseStreamingPixelStorage();
    EXPECT_TRUE(frame.depthMap.isNull());
    EXPECT_TRUE(frame.confidence.isNull());
    EXPECT_TRUE(frame.geometrySourceMask.isNull());
    EXPECT_TRUE(frame.validMask.isNull());
    EXPECT_FALSE(frame.supportRegionMask.isNull());
    EXPECT_TRUE(frame.intermediatePyramidLevels.empty());
    EXPECT_EQ(frame.pyramidLevels.size(), 1);
    EXPECT_EQ(frame.geometrySourceViewIndices, (std::vector<int>{0, 2}));
    EXPECT_EQ(frame.preparedRasterSize, cv::Size(640, 480));
    EXPECT_EQ(frame.sourceViewShortfall, 1);
    EXPECT_TRUE(frame.targetedGapRecoveredMaskExpected);
    EXPECT_EQ(frame.depthCompleteness.publishedPostConsistencyValidCount, 8);
    EXPECT_FLOAT_EQ(reader->at<float>(0, 0), 10.0f);
    frame.releaseStreamingPixelStorage();
    frame.releasePixelStorage();
    EXPECT_TRUE(frame.supportRegionMask.isNull());
    EXPECT_EQ(frame.refViewIdx, 3);
    EXPECT_TRUE(frame.success);
}

TEST(DepthFrameContract, StreamingReleaseDropsEveryOptionalPixelProduct)
{
    DepthFrameResult frame;
    const std::vector<QSharedPointer<cv::Mat>*> pixel_products = {&frame.depthMap,
                                                                  &frame.confidence,
                                                                  &frame.photometricConfidence,
                                                                  &frame.geometricConfidence,
                                                                  &frame.normalMap,
                                                                  &frame.supportCount,
                                                                  &frame.photometricSourceMask,
                                                                  &frame.geometrySupportCount,
                                                                  &frame.geometrySourceMask,
                                                                  &frame.inverseDepthMean,
                                                                  &frame.inverseDepthRelativeSpread,
                                                                  &frame.adaptiveGeometrySupportWeight,
                                                                  &frame.adaptiveGeometryEffectiveViewCount,
                                                                  &frame.adaptiveGeometryConflictRatio,
                                                                  &frame.depthLayerReliabilityClass,
                                                                  &frame.crossViewRepairedMask,
                                                                  &frame.targetedGapRecoveredMask,
                                                                  &frame.residualReestimatedMask,
                                                                  &frame.learnedCandidateAcceptedMask,
                                                                  &frame.depthProvenance,
                                                                  &frame.missingReasonMap,
                                                                  &frame.validMask};
    for (auto* product : pixel_products)
    {
        *product = QSharedPointer<cv::Mat>::create(2, 2, CV_32FC1, cv::Scalar(1));
    }
    frame.geometryRerankMaps = QSharedPointer<DepthGeometryHypothesisRerankMaps>::create();
    frame.releaseStreamingPixelStorage();
    for (const auto* product : pixel_products)
    {
        EXPECT_TRUE(product->isNull());
    }
    EXPECT_TRUE(frame.geometryRerankMaps.isNull());
}
