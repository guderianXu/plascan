#include "geometry/MatchGeometryVerifier.h"

#include <gtest/gtest.h>

namespace xjw::image_matching
{
namespace
{

TEST(MatchGeometryVerifierTest, KeepsConsistentEpipolarMatchesAndStoresResiduals)
{
    FeatureSet features0;
    FeatureSet features1;
    features0.imageWidth = features1.imageWidth = 1000;
    features0.imageHeight = features1.imageHeight = 800;

    MatchResult matches;
    for (int index = 0; index < 30; ++index)
    {
        const float x = 50.0f + static_cast<float>((index * 73) % 700);
        const float y = 40.0f + static_cast<float>((index * 47) % 600);
        features0.keypoints.emplace_back(cv::Point2f(x, y), 3.0f);
        features1.keypoints.emplace_back(cv::Point2f(x + 15.0f + index * 0.2f,
                                                     y + (index % 2 == 0 ? 0.05f : -0.05f)),
                                         3.0f);
        matches.cvMatches.emplace_back(index, index, 0.05f);
    }
    for (int index = 0; index < 6; ++index)
    {
        const int featureIndex = 30 + index;
        features0.keypoints.emplace_back(cv::Point2f(100.0f + index * 80.0f,
                                                     100.0f + index * 50.0f),
                                         3.0f);
        features1.keypoints.emplace_back(cv::Point2f(700.0f - index * 30.0f,
                                                     700.0f - index * 70.0f),
                                         3.0f);
        matches.cvMatches.emplace_back(featureIndex, featureIndex, 0.5f);
    }
    matches.numMatches = static_cast<int>(matches.cvMatches.size());

    MatchGeometryOptions options;
    options.reprojectionThresholdPixels = 1.0;
    options.minimumInliers = 20;
    const MatchGeometryResult result =
        MatchGeometryVerifier::verify(matches, features0, features1, options);

    ASSERT_TRUE(result.modelEstimated);
    EXPECT_TRUE(result.passed);
    EXPECT_GE(result.inlierCount, 28);
    EXPECT_LT(result.inlierCount, 36);
    ASSERT_EQ(result.residualPixels.size(), matches.cvMatches.size());
    EXPECT_GE(result.residualPixels.front(), 0.0f);
    EXPECT_LT(result.residualPixels.front(), 1.0f);
}

TEST(MatchGeometryVerifierTest, RejectsPairBelowMinimumInliers)
{
    FeatureSet features0;
    FeatureSet features1;
    features0.imageWidth = features1.imageWidth = 640;
    features0.imageHeight = features1.imageHeight = 480;
    MatchResult matches;
    for (int index = 0; index < 8; ++index)
    {
        features0.keypoints.emplace_back(cv::Point2f(index * 20.0f + 10.0f,
                                                     index * 13.0f + 20.0f),
                                         3.0f);
        features1.keypoints.emplace_back(cv::Point2f(index * 17.0f + 200.0f,
                                                     400.0f - index * 31.0f),
                                         3.0f);
        matches.cvMatches.emplace_back(index, index, 0.5f);
    }
    matches.numMatches = 8;

    MatchGeometryOptions options;
    options.minimumInliers = 20;
    const MatchGeometryResult result =
        MatchGeometryVerifier::verify(matches, features0, features1, options);
    EXPECT_FALSE(result.passed);
}

} // namespace
} // namespace xjw::image_matching
