#include "orb_binary/OrbBinaryAlgorithm.h"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <vector>

namespace xjw::image_matching
{
namespace
{

FeatureSet makeBinaryFeatures(const std::vector<std::vector<std::uint8_t>> &rows)
{
    FeatureSet features;
    features.imageWidth = 128;
    features.imageHeight = 128;
    features.sourceAlgorithm = kOrbBinaryAlgorithmId;
    features.computeBackend = "test";
    features.descriptors = cv::Mat::zeros(
        static_cast<int>(rows.size()), static_cast<int>(rows.front().size()), CV_8U);
    for (int row = 0; row < static_cast<int>(rows.size()); ++row)
    {
        features.keypoints.emplace_back(
            10.0f + static_cast<float>(row * 20), 20.0f, 5.0f);
        features.scores.push_back(1.0f);
        for (int column = 0; column < static_cast<int>(rows[static_cast<std::size_t>(row)].size()); ++column)
        {
            features.descriptors.at<std::uint8_t>(row, column) =
                rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)];
        }
    }
    return features;
}

TEST(OrbBinaryAlgorithmTest, ExtractsConsistentBinaryDescriptors)
{
    cv::Mat image = cv::Mat::zeros(256, 256, CV_8U);
    cv::rectangle(image, cv::Rect(32, 32, 72, 64), cv::Scalar(220), cv::FILLED);
    cv::circle(image, cv::Point(180, 78), 35, cv::Scalar(180), 5);
    cv::line(image, cv::Point(20, 210), cv::Point(230, 145), cv::Scalar(255), 6);

    ImageFeatureInput input;
    input.grayImage = image;
    input.originalWidth = image.cols;
    input.originalHeight = image.rows;
    ImageMatchingRuntimeConfig config;
    config.maxKeypoints = 200;
    OrbBinaryAlgorithm algorithm(config);

    const FeatureSet features = algorithm.extract(input);
    ASSERT_FALSE(features.empty());
    EXPECT_TRUE(features.isConsistent());
    EXPECT_EQ(features.descriptors.type(), CV_8U);
    EXPECT_EQ(features.sourceAlgorithm, kOrbBinaryAlgorithmId);
    EXPECT_LE(features.size(), 200);
}

TEST(OrbBinaryAlgorithmTest, AppliesMutualRatioGateToHammingMatches)
{
    const FeatureSet features0 = makeBinaryFeatures({
        {0x00, 0x00, 0x00, 0x00},
        {0xff, 0xff, 0xff, 0xff},
        {0x0f, 0x0f, 0x0f, 0x0f}});
    const FeatureSet features1 = makeBinaryFeatures({
        {0x00, 0x00, 0x00, 0x01},
        {0xff, 0xff, 0xff, 0xfe},
        {0xf0, 0xf0, 0xf0, 0xf0}});

    ImageMatchingRuntimeConfig config;
    config.siftMaximumRatio = 0.8f;
    OrbBinaryAlgorithm algorithm(config);
    const MatchResult result = algorithm.matchFeatures(features0, features1);

    ASSERT_EQ(result.numMatches, 2);
    EXPECT_EQ(result.matches0[0], 0);
    EXPECT_EQ(result.matches0[1], 1);
    EXPECT_EQ(result.matches0[2], -1);
    EXPECT_GT(result.matchingScores0[0], 0.9f);
}

} // namespace
} // namespace xjw::image_matching
