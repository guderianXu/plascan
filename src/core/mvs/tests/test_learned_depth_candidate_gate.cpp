#include "LearnedDepthCandidateGate.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cstdint>

namespace
{

TEST(LearnedDepthCandidateGateTest, FillsOnlyIndependentGeometrySupportedPixel)
{
    cv::Mat depth(3, 3, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat confidence(3, 3, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat candidate(3, 3, CV_32FC1, cv::Scalar(10.0f));
    cv::Mat candidate_confidence(3, 3, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat support(3, 3, CV_16UC1, cv::Scalar(1));
    cv::Mat inverse_mean(3, 3, CV_32FC1, cv::Scalar(0.1f));
    cv::Mat spread(3, 3, CV_32FC1, cv::Scalar(0.004f));
    support.at<std::uint16_t>(1, 1) = 4;

    cv::Mat accepted;
    const auto stats = xjw::mvs::gateLearnedDepthCandidate(
        depth,
        confidence,
        candidate,
        candidate_confidence,
        support,
        inverse_mean,
        spread,
        &accepted);

    EXPECT_TRUE(stats.validInputs);
    EXPECT_EQ(stats.acceptedPixelCount, 1);
    EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 10.0f);
    EXPECT_EQ(cv::countNonZero(accepted), 1);
}

TEST(LearnedDepthCandidateGateTest, RejectsConfidentButGeometricallyWrongDepth)
{
    cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat confidence(2, 2, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat candidate(2, 2, CV_32FC1, cv::Scalar(12.0f));
    cv::Mat candidate_confidence(2, 2, CV_32FC1, cv::Scalar(0.99f));
    cv::Mat support(2, 2, CV_16UC1, cv::Scalar(4));
    cv::Mat inverse_mean(2, 2, CV_32FC1, cv::Scalar(0.1f));
    cv::Mat spread(2, 2, CV_32FC1, cv::Scalar(0.002f));

    const auto stats = xjw::mvs::gateLearnedDepthCandidate(
        depth,
        confidence,
        candidate,
        candidate_confidence,
        support,
        inverse_mean,
        spread,
        nullptr);

    EXPECT_EQ(stats.acceptedPixelCount, 0);
    EXPECT_EQ(stats.rejectedDepthDifferenceCount, 4);
    EXPECT_EQ(cv::countNonZero(depth > 0.0f), 0);
}

} // namespace
