#include <gtest/gtest.h>
#include "DisparityValidator.h"

#include <cmath>
#include <limits>

using namespace xjw::dense_match;

TEST(DisparityValidatorTest, LRCheck_Consistent_Passes)
{
    cv::Mat dispLR(4, 20, CV_32FC1, cv::Scalar(3.0f));
    cv::Mat dispRL(4, 20, CV_32FC1, cv::Scalar(-3.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 0.25f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < valid.rows; ++y)
    {
        for (int x = 0; x < valid.cols; ++x)
        {
            EXPECT_EQ(valid.at<uchar>(y, x), x >= 3 ? 1 : 0);
        }
    }
}

TEST(DisparityValidatorTest, LRCheck_Inconsistent_Fails)
{
    cv::Mat dispLR(5, 12, CV_32FC1, cv::Scalar(3.0f));
    cv::Mat dispRL(5, 12, CV_32FC1, cv::Scalar(3.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 1.0f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < valid.rows; ++y)
        for (int x = 0; x < valid.cols; ++x)
            EXPECT_EQ(valid.at<uchar>(y, x), 0);
}

TEST(DisparityValidatorTest, LRCheck_RequiresBothMatcherMasks)
{
    cv::Mat dispLR(1, 8, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat dispRL(1, 8, CV_32FC1, cv::Scalar(-2.0f));
    cv::Mat validLR(1, 8, CV_8UC1, cv::Scalar(1));
    cv::Mat validRL(1, 8, CV_8UC1, cv::Scalar(1));
    validLR.at<uchar>(0, 5) = 0;
    validRL.at<uchar>(0, 2) = 0; // left x=4 projects here

    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 0.25f;
    DisparityValidator validator(cfg);
    const cv::Mat valid = validator.checkLRConsistency(
        dispLR,
        dispRL,
        validLR,
        validRL);

    EXPECT_EQ(valid.at<uchar>(0, 4), 0);
    EXPECT_EQ(valid.at<uchar>(0, 5), 0);
    EXPECT_EQ(valid.at<uchar>(0, 6), 1);
}

TEST(DisparityValidatorTest, MedianFilter_SmoothesOutlier)
{
    cv::Mat disp(5, 5, CV_32FC1, cv::Scalar(5.0f));
    disp.at<float>(2, 2) = 100.0f;
    DenseMatchConfig cfg;
    cfg.medianFilterSize = 3;
    DisparityValidator validator(cfg);
    cv::Mat filtered = validator.medianFilter(disp, 3);
    EXPECT_NEAR(filtered.at<float>(2, 2), 5.0, 95.0);
}

TEST(DisparityValidatorTest, Validate_OutputTypes)
{
    cv::Mat disp(16, 16, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat conf(16, 16, CV_32FC1, cv::Scalar(0.8f));
    DenseMatchConfig cfg;
    DisparityValidator validator(cfg);
    auto result = validator.validate(disp, conf);
    EXPECT_EQ(result.disparity.type(), CV_32FC1);
    EXPECT_EQ(result.confidence.type(), CV_32FC1);
    EXPECT_EQ(result.validMask.type(), CV_8UC1);
    EXPECT_EQ(result.validMask.size(), disp.size());
}

TEST(DisparityValidatorTest, Validate_PreservesRawMatcherValidity)
{
    cv::Mat disparity(1, 3, CV_32FC1, cv::Scalar(4.0f));
    cv::Mat confidence(1, 3, CV_32FC1, cv::Scalar(0.75f));
    cv::Mat rawValid(1, 3, CV_8UC1, cv::Scalar(1));
    rawValid.at<uchar>(0, 1) = 0;
    DenseMatchConfig config;
    config.medianFilterSize = 0;
    DisparityValidator validator(config);

    const DisparityResult result = validator.validate(disparity, confidence, rawValid);

    EXPECT_EQ(result.validMask.at<uchar>(0, 0), 1);
    EXPECT_EQ(result.validMask.at<uchar>(0, 1), 0);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(result.confidence.at<float>(0, 1), 0.0f);
}

TEST(DisparityValidatorTest, Validate_MedianIgnoresInvalidZeroNeighbors)
{
    cv::Mat disparity(3, 3, CV_32FC1, cv::Scalar(0.0f));
    disparity.at<float>(1, 1) = -4.0f;
    const cv::Mat confidence(3, 3, CV_32FC1, cv::Scalar(0.75f));
    cv::Mat rawValid(3, 3, CV_8UC1, cv::Scalar(0));
    rawValid.at<uchar>(1, 1) = 1;

    DenseMatchConfig config;
    config.medianFilterSize = 3;
    DisparityValidator validator(config);
    const DisparityResult result = validator.validate(disparity, confidence, rawValid);

    EXPECT_EQ(result.validMask.at<uchar>(1, 1), 1);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(1, 1), -4.0f);
    EXPECT_EQ(cv::countNonZero(result.validMask), 1);
}

TEST(DisparityValidatorTest, Validate_MedianTreatsZeroAndNegativeDisparitiesAsValidSamples)
{
    cv::Mat disparity(1, 3, CV_32FC1);
    disparity.at<float>(0, 0) = -4.0f;
    disparity.at<float>(0, 1) = 0.0f;
    disparity.at<float>(0, 2) = -2.0f;
    const cv::Mat confidence(1, 3, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat rawValid(1, 3, CV_8UC1, cv::Scalar(1));

    DenseMatchConfig config;
    config.medianFilterSize = 3;
    DisparityValidator validator(config);
    const DisparityResult result = validator.validate(disparity, confidence, rawValid);

    EXPECT_EQ(result.validMask.at<uchar>(0, 1), 1);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(0, 1), -2.0f);
}

TEST(DisparityValidatorTest, Validate_InvalidCenterRemainsInvalidWithValidNeighbors)
{
    const cv::Mat disparity(3, 3, CV_32FC1, cv::Scalar(7.0f));
    const cv::Mat confidence(3, 3, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat rawValid(3, 3, CV_8UC1, cv::Scalar(1));
    rawValid.at<uchar>(1, 1) = 0;

    DenseMatchConfig config;
    config.medianFilterSize = 3;
    DisparityValidator validator(config);
    const DisparityResult result = validator.validate(disparity, confidence, rawValid);

    EXPECT_EQ(result.validMask.at<uchar>(1, 1), 0);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(1, 1), 0.0f);
    EXPECT_FLOAT_EQ(result.confidence.at<float>(1, 1), 0.0f);
}

TEST(DisparityValidatorTest, MaskedMedianPreservesCenterWhenNoFiniteSampleExists)
{
    cv::Mat disparity(3, 3, CV_32FC1, cv::Scalar(0.0f));
    disparity.at<float>(1, 1) = std::numeric_limits<float>::quiet_NaN();
    cv::Mat valid(3, 3, CV_8UC1, cv::Scalar(0));
    valid.at<uchar>(1, 1) = 1;

    DenseMatchConfig config;
    DisparityValidator validator(config);
    const cv::Mat filtered = validator.medianFilter(disparity, valid, 3);

    EXPECT_TRUE(std::isnan(filtered.at<float>(1, 1)));
}

TEST(DisparityValidatorTest, ExtremeMaskedMedianKernelUsesBoundedImageCapacity)
{
    DenseMatchConfig config;
    DisparityValidator validator(config);
    const cv::Mat disparity(1, 1, CV_32FC1, cv::Scalar(7.0f));
    const cv::Mat valid(1, 1, CV_8UC1, cv::Scalar(1));

    const cv::Mat filtered = validator.medianFilter(
        disparity,
        valid,
        std::numeric_limits<int>::max());

    ASSERT_EQ(filtered.size(), disparity.size());
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 0), 7.0f);
}

TEST(DisparityValidatorTest, ImageSupportMaskRemovesBlackAndOutOfBoundsMatches)
{
    DenseMatchConfig cfg;
    cfg.supportIntensityThreshold = 5;
    DisparityValidator validator(cfg);

    DisparityResult result;
    result.disparity = cv::Mat(1, 5, CV_32FC1, cv::Scalar(2.0f));
    result.confidence = cv::Mat(1, 5, CV_32FC1, cv::Scalar(1.0f));
    result.validMask = cv::Mat(1, 5, CV_8UC1, cv::Scalar(1));

    cv::Mat left(1, 5, CV_8UC1);
    left.at<uchar>(0, 0) = 10;  // right projection out of bounds
    left.at<uchar>(0, 1) = 10;  // right projection out of bounds
    left.at<uchar>(0, 2) = 0;   // invalid left support
    left.at<uchar>(0, 3) = 10;  // only valid pixel
    left.at<uchar>(0, 4) = 10;  // invalid right support

    cv::Mat right(1, 5, CV_8UC1);
    right.at<uchar>(0, 0) = 10;
    right.at<uchar>(0, 1) = 10; // x=3 projects here
    right.at<uchar>(0, 2) = 0;  // x=4 projects here
    right.at<uchar>(0, 3) = 10;
    right.at<uchar>(0, 4) = 10;

    validator.applyImageSupportMask(result, left, right);

    EXPECT_EQ(result.validMask.at<uchar>(0, 0), 0);
    EXPECT_EQ(result.validMask.at<uchar>(0, 1), 0);
    EXPECT_EQ(result.validMask.at<uchar>(0, 2), 0);
    EXPECT_EQ(result.validMask.at<uchar>(0, 3), 1);
    EXPECT_EQ(result.validMask.at<uchar>(0, 4), 0);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(0, 3), 2.0f);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(0, 4), 0.0f);
}

TEST(DisparityValidatorTest, ImageSupportAcceptsFiniteZeroAndNegativeDisparities)
{
    DenseMatchConfig config;
    config.supportIntensityThreshold = 0;
    DisparityValidator validator(config);

    DisparityResult result;
    result.disparity = cv::Mat(1, 4, CV_32FC1);
    result.disparity.at<float>(0, 0) = -2.0f;
    result.disparity.at<float>(0, 1) = -1.0f;
    result.disparity.at<float>(0, 2) = 0.0f;
    result.disparity.at<float>(0, 3) = 1.0f;
    result.confidence = cv::Mat(1, 4, CV_32FC1, cv::Scalar(1.0f));
    result.validMask = cv::Mat(1, 4, CV_8UC1, cv::Scalar(1));
    const cv::Mat left(1, 4, CV_8UC1, cv::Scalar(50));
    const cv::Mat right(1, 4, CV_8UC1, cv::Scalar(50));

    validator.applyImageSupportMask(result, left, right);

    EXPECT_EQ(result.validMask.at<uchar>(0, 0), 1); // right x=2
    EXPECT_EQ(result.validMask.at<uchar>(0, 1), 1); // right x=2
    EXPECT_EQ(result.validMask.at<uchar>(0, 2), 1); // right x=2
    EXPECT_EQ(result.validMask.at<uchar>(0, 3), 1); // right x=2
}
