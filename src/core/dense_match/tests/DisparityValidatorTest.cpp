#include <gtest/gtest.h>
#include "DisparityValidator.h"

using namespace xjw::dense_match;

TEST(DisparityValidatorTest, LRCheck_Consistent_Passes)
{
    cv::Mat dispLR(20, 20, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat dispRL(20, 20, CV_32FC1, cv::Scalar(0.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 1.0f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < 20; ++y)
        for (int x = 0; x < 20; ++x)
            EXPECT_EQ(valid.at<uchar>(y, x), 1);
}

TEST(DisparityValidatorTest, LRCheck_Inconsistent_Fails)
{
    cv::Mat dispLR(5, 5, CV_32FC1, cv::Scalar(10.0f));
    cv::Mat dispRL(5, 5, CV_32FC1, cv::Scalar(1.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 1.0f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x)
            EXPECT_EQ(valid.at<uchar>(y, x), 0);
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
