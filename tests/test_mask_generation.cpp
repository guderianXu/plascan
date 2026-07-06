#include "MaskGenerator.h"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

namespace
{

cv::Mat makeAsteroidLikeImage()
{
    cv::Mat image(80, 100, CV_8UC1, cv::Scalar(0));
    cv::ellipse(image, cv::Point(50, 40), cv::Size(22, 16), 12.0, 0.0, 360.0, cv::Scalar(210), -1);
    cv::circle(image, cv::Point(36, 32), 4, cv::Scalar(165), -1);
    cv::circle(image, cv::Point(61, 49), 3, cv::Scalar(245), -1);
    return image;
}

} // namespace

TEST(MaskGeneratorTest, BlackBackgroundMethodMasksBackgroundAndKeepsTarget)
{
    xjw::mask::MaskGenerationOptions options;
    options.method = xjw::mask::MaskGenerationMethod::BlackBackground;
    options.threshold = -1.0;
    options.minComponentArea = 20;
    options.morphologyRadius = 1;

    const cv::Mat mask = xjw::mask::generateMask(makeAsteroidLikeImage(), options);

    ASSERT_EQ(mask.type(), CV_8UC1);
    ASSERT_EQ(mask.rows, 80);
    ASSERT_EQ(mask.cols, 100);
    EXPECT_EQ(mask.at<uchar>(0, 0), 255);
    EXPECT_EQ(mask.at<uchar>(40, 50), 0);
    EXPECT_EQ(mask.at<uchar>(49, 61), 0);
}

TEST(MaskGeneratorTest, ContoursFollowUnmaskedForegroundBoundary)
{
    xjw::mask::MaskGenerationOptions options;
    options.method = xjw::mask::MaskGenerationMethod::BlackBackground;
    options.threshold = -1.0;
    options.minComponentArea = 20;
    options.morphologyRadius = 1;

    const cv::Mat mask = xjw::mask::generateMask(makeAsteroidLikeImage(), options);
    const auto contours = xjw::mask::extractMaskContours(mask, true);

    ASSERT_FALSE(contours.empty());
    EXPECT_GT(contours.front().size(), 20);
}

TEST(MaskComposerTest, OperationsMatchMetashapeStyleMaskComposition)
{
    cv::Mat existing(3, 3, CV_8UC1, cv::Scalar(0));
    cv::Mat generated(3, 3, CV_8UC1, cv::Scalar(0));
    existing.at<uchar>(1, 1) = 255;
    generated.at<uchar>(1, 1) = 255;
    generated.at<uchar>(0, 0) = 255;

    const cv::Mat replaced = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Replace);
    const cv::Mat united = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Union);
    const cv::Mat intersected = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Intersection);
    const cv::Mat differed = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Difference);

    EXPECT_EQ(replaced.at<uchar>(0, 0), 255);
    EXPECT_EQ(united.at<uchar>(0, 0), 255);
    EXPECT_EQ(intersected.at<uchar>(0, 0), 0);
    EXPECT_EQ(intersected.at<uchar>(1, 1), 255);
    EXPECT_EQ(differed.at<uchar>(0, 0), 0);
    EXPECT_EQ(differed.at<uchar>(1, 1), 0);
}
