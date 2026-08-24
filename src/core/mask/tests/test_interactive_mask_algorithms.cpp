#include "InteractiveMaskAlgorithms.h"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

TEST(InteractiveMaskAlgorithmsTest, RectangleCanAddAndEraseExclusion)
{
    cv::Mat mask = cv::Mat::zeros(20, 30, CV_8UC1);
    const cv::Mat selection = xjw::mask::rectangleSelection(mask.size(), cv::Rect(4, 5, 8, 6));

    xjw::mask::applySelectionToMask(&mask, selection, true);
    EXPECT_EQ(cv::countNonZero(mask), 48);
    xjw::mask::applySelectionToMask(&mask, selection, false);
    EXPECT_EQ(cv::countNonZero(mask), 0);
}

TEST(InteractiveMaskAlgorithmsTest, MagicWandKeepsConnectedSimilarRegion)
{
    cv::Mat image(20, 30, CV_8UC3, cv::Scalar(20, 20, 20));
    image(cv::Rect(2, 3, 8, 9)).setTo(cv::Scalar(120, 125, 130));
    image(cv::Rect(20, 3, 8, 9)).setTo(cv::Scalar(120, 125, 130));

    const cv::Mat selection = xjw::mask::magicWandSelection(image, cv::Point(4, 5), 10);

    EXPECT_EQ(cv::countNonZero(selection), 72);
    EXPECT_EQ(selection.at<unsigned char>(5, 22), 0);
}

TEST(InteractiveMaskAlgorithmsTest, SmartBrushHonorsRadiusAndColorBoundary)
{
    cv::Mat image(31, 31, CV_8UC3, cv::Scalar(10, 10, 10));
    image(cv::Rect(16, 0, 15, 31)).setTo(cv::Scalar(240, 240, 240));

    const cv::Mat selection = xjw::mask::smartBrushSelection(image, {cv::Point(14, 15)}, 8, 15);

    EXPECT_NE(selection.at<unsigned char>(15, 10), 0);
    EXPECT_EQ(selection.at<unsigned char>(15, 18), 0);
}

TEST(InteractiveMaskAlgorithmsTest, ScissorsPathUsesStrongNearbyEdge)
{
    cv::Mat image(40, 50, CV_8UC3, cv::Scalar(10, 10, 10));
    image(cv::Rect(0, 18, 50, 22)).setTo(cv::Scalar(245, 245, 245));

    const std::vector<cv::Point> path = xjw::mask::edgeSnappedPath(image, cv::Point(3, 15), cv::Point(46, 15), 8);

    ASSERT_GT(path.size(), 20u);
    int nearEdge = 0;
    for (const cv::Point& point : path)
    {
        nearEdge += std::abs(point.y - 18) <= 1 ? 1 : 0;
    }
    EXPECT_GT(nearEdge, static_cast<int>(path.size() / 2));
}
