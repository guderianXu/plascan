#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

#include "TextureSeamLeveling.h"

namespace
{

float srgbToLinear(std::uint8_t value)
{
    const float normalized = static_cast<float>(value) / 255.0f;
    return normalized <= 0.04045f
        ? normalized / 12.92f
        : std::pow((normalized + 0.055f) / 1.055f, 2.4f);
}

float linearDifference(const cv::Vec3b &first, const cv::Vec3b &second)
{
    return std::fabs(
        srgbToLinear(first[0]) - srgbToLinear(second[0]));
}

TEST(TextureSeamLevelingTest, ReducesSharedEdgeDifferenceWithoutChangingInterior)
{
    cv::Mat atlas(12, 24, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat chart_index(12, 24, CV_32SC1, cv::Scalar(-1));
    atlas(cv::Rect(1, 1, 8, 10)).setTo(cv::Scalar(80, 80, 80));
    atlas(cv::Rect(15, 1, 8, 10)).setTo(cv::Scalar(180, 180, 180));
    chart_index(cv::Rect(1, 1, 8, 10)).setTo(0);
    chart_index(cv::Rect(15, 1, 8, 10)).setTo(1);

    const cv::Vec3b first_before = atlas.at<cv::Vec3b>(6, 8);
    const cv::Vec3b second_before = atlas.at<cv::Vec3b>(6, 15);
    const cv::Vec3b first_interior_before = atlas.at<cv::Vec3b>(6, 5);
    const cv::Vec3b second_interior_before = atlas.at<cv::Vec3b>(6, 18);
    const float difference =
        srgbToLinear(second_before[0]) - srgbToLinear(first_before[0]);
    const xjw::mesh::texture_v4::TextureSeamConstraint constraint{
        0, 1, cv::Vec3f(difference, difference, difference), 9.0f};

    const auto stats = xjw::mesh::texture_v4::applyTextureSeamLeveling(
        &atlas, chart_index, std::vector{constraint}, 2, 2, 0.20f);

    EXPECT_EQ(stats.constraintCount, 1);
    EXPECT_EQ(stats.connectedChartCount, 2);
    EXPECT_EQ(stats.adjustedChartCount, 2);
    EXPECT_GT(stats.adjustedPixelCount, 0);
    EXPECT_GT(stats.maximumAbsoluteLinearCorrection, 0.0f);
    EXPECT_LT(
        linearDifference(
            atlas.at<cv::Vec3b>(6, 8), atlas.at<cv::Vec3b>(6, 15)),
        linearDifference(first_before, second_before));
    EXPECT_EQ(atlas.at<cv::Vec3b>(6, 5), first_interior_before);
    EXPECT_EQ(atlas.at<cv::Vec3b>(6, 18), second_interior_before);
}

TEST(TextureSeamLevelingTest, InvalidOrDisconnectedConstraintsDoNotModifyAtlas)
{
    cv::Mat atlas(4, 4, CV_8UC3, cv::Scalar(64, 96, 128));
    const cv::Mat original = atlas.clone();
    cv::Mat chart_index(4, 4, CV_32SC1, cv::Scalar(0));
    const xjw::mesh::texture_v4::TextureSeamConstraint invalid{
        0, 0, cv::Vec3f(1.0f, 1.0f, 1.0f), 1.0f};

    const auto stats = xjw::mesh::texture_v4::applyTextureSeamLeveling(
        &atlas, chart_index, std::vector{invalid}, 1, 2, 0.08f);

    EXPECT_EQ(stats.constraintCount, 0);
    EXPECT_EQ(cv::countNonZero(atlas.reshape(1) != original.reshape(1)), 0);
}

TEST(TextureSeamLevelingTest, GlobalStrengthCorrectsChartInteriors)
{
    cv::Mat atlas(12, 24, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat chart_index(12, 24, CV_32SC1, cv::Scalar(-1));
    atlas(cv::Rect(1, 1, 8, 10)).setTo(cv::Scalar(80, 80, 80));
    atlas(cv::Rect(15, 1, 8, 10)).setTo(cv::Scalar(180, 180, 180));
    chart_index(cv::Rect(1, 1, 8, 10)).setTo(0);
    chart_index(cv::Rect(15, 1, 8, 10)).setTo(1);
    const cv::Vec3b first_interior_before = atlas.at<cv::Vec3b>(6, 5);
    const cv::Vec3b second_interior_before = atlas.at<cv::Vec3b>(6, 18);
    const float difference = srgbToLinear(180) - srgbToLinear(80);
    const xjw::mesh::texture_v4::TextureSeamConstraint constraint{
        0, 1, cv::Vec3f(difference, difference, difference), 9.0f};

    const auto stats = xjw::mesh::texture_v4::applyTextureSeamLeveling(
        &atlas,
        chart_index,
        std::vector{constraint},
        2,
        2,
        0.20f,
        0.5f);

    EXPECT_EQ(stats.adjustedChartCount, 2);
    EXPECT_NE(atlas.at<cv::Vec3b>(6, 5), first_interior_before);
    EXPECT_NE(atlas.at<cv::Vec3b>(6, 18), second_interior_before);
    EXPECT_LT(
        linearDifference(
            atlas.at<cv::Vec3b>(6, 5), atlas.at<cv::Vec3b>(6, 18)),
        linearDifference(first_interior_before, second_interior_before));
}

} // namespace
