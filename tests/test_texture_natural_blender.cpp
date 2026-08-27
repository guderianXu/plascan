#include "TextureNaturalBlender.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <cmath>

namespace
{

    double linearValue(std::uint8_t encoded)
    {
        const double normalized = encoded / 255.0;
        return normalized <= 0.04045 ? normalized / 12.92 : std::pow((normalized + 0.055) / 1.055, 2.4);
    }

    double checkerContrastLinear(const cv::Mat& image)
    {
        double total = 0.0;
        int count = 0;
        for (int row = 0; row < image.rows; ++row)
        {
            for (int column = 0; column + 1 < image.cols; ++column)
            {
                total += std::abs(linearValue(image.at<cv::Vec3b>(row, column)[0]) -
                                  linearValue(image.at<cv::Vec3b>(row, column + 1)[0]));
                ++count;
            }
        }
        return count > 0 ? total / count : 0.0;
    }

    double meanLinear(const cv::Mat& image)
    {
        double total = 0.0;
        for (int row = 0; row < image.rows; ++row)
        {
            for (int column = 0; column < image.cols; ++column)
            {
                total += linearValue(image.at<cv::Vec3b>(row, column)[0]);
            }
        }
        return total / image.total();
    }

} // namespace

TEST(TextureNaturalBlenderTest, ReplacesLowFrequencyWhileKeepingWinnerDetail)
{
    constexpr int size = 128;
    cv::Mat primary(size, size, CV_8UC3);
    cv::Mat robust(size, size, CV_8UC3);
    for (int row = 0; row < size; ++row)
    {
        for (int column = 0; column < size; ++column)
        {
            const std::uint8_t value = (row + column) % 2 == 0 ? 80 : 140;
            primary.at<cv::Vec3b>(row, column) = cv::Vec3b(value, value, value);
            robust.at<cv::Vec3b>(row, column) = cv::Vec3b(value + 20, value + 20, value + 20);
        }
    }
    cv::Mat atlas = robust.clone();
    cv::Mat mask(size, size, CV_8UC1, cv::Scalar(255));

    const auto stats = xjw::mesh::texture_v4::applyTextureNaturalBlend(
        &atlas, primary, mask, mask, cv::Rect(0, 0, size, size), 5, 1.0f);

    EXPECT_EQ(stats.pyramidLevels, 5);
    EXPECT_EQ(stats.correctedPixelCount, size * size);
    EXPECT_GT(stats.meanAbsoluteLinearCorrection, 0.01);
    EXPECT_NEAR(meanLinear(atlas), meanLinear(robust), 0.002);
    EXPECT_NEAR(checkerContrastLinear(atlas), checkerContrastLinear(primary), 0.004);
}

TEST(TextureNaturalBlenderTest, DoesNotWriteOutsidePrimaryAndFilledMask)
{
    cv::Mat primary(64, 64, CV_8UC3, cv::Scalar(90, 90, 90));
    cv::Mat atlas(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
    const cv::Rect content(16, 16, 32, 32);
    atlas(content).setTo(cv::Scalar(120, 120, 120));
    mask(content).setTo(255);

    const auto stats =
        xjw::mesh::texture_v4::applyTextureNaturalBlend(&atlas, primary, mask, mask, cv::Rect(0, 0, 64, 64), 5, 1.0f);

    EXPECT_EQ(stats.correctedPixelCount, content.area());
    EXPECT_EQ(atlas.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
    EXPECT_NE(atlas.at<cv::Vec3b>(32, 32), cv::Vec3b(0, 0, 0));
}
