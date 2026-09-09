#include "TextureNaturalBlender.h"

#include <metashape_texture/recovered_kernels.hpp>
#include <metashape_texture/texture_pipeline.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

using namespace xjw::mesh::texture_v4;

TEST(TextureNaturalBlenderTest, SingleCameraReconstructsOriginalAcrossAllBands)
{
    cv::Mat image(73, 91, CV_8UC3);
    cv::RNG random(42);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    const auto pyramid = buildTextureSourcePyramid(image, mask, mask);
    EXPECT_TRUE(pyramid.lowFrequency[0].empty());
    EXPECT_EQ(pyramid.lowFrequency[1].size(), cv::Size(23, 19));
    for (int row = 0; row < image.rows; row += 7)
    {
        for (int column = 0; column < image.cols; column += 7)
        {
            const cv::Vec3b expected = image.at<cv::Vec3b>(row, column);
            const std::array<TexturePyramidSample, 1> samples{
                {{&pyramid, cv::Point2d(column, row), cv::Vec3f(expected), 1.0f, true}}};
            EXPECT_EQ(textureLinearToSrgb(blendTexturePyramidSamples(samples, true, 36.0f)), expected);
        }
    }
}

TEST(TextureNaturalBlenderTest, KeepsWinnerDetailWhileMixingCoarseIllumination)
{
    cv::Mat winner(128, 128, CV_8UC3);
    for (int row = 0; row < winner.rows; ++row)
    {
        for (int column = 0; column < winner.cols; ++column)
        {
            const int value = (row + column) % 2 == 0 ? 80 : 140;
            winner.at<cv::Vec3b>(row, column) = cv::Vec3b(value, value, value);
        }
    }
    cv::Mat other(winner.size(), CV_8UC3, cv::Scalar(160, 160, 160));
    cv::Mat support(winner.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat no_winner(winner.size(), CV_8UC1, cv::Scalar(0));
    const auto first = buildTextureSourcePyramid(winner, support, support);
    const auto second = buildTextureSourcePyramid(other, support, no_winner);
    std::array<float, 2> original;
    std::array<float, 2> blended;
    for (int index = 0; index < 2; ++index)
    {
        const cv::Vec3f encoded(winner.at<cv::Vec3b>(64, 64 + index));
        const std::array<TexturePyramidSample, 2> samples{
            {{&first, cv::Point2d(64 + index, 64), encoded, 1.0f, true},
             {&second, cv::Point2d(64 + index, 64), cv::Vec3f(160, 160, 160), 1.0f, false}}};
        original[index] = textureSrgbToLinear(encoded)[0];
        blended[index] = blendTexturePyramidSamples(samples, false, 10000.0f)[0];
    }
    EXPECT_GT(blended[0], original[0]);
    EXPECT_NEAR(blended[1] - blended[0], original[1] - original[0], 1.0e-5f);
}

TEST(TextureNaturalBlenderTest, MaskedBackgroundDoesNotContaminateCoarseBands)
{
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat support(64, 64, CV_8UC1, cv::Scalar(0));
    image(cv::Rect(16, 16, 32, 32)).setTo(cv::Scalar(120, 120, 120));
    support(cv::Rect(16, 16, 32, 32)).setTo(255);
    const auto pyramid = buildTextureSourcePyramid(image, support, support);
    const float expected = textureSrgbToLinear(cv::Vec3f(120, 120, 120))[0];
    for (int level = 1; level < kTextureBlendLevels; ++level)
    {
        const cv::Mat& color = pyramid.lowFrequency[level];
        const cv::Vec3f center = color.at<cv::Vec3f>(color.rows / 2, color.cols / 2);
        for (int channel = 0; channel < 3; ++channel)
        {
            EXPECT_NEAR(center[channel], expected, 1.0e-5f);
        }
    }
}

TEST(TextureNaturalBlenderTest, RejectsGhostEvenWhenItWasPrimary)
{
    const std::array<TexturePyramidSample, 3> samples{{{nullptr, {}, cv::Vec3f(255, 0, 0), 1.0f, true},
                                                       {nullptr, {}, cv::Vec3f(100, 100, 100), 1.0f, false},
                                                       {nullptr, {}, cv::Vec3f(100, 100, 100), 1.0f, false}}};
    std::uint64_t rejected = 0;
    const cv::Vec3b color = textureLinearToSrgb(blendTexturePyramidSamples(samples, true, 36.0f, &rejected));
    EXPECT_EQ(rejected, 1U);
    EXPECT_EQ(color, cv::Vec3b(100, 100, 100));
}

TEST(TextureNaturalBlenderTest, CancellationDoesNotReturnPartialPyramid)
{
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(255));
    int calls = 0;
    const auto pyramid = buildTextureSourcePyramid(image, mask, mask, 1.0f, [&calls]() { return ++calls >= 3; });
    EXPECT_TRUE(pyramid.weight[0].empty());
}

TEST(TextureNaturalBlenderTest, HandlesOnePixelAndEmptySupport)
{
    cv::Mat image(1, 1, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat mask(1, 1, CV_8UC1, cv::Scalar(0));
    const auto pyramid = buildTextureSourcePyramid(image, mask, mask);
    EXPECT_EQ(pyramid.lowFrequency.back().at<cv::Vec3f>(0, 0), cv::Vec3f(0, 0, 0));
    EXPECT_EQ(pyramid.weight.back().at<float>(0, 0), 0.0f);
}

TEST(TextureNaturalBlenderTest, RejectsInvalidMaskShape)
{
    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat mask(4, 4, CV_8UC1, cv::Scalar(255));
    EXPECT_THROW(buildTextureSourcePyramid(image, mask, mask), std::invalid_argument);
}

TEST(TextureNaturalBlenderTest, NormalizesOnlySupportedSeamFootprint)
{
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat support(2, 2, CV_8UC1, cv::Scalar(0));
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(204, 153, 102);
    support.at<std::uint8_t>(0, 0) = 255;
    cv::Vec3f color;
    ASSERT_TRUE(sampleSupportedBilinear(image, support, 0.5, 0.5, &color));
    EXPECT_EQ(color, cv::Vec3f(204, 153, 102));
}

TEST(TextureNaturalBlenderTest, TreatsSupportedBlackAsValid)
{
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat support(2, 2, CV_8UC1, cv::Scalar(0));
    support.at<std::uint8_t>(0, 0) = 255;
    cv::Vec3f color;
    ASSERT_TRUE(sampleSupportedBilinear(image, support, 0.5, 0.5, &color));
    EXPECT_EQ(color, cv::Vec3f(0, 0, 0));
}

TEST(TextureNaturalBlenderTest, RecoveredComponentKernelRemovesOnlySmallIslands)
{
    metashape_texture::Image<std::uint8_t> input(5, 3);
    input(0, 0) = 255;
    input(2, 1) = 255;
    input(3, 1) = 255;
    input(2, 2) = 255;

    const metashape_texture::Image<std::uint8_t> filtered =
        metashape_texture::recovered::remove_small_components_8(input, 2U);

    EXPECT_EQ(filtered(0, 0), 0);
    EXPECT_EQ(filtered(2, 1), 255);
    EXPECT_EQ(filtered(3, 1), 255);
    EXPECT_EQ(filtered(2, 2), 255);
}

TEST(TextureNaturalBlenderTest, RecoveredPipelinePreservesExplicitInternalUvPrecision)
{
    metashape_texture::PipelineInput input;
    EXPECT_FALSE(input.precise_output_uv);
    input.precise_output_uv = true;
    EXPECT_TRUE(input.precise_output_uv);
}
