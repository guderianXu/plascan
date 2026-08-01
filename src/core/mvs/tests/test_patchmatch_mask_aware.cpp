#include "PatchMatchCUDA.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

constexpr int kWidth = 64;
constexpr int kHeight = 48;
constexpr int kDisparity = 5;
constexpr float kExpectedDepth = 10.0f;

xjw::Camera makeCamera(double center_x)
{
    xjw::Camera camera;
    camera.setIntrinsics(50.0, 50.0, kWidth * 0.5, kHeight * 0.5);
    camera.setPose(
        std::array<double, 9>{1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0},
        std::array<double, 3>{center_x, 0.0, 0.0});
    return camera.normalizedForPositiveDepth();
}

cv::Mat makeReferenceImage()
{
    cv::Mat image(kHeight, kWidth, CV_8U, cv::Scalar(0));
    for (int row = 10; row < 38; ++row)
    {
        for (int column = 12; column < 52; ++column)
        {
            const double value = 128.0 +
                55.0 * std::sin(0.31 * static_cast<double>(column)) +
                45.0 * std::cos(0.27 * static_cast<double>(row));
            image.at<std::uint8_t>(row, column) = static_cast<std::uint8_t>(
                std::clamp(value, 1.0, 255.0));
        }
    }
    return image;
}

cv::Mat makeReferenceMask()
{
    cv::Mat mask(kHeight, kWidth, CV_8U, cv::Scalar(0));
    mask(cv::Rect(12, 10, 40, 28)).setTo(cv::Scalar(255));
    return mask;
}

cv::Mat shiftLeft(const cv::Mat &input)
{
    cv::Mat output(input.size(), input.type(), cv::Scalar(0));
    input(cv::Rect(kDisparity, 0, kWidth - kDisparity, kHeight))
        .copyTo(output(cv::Rect(0, 0, kWidth - kDisparity, kHeight)));
    return output;
}

struct EstimateResult
{
    cv::Mat depth;
    cv::Mat confidence;
};

EstimateResult estimateMaskedPlane(bool use_cuda)
{
    const cv::Mat reference = makeReferenceImage();
    const cv::Mat source = shiftLeft(reference);
    const cv::Mat reference_mask = makeReferenceMask();
    const cv::Mat source_mask = shiftLeft(reference_mask);

    xjw::mvs::PatchMatchConfig config;
    config.useCuda = use_cuda;
    config.cudaFallbackToCpu = false;
    config.downsampleFactor = 1;
    config.numIterations = 2;
    config.patchHalf = 2;
    config.confidenceThresh = 0.05f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;

    cv::Mat hint(kHeight, kWidth, CV_32F, cv::Scalar(kExpectedDepth));
    cv::Mat radius(kHeight, kWidth, CV_32F, cv::Scalar(0.25f));
    EstimateResult result;
    std::string error;
    const std::vector<cv::Mat> source_masks{source_mask};
    EXPECT_TRUE(xjw::mvs::PatchMatchDepthEstimator::estimate(
        reference,
        std::vector<cv::Mat>{source},
        makeCamera(0.0),
        std::vector<xjw::Camera>{makeCamera(1.0)},
        5.0f,
        15.0f,
        config,
        result.depth,
        &result.confidence,
        &error,
        &hint,
        &radius,
        &reference_mask,
        &source_masks)) << error;
    return result;
}

double validRatio(const cv::Mat &depth, const cv::Mat &mask)
{
    cv::Mat valid = depth > 0.0f;
    cv::bitwise_and(valid, mask, valid);
    return static_cast<double>(cv::countNonZero(valid)) /
           static_cast<double>(std::max(1, cv::countNonZero(mask)));
}

TEST(PatchMatchMaskAwareTest, CpuKeepsDepthInsideReferenceMaskOnly)
{
    const EstimateResult result = estimateMaskedPlane(false);
    ASSERT_FALSE(result.depth.empty());

    const cv::Mat reference_mask = makeReferenceMask();
    cv::Mat outside_mask;
    cv::bitwise_not(reference_mask, outside_mask);
    cv::Mat outside_depth;
    result.depth.copyTo(outside_depth, outside_mask);
    EXPECT_EQ(cv::countNonZero(outside_depth > 0.0f), 0);
    EXPECT_GT(validRatio(result.depth, reference_mask), 0.25);
}

TEST(PatchMatchMaskAwareTest, CpuAndCudaApplyEquivalentMaskSemanticsWhenAvailable)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA is unavailable";
    }

    const EstimateResult cpu = estimateMaskedPlane(false);
    const EstimateResult gpu = estimateMaskedPlane(true);
    ASSERT_FALSE(cpu.depth.empty());
    ASSERT_FALSE(gpu.depth.empty());

    const cv::Mat reference_mask = makeReferenceMask();
    const double cpu_ratio = validRatio(cpu.depth, reference_mask);
    const double gpu_ratio = validRatio(gpu.depth, reference_mask);
    EXPECT_GT(cpu_ratio, 0.25);
    EXPECT_GT(gpu_ratio, 0.25);
    EXPECT_NEAR(cpu_ratio, gpu_ratio, 0.30);

    cv::Mat outside_mask;
    cv::bitwise_not(reference_mask, outside_mask);
    cv::Mat gpu_outside_depth;
    gpu.depth.copyTo(gpu_outside_depth, outside_mask);
    EXPECT_EQ(cv::countNonZero(gpu_outside_depth > 0.0f), 0);
}

} // namespace
