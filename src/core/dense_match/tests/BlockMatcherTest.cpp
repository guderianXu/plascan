// =============================================================================
// 文件: BlockMatcherTest.cpp
// 功能: WTA块匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>

#include "BlockMatcher.h"
#include "CostFunctions.h"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <utility>

using namespace xjw::dense_match;

namespace
{

    std::pair<cv::Mat, cv::Mat> makeShiftedPair(int width, int height, int disparity)
    {
        cv::Mat left(height, width, CV_8UC1);
        cv::Mat right(height, width, CV_8UC1);
        cv::RNG rng(0x31a5u + static_cast<unsigned>(disparity + 32));
        rng.fill(left, cv::RNG::UNIFORM, 0, 256);
        rng.fill(right, cv::RNG::UNIFORM, 0, 256);

        if (disparity >= 0)
        {
            left(cv::Rect(disparity, 0, width - disparity, height))
                .copyTo(right(cv::Rect(0, 0, width - disparity, height)));
        }
        else
        {
            const int rightOffset = -disparity;
            left(cv::Rect(0, 0, width - rightOffset, height))
                .copyTo(right(cv::Rect(rightOffset, 0, width - rightOffset, height)));
        }
        return {left, right};
    }

    DenseMatchConfig cpuBlockConfig(int minDisparity, int maxDisparity)
    {
        DenseMatchConfig config;
        config.algorithm = StereoAlgorithm::BlockMatch;
        config.costFunc = CostFunction::AbsoluteDifference;
        config.subpixel = SubpixelMode::None;
        config.minDisparity = minDisparity;
        config.maxDisparity = maxDisparity;
        config.corrKernelW = 7;
        config.corrKernelH = 7;
        config.useCuda = false;
        return config;
    }

    void expectAccurateDisparity(const DisparityResult& result, cv::Rect region, float expected)
    {
        int validCount = 0;
        int accurateCount = 0;
        for (int y = region.y; y < region.y + region.height; ++y)
        {
            for (int x = region.x; x < region.x + region.width; ++x)
            {
                if (result.validMask.at<uchar>(y, x) == 0)
                {
                    continue;
                }
                ++validCount;
                if (std::fabs(result.disparity.at<float>(y, x) - expected) <= 0.5f)
                {
                    ++accurateCount;
                }
            }
        }
        ASSERT_GT(validCount, static_cast<int>(region.area() * 0.99));
        EXPECT_GE(accurateCount, static_cast<int>(validCount * 0.99));
    }

} // namespace

TEST(BlockMatcherTest, ZeroShift_ProducesZeroDisparity)
{
    auto [left, right] = makeShiftedPair(96, 64, 0);
    BlockMatcher matcher(cpuBlockConfig(0, 16));
    const DisparityResult result = matcher.compute(left, right);

    ASSERT_FALSE(result.disparity.empty());
    expectAccurateDisparity(result, cv::Rect(24, 12, 48, 40), 0.0f);
}

TEST(BlockMatcherTest, PositiveShift_UsesLeftReferenceDisparity)
{
    constexpr int disparity = 8;
    auto [left, right] = makeShiftedPair(128, 96, disparity);
    BlockMatcher matcher(cpuBlockConfig(0, 32));
    const DisparityResult result = matcher.compute(left, right);

    expectAccurateDisparity(result, cv::Rect(32, 16, 64, 64), disparity);
}

TEST(BlockMatcherTest, NegativeShift_UsesSameLeftReferenceConvention)
{
    constexpr int disparity = -6;
    auto [left, right] = makeShiftedPair(128, 96, disparity);
    BlockMatcher matcher(cpuBlockConfig(-16, 1));
    const DisparityResult result = matcher.compute(left, right);

    expectAccurateDisparity(result, cv::Rect(24, 16, 64, 64), disparity);
}

TEST(BlockMatcherTest, InvalidBoundaryHypothesesCannotWin)
{
    const cv::Mat left(3, 8, CV_8UC1, cv::Scalar(20));
    const cv::Mat right(3, 8, CV_8UC1, cv::Scalar(30));
    DenseMatchConfig config = cpuBlockConfig(0, 4);
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    BlockMatcher matcher(config);
    const DisparityResult result = matcher.compute(left, right);

    // At x=0 only disparity zero is geometrically legal.  The old zero-filled
    // invalid hypotheses incorrectly won here.
    EXPECT_EQ(result.validMask.at<uchar>(1, 0), 1);
    EXPECT_FLOAT_EQ(result.disparity.at<float>(1, 0), 0.0f);
    EXPECT_TRUE(std::isfinite(result.confidence.at<float>(1, 0)));
    EXPECT_FLOAT_EQ(result.confidence.at<float>(1, 0), 1.0f);
}

TEST(BlockMatcherTest, PixelWithoutLegalHypothesisIsInvalid)
{
    const cv::Mat image(3, 8, CV_8UC1, cv::Scalar(20));
    DenseMatchConfig config = cpuBlockConfig(3, 6);
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    BlockMatcher matcher(config);
    const DisparityResult result = matcher.compute(image, image);

    EXPECT_EQ(result.validMask.at<uchar>(1, 0), 0);
    EXPECT_FLOAT_EQ(result.confidence.at<float>(1, 0), 0.0f);
}

TEST(BlockMatcherTest, TiedBestHypothesesAreMarkedAmbiguous)
{
    const cv::Mat image(5, 16, CV_8UC1, cv::Scalar(128));
    DenseMatchConfig config = cpuBlockConfig(0, 8);
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    BlockMatcher matcher(config);
    const DisparityResult result = matcher.compute(image, image);

    EXPECT_EQ(result.validMask.at<uchar>(2, 10), 0);
    EXPECT_FLOAT_EQ(result.confidence.at<float>(2, 10), 0.0f);
}

TEST(BlockMatcherTest, CensusWithoutEvidenceCannotProduceAWinner)
{
    const cv::Mat image(5, 9, CV_8UC1, cv::Scalar(128));
    const std::array<std::pair<CostFunction, int>, 2> cases = {
        {{CostFunction::CensusTransform, 1}, {CostFunction::TernaryCensusTransform, 3}}};

    for (const auto& [function, kernelSize] : cases)
    {
        SCOPED_TRACE(static_cast<int>(function));
        DenseMatchConfig config = cpuBlockConfig(0, 1);
        config.costFunc = function;
        config.corrKernelW = kernelSize;
        config.corrKernelH = kernelSize;
        const DisparityResult result = BlockMatcher(config).compute(image, image);

        EXPECT_EQ(cv::countNonZero(result.validMask), 0);
        EXPECT_EQ(cv::countNonZero(result.confidence), 0);
    }
}

TEST(BlockMatcherTest, OutputHasCorrectDimensions)
{
    auto [left, right] = makeShiftedPair(80, 100, 3);
    BlockMatcher matcher(cpuBlockConfig(0, 16));
    const DisparityResult result = matcher.compute(left, right);

    EXPECT_EQ(result.disparity.size(), left.size());
    EXPECT_EQ(result.confidence.size(), left.size());
    EXPECT_EQ(result.validMask.size(), left.size());
}

#ifdef DM_ENABLE_CUDA
TEST(BlockMatcherCudaParityTest, CpuAndCudaProduceSameDisparityAndValidity)
{
    if (!isCostVolumeCUDAAvailable())
    {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const std::array<CostFunction, 5> functions = {CostFunction::AbsoluteDifference,
                                                   CostFunction::SquaredDifference,
                                                   CostFunction::NormalizedCrossCorr,
                                                   CostFunction::CensusTransform,
                                                   CostFunction::TernaryCensusTransform};
    for (const int disparity : {5, -4})
    {
        auto [left, right] = makeShiftedPair(64, 40, disparity);
        for (const CostFunction function : functions)
        {
            SCOPED_TRACE(::testing::Message() << "disparity=" << disparity << " cost=" << static_cast<int>(function));
            DenseMatchConfig cpuConfig = cpuBlockConfig(disparity < 0 ? -8 : 0, disparity < 0 ? 1 : 10);
            cpuConfig.costFunc = function;
            cpuConfig.corrKernelW = 5;
            cpuConfig.corrKernelH = 3;
            DenseMatchConfig cudaConfig = cpuConfig;
            cudaConfig.useCuda = true;

            const DisparityResult cpu = BlockMatcher(cpuConfig).compute(left, right);
            const DisparityResult cuda = BlockMatcher(cudaConfig).compute(left, right);
            cv::Mat validityDifference;
            cv::bitwise_xor(cpu.validMask, cuda.validMask, validityDifference);
            EXPECT_EQ(cv::countNonZero(validityDifference), 0);
            EXPECT_LE(cv::norm(cpu.disparity, cuda.disparity, cv::NORM_INF), 1.0e-5);
            EXPECT_LE(cv::norm(cpu.confidence, cuda.confidence, cv::NORM_INF), 1.0e-4);
        }
    }
}
#else
TEST(BlockMatcherCudaParityTest, SkippedWhenCudaBackendIsNotBuilt)
{
    GTEST_SKIP() << "dense_match CUDA backend is not built";
}
#endif

#ifdef DM_ENABLE_OPENCL
TEST(BlockMatcherOpenClParityTest, ResidentPipelineMatchesCpuWithParabolicSubpixel)
{
    if (!isCostVolumeOpenCLAvailable())
    {
        GTEST_SKIP() << "OpenCL GPU device is not available";
    }

    auto [left, right] = makeShiftedPair(72, 48, 5);
    DenseMatchConfig cpuConfig = cpuBlockConfig(0, 12);
    cpuConfig.subpixel = SubpixelMode::Parabola;
    DenseMatchConfig openClConfig = cpuConfig;
    openClConfig.computeBackend = DenseMatchComputeBackend::OpenCl;

    const DisparityResult cpu = BlockMatcher(cpuConfig).compute(left, right);
    const DisparityResult openCl = BlockMatcher(openClConfig).compute(left, right);
    cv::Mat validityDifference;
    cv::bitwise_xor(cpu.validMask, openCl.validMask, validityDifference);
    EXPECT_EQ(cv::countNonZero(validityDifference), 0);
    EXPECT_LE(cv::norm(cpu.disparity, openCl.disparity, cv::NORM_INF), 2.0e-4);
    EXPECT_LE(cv::norm(cpu.confidence, openCl.confidence, cv::NORM_INF), 2.0e-4);
}
#else
TEST(BlockMatcherOpenClParityTest, SkippedWhenOpenClBackendIsNotBuilt)
{
    GTEST_SKIP() << "dense_match OpenCL backend is not built";
}
#endif
