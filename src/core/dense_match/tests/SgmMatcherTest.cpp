// =============================================================================
// 文件: SgmMatcherTest.cpp
// 功能: SGM匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>

#include "SgmMatcher.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

using namespace xjw::dense_match;

namespace
{

std::pair<cv::Mat, cv::Mat> makeSgmShiftedPair(int width, int height, int disparity)
{
    cv::Mat left(height, width, CV_8UC1);
    cv::Mat right(height, width, CV_8UC1);
    cv::RNG rng(0x713bu + static_cast<unsigned>(disparity + 32));
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

DenseMatchConfig cpuSgmConfig(int minDisparity, int maxDisparity)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::SemiGlobalMatch;
    config.costFunc = CostFunction::AbsoluteDifference;
    config.subpixel = SubpixelMode::None;
    config.minDisparity = minDisparity;
    config.maxDisparity = maxDisparity;
    config.corrKernelW = 5;
    config.corrKernelH = 5;
    config.p1 = 8;
    config.p2 = 32;
    config.sgmDirections = 8;
    config.useCuda = false;
    return config;
}

void expectSgmAccuracy(const DisparityResult &result, cv::Rect region, float expected)
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

const std::array<SgmDirection, 8> kTestDirections = {{
    {1, 0},
    {-1, 0},
    {0, 1},
    {0, -1},
    {1, 1},
    {-1, -1},
    {1, -1},
    {-1, 1}}};

} // namespace

TEST(SgmMatcherTest, IdenticalRandomImages_ProduceZeroDisparity)
{
    auto [left, right] = makeSgmShiftedPair(96, 64, 0);
    SgmMatcher matcher(cpuSgmConfig(0, 16));
    const DisparityResult result = matcher.compute(left, right);

    ASSERT_FALSE(result.disparity.empty());
    expectSgmAccuracy(result, cv::Rect(24, 12, 48, 40), 0.0f);
}

TEST(SgmMatcherTest, PositiveAndNegativeShiftsUseLeftReferenceConvention)
{
    for (const int disparity : {7, -5})
    {
        SCOPED_TRACE(disparity);
        auto [left, right] = makeSgmShiftedPair(128, 96, disparity);
        const int minimum = disparity < 0 ? -16 : 0;
        const int maximum = disparity < 0 ? 1 : 24;
        SgmMatcher matcher(cpuSgmConfig(minimum, maximum));
        const DisparityResult result = matcher.compute(left, right);
        expectSgmAccuracy(result, cv::Rect(28, 16, 64, 64), disparity);
    }
}

TEST(SgmMatcherTest, DirectionsHaveIndependentPathStateForOneFourAndEightPaths)
{
    cv::Mat left(6, 9, CV_8UC1);
    cv::Mat right(6, 9, CV_8UC1);
    cv::RNG rng(0x9213u);
    rng.fill(left, cv::RNG::UNIFORM, 0, 256);
    rng.fill(right, cv::RNG::UNIFORM, 0, 256);
    const CostVolume costVolume = computeCostVolume(
        left,
        right,
        -2,
        4,
        3,
        3,
        CostFunction::AbsoluteDifference);

    for (const int directionCount : {1, 4, 8})
    {
        SCOPED_TRACE(directionCount);
        const std::vector<SgmDirection> directions(
            kTestDirections.begin(),
            kTestDirections.begin() + directionCount);
        const CostVolume combined = aggregateSgmCostVolume(costVolume, 3, 11, directions);

        std::vector<CostVolume> independent;
        independent.reserve(static_cast<std::size_t>(directionCount));
        for (const SgmDirection direction : directions)
        {
            independent.push_back(aggregateSgmCostVolume(costVolume, 3, 11, {direction}));
        }

        for (std::size_t disparityIndex = 0;
             disparityIndex < combined.size();
             ++disparityIndex)
        {
            for (int y = 0; y < left.rows; ++y)
            {
                for (int x = 0; x < left.cols; ++x)
                {
                    if (!combined.isValid(disparityIndex, y, x))
                    {
                        EXPECT_FLOAT_EQ(
                            combined[disparityIndex].at<float>(y, x),
                            kInvalidCost);
                        continue;
                    }

                    float expected = 0.0f;
                    for (const CostVolume &singleDirection : independent)
                    {
                        expected += singleDirection[disparityIndex].at<float>(y, x);
                    }
                    EXPECT_NEAR(
                        combined[disparityIndex].at<float>(y, x),
                        expected,
                        1.0e-4f);
                }
            }
        }
    }
}

TEST(SgmMatcherTest, AggregateIsIndependentOfDirectionTraversalOrder)
{
    cv::Mat left(7, 10, CV_8UC1);
    cv::Mat right(7, 10, CV_8UC1);
    cv::RNG rng(0x41ffu);
    rng.fill(left, cv::RNG::UNIFORM, 0, 256);
    rng.fill(right, cv::RNG::UNIFORM, 0, 256);
    const CostVolume costVolume = computeCostVolume(
        left,
        right,
        -2,
        5,
        3,
        3,
        CostFunction::CensusTransform);

    std::vector<SgmDirection> forward(kTestDirections.begin(), kTestDirections.end());
    std::vector<SgmDirection> reverse = forward;
    std::reverse(reverse.begin(), reverse.end());
    const CostVolume forwardAggregate = aggregateSgmCostVolume(costVolume, 2, 8, forward);
    const CostVolume reverseAggregate = aggregateSgmCostVolume(costVolume, 2, 8, reverse);

    for (std::size_t disparityIndex = 0;
         disparityIndex < forwardAggregate.size();
         ++disparityIndex)
    {
        EXPECT_LE(
            cv::norm(
                forwardAggregate[disparityIndex],
                reverseAggregate[disparityIndex],
                cv::NORM_INF,
                forwardAggregate.hypothesisValidMask(disparityIndex)),
            1.0e-5);
    }
}

TEST(SgmMatcherTest, BoundaryPixelWithoutLegalHypothesisIsInvalid)
{
    const cv::Mat image(8, 16, CV_8UC1, cv::Scalar(80));
    DenseMatchConfig config = cpuSgmConfig(4, 8);
    SgmMatcher matcher(config);
    const DisparityResult result = matcher.compute(image, image);

    EXPECT_EQ(result.validMask.at<uchar>(4, 0), 0);
    EXPECT_FLOAT_EQ(result.confidence.at<float>(4, 0), 0.0f);
}

TEST(SgmMatcherTest, CensusWithoutEvidenceRemainsInvalidAfterAggregation)
{
    const cv::Mat image(5, 9, CV_8UC1, cv::Scalar(128));
    const std::array<std::pair<CostFunction, int>, 2> cases = {{
        {CostFunction::CensusTransform, 1},
        {CostFunction::TernaryCensusTransform, 3}}};

    for (const auto &[function, kernelSize] : cases)
    {
        SCOPED_TRACE(static_cast<int>(function));
        DenseMatchConfig config = cpuSgmConfig(0, 1);
        config.costFunc = function;
        config.corrKernelW = kernelSize;
        config.corrKernelH = kernelSize;
        const DisparityResult result = SgmMatcher(config).compute(image, image);

        EXPECT_EQ(cv::countNonZero(result.validMask), 0);
        EXPECT_EQ(cv::countNonZero(result.confidence), 0);
    }
}

TEST(SgmMatcherTest, OutputHasValidMask)
{
    auto [left, right] = makeSgmShiftedPair(64, 64, 3);
    SgmMatcher matcher(cpuSgmConfig(0, 16));
    const DisparityResult result = matcher.compute(left, right);

    EXPECT_EQ(result.disparity.size(), left.size());
    EXPECT_EQ(result.confidence.size(), left.size());
    EXPECT_EQ(result.validMask.size(), left.size());
    EXPECT_EQ(result.validMask.type(), CV_8UC1);
}
