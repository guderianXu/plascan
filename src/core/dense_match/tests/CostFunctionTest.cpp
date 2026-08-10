// =============================================================================
// 文件: CostFunctionTest.cpp
// 功能: 代价函数单元测试
// =============================================================================
#include <gtest/gtest.h>

#include "CostFunctions.h"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace xjw::dense_match;

namespace
{

void expectValidCostsNear(const CostVolume &volume, float expected, float tolerance)
{
    for (std::size_t disparityIndex = 0; disparityIndex < volume.size(); ++disparityIndex)
    {
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(
            volume[disparityIndex],
            &minimum,
            &maximum,
            nullptr,
            nullptr,
            volume.hypothesisValidMask(disparityIndex));
        EXPECT_NEAR(minimum, expected, tolerance);
        EXPECT_NEAR(maximum, expected, tolerance);
    }
}

} // namespace

class CostFunctionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        leftWhite = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        rightWhite = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        leftBlack = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
        rightBlack = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
    }

    cv::Mat leftWhite;
    cv::Mat rightWhite;
    cv::Mat leftBlack;
    cv::Mat rightBlack;
};

TEST_F(CostFunctionTest, AD_IdenticalImages_ZeroCostForValidHypotheses)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        5,
        3,
        3,
        CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 5u);
    expectValidCostsNear(volume, 0.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, AD_MaxDifference_Produces255)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightBlack,
        0,
        1,
        1,
        1,
        CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 1u);
    expectValidCostsNear(volume, 255.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, SD_IdenticalImages_ZeroCostForValidHypotheses)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        5,
        3,
        3,
        CostFunction::SquaredDifference);
    ASSERT_EQ(volume.size(), 5u);
    expectValidCostsNear(volume, 0.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, NCC_FlatIdenticalWindows_ZeroCost)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        1,
        3,
        3,
        CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    expectValidCostsNear(volume, 0.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, NCC_FlatDifferentWindows_ProducesCostTwo)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightBlack,
        0,
        1,
        3,
        3,
        CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    expectValidCostsNear(volume, 2.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, Census_Identical_ZeroCost)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        1,
        3,
        3,
        CostFunction::CensusTransform);
    ASSERT_EQ(volume.size(), 1u);
    expectValidCostsNear(volume, 0.0f, 1.0e-5f);
}

TEST_F(CostFunctionTest, Census_OneByOneWindowIsFiniteInvalidWithoutEvidence)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        1,
        1,
        1,
        CostFunction::CensusTransform);

    ASSERT_TRUE(volume.isValid(0, 2, 2));
    EXPECT_TRUE(std::isfinite(volume[0].at<float>(2, 2)));
    EXPECT_FLOAT_EQ(volume[0].at<float>(2, 2), kInvalidCost);
    EXPECT_FALSE(selectBestDisparity(volume, 2, 2).valid);
}

TEST_F(CostFunctionTest, Ternary_AllMaskedComparisonsAreFiniteInvalid)
{
    const CostVolume volume = computeCostVolume(
        leftWhite,
        rightWhite,
        0,
        1,
        3,
        3,
        CostFunction::TernaryCensusTransform);

    ASSERT_TRUE(volume.isValid(0, 2, 2));
    EXPECT_TRUE(std::isfinite(volume[0].at<float>(2, 2)));
    EXPECT_FLOAT_EQ(volume[0].at<float>(2, 2), kInvalidCost);
    EXPECT_FALSE(selectBestDisparity(volume, 2, 2).valid);
}

TEST(CostFunctionRangeTest, UsesLeftReferenceInclusiveExclusiveConvention)
{
    const DisparityIndexRange leftEdge = validDisparityIndexRangeForLeftX(0, 5, -2, 4);
    EXPECT_EQ(leftEdge.begin, 0);
    EXPECT_EQ(leftEdge.end, 3); // disparities -2, -1, 0

    const DisparityIndexRange rightEdge = validDisparityIndexRangeForLeftX(4, 5, -2, 4);
    EXPECT_EQ(rightEdge.begin, 2);
    EXPECT_EQ(rightEdge.end, 6); // disparities 0, 1, 2, 3

    const DisparityIndexRange positiveAtLeft = validDisparityIndexRangeForLeftX(0, 5, 0, 4);
    EXPECT_EQ(positiveAtLeft.begin, 0);
    EXPECT_EQ(positiveAtLeft.end, 1); // maxDisparity is exclusive
}

TEST(CostFunctionRangeTest, PositiveDisparitySamplesRightAtLeftXMinusDisparity)
{
    cv::Mat left(1, 6, CV_8UC1);
    cv::Mat right(1, 6, CV_8UC1, cv::Scalar(0));
    for (int x = 0; x < left.cols; ++x)
    {
        left.at<uchar>(0, x) = static_cast<uchar>(10 + 20 * x);
    }
    left(cv::Rect(2, 0, 4, 1)).copyTo(right(cv::Rect(0, 0, 4, 1)));

    const CostVolume volume = computeCostVolume(
        left,
        right,
        0,
        4,
        1,
        1,
        CostFunction::AbsoluteDifference);
    EXPECT_FLOAT_EQ(volume[2].at<float>(0, 4), 0.0f);
    EXPECT_GT(volume[1].at<float>(0, 4), 0.0f);
}

TEST(CostFunctionRangeTest, NegativeDisparitySamplesRightOfLeftPixel)
{
    cv::Mat left(1, 6, CV_8UC1);
    cv::Mat right(1, 6, CV_8UC1, cv::Scalar(0));
    for (int x = 0; x < left.cols; ++x)
    {
        left.at<uchar>(0, x) = static_cast<uchar>(10 + 20 * x);
    }
    left(cv::Rect(0, 0, 4, 1)).copyTo(right(cv::Rect(2, 0, 4, 1)));

    const CostVolume volume = computeCostVolume(
        left,
        right,
        -3,
        1,
        1,
        1,
        CostFunction::AbsoluteDifference);
    EXPECT_FLOAT_EQ(volume[1].at<float>(0, 1), 0.0f); // index 1 is disparity -2
    EXPECT_GT(volume[2].at<float>(0, 1), 0.0f);
}

TEST(CostFunctionRangeTest, InvalidHypothesesUseFiniteSentinelAndExplicitMask)
{
    const cv::Mat image(2, 5, CV_8UC1, cv::Scalar(17));
    const CostVolume volume = computeCostVolume(
        image,
        image,
        0,
        4,
        1,
        1,
        CostFunction::AbsoluteDifference);

    EXPECT_TRUE(volume.isValid(0, 0, 0));
    EXPECT_FALSE(volume.isValid(1, 0, 0));
    EXPECT_TRUE(std::isfinite(volume[1].at<float>(0, 0)));
    EXPECT_FLOAT_EQ(volume[1].at<float>(0, 0), kInvalidCost);
    EXPECT_EQ(volume.pixelValidMask().at<uchar>(0, 0), 1);
}

TEST(CostFunctionRangeTest, VolumeCarriesRangeAndDimensions)
{
    const cv::Mat image(5, 7, CV_8UC1, cv::Scalar(42));
    const CostVolume volume = computeCostVolume(
        image,
        image,
        -3,
        5,
        3,
        3,
        CostFunction::AbsoluteDifference);

    EXPECT_EQ(volume.minDisparity(), -3);
    EXPECT_EQ(volume.maxDisparity(), 5);
    ASSERT_EQ(volume.size(), 8u);
    for (const cv::Mat &slice : volume)
    {
        EXPECT_EQ(slice.size(), image.size());
    }
}

TEST(CostVolumeBufferLayoutTest, ComputesAllElementAndByteCounts)
{
    const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(7, 5, -3, 5);

    EXPECT_EQ(layout.numDisparities, 8);
    EXPECT_EQ(layout.planeElementCount, 35u);
    EXPECT_EQ(layout.imageBytes, 35u * sizeof(uchar));
    EXPECT_EQ(layout.planeBytes, 35u * sizeof(float));
    EXPECT_EQ(layout.volumeElementCount, 280u);
    EXPECT_EQ(layout.volumeBytes, 280u * sizeof(float));
}

TEST(CostVolumeBufferLayoutTest, RejectsSignedDisparitySpanOverflow)
{
    try
    {
        static_cast<void>(checkedCostVolumeBufferLayout(
            1,
            1,
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::max()));
        FAIL() << "Expected disparity span validation to fail";
    }
    catch (const std::overflow_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("disparity count"), std::string::npos);
    }
}

TEST(CostVolumeBufferLayoutTest, RejectsSizeProductOverflowWithoutAllocating)
{
    EXPECT_THROW(
        static_cast<void>(checkedCostVolumeBufferLayout(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            0,
            5)),
        std::overflow_error);
}

TEST(CostVolumeBufferLayoutTest, RejectsByteCountOverflowWithoutAllocating)
{
    const int width = std::numeric_limits<int>::max();
    const int height = sizeof(std::size_t) > sizeof(std::uint32_t)
        ? std::numeric_limits<int>::max()
        : 1;
    const int maxDisparity = sizeof(std::size_t) > sizeof(std::uint32_t) ? 2 : 1;

    try
    {
        static_cast<void>(checkedCostVolumeBufferLayout(
            width,
            height,
            0,
            maxDisparity));
        FAIL() << "Expected byte-count validation to fail";
    }
    catch (const std::overflow_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("byte count"), std::string::npos);
    }
}

TEST(CostVolumeBufferLayoutTest, RejectsInvalidDimensionsAndRange)
{
    EXPECT_THROW(
        static_cast<void>(checkedCostVolumeBufferLayout(0, 5, 0, 1)),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(checkedCostVolumeBufferLayout(5, 5, 1, 1)),
        std::invalid_argument);
}

#ifdef DM_ENABLE_CUDA
TEST(CostFunctionCudaValidationTest, RejectsExtremeDisparityRangeBeforeCudaWork)
{
    const cv::Mat image(1, 1, CV_8UC1, cv::Scalar(0));
    EXPECT_THROW(
        static_cast<void>(computeCostVolumeCUDA(
            image,
            image,
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::max(),
            1,
            1,
            CostFunction::AbsoluteDifference)),
        std::overflow_error);
}

TEST(CostFunctionCudaParityTest, CpuAndCudaAgreeForAllCostsAndValidity)
{
    if (!isCostVolumeCUDAAvailable())
    {
        GTEST_SKIP() << "CUDA device is not available";
    }

    cv::Mat left(7, 11, CV_8UC1);
    cv::Mat right(7, 11, CV_8UC1);
    cv::RNG rng(0x5a17u);
    rng.fill(left, cv::RNG::UNIFORM, 0, 256);
    rng.fill(right, cv::RNG::UNIFORM, 0, 256);
    const std::array<CostFunction, 5> functions = {
        CostFunction::AbsoluteDifference,
        CostFunction::SquaredDifference,
        CostFunction::NormalizedCrossCorr,
        CostFunction::CensusTransform,
        CostFunction::TernaryCensusTransform};

    for (const CostFunction function : functions)
    {
        SCOPED_TRACE(static_cast<int>(function));
        const CostVolume cpu = computeCostVolume(left, right, -3, 5, 5, 3, function);
        const CostVolume cuda = computeCostVolumeCUDA(left, right, -3, 5, 5, 3, function);
        ASSERT_EQ(cuda.size(), cpu.size());
        for (std::size_t disparityIndex = 0; disparityIndex < cpu.size(); ++disparityIndex)
        {
            cv::Mat validityDifference;
            cv::bitwise_xor(
                cpu.hypothesisValidMask(disparityIndex),
                cuda.hypothesisValidMask(disparityIndex),
                validityDifference);
            EXPECT_EQ(cv::countNonZero(validityDifference), 0);

            const float tolerance = function == CostFunction::NormalizedCrossCorr
                ? 1.0e-4f
                : 1.0e-5f;
            for (int y = 0; y < left.rows; ++y)
            {
                for (int x = 0; x < left.cols; ++x)
                {
                    EXPECT_NEAR(
                        cuda[disparityIndex].at<float>(y, x),
                        cpu[disparityIndex].at<float>(y, x),
                        tolerance);
                }
            }
        }
    }
}

TEST(CostFunctionCudaParityTest, FlatNccAndNoEvidenceCensusSemanticsAgree)
{
    if (!isCostVolumeCUDAAvailable())
    {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const cv::Mat flatLeft(5, 7, CV_8UC1, cv::Scalar(80));
    const cv::Mat flatRight(5, 7, CV_8UC1, cv::Scalar(120));
    const std::array<std::pair<CostFunction, int>, 3> cases = {{
        {CostFunction::NormalizedCrossCorr, 3},
        {CostFunction::CensusTransform, 1},
        {CostFunction::TernaryCensusTransform, 3}}};
    for (const auto &[function, kernelSize] : cases)
    {
        const CostVolume cpu = computeCostVolume(
            flatLeft,
            flatRight,
            0,
            2,
            kernelSize,
            kernelSize,
            function);
        const CostVolume cuda = computeCostVolumeCUDA(
            flatLeft,
            flatRight,
            0,
            2,
            kernelSize,
            kernelSize,
            function);
        for (std::size_t disparityIndex = 0; disparityIndex < cpu.size(); ++disparityIndex)
        {
            for (int y = 0; y < flatLeft.rows; ++y)
            {
                for (int x = 0; x < flatLeft.cols; ++x)
                {
                    if (function != CostFunction::NormalizedCrossCorr
                        && cpu.isValid(disparityIndex, y, x))
                    {
                        EXPECT_FLOAT_EQ(
                            cpu[disparityIndex].at<float>(y, x),
                            kInvalidCost);
                    }
                    EXPECT_NEAR(
                        cuda[disparityIndex].at<float>(y, x),
                        cpu[disparityIndex].at<float>(y, x),
                        1.0e-5f);
                }
            }
        }
    }
}
#else
TEST(CostFunctionCudaParityTest, SkippedWhenCudaBackendIsNotBuilt)
{
    GTEST_SKIP() << "dense_match CUDA backend is not built";
}
#endif
