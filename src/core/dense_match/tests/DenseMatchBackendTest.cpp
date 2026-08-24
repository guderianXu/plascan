#include <gtest/gtest.h>

#include "CostFunctions.h"
#include "DenseMatchBackend.h"
#include "DenseMatchConfig.h"
#include "DenseMatchService.h"
#include "SubpixelRefiner.h"

#include <opencv2/core.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xjw::dense_match;

namespace
{

    DisparityResult selectOnCpu(const CostVolume& volume, SubpixelMode subpixel)
    {
        DisparityResult result;
        const cv::Size size = volume[0].size();
        result.disparity = cv::Mat(size, CV_32FC1, cv::Scalar(0));
        result.confidence = cv::Mat(size, CV_32FC1, cv::Scalar(0));
        result.validMask = cv::Mat(size, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                const BestDisparity best = selectBestDisparity(volume, y, x);
                if (!best.valid)
                {
                    continue;
                }
                result.disparity.at<float>(y, x) = static_cast<float>(best.disparity);
                result.confidence.at<float>(y, x) = best.confidence;
                result.validMask.at<uchar>(y, x) = 1;
            }
        }

        DenseMatchConfig config;
        config.subpixel = subpixel;
        result.disparity = SubpixelRefiner(config).refine(
            result.disparity, volume, volume.minDisparity(), volume.maxDisparity(), result.validMask);
        return result;
    }

    CostVolume makeSelectionVolume()
    {
        CostVolume volume(-2, 5, cv::Size(12, 4));
        for (std::size_t disparityIndex = 0; disparityIndex < volume.size(); ++disparityIndex)
        {
            const float disparity = static_cast<float>(volume.minDisparity() + static_cast<int>(disparityIndex));
            for (int y = 0; y < volume[disparityIndex].rows; ++y)
            {
                for (int x = 0; x < volume[disparityIndex].cols; ++x)
                {
                    if (!volume.isValid(disparityIndex, y, x))
                    {
                        continue;
                    }
                    const float optimum = x >= 6 ? 1.25f : -0.35f;
                    const float difference = disparity - optimum;
                    volume[disparityIndex].at<float>(y, x) = difference * difference + static_cast<float>(y) * 0.001f;
                }
            }
        }
        return volume;
    }

    void expectSameSelection(const DisparityResult& expected, const DisparityResult& actual)
    {
        cv::Mat maskDifference;
        cv::bitwise_xor(expected.validMask, actual.validMask, maskDifference);
        EXPECT_EQ(cv::countNonZero(maskDifference), 0);
        EXPECT_LE(cv::norm(expected.disparity, actual.disparity, cv::NORM_INF), 1.0e-5);
        EXPECT_LE(cv::norm(expected.confidence, actual.confidence, cv::NORM_INF), 1.0e-5);
    }

} // namespace

TEST(DenseMatchBackendTest, ParsesCanonicalNamesCaseInsensitively)
{
    EXPECT_EQ(parseDenseMatchComputeBackend(" auto "), DenseMatchComputeBackend::Automatic);
    EXPECT_EQ(parseDenseMatchComputeBackend("CPU"), DenseMatchComputeBackend::Cpu);
    EXPECT_EQ(parseDenseMatchComputeBackend("Cuda"), DenseMatchComputeBackend::Cuda);
    EXPECT_EQ(parseDenseMatchComputeBackend("OPENCL"), DenseMatchComputeBackend::OpenCl);
    EXPECT_THROW(static_cast<void>(parseDenseMatchComputeBackend("vulkan")), std::invalid_argument);
}

TEST(DenseMatchBackendTest, CpuIsAlwaysAvailableAndExplicitAcceleratorsAreStrict)
{
    EXPECT_TRUE(isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::Cpu));
    EXPECT_EQ(resolveDenseMatchComputeBackend(DenseMatchComputeBackend::Cpu), DenseMatchComputeBackend::Cpu);
    EXPECT_THROW(static_cast<void>(resolveDenseMatchComputeBackend(
                     DenseMatchComputeBackend::Cuda, std::numeric_limits<int>::max(), 0)),
                 std::runtime_error);
    EXPECT_THROW(static_cast<void>(resolveDenseMatchComputeBackend(
                     DenseMatchComputeBackend::OpenCl, 0, std::numeric_limits<int>::max())),
                 std::runtime_error);
}

TEST(DenseMatchBackendTest, LegacyDisabledCudaForcesCpuOnlyForAutomaticMode)
{
    DenseMatchConfig config;
    config.computeBackend = DenseMatchComputeBackend::Automatic;
    config.useCuda = false;
    EXPECT_EQ(resolveDenseMatchComputeBackend(config), DenseMatchComputeBackend::Cpu);

    config.computeBackend = DenseMatchComputeBackend::Cpu;
    config.useCuda = true;
    EXPECT_EQ(resolveDenseMatchComputeBackend(config), DenseMatchComputeBackend::Cpu);
}

TEST(DenseMatchBackendTest, AutomaticSelectionUsesDocumentedPriority)
{
    const DenseMatchComputeBackend resolved =
        resolveDenseMatchComputeBackend(DenseMatchComputeBackend::Automatic, 0, 0);
    if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::Cuda, 0))
    {
        EXPECT_EQ(resolved, DenseMatchComputeBackend::Cuda);
    }
    else if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::OpenCl, 0))
    {
        EXPECT_EQ(resolved, DenseMatchComputeBackend::OpenCl);
    }
    else
    {
        EXPECT_EQ(resolved, DenseMatchComputeBackend::Cpu);
    }
}

TEST(DenseMatchBackendTest, RuntimeFailuresRetryCudaThenOpenClThenCpu)
{
    const std::vector<DenseMatchComputeBackend> candidates = {
        DenseMatchComputeBackend::Cuda, DenseMatchComputeBackend::OpenCl, DenseMatchComputeBackend::Cpu};
    std::vector<DenseMatchComputeBackend> attempted;
    DenseMatchExecutionReport report;
    std::string error;

    const bool succeeded = detail::runDenseMatchBackendAttempts(
        DenseMatchComputeBackend::Automatic,
        candidates,
        {},
        [&](const DenseMatchComputeBackend backend, std::string* attemptError)
        {
            attempted.push_back(backend);
            if (backend == DenseMatchComputeBackend::Cuda)
            {
                *attemptError = "cuda runtime failure";
                return false;
            }
            if (backend == DenseMatchComputeBackend::OpenCl)
            {
                *attemptError = "opencl runtime failure";
                return false;
            }
            return true;
        },
        &report,
        &error);

    EXPECT_TRUE(succeeded);
    EXPECT_EQ(attempted, candidates);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(report.requestedBackend, DenseMatchComputeBackend::Automatic);
    EXPECT_EQ(report.actualBackend, DenseMatchComputeBackend::Cpu);
    EXPECT_TRUE(report.workSubmitted);
    EXPECT_TRUE(report.fallbackUsed);
    EXPECT_NE(report.fallbackReason.find("cuda runtime failure"), std::string::npos);
    EXPECT_NE(report.fallbackReason.find("opencl runtime failure"), std::string::npos);
}

TEST(DenseMatchBackendTest, ExplicitRuntimeFailureDoesNotTryAnotherBackend)
{
    const std::vector<DenseMatchComputeBackend> candidates = {DenseMatchComputeBackend::Cuda,
                                                              DenseMatchComputeBackend::Cpu};
    std::vector<DenseMatchComputeBackend> attempted;
    DenseMatchExecutionReport report;
    std::string error;

    const bool succeeded = detail::runDenseMatchBackendAttempts(
        DenseMatchComputeBackend::Cuda,
        candidates,
        {},
        [&](const DenseMatchComputeBackend backend, std::string* attemptError)
        {
            attempted.push_back(backend);
            *attemptError = "cuda runtime failure";
            return false;
        },
        &report,
        &error);

    EXPECT_FALSE(succeeded);
    ASSERT_EQ(attempted.size(), 1U);
    EXPECT_EQ(attempted.front(), DenseMatchComputeBackend::Cuda);
    EXPECT_EQ(report.actualBackend, DenseMatchComputeBackend::Cuda);
    EXPECT_FALSE(report.fallbackUsed);
    EXPECT_TRUE(report.fallbackReason.empty());
    EXPECT_EQ(error, "cuda runtime failure");
}

TEST(DenseMatchBackendTest, ExplicitUnavailableDeviceDoesNotFallBackToCpu)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::BlockMatch;
    config.computeBackend = DenseMatchComputeBackend::OpenCl;
    config.openClDevice = std::numeric_limits<int>::max();
    config.useCuda = false;
    config.minDisparity = 0;
    config.maxDisparity = 1;
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    config.subpixel = SubpixelMode::None;
    config.medianFilterSize = 0;
    config.supportIntensityThreshold = 0;
    DenseMatchService service(config);
    const cv::Mat image(2, 2, CV_8UC1, cv::Scalar(32));

    const DisparityResult result = service.process(image, image);

    EXPECT_TRUE(result.disparity.empty());
    EXPECT_NE(service.lastError().find("unavailable"), std::string::npos);
    EXPECT_EQ(service.executionReport().requestedBackend, DenseMatchComputeBackend::OpenCl);
    EXPECT_EQ(service.executionReport().actualBackend, DenseMatchComputeBackend::OpenCl);
    EXPECT_FALSE(service.executionReport().workSubmitted);
    EXPECT_FALSE(service.executionReport().fallbackUsed);
    EXPECT_TRUE(service.executionReport().fallbackReason.empty());
}

TEST(DenseMatchBackendTest, AutomaticUnavailableAcceleratorsFallBackToCpuAndReportWhy)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::BlockMatch;
    config.computeBackend = DenseMatchComputeBackend::Automatic;
    config.useCuda = true;
    config.cudaDevice = std::numeric_limits<int>::max();
    config.openClDevice = std::numeric_limits<int>::max();
    config.minDisparity = 0;
    config.maxDisparity = 1;
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    config.subpixel = SubpixelMode::None;
    config.medianFilterSize = 0;
    config.supportIntensityThreshold = 0;
    DenseMatchService service(config);
    const cv::Mat image(2, 2, CV_8UC1, cv::Scalar(32));

    const DisparityResult result = service.process(image, image);

    ASSERT_FALSE(result.disparity.empty()) << service.lastError();
    const DenseMatchExecutionReport& report = service.executionReport();
    EXPECT_EQ(report.requestedBackend, DenseMatchComputeBackend::Automatic);
    EXPECT_EQ(report.actualBackend, DenseMatchComputeBackend::Cpu);
    EXPECT_EQ(report.deviceIndex, -1);
    EXPECT_TRUE(report.workSubmitted);
    EXPECT_TRUE(report.fallbackUsed);
    EXPECT_NE(report.fallbackReason.find("cuda"), std::string::npos);
    EXPECT_NE(report.fallbackReason.find("opencl"), std::string::npos);
    EXPECT_NE(report.fallbackReason.find("unavailable"), std::string::npos);
}

TEST(DenseMatchBackendTest, LegacyCpuOnlyAutomaticModeDoesNotClaimFallback)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::BlockMatch;
    config.computeBackend = DenseMatchComputeBackend::Automatic;
    config.useCuda = false;
    config.minDisparity = 0;
    config.maxDisparity = 1;
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    config.subpixel = SubpixelMode::None;
    config.medianFilterSize = 0;
    config.supportIntensityThreshold = 0;
    DenseMatchService service(config);
    const cv::Mat image(2, 2, CV_8UC1, cv::Scalar(32));

    const DisparityResult result = service.process(image, image);

    ASSERT_FALSE(result.disparity.empty()) << service.lastError();
    EXPECT_EQ(service.executionReport().actualBackend, DenseMatchComputeBackend::Cpu);
    EXPECT_TRUE(service.executionReport().workSubmitted);
    EXPECT_FALSE(service.executionReport().fallbackUsed);
    EXPECT_TRUE(service.executionReport().fallbackReason.empty());
}

#ifdef DM_ENABLE_CUDA
TEST(DenseMatchBackendCudaTest, WtaConfidenceAndParabolaMatchCpu)
{
    if (!isCostVolumeCUDAAvailable())
    {
        GTEST_SKIP() << "CUDA device is not available";
    }
    const CostVolume volume = makeSelectionVolume();
    expectSameSelection(selectOnCpu(volume, SubpixelMode::Parabola),
                        selectCostVolumeCUDA(volume, SubpixelMode::Parabola));
}
#else
TEST(DenseMatchBackendCudaTest, BackendIsUnavailableWhenNotBuilt)
{
    EXPECT_FALSE(isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::Cuda));
}
#endif

#ifdef DM_ENABLE_OPENCL
TEST(DenseMatchBackendOpenClTest, CostVolumeMatchesCpuForAllCostFunctions)
{
    if (!isCostVolumeOpenCLAvailable())
    {
        GTEST_SKIP() << "OpenCL GPU device is not available";
    }

    cv::Mat left(7, 11, CV_8UC1);
    cv::Mat right(7, 11, CV_8UC1);
    cv::RNG random(0x93abu);
    random.fill(left, cv::RNG::UNIFORM, 0, 256);
    random.fill(right, cv::RNG::UNIFORM, 0, 256);
    const std::array<CostFunction, 5> functions = {CostFunction::AbsoluteDifference,
                                                   CostFunction::SquaredDifference,
                                                   CostFunction::NormalizedCrossCorr,
                                                   CostFunction::CensusTransform,
                                                   CostFunction::TernaryCensusTransform};
    for (const CostFunction function : functions)
    {
        SCOPED_TRACE(static_cast<int>(function));
        const CostVolume cpu = computeCostVolume(left, right, -2, 4, 5, 3, function);
        const CostVolume openCl = computeCostVolumeOpenCL(left, right, -2, 4, 5, 3, function);
        ASSERT_EQ(openCl.size(), cpu.size());
        const double tolerance = function == CostFunction::SquaredDifference     ? 2.0e-3
                                 : function == CostFunction::NormalizedCrossCorr ? 2.0e-4
                                 : function == CostFunction::AbsoluteDifference  ? 2.0e-5
                                                                                 : 1.0e-5;
        for (std::size_t index = 0; index < cpu.size(); ++index)
        {
            EXPECT_LE(cv::norm(cpu[index], openCl[index], cv::NORM_INF), tolerance);
        }
    }
}

TEST(DenseMatchBackendOpenClTest, WtaConfidenceAndParabolaMatchCpu)
{
    if (!isCostVolumeOpenCLAvailable())
    {
        GTEST_SKIP() << "OpenCL GPU device is not available";
    }
    const CostVolume volume = makeSelectionVolume();
    expectSameSelection(selectOnCpu(volume, SubpixelMode::Parabola),
                        selectCostVolumeOpenCL(volume, SubpixelMode::Parabola));
}

TEST(DenseMatchBackendOpenClTest, ExtremeNegativeDisparityDoesNotOverflowRightCoordinate)
{
    if (!isCostVolumeOpenCLAvailable())
    {
        GTEST_SKIP() << "OpenCL GPU device is not available";
    }

    const cv::Mat image(2, 2, CV_8UC1, cv::Scalar(32));
    const int minimum = std::numeric_limits<int>::min();
    const CostVolume volume =
        computeCostVolumeOpenCL(image, image, minimum, minimum + 1, 1, 1, CostFunction::AbsoluteDifference);

    ASSERT_EQ(volume.size(), 1U);
    EXPECT_EQ(cv::countNonZero(volume.hypothesisValidMask(0)), 0);
    double minimum_cost = 0.0;
    cv::minMaxLoc(volume[0], &minimum_cost, nullptr);
    EXPECT_GE(minimum_cost, static_cast<double>(kInvalidCost));
}
#else
TEST(DenseMatchBackendOpenClTest, BackendIsUnavailableWhenNotBuilt)
{
    EXPECT_FALSE(isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::OpenCl));
}
#endif
