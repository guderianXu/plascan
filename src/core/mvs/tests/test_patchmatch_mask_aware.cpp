#include "PatchMatchCUDA.h"
#include "PatchMatchOpenCLKernels.h"
#include "PatchMatchPhotometricCost.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string>
#include <thread>
#include <vector>

namespace
{

TEST(PatchMatchOpenClKernelContractTest, RetainsQualitySamplingWithLocalReferenceTile)
{
    const std::string prefix = xjw::mvs::detail::kPatchMatchOpenClSourcePrefix;
    const std::string main = xjw::mvs::detail::kPatchMatchOpenClSourceMain;
    const std::string build_options =
        xjw::mvs::detail::kPatchMatchOpenClBuildOptions;

    EXPECT_EQ(build_options, "-cl-mad-enable");
    EXPECT_EQ(build_options.find("fast-relaxed-math"), std::string::npos);
    EXPECT_EQ(prefix.find("native_rsqrt"), std::string::npos);
    EXPECT_NE(prefix.find("sqrt(variance_product)"), std::string::npos);
    EXPECT_NE(prefix.find("reference-mask exclusions"), std::string::npos);
    EXPECT_NE(prefix.find("int step = 1;"), std::string::npos);
    EXPECT_NE(main.find("__local float reference_tile"), std::string::npos);
    EXPECT_NE(main.find("barrier(CLK_LOCAL_MEM_FENCE)"), std::string::npos);
    EXPECT_NE(main.find("coarse_samples = clamp(depth_sample_count, 16, 96)"),
              std::string::npos);
    EXPECT_NE(main.find("for (int refine_index = -6; refine_index <= 6; ++refine_index)"),
              std::string::npos);
}

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

EstimateResult estimateMaskedPlane(xjw::mvs::PatchMatchBackend backend,
                                   int opencl_device_index = -1)
{
    const cv::Mat reference = makeReferenceImage();
    const cv::Mat source = shiftLeft(reference);
    const cv::Mat reference_mask = makeReferenceMask();
    const cv::Mat source_mask = shiftLeft(reference_mask);

    xjw::mvs::PatchMatchConfig config;
    config.backend = backend;
    config.useCuda = backend == xjw::mvs::PatchMatchBackend::Cuda;
    config.cudaFallbackToCpu = false;
    config.openClFallbackToCpu = false;
    config.openClDeviceIndex = opencl_device_index;
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
    const EstimateResult result = estimateMaskedPlane(xjw::mvs::PatchMatchBackend::Cpu);
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

    const EstimateResult cpu = estimateMaskedPlane(xjw::mvs::PatchMatchBackend::Cpu);
    const EstimateResult gpu = estimateMaskedPlane(xjw::mvs::PatchMatchBackend::Cuda);
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

TEST(PatchMatchMaskAwareTest, OpenClEstimatesMaskedPlaneWhenAvailable)
{
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL GPU is unavailable";
    }

    for (const xjw::mvs::OpenClDeviceInfo &device : devices)
    {
        SCOPED_TRACE(device.vendor + " " + device.name);
        const EstimateResult result = estimateMaskedPlane(
            xjw::mvs::PatchMatchBackend::OpenCl, device.index);
        ASSERT_FALSE(result.depth.empty());
        ASSERT_EQ(result.depth.type(), CV_32F);

        const cv::Mat reference_mask = makeReferenceMask();
        EXPECT_GT(validRatio(result.depth, reference_mask), 0.25);

        cv::Mat outside_mask;
        cv::bitwise_not(reference_mask, outside_mask);
        cv::Mat outside_depth;
        result.depth.copyTo(outside_depth, outside_mask);
        EXPECT_EQ(cv::countNonZero(outside_depth > 0.0f), 0);

        std::vector<float> valid_depths;
        for (int row = 0; row < result.depth.rows; ++row)
        {
            for (int column = 0; column < result.depth.cols; ++column)
            {
                const float depth = result.depth.at<float>(row, column);
                if (reference_mask.at<std::uint8_t>(row, column) != 0 && depth > 0.0f)
                {
                    valid_depths.push_back(depth);
                }
            }
        }
        ASSERT_FALSE(valid_depths.empty());
        const auto median = valid_depths.begin() + valid_depths.size() / 2;
        std::nth_element(valid_depths.begin(), median, valid_depths.end());
        EXPECT_NEAR(*median, kExpectedDepth, 1.0f);
    }

    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchMaskAwareTest, OpenClConcurrentWorkersSerializeAndReuseCachedInputs)
{
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL GPU is unavailable";
    }

    for (const xjw::mvs::OpenClDeviceInfo &device : devices)
    {
        SCOPED_TRACE(device.vendor + " " + device.name);
        std::array<EstimateResult, 6> results;
        for (std::size_t round = 0; round < 2; ++round)
        {
            std::array<std::thread, 3> workers;
            for (std::size_t worker_index = 0; worker_index < workers.size(); ++worker_index)
            {
                workers[worker_index] = std::thread([&, round, worker_index]()
                {
                    results[round * workers.size() + worker_index] = estimateMaskedPlane(
                        xjw::mvs::PatchMatchBackend::OpenCl, device.index);
                });
            }
            for (std::thread &worker : workers)
            {
                worker.join();
            }
        }

        const cv::Mat reference_mask = makeReferenceMask();
        for (const EstimateResult &result : results)
        {
            ASSERT_FALSE(result.depth.empty());
            EXPECT_GT(validRatio(result.depth, reference_mask), 0.25);
        }
    }
    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchPhotometricUniquenessTest, KeepsDistinctDepthHypothesisAtFullConfidence)
{
    EXPECT_FLOAT_EQ(
        xjw::mvs::photometricUniquenessConfidenceScale(
            0.92f, 0.80f, 0.03f, 0.50f),
        1.0f);
}

TEST(PatchMatchPhotometricUniquenessTest, SoftlyPenalizesAmbiguousDepthHypothesis)
{
    EXPECT_NEAR(
        xjw::mvs::photometricUniquenessConfidenceScale(
            0.92f, 0.91f, 0.03f, 0.50f),
        2.0f / 3.0f,
        1e-5f);
    EXPECT_FLOAT_EQ(
        xjw::mvs::photometricUniquenessConfidenceScale(
            0.90f, 0.90f, 0.03f, 0.50f),
        0.50f);
}

TEST(PatchMatchPhotometricUniquenessTest, InvalidMarginLeavesConfidenceUnchanged)
{
    EXPECT_FLOAT_EQ(
        xjw::mvs::photometricUniquenessConfidenceScale(
            0.50f, 0.50f, 0.0f, 0.25f),
        1.0f);
}

} // namespace
