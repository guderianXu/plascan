#include "PatchMatchCUDA.h"
#include "PatchMatchOpenCLKernels.h"
#include "PatchMatchPhotometricCost.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace
{

TEST(PatchMatchOpenClKernelContractTest, RetainsQualitySamplingWithLocalReferenceTile)
{
    const std::string prefix = xjw::mvs::detail::kPatchMatchOpenClSourcePrefix;
    const std::string main = xjw::mvs::detail::kPatchMatchOpenClSourceMain;
    const std::string propagation =
        xjw::mvs::detail::kPatchMatchOpenClSourcePropagation;
    const std::string finalize =
        xjw::mvs::detail::kPatchMatchOpenClSourceFinalize;
    const std::string build_options =
        xjw::mvs::detail::kPatchMatchOpenClBuildOptions;

    EXPECT_EQ(build_options, "-cl-mad-enable");
    EXPECT_EQ(build_options.find("fast-relaxed-math"), std::string::npos);
    EXPECT_EQ(prefix.find("native_rsqrt"), std::string::npos);
    EXPECT_NE(prefix.find("sqrt(variance_product)"), std::string::npos);
    EXPECT_NE(prefix.find("reference-mask exclusions"), std::string::npos);
    EXPECT_NE(prefix.find("int step = clamp(patch_step, 1, 3);"),
              std::string::npos);
    EXPECT_NE(prefix.find("float plane_distance = depth"), std::string::npos);
    EXPECT_NE(prefix.find("compose_plane_homography"), std::string::npos);
    EXPECT_NE(prefix.find("projected_x += projected_x_step"), std::string::npos);
    EXPECT_EQ(prefix.find("local_depth_sample_count"), std::string::npos);
    EXPECT_NE(prefix.find("propagated_plane_depth"), std::string::npos);
    EXPECT_EQ(main.find("float scores[MAX_SOURCES]"), std::string::npos);
    EXPECT_NE(main.find("float strongest[MAX_ROBUST_SUPPORT]"), std::string::npos);
    EXPECT_NE(main.find("__local float reference_tile"), std::string::npos);
    EXPECT_NE(main.find("barrier(CLK_LOCAL_MEM_FENCE)"), std::string::npos);
    EXPECT_NE(main.find("coarse_samples = clamp(depth_sample_count, 16, 96)"),
              std::string::npos);
    EXPECT_NE(main.find("for (int refine_index = -6; refine_index <= 6; ++refine_index)"),
              std::string::npos);
    EXPECT_NE(main.find("source_count, patch_half, 1"), std::string::npos);
    EXPECT_NE(main.find("__kernel void initialize_planes"), std::string::npos);
    EXPECT_NE(main.find("0.05f * hint_depth[index]"), std::string::npos);
    EXPECT_NE(main.find("__global float4 *normal_output"), std::string::npos);
    EXPECT_NE(propagation.find("__kernel void propagate_planes"), std::string::npos);
    EXPECT_NE(propagation.find("compact_x * 2 + ((y + checkerboard) & 1)"),
              std::string::npos);
    EXPECT_NE(propagation.find("CHECKERBOARD_TILE_WIDTH"), std::string::npos);
    EXPECT_NE(propagation.find("same_plane_hypothesis"), std::string::npos);
    EXPECT_NE(propagation.find("random_facing_normal"), std::string::npos);
    EXPECT_NE(propagation.find("float normal_only_score"), std::string::npos);
    EXPECT_NE(propagation.find("0.05f * hint_depth[index]"), std::string::npos);
    EXPECT_NE(propagation.find("source_count, patch_half, 1"), std::string::npos);
    EXPECT_NE(finalize.find("__kernel void finalize_planes"), std::string::npos);
    EXPECT_EQ(finalize.find("0.05f * hint_depth[index]"), std::string::npos);
    EXPECT_EQ(finalize.find("source_count >= 3"), std::string::npos);
    EXPECT_NE(finalize.find("uniqueness_minimum_margin"), std::string::npos);
    EXPECT_NE(finalize.find("best_depth * uniqueness_relative_step * 0.25f"),
              std::string::npos);
    EXPECT_NE(finalize.find("best_depth - lower_depth >= minimum_distinct_depth"),
              std::string::npos);
    EXPECT_NE(finalize.find("upper_depth - best_depth >= minimum_distinct_depth"),
              std::string::npos);
    EXPECT_EQ(finalize.find("local_near, local_far"), std::string::npos);
    EXPECT_NE(finalize.find("source_count, patch_half, 1"), std::string::npos);
    EXPECT_NE(finalize.find("float confidence = best_score"), std::string::npos);
    EXPECT_EQ(finalize.find("best_score = robust_depth_score"), std::string::npos);
}

TEST(PatchMatchBackendAlignmentTest, UsesOneSharedSearchBudget)
{
    EXPECT_EQ(xjw::mvs::patchMatchDepthSampleCount(1), 32);
    EXPECT_EQ(xjw::mvs::patchMatchDepthSampleCount(16), 64);
    EXPECT_EQ(xjw::mvs::patchMatchDepthSampleCount(40), 96);
    EXPECT_EQ(xjw::mvs::patchMatchPropagationIterationCount(1), 2);
    EXPECT_EQ(xjw::mvs::patchMatchPropagationIterationCount(16), 4);
    EXPECT_EQ(xjw::mvs::patchMatchPropagationIterationCount(40), 4);
    EXPECT_FLOAT_EQ(xjw::mvs::kDefaultPatchMatchHintRadiusRatio, 0.05f);
}

// Odd dimensions exercise the compact checkerboard tail work-group on both axes.
constexpr int kWidth = 65;
constexpr int kHeight = 49;
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
                                   int opencl_device_index = -1,
                                   bool include_optional_inputs = true,
                                   int downsample_factor = 1,
                                   bool return_native_resolution = false,
                                   float hint_center = kExpectedDepth,
                                   float hint_radius = 0.25f)
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
    config.downsampleFactor = downsample_factor;
    config.returnNativeResolution = return_native_resolution;
    config.numIterations = 2;
    config.patchHalf = 2;
    config.confidenceThresh = 0.05f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;

    cv::Mat hint(kHeight, kWidth, CV_32F, cv::Scalar(hint_center));
    cv::Mat radius(kHeight, kWidth, CV_32F, cv::Scalar(hint_radius));
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
        include_optional_inputs ? &hint : nullptr,
        include_optional_inputs ? &radius : nullptr,
        include_optional_inputs ? &reference_mask : nullptr,
        include_optional_inputs ? &source_masks : nullptr)) << error;
    return result;
}

double validRatio(const cv::Mat &depth, const cv::Mat &mask)
{
    cv::Mat valid = depth > 0.0f;
    cv::bitwise_and(valid, mask, valid);
    return static_cast<double>(cv::countNonZero(valid)) /
           static_cast<double>(std::max(1, cv::countNonZero(mask)));
}

double medianRelativeDepthDifference(const cv::Mat &lhs,
                                     const cv::Mat &rhs,
                                     const cv::Mat &mask)
{
    std::vector<float> differences;
    for (int row = 0; row < lhs.rows; ++row)
    {
        for (int column = 0; column < lhs.cols; ++column)
        {
            const float lhs_depth = lhs.at<float>(row, column);
            const float rhs_depth = rhs.at<float>(row, column);
            if (mask.at<std::uint8_t>(row, column) == 0 ||
                lhs_depth <= 0.0f || rhs_depth <= 0.0f)
            {
                continue;
            }
            differences.push_back(
                std::abs(lhs_depth - rhs_depth) /
                std::max(1e-6f, 0.5f * (lhs_depth + rhs_depth)));
        }
    }

    if (differences.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    const auto median = differences.begin() + differences.size() / 2;
    std::nth_element(differences.begin(), median, differences.end());
    return *median;
}

double medianAbsoluteConfidenceDifference(const EstimateResult &lhs,
                                          const EstimateResult &rhs,
                                          const cv::Mat &mask)
{
    std::vector<float> differences;
    for (int row = 0; row < lhs.depth.rows; ++row)
    {
        for (int column = 0; column < lhs.depth.cols; ++column)
        {
            if (mask.at<std::uint8_t>(row, column) == 0 ||
                lhs.depth.at<float>(row, column) <= 0.0f ||
                rhs.depth.at<float>(row, column) <= 0.0f)
            {
                continue;
            }
            differences.push_back(std::abs(
                lhs.confidence.at<float>(row, column) -
                rhs.confidence.at<float>(row, column)));
        }
    }
    if (differences.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    const auto median = differences.begin() + differences.size() / 2;
    std::nth_element(differences.begin(), median, differences.end());
    return *median;
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

TEST(PatchMatchMaskAwareTest, OpenClSupportsNativeResolutionWithoutOptionalInputs)
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
            xjw::mvs::PatchMatchBackend::OpenCl,
            device.index,
            false,
            2,
            true);
        ASSERT_FALSE(result.depth.empty());
        EXPECT_EQ(result.depth.size(), cv::Size(kWidth / 2, kHeight / 2));
        EXPECT_EQ(result.confidence.size(), result.depth.size());
    }

    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchBackendAlignmentTest, CudaAndOpenClConvergeOnSameMaskedPlaneWhenAvailable)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA is unavailable";
    }
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL GPU is unavailable";
    }

    const EstimateResult cuda = estimateMaskedPlane(xjw::mvs::PatchMatchBackend::Cuda);
    ASSERT_FALSE(cuda.depth.empty());
    const cv::Mat reference_mask = makeReferenceMask();
    const double cuda_ratio = validRatio(cuda.depth, reference_mask);
    for (const xjw::mvs::OpenClDeviceInfo &device : devices)
    {
        SCOPED_TRACE(device.vendor + " " + device.name);
        const EstimateResult opencl = estimateMaskedPlane(
            xjw::mvs::PatchMatchBackend::OpenCl, device.index);
        ASSERT_FALSE(opencl.depth.empty());
        EXPECT_NEAR(validRatio(opencl.depth, reference_mask), cuda_ratio, 0.02);
        EXPECT_LT(
            medianRelativeDepthDifference(cuda.depth, opencl.depth, reference_mask),
            0.02);
    }
    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchBackendAlignmentTest, CudaAndOpenClAlignAtHintUpperBoundaryWhenAvailable)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA is unavailable";
    }
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL GPU is unavailable";
    }

    // The true depth is exactly the upper propagated-prior boundary. The
    // uniqueness probes must still use the global search interval on every
    // backend instead of comparing the best plane with a clamped duplicate.
    const EstimateResult cuda = estimateMaskedPlane(
        xjw::mvs::PatchMatchBackend::Cuda,
        -1,
        true,
        1,
        false,
        9.75f,
        0.25f);
    ASSERT_FALSE(cuda.depth.empty());
    const cv::Mat reference_mask = makeReferenceMask();
    const double cuda_ratio = validRatio(cuda.depth, reference_mask);
    for (const xjw::mvs::OpenClDeviceInfo &device : devices)
    {
        SCOPED_TRACE(device.vendor + " " + device.name);
        const EstimateResult opencl = estimateMaskedPlane(
            xjw::mvs::PatchMatchBackend::OpenCl,
            device.index,
            true,
            1,
            false,
            9.75f,
            0.25f);
        ASSERT_FALSE(opencl.depth.empty());
        EXPECT_NEAR(validRatio(opencl.depth, reference_mask), cuda_ratio, 0.02);
        EXPECT_LT(
            medianRelativeDepthDifference(cuda.depth, opencl.depth, reference_mask),
            0.02);
        EXPECT_LT(
            medianAbsoluteConfidenceDifference(cuda, opencl, reference_mask),
            0.10);
    }
    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchMaskAwareTest, OpenClConcurrentWorkersPipelineAndReuseCachedInputs)
{
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL GPU is unavailable";
    }

    xjw::mvs::PatchMatchDepthEstimator::resetOpenClExecutionStats();
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
    const std::vector<xjw::mvs::OpenClExecutionStats> execution_stats =
        xjw::mvs::PatchMatchDepthEstimator::openClExecutionStats();
    ASSERT_EQ(execution_stats.size(), devices.size());
    for (const xjw::mvs::OpenClExecutionStats &stats : execution_stats)
    {
        EXPECT_GE(stats.callCount, 6U);
        EXPECT_GT(stats.wallSpanMilliseconds, 0.0);
        EXPECT_GT(stats.kernelActiveMilliseconds, 0.0);
        EXPECT_GE(stats.interCallIdleMilliseconds, 0.0);
        EXPECT_GE(stats.queueNonKernelMilliseconds, 0.0);
        EXPECT_GE(stats.queueOccupancyRatio, stats.kernelDutyRatio);
        EXPECT_GT(stats.kernelDutyRatio, 0.0);
        EXPECT_LE(stats.queueOccupancyRatio, 1.0);
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
