#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "PatchMatchHostUtils.h"
#include "FramePinholeCamera.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <thread>

namespace
{

struct CudaPatchMatchRunStats
{
    int validCount = 0;
    int roiPixelCount = 0;
    double meanDepth = 0.0;
    double meanConfidence = 0.0;
    double depthRmse = 0.0;
    double elapsedMs = 0.0;
};

xjw::FramePinholeCamera makeCamera(double fu,
                                   double fv,
                                   double cu,
                                   double cv,
                                   int uDir,
                                   int vDir,
                                   const double r_wc[9],
                                   const double center[3],
                                   bool depthAxisFlipped)
{
    xjw::FramePinholeCamera camera;
    std::array<double, 9> rotation{{r_wc[0], r_wc[1], r_wc[2], r_wc[3], r_wc[4], r_wc[5], r_wc[6], r_wc[7], r_wc[8]}};
    std::array<double, 3> cameraCenter{{center[0], center[1], center[2]}};
    camera.setIntrinsics(fu, fv, cu, cv);
    camera.setPose(rotation, cameraCenter);
    camera.setAxisDirections(uDir, vDir);
    camera.setDepthAxisFlipped(depthAxisFlipped);
    return camera;
}

cv::Mat makeTexturedImage(int width, int height)
{
    cv::Mat image(height, width, CV_8U);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double value = 128.0 + 42.0 * std::sin(0.19 * static_cast<double>(x)) +
                                 37.0 * std::cos(0.23 * static_cast<double>(y)) +
                                 (((x / 6) + (y / 5)) % 2 == 0 ? 18.0 : -18.0);
            image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(value, 0.0, 255.0));
        }
    }
    return image;
}

cv::Mat makeShiftedImage(const cv::Mat& image, int disparity)
{
    cv::Mat shifted(image.rows, image.cols, image.type(), cv::Scalar::all(0));
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const int sampleX = x + disparity;
            if (sampleX >= 0 && sampleX < image.cols)
            {
                shifted.at<uint8_t>(y, x) = image.at<uint8_t>(y, sampleX);
            }
        }
    }
    return shifted;
}

CudaPatchMatchRunStats executeCudaPatchMatchCase(const cv::Mat& refGray,
                                                 const cv::Mat& srcGray,
                                                 const xjw::FramePinholeCamera& refCam,
                                                 const xjw::FramePinholeCamera& srcCam,
                                                 bool useParallelSweep,
                                                 int iterations,
                                                 double expectedDepth = 10.0)
{
    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cuda;
    config.cudaUseParallelSweep = useParallelSweep;
    config.downsampleFactor = 2;
    config.numIterations = iterations;
    config.patchHalf = 2;
    config.confidenceThresh = 0.05f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;

    cv::Mat depthMap;
    cv::Mat confidenceMap;
    std::string errorMessage;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(refGray,
                                                                 std::vector<cv::Mat>{srcGray},
                                                                 refCam,
                                                                 std::vector<xjw::FramePinholeCamera>{srcCam},
                                                                 5.0f,
                                                                 15.0f,
                                                                 config,
                                                                 depthMap,
                                                                 &confidenceMap,
                                                                 &errorMessage,
                                                                 nullptr);
    const auto stop = std::chrono::steady_clock::now();

    EXPECT_TRUE(ok) << errorMessage;
    if (!ok)
    {
        return {};
    }
    EXPECT_EQ(depthMap.size(), refGray.size());
    EXPECT_EQ(confidenceMap.size(), refGray.size());

    const cv::Rect roi(refGray.cols / 6, refGray.rows / 6, refGray.cols * 2 / 3, refGray.rows * 2 / 3);
    int validCount = 0;
    double depthSum = 0.0;
    double confidenceSum = 0.0;
    double squaredDepthErrorSum = 0.0;
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
        for (int x = roi.x; x < roi.x + roi.width; ++x)
        {
            const float depth = depthMap.at<float>(y, x);
            if (depth > 0.0f)
            {
                ++validCount;
                depthSum += depth;
                confidenceSum += confidenceMap.at<float>(y, x);
                const double depthError = static_cast<double>(depth) - expectedDepth;
                squaredDepthErrorSum += depthError * depthError;
            }
        }
    }

    CudaPatchMatchRunStats stats;
    stats.validCount = validCount;
    stats.roiPixelCount = roi.area();
    if (validCount > 0)
    {
        stats.meanDepth = depthSum / static_cast<double>(validCount);
        stats.meanConfidence = confidenceSum / static_cast<double>(validCount);
        stats.depthRmse = std::sqrt(squaredDepthErrorSum / static_cast<double>(validCount));
    }
    stats.elapsedMs = std::chrono::duration<double, std::milli>(stop - start).count();
    return stats;
}

} // namespace

TEST(PatchMatchCpuRegressionTest, RecoversFrontoParallelPlaneAtExpectedDepth)
{
    constexpr int IMAGE_WIDTH = 96;
    constexpr int IMAGE_HEIGHT = 72;
    constexpr float EXPECTED_DEPTH = 10.0f;
    constexpr double FOCAL = 80.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 8;

    cv::Mat refGray = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::Mat srcGray = makeShiftedImage(refGray, DISPARITY);
    cv::GaussianBlur(srcGray, srcGray, cv::Size(3, 3), 0.0);

    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};

    const auto refCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, refCenter, false)
            .normalizedForPositiveDepth();
    const auto srcCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, srcCenter, false)
            .normalizedForPositiveDepth();

    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.downsampleFactor = 2;
    config.patchHalf = 2;
    config.confidenceThresh = 0.15f;
    config.doMedianBlur = true;
    config.medianKernelSize = 3;
    config.doBilateralFilter = true;
    config.bilateralD = 5;
    config.bilateralSigmaColor = 25.0f;
    config.bilateralSigmaSpace = 3.0f;

    cv::Mat depthMap;
    cv::Mat confidenceMap;
    std::string errorMessage;
    const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(refGray,
                                                                 std::vector<cv::Mat>{srcGray},
                                                                 refCam,
                                                                 std::vector<xjw::FramePinholeCamera>{srcCam},
                                                                 5.0f,
                                                                 15.0f,
                                                                 config,
                                                                 depthMap,
                                                                 &confidenceMap,
                                                                 &errorMessage,
                                                                 nullptr);

    ASSERT_TRUE(ok) << errorMessage;
    ASSERT_EQ(depthMap.size(), refGray.size());
    ASSERT_EQ(confidenceMap.size(), refGray.size());

    const cv::Rect roi(16, 12, IMAGE_WIDTH - 32, IMAGE_HEIGHT - 24);
    int validCount = 0;
    double depthSum = 0.0;
    double confSum = 0.0;
    const int pixelCount = roi.width * roi.height;

    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
        for (int x = roi.x; x < roi.x + roi.width; ++x)
        {
            const float depth = depthMap.at<float>(y, x);
            if (depth > 0.0f)
            {
                ++validCount;
                depthSum += depth;
                confSum += confidenceMap.at<float>(y, x);
            }
        }
    }

    ASSERT_GT(validCount, pixelCount * 0.55) << "CPU PatchMatch should recover most planar pixels";

    const double meanDepth = depthSum / static_cast<double>(validCount);
    const double meanConfidence = confSum / static_cast<double>(validCount);

    EXPECT_NEAR(meanDepth, EXPECTED_DEPTH, 1.0) << "Recovered CPU depth should stay close to the expected plane depth";
    EXPECT_GT(meanConfidence, 0.35) << "Recovered CPU depth should have meaningful confidence";
}

TEST(PatchMatchCpuCancellationTest, StopsPromptlyInsideLongPixelSweeps)
{
    constexpr int IMAGE_WIDTH = 640;
    constexpr int IMAGE_HEIGHT = 480;
    constexpr double FOCAL = 520.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 12;

    const cv::Mat reference = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    const cv::Mat source = makeShiftedImage(reference, DISPARITY);
    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double reference_center[3] = {0.0, 0.0, 0.0};
    const double source_center[3] = {BASELINE, 0.0, 0.0};
    const auto reference_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, reference_center, false)
            .normalizedForPositiveDepth();
    const auto source_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, source_center, false)
            .normalizedForPositiveDepth();

    std::atomic_bool cancelled{false};
    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.downsampleFactor = 1;
    config.numIterations = 8;
    config.patchHalf = 3;
    config.cpuThreadCount = 4;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;
    config.cancelFlag = &cancelled;

    auto future = std::async(std::launch::async,
                             [&]()
                             {
                                 cv::Mat depth;
                                 std::string error;
                                 const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
                                     reference,
                                     std::vector<cv::Mat>{source},
                                     reference_camera,
                                     std::vector<xjw::FramePinholeCamera>{source_camera},
                                     5.0f,
                                     15.0f,
                                     config,
                                     depth,
                                     nullptr,
                                     &error,
                                     nullptr);
                                 return std::make_pair(ok, error);
                             });

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    const auto cancel_started = std::chrono::steady_clock::now();
    cancelled.store(true, std::memory_order_relaxed);

    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    const auto [ok, error] = future.get();
    const auto cancellation_latency = std::chrono::steady_clock::now() - cancel_started;
    EXPECT_FALSE(ok);
    EXPECT_NE(error.find("cancelled"), std::string::npos);
    EXPECT_LT(cancellation_latency, std::chrono::seconds(2));
}

TEST(PatchMatchCudaCancellationTest, StopsPromptlyAtTiledKernelCheckpoints)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    constexpr int IMAGE_WIDTH = 1280;
    constexpr int IMAGE_HEIGHT = 960;
    constexpr double FOCAL = 1040.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 52;

    const cv::Mat reference = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    const cv::Mat source = makeShiftedImage(reference, DISPARITY);
    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double reference_center[3] = {0.0, 0.0, 0.0};
    const double source_center[3] = {BASELINE, 0.0, 0.0};
    const auto reference_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, reference_center, false)
            .normalizedForPositiveDepth();
    const auto source_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, source_center, false)
            .normalizedForPositiveDepth();

    std::atomic_bool cancelled{false};
    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cuda;
    config.downsampleFactor = 1;
    config.numIterations = 16;
    config.patchHalf = 7;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;
    config.cancelFlag = &cancelled;

    auto future = std::async(std::launch::async,
                             [&]()
                             {
                                 cv::Mat depth;
                                 std::string error;
                                 const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
                                     reference,
                                     std::vector<cv::Mat>{source},
                                     reference_camera,
                                     std::vector<xjw::FramePinholeCamera>{source_camera},
                                     5.0f,
                                     30.0f,
                                     config,
                                     depth,
                                     nullptr,
                                     &error,
                                     nullptr);
                                 return std::make_pair(ok, error);
                             });

    std::this_thread::sleep_for(std::chrono::milliseconds(75));
    const auto cancel_started = std::chrono::steady_clock::now();
    cancelled.store(true, std::memory_order_relaxed);

    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    const auto [ok, error] = future.get();
    const auto cancellation_latency = std::chrono::steady_clock::now() - cancel_started;
    EXPECT_FALSE(ok);
    EXPECT_NE(error.find("cancelled"), std::string::npos);
    EXPECT_LT(cancellation_latency, std::chrono::seconds(2));
}

TEST(PatchMatchOpenClCancellationTest, StopsPromptlyAtTiledKernelCheckpoints)
{
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices = xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    if (devices.empty())
    {
        GTEST_SKIP() << "OpenCL PatchMatch is not available in this environment";
    }

    constexpr int IMAGE_WIDTH = 640;
    constexpr int IMAGE_HEIGHT = 480;
    constexpr double FOCAL = 520.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 26;

    const cv::Mat reference = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    const cv::Mat source = makeShiftedImage(reference, DISPARITY);
    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double reference_center[3] = {0.0, 0.0, 0.0};
    const double source_center[3] = {BASELINE, 0.0, 0.0};
    const auto reference_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, reference_center, false)
            .normalizedForPositiveDepth();
    const auto source_camera =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, source_center, false)
            .normalizedForPositiveDepth();

    for (const xjw::mvs::OpenClDeviceInfo& device : devices)
    {
        SCOPED_TRACE(device.vendor + " " + device.name);
        std::string preparation_error;
        ASSERT_TRUE(xjw::mvs::PatchMatchDepthEstimator::prepareOpenClDevice(device.index, &preparation_error))
            << preparation_error;

        std::atomic_bool cancelled{false};
        xjw::mvs::PatchMatchConfig config;
        config.backend = xjw::mvs::PatchMatchBackend::OpenCl;
        config.openClDeviceIndex = device.index;
        config.downsampleFactor = 1;
        config.numIterations = 16;
        config.patchHalf = 7;
        config.doMedianBlur = false;
        config.doBilateralFilter = false;
        config.cancelFlag = &cancelled;

        auto future = std::async(std::launch::async,
                                 [&]()
                                 {
                                     cv::Mat depth;
                                     std::string error;
                                     const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
                                         reference,
                                         std::vector<cv::Mat>{source},
                                         reference_camera,
                                         std::vector<xjw::FramePinholeCamera>{source_camera},
                                         5.0f,
                                         30.0f,
                                         config,
                                         depth,
                                         nullptr,
                                         &error,
                                         nullptr);
                                     return std::make_pair(ok, error);
                                 });

        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        const auto cancel_started = std::chrono::steady_clock::now();
        cancelled.store(true, std::memory_order_relaxed);

        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        const auto [ok, error] = future.get();
        const auto cancellation_latency = std::chrono::steady_clock::now() - cancel_started;
        EXPECT_FALSE(ok);
        EXPECT_NE(error.find("cancelled"), std::string::npos);
        EXPECT_LT(cancellation_latency, std::chrono::seconds(2));
    }
    xjw::mvs::PatchMatchDepthEstimator::cleanupOpenClResources();
}

TEST(PatchMatchCpuRegressionTest, FrozenGeometryGuidanceEmitsSeparatePhotometricSourceMask)
{
    constexpr int width = 80;
    constexpr int height = 60;
    constexpr float expected_depth = 10.0f;
    constexpr double focal = 80.0;
    constexpr double baseline = 1.0;
    constexpr int disparity = 8;
    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double reference_center[3] = {0.0, 0.0, 0.0};
    const double source_center[3] = {baseline, 0.0, 0.0};
    const xjw::FramePinholeCamera reference_camera =
        makeCamera(focal, focal, width * 0.5, height * 0.5, 1, 1, identity, reference_center, false)
            .normalizedForPositiveDepth();
    const xjw::FramePinholeCamera source_camera =
        makeCamera(focal, focal, width * 0.5, height * 0.5, 1, 1, identity, source_center, false)
            .normalizedForPositiveDepth();
    const cv::Mat reference = makeTexturedImage(width, height);
    const cv::Mat source = makeShiftedImage(reference, disparity);

    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.downsampleFactor = 1;
    config.numIterations = 2;
    config.patchHalf = 2;
    config.confidenceThresh = 0.05f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;
    cv::Mat hint(height, width, CV_32F, cv::Scalar(expected_depth));
    cv::Mat hint_radius(height, width, CV_32F, cv::Scalar(0.3f));
    cv::Mat frozen_source_depth(height, width, CV_32F, cv::Scalar(expected_depth));
    const std::vector<cv::Mat> source_depths{frozen_source_depth};
    xjw::mvs::PatchMatchAuxiliaryInput auxiliary_input;
    auxiliary_input.sourceDepthMaps = &source_depths;
    cv::Mat source_selection;
    xjw::mvs::PatchMatchAuxiliaryOutput auxiliary_output;
    auxiliary_output.photometricSourceMask = &source_selection;

    cv::Mat guided_depth;
    cv::Mat guided_confidence;
    std::string error;
    ASSERT_TRUE(xjw::mvs::PatchMatchDepthEstimator::estimate(reference,
                                                             std::vector<cv::Mat>{source},
                                                             reference_camera,
                                                             std::vector<xjw::FramePinholeCamera>{source_camera},
                                                             5.0f,
                                                             15.0f,
                                                             config,
                                                             guided_depth,
                                                             &guided_confidence,
                                                             &error,
                                                             &hint,
                                                             &hint_radius,
                                                             nullptr,
                                                             nullptr,
                                                             &auxiliary_input,
                                                             &auxiliary_output))
        << error;

    ASSERT_EQ(source_selection.type(), CV_32SC1);
    ASSERT_EQ(source_selection.size(), guided_depth.size());
    const cv::Mat valid = guided_depth > 0.0f;
    ASSERT_GT(cv::countNonZero(valid), width * height / 3);
    for (int row = 0; row < height; ++row)
    {
        const float* depth_row = guided_depth.ptr<float>(row);
        const float* confidence_row = guided_confidence.ptr<float>(row);
        const std::int32_t* selection_row = source_selection.ptr<std::int32_t>(row);
        for (int column = 0; column < width; ++column)
        {
            if (!(depth_row[column] > 0.0f))
            {
                EXPECT_EQ(selection_row[column], 0);
                continue;
            }
            EXPECT_EQ(selection_row[column], 1);
            EXPECT_GE(confidence_row[column], 0.0f);
            EXPECT_LE(confidence_row[column], 1.0f);
        }
    }
}

TEST(PatchMatchHostCameraDataTest, PreservesRelativePoseAfterLargeCommonWorldTranslation)
{
    constexpr double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {1.0, -0.25, 0.5};
    constexpr double translated_reference_center[3] = {100000000.0, -200000000.0, 300000000.0};
    constexpr double translated_source_center[3] = {100000001.0, -200000000.25, 300000000.5};

    const auto reference =
        makeCamera(800.0, 790.0, 320.0, 240.0, 1, 1, identity, reference_center, false).normalizedForPositiveDepth();
    const auto source =
        makeCamera(810.0, 805.0, 318.0, 242.0, 1, 1, identity, source_center, false).normalizedForPositiveDepth();
    const auto translated_reference =
        makeCamera(800.0, 790.0, 320.0, 240.0, 1, 1, identity, translated_reference_center, false)
            .normalizedForPositiveDepth();
    const auto translated_source =
        makeCamera(810.0, 805.0, 318.0, 242.0, 1, 1, identity, translated_source_center, false)
            .normalizedForPositiveDepth();

    const auto local_data = xjw::mvs::buildPatchMatchSourceCameraData(reference, source, 2);
    const auto translated_data = xjw::mvs::buildPatchMatchSourceCameraData(translated_reference, translated_source, 2);

    for (std::size_t index = 0; index < local_data.size(); ++index)
    {
        EXPECT_FLOAT_EQ(translated_data[index], local_data[index]) << "camera data element " << index;
    }
    EXPECT_FLOAT_EQ(local_data[13], -1.0f);
    EXPECT_FLOAT_EQ(local_data[14], 0.25f);
    EXPECT_FLOAT_EQ(local_data[15], -0.5f);
}

TEST(PatchMatchCpuRegressionTest, DepthIsInvariantToLargeCommonWorldTranslation)
{
    constexpr int image_width = 96;
    constexpr int image_height = 72;
    constexpr double focal = 80.0;
    constexpr int disparity = 8;
    constexpr double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    constexpr double reference_center[3] = {0.0, 0.0, 0.0};
    constexpr double source_center[3] = {1.0, 0.0, 0.0};
    constexpr double translated_reference_center[3] = {100000000.0, -200000000.0, 300000000.0};
    constexpr double translated_source_center[3] = {100000001.0, -200000000.0, 300000000.0};

    const cv::Mat reference_image = makeTexturedImage(image_width, image_height);
    cv::Mat source_image = makeShiftedImage(reference_image, disparity);
    cv::GaussianBlur(source_image, source_image, cv::Size(3, 3), 0.0);

    const auto reference =
        makeCamera(focal, focal, image_width * 0.5, image_height * 0.5, 1, 1, identity, reference_center, false)
            .normalizedForPositiveDepth();
    const auto source =
        makeCamera(focal, focal, image_width * 0.5, image_height * 0.5, 1, 1, identity, source_center, false)
            .normalizedForPositiveDepth();
    const auto translated_reference =
        makeCamera(
            focal, focal, image_width * 0.5, image_height * 0.5, 1, 1, identity, translated_reference_center, false)
            .normalizedForPositiveDepth();
    const auto translated_source =
        makeCamera(focal, focal, image_width * 0.5, image_height * 0.5, 1, 1, identity, translated_source_center, false)
            .normalizedForPositiveDepth();

    xjw::mvs::PatchMatchConfig config;
    config.backend = xjw::mvs::PatchMatchBackend::Cpu;
    config.downsampleFactor = 2;
    config.numIterations = 4;
    config.patchHalf = 2;
    config.confidenceThresh = 0.15f;
    config.doMedianBlur = false;
    config.doBilateralFilter = false;

    const auto estimate = [&](const xjw::FramePinholeCamera& reference_camera,
                              const xjw::FramePinholeCamera& source_camera,
                              cv::Mat* depth,
                              cv::Mat* confidence)
    {
        std::string error;
        EXPECT_TRUE(xjw::mvs::PatchMatchDepthEstimator::estimate(reference_image,
                                                                 std::vector<cv::Mat>{source_image},
                                                                 reference_camera,
                                                                 std::vector<xjw::FramePinholeCamera>{source_camera},
                                                                 5.0f,
                                                                 15.0f,
                                                                 config,
                                                                 *depth,
                                                                 confidence,
                                                                 &error))
            << error;
    };

    cv::Mat local_depth;
    cv::Mat local_confidence;
    cv::Mat translated_depth;
    cv::Mat translated_confidence;
    estimate(reference, source, &local_depth, &local_confidence);
    estimate(translated_reference, translated_source, &translated_depth, &translated_confidence);

    ASSERT_FALSE(local_depth.empty());
    ASSERT_EQ(translated_depth.size(), local_depth.size());
    EXPECT_DOUBLE_EQ(cv::norm(local_depth, translated_depth, cv::NORM_INF), 0.0);
    EXPECT_DOUBLE_EQ(cv::norm(local_confidence, translated_confidence, cv::NORM_INF), 0.0);
}

TEST(PatchMatchDepthPostprocessTest, PreservesInvalidHolesAndDoesNotPullEdgesTowardZero)
{
    cv::Mat depth(15, 17, CV_32FC1, cv::Scalar(0.0f));
    depth(cv::Rect(2, 3, 6, 9)).setTo(10.0f);
    depth(cv::Rect(9, 3, 6, 9)).setTo(20.0f);
    depth.at<float>(7, 4) = 100.0f;
    depth.at<float>(7, 11) = 200.0f;
    depth.at<float>(6, 5) = 0.0f;
    const cv::Mat original_valid_mask = depth > 0.0f;

    xjw::mvs::PatchMatchConfig config;
    config.doMedianBlur = true;
    config.medianKernelSize = 5;
    config.doBilateralFilter = true;
    config.bilateralD = 9;
    config.bilateralSigmaColor = 50.0f;
    config.bilateralSigmaSpace = 5.0f;
    xjw::mvs::postprocessPatchMatchDepth(depth, config);

    const cv::Mat filtered_valid_mask = depth > 0.0f;
    EXPECT_EQ(cv::countNonZero(original_valid_mask != filtered_valid_mask), 0);
    EXPECT_FLOAT_EQ(depth.at<float>(6, 5), 0.0f);
    EXPECT_NEAR(depth.at<float>(7, 4), 10.0f, 0.05f);
    EXPECT_NEAR(depth.at<float>(7, 11), 20.0f, 0.1f);
    EXPECT_NEAR(depth.at<float>(7, 7), 10.0f, 0.05f);
    EXPECT_NEAR(depth.at<float>(7, 9), 20.0f, 0.1f);
}

TEST(PatchMatchDepthPostprocessTest, IsInvariantToUniformDepthScale)
{
    cv::Mat base(19, 23, CV_32FC1, cv::Scalar(0.0f));
    for (int row = 2; row < base.rows - 2; ++row)
    {
        for (int column = 2; column < base.cols - 2; ++column)
        {
            if ((row + 2 * column) % 11 == 0)
            {
                continue;
            }
            base.at<float>(row, column) = column < base.cols / 2 ? 8.0f + 0.02f * static_cast<float>(row)
                                                                 : 16.0f + 0.03f * static_cast<float>(row);
        }
    }

    cv::Mat small = base * 1.0e-4f;
    cv::Mat large = base * 1.0e4f;
    xjw::mvs::PatchMatchConfig config;
    config.doMedianBlur = true;
    config.medianKernelSize = 5;
    config.doBilateralFilter = true;
    config.bilateralD = 7;
    config.bilateralSigmaColor = 40.0f;
    config.bilateralSigmaSpace = 3.0f;

    xjw::mvs::postprocessPatchMatchDepth(base, config);
    xjw::mvs::postprocessPatchMatchDepth(small, config);
    xjw::mvs::postprocessPatchMatchDepth(large, config);
    small *= 1.0e4f;
    large *= 1.0e-4f;

    EXPECT_LT(cv::norm(base, small, cv::NORM_INF), 2.0e-4);
    EXPECT_LT(cv::norm(base, large, cv::NORM_INF), 2.0e-4);
    EXPECT_EQ(cv::countNonZero((base > 0.0f) != (small > 0.0f)), 0);
    EXPECT_EQ(cv::countNonZero((base > 0.0f) != (large > 0.0f)), 0);
}

TEST(PatchMatchDepthPostprocessTest, ReferenceGuidancePreservesVisibleDepthBoundary)
{
    constexpr int width = 31;
    constexpr int height = 17;
    constexpr int boundary = width / 2;
    cv::Mat original(height, width, CV_32FC1, cv::Scalar(10.0f));
    original.colRange(boundary, width).setTo(11.0f);

    cv::Mat guide(height, width, CV_8UC3, cv::Scalar(20, 20, 20));
    guide.colRange(boundary, width).setTo(cv::Scalar(235, 235, 235));

    xjw::mvs::PatchMatchConfig config;
    config.doMedianBlur = false;
    config.doBilateralFilter = true;
    config.bilateralD = 9;
    config.bilateralSigmaColor = 500.0f;
    config.bilateralSigmaSpace = 5.0f;
    config.bilateralSigmaGuidance = 0.05f;

    cv::Mat depth_only = original.clone();
    config.enableReferenceGuidedFilter = false;
    xjw::mvs::postprocessPatchMatchDepth(depth_only, config, guide);

    cv::Mat guided = original.clone();
    config.enableReferenceGuidedFilter = true;
    xjw::mvs::postprocessPatchMatchDepth(guided, config, guide);

    const float depth_only_left_error =
        std::abs(depth_only.at<float>(height / 2, boundary - 1) - 10.0f);
    const float guided_left_error =
        std::abs(guided.at<float>(height / 2, boundary - 1) - 10.0f);
    const float depth_only_right_error =
        std::abs(depth_only.at<float>(height / 2, boundary) - 11.0f);
    const float guided_right_error =
        std::abs(guided.at<float>(height / 2, boundary) - 11.0f);

    EXPECT_GT(depth_only_left_error, 0.20f);
    EXPECT_GT(depth_only_right_error, 0.20f);
    EXPECT_LT(guided_left_error, 1.0e-3f);
    EXPECT_LT(guided_right_error, 1.0e-3f);
}

CudaPatchMatchRunStats runCudaPatchMatchSmallPlane(bool useParallelSweep)
{
    constexpr int IMAGE_WIDTH = 96;
    constexpr int IMAGE_HEIGHT = 72;
    constexpr double FOCAL = 80.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 8;

    cv::Mat refGray = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::Mat srcGray = makeShiftedImage(refGray, DISPARITY);
    cv::GaussianBlur(srcGray, srcGray, cv::Size(3, 3), 0.0);

    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};

    const auto refCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, refCenter, false)
            .normalizedForPositiveDepth();
    const auto srcCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, srcCenter, false)
            .normalizedForPositiveDepth();

    CudaPatchMatchRunStats stats = executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, useParallelSweep, 3);
    const cv::Rect roi(16, 12, IMAGE_WIDTH - 32, IMAGE_HEIGHT - 24);
    const int pixelCount = roi.width * roi.height;
    EXPECT_GT(stats.validCount, pixelCount * 0.10) << "CUDA PatchMatch should produce a non-trivial valid depth region";
    return stats;
}

TEST(PatchMatchCudaRegressionTest, ParallelSweepProducesDepthForSmallPlane)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }
    runCudaPatchMatchSmallPlane(true);
}

TEST(PatchMatchCudaRegressionTest, LegacySweepFallbackProducesDepthForSmallPlane)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }
    runCudaPatchMatchSmallPlane(false);
}

TEST(PatchMatchCudaRegressionTest, ReusesWorkspaceAcrossSequentialFrames)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    const CudaPatchMatchRunStats first = runCudaPatchMatchSmallPlane(true);
    const CudaPatchMatchRunStats second = runCudaPatchMatchSmallPlane(true);
    EXPECT_EQ(first.validCount, second.validCount);
    EXPECT_DOUBLE_EQ(first.meanDepth, second.meanDepth);
    EXPECT_DOUBLE_EQ(first.meanConfidence, second.meanConfidence);
    xjw::mvs::PatchMatchDepthEstimator::cleanupGpuImageCache();
}

TEST(PatchMatchCudaRegressionTest, ConcurrentHostFrameSlotsRemainIsolated)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    auto first_future = std::async(std::launch::async, []() { return runCudaPatchMatchSmallPlane(true); });
    auto second_future = std::async(std::launch::async, []() { return runCudaPatchMatchSmallPlane(true); });

    const CudaPatchMatchRunStats first = first_future.get();
    const CudaPatchMatchRunStats second = second_future.get();
    EXPECT_GT(first.validCount, 0);
    EXPECT_EQ(first.validCount, second.validCount);
    xjw::mvs::PatchMatchDepthEstimator::cleanupGpuImageCache();
}

TEST(PatchMatchCudaRegressionTest, RejectsOutOfRangeDeviceIndex)
{
    const int device_count = xjw::mvs::PatchMatchDepthEstimator::cudaDeviceCount();
    if (device_count == 0)
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    std::string error;
    EXPECT_FALSE(xjw::mvs::PatchMatchDepthEstimator::reserveGpuWorkspace(1024, 1, false, false, &error, device_count));
    EXPECT_NE(error.find("device index"), std::string::npos);
}

TEST(PatchMatchCudaBenchmarkTest, DISABLED_CompareParallelAndLegacySweepAfterWarmup)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    constexpr int IMAGE_WIDTH = 640;
    constexpr int IMAGE_HEIGHT = 480;
    constexpr double FOCAL = 520.0;
    constexpr double BASELINE = 1.0;
    constexpr int DISPARITY = 52;

    cv::Mat refGray = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::Mat srcGray = makeShiftedImage(refGray, DISPARITY);
    cv::GaussianBlur(srcGray, srcGray, cv::Size(3, 3), 0.0);

    const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};
    const auto refCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, refCenter, false)
            .normalizedForPositiveDepth();
    const auto srcCam =
        makeCamera(FOCAL, FOCAL, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5, 1, 1, identity, srcCenter, false)
            .normalizedForPositiveDepth();

    (void)executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, true, 4);
    (void)executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, false, 4);

    const CudaPatchMatchRunStats parallelStats = executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, true, 4);
    const CudaPatchMatchRunStats legacyStats = executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, false, 4);

    const int pixelCount = (IMAGE_WIDTH * 2 / 3) * (IMAGE_HEIGHT * 2 / 3);
    EXPECT_GT(parallelStats.validCount, pixelCount * 0.10);
    EXPECT_GT(legacyStats.validCount, pixelCount * 0.10);
    EXPECT_NEAR(parallelStats.meanDepth, 10.0, 1.0);
    EXPECT_LT(parallelStats.depthRmse, 1.0);
    EXPECT_NEAR(legacyStats.meanDepth, 10.0, 0.1);
    EXPECT_LT(legacyStats.depthRmse, 0.1);
    std::fprintf(stderr,
                 "[PatchMatchCudaBenchmark] parallel=%.2f ms legacy=%.2f ms speedup=%.2fx\n",
                 parallelStats.elapsedMs,
                 legacyStats.elapsedMs,
                 legacyStats.elapsedMs / std::max(1e-6, parallelStats.elapsedMs));
    std::fprintf(stdout,
                 "PATCHMATCH_BASELINE_JSON={\"image_width\":%d,\"image_height\":%d,"
                 "\"iterations\":4,\"parallel\":{\"elapsed_ms\":%.3f,\"valid_count\":%d,"
                 "\"roi_pixel_count\":%d,\"mean_depth\":%.6f,\"mean_confidence\":%.6f,"
                 "\"depth_rmse\":%.6f},"
                 "\"legacy\":{\"elapsed_ms\":%.3f,\"valid_count\":%d,"
                 "\"roi_pixel_count\":%d,\"mean_depth\":%.6f,\"mean_confidence\":%.6f,"
                 "\"depth_rmse\":%.6f}}\n",
                 IMAGE_WIDTH,
                 IMAGE_HEIGHT,
                 parallelStats.elapsedMs,
                 parallelStats.validCount,
                 parallelStats.roiPixelCount,
                 parallelStats.meanDepth,
                 parallelStats.meanConfidence,
                 parallelStats.depthRmse,
                 legacyStats.elapsedMs,
                 legacyStats.validCount,
                 legacyStats.roiPixelCount,
                 legacyStats.meanDepth,
                 legacyStats.meanConfidence,
                 legacyStats.depthRmse);
}
