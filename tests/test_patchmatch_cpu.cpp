#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "Camera.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>

namespace
{

struct CudaPatchMatchRunStats
{
    int validCount = 0;
    double elapsedMs = 0.0;
};

xjw::Camera makeCamera(double fu,
                       double fv,
                       double cu,
                       double cv,
                       int uDir,
                       int vDir,
                       const double r_wc[9],
                       const double center[3],
                       bool depthAxisFlipped)
{
    xjw::Camera camera;
    std::array<double, 9> rotation{{
        r_wc[0], r_wc[1], r_wc[2],
        r_wc[3], r_wc[4], r_wc[5],
        r_wc[6], r_wc[7], r_wc[8]
    }};
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
            const double value = 128.0
                + 42.0 * std::sin(0.19 * static_cast<double>(x))
                + 37.0 * std::cos(0.23 * static_cast<double>(y))
                + (((x / 6) + (y / 5)) % 2 == 0 ? 18.0 : -18.0);
            image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(value, 0.0, 255.0));
        }
    }
    return image;
}

cv::Mat makeShiftedImage(const cv::Mat &image, int disparity)
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

CudaPatchMatchRunStats executeCudaPatchMatchCase(
    const cv::Mat &refGray,
    const cv::Mat &srcGray,
    const xjw::Camera &refCam,
    const xjw::Camera &srcCam,
    bool useParallelSweep,
    int iterations)
{
    xjw::mvs::PatchMatchConfig config;
    config.useCuda = true;
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
    const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        refGray,
        std::vector<cv::Mat>{srcGray},
        refCam,
        std::vector<xjw::Camera>{srcCam},
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

    const cv::Rect roi(refGray.cols / 6,
                       refGray.rows / 6,
                       refGray.cols * 2 / 3,
                       refGray.rows * 2 / 3);
    int validCount = 0;
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
        for (int x = roi.x; x < roi.x + roi.width; ++x)
        {
            if (depthMap.at<float>(y, x) > 0.0f)
            {
                ++validCount;
            }
        }
    }

    CudaPatchMatchRunStats stats;
    stats.validCount = validCount;
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

    const double identity[9] = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};

    const auto refCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, refCenter,
                                   false).normalizedForPositiveDepth();
    const auto srcCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, srcCenter,
                                   false).normalizedForPositiveDepth();

    xjw::mvs::PatchMatchConfig config;
    config.useCuda = false;
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
    const bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        refGray,
        std::vector<cv::Mat>{srcGray},
        refCam,
        std::vector<xjw::Camera>{srcCam},
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

    EXPECT_NEAR(meanDepth, EXPECTED_DEPTH, 1.0)
        << "Recovered CPU depth should stay close to the expected plane depth";
    EXPECT_GT(meanConfidence, 0.35)
        << "Recovered CPU depth should have meaningful confidence";
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

    const double identity[9] = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};

    const auto refCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, refCenter,
                                   false).normalizedForPositiveDepth();
    const auto srcCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, srcCenter,
                                   false).normalizedForPositiveDepth();

    CudaPatchMatchRunStats stats = executeCudaPatchMatchCase(refGray,
                                                             srcGray,
                                                             refCam,
                                                             srcCam,
                                                             useParallelSweep,
                                                             3);
    const cv::Rect roi(16, 12, IMAGE_WIDTH - 32, IMAGE_HEIGHT - 24);
    const int pixelCount = roi.width * roi.height;
    EXPECT_GT(stats.validCount, pixelCount * 0.10)
        << "CUDA PatchMatch should produce a non-trivial valid depth region";
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
    xjw::mvs::PatchMatchDepthEstimator::cleanupGpuImageCache();
}

TEST(PatchMatchCudaRegressionTest, ConcurrentHostFrameSlotsRemainIsolated)
{
    if (!xjw::mvs::PatchMatchDepthEstimator::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA PatchMatch is not available in this environment";
    }

    auto first_future = std::async(std::launch::async, []()
    {
        return runCudaPatchMatchSmallPlane(true);
    });
    auto second_future = std::async(std::launch::async, []()
    {
        return runCudaPatchMatchSmallPlane(true);
    });

    const CudaPatchMatchRunStats first = first_future.get();
    const CudaPatchMatchRunStats second = second_future.get();
    EXPECT_GT(first.validCount, 0);
    EXPECT_EQ(first.validCount, second.validCount);
    xjw::mvs::PatchMatchDepthEstimator::cleanupGpuImageCache();
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
    constexpr int DISPARITY = 26;

    cv::Mat refGray = makeTexturedImage(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::Mat srcGray = makeShiftedImage(refGray, DISPARITY);
    cv::GaussianBlur(srcGray, srcGray, cv::Size(3, 3), 0.0);

    const double identity[9] = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
    const double refCenter[3] = {0.0, 0.0, 0.0};
    const double srcCenter[3] = {BASELINE, 0.0, 0.0};
    const auto refCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, refCenter,
                                   false).normalizedForPositiveDepth();
    const auto srcCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, srcCenter,
                                   false).normalizedForPositiveDepth();

    (void)executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, true, 4);
    (void)executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, false, 4);

    const CudaPatchMatchRunStats parallelStats =
        executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, true, 4);
    const CudaPatchMatchRunStats legacyStats =
        executeCudaPatchMatchCase(refGray, srcGray, refCam, srcCam, false, 4);

    const int pixelCount = (IMAGE_WIDTH * 2 / 3) * (IMAGE_HEIGHT * 2 / 3);
    EXPECT_GT(parallelStats.validCount, pixelCount * 0.10);
    EXPECT_GT(legacyStats.validCount, pixelCount * 0.10);
    std::fprintf(stderr,
                 "[PatchMatchCudaBenchmark] parallel=%.2f ms legacy=%.2f ms speedup=%.2fx\n",
                 parallelStats.elapsedMs,
                 legacyStats.elapsedMs,
                 legacyStats.elapsedMs / std::max(1e-6, parallelStats.elapsedMs));
}
