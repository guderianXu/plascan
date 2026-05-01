#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "Camera.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>

namespace
{

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
                                   false).toPositiveDepthModel();
    const auto srcCam = makeCamera(FOCAL, FOCAL,
                                   IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5,
                                   1, 1,
                                   identity, srcCenter,
                                   false).toPositiveDepthModel();

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
        std::vector<xjw::mvs::PositiveDepthCameraModel>{srcCam},
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