#include "MaskGenerator.h"
#include "u2net/U2NetMaskGenerator.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <vector>

namespace
{

cv::Mat makeAsteroidLikeImage()
{
    cv::Mat image(80, 100, CV_8UC1, cv::Scalar(0));
    cv::ellipse(image, cv::Point(50, 40), cv::Size(22, 16), 12.0, 0.0, 360.0, cv::Scalar(210), -1);
    cv::circle(image, cv::Point(36, 32), 4, cv::Scalar(165), -1);
    cv::circle(image, cv::Point(61, 49), 3, cv::Scalar(245), -1);
    return image;
}

std::filesystem::path modelPath(const char *name)
{
    return std::filesystem::path(TEST_DATA_DIR).parent_path() / "resources" / "models" / name;
}

std::filesystem::path u2netModelPath()
{
    const char *envPath = std::getenv("PLASCAN_U2NET_MODEL");
    if (envPath && std::filesystem::exists(envPath))
    {
        return std::filesystem::path(envPath);
    }

    const std::filesystem::path bundled = modelPath("U2Net_v1.onnx");
    if (std::filesystem::exists(bundled))
    {
        return bundled;
    }

    const char *modelDir = std::getenv("PLASCAN_MODEL_DIR");
    if (modelDir)
    {
        const std::filesystem::path candidate = std::filesystem::path(modelDir) / "U2Net_v1.onnx";
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return bundled;
}

cv::Mat readImageFromFileBytes(const std::filesystem::path &path, int flags = cv::IMREAD_UNCHANGED)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        return {};
    }

    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.empty())
    {
        return {};
    }

    return cv::imdecode(bytes, flags);
}

cv::Mat toBgr8ForU2NetTest(const cv::Mat &image)
{
    cv::Mat image8;
    if (image.depth() == CV_8U)
    {
        image8 = image;
    }
    else
    {
        cv::normalize(image, image8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::Mat bgr;
    if (image8.channels() == 1)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_GRAY2BGR);
    }
    else if (image8.channels() == 4)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_BGRA2BGR);
    }
    else
    {
        bgr = image8.clone();
    }
    return bgr;
}

cv::Mat makeU2NetBlobForTest(const cv::Mat &image)
{
    const cv::Mat bgr = toBgr8ForU2NetTest(image);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(320, 320), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    const double mean[3] = {0.485, 0.456, 0.406};
    const double stddev[3] = {0.229, 0.224, 0.225};
    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean[i]) / stddev[i];
    }
    cv::merge(channels, normalized);

    return cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
}

cv::Mat firstOutputPlaneForTest(const cv::Mat &output)
{
    if (output.dims == 4)
    {
        return cv::Mat(output.size[2], output.size[3], CV_32F, const_cast<float *>(output.ptr<float>())).clone();
    }
    if (output.dims == 3)
    {
        return cv::Mat(output.size[1], output.size[2], CV_32F, const_cast<float *>(output.ptr<float>())).clone();
    }
    return output.clone();
}

cv::Mat maskFromU2NetOutputForTest(const cv::Mat &image, const cv::Mat &output)
{
    cv::Mat probability = firstOutputPlaneForTest(output);
    probability.convertTo(probability, CV_32F);

    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(probability, &minValue, &maxValue);
    if (maxValue - minValue > 1e-6)
    {
        probability = (probability - minValue) / (maxValue - minValue);
    }
    else
    {
        probability.setTo(0.0f);
    }

    cv::Mat fullResolution;
    cv::resize(probability, fullResolution, image.size(), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat foregroundFloat;
    cv::threshold(fullResolution, foregroundFloat, 0.5, 255.0, cv::THRESH_BINARY);

    cv::Mat foreground;
    foregroundFloat.convertTo(foreground, CV_8U);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(foreground, foreground, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int components = cv::connectedComponentsWithStats(foreground, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered = cv::Mat::zeros(foreground.size(), CV_8UC1);
    if (components > 1)
    {
        int largestLabel = 1;
        int largestArea = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int label = 2; label < components; ++label)
        {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area > largestArea)
            {
                largestArea = area;
                largestLabel = label;
            }
        }
        if (largestArea >= 64)
        {
            filtered.setTo(255, labels == largestLabel);
        }
    }

    cv::Mat mask;
    cv::bitwise_not(filtered, mask);
    return mask;
}

void writeMaskOverlay(const std::filesystem::path &path, const cv::Mat &image, const cv::Mat &mask)
{
    cv::Mat image8;
    if (image.depth() == CV_8U)
    {
        image8 = image;
    }
    else
    {
        cv::normalize(image, image8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::Mat bgr;
    if (image8.channels() == 1)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_GRAY2BGR);
    }
    else if (image8.channels() == 4)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_BGRA2BGR);
    }
    else
    {
        bgr = image8.clone();
    }

    const std::vector<std::vector<cv::Point>> contours = xjw::mask::extractMaskContours(mask);
    cv::drawContours(bgr, contours, -1, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::drawContours(bgr, contours, -1, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::imwrite(path.string(), bgr);
}

} // namespace

TEST(MaskGeneratorTest, BlackBackgroundMethodMasksBackgroundAndKeepsTarget)
{
    xjw::mask::MaskGenerationOptions options;
    options.method = xjw::mask::MaskGenerationMethod::BlackBackground;
    options.threshold = -1.0;
    options.minComponentArea = 20;
    options.morphologyRadius = 1;

    const cv::Mat mask = xjw::mask::generateMask(makeAsteroidLikeImage(), options);

    ASSERT_EQ(mask.type(), CV_8UC1);
    ASSERT_EQ(mask.rows, 80);
    ASSERT_EQ(mask.cols, 100);
    EXPECT_EQ(mask.at<uchar>(0, 0), 255);
    EXPECT_EQ(mask.at<uchar>(40, 50), 0);
    EXPECT_EQ(mask.at<uchar>(49, 61), 0);
}

TEST(MaskGeneratorTest, ContoursFollowUnmaskedForegroundBoundary)
{
    xjw::mask::MaskGenerationOptions options;
    options.method = xjw::mask::MaskGenerationMethod::BlackBackground;
    options.threshold = -1.0;
    options.minComponentArea = 20;
    options.morphologyRadius = 1;

    const cv::Mat mask = xjw::mask::generateMask(makeAsteroidLikeImage(), options);
    const auto contours = xjw::mask::extractMaskContours(mask, true);

    ASSERT_FALSE(contours.empty());
    EXPECT_GT(contours.front().size(), 20);
}

TEST(MaskComposerTest, OperationsMatchMetashapeStyleMaskComposition)
{
    cv::Mat existing(3, 3, CV_8UC1, cv::Scalar(0));
    cv::Mat generated(3, 3, CV_8UC1, cv::Scalar(0));
    existing.at<uchar>(1, 1) = 255;
    generated.at<uchar>(1, 1) = 255;
    generated.at<uchar>(0, 0) = 255;

    const cv::Mat replaced = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Replace);
    const cv::Mat united = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Union);
    const cv::Mat intersected = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Intersection);
    const cv::Mat differed = xjw::mask::composeMasks(existing, generated, xjw::mask::MaskOperation::Difference);

    EXPECT_EQ(replaced.at<uchar>(0, 0), 255);
    EXPECT_EQ(united.at<uchar>(0, 0), 255);
    EXPECT_EQ(intersected.at<uchar>(0, 0), 0);
    EXPECT_EQ(intersected.at<uchar>(1, 1), 255);
    EXPECT_EQ(differed.at<uchar>(0, 0), 0);
    EXPECT_EQ(differed.at<uchar>(1, 1), 0);
}

TEST(U2NetMaskGeneratorTest, ReportsOpenCvDnnDeviceCapabilities)
{
    const xjw::mask::U2NetDnnCapabilities capabilities = xjw::mask::detectU2NetDnnCapabilities();

    EXPECT_TRUE(capabilities.hasCpu);
    EXPECT_FALSE(capabilities.summary.empty());
    EXPECT_NE(capabilities.summary.find("OpenCV CUDA build="), std::string::npos);
    EXPECT_NE(capabilities.summary.find("DNN CUDA target="), std::string::npos);
    EXPECT_NE(capabilities.summary.find("DNN CUDA backend="), std::string::npos);
    if (capabilities.hasDnnCudaBackend)
    {
        EXPECT_TRUE(capabilities.opencvBuiltWithCuda);
    }
    if (capabilities.cudaDeviceCount > 0)
    {
        EXPECT_TRUE(capabilities.hasCudaDevice);
        EXPECT_TRUE(capabilities.opencvBuiltWithCuda);
    }
}

TEST(U2NetMaskGeneratorTest, ModelFileNamesMatchBundledMetashapeStyleName)
{
    const xjw::mask::U2NetMaskGeneratorConfig config;

    EXPECT_EQ(xjw::mask::u2netDefaultModelFileName(), "U2Net_v1.onnx");
    EXPECT_EQ(config.inputSize, 320);
    EXPECT_FLOAT_EQ(config.foregroundThreshold, 0.5f);
    EXPECT_FALSE(config.allowDeviceFallback);
}

TEST(U2NetMaskGeneratorIntegrationTest, OnnxModelRunsOnCpuWhenPresent)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const cv::Mat image = makeAsteroidLikeImage();

    xjw::mask::U2NetMaskGeneratorConfig cpuConfig;
    cpuConfig.modelPath = onnxPath.string();
    cpuConfig.useCuda = false;
    cpuConfig.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator cpuGenerator(cpuConfig);
    const xjw::mask::U2NetMaskResult cpuResult = cpuGenerator.generate(image);

    ASSERT_FALSE(cpuResult.mask.empty());
    EXPECT_FALSE(cpuResult.usedCuda);
    EXPECT_EQ(cpuResult.mask.size(), image.size());
    EXPECT_GT(cv::countNonZero(cpuResult.mask == 0), 0);
}

TEST(U2NetMaskGeneratorIntegrationTest, UsesFirstFusedOutputForMultiOutputOnnx)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const cv::Mat image = makeAsteroidLikeImage();

    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxPath.string());
    ASSERT_FALSE(net.empty());
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setInput(makeU2NetBlobForTest(image));

    std::vector<cv::Mat> outputs;
    const std::vector<std::string> outputNames = net.getUnconnectedOutLayersNames();
    net.forward(outputs, outputNames);
    ASSERT_GT(outputs.size(), 1u) << "U2Net_v1.onnx should expose fused output plus side outputs.";

    const cv::Mat expectedFusedMask = maskFromU2NetOutputForTest(image, outputs.front());

    xjw::mask::U2NetMaskGeneratorConfig cpuConfig;
    cpuConfig.modelPath = onnxPath.string();
    cpuConfig.useCuda = false;
    cpuConfig.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator cpuGenerator(cpuConfig);
    const xjw::mask::U2NetMaskResult cpuResult = cpuGenerator.generate(image);

    ASSERT_FALSE(cpuResult.mask.empty());
    EXPECT_EQ(cv::countNonZero(cpuResult.mask != expectedFusedMask), 0);
}

TEST(U2NetMaskGeneratorIntegrationTest, OnnxModelRunsOnCudaWhenBackendAvailable)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const cv::Mat image = makeAsteroidLikeImage();
    const xjw::mask::U2NetDnnCapabilities capabilities = xjw::mask::detectU2NetDnnCapabilities();
    if (!capabilities.hasDnnCudaBackend)
    {
        GTEST_SKIP() << "OpenCV DNN CUDA backend is not available for U2Net: " << capabilities.summary;
    }

    xjw::mask::U2NetMaskGeneratorConfig cudaConfig;
    cudaConfig.modelPath = onnxPath.string();
    cudaConfig.useCuda = true;
    cudaConfig.allowDeviceFallback = false;
    try
    {
        xjw::mask::U2NetMaskGenerator cudaGenerator(cudaConfig);
        const xjw::mask::U2NetMaskResult cudaResult = cudaGenerator.generate(image);

        ASSERT_FALSE(cudaResult.mask.empty());
        EXPECT_TRUE(cudaResult.usedCuda);
        EXPECT_EQ(cudaResult.mask.size(), image.size());
        EXPECT_GT(cv::countNonZero(cudaResult.mask == 0), 0);
    }
    catch (const std::exception &error)
    {
        FAIL() << "OpenCV is CUDA-built and a CUDA device is visible, but U2Net CUDA execution failed: "
               << error.what() << "; " << capabilities.summary;
    }
}

TEST(U2NetMaskGeneratorIntegrationTest, RejectsCudaWithoutDnnBackendInsteadOfSilentlyUsingCpu)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const xjw::mask::U2NetDnnCapabilities capabilities = xjw::mask::detectU2NetDnnCapabilities();
    if (capabilities.hasDnnCudaBackend)
    {
        GTEST_SKIP() << "OpenCV DNN CUDA is available; rejection is covered by CPU-only builds.";
    }

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = onnxPath.string();
    config.useCuda = true;
    config.allowDeviceFallback = false;

    try
    {
        const xjw::mask::U2NetMaskGenerator generator(config);
        FAIL() << "CUDA request must not silently use OpenCV DNN CPU.";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("OpenCV DNN CUDA backend is not available"),
                  std::string::npos);
    }
}

TEST(U2NetMaskGeneratorDebugTest, DumpsExternalImageWhenRequested)
{
    const char *imageEnv = std::getenv("PLASCAN_U2NET_DEBUG_IMAGE");
    const char *outputDirEnv = std::getenv("PLASCAN_U2NET_DEBUG_OUTPUT_DIR");
    if (!imageEnv || !outputDirEnv)
    {
        GTEST_SKIP() << "Set PLASCAN_U2NET_DEBUG_IMAGE and PLASCAN_U2NET_DEBUG_OUTPUT_DIR to dump U2Net output.";
    }

    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const std::filesystem::path imagePath(imageEnv);
    const cv::Mat image = readImageFromFileBytes(imagePath);
    ASSERT_FALSE(image.empty()) << "Failed to decode debug image: " << imagePath.string();

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = onnxPath.string();
    config.useCuda = false;
    config.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator generator(config);
    const xjw::mask::U2NetMaskResult result = generator.generate(image);

    ASSERT_FALSE(result.mask.empty());
    EXPECT_EQ(result.mask.size(), image.size());

    const std::filesystem::path outputDir(outputDirEnv);
    std::filesystem::create_directories(outputDir);
    const std::filesystem::path maskPath = outputDir / "u2net_cpp_mask.png";
    const std::filesystem::path overlayPath = outputDir / "u2net_cpp_overlay.png";
    ASSERT_TRUE(cv::imwrite(maskPath.string(), result.mask));
    writeMaskOverlay(overlayPath, image, result.mask);
}
