#include "MaskGenerator.h"
#include "birefnet/BiRefNetImageProcessing.h"
#include "birefnet/BiRefNetMaskGenerator.h"
#include "u2net/U2NetMaskGenerator.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
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

    std::filesystem::path modelPath(const char* name)
    {
        return std::filesystem::path(TEST_DATA_DIR).parent_path() / "resources" / "models" / name;
    }

    std::filesystem::path u2netModelPath()
    {
        const char* envPath = std::getenv("PLASCAN_U2NET_MODEL");
        if (envPath && std::filesystem::exists(envPath))
        {
            return std::filesystem::path(envPath);
        }

        const std::filesystem::path bundled = modelPath("U2Net_v1.onnx");
        if (std::filesystem::exists(bundled))
        {
            return bundled;
        }

        const char* modelDir = std::getenv("PLASCAN_MODEL_DIR");
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

    std::string u2netTensorRtCacheDirectory()
    {
        const char* configured = std::getenv("PLASCAN_U2NET_ENGINE_CACHE");
        if (configured && *configured)
        {
            return configured;
        }
        return (std::filesystem::temp_directory_path() / "PlaScanU2NetTestEngines").string();
    }

    bool isPathInsideDirectory(const std::filesystem::path& path,
                               const std::filesystem::path& directory)
    {
        std::error_code pathError;
        const std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(path, pathError);
        if (pathError)
        {
            return false;
        }

        std::error_code directoryError;
        const std::filesystem::path canonicalDirectory =
            std::filesystem::weakly_canonical(directory, directoryError);
        if (directoryError)
        {
            return false;
        }

        const std::filesystem::path relative = canonicalPath.lexically_relative(canonicalDirectory);
        if (relative.empty())
        {
            return canonicalPath == canonicalDirectory;
        }
        const auto first = relative.begin();
        return !relative.is_absolute() && first != relative.end() && *first != "..";
    }

    double foregroundMaskIou(const cv::Mat& lhs, const cv::Mat& rhs)
    {
        cv::Mat lhsForeground = lhs == 0;
        cv::Mat rhsForeground = rhs == 0;
        cv::Mat intersection;
        cv::Mat unionMask;
        cv::bitwise_and(lhsForeground, rhsForeground, intersection);
        cv::bitwise_or(lhsForeground, rhsForeground, unionMask);
        const int unionPixels = cv::countNonZero(unionMask);
        return unionPixels == 0 ? 1.0 : static_cast<double>(cv::countNonZero(intersection)) / unionPixels;
    }

    cv::Mat readImageFromFileBytes(const std::filesystem::path& path, int flags = cv::IMREAD_UNCHANGED)
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

    cv::Mat toBgr8ForU2NetTest(const cv::Mat& image)
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

    cv::Mat makeU2NetBlobForTest(const cv::Mat& image)
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

    cv::Mat firstOutputPlaneForTest(const cv::Mat& output)
    {
        if (output.dims == 4)
        {
            return cv::Mat(output.size[2], output.size[3], CV_32F, const_cast<float*>(output.ptr<float>())).clone();
        }
        if (output.dims == 3)
        {
            return cv::Mat(output.size[1], output.size[2], CV_32F, const_cast<float*>(output.ptr<float>())).clone();
        }
        return output.clone();
    }

    cv::Mat maskFromU2NetOutputForTest(const cv::Mat& image, const cv::Mat& output)
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

    void writeMaskOverlay(const std::filesystem::path& path, const cv::Mat& image, const cv::Mat& mask)
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

TEST(BiRefNetImageProcessingTest, LetterboxesWithoutChangingSourceAspectRatio)
{
    const cv::Mat image(80, 100, CV_8UC3, cv::Scalar(10, 20, 30));
    xjw::mask::BiRefNetLetterbox letterbox;

    const cv::Mat blob = xjw::mask::makeBiRefNetBlob(
        image, xjw::mask::kBiRefNetDynamicInputSize, &letterbox);

    ASSERT_FALSE(blob.empty());
    ASSERT_EQ(blob.dims, 4);
    EXPECT_EQ(blob.size[0], 1);
    EXPECT_EQ(blob.size[1], 3);
    EXPECT_EQ(blob.size[2], xjw::mask::kBiRefNetDynamicInputSize);
    EXPECT_EQ(blob.size[3], xjw::mask::kBiRefNetDynamicInputSize);
    EXPECT_EQ(letterbox.sourceSize, image.size());
    EXPECT_EQ(letterbox.resizedSize.width, 1024);
    EXPECT_EQ(letterbox.resizedSize.height, 819);
    EXPECT_EQ(letterbox.left, 0);
    EXPECT_EQ(letterbox.top, 102);
    EXPECT_TRUE(letterbox.isValid());
}

TEST(BiRefNetImageProcessingTest, ConvertsRawLogitsWithSigmoidWithoutPerImageMinMax)
{
    const int shape[] = {1, 1, 1, 3};
    cv::Mat logits(4, shape, CV_32F);
    logits.ptr<float>()[0] = -2.0f;
    logits.ptr<float>()[1] = 0.0f;
    logits.ptr<float>()[2] = 2.0f;

    const cv::Mat probability = xjw::mask::biRefNetProbabilityFromOutput(logits);

    ASSERT_EQ(probability.rows, 1);
    ASSERT_EQ(probability.cols, 3);
    EXPECT_NEAR(probability.at<float>(0, 0), 0.1192029f, 1e-6f);
    EXPECT_NEAR(probability.at<float>(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(probability.at<float>(0, 2), 0.8807971f, 1e-6f);
}

TEST(BiRefNetImageProcessingTest, RestoresLetterboxAndKeepsSeparateForegroundRegions)
{
    cv::Mat probability(1024, 1024, CV_32F, cv::Scalar(0.0f));
    probability(cv::Rect(100, 350, 160, 180)).setTo(0.9f);
    probability(cv::Rect(700, 420, 120, 140)).setTo(0.9f);
    const xjw::mask::BiRefNetLetterbox letterbox{
        cv::Size(200, 100), cv::Size(1024, 512), 1024, 0, 256};

    const cv::Mat mask = xjw::mask::makeBiRefNetMask(
        probability, letterbox, 0.5f, 0, 4, false);

    ASSERT_EQ(mask.size(), letterbox.sourceSize);
    EXPECT_EQ(mask.type(), CV_8UC1);
    EXPECT_EQ(mask.at<uchar>(40, 35), 0);
    EXPECT_EQ(mask.at<uchar>(50, 145), 0);
    EXPECT_EQ(mask.at<uchar>(5, 100), 255);
}

TEST(BiRefNetMaskGeneratorTest, ContractIsFixed1024WithCudaAndCpuBackends)
{
    const xjw::mask::BiRefNetMaskGeneratorConfig config;

    EXPECT_EQ(xjw::mask::biRefNetDynamicDefaultModelFileName(), "BiRefNet_dynamic_1024.onnx");
    EXPECT_EQ(config.backend, xjw::mask::BiRefNetBackendType::Auto);
    EXPECT_EQ(config.inputSize, 1024);
    EXPECT_FLOAT_EQ(config.foregroundThreshold, 0.5f);
    EXPECT_EQ(config.morphologyRadius, 0);
    EXPECT_FALSE(config.keepLargestComponent);
    EXPECT_TRUE(config.preferFp16);
    EXPECT_TRUE(config.allowDeviceFallback);
    EXPECT_EQ(xjw::mask::parseBiRefNetBackendType("cuda"),
              xjw::mask::BiRefNetBackendType::TensorRt);
    EXPECT_EQ(xjw::mask::parseBiRefNetBackendType("cpu"),
              xjw::mask::BiRefNetBackendType::OnnxRuntimeCpu);
    EXPECT_EQ(xjw::mask::biRefNetBackendTypeToken(
                  xjw::mask::BiRefNetBackendType::OnnxRuntimeCpu),
              "onnxruntime_cpu");
}

TEST(BiRefNetMaskGeneratorTest, ReportsCpuAndSelectedCudaCapabilities)
{
    const xjw::mask::BiRefNetInferenceCapabilities capabilities =
        xjw::mask::detectBiRefNetInferenceCapabilities(0);
    EXPECT_TRUE(capabilities.hasOnnxRuntimeCpu);
    EXPECT_NE(capabilities.summary.find("ONNX Runtime CPU available"), std::string::npos);
    EXPECT_NE(capabilities.summary.find("CUDA devices="), std::string::npos);
}

TEST(BiRefNetMaskGeneratorIntegrationTest, OnnxModelRunsOnOnnxRuntimeCpuWhenExplicitlyEnabled)
{
    const char* enabled = std::getenv("PLASCAN_BIREFNET_CPU_INTEGRATION");
    if (!enabled || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "Set PLASCAN_BIREFNET_CPU_INTEGRATION=1 to run the real BiRefNet CPU test.";
    }
    const char* model = std::getenv("PLASCAN_BIREFNET_MODEL");
    ASSERT_NE(model, nullptr);
    ASSERT_TRUE(std::filesystem::is_regular_file(model));

    xjw::mask::BiRefNetMaskGeneratorConfig config;
    config.modelPath = model;
    config.backend = xjw::mask::BiRefNetBackendType::OnnxRuntimeCpu;
    xjw::mask::BiRefNetMaskGenerator generator(config);
    const cv::Mat image = makeAsteroidLikeImage();
    const xjw::mask::BiRefNetMaskResult result = generator.generate(image);
    ASSERT_FALSE(result.mask.empty());
    EXPECT_EQ(result.mask.size(), image.size());
    EXPECT_FALSE(result.usedCuda);
    EXPECT_EQ(result.actualBackend, xjw::mask::BiRefNetBackendType::OnnxRuntimeCpu);
    EXPECT_EQ(result.precision, xjw::mask::BiRefNetInferencePrecision::Fp32);
}

TEST(BiRefNetMaskGeneratorIntegrationTest, OnnxModelRunsOnTensorRtWhenExplicitlyEnabled)
{
    const char* enabled = std::getenv("PLASCAN_BIREFNET_INTEGRATION");
    if (!enabled || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "Set PLASCAN_BIREFNET_INTEGRATION=1 to run the real BiRefNet TensorRT test.";
    }

    const char* model = std::getenv("PLASCAN_BIREFNET_MODEL");
    if (!model || !*model)
    {
        FAIL() << "PLASCAN_BIREFNET_MODEL must name the fixed 1024x1024 BiRefNet ONNX model.";
    }
    const std::filesystem::path onnxPath(model);
    if (!std::filesystem::is_regular_file(onnxPath))
    {
        FAIL() << "PLASCAN_BIREFNET_MODEL does not exist or is not a file: " << onnxPath.string();
    }

    const char* cache = std::getenv("PLASCAN_BIREFNET_ENGINE_CACHE");
    if (!cache || !*cache)
    {
        FAIL() << "PLASCAN_BIREFNET_ENGINE_CACHE must name an isolated writable engine cache.";
    }
    const std::filesystem::path cacheDirectory(cache);

    const xjw::mask::BiRefNetInferenceCapabilities capabilities =
        xjw::mask::detectBiRefNetInferenceCapabilities();
    if (!capabilities.tensorRtAvailable)
    {
        FAIL() << "PLASCAN_BIREFNET_INTEGRATION=1 requires TensorRT/CUDA: "
               << capabilities.summary;
    }

    xjw::mask::BiRefNetMaskGeneratorConfig config;
    config.modelPath = onnxPath.string();
    config.backend = xjw::mask::BiRefNetBackendType::TensorRt;
    config.engineCacheDirectory = cacheDirectory.string();

    try
    {
        const cv::Mat image = makeAsteroidLikeImage();
        xjw::mask::BiRefNetMaskGenerator generator(config);
        const xjw::mask::BiRefNetMaskResult result = generator.generate(image);

        std::cout << "[BiRefNetTensorRtTest] engine_reused="
                  << (result.engineReused ? "yes" : "no")
                  << "; engine_path=" << result.enginePath << '\n';

        ASSERT_FALSE(result.mask.empty());
        EXPECT_EQ(result.mask.type(), CV_8UC1);
        EXPECT_EQ(result.mask.size(), image.size());
        EXPECT_TRUE(result.usedCuda);
        EXPECT_EQ(result.requestedBackend, xjw::mask::BiRefNetBackendType::TensorRt);
        EXPECT_EQ(result.actualBackend, xjw::mask::BiRefNetBackendType::TensorRt);
        EXPECT_TRUE(result.precision == xjw::mask::BiRefNetInferencePrecision::Fp16 ||
                    result.precision == xjw::mask::BiRefNetInferencePrecision::Fp32);
        EXPECT_EQ(result.outputName, "output_image");
        EXPECT_EQ(result.modelSha256.size(), 64U);
        EXPECT_EQ(result.modelSha256, generator.modelSha256());
        ASSERT_FALSE(result.enginePath.empty());
        EXPECT_TRUE(std::filesystem::is_regular_file(result.enginePath));
        EXPECT_TRUE(isPathInsideDirectory(result.enginePath, cacheDirectory));
        EXPECT_FALSE(isPathInsideDirectory(result.enginePath, onnxPath.parent_path()));
    }
    catch (const std::exception& error)
    {
        FAIL() << "BiRefNet TensorRT integration failed: " << error.what() << "; "
               << capabilities.summary;
    }
}

TEST(U2NetMaskGeneratorTest, ReportsCpuAndTensorRtCapabilities)
{
    const xjw::mask::U2NetInferenceCapabilities capabilities = xjw::mask::detectU2NetInferenceCapabilities();

    EXPECT_TRUE(capabilities.hasOpenCvCpu);
    EXPECT_FALSE(capabilities.summary.empty());
    EXPECT_NE(capabilities.summary.find("OpenCV DNN CPU available"), std::string::npos);
    EXPECT_NE(capabilities.summary.find("TensorRT compiled="), std::string::npos);
    EXPECT_EQ(capabilities.summary.find("OpenCV CUDA"), std::string::npos);
    if (capabilities.tensorRtAvailable)
    {
        EXPECT_TRUE(capabilities.tensorRtCompiled);
        EXPECT_TRUE(capabilities.hasCudaDevice);
    }
    if (capabilities.cudaDeviceCount > 0)
    {
        EXPECT_TRUE(capabilities.hasCudaDevice);
    }
}

TEST(U2NetMaskGeneratorTest, BackendTokensIncludeLegacyCudaMigration)
{
    EXPECT_EQ(xjw::mask::u2netBackendTypeToken(xjw::mask::U2NetBackendType::Auto), "auto");
    EXPECT_EQ(xjw::mask::u2netBackendTypeToken(xjw::mask::U2NetBackendType::TensorRt), "tensorrt");
    EXPECT_EQ(xjw::mask::u2netBackendTypeToken(xjw::mask::U2NetBackendType::OpenCvCpu), "opencv_cpu");

    EXPECT_EQ(xjw::mask::parseU2NetBackendType("cuda"), xjw::mask::U2NetBackendType::TensorRt);
    EXPECT_EQ(xjw::mask::parseU2NetBackendType("TensorRT"), xjw::mask::U2NetBackendType::TensorRt);
    EXPECT_EQ(xjw::mask::parseU2NetBackendType("cpu"), xjw::mask::U2NetBackendType::OpenCvCpu);
    EXPECT_FALSE(xjw::mask::parseU2NetBackendType("invalid").has_value());
}

TEST(U2NetMaskGeneratorTest, ModelFileNamesMatchBundledMetashapeStyleName)
{
    const xjw::mask::U2NetMaskGeneratorConfig config;

    EXPECT_EQ(xjw::mask::u2netDefaultModelFileName(), "U2Net_v1.onnx");
    EXPECT_EQ(config.backend, xjw::mask::U2NetBackendType::Auto);
    EXPECT_EQ(config.inputSize, 320);
    EXPECT_FLOAT_EQ(config.foregroundThreshold, 0.5f);
    EXPECT_FALSE(config.allowDeviceFallback);
    EXPECT_TRUE(config.preferFp16);
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
    cpuConfig.backend = xjw::mask::U2NetBackendType::OpenCvCpu;
    cpuConfig.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator cpuGenerator(cpuConfig);
    const xjw::mask::U2NetMaskResult cpuResult = cpuGenerator.generate(image);

    ASSERT_FALSE(cpuResult.mask.empty());
    EXPECT_FALSE(cpuResult.usedCuda);
    EXPECT_EQ(cpuResult.actualBackend, xjw::mask::U2NetBackendType::OpenCvCpu);
    EXPECT_EQ(cpuResult.precision, xjw::mask::U2NetInferencePrecision::Fp32);
    EXPECT_EQ(cpuResult.deviceLabel, "OpenCV CPU");
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
    cpuConfig.backend = xjw::mask::U2NetBackendType::OpenCvCpu;
    cpuConfig.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator cpuGenerator(cpuConfig);
    const xjw::mask::U2NetMaskResult cpuResult = cpuGenerator.generate(image);

    ASSERT_FALSE(cpuResult.mask.empty());
    EXPECT_EQ(cv::countNonZero(cpuResult.mask != expectedFusedMask), 0);
}

TEST(U2NetMaskGeneratorIntegrationTest, OnnxModelRunsOnTensorRtWhenAvailable)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const cv::Mat image = makeAsteroidLikeImage();
    const xjw::mask::U2NetInferenceCapabilities capabilities = xjw::mask::detectU2NetInferenceCapabilities();
    std::cout << "[U2NetTensorRtTest] capabilities=" << capabilities.summary << '\n';
    if (!capabilities.tensorRtAvailable)
    {
        GTEST_SKIP() << "TensorRT is not available for U2Net: " << capabilities.summary;
    }
    std::cout << "[U2NetTensorRtTest] TensorRT available; running real GPU inference.\n";

    xjw::mask::U2NetMaskGeneratorConfig tensorRtConfig;
    tensorRtConfig.modelPath = onnxPath.string();
    tensorRtConfig.backend = xjw::mask::U2NetBackendType::TensorRt;
    tensorRtConfig.allowDeviceFallback = false;
    tensorRtConfig.engineCacheDirectory = u2netTensorRtCacheDirectory();
    try
    {
        xjw::mask::U2NetMaskGenerator tensorRtGenerator(tensorRtConfig);
        const xjw::mask::U2NetMaskResult tensorRtResult = tensorRtGenerator.generate(image);
        std::cout << "[U2NetTensorRtTest] actual_backend="
                  << xjw::mask::u2netBackendTypeToken(tensorRtResult.actualBackend)
                  << "; device=" << tensorRtResult.deviceLabel
                  << "; precision=" << xjw::mask::u2netInferencePrecisionToken(tensorRtResult.precision)
                  << "; fused_output=" << tensorRtResult.fusedOutputName
                  << "; engine_reused=" << (tensorRtResult.engineReused ? "yes" : "no") << '\n';

        ASSERT_FALSE(tensorRtResult.mask.empty());
        EXPECT_TRUE(tensorRtResult.usedCuda);
        EXPECT_EQ(tensorRtResult.actualBackend, xjw::mask::U2NetBackendType::TensorRt);
        EXPECT_TRUE(tensorRtResult.precision == xjw::mask::U2NetInferencePrecision::Fp16 ||
                    tensorRtResult.precision == xjw::mask::U2NetInferencePrecision::Fp32);
        EXPECT_EQ(tensorRtResult.mask.size(), image.size());
        EXPECT_GT(cv::countNonZero(tensorRtResult.mask == 0), 0);
        EXPECT_FALSE(tensorRtResult.enginePath.empty());
        EXPECT_EQ(tensorRtResult.fusedOutputName, "1959");
        EXPECT_FALSE(tensorRtResult.environmentSummary.empty());
        EXPECT_EQ(tensorRtResult.modelSha256.size(), 64U);
        EXPECT_NE(std::filesystem::weakly_canonical(tensorRtResult.enginePath).parent_path(),
                  std::filesystem::weakly_canonical(onnxPath).parent_path());

        xjw::mask::U2NetMaskGeneratorConfig cpuConfig;
        cpuConfig.modelPath = onnxPath.string();
        cpuConfig.backend = xjw::mask::U2NetBackendType::OpenCvCpu;
        xjw::mask::U2NetMaskGenerator cpuGenerator(cpuConfig);
        const xjw::mask::U2NetMaskResult cpuResult = cpuGenerator.generate(image);
        const double minimumIou = tensorRtResult.precision == xjw::mask::U2NetInferencePrecision::Fp16 ? 0.99 : 0.995;
        EXPECT_GE(foregroundMaskIou(tensorRtResult.mask, cpuResult.mask), minimumIou);
    }
    catch (const std::exception& error)
    {
        FAIL() << "TensorRT/CUDA is available, but U2Net TensorRT execution failed: " << error.what() << "; "
               << capabilities.summary;
    }
}

TEST(U2NetMaskGeneratorIntegrationTest, RejectsForcedTensorRtWhenUnavailable)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const xjw::mask::U2NetInferenceCapabilities capabilities = xjw::mask::detectU2NetInferenceCapabilities();
    if (capabilities.tensorRtAvailable)
    {
        GTEST_SKIP() << "TensorRT is available; rejection is covered by CPU-only builds.";
    }

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = onnxPath.string();
    config.backend = xjw::mask::U2NetBackendType::TensorRt;
    config.allowDeviceFallback = false;

    try
    {
        const xjw::mask::U2NetMaskGenerator generator(config);
        FAIL() << "A forced TensorRT request must not silently use OpenCV DNN CPU.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("TensorRT"), std::string::npos);
    }
}

TEST(U2NetMaskGeneratorIntegrationTest, AutoFallsBackToOpenCvCpuWhenTensorRtUnavailable)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }
    if (xjw::mask::detectU2NetInferenceCapabilities().tensorRtAvailable)
    {
        GTEST_SKIP() << "TensorRT is available; automatic GPU selection is covered by the TensorRT test.";
    }

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = onnxPath.string();
    config.backend = xjw::mask::U2NetBackendType::Auto;
    config.allowDeviceFallback = false;
    xjw::mask::U2NetMaskGenerator generator(config);
    const xjw::mask::U2NetMaskResult result = generator.generate(makeAsteroidLikeImage());

    EXPECT_EQ(result.actualBackend, xjw::mask::U2NetBackendType::OpenCvCpu);
    EXPECT_TRUE(result.deviceFallback);
    EXPECT_FALSE(result.fallbackReason.empty());
}

TEST(U2NetMaskGeneratorDebugTest, DumpsExternalImageWhenRequested)
{
    const char* imageEnv = std::getenv("PLASCAN_U2NET_DEBUG_IMAGE");
    const char* outputDirEnv = std::getenv("PLASCAN_U2NET_DEBUG_OUTPUT_DIR");
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
    config.backend = xjw::mask::U2NetBackendType::OpenCvCpu;
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
