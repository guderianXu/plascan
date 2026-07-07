#include "MaskGenerator.h"
#include "Sam21MaskGenerator.h"
#include "u2net/U2NetMaskGenerator.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <opencv2/imgproc.hpp>

#include <filesystem>

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

bool allModelsExist(const std::initializer_list<const char *> names)
{
    for (const char *name : names)
    {
        if (!std::filesystem::exists(modelPath(name)))
        {
            return false;
        }
    }
    return true;
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

TEST(Sam21MaskGeneratorTest, ModelNamesSeparateEncoderDecoderAndCpuCuda)
{
    const auto cpuNames = xjw::mask::sam21TorchScriptModelNames(
        xjw::mask::Sam21ModelVariant::Tiny,
        false);
    const auto cudaNames = xjw::mask::sam21TorchScriptModelNames(
        xjw::mask::Sam21ModelVariant::Tiny,
        true);

    EXPECT_EQ(cpuNames.encoder, "sam21_hiera_tiny_encoder_cpu.pt");
    EXPECT_EQ(cpuNames.decoder, "sam21_hiera_tiny_decoder_cpu.pt");
    EXPECT_EQ(cudaNames.encoder, "sam21_hiera_tiny_encoder_cuda.pt");
    EXPECT_EQ(cudaNames.decoder, "sam21_hiera_tiny_decoder_cuda.pt");
}

TEST(Sam21MaskGeneratorTest, FullImageBoxPromptDoesNotDependOnBlackBackground)
{
    const cv::Size imageSize(640, 480);
    const xjw::mask::Sam21Prompt prompt = xjw::mask::Sam21Prompt::fullImageBox(imageSize);

    ASSERT_TRUE(prompt.hasBox);
    EXPECT_FLOAT_EQ(prompt.box.x, 0.0f);
    EXPECT_FLOAT_EQ(prompt.box.y, 0.0f);
    EXPECT_FLOAT_EQ(prompt.box.width, 640.0f);
    EXPECT_FLOAT_EQ(prompt.box.height, 480.0f);
    EXPECT_TRUE(prompt.points.empty());
}

TEST(Sam21MaskGeneratorTest, AutoBoxUsesForegroundBoundsForDarkBorderImages)
{
    const cv::Mat image = makeAsteroidLikeImage();
    const xjw::mask::Sam21Prompt prompt = xjw::mask::Sam21Prompt::autoBox(image, 0.0f);

    ASSERT_TRUE(prompt.hasBox);
    EXPECT_GT(prompt.box.x, 0.0f);
    EXPECT_GT(prompt.box.y, 0.0f);
    EXPECT_LT(prompt.box.width, static_cast<float>(image.cols));
    EXPECT_LT(prompt.box.height, static_cast<float>(image.rows));
    EXPECT_LE(prompt.box.x, 28.0f);
    EXPECT_LE(prompt.box.y, 23.0f);
    EXPECT_GE(prompt.box.x + prompt.box.width, 73.0f);
    EXPECT_GE(prompt.box.y + prompt.box.height, 57.0f);
}

TEST(Sam21MaskGeneratorTest, AutoBoxFallsBackToFullImageWhenBorderIsNotDark)
{
    cv::Mat image(80, 100, CV_8UC3, cv::Scalar(120, 130, 140));
    cv::rectangle(image, cv::Rect(30, 20, 40, 30), cv::Scalar(200, 210, 220), -1);

    const xjw::mask::Sam21Prompt prompt = xjw::mask::Sam21Prompt::autoBox(image);

    ASSERT_TRUE(prompt.hasBox);
    EXPECT_FLOAT_EQ(prompt.box.x, 0.0f);
    EXPECT_FLOAT_EQ(prompt.box.y, 0.0f);
    EXPECT_FLOAT_EQ(prompt.box.width, 100.0f);
    EXPECT_FLOAT_EQ(prompt.box.height, 80.0f);
}

TEST(Sam21MaskGeneratorIntegrationTest, TorchScriptModelsRunOnCpuAndCudaWhenPresent)
{
    constexpr const char *cpuEncoder = "sam21_hiera_tiny_encoder_cpu.pt";
    constexpr const char *cpuDecoder = "sam21_hiera_tiny_decoder_cpu.pt";
    constexpr const char *cudaEncoder = "sam21_hiera_tiny_encoder_cuda.pt";
    constexpr const char *cudaDecoder = "sam21_hiera_tiny_decoder_cuda.pt";

    if (!allModelsExist({cpuEncoder, cpuDecoder, cudaEncoder, cudaDecoder}))
    {
        GTEST_SKIP() << "SAM2.1 TorchScript models are not available in resources/models.";
    }

    const cv::Mat image = makeAsteroidLikeImage();
    const xjw::mask::Sam21Prompt prompt = xjw::mask::Sam21Prompt::autoBox(image);

    xjw::mask::Sam21MaskGeneratorConfig cpuConfig;
    cpuConfig.encoderModelPath = modelPath(cpuEncoder).string();
    cpuConfig.decoderModelPath = modelPath(cpuDecoder).string();
    cpuConfig.useCuda = false;
    cpuConfig.allowDeviceFallback = false;
    xjw::mask::Sam21MaskGenerator cpuGenerator(cpuConfig);
    const xjw::mask::Sam21MaskResult cpuResult = cpuGenerator.generate(image, prompt);

    ASSERT_FALSE(cpuResult.mask.empty());
    EXPECT_FALSE(cpuResult.usedCuda);
    EXPECT_EQ(cpuResult.mask.size(), image.size());
    EXPECT_GT(cv::countNonZero(cpuResult.mask == 0), 0);

    xjw::mask::Sam21MaskGeneratorConfig cudaConfig;
    cudaConfig.encoderModelPath = modelPath(cudaEncoder).string();
    cudaConfig.decoderModelPath = modelPath(cudaDecoder).string();
    cudaConfig.cpuEncoderModelPath = modelPath(cpuEncoder).string();
    cudaConfig.cpuDecoderModelPath = modelPath(cpuDecoder).string();
    cudaConfig.useCuda = true;
    cudaConfig.allowDeviceFallback = false;

    try
    {
        xjw::mask::Sam21MaskGenerator cudaGenerator(cudaConfig);
        const xjw::mask::Sam21MaskResult cudaResult = cudaGenerator.generate(image, prompt);

        ASSERT_FALSE(cudaResult.mask.empty());
        EXPECT_TRUE(cudaResult.usedCuda);
        EXPECT_EQ(cudaResult.mask.size(), image.size());
        EXPECT_GT(cv::countNonZero(cudaResult.mask == 0), 0);
    }
    catch (const std::exception &error)
    {
        GTEST_SKIP() << "SAM2.1 CUDA runtime is not available: " << error.what();
    }
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
    EXPECT_TRUE(config.allowDeviceFallback);
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

TEST(U2NetMaskGeneratorIntegrationTest, OnnxModelRunsOnCudaWhenBackendAvailable)
{
    const std::filesystem::path onnxPath = u2netModelPath();
    if (!std::filesystem::exists(onnxPath))
    {
        GTEST_SKIP() << "U2Net_v1.onnx is not available in resources/models or PLASCAN_U2NET_MODEL.";
    }

    const cv::Mat image = makeAsteroidLikeImage();

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
        GTEST_SKIP() << "OpenCV DNN CUDA backend is not available for U2Net: " << error.what();
    }
}
