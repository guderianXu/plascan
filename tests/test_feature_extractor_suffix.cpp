// =============================================================================
// test_feature_extractor_suffix.cpp — 多提取器后缀映射 + 自动检测 + 工厂
// 不依赖 Qt (避免 slots/signals 宏与 LibTorch 冲突)
// =============================================================================
#include <gtest/gtest.h>

#include "ExtractorFactory.h"
#include "CudaSiftFeatureExtractor.h"
#include "FeatureData.h"
#include "TraditionalFeatureExtractor.h"

#include <opencv2/imgproc.hpp>

#include <string>

// ── 1. 后缀映射 (验证与 FeatureFileIO.h 中 ExtractorSuffix 一致) ──

// 内联拷贝 FeatureFileIO.h 的 forAlgorithm 逻辑, 避免 Qt 依赖
static const char *suffixForAlgorithm(const std::string &algo)
{
    if (algo == "disk")       return ".dsk";
    if (algo == "aliked")     return ".alk";
    if (algo == "sift")       return ".sift";
    if (algo == "orb")        return ".orb";
    if (algo == "akaze")      return ".akz";
    if (algo == "surf")       return ".surf";
    if (algo == "dedode")     return ".dedode";
    return ".sp";
}

TEST(ExtractorSuffixTest, ForAlgorithmReturnsCorrectSuffix)
{
    EXPECT_STREQ(suffixForAlgorithm("superpoint"), ".sp");
    EXPECT_STREQ(suffixForAlgorithm("disk"),       ".dsk");
    EXPECT_STREQ(suffixForAlgorithm("aliked"),     ".alk");
    EXPECT_STREQ(suffixForAlgorithm("sift"),       ".sift");
    EXPECT_STREQ(suffixForAlgorithm("orb"),        ".orb");
    EXPECT_STREQ(suffixForAlgorithm("akaze"),      ".akz");
    EXPECT_STREQ(suffixForAlgorithm("surf"),       ".surf");
    EXPECT_STREQ(suffixForAlgorithm("dedode"),     ".dedode");
}

TEST(ExtractorSuffixTest, UnknownAlgorithmReturnsDotSp)
{
    EXPECT_STREQ(suffixForAlgorithm("unknown_algo"), ".sp");
}

// ── 2. 自动后缀检测 (模拟 cli_feature_match.cpp) ──

TEST(SuffixDetectionTest, AutoDetectFromPath)
{
    auto autoMatcher = [](const std::string &spPath) -> std::string
    {
        auto pos = spPath.rfind('.');
        if (pos == std::string::npos) return "superglue";
        std::string ext = spPath.substr(pos);
        if (ext == ".sp" || ext == ".dedode") return "superglue";
        if (ext == ".dsk" || ext == ".alk" || ext == ".sift") return "bf";
        return "superglue";
    };

    EXPECT_EQ(autoMatcher("img.sp"),     "superglue");
    EXPECT_EQ(autoMatcher("img.dedode"), "superglue");
    EXPECT_EQ(autoMatcher("img.dsk"),    "bf");
    EXPECT_EQ(autoMatcher("img.alk"),    "bf");
    EXPECT_EQ(autoMatcher("img.sift"),   "bf");
    EXPECT_EQ(autoMatcher("img.orb"),    "superglue");
    EXPECT_EQ(autoMatcher("img.xyz"),    "superglue");
}

// ── 3. 算法名归一化 ──

TEST(AlgorithmNormalizationTest, BasicNormalization)
{
    using xjw::feature_extractors::TraditionalFeatureExtractor;
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("SuperPoint"), "superpoint");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("DISK"),       "disk");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("SIFT"),       "sift");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("orb"),        "orb");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("surf"),       "surf");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("aliked"),     "aliked");
    // Unknown algorithms fall back to "superpoint" (default)
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("akaze"),      "akaze");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("unknown"),    "superpoint");
}

TEST(AlgorithmNormalizationTest, IsTraditionalAlgorithm)
{
    using xjw::feature_extractors::TraditionalFeatureExtractor;
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("sift"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("orb"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("surf"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("akaze"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("superpoint"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("disk"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("aliked"));
}

TEST(TraditionalFeatureExtractorTest, AkazeProducesFeatures)
{
    cv::Mat image(240, 320, CV_8UC1, cv::Scalar(20));
    cv::circle(image, cv::Point(80, 80), 30, cv::Scalar(220), -1);
    cv::rectangle(image, cv::Rect(180, 60, 80, 80), cv::Scalar(180), -1);
    cv::line(image, cv::Point(40, 200), cv::Point(280, 180), cv::Scalar(240), 3);

    SuperPointConfig cfg;
    cfg.max_num_keypoints = 256;
    cfg.remove_borders = 0;
    cfg.grayscale_min = 0.0f;
    cfg.grayscale_max = 1.0f;

    const FeatureOutput output =
        xjw::feature_extractors::TraditionalFeatureExtractor::detect(image, cfg, "akaze");

    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(output.descriptors.defined());
    EXPECT_EQ(output.descriptors.size(0), static_cast<int64_t>(output.keypoints.size()));
}

TEST(TraditionalFeatureExtractorTest, SiftKeepsNativeDescriptorDim)
{
    cv::Mat image(240, 320, CV_8UC1, cv::Scalar(40));
    cv::circle(image, cv::Point(90, 90), 35, cv::Scalar(230), -1);
    cv::rectangle(image, cv::Rect(170, 55, 85, 95), cv::Scalar(175), -1);
    cv::line(image, cv::Point(30, 205), cv::Point(290, 175), cv::Scalar(245), 4);

    SuperPointConfig cfg;
    cfg.max_num_keypoints = 256;
    cfg.descriptor_dim = 256;
    cfg.remove_borders = 0;
    cfg.grayscale_min = 0.0f;
    cfg.grayscale_max = 1.0f;

    const FeatureOutput output =
        xjw::feature_extractors::TraditionalFeatureExtractor::detect(image, cfg, "sift");

    ASSERT_FALSE(output.empty());
    ASSERT_TRUE(output.descriptors.defined());
    EXPECT_EQ(output.descriptors.size(0), static_cast<int64_t>(output.keypoints.size()));
    EXPECT_EQ(output.descriptors.size(1), 128);
}

TEST(TraditionalFeatureExtractorTest, SiftCudaRequestUsesOpenCvSiftAndKeepsNativeDescriptorDim)
{
    cv::Mat image(240, 320, CV_8UC1, cv::Scalar(40));
    cv::circle(image, cv::Point(90, 90), 35, cv::Scalar(230), -1);
    cv::rectangle(image, cv::Rect(170, 55, 85, 95), cv::Scalar(175), -1);
    cv::line(image, cv::Point(30, 205), cv::Point(290, 175), cv::Scalar(245), 4);

    SuperPointConfig cfg;
    cfg.max_num_keypoints = 256;
    cfg.descriptor_dim = 256;
    cfg.remove_borders = 0;
    cfg.grayscale_min = 0.0f;
    cfg.grayscale_max = 1.0f;
    cfg.allow_device_fallback = true;

    const FeatureOutput output =
        xjw::feature_extractors::TraditionalFeatureExtractor::detect(image, cfg, "sift", true, 0);

    ASSERT_FALSE(output.empty());
    ASSERT_TRUE(output.descriptors.defined());
    EXPECT_EQ(output.descriptors.size(0), static_cast<int64_t>(output.keypoints.size()));
    EXPECT_EQ(output.descriptors.size(1), 128);
}

TEST(TraditionalFeatureExtractorTest, FactoryKeepsSiftExtractionOnOpenCvWhenCudaRequested)
{
    cv::Mat image(240, 320, CV_8UC1, cv::Scalar(40));
    cv::circle(image, cv::Point(90, 90), 35, cv::Scalar(230), -1);
    cv::rectangle(image, cv::Rect(170, 55, 85, 95), cv::Scalar(175), -1);
    cv::line(image, cv::Point(30, 205), cv::Point(290, 175), cv::Scalar(245), 4);

    ExtractorConfig cfg;
    cfg.maxKeypoints = 256;
    cfg.removeBorder = 0;
    cfg.useCuda = true;
    cfg.cudaDevice = 0;

    auto extractor = xjw::feature_extractors::createExtractor("sift", cfg);
    const FeatureOutput output = extractor->extract(image);

    ASSERT_FALSE(output.empty());
    ASSERT_TRUE(output.descriptors.defined());
    EXPECT_EQ(output.descriptors.size(0), static_cast<int64_t>(output.keypoints.size()));
    EXPECT_EQ(output.descriptors.size(1), 128);
}

TEST(TraditionalFeatureExtractorTest, CudaSiftAvailabilityQueryDoesNotThrow)
{
    (void)xjw::feature_extractors::isCudaSiftAvailable();
    SUCCEED();
}

TEST(TraditionalFeatureExtractorTest, CudaSiftReportsNonZeroDetectionScores)
{
    if (!xjw::feature_extractors::isCudaSiftAvailable())
    {
        GTEST_SKIP() << "CUDA SIFT is not available on this machine";
    }

    cv::Mat image(512, 512, CV_8UC1);
    cv::randu(image, 20, 235);
    cv::circle(image, cv::Point(140, 160), 60, cv::Scalar(245), -1);
    cv::rectangle(image, cv::Rect(270, 95, 120, 160), cv::Scalar(35), -1);
    cv::line(image, cv::Point(40, 430), cv::Point(470, 360), cv::Scalar(250), 5);

    SuperPointConfig cfg;
    cfg.max_num_keypoints = 512;
    cfg.remove_borders = 0;
    cfg.grayscale_min = 0.0f;
    cfg.grayscale_max = 1.0f;

    const FeatureOutput output = xjw::feature_extractors::detectCudaSift(image, cfg, 0);

    ASSERT_FALSE(output.empty());
    ASSERT_FALSE(output.scores.empty());
    const auto max_score = *std::max_element(output.scores.begin(), output.scores.end());
    EXPECT_GT(max_score, 0.0f);
}

TEST(TraditionalFeatureExtractorTest, SiftDescriptorsUseRootSiftNormalization)
{
    cv::Mat descriptor(1, 2, CV_32F);
    descriptor.at<float>(0, 0) = 1.0f;
    descriptor.at<float>(0, 1) = 3.0f;

    const torch::Tensor tensor =
        xjw::feature_extractors::FeatureData::cvDescriptorsToTensor(descriptor, 2, "sift");

    ASSERT_TRUE(tensor.defined());
    ASSERT_EQ(tensor.size(0), 1);
    ASSERT_EQ(tensor.size(1), 2);
    EXPECT_NEAR(tensor[0][0].item<float>(), 0.5f, 1e-5f);
    EXPECT_NEAR(tensor[0][1].item<float>(), 0.8660254f, 1e-5f);
}

TEST(FeatureOutputPostprocessTest, GrayscaleRangeFiltersTensorOutput)
{
    cv::Mat gray(4, 4, CV_8UC1, cv::Scalar(0));
    gray.at<uchar>(0, 0) = 25;
    gray.at<uchar>(1, 1) = 115;
    gray.at<uchar>(2, 2) = 204;
    gray.at<uchar>(3, 3) = 242;

    const torch::Tensor kpts = torch::tensor(
        {{{0.0f, 0.0f},
          {1.0f, 1.0f},
          {2.0f, 2.0f},
          {3.0f, 3.0f},
          {99.0f, 99.0f}}},
        torch::kFloat32);
    const torch::Tensor descs = torch::arange(10, torch::kFloat32).reshape({1, 5, 2});
    const torch::Tensor scores = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}}, torch::kFloat32);

    const FeatureOutput output = tensorToFeatureOutput(
        kpts,
        descs,
        scores,
        0.0f,
        1.0f,
        -1,
        &gray,
        0.4f,
        0.85f);

    ASSERT_EQ(output.keypoints.size(), 2u);
    EXPECT_FLOAT_EQ(output.keypoints[0].pt.x, 1.0f);
    EXPECT_FLOAT_EQ(output.keypoints[0].pt.y, 1.0f);
    EXPECT_FLOAT_EQ(output.keypoints[1].pt.x, 2.0f);
    EXPECT_FLOAT_EQ(output.keypoints[1].pt.y, 2.0f);
    EXPECT_EQ(output.scores.size(), 2u);
    EXPECT_FLOAT_EQ(output.scores[0], 0.2f);
    EXPECT_FLOAT_EQ(output.scores[1], 0.3f);
    ASSERT_TRUE(output.descriptors.defined());
    ASSERT_EQ(output.descriptors.size(0), 2);
    ASSERT_EQ(output.descriptors.size(1), 2);
    EXPECT_FLOAT_EQ(output.descriptors[0][0].item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(output.descriptors[1][0].item<float>(), 4.0f);
}

// ── 4. ExtractorConfig 默认值 ──

TEST(ExtractorConfigTest, DefaultValuesAreReasonable)
{
    ExtractorConfig cfg;
    EXPECT_EQ(cfg.maxKeypoints, 4096);
    EXPECT_FLOAT_EQ(cfg.detThreshold, 0.003f);
    EXPECT_EQ(cfg.nmsRadius, 3);
    EXPECT_EQ(cfg.removeBorder, 4);
    EXPECT_EQ(cfg.maxImageDim, 2048);
    EXPECT_FLOAT_EQ(cfg.grayscaleMin, 0.0f);
    EXPECT_FLOAT_EQ(cfg.grayscaleMax, 1.0f);
    EXPECT_TRUE(cfg.useCuda);
    EXPECT_EQ(cfg.cudaDevice, 0);
}
