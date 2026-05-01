// =============================================================================
// 文件: test_aliked.cpp
// 功能: ALIKED 提取器测试
// =============================================================================
#include "AlikedExtractor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace xjw::feature_extractors;

TEST(AlikedExtractorTest, ModelLoads)
{
    AlikedConfig cfg;
    cfg.modelPath = std::string(MODEL_DIR) + "/aliked_extractor_cpu_480.pt";
    cfg.useCuda   = false;
    try {
        AlikedExtractor ext(cfg);
        SUCCEED();
    } catch (const std::exception &e) {
        FAIL() << "Load failed: " << e.what();
    }
}

TEST(AlikedExtractorTest, ExtractOnTestImage)
{
    cv::Mat img = cv::imread(
        std::string(TESTDATA_DIR) + "/1.tif", cv::IMREAD_GRAYSCALE);
    if (img.empty()) GTEST_SKIP() << "test image not found";

    float s = 640.0f / std::max(img.cols, img.rows);
    cv::resize(img, img, cv::Size(), s, s);

    AlikedConfig cfg;
    cfg.modelPath   = std::string(MODEL_DIR) + "/aliked_extractor_cpu_480.pt";
    cfg.useCuda     = false;
    cfg.maxImageDim = 0;
    AlikedExtractor ext(cfg);
    auto result = ext.extract(img);

    EXPECT_GT(result.keypoints.size(), 0u);
    EXPECT_EQ(result.keypoints.size(), result.scores.size());
    if (!result.keypoints.empty())
        EXPECT_EQ(result.descriptors.size(1), 128);
}

TEST(AlikedExtractorTest, CoordinatesInRange)
{
    cv::Mat img = cv::imread(
        std::string(TESTDATA_DIR) + "/1.tif", cv::IMREAD_GRAYSCALE);
    if (img.empty()) GTEST_SKIP() << "test image not found";

    float s = 640.0f / std::max(img.cols, img.rows);
    cv::resize(img, img, cv::Size(), s, s);

    AlikedConfig cfg;
    cfg.modelPath   = std::string(MODEL_DIR) + "/aliked_extractor_cpu_480.pt";
    cfg.useCuda     = false;
    cfg.maxImageDim = 0;
    AlikedExtractor ext(cfg);
    auto result = ext.extract(img);

    for (const auto &kp : result.keypoints)
    {
        EXPECT_GE(kp.pt.x, 0.0f);
        EXPECT_GE(kp.pt.y, 0.0f);
        EXPECT_LE(kp.pt.x, static_cast<float>(img.cols));
        EXPECT_LE(kp.pt.y, static_cast<float>(img.rows));
    }
}
