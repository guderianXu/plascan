// =============================================================================
// 文件: test_disk.cpp
// 功能: DISK 提取器测试
// =============================================================================
#include "DiskExtractor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

using namespace xjw::feature_extractors;

class DiskExtractorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 测试影像在 testdata/
        testImg = cv::imread(
            std::string(TESTDATA_DIR) + "/1.tif", cv::IMREAD_GRAYSCALE);
        if (!testImg.empty())
        {
            // 缩到 1200px 方便测试
            float s = 1200.0f / std::max(testImg.cols, testImg.rows);
            cv::resize(testImg, testImg, cv::Size(), s, s);
        }
    }
    cv::Mat testImg;
};

TEST_F(DiskExtractorTest, ModelLoads)
{
    if (testImg.empty()) GTEST_SKIP() << "test image not found";
    DiskConfig cfg;
    cfg.modelPath = std::string(MODEL_DIR) + "/disk_extractor_cuda_1200.pt";
    cfg.useCuda   = false;  // CPU test
    cfg.maxImageDim = 0;    // no further resize
    try {
        DiskExtractor ext(cfg);
        SUCCEED();
    } catch (const std::exception &e) {
        FAIL() << "Model load failed: " << e.what();
    }
}

TEST_F(DiskExtractorTest, ExtractReturnsKeypoints)
{
    if (testImg.empty()) GTEST_SKIP() << "test image not found";
    DiskConfig cfg;
    cfg.modelPath  = std::string(MODEL_DIR) + "/disk_extractor_cuda_1200.pt";
    cfg.useCuda    = false;
    cfg.maxImageDim = 0;
    DiskExtractor ext(cfg);
    auto result = ext.extract(testImg);
    EXPECT_GT(result.keypoints.size(), 0u);
    EXPECT_EQ(result.keypoints.size(), result.scores.size());
}

TEST_F(DiskExtractorTest, DescriptorsCorrectShape)
{
    if (testImg.empty()) GTEST_SKIP() << "test image not found";
    DiskConfig cfg;
    cfg.modelPath  = std::string(MODEL_DIR) + "/disk_extractor_cuda_1200.pt";
    cfg.useCuda    = false;
    cfg.maxImageDim = 0;
    DiskExtractor ext(cfg);
    auto result = ext.extract(testImg);
    ASSERT_GT(result.keypoints.size(), 0u);
    EXPECT_EQ(result.descriptors.size(0), static_cast<long>(result.keypoints.size()));
    EXPECT_EQ(result.descriptors.size(1), 128);  // DISK desc dim
}

TEST_F(DiskExtractorTest, MaxKeypointsRespected)
{
    if (testImg.empty()) GTEST_SKIP() << "test image not found";
    DiskConfig cfg;
    cfg.modelPath   = std::string(MODEL_DIR) + "/disk_extractor_cuda_1200.pt";
    cfg.useCuda     = false;
    cfg.maxImageDim = 0;
    cfg.maxKeypoints = 100;
    DiskExtractor ext(cfg);
    auto result = ext.extract(testImg);
    EXPECT_LE(result.keypoints.size(), 100u);
}

TEST_F(DiskExtractorTest, CoordinatesInImageRange)
{
    if (testImg.empty()) GTEST_SKIP() << "test image not found";
    DiskConfig cfg;
    cfg.modelPath  = std::string(MODEL_DIR) + "/disk_extractor_cuda_1200.pt";
    cfg.useCuda    = false;
    cfg.maxImageDim = 0;
    DiskExtractor ext(cfg);
    auto result = ext.extract(testImg);
    for (const auto &kp : result.keypoints)
    {
        EXPECT_GE(kp.pt.x, 0.0f);
        EXPECT_GE(kp.pt.y, 0.0f);
        EXPECT_LE(kp.pt.x, static_cast<float>(testImg.cols));
        EXPECT_LE(kp.pt.y, static_cast<float>(testImg.rows));
    }
}
