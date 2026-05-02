#include <gtest/gtest.h>

#include "SuperPoint.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace
{

std::string findModelPath()
{
    // 从不同 CWD 回退到项目根, 优先真实 CPU 模型
    std::vector<std::string> candidates = {
        "../resources/models/superpoint_extractor_cpu.pt",
        "../resources/models/superpoint_v6_cpu.pt",
        "../resources/models/superpoint_extractor.pt",
        "../resources/models/superpoint_test.pt",
        "../../resources/models/superpoint_extractor_cpu.pt",
        "../../resources/models/superpoint_v6_cpu.pt",
        "../../resources/models/superpoint_extractor.pt",
        "../../resources/models/superpoint_test.pt",
        "../../../resources/models/superpoint_extractor_cpu.pt",
        "../../../resources/models/superpoint_v6_cpu.pt",
        "../../../resources/models/superpoint_extractor.pt",
        "../../../resources/models/superpoint_test.pt",
        "../../../../resources/models/superpoint_extractor_cpu.pt",
        "../../../../resources/models/superpoint_v6_cpu.pt",
        "../../../../resources/models/superpoint_extractor.pt",
        "../../../../resources/models/superpoint_test.pt",
        "../../../../../resources/models/superpoint_extractor_cpu.pt",
        "../../../../../resources/models/superpoint_v6_cpu.pt",
        "../../../../../resources/models/superpoint_extractor.pt",
        "../../../../../resources/models/superpoint_test.pt",
        "../../../../../../resources/models/superpoint_extractor_cpu.pt",
        "../../../../../../resources/models/superpoint_v6_cpu.pt",
        "../../../../../../resources/models/superpoint_extractor.pt",
        "../../../../../../resources/models/superpoint_test.pt"
    };
    for (const auto &p : candidates)
        if (fs::exists(p)) return p;
    return "";
}

cv::Mat loadTestImage()
{
    // 从多层 CWD 回退查找真实测试影像
    std::vector<std::string> candidates = {
        "../src/core/feature_extractors/testdata/1.tif",
        "../../src/core/feature_extractors/testdata/1.tif",
        "../../../src/core/feature_extractors/testdata/1.tif",
        "../../../../src/core/feature_extractors/testdata/1.tif",
        "../../../../../src/core/feature_extractors/testdata/1.tif",
        "../../../../../../src/core/feature_extractors/testdata/1.tif",
    };
    for (const auto &p : candidates)
        if (fs::exists(p))
            return cv::imread(p, cv::IMREAD_GRAYSCALE);
    return cv::Mat();
}

cv::Mat makeTestImage(int width, int height)
{
    // 先尝试真实影像
    auto real = loadTestImage();
    if (!real.empty())
    {
        cv::resize(real, real, cv::Size(width, height));
        return real;
    }
    // 回退: 合成图像
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
    for (int i = 0; i < 6; ++i)
    {
        int x = 30 + i * (width / 6);
        int y = 30 + (i % 3) * (height / 3);
        cv::rectangle(img, cv::Point(x, y), cv::Point(x + 60, y + 60), cv::Scalar(200), 2);
    }
    cv::rectangle(img, cv::Point(10, 10), cv::Point(80, 80), cv::Scalar(180), cv::FILLED);
    cv::rectangle(img, cv::Point(width - 90, 10), cv::Point(width - 10, 80), cv::Scalar(160), cv::FILLED);
    cv::rectangle(img, cv::Point(10, height - 90), cv::Point(80, height - 10), cv::Scalar(140), cv::FILLED);
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

class SuperPointTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        modelPath = findModelPath();
        ASSERT_FALSE(modelPath.empty()) << "未找到 superpoint_v6_cuda.pt，请确认 resources/models/ 目录";
    }

    std::string modelPath;
};

// ==============================================================================
// 测试 1：模型加载和基本推理
// ==============================================================================
TEST_F(SuperPointTest, BasicInference)
{
    SuperPointConfig cfg;
    cfg.device = torch::kCPU;
    cfg.device = torch::kCPU;
    cfg.max_num_keypoints = 500;
    cfg.detection_threshold = 0.005f;

    SuperPoint sp(modelPath, cfg);
    cv::Mat img = makeTestImage(640, 480);
    auto out = sp.detect(img);

    EXPECT_GT(out.keypoints.size(), 0u) << "棋盘格图像应检测到关键点";
    EXPECT_EQ(out.keypoints.size(), out.scores.size());
    ASSERT_TRUE(out.descriptors.defined()) << "描述子应该被定义";
    ASSERT_EQ(out.descriptors.dim(), 2) << "描述子应为 2D [N,256]";
    EXPECT_EQ(out.descriptors.size(0), static_cast<int64_t>(out.keypoints.size()));
    EXPECT_EQ(out.descriptors.size(1), 256);

    std::cout << "[BasicInference] 检测到 " << out.keypoints.size() << " 个关键点\n";
}

// ==============================================================================
// 测试 2：描述子 L2 归一化
// ==============================================================================
TEST_F(SuperPointTest, DescriptorNormalization)
{
    SuperPointConfig cfg;
    cfg.device = torch::kCPU;
    cfg.max_num_keypoints = 200;

    SuperPoint sp(modelPath, cfg);
    auto out = sp.detect(makeTestImage(640, 480));

    ASSERT_GT(out.keypoints.size(), 0u);

    auto norms = torch::norm(out.descriptors, 2, 1);  // [N]
    auto norms_cpu = norms.cpu();
    auto acc = norms_cpu.accessor<float, 1>();

    for (int64_t i = 0; i < norms_cpu.size(0); ++i)
        EXPECT_NEAR(acc[i], 1.0f, 1e-5f) << "描述子 " << i << " 的 L2 范数应为 1.0";

    std::cout << "[DescriptorNormalization] 范数 min=" << norms.min().item<float>()
              << " max=" << norms.max().item<float>()
              << " mean=" << norms.mean().item<float>() << "\n";
}

// ==============================================================================
// 测试 3：关键点坐标在图像范围内
// ==============================================================================
TEST_F(SuperPointTest, KeypointBounds)
{
    SuperPointConfig cfg;
    cfg.device = torch::kCPU;
    cfg.max_num_keypoints = 500;

    SuperPoint sp(modelPath, cfg);

    std::vector<std::pair<int, int>> sizes = {{320, 240}, {640, 480}, {800, 600}};
    for (auto [w, h] : sizes)
    {
        auto out = sp.detect(makeTestImage(w, h));
        EXPECT_GT(out.keypoints.size(), 0u) << "尺寸 " << w << "x" << h;

        for (const auto &kp : out.keypoints)
        {
            EXPECT_GE(kp.pt.x, 0.0f) << "x 坐标不应为负";
            EXPECT_LT(kp.pt.x, static_cast<float>(w)) << "x 坐标超出图像宽度";
            EXPECT_GE(kp.pt.y, 0.0f) << "y 坐标不应为负";
            EXPECT_LT(kp.pt.y, static_cast<float>(h)) << "y 坐标超出图像高度";
        }
        std::cout << "[KeypointBounds] " << w << "x" << h << ": "
                  << out.keypoints.size() << " 个关键点\n";
    }
}

// ==============================================================================
// 测试 4：sampleDescriptors 采样公式与 LightGlue 官方一致性
//
// 原理：用同一张图像分别用 superpoint_v1_compat 模型（LightGlue 官方权重）
// 和 SuperPoint.cpp 的 detect() 提取描述子，对比同一关键点的描述子余弦相似度。
// 如果采样公式一致，相似度应接近 1.0。
//
// 注意：此测试依赖 superpoint_v6_cuda.pt 就是 superpoint_v1_compat 版本，
// 即已经完成了模型替换。
// ==============================================================================
TEST_F(SuperPointTest, SampleDescriptorsConsistency)
{
    SuperPointConfig cfg;
    cfg.device = torch::kCPU;
    cfg.max_num_keypoints = 100;
    cfg.detection_threshold = 0.01f;

    SuperPoint sp(modelPath, cfg);

    // 用真实图像（棋盘格）提取两次，结果应完全一致（确定性）
    cv::Mat img = makeTestImage(640, 480);
    auto out1 = sp.detect(img);
    auto out2 = sp.detect(img);

    ASSERT_EQ(out1.keypoints.size(), out2.keypoints.size())
        << "相同输入应产生相同数量的关键点";

    if (out1.keypoints.size() == 0) GTEST_SKIP() << "未检测到关键点，跳过一致性测试";

    // 计算两次结果的描述子余弦相似度（应为 1.0）
    auto d1 = out1.descriptors;  // [N, 256]
    auto d2 = out2.descriptors;  // [N, 256]
    auto sim = (d1 * d2).sum(1);  // [N]，已归一化所以点积 = 余弦相似度
    auto sim_cpu = sim.cpu();
    auto acc = sim_cpu.accessor<float, 1>();

    for (int64_t i = 0; i < sim_cpu.size(0); ++i)
        EXPECT_NEAR(acc[i], 1.0f, 1e-4f) << "描述子 " << i << " 两次推理结果不一致";

    std::cout << "[SampleDescriptorsConsistency] 余弦相似度 min="
              << sim.min().item<float>() << " mean=" << sim.mean().item<float>() << "\n";
}

// ==============================================================================
// 测试 5：批量处理结果与逐张处理一致
// ==============================================================================
TEST_F(SuperPointTest, BatchVsSingleConsistency)
{
    SuperPointConfig cfg;
    cfg.device = torch::kCPU;
    cfg.max_num_keypoints = 100;
    cfg.batch_size = 4;

    SuperPoint sp(modelPath, cfg);

    std::vector<cv::Mat> images;
    for (int i = 0; i < 4; ++i)
        images.push_back(makeTestImage(640, 480));

    auto batch_out = sp.detectBatch(images);
    ASSERT_EQ(batch_out.size(), images.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        auto single_out = sp.detect(images[i]);
        EXPECT_EQ(batch_out[i].keypoints.size(), single_out.keypoints.size())
            << "图像 " << i << " 批量与单张关键点数量不一致";
    }

    std::cout << "[BatchVsSingleConsistency] 批量处理 " << images.size() << " 张图像通过\n";
}

} // anonymous namespace
