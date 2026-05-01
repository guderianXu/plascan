// =============================================================================
// 文件: LoFTRMatcher.h
// 功能: LoFTR 端到端匹配器 C++ wrapper — 图像 → 匹配点 (无需特征提取)
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace xjw::feature_match
{

struct LoFTRConfig
{
    std::string modelPath;
    bool useCuda = true;
    float matchThreshold = 0.2f;
    int maxImageDim = 2048;   // 超过则降采样, 适配GPU显存
};

struct LoFTRResult
{
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    std::vector<float> confidences;
    int numMatches = 0;
    float scale = 1.0f;       // 降采样倍数
};

class LoFTRMatcher
{
public:
    explicit LoFTRMatcher(const LoFTRConfig &cfg);
    LoFTRResult match(const cv::Mat &img0, const cv::Mat &img1);

private:
    LoFTRConfig m_cfg;
    torch::jit::script::Module m_model;
    torch::Device m_device{torch::kCPU};
};

} // namespace xjw::feature_match
