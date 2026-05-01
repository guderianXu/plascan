// =============================================================================
// 文件: AlikedExtractor.h
// 功能: ALIKED 特征提取器 C++ LibTorch wrapper
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

#include "FeatureOutput.h"
#include "IExtractor.h"

namespace xjw::feature_extractors
{

struct AlikedConfig
{
    std::string modelPath;
    int    maxKeypoints  = 2048;
    float  scoreThreshold = 0.0f;
    int    maxImageDim    = 1600;
    bool   useCuda        = true;
    int    cudaDevice     = 0;
};

class AlikedExtractor : public IExtractor
{
public:
    explicit AlikedExtractor(const AlikedConfig &cfg);
    FeatureOutput extract(const cv::Mat &grayImage);
    std::string algorithmName() const override { return "aliked"; }

private:
    AlikedConfig m_cfg;
    torch::jit::script::Module m_model;
    torch::Device m_device{torch::kCPU};
};

} // namespace xjw::feature_extractors
