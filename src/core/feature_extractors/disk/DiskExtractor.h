// =============================================================================
// 文件: DiskExtractor.h
// 功能: DISK 特征提取器 C++ LibTorch wrapper
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

struct FeatureOutput;

namespace xjw::feature_extractors
{

struct DiskConfig
{
    std::string modelPath;
    int    maxKeypoints  = 2048;
    float  scoreThreshold = 0.0f;
    int    maxImageDim    = 1600;
    bool   useCuda        = true;
    int    cudaDevice     = 0;
};

class DiskExtractor
{
public:
    explicit DiskExtractor(const DiskConfig &cfg);
    FeatureOutput extract(const cv::Mat &grayImage);

private:
    DiskConfig m_cfg;
    torch::jit::script::Module m_model;
    torch::Device m_device{torch::kCPU};
};

} // namespace xjw::feature_extractors
