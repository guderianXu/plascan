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

#include "FeatureOutput.h"
#include "IExtractor.h"

namespace xjw::feature_extractors
{

struct DiskConfig
{
    std::string modelPath;
    int    maxKeypoints  = 2048;
    float  scoreThreshold = 0.0f;
    int    maxImageDim    = 1600;
    float  grayscaleMin   = 0.0f;
    float  grayscaleMax   = 1.0f;
    bool   useCuda        = true;
    int    cudaDevice     = 0;
};

class DiskExtractor : public IExtractor
{
public:
    explicit DiskExtractor(const DiskConfig &cfg);
    FeatureOutput extract(const cv::Mat &grayImage);
    std::string algorithmName() const override { return "disk"; }

private:
    DiskConfig _config;
    torch::jit::script::Module _model;
    torch::Device _device{torch::kCPU};
};

} // namespace xjw::feature_extractors
