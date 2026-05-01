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

namespace xjw::feature_extractors
{

struct DiskConfig
{
    std::string modelPath;
    int    maxKeypoints  = 2048;
    float  scoreThreshold = 0.0f;
    int    maxImageDim    = 1600;      // 超则降采样
    bool   useCuda        = true;
    int    cudaDevice     = 0;
};

struct DiskOutput
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<float>        scores;
    torch::Tensor             descriptors;  // [N, 128] float32
    float                     scale = 1.0f;  // 降采样时的缩放因子
};

class DiskExtractor
{
public:
    explicit DiskExtractor(const DiskConfig &cfg);
    DiskOutput extract(const cv::Mat &grayImage);

private:
    DiskConfig m_cfg;
    torch::jit::script::Module m_model;
    torch::Device m_device{torch::kCPU};
};

} // namespace xjw::feature_extractors
