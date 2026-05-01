// =============================================================================
// 文件: AlikedExtractor.cpp
// 功能: ALIKED 特征提取器 LibTorch 实现
// =============================================================================
#include "AlikedExtractor.h"
#include "FeatureOutput.h"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xjw::feature_extractors
{

AlikedExtractor::AlikedExtractor(const AlikedConfig &cfg) : m_cfg(cfg)
{
    m_device = (cfg.useCuda && torch::cuda::is_available())
                   ? torch::Device(torch::kCUDA, cfg.cudaDevice)
                   : torch::Device(torch::kCPU);
    m_model = torch::jit::load(cfg.modelPath, m_device);
    m_model.eval();
}

FeatureOutput AlikedExtractor::extract(const cv::Mat &grayImage)
{
    CV_Assert(grayImage.type() == CV_8UC1);

    int origW = grayImage.cols, origH = grayImage.rows;
    float scale = 1.0f;
    cv::Mat img = grayImage.clone();

    int maxSide = std::max(origW, origH);
    if (m_cfg.maxImageDim > 0 && maxSide > m_cfg.maxImageDim)
    {
        scale = static_cast<float>(m_cfg.maxImageDim) / maxSide;
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat f;
    img.convertTo(f, CV_32FC1, 1.0 / 255.0);
    auto input = torch::from_blob(f.data, {1, 1, f.rows, f.cols},
                                  torch::kFloat32).clone().to(m_device);

    auto outputs = m_model.forward({input}).toTuple();
    auto kpts   = outputs->elements()[0].toTensor();
    auto descs  = outputs->elements()[1].toTensor();
    auto scores = outputs->elements()[2].toTensor();

    return tensorToFeatureOutput(kpts, descs, scores, m_cfg.scoreThreshold, scale);
}

} // namespace xjw::feature_extractors
