// =============================================================================
// 文件: AlikedExtractor.cpp
// 功能: ALIKED 特征提取器 LibTorch 实现
// =============================================================================
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "AlikedExtractor.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "FeatureOutput.h"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xjw::feature_extractors
{

AlikedExtractor::AlikedExtractor(const AlikedConfig &cfg) : _config(cfg)
{
    _device = (cfg.useCuda && torch::cuda::is_available())
                   ? torch::Device(torch::kCUDA, cfg.cudaDevice)
                   : torch::Device(torch::kCPU);
    _model = torch::jit::load(cfg.modelPath, _device);
    _model.eval();
}

FeatureOutput AlikedExtractor::extract(const cv::Mat &grayImage)
{
    CV_Assert(grayImage.type() == CV_8UC1);

    int origW = grayImage.cols, origH = grayImage.rows;
    float scale = 1.0f;
    cv::Mat img = grayImage.clone();

    int maxSide = std::max(origW, origH);
    if (_config.maxImageDim > 0 && maxSide > _config.maxImageDim)
    {
        scale = static_cast<float>(_config.maxImageDim) / maxSide;
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat f;
    img.convertTo(f, CV_32FC1, 1.0 / 255.0);
    auto input = torch::from_blob(f.data, {1, 1, f.rows, f.cols},
                                  torch::kFloat32).clone().to(_device);

    auto orig_wh = torch::tensor({static_cast<float>(origW), static_cast<float>(origH)},
        torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    auto outputs = _model.forward({input, orig_wh}).toTuple();
    auto kpts   = outputs->elements()[0].toTensor();
    auto descs  = outputs->elements()[1].toTensor();
    auto scores = outputs->elements()[2].toTensor();

    return tensorToFeatureOutput(kpts, descs, scores,
                                 _config.scoreThreshold,
                                 scale,
                                 _config.maxKeypoints,
                                 &grayImage,
                                 _config.grayscaleMin,
                                 _config.grayscaleMax);
}

} // namespace xjw::feature_extractors
