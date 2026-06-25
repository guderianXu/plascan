// =============================================================================
// 文件: LoFTRMatcher.cpp
// 功能: LoFTR 端到端匹配器实现
// =============================================================================
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "LoFTRMatcher.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xjw::feature_match
{

LoFTRMatcher::LoFTRMatcher(const LoFTRConfig &cfg) : _config(cfg)
{
    _device = (cfg.useCuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
    _model  = torch::jit::load(cfg.modelPath, _device);
    _model.eval();
}

static torch::Tensor toTensor(const cv::Mat &gray, torch::Device device)
{
    cv::Mat f;
    gray.convertTo(f, CV_32FC1, 1.0 / 255.0);
    auto t = torch::from_blob(f.data, {1, 1, f.rows, f.cols},
                              torch::kFloat32).clone();
    return t.to(device);
}

LoFTRResult LoFTRMatcher::match(const cv::Mat &img0, const cv::Mat &img1)
{
    CV_Assert(img0.type() == CV_8UC1 && img1.type() == CV_8UC1);

    int ow0 = img0.cols, oh0 = img0.rows;
    int ow1 = img1.cols, oh1 = img1.rows;
    int maxSide = std::max({ow0, oh0, ow1, oh1});
    float scale = 1.0f;

    cv::Mat left  = img0.clone();
    cv::Mat right = img1.clone();

    if (_config.maxImageDim > 0 && maxSide > _config.maxImageDim)
    {
        scale = static_cast<float>(_config.maxImageDim) / maxSide;
        cv::resize(left,  left,  cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(right, right, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    auto t0 = toTensor(left,  _device);
    auto t1 = toTensor(right, _device);

    // forward(img0, img1) → (mkpts0, mkpts1, mconf)
    auto output = _model.forward({t0, t1}).toTuple();
    auto mkpts0 = output->elements()[0].toTensor().to(torch::kCPU);
    auto mkpts1 = output->elements()[1].toTensor().to(torch::kCPU);
    auto mconf  = output->elements()[2].toTensor().to(torch::kCPU);

    int N = static_cast<int>(mkpts0.size(0));
    LoFTRResult result;
    result.scale = scale;
    result.numMatches = N;
    result.pts0.reserve(N);
    result.pts1.reserve(N);
    result.confidences.reserve(N);

    auto a0 = mkpts0.accessor<float, 2>();
    auto a1 = mkpts1.accessor<float, 2>();
    auto ac = mconf.accessor<float, 2>();

    float thresh = _config.matchThreshold;
    for (int i = 0; i < N; ++i)
    {
        float conf = ac[i][0];
        if (conf < thresh) continue;

        float x0 = a0[i][0] / scale;
        float y0 = a0[i][1] / scale;
        float x1 = a1[i][0] / scale;
        float y1 = a1[i][1] / scale;

        result.pts0.emplace_back(x0, y0);
        result.pts1.emplace_back(x1, y1);
        result.confidences.push_back(conf);
    }
    result.numMatches = static_cast<int>(result.pts0.size());

    return result;
}

} // namespace xjw::feature_match
