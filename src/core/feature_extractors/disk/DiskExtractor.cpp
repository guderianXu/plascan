// =============================================================================
// 文件: DiskExtractor.cpp
// 功能: DISK 特征提取器 LibTorch 实现
// =============================================================================
#include "DiskExtractor.h"
#include "SuperPoint.h"  // FeatureOutput 完整定义
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xjw::feature_extractors
{

DiskExtractor::DiskExtractor(const DiskConfig &cfg) : m_cfg(cfg)
{
    m_device = (cfg.useCuda && torch::cuda::is_available())
                   ? torch::Device(torch::kCUDA, cfg.cudaDevice)
                   : torch::Device(torch::kCPU);
    m_model = torch::jit::load(cfg.modelPath, m_device);
    m_model.eval();
}

FeatureOutput DiskExtractor::extract(const cv::Mat &grayImage)
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

    // 转 tensor [1,1,H,W] float32 normalized
    cv::Mat f;
    img.convertTo(f, CV_32FC1, 1.0 / 255.0);
    auto input = torch::from_blob(f.data, {1, 1, f.rows, f.cols},
                                  torch::kFloat32).clone().to(m_device);

    // 推理: forward(image) → (keypoints, descriptors, scores)
    auto outputs = m_model.forward({input}).toTuple();
    auto kpts    = outputs->elements()[0].toTensor().to(torch::kCPU);  // [1,N,2]
    auto descs   = outputs->elements()[1].toTensor().to(torch::kCPU);  // [1,N,D]
    auto scores  = outputs->elements()[2].toTensor().to(torch::kCPU);  // [1,N]

    int N = static_cast<int>(kpts.size(1));
    auto kptAcc = kpts.accessor<float, 3>();
    auto scoreAcc = scores.accessor<float, 2>();

    FeatureOutput result;

    for (int i = 0; i < N; ++i)
    {
        float conf = scoreAcc[0][i];
        if (conf < m_cfg.scoreThreshold) continue;

        cv::KeyPoint kp;
        kp.pt.x = kptAcc[0][i][0] / scale;
        kp.pt.y = kptAcc[0][i][1] / scale;
        kp.response = conf;
        kp.size = 1.0f;
        result.keypoints.push_back(kp);
        result.scores.push_back(conf);
    }

    if (!result.keypoints.empty())
    {
        auto descAcc = descs.accessor<float, 3>();
        int D = static_cast<int>(descs.size(2));
        int M = static_cast<int>(result.keypoints.size());
        result.descriptors = torch::empty({M, D}, torch::kFloat32);
        auto resAcc = result.descriptors.accessor<float, 2>();
        int outIdx = 0;
        for (int i = 0; i < N; ++i)
        {
            if (scoreAcc[0][i] < m_cfg.scoreThreshold) continue;
            for (int d = 0; d < D; ++d)
                resAcc[outIdx][d] = descAcc[0][i][d];
            ++outIdx;
        }
    }

    return result;
}

} // namespace xjw::feature_extractors
