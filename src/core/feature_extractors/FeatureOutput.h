// =============================================================================
// 文件: FeatureOutput.h
// 功能: 通用特征提取输出结构 (所有提取器共用) + tensor→FeatureOutput 转换
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct FeatureOutput
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<float>        scores;
    torch::Tensor             descriptors;   // [N, D] float32, L2 normalized
    int                       imageWidth = 0;
    int                       imageHeight = 0;

    bool empty() const { return keypoints.empty(); }
    int  count() const { return static_cast<int>(keypoints.size()); }
    int  descDim() const { return descriptors.defined() ? descriptors.size(1) : 0; }
};

inline float normalizedGrayAt(const cv::Mat &grayImage, int x, int y)
{
    switch (grayImage.depth())
    {
    case CV_8U:
        return static_cast<float>(grayImage.at<uchar>(y, x)) / 255.0f;
    case CV_16U:
        return static_cast<float>(grayImage.at<uint16_t>(y, x)) / 65535.0f;
    case CV_32F:
        return grayImage.at<float>(y, x);
    default:
        return 0.0f;
    }
}

inline bool passesGrayscaleRange(const cv::Mat *grayImage, float x, float y,
                                 float grayscaleMin, float grayscaleMax)
{
    if (!grayImage || grayImage->empty() || grayImage->channels() != 1)
    {
        return true;
    }
    if (grayscaleMin <= 0.0f && grayscaleMax >= 1.0f)
    {
        return true;
    }
    if (grayscaleMin > grayscaleMax)
    {
        return false;
    }

    const int xi = static_cast<int>(std::round(x));
    const int yi = static_cast<int>(std::round(y));
    if (xi < 0 || yi < 0 || xi >= grayImage->cols || yi >= grayImage->rows)
    {
        return false;
    }

    const float gray = normalizedGrayAt(*grayImage, xi, yi);
    return gray >= grayscaleMin && gray <= grayscaleMax;
}

// 通用转换: 模型输出 tensor [1,N,2]/[1,N,D]/[1,N] → FeatureOutput
inline FeatureOutput tensorToFeatureOutput(
    const torch::Tensor &kpts,      // [1,N,2] float32
    const torch::Tensor &descs,     // [1,N,D] float32
    const torch::Tensor &scores,    // [1,N]   float32
    float scoreThreshold = 0.0f,
    float coordScale     = 1.0f,
    int maxKeypoints     = -1,
    const cv::Mat *grayImage = nullptr,
    float grayscaleMin  = 0.0f,
    float grayscaleMax  = 1.0f)
{
    int N = static_cast<int>(kpts.size(1));
    auto kptsCpu   = kpts.to(torch::kCPU);
    auto scoresCpu = scores.to(torch::kCPU);
    auto kptAcc    = kptsCpu.accessor<float, 3>();
    auto scoreAcc  = scoresCpu.accessor<float, 2>();

    FeatureOutput result;
    if (grayImage)
    {
        result.imageWidth = grayImage->cols;
        result.imageHeight = grayImage->rows;
    }
    std::vector<int> keptIndices;
    keptIndices.reserve(maxKeypoints > 0 ? std::min(maxKeypoints, N) : N);
    for (int i = 0; i < N; ++i)
    {
        float conf = scoreAcc[0][i];
        if (conf < scoreThreshold) continue;

        cv::KeyPoint kp;
        kp.pt.x = kptAcc[0][i][0] / coordScale;
        kp.pt.y = kptAcc[0][i][1] / coordScale;
        if (!passesGrayscaleRange(grayImage, kp.pt.x, kp.pt.y, grayscaleMin, grayscaleMax)) continue;
        if (maxKeypoints > 0 && static_cast<int>(keptIndices.size()) >= maxKeypoints) break;

        kp.response = conf;
        kp.size = 1.0f;
        result.keypoints.push_back(kp);
        result.scores.push_back(conf);
        keptIndices.push_back(i);
    }

    if (!result.empty())
    {
        auto descCpu = descs.to(torch::kCPU).contiguous();
        auto descAcc = descCpu.accessor<float, 3>();
        int D = static_cast<int>(descs.size(2));
        int M = static_cast<int>(keptIndices.size());
        result.descriptors = torch::empty({M, D}, torch::kFloat32);
        auto resAcc = result.descriptors.accessor<float, 2>();
        for (int outIdx = 0; outIdx < M; ++outIdx)
        {
            const int i = keptIndices[static_cast<std::size_t>(outIdx)];
            for (int d = 0; d < D; ++d)
                resAcc[outIdx][d] = descAcc[0][i][d];
        }
    }

    return result;
}
