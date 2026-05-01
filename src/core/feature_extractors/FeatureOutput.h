// =============================================================================
// 文件: FeatureOutput.h
// 功能: 通用特征提取输出结构 (所有提取器共用) + tensor→FeatureOutput 转换
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <vector>

struct FeatureOutput
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<float>        scores;
    torch::Tensor             descriptors;   // [N, D] float32, L2 normalized

    bool empty() const { return keypoints.empty(); }
    int  count() const { return static_cast<int>(keypoints.size()); }
    int  descDim() const { return descriptors.defined() ? descriptors.size(1) : 0; }
};

// 通用转换: 模型输出 tensor [1,N,2]/[1,N,D]/[1,N] → FeatureOutput
inline FeatureOutput tensorToFeatureOutput(
    const torch::Tensor &kpts,      // [1,N,2] float32
    const torch::Tensor &descs,     // [1,N,D] float32
    const torch::Tensor &scores,    // [1,N]   float32
    float scoreThreshold = 0.0f,
    float coordScale     = 1.0f)
{
    int N = static_cast<int>(kpts.size(1));
    auto kptsCpu   = kpts.to(torch::kCPU);
    auto scoresCpu = scores.to(torch::kCPU);
    auto kptAcc    = kptsCpu.accessor<float, 3>();
    auto scoreAcc  = scoresCpu.accessor<float, 2>();

    FeatureOutput result;
    for (int i = 0; i < N; ++i)
    {
        float conf = scoreAcc[0][i];
        if (conf < scoreThreshold) continue;

        cv::KeyPoint kp;
        kp.pt.x = kptAcc[0][i][0] / coordScale;
        kp.pt.y = kptAcc[0][i][1] / coordScale;
        kp.response = conf;
        kp.size = 1.0f;
        result.keypoints.push_back(kp);
        result.scores.push_back(conf);
    }

    if (!result.empty())
    {
        auto descCpu = descs.to(torch::kCPU).contiguous();
        auto descAcc = descCpu.accessor<float, 3>();
        int D = static_cast<int>(descs.size(2));
        int M = result.count();
        result.descriptors = torch::empty({M, D}, torch::kFloat32);
        auto resAcc = result.descriptors.accessor<float, 2>();
        int outIdx = 0;
        for (int i = 0; i < N; ++i)
        {
            if (scoreAcc[0][i] < scoreThreshold) continue;
            for (int d = 0; d < D; ++d)
                resAcc[outIdx][d] = descAcc[0][i][d];
            ++outIdx;
        }
    }

    return result;
}
