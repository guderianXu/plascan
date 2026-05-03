// =============================================================================
// 文件: SuperPointUtils.h
// 功能: SuperPoint 辅助函数 (NMS, 边界检查) — 多编译单元共享
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <vector>

// 邻域判断：检查特征点周围邻域是否有纯黑像素
inline bool isNearBlackBoundary(const cv::Mat &gray_image, int x, int y,
                                int radius, float threshold)
{
    if (gray_image.empty() || gray_image.type() != CV_32F)
    {
        return false;
    }

    const int h = gray_image.rows;
    const int w = gray_image.cols;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

            float val = gray_image.at<float>(ny, nx);
            if (val <= threshold)
            {
                return true;
            }
        }
    }
    return false;
}

// 稀疏 NMS: 按分数排序, 贪心保留, 抑制半径内低分点
inline void applySparseNmsByRadius(std::vector<cv::KeyPoint> &keypoints,
                                   std::vector<float>   &scores,
                                   std::vector<int64_t> &indices,
                                   int nmsRadius)
{
    if (nmsRadius <= 0 || keypoints.empty()) return;
    const float r2 = static_cast<float>(nmsRadius * nmsRadius);

    std::vector<size_t> sortIdx(keypoints.size());
    for (size_t i = 0; i < sortIdx.size(); ++i) sortIdx[i] = i;
    std::sort(sortIdx.begin(), sortIdx.end(),
              [&](size_t a, size_t b)
              { return scores[a] > scores[b]; });

    std::vector<cv::KeyPoint> keptKpts;
    std::vector<float> keptScores;
    std::vector<int64_t> keptIndices;
    keptKpts.reserve(keypoints.size());
    keptScores.reserve(keypoints.size());
    keptIndices.reserve(keypoints.size());

    for (size_t i = 0; i < sortIdx.size(); ++i)
    {
        size_t idx = sortIdx[i];
        const auto &p = keypoints[idx];
        bool suppressed = false;
        for (const auto &kk : keptKpts)
        {
            const float dx = p.pt.x - kk.pt.x;
            const float dy = p.pt.y - kk.pt.y;
            if (dx * dx + dy * dy <= r2)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
        {
            keptKpts.push_back(keypoints[static_cast<size_t>(idx)]);
            keptScores.push_back(scores[static_cast<size_t>(idx)]);
            keptIndices.push_back(indices[static_cast<size_t>(idx)]);
        }
    }

    keypoints.swap(keptKpts);
    scores.swap(keptScores);
    indices.swap(keptIndices);
}
