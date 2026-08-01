#pragma once

/**
 * @file AdaptiveFocalSearch.h
 * @brief 无可靠内参时对多个焦距初始化模型进行轻量择优。
 */

#include <QVector>

namespace xjw::aerial_triangulation
{

/// 一个焦距尺度候选的最小比较摘要。
struct AdaptiveFocalCandidate
{
    double focalScale = 1.0; ///< `focalPixels / max(imageWidth,imageHeight)`。
    bool success = false; ///< 单次 SfM 是否通过自身质量门控。
    int registeredImages = 0; ///< 位姿恢复覆盖数。
    int points3D = 0; ///< 稀疏点数。
    double meanReprojectionError = 0.0; ///< 像素 RMS/平均误差指标。
};

/// 兼容的简单候选择优器；完整生产排序使用 SfmSearchPolicy。
class AdaptiveFocalSearch
{
public:
    /// 返回 candidates 下标；优先成功、注册覆盖、点数和较低误差。
    static int selectBestCandidate(const QVector<AdaptiveFocalCandidate> &candidates,
                                   int totalImages);
};

} // namespace xjw::aerial_triangulation
