#pragma once

/**
 * @file SimilarityGaugeNormalizer.h
 * @brief 在自由单目 BA 后恢复约定的平移原点和尺度规范。
 *
 * 单目重建对全局 Sim(3) 不可观。若求解器释放了相机中心，仅固定一个相机仍不能
 * 约束尺度。本工具用 BA 前两个相机中心的基线长度恢复尺度，并把锚点中心恢复到
 * 原位置；它不假设相机沿圆轨迹运动。
 */

#include "BundleAdjust.h"
#include "FramePinholeCamera.h"

#include <string>
#include <vector>

namespace xjw
{

struct SimilarityGaugeNormalizationResult
{
    bool applied = false; ///< 是否成功原子应用到全部相机和有限三维点。
    double scale = 1.0; ///< reference baseline / refined baseline。
    std::string reason; ///< 稳定机器可读结果或拒绝原因。
};

/**
 * @brief 用 BA 前的一条相机基线恢复单目重建的尺度规范。
 *
 * 第一台相机的中心恢复到 BA 前的位置，所有其它相机中心和三维点相对该中心
 * 做同一尺度变换。相机旋转和内参保持不变，因此不会改变重投影几何。
 */
SimilarityGaugeNormalizationResult normalizeSimilarityGauge(
    const std::vector<FramePinholeCamera> &referenceCameras,
    int anchorCameraIndex,
    int scaleCameraIndex,
    std::vector<FramePinholeCamera> *refinedCameras,
    std::vector<BARefinedPoint> *refinedPoints);

} // namespace xjw
