#pragma once

/**
 * @file TriangulationQuality.h
 * @brief BA/SfM 共用的交会几何质量计算。
 *
 * 本层集中使用 CameraBaseline、统一投影模型和 Intersection，避免各调用点
 * 分别实现基线角、正深度或相机方向回退而产生不一致。
 */

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <limits>
#include <vector>

namespace xjw
{

struct PairIntersectionCandidate
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}}; ///< 世界坐标交会点。
    double rmsReprojectionPx = std::numeric_limits<double>::infinity(); ///< 双视 RMS。
    bool valid = false; ///< 至少一个方向假设产生有限点和有限重投影误差。
};

/**
 * @brief 返回轨迹所有有效相机对中的最小三角化角。
 *
 * 采用最小值是保守质量指标：任一参与观测的极弱基线都会降低该 track 可信度。
 * 没有两台正深度相机形成有效几何时返回 0。
 */
double minimumTriangulationAngleDeg(const std::vector<Camera> &cameras,
                                    const BATrack &track,
                                    const std::array<double, 3> &worldPoint);

/// 计算同一世界点在两幅影像上的均方根像素重投影误差。
double pairRmsReprojectionErrorPx(const Camera &cameraA,
                                  const std::array<double, 2> &pixelA,
                                  const Camera &cameraB,
                                  const std::array<double, 2> &pixelB,
                                  const std::array<double, 3> &worldPoint);

/**
 * @brief 在历史深度轴可能错误时尝试四种双相机方向组合并选择最低 RMS 候选。
 *
 * 该回退只用于生成可继续优化的初值，不会修改输入 Camera。最终解仍必须经过
 * 正深度、基线角和重投影门控，不能把“某方向可投影”等同于相机元数据正确。
 */
PairIntersectionCandidate triangulatePairWithDirectionFallback(
    const Camera &cameraA,
    const std::array<double, 2> &pixelA,
    const Camera &cameraB,
    const std::array<double, 2> &pixelB);

} // namespace xjw
