#pragma once

/**
 * @file BundleAdjustPlaMatrixProjection.h
 * @brief PlaMatrix BA 使用的固定/共享 Brown 内参重投影解析线性化。
 */

#include "BundleAdjust.h"

#include <array>

namespace xjw::detail::plamatrix_ba
{

/// 一条二维重投影观测的原始残差、解析雅可比和鲁棒法方程权重。
struct ObservationLinearization
{
    std::array<double, 2> residual{{0.0, 0.0}};
    std::array<double, 6> pointJacobian{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 12> cameraJacobian{{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 18> intrinsicJacobian{{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    double normalWeight = 1.0;
    double robustCost = 0.0;
};

/**
 * @brief 按 FramePinholeCamera 投影语义线性化单条观测。
 *
 * cameraJacobian 对应局部参数 `[wx, wy, wz, dCx, dCy, dCz]`，旋转增量左乘
 * camera-to-world 旋转，中心增量位于世界坐标系。返回 false 表示点不在物理前方、
 * 输入非法或投影/雅可比非有限。
 */
bool linearizeObservation(const FramePinholeCamera& camera,
                          const std::array<double, 3>& point,
                          const BAObservation& observation,
                          double huber_delta,
                          ObservationLinearization* linearization);

/// Linearize reprojection with the shared nine-parameter Brown-Conrady model.
bool linearizeObservationWithSharedIntrinsics(
    const FramePinholeCamera& camera,
    const FramePinholeCamera& reference_camera,
    const std::array<double, 9>& shared_intrinsics,
    const BAIntrinsicParameterMask& active_parameters,
    const std::array<double, 3>& point,
    const BAObservation& observation,
    double huber_delta,
    ObservationLinearization* linearization);

} // namespace xjw::detail::plamatrix_ba
