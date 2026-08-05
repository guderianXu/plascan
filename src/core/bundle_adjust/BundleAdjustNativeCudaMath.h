#pragma once

/**
 * @file BundleAdjustNativeCudaMath.h
 * @brief native CUDA 投影与线性化公式的 CPU 参考实现。
 *
 * 该实现与设备核保持同一坐标/畸变/鲁棒权重约定，用于单元测试、主机代价复核和
 * CUDA 结果诊断。修改设备公式时必须同步修改此处并运行投影与后端一致性测试。
 */

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct ProjectionResult
{
    bool ok = false; ///< 点位于相机前方且投影结果有限。
    double pixel[2] = {0.0, 0.0}; ///< 预测像素 [u, v]。
};

/**
 * @brief 单条观测的加权线性化结果。
 *
 * residual 和雅可比均已乘以 `sqrt(observationWeight * huberWeight)`。
 * `jp` 为按行存储的 2x3 点雅可比。
 */
struct ObservationLinearization
{
    double residual[2] = {0.0, 0.0};
    double jp[6] = {0.0};
    double weightedCost = 0.0;
};

/// 使用与 Camera::projectWorldPoint 一致的正深度和 Brown-Conrady 模型投影世界点。
ProjectionResult projectHost(const HostCamera &camera, const std::array<double, 3> &point);

/// 计算像素 [u,v] 对世界点 [X,Y,Z] 的 2x3 解析雅可比，按行写入 6 个 double。
bool pointProjectionJacobianHost(const HostCamera &camera,
                                 const std::array<double, 3> &point,
                                 double jacobian[6]);

/**
 * @brief 线性化一条重投影观测。
 *
 * Huber 权重按二维残差范数计算，而不是分别裁剪 u/v 分量。返回 false 表示输入非法、
 * 点位于相机后方或投影/雅可比非有限，该观测不得进入正规方程。
 */
bool linearizeObservationHost(const HostCamera &camera,
                              const std::array<double, 3> &point,
                              double observedU,
                              double observedV,
                              double weight,
                              double huberDelta,
                              ObservationLinearization *out);

} // namespace xjw::detail::native_cuda
