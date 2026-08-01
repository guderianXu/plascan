#pragma once

/**
 * @file BundleAdjustCeres.h
 * @brief Ceres BA 后端的内部入口和能力探测。
 *
 * Ceres 后端负责联合相机/点优化、共享焦距、位姿先验及物方约束。`requestGpu`
 * 只表示请求 CUDA 线性代数；求解计划仍可根据问题规模和显存预算回退 CPU。
 */

#include "BundleAdjust.h"

namespace xjw::detail
{

/// 当前构建是否链接 Ceres。
bool isCeresBackendCompiled();

/// 当前 Ceres 是否带可用的 CUDA dense linear algebra 支持。
bool isCeresCudaBackendCompiled();

/**
 * @brief 构建 Ceres Problem 并执行所请求的 BA。
 *
 * `options` 必须已经通过 validateAndNormalizeBundleAdjustOptions。
 * 返回结果保留实际后端、回退原因和求解统计，随后由公共层执行统一质量门控。
 */
BAResult optimizePointsWithCeres(const std::vector<Camera> &cameras,
                                 const std::vector<BATrack> &tracks,
                                 const BAOptions &options,
                                 bool requestGpu);

} // namespace xjw::detail
