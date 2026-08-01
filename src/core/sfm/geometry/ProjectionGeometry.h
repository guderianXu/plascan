#pragma once

/**
 * @file ProjectionGeometry.h
 * @brief SfM 质量评估使用的统一投影包装。
 *
 * 正常路径要求 Camera 的正深度约定成立。为读取历史工程中深度轴元数据不完整的
 * 相机，本层保留 signed fallback 并显式标记；调用方可在最终质量门控中拒绝该结果，
 * 而不是悄悄把相机后方点当成正常正深度点。
 */

#include "Camera.h"

#include <array>

namespace xjw
{

struct ProjectionResult
{
    bool success = false; ///< 至少获得了有限像素投影。
    bool usedSignedFallback = false; ///< true 表示严格正深度投影失败。
    std::array<double, 2> pixel{{0.0, 0.0}}; ///< 预测像素 [u,v]。
    double positiveDepth = 0.0; ///< 按 Camera 前向轴定义的有符号深度。
};

/// 先执行严格正深度投影，失败时再尝试仅用于诊断/兼容的有符号投影。
ProjectionResult projectForReprojection(const Camera &camera,
                                        const std::array<double, 3> &worldPoint);

/// 返回二维欧氏像素误差；无法投影时返回正无穷。
double reprojectionErrorPx(const Camera &camera,
                           const std::array<double, 3> &worldPoint,
                           const std::array<double, 2> &observedPixel);

} // namespace xjw
