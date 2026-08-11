#pragma once

/**
 * @file BundleAdjustNativeCuda.h
 * @brief PlaScan 自有 CUDA 点优化后端的内部入口。
 *
 * 此后端针对“相机固定、仅优化三维点”的可分离 BA；它不是完整 Schur 联合 BA。
 * 不支持的物方约束和相机参数优化必须显式返回 UnsupportedConfiguration，
 * 由公共调度层决定是否回退 Ceres/CPU。
 */

#include "BundleAdjust.h"

namespace xjw::detail
{

/// 编译期是否包含 CUDA 实现；无 CUDA 构建中仍提供稳定符号。
bool isNativeCudaBackendCompiled();

/// 检查设备编号、驱动和运行时是否可用，并返回可读诊断。
bool isNativeCudaRuntimeAvailable(int deviceId, std::string *message);

/**
 * @brief 使用 native CUDA 优化独立三维点块。
 *
 * 返回值尚需经过 finalizeBundleAdjustResult 的统一质量复核；调用方不得只依据
 * CUDA 核成功标志认定摄影测量解可用。
 */
BAResult optimizePointsWithNativeCuda(const std::vector<FramePinholeCamera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options);

} // namespace xjw::detail
