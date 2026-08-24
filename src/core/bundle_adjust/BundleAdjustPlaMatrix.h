#pragma once

/**
 * @file BundleAdjustPlaMatrix.h
 * @brief PlaMatrix 相机、三维点、共享 Brown 内参与物方约束联合 BA 后端入口。
 */

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"
#include "BundleAdjustResult.h"
#include "BundleAdjustTypes.h"

#include <string>

namespace xjw::detail
{

/// 检查 PlaMatrix CPU/CUDA/OpenCL 线性求解后端及所选设备是否可用。
bool isPlaMatrixBackendAvailable(BABackend backend,
                                 int device_index,
                                 std::string* message = nullptr);

/// 使用 PlaMatrix 块法方程、Schur 消元和 LM 阻尼执行完整联合 BA。
BAResult optimizePointsWithPlaMatrix(const std::vector<FramePinholeCamera>& cameras,
                                     const std::vector<BATrack>& tracks,
                                     const BAOptions& options);

} // namespace xjw::detail
