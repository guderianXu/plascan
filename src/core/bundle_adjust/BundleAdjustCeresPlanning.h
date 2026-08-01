#pragma once

/**
 * @file BundleAdjustCeresPlanning.h
 * @brief 根据问题规模和显存预算选择 Ceres 线性求解器。
 *
 * 这里只生成计划，不创建 Ceres Problem，也不访问 BA 状态。将策略独立出来后，
 * 后端选择可以在没有 GPU 的单元测试中稳定验证。
 */

#include "BundleAdjust.h"

#include <cstdint>
#include <string>

namespace xjw::detail
{

/// PlaScan 使用的 Ceres 求解器类别；名称与 Ceres LinearSolverType 一一映射。
enum class BACeresSolverKind
{
    DenseQr,
    DenseSchur,
    SparseSchur,
    IterativeSchur,
};

struct BACeresSolverPlan
{
    BACeresSolverKind solver = BACeresSolverKind::DenseSchur; ///< 最终选中的线性求解器。
    bool useCuda = false; ///< true 时为 Ceres dense 求解器启用 CUDA。
    std::uint64_t estimatedCudaBytes = 0; ///< 保守峰值显存估计。
    std::string reason; ///< 显式选择、自动选择或回退原因。
};

/**
 * @brief 保守估计 Ceres dense CUDA 求解的峰值工作集。
 *
 * 估算包含约化相机法方程、分解临时区、观测雅可比和点块暂存。它不是 CUDA
 * 分配器的精确值，只用于在已知会爆显存时提前选择 CPU 稀疏/迭代求解器。
 *
 * @param variableCameraCount 需要优化位姿的相机数量，每台按 6 自由度估算。
 * @param calibrationParameterCount 共享或分组内参的总参数数。
 * @param activeTrackCount 进入问题的三维点块数量。
 * @param observationCount 二维观测残差块数量。
 */
std::uint64_t estimateCeresDenseCudaBytes(int variableCameraCount,
                                         int calibrationParameterCount,
                                         int activeTrackCount,
                                         int observationCount);

/**
 * @brief 根据问题规模、用户策略和当前空闲显存生成 Ceres 求解计划。
 *
 * Auto 策略下，小规模联合 BA 使用 DenseSchur，中等规模使用 SparseSchur，
 * 更大规模使用 IterativeSchur；纯点优化使用 DenseQr。请求 GPU 时仅 dense
 * 路径可启用 CUDA，并需通过 `maxCeresCudaMemoryFraction` 显存门控。
 */
BACeresSolverPlan planCeresSolver(const BAOptions &options,
                                  int variableCameraCount,
                                  int calibrationParameterCount,
                                  int activeTrackCount,
                                  int observationCount,
                                  bool requestGpu,
                                  std::uint64_t cudaFreeBytes);

/// 返回稳定的日志/测试名称，不返回 Ceres 枚举文本。
const char *ceresSolverKindName(BACeresSolverKind solver);

} // namespace xjw::detail
