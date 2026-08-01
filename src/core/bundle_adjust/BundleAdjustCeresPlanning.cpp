#include "BundleAdjustCeresPlanning.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::detail
{
namespace
{

std::uint64_t saturatingMultiply(std::uint64_t lhs, std::uint64_t rhs)
{
    if (lhs == 0 || rhs == 0)
    {
        return 0;
    }
    if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
    {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs * rhs;
}

std::uint64_t saturatingAdd(std::uint64_t lhs, std::uint64_t rhs)
{
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs)
    {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs + rhs;
}

BACeresSolverKind cpuSolverForScale(const BAOptions &options,
                                    int variableCameraCount,
                                    bool pointOnly)
{
    // 纯点问题没有相机 Schur 块，DenseQr 更直接；联合 BA 才按可变相机数
    // 从 dense、sparse 到 iterative Schur 逐级扩展。
    if (pointOnly)
    {
        return BACeresSolverKind::DenseQr;
    }
    if (variableCameraCount <= std::max(1, options.maxDenseSchurCameras))
    {
        return BACeresSolverKind::DenseSchur;
    }
    if (variableCameraCount <= std::max(
            options.maxDenseSchurCameras,
            options.maxSparseSchurCameras))
    {
        return BACeresSolverKind::SparseSchur;
    }
    return BACeresSolverKind::IterativeSchur;
}

} // namespace

std::uint64_t estimateCeresDenseCudaBytes(int variableCameraCount,
                                         int calibrationParameterCount,
                                         int activeTrackCount,
                                         int observationCount)
{
    const std::uint64_t cameraParameters =
        static_cast<std::uint64_t>(std::max(0, variableCameraCount)) * 6ULL +
        static_cast<std::uint64_t>(std::max(0, calibrationParameterCount));
    // CUDA dense 路径的主导项是消元后的相机/标定参数稠密矩阵，空间复杂度 O(n^2)。
    const std::uint64_t reducedMatrixBytes =
        saturatingMultiply(
            saturatingMultiply(cameraParameters, cameraParameters),
            sizeof(double));

    // Dense 正规方程、分解副本和求解临时区通常会同时驻留。
    std::uint64_t estimate = saturatingMultiply(reducedMatrixBytes, 3ULL);
    estimate = saturatingAdd(
        estimate,
        saturatingMultiply(
            static_cast<std::uint64_t>(std::max(0, observationCount)),
            256ULL));
    estimate = saturatingAdd(
        estimate,
        saturatingMultiply(
            static_cast<std::uint64_t>(std::max(0, activeTrackCount)),
            192ULL));

    // 给 Ceres/CUDA 上下文、cuSOLVER 和分配器碎片保留固定余量。
    return saturatingAdd(estimate, 64ULL * 1024ULL * 1024ULL);
}

BACeresSolverPlan planCeresSolver(const BAOptions &options,
                                  int variableCameraCount,
                                  int calibrationParameterCount,
                                  int activeTrackCount,
                                  int observationCount,
                                  bool requestGpu,
                                  std::uint64_t cudaFreeBytes)
{
    BACeresSolverPlan plan;
    const bool pointOnly =
        variableCameraCount <= 0 && calibrationParameterCount <= 0;
    const BACeresSolverKind automaticCpuSolver =
        cpuSolverForScale(options, variableCameraCount, pointOnly);

    // 显式 CPU 策略立即返回；Auto 或显式 CUDA 还需要经过设备和显存门控。
    switch (options.ceresLinearSolver)
    {
    case BACeresLinearSolver::DenseSchurCpu:
        plan.solver = pointOnly
                          ? BACeresSolverKind::DenseQr
                          : BACeresSolverKind::DenseSchur;
        plan.reason = "显式选择 dense CPU";
        return plan;
    case BACeresLinearSolver::SparseSchurCpu:
        plan.solver = pointOnly
                          ? BACeresSolverKind::DenseQr
                          : BACeresSolverKind::SparseSchur;
        plan.reason = "显式选择 sparse Schur CPU";
        return plan;
    case BACeresLinearSolver::IterativeSchurCpu:
        plan.solver = pointOnly
                          ? BACeresSolverKind::DenseQr
                          : BACeresSolverKind::IterativeSchur;
        plan.reason = "显式选择 iterative Schur CPU";
        return plan;
    case BACeresLinearSolver::DenseSchurCuda:
        plan.solver = pointOnly
                          ? BACeresSolverKind::DenseQr
                          : BACeresSolverKind::DenseSchur;
        break;
    case BACeresLinearSolver::Auto:
        plan.solver = requestGpu
                          ? (pointOnly
                                 ? BACeresSolverKind::DenseQr
                                 : BACeresSolverKind::DenseSchur)
                          : automaticCpuSolver;
        break;
    }

    if (!requestGpu)
    {
        plan.useCuda = false;
        plan.solver = automaticCpuSolver;
        plan.reason = "未请求 CUDA，按问题规模选择 CPU 求解器";
        return plan;
    }

    plan.estimatedCudaBytes =
        estimateCeresDenseCudaBytes(
            variableCameraCount,
            calibrationParameterCount,
            activeTrackCount,
            observationCount);
    const double memoryFraction =
        std::clamp(options.maxCeresCudaMemoryFraction, 0.01, 1.0);
    const long double usableCudaBytes =
        static_cast<long double>(cudaFreeBytes) * memoryFraction;
    // 只使用用户允许的空闲显存比例，给 GUI 渲染、Torch 和 CUDA 上下文保留空间。
    if (cudaFreeBytes > 0 &&
        static_cast<long double>(plan.estimatedCudaBytes) > usableCudaBytes)
    {
        plan.useCuda = false;
        plan.solver = automaticCpuSolver;
        plan.reason = "预计 dense CUDA 工作集超过空闲显存预算，回退 CPU";
        return plan;
    }

    plan.useCuda = true;
    plan.reason = cudaFreeBytes > 0
                      ? "dense CUDA 工作集通过显存预算"
                      : "CUDA 空闲显存不可查询，沿用显式 GPU 请求";
    return plan;
}

const char *ceresSolverKindName(BACeresSolverKind solver)
{
    switch (solver)
    {
    case BACeresSolverKind::DenseQr:
        return "dense_qr";
    case BACeresSolverKind::DenseSchur:
        return "dense_schur";
    case BACeresSolverKind::SparseSchur:
        return "sparse_schur";
    case BACeresSolverKind::IterativeSchur:
        return "iterative_schur";
    }
    return "unknown";
}

} // namespace xjw::detail
