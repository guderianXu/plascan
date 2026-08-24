#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace xjw
{

    /**
     * @brief BA 求解后端。
     *
     * LegacyCpu 为项目原有的手写 CPU/OpenMP 求解器；PlaMatrixCpu/Cuda/OpenCl
     * 共用完整联合 BA，只切换约化 Schur 的线性代数后端。
     */
    enum class BABackend
    {
        Auto,
        LegacyCpu,
        PlaMatrixCpu,
        PlaMatrixCuda,
        PlaMatrixOpenCl,
    };

    /**
     * @brief BA 求解终止状态。
     *
     * solutionUsable 只在 Success 或可接受的 NoConvergence 下为 true。调用方必须先
     * 检查该状态，再把相机和三维点写回重建，避免取消或数值失败后的中间状态污染工程。
     */
    enum class BASolveStatus
    {
        NotRun,
        Success,
        NoConvergence,
        Cancelled,
        InvalidInput,
        UnsupportedConfiguration,
        BackendUnavailable,
        NumericalFailure,
    };

    /**
     * @brief 后端真实能力声明，用于调度器拒绝能力不完整的执行路径。
     */
    struct BABackendCapabilities
    {
        bool optimizesPoints = false;
        bool refinesCameraPose = false;
        bool refinesSharedFocalLength = false;
        bool refinesSharedFocalAspectRatio = false;
        bool refinesSharedPrincipalPoint = false;
        bool refinesSharedRadialDistortion = false;
        bool supportsSoftConstraints = false;
        bool supportsLaserRangeConstraints = false;
    };

    /**
     * @brief 联合 BA 的相似变换规范（gauge）处理策略。
     *
     * AutoAnchor 会在缺少绝对约束时自动固定两台具有非退化基线的相机；
     * RequireExplicitGauge 用于严格检查调用方是否完整提供规范约束；
     * CallerManaged 用于 SfM 协调器等在求解后自行执行 Sim(3) 归一化的调用方。
     */
    enum class BAGaugePolicy
    {
        AutoAnchor,
        RequireExplicitGauge,
        CallerManaged,
    };

    /**
     * @brief 共享相机模型参数在联合 BA 内参块中的稳定索引。
     *
     * 顺序必须与 BundleAdjustProjection::projectWithPoseDeltaAndSharedIntrinsics
     * 保持一致。显式枚举让自适应相机模型可以逐参数冻结，而不是把所有高阶
     * Brown-Conrady 系数作为一个不可分割的开关。
     */
    enum class BAIntrinsicParameter : std::uint8_t
    {
        FocalLength = 0,
        FocalAspectRatio,
        PrincipalPointX,
        PrincipalPointY,
        RadialK1,
        RadialK2,
        RadialK3,
        TangentialP1,
        TangentialP2,
        Count,
    };

    inline constexpr std::size_t kBAIntrinsicParameterCount = static_cast<std::size_t>(BAIntrinsicParameter::Count);
    using BAIntrinsicParameterMask = std::array<bool, kBAIntrinsicParameterCount>;

    /**
     * @brief BA 可用问题规模摘要，用于自动后端选择和日志诊断。
     *
     * track 和 observation 只统计具有有限初值、正权重有限像点且至少被两台
     * 不同相机观测的轨迹，避免无效输入抬高 CPU/CUDA 选择阈值。
     */
    struct BAProblemStats
    {
        int cameraCount = 0;
        int trackCount = 0;
        int observationCount = 0;
    };

    struct BABackendDecision
    {
        BABackend backend = BABackend::LegacyCpu;
        std::string reason;
    };

} // namespace xjw
