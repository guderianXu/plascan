#pragma once

#include "BundleAdjustProblem.h"
#include "BundleAdjustTypes.h"
#include "FramePinholeCamera.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw
{

    /**
     * @brief 单个轨迹/三维点的优化结果。
     *
     * 记录优化前后重投影 RMS 误差及收敛状态，方便质量分析。
     */
    struct BARefinedPoint
    {
        bool valid = false;                           ///< 优化结果是否有效（点坐标有限且 rmsAfter 有限）
        bool converged = false;                       ///< 是否在容差内收敛
        std::array<double, 3> point{{0.0, 0.0, 0.0}}; ///< 优化后的三维坐标
        double rmsBefore = 0.0;                       ///< 优化前重投影 RMS 误差（像素）
        double rmsAfter = 0.0;                        ///< 优化后重投影 RMS 误差（像素）
        int iterations = 0;                           ///< 实际迭代次数
    };

    /**
     * @brief 独立行星激光测高 shot 的优化和质量统计结果。
     *
     * 数组顺序与 BAOptions::laserRangeConstraints 严格一致，shotId、时间和 sourceIndex
     * 原样保留，便于上层报告和将来的时变轨迹扩展。
     */
    struct BARefinedLaserRangeShot
    {
        bool valid = false;
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        BALaserPointMode pointMode = BALaserPointMode::Unspecified;
        std::string shotId;
        double ephemerisTimeSeconds = 0.0;
        int sourceIndex = -1;
        double computedRangeBeforeMeters = 0.0;
        double computedRangeAfterMeters = 0.0;
        double residualBeforeMeters = 0.0;
        double residualAfterMeters = 0.0;
        double normalizedResidualAfter = 0.0;
    };

    /**
     * @brief 整个光束法平差的综合结果结构。
     *
     * 包含所有轨迹的优化结果、全局统计指标和优化后的相机列表。
     */
    struct BAResult
    {
        BABackend requestedBackend = BABackend::LegacyCpu;  ///< 调用方请求的 BA 后端
        BABackend usedBackend = BABackend::LegacyCpu;       ///< 实际执行的 BA 后端
        BASolveStatus solveStatus = BASolveStatus::NotRun;  ///< 求解终止状态
        bool solutionUsable = false;                        ///< 相机和点结果是否允许写回重建
        bool usedGpu = false;                               ///< 本次 BA 是否实际启用了 GPU 求解
        bool backendFallback = false;                       ///< 请求后端不可用时是否发生回退
        std::string backendMessage;                         ///< 后端选择/回退说明，便于 GUI 和日志展示
        std::string backendSelectionReason;                 ///< Auto 后端选择、拒绝或回退原因
        bool qualityGateRejected = false;                   ///< Auto 候选后端是否被质量门控拒绝
        std::string qualityGateMessage;                     ///< 质量门控拒绝细节
        double validTrackRatio = 0.0;                       ///< optimizedTracks / totalTracks
        double setupSeconds = 0.0;                          ///< 问题构建或前处理耗时
        double solveSeconds = 0.0;                          ///< 非线性求解主体耗时
        double postprocessSeconds = 0.0;                    ///< 统一 RMS、正深度与离群点质量检查耗时
        double totalSeconds = 0.0;                          ///< BA 后端总耗时
        int observationCount = 0;                           ///< 实际进入当前后端的有效观测数
        double plaMatrixInitialCost = 0.0;                  ///< PlaMatrix 初始鲁棒目标函数值
        double plaMatrixFinalCost = 0.0;                    ///< PlaMatrix 最终鲁棒目标函数值
        int plaMatrixAcceptedSteps = 0;                     ///< PlaMatrix 接受的 LM trial step 数
        int plaMatrixRejectedSteps = 0;                     ///< PlaMatrix 拒绝的 LM trial step 数
        int plaMatrixLinearizations = 0;                    ///< 实际构建 Jacobian/法方程的次数
        int plaMatrixObjectiveEvaluations = 0;              ///< 完整目标函数遍历次数（含线性化）
        int plaMatrixRejectedInitialTracks = 0;             ///< PlaMatrix 初始 gross gate 拒绝的 track 数
        std::string plaMatrixLinearSolverName = "none";     ///< CPU/CUDA/OpenCL Schur PCG 名称
        std::string plaMatrixPreconditionerName = "none";   ///< 实际使用的 Schur 预条件器
        std::string plaMatrixDeviceName;                    ///< 实际执行 Schur PCG 的加速设备名
        int plaMatrixLinearIterations = 0;                  ///< 所有 LM trial 累计 PCG 迭代数
        int plaMatrixDenseFallbacks = 0;                    ///< 稠密 Schur 失败后切换 PCG 的次数
        std::string plaMatrixDenseFallbackMessage;          ///< 最近一次稠密 Schur 回退原因
        double plaMatrixAssemblySeconds = 0.0;              ///< Jacobian 与块法方程装配累计耗时
        double plaMatrixObjectiveSeconds = 0.0;             ///< 非线性 trial 目标函数复算累计耗时
        double plaMatrixTrialStateSeconds = 0.0;            ///< trial 状态复制与增量应用累计耗时
        double plaMatrixLinearToleranceMinimum = 0.0;       ///< 实际使用的最小 PCG 相对容差
        double plaMatrixLinearToleranceMaximum = 0.0;       ///< 实际使用的最大 PCG 相对容差
        int plaMatrixSchurPatternBuilds = 0;                ///< 加速后端 Schur CSR pattern 构建次数
        int plaMatrixSchurPatternReuses = 0;                ///< 加速后端 Schur CSR pattern 复用次数
        bool plaMatrixSchurAssemblyOnDevice = false;        ///< Schur CSR 数值是否由 CUDA/OpenCL 设备装配
        bool plaMatrixMixedPrecisionUsed = false;           ///< 是否使用 FP32 初解和 FP64 校正
        double plaMatrixSmallBlockInverseSeconds = 0.0;     ///< 小型消元/预条件块求逆累计耗时
        double plaMatrixSchurAccumulationSeconds = 0.0;     ///< Schur 数值累加累计耗时
        double plaMatrixCsrConversionSeconds = 0.0;         ///< CSR 构建及稠密转换累计耗时
        double plaMatrixSchurAssemblySeconds = 0.0;         ///< Schur 完整装配累计耗时
        double plaMatrixCholeskyFactorizationSeconds = 0.0; ///< 稠密 Cholesky 分解累计耗时
        double plaMatrixTriangularSolveSeconds = 0.0;       ///< 稠密三角回代累计耗时
        double plaMatrixResidualCheckSeconds = 0.0;         ///< 线性残差检查累计耗时
        double plaMatrixLinearSolveSeconds = 0.0;           ///< 所有 LM trial 累计线性求解耗时
        double plaMatrixBackSubstitutionSeconds = 0.0;      ///< 消元变量回代累计耗时

        int totalTracks = 0;                        ///< 输入轨迹总数
        int optimizedTracks = 0;                    ///< 成功优化的轨迹数
        double meanRmsBefore = 0.0;                 ///< 优化前所有轨迹重投影 RMS 的均值
        double meanRmsAfter = 0.0;                  ///< 优化后所有轨迹重投影 RMS 的均值
        int refinedCameraCount = 0;                 ///< 最后一轮中实际被更新的相机数量
        int refinedIntrinsicCount = 0;              ///< 至少一项共享内参发生更新的相机数量
        int refinedCalibrationGroupCount = 0;       ///< 实际参与共享内参优化的标定组数量
        int selfCalibrationStagesRun = 0;           ///< 本次自标定实际执行的求解阶段数
        double refinedSharedFocalScale = 1.0;       ///< 优化后焦距相对稳定参考焦距的平均尺度
        double refinedSharedFocalAspectScale = 1.0; ///< 优化后 fy/fx 相对稳定参考比例的平均倍率
        double refinedSharedPrincipalOffsetX = 0.0; ///< 相对稳定参考的标定组平均主点 X 偏移（像素）
        double refinedSharedPrincipalOffsetY = 0.0; ///< 相对稳定参考的标定组平均主点 Y 偏移（像素）
        double refinedSharedRadialK1 = 0.0;         ///< 优化后的标定组平均 Brown-Conrady k1。
        double refinedSharedRadialK2 = 0.0;         ///< 优化后的标定组平均 Brown-Conrady k2。
        double refinedSharedRadialK3 = 0.0;         ///< 优化后的标定组平均 Brown-Conrady k3。
        double refinedSharedTangentialP1 = 0.0;     ///< 优化后的标定组平均 Brown-Conrady p1。
        double refinedSharedTangentialP2 = 0.0;     ///< 优化后的标定组平均 Brown-Conrady p2。

        int laserConstraintCount = 0;         ///< 参与统计/优化的 LiDAR 点到面约束数量
        double laserRmsBeforeMeters = 0.0;    ///< 优化前 LiDAR 点到面 RMS（米）
        double laserRmsAfterMeters = 0.0;     ///< 优化后 LiDAR 点到面 RMS（米）
        double laserMedianBeforeMeters = 0.0; ///< 优化前 LiDAR 点到面绝对距离中位数（米）
        double laserMedianAfterMeters = 0.0;  ///< 优化后 LiDAR 点到面绝对距离中位数（米）

        int laserRangeConstraintCount = 0;      ///< 真正进入问题的独立激光测距 shot 数量
        double laserRangeRmsBeforeMeters = 0.0; ///< 优化前原始测距残差 RMS（米）
        double laserRangeRmsAfterMeters = 0.0;  ///< 优化后原始测距残差 RMS（米）

        int controlPointConstraintCount = 0;      ///< 参与统计/优化的控制点约束数量
        double controlPointRmsBeforeMeters = 0.0; ///< 优化前控制点 3D RMS（米）
        double controlPointRmsAfterMeters = 0.0;  ///< 优化后控制点 3D RMS（米）

        int scaleBarConstraintCount = 0;      ///< 参与统计/优化的比例尺约束数量
        double scaleBarRmsBeforeMeters = 0.0; ///< 优化前比例尺长度 RMS（米）
        double scaleBarRmsAfterMeters = 0.0;  ///< 优化后比例尺长度 RMS（米）

        std::vector<BARefinedPoint> points; ///< 每条轨迹对应的点优化结果（与输入 tracks 索引一一对应）
        std::vector<BARefinedLaserRangeShot> laserRangeShots; ///< 与输入独立测距 shot 一一对应
        std::vector<FramePinholeCamera> refinedCameras;       ///< 优化后的相机列表（与输入 cameras 长度相同）
    };

} // namespace xjw
