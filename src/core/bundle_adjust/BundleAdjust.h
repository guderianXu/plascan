#pragma once

// ============================================================
// 文件：BundleAdjust.h
// 功能：定义光束法平差（Bundle Adjustment）相关数据结构和核心类。
//
// 算法概述：
//   - Legacy CPU：采用点、相机交替优化，高斯牛顿/LM 与有限差分雅可比
//   - Ceres CPU/CUDA：联合优化三维点、相机位姿及可选标定组共享焦距
//   - Native CUDA：显式固定相机的三维点块优化，不冒充完整联合 BA
//   - 所有后端统一执行正深度、离群点、约束质量和结果可写回检查
// ============================================================

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Camera.h"

namespace xjw
{

/**
 * @brief BA 求解后端。
 *
 * LegacyCpu 为项目原有的手写 CPU/OpenMP 求解器；CeresCpu/CeresCuda 使用
 * Ceres 构建联合非线性最小二乘问题，其中 CeresCuda 仅在 Ceres 本身
 * 编译了 CUDA 支持时才会实际启用 GPU 线性代数求解；NativeCuda 当前
 * 只负责固定相机的三维点块优化。
 */
enum class BABackend
{
    Auto,
    LegacyCpu,
    CeresCpu,
    CeresCuda,
    NativeCuda,
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
};

/**
 * @brief Ceres 内部线性求解策略。
 */
enum class BACeresLinearSolver
{
    Auto,
    DenseSchurCpu,
    DenseSchurCuda,
    SparseSchurCpu,
    IterativeSchurCpu,
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

/**
 * @brief 单条观测：某个三维点在某台相机中对应的像点坐标。
 *
 * 每条观测将一个三维轨迹点与某台相机中的匹配点关联起来，
 * 是重投影误差计算的最小单元。
 */
struct BAObservation
{
    int cameraIndex = -1; ///< 观测所属相机的索引（对应 cameras 向量下标）
    double u = 0.0;       ///< 相机像平面上的 u（列）坐标（像素）
    double v = 0.0;       ///< 相机像平面上的 v（行）坐标（像素）
    double weight = 1.0;  ///< 观测置信权重，通常来自特征尺度、匹配分数或 track confidence
};

/**
 * @brief LiDAR 局部平面约束：约束 BA 三维点靠近激光点云中的局部平面。
 */
struct BALaserPlaneConstraint
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    std::array<double, 3> normal{{0.0, 0.0, 1.0}};
    double weight = 1.0;
    double initialSignedDistance = 0.0;
    int sourceFrameIndex = -1;
};

/**
 * @brief 测绘控制点软约束：约束 BA 三维点靠近已知物方控制点坐标。
 */
struct BAControlPointConstraint
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    double sigmaMeters = 1.0;
    double weight = 1.0;
    int sourceIndex = -1;
};

/**
 * @brief 比例尺/标尺软约束：约束两条 BA track 之间的物方距离。
 */
struct BAScaleBarConstraint
{
    int trackIndexA = -1;
    int trackIndexB = -1;
    double measuredDistanceMeters = 0.0;
    double sigmaMeters = 1.0;
    double weight = 1.0;
    int sourceIndex = -1;
};

/**
 * @brief 相机位姿软先验：用于已知外参不完全可靠时约束 BA 不发生无意义漂移。
 */
struct BACameraPosePrior
{
    bool enabled = false;
    std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0}};
    std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
    double positionSigmaMeters = 1.0;
    double rotationSigmaDegrees = 2.0;
};

/**
 * @brief 相机中心参考层软约束：只限制光心沿参考法向的低频漂移。
 *
 * `referenceSignedDistances` 非空时，每台相机保留进入 BA 前相对参考面的原始
 * 法向偏移，因此不会把真实弧形或起伏轨迹强制压平。空数组保留旧的零距离平面
 * 约束语义，供明确知道相机应共面的调用方使用。
 */
struct BACameraPlaneConstraint
{
    bool enabled = false;
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    std::array<double, 3> normal{{0.0, 0.0, 1.0}};
    std::vector<double> referenceSignedDistances;
    double sigmaMeters = 1.0;
    double weight = 1.0;
};

/**
 * @brief 轨迹：一个三维点及其在多幅图像中的观测集合。
 *
 * 轨迹表示多相机共观的同一个物方点，是光束法平差的核心数据单元。
 * 每条轨迹至少需要 2 个观测才能参与优化。
 */
struct BATrack
{
    std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}}; ///< 三维点的初始坐标（优化起始值）
    std::vector<BAObservation> observations;              ///< 所有相机中对该点的观测列表
    std::vector<BALaserPlaneConstraint> laserPlaneConstraints; ///< 可选 LiDAR 点到面软约束
    std::vector<BAControlPointConstraint> controlPointConstraints; ///< 可选 GCP/控制点软约束
};

/**
 * @brief 优化选项：控制光束法平差迭代策略和收敛条件的参数集。
 */
struct BAOptions
{
    BABackend backend = BABackend::LegacyCpu; ///< BA 求解后端，默认保持原有 CPU 行为
    int maxIterations = 20;       ///< 外层交替优化最大迭代轮数（点+相机各一次为一轮）
    int maxPointIterations = 12;  ///< 每轮内点位置优化的最大迭代次数
    int maxCameraIterations = 10; ///< 每轮内相机位姿优化的最大迭代次数
    bool refineCameraPose = true; ///< 是否同时优化相机位姿（false 则仅优化三维点）

    // ── 内参自标定 ────────────────────────────────────────────────────────
    /// 是否优化所有相机共享的焦距尺度。该选项面向无相机文件/无 EXIF 的空三，
    /// 当前只释放 fu/fv 的公共 scale，主点和畸变保持固定，避免弱几何下过拟合。
    bool refineSharedFocalLength = false;
    /// 是否优化标定组共享的 fy/fx 比例。仅由 Ceres 联合 BA 实现；默认关闭，
    /// 避免弱几何数据把外参误差吸收到像素宽高比中。
    bool refineSharedFocalAspectRatio = false;
    /// 是否优化标定组共享的主点偏移量。偏移相对于每台相机输入主点应用，
    /// 因而同一标定组可以保持各自分辨率对应的基础主点。
    bool refineSharedPrincipalPoint = false;
    /// 是否优化标定组共享的一、二阶径向畸变。仅 Ceres 联合 BA 实现；应在
    /// 焦距已有可信先验时使用，避免 k1/k2 与焦距、高程在弱几何下共同漂移。
    bool refineSharedRadialDistortion = false;
    /// 共享焦距相对输入焦距的最小尺度。
    double minSharedFocalScale = 0.5;
    /// 共享焦距相对输入焦距的最大尺度。
    double maxSharedFocalScale = 4.0;
    /// fy/fx 相对标定组输入中位数允许的最小/最大倍率。
    double minSharedFocalAspectScale = 0.85;
    double maxSharedFocalAspectScale = 1.18;
    /// 主点最大偏移，以标定组参考焦距的比例表示。
    double maxSharedPrincipalPointOffsetFraction = 0.08;
    /// 扩展内参自标定的弱先验标准差。主点按参考焦距比例，宽高比按 log 比例。
    double sharedPrincipalPointPriorSigmaFraction = 0.04;
    double sharedFocalAspectPriorSigma = 0.08;
    /// Brown-Conrady k1/k2 的绝对边界和零中心弱先验标准差。
    double maxSharedRadialK1Abs = 0.35;
    double maxSharedRadialK2Abs = 0.20;
    double sharedRadialK1PriorSigma = 0.15;
    double sharedRadialK2PriorSigma = 0.08;
    /// 单次 LM 试探允许的最大焦距倍率，限制内参更新步长。
    double maxSharedFocalStepScale = 1.20;
    /// 每轮外层 BA 中共享焦距优化的最大内部迭代次数。
    int maxSharedFocalIterations = 6;
    /// 每台相机所属的内参标定组。为空表示所有相机共享组 0；非空时长度必须
    /// 与 cameras 相同，同组相机共享一个焦距参数，不同组独立自标定。
    std::vector<int> cameraCalibrationGroupIds;
    /// 是否先固定焦距稳定相机/三维点，再释放各标定组焦距。
    bool stageSharedFocalRefinement = true;
    /// 分阶段自标定中用于固定焦距预热的迭代比例。
    double sharedFocalWarmupFraction = 0.35;

    double huberDelta = 3.0;        ///< Huber 损失阈值（像素），残差>delta 则降低权重以抑制粗差
    double finiteDiffEps = 1e-6;   ///< 有限差分步长（中央差分: ±eps），用于近似雅可比
    double damping = 1e-3;         ///< Levenberg-Marquardt 初始阻尼因子（自适应调整）
    double stepTolerance = 1e-8;   ///< 收敛判断阈值：步长小于此值则认为已收敛

    // ── LiDAR 点到面软约束 ────────────────────────────────────────────────
    bool enableLaserPlaneConstraints = false; ///< 是否启用 BATrack 上挂载的 LiDAR 点到面约束
    double laserPlaneWeight = 1.0;            ///< LiDAR 残差全局权重，单位相当于 1/m
    double laserHuberDeltaMeters = 0.2;       ///< LiDAR 点到面 Huber 阈值（米）

    // ── 测绘控制点软约束 ────────────────────────────────────────────────
    bool enableControlPointConstraints = false; ///< 是否启用 BATrack 上挂载的控制点约束
    double controlPointWeight = 1.0;            ///< 控制点残差全局权重
    double controlPointHuberDeltaMeters = 0.2;  ///< 控制点 3D 距离 Huber 阈值（米）

    // ── 比例尺/标尺软约束 ────────────────────────────────────────────────
    bool enableScaleBarConstraints = false;       ///< 是否启用两点间距离约束
    double scaleBarWeight = 1.0;                  ///< 比例尺残差全局权重
    double scaleBarHuberDeltaMeters = 0.2;        ///< 比例尺长度残差 Huber 阈值（米）
    std::vector<BAScaleBarConstraint> scaleBarConstraints; ///< 与 tracks 下标关联的比例尺约束

    // ── 相机位姿软先验 ───────────────────────────────────────────────────
    std::vector<BACameraPosePrior> cameraPosePriors; ///< 与 cameras 同序的可选外参软先验
    double cameraPosePriorWeight = 1000.0;           ///< 位姿先验整体权重
    double cameraPosePriorHuberDelta = 3.0;          ///< 位姿先验 Huber 阈值（归一化残差）

    // ── 航测相机中心平面软约束 ───────────────────────────────────────────
    BACameraPlaneConstraint cameraPlaneConstraint;  ///< 所有可变相机共享的光心平面约束
    double cameraPlaneHuberDelta = 3.0;              ///< 法向归一化残差的 Huber 阈值

    // ── Gauge 固定 ──────────────────────────────────────────────────────────
    /// 联合 BA 的规范处理策略。默认自动锚定，避免直接调用产生奇异法方程。
    BAGaugePolicy gaugePolicy = BAGaugePolicy::AutoAnchor;
    /// 固定这些索引对应的相机位姿（不参与 camera 优化阶段）。
    /// 仅固定一个相机只能消除整体旋转和平移；无绝对尺度约束时还必须固定第二个相机
    /// 或提供比例尺/控制点约束，才能消除完整的 7 自由度 gauge。
    std::vector<int> fixedCameraIndices;

    // ── 离群点过滤 ─────────────────────────────────────────────────────────
    /// 是否在每轮 BA 结束后自动过滤高重投影误差点。
    bool enablePointFilter = true;
    /// 过滤阈值（像素）：rmsAfter 超过此值的点会被标记为 invalid。
    double filterMaxReprojError = 2.5;  ///< 重投影误差上限（像素），超过此值的点标记为无效
    /// 过滤倍率：同时过滤 rmsAfter > filterSigmaFactor × 全局中位数 rms 的点。
    /// 设为 0 则禁用此规则，仅使用固定阈值。
    double filterSigmaFactor = 3.0;
    // ── 并行计算 ───────────────────────────────────────────
    /// OpenMP 线程数；0 表示使用系统默认（OMP_NUM_THREADS 或 CPU 核心数）。
    int numThreads = 0;
    /// 是否把每一轮 BA 迭代写入 INFO 日志；候选粗筛关闭以避免日志洪泛。
    bool logIterationProgress = true;

    // ── Ceres/GPU 后端 ─────────────────────────────────────────────────────
    /// Ceres CUDA 求解使用的 GPU 设备 ID。当前仅在 Ceres 编译了 CUDA 支持时生效。
    int ceresCudaDevice = 0;
    /// 自研 CUDA BA 使用的 GPU 设备 ID。
    int nativeCudaDevice = 0;
    /// native_cuda 每个点块 LM step 允许的最大三维位移范数。
    double nativeCudaMaxPointStepNorm = 1.0;
    /// 参考 COLMAP：问题太小时 GPU 数据搬运开销通常不划算，低于该相机数则回退 Ceres CPU。
    int minCeresCudaCameras = 50;
    /// 低于该观测数时，即使 CUDA 可用也优先使用 CPU，避免 GPU 调度/搬运开销大于收益。
    int minCeresCudaObservations = 500000;
    /// 低于该观测数时，Auto 下的 Ceres CPU 通常也不如 legacy 交替 BA 划算。
    int minCeresCpuObservations = 50000;
    /// point-only Ceres 使用 dense QR；超出该观测规模时默认回退 legacy，避免大矩阵内存/稳定性问题。
    int maxCeresPointOnlyObservations = 100000;
    /// Ceres 线性求解策略。Auto 在 CPU 上按问题规模选择 Dense/Sparse/Iterative
    /// Schur；CUDA 当前使用 Dense Schur/Dense QR，并受显存预算门禁保护。
    BACeresLinearSolver ceresLinearSolver = BACeresLinearSolver::Auto;
    /// Auto 模式下使用 DENSE_SCHUR 的最大可变相机数量。
    int maxDenseSchurCameras = 200;
    /// Auto 模式下使用 SPARSE_SCHUR 的最大可变相机数量；更大问题使用 ITERATIVE_SCHUR。
    int maxSparseSchurCameras = 2000;
    /// Ceres CUDA 预计工作集最多占当前空闲显存的比例，超限时回退 CPU 求解器。
    double maxCeresCudaMemoryFraction = 0.70;
    /// 请求 Ceres/GPU 后端不可用时是否回退到可用后端。
    bool allowBackendFallback = true;
    /// Auto 后端是否启用质量门控；显式指定 Ceres/CUDA 时不自动替用户回退质量。
    bool enableBackendQualityGate = true;
    /// Auto 候选后端允许的最大 RMS 增长倍率；超过则回退 legacy。
    double maxAcceptedRmsGrowth = 1.25;
    /// Auto 候选后端最少有效 track 比例；低于该比例则回退 legacy。
    double minAcceptedValidTrackRatio = 0.60;
    /// LiDAR/GCP/比例尺等物方约束允许的最大 RMS 增长倍率。
    double maxAcceptedConstraintRmsGrowth = 1.25;
    /// Auto 候选为 Ceres/CUDA 时是否额外运行一次 legacy 对照。
    /// 默认关闭，避免每轮 BA 执行两套完整求解；基准测试或诊断时可显式开启。
    bool compareAutoBackendWithLegacy = false;

    // ── 任务控制 ──────────────────────────────────────────────────────────
    /// 外部取消标志；GUI/CLI 长任务可在外层迭代边界中止 BA。
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    /// 外层迭代进度回调，返回 false 表示调用方要求停止后续迭代。
    std::function<bool(int currentIteration, int maxIterations, double avgRms, int validPoints)> progressCallback;
};

/**
 * @brief 单个轨迹/三维点的优化结果。
 *
 * 记录优化前后重投影 RMS 误差及收敛状态，方便谨歌分析。
 */
struct BARefinedPoint
{
    bool valid = false;                              ///< 优化结果是否有效（点坐标有限且 rmsAfter 有限）
    bool converged = false;                          ///< 是否在容差内收敛
    std::array<double, 3> point{{0.0, 0.0, 0.0}};   ///< 优化后的三维坐标
    double rmsBefore = 0.0;  ///< 优化前重投影 RMS 误差（像素）
    double rmsAfter = 0.0;   ///< 优化后重投影 RMS 误差（像素）
    int iterations = 0;      ///< 实际迭代次数
};

/**
 * @brief 整个光束法平差的综合结果结构。
 *
 * 包含所有轨迹的优化结果、全局统计指标和优化后的相机列表。
 */
struct BAResult
{
    BABackend requestedBackend = BABackend::LegacyCpu; ///< 调用方请求的 BA 后端
    BABackend usedBackend = BABackend::LegacyCpu;      ///< 实际执行的 BA 后端
    BASolveStatus solveStatus = BASolveStatus::NotRun; ///< 求解终止状态
    bool solutionUsable = false;                       ///< 相机和点结果是否允许写回重建
    bool usedGpu = false;                              ///< 本次 BA 是否实际启用了 GPU 求解
    bool backendFallback = false;                      ///< 请求后端不可用时是否发生回退
    std::string backendMessage;                        ///< 后端选择/回退说明，便于 GUI 和日志展示
    std::string backendSelectionReason;                ///< Auto 后端选择、拒绝或回退原因
    bool qualityGateRejected = false;                  ///< Auto 候选后端是否被质量门控拒绝
    std::string qualityGateMessage;                    ///< 质量门控拒绝细节
    double validTrackRatio = 0.0;                      ///< optimizedTracks / totalTracks
    std::string ceresLinearSolverName = "none";        ///< Ceres 实际线性求解器名称，legacy 为 none
    std::uint64_t ceresEstimatedCudaBytes = 0;         ///< Ceres dense CUDA 工作集保守估算字节数
    std::uint64_t ceresCudaFreeBytes = 0;              ///< 求解前查询到的 CUDA 空闲显存
    double setupSeconds = 0.0;                         ///< Ceres 问题构建耗时或 legacy 前处理耗时
    double solveSeconds = 0.0;                         ///< 非线性求解主体耗时
    double totalSeconds = 0.0;                         ///< BA 后端总耗时
    int observationCount = 0;                          ///< 实际进入当前后端的有效观测数
    double nativeCudaInitialCost = 0.0;                ///< native_cuda 优化前加权重投影代价
    double nativeCudaFinalCost = 0.0;                  ///< native_cuda 优化后加权重投影代价
    int nativeCudaAcceptedSteps = 0;                   ///< native_cuda 接受的 LM trial step 数
    int nativeCudaRejectedSteps = 0;                   ///< native_cuda 拒绝的 LM trial step 数
    int nativeCudaActiveCameras = 0;                   ///< native_cuda 工作集中的活动相机数
    int nativeCudaActiveTracks = 0;                    ///< native_cuda 工作集中的活动 track 数
    int nativeCudaActiveObservations = 0;              ///< native_cuda 工作集中的活动观测数
    double nativeCudaUploadSeconds = 0.0;              ///< native_cuda H2D 上传和设备缓冲准备耗时
    double nativeCudaKernelSeconds = 0.0;              ///< native_cuda CUDA kernel 同步耗时
    double nativeCudaDownloadSeconds = 0.0;            ///< native_cuda D2H 结果回传耗时
    double nativeCudaHostCostSeconds = 0.0;            ///< native_cuda CPU 侧代价统计耗时
    double nativeCudaDeviceSelectSeconds = 0.0;        ///< native_cuda cudaSetDevice 耗时
    double nativeCudaStagingSeconds = 0.0;             ///< native_cuda CPU 侧设备输入打包耗时
    double nativeCudaReleaseSeconds = 0.0;             ///< native_cuda 设备缓冲释放耗时

    int totalTracks = 0;        ///< 输入轨迹总数
    int optimizedTracks = 0;    ///< 成功优化的轨迹数
    double meanRmsBefore = 0.0; ///< 优化前所有轩迹重投影 RMS 的均値
    double meanRmsAfter = 0.0;  ///< 优化后所有轨迹重投影 RMS 的均値
    int refinedCameraCount = 0; ///< 最后一轮中实际被更新的相机数量
    int refinedIntrinsicCount = 0;        ///< 发生共享焦距更新的相机数量
    int refinedCalibrationGroupCount = 0; ///< 实际参与焦距优化的标定组数量
    int selfCalibrationStagesRun = 0;     ///< 本次自标定实际执行的求解阶段数
    double refinedSharedFocalScale = 1.0; ///< 优化后焦距相对输入焦距的平均尺度
    double refinedSharedFocalAspectScale = 1.0; ///< 优化后 fy/fx 相对输入比例的平均倍率
    double refinedSharedPrincipalOffsetX = 0.0; ///< 优化后的标定组平均主点 X 偏移（像素）
    double refinedSharedPrincipalOffsetY = 0.0; ///< 优化后的标定组平均主点 Y 偏移（像素）
    double refinedSharedRadialK1 = 0.0; ///< 优化后的标定组平均 Brown-Conrady k1。
    double refinedSharedRadialK2 = 0.0; ///< 优化后的标定组平均 Brown-Conrady k2。

    int laserConstraintCount = 0;          ///< 参与统计/优化的 LiDAR 点到面约束数量
    double laserRmsBeforeMeters = 0.0;     ///< 优化前 LiDAR 点到面 RMS（米）
    double laserRmsAfterMeters = 0.0;      ///< 优化后 LiDAR 点到面 RMS（米）
    double laserMedianBeforeMeters = 0.0;  ///< 优化前 LiDAR 点到面绝对距离中位数（米）
    double laserMedianAfterMeters = 0.0;   ///< 优化后 LiDAR 点到面绝对距离中位数（米）

    int controlPointConstraintCount = 0;       ///< 参与统计/优化的控制点约束数量
    double controlPointRmsBeforeMeters = 0.0;  ///< 优化前控制点 3D RMS（米）
    double controlPointRmsAfterMeters = 0.0;   ///< 优化后控制点 3D RMS（米）

    int scaleBarConstraintCount = 0;       ///< 参与统计/优化的比例尺约束数量
    double scaleBarRmsBeforeMeters = 0.0;  ///< 优化前比例尺长度 RMS（米）
    double scaleBarRmsAfterMeters = 0.0;   ///< 优化后比例尺长度 RMS（米）

    std::vector<BARefinedPoint> points;   ///< 每条轨迹对应的点优化结果（与输入 tracks 索引一一对应）
    std::vector<Camera> refinedCameras;   ///< 优化后的相机列表（与输入 cameras 長度相同）
};

/**
 * @brief 光束法平差核心类（静态接口）。
 *
 * 提供一个静态方法 optimizePoints，负责接收相机列表和轨迹列表，
 * 返回优化后的点坐标、相机位姿及统计信息。
 */
class BundleAdjust
{
public:
    /// 返回指定 BA 后端在当前编译产物中是否可用。
    static bool isBackendAvailable(BABackend backend);

    /// 返回后端稳定名称，供日志、JSON 和测试输出使用。
    static const char *backendName(BABackend backend);

    /// 返回求解状态稳定名称，供日志、JSON 和测试输出使用。
    static const char *solveStatusName(BASolveStatus status);

    /// 返回后端当前实现的真实能力，不能根据名称推断 CUDA 后端一定支持联合 BA。
    static BABackendCapabilities backendCapabilities(BABackend backend);

    /// 统计 BA 实际可用的问题规模。
    static BAProblemStats summarizeProblem(const std::vector<Camera> &cameras,
                                           const std::vector<BATrack> &tracks);

    /// 根据问题规模与配置选择实际执行后端。
    static BABackend selectBackendForProblem(const BAProblemStats &stats,
                                             const BAOptions &options);

    /// 返回自动后端及机器可读原因，供日志解释 CPU/CUDA 选择。
    static BABackendDecision decideBackendForProblem(const BAProblemStats &stats,
                                                      const BAOptions &options);

    /**
     * @brief 执行光束法平差优化（交替优化点位置和相机位姿）。
     *
     * 优化流程：
     *   1. 初始化点集，计算前置重投影误差
     *   2. 外层迭代 maxIterations 轮：
     *      a) 对每条轨迹优化三维点坐标
     *      b) （可选）对每台相机优化 6-DOF 位姿
     *   3. 收集全局统计数据
     *
     * @param cameras  初始相机列表（优化过程中将作副本修改）
     * @param tracks   轨迹列表，每条轨迹包含初始三维点和多相机观测
     * @param options  优化选项（可选，默认使用 BAOptions）
     * @return         BAResult 结个，包含优化后点坐标、相机位姿及误差统计
     */
    static BAResult optimizePoints(const std::vector<Camera> &cameras,
                                   const std::vector<BATrack> &tracks,
                                   const BAOptions &options = BAOptions());
};

} // namespace xjw
