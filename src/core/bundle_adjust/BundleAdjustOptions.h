#pragma once

#include "BundleAdjustProblem.h"
#include "BundleAdjustTypes.h"
#include "FramePinholeCamera.h"

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

namespace xjw
{

    /**
     * @brief 优化选项：控制光束法平差迭代策略和收敛条件的参数集。
     */
    struct BAOptions
    {
        static constexpr int kAutoBackendPolicyVersion = 2;
        static constexpr int kDefaultMinPlaMatrixCudaCameras = 128;
        static constexpr int kDefaultMinPlaMatrixCudaObservations = 30000;
        static constexpr int kDefaultMinPlaMatrixOpenClCameras = 160;
        static constexpr int kDefaultMinPlaMatrixOpenClObservations = 50000;
        static constexpr int kDefaultMinPlaMatrixDenseCameras = 120;
        static constexpr int kDefaultMinPlaMatrixCudaDenseObservations = 150000;
        static constexpr int kDefaultMinPlaMatrixOpenClDenseObservations = 200000;
        static constexpr int kLegacyMinPlaMatrixGpuCameras = 24;
        static constexpr int kLegacyMinPlaMatrixGpuObservations = 30000;

        BABackend backend = BABackend::PlaMatrixCpu; ///< CPU 默认使用参考兼容的联合 Schur BA
        int maxIterations = 20;      ///< 联合非线性求解最大迭代数；参考 SfM 按阶段覆盖为 20 或 10
        int maxPointIterations = 12; ///< 每轮内点位置优化的最大迭代次数
        int maxCameraIterations = 10; ///< 每轮内相机位姿优化的最大迭代次数
        bool refineCameraPose = true; ///< 是否同时优化相机位姿（false 则仅优化三维点）

        // ── 内参自标定 ────────────────────────────────────────────────────────
        /// 是否优化所有相机共享的焦距尺度。该选项面向无相机文件/无 EXIF 的空三，
        /// 当前只释放 fu/fv 的公共 scale，主点和畸变保持固定，避免弱几何下过拟合。
        bool refineSharedFocalLength = false;
        /// 是否优化标定组共享的 fy/fx 比例。由联合 BA 实现；默认关闭，
        /// 避免弱几何数据把外参误差吸收到像素宽高比中。
        bool refineSharedFocalAspectRatio = false;
        /// 是否优化标定组共享的主点偏移量。偏移相对于每台相机输入主点应用，
        /// 因而同一标定组可以保持各自分辨率对应的基础主点。
        bool refineSharedPrincipalPoint = false;
        /// 是否优化标定组共享的完整 Brown-Conrady 镜头畸变
        /// (k1/k2/k3/p1/p2)。应配合焦距弱先验和分阶段
        /// 求解使用，避免镜头参数与外参在弱几何下同时漂移。
        bool refineSharedRadialDistortion = false;
        /// 是否在径向畸变自标定中继续释放 k2/k3/p1/p2。关闭时仅优化 k1，
        /// 适用于近乎平行的对地航摄块，避免高阶畸变与场景穹顶互相补偿。
        bool refineSharedHighOrderDistortion = true;
        /// 共享焦距是否来自批次一致的可信 EXIF/固定镜头先验。仅在航摄穹顶保护中
        /// 允许固定该焦距并单独估计低阶 k1；普通无标定问题仍要求焦距共同参与。
        bool hasTrustedSharedFocalPrior = false;
        /// 是否使用逐参数共享内参掩码。false 保持上述兼容开关的旧行为；true 时
        /// 兼容开关定义允许的最大模型，本掩码再冻结可靠性不足的单个参数。
        bool useSharedIntrinsicParameterMask = false;
        BAIntrinsicParameterMask sharedIntrinsicParameterMask{{true, true, true, true, true, true, true, true, true}};
        /// 共享焦距相对输入焦距的最小尺度。
        double minSharedFocalScale = 0.5;
        /// 共享焦距相对输入焦距的最大尺度。
        double maxSharedFocalScale = 4.0;
        /// fy/fx 相对标定组输入中位数允许的最小/最大倍率。
        double minSharedFocalAspectScale = 0.85;
        double maxSharedFocalAspectScale = 1.18;
        /// 主点最大偏移，以标定组参考焦距的比例表示。
        double maxSharedPrincipalPointOffsetFraction = 0.08;
        /// 扩展内参自标定的弱先验标准差。焦距按相对比例、宽高比按 log 比例，主点按参考焦距比例。
        double sharedFocalPriorSigma = 0.35;
        double sharedPrincipalPointPriorSigmaFraction = 0.04;
        double sharedFocalAspectPriorSigma = 0.08;
        /// Brown-Conrady 畸变参数的绝对边界和输入标定中心弱先验标准差。
        double maxSharedRadialK1Abs = 0.35;
        double maxSharedRadialK2Abs = 0.35;
        double maxSharedRadialK3Abs = 0.35;
        double maxSharedTangentialP1Abs = 0.02;
        double maxSharedTangentialP2Abs = 0.02;
        double sharedRadialK1PriorSigma = 0.15;
        double sharedRadialK2PriorSigma = 0.15;
        double sharedRadialK3PriorSigma = 0.15;
        double sharedTangentialP1PriorSigma = 0.01;
        double sharedTangentialP2PriorSigma = 0.01;
        /// 窄视场下把 k1/p1/p2 系数换算到统一参考像场的尺度。默认 1；
        /// 自适应模型按观测半径设置，求解器在边界和先验中一次性应用，避免多轮复合。
        double sharedLowOrderDistortionScale = 1.0;
        /// 每台相机所属的内参标定组。为空表示所有相机共享组 0；非空时长度必须
        /// 与 cameras 相同，同组相机共享一个内参参数块，不同组独立自标定。
        std::vector<int> cameraCalibrationGroupIds;
        /// 共享内参的稳定参考相机。为空时以本次输入 cameras 为参考；非空时必须与
        /// cameras 等长且 reference[i] 对应 cameras[i] 的同一像素坐标系。多轮 BA 可
        /// 更新求解初值，同时继续相对同一参考设置焦距/宽高比范围、主点偏移和弱先验；
        /// Brown-Conrady 畸变硬边界仍是绝对范围。
        std::vector<FramePinholeCamera> sharedIntrinsicReferenceCameras;
        double huberDelta = 3.0;     ///< 兼容字段；统一参考 BA 的影像残差固定使用普通最小二乘
        double finiteDiffEps = 1e-6; ///< 有限差分步长（中央差分: ±eps），用于近似雅可比
        double damping = 1e-3;       ///< 旧 LM 工程字段；参考 BA 固定从零阻尼开始
        double stepTolerance = 1e-8; ///< 旧 LM 工程字段；参考 BA 固定使用 1e-6 RMS 更新阈值

        // ── 参考 BA 数值策略 ──────────────────────────────────────────────────
        /// PlaMatrix CPU/CUDA/OpenCL 统一使用“对齐照片”兼容策略：尺度白化、
        /// 普通最小二乘、参考局部参数化、零阻尼起步和 Armijo 回溯。
        /// 自由网固定锚相机；负值表示没有参考式定长基线 gauge。
        int referenceGaugeAnchorCameraIndex = -1;
        /// 自由网定长基线的另一台相机；其光心更新后投影回固定半径球面。
        int referenceGaugeScaleCameraIndex = -1;
        /// 定长基线半径。必须有限且大于零才启用参考式 gauge。
        double referenceGaugeBaseline = 0.0;
        /// Armijo 充分下降系数和最大回溯次数，与参考流程一致。
        double referenceArmijoCoefficient = 1.0e-3;
        int referenceLineSearchSteps = 34;
        /// 参考空三主链在并行线性化阶段完成点局部 Schur 消元，并在求解后
        /// 重新线性化完成点回代。普通独立 BA 和带物方扩展约束的任务保持旧路径。
        bool useReferenceOnlineSchur = false;
        /// SfM 自标定轮次启用参考实现的参数释放过渡先验。独立 BA 默认关闭，
        /// 由增量 SfM 调度器携带上一轮真正提交的参数掩码。
        bool useReferenceCalibrationTransitionPrior = false;
        BAIntrinsicParameterMask referencePreviousIntrinsicParameterMask{};

        // ── LiDAR 点到面软约束 ────────────────────────────────────────────────
        bool enableLaserPlaneConstraints = false; ///< 是否启用 BATrack 上挂载的 LiDAR 点到面约束
        double laserPlaneWeight = 1.0;            ///< LiDAR 统计权重，通常取 1/sigma^2，单位 1/m^2
        double laserHuberDeltaMeters = 0.2;       ///< LiDAR 点到面 Huber 阈值（米）

        // ── 行星激光测高独立测距约束 ──────────────────────────────────────────
        bool enableLaserRangeConstraints = false; ///< 是否启用独立 shot 的相机-落点测距约束
        double laserRangeWeight = 1.0;            ///< 所有 shot 的全局统计权重
        double laserRangeHuberDelta = 3.0;        ///< 标准差归一化测距残差的 Huber 阈值
        std::vector<BALaserRangeConstraint> laserRangeConstraints; ///< 不属于 BATrack 的独立测距 shot

        // ── 测绘控制点软约束 ────────────────────────────────────────────────
        bool enableControlPointConstraints = false; ///< 是否启用 BATrack 上挂载的控制点约束
        double controlPointWeight = 1.0;            ///< 控制点残差全局权重
        double controlPointHuberDeltaMeters = 0.2;  ///< 控制点 3D 距离 Huber 阈值（米）

        // ── 比例尺/标尺软约束 ────────────────────────────────────────────────
        bool enableScaleBarConstraints = false;                ///< 是否启用两点间距离约束
        double scaleBarWeight = 1.0;                           ///< 比例尺残差全局权重
        double scaleBarHuberDeltaMeters = 0.2;                 ///< 比例尺长度残差 Huber 阈值（米）
        std::vector<BAScaleBarConstraint> scaleBarConstraints; ///< 与 tracks 下标关联的比例尺约束

        // ── 相机位姿软先验 ───────────────────────────────────────────────────
        std::vector<BACameraPosePrior> cameraPosePriors; ///< 与 cameras 同序的可选外参软先验
        double cameraPosePriorWeight = 1000.0;           ///< 位姿先验整体权重
        double cameraPosePriorHuberDelta = 3.0;          ///< 位姿先验 Huber 阈值（归一化残差）

        // ── 航测相机中心平面软约束 ───────────────────────────────────────────
        BACameraPlaneConstraint cameraPlaneConstraint; ///< 所有可变相机共享的光心平面约束
        double cameraPlaneHuberDelta = 3.0;            ///< 法向归一化残差的 Huber 阈值

        // ── Gauge 固定 ──────────────────────────────────────────────────────────
        /// 联合 BA 的规范处理策略。默认自动锚定，避免直接调用产生奇异法方程。
        BAGaugePolicy gaugePolicy = BAGaugePolicy::AutoAnchor;
        /// 固定这些索引对应的相机位姿（不参与 camera 优化阶段）。
        /// 仅固定一个相机只能消除整体旋转和平移；无绝对尺度约束时还必须固定第二个相机
        /// 或提供比例尺/控制点约束，才能消除完整的 7 自由度 gauge。
        std::vector<int> fixedCameraIndices;
        /// 固定这些索引对应的三维点，但仍保留其重投影残差对相机的约束。
        /// 主要用于分块 BA 的跨块轨迹，避免各块独立改写同一个边界点。
        std::vector<int> fixedTrackIndices;

        // ── 离群点过滤 ─────────────────────────────────────────────────────────
        /// 是否在每轮 BA 结束后自动过滤高重投影误差点。
        bool enablePointFilter = true;
        /// 过滤阈值（像素）：rmsAfter 超过此值的点会被标记为 invalid。
        double filterMaxReprojError = 2.5; ///< 重投影误差上限（像素），超过此值的点标记为无效
        /// 过滤倍率：同时过滤 rmsAfter > filterSigmaFactor × 全局中位数 rms 的点。
        /// 设为 0 则禁用此规则，仅使用固定阈值。
        double filterSigmaFactor = 3.0;

        // ── 并行计算 ───────────────────────────────────────────────────────────
        /// OpenMP 线程数；0 表示使用系统默认（OMP_NUM_THREADS 或 CPU 核心数）。
        int numThreads = 0;
        /// 是否把每一轮 BA 迭代写入 INFO 日志；候选粗筛关闭以避免日志洪泛。
        bool logIterationProgress = true;

        // ── 后端与 GPU ─────────────────────────────────────────────────────────
        /// PlaMatrix CUDA/OpenCL Schur PCG 使用的稳定设备索引。CPU 后端忽略此项。
        int plaMatrixDevice = 0;
        /// 实验性混合精度：先用 FP32 PCG 生成初值，再由 FP64 PCG 校正；
        /// 默认关闭，只有目标 GPU 上基准确认收益后才启用。
        bool enablePlaMatrixMixedPrecision = false;
        /// Auto 常规规模选择 PlaMatrix CUDA 所需的最小相机数和观测数。
        int minPlaMatrixCudaCameras = kDefaultMinPlaMatrixCudaCameras;
        int minPlaMatrixCudaObservations = kDefaultMinPlaMatrixCudaObservations;
        /// Auto 常规规模选择 PlaMatrix OpenCL 所需的最小相机数和观测数。
        int minPlaMatrixOpenClCameras = kDefaultMinPlaMatrixOpenClCameras;
        int minPlaMatrixOpenClObservations = kDefaultMinPlaMatrixOpenClObservations;
        /// 高密度覆盖规则的共同最小相机数；低于此值始终保留 CPU，避免设备启动开销主导。
        int minPlaMatrixDenseCameras = kDefaultMinPlaMatrixDenseCameras;
        /// 达到高密度相机门槛后，CUDA/OpenCL 各自所需的最小观测数。
        int minPlaMatrixCudaDenseObservations = kDefaultMinPlaMatrixCudaDenseObservations;
        int minPlaMatrixOpenClDenseObservations = kDefaultMinPlaMatrixOpenClDenseObservations;
        /// 联合问题装配前允许的初始 track 重投影 RMS 上限（像素）；<=0 表示关闭。
        /// 该门控仅剔除会让联合试探步进入硬正深度惩罚的 gross outlier。
        double maxInitialTrackRms = 100.0;
        /// 旧 CPU 自动切换阈值；保留工程兼容，参考 CPU 固定使用直接 Cholesky。
        int maxDenseSchurCameras = 200;
        /// CUDA/OpenCL Schur PCG 的连续相机块 cluster-Jacobi 大小；1 使用 block-Jacobi。
        int plaMatrixPreconditionerClusterSize = 1;
        /// 请求加速后端不可用或求解失败时是否允许回退到语义等价的 CPU 后端。
        bool allowBackendFallback = true;
        /// Auto 后端是否启用质量门控；显式指定后端时不自动替用户回退质量。
        bool enableBackendQualityGate = true;
        /// Auto 候选后端允许的最大 RMS 增长倍率；超过则回退 legacy。
        double maxAcceptedRmsGrowth = 1.25;
        /// Auto 候选后端最少有效 track 比例；低于该比例则回退 legacy。
        double minAcceptedValidTrackRatio = 0.60;
        /// LiDAR/GCP/比例尺等物方约束允许的最大 RMS 增长倍率。
        double maxAcceptedConstraintRmsGrowth = 1.25;
        /// Auto 候选为加速后端时是否额外运行一次 legacy 对照。
        /// 默认关闭，避免每轮 BA 执行两套完整求解；基准测试或诊断时可显式开启。
        bool compareAutoBackendWithLegacy = false;

        // ── 任务控制 ──────────────────────────────────────────────────────────
        /// 外部取消标志；GUI/CLI 长任务可在外层迭代边界中止 BA。
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        /// 外层迭代进度回调，返回 false 表示调用方要求停止后续迭代。
        std::function<bool(int currentIteration, int maxIterations, double avgRms, int validPoints)> progressCallback;
    };

    /// 同时遵守兼容开关与逐参数掩码，作为所有 BA 后端和调度层的统一生效判定。
    inline bool sharedIntrinsicParameterEnabled(const BAOptions& options, BAIntrinsicParameter parameter)
    {
        bool enabled = false;
        switch (parameter)
        {
        case BAIntrinsicParameter::FocalLength:
            enabled = options.refineSharedFocalLength;
            break;
        case BAIntrinsicParameter::FocalAspectRatio:
            enabled = options.refineSharedFocalAspectRatio;
            break;
        case BAIntrinsicParameter::PrincipalPointX:
        case BAIntrinsicParameter::PrincipalPointY:
            enabled = options.refineSharedPrincipalPoint;
            break;
        case BAIntrinsicParameter::RadialK1:
            enabled = options.refineSharedRadialDistortion;
            break;
        case BAIntrinsicParameter::RadialK2:
        case BAIntrinsicParameter::RadialK3:
        case BAIntrinsicParameter::TangentialP1:
        case BAIntrinsicParameter::TangentialP2:
            enabled = options.refineSharedRadialDistortion && options.refineSharedHighOrderDistortion;
            break;
        case BAIntrinsicParameter::Count:
            return false;
        }

        const std::size_t index = static_cast<std::size_t>(parameter);
        return enabled && (!options.useSharedIntrinsicParameterMask || options.sharedIntrinsicParameterMask[index]);
    }

} // namespace xjw
