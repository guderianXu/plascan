// =============================================================================
// 文件: MvsTypes.h
// 模块: MVS (Multi-View Stereo)
// 说明:
//   MVS 管线的所有公共数据结构定义。
//
//   核心设计原则（COLMAP 方法）：
//     - 所有相机参数在进入 MVS 之前统一转换为 COLMAP 正深度约定
//     - Z_cam > 0 恒成立，不再需要追踪 depthFlippedZ / uDir / vDir 等 hack
//
//   深度图约定：
//     depthMap(v, u) = Z_cam（正值，代表相机坐标系 Z 轴距离）
//     0.0 = 无效
// =============================================================================
#pragma once

#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdint>
#include <atomic>

#include "camera/Camera.h"
#include "DepthPoseRefinementStage.h"

#include <opencv2/core.hpp>
#include <plapoint/core/processing_policy.h>

namespace xjw
{
namespace mvs
{

// =============================================================================
// PatchMatch 配置
// =============================================================================
enum class PatchMatchBackend
{
    Auto,
    Cpu,
    Cuda,
    OpenCl
};

inline const char *patchMatchBackendId(PatchMatchBackend backend) noexcept
{
    switch (backend)
    {
    case PatchMatchBackend::Auto:
        return "auto";
    case PatchMatchBackend::Cpu:
        return "cpu";
    case PatchMatchBackend::Cuda:
        return "cuda";
    case PatchMatchBackend::OpenCl:
        return "opencl";
    }
    return "unknown";
}

struct PatchMatchConfig
{
    int   numIterations      = 16;      ///< PatchMatch 迭代次数（16轮保证宽深度范围下充分收敛）
    int   patchHalf          = 7;       ///< NCC 块半径（块大小 = 2r+1 = 15×15），更大→更鲁棒
    int   numSourceViews     = 6;
    int   cpuThreadCount     = 1;       ///< CPU 路径的像素级并行线程数
    float confidenceThresh   = 0.60f;   ///< 多图生产阈值；少视图/快速预览由调用方显式降低
    float minimumMaskedPatchSupportRatio = 0.35f; ///< mask-aware NCC 最小双边有效样本比例
    bool  enablePhotometricUniqueness = true; ///< 用相邻竞争深度抑制重复纹理/低纹理的歧义解
    float photometricUniquenessRelativeDepthStep = 0.01f; ///< 竞争深度相对偏移（正负各一次）
    float photometricUniquenessMinimumMargin = 0.03f; ///< 最优 NCC 与竞争深度 NCC 的最小可信间隔
    float photometricUniquenessMinimumConfidenceScale = 0.50f; ///< 完全歧义时保留的置信度比例
    bool  useCuda            = true;    ///< 旧版兼容字段；Auto 下 false 强制 CPU，显式 backend 优先
    PatchMatchBackend backend = PatchMatchBackend::Auto;
    int   downsampleFactor   = 2;       ///< 降采样因子（2=半分辨率，速度提升约4倍）
    bool  returnNativeResolution = false; ///< 仅返回 PatchMatch 工作分辨率；默认保持对外全尺寸契约
    bool  doMedianBlur       = true;
    int   medianKernelSize   = 5;
    bool  doBilateralFilter  = true;
    int   bilateralD         = 9;
    float bilateralSigmaColor= 50.f;
    float bilateralSigmaSpace= 5.f;

    // 几何一致性检查（PatchMatch 第二趟）
    bool  geomConsistency      = true;   ///< 是否启用几何一致性检查
    float geomConsistencyMaxErr= 1.0f;   ///< 允许的最大质心深度偏差（像素）

    // CUDA 线程块尺寸（调优不同 GPU 架构时修改）
    int   cudaBlockW           = 16;     ///< 2D kernel 线程块宽度（必须为 2 的幂）
    int   cudaBlockH           = 16;     ///< 2D kernel 线程块高度（必须为 2 的幂）
    int   cudaBlockSweep       = 32;     ///< 1D sweep kernel 线程块大小（必须为 2 的幂）
    int   cudaDeviceIndex      = -1;     ///< CUDA 设备编号；-1 使用当前设备
    int   openClDeviceIndex    = -1;     ///< OpenCL GPU 全局编号；-1 使用首个 GPU

    bool  epipolarRectified    = false;  ///< 图像已极线校正，偏向水平传播
    bool  cudaUseParallelSweep = true;   ///< CUDA 使用棋盘格像素级并行传播；false 时回退传统行列 sweep
    bool  cudaFallbackToCpu    = false;  ///< CUDA 失败时默认明确报错；Auto 只在任务开始前按优先级选择后端
    bool  openClFallbackToCpu  = false;  ///< OpenCL 失败时默认明确报错，避免 GPU worker 暗中占用 CPU
    const std::atomic_bool *cancelFlag = nullptr; ///< 非拥有取消标志；用于长 PatchMatch 循环协作退出
};

// =============================================================================
// 融合配置
// =============================================================================
struct FusionConfig
{
    int   minConsistentViews = 3;      ///< 至少有多少视图一致才保留像素
    float relDepthThresh     = 0.03f;  ///< 深度相对误差阈值（收紧→更严格一致性）
    float pixelThresh        = 1.5f;   ///< 投影误差阈值（像素），收紧→减少噎点
    float confidenceThresh   = 0.65f;  ///< 融合前二次置信度过滤阈值（生产点云默认较严格）
    bool  enableAdaptiveConfidenceFilter = true; ///< 低置信满幅深度图自动进入严格过滤
    float adaptiveFullCoverageThreshold = 0.95f; ///< 有效覆盖率超过该值时检查低置信满幅风险
    float adaptiveLowMeanConfidenceThreshold = 0.65f; ///< 平均置信度低于该值视为可疑满幅深度
    float adaptiveStrictConfidenceThreshold = 0.65f; ///< 可疑满幅深度图使用的最低融合阈值
    bool  enableGeometrySupportedLowConfidenceRetention = true; ///< 多视几何一致时保留低置信实测深度
    float geometrySupportedMinimumConfidence = 0.35f; ///< 几何保留仍接受的最低原始置信度
    int   geometrySupportedMinimumObservationCount = 3; ///< 至少参考帧加两个来源共同确认
    float geometrySupportedMaximumInverseDepthSpread = 0.006f; ///< 几何保留允许的最大逆深度相对离散度
    float geometrySupportedMinimumAdaptiveSupportWeight = 0.50f; ///< 连续几何支持权重下限
    float geometrySupportedMinimumAdaptiveEffectiveViews = 2.0f; ///< 连续证据有效来源数下限
    float geometrySupportedMaximumAdaptiveConflictRatio = 0.45f; ///< 连续证据冲突比例上限
    bool  doSigmaFusion      = true;   ///< 是否做 sigma 加权深度融合
    float sigmaMultiplier    = 2.0f;   ///< sigma 乘数放宽→少剔除内点
    bool  doInpaint          = true;   ///< 对小洞做 inpaint（填补）
    float inpaintRadiusFactor= 3.0f;
    int   inpaintRadius      = 5;      ///< inpaint 搜索半径（像素），加大→填补更多小洞
    bool  enableLocalDepthOutlierFilter = true;  ///< 融合前剔除局部中值不一致的深度突刺
    int   localDepthOutlierKernelSize = 3;       ///< 局部中值窗口，必须为奇数
    float localDepthOutlierRelThresh = 0.25f;    ///< 与局部中值的相对深度差超过该值时视为离群
    float maxLocalDepthOutlierRemovalRatio = 0.20f; ///< 单帧最多允许移除比例，超过则回退
    bool  enableSpeckleFilter = true;            ///< 是否剔除孤立小连通域深度斑点
    int   minSpeckleComponentArea = 16;          ///< 小于该面积的有效深度连通域视为 speckle
    float maxSpeckleRemovalRatio = 0.20f;        ///< 单帧 speckle 最多允许移除比例，超过则回退
};

// =============================================================================
// 融合前低置信保留所需的跨视几何证据
// =============================================================================
struct DepthPostProcessEvidence
{
    cv::Mat geometrySupportCount; ///< 参考帧与一致来源的观测总数 (CV_16U)
    cv::Mat inverseDepthRelativeSpread; ///< 一致观测逆深度相对标准差 (CV_32F)
    cv::Mat adaptiveSupportWeight; ///< 连续跨视支持权重 (CV_32F)，可为空
    cv::Mat adaptiveEffectiveViewCount; ///< 连续证据有效来源数 (CV_32F)，可为空
    cv::Mat adaptiveConflictRatio; ///< 连续证据冲突比例 (CV_32F)，可为空
};

// =============================================================================
// 深度图后处理统计
// =============================================================================
struct DepthPostProcessStats
{
    int validBeforePostprocess = 0;       ///< 后处理前有效深度像素数
    int validAfterConfidenceFilter = 0;   ///< 置信度过滤后有效深度像素数
    int lowConfidenceCandidateCount = 0;  ///< 低于阈值、进入置信度判定的像素数
    int geometrySupportedLowConfidenceRetained = 0; ///< 因强多视几何证据而保留的低置信实测像素数
    int confidenceRemoved = 0;            ///< 置信度过滤移除像素数
    int localDepthOutlierRemoved = 0;     ///< 局部深度离群过滤移除像素数
    int smallComponentRemoved = 0;        ///< 小连通域 speckle 过滤移除像素数
    int speckleRemoved = 0;               ///< speckle 过滤移除像素数（对外报告字段）
    int edgeConfidenceRemoved = 0;        ///< 边缘/低纹理置信度过滤移除像素数
    int geomConsistencyRemoved = 0;       ///< 多视几何一致性过滤移除像素数
    int validAfterPostprocess = 0;        ///< 所有后处理后有效深度像素数
    float effectiveConfidenceThreshold = 0.0f; ///< 实际使用的融合前置信度阈值
};

// =============================================================================
// 融合输入帧
// =============================================================================
struct FusionFrameInput
{
    cv::Mat   depthMap;    ///< CV_32F, 正深度（COLMAP 约定），0=无效
    cv::Mat   normalMap;   ///< CV_32FC3, 法线（相机坐标系），可为空
    cv::Mat   confidence;  ///< CV_32F, [0,1]，可为空
    cv::Mat   validMask;   ///< CV_8U, 项目/内容蒙版传播后的权威有效区域
    cv::Mat   supportCount; ///< CV_16U, PatchMatch 多视支持计数
    Camera    cameraModel;  ///< 与深度栅格对应的正深度、零畸变工作相机
    Camera    sourceCamera; ///< 与 imagePath 原始像素对应的完整相机，用于融合取色预处理
    int       viewIndex = -1; ///< 原始 CameraView 下标，用于将 source plan 重映射到融合帧下标
    int       imgW = 0;
    int       imgH = 0;
    std::string imagePath; ///< 原始彩色图像路径（用于取色）
    std::vector<int> sourceImageIndices; ///< MVS source plan 指定的重叠影像索引；为空时回退相机中心近邻
    DepthPostProcessStats depthPostprocess; ///< 融合前深度图后处理统计
};

// =============================================================================
// 稠密点（含颜色）
// =============================================================================
struct DensePoint
{
    float   x = 0.f, y = 0.f, z = 0.f;
    uint8_t r = 0,   g = 0,   b = 0;
};

enum class MvsSceneProfile
{
    Auto,
    OrbitalObject,
    AerialTerrain,
    Custom
};

enum class DepthFilterMode
{
    Mild,
    Moderate,
    Aggressive
};

struct DepthPyramidLevelConfig
{
    int level = 1;
    PatchMatchConfig patchMatch;
    int minSupportViews = 2;
    float radiusScale = 1.0f;
};

struct DepthPyramidConfig
{
    std::array<DepthPyramidLevelConfig, 3> levels;
    int activeLevelCount = 3;
    MvsSceneProfile sceneProfile = MvsSceneProfile::Auto;
    DepthFilterMode filterMode = DepthFilterMode::Moderate;
    bool saveIntermediateLevels = false;
    std::string degradedReason;
};

struct MvsSourcePairQuality
{
    std::string imageA;
    std::string imageB;
    int totalMatches = 0;
    int geometricInliers = 0;
    bool verified = false;
    bool hasVerificationStatistics = false;
    float geometricCoverage = 0.0f;
    std::string verificationReason;
};

// =============================================================================
// 深度图生成配置
// =============================================================================
struct DepthGenConfig
{
    PatchMatchConfig patchMatch;
    FusionConfig     fusion;
    plapoint::ProcessingDevice pointCloudProcessingDevice =
        plapoint::ProcessingDevice::Auto; ///< 点云阶段独立后端；Auto 按 CUDA、OpenCL、CPU 选择
    std::string qualityProfile = "medium"; ///< 用户请求的深度质量档位，用于产物审计
    int   numSourceViews        = 4;
    int   configuredSourceViewCount = 0; ///< 场景自适应限制前的源视角请求；0 表示与 numSourceViews 相同
    int   totalCpuThreadBudget  = 0;     ///< 独占 CPU 后处理阶段的总线程预算；0 表示由帧调度配置推导
    int   cpuWorkerCount        = 1;     ///< 每个 CPU 帧 worker 内部的像素级线程数
    int   gpuFrameWorkerCount   = 2;     ///< GPU 主机准备槽下限；默认双缓冲，设备 kernel 仍单路串行
    int   cpuFrameWorkerCount   = 1;     ///< CPU 路径的帧级并发数
    float zNearScale            = 0.75f;  ///< IQR 内 2%ile × 0.75（原0.5太宽致PatchMatch难收敛）
    float zFarScale             = 1.5f;   ///< IQR 内 98%ile × 1.5（原2.0产生6x搜索比，现3x更合理）
    bool  runDepthEstimation    = true;
    bool  runFusion             = true;
    bool  saveIntermediateDepthMaps = false;
    bool  saveIntermediatePyramidLevels = false; ///< Debug：保存 Level 2/3 原始深度结果
    std::string intermediateDir = "";
    bool  adaptiveDepthCacheMemory = true;      ///< 根据系统内存自动决定是否常驻 full-res 深度帧
    float maxDepthCacheRamFraction = 0.60f;     ///< full-res 深度帧缓存最多使用物理内存比例
    uint64_t minFreeRamBytes = 2ull * 1024ull * 1024ull * 1024ull; ///< 运行时保留给系统/临时 Mat 的空闲内存
    std::string inputSignature; ///< 上游空三/相机解代次；变化时必须使工作区深度缓存失效
    std::vector<MvsSourcePairQuality> sourcePairQualities; ///< 直接影像对匹配/几何验证质量
    bool requireVerifiedSourcePairs = false; ///< 有 pair 质量时，MVS source 必须来自已验证匹配对
    int minSourcePairGeometricInliers = 20;  ///< source pair 几何内点最低门槛
    MvsSceneProfile sceneProfile = MvsSceneProfile::Auto; ///< Auto 时根据相机与稀疏云几何分类
    DepthFilterMode depthFilterMode = DepthFilterMode::Moderate; ///< 显式过滤预设
    bool adaptiveDepthFilterMode = true; ///< Auto 场景下航测用中等、环拍物体用温和过滤
    bool enableAdaptiveGeometryEvidence = true; ///< 环拍场景生成连续证据并执行投影主深度层选择
    DepthPoseRefinementOptions depthPoseRefinement; ///< 默认关闭；仅输出派生相机候选和诊断，不覆盖项目相机
    int crossViewHoleRepairSourceCount = 8; ///< 环拍内部孔洞修复使用的邻帧数，不改变 PatchMatch 源视图
    bool enableTargetedGapRecovery = true; ///< 环拍缺口用两个最优来源执行受先验约束的定向 PatchMatch
    int targetedGapRecoverySourceCount = 6;
    int targetedGapRecoveryHypothesisCount = 2;
    float targetedGapRecoveryConfidence = 0.28f;
    float targetedGapRecoveryPriorRelativeDifference = 0.18f;
    float targetedGapRecoveryConsensusInverseDepthSpread = 0.025f;
    float targetedGapRecoveryConsensusPriorRelativeDifference = 0.35f;
    bool enableTargetedGapSurfacePrior = false;
    float targetedGapSurfacePriorMaximumAnchorSpread = 0.12f;
    float targetedGapSurfacePriorMaximumFitResidual = 0.025f;
    int targetedGapRecoveryMaximumPriorDistancePixels = 128;
    bool enablePostConsistencyResidualReestimation = true;
    int postConsistencyResidualSourceCount = 8;
    float postConsistencyResidualConfidence = 0.30f;
    float postConsistencyResidualMaximumLayerSpread = 0.025f;
    float postConsistencyResidualMaximumPriorRadius = 0.08f;
    bool enableTwoSourceCrossViewGrowth = false; ///< 实验：从三源强核心受控恢复稳定两源缺口
    int twoSourceGrowthDistancePixels = 3;
    float twoSourceGrowthInverseDepthSpread = 0.01f;
    float twoSourceGrowthNormalAngleDegrees = 15.0f;
    int twoSourceGrowthMaximumComponentArea = 64;
};

// =============================================================================
// 输入相机视图（图像元数据 + 唯一相机表达）
// =============================================================================
struct CameraView
{
    std::string imagePath;
    std::string validRegionMaskPath; ///< 项目蒙版路径；文件中非零=排除，MVS 内部转换为 255=有效
    int         imageWidth  = 0;
    int         imageHeight = 0;
    xjw::Camera camera;
};

// =============================================================================
// 稀疏点云预处理结果（供深度范围估计）
// =============================================================================
struct SparseCloud
{
    std::vector<std::array<float,3>> points;
    std::array<float,3> minPt = {}, maxPt = {};
};

struct MvsSceneClassification
{
    MvsSceneProfile profile = MvsSceneProfile::OrbitalObject;
    float cameraCenterLinearity = 0.0f;
    float opticalAxisConvergence = 0.0f;
    float planeThicknessRatio = 1.0f;
    float downLookingConsistency = 0.0f;
    std::string reason;
};

// =============================================================================
// 稀疏点在参考影像上的投影样本
// =============================================================================
struct ProjectedSparseDepthSample
{
    float uNorm = 0.0f;  ///< 归一化列坐标，后续可缩放到任意同几何工作分辨率
    float vNorm = 0.0f;  ///< 归一化行坐标，后续可缩放到任意同几何工作分辨率
    float depth = 0.0f;  ///< 正深度 Z_cam
};

// =============================================================================
// 进度回调
// =============================================================================
using MvsProgressCallback = std::function<void(const std::string &, float)>;

} // namespace mvs
} // namespace xjw
