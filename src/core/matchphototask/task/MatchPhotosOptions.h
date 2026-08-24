#pragma once

#include "PairSelectionPolicy.h"

#include <QString>

namespace xjw
{
namespace matchphotos
{

// 面向用户的预设应映射到合理默认值，而不是在主流程 UI 中暴露每个检测器
// 和匹配器的细碎参数。
enum class MatchPhotosProfile
{
    Auto,
    Fast,
    HighAccuracy,
    DifficultTexture,
    CpuCompatible,
    CudaAccelerated
};

enum class ComputeDevice
{
    Auto,
    Cpu,
    Cuda,
    OpenCl,
    Metal
};

enum class ReferencePreselectionGeometry
{
    GroundFootprint,
    SparseScene
};

enum class GuidedMatchingMode
{
    Disabled,
    Automatic,
    Forced
};

inline QString guidedMatchingModeName(GuidedMatchingMode mode)
{
    switch (mode)
    {
    case GuidedMatchingMode::Disabled:
        return QStringLiteral("off");
    case GuidedMatchingMode::Automatic:
        return QStringLiteral("auto");
    case GuidedMatchingMode::Forced:
        return QStringLiteral("force");
    }
    return QStringLiteral("off");
}

inline bool guidedMatchingEnabled(GuidedMatchingMode mode)
{
    return mode != GuidedMatchingMode::Disabled;
}

inline GuidedMatchingMode guidedMatchingModeFromName(
    const QString &name,
    GuidedMatchingMode fallback = GuidedMatchingMode::Disabled)
{
    const QString normalized = name.trimmed().toLower();
    if (normalized == QLatin1String("off") || normalized == QLatin1String("disabled"))
    {
        return GuidedMatchingMode::Disabled;
    }
    if (normalized == QLatin1String("auto") || normalized == QLatin1String("automatic"))
    {
        return GuidedMatchingMode::Automatic;
    }
    if (normalized == QLatin1String("force") || normalized == QLatin1String("forced"))
    {
        return GuidedMatchingMode::Forced;
    }
    return fallback;
}

struct MatchPhotosOptions
{
    MatchPhotosProfile profile = MatchPhotosProfile::Auto;
    ComputeDevice device = ComputeDevice::Auto;

    // 推荐指定便携 LightGlue ONNX；历史本机 `.engine` 仅作为兼容输入。
    QString lightGlueTensorRtEnginePath;
    // LoMa-R schema 2 清单绑定特征/匹配 ONNX，运行时据本机环境生成 engine。
    QString lomaRTensorRtPackagePath;
    // 0 表示按 GPU 总显存与关键点上限自动选择 1024/2048/3840 bucket；
    // 非 0 值表示用户明确指定档位。显式 manifest 路径始终优先于档位选择。
    int lomaRKeypointBudget = 0;

    // 影像对规划保持显式配置，方便调用方复用 overlap 结果，
    // 或在测试、批处理中强制使用确定性模式。
    PairSelectionPolicy pairPolicy = makePairSelectionPolicy(PairSelectionPreset::Auto);

    // 组合算法通过统一注册表选择。当前注册 auto_sift、sift_lightglue 和 loma_r；
    // 新实现只需实现 IImageMatchingAlgorithm 并注册，不再增加特征/匹配双重 token。
    QString algorithmId = QStringLiteral("auto_sift");
    // 蒙版应用阶段：none=不使用，keypoints=提取后过滤关键点，tiepoints=匹配后过滤连接点。
    // 项目蒙版约定为 0 表示有效区域，非 0 表示排除区域。
    QString maskApplyMode = QStringLiteral("none");

    int maxImageDim = 2048;
    int maxKeypoints = 0;
    int keypointLimitPerMegapixel = 0;
    int cudaDevice = 0;
    // 0 表示根据可用显存自动选择；CPU 路径始终按 1 处理。
    int cudaParallelPairs = 0;
    // CUDA SIFT 流水线最多预读的影像数，避免大 TIFF 占满主机内存。
    int featurePrefetchDepth = 2;
    float matchThreshold = 0.15f;
    // SIFT 最近邻/次近邻距离比。自适应模式把该值作为宽松上限，在候选充足时
    // 根据当前像对的双向互检分布自动收紧；小像对保留上限，避免少量匹配被裁空。
    float siftMaximumRatio = 0.98f;
    float siftMinimumAdaptiveRatio = 0.78f;
    bool adaptiveSiftRatio = true;
    double geometryReprojThreshold = 1.5;
    int geometryMinInliers = 20;
    double geometryMinInlierRatio = 0.18;
    double geometryMinGridCoverage = 0.12;
    int geometryGridColumns = 4;
    int geometryGridRows = 4;
    int geometryMaxIterations = 10000;
    int maxTiePointsPerImage = 4000;
    int maxTiePointsPerGridCell = 500;
    int tiePointGridColumns = 4;
    int tiePointGridRows = 4;
    bool enableGeometryVerification = true;
    bool enableTrackBuild = true;
    // Automatic 只补救几何可靠但支持度、内点率或覆盖率偏弱的像对；Forced
    // 对所有具备可靠基础矩阵或可信参考位姿的像对执行引导搜索。
    GuidedMatchingMode guidedMatchingMode = GuidedMatchingMode::Disabled;
    bool guidedUseReferenceCameraPoses = false;
    bool guidedRequireMultiViewConsistency = true;
    bool useExplicitKeypointLimit = false;
    bool useGenericPreselection = true;
    bool useReferencePreselection = false;
    // 导入外方位默认按地面/球面覆盖分析；已有 SfM 位姿必须使用稀疏场景
    // 共视和视锥重叠，不能把任意三维/环拍场景强行投影到地球表面。
    ReferencePreselectionGeometry referencePreselectionGeometry =
        ReferencePreselectionGeometry::GroundFootprint;
    bool excludeStationaryTiePoints = true;
    // 复用与当前影像指纹、算法版本和配置指纹完全一致的 `.pimatch` 数据。
    // 特征本身不再持久化，因此“重置对齐 + 复用匹配”不会重新提取 SIFT。
    bool reuseExistingMatches = true;
    float stationaryTiePointMaxPixelMotion = 1.0f;

    // 项目蒙版是排除概率（0=有效，255=确定排除）。只硬裁高置信度且位于
    // 排除区内部的点；边界和不确定区域以软权重保留。
    float maskHardExclusionThreshold = 0.90f;
    float maskMinimumTiepointWeight = 0.20f;
    int maskRelaxationRadius = 2;

    // 在特征、匹配、几何验证和轨迹阶段接入现有核心模块前，
    // 框架默认先以 plan-only 方式运行。
    bool planOnly = true;
};

} // namespace matchphotos
} // namespace xjw
