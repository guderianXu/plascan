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
    Cuda
};

struct MatchPhotosOptions
{
    MatchPhotosProfile profile = MatchPhotosProfile::Auto;
    ComputeDevice device = ComputeDevice::Auto;

    // LightGlue 仅使用 TensorRT。engine 与目标 GPU 和 TensorRT 版本绑定。
    QString lightGlueTensorRtEnginePath;
    // LoMa-R 使用一个 JSON 清单同时绑定 DaD + DeDoDe-G 特征引擎和
    // LoMa-R 匹配引擎，避免两个不兼容 engine 被误组合。
    QString lomaRTensorRtPackagePath;

    // 影像对规划保持显式配置，方便调用方复用 overlap 结果，
    // 或在测试、批处理中强制使用确定性模式。
    PairSelectionPolicy pairPolicy = makePairSelectionPolicy(PairSelectionPreset::Auto);

    // 组合算法通过统一注册表选择。当前只注册 sift_lightglue；以后增加算法时，
    // 新实现只需实现 IImageMatchingAlgorithm 并注册，不再增加特征/匹配双重 token。
    QString algorithmId = QStringLiteral("sift_lightglue");
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
    double geometryReprojThreshold = 1.5;
    int geometryMinInliers = 20;
    int geometryMaxIterations = 10000;
    int maxTiePointsPerImage = 4000;
    int maxTiePointsPerGridCell = 500;
    int tiePointGridColumns = 4;
    int tiePointGridRows = 4;
    bool enableGeometryVerification = true;
    bool enableTrackBuild = true;
    bool enableGuidedMatching = false;
    bool useExplicitKeypointLimit = false;
    bool useGenericPreselection = true;
    bool useReferencePreselection = false;
    bool excludeStationaryTiePoints = true;
    // 复用与当前影像指纹、算法版本和配置指纹完全一致的 `.pimatch` 数据。
    // 特征本身不再持久化，因此“重置对齐 + 复用匹配”不会重新提取 SIFT。
    bool reuseExistingMatches = true;
    float stationaryTiePointMaxPixelMotion = 1.0f;

    // 在特征、匹配、几何验证和轨迹阶段接入现有核心模块前，
    // 框架默认先以 plan-only 方式运行。
    bool planOnly = true;
};

} // namespace matchphotos
} // namespace xjw
