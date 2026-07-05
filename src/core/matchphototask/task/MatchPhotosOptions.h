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

    // 影像对规划保持显式配置，方便调用方复用 overlap 结果，
    // 或在测试、批处理中强制使用确定性模式。
    PairSelectionPolicy pairPolicy = makePairSelectionPolicy(PairSelectionPreset::Auto);

    // 算法名有意与 feature_extractors 和 feature_match 工厂保持一致。
    // 阶段层后续会把这些字符串转换为具体实现。
    QString featureAlgorithm = QStringLiteral("sift");
    QString matcherAlgorithm = QStringLiteral("lightglue");

    int maxImageDim = 2048;
    int maxKeypoints = 0;
    int cudaDevice = 0;
    float matchThreshold = 0.15f;
    double geometryReprojThreshold = 1.5;
    int geometryMinInliers = 20;
    int maxTiePointsPerImage = 4000;
    int maxTiePointsPerGridCell = 500;
    int tiePointGridColumns = 4;
    int tiePointGridRows = 4;
    bool enableGeometryVerification = true;
    bool enableTrackBuild = true;
    bool enableGuidedMatching = false;
    bool reuseExistingFeatures = true;

    // 在特征、匹配、几何验证和轨迹阶段接入现有核心模块前，
    // 框架默认先以 plan-only 方式运行。
    bool planOnly = true;
};

} // namespace matchphotos
} // namespace xjw
