#include "MatchPhotosAlgorithmSelector.h"

#include <algorithm>

namespace xjw
{
namespace matchphotos
{
namespace
{

QString profileId(MatchPhotosProfile profile)
{
    switch (profile)
    {
    case MatchPhotosProfile::Fast:
        return QStringLiteral("fast");
    case MatchPhotosProfile::HighAccuracy:
        return QStringLiteral("high_accuracy");
    case MatchPhotosProfile::DifficultTexture:
        return QStringLiteral("difficult_texture");
    case MatchPhotosProfile::CpuCompatible:
        return QStringLiteral("cpu_compatible");
    case MatchPhotosProfile::CudaAccelerated:
        return QStringLiteral("cuda_accelerated");
    case MatchPhotosProfile::Auto:
        return QStringLiteral("auto");
    }
    return QStringLiteral("auto");
}

bool shouldPreferCuda(const MatchPhotosOptions &options)
{
    if (options.device == ComputeDevice::Cpu ||
        (options.device == ComputeDevice::Auto &&
         options.profile == MatchPhotosProfile::CpuCompatible))
    {
        return false;
    }

    // Auto 表示优先使用可用加速设备，而不是默认 CPU。运行阶段会在 CUDA 不可用时
    // 自动回退；只有用户显式选择 CUDA 时才禁止静默回退。
    return true;
}

int defaultMaxKeypoints(MatchPhotosProfile profile)
{
    switch (profile)
    {
    case MatchPhotosProfile::Fast:
        return 4096;
    case MatchPhotosProfile::HighAccuracy:
    case MatchPhotosProfile::DifficultTexture:
        return 12000;
    case MatchPhotosProfile::Auto:
    case MatchPhotosProfile::CpuCompatible:
    case MatchPhotosProfile::CudaAccelerated:
        return 8192;
    }
    return 8192;
}

int resolveMaxKeypoints(const MatchPhotosOptions &options)
{
    if (options.useExplicitKeypointLimit)
    {
        return std::max(0, options.maxKeypoints);
    }
    if (options.maxKeypoints > 0)
    {
        return options.maxKeypoints;
    }
    return defaultMaxKeypoints(options.profile);
}

} // namespace

MatchPhotosAlgorithmPlan MatchPhotosAlgorithmSelector::select(const MatchPhotosOptions &options)
{
    MatchPhotosAlgorithmPlan plan;
    plan.strategyId = QStringLiteral("metashape_like_%1_sift_lightglue").arg(profileId(options.profile));
    plan.displayName = QStringLiteral("类 Metashape 自动匹配：SIFT + LightGlue");

    // 当前主线固定为 SIFT + LightGlue：
    // SIFT 提供尺度和旋转鲁棒性，LightGlue 负责更强的学习型特征匹配。
    plan.featureAlgorithm = QStringLiteral("sift");
    plan.featureSuffix = QStringLiteral(".sift");
    plan.matcherAlgorithm = QStringLiteral("lightglue");
    plan.fallbackMatcherAlgorithm = QStringLiteral("sift_bf_l2");
    plan.needsFeatureStage = true;
    plan.endToEndMatcher = false;
    plan.preferCuda = shouldPreferCuda(options);
    plan.rotationRobust = true;
    // 指导匹配是用户可见的显式开关。质量档只调整数值预算，不能覆盖未勾选状态，
    // 否则匹配 sidecar 与后续 SfM 的缓存契约会出现同一任务内不一致。
    plan.enableGuidedMatching = options.enableGuidedMatching;
    plan.maxImageDim = options.maxImageDim;
    plan.maxKeypoints = resolveMaxKeypoints(options);
    plan.reason = QStringLiteral("采用 SIFT 作为特征提取器以保留尺度和旋转鲁棒性，"
                                 "再使用 LightGlue 对 .sift 特征进行学习型匹配。");
    plan.fallbackReason = QStringLiteral("LightGlue 模型或运行环境不可用时，可回退到 SIFT BF-L2，"
                                         "保证 CPU 环境仍能完成匹配。");
    return plan;
}

} // namespace matchphotos
} // namespace xjw
