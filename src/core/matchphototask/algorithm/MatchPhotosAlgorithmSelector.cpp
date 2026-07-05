#include "MatchPhotosAlgorithmSelector.h"

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
    if (options.device == ComputeDevice::Cpu)
    {
        return false;
    }
    if (options.device == ComputeDevice::Cuda)
    {
        return true;
    }
    return options.profile == MatchPhotosProfile::CudaAccelerated;
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

bool defaultGuidedMatching(MatchPhotosProfile profile, bool requested)
{
    return requested ||
        profile == MatchPhotosProfile::HighAccuracy ||
        profile == MatchPhotosProfile::DifficultTexture;
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
    plan.enableGuidedMatching = defaultGuidedMatching(options.profile, options.enableGuidedMatching);
    plan.maxImageDim = options.maxImageDim;
    plan.maxKeypoints = options.maxKeypoints > 0 ? options.maxKeypoints : defaultMaxKeypoints(options.profile);
    plan.reason = QStringLiteral("采用 SIFT 作为特征提取器以保留尺度和旋转鲁棒性，"
                                 "再使用 LightGlue 对 .sift 特征进行学习型匹配。");
    plan.fallbackReason = QStringLiteral("LightGlue 模型或运行环境不可用时，可回退到 SIFT BF-L2，"
                                         "保证 CPU 环境仍能完成匹配。");
    return plan;
}

} // namespace matchphotos
} // namespace xjw
