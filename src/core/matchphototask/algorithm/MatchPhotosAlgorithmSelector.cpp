#include "MatchPhotosAlgorithmSelector.h"

#include "ImageMatchingRegistry.h"

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
    plan.algorithmId = options.algorithmId.trimmed().toLower();
    if (plan.algorithmId.isEmpty())
    {
        plan.algorithmId = QStringLiteral("sift_lightglue");
    }
    plan.strategyId = QStringLiteral("metashape_like_%1_%2")
                          .arg(profileId(options.profile), plan.algorithmId);

    // 注册表是算法扩展的唯一入口。选择器不再维护独立的 if/else 工厂列表，
    // 因而新算法不会迫使 GUI、空三或缓存格式同步增加自由字符串分支。
    const std::vector<image_matching::ImageMatchingAlgorithmDescriptor> descriptors =
        image_matching::ImageMatchingRegistry::descriptors();
    const auto descriptor = std::find_if(
        descriptors.cbegin(), descriptors.cend(),
        [&](const image_matching::ImageMatchingAlgorithmDescriptor &candidate)
        {
            return candidate.id.compare(plan.algorithmId, Qt::CaseInsensitive) == 0;
        });
    if (descriptor == descriptors.cend())
    {
        plan.validationError = QStringLiteral("未注册的影像匹配算法: %1")
                                   .arg(plan.algorithmId);
        return plan;
    }

    plan.displayName = descriptor->displayName;
    plan.algorithmVersion = descriptor->version;
    plan.requiresCuda = descriptor->requiresCuda;
    plan.extractsFeaturesInMemory =
        descriptor->inputModel == image_matching::AlgorithmInputModel::ReusableFeatures;
    if (plan.requiresCuda && options.device == ComputeDevice::Cpu)
    {
        plan.validationError = QStringLiteral("%1 需要 CUDA，不能使用 CPU 设备")
                                   .arg(plan.displayName);
        return plan;
    }
    plan.valid = true;
    plan.preferCuda = shouldPreferCuda(options);
    plan.rotationRobust = true;
    // 指导匹配是用户可见的显式开关。质量档只调整数值预算，不能覆盖未勾选状态，
    // 否则匹配 sidecar 与后续 SfM 的缓存契约会出现同一任务内不一致。
    plan.enableGuidedMatching = options.enableGuidedMatching;
    plan.maxImageDim = options.maxImageDim;
    plan.maxKeypoints = resolveMaxKeypoints(options);
    plan.reason = QStringLiteral("CUDA SIFT 在任务内存中提取尺度/旋转鲁棒特征，"
                                 "TensorRT LightGlue 完成匹配；只持久化最终 .pimatch 观测。");
    return plan;
}

} // namespace matchphotos
} // namespace xjw
