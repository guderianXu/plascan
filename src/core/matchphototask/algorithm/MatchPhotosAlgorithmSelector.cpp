#include "MatchPhotosAlgorithmSelector.h"

#include "ImageMatchingRegistry.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"
#include "sift/AutoSiftAlgorithm.h"

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
        options.device == ComputeDevice::OpenCl ||
        options.device == ComputeDevice::Metal ||
        (options.device == ComputeDevice::Auto &&
         options.profile == MatchPhotosProfile::CpuCompatible))
    {
        return false;
    }

    // Auto 表示优先使用可用加速设备，而不是默认 CPU。运行阶段会解析实际后端；
    // 用户显式选择任一设备时都不允许静默替换。
    return true;
}

image_matching::SiftComputeBackend plaMatchRequestedBackend(ComputeDevice device)
{
    switch (device)
    {
    case ComputeDevice::Auto:
        return image_matching::SiftComputeBackend::Automatic;
    case ComputeDevice::Cpu:
        return image_matching::SiftComputeBackend::Cpu;
    case ComputeDevice::Cuda:
        return image_matching::SiftComputeBackend::Cuda;
    case ComputeDevice::OpenCl:
        return image_matching::SiftComputeBackend::OpenCl;
    case ComputeDevice::Metal:
        return image_matching::SiftComputeBackend::Metal;
    }
    return image_matching::SiftComputeBackend::Automatic;
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

float resolveSiftDetectionThreshold(MatchPhotosProfile profile)
{
    // cudaSift 的阈值会在 SiftFeatureExtractor 内转换到其原生量纲。
    // 快速模式减少低响应点，其余模式保留低纹理摄影测量所需的较密检测。
    return profile == MatchPhotosProfile::Fast ? 0.003f : 0.0005f;
}

float resolveSiftContrastThreshold(MatchPhotosProfile profile)
{
    return profile == MatchPhotosProfile::Fast ? 0.04f : 0.02f;
}

} // namespace

MatchPhotosAlgorithmPlan MatchPhotosAlgorithmSelector::select(const MatchPhotosOptions &options)
{
    MatchPhotosAlgorithmPlan plan;
    plan.algorithmId = options.algorithmId.trimmed().toLower();
    if (plan.algorithmId.isEmpty())
    {
        plan.algorithmId = QStringLiteral("plamatch_hct");
    }
    plan.strategyId = QStringLiteral("metashape_like_%1_%2_%3")
                          .arg(profileId(options.profile),
                               alignmentAccuracyName(options.accuracy),
                               plan.algorithmId);

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
    plan.executionBackend = descriptor->requiresCuda
        ? image_matching::SiftComputeBackend::Cuda
        : image_matching::SiftComputeBackend::Automatic;
    plan.extractsFeaturesInMemory = descriptor->inputModel == image_matching::AlgorithmInputModel::ReusableFeatures;
    plan.requiresColorInput = descriptor->requiresColorInput;
    plan.suppliesCoarsePairPreselection = descriptor->suppliesCoarsePairPreselection;
    plan.supportsBatchFeatureMatching = descriptor->supportsBatchFeatureMatching;
    if (plan.requiresCuda && options.device != ComputeDevice::Auto && options.device != ComputeDevice::Cuda)
    {
        plan.validationError = QStringLiteral("%1 需要 CUDA，不能使用当前计算设备").arg(plan.displayName);
        return plan;
    }
    plan.valid = true;
    plan.preferCuda = shouldPreferCuda(options);
    plan.rotationRobust = true;
    // 引导策略是用户可见的显式三态。性能 profile 与五档 Accuracy 都不能覆盖该策略，
    // 否则匹配 sidecar 与后续 SfM 的缓存契约会出现同一任务内不一致。
    plan.guidedMatchingMode = options.guidedMatchingMode;
    plan.alignmentDownscale = alignmentAccuracyDownscale(options.accuracy);
    plan.maxImageDim = options.maxImageDim;
    plan.maxKeypoints = resolveMaxKeypoints(options);
    plan.siftDetectionThreshold = resolveSiftDetectionThreshold(options.profile);
    plan.siftContrastThreshold = resolveSiftContrastThreshold(options.profile);
    plan.lowTextureRecovery = plan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId) &&
                              options.profile != MatchPhotosProfile::Fast;
    if (plan.algorithmId == QLatin1String(image_matching::kPlaMatchHctAlgorithmId))
    {
        const bool force_cpu_profile =
            options.device == ComputeDevice::Auto && options.profile == MatchPhotosProfile::CpuCompatible;
        const image_matching::PlaMatchHctBackendResolution resolution = image_matching::resolvePlaMatchHctBackend(
            force_cpu_profile ? image_matching::SiftComputeBackend::Cpu : plaMatchRequestedBackend(options.device),
            options.cudaDevice);
        if (!resolution.valid)
        {
            plan.valid = false;
            plan.validationError = resolution.errorMessage;
            return plan;
        }
        plan.executionBackend = resolution.backend;
        plan.preferCuda = resolution.backend == image_matching::SiftComputeBackend::Cuda;
        plan.backendFallback =
            options.device == ComputeDevice::Auto && resolution.backend == image_matching::SiftComputeBackend::Cpu;
        plan.computeDeviceName = resolution.deviceName;
        plan.computeDeviceDisplayName = resolution.displayName;
        plan.backendReason = QStringLiteral("PlaMatch-HCT 计算设备：%1").arg(resolution.displayName);
    }
    if (plan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        plan.featureSchemaVersion = 3;
    }
    else if (plan.algorithmId == QLatin1String(image_matching::kPlaMatchHctAlgorithmId))
    {
        plan.featureSchemaVersion = 2;
    }
    plan.reason = QStringLiteral("%1 在任务内存中提取并匹配特征；只持久化最终 "
                                 ".pimatch 观测，不写中间特征文件。")
                      .arg(plan.displayName);
    return plan;
}

MatchPhotosAlgorithmPlan
MatchPhotosAlgorithmSelector::resolveExecutionBackend(const MatchPhotosOptions& options,
                                                      MatchPhotosAlgorithmPlan plan,
                                                      image_matching::SiftComputeBackend resolvedBackend,
                                                      int deviceIndex)
{
    if (!plan.valid ||
        plan.algorithmId != QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        return plan;
    }

    plan.executionBackend = resolvedBackend;
    plan.requiresCuda = false;
    plan.preferCuda = resolvedBackend == image_matching::SiftComputeBackend::Cuda;
    plan.backendFallback = options.device == ComputeDevice::Auto &&
        resolvedBackend == image_matching::SiftComputeBackend::Cpu;
    plan.computeDeviceName = image_matching::siftBackendDeviceName(resolvedBackend, deviceIndex);
    plan.computeDeviceDisplayName =
        image_matching::siftBackendRuntimeDisplayName(resolvedBackend, deviceIndex);
    plan.backendReason = QStringLiteral("SIFT 计算设备：%1")
                             .arg(plan.computeDeviceDisplayName);
    return plan;
}

} // namespace matchphotos
} // namespace xjw
