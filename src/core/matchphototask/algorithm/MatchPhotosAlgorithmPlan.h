#pragma once

#include <QString>

#include "MatchPhotosOptions.h"
#include "sift/SiftBackendType.h"

#include <cstdint>

namespace xjw
{
namespace matchphotos
{

// MatchPhotosTask 的已解析算法计划。算法身份由注册表中的 id + version 唯一确定，
// 不再暴露“特征算法 + 匹配算法 + 文件后缀”的历史组合。
struct MatchPhotosAlgorithmPlan
{
    QString strategyId;
    QString displayName;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;

    bool valid = false;
    bool extractsFeaturesInMemory = true;
    bool requiresCuda = true;
    bool preferCuda = false;
    bool rotationRobust = false;
    bool requiresColorInput = false;
    bool suppliesCoarsePairPreselection = false;
    bool supportsBatchFeatureMatching = false;
    GuidedMatchingMode guidedMatchingMode = GuidedMatchingMode::Disabled;
    image_matching::SiftComputeBackend executionBackend =
        image_matching::SiftComputeBackend::Automatic;
    bool backendFallback = false;
    QString backendReason;
    QString computeDeviceName;
    QString computeDeviceDisplayName;

    int alignmentDownscale = 1;
    int maxImageDim = 0;
    int maxKeypoints = 8192;
    // 已解析的特征提取配置必须与匹配缓存键共用，避免 profile 改变实际
    // 检测行为后仍误命中旧的 `.pimatch`。
    int featureSchemaVersion = 2;
    int featureRemoveBorders = 16;
    float siftDetectionThreshold = 0.0005f;
    float siftContrastThreshold = 0.02f;
    bool lowTextureRecovery = false;

    QString reason;
    QString validationError;
};

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan);

} // namespace matchphotos
} // namespace xjw
