#pragma once

#include <QString>

#include <cstdint>

namespace xjw
{
namespace matchphotos
{

enum class MatchPhotosExecutionBackend
{
    Cuda,
    Cpu
};

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
    bool enableGuidedMatching = false;
    MatchPhotosExecutionBackend executionBackend = MatchPhotosExecutionBackend::Cuda;
    bool backendFallback = false;
    QString backendReason;

    int maxImageDim = 2048;
    int maxKeypoints = 8192;
    // 已解析的特征提取配置必须与匹配缓存键共用，避免 profile 改变实际
    // 检测行为后仍误命中旧的 `.pimatch`。
    int featureSchemaVersion = 1;
    int featureRemoveBorders = 16;
    float siftDetectionThreshold = 0.0005f;

    QString reason;
    QString validationError;
};

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan);

} // namespace matchphotos
} // namespace xjw
