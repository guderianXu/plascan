#pragma once

#include <QString>

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
    bool enableGuidedMatching = false;

    int maxImageDim = 2048;
    int maxKeypoints = 8192;

    QString reason;
    QString validationError;
};

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan);

} // namespace matchphotos
} // namespace xjw
