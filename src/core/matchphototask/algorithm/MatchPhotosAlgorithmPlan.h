#pragma once

#include <QString>

namespace xjw
{
namespace matchphotos
{

// MatchPhotosTask 的算法计划。这里不实现具体算法，只描述本轮匹配照片流程
// 应该调用哪个特征提取器、哪个匹配器，以及为什么这么选。
struct MatchPhotosAlgorithmPlan
{
    QString strategyId;
    QString displayName;

    QString featureAlgorithm;
    QString featureSuffix;
    QString matcherAlgorithm;
    QString fallbackMatcherAlgorithm;

    bool needsFeatureStage = true;
    bool endToEndMatcher = false;
    bool preferCuda = false;
    bool rotationRobust = false;
    bool enableGuidedMatching = false;

    int maxImageDim = 2048;
    int maxKeypoints = 8192;

    QString reason;
    QString fallbackReason;
};

QString algorithmPlanSummary(const MatchPhotosAlgorithmPlan &plan);

} // namespace matchphotos
} // namespace xjw
