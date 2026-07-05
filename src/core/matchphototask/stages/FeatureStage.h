#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 特征检测和描述子提取阶段边界。
// MatchPhotosTask 从影像对规划扩展出去后，这里会调用 feature_extractors。
class FeatureStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options,
                               const MatchPhotosAlgorithmPlan &algorithmPlan,
                               std::vector<MatchPhotosFeatureRecord> *featureRecords) const;
};

} // namespace matchphotos
} // namespace xjw
