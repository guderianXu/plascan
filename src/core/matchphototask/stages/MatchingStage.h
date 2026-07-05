#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 两两匹配阶段边界。它消费 PairSelectionResult，
// 后续实现可以只匹配已选 pair，而不是重新规划候选对。
class MatchingStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options,
                               const MatchPhotosAlgorithmPlan &algorithmPlan,
                               const PairSelectionResult &pairSelection,
                               std::vector<MatchPhotosMatchRecord> *matchRecords) const;
};

} // namespace matchphotos
} // namespace xjw
