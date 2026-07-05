#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <vector>

namespace xjw
{
namespace matchphotos
{

// F/E/H 几何验证和内点过滤阶段边界。
// 它独立于 MatchingStage，便于传统匹配器和学习型匹配器共用几何检查。
class GeometryVerifyStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options,
                               std::vector<MatchPhotosMatchRecord> *matchRecords) const;
};

} // namespace matchphotos
} // namespace xjw
