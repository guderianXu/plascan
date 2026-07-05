#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 将验证后的两两匹配合并为多视图轨迹的阶段边界。
// 这里应复用 sfm/tracks，而不是重新实现一套轨迹构建器。
class TrackBuildStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options) const;
};

} // namespace matchphotos
} // namespace xjw
