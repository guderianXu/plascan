#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <vector>

namespace xjw
{
namespace matchphotos
{

// 连接点轨迹阶段边界。最终多视图 track 管理由 TiePointTrackManager 负责。
class TrackBuildStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options,
                               const std::vector<MatchPhotosMatchRecord> &matchRecords,
                               MatchPhotosResult *result) const;
};

} // namespace matchphotos
} // namespace xjw
