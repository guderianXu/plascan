#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
namespace matchphotos
{

// 可选的第二轮引导匹配阶段。
// 在初始几何关系建立后，这里可以追加引导匹配对或修补弱轨迹。
class GuidedMatchStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options) const;
};

} // namespace matchphotos
} // namespace xjw
