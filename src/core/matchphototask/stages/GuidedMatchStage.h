#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

namespace xjw
{
    namespace matchphotos
    {

        // 初始双视几何建立后的第二轮 SIFT 引导匹配，必须在轨迹构建和落盘之前运行。
        class GuidedMatchStage
        {
        public:
            MatchPhotosStageReport run(const MatchPhotosContext& context,
                                       const MatchPhotosOptions& options,
                                       const MatchPhotosAlgorithmPlan& algorithmPlan,
                                       std::vector<MatchPhotosMatchRecord>* matchRecords) const;
        };

    } // namespace matchphotos
} // namespace xjw
