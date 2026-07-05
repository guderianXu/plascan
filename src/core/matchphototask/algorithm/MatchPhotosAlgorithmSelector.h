#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosOptions.h"

namespace xjw
{
namespace matchphotos
{

// 将用户预设和设备选择转换成具体算法计划。
// 当前策略固定以 SIFT + LightGlue 为主线，避免在主流程里暴露算法商店。
class MatchPhotosAlgorithmSelector
{
public:
    static MatchPhotosAlgorithmPlan select(const MatchPhotosOptions &options);
};

} // namespace matchphotos
} // namespace xjw
