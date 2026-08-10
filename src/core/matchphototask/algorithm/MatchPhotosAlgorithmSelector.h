#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosOptions.h"

namespace xjw
{
namespace matchphotos
{

// 将用户预设和设备选择转换成具体算法计划。
// 策略通过注册表解析算法 ID，不在主流程里硬编码特征/匹配组合。
class MatchPhotosAlgorithmSelector
{
public:
    static MatchPhotosAlgorithmPlan select(const MatchPhotosOptions &options);
    static MatchPhotosAlgorithmPlan resolveExecutionBackend(
        const MatchPhotosOptions &options,
        MatchPhotosAlgorithmPlan plan,
        bool cudaSiftAvailable);
};

} // namespace matchphotos
} // namespace xjw
