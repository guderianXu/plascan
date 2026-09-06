#pragma once

#include "IncrementalSfm.h"

#include <optional>

namespace xjw
{

    /**
     * @brief “对齐照片”式大型相机块执行器。
     *
     * 返回空值表示当前网络保持单块并应继续普通 IncrementalSfm；有值表示已经完成
     * 所有独立块重建、稳健 Sim3 合并和最终全局增长。
     */
    class IndependentBlockReconstructor
    {
    public:
        explicit IndependentBlockReconstructor(IncrementalSfm& owner);

        std::optional<IncrementalSfmResult> runIfNeeded(SfmProgressCallback progressCb);

    private:
        IncrementalSfm& _owner;
    };

} // namespace xjw
