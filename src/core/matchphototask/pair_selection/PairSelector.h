#pragma once

#include "PairSelectionPolicy.h"
#include "PairTypes.h"

namespace xjw
{
    namespace matchphotos
    {

        // Auto 模式的通用/参考预选由 PlaMatchHctPairPreselector 统一负责。
        // PairSelector 只保留无需描述子的显式全量、序列和手动模式。
        struct PairSelectionInput
        {
            QStringList images;

            // 手动影像对键使用 PairTypes::makePairKey 的标准 "pathA\npathB" 格式，
            // 这样 GUI、CLI 和 task 层可以直接共享。
            QStringList manualPairKeys;
        };

        // PairSelector 不提取特征，也不做几何验证。
        class PairSelector
        {
        public:
            static PairSelectionResult
            select(const PairSelectionInput& input, const PairSelectionPolicy& policy, QString* errorMessage = nullptr);
        };

    } // namespace matchphotos
} // namespace xjw
