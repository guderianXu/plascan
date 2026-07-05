#pragma once

#include "OverlapAnalyzer.h"
#include "PairSelectionPolicy.h"
#include "PairTypes.h"
#include "VocabularyOverlapRetriever.h"

#include <array>
#include <vector>

namespace xjw
{
namespace matchphotos
{

// 输入刻意只接收“已经算好的信号”。相机足迹投影、词汇召回等重计算
// 继续留在 overlap/ 模块中，PairSelector 只负责合并结果。
struct PairSelectionInput
{
    QStringList images;

    // 手动影像对键使用 PairTypes::makePairKey 的标准 "pathA\npathB" 格式，
    // 这样 GUI、CLI 和 task 层可以直接共享。
    QStringList manualPairKeys;
    std::vector<std::array<int, 2>> knownCameraOverlapPairs;

    // 可选的外部结果借用指针。PairSelector 只在 select() 调用期间读取。
    const OverlapAnalysisResult *cameraOverlapResult = nullptr;
    const VocabularyOverlapResult *vocabularyOverlapResult = nullptr;
};

// PairSelector 是独立候选生成器和匹配阶段之间的合并层。
// 它不提取特征，也不做几何验证。
class PairSelector
{
public:
    static PairSelectionResult select(const PairSelectionInput &input,
                                      const PairSelectionPolicy &policy,
                                      QString *errorMessage = nullptr);
};

} // namespace matchphotos
} // namespace xjw
