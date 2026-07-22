#pragma once

namespace xjw
{
namespace matchphotos
{

// 控制影像对规划的形态。算法选择属于 MatchPhotosOptions；
// 这里的策略只决定“哪些影像对值得尝试匹配”。
enum class PairSelectionMode
{
    Auto,
    Exhaustive,
    Sequence,
    ManualOnly
};

enum class PairSelectionPreset
{
    Auto,
    Fast,
    HighAccuracy,
    CpuCompatible,
    DifficultTexture
};

struct PairSelectionPolicy
{
    PairSelectionMode mode = PairSelectionMode::Auto;

    // 小项目可以承受全量匹配；大项目会回退到重叠候选、序列窗口等先验。
    int exhaustiveMaxImages = 20;
    int sequenceWindow = 4;
    /// 序列是否首尾闭环。转台/环拍数据可显式启用，以补充首尾跨界像对。
    bool closeSequenceLoop = false;

    // 0 表示不限制。后续可用于快速预览，或先匹配高优先级 pair 的分阶段流程。
    int maxPairs = 0;

    // 这些开关允许 MatchPhotosTask 复用已生成的 overlap 结果，
    // 而不是强制每个流程都运行较重的候选生成器。
    bool includeCameraOverlap = true;
    bool includeVocabularyOverlap = true;
    bool useSequenceFallback = true;
};

PairSelectionPolicy makePairSelectionPolicy(PairSelectionPreset preset);

} // namespace matchphotos
} // namespace xjw
