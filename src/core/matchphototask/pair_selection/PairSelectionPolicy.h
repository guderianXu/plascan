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

    // 0 表示不额外裁剪。参考位姿可能因错误地面模型或已有弯曲而把大部分影像
    // 判为重叠；按每张影像的最高分邻居投票可以把候选规模稳定在 O(N * K)。
    int cameraOverlapTopKPerImage = 0;

    // 同时启用参考与通用预选时，词汇结果只应作为少量闭环补充，不能重新把
    // 有界的位姿候选扩张成接近全量匹配。0 保持通用匹配工作流的既有行为。
    int vocabularyTopKPerImage = 0;
};

PairSelectionPolicy makePairSelectionPolicy(PairSelectionPreset preset);

} // namespace matchphotos
} // namespace xjw
