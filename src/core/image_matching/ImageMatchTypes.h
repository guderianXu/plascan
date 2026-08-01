#pragma once

/**
 * @file ImageMatchTypes.h
 * @brief 影像匹配模块在内存与二进制文件之间共享的数据契约。
 *
 * 这里刻意不保存描述子。描述子只在一次匹配任务的内存缓存中存在，任务完成后
 * 即释放；持久化结果只保留 SfM、连接点查看和质量分析真正需要的观测与统计。
 * 这种边界避免下游重新依赖某一种特征描述子格式，也让以后增加新的匹配算法时
 * 不必修改空三和 GUI 的读取代码。
 */

#include <QByteArray>
#include <QString>

#include <array>
#include <cstdint>
#include <vector>

namespace xjw::image_matching
{

/// 当前唯一受支持的 `.pimatch` 容器版本。旧格式不会在核心流程中隐式兼容。
inline constexpr std::uint32_t kImageMatchFormatVersion = 1;

/// 二进制文件扩展名；一个参与匹配的影像对应一个该类型文件。
inline constexpr const char *kImageMatchFileSuffix = ".pimatch";

/**
 * @brief 描述影像身份及用于缓存失效判断的轻量指纹。
 *
 * stableId 由规范路径生成，负责文件命名和邻接关系；fileSize/modifiedTimeMs
 * 用于发现影像内容已经变化。它们不是摄影测量 ImageId，不能参与数组索引。
 */
struct ImageIdentity
{
    QString stableId;
    QString path;
    QString displayName;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t fileSize = 0;
    std::int64_t modifiedTimeMs = 0;

    bool isValid() const;
};

/**
 * @brief 当前影像中被至少一个匹配引用的局部特征观测。
 *
 * featureId 只在同一个算法版本与配置指纹内稳定。不同算法或不同提取参数都可能
 * 从 0 重新编号，因此观测必须由 NeighborMatchBlock 按算法变体局部持有，不能放在
 * 整个影像分片的共享命名空间中。
 */
struct KeypointObservation
{
    std::uint32_t featureId = 0;
    float x = 0.0f;
    float y = 0.0f;
    float scale = 1.0f;
    float orientation = -1.0f;
    float response = 0.0f;
};

/// 每条匹配的处理状态。位标志允许以后增加新的质量阶段而不改变记录布局。
enum class MatchRecordFlag : std::uint32_t
{
    None = 0,
    GeometryInlier = 1U << 0U,
    InTiePointTrack = 1U << 1U,
    MaskAccepted = 1U << 2U,
    Guided = 1U << 3U
};

constexpr MatchRecordFlag operator|(MatchRecordFlag left, MatchRecordFlag right)
{
    return static_cast<MatchRecordFlag>(static_cast<std::uint32_t>(left) |
                                        static_cast<std::uint32_t>(right));
}

constexpr MatchRecordFlag operator&(MatchRecordFlag left, MatchRecordFlag right)
{
    return static_cast<MatchRecordFlag>(static_cast<std::uint32_t>(left) &
                                        static_cast<std::uint32_t>(right));
}

constexpr bool hasFlag(MatchRecordFlag value, MatchRecordFlag flag)
{
    return (value & flag) != MatchRecordFlag::None;
}

/**
 * @brief 从当前分片的一个观测指向相邻影像观测的匹配边。
 *
 * 当前影像坐标通过 ownerFeatureId 在所属邻接块的 ownerObservations 中查找。相邻影像坐标直接
 * 保存在记录内，因此查看当前影像的全部匹配只需读取一个文件，不必随机访问
 * 其他影像分片。residualPixels 是几何模型下的像素残差；尚未计算时为负数。
 */
struct MatchRecord
{
    std::uint32_t ownerFeatureId = 0;
    std::uint32_t peerFeatureId = 0;
    float peerX = 0.0f;
    float peerY = 0.0f;
    float confidence = 0.0f;
    float residualPixels = -1.0f;
    MatchRecordFlag flags = MatchRecordFlag::None;
};

enum class GeometryModel : std::uint8_t
{
    None = 0,
    Fundamental = 1,
    Essential = 2,
    Homography = 3,
    Affine = 4
};

/**
 * @brief 当前影像与一个相邻影像的一种算法变体。
 *
 * algorithmId、algorithmVersion、configFingerprint 与 modelFingerprint 一起构成
 * 完整缓存键。以后新增匹配算法或更新模型权重时，可以在同一个影像文件中并存
 * 多个变体，而无需在文件名中编码算法。
 */
struct NeighborMatchBlock
{
    ImageIdentity peer;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    QByteArray configFingerprint;
    QByteArray modelFingerprint;
    std::int64_t createdTimeMs = 0;

    std::uint32_t rawMatchCount = 0;
    std::uint32_t geometryInlierCount = 0;
    std::uint32_t tiePointMatchCount = 0;
    bool geometryPassed = false;
    GeometryModel geometryModel = GeometryModel::None;
    std::array<double, 9> geometryMatrix{};
    // 本影像观测表属于当前算法变体。相同观测可被该变体下的多个像对块分别保存，
    // 以换取每个 `.pimatch` 分片和每个邻接块都能独立读取，不依赖全局特征文件。
    std::vector<KeypointObservation> ownerObservations;
    std::vector<MatchRecord> matches;

    bool isCompatible(const QString &requestedAlgorithmId,
                      std::uint32_t requestedAlgorithmVersion,
                      const QByteArray &requestedConfigFingerprint,
                      const QByteArray &requestedModelFingerprint = {}) const;
    const KeypointObservation *findOwnerObservation(std::uint32_t featureId) const;
    void normalize();
};

/**
 * @brief 一个影像对应的完整匹配分片。
 *
 * neighbors 可以包含多个相邻影像以及同一像对的多种算法变体。每个邻接块独立
 * 管理本影像观测的 featureId 命名空间；写入前会进行排序和去重，以保证相同输入
 * 生成稳定的二进制内容。
 */
struct ImageMatchShard
{
    ImageIdentity owner;
    std::vector<NeighborMatchBlock> neighbors;

    const NeighborMatchBlock *findNeighbor(const QString &peerStableId,
                                           const QString &algorithmId,
                                           std::uint32_t algorithmVersion,
                                           const QByteArray &configFingerprint,
                                           const QByteArray &modelFingerprint = {}) const;
    void normalize();
};

/**
 * @brief 匹配和几何验证阶段使用的对称像对结果。
 *
 * 该结构只在任务内存中存在。ImageMatchRepository 会把它分别转换成 image0 和
 * image1 的有向邻接块，并在写入第二个分片时自动交换观测方向。
 */
struct PairCorrespondence
{
    KeypointObservation observation0;
    KeypointObservation observation1;
    float confidence = 0.0f;
    float residualPixels = -1.0f;
    MatchRecordFlag flags = MatchRecordFlag::None;
};

struct PairMatchData
{
    ImageIdentity image0;
    ImageIdentity image1;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    QByteArray configFingerprint;
    QByteArray modelFingerprint;
    std::int64_t createdTimeMs = 0;

    std::uint32_t rawMatchCount = 0;
    std::uint32_t geometryInlierCount = 0;
    std::uint32_t tiePointMatchCount = 0;
    bool geometryPassed = false;
    GeometryModel geometryModel = GeometryModel::None;
    std::array<double, 9> geometryMatrix{};
    std::vector<PairCorrespondence> correspondences;
};

} // namespace xjw::image_matching
