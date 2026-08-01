#include "ProjectMatchInputReader.h"

#include "ImageMatchFile.h"
#include "ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QMap>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::core::project
{
namespace
{

using xjw::image_matching::ImageMatchFile;
using xjw::image_matching::ImageMatchShard;
using xjw::image_matching::MatchRecordFlag;
using xjw::image_matching::NeighborMatchBlock;

/**
 * @brief 一个无向像对当前最优的持久化算法变体。
 *
 * 对称分片会各保存一份有向邻接块，同一像对也允许多个配置指纹并存。读取层先
 * 把每个候选转换为 SfM 的统一方向，再按几何内点数、连接点数和原始匹配数排序，
 * 最终只向轨迹构建器提交一个变体，避免重复观测改变 BA 权重。
 */
struct PairCandidate
{
    ProjectMatchPair pair;
    int geometricInlierCount = 0;
    int tiePointMatchCount = 0;
    int rawMatchCount = 0;
    std::int64_t createdTimeMs = 0;
};

QString normalizedPath(const QString &path)
{
    return xjw::common::project::normalizePath(path);
}

QString canonicalPairKey(const QString &left, const QString &right)
{
    const QString normalized_left = normalizedPath(left);
    const QString normalized_right = normalizedPath(right);
    if (normalized_left.isEmpty() || normalized_right.isEmpty() ||
        normalized_left == normalized_right)
    {
        return QString();
    }
    return normalized_left < normalized_right
        ? normalized_left + QLatin1Char('\n') + normalized_right
        : normalized_right + QLatin1Char('\n') + normalized_left;
}

bool betterCandidate(const PairCandidate &left, const PairCandidate &right)
{
    if (left.geometricInlierCount != right.geometricInlierCount)
    {
        return left.geometricInlierCount > right.geometricInlierCount;
    }
    if (left.tiePointMatchCount != right.tiePointMatchCount)
    {
        return left.tiePointMatchCount > right.tiePointMatchCount;
    }
    if (left.rawMatchCount != right.rawMatchCount)
    {
        return left.rawMatchCount > right.rawMatchCount;
    }
    return left.createdTimeMs > right.createdTimeMs;
}

/**
 * @brief 把 owner->peer 邻接块转换为 SfM 像对。
 *
 * 只保留通过几何模型验证的记录。owner 坐标由分片 observations 按稳定 featureId
 * 查找，peer 坐标直接来自邻接记录；两端 featureId 可跨多个像对合并成多视轨迹。
 */
PairCandidate makePairCandidate(const ImageMatchShard &shard,
                                const NeighborMatchBlock &block,
                                int owner_camera_index,
                                int peer_camera_index)
{
    PairCandidate candidate;
    candidate.pair.cameraIndexA = owner_camera_index;
    candidate.pair.cameraIndexB = peer_camera_index;
    candidate.pair.indexed = true;
    candidate.geometricInlierCount = static_cast<int>(block.geometryInlierCount);
    candidate.tiePointMatchCount = static_cast<int>(block.tiePointMatchCount);
    candidate.rawMatchCount = static_cast<int>(block.rawMatchCount);
    candidate.createdTimeMs = block.createdTimeMs;
    candidate.pair.observations.reserve(block.geometryInlierCount);

    for (const xjw::image_matching::MatchRecord &match : block.matches)
    {
        if (!xjw::image_matching::hasFlag(match.flags, MatchRecordFlag::GeometryInlier))
        {
            continue;
        }
        const xjw::image_matching::KeypointObservation *owner_observation =
            block.findOwnerObservation(match.ownerFeatureId);
        if (!owner_observation)
        {
            continue;
        }

        ProjectMatchObservationPair observation;
        observation.pixelA = {static_cast<double>(owner_observation->x),
                              static_cast<double>(owner_observation->y)};
        observation.pixelB = {static_cast<double>(match.peerX),
                              static_cast<double>(match.peerY)};
        observation.featureA = static_cast<xjw::FeatureIdx>(match.ownerFeatureId);
        observation.featureB = static_cast<xjw::FeatureIdx>(match.peerFeatureId);
        observation.score = std::clamp(static_cast<double>(match.confidence), 0.0, 1.0);
        candidate.pair.observations.push_back(observation);
    }
    return candidate;
}

} // namespace

int cameraIndexForImageToken(const QString &imageToken,
                             const QMap<QString, int> &cameraIndexByPath)
{
    if (imageToken.trimmed().isEmpty())
    {
        return -1;
    }

    const QString normalized_token = normalizedPath(imageToken);
    const auto direct = cameraIndexByPath.constFind(normalized_token);
    if (direct != cameraIndexByPath.constEnd())
    {
        return direct.value();
    }

    // 工程中的影像路径可能经过移动或打包。受控 token 匹配仅在规范路径无法命中
    // 时启用，并沿用 ProjectCommonUtils 对同名歧义的约束。
    for (auto it = cameraIndexByPath.constBegin(); it != cameraIndexByPath.constEnd(); ++it)
    {
        if (xjw::common::project::pathTokenMatchesImage(imageToken, it.key()))
        {
            return it.value();
        }
    }
    return -1;
}

bool readProjectMatchInput(const QJsonObject &meta,
                           const QStringList &selectedImages,
                           int minMatches,
                           ProjectMatchInput *input)
{
    if (!input)
    {
        return false;
    }
    *input = {};

    // 第一阶段：按 selectedImages 的集合约束建立当前相机索引。数组实际顺序沿用
    // 工程 images，以保证相机 JSON、影像路径和后续事务式回写始终一一对应。
    QSet<QString> selected_normalized;
    for (const QString &path : selectedImages)
    {
        selected_normalized.insert(normalizedPath(path));
    }

    const QJsonArray image_array = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : image_array)
    {
        const QJsonObject object = value.toObject();
        const QString normalized_path =
            normalizedPath(object.value(QStringLiteral("path")).toString());
        if (!selected_normalized.contains(normalized_path))
        {
            continue;
        }

        xjw::Camera camera;
        if (!xjw::common::project::cameraFromJson(
                object.value(QStringLiteral("camera")).toObject(), &camera))
        {
            continue;
        }

        input->cameraIndexByPath[normalized_path] = static_cast<int>(input->cameras.size());
        input->cameras.push_back(camera);
        input->imagePathByIndex.append(normalized_path);
        input->beforeCamMeta.insert(
            normalized_path, object.value(QStringLiteral("camera")).toObject());
    }

    // 第二阶段：每个 image_match_results 记录只指向一幅影像的唯一分片。使用
    // output 路径去重后读取，不扫描目录，也不依赖文件名编码影像对。
    QSet<QString> visited_files;
    QMap<QString, PairCandidate> best_pairs;
    const QJsonArray match_results =
        meta.value(QStringLiteral("image_match_results")).toArray();
    for (const QJsonValue &value : match_results)
    {
        const QString output_path =
            QFileInfo(value.toObject().value(QStringLiteral("output")).toString())
                .absoluteFilePath();
        const QString normalized_output = normalizedPath(output_path);
        if (normalized_output.isEmpty() || visited_files.contains(normalized_output) ||
            !QFileInfo::exists(output_path))
        {
            continue;
        }
        visited_files.insert(normalized_output);

        ImageMatchShard shard;
        QString read_error;
        if (!ImageMatchFile::read(output_path, &shard, &read_error))
        {
            continue;
        }
        const int owner_index = cameraIndexForImageToken(
            shard.owner.path, input->cameraIndexByPath);
        if (owner_index < 0)
        {
            continue;
        }

        for (const NeighborMatchBlock &block : shard.neighbors)
        {
            if (!block.geometryPassed)
            {
                continue;
            }
            const int peer_index = cameraIndexForImageToken(
                block.peer.path, input->cameraIndexByPath);
            const QString pair_key = canonicalPairKey(shard.owner.path, block.peer.path);
            if (peer_index < 0 || peer_index == owner_index || pair_key.isEmpty())
            {
                continue;
            }

            PairCandidate candidate = makePairCandidate(
                shard, block, owner_index, peer_index);
            if (minMatches > 0 &&
                static_cast<int>(candidate.pair.observations.size()) < minMatches)
            {
                continue;
            }

            const auto existing = best_pairs.constFind(pair_key);
            if (existing == best_pairs.constEnd() || betterCandidate(candidate, existing.value()))
            {
                best_pairs[pair_key] = std::move(candidate);
            }
        }
    }

    // 第三阶段：QMap 的规范 pair key 保证输出顺序稳定，便于复现实验和单元测试。
    input->pairs.reserve(static_cast<std::size_t>(best_pairs.size()));
    for (auto it = best_pairs.begin(); it != best_pairs.end(); ++it)
    {
        input->indexedObservationCount +=
            static_cast<int>(it.value().pair.observations.size());
        input->pairs.push_back(std::move(it.value().pair));
    }
    return true;
}

} // namespace xjw::core::project
