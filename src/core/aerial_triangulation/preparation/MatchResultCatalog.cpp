#include "preparation/MatchResultCatalog.h"

#include "ImageMatchFile.h"

#include <QDir>
#include <QFileInfo>
#include <QMap>

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw::aerial_triangulation
{
namespace
{

QString normalizedPath(const QString &path)
{
    QString value = QDir::cleanPath(QFileInfo(path.trimmed()).absoluteFilePath());
#if defined(Q_OS_WIN)
    value = value.toLower();
#endif
    return value;
}

bool sameImage(const QString &left, const QString &right)
{
    return !left.trimmed().isEmpty() && normalizedPath(left) == normalizedPath(right);
}

bool imageInSet(const QString &image, const QStringList &set)
{
    if (set.isEmpty())
    {
        return true;
    }
    return std::any_of(set.cbegin(), set.cend(),
                       [&](const QString &candidate)
                       {
                           return sameImage(image, candidate);
                       });
}

bool pairPassesFilter(const QString &imageA,
                      const QString &imageB,
                      const MatchResultCatalogConfig &config)
{
    if (!config.targetImagePath.trimmed().isEmpty() &&
        !sameImage(imageA, config.targetImagePath) &&
        !sameImage(imageB, config.targetImagePath))
    {
        return false;
    }
    return imageInSet(imageA, config.targetImagePaths) &&
           imageInSet(imageB, config.targetImagePaths);
}

QString variantKey(const MatchVariant &variant)
{
    return variant.algorithmId.trimmed().toLower() + QLatin1Char('\n') +
        QString::number(variant.algorithmVersion) + QLatin1Char('\n') +
        QString::fromLatin1(variant.configFingerprint.toHex());
}

bool betterVariant(const MatchVariant &left, const MatchVariant &right)
{
    if (left.compatible != right.compatible)
    {
        return left.compatible;
    }
    if (left.geometryPassed != right.geometryPassed)
    {
        return left.geometryPassed;
    }
    if (left.geometricVerifiedInliers != right.geometricVerifiedInliers)
    {
        return left.geometricVerifiedInliers > right.geometricVerifiedInliers;
    }
    if (left.totalMatches != right.totalMatches)
    {
        return left.totalMatches > right.totalMatches;
    }
    return left.modifiedTime > right.modifiedTime;
}

/**
 * @brief 统计几何内点在像面的空间覆盖率。
 *
 * 单纯依赖内点数量会高估集中在局部重复纹理上的像对。这里把两幅影像分别划分为
 * 4x4 网格，计算被几何内点覆盖的格子比例后取平均。坐标和内点标志均直接来自
 * `.pimatch`，无需重新读取影像、特征文件或运行几何验证。
 */
double geometricGridCoverage(const image_matching::ImageMatchShard &shard,
                             const image_matching::NeighborMatchBlock &block)
{
    constexpr int gridColumns = 4;
    constexpr int gridRows = 4;
    std::array<bool, gridColumns * gridRows> ownerOccupied{};
    std::array<bool, gridColumns * gridRows> peerOccupied{};

    const float ownerWidth = static_cast<float>(std::max<std::uint32_t>(1U, shard.owner.width));
    const float ownerHeight = static_cast<float>(std::max<std::uint32_t>(1U, shard.owner.height));
    const float peerWidth = static_cast<float>(std::max<std::uint32_t>(1U, block.peer.width));
    const float peerHeight = static_cast<float>(std::max<std::uint32_t>(1U, block.peer.height));

    auto occupy = [](float x,
                     float y,
                     float width,
                     float height,
                     std::array<bool, gridColumns * gridRows> *grid)
    {
        if (!grid || !std::isfinite(x) || !std::isfinite(y))
        {
            return;
        }
        const int column = std::clamp(
            static_cast<int>(std::floor(x / width * gridColumns)), 0, gridColumns - 1);
        const int row = std::clamp(
            static_cast<int>(std::floor(y / height * gridRows)), 0, gridRows - 1);
        (*grid)[static_cast<std::size_t>(row * gridColumns + column)] = true;
    };

    for (const image_matching::MatchRecord &match : block.matches)
    {
        if (!image_matching::hasFlag(match.flags,
                                     image_matching::MatchRecordFlag::GeometryInlier))
        {
            continue;
        }
        const image_matching::KeypointObservation *owner =
            block.findOwnerObservation(match.ownerFeatureId);
        if (!owner)
        {
            continue;
        }
        occupy(owner->x, owner->y, ownerWidth, ownerHeight, &ownerOccupied);
        occupy(match.peerX, match.peerY, peerWidth, peerHeight, &peerOccupied);
    }

    const auto occupiedRatio = [](const auto &grid)
    {
        return static_cast<double>(std::count(grid.cbegin(), grid.cend(), true)) /
            static_cast<double>(grid.size());
    };
    return 0.5 * (occupiedRatio(ownerOccupied) + occupiedRatio(peerOccupied));
}

void chooseBestVariant(MatchPairGroup *group)
{
    group->bestVariantIndex = -1;
    for (int index = 0; index < group->variants.size(); ++index)
    {
        if (!group->variants.at(index).compatible)
        {
            continue;
        }
        if (group->bestVariantIndex < 0 ||
            betterVariant(group->variants.at(index),
                          group->variants.at(group->bestVariantIndex)))
        {
            group->bestVariantIndex = index;
        }
    }
}

} // namespace

MatchResultCatalog::MatchResultCatalog(const MatchResultCatalogConfig &config)
    : _config(config)
{
}

QString MatchResultCatalog::canonicalPairKey(const QString &imageA,
                                             const QString &imageB)
{
    const QString left = normalizedPath(imageA);
    const QString right = normalizedPath(imageB);
    if (left.isEmpty() || right.isEmpty() || left == right)
    {
        return QString();
    }
    return left < right
        ? left + QLatin1Char('\n') + right
        : right + QLatin1Char('\n') + left;
}

QString MatchResultCatalog::algorithmDisplayLabel(const MatchVariant &variant)
{
    QString label = variant.algorithmId.trimmed();
    if (label.isEmpty())
    {
        return QStringLiteral("未知算法");
    }
    label.replace(QLatin1Char('_'), QLatin1Char('-'));
    return variant.algorithmVersion > 0
        ? QStringLiteral("%1 v%2").arg(label).arg(variant.algorithmVersion)
        : label;
}

MatchResultCatalogSummary MatchResultCatalog::scan() const
{
    MatchResultCatalogSummary summary;
    const QDir directory(_config.matchDirectory);
    const QString pattern = QStringLiteral("*") +
        QString::fromLatin1(image_matching::kImageMatchFileSuffix);
    const QFileInfoList files = directory.entryInfoList(
        QStringList{pattern}, QDir::Files, QDir::Name);
    summary.matchFileCount = files.size();

    QMap<QString, MatchPairGroup> groups;
    int processed = 0;
    for (const QFileInfo &fileInfo : files)
    {
        image_matching::ImageMatchShard shard;
        QString readError;
        if (!image_matching::ImageMatchFile::read(fileInfo.absoluteFilePath(),
                                                   &shard,
                                                   &readError))
        {
            ++summary.incompatibleVariantCount;
            ++processed;
            if (_config.progressCallback)
            {
                _config.progressCallback(processed, files.size());
            }
            continue;
        }

        for (const image_matching::NeighborMatchBlock &block : shard.neighbors)
        {
            if (!pairPassesFilter(shard.owner.path, block.peer.path, _config))
            {
                continue;
            }
            const QString pairKey = canonicalPairKey(shard.owner.path, block.peer.path);
            if (pairKey.isEmpty())
            {
                continue;
            }

            MatchPairGroup &group = groups[pairKey];
            group.pairKey = pairKey;
            if (normalizedPath(shard.owner.path) <= normalizedPath(block.peer.path))
            {
                group.imageA = shard.owner.path;
                group.imageB = block.peer.path;
            }
            else
            {
                group.imageA = block.peer.path;
                group.imageB = shard.owner.path;
            }

            MatchVariant candidate;
            candidate.imageA = group.imageA;
            candidate.imageB = group.imageB;
            candidate.algorithmId = block.algorithmId;
            candidate.algorithmVersion = block.algorithmVersion;
            candidate.configFingerprint = block.configFingerprint;
            candidate.matchFilePath = fileInfo.absoluteFilePath();
            candidate.peerMatchFilePath = image_matching::ImageMatchFile::filePathForImage(
                directory.absolutePath(), block.peer.path);
            candidate.totalMatches = static_cast<int>(block.rawMatchCount);
            candidate.geometricVerifiedInliers =
                static_cast<int>(block.geometryInlierCount);
            candidate.tiePointMatches = static_cast<int>(block.tiePointMatchCount);
            candidate.geometricCoverage = geometricGridCoverage(shard, block);
            candidate.geometryPassed = block.geometryPassed;
            candidate.compatible = true;
            candidate.status = QStringLiteral("compatible");
            candidate.modifiedTime = fileInfo.lastModified();

            const QString key = variantKey(candidate);
            auto existing = std::find_if(
                group.variants.begin(), group.variants.end(),
                [&](const MatchVariant &variant)
                {
                    return variantKey(variant) == key;
                });
            if (existing == group.variants.end())
            {
                group.variants.push_back(std::move(candidate));
            }
            else if (sameImage(shard.owner.path, group.imageA))
            {
                // 对称分片包含同一逻辑变体。固定选择规范 imageA 的分片，使目录
                // 扫描顺序不会改变 GUI 选中的文件路径。
                *existing = std::move(candidate);
            }
        }

        ++processed;
        if (_config.progressCallback)
        {
            _config.progressCallback(processed, files.size());
        }
    }

    summary.pairGroups.reserve(groups.size());
    for (auto it = groups.begin(); it != groups.end(); ++it)
    {
        MatchPairGroup group = it.value();
        std::sort(group.variants.begin(), group.variants.end(),
                  [](const MatchVariant &left, const MatchVariant &right)
                  {
                      return variantKey(left) < variantKey(right);
                  });
        chooseBestVariant(&group);
        summary.variantCount += group.variants.size();
        summary.compatibleVariantCount += static_cast<int>(std::count_if(
            group.variants.cbegin(), group.variants.cend(),
            [](const MatchVariant &variant)
            {
                return variant.compatible;
            }));
        summary.pairGroups.push_back(std::move(group));
    }
    summary.pairGroupCount = summary.pairGroups.size();
    return summary;
}

} // namespace xjw::aerial_triangulation
