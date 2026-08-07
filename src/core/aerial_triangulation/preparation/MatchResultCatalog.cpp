#include "preparation/MatchResultCatalog.h"

#include "ImageMatchIndexFile.h"

#include <QDir>
#include <QFileInfo>
#include <QMap>
#include <QSet>

#include <algorithm>
#include <atomic>
#include <future>
#include <mutex>
#include <thread>

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

struct TargetImageFilter
{
    QString targetStableId;
    QString targetNormalizedPath;
    QSet<QString> targetStableIds;
    QSet<QString> targetNormalizedPaths;

    bool contains(const image_matching::ImageIdentity &identity) const
    {
        return targetStableIds.contains(identity.stableId) ||
            targetNormalizedPaths.contains(normalizedPath(identity.path));
    }

    bool isTarget(const image_matching::ImageIdentity &identity) const
    {
        return identity.stableId == targetStableId ||
            normalizedPath(identity.path) == targetNormalizedPath;
    }

    bool accepts(const image_matching::ImageIdentity &owner,
                 const image_matching::ImageIdentity &peer) const
    {
        if (!targetStableId.isEmpty() &&
            !isTarget(owner) && !isTarget(peer))
        {
            return false;
        }
        return targetStableIds.isEmpty() ||
            (contains(owner) && contains(peer));
    }
};

TargetImageFilter makeTargetImageFilter(const MatchResultCatalogConfig &config)
{
    TargetImageFilter filter;
    if (!config.targetImagePath.trimmed().isEmpty())
    {
        filter.targetStableId = image_matching::ImageMatchFile::stableImageId(
            config.targetImagePath);
        filter.targetNormalizedPath = normalizedPath(config.targetImagePath);
    }
    filter.targetStableIds.reserve(config.targetImagePaths.size());
    for (const QString &path : config.targetImagePaths)
    {
        if (!path.trimmed().isEmpty())
        {
            filter.targetStableIds.insert(image_matching::ImageMatchFile::stableImageId(path));
            filter.targetNormalizedPaths.insert(normalizedPath(path));
        }
    }
    return filter;
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

int catalogWorkerCount(const MatchResultCatalogConfig &config, int fileCount)
{
    if (fileCount <= 1)
    {
        return std::max(0, fileCount);
    }
    const int requested = config.maxConcurrency > 0
        ? config.maxConcurrency
        : static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    return std::clamp(requested, 1, std::min(8, fileCount));
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

    struct LoadedIndex
    {
        bool valid = false;
        image_matching::ImageMatchFileIndex index;
        image_matching::ImageMatchIndexLoadSource source =
            image_matching::ImageMatchIndexLoadSource::RebuiltFromMatchFile;
    };

    QVector<LoadedIndex> loaded(files.size());
    std::atomic<int> nextFile{0};
    int processedFiles = 0;
    std::mutex progressMutex;
    auto worker = [&]()
    {
        while (true)
        {
            const int fileIndex = nextFile.fetch_add(1, std::memory_order_relaxed);
            if (fileIndex >= files.size())
            {
                break;
            }
            QString readError;
            loaded[fileIndex].valid = image_matching::ImageMatchIndexFile::load(
                files.at(fileIndex).absoluteFilePath(),
                &loaded[fileIndex].index,
                &loaded[fileIndex].source,
                &readError);

            {
                const std::lock_guard<std::mutex> lock(progressMutex);
                ++processedFiles;
                if (_config.progressCallback)
                {
                    _config.progressCallback(processedFiles, files.size());
                }
            }
        }
    };

    const int workerCount = catalogWorkerCount(_config, files.size());
    if (workerCount == 1)
    {
        worker();
    }
    else
    {
        std::vector<std::future<void>> futures;
        futures.reserve(static_cast<std::size_t>(workerCount));
        for (int workerIndex = 0; workerIndex < workerCount; ++workerIndex)
        {
            futures.push_back(std::async(std::launch::async, worker));
        }
        for (auto &future : futures)
        {
            future.get();
        }
    }

    QMap<QString, MatchPairGroup> groups;
    const TargetImageFilter targetFilter = makeTargetImageFilter(_config);
    for (int fileIndex = 0; fileIndex < files.size(); ++fileIndex)
    {
        const QFileInfo &fileInfo = files.at(fileIndex);
        const LoadedIndex &file = loaded.at(fileIndex);
        if (!file.valid)
        {
            ++summary.incompatibleVariantCount;
            continue;
        }
        switch (file.source)
        {
        case image_matching::ImageMatchIndexLoadSource::MemoryCache:
            ++summary.memoryIndexHitCount;
            break;
        case image_matching::ImageMatchIndexLoadSource::PersistentIndex:
            ++summary.persistentIndexHitCount;
            break;
        case image_matching::ImageMatchIndexLoadSource::RebuiltFromMatchFile:
            ++summary.rebuiltIndexCount;
            break;
        }

        const image_matching::ImageMatchFileIndex &index = file.index;
        for (const image_matching::ImageMatchNeighborIndex &neighbor : index.neighbors)
        {
            if (!targetFilter.accepts(index.owner, neighbor.peer))
            {
                continue;
            }
            const QString pairKey = canonicalPairKey(index.owner.path, neighbor.peer.path);
            if (pairKey.isEmpty())
            {
                continue;
            }

            MatchPairGroup &group = groups[pairKey];
            group.pairKey = pairKey;
            const bool ownerIsCanonical =
                normalizedPath(index.owner.path) <= normalizedPath(neighbor.peer.path);
            if (ownerIsCanonical)
            {
                group.imageA = index.owner.path;
                group.imageB = neighbor.peer.path;
            }
            else
            {
                group.imageA = neighbor.peer.path;
                group.imageB = index.owner.path;
            }

            MatchVariant candidate;
            candidate.imageA = group.imageA;
            candidate.imageB = group.imageB;
            candidate.algorithmId = neighbor.algorithmId;
            candidate.algorithmVersion = neighbor.algorithmVersion;
            candidate.configFingerprint = neighbor.configFingerprint;
            candidate.matchFilePath = fileInfo.absoluteFilePath();
            candidate.peerMatchFilePath = image_matching::ImageMatchFile::filePathForImage(
                directory.absolutePath(), neighbor.peer.path);
            candidate.totalMatches = static_cast<int>(neighbor.rawMatchCount);
            candidate.geometricVerifiedInliers =
                static_cast<int>(neighbor.geometryInlierCount);
            candidate.tiePointMatches = static_cast<int>(neighbor.tiePointMatchCount);
            candidate.geometricCoverage = neighbor.geometricCoverage;
            candidate.geometryPassed = neighbor.geometryPassed;
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
            else if (ownerIsCanonical)
            {
                // 对称分片包含同一逻辑变体。固定选择规范 imageA 的分片，使目录
                // 扫描顺序不会改变 GUI 选中的文件路径。
                *existing = std::move(candidate);
            }
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
