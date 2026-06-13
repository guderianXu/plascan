#pragma once

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace xjw
{
namespace gui
{

struct SfmPairPlannerOptions
{
    bool restrictPairs = false;
    QStringList allowedPairs;
    bool autoRestrictKnownCameraPairs = true;
    int knownCameraPairWindow = 4;
    int knownCameraAllPairsMaxImages = 20;
    int knownCameraSpatialNeighborCount = 8;
    std::vector<std::array<double, 3>> knownCameraCenters;
    std::vector<std::array<int, 2>> knownCameraOverlapPairs;
};

struct SfmPairPlan
{
    bool restrictPairs = false;
    bool autoRestricted = false;
    bool usedCameraOverlapPairs = false;
    bool usedSpatialCameraCenters = false;
    int allPairCount = 0;
    int knownCameraPairWindow = 0;
    int knownCameraSpatialNeighborCount = 0;
    int knownCameraOverlapPairCount = 0;
    QStringList allowedPairKeys;
};

inline QString canonicalSfmPath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

inline QString canonicalSfmPairKey(const QString &pathA, const QString &pathB)
{
    const QString normA = canonicalSfmPath(pathA);
    const QString normB = canonicalSfmPath(pathB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return (normA < normB)
        ? (normA + QStringLiteral("\n") + normB)
        : (normB + QStringLiteral("\n") + normA);
}

inline QStringList uniqueNonEmptyPairKeys(const QStringList &pairKeys)
{
    QStringList result;
    QSet<QString> seen;
    for (const QString &pairKey : pairKeys)
    {
        const QString trimmedKey = pairKey.trimmed();
        if (trimmedKey.isEmpty() || seen.contains(trimmedKey))
        {
            continue;
        }
        seen.insert(trimmedKey);
        result.append(trimmedKey);
    }
    return result;
}

inline bool hasCompleteCameraPathList(const QStringList &images, const QStringList &cameraPaths)
{
    if (images.isEmpty() || cameraPaths.size() != images.size())
    {
        return false;
    }

    for (int i = 0; i < images.size(); ++i)
    {
        if (canonicalSfmPath(images.at(i)).isEmpty() || canonicalSfmPath(cameraPaths.at(i)).isEmpty())
        {
            return false;
        }
    }
    return true;
}

inline bool hasCompleteKnownCameraCenters(int imageCount, const std::vector<std::array<double, 3>> &centers)
{
    if (imageCount <= 0 || centers.size() != static_cast<std::size_t>(imageCount))
    {
        return false;
    }

    for (const auto &center : centers)
    {
        if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2]))
        {
            return false;
        }
    }

    return true;
}

inline double squaredCenterDistance(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

inline void appendUniqueSfmPairKey(const QStringList &images,
                                   int indexA,
                                   int indexB,
                                   QSet<QString> *seen,
                                   QStringList *pairKeys)
{
    if (!seen || !pairKeys || indexA == indexB)
    {
        return;
    }

    const QString pairKey = canonicalSfmPairKey(images.at(indexA), images.at(indexB));
    if (pairKey.isEmpty() || seen->contains(pairKey))
    {
        return;
    }

    seen->insert(pairKey);
    pairKeys->append(pairKey);
}

inline SfmPairPlan planSfmMatchPairs(
    const QStringList &images,
    const QStringList &cameraPaths,
    const SfmPairPlannerOptions &options)
{
    SfmPairPlan plan;
    const int imageCount = images.size();
    plan.allPairCount = imageCount > 1 ? (imageCount * (imageCount - 1)) / 2 : 0;

    if (options.restrictPairs)
    {
        plan.restrictPairs = true;
        plan.allowedPairKeys = uniqueNonEmptyPairKeys(options.allowedPairs);
        return plan;
    }

    if (!options.autoRestrictKnownCameraPairs ||
        imageCount <= std::max(0, options.knownCameraAllPairsMaxImages) ||
        !hasCompleteCameraPathList(images, cameraPaths))
    {
        return plan;
    }

    const int window = std::max(1, options.knownCameraPairWindow);
    const int spatialNeighborCount = std::max(0, options.knownCameraSpatialNeighborCount);
    plan.restrictPairs = true;
    plan.autoRestricted = true;
    plan.knownCameraPairWindow = window;
    plan.knownCameraSpatialNeighborCount = spatialNeighborCount;

    QSet<QString> seen;
    for (const auto &pair : options.knownCameraOverlapPairs)
    {
        const int indexA = pair[0];
        const int indexB = pair[1];
        if (indexA < 0 || indexA >= imageCount || indexB < 0 || indexB >= imageCount)
        {
            continue;
        }
        appendUniqueSfmPairKey(images, indexA, indexB, &seen, &plan.allowedPairKeys);
    }

    if (!plan.allowedPairKeys.isEmpty())
    {
        plan.usedCameraOverlapPairs = true;
        plan.knownCameraOverlapPairCount = plan.allowedPairKeys.size();
        return plan;
    }

    for (int i = 0; i < imageCount; ++i)
    {
        const int last = std::min(imageCount - 1, i + window);
        for (int j = i + 1; j <= last; ++j)
        {
            appendUniqueSfmPairKey(images, i, j, &seen, &plan.allowedPairKeys);
        }
    }

    if (spatialNeighborCount > 0 && hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters))
    {
        plan.usedSpatialCameraCenters = true;
        for (int i = 0; i < imageCount; ++i)
        {
            std::vector<std::pair<double, int>> neighbors;
            neighbors.reserve(static_cast<std::size_t>(imageCount - 1));
            for (int j = 0; j < imageCount; ++j)
            {
                if (i == j)
                {
                    continue;
                }

                neighbors.emplace_back(squaredCenterDistance(options.knownCameraCenters[static_cast<std::size_t>(i)],
                                                             options.knownCameraCenters[static_cast<std::size_t>(j)]),
                                       j);
            }

            std::sort(neighbors.begin(), neighbors.end(), [](const auto &lhs, const auto &rhs)
            {
                if (lhs.first == rhs.first)
                {
                    return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
            });

            const int keep = std::min(spatialNeighborCount, static_cast<int>(neighbors.size()));
            for (int n = 0; n < keep; ++n)
            {
                appendUniqueSfmPairKey(images, i, neighbors[static_cast<std::size_t>(n)].second, &seen,
                                       &plan.allowedPairKeys);
            }
        }
    }

    return plan;
}

} // namespace gui
} // namespace xjw
