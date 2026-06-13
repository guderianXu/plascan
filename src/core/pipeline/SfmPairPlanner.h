#pragma once

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QString>
#include <QStringList>

#include <algorithm>

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
};

struct SfmPairPlan
{
    bool restrictPairs = false;
    bool autoRestricted = false;
    int allPairCount = 0;
    int knownCameraPairWindow = 0;
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
    plan.restrictPairs = true;
    plan.autoRestricted = true;
    plan.knownCameraPairWindow = window;

    QSet<QString> seen;
    for (int i = 0; i < imageCount; ++i)
    {
        const int last = std::min(imageCount - 1, i + window);
        for (int j = i + 1; j <= last; ++j)
        {
            const QString pairKey = canonicalSfmPairKey(images.at(i), images.at(j));
            if (pairKey.isEmpty() || seen.contains(pairKey))
            {
                continue;
            }
            seen.insert(pairKey);
            plan.allowedPairKeys.append(pairKey);
        }
    }

    return plan;
}

} // namespace gui
} // namespace xjw
