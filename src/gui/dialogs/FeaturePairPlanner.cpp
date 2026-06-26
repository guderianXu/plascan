#include "FeaturePairPlanner.h"

#include "SfmPairPlanner.h"

#include <QFileInfo>
#include <QHash>
#include <QSet>

#include <algorithm>

namespace xjw::gui
{

namespace
{

QString guiPairKey(const QString &lhs, const QString &rhs)
{
    if (lhs == rhs)
    {
        return {};
    }
    return (lhs < rhs)
        ? QStringLiteral("%1__%2").arg(lhs, rhs)
        : QStringLiteral("%1__%2").arg(rhs, lhs);
}

QString pipelinePairKey(const QString &lhs, const QString &rhs)
{
    if (lhs == rhs)
    {
        return {};
    }
    return QStringLiteral("%1|%2").arg(lhs, rhs);
}

QStringList convertCanonicalPairsToGuiPairs(const QStringList &canonicalPairs,
                                            const QHash<QString, QString> &nameByCanonicalPath)
{
    QStringList pairs;
    QSet<QString> seen;
    pairs.reserve(canonicalPairs.size());
    for (const QString &canonicalPair : canonicalPairs)
    {
        const QStringList parts = canonicalPair.split(QLatin1Char('\n'));
        if (parts.size() != 2)
        {
            continue;
        }

        const QString lhs = nameByCanonicalPath.value(parts.at(0));
        const QString rhs = nameByCanonicalPath.value(parts.at(1));
        const QString pair = guiPairKey(lhs, rhs);
        if (pair.isEmpty() || seen.contains(pair))
        {
            continue;
        }
        seen.insert(pair);
        pairs.append(pair);
    }
    return pairs;
}

QStringList convertCanonicalPairsToPipelinePairs(const QStringList &canonicalPairs,
                                                 const QHash<QString, QString> &pathByCanonicalPath)
{
    QStringList pairs;
    QSet<QString> seen;
    pairs.reserve(canonicalPairs.size());
    for (const QString &canonicalPair : canonicalPairs)
    {
        const QStringList parts = canonicalPair.split(QLatin1Char('\n'));
        if (parts.size() != 2)
        {
            continue;
        }

        const QString lhs = pathByCanonicalPath.value(parts.at(0));
        const QString rhs = pathByCanonicalPath.value(parts.at(1));
        const QString pair = pipelinePairKey(lhs, rhs);
        if (pair.isEmpty() || seen.contains(pair))
        {
            continue;
        }
        seen.insert(pair);
        pairs.append(pair);
    }
    return pairs;
}

QStringList makeExhaustiveOrWindowPairs(const QStringList &names, int window)
{
    QStringList pairs;
    for (int i = 0; i < names.size(); ++i)
    {
        const int end = std::min(static_cast<int>(names.size()), i + window + 1);
        for (int j = i + 1; j < end; ++j)
        {
            pairs.append(guiPairKey(names.at(i), names.at(j)));
        }
    }
    return pairs;
}

QStringList makeExhaustiveOrWindowPathPairs(const QStringList &paths, int window)
{
    QStringList pairs;
    for (int i = 0; i < paths.size(); ++i)
    {
        const int end = std::min(static_cast<int>(paths.size()), i + window + 1);
        for (int j = i + 1; j < end; ++j)
        {
            pairs.append(pipelinePairKey(paths.at(i), paths.at(j)));
        }
    }
    return pairs;
}

} // namespace

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options)
{
    QStringList names;
    names.reserve(imageBaseNames.size());
    for (const QString &name : imageBaseNames)
    {
        const QString trimmed = name.trimmed();
        if (!trimmed.isEmpty())
        {
            names.append(trimmed);
        }
    }

    const int count = names.size();
    if (count < 2)
    {
        return {};
    }

    const bool exhaustive = count <= std::max(2, options.exhaustiveMaxImages);
    const int window = exhaustive ? (count - 1) : std::max(1, options.sequentialWindow);
    if (exhaustive &&
        options.knownCameraOverlapPairs.empty() &&
        options.knownCameraCenters.empty())
    {
        return makeExhaustiveOrWindowPairs(names, window);
    }

    QStringList plannerImages;
    plannerImages.reserve(names.size());
    QHash<QString, QString> nameByCanonicalPath;
    for (const QString &name : names)
    {
        plannerImages.append(name);
        nameByCanonicalPath.insert(canonicalSfmPath(name), name);
    }

    SfmPairPlannerOptions pairOptions;
    pairOptions.autoRestrictKnownCameraPairs = !exhaustive;
    pairOptions.knownCameraAllPairsMaxImages = std::max(2, options.exhaustiveMaxImages);
    pairOptions.knownCameraPairWindow = std::max(1, options.sequentialWindow);
    pairOptions.knownCameraSpatialNeighborCount = std::max(0, options.spatialNeighborCount);
    pairOptions.knownCameraCenters = options.knownCameraCenters;
    pairOptions.knownCameraOverlapPairs = options.knownCameraOverlapPairs;
    pairOptions.knownCameraOverlapMaxExpansion = options.knownCameraOverlapMaxExpansion;

    const SfmPairPlan pairPlan = planSfmMatchPairs(plannerImages, QStringList(), pairOptions);
    if (!pairPlan.restrictPairs)
    {
        return makeExhaustiveOrWindowPairs(names, window);
    }

    QStringList pairs = convertCanonicalPairsToGuiPairs(pairPlan.allowedPairKeys, nameByCanonicalPath);
    if (pairs.isEmpty())
    {
        return makeExhaustiveOrWindowPairs(names, window);
    }

    return pairs;
}

QStringList planFeatureMatchPairPaths(const QStringList &imagePaths,
                                      const FeaturePairPlannerOptions &options)
{
    QStringList paths;
    paths.reserve(imagePaths.size());
    for (const QString &path : imagePaths)
    {
        const QString trimmed = path.trimmed();
        if (!trimmed.isEmpty())
        {
            paths.append(trimmed);
        }
    }

    const int count = paths.size();
    if (count < 2)
    {
        return {};
    }

    const bool exhaustive = count <= std::max(2, options.exhaustiveMaxImages);
    const int window = exhaustive ? (count - 1) : std::max(1, options.sequentialWindow);
    if (exhaustive &&
        options.knownCameraOverlapPairs.empty() &&
        options.knownCameraCenters.empty())
    {
        return makeExhaustiveOrWindowPathPairs(paths, window);
    }

    QHash<QString, QString> pathByCanonicalPath;
    for (const QString &path : paths)
    {
        pathByCanonicalPath.insert(canonicalSfmPath(path), path);
    }

    SfmPairPlannerOptions pairOptions;
    pairOptions.autoRestrictKnownCameraPairs = !exhaustive;
    pairOptions.knownCameraAllPairsMaxImages = std::max(2, options.exhaustiveMaxImages);
    pairOptions.knownCameraPairWindow = std::max(1, options.sequentialWindow);
    pairOptions.knownCameraSpatialNeighborCount = std::max(0, options.spatialNeighborCount);
    pairOptions.knownCameraCenters = options.knownCameraCenters;
    pairOptions.knownCameraOverlapPairs = options.knownCameraOverlapPairs;
    pairOptions.knownCameraOverlapMaxExpansion = options.knownCameraOverlapMaxExpansion;

    const SfmPairPlan pairPlan = planSfmMatchPairs(paths, QStringList(), pairOptions);
    if (!pairPlan.restrictPairs)
    {
        return makeExhaustiveOrWindowPathPairs(paths, window);
    }

    QStringList pairs = convertCanonicalPairsToPipelinePairs(pairPlan.allowedPairKeys, pathByCanonicalPath);
    if (pairs.isEmpty())
    {
        return makeExhaustiveOrWindowPathPairs(paths, window);
    }

    return pairs;
}

} // namespace xjw::gui
