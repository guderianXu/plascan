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

SfmPairPlannerOptions makeSfmPairPlannerOptions(const FeaturePairPlannerOptions &options, bool exhaustive)
{
    SfmPairPlannerOptions pairOptions;
    pairOptions.autoRestrictKnownCameraPairs = !exhaustive;
    pairOptions.knownCameraAllPairsMaxImages = std::max(2, options.exhaustiveMaxImages);
    pairOptions.knownCameraPairWindow = std::max(1, options.sequentialWindow);
    pairOptions.knownCameraSpatialNeighborCount = std::max(0, options.spatialNeighborCount);
    pairOptions.knownCameraCenters = options.knownCameraCenters;
    pairOptions.knownCameraOverlapPairs = options.knownCameraOverlapPairs;
    pairOptions.knownCameraOverlapMaxExpansion = options.knownCameraOverlapMaxExpansion;
    return pairOptions;
}

} // namespace

FeaturePairPlan planFeatureMatchPairPlan(const QStringList &imageBaseNames,
                                         const FeaturePairPlannerOptions &options)
{
    FeaturePairPlan result;
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
        return result;
    }

    const bool exhaustive = count <= std::max(2, options.exhaustiveMaxImages);
    const int window = exhaustive ? (count - 1) : std::max(1, options.sequentialWindow);
    if (exhaustive &&
        options.knownCameraOverlapPairs.empty() &&
        options.knownCameraCenters.empty())
    {
        result.pairs = makeExhaustiveOrWindowPairs(names, window);
        return result;
    }

    QStringList plannerImages;
    plannerImages.reserve(names.size());
    QHash<QString, QString> nameByCanonicalPath;
    for (const QString &name : names)
    {
        plannerImages.append(name);
        nameByCanonicalPath.insert(canonicalSfmPath(name), name);
    }

    result.corePlan = planSfmMatchPairs(plannerImages,
                                        QStringList(),
                                        makeSfmPairPlannerOptions(options, exhaustive));
    if (!result.corePlan.restrictPairs)
    {
        result.pairs = makeExhaustiveOrWindowPairs(names, window);
        return result;
    }

    result.corePairKeys = result.corePlan.allowedPairKeys;
    result.pairs = convertCanonicalPairsToGuiPairs(result.corePairKeys, nameByCanonicalPath);
    if (result.pairs.isEmpty())
    {
        result.pairs = makeExhaustiveOrWindowPairs(names, window);
        result.corePairKeys.clear();
    }

    return result;
}

FeaturePairPlan planFeatureMatchPairPathPlan(const QStringList &imagePaths,
                                             const FeaturePairPlannerOptions &options)
{
    FeaturePairPlan result;
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
        return result;
    }

    const bool exhaustive = count <= std::max(2, options.exhaustiveMaxImages);
    const int window = exhaustive ? (count - 1) : std::max(1, options.sequentialWindow);
    if (exhaustive &&
        options.knownCameraOverlapPairs.empty() &&
        options.knownCameraCenters.empty())
    {
        result.pairs = makeExhaustiveOrWindowPathPairs(paths, window);
        return result;
    }

    QHash<QString, QString> pathByCanonicalPath;
    for (const QString &path : paths)
    {
        pathByCanonicalPath.insert(canonicalSfmPath(path), path);
    }

    result.corePlan = planSfmMatchPairs(paths,
                                        QStringList(),
                                        makeSfmPairPlannerOptions(options, exhaustive));
    if (!result.corePlan.restrictPairs)
    {
        result.pairs = makeExhaustiveOrWindowPathPairs(paths, window);
        return result;
    }

    result.corePairKeys = result.corePlan.allowedPairKeys;
    result.pairs = convertCanonicalPairsToPipelinePairs(result.corePairKeys, pathByCanonicalPath);
    if (result.pairs.isEmpty())
    {
        result.pairs = makeExhaustiveOrWindowPathPairs(paths, window);
        result.corePairKeys.clear();
    }

    return result;
}

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options)
{
    return planFeatureMatchPairPlan(imageBaseNames, options).pairs;
}

QStringList planFeatureMatchPairPaths(const QStringList &imagePaths,
                                      const FeaturePairPlannerOptions &options)
{
    return planFeatureMatchPairPathPlan(imagePaths, options).pairs;
}

} // namespace xjw::gui
