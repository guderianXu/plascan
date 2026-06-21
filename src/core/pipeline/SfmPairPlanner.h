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
    std::vector<std::array<double, 3>> knownCameraViewingDirections;
    std::vector<std::array<int, 2>> knownCameraOverlapPairs;
    double knownCameraOverlapMaxExpansion = 2.0;
};

struct SfmPairCandidate
{
    int indexA = -1;
    int indexB = -1;
    QString pairKey;
    QStringList sourceTypes;
    double priorityScore = 0.0;
    double overlapScore = 0.0;
    double sequenceScore = 0.0;
    double spatialScore = 0.0;
    double baselineScore = 0.0;
    double orientationScore = 0.0;
    int sequenceDistance = 0;
    double centerDistance = -1.0;
    double orientationAngleDeg = -1.0;
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
    std::vector<SfmPairCandidate> pairCandidates;
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

inline bool hasCompleteSfmImageList(const QStringList &images)
{
    if (images.isEmpty())
    {
        return false;
    }

    for (const QString &image : images)
    {
        if (canonicalSfmPath(image).isEmpty())
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

inline double vectorNorm3(const std::array<double, 3> &v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline bool hasCompleteKnownCameraViewingDirections(
    int imageCount,
    const std::vector<std::array<double, 3>> &directions)
{
    if (imageCount <= 0 || directions.size() != static_cast<std::size_t>(imageCount))
    {
        return false;
    }

    for (const auto &direction : directions)
    {
        if (!std::isfinite(direction[0]) ||
            !std::isfinite(direction[1]) ||
            !std::isfinite(direction[2]) ||
            vectorNorm3(direction) <= 1e-12)
        {
            return false;
        }
    }

    return true;
}

struct SfmPairGeometryScores
{
    double baselineScore = 0.0;
    double orientationScore = 0.0;
    double centerDistance = -1.0;
    double orientationAngleDeg = -1.0;
};

inline SfmPairGeometryScores computeSfmPairGeometryScores(const SfmPairPlannerOptions &options,
                                                          int imageCount,
                                                          int indexA,
                                                          int indexB)
{
    SfmPairGeometryScores scores;
    if (indexA < 0 || indexB < 0 || indexA >= imageCount || indexB >= imageCount || indexA == indexB)
    {
        return scores;
    }

    if (hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters))
    {
        scores.centerDistance = std::sqrt(std::max(0.0,
            squaredCenterDistance(options.knownCameraCenters[static_cast<std::size_t>(indexA)],
                                  options.knownCameraCenters[static_cast<std::size_t>(indexB)])));
        scores.baselineScore = scores.centerDistance > 0.0
            ? std::clamp(scores.centerDistance / (scores.centerDistance + 1.0), 0.0, 1.0)
            : 0.0;
    }

    if (hasCompleteKnownCameraViewingDirections(imageCount, options.knownCameraViewingDirections))
    {
        const auto &dirA = options.knownCameraViewingDirections[static_cast<std::size_t>(indexA)];
        const auto &dirB = options.knownCameraViewingDirections[static_cast<std::size_t>(indexB)];
        const double normA = vectorNorm3(dirA);
        const double normB = vectorNorm3(dirB);
        if (normA > 1e-12 && normB > 1e-12)
        {
            const double dot = std::clamp((dirA[0] * dirB[0] + dirA[1] * dirB[1] + dirA[2] * dirB[2]) /
                                          (normA * normB),
                                          -1.0,
                                          1.0);
            scores.orientationScore = std::max(0.0, dot);
            scores.orientationAngleDeg = std::acos(dot) * 180.0 / M_PI;
        }
    }

    return scores;
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

inline SfmPairCandidate *findSfmPairCandidate(std::vector<SfmPairCandidate> *candidates,
                                              const QString &pairKey)
{
    if (!candidates)
    {
        return nullptr;
    }

    for (SfmPairCandidate &candidate : *candidates)
    {
        if (candidate.pairKey == pairKey)
        {
            return &candidate;
        }
    }
    return nullptr;
}

inline void appendSfmPairSource(SfmPairCandidate *candidate, const QString &sourceType)
{
    if (!candidate || sourceType.isEmpty() || candidate->sourceTypes.contains(sourceType))
    {
        return;
    }
    candidate->sourceTypes.append(sourceType);
}

inline void addOrUpdateSfmPairCandidate(SfmPairPlan *plan,
                                        const QStringList &images,
                                        int indexA,
                                        int indexB,
                                        const QString &sourceType,
                                        double priorityScore,
                                        double overlapScore = 0.0,
                                        double sequenceScore = 0.0,
                                        double spatialScore = 0.0,
                                        int sequenceDistance = 0,
                                        double centerDistance = -1.0,
                                        double baselineScore = 0.0,
                                        double orientationScore = 0.0,
                                        double orientationAngleDeg = -1.0)
{
    if (!plan || indexA == indexB || indexA < 0 || indexB < 0 ||
        indexA >= images.size() || indexB >= images.size())
    {
        return;
    }

    const QString pairKey = canonicalSfmPairKey(images.at(indexA), images.at(indexB));
    if (pairKey.isEmpty())
    {
        return;
    }

    SfmPairCandidate *candidate = findSfmPairCandidate(&plan->pairCandidates, pairKey);
    if (!candidate)
    {
        SfmPairCandidate created;
        created.indexA = std::min(indexA, indexB);
        created.indexB = std::max(indexA, indexB);
        created.pairKey = pairKey;
        plan->pairCandidates.push_back(created);
        candidate = &plan->pairCandidates.back();
    }

    appendSfmPairSource(candidate, sourceType);
    candidate->priorityScore += std::max(0.0, priorityScore);
    candidate->overlapScore = std::max(candidate->overlapScore, overlapScore);
    candidate->sequenceScore = std::max(candidate->sequenceScore, sequenceScore);
    candidate->spatialScore = std::max(candidate->spatialScore, spatialScore);
    candidate->baselineScore = std::max(candidate->baselineScore, baselineScore);
    candidate->orientationScore = std::max(candidate->orientationScore, orientationScore);
    if (sequenceDistance > 0 &&
        (candidate->sequenceDistance == 0 || sequenceDistance < candidate->sequenceDistance))
    {
        candidate->sequenceDistance = sequenceDistance;
    }
    if (centerDistance >= 0.0 &&
        (candidate->centerDistance < 0.0 || centerDistance < candidate->centerDistance))
    {
        candidate->centerDistance = centerDistance;
    }
    if (orientationAngleDeg >= 0.0 &&
        (candidate->orientationAngleDeg < 0.0 || orientationAngleDeg < candidate->orientationAngleDeg))
    {
        candidate->orientationAngleDeg = orientationAngleDeg;
    }
}

inline void addManualSfmPairCandidate(SfmPairPlan *plan, const QString &pairKey)
{
    if (!plan || pairKey.trimmed().isEmpty())
    {
        return;
    }

    SfmPairCandidate candidate;
    candidate.pairKey = pairKey.trimmed();
    candidate.priorityScore = 10.0;
    candidate.sourceTypes.append(QStringLiteral("manual_restricted"));
    plan->pairCandidates.push_back(candidate);
}

inline void finalizeSfmPairPlan(SfmPairPlan *plan)
{
    if (!plan)
    {
        return;
    }

    std::sort(plan->pairCandidates.begin(), plan->pairCandidates.end(),
              [](const SfmPairCandidate &lhs, const SfmPairCandidate &rhs)
    {
        if (lhs.priorityScore != rhs.priorityScore)
        {
            return lhs.priorityScore > rhs.priorityScore;
        }
        return lhs.pairKey < rhs.pairKey;
    });

    plan->allowedPairKeys.clear();
    for (const SfmPairCandidate &candidate : plan->pairCandidates)
    {
        if (!candidate.pairKey.isEmpty())
        {
            plan->allowedPairKeys.append(candidate.pairKey);
        }
    }
}

inline int slidingWindowPairCount(int imageCount, int window)
{
    if (imageCount <= 1 || window <= 0)
    {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < imageCount; ++i)
    {
        count += std::min(window, imageCount - i - 1);
    }
    return count;
}

inline bool acceptsKnownCameraOverlapPairs(int imageCount,
                                           int window,
                                           int spatialNeighborCount,
                                           int overlapPairCount,
                                           double maxExpansion)
{
    if (overlapPairCount <= 0)
    {
        return false;
    }

    const int sequenceBudget = slidingWindowPairCount(imageCount, window);
    const int spatialBudget = std::max(0, imageCount * std::max(0, spatialNeighborCount));
    const int boundedBudget = std::max(1, sequenceBudget + spatialBudget);
    const double expansion = std::max(1.0, maxExpansion);
    return static_cast<double>(overlapPairCount) <= static_cast<double>(boundedBudget) * expansion;
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
        for (const QString &pairKey : plan.allowedPairKeys)
        {
            addManualSfmPairCandidate(&plan, pairKey);
        }
        return plan;
    }

    const bool hasRestrictionInputs =
        hasCompleteCameraPathList(images, cameraPaths) ||
        hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters) ||
        !options.knownCameraOverlapPairs.empty();

    if (!options.autoRestrictKnownCameraPairs ||
        imageCount <= std::max(0, options.knownCameraAllPairsMaxImages) ||
        !hasCompleteSfmImageList(images) ||
        !hasRestrictionInputs)
    {
        return plan;
    }

    const int window = std::max(1, options.knownCameraPairWindow);
    const int spatialNeighborCount = std::max(0, options.knownCameraSpatialNeighborCount);
    plan.restrictPairs = true;
    plan.autoRestricted = true;
    plan.knownCameraPairWindow = window;
    plan.knownCameraSpatialNeighborCount = spatialNeighborCount;

    for (const auto &pair : options.knownCameraOverlapPairs)
    {
        const int indexA = pair[0];
        const int indexB = pair[1];
        if (indexA < 0 || indexA >= imageCount || indexB < 0 || indexB >= imageCount)
        {
            continue;
        }
        const SfmPairGeometryScores geometry =
            computeSfmPairGeometryScores(options, imageCount, indexA, indexB);
        addOrUpdateSfmPairCandidate(&plan,
                                    images,
                                    indexA,
                                    indexB,
                                    QStringLiteral("known_camera_overlap"),
                                    100.0 + 15.0 * geometry.orientationScore + 5.0 * geometry.baselineScore,
                                    1.0,
                                    0.0,
                                    0.0,
                                    0,
                                    geometry.centerDistance,
                                    geometry.baselineScore,
                                    geometry.orientationScore,
                                    geometry.orientationAngleDeg);
    }

    if (!plan.pairCandidates.empty())
    {
        plan.knownCameraOverlapPairCount = static_cast<int>(plan.pairCandidates.size());
        if (acceptsKnownCameraOverlapPairs(imageCount,
                                           window,
                                           spatialNeighborCount,
                                           plan.knownCameraOverlapPairCount,
                                           options.knownCameraOverlapMaxExpansion))
        {
            plan.usedCameraOverlapPairs = true;
            finalizeSfmPairPlan(&plan);
            return plan;
        }

        plan.pairCandidates.clear();
        plan.knownCameraOverlapPairCount = 0;
    }

    for (int i = 0; i < imageCount; ++i)
    {
        const int last = std::min(imageCount - 1, i + window);
        for (int j = i + 1; j <= last; ++j)
        {
            const int sequenceDistance = j - i;
            const double sequenceScore =
                static_cast<double>(window - sequenceDistance + 1) / static_cast<double>(window);
            const SfmPairGeometryScores geometry =
                computeSfmPairGeometryScores(options, imageCount, i, j);
            addOrUpdateSfmPairCandidate(&plan,
                                        images,
                                        i,
                                        j,
                                        QStringLiteral("sequence_window"),
                                        40.0 + 10.0 * sequenceScore +
                                            15.0 * geometry.orientationScore +
                                            5.0 * geometry.baselineScore,
                                        0.0,
                                        sequenceScore,
                                        0.0,
                                        sequenceDistance,
                                        geometry.centerDistance,
                                        geometry.baselineScore,
                                        geometry.orientationScore,
                                        geometry.orientationAngleDeg);
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
                const auto &neighbor = neighbors[static_cast<std::size_t>(n)];
                const double spatialScore =
                    static_cast<double>(keep - n) / static_cast<double>(std::max(1, keep));
                const SfmPairGeometryScores geometry =
                    computeSfmPairGeometryScores(options, imageCount, i, neighbor.second);
                addOrUpdateSfmPairCandidate(&plan,
                                            images,
                                            i,
                                            neighbor.second,
                                            QStringLiteral("known_camera_spatial_neighbors"),
                                            60.0 + 10.0 * spatialScore +
                                                25.0 * geometry.orientationScore +
                                                5.0 * geometry.baselineScore,
                                            0.0,
                                            0.0,
                                            spatialScore,
                                            0,
                                            geometry.centerDistance,
                                            geometry.baselineScore,
                                            geometry.orientationScore,
                                            geometry.orientationAngleDeg);
            }
        }
    }

    finalizeSfmPairPlan(&plan);
    return plan;
}

} // namespace gui
} // namespace xjw
