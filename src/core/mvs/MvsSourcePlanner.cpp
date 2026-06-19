#include "MvsSourcePlanner.h"

#include <QJsonArray>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace xjw
{
namespace mvs
{

namespace
{

MvsSourcePlanEntry makeEntry(const MvsSourceCandidate &candidate,
                             const MvsSourcePlannerOptions &options)
{
    MvsSourcePlanEntry entry;
    entry.viewIndex = candidate.viewIndex;
    entry.sharedTracks = std::max(0, candidate.sharedTracks);
    entry.geometricInliers = std::max(0, candidate.geometricInliers);
    entry.medianTriangulationAngleDeg = std::max(0.0f, candidate.medianTriangulationAngleDeg);
    entry.coverageScore = std::clamp(candidate.coverageScore, 0.0f, 1.0f);
    entry.baselineScore = std::clamp(candidate.baselineScore, 0.0f, 1.0f);
    entry.knownOverlap = candidate.knownOverlap;
    entry.sequenceDistance = candidate.sequenceDistance > 0
        ? candidate.sequenceDistance
        : std::abs(candidate.viewIndex - options.refIndex);
    return entry;
}

bool isAngleOutlier(const MvsSourcePlanEntry &entry,
                    const MvsSourcePlannerOptions &options)
{
    if (!options.rejectAngleOutliers)
    {
        return false;
    }
    if (entry.medianTriangulationAngleDeg <= 0.0f)
    {
        return entry.sharedTracks > 0 || entry.geometricInliers > 0;
    }
    return entry.medianTriangulationAngleDeg < options.minTriangulationAngleDeg
        || entry.medianTriangulationAngleDeg > options.maxTriangulationAngleDeg;
}

float angleWeight(const MvsSourcePlanEntry &entry,
                  const MvsSourcePlannerOptions &options)
{
    const float angle = entry.medianTriangulationAngleDeg;
    if (angle <= 0.0f)
    {
        return entry.knownOverlap ? 0.5f : 0.25f;
    }
    if (angle < options.minTriangulationAngleDeg)
    {
        return 0.25f;
    }
    if (angle > options.maxTriangulationAngleDeg)
    {
        return 0.50f;
    }
    return 1.0f;
}

float computeScore(const MvsSourcePlanEntry &entry,
                   const MvsSourcePlannerOptions &options)
{
    const int evidence = std::max(entry.sharedTracks, entry.geometricInliers);
    const float baseEvidence = evidence > 0 ? static_cast<float>(evidence)
                                            : (entry.knownOverlap ? 1.0f : 0.0f);
    if (baseEvidence <= 0.0f)
    {
        return -std::numeric_limits<float>::infinity();
    }

    const float geomBoost = static_cast<float>(entry.geometricInliers) * 0.25f;
    const float coverageBoost = baseEvidence * entry.coverageScore * 0.50f;
    const float baselineBoost = baseEvidence * entry.baselineScore * 0.25f;
    const float knownOverlapBoost = entry.knownOverlap ? 0.10f : 0.0f;
    const float sequencePenalty = 0.001f * static_cast<float>(std::max(0, entry.sequenceDistance));

    return baseEvidence * angleWeight(entry, options)
        + geomBoost
        + coverageBoost
        + baselineBoost
        + knownOverlapBoost
        - sequencePenalty;
}

bool sourceEntryLess(const MvsSourcePlanEntry &lhs,
                     const MvsSourcePlanEntry &rhs)
{
    if (lhs.score != rhs.score)
    {
        return lhs.score > rhs.score;
    }
    if (lhs.geometricInliers != rhs.geometricInliers)
    {
        return lhs.geometricInliers > rhs.geometricInliers;
    }
    if (lhs.sharedTracks != rhs.sharedTracks)
    {
        return lhs.sharedTracks > rhs.sharedTracks;
    }
    if (lhs.sequenceDistance != rhs.sequenceDistance)
    {
        return lhs.sequenceDistance < rhs.sequenceDistance;
    }
    return lhs.viewIndex < rhs.viewIndex;
}

std::vector<MvsSourcePlanEntry> sequenceFallbackEntries(const MvsSourcePlannerOptions &options)
{
    std::vector<MvsSourcePlanEntry> entries;
    if (!options.allowSequenceFallback || options.viewCount <= 1
        || options.refIndex < 0 || options.refIndex >= options.viewCount
        || options.maxSources <= 0)
    {
        return entries;
    }

    entries.reserve(static_cast<size_t>(options.maxSources));
    for (int delta = 1;
         delta <= options.viewCount && static_cast<int>(entries.size()) < options.maxSources;
         ++delta)
    {
        for (int sign : {-1, 1})
        {
            const int viewIndex = options.refIndex + sign * delta;
            if (viewIndex < 0 || viewIndex >= options.viewCount)
            {
                continue;
            }

            MvsSourcePlanEntry entry;
            entry.viewIndex = viewIndex;
            entry.sequenceDistance = delta;
            entry.sequenceFallback = true;
            entry.score = -static_cast<float>(delta);
            entries.push_back(entry);
            if (static_cast<int>(entries.size()) >= options.maxSources)
            {
                break;
            }
        }
    }
    return entries;
}

} // namespace

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options)
{
    MvsSourcePlan plan;
    if (options.maxSources <= 0)
    {
        return plan;
    }

    std::unordered_map<int, MvsSourcePlanEntry> bestByView;
    bestByView.reserve(candidates.size());

    for (const MvsSourceCandidate &candidate : candidates)
    {
        MvsSourcePlanEntry entry = makeEntry(candidate, options);
        if (entry.viewIndex < 0 || entry.viewIndex >= options.viewCount)
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::InvalidIndex});
            continue;
        }
        if (entry.viewIndex == options.refIndex)
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::Self});
            continue;
        }
        if (isAngleOutlier(entry, options))
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::TriangulationAngle});
            continue;
        }

        entry.score = computeScore(entry, options);
        if (!std::isfinite(entry.score) || entry.score <= 0.0f)
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::NoEvidence});
            continue;
        }

        auto it = bestByView.find(entry.viewIndex);
        if (it == bestByView.end())
        {
            bestByView.emplace(entry.viewIndex, entry);
            continue;
        }

        if (sourceEntryLess(entry, it->second))
        {
            plan.rejected.push_back({it->second, MvsSourceRejectReason::Duplicate});
            it->second = entry;
        }
        else
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::Duplicate});
        }
    }

    plan.selected.reserve(bestByView.size());
    for (const auto &item : bestByView)
    {
        plan.selected.push_back(item.second);
    }
    std::sort(plan.selected.begin(), plan.selected.end(), sourceEntryLess);
    if (static_cast<int>(plan.selected.size()) > options.maxSources)
    {
        plan.selected.resize(static_cast<size_t>(options.maxSources));
    }

    if (plan.selected.empty())
    {
        plan.selected = sequenceFallbackEntries(options);
        plan.usedSequenceFallback = !plan.selected.empty();
    }

    return plan;
}

QJsonObject mvsSourcePlanEntryToJson(const MvsSourcePlanEntry &entry)
{
    QJsonObject object;
    object.insert(QStringLiteral("view_index"), entry.viewIndex);
    object.insert(QStringLiteral("shared_tracks"), entry.sharedTracks);
    object.insert(QStringLiteral("geometric_inliers"), entry.geometricInliers);
    object.insert(QStringLiteral("median_angle_deg"), entry.medianTriangulationAngleDeg);
    object.insert(QStringLiteral("coverage_score"), entry.coverageScore);
    object.insert(QStringLiteral("baseline_score"), entry.baselineScore);
    object.insert(QStringLiteral("sequence_distance"), entry.sequenceDistance);
    object.insert(QStringLiteral("known_overlap"), entry.knownOverlap);
    object.insert(QStringLiteral("sequence_fallback"), entry.sequenceFallback);
    object.insert(QStringLiteral("score"), entry.score);
    return object;
}

} // namespace mvs
} // namespace xjw
