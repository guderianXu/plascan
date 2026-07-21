#include "MvsSourcePlanner.h"

#include <QJsonArray>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

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
    entry.verifiedPairGeometry = candidate.verifiedPairGeometry;
    entry.tier = candidate.verifiedPairGeometry
        ? MvsSourceTier::VerifiedPair
        : MvsSourceTier::TrackGeometryBackfill;
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
    if (options.softMaxTriangulationAngleDeg > options.minTriangulationAngleDeg &&
        angle > options.softMaxTriangulationAngleDeg)
    {
        const float range = std::max(
            1.0f,
            options.maxTriangulationAngleDeg - options.softMaxTriangulationAngleDeg);
        const float t = std::clamp((angle - options.softMaxTriangulationAngleDeg) / range, 0.0f, 1.0f);
        return 0.75f - 0.25f * t;
    }
    if (options.preferredTriangulationAngleDeg > options.minTriangulationAngleDeg)
    {
        const float normalizedDistance = std::fabs(angle - options.preferredTriangulationAngleDeg) /
            std::max(1.0f, options.preferredTriangulationAngleDeg);
        return std::clamp(1.15f - 0.25f * normalizedDistance, 0.85f, 1.15f);
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

float computeSourceQualityScore(const MvsSourcePlanEntry &entry,
                                const MvsSourcePlannerOptions &options)
{
    const int evidence = std::max(entry.sharedTracks, entry.geometricInliers);
    float evidenceScore = evidence > 0
        ? 1.0f - std::exp(-static_cast<float>(evidence) / 80.0f)
        : (entry.knownOverlap ? 0.20f : 0.0f);
    const float angleScore = std::clamp(angleWeight(entry, options), 0.0f, 1.0f);
    float baselineScore = entry.baselineScore;
    if (entry.medianTriangulationAngleDeg > options.softMaxTriangulationAngleDeg)
    {
        evidenceScore *= angleScore;
        baselineScore *= angleScore;
    }
    const float overlapScore = entry.knownOverlap ? 1.0f : 0.0f;
    return std::clamp(0.35f * evidenceScore +
                          0.20f * entry.coverageScore +
                          0.10f * baselineScore +
                          0.25f * angleScore +
                          0.10f * overlapScore,
                      0.0f,
                      1.0f);
}

bool failsQualityGate(const MvsSourcePlanEntry &entry,
                      const MvsSourcePlannerOptions &options)
{
    if (!options.allowWeakKnownOverlap
        && entry.knownOverlap
        && entry.sharedTracks <= 0
        && entry.geometricInliers <= 0)
    {
        return true;
    }
    if (options.minSharedTracks > 0 && entry.sharedTracks < options.minSharedTracks)
    {
        return true;
    }
    if (options.minGeometricInliers > 0 && entry.geometricInliers < options.minGeometricInliers)
    {
        return true;
    }
    if (options.requireVerifiedPairGeometry && !entry.verifiedPairGeometry)
    {
        return true;
    }
    if (options.minSourceQualityScore > 0.0f && entry.sourceQualityScore < options.minSourceQualityScore)
    {
        return true;
    }
    return false;
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
            entry.tier = MvsSourceTier::SequenceFallback;
            entry.score = -static_cast<float>(delta);
            entry.sourceQualityScore = std::clamp(0.10f / static_cast<float>(delta), 0.0f, 1.0f);
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
    plan.requestedSourceCount = std::max(0, options.maxSources);
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
        entry.sourceQualityScore = computeSourceQualityScore(entry, options);
        if (!std::isfinite(entry.score) || entry.score <= 0.0f)
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::NoEvidence});
            continue;
        }
        if (failsQualityGate(entry, options))
        {
            plan.rejected.push_back({entry, MvsSourceRejectReason::LowQuality});
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

    plan.sourceViewShortfall = std::max(
        0,
        plan.requestedSourceCount - static_cast<int>(plan.selected.size()));

    return plan;
}

MvsSourcePlan planMvsSourceViewsVerifiedFirst(
    const std::vector<MvsSourceCandidate> &candidates,
    const MvsSourcePlannerOptions &options)
{
    MvsSourcePlannerOptions verifiedOptions = options;
    verifiedOptions.requireVerifiedPairGeometry = true;
    verifiedOptions.allowSequenceFallback = false;
    MvsSourcePlan result = planMvsSourceViews(candidates, verifiedOptions);
    result.requestedSourceCount = std::max(0, options.maxSources);
    for (MvsSourcePlanEntry &entry : result.selected)
    {
        entry.tier = MvsSourceTier::VerifiedPair;
    }

    if (static_cast<int>(result.selected.size()) >= result.requestedSourceCount)
    {
        result.sourceViewShortfall = 0;
        return result;
    }

    std::unordered_set<int> selectedViews;
    selectedViews.reserve(result.selected.size());
    for (const MvsSourcePlanEntry &entry : result.selected)
    {
        selectedViews.insert(entry.viewIndex);
    }

    std::vector<MvsSourceCandidate> remaining;
    remaining.reserve(candidates.size());
    for (const MvsSourceCandidate &candidate : candidates)
    {
        if (selectedViews.find(candidate.viewIndex) == selectedViews.end())
        {
            remaining.push_back(candidate);
        }
    }

    MvsSourcePlannerOptions backfillOptions = options;
    backfillOptions.requireVerifiedPairGeometry = false;
    backfillOptions.allowSequenceFallback = false;
    backfillOptions.allowWeakKnownOverlap = false;
    backfillOptions.rejectAngleOutliers = true;
    backfillOptions.minTriangulationAngleDeg = std::max(
        0.2f, options.minTriangulationAngleDeg);
    backfillOptions.maxTriangulationAngleDeg = options.maxTriangulationAngleDeg;
    backfillOptions.minSharedTracks = 20;
    // A backfill view may be absent from the verified-pair tier because its
    // pair verification did not meet the production threshold.  It still
    // needs direct geometric evidence of its own: projected sparse overlap
    // alone is not sufficient and previously admitted zero-inlier views into
    // PatchMatch for weak Temple frames.
    backfillOptions.minGeometricInliers = 1;
    backfillOptions.minSourceQualityScore = 0.35f;
    backfillOptions.maxSources = result.requestedSourceCount -
        static_cast<int>(result.selected.size());
    MvsSourcePlan backfill = planMvsSourceViews(remaining, backfillOptions);
    for (MvsSourcePlanEntry &entry : backfill.selected)
    {
        entry.tier = MvsSourceTier::TrackGeometryBackfill;
        result.selected.push_back(entry);
    }
    result.rejected = std::move(backfill.rejected);
    result.usedSequenceFallback = false;
    result.sourceViewShortfall = std::max(
        0,
        result.requestedSourceCount - static_cast<int>(result.selected.size()));
    return result;
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
    object.insert(QStringLiteral("verified_pair_geometry"), entry.verifiedPairGeometry);
    object.insert(QStringLiteral("sequence_fallback"), entry.sequenceFallback);
    QString sourceTier = QStringLiteral("verified_pair");
    if (entry.tier == MvsSourceTier::TrackGeometryBackfill)
    {
        sourceTier = QStringLiteral("track_geometry_backfill");
    }
    else if (entry.tier == MvsSourceTier::SequenceFallback)
    {
        sourceTier = QStringLiteral("sequence_fallback");
    }
    object.insert(QStringLiteral("source_tier"), sourceTier);
    object.insert(QStringLiteral("score"), entry.score);
    object.insert(QStringLiteral("source_quality_score"), entry.sourceQualityScore);
    return object;
}

} // namespace mvs
} // namespace xjw
