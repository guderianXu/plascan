#include "MvsSourcePlanner.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace xjw
{
namespace mvs
{

namespace
{

QString normalizedSourceImagePath(const std::string &path)
{
    const QString rawPath = QString::fromStdString(path).trimmed();
    if (rawPath.isEmpty())
    {
        return QString();
    }

    const QFileInfo info(rawPath);
    QString absolutePath = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
    if (absolutePath.isEmpty())
    {
        absolutePath = rawPath;
    }
    return QDir::cleanPath(absolutePath)
        .replace(QLatin1Char('\\'), QLatin1Char('/'))
        .toCaseFolded();
}

QString sourcePairKey(const std::string &imageA, const std::string &imageB)
{
    const QString keyA = normalizedSourceImagePath(imageA);
    const QString keyB = normalizedSourceImagePath(imageB);
    if (keyA.isEmpty() || keyB.isEmpty() || keyA == keyB)
    {
        return QString();
    }
    return keyA < keyB
        ? keyA + QLatin1Char('\n') + keyB
        : keyB + QLatin1Char('\n') + keyA;
}

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
    entry.verificationStatus = candidate.verifiedPairGeometry
        ? MvsSourceVerificationStatus::Verified
        : candidate.verificationStatus;
    entry.pairTotalMatches = std::max(0, candidate.pairTotalMatches);
    entry.pairInlierRatio = entry.pairTotalMatches > 0
        ? std::clamp(
              static_cast<float>(entry.geometricInliers) /
                  static_cast<float>(entry.pairTotalMatches),
              0.0f,
              1.0f)
        : 0.0f;
    entry.pairCoverageScore =
        std::clamp(candidate.pairCoverageScore, 0.0f, 1.0f);
    entry.verificationReason = candidate.verificationReason;
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

bool sourceSoftRankingEnabled(const MvsSourcePlannerOptions &options)
{
    return std::isfinite(options.sourceAngleSoftRankingStrength) &&
        options.sourceAngleSoftRankingStrength > 0.0f;
}

float normalizedSoftAnglePenalty(
    const MvsSourcePlanEntry &entry,
    const MvsSourcePlannerOptions &options)
{
    const float angle = entry.medianTriangulationAngleDeg;
    const float soft_maximum = options.softMaxTriangulationAngleDeg;
    const float maximum = options.maxTriangulationAngleDeg;
    if (!std::isfinite(angle) || !std::isfinite(soft_maximum) ||
        !std::isfinite(maximum) || angle <= soft_maximum ||
        maximum <= soft_maximum)
    {
        return 0.0f;
    }

    return std::clamp(
        (angle - soft_maximum) / (maximum - soft_maximum),
        0.0f,
        1.0f);
}

float adjustedSourceScore(const MvsSourcePlanEntry &entry,
                          const MvsSourcePlannerOptions &options,
                          float normalized_penalty)
{
    if (!sourceSoftRankingEnabled(options) || normalized_penalty <= 0.0f)
    {
        return entry.score;
    }

    return entry.score * std::exp(
        -options.sourceAngleSoftRankingStrength * normalized_penalty);
}

bool adjustedSourceEntryLess(const MvsSourcePlanEntry &lhs,
                             const MvsSourcePlanEntry &rhs)
{
    if (lhs.adjustedScore != rhs.adjustedScore)
    {
        return lhs.adjustedScore > rhs.adjustedScore;
    }
    return sourceEntryLess(lhs, rhs);
}

bool sameSourceSelection(const std::vector<MvsSourcePlanEntry> &lhs,
                         const std::vector<MvsSourcePlanEntry> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index].viewIndex != rhs[index].viewIndex ||
            lhs[index].tier != rhs[index].tier)
        {
            return false;
        }
    }
    return true;
}

bool sameSelectedViewSet(const std::vector<MvsSourcePlanEntry> &lhs,
                         const std::vector<MvsSourcePlanEntry> &rhs)
{
    std::vector<int> lhs_views;
    std::vector<int> rhs_views;
    lhs_views.reserve(lhs.size());
    rhs_views.reserve(rhs.size());
    for (const MvsSourcePlanEntry &entry : lhs)
    {
        lhs_views.push_back(entry.viewIndex);
    }
    for (const MvsSourcePlanEntry &entry : rhs)
    {
        rhs_views.push_back(entry.viewIndex);
    }
    std::sort(lhs_views.begin(), lhs_views.end());
    std::sort(rhs_views.begin(), rhs_views.end());
    return lhs_views == rhs_views;
}

bool sameSelectedViewOrder(const std::vector<MvsSourcePlanEntry> &lhs,
                           const std::vector<MvsSourcePlanEntry> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index].viewIndex != rhs[index].viewIndex)
        {
            return false;
        }
    }
    return true;
}

bool containsSourceIdentity(const std::vector<MvsSourcePlanEntry> &entries,
                            int view_index,
                            MvsSourceTier tier)
{
    return std::any_of(
        entries.cbegin(),
        entries.cend(),
        [view_index, tier](const MvsSourcePlanEntry &entry)
        {
            return entry.viewIndex == view_index && entry.tier == tier;
        });
}

void annotateSourceRanking(
    std::vector<MvsSourcePlanEntry> &legacy_order,
    std::vector<MvsSourcePlanEntry> &adjusted_order,
    const MvsSourcePlannerOptions &options)
{
    struct RankValues
    {
        float adjustedScore = 0.0f;
        float normalizedPenalty = 0.0f;
        int legacyRank = -1;
        int adjustedRank = -1;
    };

    std::unordered_map<int, RankValues> values_by_view;
    values_by_view.reserve(legacy_order.size());
    for (std::size_t index = 0; index < legacy_order.size(); ++index)
    {
        const float penalty = normalizedSoftAnglePenalty(
            legacy_order[index], options);
        values_by_view.emplace(
            legacy_order[index].viewIndex,
            RankValues{
                adjustedSourceScore(legacy_order[index], options, penalty),
                penalty,
                static_cast<int>(index),
                -1});
    }

    adjusted_order = legacy_order;
    for (MvsSourcePlanEntry &entry : adjusted_order)
    {
        const RankValues &values = values_by_view.at(entry.viewIndex);
        entry.adjustedScore = values.adjustedScore;
        entry.normalizedSoftAnglePenalty = values.normalizedPenalty;
        entry.rankingSoftMaximumDegrees =
            options.softMaxTriangulationAngleDeg;
        entry.rankingEffectiveMaximumDegrees =
            options.maxTriangulationAngleDeg;
        entry.legacyRankWithinTier = values.legacyRank;
        entry.sourceRankingAudited = true;
    }
    std::sort(
        adjusted_order.begin(),
        adjusted_order.end(),
        adjustedSourceEntryLess);
    for (std::size_t index = 0; index < adjusted_order.size(); ++index)
    {
        values_by_view.at(adjusted_order[index].viewIndex).adjustedRank =
            static_cast<int>(index);
    }

    const auto apply_values = [&values_by_view, &options](
                                  MvsSourcePlanEntry &entry)
    {
        const RankValues &values = values_by_view.at(entry.viewIndex);
        entry.adjustedScore = values.adjustedScore;
        entry.normalizedSoftAnglePenalty = values.normalizedPenalty;
        entry.rankingSoftMaximumDegrees =
            options.softMaxTriangulationAngleDeg;
        entry.rankingEffectiveMaximumDegrees =
            options.maxTriangulationAngleDeg;
        entry.legacyRankWithinTier = values.legacyRank;
        entry.adjustedRankWithinTier = values.adjustedRank;
        entry.sourceRankingAudited = true;
    };
    for (MvsSourcePlanEntry &entry : legacy_order)
    {
        apply_values(entry);
    }
    for (MvsSourcePlanEntry &entry : adjusted_order)
    {
        apply_values(entry);
    }
}

void finalizeAuditedSourceSelection(
    MvsSourcePlan &plan,
    std::vector<MvsSourcePlanEntry> legacy_order,
    const MvsSourcePlannerOptions &options)
{
    std::sort(legacy_order.begin(), legacy_order.end(), sourceEntryLess);
    std::vector<MvsSourcePlanEntry> control_order = legacy_order;
    std::vector<MvsSourcePlanEntry> unused_control_adjusted_order;
    MvsSourcePlannerOptions control_options = options;
    control_options.sourceAngleSoftRankingStrength = 0.0f;
    annotateSourceRanking(
        control_order, unused_control_adjusted_order, control_options);

    std::vector<MvsSourcePlanEntry> treatment_legacy_order = legacy_order;
    std::vector<MvsSourcePlanEntry> adjusted_order;
    annotateSourceRanking(treatment_legacy_order, adjusted_order, options);

    const std::size_t selection_count = std::min(
        control_order.size(),
        static_cast<std::size_t>(std::max(0, options.maxSources)));
    plan.controlSelected.assign(
        control_order.cbegin(), control_order.cbegin() + selection_count);

    const bool has_alternative =
        control_order.size() > static_cast<std::size_t>(options.maxSources);
    if (!sourceSoftRankingEnabled(options) || !has_alternative)
    {
        adjusted_order = treatment_legacy_order;
    }
    plan.treatmentSelected.assign(
        adjusted_order.cbegin(), adjusted_order.cbegin() + selection_count);
    plan.selectedCountInvariant =
        plan.controlSelected.size() == plan.treatmentSelected.size();

    const bool control_is_complete =
        static_cast<int>(plan.controlSelected.size()) == plan.requestedSourceCount;
    if (!plan.selectedCountInvariant)
    {
        plan.selected = plan.controlSelected;
        plan.sourceRankingDecisionReason =
            "selected_count_mismatch_fallback";
    }
    else if (sourceSoftRankingEnabled(options) && !control_is_complete)
    {
        plan.selected = plan.controlSelected;
        plan.sourceRankingDecisionReason = "control_source_shortfall";
    }
    else
    {
        plan.selected = plan.treatmentSelected;
        plan.sourceRankingApplied = sourceSoftRankingEnabled(options) &&
            !sameSourceSelection(plan.controlSelected, plan.treatmentSelected);
        if (!sourceSoftRankingEnabled(options))
        {
            plan.sourceRankingDecisionReason = "legacy_score_control";
        }
        else if (!has_alternative)
        {
            plan.sourceRankingDecisionReason = "insufficient_alternatives";
        }
        else if (plan.sourceRankingApplied)
        {
            plan.sourceRankingDecisionReason = "selection_changed";
        }
        else
        {
            plan.sourceRankingDecisionReason = "ranking_unchanged";
        }
    }

    plan.sourceRankingAudited = true;
    plan.controlRankingCandidates.reserve(control_order.size());
    for (const MvsSourcePlanEntry &entry : control_order)
    {
        plan.controlRankingCandidates.push_back({
            entry,
            containsSourceIdentity(
                plan.controlSelected, entry.viewIndex, entry.tier)});
    }
    plan.rankingCandidates.reserve(treatment_legacy_order.size());
    for (const MvsSourcePlanEntry &entry : treatment_legacy_order)
    {
        plan.rankingCandidates.push_back({
            entry,
            containsSourceIdentity(
                plan.treatmentSelected, entry.viewIndex, entry.tier)});
    }
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
            entry.verificationStatus =
                MvsSourceVerificationStatus::SequenceFallback;
            entry.verificationReason = "no_geometry_candidate";
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

bool isSafeFailedPairGeometryBackfill(
    const MvsSourcePlanEntry &entry,
    const MvsSourcePlannerOptions &options)
{
    if (!options.allowFailedPairBackfill
        || entry.verificationStatus != MvsSourceVerificationStatus::Failed
        || entry.verifiedPairGeometry
        || entry.pairTotalMatches <= 0
        || entry.geometricInliers <= 0
        || entry.geometricInliers > entry.pairTotalMatches)
    {
        return false;
    }

    const int required_inliers = std::max(
        options.failedPairBackfillMinimumInliers,
        options.minGeometricInliers);
    const int required_matches = std::max(
        options.failedPairBackfillMinimumMatches,
        required_inliers);
    const int required_shared_tracks = std::max(
        options.failedPairBackfillMinimumSharedTracks,
        options.minSharedTracks);
    if (entry.geometricInliers < required_inliers
        || entry.pairTotalMatches < required_matches
        || entry.sharedTracks < required_shared_tracks
        || entry.pairCoverageScore <
            options.failedPairBackfillMinimumCoverage)
    {
        return false;
    }

    // A raw ratio is over-confident for the small failed-pair samples that
    // occur in short orbital sequences.  The 90% Wilson lower bound keeps a
    // pair only when its direct geometric support remains credible after
    // accounting for sample size.
    constexpr double kWilson90Z = 1.6448536269514722;
    const double sample_count =
        static_cast<double>(entry.pairTotalMatches);
    const double success_ratio =
        static_cast<double>(entry.geometricInliers) / sample_count;
    const double z_squared = kWilson90Z * kWilson90Z;
    const double denominator = 1.0 + z_squared / sample_count;
    const double center = success_ratio + z_squared / (2.0 * sample_count);
    const double radius = kWilson90Z * std::sqrt(
        success_ratio * (1.0 - success_ratio) / sample_count
        + z_squared / (4.0 * sample_count * sample_count));
    const double wilson_lower_bound = (center - radius) / denominator;
    return wilson_lower_bound >=
        static_cast<double>(
            options.failedPairBackfillMinimumWilsonLowerBound);
}

MvsSourcePlan planMvsSourceViewsImpl(
    const std::vector<MvsSourceCandidate> &candidates,
    const MvsSourcePlannerOptions &options,
    bool allowFailedPairGeometryBackfill)
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
        if (entry.verificationStatus == MvsSourceVerificationStatus::Failed)
        {
            if (!allowFailedPairGeometryBackfill
                || !isSafeFailedPairGeometryBackfill(entry, options))
            {
                plan.rejected.push_back({entry, MvsSourceRejectReason::LowQuality});
                continue;
            }
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

    const bool audit_source_ranking = options.auditSourceRanking ||
        sourceSoftRankingEnabled(options);
    if (audit_source_ranking)
    {
        std::vector<MvsSourcePlanEntry> qualified_entries;
        qualified_entries.reserve(bestByView.size());
        for (const auto &item : bestByView)
        {
            qualified_entries.push_back(item.second);
        }
        finalizeAuditedSourceSelection(
            plan, std::move(qualified_entries), options);
    }
    else
    {
        // Keep the default-off path byte-for-byte equivalent to the legacy
        // planner: the experiment must not perturb ordering arithmetic, JSON,
        // or cache hashes unless explicitly enabled.
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
    }

    if (plan.selected.empty())
    {
        plan.selected = sequenceFallbackEntries(options);
        plan.usedSequenceFallback = !plan.selected.empty();
        if (audit_source_ranking)
        {
            for (std::size_t index = 0; index < plan.selected.size(); ++index)
            {
                MvsSourcePlanEntry &entry = plan.selected[index];
                entry.adjustedScore = entry.score;
                entry.normalizedSoftAnglePenalty = 0.0f;
                entry.rankingSoftMaximumDegrees =
                    options.softMaxTriangulationAngleDeg;
                entry.rankingEffectiveMaximumDegrees =
                    options.maxTriangulationAngleDeg;
                entry.legacyRankWithinTier = static_cast<int>(index);
                entry.adjustedRankWithinTier = static_cast<int>(index);
                entry.sourceRankingAudited = true;
            }
            plan.controlSelected = plan.selected;
            plan.treatmentSelected = plan.selected;
            plan.selectedCountInvariant = true;
            plan.sourceRankingApplied = false;
            plan.sourceRankingDecisionReason = "sequence_fallback_unchanged";
        }
    }

    plan.sourceViewShortfall = std::max(
        0,
        plan.requestedSourceCount - static_cast<int>(plan.selected.size()));

    return plan;
}

void setAuditedPlanTier(MvsSourcePlan &plan, MvsSourceTier tier)
{
    for (MvsSourcePlanEntry &entry : plan.controlSelected)
    {
        entry.tier = tier;
    }
    for (MvsSourcePlanEntry &entry : plan.treatmentSelected)
    {
        entry.tier = tier;
    }
    for (MvsSourceRankingAuditEntry &audit : plan.rankingCandidates)
    {
        audit.candidate.tier = tier;
    }
    for (MvsSourceRankingAuditEntry &audit : plan.controlRankingCandidates)
    {
        audit.candidate.tier = tier;
    }
}

void appendAuditedPlan(
    MvsSourcePlan &destination,
    MvsSourcePlan &source,
    MvsSourceTier tier)
{
    if (!source.sourceRankingAudited)
    {
        return;
    }

    setAuditedPlanTier(source, tier);
    destination.sourceRankingAudited = true;
    destination.controlRankingCandidates.insert(
        destination.controlRankingCandidates.end(),
        std::make_move_iterator(source.controlRankingCandidates.begin()),
        std::make_move_iterator(source.controlRankingCandidates.end()));
    destination.rankingCandidates.insert(
        destination.rankingCandidates.end(),
        std::make_move_iterator(source.rankingCandidates.begin()),
        std::make_move_iterator(source.rankingCandidates.end()));
}

void finalizeSinglePassRankingPlan(
    MvsSourcePlan &plan,
    const MvsSourcePlannerOptions &options)
{
    if (!plan.sourceRankingAudited && !options.auditSourceRanking &&
        !sourceSoftRankingEnabled(options))
    {
        return;
    }

    plan.sourceRankingAudited = true;
    plan.controlSelected = plan.selected;
    plan.treatmentSelected = plan.selected;
    plan.selectedCountInvariant = true;
    plan.sourceRankingApplied = false;
    plan.sourceRankingDecisionReason = sourceSoftRankingEnabled(options)
        ? "single_pass_treatment"
        : "legacy_score_control";
}

} // namespace

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options)
{
    return planMvsSourceViewsImpl(candidates, options, false);
}

static MvsSourcePlan planMvsSourceViewsVerifiedFirstSinglePass(
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
    setAuditedPlanTier(result, MvsSourceTier::VerifiedPair);

    if (static_cast<int>(result.selected.size()) >= result.requestedSourceCount)
    {
        result.sourceViewShortfall = 0;
        finalizeSinglePassRankingPlan(result, options);
        return result;
    }

    std::unordered_set<int> selectedViews;
    selectedViews.reserve(result.selected.size());
    for (const MvsSourcePlanEntry &entry : result.selected)
    {
        selectedViews.insert(entry.viewIndex);
    }

    std::vector<MvsSourceCandidate> ordinary_backfill_candidates;
    std::vector<MvsSourceCandidate> failed_backfill_candidates;
    ordinary_backfill_candidates.reserve(candidates.size());
    failed_backfill_candidates.reserve(candidates.size());
    for (const MvsSourceCandidate &candidate : candidates)
    {
        if (selectedViews.find(candidate.viewIndex) != selectedViews.end())
        {
            continue;
        }
        if (candidate.verificationStatus ==
                MvsSourceVerificationStatus::MissingStatistics &&
            candidate.pairTotalMatches <
                std::max(
                    1,
                    options.minMissingStatisticsPairMatches))
        {
            result.rejected.push_back(
                {makeEntry(candidate, options),
                 MvsSourceRejectReason::LowQuality});
            continue;
        }
        if (candidate.verificationStatus ==
            MvsSourceVerificationStatus::Failed)
        {
            failed_backfill_candidates.push_back(candidate);
        }
        else
        {
            ordinary_backfill_candidates.push_back(candidate);
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
    MvsSourcePlan ordinary_backfill = planMvsSourceViewsImpl(
        ordinary_backfill_candidates,
        backfillOptions,
        false);
    for (MvsSourcePlanEntry &entry : ordinary_backfill.selected)
    {
        entry.tier = MvsSourceTier::TrackGeometryBackfill;
        result.selected.push_back(entry);
        selectedViews.insert(entry.viewIndex);
    }
    appendAuditedPlan(
        result,
        ordinary_backfill,
        MvsSourceTier::TrackGeometryBackfill);
    result.rejected.insert(
        result.rejected.end(),
        std::make_move_iterator(ordinary_backfill.rejected.begin()),
        std::make_move_iterator(ordinary_backfill.rejected.end()));

    const int failed_backfill_target = std::min(
        result.requestedSourceCount,
        std::max(0, options.failedPairBackfillMaximumTotalSources));
    if (options.allowFailedPairBackfill
        && static_cast<int>(result.selected.size()) < failed_backfill_target)
    {
        MvsSourcePlannerOptions failed_backfill_options = backfillOptions;
        failed_backfill_options.allowFailedPairBackfill = true;
        failed_backfill_options.maxTriangulationAngleDeg = std::min(
            backfillOptions.maxTriangulationAngleDeg,
            options.failedPairBackfillMaximumAngleDeg);
        failed_backfill_options.maxSources = failed_backfill_target -
            static_cast<int>(result.selected.size());
        MvsSourcePlan failed_backfill = planMvsSourceViewsImpl(
            failed_backfill_candidates,
            failed_backfill_options,
            true);
        for (MvsSourcePlanEntry &entry : failed_backfill.selected)
        {
            entry.tier = MvsSourceTier::TrackGeometryBackfill;
            result.selected.push_back(entry);
            selectedViews.insert(entry.viewIndex);
        }
        appendAuditedPlan(
            result,
            failed_backfill,
            MvsSourceTier::TrackGeometryBackfill);
        result.rejected.insert(
            result.rejected.end(),
            std::make_move_iterator(failed_backfill.rejected.begin()),
            std::make_move_iterator(failed_backfill.rejected.end()));
    }
    else if (!options.allowStrictFailedPairBackfill)
    {
        for (const MvsSourceCandidate &candidate : failed_backfill_candidates)
        {
            result.rejected.push_back(
                {makeEntry(candidate, options),
                 MvsSourceRejectReason::LowQuality});
        }
    }

    if (options.allowStrictFailedPairBackfill
        && static_cast<int>(result.selected.size()) < result.requestedSourceCount)
    {
        std::vector<MvsSourceCandidate> strict_candidates;
        strict_candidates.reserve(failed_backfill_candidates.size());
        for (const MvsSourceCandidate &candidate : failed_backfill_candidates)
        {
            if (selectedViews.find(candidate.viewIndex) == selectedViews.end())
            {
                strict_candidates.push_back(candidate);
            }
        }

        MvsSourcePlannerOptions strict_options = backfillOptions;
        strict_options.allowFailedPairBackfill = true;
        strict_options.failedPairBackfillMinimumInliers =
            options.strictFailedPairBackfillMinimumInliers;
        strict_options.failedPairBackfillMinimumMatches =
            options.strictFailedPairBackfillMinimumMatches;
        strict_options.failedPairBackfillMinimumSharedTracks =
            options.strictFailedPairBackfillMinimumSharedTracks;
        strict_options.failedPairBackfillMinimumCoverage =
            options.strictFailedPairBackfillMinimumCoverage;
        strict_options.failedPairBackfillMinimumWilsonLowerBound =
            options.strictFailedPairBackfillMinimumWilsonLowerBound;
        strict_options.failedPairBackfillMaximumAngleDeg =
            options.strictFailedPairBackfillMaximumAngleDeg;
        strict_options.maxTriangulationAngleDeg = std::min(
            backfillOptions.maxTriangulationAngleDeg,
            options.strictFailedPairBackfillMaximumAngleDeg);
        strict_options.maxSources = result.requestedSourceCount -
            static_cast<int>(result.selected.size());
        MvsSourcePlan strict_backfill = planMvsSourceViewsImpl(
            strict_candidates,
            strict_options,
            true);
        for (MvsSourcePlanEntry &entry : strict_backfill.selected)
        {
            entry.tier = MvsSourceTier::StrictPairAuditBackfill;
            entry.verificationReason += entry.verificationReason.empty()
                ? "strict_pair_audit_backfill"
                : ";strict_pair_audit_backfill";
            result.selected.push_back(entry);
        }
        appendAuditedPlan(
            result,
            strict_backfill,
            MvsSourceTier::StrictPairAuditBackfill);
        result.rejected.insert(
            result.rejected.end(),
            std::make_move_iterator(strict_backfill.rejected.begin()),
            std::make_move_iterator(strict_backfill.rejected.end()));
    }
    result.usedSequenceFallback = false;
    result.sourceViewShortfall = std::max(
        0,
        result.requestedSourceCount - static_cast<int>(result.selected.size()));
    finalizeSinglePassRankingPlan(result, options);
    return result;
}

MvsSourcePlan planMvsSourceViewsVerifiedFirst(
    const std::vector<MvsSourceCandidate> &candidates,
    const MvsSourcePlannerOptions &options)
{
    if (!sourceSoftRankingEnabled(options))
    {
        return planMvsSourceViewsVerifiedFirstSinglePass(
            candidates, options);
    }

    MvsSourcePlannerOptions control_options = options;
    control_options.sourceAngleSoftRankingStrength = 0.0f;
    control_options.auditSourceRanking = true;
    MvsSourcePlan control = planMvsSourceViewsVerifiedFirstSinglePass(
        candidates, control_options);
    MvsSourcePlan treatment = planMvsSourceViewsVerifiedFirstSinglePass(
        candidates, options);

    MvsSourcePlan result = std::move(treatment);
    result.controlSelected = control.selected;
    result.treatmentSelected = result.selected;
    result.controlRankingCandidates = std::move(control.rankingCandidates);
    result.selectedCountInvariant =
        result.controlSelected.size() == result.treatmentSelected.size();
    const bool control_is_complete =
        static_cast<int>(result.controlSelected.size()) ==
        result.requestedSourceCount;

    for (MvsSourceRankingAuditEntry &audit :
         result.controlRankingCandidates)
    {
        audit.selectedByPlan = containsSourceIdentity(
            result.controlSelected,
            audit.candidate.viewIndex,
            audit.candidate.tier);
    }
    for (MvsSourceRankingAuditEntry &audit : result.rankingCandidates)
    {
        audit.selectedByPlan = containsSourceIdentity(
            result.treatmentSelected,
            audit.candidate.viewIndex,
            audit.candidate.tier);
    }

    if (!result.selectedCountInvariant)
    {
        result.selected = result.controlSelected;
        result.sourceRankingApplied = false;
        result.sourceRankingDecisionReason =
            "selected_count_mismatch_fallback";
    }
    else if (!control_is_complete)
    {
        result.selected = result.controlSelected;
        result.sourceRankingApplied = false;
        result.sourceRankingDecisionReason = "control_source_shortfall";
    }
    else
    {
        result.selected = result.treatmentSelected;
        result.sourceRankingApplied = !sameSourceSelection(
            result.controlSelected, result.treatmentSelected);
        result.sourceRankingDecisionReason = result.sourceRankingApplied
            ? "selection_changed"
            : "ranking_unchanged";
    }
    result.sourceRankingAudited = true;
    result.sourceViewShortfall = std::max(
        0,
        result.requestedSourceCount - static_cast<int>(result.selected.size()));
    return result;
}

std::vector<int> planMvsRepairSourceViews(
    const std::vector<int> &preferredSources,
    const std::vector<bool> &sourceEligibility,
    int refIndex,
    int requestedSourceCount)
{
    const int view_count = static_cast<int>(sourceEligibility.size());
    if (refIndex < 0 || refIndex >= view_count || view_count <= 1)
    {
        return {};
    }

    const int maximum_source_count = std::min(16, view_count - 1);
    const int minimum_source_count = std::min(2, maximum_source_count);
    const int target_count = std::clamp(
        requestedSourceCount, minimum_source_count, maximum_source_count);

    std::vector<int> sources;
    sources.reserve(static_cast<std::size_t>(target_count));
    auto append_source = [&](int source_index)
    {
        if (static_cast<int>(sources.size()) >= target_count ||
            source_index < 0 || source_index >= view_count || source_index == refIndex ||
            !sourceEligibility[static_cast<std::size_t>(source_index)] ||
            std::find(sources.begin(), sources.end(), source_index) != sources.end())
        {
            return;
        }
        sources.push_back(source_index);
    };

    for (int source_index : preferredSources)
    {
        append_source(source_index);
        if (static_cast<int>(sources.size()) >= target_count)
        {
            return sources;
        }
    }

    for (int distance = 1;
         distance < view_count && static_cast<int>(sources.size()) < target_count;
         ++distance)
    {
        append_source((refIndex - distance + view_count) % view_count);
        append_source((refIndex + distance) % view_count);
    }
    return sources;
}

std::vector<MvsSourcePairQuality> filterMvsSourcePairQualitiesForImages(
    const std::vector<MvsSourcePairQuality> &qualities,
    const std::vector<std::string> &imagePaths)
{
    std::set<QString> activeImages;
    std::map<QString, std::string> activePathByNormalizedPath;
    std::map<QString, std::string> uniqueActivePathByFileName;
    std::set<QString> ambiguousFileNames;
    for (const std::string &imagePath : imagePaths)
    {
        const QString normalizedPath = normalizedSourceImagePath(imagePath);
        if (!normalizedPath.isEmpty())
        {
            activeImages.insert(normalizedPath);
            activePathByNormalizedPath[normalizedPath] = imagePath;

            const QString fileName = QFileInfo(
                QString::fromStdString(imagePath)).fileName().toCaseFolded();
            if (!fileName.isEmpty())
            {
                if (uniqueActivePathByFileName.contains(fileName))
                {
                    uniqueActivePathByFileName.erase(fileName);
                    ambiguousFileNames.insert(fileName);
                }
                else if (!ambiguousFileNames.contains(fileName))
                {
                    uniqueActivePathByFileName[fileName] = imagePath;
                }
            }
        }
    }

    // A pair audit often travels together with an image set to a new workspace.
    // Only enable file-name rebinding when the entire audit is detached from the
    // current absolute paths.  If any endpoint still matches exactly, stale
    // records from images removed by the user must remain stale.
    const bool hasAnyExactEndpoint = std::any_of(
        qualities.cbegin(),
        qualities.cend(),
        [&activeImages](const MvsSourcePairQuality &quality)
        {
            return activeImages.contains(normalizedSourceImagePath(quality.imageA))
                || activeImages.contains(normalizedSourceImagePath(quality.imageB));
        });

    const auto resolveActivePath = [&](const std::string &storedPath) -> std::string
    {
        const QString normalizedPath = normalizedSourceImagePath(storedPath);
        const auto exact = activePathByNormalizedPath.find(normalizedPath);
        if (exact != activePathByNormalizedPath.end())
        {
            return exact->second;
        }
        if (hasAnyExactEndpoint)
        {
            return {};
        }

        const QString fileName = QFileInfo(
            QString::fromStdString(storedPath)).fileName().toCaseFolded();
        const auto relocated = uniqueActivePathByFileName.find(fileName);
        return relocated != uniqueActivePathByFileName.end()
            ? relocated->second
            : std::string{};
    };

    std::map<QString, MvsSourcePairQuality> bestByPair;
    for (const MvsSourcePairQuality &quality : qualities)
    {
        const std::string imageA = resolveActivePath(quality.imageA);
        const std::string imageB = resolveActivePath(quality.imageB);
        if (imageA.empty() || imageB.empty())
        {
            continue;
        }

        const QString key = sourcePairKey(imageA, imageB);
        if (key.isEmpty())
        {
            continue;
        }

        MvsSourcePairQuality reboundQuality = quality;
        reboundQuality.imageA = imageA;
        reboundQuality.imageB = imageB;

        auto it = bestByPair.find(key);
        if (it == bestByPair.end()
            || reboundQuality.geometricInliers > it->second.geometricInliers
            || (reboundQuality.geometricInliers == it->second.geometricInliers
                && reboundQuality.hasVerificationStatistics
                && !it->second.hasVerificationStatistics))
        {
            bestByPair[key] = std::move(reboundQuality);
        }
    }

    std::vector<MvsSourcePairQuality> filtered;
    filtered.reserve(bestByPair.size());
    for (auto &item : bestByPair)
    {
        filtered.push_back(std::move(item.second));
    }
    return filtered;
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
    QString verificationStatus = QStringLiteral("not_requested");
    if (entry.verificationStatus == MvsSourceVerificationStatus::Verified)
    {
        verificationStatus = QStringLiteral("verified");
    }
    else if (entry.verificationStatus == MvsSourceVerificationStatus::Failed)
    {
        verificationStatus = QStringLiteral("failed");
    }
    else if (entry.verificationStatus ==
             MvsSourceVerificationStatus::MissingStatistics)
    {
        verificationStatus = QStringLiteral("missing_statistics");
    }
    else if (entry.verificationStatus ==
             MvsSourceVerificationStatus::SequenceFallback)
    {
        verificationStatus = QStringLiteral("sequence_fallback");
    }
    object.insert(QStringLiteral("verification_status"), verificationStatus);
    object.insert(QStringLiteral("pair_total_matches"), entry.pairTotalMatches);
    object.insert(QStringLiteral("pair_inlier_ratio"), entry.pairInlierRatio);
    object.insert(QStringLiteral("pair_coverage_score"), entry.pairCoverageScore);
    object.insert(
        QStringLiteral("verification_reason"),
        QString::fromStdString(entry.verificationReason));
    object.insert(QStringLiteral("sequence_fallback"), entry.sequenceFallback);
    QString sourceTier = QStringLiteral("verified_pair");
    if (entry.tier == MvsSourceTier::TrackGeometryBackfill)
    {
        sourceTier = QStringLiteral("track_geometry_backfill");
    }
    else if (entry.tier == MvsSourceTier::StrictPairAuditBackfill)
    {
        sourceTier = QStringLiteral("strict_pair_audit_backfill");
    }
    else if (entry.tier == MvsSourceTier::SequenceFallback)
    {
        sourceTier = QStringLiteral("sequence_fallback");
    }
    object.insert(QStringLiteral("source_tier"), sourceTier);
    object.insert(QStringLiteral("score"), entry.score);
    object.insert(QStringLiteral("source_quality_score"), entry.sourceQualityScore);
    if (entry.sourceRankingAudited)
    {
        object.insert(QStringLiteral("legacy_score"), entry.score);
        object.insert(QStringLiteral("adjusted_score"), entry.adjustedScore);
        object.insert(
            QStringLiteral("normalized_soft_angle_penalty"),
            entry.normalizedSoftAnglePenalty);
        object.insert(
            QStringLiteral("ranking_soft_maximum_degrees"),
            entry.rankingSoftMaximumDegrees);
        object.insert(
            QStringLiteral("ranking_effective_maximum_degrees"),
            entry.rankingEffectiveMaximumDegrees);
        object.insert(
            QStringLiteral("legacy_rank_within_tier"),
            entry.legacyRankWithinTier);
        object.insert(
            QStringLiteral("adjusted_rank_within_tier"),
            entry.adjustedRankWithinTier);
    }
    return object;
}

QJsonObject mvsSourceAngleDiagnosticsToJson(
    const MvsSourceAnglePolicy &policy,
    const MvsSourcePlan &plan)
{
    float selected_maximum_degrees = 0.0f;
    for (const MvsSourcePlanEntry &entry : plan.selected)
    {
        if (std::isfinite(entry.medianTriangulationAngleDeg) &&
            entry.medianTriangulationAngleDeg > 0.0f)
        {
            selected_maximum_degrees = std::max(
                selected_maximum_degrees,
                entry.medianTriangulationAngleDeg);
        }
    }

    const int angle_rejected_candidate_count = static_cast<int>(
        std::count_if(
            plan.rejected.cbegin(),
            plan.rejected.cend(),
            [](const MvsSourceRejectedCandidate &rejected)
            {
                return rejected.reason ==
                    MvsSourceRejectReason::TriangulationAngle;
            }));

    return QJsonObject{
        {QStringLiteral("scope"),
         QStringLiteral("patchmatch_source_plan")},
        {QStringLiteral("configured_cap_degrees"),
         policy.configuredCapDegrees},
        {QStringLiteral("scene_maximum_degrees"),
         policy.sceneMaximumDegrees},
        {QStringLiteral("effective_maximum_degrees"),
         policy.effectiveMaximumDegrees},
        {QStringLiteral("cap_enabled"), policy.capEnabled},
        {QStringLiteral("cap_applied"), policy.capApplied},
        {QStringLiteral("only_tightens_scene_maximum"), true},
        {QStringLiteral("sequence_fallback_allowed"),
         policy.sequenceFallbackAllowed},
        {QStringLiteral("sequence_fallback_disabled_reason"),
         policy.capEnabled
             ? QStringLiteral("explicit_source_angle_cap")
             : QString()},
        {QStringLiteral("selected_source_count"),
         static_cast<int>(plan.selected.size())},
        {QStringLiteral("selected_maximum_degrees"),
         selected_maximum_degrees},
        {QStringLiteral("angle_rejected_candidate_count"),
         angle_rejected_candidate_count},
        {QStringLiteral("policy"),
         QStringLiteral("min_scene_maximum_and_configured_cap")}};
}

QJsonObject mvsSourceRankingDiagnosticsToJson(
    const MvsSourceRankingPolicy &policy,
    const MvsSourcePlan &plan)
{
    const auto entries_to_json = [](const auto &entries)
    {
        QJsonArray result;
        for (const auto &entry : entries)
        {
            result.append(mvsSourcePlanEntryToJson(entry));
        }
        return result;
    };

    const auto candidates_to_json = [](const auto &audit_entries)
    {
        QJsonArray result;
        for (const MvsSourceRankingAuditEntry &audit : audit_entries)
        {
            QJsonObject candidate = mvsSourcePlanEntryToJson(audit.candidate);
            candidate.insert(
                QStringLiteral("selected_by_plan"),
                audit.selectedByPlan);
            result.append(candidate);
        }
        return result;
    };
    const auto unique_qualified_view_count = [](const auto &audit_entries)
    {
        std::set<int> qualified_views;
        for (const MvsSourceRankingAuditEntry &audit : audit_entries)
        {
            qualified_views.insert(audit.candidate.viewIndex);
        }
        return static_cast<int>(qualified_views.size());
    };

    return QJsonObject{
        {QStringLiteral("scope"), QStringLiteral("patchmatch_source_plan")},
        {QStringLiteral("enabled"),
         policy.evaluateCompleteVisibilityCandidatePool},
        {QStringLiteral("candidate_pool_policy"),
         QStringLiteral("complete_evaluated_visibility_candidate_pool")},
        {QStringLiteral("visibility_graph_scope"),
         policy.visibilityGraphCoversAllViewPairs
             ? QStringLiteral("all_co_visible_or_required_pairs")
             : QStringLiteral("deterministic_bounded_graph")},
        {QStringLiteral("view_count"), policy.viewCount},
        {QStringLiteral("visibility_candidate_count"),
         policy.visibilityCandidateCount},
        {QStringLiteral("legacy_evaluated_candidate_count"),
         policy.legacyEvaluatedCandidateCount},
        {QStringLiteral("complete_evaluated_candidate_count"),
         policy.completeEvaluatedCandidateCount},
        {QStringLiteral("control_qualified_candidate_count"),
         unique_qualified_view_count(plan.controlRankingCandidates)},
        {QStringLiteral("control_qualified_tier_entry_count"),
         static_cast<int>(plan.controlRankingCandidates.size())},
        {QStringLiteral("treatment_qualified_candidate_count"),
         unique_qualified_view_count(plan.rankingCandidates)},
        {QStringLiteral("treatment_qualified_tier_entry_count"),
         static_cast<int>(plan.rankingCandidates.size())},
        {QStringLiteral("soft_ranking_strength"),
         policy.softRankingStrength},
        {QStringLiteral("soft_maximum_degrees"),
         policy.softMaximumDegrees},
        {QStringLiteral("effective_maximum_degrees"),
         policy.effectiveMaximumDegrees},
        {QStringLiteral("soft_ranking_formula"),
         QStringLiteral("legacy_score*exp(-strength*t)")},
        {QStringLiteral("control_selected_count"),
         static_cast<int>(plan.controlSelected.size())},
        {QStringLiteral("treatment_selected_count"),
         static_cast<int>(plan.treatmentSelected.size())},
        {QStringLiteral("requested_source_count"),
         plan.requestedSourceCount},
        {QStringLiteral("count_invariant"),
         plan.selectedCountInvariant},
        {QStringLiteral("selection_changed"),
         !sameSourceSelection(
             plan.controlSelected, plan.treatmentSelected)},
        {QStringLiteral("selected_view_set_changed"),
         !sameSelectedViewSet(
             plan.controlSelected, plan.treatmentSelected)},
        {QStringLiteral("selected_order_changed"),
         !sameSelectedViewOrder(
             plan.controlSelected, plan.treatmentSelected)},
        {QStringLiteral("applied"), plan.sourceRankingApplied},
        {QStringLiteral("decision_reason"),
         QString::fromStdString(plan.sourceRankingDecisionReason)},
        {QStringLiteral("control_selected"),
         entries_to_json(plan.controlSelected)},
        {QStringLiteral("treatment_selected"),
         entries_to_json(plan.treatmentSelected)},
        {QStringLiteral("control_candidate_ranking"),
         candidates_to_json(plan.controlRankingCandidates)},
        {QStringLiteral("treatment_candidate_ranking"),
         candidates_to_json(plan.rankingCandidates)}};
}

bool validateMvsSourceRankingConfiguration(
    bool evaluate_complete_visibility_candidate_pool,
    float soft_ranking_strength,
    float source_maximum_angle_degrees_cap,
    std::string *error_message)
{
    const auto fail = [error_message](const char *message)
    {
        if (error_message)
        {
            *error_message = message;
        }
        return false;
    };

    if (!std::isfinite(soft_ranking_strength) ||
        soft_ranking_strength < 0.0f || soft_ranking_strength > 4.0f)
    {
        return fail("source angle soft ranking strength must be finite and within [0,4]");
    }
    if (!std::isfinite(source_maximum_angle_degrees_cap) ||
        source_maximum_angle_degrees_cap < 0.0f ||
        source_maximum_angle_degrees_cap > 90.0f)
    {
        return fail("hard source angle cap must be finite and within [0,90]");
    }
    if (soft_ranking_strength > 0.0f &&
        !evaluate_complete_visibility_candidate_pool)
    {
        return fail("source angle soft ranking requires the complete visibility candidate pool");
    }
    if (soft_ranking_strength > 0.0f &&
        source_maximum_angle_degrees_cap != 0.0f)
    {
        return fail("source angle soft ranking cannot be combined with a hard source angle cap");
    }
    if (error_message)
    {
        error_message->clear();
    }
    return true;
}

} // namespace mvs
} // namespace xjw
