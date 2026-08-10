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

} // namespace

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options)
{
    return planMvsSourceViewsImpl(candidates, options, false);
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
    }
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
        }
        result.rejected.insert(
            result.rejected.end(),
            std::make_move_iterator(failed_backfill.rejected.begin()),
            std::make_move_iterator(failed_backfill.rejected.end()));
    }
    else
    {
        for (const MvsSourceCandidate &candidate : failed_backfill_candidates)
        {
            result.rejected.push_back(
                {makeEntry(candidate, options),
                 MvsSourceRejectReason::LowQuality});
        }
    }
    result.usedSequenceFallback = false;
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
