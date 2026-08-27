#pragma once

#include "MvsTypes.h"

#include <QJsonObject>

#include <string>
#include <vector>

namespace xjw
{
namespace mvs
{

enum class MvsSourceRejectReason
{
    None,
    InvalidIndex,
    Self,
    Duplicate,
    NoEvidence,
    TriangulationAngle,
    LowQuality
};

enum class MvsSourceTier
{
    VerifiedPair,
    TrackGeometryBackfill,
    StrictPairAuditBackfill,
    SequenceFallback
};

enum class MvsSourceVerificationStatus
{
    Verified,
    Failed,
    MissingStatistics,
    NotRequested,
    SequenceFallback
};

struct MvsSourceCandidate
{
    int viewIndex = -1;
    int sharedTracks = 0;
    int geometricInliers = 0;
    float medianTriangulationAngleDeg = 0.0f;
    float coverageScore = 0.0f;
    float baselineScore = 0.0f;
    int sequenceDistance = 0;
    bool knownOverlap = false;
    bool verifiedPairGeometry = false;
    MvsSourceVerificationStatus verificationStatus =
        MvsSourceVerificationStatus::NotRequested;
    int pairTotalMatches = 0;
    float pairCoverageScore = 0.0f;
    std::string verificationReason;
};

struct MvsSourcePlanEntry
{
    int viewIndex = -1;
    int sharedTracks = 0;
    int geometricInliers = 0;
    float medianTriangulationAngleDeg = 0.0f;
    float coverageScore = 0.0f;
    float baselineScore = 0.0f;
    int sequenceDistance = 0;
    bool knownOverlap = false;
    bool verifiedPairGeometry = false;
    bool sequenceFallback = false;
    MvsSourceTier tier = MvsSourceTier::VerifiedPair;
    MvsSourceVerificationStatus verificationStatus =
        MvsSourceVerificationStatus::NotRequested;
    int pairTotalMatches = 0;
    float pairInlierRatio = 0.0f;
    float pairCoverageScore = 0.0f;
    std::string verificationReason;
    float score = 0.0f;
    float sourceQualityScore = 0.0f;
    float adjustedScore = 0.0f;
    float normalizedSoftAnglePenalty = 0.0f;
    float rankingSoftMaximumDegrees = 0.0f;
    float rankingEffectiveMaximumDegrees = 0.0f;
    int legacyRankWithinTier = -1;
    int adjustedRankWithinTier = -1;
    bool sourceRankingAudited = false;
};

struct MvsSourceRankingAuditEntry
{
    MvsSourcePlanEntry candidate;
    bool selectedByPlan = false;
};

struct MvsSourceRejectedCandidate
{
    MvsSourcePlanEntry candidate;
    MvsSourceRejectReason reason = MvsSourceRejectReason::None;
};

struct MvsSourcePlannerOptions
{
    int refIndex = -1;
    int viewCount = 0;
    int maxSources = 4;
    bool allowSequenceFallback = true;
    bool rejectAngleOutliers = false;
    float minTriangulationAngleDeg = 0.2f;
    float maxTriangulationAngleDeg = 35.0f;
    float preferredTriangulationAngleDeg = 10.0f;
    float softMaxTriangulationAngleDeg = 25.0f;
    int minSharedTracks = 0;
    int minGeometricInliers = 0;
    int minMissingStatisticsPairMatches = 16;
    bool allowFailedPairBackfill = false;
    int failedPairBackfillMaximumTotalSources = 3;
    int failedPairBackfillMinimumInliers = 12;
    int failedPairBackfillMinimumMatches = 14;
    int failedPairBackfillMinimumSharedTracks = 20;
    float failedPairBackfillMinimumCoverage = 0.1875f;
    float failedPairBackfillMinimumWilsonLowerBound = 0.50f;
    float failedPairBackfillMaximumAngleDeg = 65.0f;
    bool allowStrictFailedPairBackfill = false;
    int strictFailedPairBackfillMinimumInliers = 24;
    int strictFailedPairBackfillMinimumMatches = 32;
    int strictFailedPairBackfillMinimumSharedTracks = 40;
    float strictFailedPairBackfillMinimumCoverage = 0.30f;
    float strictFailedPairBackfillMinimumWilsonLowerBound = 0.65f;
    float strictFailedPairBackfillMaximumAngleDeg = 55.0f;
    float minSourceQualityScore = 0.0f;
    bool allowWeakKnownOverlap = true;
    bool requireVerifiedPairGeometry = false;
    float sourceAngleSoftRankingStrength = 0.0f;
    bool auditSourceRanking = false;
};

struct MvsSourcePlan
{
    std::vector<MvsSourcePlanEntry> selected;
    std::vector<MvsSourceRejectedCandidate> rejected;
    bool usedSequenceFallback = false;
    int requestedSourceCount = 0;
    int sourceViewShortfall = 0;
    std::vector<MvsSourcePlanEntry> controlSelected;
    std::vector<MvsSourcePlanEntry> treatmentSelected;
    std::vector<MvsSourceRankingAuditEntry> controlRankingCandidates;
    std::vector<MvsSourceRankingAuditEntry> rankingCandidates;
    bool sourceRankingAudited = false;
    bool sourceRankingApplied = false;
    bool selectedCountInvariant = true;
    std::string sourceRankingDecisionReason;
};

struct MvsSourceRankingPolicy
{
    bool evaluateCompleteVisibilityCandidatePool = false;
    bool visibilityGraphCoversAllViewPairs = false;
    int viewCount = 0;
    int visibilityCandidateCount = 0;
    int legacyEvaluatedCandidateCount = 0;
    int completeEvaluatedCandidateCount = 0;
    float softRankingStrength = 0.0f;
    float softMaximumDegrees = 25.0f;
    float effectiveMaximumDegrees = 35.0f;
};

struct MvsSourceAnglePolicy
{
    float configuredCapDegrees = 0.0f;
    float sceneMaximumDegrees = 0.0f;
    float effectiveMaximumDegrees = 0.0f;
    bool capEnabled = false;
    bool capApplied = false;
    bool sequenceFallbackAllowed = true;
};

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options);
MvsSourcePlan planMvsSourceViewsVerifiedFirst(
    const std::vector<MvsSourceCandidate> &candidates,
    const MvsSourcePlannerOptions &options);

std::vector<int> planMvsRepairSourceViews(
    const std::vector<int> &preferredSources,
    const std::vector<bool> &sourceEligibility,
    int refIndex,
    int requestedSourceCount);

/// Resolves the cross-camera consensus budget independently from PatchMatch's
/// smaller photometric source set. Orbital scenes use the full 16-bit source
/// evidence mask when enough registered views are available.
int recommendedMvsCrossViewSourceCount(
    MvsSceneProfile sceneProfile,
    int configuredSourceCount,
    int viewCount);

std::vector<MvsSourcePairQuality> filterMvsSourcePairQualitiesForImages(
    const std::vector<MvsSourcePairQuality> &qualities,
    const std::vector<std::string> &imagePaths);

QJsonObject mvsSourcePlanEntryToJson(const MvsSourcePlanEntry &entry);
QJsonObject mvsSourceAngleDiagnosticsToJson(
    const MvsSourceAnglePolicy &policy,
    const MvsSourcePlan &plan);
QJsonObject mvsSourceRankingDiagnosticsToJson(
    const MvsSourceRankingPolicy &policy,
    const MvsSourcePlan &plan);

bool validateMvsSourceRankingConfiguration(
    bool evaluate_complete_visibility_candidate_pool,
    float soft_ranking_strength,
    float source_maximum_angle_degrees_cap,
    std::string *error_message = nullptr);

} // namespace mvs
} // namespace xjw
