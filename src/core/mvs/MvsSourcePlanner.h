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
    float minSourceQualityScore = 0.0f;
    bool allowWeakKnownOverlap = true;
    bool requireVerifiedPairGeometry = false;
};

struct MvsSourcePlan
{
    std::vector<MvsSourcePlanEntry> selected;
    std::vector<MvsSourceRejectedCandidate> rejected;
    bool usedSequenceFallback = false;
    int requestedSourceCount = 0;
    int sourceViewShortfall = 0;
};

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options);
MvsSourcePlan planMvsSourceViewsVerifiedFirst(
    const std::vector<MvsSourceCandidate> &candidates,
    const MvsSourcePlannerOptions &options);

std::vector<MvsSourcePairQuality> filterMvsSourcePairQualitiesForImages(
    const std::vector<MvsSourcePairQuality> &qualities,
    const std::vector<std::string> &imagePaths);

QJsonObject mvsSourcePlanEntryToJson(const MvsSourcePlanEntry &entry);

} // namespace mvs
} // namespace xjw
