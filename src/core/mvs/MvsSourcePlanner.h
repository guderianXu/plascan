#pragma once

#include <QJsonObject>

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
    float minSourceQualityScore = 0.0f;
    bool allowWeakKnownOverlap = true;
    bool requireVerifiedPairGeometry = false;
};

struct MvsSourcePlan
{
    std::vector<MvsSourcePlanEntry> selected;
    std::vector<MvsSourceRejectedCandidate> rejected;
    bool usedSequenceFallback = false;
};

MvsSourcePlan planMvsSourceViews(const std::vector<MvsSourceCandidate> &candidates,
                                 const MvsSourcePlannerOptions &options);

QJsonObject mvsSourcePlanEntryToJson(const MvsSourcePlanEntry &entry);

} // namespace mvs
} // namespace xjw
