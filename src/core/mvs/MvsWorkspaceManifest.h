#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace xjw::mvs
{

struct DepthGenConfig;

struct MvsDepthFrameQualification
{
    QString acceptance;
    bool fusionEligible = false;
    bool reclassified = false;
    /// The frame qualified through its final hard multi-view core because the
    /// continuous adaptive residual was not calibrated for this capture.  TSDF
    /// must then consume the discrete support/spread maps, not the conflicting
    /// adaptive maps that triggered the fallback.
    bool useDiscreteGeometryFallback = false;
};

// Increment whenever a production depth algorithm change makes persisted
// depth maps unsuitable for transparent reuse by a newer build.
inline constexpr int kMvsDepthAlgorithmRevision = 38;
/// Revision 37 persists the exact source-view ordinal table used by the
/// per-pixel geometry-source mask. Revision 36 stored only the shorter
/// PatchMatch source list even though orbital consistency and measured repair
/// could encode up to sixteen repair sources, so its source bits cannot be
/// decoded reliably by TSDF meshing.
inline constexpr int kMvsMinimumModelCompatibleRevision = 37;
inline constexpr int kMvsAdaptiveGeometryEvidenceRevision = 13;
inline constexpr int kMvsAdaptiveGeometryConflictRatioRevision = 14;
inline constexpr int kMvsDepthProvenanceRevision = 17;
inline constexpr int kMvsGeometrySupportedLowConfidenceRevision = 18;
inline constexpr int kMvsMultiHypothesisTargetedRecoveryRevision = 19;
inline constexpr int kMvsSurfaceAwareTargetedRecoveryRevision = 20;
inline constexpr int kMvsPostConsistencyResidualReestimationRevision = 21;
inline constexpr int kMvsGeometryFusionSupportRevision = 34;
inline constexpr int kMvsDiscreteGeometryCoreRatioRevision = 35;
inline constexpr int kMvsSparseAbsoluteDepthResidualRevision = 36;
inline constexpr int kMvsGeometrySourceOrdinalRevision = 37;
inline constexpr int kMvsRobustPhotometricAndLearnedCandidateRevision = 38;

struct MvsDepthFrameRecord
{
    int refIndex = -1;
    QString refImage;
    QStringList sourceImages;
    QVector<int> sourceIndices;
    QVector<int> geometrySourceIndices;
    QJsonArray sourcePlan;
    QString qualityProfile;
    int configuredSourceViewCount = 0;
    int sourceViewCount = 0;
    int requestedSourceViewCount = 0;
    int sourceViewShortfall = 0;
    QString sourceViewShortfallReason;
    double meanSourceQualityScore = 0.0;
    double minSourceQualityScore = 0.0;
    double meanDepthConfidence = 0.0;
    double effectivePatchMatchConfidenceThreshold = 0.0;
    int validPixelCount = 0;
    double validCoverage = -1.0;
    QJsonObject depthQuality;
    QJsonObject depthCompleteness;
    QJsonObject missingReasonSummary;
    QJsonObject crossViewRepairDiagnostics;
    QJsonObject targetedGapRecoveryDiagnostics;
    QJsonObject residualReestimationDiagnostics;
    QJsonObject learnedCandidateDiagnostics;
    QJsonObject depthProvenanceSummary;
    QJsonObject geometryEvidenceDiagnostics;
    QJsonObject poseRefinementDiagnostics;
    QJsonObject derivedCameraModel;
    QJsonObject qualityDecision;
    QJsonArray pyramidLevels;
    QString maskSource;
    double maskCoverage = -1.0;
    int selectedLevel = 0;
    QString fallbackReason;
    int pyramidRequestedLevelCount = 3;
    int pyramidActiveLevelCount = 0;
    int pyramidMinimumShortSide = 0;
    QString pyramidDegradedReason;
    QString sceneProfile;
    QString filterMode;
    QString acceptance;
    bool fusionEligible = false;
    bool fusionEligibilityKnown = false;
    QJsonObject depthPostprocess;
    QJsonObject cameraModel;
    QString status;
    QString device;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
    QString rawGeometrySupportPath;
    QString rawAdaptiveGeometrySupportWeightPath;
    QString rawAdaptiveGeometryEffectiveViewCountPath;
    QString rawAdaptiveGeometryConflictRatioPath;
    QString rawAdaptiveGeometryConflictWeightPath; ///< Legacy revision-13 absolute conflict mass.
    QString rawGeometrySourceMaskPath;
    QString rawInverseDepthMeanPath;
    QString rawInverseDepthSpreadPath;
    QString crossViewRepairedMaskPath;
    QString targetedGapRecoveredMaskPath;
    QString residualReestimatedMaskPath;
    QString depthProvenancePath;
    QString validMaskPath;
    QString supportMaskPath;
    QString missingReasonPath;
    QString missingReasonPreviewPath;
    int gridWidth = 0;
    int gridHeight = 0;
    qint64 elapsedMs = 0;
    QString error;
    QString configHash;
    int algorithmRevision = 0;

    QJsonObject toJson() const;
    static MvsDepthFrameRecord fromJson(const QJsonObject &object);
};

MvsDepthFrameQualification qualifyMvsDepthFrameArtifact(
    const QJsonObject &artifact);

class MvsWorkspaceManifest
{
public:
    bool load(const QString &path, QString *errorMsg = nullptr);
    bool saveAtomic(const QString &path, QString *errorMsg = nullptr) const;

    void clear();

    QString configHash() const;
    void setConfigHash(const QString &hash);

    const QVector<MvsDepthFrameRecord> &frames() const;
    QVector<MvsDepthFrameRecord> completedFramesSortedByName() const;

    void upsertFrame(const MvsDepthFrameRecord &record);
    void markRunning(int refIndex, const QString &refImage, const QString &configHash);
    void markCompleted(const MvsDepthFrameRecord &record);
    void markFailed(int refIndex, const QString &error);
    void updatePoseRefinement(int refIndex,
                              const QJsonObject &diagnostics,
                              const QJsonObject &derivedCameraModel = {});

    bool hasReusableCompletedFrame(int refIndex, const QString &configHash) const;
    QJsonObject toJson() const;

private:
    int findFrameIndex(int refIndex) const;

    QString _configHash;
    QVector<MvsDepthFrameRecord> _frames;
};

QString makeMvsDepthConfigHash(const DepthGenConfig &config, int viewCount);

} // namespace xjw::mvs
