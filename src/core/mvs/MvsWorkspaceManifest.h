#pragma once

#include "DepthFrameQualificationPolicy.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

#include <vector>

namespace xjw::mvs
{

struct DepthGenConfig;
struct CameraView;
struct SparseCloud;

struct MvsDepthFrameQualification
{
    QString acceptance;
    bool fusionEligible = false;
    bool fusionEligibilityKnown = false;
    DepthFrameRole role = DepthFrameRole::Excluded;
    /// The frame qualified through its final hard multi-view core because the
    /// continuous adaptive residual was not calibrated for this capture.  TSDF
    /// must then consume the discrete support/spread maps, not the conflicting
    /// adaptive maps that triggered the fallback.
    bool useDiscreteGeometryFallback = false;
};

// Increment whenever a production depth algorithm change makes persisted
// depth maps unsuitable for transparent reuse by a newer build.
inline constexpr int kMvsDepthAlgorithmRevision = 52;
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
/// Revision 39 persists the exact full-resolution raster and camera used by
/// MVS.  Earlier manifests paired a distorted ref_image with a zero-distortion
/// working camera, so replay and mesh texturing could sample the wrong pixels.
inline constexpr int kMvsPreparedRasterProvenanceRevision = 39;
/// Revision 40 makes automatic scene classification fail closed to the
/// general/custom profile unless the camera layout passes an explicit orbital
/// ring gate. It also freezes consistency source plans before frame acceptance
/// updates and normalizes an all-zero/no-source geometry mask by omitting both
/// the mask path and its empty ordinal table.
inline constexpr int kMvsGeneralSceneAndStableSourcePlanRevision = 40;
/// Revision 41 makes Custom acceptance fail closed when neither sparse
/// absolute-depth evidence nor a strong discrete multi-view core is present.
/// It also distinguishes semantic project support masks from content and
/// prepared-raster validity masks before applying normalized coverage gates.
inline constexpr int kMvsCustomGeometryQualityGateRevision = 41;
/// Revision 42 records the initial admission decision and distinguishes an
/// actual complete-pool source-plan replacement from indirect cross-frame
/// effects before granting or removing a Primary role.
inline constexpr int kMvsCompletePoolAdmissionRevision = 42;
/// Revision 43 evaluates reduced-grid zero-radius consistency over the bounded
/// nearest subpixel footprint and preserves one audited boundary shell without
/// expanding the protection band to a full reduced-grid pixel.
inline constexpr int kMvsNativeGridQualityRevision = 43;
/// Revision 44 records independent photometric and geometry confidence for
/// evidence-guided correction, requires corrected pixels to carry causal
/// geometry evidence before relative-retention loss can be explained, and
/// keeps one inward measured boundary shell on reduced native grids.
inline constexpr int kMvsCausalAdmissionEvidenceRevision = 44;
/// Revision 45 adds a per-pixel photometric source bitset, asymmetric
/// near/far checkerboard propagation with a local surface-normal candidate,
/// and a frozen-source-depth geometric guidance pass. The guidance objective
/// remains separate from independent geometric confidence.
inline constexpr int kMvsJointViewAndGeometricGuidanceRevision = 45;
/// Revision 46 fingerprints the effective image and mask contents, validates
/// persisted evidence before reuse, restores the adaptive backend's auxiliary
/// source-mask/source-depth contract, and keeps observed consistency retention
/// separate from any diagnostic publication fallback.
inline constexpr int kMvsCacheAndConsistencyAuditRevision = 46;
/// Revision 47 makes the initial write a non-publishable checkpoint, records
/// whether consistency and frozen-depth guidance were expected and actually
/// completed, and preserves the unavailable single-view consistency state.
inline constexpr int kMvsDurableDepthPublicationRevision = 47;
/// Revision 48 prevents a fast but quality-rejected OpenCL device from being
/// rewarded as useful throughput during automatic CUDA+OpenCL scheduling.
/// It also aligns the sparse absolute-depth primary threshold for trusted
/// orbital captures with the selected consistency-filter strength, while
/// generic captures retain the strict half-percent threshold. Existing depth
/// results must be recomputed under both corrected policies.
inline constexpr int kMvsQualityAwareHeterogeneousSchedulingRevision = 48;
/// Revision 49 blocks targeted orbital gap recovery when independent sparse
/// anchors already classify the native depth as non-primary, and tightens the
/// default recovery confidence, consensus spread, and prior search radius.
inline constexpr int kMvsSparseAnchoredGapRecoveryRevision = 49;
/// Revision 50 replaces source-count majority voting with a bounded adaptive
/// three-to-four-view photometric inlier set and keeps strict confidence-filter
/// output when a frame collapses instead of restoring the rejected depth map.
inline constexpr int kMvsAdaptivePhotometricInlierRevision = 50;
/// Revision 51 adds a deterministic final plane-propagation pass after random
/// refinement and optional reference-raster guidance for the scale-invariant
/// depth bilateral filter. The guidance remains opt-in after dataset A/B.
inline constexpr int kMvsFinalPropagationAndGuidedFilterRevision = 51;

struct MvsDepthFrameRecord
{
    int refIndex = -1;
    QString refImage;
    QString preparedImage;
    QString preparedValidMaskPath;
    QJsonObject preparedCameraModel;
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
    bool consistencyPublicationExpected = false;
    bool geometricGuidancePassExpected = false;
    bool geometricGuidancePassApplied = false;
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
    DepthFrameRole role = DepthFrameRole::Excluded;
    QJsonObject depthPostprocess;
    QJsonObject cameraModel;
    QString status;
    QString device;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
    QString rawPhotometricSourceMaskPath;
    QString rawGeometrySupportPath;
    QString rawAdaptiveGeometrySupportWeightPath;
    QString rawAdaptiveGeometryEffectiveViewCountPath;
    QString rawAdaptiveGeometryConflictRatioPath;
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
    bool effectiveNativeFinalDepthGrid = false;
    QJsonObject pixelDomainDiagnostics;
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
QString makeMvsDepthInputHash(const DepthGenConfig &config,
                              const std::vector<CameraView> &views,
                              const SparseCloud &sparse);

} // namespace xjw::mvs
