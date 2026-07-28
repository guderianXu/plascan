#pragma once

#include "Camera.h"
#include "DepthMapMeshBuilder.h"
#include "MeshTypes.h"

#include <QJsonObject>
#include <QString>
#include <QVector>

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::mesh
{

struct DepthTsdfFrame
{
    int refIndex = -1;
    QString refImage;
    Camera camera;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat geometrySupportCount;
    cv::Mat geometrySourceMask;
    cv::Mat inverseDepthMean;
    cv::Mat inverseDepthRelativeSpread;
    cv::Mat crossViewRepairedMask;
    QVector<int> sourceIndices;
    cv::Mat depthValidMask;
    cv::Mat supportMask;
    cv::Mat colorBgr;
    float frameQualityWeight = 1.0f;
    bool auxiliarySurfaceOnly = false;
};

struct DepthTsdfOptions
{
    int resolution = 320;
    float truncationVoxels = 7.5f;
    float surfaceSupportBandVoxels = 0.0f;
    float minimumConfidence = 0.25f;
    float minimumVoxelWeight = 1.0f;
    float minimumSingleObservationWeight = 0.70f;
    float minimumGeometryVerifiedObservationWeight = 0.85f;
    int minimumGeometrySupportCount = 4;
    bool allowGeometryVerifiedSingleObservation = false;
    bool enableGeometrySingleViewNeighborhoodGuard = false;
    int minimumGeometrySingleViewNeighborCount = 2;
    int geometrySingleViewGrowthPasses = 2;
    float maximumGeometrySingleViewNeighborTsdfDelta = 0.35f;
    bool enableDiscontinuityAwareSampling = false;
    float maximumInterpolationRelativeDepthSpread = 0.02f;
    float maximumObservationInverseDepthSpread = 0.0f;
    bool allowInvalidNearestPixelRecovery = true;
    float maximumInvalidNearestPixelRecoveryInverseDepthSpread = 0.0f;
    bool enableCrossViewConsensusDepth = false;
    float maximumCrossViewConsensusInverseDepthSpread = 0.02f;
    bool crossViewConsensusContourBandOnly = false;
    bool enableRobustFrameQualityWeighting = false;
    float robustFrameQualityMinimumMultiplier = 0.35f;
    float robustFrameQualityMadFloor = 0.015f;
    float robustFrameQualityPenaltyOnset = 0.50f;
    float robustFrameQualityPenaltyStrength = 0.45f;
    bool enableRobustFrameQualityRejection = false;
    float robustFrameQualityRejectionSigma = 1.75f;
    float robustFrameQualityMaximumRejectedRatio = 0.25f;
    int robustFrameQualityMinimumRetainedFrames = 5;
    bool enableOrbitalFrameCoverageProtection = false;
    float maximumOrbitalAngularGapRatio = 2.0f;
    float validationOnlyFrameWeightMultiplier = 0.35f;
    float coverageProtectedFrameMinimumMultiplier = 0.20f;
    bool enableSurfacePatchSupport = false;
    bool enableContourBandZeroCrossingSupport = false;
    bool collectZeroCrossingDiagnostics = false;
    bool enableConsistentIsoSurfaceExtraction = false;
    bool enableMc33IsoSurfaceExtraction = false;
    bool mc33RequireSupportedSignChange = true;
    bool enableGeometryZeroCrossingRecovery = false;
    int geometryZeroCrossingMinimumSupportedCorners = 2;
    int geometryZeroCrossingMinimumCellVotes = 2;
    bool enableGeometryZeroCrossingCellSheets = false;
    int minimumGeometryZeroCrossingSheetCells = 3;
    int minimumGeometryZeroCrossingSheetAnchorCells = 2;
    float maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf = 1.0f;
    bool enableGlobalImplicitRegularization = false;
    int implicitRegularizationLevels = 2;
    int implicitRegularizationPassesPerLevel = 1;
    float implicitRegularizationSmoothness = 0.30f;
    float implicitRegularizationDataFidelity = 1.0f;
    float implicitRegularizationMaximumUpdate = 0.10f;
    float implicitRegularizationEdgeThreshold = 0.35f;
    bool implicitRegularizationRecoverAxialGaps = false;
    int implicitRegularizationMinimumBridgeAxes = 1;
    float implicitRegularizationMaximumBridgePredictionDelta = 0.20f;
    bool enableAdaptiveTgvRegularization = false;
    int adaptiveTgvMaximumMergeLevel = 8;
    float adaptiveTgvMinimumMergeAbsoluteField = 0.55f;
    float adaptiveTgvMaximumMergeFieldRange = 0.15f;
    float adaptiveTgvMaximumActiveAbsoluteField = 0.85f;
    int adaptiveTgvMaximumIterations = 120;
    int adaptiveTgvMinimumIterations = 20;
    float adaptiveTgvFirstOrderWeight = 0.12f;
    float adaptiveTgvSecondOrderWeight = 0.08f;
    float adaptiveTgvDataFidelity = 0.08f;
    float adaptiveTgvPrimalStep = 0.12f;
    float adaptiveTgvDualStep = 0.12f;
    float adaptiveTgvConvergenceTolerance = 1.0e-4f;
    bool adaptiveTgvUseGlobalVisibilityField = false;
    bool adaptiveTgvRecoverUnsupportedSamples = true;
    int adaptiveTgvRecoveryPasses = 2;
    int adaptiveTgvMinimumRecoveryNeighbors = 2;
    float adaptiveTgvMaximumRecoveryConflictRatio = 0.20f;
    float minimumSurfacePatchObservationWeight = 0.60f;
    int minimumSurfacePatchSourceCount = 2;
    int minimumSurfacePatchCoreNeighborCount = 3;
    int surfacePatchGrowthPasses = 1;
    float maximumSurfacePatchInverseDepthSpread = 0.015f;
    float maximumSurfacePatchNormalAngleDegrees = 20.0f;
    float maximumSurfacePatchAbsoluteTsdf = 0.45f;
    float maximumContourBandAbsoluteTsdf = 0.45f;
    float minimumSurfacePatchWeightRatio = 0.10f;
    int minimumInputFrames = 3;
    int minimumDistinctCameraSupport = 2;
    int minimumComponentFaces = 64;
    float minimumComponentFaceRatio = 0.025f;
    bool enableSupportMaskFreeSpaceCarving = false;
    bool enableSurfaceEvidenceFreeSpaceVeto = true;
    float supportMaskFreeSpaceWeight = 0.25f;
    int minimumSupportMaskFreeSpaceViews = 1;
    float maximumFreeSpaceVoxels = 36.0f;
    int depthValidBoundaryErosionPixels = 0;
    bool enableGeometryVerifiedBoundaryRecovery = false;
    int minimumBoundaryRecoveryGeometrySupport = 4;
    float maximumBoundaryRecoveryInverseDepthSpread = 0.01f;
    bool fillSmallBoundaryHoles = false;
    bool splitPinchedBoundaryVertices = true;
    int maximumHoleBoundaryEdges = 16;
    float maximumHoleDiameterVoxels = 4.0f;
    bool enableSilhouetteAwareFinalHoleFill = false;
    int finalHoleFillMaximumBoundaryEdges = 128;
    float finalHoleFillMaximumDiameterVoxels = 32.0f;
    float finalHoleFillMaximumFaceGrowthRatio = 0.10f;
    float finalHoleFillMaximumSliverRatio = 0.05f;
    bool enableVisibilityConstrainedFinalHoleFill = false;
    int visibilityHoleFillMinimumSupportingViews = 2;
    int visibilityHoleFillMaximumConflictViews = 0;
    float visibilityHoleFillDepthToleranceVoxels = 10.0f;
    float visibilityHoleFillStrongSilhouetteRatio = 0.35f;
    bool enableTinyBoundaryLoopCollapse = false;
    int tinyBoundaryLoopCollapseMaximumEdges = 8;
    float tinyBoundaryLoopCollapseMaximumDiameterVoxels = 3.0f;
    float tinyBoundaryLoopCollapseMaximumEdgeVoxels = 0.35f;
    int tinyBoundaryLoopCollapseMaximumPasses = 4;
    int boundarySmoothingIterations = 0;
    float boundarySmoothingLambda = 0.20f;
    float maximumBoundarySmoothingDisplacementVoxels = 0.35f;
    int surfaceDenoisingIterations = 0;
    float surfaceDenoisingLambda = 0.45f;
    float maximumSurfaceDenoisingDisplacementVoxels = 0.20f;
    float maximumSurfaceDenoisingNormalAngleDegrees = 30.0f;
    int surfaceDenoisingBoundaryProtectionRings = 1;
    bool enablePostSimplificationSurfaceDenoising = true;
    bool trimWeakBoundaryTips = false;
    int weakBoundaryTipTrimPasses = 1;
    bool calculateVertexColors = true;
    bool compensateColorExposure = false;
    bool coherentFacePrimaryViewColors = false;
    bool enableQuadricSimplification = false;
    bool enableOpenMeshSimplification = false;
    float openMeshMaximumNormalDeviationDegrees = 180.0f;
    float openMeshMaximumNormalFlippingDegrees = 75.0f;
    int openMeshSmoothingIterations = 2;
    float openMeshSmoothingMaximumDisplacementVoxels = 0.40f;
    float openMeshSmoothingFeatureAngleDegrees = 120.0f;
    int openMeshNotificationInterval = 4096;
    bool enableVoxelFallbackSimplification = true;
    bool enableVoxelFallbackQemPolish = false;
    int voxelFallbackMinimumProtectedBoundaryVertices = 1;
    float voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels = 0.0f;
    float voxelFallbackMaximumNormalClusterAngleDegrees = 180.0f;
    float voxelFallbackInitialClusterFactor = 1.0f;
    bool enableVoxelFallbackMultiViewSilhouetteProtection = true;
    int voxelFallbackMinimumSilhouetteViews = 2;
    int voxelFallbackSilhouetteBandPixels = 2;
    float voxelFallbackSilhouetteDepthToleranceVoxels = 8.0f;
    float voxelFallbackMaximumSliverRatio = 0.08f;
    bool enableTriangleQualityOptimization = false;
    int triangleQualityOptimizationMaximumPasses = 4;
    float triangleQualityMinimumAspectImprovementRatio = 0.01f;
    float triangleQualityMaximumFeatureAngleDegrees = 45.0f;
    float triangleQualityMaximumNormalDeviationDegrees = 35.0f;
    bool enableTriangleQualityTangentialRelaxation = true;
    int triangleQualityTangentialRelaxationPasses = 3;
    float triangleQualityTangentialRelaxationLambda = 0.45f;
    float triangleQualityTangentialMaximumDisplacementEdgeRatio = 0.20f;
    bool enableTriangleQualityIsotropicRemeshing = false;
    int triangleQualityIsotropicRemeshingPasses = 2;
    float triangleQualityIsotropicShortEdgeRatio = 0.25f;
    float triangleQualityIsotropicLongEdgeRatio = 2.0f;
    float triangleQualityIsotropicMaximumFaceGrowthRatio = 0.08f;
    float topologyQualityMaximumBoundaryEdgeRatio = 0.01f;
    float topologyQualityMaximumHighAspectFaceRatio = 0.02f;
    float topologyQualityMaximumExtremeAspectFaceRatio = 0.005f;
    int topologyQualityMaximumTopologicalComplexity = 128;
    float topologyQualityMaximumClosedGenus = 64.0f;
    bool enableDepthCompletenessDiagnostics = false;
    bool enforceDepthCompletenessGate = false;
    int depthCompletenessMaximumSamplesPerFrame = 6000;
    float depthCompletenessToleranceVoxels = 4.0f;
    float minimumDepthCompletenessP10Recall = 0.40f;
    float minimumDepthCompletenessMedianRecall = 0.75f;
    int simplifyTargetFaces = 0;
    int simplificationMaximumPasses = 96;
    float simplificationFeatureAngleDegrees = 65.0f;
    float simplificationMaximumNormalDeviationDegrees = 65.0f;
    int simplificationMinimumSharpEdgeEndpointDegree = 3;
    bool simplifySimpleOpenBoundaries = true;
    float maximumSimplificationBoundaryEdgeGrowthRatio = 0.10f;
    int workerCount = 0;
    std::uint64_t availableMemoryBytes = 0;
    std::function<bool()> isCancelled;
    std::function<void(const QString &, int)> progress;
};

struct DepthTsdfLayout
{
    bool ok = false;
    std::array<float, 3> boundsMin{};
    std::array<float, 3> boundsMax{};
    std::array<int, 3> cells{};
    std::array<float, 3> voxelSize{};
    std::uint64_t sampleCount = 0;
    std::uint64_t requiredBytes = 0;
};

struct DepthTsdfStatistics
{
    int inputFrameCount = 0;
    int acceptedFrameCount = 0;
    std::uint64_t integratedVoxelUpdates = 0;
    std::uint64_t rejectedProjectionCount = 0;
    std::uint64_t rejectedSupportMaskCount = 0;
    std::uint64_t supportMaskFreeSpaceUpdateCount = 0;
    std::uint64_t rejectedDepthValidCount = 0;
    std::uint64_t rejectedDepthCount = 0;
    std::uint64_t rejectedConfidenceCount = 0;
    std::uint64_t subpixelObservationCount = 0;
    std::uint64_t recoveredNeighborObservationCount = 0;
    std::uint64_t discontinuityRejectedCandidateCount = 0;
    std::uint64_t rejectedGeometryConsistencyCount = 0;
    std::uint64_t rejectedInvalidNearestPixelRecoveryCount = 0;
    std::uint64_t crossViewConsensusDepthObservationCount = 0;
    std::uint64_t crossViewConsensusContourBandPixelCount = 0;
    std::uint64_t supportedSampleCount = 0;
    std::uint64_t singleViewSupportedSampleCount = 0;
    std::uint64_t geometryVerifiedSingleViewSupportedSampleCount = 0;
    std::uint64_t geometrySingleViewNeighborhoodCandidateCount = 0;
    std::uint64_t geometrySingleViewNeighborhoodAcceptedCount = 0;
    std::uint64_t geometrySingleViewNeighborhoodRejectedCount = 0;
    std::uint64_t multiViewSupportedSampleCount = 0;
    std::uint64_t rejectedAccumulatedWeightCount = 0;
    std::uint64_t rejectedSingleObservationWeightCount = 0;
    std::uint64_t surfacePatchRecoveredSampleCount = 0;
    std::uint64_t surfacePatchConsideredSampleCount = 0;
    std::uint64_t surfacePatchRejectedWeightCount = 0;
    std::uint64_t surfacePatchRejectedNormalCount = 0;
    std::uint64_t surfacePatchRejectedSourceOverlapCount = 0;
    std::uint64_t surfacePatchRejectedDepthSpreadCount = 0;
    std::uint64_t surfacePatchRejectedFreeSpaceCount = 0;
    int surfacePatchCreatedComponentCount = 0;
    bool effectiveContourBandZeroCrossingSupport = false;
    std::uint64_t contourBandZeroCrossingConsideredSampleCount = 0;
    std::uint64_t contourBandZeroCrossingRecoveredSampleCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedNoContourCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedWeightCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedSourceOverlapCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedDepthSpreadCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedFreeSpaceCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedSurfaceWeightRatioCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedAbsoluteTsdfCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedNeighborhoodCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedGeometrySupportCount = 0;
    std::uint64_t contourBandZeroCrossingRejectedNoSignPairCount = 0;
    float effectiveMaximumContourBandAbsoluteTsdf = 0.0f;
    bool effectiveZeroCrossingDiagnostics = false;
    bool effectiveConsistentIsoSurfaceExtraction = false;
    bool effectiveMc33IsoSurfaceExtraction = false;
    bool effectiveMc33RequireSupportedSignChange = false;
    std::uint64_t mc33SupportMaskedSampleCount = 0;
    std::uint64_t mc33RejectedUnsupportedCellFaceCount = 0;
    int initialDegenerateRemovedFaceCount = 0;
    int componentFilterRemovedFaceCount = 0;
    bool effectiveGeometryZeroCrossingRecovery = false;
    std::uint64_t geometryZeroCrossingCandidateSampleCount = 0;
    std::uint64_t geometryZeroCrossingRecoveredSampleCount = 0;
    bool effectiveGeometryZeroCrossingCellSheets = false;
    std::uint64_t geometryZeroCrossingSheetCandidateCellCount = 0;
    std::uint64_t geometryZeroCrossingSheetAcceptedCellCount = 0;
    int geometryZeroCrossingSheetComponentCount = 0;
    int geometryZeroCrossingSheetAcceptedComponentCount = 0;
    int geometryZeroCrossingSheetRejectedSmallComponentCount = 0;
    int geometryZeroCrossingSheetRejectedAnchorComponentCount = 0;
    float effectiveMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf =
        0.0f;
    bool effectiveGlobalImplicitRegularization = false;
    int effectiveImplicitRegularizationLevels = 0;
    int effectiveImplicitRegularizationPassesPerLevel = 0;
    float effectiveImplicitRegularizationSmoothness = 0.0f;
    float effectiveImplicitRegularizationDataFidelity = 0.0f;
    float effectiveImplicitRegularizationMaximumUpdate = 0.0f;
    float effectiveImplicitRegularizationEdgeThreshold = 0.0f;
    bool effectiveImplicitRegularizationRecoverAxialGaps = false;
    int effectiveImplicitRegularizationMinimumBridgeAxes = 0;
    float effectiveImplicitRegularizationMaximumBridgePredictionDelta = 0.0f;
    std::uint64_t implicitRegularizationBridgeCandidateCount = 0;
    std::uint64_t implicitRegularizationRecoveredSampleCount = 0;
    std::uint64_t implicitRegularizationUpdateOperationCount = 0;
    double implicitRegularizationMeanAbsoluteUpdate = 0.0;
    float implicitRegularizationMaximumAbsoluteUpdate = 0.0f;
    std::int64_t implicitRegularizationElapsedMs = 0;
    bool effectiveAdaptiveTgvRegularization = false;
    bool effectiveAdaptiveTgvGlobalVisibilityField = false;
    std::uint64_t adaptiveTgvHistogramSampleCount = 0;
    std::uint64_t adaptiveTgvInputActiveSampleCount = 0;
    std::uint64_t adaptiveTgvLeafCount = 0;
    std::uint64_t adaptiveTgvMergedNodeCount = 0;
    std::uint64_t adaptiveTgvBalanceSplitCount = 0;
    std::uint64_t adaptiveTgvGlobalVisibilitySampleCount = 0;
    std::uint64_t adaptiveTgvRecoveredSampleCount = 0;
    float effectiveAdaptiveTgvMaximumActiveAbsoluteField = 0.0f;
    bool adaptiveTgvTwoToOneBalanced = false;
    int adaptiveTgvIterationCount = 0;
    double adaptiveTgvInitialMeanAbsoluteCurvature = 0.0;
    double adaptiveTgvFinalMeanAbsoluteCurvature = 0.0;
    double adaptiveTgvFinalMeanAbsoluteUpdate = 0.0;
    std::int64_t adaptiveTgvOctreeElapsedMs = 0;
    std::int64_t adaptiveTgvSolverElapsedMs = 0;
    std::uint64_t zeroCrossingObservedCellCount = 0;
    std::uint64_t zeroCrossingRawCandidateCellCount = 0;
    std::uint64_t zeroCrossingExtractableCellCount = 0;
    std::uint64_t zeroCrossingSuppressedBySupportCellCount = 0;
    std::uint64_t zeroCrossingPositiveOnlySupportedCellCount = 0;
    std::uint64_t zeroCrossingNegativeOnlySupportedCellCount = 0;
    std::uint64_t zeroCrossingPartiallySupportedCellCount = 0;
    std::uint64_t zeroCrossingFullyUnsupportedObservedCellCount = 0;
    float effectiveMinimumVoxelWeight = 0.0f;
    float effectiveMinimumSingleObservationWeight = 0.0f;
    float effectiveMinimumGeometryVerifiedObservationWeight = 0.0f;
    int effectiveMinimumGeometrySupportCount = 0;
    bool effectiveAllowGeometryVerifiedSingleObservation = false;
    bool effectiveGeometrySingleViewNeighborhoodGuard = false;
    int effectiveMinimumGeometrySingleViewNeighborCount = 0;
    int effectiveGeometrySingleViewGrowthPasses = 0;
    float effectiveMaximumGeometrySingleViewNeighborTsdfDelta = 0.0f;
    bool effectiveDiscontinuityAwareSampling = false;
    float effectiveMaximumInterpolationRelativeDepthSpread = 0.0f;
    float effectiveMaximumObservationInverseDepthSpread = 0.0f;
    bool effectiveAllowInvalidNearestPixelRecovery = true;
    float effectiveMaximumInvalidNearestPixelRecoveryInverseDepthSpread = 0.0f;
    bool effectiveCrossViewConsensusDepth = false;
    float effectiveMaximumCrossViewConsensusInverseDepthSpread = 0.0f;
    bool effectiveCrossViewConsensusContourBandOnly = false;
    bool effectiveRobustFrameQualityWeighting = false;
    int robustFrameQualityDownweightedFrameCount = 0;
    float robustFrameQualityMedian = 0.0f;
    float robustFrameQualityScale = 0.0f;
    float robustFrameQualityMinimumEffectiveWeight = 1.0f;
    bool effectiveRobustFrameQualityRejection = false;
    int robustFrameQualityRejectedFrameCount = 0;
    QVector<int> robustFrameQualityRejectedRefIndices;
    int auxiliarySurfaceOnlyFrameCount = 0;
    QVector<int> auxiliarySurfaceOnlyRefIndices;
    bool effectiveOrbitalFrameCoverageProtection = false;
    int orbitalCoverageProtectedFrameCount = 0;
    QVector<int> orbitalCoverageProtectedRefIndices;
    double orbitalMedianAngularSpacingDegrees = 0.0;
    double orbitalMaximumAngularGapDegrees = 0.0;
    double orbitalMaximumAngularGapRatio = 0.0;
    std::uint64_t auxiliaryOutsideSurfaceBandRejectedCount = 0;
    bool effectiveSurfacePatchSupport = false;
    float effectiveMinimumSurfacePatchObservationWeight = 0.0f;
    int effectiveMinimumSurfacePatchSourceCount = 0;
    int effectiveMinimumSurfacePatchCoreNeighborCount = 0;
    int effectiveSurfacePatchGrowthPasses = 0;
    int surfacePatchExecutedGrowthPassCount = 0;
    float effectiveMaximumSurfacePatchInverseDepthSpread = 0.0f;
    float effectiveMaximumSurfacePatchNormalAngleDegrees = 0.0f;
    float effectiveMaximumSurfacePatchAbsoluteTsdf = 0.0f;
    float effectiveMinimumSurfacePatchWeightRatio = 0.0f;
    int effectiveMinimumDistinctCameraSupport = 0;
    float effectiveTruncationVoxels = 0.0f;
    float effectiveSurfaceSupportBandVoxels = 0.0f;
    float effectiveMaximumFreeSpaceVoxels = 0.0f;
    int effectiveMinimumSupportMaskFreeSpaceViews = 0;
    bool effectiveSurfaceEvidenceFreeSpaceVeto = false;
    std::uint64_t supportMaskFreeSpaceSurfaceVetoCount = 0;
    int effectiveDepthValidBoundaryErosionPixels = 0;
    bool effectiveGeometryVerifiedBoundaryRecovery = false;
    int effectiveMinimumBoundaryRecoveryGeometrySupport = 0;
    float effectiveMaximumBoundaryRecoveryInverseDepthSpread = 0.0f;
    std::uint64_t boundaryRecoveredDepthValidPixelCount = 0;
    int boundaryEdgeCountBefore = 0;
    int boundaryEdgeCountAfter = 0;
    int marchingCubesVertexCount = 0;
    int marchingCubesFaceCount = 0;
    int marchingCubesBoundaryEdgeCount = 0;
    std::uint64_t isoSurfaceAmbiguousFaceCount = 0;
    std::uint64_t isoSurfaceTopologyAdjustedCellCount = 0;
    std::uint64_t isoSurfaceDeciderTieCount = 0;
    std::uint64_t isoSurfaceMultipleLoopCellCount = 0;
    std::uint64_t isoSurfaceEdgeVertexCacheHitCount = 0;
    std::uint64_t isoSurfaceEdgeVertexCacheMissCount = 0;
    std::uint64_t isoSurfaceInteriorLoopVertexCount = 0;
    std::uint64_t isoSurfaceRejectedDegenerateFaceCount = 0;
    std::uint64_t isoSurfaceUnresolvedCellCount = 0;
    int componentFilteredBoundaryEdgeCount = 0;
    int weakBoundaryTrimmedBoundaryEdgeCount = 0;
    int topologyCleanedBoundaryEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionNoObservationEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionInsufficientSourceEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionDepthSpreadRejectedEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionSurfaceWeightRejectedEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionAbsoluteTsdfRejectedEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionSupportGateRejectedEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionExtractionOrPostprocessEdgeCount = 0;
    std::uint64_t topologyCleanedAttributionUnclassifiedEdgeCount = 0;
    std::int64_t marchingCubesElapsedMs = 0;
    std::int64_t meshCleanupElapsedMs = 0;
    std::int64_t meshSimplificationElapsedMs = 0;
    std::int64_t meshColorizationElapsedMs = 0;
    std::int64_t postIntegrationElapsedMs = 0;
    int componentFilteredFaceCount = 0;
    int preSimplificationFaceCount = 0;
    int postSimplificationFaceCount = 0;
    int postSimplificationComponentFilterRemovedFaceCount = 0;
    int compactedUnusedVertexCount = 0;
    int removedDuplicateFaceCount = 0;
    int removedNonManifoldFaceCount = 0;
    int splitPinchedBoundaryVertexCount = 0;
    int filledBoundaryHoleCount = 0;
    int addedHoleFillFaceCount = 0;
    bool effectiveSilhouetteAwareFinalHoleFill = false;
    bool finalHoleFillAttempted = false;
    bool finalHoleFillAccepted = false;
    bool finalHoleFillTriangleQualityRejected = false;
    bool finalHoleFillPostSimplificationAttempted = false;
    bool finalHoleFillPostSimplificationAccepted = false;
    int finalHoleFillPostSimplificationInputFaceCount = 0;
    int finalHoleFillPostSimplificationOutputFaceCount = 0;
    int finalHoleFillPostSimplificationCollapsedEdgeCount = 0;
    int finalHoleFillPostSimplificationBoundaryEdgeCountBefore = 0;
    int finalHoleFillPostSimplificationBoundaryEdgeCountAfter = 0;
    bool effectiveVisibilityConstrainedFinalHoleFill = false;
    int visibilityHoleFillConsideredLoopCount = 0;
    int visibilityHoleFillReleasedLoopCount = 0;
    int visibilityHoleFillRejectedSupportLoopCount = 0;
    int visibilityHoleFillRejectedConflictLoopCount = 0;
    bool effectiveTinyBoundaryLoopCollapse = false;
    bool tinyBoundaryLoopCollapseAttempted = false;
    bool tinyBoundaryLoopCollapseAccepted = false;
    int tinyBoundaryLoopCollapsePassCount = 0;
    int tinyBoundaryLoopCollapsedEdgeCount = 0;
    int tinyBoundaryLoopBoundaryEdgeCountBefore = 0;
    int tinyBoundaryLoopBoundaryEdgeCountAfter = 0;
    int tinyBoundaryLoopNonManifoldEdgeCountBefore = 0;
    int tinyBoundaryLoopNonManifoldEdgeCountAfter = 0;
    double tinyBoundaryLoopHighAspectRatioBefore = 0.0;
    double tinyBoundaryLoopHighAspectRatioAfter = 0.0;
    int finalHoleFillProtectedSilhouetteVertexCount = 0;
    int finalHoleFillProtectedHoleCount = 0;
    int finalHoleFillFilledHoleCount = 0;
    int finalHoleFillAddedFaceCount = 0;
    int finalHoleFillBoundaryEdgeCountBefore = 0;
    int finalHoleFillBoundaryEdgeCountAfter = 0;
    int finalHoleFillNonManifoldEdgeCountBefore = 0;
    int finalHoleFillNonManifoldEdgeCountAfter = 0;
    double finalHoleFillSliverRatioBefore = 0.0;
    double finalHoleFillSliverRatioAfter = 0.0;
    bool residualMicroHoleFillAttempted = false;
    bool residualMicroHoleFillAccepted = false;
    int residualMicroHoleFillProtectedHoleCount = 0;
    int residualMicroHoleFillFilledHoleCount = 0;
    int residualMicroHoleFillAddedFaceCount = 0;
    int residualMicroHoleFillBoundaryEdgeCountBefore = 0;
    int residualMicroHoleFillBoundaryEdgeCountAfter = 0;
    int residualMicroHoleFillNonManifoldEdgeCountBefore = 0;
    int residualMicroHoleFillNonManifoldEdgeCountAfter = 0;
    double residualMicroHoleFillSliverRatioBefore = 0.0;
    double residualMicroHoleFillSliverRatioAfter = 0.0;
    int smoothedBoundaryVertexCount = 0;
    int smoothedSurfaceVertexCount = 0;
    int effectiveSurfaceDenoisingIterations = 0;
    float effectiveSurfaceDenoisingLambda = 0.0f;
    float effectiveMaximumSurfaceDenoisingDisplacementVoxels = 0.0f;
    float effectiveMaximumSurfaceDenoisingNormalAngleDegrees = 0.0f;
    int effectiveSurfaceDenoisingBoundaryProtectionRings = 0;
    bool effectivePostSimplificationSurfaceDenoising = false;
    bool postSimplificationSurfaceDenoisingAttempted = false;
    bool postSimplificationSurfaceDenoisingAccepted = false;
    int postSimplificationSmoothedSurfaceVertexCount = 0;
    int postSimplificationTangentialRelaxedVertexCount = 0;
    double postSimplificationHighAspectFaceRatioBefore = 0.0;
    double postSimplificationHighAspectFaceRatioAfter = 0.0;
    double postSimplificationExtremeAspectFaceRatioBefore = 0.0;
    double postSimplificationExtremeAspectFaceRatioAfter = 0.0;
    double postSimplificationNormalAngleMedianBefore = 0.0;
    double postSimplificationNormalAngleMedianAfter = 0.0;
    double postSimplificationNormalAngleP90Before = 0.0;
    double postSimplificationNormalAngleP90After = 0.0;
    double postSimplificationNormalAngleOver30RatioBefore = 0.0;
    double postSimplificationNormalAngleOver30RatioAfter = 0.0;
    int weakBoundaryTipVertexCount = 0;
    int candidateWeakBoundaryTipFaceCount = 0;
    int trimmedWeakBoundaryTipFaceCount = 0;
    std::uint64_t colorCandidateObservationCount = 0;
    std::uint64_t colorRejectedProjectionCount = 0;
    std::uint64_t colorRejectedMaskCount = 0;
    std::uint64_t colorRejectedDepthCount = 0;
    std::uint64_t colorRejectedVisibilityCount = 0;
    std::uint64_t colorRejectedViewAngleCount = 0;
    std::uint64_t colorRejectedOutlierCount = 0;
    int reliablyColoredVertexCount = 0;
    int bestViewFallbackColorVertexCount = 0;
    int propagatedColorVertexCount = 0;
    int fallbackColorVertexCount = 0;
    int cleanedColorSpeckleVertexCount = 0;
    bool effectiveColorExposureCompensation = false;
    bool effectiveCoherentFacePrimaryViewColors = false;
    int coherentPrimaryViewFaceCount = 0;
    int coherentPrimaryViewVertexCount = 0;
    bool openMeshSimplificationAttempted = false;
    bool effectiveOpenMeshSimplification = false;
    bool openMeshSimplificationAccepted = false;
    bool openMeshSimplificationReachedTarget = false;
    bool openMeshSimplificationCancelled = false;
    int openMeshSimplificationInputVertexCount = 0;
    int openMeshSimplificationInputFaceCount = 0;
    int openMeshSimplificationOutputVertexCount = 0;
    int openMeshSimplificationOutputFaceCount = 0;
    int openMeshSimplificationCollapsedVertexCount = 0;
    int openMeshSimplificationRejectedInputFaceCount = 0;
    int openMeshInconsistentSharedEdgeCountBefore = 0;
    int openMeshReorientedInputFaceCount = 0;
    int openMeshRemovedContradictoryFaceCount = 0;
    int openMeshOrientationConflictCount = 0;
    bool openMeshSmoothingApplied = false;
    int openMeshBoundaryEdgeCountBefore = 0;
    int openMeshBoundaryEdgeCountAfter = 0;
    int openMeshNonManifoldEdgeCountBefore = 0;
    int openMeshNonManifoldEdgeCountAfter = 0;
    QString openMeshSimplificationError;
    bool effectiveQuadricSimplification = false;
    bool quadricSimplificationAccepted = false;
    bool quadricSimplificationBoundarySafetyRejected = false;
    int quadricBoundaryEdgeCountBefore = 0;
    int quadricBoundaryEdgeCountAfter = 0;
    int quadricDanglingBoundaryVertexCountBefore = 0;
    int quadricDanglingBoundaryVertexCountAfter = 0;
    int quadricNonManifoldEdgeCountBefore = 0;
    int quadricNonManifoldEdgeCountAfter = 0;
    int requestedSimplifyTargetFaces = 0;
    int quadricCollapsedEdgeCount = 0;
    int quadricRejectedBoundaryEdgeCount = 0;
    int quadricRejectedFeatureEdgeCount = 0;
    int quadricRejectedTopologyEdgeCount = 0;
    int quadricRejectedFlipEdgeCount = 0;
    int quadricSimplifyPassCount = 0;
    bool quadricSimplifyReachedTarget = false;
    bool quadricSimplifyStoppedByStagnation = false;
    bool voxelFallbackAttempted = false;
    bool effectiveVoxelFallbackSimplification = true;
    bool voxelFallbackAccepted = false;
    bool voxelFallbackPreservedOpenBoundaries = false;
    int voxelFallbackInputFaceCount = 0;
    int voxelFallbackOutputFaceCount = 0;
    int voxelFallbackBoundaryEdgeCountBefore = 0;
    int voxelFallbackBoundaryEdgeCountAfter = 0;
    int voxelFallbackNonManifoldEdgeCountAfter = 0;
    bool effectiveVoxelFallbackQemPolish = false;
    int effectiveVoxelFallbackMinimumProtectedBoundaryVertices = 1;
    float effectiveVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels = 0.0f;
    float effectiveVoxelFallbackMaximumNormalClusterAngleDegrees = 180.0f;
    float effectiveVoxelFallbackInitialClusterFactor = 1.0f;
    bool effectiveVoxelFallbackMultiViewSilhouetteProtection = false;
    int effectiveVoxelFallbackMinimumSilhouetteViews = 0;
    int effectiveVoxelFallbackSilhouetteBandPixels = 0;
    float effectiveVoxelFallbackSilhouetteDepthToleranceVoxels = 0.0f;
    int voxelFallbackProtectedSilhouetteVertexCount = 0;
    int voxelFallbackSliverFaceCountBefore = 0;
    int voxelFallbackSliverFaceCountAfter = 0;
    double voxelFallbackSliverRatioBefore = 0.0;
    double voxelFallbackSliverRatioAfter = 0.0;
    bool voxelFallbackTriangleQualityRejected = false;
    bool voxelFallbackQemPolishAttempted = false;
    bool voxelFallbackQemPolishAccepted = false;
    int voxelFallbackQemPolishInputFaceCount = 0;
    int voxelFallbackQemPolishOutputFaceCount = 0;
    int voxelFallbackQemPolishCollapsedEdgeCount = 0;
    bool effectiveTriangleQualityOptimization = false;
    bool triangleQualityOptimizationAttempted = false;
    bool triangleQualityOptimizationAccepted = false;
    int triangleQualityOptimizationPassCount = 0;
    int triangleQualityOptimizationFlippedEdgeCount = 0;
    int triangleQualityTangentialRelaxationPassCount = 0;
    int triangleQualityTangentialRelaxedVertexCount = 0;
    int triangleQualityIsotropicRemeshingPassCount = 0;
    int triangleQualityIsotropicCollapsedEdgeCount = 0;
    int triangleQualityIsotropicSplitEdgeCount = 0;
    int triangleQualityOptimizationInputFaceCount = 0;
    int triangleQualityOptimizationOutputFaceCount = 0;
    double triangleQualityHighAspectFaceRatioBefore = 0.0;
    double triangleQualityHighAspectFaceRatioAfter = 0.0;
    double triangleQualityExtremeAspectFaceRatioBefore = 0.0;
    double triangleQualityExtremeAspectFaceRatioAfter = 0.0;
    int topologyQualityUniqueEdgeCount = 0;
    int topologyQualityReferencedVertexCount = 0;
    int topologyQualityEulerCharacteristic = 0;
    int topologyQualityTopologicalComplexity = 0;
    double topologyQualityClosedGenusEstimate = 0.0;
    bool topologyQualityClosedTopologyEvaluated = false;
    int topologyQualityBoundaryEdgeCount = 0;
    int topologyQualityNonManifoldEdgeCount = 0;
    int topologyQualityComponentCount = 0;
    int topologyQualityHighAspectFaceCount = 0;
    int topologyQualityExtremeAspectFaceCount = 0;
    double topologyQualityBoundaryEdgeRatio = 0.0;
    double topologyQualityLargestComponentFaceRatio = 0.0;
    double topologyQualityHighAspectFaceRatio = 0.0;
    double topologyQualityExtremeAspectFaceRatio = 0.0;
    int topologyQualityAdjacentFacePairCount = 0;
    double topologyQualityAdjacentNormalAngleMedianDegrees = 0.0;
    double topologyQualityAdjacentNormalAngleP90Degrees = 0.0;
    double topologyQualityAdjacentNormalAngleOver30Ratio = 0.0;
    bool topologyQualityStrictGatePassed = false;
    bool effectiveDepthCompletenessDiagnostics = false;
    bool effectiveDepthCompletenessGateEnforcement = false;
    bool depthCompletenessAvailable = false;
    bool depthCompletenessGatePassed = false;
    double depthCompletenessTolerance = 0.0;
    std::uint64_t depthCompletenessSampledPointCount = 0;
    std::uint64_t depthCompletenessExplainedPointCount = 0;
    double depthCompletenessAggregateRecall = 0.0;
    double depthCompletenessMinimumFrameRecall = 0.0;
    double depthCompletenessP10FrameRecall = 0.0;
    double depthCompletenessMedianFrameRecall = 0.0;
    QVector<int> depthCompletenessRefIndices;
    QVector<double> depthCompletenessFrameRecalls;
    std::uint64_t boundaryAttributionEdgeCount = 0;
    std::uint64_t boundaryAttributionNoObservationEdgeCount = 0;
    std::uint64_t boundaryAttributionInsufficientSourceEdgeCount = 0;
    std::uint64_t boundaryAttributionDepthSpreadRejectedEdgeCount = 0;
    std::uint64_t boundaryAttributionSurfaceWeightRejectedEdgeCount = 0;
    std::uint64_t boundaryAttributionAbsoluteTsdfRejectedEdgeCount = 0;
    std::uint64_t boundaryAttributionSupportGateRejectedEdgeCount = 0;
    std::uint64_t boundaryAttributionExtractionOrPostprocessEdgeCount = 0;
    std::uint64_t boundaryAttributionUnclassifiedEdgeCount = 0;
    int vertexCount = 0;
    int faceCount = 0;
    int componentCount = 0;
    double largestComponentFaceRatio = 0.0;
    std::vector<std::size_t> componentFaceCounts;
    std::vector<MeshConnectivityStats::Component> components;
};

struct DepthTsdfResult
{
    bool ok = false;
    QString errorMessage;
    DepthTsdfLayout layout;
    DepthTsdfStatistics statistics;
    TriMesh mesh;
    TriMesh boundaryAttributionDebugMesh;
};

struct DepthTsdfFrameLoadResult
{
    bool ok = false;
    QString errorMessage;
    QVector<DepthTsdfFrame> frames;
};

struct DepthTsdfBoundsResult
{
    bool ok = false;
    QString errorMessage;
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
    std::uint64_t sampleCount = 0;
};

enum class DepthTsdfObservationFailure
{
    None,
    Projection,
    SupportMask,
    DepthValid,
    Depth,
    Confidence,
    GeometryConsistency
};

struct DepthTsdfObservationSample
{
    bool valid = false;
    float depth = 0.0f;
    float confidence = 0.0f;
    std::uint16_t geometrySupportCount = 0;
    std::uint16_t geometrySourceMask = 0;
    float inverseDepthRelativeSpread = 0.0f;
    int contributingPixelCount = 0;
    int discontinuityRejectedPixelCount = 0;
    bool recoveredFromInvalidNearestPixel = false;
    bool rejectedInvalidNearestPixelRecovery = false;
    bool usedCrossViewConsensusDepth = false;
    DepthTsdfObservationFailure failure = DepthTsdfObservationFailure::Projection;
};

struct DepthTsdfZeroCrossingStatistics
{
    std::uint64_t observedCellCount = 0;
    std::uint64_t rawCandidateCellCount = 0;
    std::uint64_t extractableCellCount = 0;
    std::uint64_t suppressedBySupportCellCount = 0;
    std::uint64_t positiveOnlySupportedCellCount = 0;
    std::uint64_t negativeOnlySupportedCellCount = 0;
    std::uint64_t partiallySupportedCellCount = 0;
    std::uint64_t fullyUnsupportedObservedCellCount = 0;
};

struct DepthTsdfZeroCrossingRecoveryStatistics
{
    std::uint64_t candidateSampleCount = 0;
    std::uint64_t recoveredSampleCount = 0;
    std::uint64_t candidateCellCount = 0;
    std::uint64_t acceptedCellCount = 0;
    int componentCount = 0;
    int acceptedComponentCount = 0;
    int rejectedSmallComponentCount = 0;
    int rejectedAnchorComponentCount = 0;
};

class DepthTsdfSurfaceBuilder
{
public:
    static DepthTsdfLayout makeLayout(const std::array<float, 3> &boundsMin,
                                      const std::array<float, 3> &boundsMax,
                                      int resolution,
                                      bool includeColor);
    static DepthTsdfResult validateAllocation(const std::array<float, 3> &boundsMin,
                                              const std::array<float, 3> &boundsMax,
                                              const DepthTsdfOptions &options);
    static DepthTsdfFrameLoadResult loadFrames(const QVector<DepthFrameArtifact> &artifacts);
    static DepthTsdfBoundsResult estimateBounds(const QVector<DepthTsdfFrame> &frames);
    static DepthTsdfObservationSample sampleObservation(
        const DepthTsdfFrame &frame,
        const cv::Mat &effectiveDepthValidMask,
        const cv::Point2d &pixel,
        float minimumConfidence,
        bool discontinuityAware,
        float maximumRelativeDepthSpread,
        float maximumObservationInverseDepthSpread = 0.0f,
        bool allowInvalidNearestPixelRecovery = true,
        float maximumInvalidNearestPixelRecoveryInverseDepthSpread = 0.0f,
        bool enableCrossViewConsensusDepth = false,
        float maximumCrossViewConsensusInverseDepthSpread = 0.02f,
        const cv::Mat &crossViewConsensusMask = cv::Mat());
    static bool isSampleSupported(float accumulatedWeight,
                                  int distinctSupportCount,
                                  float maximumObservationWeight,
                                  const DepthTsdfOptions &options,
                                  bool *singleView = nullptr,
                                  bool *multiView = nullptr,
                                  int maximumGeometrySupportCount = 0,
                                  bool *geometryVerifiedSingleView = nullptr);
    static QVector<float> robustFrameQualityWeights(
        const QVector<float> &rawWeights,
        float minimumMultiplier = 0.35f,
        float madFloor = 0.015f,
        float penaltyOnset = 0.50f,
        float penaltyStrength = 0.45f,
        float *median = nullptr,
        float *scale = nullptr);
    static int growGeometryVerifiedSingleViewSamples(
        const DepthTsdfLayout &layout,
        const std::vector<float> &tsdf,
        const std::vector<std::size_t> &candidateIndices,
        int minimumNeighborCount,
        int passes,
        float maximumTsdfDelta,
        std::vector<std::uint8_t> *supported);
    static DepthTsdfZeroCrossingStatistics analyzeZeroCrossings(
        const DepthTsdfLayout &layout,
        const std::vector<float> &tsdf,
        const std::vector<float> &weight,
        const std::vector<std::uint8_t> &supported);
    static DepthTsdfZeroCrossingRecoveryStatistics
        recoverGeometryVerifiedZeroCrossingSamples(
            const DepthTsdfLayout &layout,
            const std::vector<float> &tsdf,
            const std::vector<float> &weight,
            const std::vector<std::uint16_t> &geometrySourceMask,
            const std::vector<std::uint8_t> &eligible,
            int minimumSupportedCorners,
            int minimumCellVotes,
            std::vector<std::uint8_t> *supported);
    static bool shouldTrimWeakBoundaryFace(int boundaryEdgeCount,
                                           int weakVertexCount);
    static bool shouldAcceptQuadricSimplification(
        int inputFaceCount,
        int outputFaceCount,
        int boundaryEdgeCountBefore,
        int boundaryEdgeCountAfter,
        float maximumBoundaryEdgeGrowthRatio);
    static DepthTsdfResult build(const QVector<DepthTsdfFrame> &frames,
                                 const DepthTsdfOptions &options);
    static QJsonObject statisticsToJson(const DepthTsdfResult &result);
};

} // namespace xjw::mesh
