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
    bool enableSurfacePatchSupport = false;
    bool enableContourBandZeroCrossingSupport = false;
    bool collectZeroCrossingDiagnostics = false;
    bool enableGeometryZeroCrossingRecovery = false;
    int geometryZeroCrossingMinimumSupportedCorners = 2;
    int geometryZeroCrossingMinimumCellVotes = 2;
    float minimumSurfacePatchObservationWeight = 0.60f;
    int minimumSurfacePatchSourceCount = 2;
    int minimumSurfacePatchCoreNeighborCount = 3;
    int surfacePatchGrowthPasses = 1;
    float maximumSurfacePatchInverseDepthSpread = 0.015f;
    float maximumSurfacePatchNormalAngleDegrees = 20.0f;
    float maximumSurfacePatchAbsoluteTsdf = 0.45f;
    float minimumSurfacePatchWeightRatio = 0.10f;
    int minimumInputFrames = 3;
    int minimumDistinctCameraSupport = 2;
    int minimumComponentFaces = 64;
    float minimumComponentFaceRatio = 0.025f;
    bool enableSupportMaskFreeSpaceCarving = false;
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
    bool trimWeakBoundaryTips = false;
    int weakBoundaryTipTrimPasses = 1;
    bool calculateVertexColors = true;
    bool compensateColorExposure = false;
    bool coherentFacePrimaryViewColors = false;
    bool enableQuadricSimplification = false;
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
    std::uint64_t contourBandZeroCrossingRejectedNoSignPairCount = 0;
    bool effectiveZeroCrossingDiagnostics = false;
    bool effectiveGeometryZeroCrossingRecovery = false;
    std::uint64_t geometryZeroCrossingCandidateSampleCount = 0;
    std::uint64_t geometryZeroCrossingRecoveredSampleCount = 0;
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
    int effectiveDepthValidBoundaryErosionPixels = 0;
    bool effectiveGeometryVerifiedBoundaryRecovery = false;
    int effectiveMinimumBoundaryRecoveryGeometrySupport = 0;
    float effectiveMaximumBoundaryRecoveryInverseDepthSpread = 0.0f;
    std::uint64_t boundaryRecoveredDepthValidPixelCount = 0;
    int boundaryEdgeCountBefore = 0;
    int boundaryEdgeCountAfter = 0;
    int marchingCubesVertexCount = 0;
    int marchingCubesFaceCount = 0;
    std::int64_t marchingCubesElapsedMs = 0;
    std::int64_t meshCleanupElapsedMs = 0;
    std::int64_t meshSimplificationElapsedMs = 0;
    std::int64_t meshColorizationElapsedMs = 0;
    std::int64_t postIntegrationElapsedMs = 0;
    int componentFilteredFaceCount = 0;
    int preSimplificationFaceCount = 0;
    int postSimplificationFaceCount = 0;
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
    int topologyQualityBoundaryEdgeCount = 0;
    int topologyQualityNonManifoldEdgeCount = 0;
    int topologyQualityComponentCount = 0;
    int topologyQualityHighAspectFaceCount = 0;
    int topologyQualityExtremeAspectFaceCount = 0;
    double topologyQualityBoundaryEdgeRatio = 0.0;
    double topologyQualityLargestComponentFaceRatio = 0.0;
    double topologyQualityHighAspectFaceRatio = 0.0;
    double topologyQualityExtremeAspectFaceRatio = 0.0;
    bool topologyQualityStrictGatePassed = false;
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
