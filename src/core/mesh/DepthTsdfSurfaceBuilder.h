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
    bool enableSurfacePatchSupport = false;
    int minimumSurfacePatchSourceCount = 2;
    int minimumSurfacePatchCoreNeighborCount = 3;
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
    bool fillSmallBoundaryHoles = false;
    int maximumHoleBoundaryEdges = 16;
    float maximumHoleDiameterVoxels = 4.0f;
    int boundarySmoothingIterations = 0;
    float boundarySmoothingLambda = 0.20f;
    float maximumBoundarySmoothingDisplacementVoxels = 0.35f;
    bool trimWeakBoundaryTips = false;
    int weakBoundaryTipTrimPasses = 1;
    bool calculateVertexColors = true;
    bool compensateColorExposure = false;
    bool coherentFacePrimaryViewColors = false;
    bool enableQuadricSimplification = false;
    int simplifyTargetFaces = 0;
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
    bool effectiveSurfacePatchSupport = false;
    int effectiveMinimumSurfacePatchSourceCount = 0;
    int effectiveMinimumSurfacePatchCoreNeighborCount = 0;
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
    int boundaryEdgeCountBefore = 0;
    int boundaryEdgeCountAfter = 0;
    int compactedUnusedVertexCount = 0;
    int filledBoundaryHoleCount = 0;
    int addedHoleFillFaceCount = 0;
    int smoothedBoundaryVertexCount = 0;
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
    int requestedSimplifyTargetFaces = 0;
    int quadricCollapsedEdgeCount = 0;
    int quadricRejectedBoundaryEdgeCount = 0;
    int quadricRejectedFeatureEdgeCount = 0;
    int quadricRejectedTopologyEdgeCount = 0;
    int quadricRejectedFlipEdgeCount = 0;
    int quadricSimplifyPassCount = 0;
    bool quadricSimplifyReachedTarget = false;
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
        float maximumCrossViewConsensusInverseDepthSpread = 0.02f);
    static bool isSampleSupported(float accumulatedWeight,
                                  int distinctSupportCount,
                                  float maximumObservationWeight,
                                  const DepthTsdfOptions &options,
                                  bool *singleView = nullptr,
                                  bool *multiView = nullptr,
                                  int maximumGeometrySupportCount = 0,
                                  bool *geometryVerifiedSingleView = nullptr);
    static int growGeometryVerifiedSingleViewSamples(
        const DepthTsdfLayout &layout,
        const std::vector<float> &tsdf,
        const std::vector<std::size_t> &candidateIndices,
        int minimumNeighborCount,
        int passes,
        float maximumTsdfDelta,
        std::vector<std::uint8_t> *supported);
    static bool shouldTrimWeakBoundaryFace(int boundaryEdgeCount,
                                           int weakVertexCount);
    static DepthTsdfResult build(const QVector<DepthTsdfFrame> &frames,
                                 const DepthTsdfOptions &options);
    static QJsonObject statisticsToJson(const DepthTsdfResult &result);
};

} // namespace xjw::mesh
