#pragma once
// Private input, allocation, evidence and postprocess stage contracts.
#include "../DepthTsdfSurfaceBuilder.h"

#include "DepthProvenance.h"

#include "DepthAuxiliaryBridgeSelector.h"
#include "DepthFrameQualificationPolicy.h"
#include "DepthMeasuredSupportConnectivity.h"
#include "DepthTsdfNarrowBandActivation.h"
#include "AdaptiveTsdfOctree.h"
#include "ConsistentIsoSurfaceExtractor.h"
#include "DepthConsensusBiasPolicy.h"
#include "DepthFrameUtils.h"
#include "MvsWorkspaceManifest.h"
#include "DepthFusionFramePolicy.h"
#include "DepthMeshCompleteness.h"
#include "DepthImplicitFieldRegularizer.h"
#include "DepthTsdfRecoveryTransaction.h"
#include "DepthTsdfCellSheetRecovery.h"
#include "DepthVisibilityHistogram.h"
#include "MeshAcquisitionGapReport.h"
#include "MeshBoundaryAttribution.h"
#include "MeshColorizer.h"
#include "MeshFaceOrientation.h"
#include "MeshQuadricSimplifier.h"
#include "NativeMeshSimplifier.h"
#include "ProcessCpuTimer.h"
#include "MeshTopologyQuality.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "SparseTgvSolver.h"
#include "SurfaceReconstructorPostprocess.h"
#include "VisualHullFieldEvaluator.h"
#include "VisibilityOccupancySurfaceBuilder.h"
#include "VisibilityOccupancyBoundaryExtractor.h"
#include "VisibilityOccupancyDistanceField.h"
#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyTsdfCompletion.h"
#include "io/PathIO.h"

#include <QJsonArray>
#include <QFileInfo>
#include <QSet>
#include <QStringList>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <plapoint/mesh/marching_cubes.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <atomic>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace xjw::mesh::tsdf_detail
{

    constexpr std::uint64_t kBaseBytesPerSample = sizeof(float) * 5 + sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t);

    constexpr std::uint64_t kColorBytesPerSample = sizeof(float) * 4;

    struct MeshBoundaryTopology
    {
        int boundaryEdgeCount = 0;
        int danglingBoundaryVertexCount = 0;
        int nonManifoldEdgeCount = 0;
    };

    struct ComparableIsoSurfaceExtraction
    {
        bool ok = false;
        QString errorMessage;
        TriMesh mesh;
        std::uint64_t mc33SupportMaskedSampleCount = 0;
        std::uint64_t mc33RejectedUnsupportedCellFaceCount = 0;
        std::uint64_t isoSurfaceAmbiguousFaceCount = 0;
        std::uint64_t isoSurfaceTopologyAdjustedCellCount = 0;
        std::uint64_t isoSurfaceDeciderTieCount = 0;
        std::uint64_t isoSurfaceMultipleLoopCellCount = 0;
        std::uint64_t isoSurfaceEdgeVertexCacheHitCount = 0;
        std::uint64_t isoSurfaceEdgeVertexCacheMissCount = 0;
        std::uint64_t isoSurfaceInteriorLoopVertexCount = 0;
        std::uint64_t isoSurfaceRejectedDegenerateFaceCount = 0;
        std::uint64_t isoSurfaceUnresolvedCellCount = 0;
    };

    struct TriangleQualitySummary
    {
        int validFaceCount = 0;
        int sliverFaceCount = 0;
        double sliverRatio = 0.0;
    };

    struct VisibilityHoleProtectionResult
    {
        std::vector<std::uint8_t> protectedVertices;
        int consideredLoopCount = 0;
        int releasedLoopCount = 0;
        int rejectedSupportLoopCount = 0;
        int rejectedConflictLoopCount = 0;
    };

    struct HoleFillPatchEvidenceResult
    {
        bool attempted = false;
        bool accepted = false;
        int vertexSampleCount = 0;
        int faceCenterSampleCount = 0;
        int acceptedSampleCount = 0;
        int rejectedSupportSampleCount = 0;
        int rejectedConflictSampleCount = 0;
        int minimumSupportingViewCount = 0;
        int maximumConflictViewCount = 0;
    };

    struct DepthUncertaintyBandEstimate
    {
        std::uint64_t sampleCount = 0;
        float p90Voxels = 0.0f;
    };

    struct WeakBoundaryTipResult
    {
        int weakVertexCount = 0;
        int candidateFaceCount = 0;
        int trimmedFaceCount = 0;
    };
    bool checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result);
    std::uint64_t availablePhysicalMemoryBytes();
    MeshBoundaryTopology boundaryTopology(const TriMesh& mesh);
    int boundaryEdgeCount(const TriMesh& mesh);
    double meshSurfaceArea(const TriMesh& mesh);
    double meshBoundsDiagonal(const TriMesh& mesh);
    ComparableIsoSurfaceExtraction extractComparableIsoSurface(const std::array<float, 3>& boundsMin,
                                                               const std::array<float, 3>& boundsMax,
                                                               const std::array<int, 3>& cells,
                                                               const std::vector<float>& field,
                                                               const std::vector<std::uint8_t>& support,
                                                               bool useMc33,
                                                               bool requireSupportedSignChange,
                                                               bool useConsistentExtractor,
                                                               const std::function<bool()>& isCancelled);
    bool meshVerticesInsidePaddedLayout(const TriMesh& mesh, const DepthTsdfLayout& layout, float paddingVoxels = 2.0f);
    TriangleQualitySummary triangleQualitySummary(const TriMesh& mesh);
    std::vector<std::uint8_t> boundaryVertexMask(const TriMesh& mesh);
    std::vector<std::uint8_t> multiViewSilhouetteBoundaryVertices(const TriMesh& mesh,
                                                                  const QVector<DepthTsdfFrame>& frames,
                                                                  const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                                  int minimumViews,
                                                                  int bandPixels,
                                                                  float depthToleranceVoxels,
                                                                  float maximumVoxelSize,
                                                                  float minimumConfidence);
    std::pair<int, int> holeFillPatchSampleViewEvidence(const MeshVertex& sample,
                                                        const QVector<DepthTsdfFrame>& frames,
                                                        const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                        float absoluteDepthTolerance,
                                                        float minimumConfidence);
    HoleFillPatchEvidenceResult validateAddedHoleFillPatchEvidence(const TriMesh& candidate,
                                                                   std::size_t baselineVertexCount,
                                                                   std::size_t baselineFaceCount,
                                                                   const QVector<DepthTsdfFrame>& frames,
                                                                   const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                                   int minimumSupportingViews,
                                                                   int maximumConflictViews,
                                                                   float depthToleranceVoxels,
                                                                   float maximumVoxelSize,
                                                                   float minimumConfidence);
    VisibilityHoleProtectionResult
    visibilityConstrainedHoleProtection(const TriMesh& mesh,
                                        const QVector<DepthTsdfFrame>& frames,
                                        const QVector<cv::Mat>& effectiveDepthValidMasks,
                                        const std::vector<std::uint8_t>& silhouetteProtectedVertices,
                                        int maximumBoundaryEdges,
                                        int minimumSupportingViews,
                                        int maximumConflictViews,
                                        float depthToleranceVoxels,
                                        float strongSilhouetteRatio,
                                        float maximumVoxelSize,
                                        float minimumConfidence);
    QString frameArtifactError(const DepthFrameArtifact& artifact, const QString& reason);
    bool loadsAsAuxiliarySurfaceOnly(const DepthFrameArtifact& artifact);
    bool canLoadDepthFrameArtifact(const DepthFrameArtifact& artifact);
    DepthAuxiliaryBridgeNode bridgeNode(const DepthFrameArtifact& artifact, int frame_index);
    DepthAuxiliaryBridgeNode bridgeNode(const DepthTsdfFrame& frame, int frame_index);
    std::uint64_t estimatedResidentDepthFrameBytes(const DepthFrameArtifact& artifact);
    QVector<DepthFrameArtifact> selectMemoryBoundedDepthFrameArtifacts(const QVector<DepthFrameArtifact>& artifacts);
    bool loadFloatMatrix(const QString& path, cv::Mat* matrix, QString* reason);
    bool loadUnsignedShortMatrix(const QString& path, cv::Mat* matrix, QString* reason);
    bool loadMask(const QString& path, const cv::Size& size, cv::Mat* mask, QString* reason, bool allow_resize = false);
    bool
    loadByteMap(const QString& path, const cv::Size& size, cv::Mat* map, QString* reason, bool allow_resize = false);
    void integrateWeighted(float* value, float* weight, float observation, float observationWeight);
    std::size_t sampleIndex(const DepthTsdfLayout& layout, int x, int y, int z);
    std::vector<std::uint8_t>
    dilateSampleMask(const DepthTsdfLayout& layout, const std::vector<std::uint8_t>& source, int passes);
    int bitCount(std::uint16_t value);
    int bitCount(DepthGeometrySourceMask value);
    DepthUncertaintyBandEstimate estimateDepthUncertaintyBand(const QVector<DepthTsdfFrame>& frames,
                                                              const DepthTsdfOptions& options,
                                                              float maximum_voxel_size);
    bool volumeNormalAt(
        const DepthTsdfLayout& layout, const std::vector<float>& tsdf, int x, int y, int z, cv::Vec3f* normal);
    WeakBoundaryTipResult trimWeakBoundaryTips(TriMesh* mesh,
                                               const DepthTsdfLayout& layout,
                                               const std::vector<std::uint16_t>& support,
                                               const std::vector<std::uint8_t>& strongAdaptiveSupport,
                                               int minimum_support,
                                               int passes,
                                               bool enabled);
} // namespace xjw::mesh::tsdf_detail
namespace xjw::mesh
{
    std::vector<std::uint8_t> buildVisualHullOccupancy(const DepthTsdfLayout& layout,
                                                       const QVector<DepthTsdfFrame>& frames,
                                                       int minimumVisibleViews,
                                                       int allowedSilhouetteViolations,
                                                       const std::function<bool()>& isCancelled);
    std::uint64_t relaxVisualHullCompletionField(const DepthTsdfLayout& layout,
                                                 const std::vector<std::uint8_t>& occupied,
                                                 const std::vector<std::uint8_t>& completionMask,
                                                 const std::vector<std::uint8_t>& supported,
                                                 int iterations,
                                                 float lambda,
                                                 float maximumUpdate,
                                                 std::vector<float>* tsdf);
} // namespace xjw::mesh
