#pragma once

#include "MeshTypes.h"

namespace xjw
{
namespace mesh
{
namespace detail
{

void simplifyVoxelMeshAdaptive(TriMesh *mesh,
                               const ReconstructionConfig &config,
                               float voxelStep);
void simplifyVoxelMeshAdaptive(TriMesh *mesh,
                               const ReconstructionConfig &config,
                               float voxelStep,
                               bool preserveOpenBoundaries,
                               int minimumProtectedBoundaryVertices = 1,
                               float maximumCollapsibleBoundaryDiameter = 0.0f,
                               float maximumNormalClusterAngleDegrees = 180.0f,
                               const std::vector<std::uint8_t> *
                                   protectedBoundaryVertices = nullptr);
void weldCoincidentVertices(TriMesh *mesh, float relativeTolerance);
void removeDegenerateFaces(TriMesh *mesh);
int compactReferencedVertices(TriMesh *mesh);
void removeSmallConnectedComponents(TriMesh *mesh,
                                    int minFaces,
                                    float minimumLargestComponentRatio = 0.0f);
int removeDuplicateFaces(TriMesh *mesh);
int removeNonManifoldFaces(TriMesh *mesh);
int splitPinchedBoundaryVertices(TriMesh *mesh);
int fillSmallBoundaryHoles(TriMesh *mesh,
                           int maxBoundaryEdges,
                           float maxBoundaryDiameter = 0.0f,
                           const std::vector<std::uint8_t> *
                               protectedBoundaryVertices = nullptr,
                           int *protectedHoleCount = nullptr,
                           bool useQualityTriangulation = false);
int collapseTinyBoundaryLoops(TriMesh *mesh,
                              int maximumBoundaryEdges,
                              float maximumBoundaryDiameter,
                              float maximumCollapseEdgeLength);
int smoothOpenBoundaryVertices(TriMesh *mesh,
                               int iterations,
                               float lambda,
                               float maximumDisplacement);
int smoothSurfaceVerticesNormalAware(TriMesh *mesh,
                                     int iterations,
                                     float lambda,
                                     float maximumDisplacement,
                                     float maximumNormalAngleDegrees,
                                     int boundaryProtectionRings);
void taubinSmooth(TriMesh *mesh, int iterations, float lambda);
void recomputeNormals(TriMesh *mesh);

} // namespace detail
} // namespace mesh
} // namespace xjw
