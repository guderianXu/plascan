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
void weldCoincidentVertices(TriMesh *mesh, float relativeTolerance);
void removeDegenerateFaces(TriMesh *mesh);
int compactReferencedVertices(TriMesh *mesh);
void removeSmallConnectedComponents(TriMesh *mesh,
                                    int minFaces,
                                    float minimumLargestComponentRatio = 0.0f);
int fillSmallBoundaryHoles(TriMesh *mesh,
                           int maxBoundaryEdges,
                           float maxBoundaryDiameter = 0.0f);
int smoothOpenBoundaryVertices(TriMesh *mesh,
                               int iterations,
                               float lambda,
                               float maximumDisplacement);
void taubinSmooth(TriMesh *mesh, int iterations, float lambda);
void recomputeNormals(TriMesh *mesh);

} // namespace detail
} // namespace mesh
} // namespace xjw
