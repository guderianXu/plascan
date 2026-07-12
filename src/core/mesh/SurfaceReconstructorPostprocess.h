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
void removeSmallConnectedComponents(TriMesh *mesh, int minFaces);
int fillSmallBoundaryHoles(TriMesh *mesh, int maxBoundaryEdges);
void taubinSmooth(TriMesh *mesh, int iterations, float lambda);
void recomputeNormals(TriMesh *mesh);

} // namespace detail
} // namespace mesh
} // namespace xjw
