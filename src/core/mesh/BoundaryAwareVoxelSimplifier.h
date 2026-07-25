#pragma once

#include "MeshTypes.h"

namespace xjw
{
namespace mesh
{
namespace detail
{

TriMesh voxelClusterSimplifyPreservingOpenBoundaries(const TriMesh &mesh,
                                                      float clusterSize,
                                                      int minimumProtectedBoundaryVertices = 1,
                                                      float maximumCollapsibleBoundaryDiameter = 0.0f,
                                                      float maximumNormalClusterAngleDegrees = 180.0f,
                                                      const std::vector<std::uint8_t> *
                                                          protectedBoundaryVertices = nullptr,
                                                      std::vector<std::uint8_t> *
                                                          outputProtectedBoundaryVertices = nullptr);

} // namespace detail
} // namespace mesh
} // namespace xjw
