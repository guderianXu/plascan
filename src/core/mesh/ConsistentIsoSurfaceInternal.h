#pragma once

#include "ConsistentIsoSurfaceExtractor.h"
#include "IsoSurfaceTopology.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xjw::mesh::detail
{

inline constexpr int kIsoSurfaceCorners[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
inline constexpr int kIsoSurfaceEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}};
inline constexpr int kIsoSurfaceFaceCorners[6][4] = {
    {0, 1, 2, 3}, {4, 5, 6, 7},
    {0, 1, 5, 4}, {3, 2, 6, 7},
    {0, 3, 7, 4}, {1, 2, 6, 5}};
inline constexpr int kIsoSurfaceFaceEdges[6][4] = {
    {0, 1, 2, 3}, {4, 5, 6, 7},
    {0, 9, 4, 8}, {2, 10, 6, 11},
    {3, 11, 7, 8}, {1, 10, 5, 9}};

std::size_t checkedIsoSurfaceSampleCount(const std::array<int, 3> &cells);

std::size_t isoSurfaceSampleIndex(const std::array<int, 3> &cells,
                                  int x,
                                  int y,
                                  int z);

std::uint64_t isoSurfaceEdgeKey(const std::array<int, 3> &cells,
                                int cellX,
                                int cellY,
                                int cellZ,
                                int edge);

GridFaceKey isoSurfaceFaceKey(int cellX, int cellY, int cellZ, int face);

void addIsoSurfaceConnection(std::array<std::vector<int>, 12> *adjacency,
                             int first,
                             int second);

int countIsoSurfaceLoops(
    const std::array<bool, 12> &activeEdges,
    const std::array<std::vector<int>, 12> &adjacency);

bool classicIsoSurfaceTriangulationMatchesFaceGraph(
    int cubeIndex,
    const std::array<bool, 12> &activeEdges,
    const std::array<std::vector<int>, 12> &adjacency);

MeshVertex interpolateIsoSurfaceVertex(
    const std::array<MeshVertex, 8> &points,
    const std::array<float, 8> &values,
    int edge,
    float isoLevel);

void appendOrientedIsoSurfaceFace(
    TriMesh *mesh,
    int first,
    int second,
    int third,
    const std::array<double, 3> &gradient,
    ConsistentIsoSurfaceStatistics *statistics);

} // namespace xjw::mesh::detail
