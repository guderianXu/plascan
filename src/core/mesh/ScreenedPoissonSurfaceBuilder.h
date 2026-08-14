#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct ScreenedPoissonOptions
{
    int depth = 9;
    float pointWeight = 4.0f;
    float samplesPerNode = 1.5f;
    float scale = 1.1f;
    int solverIterations = 8;
    float cgSolverAccuracy = 1.0e-3f;
    bool linearFit = false;
    bool forceManifold = true;
    bool verbose = false;
};

struct ScreenedPoissonStatistics
{
    std::size_t inputSampleCount = 0;
    std::size_t acceptedSampleCount = 0;
    std::size_t rejectedSampleCount = 0;
    std::size_t outputPolygonCount = 0;
    std::size_t skippedPolygonCount = 0;
    std::size_t outputVertexCount = 0;
    std::size_t outputTriangleCount = 0;
};

struct ScreenedPoissonResult
{
    bool ok = false;
    std::string error;
    TriMesh mesh;
    ScreenedPoissonStatistics statistics;
};

/**
 * @brief Reconstructs a closed triangle mesh with the official Screened
 * Poisson implementation from oriented points.
 *
 * Invalid samples and zero-length normals are discarded. The point weight
 * must remain strictly positive so that this adapter cannot silently fall
 * back to unscreened Poisson reconstruction.
 */
class ScreenedPoissonSurfaceBuilder
{
public:
    static ScreenedPoissonResult build(
        const std::vector<std::array<float, 3>> &points,
        const std::vector<std::array<float, 3>> &normals,
        const ScreenedPoissonOptions &options = {});
};

} // namespace xjw::mesh
