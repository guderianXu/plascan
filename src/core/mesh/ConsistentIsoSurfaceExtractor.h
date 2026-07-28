#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct ConsistentIsoSurfaceOptions
{
    float isoLevel = 0.0f;
    double ambiguityEpsilon = 1.0e-12;
    std::function<bool()> isCancelled;
    std::function<void(double)> progress;
};

struct ConsistentIsoSurfaceStatistics
{
    std::uint64_t visitedCellCount = 0;
    std::uint64_t activeCellCount = 0;
    std::uint64_t uniqueAmbiguousFaceCount = 0;
    std::uint64_t topologyAdjustedCellCount = 0;
    std::uint64_t deciderTieCount = 0;
    std::uint64_t multipleLoopCellCount = 0;
    std::uint64_t edgeVertexCacheHitCount = 0;
    std::uint64_t edgeVertexCacheMissCount = 0;
    std::uint64_t interiorLoopVertexCount = 0;
    std::uint64_t rejectedDegenerateFaceCount = 0;
    std::uint64_t unresolvedCellCount = 0;
};

struct ConsistentIsoSurfaceResult
{
    bool ok = false;
    bool cancelled = false;
    std::string errorMessage;
    TriMesh mesh;
    ConsistentIsoSurfaceStatistics statistics;
};

class ConsistentIsoSurfaceExtractor
{
public:
    static ConsistentIsoSurfaceResult extract(
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::array<int, 3> &cells,
        const std::vector<float> &field,
        const std::vector<std::uint8_t> &extractionSupport = {},
        const ConsistentIsoSurfaceOptions &options = {});
};

} // namespace xjw::mesh
