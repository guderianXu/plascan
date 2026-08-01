#pragma once

#include "MeshTypes.h"

#include <cstdint>
#include <functional>
#include <string>

namespace xjw::mesh
{

struct VisibilityOccupancyCarrierSubdivisionOptions
{
    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyCarrierSubdivisionStatistics
{
    std::uint64_t inputVertexCount = 0;
    std::uint64_t inputFaceCount = 0;
    std::uint64_t validatedFaceCount = 0;
    std::uint64_t uniqueInputEdgeCount = 0;
    std::uint64_t createdMidpointVertexCount = 0;
    std::uint64_t subdividedFaceCount = 0;
    std::uint64_t outputVertexCount = 0;
    std::uint64_t outputFaceCount = 0;
};

struct VisibilityOccupancyCarrierSubdivisionResult
{
    bool ok = false;
    bool cancelled = false;
    std::string errorMessage;
    TriMesh mesh;
    VisibilityOccupancyCarrierSubdivisionStatistics statistics;
};

/**
 * @brief Uniformly subdivides every carrier triangle into four triangles.
 *
 * Original vertices retain their indices. One midpoint is created for every
 * globally unique undirected edge in sorted edge order, making the output
 * deterministic. Face winding and topology are preserved for valid input.
 */
class VisibilityOccupancyCarrierSubdivider
{
public:
    static VisibilityOccupancyCarrierSubdivisionResult subdivide(
        const TriMesh &mesh,
        const VisibilityOccupancyCarrierSubdivisionOptions &options = {});
};

} // namespace xjw::mesh
