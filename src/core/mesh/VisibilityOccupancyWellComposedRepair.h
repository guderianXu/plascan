#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyWellComposedRepairOptions
{
    int maximumPasses = 8;
    std::size_t maximumFilledSampleCount = 1024;
    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyWellComposedRepairStatistics
{
    std::uint64_t sampleCount = 0;
    std::uint64_t filledSampleCount = 0;
    std::uint64_t protectedFillRefusalCount = 0;
    std::uint64_t maximumFillRefusalCount = 0;
    std::uint64_t protectedExteriorSampleCountBefore = 0;
    std::uint64_t protectedExteriorSampleCountAfter = 0;
    int attemptedPassCount = 0;
    int acceptedPassCount = 0;
    int rolledBackEulerPassCount = 0;
    int rolledBackProtectedReachabilityPassCount = 0;
    std::uint64_t rejectedEulerSampleCount = 0;
    std::uint64_t rejectedProtectedReachabilitySampleCount = 0;
    int bodyEulerBefore = 0;
    int bodyEulerAfter = 0;
    std::uint64_t edgeCheckerboardCountBefore = 0;
    std::uint64_t vertexOccupiedComponentDefectCountBefore = 0;
    std::uint64_t vertexEmptyComponentDefectCountBefore = 0;
    std::uint64_t remainingEdgeCheckerboardCount = 0;
    std::uint64_t remainingVertexOccupiedComponentDefectCount = 0;
    std::uint64_t remainingVertexEmptyComponentDefectCount = 0;
};

struct VisibilityOccupancyWellComposedRepairResult
{
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::vector<std::uint8_t> occupied;
    VisibilityOccupancyWellComposedRepairStatistics statistics;
};

/**
 * @brief Repairs local cubical configurations that do not have manifold links.
 *
 * Only empty, unprotected samples are filled. Each accepted pass preserves the
 * cubical-body Euler characteristic monotonically and keeps every initially
 * exterior-reachable protected empty sample connected to the exterior.
 */
class VisibilityOccupancyWellComposedRepair
{
public:
    static VisibilityOccupancyWellComposedRepairResult repair(
        const std::array<int, 3> &dimensions,
        const std::vector<std::uint8_t> &occupied,
        const std::vector<std::uint8_t> &protectedEmpty,
        const VisibilityOccupancyWellComposedRepairOptions &options = {});
};

} // namespace xjw::mesh
