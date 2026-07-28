#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct Mc33IsoSurfaceOptions
{
    float isoLevel = 0.0f;
    bool requireSupportedSignChange = false;
    std::function<bool()> isCancelled;
};

struct Mc33IsoSurfaceStatistics
{
    std::uint64_t inputSampleCount = 0;
    std::uint64_t supportMaskedSampleCount = 0;
    std::uint64_t rejectedUnsupportedCellFaceCount = 0;
    std::uint64_t outputVertexCount = 0;
    std::uint64_t outputFaceCount = 0;
};

struct Mc33IsoSurfaceResult
{
    bool ok = false;
    bool cancelled = false;
    std::string errorMessage;
    TriMesh mesh;
    Mc33IsoSurfaceStatistics statistics;
};

class Mc33IsoSurfaceExtractor
{
public:
    static bool isAvailable();

    static Mc33IsoSurfaceResult extract(
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::array<int, 3> &cells,
        const std::vector<float> &field,
        const std::vector<std::uint8_t> &extractionSupport = {},
        const Mc33IsoSurfaceOptions &options = {});
};

} // namespace xjw::mesh
