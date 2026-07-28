#pragma once

#include <array>
#include <cstdint>

namespace xjw::mesh
{

enum class GridAxis : std::uint8_t
{
    X = 0,
    Y = 1,
    Z = 2
};

struct GridFaceKey
{
    GridAxis axis = GridAxis::X;
    int x = 0;
    int y = 0;
    int z = 0;
};

struct IsoSurfaceFaceDecision
{
    bool ambiguous = false;
    bool connectEdge01And23 = true;
    bool usedTieBreak = false;
    double determinant = 0.0;
};

IsoSurfaceFaceDecision decideIsoSurfaceFace(
    const std::array<float, 4> &values,
    float isoLevel,
    const GridFaceKey &key,
    double relativeEpsilon = 1.0e-12);

std::uint64_t encodeGridFaceKey(const GridFaceKey &key,
                                const std::array<int, 3> &cells);

} // namespace xjw::mesh
