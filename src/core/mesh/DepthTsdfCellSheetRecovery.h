#pragma once

#include "DepthTsdfSurfaceBuilder.h"

#include <cstdint>
#include <vector>

namespace xjw::mesh
{

DepthTsdfZeroCrossingRecoveryStatistics
recoverGeometryVerifiedZeroCrossingCellSheets(
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint8_t> &eligible,
    int minimumSupportedCorners,
    int minimumSheetCells,
    int minimumSheetAnchorCells,
    float maximumSingleVoteAbsoluteTsdf,
    std::vector<std::uint8_t> *supported,
    bool requireBoundaryReduction = false);

} // namespace xjw::mesh
