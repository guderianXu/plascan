#pragma once

#include "PlanetaryLineScanBundleAdjustInternal.h"

#include <plamatrix/optimization/block_schur.h>

#include <vector>

namespace xjw
{
namespace lidar
{
namespace detail
{
namespace plamatrix_linescan
{

std::vector<int> makeLaserBlocks(
    const PlanetaryLineScanBaWorkingSet &workingSet);

double evaluateObjective(const PlanetaryLineScanBaWorkingSet &workingSet,
                         const PlanetaryLineScanBaOptions &options);

plamatrix::BlockNormalEquations<double> buildEquations(
    const PlanetaryLineScanBaWorkingSet &workingSet,
    const PlanetaryLineScanBaOptions &options,
    const std::vector<int> &laserBlocks);

double maximumStepNorm(const std::vector<double> &primaryStep,
                       const std::vector<double> &eliminatedStep);

void applyStep(const std::vector<int> &laserBlocks,
               const std::vector<double> &primaryStep,
               const std::vector<double> &eliminatedStep,
               PlanetaryLineScanBaWorkingSet *workingSet);

} // namespace plamatrix_linescan
} // namespace detail
} // namespace lidar
} // namespace xjw
