#pragma once

#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/optimization/block_schur.h>

namespace xjw::detail::plamatrix_ba
{

struct OptimizationState
{
    std::vector<FramePinholeCamera> cameras;
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<double, 3>> laserPoints;
    std::vector<IntrinsicGroupState> intrinsicGroups;
};

OptimizationState initializeState(const std::vector<FramePinholeCamera>& cameras,
                                  const std::vector<BATrack>& tracks,
                                  const BAOptions& options,
                                  const ActiveProblem& active);

plamatrix::BlockNormalEquations<double> buildNormalEquations(
    const std::vector<FramePinholeCamera>& input_cameras,
    const std::vector<BATrack>& tracks,
    const BAOptions& options,
    const ActiveProblem& active,
    const OptimizationState& state,
    int iteration,
    double* objective_cost = nullptr);

double evaluateObjective(const std::vector<FramePinholeCamera>& input_cameras,
                         const std::vector<BATrack>& tracks,
                         const BAOptions& options,
                         const ActiveProblem& active,
                         const OptimizationState& state,
                         int iteration);

double maximumStepNorm(const std::vector<double>& primary_step,
                       const std::vector<double>& eliminated_step);

void applyStep(const ActiveProblem& active,
               const std::vector<double>& primary_step,
               const std::vector<double>& eliminated_step,
               OptimizationState* state);

} // namespace xjw::detail::plamatrix_ba
