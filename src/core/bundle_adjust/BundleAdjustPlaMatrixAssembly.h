#pragma once

#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/optimization/block_schur.h>

#include <exception>
#include <memory>

namespace xjw::detail::plamatrix_ba
{

    struct OptimizationState
    {
        std::vector<FramePinholeCamera> cameras;
        std::vector<std::array<double, 3>> points;
        std::vector<std::array<double, 3>> laserPoints;
        std::vector<IntrinsicGroupState> intrinsicGroups;
    };

    struct NormalEquationAssemblyWorkspace
    {
        explicit NormalEquationAssemblyWorkspace(const ActiveProblem& active);

        plamatrix::BlockNormalEquations<double> equations;
        std::vector<std::unique_ptr<plamatrix::BlockNormalEquations<double>>> partialEquations;
        std::vector<double> partialCosts;
        std::vector<std::exception_ptr> errors;
        std::vector<std::size_t> trackBoundaries;
        int partitionThreadCount = 0;
    };

    OptimizationState initializeState(const std::vector<FramePinholeCamera>& cameras,
                                      const std::vector<BATrack>& tracks,
                                      const BAOptions& options,
                                      const ActiveProblem& active);

    void buildNormalEquations(const std::vector<FramePinholeCamera>& input_cameras,
                              const std::vector<BATrack>& tracks,
                              const BAOptions& options,
                              const ActiveProblem& active,
                              const OptimizationState& state,
                              int iteration,
                              NormalEquationAssemblyWorkspace* workspace,
                              double* objective_cost = nullptr);

    double evaluateObjective(const std::vector<FramePinholeCamera>& input_cameras,
                             const std::vector<BATrack>& tracks,
                             const BAOptions& options,
                             const ActiveProblem& active,
                             const OptimizationState& state,
                             int iteration);

    double maximumStepNorm(const std::vector<double>& primary_step, const std::vector<double>& eliminated_step);

    BAIntrinsicParameterMask committedReferenceIntrinsicParameters(const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   double final_cost);

    void applyStep(const ActiveProblem& active,
                   const std::vector<double>& primary_step,
                   const std::vector<double>& eliminated_step,
                   OptimizationState* state,
                   const BAOptions& options,
                   double step_scale = 1.0);

} // namespace xjw::detail::plamatrix_ba
