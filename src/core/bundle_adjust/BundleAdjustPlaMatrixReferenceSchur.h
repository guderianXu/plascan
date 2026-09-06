#pragma once

#include "BundleAdjustPlaMatrixAssembly.h"

#include <plamatrix/optimization/block_schur.h>

#include <exception>
#include <utility>

namespace xjw::detail::plamatrix_ba
{

    struct ReferenceSchurWorkspace
    {
        explicit ReferenceSchurWorkspace(const ActiveProblem& active);

        plamatrix::BlockNormalEquations<double> reducedEquations;
        std::vector<std::vector<double>> partialDiagonals;
        std::vector<std::vector<double>> partialDirectRhs;
        std::vector<std::vector<double>> partialSchurRhs;
        std::vector<std::vector<double>> sharedOffDiagonalSlots;
        std::vector<std::pair<plamatrix::Index, plamatrix::Index>> offDiagonalBlocks;
        std::vector<int> offDiagonalLookup;
        std::vector<double> directPrimaryRhs;
        std::vector<double> partialCosts;
        std::vector<char> partialPointSystemSingular;
        std::vector<std::exception_ptr> errors;
        std::vector<std::size_t> trackBoundaries;
        int partitionThreadCount = 0;
    };

    struct ReferenceSchurBuildResult
    {
        double objectiveCost = 0.0;
        bool pointSystemSingular = false;
    };

    struct ReferenceSchurBackSubstitutionResult
    {
        double directionalDecrease = 0.0;
        bool success = true;
    };

    bool canUseReferenceOnlineSchur(const BAOptions& options, const ActiveProblem& active);

    ReferenceSchurBuildResult buildReferenceReducedNormalEquations(const std::vector<FramePinholeCamera>& input_cameras,
                                                                   const std::vector<BATrack>& tracks,
                                                                   const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   int iteration,
                                                                   double damping,
                                                                   ReferenceSchurWorkspace* workspace);

    ReferenceSchurBackSubstitutionResult
    recoverReferencePointSteps(const std::vector<FramePinholeCamera>& input_cameras,
                               const std::vector<BATrack>& tracks,
                               const BAOptions& options,
                               const ActiveProblem& active,
                               const OptimizationState& state,
                               int iteration,
                               double damping,
                               const std::vector<double>& primary_step,
                               const std::vector<double>& direct_primary_rhs,
                               std::vector<double>* eliminated_step);

} // namespace xjw::detail::plamatrix_ba
