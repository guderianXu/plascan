#pragma once

#include "BundleAdjustPlaMatrixAssembly.h"
#include "BundleAdjustPlaMatrixProjection.h"

namespace xjw::detail::plamatrix_ba::assembly_detail
{

    struct ObservationPrimaryTerms
    {
        std::array<plamatrix::Index, 2> blocks{};
        std::array<std::array<double, 18>, 2> jacobians{};
        std::size_t count = 0;
    };

    ObservationPrimaryTerms observationPrimaryTerms(const BAOptions& options,
                                                    const ActiveProblem& active,
                                                    const OptimizationState& state,
                                                    std::size_t camera_index,
                                                    const ObservationLinearization& linearization);

    void addPointResidual(plamatrix::BlockNormalEquations<double>* equations,
                          int primary_block,
                          int eliminated_block,
                          const double* point_jacobian,
                          const double* residual,
                          int residual_size,
                          double weight);

    void addObservation(plamatrix::BlockNormalEquations<double>* equations,
                        const BAOptions& options,
                        const ActiveProblem& active,
                        const OptimizationState& state,
                        std::size_t camera_index,
                        int point_primary_block,
                        int point_eliminated_block,
                        const ObservationLinearization& linearization);

    bool linearizeImageObservation(const std::vector<FramePinholeCamera>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   std::size_t camera_index,
                                   const std::array<double, 3>& point,
                                   const BAObservation& observation,
                                   int iteration,
                                   ObservationLinearization* output);

    double assembleSurveyResiduals(const std::vector<FramePinholeCamera>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   int iteration,
                                   plamatrix::BlockNormalEquations<double>* equations);

} // namespace xjw::detail::plamatrix_ba::assembly_detail
