#include "BundleAdjustPlaMatrixAssembly.h"

#include "BundleAdjustPlaMatrixAssemblyInternal.h"
#include "BundleAdjustPlaMatrixConstraints.h"
#include "BundleAdjustValidation.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <omp.h>

namespace xjw::detail::plamatrix_ba
{
    namespace assembly_detail
    {

        std::array<double, 54> paddedPointJacobian(const double* point_jacobian, int residual_size)
        {
            std::array<double, 54> padded{};
            for (int row = 0; row < residual_size; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    padded[static_cast<std::size_t>(row * kPrimaryBlockSize + column)] =
                        point_jacobian[row * kEliminatedBlockSize + column];
                }
            }
            return padded;
        }

        void addPointResidual(plamatrix::BlockNormalEquations<double>* equations,
                              int primary_block,
                              int eliminated_block,
                              const double* point_jacobian,
                              const double* residual,
                              int residual_size,
                              double weight)
        {
            if (!equations)
            {
                return;
            }
            if (primary_block >= 0)
            {
                const auto padded = paddedPointJacobian(point_jacobian, residual_size);
                equations->addPrimaryResidualBlock(primary_block, padded.data(), residual, residual_size, weight);
            }
            else if (eliminated_block >= 0)
            {
                equations->addEliminatedResidualBlock(
                    eliminated_block, point_jacobian, residual, residual_size, weight);
            }
        }

        void addObservation(plamatrix::BlockNormalEquations<double>* equations,
                            const ActiveProblem& active,
                            std::size_t camera_index,
                            int point_primary_block,
                            int point_eliminated_block,
                            const ObservationLinearization& linearization)
        {
            if (!equations)
            {
                return;
            }
            std::array<plamatrix::Index, 3> primary_blocks{};
            std::array<const double*, 3> primary_jacobians{};
            std::size_t primary_block_count = 0;
            std::array<double, 18> camera_jacobian{};
            for (int row = 0; row < 2; ++row)
            {
                std::copy_n(
                    linearization.cameraJacobian.data() + row * 6, 6, camera_jacobian.data() + row * kPrimaryBlockSize);
            }
            if (active.cameraBlock[camera_index] >= 0)
            {
                primary_blocks[primary_block_count] = active.cameraBlock[camera_index];
                primary_jacobians[primary_block_count++] = camera_jacobian.data();
            }
            if (active.intrinsicBlockByCamera[camera_index] >= 0)
            {
                primary_blocks[primary_block_count] = active.intrinsicBlockByCamera[camera_index];
                primary_jacobians[primary_block_count++] = linearization.intrinsicJacobian.data();
            }
            const auto point_primary_jacobian = paddedPointJacobian(linearization.pointJacobian.data(), 2);
            if (point_primary_block >= 0)
            {
                primary_blocks[primary_block_count] = point_primary_block;
                primary_jacobians[primary_block_count++] = point_primary_jacobian.data();
            }

            if (point_eliminated_block >= 0 && primary_block_count > 0)
            {
                equations->addResidualBlocks(primary_blocks.data(),
                                             primary_jacobians.data(),
                                             primary_block_count,
                                             point_eliminated_block,
                                             linearization.pointJacobian.data(),
                                             linearization.residual.data(),
                                             2,
                                             linearization.normalWeight);
            }
            else if (point_eliminated_block >= 0)
            {
                equations->addEliminatedResidualBlock(point_eliminated_block,
                                                      linearization.pointJacobian.data(),
                                                      linearization.residual.data(),
                                                      2,
                                                      linearization.normalWeight);
            }
            else if (primary_block_count > 0)
            {
                equations->addPrimaryResidualBlocks(primary_blocks.data(),
                                                    primary_jacobians.data(),
                                                    primary_block_count,
                                                    linearization.residual.data(),
                                                    2,
                                                    linearization.normalWeight);
            }
        }

        bool linearizeImageObservation(const std::vector<FramePinholeCamera>& input_cameras,
                                       const BAOptions& options,
                                       const ActiveProblem& active,
                                       const OptimizationState& state,
                                       std::size_t camera_index,
                                       const std::array<double, 3>& point,
                                       const BAObservation& observation,
                                       int iteration,
                                       ObservationLinearization* output)
        {
            if (state.intrinsicGroups.empty())
            {
                return linearizeObservation(
                    state.cameras[camera_index], point, observation, options.huberDelta, output);
            }
            const std::size_t group_index = static_cast<std::size_t>(active.calibrationGroupByCamera[camera_index]);
            const auto active_parameters =
                activeIntrinsicParameters(options, state.intrinsicGroups[group_index].enabled, iteration);
            const auto& references = options.sharedIntrinsicReferenceCameras.empty()
                                         ? input_cameras
                                         : options.sharedIntrinsicReferenceCameras;
            return linearizeObservationWithSharedIntrinsics(state.cameras[camera_index],
                                                            references[camera_index],
                                                            state.intrinsicGroups[group_index].parameters,
                                                            active_parameters,
                                                            point,
                                                            observation,
                                                            options.huberDelta,
                                                            output);
        }

    } // namespace assembly_detail

    namespace
    {

        double assembleTrackRange(const std::vector<FramePinholeCamera>& input_cameras,
                                  const std::vector<BATrack>& tracks,
                                  const BAOptions& options,
                                  const ActiveProblem& active,
                                  const OptimizationState& state,
                                  int iteration,
                                  std::size_t begin,
                                  std::size_t end,
                                  plamatrix::BlockNormalEquations<double>* equations)
        {
            double cost = 0.0;
            for (std::size_t track_index = begin; track_index < end; ++track_index)
            {
                if (!active.activeTrack[track_index])
                {
                    continue;
                }
                const auto& point = state.points[track_index];
                for (const auto& observation : tracks[track_index].observations)
                {
                    if (!observationIsUsable(observation, state.cameras.size()))
                    {
                        continue;
                    }
                    const std::size_t camera_index = static_cast<std::size_t>(observation.cameraIndex);
                    ObservationLinearization linearization;
                    if (!assembly_detail::linearizeImageObservation(input_cameras,
                                                                    options,
                                                                    active,
                                                                    state,
                                                                    camera_index,
                                                                    point,
                                                                    observation,
                                                                    iteration,
                                                                    &linearization))
                    {
                        throw std::runtime_error("PlaMatrix BA 线性化失败：活动轨迹产生了非法投影");
                    }
                    cost += linearization.robustCost;
                    assembly_detail::addObservation(equations,
                                                    active,
                                                    camera_index,
                                                    active.trackPrimaryBlock[track_index],
                                                    active.trackBlock[track_index],
                                                    linearization);
                }
                ConstraintLinearization constraint;
                if (options.enableLaserPlaneConstraints)
                {
                    for (const auto& laser : tracks[track_index].laserPlaneConstraints)
                    {
                        if (linearizeLaserPlane(laser, point, options, &constraint))
                        {
                            cost += constraint.robustCost;
                            assembly_detail::addPointResidual(equations,
                                                              active.trackPrimaryBlock[track_index],
                                                              active.trackBlock[track_index],
                                                              constraint.pointJacobian.data(),
                                                              constraint.residual.data(),
                                                              constraint.residualSize,
                                                              constraint.normalWeight);
                        }
                    }
                }
                if (options.enableControlPointConstraints)
                {
                    for (const auto& control : tracks[track_index].controlPointConstraints)
                    {
                        if (linearizeControlPoint(control, point, options, &constraint))
                        {
                            cost += constraint.robustCost;
                            assembly_detail::addPointResidual(equations,
                                                              active.trackPrimaryBlock[track_index],
                                                              active.trackBlock[track_index],
                                                              constraint.pointJacobian.data(),
                                                              constraint.residual.data(),
                                                              constraint.residualSize,
                                                              constraint.normalWeight);
                        }
                    }
                }
            }
            return cost;
        }

        double assembleTrackResiduals(const std::vector<FramePinholeCamera>& input_cameras,
                                      const std::vector<BATrack>& tracks,
                                      const BAOptions& options,
                                      const ActiveProblem& active,
                                      const OptimizationState& state,
                                      int iteration,
                                      plamatrix::BlockNormalEquations<double>* equations,
                                      NormalEquationAssemblyWorkspace* workspace)
        {
            const int requested_threads = options.numThreads > 0 ? options.numThreads : omp_get_max_threads();
            const int thread_count = std::min<int>(std::max(1, requested_threads), static_cast<int>(tracks.size()));
            if (thread_count <= 1 || tracks.size() < 128)
            {
                return assembleTrackRange(
                    input_cameras, tracks, options, active, state, iteration, 0, tracks.size(), equations);
            }

            std::vector<std::size_t> local_boundaries;
            auto* boundaries = workspace ? &workspace->trackBoundaries : &local_boundaries;
            if (!workspace || workspace->partitionThreadCount != thread_count ||
                boundaries->size() != static_cast<std::size_t>(thread_count + 1))
            {
                std::vector<std::size_t> prefix_cost(tracks.size() + 1, 0);
                for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
                {
                    std::size_t work = 0;
                    if (active.activeTrack[track_index])
                    {
                        work = static_cast<std::size_t>(
                            std::count_if(tracks[track_index].observations.cbegin(),
                                          tracks[track_index].observations.cend(),
                                          [&](const BAObservation& observation)
                                          { return observationIsUsable(observation, state.cameras.size()); }));
                        if (options.enableLaserPlaneConstraints)
                        {
                            work += tracks[track_index].laserPlaneConstraints.size();
                        }
                        if (options.enableControlPointConstraints)
                        {
                            work += tracks[track_index].controlPointConstraints.size();
                        }
                    }
                    prefix_cost[track_index + 1] = prefix_cost[track_index] + std::max<std::size_t>(1, work);
                }
                boundaries->assign(static_cast<std::size_t>(thread_count + 1), tracks.size());
                (*boundaries)[0] = 0;
                const std::size_t total_work = prefix_cost.back();
                for (int thread = 1; thread < thread_count; ++thread)
                {
                    const std::size_t target =
                        total_work * static_cast<std::size_t>(thread) / static_cast<std::size_t>(thread_count);
                    (*boundaries)[static_cast<std::size_t>(thread)] = static_cast<std::size_t>(
                        std::lower_bound(prefix_cost.begin(), prefix_cost.end(), target) - prefix_cost.begin());
                }
                if (workspace)
                {
                    workspace->partitionThreadCount = thread_count;
                }
            }

            std::vector<double> local_partial_costs;
            std::vector<std::exception_ptr> local_errors;
            auto* partial_costs = workspace ? &workspace->partialCosts : &local_partial_costs;
            auto* errors = workspace ? &workspace->errors : &local_errors;
            partial_costs->assign(static_cast<std::size_t>(thread_count), 0.0);
            errors->assign(static_cast<std::size_t>(thread_count), {});
            std::vector<std::unique_ptr<plamatrix::BlockNormalEquations<double>>> local_partial_equations;
            auto* partial_equations = workspace ? &workspace->partialEquations : &local_partial_equations;
            if (equations)
            {
                while (static_cast<int>(partial_equations->size()) < thread_count)
                {
                    partial_equations->push_back(std::make_unique<plamatrix::BlockNormalEquations<double>>(
                        active.primaryBlockCount,
                        active.trackBlockCount + active.laserBlockCount,
                        kPrimaryBlockSize,
                        kEliminatedBlockSize));
                }
                for (int thread = 0; thread < thread_count; ++thread)
                {
                    (*partial_equations)[static_cast<std::size_t>(thread)]->clearValues();
                }
            }

#pragma omp parallel for num_threads(thread_count) schedule(static, 1)
            for (int thread = 0; thread < thread_count; ++thread)
            {
                const std::size_t begin = (*boundaries)[static_cast<std::size_t>(thread)];
                const std::size_t end = (*boundaries)[static_cast<std::size_t>(thread + 1)];
                try
                {
                    (*partial_costs)[static_cast<std::size_t>(thread)] = assembleTrackRange(
                        input_cameras,
                        tracks,
                        options,
                        active,
                        state,
                        iteration,
                        begin,
                        end,
                        equations ? (*partial_equations)[static_cast<std::size_t>(thread)].get() : nullptr);
                }
                catch (...)
                {
                    (*errors)[static_cast<std::size_t>(thread)] = std::current_exception();
                }
            }

            for (int thread = 0; thread < thread_count; ++thread)
            {
                const auto index = static_cast<std::size_t>(thread);
                if ((*errors)[index])
                {
                    std::rethrow_exception((*errors)[index]);
                }
                if (equations)
                {
                    equations->mergeFrom(*(*partial_equations)[index]);
                }
            }
            return std::accumulate(partial_costs->begin(), partial_costs->end(), 0.0);
        }

        double assembleAll(const std::vector<FramePinholeCamera>& input_cameras,
                           const std::vector<BATrack>& tracks,
                           const BAOptions& options,
                           const ActiveProblem& active,
                           const OptimizationState& state,
                           int iteration,
                           plamatrix::BlockNormalEquations<double>* equations,
                           NormalEquationAssemblyWorkspace* workspace)
        {
            return assembleTrackResiduals(
                       input_cameras, tracks, options, active, state, iteration, equations, workspace) +
                   assembly_detail::assembleSurveyResiduals(
                       input_cameras, options, active, state, iteration, equations);
        }

    } // namespace

    OptimizationState initializeState(const std::vector<FramePinholeCamera>& cameras,
                                      const std::vector<BATrack>& tracks,
                                      const BAOptions& options,
                                      const ActiveProblem& active)
    {
        OptimizationState state;
        state.cameras = cameras;
        state.points.reserve(tracks.size());
        for (const auto& track : tracks)
        {
            state.points.push_back(track.initialPoint);
        }
        state.laserPoints.reserve(options.laserRangeConstraints.size());
        for (const auto& shot : options.laserRangeConstraints)
        {
            state.laserPoints.push_back(shot.initialPoint);
        }
        state.intrinsicGroups = initializeIntrinsicGroups(cameras, options, active);
        return state;
    }

    NormalEquationAssemblyWorkspace::NormalEquationAssemblyWorkspace(const ActiveProblem& active)
        : equations(active.primaryBlockCount,
                    active.trackBlockCount + active.laserBlockCount,
                    kPrimaryBlockSize,
                    kEliminatedBlockSize)
    {
    }

    void buildNormalEquations(const std::vector<FramePinholeCamera>& input_cameras,
                              const std::vector<BATrack>& tracks,
                              const BAOptions& options,
                              const ActiveProblem& active,
                              const OptimizationState& state,
                              int iteration,
                              NormalEquationAssemblyWorkspace* workspace,
                              double* objective_cost)
    {
        if (!workspace)
        {
            throw std::invalid_argument("PlaMatrix BA 法方程工作区不能为空");
        }
        workspace->equations.clearValues();
        const double cost =
            assembleAll(input_cameras, tracks, options, active, state, iteration, &workspace->equations, workspace);
        if (objective_cost)
        {
            *objective_cost = cost;
        }
    }

    double evaluateObjective(const std::vector<FramePinholeCamera>& input_cameras,
                             const std::vector<BATrack>& tracks,
                             const BAOptions& options,
                             const ActiveProblem& active,
                             const OptimizationState& state,
                             int iteration)
    {
        return assembleAll(input_cameras, tracks, options, active, state, iteration, nullptr, nullptr);
    }

    double maximumStepNorm(const std::vector<double>& primary_step, const std::vector<double>& eliminated_step)
    {
        double maximum = 0.0;
        const auto update_maximum = [&](const std::vector<double>& step, int block_size)
        {
            for (std::size_t offset = 0; offset < step.size(); offset += block_size)
            {
                double squared = 0.0;
                for (int index = 0; index < block_size; ++index)
                {
                    squared +=
                        step[offset + static_cast<std::size_t>(index)] * step[offset + static_cast<std::size_t>(index)];
                }
                maximum = std::max(maximum, std::sqrt(squared));
            }
        };
        update_maximum(primary_step, kPrimaryBlockSize);
        update_maximum(eliminated_step, kEliminatedBlockSize);
        return maximum;
    }

    void applyStep(const ActiveProblem& active,
                   const std::vector<double>& primary_step,
                   const std::vector<double>& eliminated_step,
                   OptimizationState* state)
    {
        for (std::size_t camera_index = 0; camera_index < state->cameras.size(); ++camera_index)
        {
            const int block = active.cameraBlock[camera_index];
            if (block >= 0)
            {
                state->cameras[camera_index].applyDeltaPose(primary_step.data() + block * kPrimaryBlockSize);
            }
        }
        applyIntrinsicStep(active, primary_step, &state->intrinsicGroups);
        for (std::size_t track_index = 0; track_index < state->points.size(); ++track_index)
        {
            const int primary_block = active.trackPrimaryBlock[track_index];
            const int eliminated_block = active.trackBlock[track_index];
            for (int axis = 0; axis < 3; ++axis)
            {
                if (primary_block >= 0)
                {
                    state->points[track_index][static_cast<std::size_t>(axis)] +=
                        primary_step[static_cast<std::size_t>(primary_block * kPrimaryBlockSize + axis)];
                }
                else if (eliminated_block >= 0)
                {
                    state->points[track_index][static_cast<std::size_t>(axis)] +=
                        eliminated_step[static_cast<std::size_t>(eliminated_block * kEliminatedBlockSize + axis)];
                }
            }
        }
        for (std::size_t shot_index = 0; shot_index < state->laserPoints.size(); ++shot_index)
        {
            const int block = active.laserBlock[shot_index];
            if (block < 0)
            {
                continue;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                state->laserPoints[shot_index][static_cast<std::size_t>(axis)] +=
                    eliminated_step[static_cast<std::size_t>(block * kEliminatedBlockSize + axis)];
            }
        }
    }

} // namespace xjw::detail::plamatrix_ba
