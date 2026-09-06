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
    namespace
    {
        bool useReferencePointParameterization(const BAOptions& options)
        {
            return !options.enableLaserPlaneConstraints && !options.enableControlPointConstraints &&
                   !options.enableLaserRangeConstraints && !options.enableScaleBarConstraints;
        }

        double referencePointScale(const std::array<double, 3>& point)
        {
            const double length = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
            const double scaled_length = length / 1000.0;
            if (!std::isfinite(scaled_length) || scaled_length == 0.0)
            {
                return 1.0;
            }
            return scaled_length / std::atan(scaled_length);
        }

        std::array<double, 3> targetCenterCoordinates(const std::array<double, 3>& value)
        {
            return {{value[0], -value[1], -value[2]}};
        }

        std::array<double, 3> cross(const std::array<double, 3>& left, const std::array<double, 3>& right)
        {
            return {{left[1] * right[2] - left[2] * right[1],
                     left[2] * right[0] - left[0] * right[2],
                     left[0] * right[1] - left[1] * right[0]}};
        }

        double norm(const std::array<double, 3>& value)
        {
            return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        }

        bool
        usesReferenceGaugeTangent(const BAOptions& options, const OptimizationState& state, std::size_t camera_index)
        {
            return options.referenceGaugeAnchorCameraIndex >= 0 && options.referenceGaugeScaleCameraIndex >= 0 &&
                   options.referenceGaugeAnchorCameraIndex < static_cast<int>(state.cameras.size()) &&
                   options.referenceGaugeScaleCameraIndex < static_cast<int>(state.cameras.size()) &&
                   camera_index == static_cast<std::size_t>(options.referenceGaugeScaleCameraIndex) &&
                   std::isfinite(options.referenceGaugeBaseline) && options.referenceGaugeBaseline > 0.0;
        }

        std::array<std::array<double, 3>, 2> referenceGaugeTargetTangentBasis(const BAOptions& options,
                                                                              const OptimizationState& state)
        {
            const auto origin = targetCenterCoordinates(
                state.cameras[static_cast<std::size_t>(options.referenceGaugeAnchorCameraIndex)].cameraCenter());
            const auto center = targetCenterCoordinates(
                state.cameras[static_cast<std::size_t>(options.referenceGaugeScaleCameraIndex)].cameraCenter());
            std::array<double, 3> unit{{center[0] - origin[0], center[1] - origin[1], center[2] - origin[2]}};
            const double length = norm(unit);
            const double inverse_length = length >= 1.0e-20 ? 1.0 / length : 0.0;
            for (double& value : unit)
            {
                value *= inverse_length;
            }

            std::size_t axis_index = std::abs(unit[0]) > std::abs(unit[1]) ? 1U : 0U;
            if (std::abs(unit[axis_index]) > std::abs(unit[2]))
            {
                axis_index = 2;
            }
            std::array<double, 3> axis{};
            axis[axis_index] = 1.0;
            std::array<double, 3> second = cross(axis, unit);
            const double second_length = norm(second);
            const double inverse_second_length = second_length >= 1.0e-20 ? 1.0 / second_length : 0.0;
            for (double& value : second)
            {
                value *= inverse_second_length;
            }
            return {{cross(unit, second), second}};
        }

        std::array<double, 3> applyReferenceGaugeTangentStep(const BAOptions& options,
                                                             const OptimizationState& state,
                                                             double first,
                                                             double second)
        {
            const auto origin = targetCenterCoordinates(
                state.cameras[static_cast<std::size_t>(options.referenceGaugeAnchorCameraIndex)].cameraCenter());
            const auto center = targetCenterCoordinates(
                state.cameras[static_cast<std::size_t>(options.referenceGaugeScaleCameraIndex)].cameraCenter());
            const auto basis = referenceGaugeTargetTangentBasis(options, state);
            const std::array<double, 3> tangent_delta{{basis[0][0] * first + basis[1][0] * second,
                                                       basis[0][1] * first + basis[1][1] * second,
                                                       basis[0][2] * first + basis[1][2] * second}};
            const std::array<double, 3> tentative{
                {center[0] + tangent_delta[0], center[1] + tangent_delta[1], center[2] + tangent_delta[2]}};
            const std::array<double, 3> direction{
                {tentative[0] - origin[0], tentative[1] - origin[1], tentative[2] - origin[2]}};
            const double updated_length = norm(direction);
            if (updated_length < 1.0e-20)
            {
                return targetCenterCoordinates(origin);
            }
            const double inverse_updated_length = 1.0 / updated_length;
            return targetCenterCoordinates(
                {{origin[0] + (direction[0] * inverse_updated_length) * options.referenceGaugeBaseline,
                  origin[1] + (direction[1] * inverse_updated_length) * options.referenceGaugeBaseline,
                  origin[2] + (direction[2] * inverse_updated_length) * options.referenceGaugeBaseline}});
        }

        void applyReferenceCameraPoseStep(FramePinholeCamera* camera, const double* delta)
        {
            auto rotation = camera->cameraToWorldRotation();
            // 转到参考 type-4 矩阵约定 R*diag(1,-1,-1)。
            for (int row = 0; row < 3; ++row)
            {
                rotation[static_cast<std::size_t>(row * 3 + 1)] *= -1.0;
                rotation[static_cast<std::size_t>(row * 3 + 2)] *= -1.0;
            }
            const double sx = std::sin(delta[0]);
            const double cx = std::cos(delta[0]);
            const double sy = std::sin(delta[1]);
            const double cy = std::cos(delta[1]);
            const double sz = std::sin(delta[2]);
            const double cz = std::cos(delta[2]);
            const std::array<double, 9> rx{{1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx}};
            const std::array<double, 9> ry{{cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy}};
            const std::array<double, 9> rz{{cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0}};
            const auto multiply = [](const std::array<double, 9>& left, const std::array<double, 9>& right)
            {
                std::array<double, 9> product{};
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        for (int inner = 0; inner < 3; ++inner)
                        {
                            product[static_cast<std::size_t>(row * 3 + column)] +=
                                left[static_cast<std::size_t>(row * 3 + inner)] *
                                right[static_cast<std::size_t>(inner * 3 + column)];
                        }
                    }
                }
                return product;
            };
            rotation = multiply(multiply(multiply(rotation, rx), ry), rz);
            for (int row = 0; row < 3; ++row)
            {
                rotation[static_cast<std::size_t>(row * 3 + 1)] *= -1.0;
                rotation[static_cast<std::size_t>(row * 3 + 2)] *= -1.0;
            }
            auto center = camera->cameraCenter();
            center[0] += delta[3];
            center[1] += delta[4];
            center[2] += delta[5];
            camera->setPose(rotation, center);
        }

    } // namespace

    namespace assembly_detail
    {

        ObservationPrimaryTerms observationPrimaryTerms(const BAOptions& options,
                                                        const ActiveProblem& active,
                                                        const OptimizationState& state,
                                                        std::size_t camera_index,
                                                        const ObservationLinearization& linearization)
        {
            ObservationPrimaryTerms terms;
            std::array<double, 18> camera_jacobian{};
            const bool constrained_center = usesReferenceGaugeTangent(options, state, camera_index);
            const auto target_basis = constrained_center ? referenceGaugeTargetTangentBasis(options, state)
                                                         : std::array<std::array<double, 3>, 2>{};
            for (int row = 0; row < 2; ++row)
            {
                std::copy_n(
                    linearization.cameraJacobian.data() + row * 6, 3, camera_jacobian.data() + row * kPrimaryBlockSize);
                if (constrained_center)
                {
                    for (std::size_t parameter = 0; parameter < 2; ++parameter)
                    {
                        const auto world_basis = targetCenterCoordinates(target_basis[parameter]);
                        for (int axis = 0; axis < 3; ++axis)
                        {
                            camera_jacobian[static_cast<std::size_t>(row * kPrimaryBlockSize + 3) + parameter] +=
                                linearization.cameraJacobian[static_cast<std::size_t>(row * 6 + 3 + axis)] *
                                world_basis[static_cast<std::size_t>(axis)];
                        }
                    }
                }
                else
                {
                    std::copy_n(linearization.cameraJacobian.data() + row * 6 + 3,
                                3,
                                camera_jacobian.data() + row * kPrimaryBlockSize + 3);
                }
            }
            if (active.cameraBlock[camera_index] >= 0)
            {
                terms.blocks[terms.count] = active.cameraBlock[camera_index];
                terms.jacobians[terms.count++] = camera_jacobian;
            }
            if (active.intrinsicBlockByCamera[camera_index] >= 0)
            {
                terms.blocks[terms.count] = active.intrinsicBlockByCamera[camera_index];
                std::copy(linearization.intrinsicJacobian.begin(),
                          linearization.intrinsicJacobian.end(),
                          terms.jacobians[terms.count].begin());
                ++terms.count;
            }
            return terms;
        }

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
                            const BAOptions& options,
                            const ActiveProblem& active,
                            const OptimizationState& state,
                            std::size_t camera_index,
                            int point_primary_block,
                            int point_eliminated_block,
                            const ObservationLinearization& linearization)
        {
            if (!equations)
            {
                return;
            }
            const ObservationPrimaryTerms terms =
                observationPrimaryTerms(options, active, state, camera_index, linearization);
            std::array<plamatrix::Index, 3> primary_blocks{};
            std::array<const double*, 3> primary_jacobians{};
            std::size_t primary_block_count = terms.count;
            for (std::size_t index = 0; index < terms.count; ++index)
            {
                primary_blocks[index] = terms.blocks[index];
                primary_jacobians[index] = terms.jacobians[index].data();
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
            constexpr double image_huber_delta = 0.0;
            const bool reference_point_parameterization = useReferencePointParameterization(options);
            if (state.intrinsicGroups.empty())
            {
                return linearizeObservation(state.cameras[camera_index],
                                            point,
                                            observation,
                                            image_huber_delta,
                                            output,
                                            true,
                                            reference_point_parameterization);
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
                                                            image_huber_delta,
                                                            output,
                                                            true,
                                                            reference_point_parameterization);
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
                                  plamatrix::BlockNormalEquations<double>* equations,
                                  int eliminated_block_offset)
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
                                                    options,
                                                    active,
                                                    state,
                                                    camera_index,
                                                    active.trackPrimaryBlock[track_index],
                                                    active.trackBlock[track_index] - eliminated_block_offset,
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
                    input_cameras, tracks, options, active, state, iteration, 0, tracks.size(), equations, 0);
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
            bool use_eliminated_shards = equations && useReferencePointParameterization(options) &&
                                         active.laserBlockCount == 0 && active.trackBlockCount > 0;
            std::vector<int> eliminated_offsets(static_cast<std::size_t>(thread_count), 0);
            std::vector<int> eliminated_counts(static_cast<std::size_t>(thread_count), 0);
            if (use_eliminated_shards)
            {
                for (int thread = 0; thread < thread_count && use_eliminated_shards; ++thread)
                {
                    const std::size_t begin = (*boundaries)[static_cast<std::size_t>(thread)];
                    const std::size_t end = (*boundaries)[static_cast<std::size_t>(thread + 1)];
                    int first_block = -1;
                    int block_count = 0;
                    for (std::size_t track_index = begin; track_index < end; ++track_index)
                    {
                        const int block = active.trackBlock[track_index];
                        if (block < 0)
                        {
                            continue;
                        }
                        if (first_block < 0)
                        {
                            first_block = block;
                        }
                        if (block != first_block + block_count)
                        {
                            use_eliminated_shards = false;
                            break;
                        }
                        ++block_count;
                    }
                    eliminated_offsets[static_cast<std::size_t>(thread)] = std::max(0, first_block);
                    eliminated_counts[static_cast<std::size_t>(thread)] = block_count;
                }
            }
            if (equations)
            {
                partial_equations->resize(static_cast<std::size_t>(thread_count));
                for (int thread = 0; thread < thread_count; ++thread)
                {
                    const auto index = static_cast<std::size_t>(thread);
                    const int eliminated_count = use_eliminated_shards
                                                     ? eliminated_counts[index]
                                                     : active.trackBlockCount + active.laserBlockCount;
                    auto& partial = (*partial_equations)[index];
                    if (!partial || partial->primaryBlockCount() != active.primaryBlockCount ||
                        partial->eliminatedBlockCount() != eliminated_count ||
                        partial->primaryBlockSize() != kPrimaryBlockSize ||
                        partial->eliminatedBlockSize() != kEliminatedBlockSize)
                    {
                        partial = std::make_unique<plamatrix::BlockNormalEquations<double>>(
                            active.primaryBlockCount, eliminated_count, kPrimaryBlockSize, kEliminatedBlockSize);
                    }
                    else
                    {
                        partial->clearValues();
                    }
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
                        equations ? (*partial_equations)[static_cast<std::size_t>(thread)].get() : nullptr,
                        use_eliminated_shards ? eliminated_offsets[static_cast<std::size_t>(thread)] : 0);
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
                    if (use_eliminated_shards)
                    {
                        equations->mergeEliminatedShardFrom(*(*partial_equations)[index], eliminated_offsets[index]);
                    }
                    else
                    {
                        equations->mergeFrom(*(*partial_equations)[index]);
                    }
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

    BAIntrinsicParameterMask committedReferenceIntrinsicParameters(const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   double final_cost)
    {
        BAIntrinsicParameterMask committed{};
        if (!options.useReferenceCalibrationTransitionPrior)
        {
            for (const auto& group : state.intrinsicGroups)
            {
                for (std::size_t parameter = 0; parameter < committed.size(); ++parameter)
                {
                    committed[parameter] = committed[parameter] || group.enabled[parameter];
                }
            }
            return committed;
        }

        committed = options.referencePreviousIntrinsicParameterMask;
        std::size_t prior_residual_count = 0;
        std::size_t intrinsic_parameter_count = 0;
        for (const auto& group : state.intrinsicGroups)
        {
            for (std::size_t parameter = 0; parameter < committed.size(); ++parameter)
            {
                prior_residual_count += group.transitionWeight[parameter] != 0.0 ? 1U : 0U;
                intrinsic_parameter_count += group.enabled[parameter] ? 1U : 0U;
            }
        }
        const std::size_t residual_count =
            static_cast<std::size_t>(std::max(0, active.observationCount)) * 2U + prior_residual_count;
        const std::size_t parameter_count = static_cast<std::size_t>(std::max(0, active.cameraBlockCount)) * 6U +
                                            intrinsic_parameter_count +
                                            static_cast<std::size_t>(std::max(0, active.activeTrackCount)) * 3U +
                                            static_cast<std::size_t>(std::max(0, active.activeLaserRangeCount)) * 3U;
        if (!std::isfinite(final_cost) || residual_count <= parameter_count)
        {
            return committed;
        }
        const double residual_sigma =
            std::sqrt(2.0 * std::max(0.0, final_cost) / static_cast<double>(residual_count - parameter_count));
        if (!std::isfinite(residual_sigma) || residual_sigma <= 0.0)
        {
            return committed;
        }

        for (const auto& group : state.intrinsicGroups)
        {
            for (std::size_t parameter = 0; parameter < committed.size(); ++parameter)
            {
                if (committed[parameter] || !group.enabled[parameter] || group.transitionWeight[parameter] == 0.0)
                {
                    continue;
                }
                const double value = group.parameters[parameter];
                const double reference = parameter == static_cast<std::size_t>(BAIntrinsicParameter::FocalLength)
                                             ? group.focalReference
                                             : group.prior[parameter];
                const double normalized_change =
                    std::abs(value - reference) * group.transitionWeight[parameter] / residual_sigma;
                if (normalized_change > 0.5)
                {
                    committed[parameter] = true;
                }
            }
        }
        return committed;
    }

    void applyStep(const ActiveProblem& active,
                   const std::vector<double>& primary_step,
                   const std::vector<double>& eliminated_step,
                   OptimizationState* state,
                   const BAOptions& options,
                   double step_scale)
    {
        std::vector<double> scaled_primary = primary_step;
        if (step_scale != 1.0)
        {
            for (double& value : scaled_primary)
            {
                value *= step_scale;
            }
        }
        for (std::size_t camera_index = 0; camera_index < state->cameras.size(); ++camera_index)
        {
            const int block = active.cameraBlock[camera_index];
            if (block >= 0)
            {
                const double* delta = scaled_primary.data() + block * kPrimaryBlockSize;
                if (usesReferenceGaugeTangent(options, *state, camera_index))
                {
                    const auto center = applyReferenceGaugeTangentStep(options, *state, delta[3], delta[4]);
                    std::array<double, 6> rotation_delta{{delta[0], delta[1], delta[2], 0.0, 0.0, 0.0}};
                    applyReferenceCameraPoseStep(&state->cameras[camera_index], rotation_delta.data());
                    state->cameras[camera_index].setCameraCenter(center);
                }
                else
                {
                    applyReferenceCameraPoseStep(&state->cameras[camera_index], delta);
                }
            }
        }
        applyIntrinsicStep(active, scaled_primary, &state->intrinsicGroups);
        const bool reference_point_parameterization = useReferencePointParameterization(options);
        for (std::size_t track_index = 0; track_index < state->points.size(); ++track_index)
        {
            const int primary_block = active.trackPrimaryBlock[track_index];
            const int eliminated_block = active.trackBlock[track_index];
            const double point_scale =
                reference_point_parameterization ? referencePointScale(state->points[track_index]) : 1.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (primary_block >= 0)
                {
                    state->points[track_index][static_cast<std::size_t>(axis)] +=
                        point_scale *
                        scaled_primary[static_cast<std::size_t>(primary_block * kPrimaryBlockSize + axis)];
                }
                else if (eliminated_block >= 0)
                {
                    state->points[track_index][static_cast<std::size_t>(axis)] +=
                        point_scale * step_scale *
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
                    step_scale * eliminated_step[static_cast<std::size_t>(block * kEliminatedBlockSize + axis)];
            }
        }
    }

} // namespace xjw::detail::plamatrix_ba
