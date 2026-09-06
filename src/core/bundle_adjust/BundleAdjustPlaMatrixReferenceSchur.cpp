#include "BundleAdjustPlaMatrixReferenceSchur.h"

#include "BundleAdjustPlaMatrixAssemblyInternal.h"
#include "BundleAdjustValidation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <omp.h>

namespace xjw::detail::plamatrix_ba
{
    namespace
    {
        struct PointNormalBlock
        {
            std::array<double, 9> hessian{};
            std::array<double, 3> rhs{};
            std::vector<plamatrix::Index> primaryBlocks;
            std::vector<std::array<double, 27>> crossBlocks;

            void clear()
            {
                hessian.fill(0.0);
                rhs.fill(0.0);
                primaryBlocks.clear();
                crossBlocks.clear();
            }

            std::array<double, 27>& crossBlock(plamatrix::Index block)
            {
                const auto found = std::find(primaryBlocks.begin(), primaryBlocks.end(), block);
                if (found != primaryBlocks.end())
                {
                    return crossBlocks[static_cast<std::size_t>(found - primaryBlocks.begin())];
                }
                primaryBlocks.push_back(block);
                crossBlocks.emplace_back();
                return crossBlocks.back();
            }
        };

        bool invertPointHessian(const std::array<double, 9>& matrix, std::array<double, 9>* inverse)
        {
            const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                                       matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                                       matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
            if (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-15)
            {
                return false;
            }
            const double scale = 1.0 / determinant;
            *inverse = {{(matrix[4] * matrix[8] - matrix[5] * matrix[7]) * scale,
                         (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * scale,
                         (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * scale,
                         (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * scale,
                         (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * scale,
                         (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * scale,
                         (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * scale,
                         (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * scale,
                         (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * scale}};
            return std::all_of(inverse->begin(), inverse->end(), [](double value) { return std::isfinite(value); });
        }

        struct ReducedAccumulator
        {
            std::vector<double>* diagonal = nullptr;
            std::vector<double>* directRhs = nullptr;
            std::vector<double>* schurRhs = nullptr;
            std::vector<double>* offDiagonal = nullptr;
            const std::vector<int>* offDiagonalLookup = nullptr;
            int primaryBlockCount = 0;

            void addGradientEntry(plamatrix::Index block, int parameter, double value, bool direct)
            {
                const std::size_t index = static_cast<std::size_t>(block * kPrimaryBlockSize + parameter);
                if (direct)
                {
                    (*directRhs)[index] -= value;
                }
                else
                {
                    (*schurRhs)[index] -= value;
                }
            }

            void addHessianEntry(plamatrix::Index row_block,
                                 int row_parameter,
                                 plamatrix::Index column_block,
                                 int column_parameter,
                                 double value)
            {
                if (row_block == column_block)
                {
                    const std::size_t index =
                        static_cast<std::size_t>(row_block * kPrimaryBlockSize * kPrimaryBlockSize +
                                                 row_parameter * kPrimaryBlockSize + column_parameter);
                    (*diagonal)[index] += value;
                    return;
                }

                const bool ordered = row_block < column_block;
                const plamatrix::Index stored_row = ordered ? row_block : column_block;
                const plamatrix::Index stored_column = ordered ? column_block : row_block;
                const int stored_row_parameter = ordered ? row_parameter : column_parameter;
                const int stored_column_parameter = ordered ? column_parameter : row_parameter;
                const std::size_t lookup_index =
                    static_cast<std::size_t>(stored_row) * static_cast<std::size_t>(primaryBlockCount) +
                    static_cast<std::size_t>(stored_column);
                const int slot = (*offDiagonalLookup)[lookup_index];
                if (slot < 0)
                {
                    throw std::logic_error("参考在线 Schur 缺少预计算的非对角块");
                }
                const std::size_t index =
                    static_cast<std::size_t>(slot * kPrimaryBlockSize * kPrimaryBlockSize +
                                             stored_row_parameter * kPrimaryBlockSize + stored_column_parameter);
                std::atomic_ref<double> entry((*offDiagonal)[index]);
                entry.fetch_add(value, std::memory_order_relaxed);
            }
        };

        void accumulateObservationPrimaryTerms(const assembly_detail::ObservationPrimaryTerms& terms,
                                               const ObservationLinearization& linearization,
                                               ReducedAccumulator* accumulator)
        {
            for (std::size_t term = 0; term < terms.count; ++term)
            {
                for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
                {
                    const double jacobian_x = terms.jacobians[term][static_cast<std::size_t>(parameter)];
                    const double jacobian_y =
                        terms.jacobians[term][static_cast<std::size_t>(kPrimaryBlockSize + parameter)];
                    if (jacobian_x == 0.0 && jacobian_y == 0.0)
                    {
                        continue;
                    }
                    double gradient = 0.0;
                    for (int row = 0; row < 2; ++row)
                    {
                        gradient +=
                            linearization.normalWeight *
                            terms.jacobians[term][static_cast<std::size_t>(row * kPrimaryBlockSize + parameter)] *
                            linearization.residual[static_cast<std::size_t>(row)];
                    }
                    accumulator->addGradientEntry(terms.blocks[term], parameter, gradient, true);
                }
            }
            for (std::size_t left = 0; left < terms.count; ++left)
            {
                for (std::size_t right = left; right < terms.count; ++right)
                {
                    for (int row = 0; row < kPrimaryBlockSize; ++row)
                    {
                        const double left_x = terms.jacobians[left][static_cast<std::size_t>(row)];
                        const double left_y = terms.jacobians[left][static_cast<std::size_t>(kPrimaryBlockSize + row)];
                        if (left_x == 0.0 && left_y == 0.0)
                        {
                            continue;
                        }
                        for (int column = 0; column < kPrimaryBlockSize; ++column)
                        {
                            const double right_x = terms.jacobians[right][static_cast<std::size_t>(column)];
                            const double right_y =
                                terms.jacobians[right][static_cast<std::size_t>(kPrimaryBlockSize + column)];
                            if (right_x == 0.0 && right_y == 0.0)
                            {
                                continue;
                            }
                            accumulator->addHessianEntry(terms.blocks[left],
                                                         row,
                                                         terms.blocks[right],
                                                         column,
                                                         linearization.normalWeight *
                                                             (left_x * right_x + left_y * right_y));
                        }
                    }
                }
            }
        }

        void accumulatePointTerms(const assembly_detail::ObservationPrimaryTerms& terms,
                                  const ObservationLinearization& linearization,
                                  PointNormalBlock* point)
        {
            for (int column = 0; column < kEliminatedBlockSize; ++column)
            {
                for (int row = 0; row < 2; ++row)
                {
                    const double point_value =
                        linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + column)];
                    point->rhs[static_cast<std::size_t>(column)] -=
                        linearization.normalWeight * point_value *
                        linearization.residual[static_cast<std::size_t>(row)];
                    for (int other = 0; other < kEliminatedBlockSize; ++other)
                    {
                        point->hessian[static_cast<std::size_t>(column * kEliminatedBlockSize + other)] +=
                            linearization.normalWeight * point_value *
                            linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + other)];
                    }
                }
            }
            for (std::size_t term = 0; term < terms.count; ++term)
            {
                auto& cross = point->crossBlock(terms.blocks[term]);
                for (int primary = 0; primary < kPrimaryBlockSize; ++primary)
                {
                    const double jacobian_x = terms.jacobians[term][static_cast<std::size_t>(primary)];
                    const double jacobian_y =
                        terms.jacobians[term][static_cast<std::size_t>(kPrimaryBlockSize + primary)];
                    if (jacobian_x == 0.0 && jacobian_y == 0.0)
                    {
                        continue;
                    }
                    for (int eliminated = 0; eliminated < kEliminatedBlockSize; ++eliminated)
                    {
                        cross[static_cast<std::size_t>(primary * kEliminatedBlockSize + eliminated)] +=
                            linearization.normalWeight *
                            (jacobian_x * linearization.pointJacobian[static_cast<std::size_t>(eliminated)] +
                             jacobian_y *
                                 linearization
                                     .pointJacobian[static_cast<std::size_t>(kEliminatedBlockSize + eliminated)]);
                    }
                }
            }
        }

        bool linearizeTrack(const std::vector<FramePinholeCamera>& input_cameras,
                            const std::vector<BATrack>& tracks,
                            const BAOptions& options,
                            const ActiveProblem& active,
                            const OptimizationState& state,
                            int iteration,
                            std::size_t track_index,
                            ReducedAccumulator* accumulator,
                            PointNormalBlock* point,
                            double* cost)
        {
            point->clear();
            const bool eliminate_point = active.trackBlock[track_index] >= 0;
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
                                                                state.points[track_index],
                                                                observation,
                                                                iteration,
                                                                &linearization))
                {
                    throw std::runtime_error("参考在线 Schur 线性化产生非法投影");
                }
                *cost += linearization.robustCost;
                const auto terms =
                    assembly_detail::observationPrimaryTerms(options, active, state, camera_index, linearization);
                if (accumulator)
                {
                    accumulateObservationPrimaryTerms(terms, linearization, accumulator);
                }
                if (eliminate_point)
                {
                    accumulatePointTerms(terms, linearization, point);
                }
            }
            return eliminate_point;
        }

        bool eliminatePoint(double damping, const PointNormalBlock& point, ReducedAccumulator* accumulator)
        {
            std::array<double, 9> damped_hessian = point.hessian;
            for (int diagonal = 0; diagonal < kEliminatedBlockSize; ++diagonal)
            {
                damped_hessian[static_cast<std::size_t>(diagonal * kEliminatedBlockSize + diagonal)] *= 1.0 + damping;
            }
            std::array<double, 9> inverse{};
            if (!invertPointHessian(damped_hessian, &inverse))
            {
                return false;
            }

            std::vector<std::array<double, 27>> reduced_cross(point.crossBlocks.size());
            std::vector<std::array<char, 9>> active_rows(point.crossBlocks.size());
            for (std::size_t block = 0; block < point.crossBlocks.size(); ++block)
            {
                for (int row = 0; row < kPrimaryBlockSize; ++row)
                {
                    const std::size_t row_offset = static_cast<std::size_t>(row * kEliminatedBlockSize);
                    active_rows[block][static_cast<std::size_t>(row)] =
                        point.crossBlocks[block][row_offset] != 0.0 ||
                        point.crossBlocks[block][row_offset + 1] != 0.0 ||
                        point.crossBlocks[block][row_offset + 2] != 0.0;
                    if (!active_rows[block][static_cast<std::size_t>(row)])
                    {
                        continue;
                    }
                    for (int column = 0; column < kEliminatedBlockSize; ++column)
                    {
                        for (int inner = 0; inner < kEliminatedBlockSize; ++inner)
                        {
                            reduced_cross[block][static_cast<std::size_t>(row * kEliminatedBlockSize + column)] +=
                                point.crossBlocks[block][static_cast<std::size_t>(row * kEliminatedBlockSize + inner)] *
                                inverse[static_cast<std::size_t>(inner * kEliminatedBlockSize + column)];
                        }
                    }
                }
                for (int row = 0; row < kPrimaryBlockSize; ++row)
                {
                    if (!active_rows[block][static_cast<std::size_t>(row)])
                    {
                        continue;
                    }
                    double gradient = 0.0;
                    for (int column = 0; column < kEliminatedBlockSize; ++column)
                    {
                        gradient +=
                            reduced_cross[block][static_cast<std::size_t>(row * kEliminatedBlockSize + column)] *
                            point.rhs[static_cast<std::size_t>(column)];
                    }
                    accumulator->addGradientEntry(point.primaryBlocks[block], row, gradient, false);
                }
            }

            for (std::size_t left = 0; left < point.crossBlocks.size(); ++left)
            {
                for (std::size_t right = left; right < point.crossBlocks.size(); ++right)
                {
                    for (int row = 0; row < kPrimaryBlockSize; ++row)
                    {
                        if (!active_rows[left][static_cast<std::size_t>(row)])
                        {
                            continue;
                        }
                        for (int column = 0; column < kPrimaryBlockSize; ++column)
                        {
                            if (!active_rows[right][static_cast<std::size_t>(column)])
                            {
                                continue;
                            }
                            double hessian = 0.0;
                            for (int inner = 0; inner < kEliminatedBlockSize; ++inner)
                            {
                                hessian -=
                                    reduced_cross[left][static_cast<std::size_t>(row * kEliminatedBlockSize + inner)] *
                                    point.crossBlocks[right]
                                                     [static_cast<std::size_t>(column * kEliminatedBlockSize + inner)];
                            }
                            accumulator->addHessianEntry(
                                point.primaryBlocks[left], row, point.primaryBlocks[right], column, hessian);
                        }
                    }
                }
            }
            return true;
        }

        double assembleReferencePriors(const BAOptions& options,
                                       const ActiveProblem& active,
                                       const OptimizationState& state,
                                       int iteration,
                                       plamatrix::BlockNormalEquations<double>* equations,
                                       std::vector<double>* direct_rhs)
        {
            double cost = 0.0;
            for (std::size_t group_index = 0; group_index < state.intrinsicGroups.size(); ++group_index)
            {
                const auto& group = state.intrinsicGroups[group_index];
                const auto active_parameters = activeIntrinsicParameters(options, group.enabled, iteration);
                std::array<double, 9> residual{};
                std::array<double, 81> jacobian{};
                for (std::size_t parameter = 0; parameter < 9; ++parameter)
                {
                    if (!active_parameters[parameter])
                    {
                        continue;
                    }
                    const bool transition_parameter =
                        parameter != static_cast<std::size_t>(BAIntrinsicParameter::FocalAspectRatio);
                    const bool use_transition = group.usesReferenceTransitionPrior && transition_parameter;
                    const double reference =
                        use_transition && parameter == static_cast<std::size_t>(BAIntrinsicParameter::FocalLength)
                            ? group.focalReference
                            : group.prior[parameter];
                    const double weight =
                        use_transition ? group.transitionWeight[parameter] : group.inverseSigma[parameter];
                    residual[parameter] = weight * (group.parameters[parameter] - reference);
                    jacobian[parameter * 9 + parameter] = weight;
                    cost += 0.5 * residual[parameter] * residual[parameter];
                }
                const int block = active.cameraBlockCount + static_cast<int>(group_index);
                equations->addPrimaryResidualBlock(block, jacobian.data(), residual.data(), 9);
                const std::size_t offset = static_cast<std::size_t>(block * kPrimaryBlockSize);
                for (std::size_t parameter = 0; parameter < 9; ++parameter)
                {
                    (*direct_rhs)[offset + parameter] -= jacobian[parameter * 9 + parameter] * residual[parameter];
                }
            }
            return cost;
        }

        void updateTrackBoundaries(const std::vector<BATrack>& tracks,
                                   const ActiveProblem& active,
                                   int thread_count,
                                   ReferenceSchurWorkspace* workspace)
        {
            if (workspace->partitionThreadCount == thread_count &&
                workspace->trackBoundaries.size() == static_cast<std::size_t>(thread_count + 1))
            {
                return;
            }
            std::vector<std::size_t> prefix_cost(tracks.size() + 1, 0);
            for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
            {
                const std::size_t work = active.activeTrack[track_index]
                                             ? std::max<std::size_t>(1, tracks[track_index].observations.size())
                                             : 1;
                prefix_cost[track_index + 1] = prefix_cost[track_index] + work;
            }
            workspace->trackBoundaries.assign(static_cast<std::size_t>(thread_count + 1), tracks.size());
            workspace->trackBoundaries[0] = 0;
            for (int thread = 1; thread < thread_count; ++thread)
            {
                const std::size_t target =
                    prefix_cost.back() * static_cast<std::size_t>(thread) / static_cast<std::size_t>(thread_count);
                workspace->trackBoundaries[static_cast<std::size_t>(thread)] = static_cast<std::size_t>(
                    std::lower_bound(prefix_cost.begin(), prefix_cost.end(), target) - prefix_cost.begin());
            }
            workspace->partitionThreadCount = thread_count;
        }

        void prepareOffDiagonalTopology(const std::vector<BATrack>& tracks,
                                        const ActiveProblem& active,
                                        ReferenceSchurWorkspace* workspace)
        {
            if (!workspace->offDiagonalLookup.empty())
            {
                return;
            }
            const int block_count = active.primaryBlockCount;
            const std::size_t lookup_size =
                static_cast<std::size_t>(block_count) * static_cast<std::size_t>(block_count);
            workspace->offDiagonalLookup.assign(lookup_size, -1);
            std::vector<char> connected(lookup_size, 0);
            const auto connect = [&](int left, int right)
            {
                if (left < 0 || right < 0 || left == right)
                {
                    return;
                }
                const int row = std::min(left, right);
                const int column = std::max(left, right);
                connected[static_cast<std::size_t>(row) * static_cast<std::size_t>(block_count) +
                          static_cast<std::size_t>(column)] = 1;
            };

            std::vector<int> track_blocks;
            for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
            {
                if (!active.activeTrack[track_index])
                {
                    continue;
                }
                track_blocks.clear();
                for (const BAObservation& observation : tracks[track_index].observations)
                {
                    if (!observationIsUsable(observation, active.cameraBlock.size()))
                    {
                        continue;
                    }
                    const std::size_t camera_index = static_cast<std::size_t>(observation.cameraIndex);
                    const int camera_block = active.cameraBlock[camera_index];
                    const int intrinsic_block = active.intrinsicBlockByCamera[camera_index];
                    connect(camera_block, intrinsic_block);
                    if (active.trackBlock[track_index] >= 0)
                    {
                        if (camera_block >= 0)
                        {
                            track_blocks.push_back(camera_block);
                        }
                        if (intrinsic_block >= 0)
                        {
                            track_blocks.push_back(intrinsic_block);
                        }
                    }
                }
                if (active.trackBlock[track_index] < 0)
                {
                    continue;
                }
                std::sort(track_blocks.begin(), track_blocks.end());
                track_blocks.erase(std::unique(track_blocks.begin(), track_blocks.end()), track_blocks.end());
                for (std::size_t left = 0; left < track_blocks.size(); ++left)
                {
                    for (std::size_t right = left + 1; right < track_blocks.size(); ++right)
                    {
                        connect(track_blocks[left], track_blocks[right]);
                    }
                }
            }

            for (int row = 0; row < block_count; ++row)
            {
                for (int column = row + 1; column < block_count; ++column)
                {
                    const std::size_t lookup = static_cast<std::size_t>(row) * static_cast<std::size_t>(block_count) +
                                               static_cast<std::size_t>(column);
                    if (!connected[lookup])
                    {
                        continue;
                    }
                    workspace->offDiagonalLookup[lookup] = static_cast<int>(workspace->offDiagonalBlocks.size());
                    workspace->offDiagonalBlocks.emplace_back(row, column);
                }
            }
        }
    } // namespace

    ReferenceSchurWorkspace::ReferenceSchurWorkspace(const ActiveProblem& active)
        : reducedEquations(active.primaryBlockCount, 0, kPrimaryBlockSize, kEliminatedBlockSize)
    {
    }

    bool canUseReferenceOnlineSchur(const BAOptions& options, const ActiveProblem& active)
    {
        return options.useReferenceOnlineSchur && active.primaryBlockCount > 0 && active.promotedTrackBlockCount == 0 &&
               active.laserBlockCount == 0 && !options.enableLaserPlaneConstraints &&
               !options.enableControlPointConstraints && !options.enableLaserRangeConstraints &&
               !options.enableScaleBarConstraints && options.cameraPosePriors.empty() &&
               !options.cameraPlaneConstraint.enabled;
    }

    ReferenceSchurBuildResult buildReferenceReducedNormalEquations(const std::vector<FramePinholeCamera>& input_cameras,
                                                                   const std::vector<BATrack>& tracks,
                                                                   const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   int iteration,
                                                                   double damping,
                                                                   ReferenceSchurWorkspace* workspace)
    {
        if (!workspace || !std::isfinite(damping) || damping < 0.0)
        {
            throw std::invalid_argument("参考在线 Schur 工作区或阻尼无效");
        }
        const int requested_threads = options.numThreads > 0 ? options.numThreads : omp_get_max_threads();
        const int thread_count = std::min<int>(std::max(1, requested_threads), static_cast<int>(tracks.size()));
        updateTrackBoundaries(tracks, active, thread_count, workspace);
        prepareOffDiagonalTopology(tracks, active, workspace);
        workspace->partialDiagonals.resize(static_cast<std::size_t>(thread_count));
        workspace->partialDirectRhs.resize(static_cast<std::size_t>(thread_count));
        workspace->partialSchurRhs.resize(static_cast<std::size_t>(thread_count));
        workspace->partialCosts.assign(static_cast<std::size_t>(thread_count), 0.0);
        workspace->partialPointSystemSingular.assign(static_cast<std::size_t>(thread_count), 0);
        workspace->errors.assign(static_cast<std::size_t>(thread_count), {});
        const std::size_t primary_dimension = static_cast<std::size_t>(active.primaryBlockCount * kPrimaryBlockSize);
        const std::size_t diagonal_dimension =
            static_cast<std::size_t>(active.primaryBlockCount * kPrimaryBlockSize * kPrimaryBlockSize);
        for (int thread = 0; thread < thread_count; ++thread)
        {
            const std::size_t index = static_cast<std::size_t>(thread);
            workspace->partialDiagonals[index].assign(diagonal_dimension, 0.0);
            workspace->partialDirectRhs[index].assign(primary_dimension, 0.0);
            workspace->partialSchurRhs[index].assign(primary_dimension, 0.0);
        }
        const int shared_slot_count = std::min(4, thread_count);
        const std::size_t off_diagonal_dimension =
            workspace->offDiagonalBlocks.size() * static_cast<std::size_t>(kPrimaryBlockSize * kPrimaryBlockSize);
        workspace->sharedOffDiagonalSlots.resize(static_cast<std::size_t>(shared_slot_count));
        for (auto& slot : workspace->sharedOffDiagonalSlots)
        {
            slot.assign(off_diagonal_dimension, 0.0);
        }

#pragma omp parallel for num_threads(thread_count) schedule(static, 1)
        for (int thread = 0; thread < thread_count; ++thread)
        {
            try
            {
                PointNormalBlock point;
                point.primaryBlocks.reserve(16);
                point.crossBlocks.reserve(16);
                const std::size_t thread_index = static_cast<std::size_t>(thread);
                ReducedAccumulator accumulator{
                    &workspace->partialDiagonals[thread_index],
                    &workspace->partialDirectRhs[thread_index],
                    &workspace->partialSchurRhs[thread_index],
                    &workspace->sharedOffDiagonalSlots[static_cast<std::size_t>(thread % shared_slot_count)],
                    &workspace->offDiagonalLookup,
                    active.primaryBlockCount,
                };
                double cost = 0.0;
                const std::size_t begin = workspace->trackBoundaries[thread_index];
                const std::size_t end = workspace->trackBoundaries[thread_index + 1];
                for (std::size_t track_index = begin; track_index < end; ++track_index)
                {
                    if (!active.activeTrack[track_index])
                    {
                        continue;
                    }
                    if (linearizeTrack(input_cameras,
                                       tracks,
                                       options,
                                       active,
                                       state,
                                       iteration,
                                       track_index,
                                       &accumulator,
                                       &point,
                                       &cost) &&
                        !eliminatePoint(damping, point, &accumulator))
                    {
                        workspace->partialPointSystemSingular[thread_index] = 1;
                    }
                }
                workspace->partialCosts[thread_index] = cost;
            }
            catch (...)
            {
                workspace->errors[static_cast<std::size_t>(thread)] = std::current_exception();
            }
        }

        workspace->reducedEquations.clearValues();
        workspace->directPrimaryRhs.assign(primary_dimension, 0.0);
        ReferenceSchurBuildResult result;
        for (int thread = 0; thread < thread_count; ++thread)
        {
            const auto index = static_cast<std::size_t>(thread);
            if (workspace->errors[index])
            {
                std::rethrow_exception(workspace->errors[index]);
            }
            for (std::size_t parameter = 0; parameter < primary_dimension; ++parameter)
            {
                workspace->directPrimaryRhs[parameter] += workspace->partialDirectRhs[index][parameter];
            }
            result.objectiveCost += workspace->partialCosts[index];
            result.pointSystemSingular =
                result.pointSystemSingular || workspace->partialPointSystemSingular[index] != 0;
        }
        for (int block = 0; block < active.primaryBlockCount; ++block)
        {
            std::array<double, 9> gradient{};
            std::array<double, 81> diagonal{};
            const std::size_t gradient_offset = static_cast<std::size_t>(block * kPrimaryBlockSize);
            const std::size_t diagonal_offset = static_cast<std::size_t>(block * kPrimaryBlockSize * kPrimaryBlockSize);
            for (int thread = 0; thread < thread_count; ++thread)
            {
                const auto& partial_diagonal = workspace->partialDiagonals[static_cast<std::size_t>(thread)];
                const auto& partial_schur_rhs = workspace->partialSchurRhs[static_cast<std::size_t>(thread)];
                for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
                {
                    gradient[static_cast<std::size_t>(parameter)] -=
                        partial_schur_rhs[gradient_offset + static_cast<std::size_t>(parameter)];
                }
                for (int parameter = 0; parameter < kPrimaryBlockSize * kPrimaryBlockSize; ++parameter)
                {
                    diagonal[static_cast<std::size_t>(parameter)] +=
                        partial_diagonal[diagonal_offset + static_cast<std::size_t>(parameter)];
                }
            }
            for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
            {
                gradient[static_cast<std::size_t>(parameter)] -=
                    workspace->directPrimaryRhs[gradient_offset + static_cast<std::size_t>(parameter)];
            }
            workspace->reducedEquations.addPrimaryGradientBlock(block, gradient.data());
            workspace->reducedEquations.addPrimaryHessianBlock(block, block, diagonal.data());
        }
        for (std::size_t block = 0; block < workspace->offDiagonalBlocks.size(); ++block)
        {
            std::array<double, 81> hessian{};
            const std::size_t offset = block * static_cast<std::size_t>(kPrimaryBlockSize * kPrimaryBlockSize);
            for (const auto& slot : workspace->sharedOffDiagonalSlots)
            {
                for (int parameter = 0; parameter < kPrimaryBlockSize * kPrimaryBlockSize; ++parameter)
                {
                    hessian[static_cast<std::size_t>(parameter)] += slot[offset + static_cast<std::size_t>(parameter)];
                }
            }
            workspace->reducedEquations.addPrimaryHessianBlock(
                workspace->offDiagonalBlocks[block].first, workspace->offDiagonalBlocks[block].second, hessian.data());
        }
        result.objectiveCost += assembleReferencePriors(
            options, active, state, iteration, &workspace->reducedEquations, &workspace->directPrimaryRhs);
        workspace->reducedEquations.finalizePrimaryDiagonal(1.0 + damping, 1.0);
        return result;
    }

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
                               std::vector<double>* eliminated_step)
    {
        ReferenceSchurBackSubstitutionResult result;
        eliminated_step->assign(static_cast<std::size_t>(active.trackBlockCount * kEliminatedBlockSize), 0.0);
        for (std::size_t parameter = 0; parameter < primary_step.size(); ++parameter)
        {
            result.directionalDecrease += primary_step[parameter] * direct_primary_rhs[parameter];
        }
        const int requested_threads = options.numThreads > 0 ? options.numThreads : omp_get_max_threads();
        const int thread_count = std::min<int>(std::max(1, requested_threads), static_cast<int>(tracks.size()));
        std::vector<double> partial_directional(static_cast<std::size_t>(thread_count), 0.0);
        std::vector<char> partial_success(static_cast<std::size_t>(thread_count), 1);
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(thread_count));

#pragma omp parallel for num_threads(thread_count) schedule(static)
        for (int thread = 0; thread < thread_count; ++thread)
        {
            try
            {
                PointNormalBlock point;
                point.primaryBlocks.reserve(16);
                point.crossBlocks.reserve(16);
                const std::size_t begin =
                    tracks.size() * static_cast<std::size_t>(thread) / static_cast<std::size_t>(thread_count);
                const std::size_t end =
                    tracks.size() * static_cast<std::size_t>(thread + 1) / static_cast<std::size_t>(thread_count);
                for (std::size_t track_index = begin; track_index < end; ++track_index)
                {
                    const int point_block = active.trackBlock[track_index];
                    if (!active.activeTrack[track_index] || point_block < 0)
                    {
                        continue;
                    }
                    double ignored_cost = 0.0;
                    linearizeTrack(input_cameras,
                                   tracks,
                                   options,
                                   active,
                                   state,
                                   iteration,
                                   track_index,
                                   nullptr,
                                   &point,
                                   &ignored_cost);
                    std::array<double, 9> damped_hessian = point.hessian;
                    for (int diagonal = 0; diagonal < kEliminatedBlockSize; ++diagonal)
                    {
                        damped_hessian[static_cast<std::size_t>(diagonal * kEliminatedBlockSize + diagonal)] *=
                            1.0 + damping;
                    }
                    std::array<double, 9> inverse{};
                    if (!invertPointHessian(damped_hessian, &inverse))
                    {
                        partial_success[static_cast<std::size_t>(thread)] = 0;
                        continue;
                    }
                    std::array<double, 3> conditioned_rhs = point.rhs;
                    for (std::size_t block = 0; block < point.primaryBlocks.size(); ++block)
                    {
                        const std::size_t primary_offset =
                            static_cast<std::size_t>(point.primaryBlocks[block] * kPrimaryBlockSize);
                        for (int eliminated = 0; eliminated < kEliminatedBlockSize; ++eliminated)
                        {
                            for (int primary = 0; primary < kPrimaryBlockSize; ++primary)
                            {
                                conditioned_rhs[static_cast<std::size_t>(eliminated)] -=
                                    point.crossBlocks[block][static_cast<std::size_t>(primary * kEliminatedBlockSize +
                                                                                      eliminated)] *
                                    primary_step[primary_offset + static_cast<std::size_t>(primary)];
                            }
                        }
                    }
                    const std::size_t output_offset = static_cast<std::size_t>(point_block * kEliminatedBlockSize);
                    for (int row = 0; row < kEliminatedBlockSize; ++row)
                    {
                        for (int column = 0; column < kEliminatedBlockSize; ++column)
                        {
                            (*eliminated_step)[output_offset + static_cast<std::size_t>(row)] +=
                                inverse[static_cast<std::size_t>(row * kEliminatedBlockSize + column)] *
                                conditioned_rhs[static_cast<std::size_t>(column)];
                        }
                        partial_directional[static_cast<std::size_t>(thread)] +=
                            point.rhs[static_cast<std::size_t>(row)] *
                            (*eliminated_step)[output_offset + static_cast<std::size_t>(row)];
                    }
                }
            }
            catch (...)
            {
                errors[static_cast<std::size_t>(thread)] = std::current_exception();
            }
        }
        for (int thread = 0; thread < thread_count; ++thread)
        {
            const auto index = static_cast<std::size_t>(thread);
            if (errors[index])
            {
                std::rethrow_exception(errors[index]);
            }
            result.directionalDecrease += partial_directional[index];
            result.success = result.success && partial_success[index] != 0;
        }
        return result;
    }

} // namespace xjw::detail::plamatrix_ba
