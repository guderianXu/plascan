#include "BundleAdjustPlaMatrix.h"

#include "BundleAdjustPlaMatrixAssembly.h"
#include "BundleAdjustPlaMatrixProblem.h"
#include "BundleAdjustPlaMatrixReferenceSchur.h"
#include "BundleAdjustPlaMatrixRuntime.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"

#include <plamatrix/optimization/block_schur.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace xjw::detail
{
    namespace
    {

        using plamatrix_ba::ActiveProblem;
        using plamatrix_ba::prepareActiveProblem;

        bool isCancelled(const BAOptions& options)
        {
            return options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed);
        }

        bool isRetryableReferenceLinearFailure(const std::exception& error)
        {
            const std::string message = error.what();
            return message.find("breakdown") != std::string::npos ||
                   message.find("positive definite") != std::string::npos ||
                   message.find("numerically SPD") != std::string::npos;
        }

        bool
        isFinalIntrinsicStage(const BAOptions& options, const plamatrix_ba::OptimizationState& state, int iteration)
        {
            return std::all_of(state.intrinsicGroups.begin(),
                               state.intrinsicGroups.end(),
                               [&](const auto& group) {
                                   return plamatrix_ba::activeIntrinsicParameters(options, group.enabled, iteration) ==
                                          group.enabled;
                               });
        }

        struct ReferenceStepRms
        {
            double global = 0.0;
            double point = 0.0;
        };

        ReferenceStepRms referenceStepRms(const BAOptions& options,
                                          const ActiveProblem& active,
                                          const plamatrix_ba::OptimizationState& state,
                                          int iteration,
                                          const std::vector<double>& primary_step,
                                          const std::vector<double>& eliminated_step)
        {
            double global_squared = 0.0;
            std::size_t global_dimension = 0;
            double point_squared = 0.0;
            std::size_t point_dimension = 0;
            const auto accumulate = [](const std::vector<double>& values,
                                       std::size_t offset,
                                       std::size_t count,
                                       double* squared,
                                       std::size_t* dimension)
            {
                for (std::size_t index = 0; index < count; ++index)
                {
                    const double value = values[offset + index];
                    *squared += value * value;
                }
                *dimension += count;
            };

            for (std::size_t camera_index = 0; camera_index < active.cameraBlock.size(); ++camera_index)
            {
                const int block = active.cameraBlock[camera_index];
                if (block >= 0)
                {
                    accumulate(primary_step,
                               static_cast<std::size_t>(block * plamatrix_ba::kPrimaryBlockSize),
                               6,
                               &global_squared,
                               &global_dimension);
                }
            }
            for (std::size_t group_index = 0; group_index < state.intrinsicGroups.size(); ++group_index)
            {
                const auto active_parameters = plamatrix_ba::activeIntrinsicParameters(
                    options, state.intrinsicGroups[group_index].enabled, iteration);
                const std::size_t offset = static_cast<std::size_t>(
                    (active.cameraBlockCount + static_cast<int>(group_index)) * plamatrix_ba::kPrimaryBlockSize);
                for (std::size_t parameter = 0; parameter < active_parameters.size(); ++parameter)
                {
                    if (active_parameters[parameter])
                    {
                        accumulate(primary_step, offset + parameter, 1, &global_squared, &global_dimension);
                    }
                }
            }
            for (std::size_t track_index = 0; track_index < active.trackPrimaryBlock.size(); ++track_index)
            {
                const int primary_block = active.trackPrimaryBlock[track_index];
                if (primary_block >= 0)
                {
                    accumulate(primary_step,
                               static_cast<std::size_t>(primary_block * plamatrix_ba::kPrimaryBlockSize),
                               3,
                               &point_squared,
                               &point_dimension);
                }
                const int eliminated_block = active.trackBlock[track_index];
                if (eliminated_block >= 0)
                {
                    accumulate(eliminated_step,
                               static_cast<std::size_t>(eliminated_block * plamatrix_ba::kEliminatedBlockSize),
                               3,
                               &point_squared,
                               &point_dimension);
                }
            }
            for (const int block : active.laserBlock)
            {
                if (block >= 0)
                {
                    accumulate(eliminated_step,
                               static_cast<std::size_t>(block * plamatrix_ba::kEliminatedBlockSize),
                               3,
                               &point_squared,
                               &point_dimension);
                }
            }
            return {global_dimension > 0 ? std::sqrt(global_squared / static_cast<double>(global_dimension)) : 0.0,
                    point_dimension > 0 ? std::sqrt(point_squared / static_cast<double>(point_dimension)) : 0.0};
        }

    } // namespace

    BAResult optimizePointsWithPlaMatrix(const std::vector<FramePinholeCamera>& cameras,
                                         const std::vector<BATrack>& tracks,
                                         const BAOptions& options)
    {
        BAResult result;
        result.requestedBackend = options.backend;
        result.usedBackend = options.backend;
        auto solver_backend = plaMatrixLinearBackend(options.backend);
        result.refinedCameras = cameras;
        result.totalTracks = static_cast<int>(tracks.size());
        result.points.resize(tracks.size());
        for (std::size_t index = 0; index < tracks.size(); ++index)
        {
            result.points[index].point = tracks[index].initialPoint;
        }
        if (cameras.empty() || tracks.empty())
        {
            result.solveStatus = BASolveStatus::InvalidInput;
            result.backendMessage = "PlaMatrix BA 输入相机或轨迹为空";
            return result;
        }
        if (isCancelled(options))
        {
            result.solveStatus = BASolveStatus::Cancelled;
            result.backendMessage = "PlaMatrix BA 在启动前被取消";
            return result;
        }

        const auto setup_start = std::chrono::steady_clock::now();
        const ActiveProblem active = prepareActiveProblem(cameras, tracks, options);
        result.setupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
        result.observationCount = active.observationCount;
        result.plaMatrixRejectedInitialTracks = active.rejectedInitialTracks;
        result.plaMatrixLinearSolverName = plaMatrixLinearBackendName(solver_backend);
        if (active.observationCount == 0 && active.activeLaserRangeCount == 0)
        {
            result.solveStatus = BASolveStatus::InvalidInput;
            result.backendMessage = "PlaMatrix BA 没有足够有效的轨迹";
            result.totalSeconds = result.setupSeconds;
            return result;
        }

        const auto solve_start = std::chrono::steady_clock::now();
        plamatrix_ba::OptimizationState state = plamatrix_ba::initializeState(cameras, tracks, options, active);
        double current_cost = std::numeric_limits<double>::quiet_NaN();
        double damping = 0.0;
        int accepted_steps = 0;
        int rejected_steps = 0;
        bool converged = active.primaryBlockCount == 0 && active.trackBlockCount + active.laserBlockCount == 0;
        bool cancelled = false;
        int iterations = 0;
        double minimum_linear_tolerance = std::numeric_limits<double>::infinity();
        double maximum_linear_tolerance = 0.0;
        plamatrix::SchurComplementSolverWorkspace<double> solver_workspace;
        plamatrix_ba::NormalEquationAssemblyWorkspace assembly_workspace(active);
        plamatrix_ba::ReferenceSchurWorkspace reference_schur_workspace(active);
        const bool use_reference_online_schur = plamatrix_ba::canUseReferenceOnlineSchur(options, active);
        result.plaMatrixReferenceOnlineSchurUsed = use_reference_online_schur;
        plamatrix_ba::OptimizationState candidate_state = state;
        std::vector<double> primary_step;
        std::vector<double> eliminated_step;

        try
        {
            if (converged)
            {
                current_cost = plamatrix_ba::evaluateObjective(cameras, tracks, options, active, state, 0);
                result.plaMatrixInitialCost = current_cost;
                ++result.plaMatrixObjectiveEvaluations;
            }

            const int iteration_budget = std::max(1, options.maxIterations);
            bool terminate = false;
            while (iterations < iteration_budget && !converged && !terminate)
            {
                if (isCancelled(options))
                {
                    cancelled = true;
                    break;
                }

                const int stage_iteration = iterations;
                double linearized_cost = 0.0;
                if (!use_reference_online_schur)
                {
                    const auto assembly_start = std::chrono::steady_clock::now();
                    plamatrix_ba::buildNormalEquations(cameras,
                                                       tracks,
                                                       options,
                                                       active,
                                                       state,
                                                       stage_iteration,
                                                       &assembly_workspace,
                                                       &linearized_cost);
                    result.plaMatrixAssemblySeconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - assembly_start).count();
                    ++result.plaMatrixLinearizations;
                    ++result.plaMatrixObjectiveEvaluations;
                    current_cost = linearized_cost;
                    if (result.plaMatrixLinearizations == 1)
                    {
                        result.plaMatrixInitialCost = current_cost;
                    }
                }

                bool relinearize = false;
                std::vector<double> retry_primary_step;
                while (iterations < iteration_budget && !converged && !relinearize)
                {
                    if (isCancelled(options))
                    {
                        cancelled = true;
                        break;
                    }

                    if (use_reference_online_schur)
                    {
                        const auto assembly_start = std::chrono::steady_clock::now();
                        const auto build =
                            plamatrix_ba::buildReferenceReducedNormalEquations(cameras,
                                                                               tracks,
                                                                               options,
                                                                               active,
                                                                               state,
                                                                               stage_iteration,
                                                                               damping,
                                                                               &reference_schur_workspace);
                        result.plaMatrixAssemblySeconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - assembly_start).count();
                        ++result.plaMatrixLinearizations;
                        ++result.plaMatrixObjectiveEvaluations;
                        current_cost = build.objectiveCost;
                        if (result.plaMatrixLinearizations == 1)
                        {
                            result.plaMatrixInitialCost = current_cost;
                        }
                        if (build.pointSystemSingular)
                        {
                            ++iterations;
                            ++rejected_steps;
                            damping = damping == 0.0 ? 1.0e-3 : damping * 10.0;
                            continue;
                        }
                    }

                    plamatrix::SchurComplementSolverOptions<double> solver_options;
                    solver_options.linearBackend = solver_backend;
                    solver_options.deviceIndex = options.plaMatrixDevice;
                    solver_options.maxIterations =
                        (solver_backend == plamatrix::SchurComplementLinearBackend::Cpu ||
                         solver_backend == plamatrix::SchurComplementLinearBackend::DenseCpu ||
                         solver_backend == plamatrix::SchurComplementLinearBackend::SparseCpu)
                            ? std::max(50, active.primaryBlockCount * 12)
                            : std::max(200, active.primaryBlockCount * 30);
                    solver_options.relativeTolerance = 1.0e-12;
                    minimum_linear_tolerance = std::min(minimum_linear_tolerance, solver_options.relativeTolerance);
                    maximum_linear_tolerance = std::max(maximum_linear_tolerance, solver_options.relativeTolerance);
                    solver_options.absoluteTolerance = 1e-12;
                    solver_options.preconditionerClusterSize = options.plaMatrixPreconditionerClusterSize;
                    solver_options.useInitialGuess = !use_reference_online_schur && !retry_primary_step.empty();
                    solver_options.useMixedPrecision =
                        options.enablePlaMatrixMixedPrecision &&
                        solver_backend == plamatrix::SchurComplementLinearBackend::Cuda &&
                        active.primaryBlockCount >= 50;
                    primary_step = use_reference_online_schur ? std::vector<double>{} : retry_primary_step;
                    eliminated_step.clear();
                    const auto& equations = use_reference_online_schur ? reference_schur_workspace.reducedEquations
                                                                       : assembly_workspace.equations;
                    plamatrix::SchurComplementSolverReport<double> solver_report;
                    try
                    {
                        solver_report =
                            plamatrix::solveDampedSchurComplement(equations,
                                                                  use_reference_online_schur ? 0.0 : damping,
                                                                  solver_options,
                                                                  solver_workspace,
                                                                  &primary_step,
                                                                  &eliminated_step);
                    }
                    catch (const std::exception& error)
                    {
                        if (!use_reference_online_schur || !isRetryableReferenceLinearFailure(error))
                        {
                            throw;
                        }
                        ++iterations;
                        ++rejected_steps;
                        damping = damping == 0.0 ? 1.0e-3 : damping * 10.0;
                        continue;
                    }
                    result.plaMatrixLinearIterations += solver_report.iterations;
                    if (!solver_report.preconditionerName.empty())
                    {
                        result.plaMatrixPreconditionerName = solver_report.preconditionerName;
                    }
                    if (solver_backend != plamatrix::SchurComplementLinearBackend::Cpu)
                    {
                        if (solver_report.schurPatternReused)
                        {
                            ++result.plaMatrixSchurPatternReuses;
                        }
                        else
                        {
                            ++result.plaMatrixSchurPatternBuilds;
                        }
                    }
                    result.plaMatrixSmallBlockInverseSeconds += solver_report.smallBlockInverseSeconds;
                    result.plaMatrixSchurAccumulationSeconds += solver_report.schurAccumulationSeconds;
                    result.plaMatrixCsrConversionSeconds += solver_report.csrConversionSeconds;
                    result.plaMatrixSchurAssemblySeconds += solver_report.schurAssemblySeconds;
                    result.plaMatrixCholeskyFactorizationSeconds += solver_report.choleskyFactorizationSeconds;
                    result.plaMatrixTriangularSolveSeconds += solver_report.triangularSolveSeconds;
                    result.plaMatrixSymbolicAnalysisSeconds += solver_report.symbolicAnalysisSeconds;
                    if (solver_report.symbolicAnalysisReused)
                    {
                        ++result.plaMatrixSymbolicAnalysisReuses;
                    }
                    result.plaMatrixResidualCheckSeconds += solver_report.residualCheckSeconds;
                    result.plaMatrixLinearSolveSeconds += solver_report.linearSolveSeconds;
                    result.plaMatrixBackSubstitutionSeconds += solver_report.backSubstitutionSeconds;
                    if (!solver_report.deviceName.empty())
                    {
                        result.plaMatrixDeviceName = solver_report.deviceName;
                        result.usedGpu = true;
                    }
                    result.plaMatrixSchurAssemblyOnDevice =
                        result.plaMatrixSchurAssemblyOnDevice || solver_report.schurAssemblyOnDevice;
                    result.plaMatrixMixedPrecisionUsed =
                        result.plaMatrixMixedPrecisionUsed || solver_report.mixedPrecisionUsed;
                    ++iterations;
                    if (!solver_report.converged)
                    {
                        retry_primary_step = use_reference_online_schur ? std::vector<double>{} : primary_step;
                        ++rejected_steps;
                        damping = damping == 0.0 ? 1.0e-3 : damping * 10.0;
                        continue;
                    }

                    double reference_directional_decrease = 0.0;
                    if (use_reference_online_schur)
                    {
                        const auto back_substitution_start = std::chrono::steady_clock::now();
                        const auto back_substitution =
                            plamatrix_ba::recoverReferencePointSteps(cameras,
                                                                     tracks,
                                                                     options,
                                                                     active,
                                                                     state,
                                                                     stage_iteration,
                                                                     damping,
                                                                     primary_step,
                                                                     reference_schur_workspace.directPrimaryRhs,
                                                                     &eliminated_step);
                        result.plaMatrixBackSubstitutionSeconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - back_substitution_start)
                                .count();
                        if (!back_substitution.success || !std::isfinite(back_substitution.directionalDecrease))
                        {
                            ++rejected_steps;
                            damping = damping == 0.0 ? 1.0e-3 : damping * 10.0;
                            continue;
                        }
                        reference_directional_decrease = back_substitution.directionalDecrease;
                    }

                    const ReferenceStepRms reference_step_rms =
                        referenceStepRms(options, active, state, stage_iteration, primary_step, eliminated_step);
                    constexpr double convergence_tolerance = 1.0e-6;
                    const bool update_is_small = reference_step_rms.global <= convergence_tolerance &&
                                                 reference_step_rms.point <= convergence_tolerance;
                    if (update_is_small && !use_reference_online_schur)
                    {
                        converged = isFinalIntrinsicStage(options, state, stage_iteration);
                        relinearize = !converged;
                        if (options.progressCallback)
                        {
                            const double rms =
                                std::sqrt(std::max(0.0, current_cost) / std::max(1, active.observationCount));
                            if (!options.progressCallback(iterations, iteration_budget, rms, active.activeTrackCount))
                            {
                                cancelled = true;
                            }
                        }
                        continue;
                    }

                    double candidate_cost = std::numeric_limits<double>::infinity();
                    double accepted_alpha = 1.0;
                    double directional_decrease = reference_directional_decrease;
                    // 所有 PlaScan 扩展约束都复用同一个目标函数，因此在完整 retraction 上
                    // 数值估计 -g^T d，比为每种因子维护第二套方向导数更不易产生语义漂移。
                    if (!use_reference_online_schur)
                    {
                        constexpr double derivative_step = 1.0e-4;
                        candidate_state = state;
                        plamatrix_ba::applyStep(
                            active, primary_step, eliminated_step, &candidate_state, options, derivative_step);
                        const double derivative_cost = plamatrix_ba::evaluateObjective(
                            cameras, tracks, options, active, candidate_state, stage_iteration);
                        ++result.plaMatrixObjectiveEvaluations;
                        if (std::isfinite(derivative_cost) && derivative_cost < current_cost)
                        {
                            directional_decrease = (current_cost - derivative_cost) / derivative_step;
                        }
                    }
                    const int line_search_steps = options.referenceLineSearchSteps;
                    bool armijo_accepted = false;
                    for (int line_search = 0; line_search < line_search_steps; ++line_search)
                    {
                        const auto trial_state_start = std::chrono::steady_clock::now();
                        candidate_state = state;
                        plamatrix_ba::applyStep(
                            active, primary_step, eliminated_step, &candidate_state, options, accepted_alpha);
                        result.plaMatrixTrialStateSeconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - trial_state_start).count();
                        const auto objective_start = std::chrono::steady_clock::now();
                        candidate_cost = plamatrix_ba::evaluateObjective(
                            cameras, tracks, options, active, candidate_state, stage_iteration);
                        result.plaMatrixObjectiveSeconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - objective_start).count();
                        ++result.plaMatrixObjectiveEvaluations;
                        const double sufficient_decrease =
                            options.referenceArmijoCoefficient * accepted_alpha * directional_decrease;
                        const double acceptance_limit =
                            use_reference_online_schur
                                ? current_cost - sufficient_decrease
                                : (directional_decrease > 0.0 ? current_cost - sufficient_decrease : current_cost);
                        if (std::isfinite(candidate_cost) && candidate_cost <= acceptance_limit &&
                            candidate_cost < current_cost)
                        {
                            armijo_accepted = true;
                            break;
                        }
                        accepted_alpha *= 0.5;
                    }
                    if (std::isfinite(candidate_cost) && candidate_cost < current_cost &&
                        (!use_reference_online_schur || armijo_accepted))
                    {
                        std::swap(state, candidate_state);
                        current_cost = candidate_cost;
                        ++accepted_steps;
                        damping /= 10.0;
                        converged = isFinalIntrinsicStage(options, state, stage_iteration) &&
                                    accepted_alpha * reference_step_rms.global <= 1.0e-6 &&
                                    accepted_alpha * reference_step_rms.point <= 1.0e-6;
                        relinearize = !converged;
                    }
                    else
                    {
                        ++rejected_steps;
                        if (use_reference_online_schur)
                        {
                            terminate = true;
                            relinearize = true;
                        }
                        else
                        {
                            retry_primary_step = primary_step;
                            damping = damping == 0.0 ? 1.0e-3 : damping * 10.0;
                        }
                    }

                    if (options.progressCallback)
                    {
                        const double rms =
                            std::sqrt(std::max(0.0, current_cost) / std::max(1, active.observationCount));
                        if (!options.progressCallback(iterations, iteration_budget, rms, active.activeTrackCount))
                        {
                            cancelled = true;
                            break;
                        }
                    }
                }
            }
        }
        catch (const std::exception& error)
        {
            result.solveStatus = BASolveStatus::NumericalFailure;
            result.backendMessage = std::string("PlaMatrix BA 数值求解失败：") + error.what();
        }

        result.solveSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        result.plaMatrixAcceptedSteps = accepted_steps;
        result.plaMatrixRejectedSteps = rejected_steps;
        result.plaMatrixLinearToleranceMinimum =
            std::isfinite(minimum_linear_tolerance) ? minimum_linear_tolerance : 0.0;
        result.plaMatrixLinearToleranceMaximum = maximum_linear_tolerance;
        result.plaMatrixFinalCost = current_cost;
        if (cancelled)
        {
            result.refinedCameras = cameras;
            for (std::size_t index = 0; index < tracks.size(); ++index)
            {
                result.points[index].point = tracks[index].initialPoint;
            }
            result.solveStatus = BASolveStatus::Cancelled;
            result.solutionUsable = false;
            result.backendMessage = "PlaMatrix BA 已取消，未发布中间解";
            result.totalSeconds = result.setupSeconds + result.solveSeconds;
            return result;
        }
        if (result.solveStatus == BASolveStatus::NumericalFailure)
        {
            result.totalSeconds = result.setupSeconds + result.solveSeconds;
            return result;
        }

        result.solveStatus = converged ? BASolveStatus::Success : BASolveStatus::NoConvergence;
        result.solutionUsable = std::isfinite(current_cost);
        result.backendMessage =
            converged ? "参考联合 BA 收敛（" + result.plaMatrixLinearSolverName + " + Armijo）"
                      : "参考联合 BA 达到迭代上限，返回可用解（" + result.plaMatrixLinearSolverName + " + Armijo）";
        if (result.plaMatrixDenseFallbacks > 0)
        {
            result.backendMessage += "；稠密 Schur 回退原因：" + result.plaMatrixDenseFallbackMessage;
        }
        result.refinedCameras = state.cameras;
        result.referenceCommittedIntrinsicParameterMask =
            plamatrix_ba::committedReferenceIntrinsicParameters(options, active, state, current_cost);
        plamatrix_ba::publishIntrinsics(
            cameras, options, active, state.intrinsicGroups, result.referenceCommittedIntrinsicParameterMask, &result);
        result.refinedCameraCount = active.cameraBlockCount;
        if (!state.intrinsicGroups.empty())
        {
            result.selfCalibrationStagesRun =
                plamatrix_ba::intrinsicStageCount(options, state.intrinsicGroups.front().enabled);
        }
        for (std::size_t index = 0; index < tracks.size(); ++index)
        {
            result.points[index].point = state.points[index];
            result.points[index].valid = active.activeTrack[index] != 0;
            result.points[index].converged = converged;
            result.points[index].iterations = iterations;
        }
        result.laserRangeShots.resize(state.laserPoints.size());
        for (std::size_t index = 0; index < state.laserPoints.size(); ++index)
        {
            result.laserRangeShots[index].point = state.laserPoints[index];
            result.laserRangeShots[index].valid = true;
        }
        const auto postprocess_start = std::chrono::steady_clock::now();
        finalizeBundleAdjustResult(cameras, tracks, options, &result);
        result.postprocessSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - postprocess_start).count();
        result.totalSeconds = result.setupSeconds + result.solveSeconds + result.postprocessSeconds;
        return result;
    }

} // namespace xjw::detail
