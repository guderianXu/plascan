#include "BundleAdjustPlaMatrix.h"

#include "BundleAdjustPlaMatrixAssembly.h"
#include "BundleAdjustPlaMatrixProblem.h"
#include "BundleAdjustPlaMatrixRuntime.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"

#include <plamatrix/optimization/block_schur.h>
#include <plamatrix/optimization/levenberg_marquardt.h>

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

// 与 Ceres 默认 function_tolerance 保持同一数量级，避免在已经达到相同
// 重投影精度后继续用拒绝步消耗迭代上限。
constexpr double kRelativeCostTolerance = 1e-6;
bool isCancelled(const BAOptions& options)
{
    return options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed);
}

bool isFinalIntrinsicStage(const BAOptions& options,
                           const plamatrix_ba::OptimizationState& state,
                           int iteration)
{
    return std::all_of(
        state.intrinsicGroups.begin(), state.intrinsicGroups.end(),
        [&](const auto& group)
        {
            return plamatrix_ba::activeIntrinsicParameters(
                       options, group.enabled, iteration) == group.enabled;
        });
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
    result.setupSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start).count();
    result.observationCount = active.observationCount;
    result.plaMatrixRejectedInitialTracks = active.rejectedInitialTracks;
    if (solver_backend == plamatrix::SchurComplementLinearBackend::Cpu &&
        active.primaryBlockCount <= std::max(1, options.maxDenseSchurCameras))
    {
        solver_backend = plamatrix::SchurComplementLinearBackend::DenseCpu;
    }
    result.plaMatrixLinearSolverName = plaMatrixLinearBackendName(solver_backend);
    if (active.observationCount == 0 && active.activeLaserRangeCount == 0)
    {
        result.solveStatus = BASolveStatus::InvalidInput;
        result.backendMessage = "PlaMatrix BA 没有足够有效的轨迹";
        result.totalSeconds = result.setupSeconds;
        return result;
    }

    const auto solve_start = std::chrono::steady_clock::now();
    plamatrix_ba::OptimizationState state = plamatrix_ba::initializeState(
        cameras, tracks, options, active);
    double current_cost = std::numeric_limits<double>::quiet_NaN();
    plamatrix::LevenbergMarquardtOptions<double> lm_options;
    lm_options.initialDamping = std::clamp(options.damping, 1e-12, 1e12);
    plamatrix::LevenbergMarquardtStrategy<double> lm(lm_options);
    bool converged = active.primaryBlockCount == 0 &&
                     active.trackBlockCount + active.laserBlockCount == 0;
    bool cancelled = false;
    int iterations = 0;
    plamatrix::SchurComplementSolverWorkspace<double> solver_workspace;

    try
    {
        if (converged)
        {
            current_cost = plamatrix_ba::evaluateObjective(
                cameras, tracks, options, active, state, 0);
            result.plaMatrixInitialCost = current_cost;
            ++result.plaMatrixObjectiveEvaluations;
        }

        const int iteration_budget = std::max(1, options.maxIterations);
        while (iterations < iteration_budget && !converged)
        {
            if (isCancelled(options))
            {
                cancelled = true;
                break;
            }

            const int stage_iteration = iterations;
            double linearized_cost = 0.0;
            const auto equations = plamatrix_ba::buildNormalEquations(
                cameras, tracks, options, active, state, stage_iteration,
                &linearized_cost);
            ++result.plaMatrixLinearizations;
            ++result.plaMatrixObjectiveEvaluations;
            current_cost = linearized_cost;
            if (result.plaMatrixLinearizations == 1)
            {
                result.plaMatrixInitialCost = current_cost;
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

                plamatrix::SchurComplementSolverOptions<double> solver_options;
                solver_options.linearBackend = solver_backend;
                solver_options.deviceIndex = options.plaMatrixDevice;
                solver_options.maxIterations =
                    (solver_backend == plamatrix::SchurComplementLinearBackend::Cpu ||
                     solver_backend == plamatrix::SchurComplementLinearBackend::DenseCpu)
                    ? std::max(50, active.primaryBlockCount * 12)
                    : std::max(200, active.primaryBlockCount * 30);
                if (stage_iteration < 2)
                {
                    solver_options.relativeTolerance = 1e-3;
                }
                else if (stage_iteration < 4)
                {
                    solver_options.relativeTolerance = 1e-5;
                }
                else
                {
                    solver_options.relativeTolerance = 1e-8;
                }
                if (iterations + 1 >= iteration_budget)
                {
                    solver_options.relativeTolerance = 1e-10;
                }
                solver_options.absoluteTolerance = 1e-12;
                solver_options.useInitialGuess = !retry_primary_step.empty();
                solver_options.useMixedPrecision =
                    options.enablePlaMatrixMixedPrecision &&
                    solver_backend == plamatrix::SchurComplementLinearBackend::Cuda &&
                    active.primaryBlockCount >= 50;
                std::vector<double> primary_step = retry_primary_step;
                std::vector<double> eliminated_step;
                const auto solver_report = plamatrix::solveDampedSchurComplement(
                    equations,
                    lm.damping(),
                    solver_options,
                    solver_workspace,
                    &primary_step,
                    &eliminated_step);
                result.plaMatrixLinearIterations += solver_report.iterations;
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
                result.plaMatrixSchurAssemblySeconds += solver_report.schurAssemblySeconds;
                result.plaMatrixLinearSolveSeconds += solver_report.linearSolveSeconds;
                if (!solver_report.deviceName.empty())
                {
                    result.plaMatrixDeviceName = solver_report.deviceName;
                    result.usedGpu = true;
                }
                result.plaMatrixSchurAssemblyOnDevice =
                    result.plaMatrixSchurAssemblyOnDevice ||
                    solver_report.schurAssemblyOnDevice;
                result.plaMatrixMixedPrecisionUsed =
                    result.plaMatrixMixedPrecisionUsed || solver_report.mixedPrecisionUsed;
                ++iterations;
                if (!solver_report.converged)
                {
                    retry_primary_step = primary_step;
                    lm.rejectStep();
                    continue;
                }

                const double step_norm = plamatrix_ba::maximumStepNorm(
                    primary_step, eliminated_step);
                if (step_norm <= options.stepTolerance)
                {
                    converged = isFinalIntrinsicStage(
                        options, state, stage_iteration);
                    relinearize = !converged;
                    continue;
                }

                auto candidate_state = state;
                plamatrix_ba::applyStep(
                    active, primary_step, eliminated_step, &candidate_state);
                const double candidate_cost = plamatrix_ba::evaluateObjective(
                    cameras, tracks, options, active, candidate_state,
                    stage_iteration);
                ++result.plaMatrixObjectiveEvaluations;
                if (std::isfinite(candidate_cost) && candidate_cost < current_cost)
                {
                    const double cost_change = current_cost - candidate_cost;
                    const double relative_cost_change =
                        cost_change / std::max(1.0, std::abs(current_cost));
                    state = std::move(candidate_state);
                    current_cost = candidate_cost;
                    lm.acceptStep();
                    converged = isFinalIntrinsicStage(
                                    options, state, stage_iteration) &&
                                relative_cost_change <= kRelativeCostTolerance;
                    relinearize = !converged;
                }
                else
                {
                    retry_primary_step = primary_step;
                    lm.rejectStep();
                }

                if (options.progressCallback)
                {
                    const double rms = std::sqrt(
                        std::max(0.0, current_cost) /
                        std::max(1, active.observationCount));
                    if (!options.progressCallback(
                            iterations,
                            iteration_budget,
                            rms,
                            active.activeTrackCount))
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

    result.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    result.plaMatrixAcceptedSteps = lm.acceptedSteps();
    result.plaMatrixRejectedSteps = lm.rejectedSteps();
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
    result.solutionUsable = std::isfinite(current_cost) &&
                            (converged || lm.acceptedSteps() > 0);
    result.backendMessage = converged
        ? "PlaMatrix 联合 BA 收敛（" + result.plaMatrixLinearSolverName + "）"
        : "PlaMatrix 联合 BA 达到迭代上限，返回可用解（" +
              result.plaMatrixLinearSolverName + "）";
    result.refinedCameras = state.cameras;
    plamatrix_ba::publishIntrinsics(
        cameras, options, active, state.intrinsicGroups, &result);
    result.refinedCameraCount = active.cameraBlockCount;
    if (!state.intrinsicGroups.empty())
    {
        result.selfCalibrationStagesRun = plamatrix_ba::intrinsicStageCount(
            options, state.intrinsicGroups.front().enabled);
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
    result.postprocessSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - postprocess_start).count();
    result.totalSeconds = result.setupSeconds + result.solveSeconds + result.postprocessSeconds;
    return result;
}

} // namespace xjw::detail
