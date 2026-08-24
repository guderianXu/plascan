#include "PlanetaryLineScanBundleAdjustInternal.h"

#include "PlanetaryLineScanBundleAdjustPlaMatrixAssembly.h"

#include <plamatrix/optimization/levenberg_marquardt.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace xjw
{
namespace lidar
{
namespace detail
{
namespace
{

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

plamatrix::SchurComplementLinearBackend linearBackend(BABackend backend)
{
    if (backend == BABackend::PlaMatrixCuda)
    {
        return plamatrix::SchurComplementLinearBackend::Cuda;
    }
    if (backend == BABackend::PlaMatrixOpenCl)
    {
        return plamatrix::SchurComplementLinearBackend::OpenCl;
    }
    return plamatrix::SchurComplementLinearBackend::Cpu;
}

const char *linearBackendName(plamatrix::SchurComplementLinearBackend backend)
{
    switch (backend)
    {
    case plamatrix::SchurComplementLinearBackend::Cpu:
        return "block_jacobi_pcg_cpu";
    case plamatrix::SchurComplementLinearBackend::DenseCpu:
        return "dense_cholesky_cpu";
    case plamatrix::SchurComplementLinearBackend::Cuda:
        return "block_jacobi_pcg_cuda";
    case plamatrix::SchurComplementLinearBackend::OpenCl:
        return "block_jacobi_pcg_opencl";
    }
    return "unknown";
}

} // namespace

bool solvePlanetaryLineScanBundleAdjustPlaMatrix(
    PlanetaryLineScanBaWorkingSet *workingSet,
    const PlanetaryLineScanBaOptions &options,
    PlanetaryLineScanBaResult *result,
    std::string *errorMessage)
{
    if (!workingSet || !result)
    {
        setError(errorMessage, "invalid planetary line-scan PlaMatrix working set");
        return false;
    }
    auto backend = linearBackend(options.backend);
    if (backend == plamatrix::SchurComplementLinearBackend::Cpu &&
        workingSet->cameraParameters.size() <=
            static_cast<std::size_t>(std::max(1, options.maxDenseSchurCameras)))
    {
        backend = plamatrix::SchurComplementLinearBackend::DenseCpu;
    }
    result->linearSolverName = linearBackendName(backend);
    const PlanetaryLineScanBaWorkingSet initialWorkingSet = *workingSet;
    const auto isCancelled = [&options]()
    {
        return options.cancelFlag &&
               options.cancelFlag->load(std::memory_order_relaxed);
    };
    if (isCancelled())
    {
        result->terminationType = "CANCELLED";
        result->backendMessage = "PlaMatrix line-scan BA cancelled before solving";
        setError(errorMessage, result->backendMessage);
        return false;
    }
    const std::vector<int> laserBlocks =
        plamatrix_linescan::makeLaserBlocks(*workingSet);
    double currentCost = plamatrix_linescan::evaluateObjective(*workingSet, options);
    plamatrix::LevenbergMarquardtStrategy<double> lm;
    plamatrix::SchurComplementSolverWorkspace<double> workspace;
    bool converged = false;
    try
    {
        int iteration = 0;
        while (iteration < options.maximumIterations && !converged)
        {
            if (isCancelled())
            {
                *workingSet = initialWorkingSet;
                result->terminationType = "CANCELLED";
                result->solutionUsable = false;
                result->backendMessage =
                    "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                setError(errorMessage, result->backendMessage);
                return false;
            }
            const auto equations = plamatrix_linescan::buildEquations(
                *workingSet, options, laserBlocks);
            bool relinearize = false;
            std::vector<double> retryPrimaryStep;
            while (iteration < options.maximumIterations && !converged && !relinearize)
            {
                if (isCancelled())
                {
                    *workingSet = initialWorkingSet;
                    result->terminationType = "CANCELLED";
                    result->solutionUsable = false;
                    result->backendMessage =
                        "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                    setError(errorMessage, result->backendMessage);
                    return false;
                }
                plamatrix::SchurComplementSolverOptions<double> solverOptions;
                solverOptions.linearBackend = backend;
                solverOptions.deviceIndex = options.plaMatrixDevice;
                solverOptions.maxIterations =
                    (backend == plamatrix::SchurComplementLinearBackend::Cpu ||
                     backend == plamatrix::SchurComplementLinearBackend::DenseCpu)
                    ? std::max(100, static_cast<int>(
                          workingSet->cameraParameters.size()) * 24)
                    : std::max(200, static_cast<int>(
                          workingSet->cameraParameters.size()) * 30);
                solverOptions.relativeTolerance = iteration < 2
                    ? 1.0e-3 : (iteration < 4 ? 1.0e-5 : 1.0e-8);
                if (iteration + 1 >= options.maximumIterations)
                {
                    solverOptions.relativeTolerance = 1.0e-10;
                }
                solverOptions.absoluteTolerance = 1.0e-12;
                solverOptions.useInitialGuess = !retryPrimaryStep.empty();
                std::vector<double> primaryStep = retryPrimaryStep;
                std::vector<double> eliminatedStep;
                auto report = plamatrix::solveDampedSchurComplement(
                    equations, lm.damping(), solverOptions, workspace,
                    &primaryStep, &eliminatedStep);
                if (!report.converged &&
                    backend == plamatrix::SchurComplementLinearBackend::DenseCpu)
                {
                    backend = plamatrix::SchurComplementLinearBackend::Cpu;
                    result->linearSolverName = linearBackendName(backend);
                    solverOptions.linearBackend = backend;
                    solverOptions.useInitialGuess = false;
                    primaryStep.clear();
                    eliminatedStep.clear();
                    report = plamatrix::solveDampedSchurComplement(
                        equations, lm.damping(), solverOptions, workspace,
                        &primaryStep, &eliminatedStep);
                }
                if (isCancelled())
                {
                    *workingSet = initialWorkingSet;
                    result->terminationType = "CANCELLED";
                    result->solutionUsable = false;
                    result->backendMessage =
                        "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                    setError(errorMessage, result->backendMessage);
                    return false;
                }
                ++iteration;
                ++result->iterations;
                if (!report.deviceName.empty())
                {
                    result->deviceName = report.deviceName;
                    result->usedGpu = true;
                }
                if (!report.converged)
                {
                    retryPrimaryStep = primaryStep;
                    lm.rejectStep();
                    continue;
                }
                if (plamatrix_linescan::maximumStepNorm(
                        primaryStep, eliminatedStep) <= 1.0e-10)
                {
                    converged = true;
                    break;
                }
                auto candidate = *workingSet;
                plamatrix_linescan::applyStep(
                    laserBlocks, primaryStep, eliminatedStep, &candidate);
                const double candidateCost =
                    plamatrix_linescan::evaluateObjective(candidate, options);
                if (std::isfinite(candidateCost) && candidateCost < currentCost)
                {
                    const double relativeChange = (currentCost - candidateCost) /
                        std::max(1.0, std::abs(currentCost));
                    *workingSet = std::move(candidate);
                    currentCost = candidateCost;
                    lm.acceptStep();
                    converged = relativeChange <= 1.0e-12;
                    relinearize = !converged;
                }
                else
                {
                    retryPrimaryStep = primaryStep;
                    lm.rejectStep();
                }
            }
        }
    }
    catch (const std::exception &error)
    {
        setError(errorMessage,
                 std::string("PlaMatrix line-scan BA failed: ") + error.what());
        result->terminationType = "NUMERICAL_FAILURE";
        result->backendMessage = errorMessage ? *errorMessage : error.what();
        return false;
    }
    result->converged = converged;
    result->solutionUsable = std::isfinite(currentCost) &&
        (converged || lm.acceptedSteps() > 0 || currentCost <= 1.0e-20);
    result->terminationType = converged ? "CONVERGENCE" : "NO_CONVERGENCE";
    result->solverBriefReport = "PlaMatrix " + result->linearSolverName +
        ": accepted=" + std::to_string(lm.acceptedSteps()) +
        ", rejected=" + std::to_string(lm.rejectedSteps());
    result->message = converged
        ? "planetary line-scan PlaMatrix bundle adjustment converged"
        : "planetary line-scan PlaMatrix bundle adjustment reached the iteration limit";
    result->backendMessage = result->message;
    if (!result->solutionUsable)
    {
        setError(errorMessage,
                 "PlaMatrix line-scan BA did not produce a usable solution");
        return false;
    }
    return true;
}

} // namespace detail
} // namespace lidar
} // namespace xjw
