#include "PlanetaryLineScanBundleAdjustPlaMatrixAssembly.h"

#include <plamatrix/optimization/robust_loss.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <omp.h>

namespace xjw
{
namespace lidar
{
namespace detail
{
namespace plamatrix_linescan
{
namespace
{

using Vector3 = PlanetaryLineScanCamera::Vector3;

bool linearizeImageObservation(
    const PlanetaryLineScanBaWorkingSet &workingSet,
    const LineScanImageObservation &observation,
    const PlanetaryLineScanBaOptions &options,
    double *residual,
    double *cameraJacobian,
    double *pointJacobian)
{
    const auto &camera = workingSet.cameraParameters[observation.cameraIndex];
    const auto &point = workingSet.tiePoints[observation.pointIndex];
    const auto &model = *workingSet.cameraModels[observation.cameraIndex];
    if (!evaluateLineScanImageObservation(
            model, observation, camera.data(), point.data(),
            options.imageSigmaPixels, residual))
    {
        return false;
    }
    for (int block = 0; block < 2; ++block)
    {
        const int blockSize = block == 0 ? 6 : 3;
        double *jacobian = block == 0 ? cameraJacobian : pointJacobian;
        for (int column = 0; column < blockSize; ++column)
        {
            const double step = block == 1
                ? options.finiteDifferencePointStepMeters
                : (column < 3
                       ? options.finiteDifferencePositionStepMeters
                       : options.finiteDifferenceAngleStepRadians);
            auto plusCamera = camera;
            auto minusCamera = camera;
            auto plusPoint = point;
            auto minusPoint = point;
            if (block == 0)
            {
                plusCamera[column] += step;
                minusCamera[column] -= step;
            }
            else
            {
                plusPoint[column] += step;
                minusPoint[column] -= step;
            }
            double plusResidual[2]{};
            double minusResidual[2]{};
            if (!evaluateLineScanImageObservation(
                    model, observation, plusCamera.data(), plusPoint.data(),
                    options.imageSigmaPixels, plusResidual) ||
                !evaluateLineScanImageObservation(
                    model, observation, minusCamera.data(), minusPoint.data(),
                    options.imageSigmaPixels, minusResidual))
            {
                return false;
            }
            for (int row = 0; row < 2; ++row)
            {
                jacobian[row * blockSize + column] =
                    (plusResidual[row] - minusResidual[row]) / (2.0 * step);
            }
        }
    }
    return true;
}

bool linearizeLaserRange(const std::array<double, 6> &camera,
                         const Vector3 &point,
                         const LineScanLaserObservation &observation,
                         double *residual,
                         double *cameraJacobian,
                         double *pointJacobian)
{
    const Vector3 delta{{
        point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
        point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
        point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
    const double range = std::hypot(std::hypot(delta[0], delta[1]), delta[2]);
    if (!(range > 1.0e-12) || !std::isfinite(range))
    {
        return false;
    }
    const double inverseSigma = 1.0 / observation.sigmaMeters;
    *residual = (range - observation.observedRangeMeters) * inverseSigma;
    std::fill_n(cameraJacobian, 6, 0.0);
    for (int axis = 0; axis < 3; ++axis)
    {
        pointJacobian[axis] = delta[axis] * inverseSigma / range;
        cameraJacobian[axis] = -pointJacobian[axis];
    }
    return std::isfinite(*residual);
}

void addPointPrior(const LineScanLaserPoint &laserPoint,
                   int laserBlock,
                   plamatrix::BlockNormalEquations<double> *equations)
{
    double residual[3]{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            residual[row] += laserPoint.sqrtInformation[row * 3 + column] *
                (laserPoint.refinedBodyFixedMeters[column] -
                 laserPoint.initialBodyFixedMeters[column]);
        }
    }
    equations->addEliminatedResidualBlock(
        laserBlock, laserPoint.sqrtInformation.data(), residual, 3);
}

double evaluateImageRange(const PlanetaryLineScanBaWorkingSet &workingSet,
                          const PlanetaryLineScanBaOptions &options,
                          std::size_t begin,
                          std::size_t end)
{
    const double imageDelta = options.imageHuberDeltaPixels /
                              options.imageSigmaPixels;
    double cost = 0.0;
    for (std::size_t index = begin; index < end; ++index)
    {
        const auto &observation = workingSet.imageObservations[index];
        double residual[2]{};
        if (!evaluateLineScanImageObservation(
                *workingSet.cameraModels[observation.cameraIndex], observation,
                workingSet.cameraParameters[observation.cameraIndex].data(),
                workingSet.tiePoints[observation.pointIndex].data(),
                options.imageSigmaPixels, residual))
        {
            return std::numeric_limits<double>::infinity();
        }
        cost += plamatrix::evaluateHuberLoss(
            residual[0] * residual[0] + residual[1] * residual[1], imageDelta).cost;
    }
    return cost;
}

void assembleImageRange(const PlanetaryLineScanBaWorkingSet &workingSet,
                        const PlanetaryLineScanBaOptions &options,
                        std::size_t begin,
                        std::size_t end,
                        plamatrix::BlockNormalEquations<double> *equations)
{
    const double imageDelta = options.imageHuberDeltaPixels /
                              options.imageSigmaPixels;
    for (std::size_t index = begin; index < end; ++index)
    {
        const auto &observation = workingSet.imageObservations[index];
        double residual[2]{};
        double cameraJacobian[12]{};
        double pointJacobian[6]{};
        if (!linearizeImageObservation(
                workingSet, observation, options, residual,
                cameraJacobian, pointJacobian))
        {
            throw std::runtime_error(
                "line-scan image projection failed during linearization");
        }
        const auto robust = plamatrix::evaluateHuberLoss(
            residual[0] * residual[0] + residual[1] * residual[1], imageDelta);
        equations->addResidualBlock(
            observation.cameraIndex, observation.pointIndex,
            cameraJacobian, pointJacobian, residual, 2, robust.weight);
    }
}

} // namespace

std::vector<int> makeLaserBlocks(const PlanetaryLineScanBaWorkingSet &workingSet)
{
    std::vector<int> blocks(workingSet.laserPoints.size(), -1);
    int nextBlock = static_cast<int>(workingSet.tiePoints.size());
    for (std::size_t index = 0; index < workingSet.laserPoints.size(); ++index)
    {
        if (workingSet.laserPoints[index].pointMode ==
            PlanetaryLaserPointMode::Constrained)
        {
            blocks[index] = nextBlock++;
        }
    }
    return blocks;
}

double evaluateObjective(const PlanetaryLineScanBaWorkingSet &workingSet,
                         const PlanetaryLineScanBaOptions &options)
{
    const int requestedThreads = options.threadCount > 0
        ? options.threadCount : omp_get_max_threads();
    const int threadCount = std::min<int>(
        std::max(1, requestedThreads),
        static_cast<int>(workingSet.imageObservations.size()));
    double cost = 0.0;
    if (threadCount <= 1 || workingSet.imageObservations.size() < 128)
    {
        cost = evaluateImageRange(
            workingSet, options, 0, workingSet.imageObservations.size());
    }
    else
    {
        std::vector<double> partialCosts(static_cast<std::size_t>(threadCount), 0.0);
#pragma omp parallel for num_threads(threadCount) schedule(static, 1)
        for (int thread = 0; thread < threadCount; ++thread)
        {
            const std::size_t begin = workingSet.imageObservations.size() *
                static_cast<std::size_t>(thread) / static_cast<std::size_t>(threadCount);
            const std::size_t end = workingSet.imageObservations.size() *
                static_cast<std::size_t>(thread + 1) / static_cast<std::size_t>(threadCount);
            partialCosts[static_cast<std::size_t>(thread)] =
                evaluateImageRange(workingSet, options, begin, end);
        }
        cost = std::accumulate(partialCosts.begin(), partialCosts.end(), 0.0);
    }
    const double angleSigma = options.cameraAngleSigmaDegrees *
                              std::acos(-1.0) / 180.0;
    for (const auto &camera : workingSet.cameraParameters)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            cost += 0.5 * std::pow(camera[axis] /
                                   options.cameraPositionSigmaMeters, 2.0);
            cost += 0.5 * std::pow(camera[axis + 3] / angleSigma, 2.0);
        }
    }
    if (!options.enableLaserRangeConstraints)
    {
        return cost;
    }
    for (const auto &observation : workingSet.laserObservations)
    {
        const auto &laserPoint = workingSet.laserPoints[observation.laserPointIndex];
        double residual = 0.0;
        double cameraJacobian[6]{};
        double pointJacobian[3]{};
        if (!linearizeLaserRange(
                workingSet.cameraParameters[observation.cameraIndex],
                laserPoint.refinedBodyFixedMeters, observation,
                &residual, cameraJacobian, pointJacobian))
        {
            return std::numeric_limits<double>::infinity();
        }
        cost += options.laserRangeWeight * plamatrix::evaluateHuberLoss(
            residual * residual, options.laserRangeHuberDeltaSigma).cost;
        if (laserPoint.pointMode == PlanetaryLaserPointMode::Constrained)
        {
            double prior[3]{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    prior[row] += laserPoint.sqrtInformation[row * 3 + column] *
                        (laserPoint.refinedBodyFixedMeters[column] -
                         laserPoint.initialBodyFixedMeters[column]);
                }
                cost += 0.5 * prior[row] * prior[row];
            }
        }
    }
    return cost;
}

plamatrix::BlockNormalEquations<double> buildEquations(
    const PlanetaryLineScanBaWorkingSet &workingSet,
    const PlanetaryLineScanBaOptions &options,
    const std::vector<int> &laserBlocks)
{
    const int constrainedLaserCount = static_cast<int>(std::count_if(
        laserBlocks.begin(), laserBlocks.end(), [](int block)
        {
            return block >= 0;
        }));
    plamatrix::BlockNormalEquations<double> equations(
        workingSet.cameraParameters.size(),
        workingSet.tiePoints.size() + constrainedLaserCount,
        6,
        3);
    const int requestedThreads = options.threadCount > 0
        ? options.threadCount : omp_get_max_threads();
    const int threadCount = std::min<int>(
        std::max(1, requestedThreads),
        static_cast<int>(workingSet.imageObservations.size()));
    if (threadCount <= 1 || workingSet.imageObservations.size() < 128)
    {
        assembleImageRange(
            workingSet, options, 0, workingSet.imageObservations.size(), &equations);
    }
    else
    {
        std::vector<std::unique_ptr<plamatrix::BlockNormalEquations<double>>> partials;
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threadCount));
        partials.reserve(static_cast<std::size_t>(threadCount));
        for (int thread = 0; thread < threadCount; ++thread)
        {
            partials.push_back(std::make_unique<plamatrix::BlockNormalEquations<double>>(
                workingSet.cameraParameters.size(),
                workingSet.tiePoints.size() + constrainedLaserCount, 6, 3));
        }
#pragma omp parallel for num_threads(threadCount) schedule(static, 1)
        for (int thread = 0; thread < threadCount; ++thread)
        {
            const std::size_t begin = workingSet.imageObservations.size() *
                static_cast<std::size_t>(thread) / static_cast<std::size_t>(threadCount);
            const std::size_t end = workingSet.imageObservations.size() *
                static_cast<std::size_t>(thread + 1) / static_cast<std::size_t>(threadCount);
            try
            {
                assembleImageRange(
                    workingSet, options, begin, end,
                    partials[static_cast<std::size_t>(thread)].get());
            }
            catch (...)
            {
                errors[static_cast<std::size_t>(thread)] = std::current_exception();
            }
        }
        for (int thread = 0; thread < threadCount; ++thread)
        {
            const auto index = static_cast<std::size_t>(thread);
            if (errors[index])
            {
                std::rethrow_exception(errors[index]);
            }
            equations.mergeFrom(*partials[index]);
        }
    }
    const double angleSigma = options.cameraAngleSigmaDegrees *
                              std::acos(-1.0) / 180.0;
    for (std::size_t index = 0; index < workingSet.cameraParameters.size(); ++index)
    {
        double residual[6]{};
        double jacobian[36]{};
        for (int axis = 0; axis < 6; ++axis)
        {
            const double inverseSigma = 1.0 /
                (axis < 3 ? options.cameraPositionSigmaMeters : angleSigma);
            residual[axis] = workingSet.cameraParameters[index][axis] * inverseSigma;
            jacobian[axis * 6 + axis] = inverseSigma;
        }
        equations.addPrimaryResidualBlock(index, jacobian, residual, 6);
    }
    if (!options.enableLaserRangeConstraints)
    {
        return equations;
    }
    for (const auto &observation : workingSet.laserObservations)
    {
        const auto &laserPoint = workingSet.laserPoints[observation.laserPointIndex];
        double residual = 0.0;
        double cameraJacobian[6]{};
        double pointJacobian[3]{};
        if (!linearizeLaserRange(
                workingSet.cameraParameters[observation.cameraIndex],
                laserPoint.refinedBodyFixedMeters, observation,
                &residual, cameraJacobian, pointJacobian))
        {
            throw std::runtime_error("line-scan laser range is numerically invalid");
        }
        const auto robust = plamatrix::evaluateHuberLoss(
            residual * residual, options.laserRangeHuberDeltaSigma);
        const double weight = options.laserRangeWeight * robust.weight;
        const int laserBlock = laserBlocks[observation.laserPointIndex];
        if (laserBlock >= 0)
        {
            equations.addResidualBlock(
                observation.cameraIndex, laserBlock,
                cameraJacobian, pointJacobian, &residual, 1, weight);
            addPointPrior(laserPoint, laserBlock, &equations);
        }
        else
        {
            equations.addPrimaryResidualBlock(
                observation.cameraIndex, cameraJacobian, &residual, 1, weight);
        }
    }
    return equations;
}

double maximumStepNorm(const std::vector<double> &primaryStep,
                       const std::vector<double> &eliminatedStep)
{
    double maximum = 0.0;
    for (double value : primaryStep)
    {
        maximum = std::max(maximum, std::abs(value));
    }
    for (double value : eliminatedStep)
    {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

void applyStep(const std::vector<int> &laserBlocks,
               const std::vector<double> &primaryStep,
               const std::vector<double> &eliminatedStep,
               PlanetaryLineScanBaWorkingSet *workingSet)
{
    for (std::size_t block = 0; block < workingSet->cameraParameters.size(); ++block)
    {
        for (int axis = 0; axis < 6; ++axis)
        {
            workingSet->cameraParameters[block][axis] += primaryStep[block * 6 + axis];
        }
    }
    for (std::size_t block = 0; block < workingSet->tiePoints.size(); ++block)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            workingSet->tiePoints[block][axis] += eliminatedStep[block * 3 + axis];
        }
    }
    for (std::size_t index = 0; index < laserBlocks.size(); ++index)
    {
        if (laserBlocks[index] < 0)
        {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            workingSet->laserPoints[index].refinedBodyFixedMeters[axis] +=
                eliminatedStep[laserBlocks[index] * 3 + axis];
        }
    }
}

} // namespace plamatrix_linescan
} // namespace detail
} // namespace lidar
} // namespace xjw
