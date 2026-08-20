#include "PlanetaryLineScanBundleAdjustInternal.h"

#include <algorithm>
#include <cmath>
#include <thread>

#ifdef PLASCAN_LIDAR_HAS_CERES
#include <ceres/ceres.h>
#endif

namespace xjw
{
namespace lidar
{
namespace detail
{
namespace
{

using Vector3 = PlanetaryLineScanCamera::Vector3;

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

#ifdef PLASCAN_LIDAR_HAS_CERES

class LineScanImageCost final : public ceres::CostFunction
{
public:
    LineScanImageCost(const PlanetaryLineScanCamera *camera,
                      LineScanImageObservation observation,
                      const PlanetaryLineScanBaOptions &options)
        : _camera(camera),
          _observation(observation),
          _imageSigmaPixels(options.imageSigmaPixels),
          _pointStepMeters(options.finiteDifferencePointStepMeters),
          _positionStepMeters(options.finiteDifferencePositionStepMeters),
          _angleStepRadians(options.finiteDifferenceAngleStepRadians)
    {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(3);
    }

    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const override
    {
        if (!evaluateLineScanImageObservation(
                *_camera, _observation, parameters[0], parameters[1],
                _imageSigmaPixels, residuals))
        {
            return false;
        }
        if (!jacobians)
        {
            return true;
        }
        if (jacobians[0] && !differentiateBlock(parameters, 0, 6, jacobians[0]))
        {
            return false;
        }
        if (jacobians[1] && !differentiateBlock(parameters, 1, 3, jacobians[1]))
        {
            return false;
        }
        return true;
    }

private:
    bool differentiateBlock(double const *const *parameters,
                            int blockIndex,
                            int blockSize,
                            double *jacobian) const
    {
        std::array<double, 6> camera{};
        std::array<double, 3> point{};
        std::copy_n(parameters[0], camera.size(), camera.begin());
        std::copy_n(parameters[1], point.size(), point.begin());
        for (int column = 0; column < blockSize; ++column)
        {
            const double step = blockIndex == 1
                ? _pointStepMeters
                : (column < 3 ? _positionStepMeters : _angleStepRadians);
            if (!(step > 0.0))
            {
                return false;
            }
            std::array<double, 6> plusCamera = camera;
            std::array<double, 6> minusCamera = camera;
            std::array<double, 3> plusPoint = point;
            std::array<double, 3> minusPoint = point;
            if (blockIndex == 0)
            {
                plusCamera[column] += step;
                minusCamera[column] -= step;
            }
            else
            {
                plusPoint[column] += step;
                minusPoint[column] -= step;
            }
            double plusResiduals[2]{};
            double minusResiduals[2]{};
            if (!evaluateLineScanImageObservation(
                    *_camera, _observation, plusCamera.data(), plusPoint.data(),
                    _imageSigmaPixels, plusResiduals) ||
                !evaluateLineScanImageObservation(
                    *_camera, _observation, minusCamera.data(), minusPoint.data(),
                    _imageSigmaPixels, minusResiduals))
            {
                return false;
            }
            for (int row = 0; row < 2; ++row)
            {
                jacobian[row * blockSize + column] =
                    (plusResiduals[row] - minusResiduals[row]) / (2.0 * step);
            }
        }
        return true;
    }

    const PlanetaryLineScanCamera *_camera = nullptr;
    LineScanImageObservation _observation;
    double _imageSigmaPixels = 1.0;
    double _pointStepMeters = 0.05;
    double _positionStepMeters = 0.05;
    double _angleStepRadians = 1.0e-7;
};

struct CameraPriorCost
{
    double inversePositionSigma = 1.0;
    double inverseAngleSigma = 1.0;

    template <typename T>
    bool operator()(const T *parameters, T *residuals) const
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            residuals[axis] = T(inversePositionSigma) * parameters[axis];
            residuals[axis + 3] = T(inverseAngleSigma) * parameters[axis + 3];
        }
        return true;
    }
};

struct LaserRangeCost
{
    Vector3 nominalCenterMeters{};
    double observedRangeMeters = 0.0;
    double scale = 1.0;

    template <typename T>
    bool operator()(const T *cameraParameters, const T *point, T *residual) const
    {
        const T dx = point[0] - T(nominalCenterMeters[0]) - cameraParameters[0];
        const T dy = point[1] - T(nominalCenterMeters[1]) - cameraParameters[1];
        const T dz = point[2] - T(nominalCenterMeters[2]) - cameraParameters[2];
        residual[0] = T(scale) *
            (ceres::sqrt(dx * dx + dy * dy + dz * dz) - T(observedRangeMeters));
        return true;
    }
};

struct PointPriorCost
{
    Vector3 initial{};
    std::array<double, 9> sqrtInformation{};

    template <typename T>
    bool operator()(const T *point, T *residuals) const
    {
        T delta[3]{};
        for (int axis = 0; axis < 3; ++axis)
        {
            delta[axis] = point[axis] - T(initial[axis]);
        }
        for (int row = 0; row < 3; ++row)
        {
            residuals[row] = T(0.0);
            for (int column = 0; column < 3; ++column)
            {
                residuals[row] += T(sqrtInformation[row * 3 + column]) * delta[column];
            }
        }
        return true;
    }
};

const char *terminationTypeName(ceres::TerminationType type)
{
    switch (type)
    {
        case ceres::CONVERGENCE:
            return "CONVERGENCE";
        case ceres::NO_CONVERGENCE:
            return "NO_CONVERGENCE";
        case ceres::FAILURE:
            return "FAILURE";
        case ceres::USER_SUCCESS:
            return "USER_SUCCESS";
        case ceres::USER_FAILURE:
            return "USER_FAILURE";
        default:
            return "UNKNOWN";
    }
}

#endif

} // namespace

bool solvePlanetaryLineScanBundleAdjustCeres(
    PlanetaryLineScanBaWorkingSet *workingSet,
    const PlanetaryLineScanBaOptions &options,
    PlanetaryLineScanBaResult *result,
    std::string *errorMessage)
{
#ifndef PLASCAN_LIDAR_HAS_CERES
    (void)workingSet;
    (void)options;
    (void)result;
    setError(errorMessage, "planetary line-scan BA requires a PlaScan build with Ceres");
    return false;
#else
    if (!workingSet || !result)
    {
        setError(errorMessage, "invalid planetary line-scan Ceres working set");
        return false;
    }

    ceres::Problem problem;
    for (const LineScanImageObservation &observation : workingSet->imageObservations)
    {
        auto *cost = new LineScanImageCost(
            workingSet->cameraModels[observation.cameraIndex], observation, options);
        ceres::LossFunction *loss = options.imageHuberDeltaPixels > 0.0
            ? static_cast<ceres::LossFunction *>(new ceres::HuberLoss(
                  options.imageHuberDeltaPixels / options.imageSigmaPixels))
            : nullptr;
        problem.AddResidualBlock(cost, loss,
                                 workingSet->cameraParameters[observation.cameraIndex].data(),
                                 workingSet->tiePoints[observation.pointIndex].data());
    }

    const double angleSigmaRadians =
        options.cameraAngleSigmaDegrees * std::acos(-1.0) / 180.0;
    for (auto &camera : workingSet->cameraParameters)
    {
        auto *cost = new ceres::AutoDiffCostFunction<CameraPriorCost, 6, 6>(
            new CameraPriorCost{1.0 / options.cameraPositionSigmaMeters,
                                1.0 / angleSigmaRadians});
        problem.AddResidualBlock(cost, nullptr, camera.data());
    }

    if (options.enableLaserRangeConstraints)
    {
        for (const LineScanLaserObservation &observation : workingSet->laserObservations)
        {
            auto &laserPoint = workingSet->laserPoints[observation.laserPointIndex];
            auto *rangeCost = new ceres::AutoDiffCostFunction<LaserRangeCost, 1, 6, 3>(
                new LaserRangeCost{observation.nominalSensorCenterMeters,
                                   observation.observedRangeMeters,
                                   1.0 / observation.sigmaMeters});
            ceres::LossFunction *baseLoss = options.laserRangeHuberDeltaSigma > 0.0
                ? static_cast<ceres::LossFunction *>(new ceres::HuberLoss(
                      options.laserRangeHuberDeltaSigma))
                : nullptr;
            // Keep the Huber transition in normalized-sigma units. Scaling the
            // residual itself would make the physical transition depend on weight.
            ceres::LossFunction *loss = new ceres::ScaledLoss(
                baseLoss, options.laserRangeWeight, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(
                rangeCost, loss,
                workingSet->cameraParameters[observation.cameraIndex].data(),
                laserPoint.refinedBodyFixedMeters.data());

            if (laserPoint.pointMode == PlanetaryLaserPointMode::Constrained)
            {
                auto *priorCost = new ceres::AutoDiffCostFunction<PointPriorCost, 3, 3>(
                    new PointPriorCost{laserPoint.initialBodyFixedMeters,
                                       laserPoint.sqrtInformation});
                problem.AddResidualBlock(priorCost, nullptr,
                                         laserPoint.refinedBodyFixedMeters.data());
            }
            else if (laserPoint.pointMode == PlanetaryLaserPointMode::Fixed)
            {
                problem.SetParameterBlockConstant(laserPoint.refinedBodyFixedMeters.data());
            }
        }
    }

    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = options.maximumIterations;
    solverOptions.linear_solver_type = ceres::DENSE_SCHUR;
    solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solverOptions.num_threads = options.threadCount > 0
        ? options.threadCount
        : std::max(1u, std::thread::hardware_concurrency());
    solverOptions.function_tolerance = 1.0e-12;
    solverOptions.gradient_tolerance = 1.0e-12;
    solverOptions.parameter_tolerance = 1.0e-10;
    solverOptions.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);
    result->iterations = static_cast<int>(summary.iterations.size());
    result->solverBriefReport = summary.BriefReport();
    result->solutionUsable = summary.IsSolutionUsable();
    result->converged = summary.termination_type == ceres::CONVERGENCE ||
                        summary.termination_type == ceres::USER_SUCCESS;
    result->terminationType = terminationTypeName(summary.termination_type);
    result->message = result->converged
        ? "planetary line-scan bundle adjustment converged"
        : (result->solutionUsable
               ? "planetary line-scan bundle adjustment produced a usable non-converged solution"
               : summary.FullReport());
    if (!result->solutionUsable)
    {
        setError(errorMessage, "Ceres line-scan BA failed: " + summary.BriefReport());
        return false;
    }
    return true;
#endif
}

} // namespace detail
} // namespace lidar
} // namespace xjw
