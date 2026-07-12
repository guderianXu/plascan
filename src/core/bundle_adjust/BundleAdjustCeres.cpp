#include "BundleAdjustCeres.h"

#include "BundleAdjustProjection.h"

#include <plamatrix/ops/vector.h>

#ifdef PLASCAN_BA_HAS_CERES
#  include <ceres/ceres.h>
#  include <ceres/internal/config.h>
#  if !defined(CERES_NO_CUDA) && __has_include(<cuda_runtime_api.h>)
#    include <cuda_runtime_api.h>
#    define PLASCAN_BA_HAS_CUDA_RUNTIME_API 1
#  endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>

namespace xjw::ba
{

ProjectionCamera makeProjectionCamera(const Camera &camera)
{
    ProjectionCamera out;
    const Camera::Intrinsics intrinsics = camera.intrinsics();
    const Camera::Distortion distortion = camera.distortion();
    out.cameraToWorldRotation = camera.cameraToWorldRotation();
    out.cameraCenter = camera.cameraCenter();
    out.focalX = intrinsics.focalX;
    out.focalY = intrinsics.focalY;
    out.principalX = intrinsics.principalX;
    out.principalY = intrinsics.principalY;
    out.radialK1 = distortion.radialK1;
    out.radialK2 = distortion.radialK2;
    out.radialK3 = distortion.radialK3;
    out.tangentialP1 = distortion.tangentialP1;
    out.tangentialP2 = distortion.tangentialP2;
    out.uAxisSign = intrinsics.uAxisSign;
    out.vAxisSign = intrinsics.vAxisSign;
    out.depthAxisFlipped = camera.depthAxisFlipped();
    return out;
}

} // namespace xjw::ba

namespace xjw::detail
{
namespace
{

double safeObservationWeight(const BAObservation &observation)
{
    return std::isfinite(observation.weight) ? std::max(0.0, observation.weight) : 1.0;
}

bool projectWithDelta(const Camera &baseCamera,
                      const double *cameraDelta,
                      const double *point,
                      double *pixel)
{
    Camera camera = baseCamera;
    camera.applyDeltaPose(cameraDelta);
    const double world[3] = {point[0], point[1], point[2]};
    return camera.projectWorldPoint(world, pixel);
}

double computeTrackRms(const std::vector<Camera> &cameras,
                       const BATrack &track,
                       const std::array<double, 3> &point)
{
    double sum = 0.0;
    int count = 0;
    for (const BAObservation &observation : track.observations)
    {
        if (observation.cameraIndex < 0 ||
            observation.cameraIndex >= static_cast<int>(cameras.size()))
        {
            continue;
        }

        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        if (!cameras[static_cast<size_t>(observation.cameraIndex)].projectWorldPoint(world, pixel))
        {
            continue;
        }

        const double du = pixel[0] - observation.u;
        const double dv = pixel[1] - observation.v;
        const double weight = safeObservationWeight(observation);
        sum += weight * (du * du + dv * dv);
        count += 2;
    }
    return count > 0 ? std::sqrt(sum / static_cast<double>(count)) : std::numeric_limits<double>::infinity();
}

bool isCameraFixed(int cameraIndex, const BAOptions &options)
{
    return std::find(options.fixedCameraIndices.begin(),
                     options.fixedCameraIndices.end(),
                     cameraIndex) != options.fixedCameraIndices.end();
}

struct ScalarDistanceStats
{
    int count = 0;
    double rms = 0.0;
    double median = 0.0;
};

std::array<double, 3> pointForStats(const std::vector<BATrack> &tracks,
                                    const std::vector<BARefinedPoint> *points,
                                    size_t index)
{
    if (points && index < points->size() && (*points)[index].valid)
    {
        return (*points)[index].point;
    }
    return tracks[index].initialPoint;
}

ScalarDistanceStats computeLaserStats(const std::vector<BATrack> &tracks,
                                      const std::vector<BARefinedPoint> *points)
{
    std::vector<double> absDistances;
    double sum2 = 0.0;
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        const auto point = pointForStats(tracks, points, ti);
        for (const BALaserPlaneConstraint &constraint : tracks[ti].laserPlaneConstraints)
        {
            const double dx = point[0] - constraint.point[0];
            const double dy = point[1] - constraint.point[1];
            const double dz = point[2] - constraint.point[2];
            const double signedDistance =
                dx * constraint.normal[0] + dy * constraint.normal[1] + dz * constraint.normal[2];
            const double absDistance = std::abs(signedDistance);
            sum2 += absDistance * absDistance;
            absDistances.push_back(absDistance);
        }
    }

    ScalarDistanceStats stats;
    stats.count = static_cast<int>(absDistances.size());
    stats.rms = stats.count > 0 ? std::sqrt(sum2 / static_cast<double>(stats.count)) : 0.0;
    if (!absDistances.empty())
    {
        std::sort(absDistances.begin(), absDistances.end());
        const size_t mid = absDistances.size() / 2;
        stats.median = (absDistances.size() % 2 == 0)
                           ? 0.5 * (absDistances[mid - 1] + absDistances[mid])
                           : absDistances[mid];
    }
    return stats;
}

ScalarDistanceStats computeControlPointStats(const std::vector<BATrack> &tracks,
                                             const std::vector<BARefinedPoint> *points)
{
    double sum2 = 0.0;
    int count = 0;
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        const auto point = pointForStats(tracks, points, ti);
        for (const BAControlPointConstraint &constraint : tracks[ti].controlPointConstraints)
        {
            const double dx = point[0] - constraint.point[0];
            const double dy = point[1] - constraint.point[1];
            const double dz = point[2] - constraint.point[2];
            sum2 += dx * dx + dy * dy + dz * dz;
            ++count;
        }
    }

    ScalarDistanceStats stats;
    stats.count = count;
    stats.rms = count > 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
    return stats;
}

ScalarDistanceStats computeScaleBarStats(const std::vector<BATrack> &tracks,
                                         const std::vector<BARefinedPoint> *points,
                                         const std::vector<BAScaleBarConstraint> &constraints)
{
    double sum2 = 0.0;
    int count = 0;
    for (const BAScaleBarConstraint &constraint : constraints)
    {
        if (constraint.trackIndexA < 0 || constraint.trackIndexB < 0 ||
            constraint.trackIndexA >= static_cast<int>(tracks.size()) ||
            constraint.trackIndexB >= static_cast<int>(tracks.size()))
        {
            continue;
        }
        const auto pointA = pointForStats(tracks, points, static_cast<size_t>(constraint.trackIndexA));
        const auto pointB = pointForStats(tracks, points, static_cast<size_t>(constraint.trackIndexB));
        const double dx = pointA[0] - pointB[0];
        const double dy = pointA[1] - pointB[1];
        const double dz = pointA[2] - pointB[2];
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double error = distance - constraint.measuredDistanceMeters;
        sum2 += error * error;
        ++count;
    }

    ScalarDistanceStats stats;
    stats.count = count;
    stats.rms = count > 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
    return stats;
}

std::array<double, 9> relativeRotationCurrentToPrior(const std::array<double, 9> &current,
                                                     const std::array<double, 9> &prior)
{
    std::array<double, 9> relative{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                sum += current[r * 3 + k] * prior[c * 3 + k];
            }
            relative[r * 3 + c] = sum;
        }
    }
    return relative;
}

std::array<double, 3> rotationLogVector(const std::array<double, 9> &rotation)
{
    const double trace = rotation[0] + rotation[4] + rotation[8];
    const double cosAngle = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
    const double angle = std::acos(cosAngle);
    const std::array<double, 3> vee{{
        rotation[7] - rotation[5],
        rotation[2] - rotation[6],
        rotation[3] - rotation[1],
    }};
    if (angle < 1e-10)
    {
        return {{0.5 * vee[0], 0.5 * vee[1], 0.5 * vee[2]}};
    }
    const double sinAngle = std::sin(angle);
    if (std::abs(sinAngle) < 1e-10)
    {
        return {{0.0, 0.0, 0.0}};
    }
    const double scale = angle / (2.0 * sinAngle);
    return {{scale * vee[0], scale * vee[1], scale * vee[2]}};
}

int countObservations(const std::vector<BATrack> &tracks)
{
    int count = 0;
    for (const BATrack &track : tracks)
    {
        count += static_cast<int>(track.observations.size());
    }
    return count;
}

#ifdef PLASCAN_BA_HAS_CERES

struct FixedCameraReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight = T(std::sqrt(safeObservationWeight(observation)));
        if (!xjw::ba::project(camera, point, pixel))
        {
            residuals[0] = sqrtWeight * T(1.0e6);
            residuals[1] = sqrtWeight * T(1.0e6);
            return true;
        }
        residuals[0] = sqrtWeight * (pixel[0] - T(observation.u));
        residuals[1] = sqrtWeight * (pixel[1] - T(observation.v));
        return true;
    }
};

struct PoseDeltaReprojectionResidual
{
    Camera camera;
    BAObservation observation;

    bool operator()(const double *const cameraDelta,
                    const double *const point,
                    double *residuals) const
    {
        double pixel[2] = {0.0, 0.0};
        const double sqrtWeight = std::sqrt(safeObservationWeight(observation));
        if (!projectWithDelta(camera, cameraDelta, point, pixel))
        {
            residuals[0] = sqrtWeight * 1.0e6;
            residuals[1] = sqrtWeight * 1.0e6;
            return true;
        }
        residuals[0] = sqrtWeight * (pixel[0] - observation.u);
        residuals[1] = sqrtWeight * (pixel[1] - observation.v);
        return true;
    }
};

struct LaserPlaneResidual
{
    BALaserPlaneConstraint constraint;
    double weight = 1.0;

    bool operator()(const double *const point, double *residuals) const
    {
        const double dx = point[0] - constraint.point[0];
        const double dy = point[1] - constraint.point[1];
        const double dz = point[2] - constraint.point[2];
        const double signedDistance =
            dx * constraint.normal[0] + dy * constraint.normal[1] + dz * constraint.normal[2];
        residuals[0] = std::sqrt(std::max(0.0, weight * constraint.weight)) * signedDistance;
        return true;
    }
};

struct ControlPointResidual
{
    BAControlPointConstraint constraint;
    double weight = 1.0;

    bool operator()(const double *const point, double *residuals) const
    {
        const double sigma = std::max(1e-9, constraint.sigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight * constraint.weight)) / sigma;
        residuals[0] = scale * (point[0] - constraint.point[0]);
        residuals[1] = scale * (point[1] - constraint.point[1]);
        residuals[2] = scale * (point[2] - constraint.point[2]);
        return true;
    }
};

struct ScaleBarResidual
{
    BAScaleBarConstraint constraint;
    double weight = 1.0;

    bool operator()(const double *const pointA,
                    const double *const pointB,
                    double *residuals) const
    {
        const double dx = pointA[0] - pointB[0];
        const double dy = pointA[1] - pointB[1];
        const double dz = pointA[2] - pointB[2];
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double sigma = std::max(1e-9, constraint.sigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight * constraint.weight)) / sigma;
        residuals[0] = scale * (distance - constraint.measuredDistanceMeters);
        return true;
    }
};

struct PosePriorResidual
{
    Camera camera;
    BACameraPosePrior prior;
    double weight = 1.0;

    bool operator()(const double *const cameraDelta, double *residuals) const
    {
        Camera current = camera;
        current.applyDeltaPose(cameraDelta);
        const double rotationSigma = std::max(1e-9, prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0);
        const double positionSigma = std::max(1e-9, prior.positionSigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight));
        const auto relative =
            relativeRotationCurrentToPrior(current.cameraToWorldRotation(), prior.cameraToWorldRotation);
        const auto rotationResidual = rotationLogVector(relative);
        const auto center = current.cameraCenter();
        residuals[0] = scale * rotationResidual[0] / rotationSigma;
        residuals[1] = scale * rotationResidual[1] / rotationSigma;
        residuals[2] = scale * rotationResidual[2] / rotationSigma;
        residuals[3] = scale * (center[0] - prior.cameraCenter[0]) / positionSigma;
        residuals[4] = scale * (center[1] - prior.cameraCenter[1]) / positionSigma;
        residuals[5] = scale * (center[2] - prior.cameraCenter[2]) / positionSigma;
        return true;
    }
};

ceres::LossFunction *makeHuberLoss(double delta)
{
    return delta > 0.0 ? new ceres::HuberLoss(delta) : nullptr;
}

class CeresBaIterationCallback final : public ceres::IterationCallback
{
public:
    explicit CeresBaIterationCallback(const BAOptions &options)
        : _options(options)
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override
    {
        if (_options.cancelFlag && _options.cancelFlag->load())
        {
            return ceres::SOLVER_ABORT;
        }
        if (_options.progressCallback)
        {
            const double rmsProxy = std::sqrt(std::max(0.0, summary.cost));
            if (!_options.progressCallback(summary.iteration + 1,
                                           std::max(1, _options.maxIterations),
                                           rmsProxy,
                                           0))
            {
                return ceres::SOLVER_ABORT;
            }
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    const BAOptions &_options;
};

bool selectCeresCudaDevice(const BAOptions &options, std::string *message)
{
#  if defined(PLASCAN_BA_HAS_CUDA_RUNTIME_API)
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess || deviceCount <= 0)
    {
        if (message)
        {
            *message = std::string("CUDA runtime 未发现可用设备: ") +
                       cudaGetErrorString(countStatus);
        }
        return false;
    }

    const int requestedDevice = std::max(0, options.ceresCudaDevice);
    if (requestedDevice >= deviceCount)
    {
        if (message)
        {
            *message = "请求的 Ceres CUDA 设备越界: device=" +
                       std::to_string(requestedDevice) +
                       ", count=" +
                       std::to_string(deviceCount);
        }
        return false;
    }

    const cudaError_t setStatus = cudaSetDevice(requestedDevice);
    if (setStatus != cudaSuccess)
    {
        if (message)
        {
            *message = std::string("设置 Ceres CUDA 设备失败: ") +
                       cudaGetErrorString(setStatus);
        }
        return false;
    }
#  else
    (void)options;
    (void)message;
#  endif
    return true;
}

#endif // PLASCAN_BA_HAS_CERES

} // namespace

bool isCeresBackendCompiled()
{
#ifdef PLASCAN_BA_HAS_CERES
    return true;
#else
    return false;
#endif
}

bool isCeresCudaBackendCompiled()
{
#if defined(PLASCAN_BA_HAS_CERES) && !defined(CERES_NO_CUDA)
    return true;
#else
    return false;
#endif
}

BAResult optimizePointsWithCeres(const std::vector<Camera> &cameras,
                                 const std::vector<BATrack> &tracks,
                                 const BAOptions &options,
                                 bool requestGpu)
{
    BAResult result;
    result.totalTracks = static_cast<int>(tracks.size());
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());
    result.requestedBackend = options.backend;
    result.usedBackend = requestGpu ? BABackend::CeresCuda : BABackend::CeresCpu;
    result.usedGpu = requestGpu;
    result.observationCount = countObservations(tracks);

#ifndef PLASCAN_BA_HAS_CERES
    result.backendMessage = "Ceres 未编译进当前目标";
    return result;
#else
    if (cameras.empty() || tracks.empty())
    {
        return result;
    }
    if (options.cancelFlag && options.cancelFlag->load())
    {
        result.backendMessage = "Ceres BA 在启动前被取消";
        return result;
    }

    bool useCeresCuda = requestGpu;
    if (options.ceresLinearSolver == BACeresLinearSolver::DenseSchurCpu ||
        options.ceresLinearSolver == BACeresLinearSolver::SparseSchurCpu)
    {
        useCeresCuda = false;
        result.usedGpu = false;
        if (requestGpu)
        {
            result.usedBackend = BABackend::CeresCpu;
            result.backendFallback = true;
        }
    }
    std::string cudaDeviceMessage;
    if (requestGpu && !selectCeresCudaDevice(options, &cudaDeviceMessage))
    {
        useCeresCuda = false;
        result.usedBackend = BABackend::CeresCpu;
        result.usedGpu = false;
        result.backendFallback = true;
    }

    std::vector<std::array<double, 6>> cameraDeltas(cameras.size());
    std::vector<std::array<double, 3>> pointParams(tracks.size());
    std::vector<std::vector<BAObservation>> validObservationsByTrack(tracks.size());
    std::vector<char> activeTrack(tracks.size(), 0);
    std::vector<char> cameraHasResidual(cameras.size(), 0);
    ceres::Problem problem;
    const auto setupStart = std::chrono::steady_clock::now();

    int activeTrackCount = 0;
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        if (!plamatrix::isFinite(plamatrix::Vec3<double>(tracks[ti].initialPoint)))
        {
            continue;
        }

        std::set<int> uniqueCameras;
        std::vector<BAObservation> validObservations;
        validObservations.reserve(tracks[ti].observations.size());
        for (const BAObservation &observation : tracks[ti].observations)
        {
            if (observation.cameraIndex < 0 ||
                observation.cameraIndex >= static_cast<int>(cameras.size()) ||
                !std::isfinite(observation.u) ||
                !std::isfinite(observation.v) ||
                !(safeObservationWeight(observation) > 0.0))
            {
                continue;
            }
            uniqueCameras.insert(observation.cameraIndex);
            validObservations.push_back(observation);
        }

        if (validObservations.size() < 2 || uniqueCameras.size() < 2)
        {
            continue;
        }

        activeTrack[ti] = 1;
        ++activeTrackCount;
        validObservationsByTrack[ti] = std::move(validObservations);
        for (const int cameraIndex : uniqueCameras)
        {
            cameraHasResidual[static_cast<size_t>(cameraIndex)] = 1;
        }
    }

    if (activeTrackCount == 0)
    {
        const auto setupEnd = std::chrono::steady_clock::now();
        result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
        result.totalSeconds = result.setupSeconds;
        result.backendMessage = "Ceres BA 没有足够有效 track";
        return result;
    }

    int variableCameraCount = 0;
    for (size_t ci = 0; ci < cameras.size(); ++ci)
    {
        cameraDeltas[ci] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        if (!options.refineCameraPose || !cameraHasResidual[ci])
        {
            continue;
        }

        problem.AddParameterBlock(cameraDeltas[ci].data(), 6);
        if (isCameraFixed(static_cast<int>(ci), options))
        {
            problem.SetParameterBlockConstant(cameraDeltas[ci].data());
        }
        else
        {
            ++variableCameraCount;
        }
    }

    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        if (!activeTrack[ti])
        {
            continue;
        }

        pointParams[ti] = tracks[ti].initialPoint;
        problem.AddParameterBlock(pointParams[ti].data(), 3);
        for (const BAObservation &observation : validObservationsByTrack[ti])
        {
            if (options.refineCameraPose)
            {
                auto *cost = new ceres::NumericDiffCostFunction<PoseDeltaReprojectionResidual,
                                                                ceres::CENTRAL,
                                                                2,
                                                                6,
                                                                3>(
                    new PoseDeltaReprojectionResidual{
                        cameras[static_cast<size_t>(observation.cameraIndex)],
                        observation});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.huberDelta),
                                         cameraDeltas[static_cast<size_t>(observation.cameraIndex)].data(),
                                         pointParams[ti].data());
            }
            else
            {
                auto *cost = new ceres::AutoDiffCostFunction<FixedCameraReprojectionResidual, 2, 3>(
                    new FixedCameraReprojectionResidual{
                        xjw::ba::makeProjectionCamera(cameras[static_cast<size_t>(observation.cameraIndex)]),
                        observation});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.huberDelta),
                                         pointParams[ti].data());
            }
        }
        if (options.enableLaserPlaneConstraints)
        {
            for (const BALaserPlaneConstraint &constraint : tracks[ti].laserPlaneConstraints)
            {
                auto *cost = new ceres::NumericDiffCostFunction<LaserPlaneResidual, ceres::CENTRAL, 1, 3>(
                    new LaserPlaneResidual{constraint, options.laserPlaneWeight});
                problem.AddResidualBlock(cost, makeHuberLoss(options.laserHuberDeltaMeters), pointParams[ti].data());
            }
        }
        if (options.enableControlPointConstraints)
        {
            for (const BAControlPointConstraint &constraint : tracks[ti].controlPointConstraints)
            {
                auto *cost = new ceres::NumericDiffCostFunction<ControlPointResidual, ceres::CENTRAL, 3, 3>(
                    new ControlPointResidual{constraint, options.controlPointWeight});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.controlPointHuberDeltaMeters),
                                         pointParams[ti].data());
            }
        }
    }

    if (options.enableScaleBarConstraints)
    {
        for (const BAScaleBarConstraint &constraint : options.scaleBarConstraints)
        {
            if (constraint.trackIndexA < 0 || constraint.trackIndexB < 0 ||
                constraint.trackIndexA >= static_cast<int>(pointParams.size()) ||
                constraint.trackIndexB >= static_cast<int>(pointParams.size()) ||
                !activeTrack[static_cast<size_t>(constraint.trackIndexA)] ||
                !activeTrack[static_cast<size_t>(constraint.trackIndexB)])
            {
                continue;
            }
            auto *cost = new ceres::NumericDiffCostFunction<ScaleBarResidual, ceres::CENTRAL, 1, 3, 3>(
                new ScaleBarResidual{constraint, options.scaleBarWeight});
            problem.AddResidualBlock(cost,
                                     makeHuberLoss(options.scaleBarHuberDeltaMeters),
                                     pointParams[static_cast<size_t>(constraint.trackIndexA)].data(),
                                     pointParams[static_cast<size_t>(constraint.trackIndexB)].data());
        }
    }

    for (size_t ci = 0; ci < cameras.size() && ci < options.cameraPosePriors.size(); ++ci)
    {
        const BACameraPosePrior &prior = options.cameraPosePriors[ci];
        if (!prior.enabled ||
            !problem.HasParameterBlock(cameraDeltas[ci].data()) ||
            problem.IsParameterBlockConstant(cameraDeltas[ci].data()))
        {
            continue;
        }
        auto *cost = new ceres::NumericDiffCostFunction<PosePriorResidual, ceres::CENTRAL, 6, 6>(
            new PosePriorResidual{cameras[ci], prior, options.cameraPosePriorWeight});
        problem.AddResidualBlock(cost, makeHuberLoss(options.cameraPosePriorHuberDelta), cameraDeltas[ci].data());
    }

    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = std::max(1, options.maxIterations);
    if (options.ceresLinearSolver == BACeresLinearSolver::SparseSchurCpu)
    {
        solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else if (!options.refineCameraPose || variableCameraCount == 0)
    {
        // 仅优化三维点时没有需要 Schur 消元的相机变量，Dense QR 更符合问题结构。
        solverOptions.linear_solver_type = ceres::DENSE_QR;
    }
    else
    {
        solverOptions.linear_solver_type = ceres::DENSE_SCHUR;
    }
    solverOptions.num_threads = options.numThreads > 0
                                    ? options.numThreads
                                    : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    solverOptions.logging_type = ceres::SILENT;
    solverOptions.minimizer_progress_to_stdout = false;
    solverOptions.update_state_every_iteration = true;
#  if !defined(CERES_NO_CUDA)
    if (useCeresCuda)
    {
        solverOptions.dense_linear_algebra_library_type = ceres::CUDA;
        result.ceresLinearSolverName =
            solverOptions.linear_solver_type == ceres::DENSE_QR
                ? "dense_qr_cuda"
                : "dense_schur_cuda";
    }
    else
    {
        result.ceresLinearSolverName =
            solverOptions.linear_solver_type == ceres::SPARSE_SCHUR
                ? "sparse_schur_cpu"
                : (solverOptions.linear_solver_type == ceres::DENSE_QR
                       ? "dense_qr_cpu"
                       : "dense_schur_cpu");
    }
#  else
    result.ceresLinearSolverName =
        solverOptions.linear_solver_type == ceres::SPARSE_SCHUR
            ? "sparse_schur_cpu"
            : (solverOptions.linear_solver_type == ceres::DENSE_QR
                   ? "dense_qr_cpu"
                   : "dense_schur_cpu");
#  endif

    CeresBaIterationCallback iterationCallback(options);
    if (options.progressCallback || options.cancelFlag)
    {
        solverOptions.callbacks.push_back(&iterationCallback);
    }

    ceres::Solver::Summary summary;
    const auto setupEnd = std::chrono::steady_clock::now();
    ceres::Solve(solverOptions, &problem, &summary);
    const auto solveEnd = std::chrono::steady_clock::now();
    result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
    result.solveSeconds = std::chrono::duration<double>(solveEnd - setupEnd).count();
    result.totalSeconds = std::chrono::duration<double>(solveEnd - setupStart).count();
    result.backendMessage = cudaDeviceMessage.empty()
                                ? summary.BriefReport()
                                : cudaDeviceMessage + "；" + summary.BriefReport();

    for (size_t ci = 0; ci < cameras.size(); ++ci)
    {
        result.refinedCameras[ci] = cameras[ci];
        result.refinedCameras[ci].applyDeltaPose(cameraDeltas[ci].data());
    }

    double sumBefore = 0.0;
    double sumAfter = 0.0;
    int countBefore = 0;
    int countAfter = 0;
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        BARefinedPoint &point = result.points[ti];
        if (!activeTrack[ti])
        {
            point.point = tracks[ti].initialPoint;
            point.valid = false;
            point.converged = false;
            point.iterations = 0;
            point.rmsBefore = computeTrackRms(cameras, tracks[ti], tracks[ti].initialPoint);
            point.rmsAfter = std::numeric_limits<double>::infinity();
            if (std::isfinite(point.rmsBefore))
            {
                sumBefore += point.rmsBefore;
                ++countBefore;
            }
            continue;
        }

        point.point = pointParams[ti];
        point.valid = plamatrix::isFinite(plamatrix::Vec3<double>(point.point));
        point.converged = summary.termination_type == ceres::CONVERGENCE ||
                          summary.termination_type == ceres::USER_SUCCESS;
        point.iterations = static_cast<int>(summary.iterations.size());
        point.rmsBefore = computeTrackRms(cameras, tracks[ti], tracks[ti].initialPoint);
        point.rmsAfter = computeTrackRms(result.refinedCameras, tracks[ti], point.point);
        // Ceres 可能把点推到所有观测相机的有效投影范围外，此时坐标仍为有限值，
        // 但 rmsAfter 为 inf；这种轨迹不能计入有效优化点或全局 RMS。
        point.valid = point.valid && std::isfinite(point.rmsAfter);
        if (std::isfinite(point.rmsBefore))
        {
            sumBefore += point.rmsBefore;
            ++countBefore;
        }
        if (point.valid)
        {
            if (options.enablePointFilter && point.rmsAfter > options.filterMaxReprojError)
            {
                point.valid = false;
            }
        }
        if (point.valid)
        {
            sumAfter += point.rmsAfter;
            ++countAfter;
            ++result.optimizedTracks;
        }
    }
    result.meanRmsBefore = countBefore > 0 ? sumBefore / static_cast<double>(countBefore) : 0.0;
    result.meanRmsAfter = countAfter > 0 ? sumAfter / static_cast<double>(countAfter) : 0.0;
    result.refinedCameraCount = options.refineCameraPose ? variableCameraCount : 0;

    if (options.enableLaserPlaneConstraints)
    {
        const ScalarDistanceStats before = computeLaserStats(tracks, nullptr);
        const ScalarDistanceStats after = computeLaserStats(tracks, &result.points);
        result.laserConstraintCount = before.count;
        result.laserRmsBeforeMeters = before.rms;
        result.laserMedianBeforeMeters = before.median;
        result.laserRmsAfterMeters = after.rms;
        result.laserMedianAfterMeters = after.median;
    }
    if (options.enableControlPointConstraints)
    {
        const ScalarDistanceStats before = computeControlPointStats(tracks, nullptr);
        const ScalarDistanceStats after = computeControlPointStats(tracks, &result.points);
        result.controlPointConstraintCount = before.count;
        result.controlPointRmsBeforeMeters = before.rms;
        result.controlPointRmsAfterMeters = after.rms;
    }
    if (options.enableScaleBarConstraints)
    {
        const ScalarDistanceStats before = computeScaleBarStats(tracks, nullptr, options.scaleBarConstraints);
        const ScalarDistanceStats after = computeScaleBarStats(tracks, &result.points, options.scaleBarConstraints);
        result.scaleBarConstraintCount = before.count;
        result.scaleBarRmsBeforeMeters = before.rms;
        result.scaleBarRmsAfterMeters = after.rms;
    }
    return result;
#endif
}

} // namespace xjw::detail
