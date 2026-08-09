/**
 * @file BundleAdjustCeres.cpp
 * @brief Ceres 联合相机、三维点和可选共享焦距 BA 后端。
 *
 * 公共 Camera 使用 camera-to-world 旋转 Rcw 和世界系中心 C；Ceres 参数块内部
 * 转换为 angle-axis world-to-camera 加平移。每条像点观测形成二维残差块，点和
 * 相机的稀疏耦合由 Schur 求解器处理。求解结果返回前仍经过与其他后端一致的
 * 正深度、轨迹有效率和 RMS 质量门控。
 */

#include "BundleAdjustCeres.h"

#include "BundleAdjustCeresPlanning.h"
#include "BundleAdjustProjection.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"

#include <plamatrix/ops/statistics.h>
#include <plamatrix/ops/vector.h>

#ifdef PLASCAN_BA_HAS_CERES
#  include <ceres/ceres.h>
#  include <ceres/internal/config.h>
#  include <ceres/rotation.h>
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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>

namespace xjw::detail
{
namespace
{

bool hasCeresSparseLinearAlgebra()
{
#if defined(PLASCAN_BA_HAS_CERES) && !defined(CERES_NO_SPARSE)
    return ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
           ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE) ||
           ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::ACCELERATE_SPARSE) ||
           ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE);
#else
    return false;
#endif
}

bool isCameraFixed(int cameraIndex, const BAOptions &options)
{
    return std::find(options.fixedCameraIndices.begin(),
                     options.fixedCameraIndices.end(),
                     cameraIndex) != options.fixedCameraIndices.end();
}

#ifdef PLASCAN_BA_HAS_CERES

// -------------------------------------------------------------------------
// 残差块定义
//
// 重投影残差按“固定/可变外参 × 固定/共享焦距”拆成四种签名，使 Ceres 只看到
// 实际需要优化的参数块。无效正深度返回大残差而不是 false，避免 Ceres 将其当作
// 残差求值失败并中止整个问题。
// -------------------------------------------------------------------------
struct FixedCameraReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight =
            T(std::sqrt(sanitizedObservationWeight(observation)));
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
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const cameraDelta,
                    const T *const point,
                    T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight =
            T(std::sqrt(sanitizedObservationWeight(observation)));
        if (!xjw::ba::projectWithPoseDelta(camera, cameraDelta, point, pixel))
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

struct FixedPoseSharedIntrinsicsReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const point,
                    const T *const sharedIntrinsics,
                    T *residuals) const
    {
        const T zeroDelta[6] = {
            T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0)};
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight =
            T(std::sqrt(sanitizedObservationWeight(observation)));
        if (!xjw::ba::projectWithPoseDeltaAndSharedIntrinsics(
                camera, zeroDelta, point, sharedIntrinsics, pixel))
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

struct PoseDeltaSharedIntrinsicsReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const cameraDelta,
                    const T *const point,
                    const T *const sharedIntrinsics,
                    T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight =
            T(std::sqrt(sanitizedObservationWeight(observation)));
        if (!xjw::ba::projectWithPoseDeltaAndSharedIntrinsics(
                camera, cameraDelta, point, sharedIntrinsics, pixel))
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

struct SharedIntrinsicsPriorResidual
{
    std::array<double, 9> reference{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 9> inverseSigma{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    template <typename T>
    bool operator()(const T *const sharedIntrinsics, T *residuals) const
    {
        for (int index = 0; index < 9; ++index)
        {
            residuals[index] = T(inverseSigma[static_cast<size_t>(index)]) *
                               (sharedIntrinsics[index] -
                                T(reference[static_cast<size_t>(index)]));
        }
        return true;
    }
};

struct LaserPlaneResidual
{
    BALaserPlaneConstraint constraint;
    double sqrtWeight = 1.0;

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        const T dx = point[0] - T(constraint.point[0]);
        const T dy = point[1] - T(constraint.point[1]);
        const T dz = point[2] - T(constraint.point[2]);
        const T signedDistance =
            dx * T(constraint.normal[0]) +
            dy * T(constraint.normal[1]) +
            dz * T(constraint.normal[2]);
        residuals[0] = T(sqrtWeight) * signedDistance;
        return true;
    }
};

struct ControlPointResidual
{
    BAControlPointConstraint constraint;
    double weight = 1.0;

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        const double sigma = std::max(1e-9, constraint.sigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight * constraint.weight)) / sigma;
        residuals[0] = T(scale) * (point[0] - T(constraint.point[0]));
        residuals[1] = T(scale) * (point[1] - T(constraint.point[1]));
        residuals[2] = T(scale) * (point[2] - T(constraint.point[2]));
        return true;
    }
};

struct ScaleBarResidual
{
    BAScaleBarConstraint constraint;
    double weight = 1.0;

    template <typename T>
    bool operator()(const T *const pointA,
                    const T *const pointB,
                    T *residuals) const
    {
        const T dx = pointA[0] - pointB[0];
        const T dy = pointA[1] - pointB[1];
        const T dz = pointA[2] - pointB[2];
        using std::sqrt;
        const T distance = sqrt(dx * dx + dy * dy + dz * dz);
        const double sigma = std::max(1e-9, constraint.sigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight * constraint.weight)) / sigma;
        residuals[0] =
            T(scale) * (distance - T(constraint.measuredDistanceMeters));
        return true;
    }
};

struct PosePriorResidual
{
    Camera camera;
    BACameraPosePrior prior;
    double weight = 1.0;

    template <typename T>
    bool operator()(const T *const cameraDelta, T *residuals) const
    {
        const double rotationSigma = std::max(1e-9, prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0);
        const double positionSigma = std::max(1e-9, prior.positionSigmaMeters);
        const double scale = std::sqrt(std::max(0.0, weight));

        T deltaRotation[9];
        xjw::ba::poseDeltaRotation(cameraDelta, deltaRotation);
        const auto baseRotation = camera.cameraToWorldRotation();
        T updatedRotation[9];
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                T value = T(0.0);
                for (int inner = 0; inner < 3; ++inner)
                {
                    value +=
                        deltaRotation[row * 3 + inner] *
                        T(baseRotation[inner * 3 + column]);
                }
                updatedRotation[row * 3 + column] = value;
            }
        }

        // SO(3) 残差使用 log(R_current * R_prior^T)，避免对位姿先验做
        // 12 次中央数值差分，同时保持旋转残差对 Ceres Jet 可导。
        T relativeRotation[9];
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                T value = T(0.0);
                for (int inner = 0; inner < 3; ++inner)
                {
                    value +=
                        updatedRotation[row * 3 + inner] *
                        T(prior.cameraToWorldRotation[
                            column * 3 + inner]);
                }
                relativeRotation[row * 3 + column] = value;
            }
        }

        T rotationResidual[3];
        ceres::RotationMatrixToAngleAxis(
            ceres::RowMajorAdapter3x3(
                static_cast<const T *>(relativeRotation)),
            rotationResidual);
        residuals[0] = T(scale / rotationSigma) * rotationResidual[0];
        residuals[1] = T(scale / rotationSigma) * rotationResidual[1];
        residuals[2] = T(scale / rotationSigma) * rotationResidual[2];
        const auto center = camera.cameraCenter();
        residuals[3] =
            T(scale / positionSigma) *
            (T(center[0]) + cameraDelta[3] -
             T(prior.cameraCenter[0]));
        residuals[4] =
            T(scale / positionSigma) *
            (T(center[1]) + cameraDelta[4] -
             T(prior.cameraCenter[1]));
        residuals[5] =
            T(scale / positionSigma) *
            (T(center[2]) + cameraDelta[5] -
             T(prior.cameraCenter[2]));
        return true;
    }
};

struct CameraPlaneResidual
{
    Camera camera;
    std::array<double, 3> point;
    std::array<double, 3> normal;
    double referenceSignedDistance = 0.0;
    double sigmaMeters = 1.0;
    double weight = 1.0;

    template <typename T>
    bool operator()(const T *const cameraDelta, T *residuals) const
    {
        const auto center = camera.cameraCenter();
        const double sigma = std::max(1.0e-9, sigmaMeters);
        const double scale =
            std::sqrt(std::max(0.0, weight)) / sigma;
        residuals[0] = T(scale) *
            (T(normal[0]) *
                 (T(center[0]) + cameraDelta[3] - T(point[0])) +
             T(normal[1]) *
                 (T(center[1]) + cameraDelta[4] - T(point[1])) +
             T(normal[2]) *
                 (T(center[2]) + cameraDelta[5] - T(point[2])) -
             T(referenceSignedDistance));
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
    CeresBaIterationCallback(const BAOptions &options,
                             int observationCount,
                             int iterationOffset,
                             int stageIterations,
                             int totalIterations)
        : _options(options),
          _observationCount(std::max(1, observationCount)),
          _iterationOffset(std::max(0, iterationOffset)),
          _stageIterations(std::max(1, stageIterations)),
          _totalIterations(std::max(1, totalIterations))
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override
    {
        if (_options.cancelFlag && _options.cancelFlag->load())
        {
            _cancelled = true;
            return ceres::SOLVER_ABORT;
        }
        if (_options.progressCallback)
        {
            // Ceres 会为初始状态额外产生 iteration=0 的摘要。GUI 只展示最多
            // maxIterations 个实际进度步，避免出现“21/20”。
            if (summary.iteration < _stageIterations)
            {
                // Ceres cost 为 0.5 * sum(r^2)。每个影像观测有两个像素残差，
                // 因而 sqrt(cost / observationCount) 是可解释的像素 RMS 代理。
                const double rmsProxy =
                    std::sqrt(std::max(0.0, summary.cost) /
                              static_cast<double>(_observationCount));
                const int currentIteration = std::min(
                    _totalIterations,
                    _iterationOffset + summary.iteration + 1);
                if (!_options.progressCallback(currentIteration,
                                                 _totalIterations,
                                                 rmsProxy,
                                                 0))
                {
                    _progressAborted = true;
                    return ceres::SOLVER_ABORT;
                }
            }
        }
        return ceres::SOLVER_CONTINUE;
    }

    bool cancelled() const
    {
        return _cancelled;
    }

    bool progressAborted() const
    {
        return _progressAborted;
    }

private:
    const BAOptions &_options;
    int _observationCount = 1;
    int _iterationOffset = 0;
    int _stageIterations = 1;
    int _totalIterations = 1;
    bool _cancelled = false;
    bool _progressAborted = false;
};

bool selectCeresCudaDevice(const BAOptions &options,
                           std::uint64_t *freeBytes,
                           std::string *message)
{
    if (freeBytes)
    {
        *freeBytes = 0;
    }
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
    std::size_t freeMemory = 0;
    std::size_t totalMemory = 0;
    const cudaError_t memoryStatus =
        cudaMemGetInfo(&freeMemory, &totalMemory);
    if (memoryStatus == cudaSuccess && freeBytes)
    {
        *freeBytes = static_cast<std::uint64_t>(freeMemory);
    }
#  else
    (void)options;
    (void)freeBytes;
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
    result.observationCount =
        summarizeUsableProblem(cameras, tracks).observationCount;
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        BARefinedPoint &point = result.points[ti];
        point.point = tracks[ti].initialPoint;
    }

#ifndef PLASCAN_BA_HAS_CERES
    result.solveStatus = BASolveStatus::BackendUnavailable;
    result.backendMessage = "Ceres 未编译进当前目标";
    return result;
#else
    if (cameras.empty() || tracks.empty())
    {
        result.solveStatus = BASolveStatus::InvalidInput;
        result.backendMessage = "Ceres BA 输入相机或 track 为空";
        return result;
    }
    if (options.cancelFlag && options.cancelFlag->load())
    {
        result.solveStatus = BASolveStatus::Cancelled;
        result.backendMessage = "Ceres BA 在启动前被取消";
        return result;
    }

    bool cudaDeviceReady = requestGpu;
    std::uint64_t cudaFreeBytes = 0;
    std::string cudaDeviceMessage;
    if (requestGpu &&
        !selectCeresCudaDevice(
            options, &cudaFreeBytes, &cudaDeviceMessage))
    {
        cudaDeviceReady = false;
        result.usedBackend = BABackend::CeresCpu;
        result.usedGpu = false;
        result.backendFallback = true;
    }

    // 阶段 1：筛选可进入问题的轨迹并建立参数块活动掩码。
    //
    // 初始点必须被至少两台不同相机正深度观测；不满足条件的 track 保留在结果
    // 对齐数组中，但不会进入 Ceres Problem。
    std::vector<std::array<double, 6>> cameraDeltas(cameras.size());
    std::vector<std::array<double, 3>> pointParams(tracks.size());
    std::vector<std::vector<BAObservation>> validObservationsByTrack(tracks.size());
    std::vector<char> activeTrack(tracks.size(), 0);
    std::vector<char> cameraHasResidual(cameras.size(), 0);
    ceres::Problem problem;
    const auto setupStart = std::chrono::steady_clock::now();

    int activeTrackCount = 0;
    int activeObservationCount = 0;
    int rejectedInitialTrackCount = 0;
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
            if (!observationIsUsable(observation, cameras.size()))
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

        bool initialPointHasPositiveDepth = true;
        double initialResidualSquared = 0.0;
        int initialResidualCount = 0;
        for (const BAObservation &observation : validObservations)
        {
            const double world[3] = {
                tracks[ti].initialPoint[0],
                tracks[ti].initialPoint[1],
                tracks[ti].initialPoint[2],
            };
            double pixel[2] = {0.0, 0.0};
            if (!cameras[static_cast<size_t>(observation.cameraIndex)]
                     .projectWorldPoint(world, pixel))
            {
                initialPointHasPositiveDepth = false;
                break;
            }
            const double du = pixel[0] - observation.u;
            const double dv = pixel[1] - observation.v;
            initialResidualSquared +=
                sanitizedObservationWeight(observation) * (du * du + dv * dv);
            initialResidualCount += 2;
        }
        if (!initialPointHasPositiveDepth)
        {
            continue;
        }
        const double initialTrackRms =
            initialResidualCount > 0
                ? std::sqrt(initialResidualSquared /
                            static_cast<double>(initialResidualCount))
                : std::numeric_limits<double>::infinity();
        if (options.maxCeresInitialTrackRms > 0.0 &&
            initialTrackRms > options.maxCeresInitialTrackRms)
        {
            ++rejectedInitialTrackCount;
            continue;
        }

        activeTrack[ti] = 1;
        ++activeTrackCount;
        activeObservationCount += static_cast<int>(validObservations.size());
        validObservationsByTrack[ti] = std::move(validObservations);
        for (const int cameraIndex : uniqueCameras)
        {
            cameraHasResidual[static_cast<size_t>(cameraIndex)] = 1;
        }
    }

    result.observationCount = activeObservationCount;
    result.ceresRejectedInitialTracks = rejectedInitialTrackCount;
    if (activeTrackCount == 0)
    {
        const auto setupEnd = std::chrono::steady_clock::now();
        result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
        result.totalSeconds = result.setupSeconds;
        result.solveStatus = BASolveStatus::InvalidInput;
        result.backendMessage = "Ceres BA 没有足够有效 track";
        return result;
    }

    // 阶段 2：相机使用零起点的局部 6-DOF 增量，而不是直接优化 9 元旋转矩阵。
    // fixedCameraIndices 在这里转换为 Ceres 常量块，真正消除 gauge 自由度。
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

    // 按镜头/焦段标定组建立独立针孔内参参数。同组共享绝对焦距、宽高比和
    // 主点相对偏移，不同组互不吸收误差。
    const bool refineSharedIntrinsics =
        options.refineSharedFocalLength ||
        options.refineSharedFocalAspectRatio ||
        options.refineSharedPrincipalPoint ||
        options.refineSharedRadialDistortion;
    std::vector<int> calibrationGroupByCamera(cameras.size(), 0);
    if (!options.cameraCalibrationGroupIds.empty())
    {
        calibrationGroupByCamera = options.cameraCalibrationGroupIds;
    }
    std::map<int, int> groupIdToParameterIndex;
    for (const int groupId : calibrationGroupByCamera)
    {
        if (groupIdToParameterIndex.count(groupId) == 0)
        {
            const int parameterIndex =
                static_cast<int>(groupIdToParameterIndex.size());
            groupIdToParameterIndex.emplace(groupId, parameterIndex);
        }
    }
    std::vector<int> calibrationParameterByCamera(cameras.size(), 0);
    std::vector<std::vector<double>> focalSamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> aspectSamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> radialK1SamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> radialK2SamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> radialK3SamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> tangentialP1SamplesByGroup(
        groupIdToParameterIndex.size());
    std::vector<std::vector<double>> tangentialP2SamplesByGroup(
        groupIdToParameterIndex.size());
    for (size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
    {
        const int parameterIndex =
            groupIdToParameterIndex.at(
                calibrationGroupByCamera[cameraIndex]);
        calibrationParameterByCamera[cameraIndex] = parameterIndex;
        if (std::isfinite(cameras[cameraIndex].focalX()) &&
            cameras[cameraIndex].focalX() > 0.0)
        {
            focalSamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(cameras[cameraIndex].focalX());
            if (std::isfinite(cameras[cameraIndex].focalY()) &&
                cameras[cameraIndex].focalY() > 0.0)
            {
                aspectSamplesByGroup[static_cast<size_t>(parameterIndex)]
                    .push_back(cameras[cameraIndex].focalY() /
                               cameras[cameraIndex].focalX());
            }
        }
        const Camera::Distortion distortion = cameras[cameraIndex].distortion();
        if (std::isfinite(distortion.radialK1))
        {
            radialK1SamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(distortion.radialK1);
        }
        if (std::isfinite(distortion.radialK2))
        {
            radialK2SamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(distortion.radialK2);
        }
        if (std::isfinite(distortion.radialK3))
        {
            radialK3SamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(distortion.radialK3);
        }
        if (std::isfinite(distortion.tangentialP1))
        {
            tangentialP1SamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(distortion.tangentialP1);
        }
        if (std::isfinite(distortion.tangentialP2))
        {
            tangentialP2SamplesByGroup[static_cast<size_t>(parameterIndex)]
                .push_back(distortion.tangentialP2);
        }
    }

    // 求解器规划必须在活动轨迹、可变相机和标定组数量明确后执行；使用原始输入
    // 数量估算会高估显存并导致不必要的 CPU 回退。
    const BACeresSolverPlan solverPlan =
        planCeresSolver(
            options,
            variableCameraCount,
            refineSharedIntrinsics
                ? static_cast<int>(groupIdToParameterIndex.size())
                : 0,
            activeTrackCount,
            result.observationCount,
            requestGpu && cudaDeviceReady,
            cudaFreeBytes,
            hasCeresSparseLinearAlgebra());
    const bool useCeresCuda = solverPlan.useCuda;
    result.ceresEstimatedCudaBytes =
        solverPlan.estimatedCudaBytes;
    result.ceresCudaFreeBytes = cudaFreeBytes;
    result.usedGpu = useCeresCuda;
    if (requestGpu && !useCeresCuda)
    {
        result.usedBackend = BABackend::CeresCpu;
        result.backendFallback = true;
        if (!cudaDeviceMessage.empty())
        {
            cudaDeviceMessage += "；";
        }
        cudaDeviceMessage += solverPlan.reason;
    }

    std::vector<double> sharedFocalReferencePixels(
        groupIdToParameterIndex.size(), 1.0);
    std::vector<double> sharedAspectReference(
        groupIdToParameterIndex.size(), 1.0);
    std::vector<std::array<double, 9>> sharedIntrinsicsParameters(
        groupIdToParameterIndex.size());
    std::vector<std::vector<int>> sharedIntrinsicConstantIndices(
        groupIdToParameterIndex.size());
    for (size_t groupIndex = 0;
         groupIndex < sharedFocalReferencePixels.size();
         ++groupIndex)
    {
        sharedFocalReferencePixels[groupIndex] =
            plamatrix::finiteMedian(
                std::move(focalSamplesByGroup[groupIndex]))
                .value_or(1.0);
        sharedAspectReference[groupIndex] =
            plamatrix::finiteMedian(
                std::move(aspectSamplesByGroup[groupIndex]))
                .value_or(1.0);
        const double radialK1Reference =
            plamatrix::finiteMedian(
                std::move(radialK1SamplesByGroup[groupIndex]))
                .value_or(0.0);
        const double radialK2Reference =
            plamatrix::finiteMedian(
                std::move(radialK2SamplesByGroup[groupIndex]))
                .value_or(0.0);
        const double radialK3Reference =
            plamatrix::finiteMedian(
                std::move(radialK3SamplesByGroup[groupIndex]))
                .value_or(0.0);
        const double tangentialP1Reference =
            plamatrix::finiteMedian(
                std::move(tangentialP1SamplesByGroup[groupIndex]))
                .value_or(0.0);
        const double tangentialP2Reference =
            plamatrix::finiteMedian(
                std::move(tangentialP2SamplesByGroup[groupIndex]))
                .value_or(0.0);
        sharedIntrinsicsParameters[groupIndex] = {{
            std::log(sharedFocalReferencePixels[groupIndex]),
            std::log(sharedAspectReference[groupIndex]),
            0.0,
            0.0,
            radialK1Reference,
            radialK2Reference,
            radialK3Reference,
            tangentialP1Reference,
            tangentialP2Reference,
        }};
    }

    if (refineSharedIntrinsics)
    {
        const double minimumScale = std::max(1e-6, options.minSharedFocalScale);
        const double maximumScale =
            std::max(minimumScale, options.maxSharedFocalScale);
        const double minimumAspectScale =
            std::max(1e-6, options.minSharedFocalAspectScale);
        const double maximumAspectScale =
            std::max(minimumAspectScale, options.maxSharedFocalAspectScale);
        for (size_t groupIndex = 0;
             groupIndex < sharedIntrinsicsParameters.size();
             ++groupIndex)
        {
            double *parameter = sharedIntrinsicsParameters[groupIndex].data();
            problem.AddParameterBlock(parameter, 9);
            std::vector<int> constantIndices;
            if (options.refineSharedFocalLength)
            {
                if (maximumScale - minimumScale <= 1e-12)
                {
                    // Ceres 要求参数上下界严格可行。零宽焦距搜索区间的语义是
                    // 固定共享焦距，而不是提交一对相等的上下界。
                    parameter[0] = std::log(
                        sharedFocalReferencePixels[groupIndex] * minimumScale);
                    constantIndices.push_back(0);
                }
                else
                {
                    problem.SetParameterLowerBound(
                        parameter,
                        0,
                        std::log(sharedFocalReferencePixels[groupIndex] * minimumScale));
                    problem.SetParameterUpperBound(
                        parameter,
                        0,
                        std::log(sharedFocalReferencePixels[groupIndex] * maximumScale));
                }
            }
            else
            {
                constantIndices.push_back(0);
            }
            if (options.refineSharedFocalAspectRatio)
            {
                problem.SetParameterLowerBound(
                    parameter,
                    1,
                    std::log(sharedAspectReference[groupIndex] * minimumAspectScale));
                problem.SetParameterUpperBound(
                    parameter,
                    1,
                    std::log(sharedAspectReference[groupIndex] * maximumAspectScale));
            }
            else
            {
                constantIndices.push_back(1);
            }
            if (options.refineSharedPrincipalPoint)
            {
                const double maxOffset =
                    sharedFocalReferencePixels[groupIndex] *
                    options.maxSharedPrincipalPointOffsetFraction;
                for (int coordinate = 2; coordinate <= 3; ++coordinate)
                {
                    problem.SetParameterLowerBound(parameter, coordinate, -maxOffset);
                    problem.SetParameterUpperBound(parameter, coordinate, maxOffset);
                }
            }
            else
            {
                constantIndices.push_back(2);
                constantIndices.push_back(3);
            }
            if (options.refineSharedRadialDistortion)
            {
                problem.SetParameterLowerBound(
                    parameter, 4, -options.maxSharedRadialK1Abs);
                problem.SetParameterUpperBound(
                    parameter, 4, options.maxSharedRadialK1Abs);
                problem.SetParameterLowerBound(
                    parameter, 5, -options.maxSharedRadialK2Abs);
                problem.SetParameterUpperBound(
                    parameter, 5, options.maxSharedRadialK2Abs);
                problem.SetParameterLowerBound(
                    parameter, 6, -options.maxSharedRadialK3Abs);
                problem.SetParameterUpperBound(
                    parameter, 6, options.maxSharedRadialK3Abs);
                problem.SetParameterLowerBound(
                    parameter, 7, -options.maxSharedTangentialP1Abs);
                problem.SetParameterUpperBound(
                    parameter, 7, options.maxSharedTangentialP1Abs);
                problem.SetParameterLowerBound(
                    parameter, 8, -options.maxSharedTangentialP2Abs);
                problem.SetParameterUpperBound(
                    parameter, 8, options.maxSharedTangentialP2Abs);
                if (!options.refineSharedHighOrderDistortion)
                {
                    constantIndices.insert(
                        constantIndices.end(), {5, 6, 7, 8});
                }
            }
            else
            {
                constantIndices.push_back(4);
                constantIndices.push_back(5);
                constantIndices.push_back(6);
                constantIndices.push_back(7);
                constantIndices.push_back(8);
            }
            if (!constantIndices.empty())
            {
                problem.SetManifold(
                    parameter,
                    new ceres::SubsetManifold(9, constantIndices));
            }
            sharedIntrinsicConstantIndices[groupIndex] = constantIndices;

            if (options.refineSharedPrincipalPoint ||
                options.refineSharedFocalAspectRatio ||
                options.refineSharedRadialDistortion)
            {
                const double principalSigma =
                    std::max(1e-6,
                             sharedFocalReferencePixels[groupIndex] *
                                 options.sharedPrincipalPointPriorSigmaFraction);
                const double aspectSigma =
                    std::max(1e-6, options.sharedFocalAspectPriorSigma);
                auto *prior =
                    new ceres::AutoDiffCostFunction<SharedIntrinsicsPriorResidual,
                                                    9,
                                                    9>(
                        new SharedIntrinsicsPriorResidual{
                            {{std::log(sharedFocalReferencePixels[groupIndex]),
                              std::log(sharedAspectReference[groupIndex]),
                              0.0,
                              0.0,
                              sharedIntrinsicsParameters[groupIndex][4],
                              sharedIntrinsicsParameters[groupIndex][5],
                              sharedIntrinsicsParameters[groupIndex][6],
                              sharedIntrinsicsParameters[groupIndex][7],
                              sharedIntrinsicsParameters[groupIndex][8]}},
                            {{1.0 / options.sharedFocalPriorSigma,
                              1.0 / aspectSigma,
                              1.0 / principalSigma,
                              1.0 / principalSigma,
                              1.0 / options.sharedRadialK1PriorSigma,
                              1.0 / options.sharedRadialK2PriorSigma,
                              1.0 / options.sharedRadialK3PriorSigma,
                              1.0 / options.sharedTangentialP1PriorSigma,
                              1.0 / options.sharedTangentialP2PriorSigma}}});
                problem.AddResidualBlock(prior, nullptr, parameter);
            }
        }
    }

    // 阶段 3：装配重投影和物方约束。
    //
    // 每条二维观测只选择一种重投影残差签名；控制点/激光平面挂到单点块，
    // 比例尺连接两个点块，位姿先验挂到相机增量块。
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
            if (options.refineCameraPose && refineSharedIntrinsics)
            {
                auto *cost =
                    new ceres::AutoDiffCostFunction<PoseDeltaSharedIntrinsicsReprojectionResidual,
                                                    2,
                                                    6,
                                                    3,
                                                    9>(
                        new PoseDeltaSharedIntrinsicsReprojectionResidual{
                            xjw::ba::makeProjectionCamera(
                                cameras[static_cast<size_t>(observation.cameraIndex)]),
                            observation});
                problem.AddResidualBlock(
                    cost,
                    makeHuberLoss(options.huberDelta),
                    cameraDeltas[static_cast<size_t>(observation.cameraIndex)].data(),
                    pointParams[ti].data(),
                    sharedIntrinsicsParameters[
                        static_cast<size_t>(
                            calibrationParameterByCamera[
                                static_cast<size_t>(
                                    observation.cameraIndex)])].data());
            }
            else if (options.refineCameraPose)
            {
                auto *cost = new ceres::AutoDiffCostFunction<PoseDeltaReprojectionResidual,
                                                             2,
                                                             6,
                                                             3>(
                    new PoseDeltaReprojectionResidual{
                        xjw::ba::makeProjectionCamera(
                            cameras[static_cast<size_t>(observation.cameraIndex)]),
                        observation});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.huberDelta),
                                         cameraDeltas[static_cast<size_t>(observation.cameraIndex)].data(),
                                         pointParams[ti].data());
            }
            else if (refineSharedIntrinsics)
            {
                auto *cost =
                    new ceres::AutoDiffCostFunction<FixedPoseSharedIntrinsicsReprojectionResidual,
                                                    2,
                                                    3,
                                                    9>(
                        new FixedPoseSharedIntrinsicsReprojectionResidual{
                            xjw::ba::makeProjectionCamera(
                                cameras[static_cast<size_t>(observation.cameraIndex)]),
                            observation});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.huberDelta),
                                         pointParams[ti].data(),
                                         sharedIntrinsicsParameters[
                                             static_cast<size_t>(
                                                 calibrationParameterByCamera[
                                                     static_cast<size_t>(
                                                         observation.cameraIndex)])].data());
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
                const double effectiveWeight =
                    std::max(0.0, options.laserPlaneWeight) *
                    std::max(0.0, constraint.weight);
                if (!(effectiveWeight > 0.0))
                {
                    continue;
                }
                const double sqrtWeight = std::sqrt(effectiveWeight);
                auto *cost = new ceres::AutoDiffCostFunction<LaserPlaneResidual, 1, 3>(
                    new LaserPlaneResidual{constraint, sqrtWeight});
                // 同时缩放残差和 Huber 阈值，等价于
                // effectiveWeight * Huber(distance, deltaMeters)，与 legacy 后端一致。
                problem.AddResidualBlock(
                    cost,
                    makeHuberLoss(options.laserHuberDeltaMeters * sqrtWeight),
                    pointParams[ti].data());
            }
        }
        if (options.enableControlPointConstraints)
        {
            for (const BAControlPointConstraint &constraint : tracks[ti].controlPointConstraints)
            {
                auto *cost = new ceres::AutoDiffCostFunction<ControlPointResidual, 3, 3>(
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
            auto *cost = new ceres::AutoDiffCostFunction<ScaleBarResidual, 1, 3, 3>(
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
        auto *cost = new ceres::AutoDiffCostFunction<PosePriorResidual, 6, 6>(
            new PosePriorResidual{cameras[ci], prior, options.cameraPosePriorWeight});
        problem.AddResidualBlock(cost, makeHuberLoss(options.cameraPosePriorHuberDelta), cameraDeltas[ci].data());
    }

    if (options.cameraPlaneConstraint.enabled)
    {
        for (size_t ci = 0; ci < cameras.size(); ++ci)
        {
            if (!problem.HasParameterBlock(cameraDeltas[ci].data()) ||
                problem.IsParameterBlockConstant(cameraDeltas[ci].data()))
            {
                continue;
            }
            const BACameraPlaneConstraint &constraint =
                options.cameraPlaneConstraint;
            const double referenceSignedDistance =
                constraint.referenceSignedDistances.empty()
                    ? 0.0
                    : constraint.referenceSignedDistances[ci];
            auto *cost = new ceres::AutoDiffCostFunction<CameraPlaneResidual, 1, 6>(
                new CameraPlaneResidual{
                    cameras[ci],
                    constraint.point,
                    constraint.normal,
                    referenceSignedDistance,
                    constraint.sigmaMeters,
                    constraint.weight});
            problem.AddResidualBlock(
                cost,
                makeHuberLoss(options.cameraPlaneHuberDelta),
                cameraDeltas[ci].data());
        }
    }

    // 阶段 4：按规划配置线性求解器。CUDA 只加速 dense 线性代数，
    // 残差装配、鲁棒核和大部分问题管理仍在 CPU。
    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = std::max(1, options.maxIterations);
    switch (solverPlan.solver)
    {
    case BACeresSolverKind::DenseQr:
        solverOptions.linear_solver_type = ceres::DENSE_QR;
        break;
    case BACeresSolverKind::DenseSchur:
        solverOptions.linear_solver_type = ceres::DENSE_SCHUR;
        break;
    case BACeresSolverKind::SparseSchur:
        solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
        break;
    case BACeresSolverKind::IterativeSchur:
        solverOptions.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solverOptions.preconditioner_type = ceres::SCHUR_JACOBI;
        break;
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
            std::string(ceresSolverKindName(solverPlan.solver)) +
            "_cuda";
    }
    else
    {
        result.ceresLinearSolverName =
            std::string(ceresSolverKindName(solverPlan.solver)) +
            "_cpu";
    }
#  else
    result.ceresLinearSolverName =
        std::string(ceresSolverKindName(solverPlan.solver)) +
        "_cpu";
#  endif

    const int totalIterationBudget = std::max(1, options.maxIterations);
    // 自标定采用“固定内参预热 -> 焦距+k1 -> 完整 Brown 模型”的分阶段策略。
    // 先让低阶径向项解释主要形变，再释放 k2/k3/p1/p2，可避免弱几何下高阶项
    // 过早吸收本应由 k1 表达的穹顶误差。
    const bool runStagedSelfCalibration =
        refineSharedIntrinsics &&
        options.stageSharedFocalRefinement &&
        options.sharedFocalWarmupFraction > 0.0 &&
        totalIterationBudget >= 2;
    int warmupIterations = 0;
    if (runStagedSelfCalibration)
    {
        warmupIterations = std::clamp(
            static_cast<int>(
                std::lround(
                    totalIterationBudget *
                    options.sharedFocalWarmupFraction)),
            1,
            totalIterationBudget - 1);
        for (auto &intrinsicsParameter : sharedIntrinsicsParameters)
        {
            problem.SetParameterBlockConstant(intrinsicsParameter.data());
        }
    }

    std::unique_ptr<CeresBaIterationCallback> warmupCallback;
    std::unique_ptr<CeresBaIterationCallback> lowOrderCallback;
    std::unique_ptr<CeresBaIterationCallback> refinementCallback;
    ceres::Solver::Summary warmupSummary;
    ceres::Solver::Summary lowOrderSummary;
    ceres::Solver::Summary summary;
    bool lowOrderStageAttempted = false;
    const auto applySharedIntrinsicManifolds =
        [&](bool lowOrderDistortionOnly)
        {
            for (size_t groupIndex = 0;
                 groupIndex < sharedIntrinsicsParameters.size();
                 ++groupIndex)
            {
                std::vector<int> constantIndices =
                    sharedIntrinsicConstantIndices[groupIndex];
                if (options.refineSharedRadialDistortion &&
                    (lowOrderDistortionOnly ||
                     !options.refineSharedHighOrderDistortion))
                {
                    constantIndices.insert(
                        constantIndices.end(), {5, 6, 7, 8});
                    std::sort(constantIndices.begin(), constantIndices.end());
                    constantIndices.erase(
                        std::unique(constantIndices.begin(), constantIndices.end()),
                        constantIndices.end());
                }
                double *parameter = sharedIntrinsicsParameters[groupIndex].data();
                problem.SetParameterBlockVariable(parameter);
                problem.SetManifold(
                    parameter,
                    constantIndices.empty()
                        ? nullptr
                        : new ceres::SubsetManifold(9, constantIndices));
            }
        };
    const auto setupEnd = std::chrono::steady_clock::now();
    if (runStagedSelfCalibration)
    {
        solverOptions.max_num_iterations = warmupIterations;
        if (options.progressCallback || options.cancelFlag)
        {
            warmupCallback = std::make_unique<CeresBaIterationCallback>(
                options,
                result.observationCount,
                0,
                warmupIterations,
                totalIterationBudget);
            solverOptions.callbacks = {warmupCallback.get()};
        }
        ceres::Solve(solverOptions, &problem, &warmupSummary);
        result.selfCalibrationStagesRun = 1;

        const bool warmupInterrupted =
            (warmupCallback &&
             (warmupCallback->cancelled() ||
              warmupCallback->progressAborted())) ||
            (options.cancelFlag && options.cancelFlag->load());
        if (warmupSummary.IsSolutionUsable() && !warmupInterrupted)
        {
            int refinementIterations =
                totalIterationBudget - warmupIterations;
            const bool runLowOrderDistortionStage =
                options.refineSharedRadialDistortion &&
                options.refineSharedHighOrderDistortion &&
                refinementIterations >= 6;
            lowOrderStageAttempted = runLowOrderDistortionStage;
            int refinementOffset = warmupIterations;
            if (runLowOrderDistortionStage)
            {
                const int lowOrderIterations =
                    std::max(2, refinementIterations / 2);
                applySharedIntrinsicManifolds(true);
                solverOptions.max_num_iterations = lowOrderIterations;
                solverOptions.callbacks.clear();
                if (options.progressCallback || options.cancelFlag)
                {
                    lowOrderCallback =
                        std::make_unique<CeresBaIterationCallback>(
                            options,
                            result.observationCount,
                            refinementOffset,
                            lowOrderIterations,
                            totalIterationBudget);
                    solverOptions.callbacks = {lowOrderCallback.get()};
                }
                ceres::Solve(solverOptions, &problem, &lowOrderSummary);
                result.selfCalibrationStagesRun = 2;
                refinementOffset += lowOrderIterations;
                refinementIterations -= lowOrderIterations;
            }

            const bool lowOrderInterrupted =
                (lowOrderCallback &&
                 (lowOrderCallback->cancelled() ||
                  lowOrderCallback->progressAborted())) ||
                (options.cancelFlag && options.cancelFlag->load());
            if (runLowOrderDistortionStage &&
                (!lowOrderSummary.IsSolutionUsable() || lowOrderInterrupted))
            {
                summary = lowOrderSummary;
            }
            else
            {
                applySharedIntrinsicManifolds(false);
                solverOptions.max_num_iterations = refinementIterations;
                solverOptions.callbacks.clear();
                if (options.progressCallback || options.cancelFlag)
                {
                    refinementCallback =
                        std::make_unique<CeresBaIterationCallback>(
                            options,
                            result.observationCount,
                            refinementOffset,
                            refinementIterations,
                            totalIterationBudget);
                    solverOptions.callbacks = {refinementCallback.get()};
                }
                ceres::Solve(solverOptions, &problem, &summary);
                result.selfCalibrationStagesRun =
                    runLowOrderDistortionStage ? 3 : 2;
            }
        }
        else
        {
            summary = warmupSummary;
        }
    }
    else
    {
        solverOptions.max_num_iterations = totalIterationBudget;
        if (options.progressCallback || options.cancelFlag)
        {
            refinementCallback =
                std::make_unique<CeresBaIterationCallback>(
                    options,
                    result.observationCount,
                    0,
                    totalIterationBudget,
                    totalIterationBudget);
            solverOptions.callbacks = {refinementCallback.get()};
        }
        ceres::Solve(solverOptions, &problem, &summary);
        result.selfCalibrationStagesRun =
            refineSharedIntrinsics ? 1 : 0;
    }
    const auto solveEnd = std::chrono::steady_clock::now();
    result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
    result.solveSeconds = std::chrono::duration<double>(solveEnd - setupEnd).count();
    result.totalSeconds = std::chrono::duration<double>(solveEnd - setupStart).count();
    const auto accumulateSummary =
        [&result](const ceres::Solver::Summary &stageSummary)
        {
            result.ceresSuccessfulSteps += stageSummary.num_successful_steps;
            result.ceresUnsuccessfulSteps += stageSummary.num_unsuccessful_steps;
        };
    if (runStagedSelfCalibration)
    {
        result.ceresInitialCost = warmupSummary.initial_cost;
        accumulateSummary(warmupSummary);
        if (result.selfCalibrationStagesRun == 3)
        {
            accumulateSummary(lowOrderSummary);
        }
        if (result.selfCalibrationStagesRun >= 2)
        {
            accumulateSummary(summary);
        }
    }
    else
    {
        result.ceresInitialCost = summary.initial_cost;
        accumulateSummary(summary);
    }
    result.ceresFinalCost = summary.final_cost;
    std::string solveReport;
    if (runStagedSelfCalibration && result.selfCalibrationStagesRun >= 2)
    {
        solveReport = "预热: " + warmupSummary.BriefReport();
        if (result.selfCalibrationStagesRun == 3)
        {
            solveReport += "；低阶自标定: " + lowOrderSummary.BriefReport();
        }
        solveReport +=
            lowOrderStageAttempted && result.selfCalibrationStagesRun == 2
                ? "；低阶自标定: " + summary.BriefReport()
                : "；完整自标定: " + summary.BriefReport();
    }
    else
    {
        solveReport = summary.BriefReport();
    }
    const std::string solverReport =
        "求解规划: solver=" + result.ceresLinearSolverName +
        "；" + solverPlan.reason;
    result.backendMessage = cudaDeviceMessage.empty()
                                ? solverReport + "；" + solveReport
                                : cudaDeviceMessage + "；" +
                                      solverReport + "；" + solveReport;

    const bool cancelledByFlag =
        (warmupCallback && warmupCallback->cancelled()) ||
        (lowOrderCallback && lowOrderCallback->cancelled()) ||
        (refinementCallback && refinementCallback->cancelled()) ||
        (options.cancelFlag && options.cancelFlag->load());
    const bool progressAborted =
        (warmupCallback && warmupCallback->progressAborted()) ||
        (lowOrderCallback && lowOrderCallback->progressAborted()) ||
        (refinementCallback && refinementCallback->progressAborted());
    if (cancelledByFlag || progressAborted)
    {
        result.solveStatus = BASolveStatus::Cancelled;
        result.solutionUsable = false;
        result.backendMessage =
            (cancelledByFlag ? "Ceres BA 已取消；" : "Ceres BA 被进度回调终止；") +
            result.backendMessage;
        return result;
    }
    if (!summary.IsSolutionUsable())
    {
        result.solveStatus = BASolveStatus::NumericalFailure;
        result.solutionUsable = false;
        result.backendMessage = "Ceres BA 解不可用，保留输入相机和三维点；" +
                                result.backendMessage + "；" + summary.message;
        return result;
    }
    if (summary.termination_type == ceres::NO_CONVERGENCE &&
        result.ceresSuccessfulSteps == 0 &&
        std::isfinite(result.ceresInitialCost) &&
        result.ceresInitialCost > 1.0e-12)
    {
        result.solveStatus = BASolveStatus::NoConvergence;
        result.solutionUsable = false;
        result.backendMessage =
            "Ceres BA 未接受任何 trial step，拒绝写回 no-op 结果；" +
            result.backendMessage;
        return result;
    }

    result.solveStatus = summary.termination_type == ceres::NO_CONVERGENCE
                             ? BASolveStatus::NoConvergence
                             : BASolveStatus::Success;
    result.solutionUsable = true;

    // 阶段 5：把局部参数解码回 PlaScan Camera/BARefinedPoint。随后必须调用统一
    // 终结器重算正深度和 RMS，Ceres 的 solution usable 不能替代摄影测量质量检查。
    double focalScaleSum = 0.0;
    double aspectScaleSum = 0.0;
    double principalOffsetXSum = 0.0;
    double principalOffsetYSum = 0.0;
    double radialK1Sum = 0.0;
    double radialK2Sum = 0.0;
    double radialK3Sum = 0.0;
    double tangentialP1Sum = 0.0;
    double tangentialP2Sum = 0.0;
    int refinedIntrinsicCount = 0;
    for (size_t ci = 0; ci < cameras.size(); ++ci)
    {
        result.refinedCameras[ci] = cameras[ci];
        result.refinedCameras[ci].applyDeltaPose(cameraDeltas[ci].data());
        if (refineSharedIntrinsics)
        {
            const Camera::Intrinsics intrinsics = result.refinedCameras[ci].intrinsics();
            const size_t groupIndex =
                static_cast<size_t>(calibrationParameterByCamera[ci]);
            const auto &parameters = sharedIntrinsicsParameters[groupIndex];
            const double focalPixels = std::exp(parameters[0]);
            const double focalAspect = std::exp(parameters[1]);
            const double principalOffsetX = parameters[2];
            const double principalOffsetY = parameters[3];
            const double radialK1 = parameters[4];
            const double radialK2 = parameters[5];
            const double radialK3 = parameters[6];
            const double tangentialP1 = parameters[7];
            const double tangentialP2 = parameters[8];
            result.refinedCameras[ci].setIntrinsics(
                focalPixels,
                focalPixels * focalAspect,
                intrinsics.principalX + principalOffsetX,
                intrinsics.principalY + principalOffsetY);
            Camera::Distortion distortion = result.refinedCameras[ci].distortion();
            distortion.radialK1 = radialK1;
            distortion.radialK2 = radialK2;
            distortion.radialK3 = radialK3;
            distortion.tangentialP1 = tangentialP1;
            distortion.tangentialP2 = tangentialP2;
            result.refinedCameras[ci].setDistortion(distortion);
            focalScaleSum +=
                focalPixels / sharedFocalReferencePixels[groupIndex];
            aspectScaleSum +=
                focalAspect / sharedAspectReference[groupIndex];
            principalOffsetXSum += principalOffsetX;
            principalOffsetYSum += principalOffsetY;
            radialK1Sum += radialK1;
            radialK2Sum += radialK2;
            radialK3Sum += radialK3;
            tangentialP1Sum += tangentialP1;
            tangentialP2Sum += tangentialP2;
            const bool changed =
                std::abs(focalPixels - intrinsics.focalX) >
                    1.0e-8 * std::max(1.0, std::abs(intrinsics.focalX)) ||
                std::abs(focalPixels * focalAspect - intrinsics.focalY) >
                    1.0e-8 * std::max(1.0, std::abs(intrinsics.focalY)) ||
                std::abs(principalOffsetX) > 1.0e-8 ||
                std::abs(principalOffsetY) > 1.0e-8 ||
                std::abs(radialK1 - cameras[ci].distortion().radialK1) > 1.0e-10 ||
                std::abs(radialK2 - cameras[ci].distortion().radialK2) > 1.0e-10 ||
                std::abs(radialK3 - cameras[ci].distortion().radialK3) > 1.0e-10 ||
                std::abs(tangentialP1 - cameras[ci].distortion().tangentialP1) > 1.0e-10 ||
                std::abs(tangentialP2 - cameras[ci].distortion().tangentialP2) > 1.0e-10;
            if (changed)
            {
                ++refinedIntrinsicCount;
            }
        }
    }

    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        BARefinedPoint &point = result.points[ti];
        if (!activeTrack[ti])
        {
            point.point = tracks[ti].initialPoint;
            point.valid = false;
            point.converged = false;
            point.iterations = 0;
            continue;
        }

        point.point = pointParams[ti];
        point.valid = plamatrix::isFinite(plamatrix::Vec3<double>(point.point));
        point.converged = summary.termination_type == ceres::CONVERGENCE ||
                          summary.termination_type == ceres::USER_SUCCESS;
        point.iterations =
            static_cast<int>(summary.iterations.size()) +
            (result.selfCalibrationStagesRun >= 2
                 ? static_cast<int>(warmupSummary.iterations.size())
                 : 0) +
            (result.selfCalibrationStagesRun == 3
                 ? static_cast<int>(lowOrderSummary.iterations.size())
                 : 0);
    }
    result.refinedCameraCount = options.refineCameraPose ? variableCameraCount : 0;
    if (refineSharedIntrinsics)
    {
        const double cameraCount = static_cast<double>(cameras.size());
        result.refinedSharedFocalScale =
            cameras.empty()
                ? 1.0
                : focalScaleSum / cameraCount;
        result.refinedSharedFocalAspectScale =
            cameras.empty() ? 1.0 : aspectScaleSum / cameraCount;
        result.refinedSharedPrincipalOffsetX =
            cameras.empty() ? 0.0 : principalOffsetXSum / cameraCount;
        result.refinedSharedPrincipalOffsetY =
            cameras.empty() ? 0.0 : principalOffsetYSum / cameraCount;
        result.refinedSharedRadialK1 =
            cameras.empty() ? 0.0 : radialK1Sum / cameraCount;
        result.refinedSharedRadialK2 =
            cameras.empty() ? 0.0 : radialK2Sum / cameraCount;
        result.refinedSharedRadialK3 =
            cameras.empty() ? 0.0 : radialK3Sum / cameraCount;
        result.refinedSharedTangentialP1 =
            cameras.empty() ? 0.0 : tangentialP1Sum / cameraCount;
        result.refinedSharedTangentialP2 =
            cameras.empty() ? 0.0 : tangentialP2Sum / cameraCount;
        result.refinedIntrinsicCount = refinedIntrinsicCount;
        result.refinedCalibrationGroupCount =
            static_cast<int>(sharedIntrinsicsParameters.size());
    }

    finalizeBundleAdjustResult(cameras, tracks, options, &result);
    return result;
#endif
}

} // namespace xjw::detail
