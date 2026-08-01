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

namespace xjw::ba
{

ProjectionCamera makeProjectionCamera(const Camera &camera)
{
    // 将功能丰富的 Camera 冻结为无虚函数投影 POD，供残差和质量复核共享。
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

double computeTrackRms(const std::vector<Camera> &cameras,
                       const BATrack &track,
                       const std::array<double, 3> &point)
{
    double sum = 0.0;
    int count = 0;
    // RMS 采用实际像素残差，不包含 Ceres 鲁棒损失缩放，便于跨后端比较。
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
        stats.median =
            plamatrix::finiteMedian(std::move(absDistances)).value_or(0.0);
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
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const cameraDelta,
                    const T *const point,
                    T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight = T(std::sqrt(safeObservationWeight(observation)));
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

struct FixedPoseSharedFocalReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const point,
                    const T *const sharedFocalLogPixels,
                    T *residuals) const
    {
        const T zeroDelta[6] = {
            T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0)};
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight = T(std::sqrt(safeObservationWeight(observation)));
        if (!xjw::ba::projectWithPoseDeltaAndSharedFocal(
                camera, zeroDelta, point, sharedFocalLogPixels, pixel))
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

struct PoseDeltaSharedFocalReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const cameraDelta,
                    const T *const point,
                    const T *const sharedFocalLogPixels,
                    T *residuals) const
    {
        T pixel[2] = {T(0.0), T(0.0)};
        const T sqrtWeight = T(std::sqrt(safeObservationWeight(observation)));
        if (!xjw::ba::projectWithPoseDeltaAndSharedFocal(
                camera, cameraDelta, point, sharedFocalLogPixels, pixel))
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

struct LaserPlaneResidual
{
    BALaserPlaneConstraint constraint;
    double weight = 1.0;

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
        residuals[0] =
            T(std::sqrt(std::max(0.0, weight * constraint.weight))) * signedDistance;
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
    result.observationCount = countObservations(tracks);
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        BARefinedPoint &point = result.points[ti];
        point.point = tracks[ti].initialPoint;
        point.rmsBefore = computeTrackRms(cameras, tracks[ti], tracks[ti].initialPoint);
        point.rmsAfter = point.rmsBefore;
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

        bool initialPointHasPositiveDepth = true;
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
        }
        if (!initialPointHasPositiveDepth)
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

    // 按镜头/焦段标定组建立独立焦距参数。同组共享绝对焦距，不同组互不吸收误差。
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
        }
    }

    // 求解器规划必须在活动轨迹、可变相机和标定组数量明确后执行；使用原始输入
    // 数量估算会高估显存并导致不必要的 CPU 回退。
    const BACeresSolverPlan solverPlan =
        planCeresSolver(
            options,
            variableCameraCount,
            options.refineSharedFocalLength
                ? static_cast<int>(groupIdToParameterIndex.size())
                : 0,
            activeTrackCount,
            result.observationCount,
            requestGpu && cudaDeviceReady,
            cudaFreeBytes);
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
    std::vector<double> sharedFocalLogPixels(
        groupIdToParameterIndex.size(), 0.0);
    for (size_t groupIndex = 0;
         groupIndex < sharedFocalReferencePixels.size();
         ++groupIndex)
    {
        sharedFocalReferencePixels[groupIndex] =
            plamatrix::finiteMedian(
                std::move(focalSamplesByGroup[groupIndex]))
                .value_or(1.0);
        sharedFocalLogPixels[groupIndex] =
            std::log(sharedFocalReferencePixels[groupIndex]);
    }

    if (options.refineSharedFocalLength)
    {
        const double minimumScale = std::max(1e-6, options.minSharedFocalScale);
        const double maximumScale =
            std::max(minimumScale, options.maxSharedFocalScale);
        for (size_t groupIndex = 0;
             groupIndex < sharedFocalLogPixels.size();
             ++groupIndex)
        {
            double *parameter = &sharedFocalLogPixels[groupIndex];
            problem.AddParameterBlock(parameter, 1);
            problem.SetParameterLowerBound(
                parameter,
                0,
                std::log(
                    sharedFocalReferencePixels[groupIndex] *
                    minimumScale));
            problem.SetParameterUpperBound(
                parameter,
                0,
                std::log(
                    sharedFocalReferencePixels[groupIndex] *
                    maximumScale));
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
            if (options.refineCameraPose && options.refineSharedFocalLength)
            {
                auto *cost =
                    new ceres::AutoDiffCostFunction<PoseDeltaSharedFocalReprojectionResidual,
                                                    2,
                                                    6,
                                                    3,
                                                    1>(
                        new PoseDeltaSharedFocalReprojectionResidual{
                            xjw::ba::makeProjectionCamera(
                                cameras[static_cast<size_t>(observation.cameraIndex)]),
                            observation});
                problem.AddResidualBlock(
                    cost,
                    makeHuberLoss(options.huberDelta),
                    cameraDeltas[static_cast<size_t>(observation.cameraIndex)].data(),
                    pointParams[ti].data(),
                    &sharedFocalLogPixels[
                        static_cast<size_t>(
                            calibrationParameterByCamera[
                                static_cast<size_t>(
                                    observation.cameraIndex)])]);
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
            else if (options.refineSharedFocalLength)
            {
                auto *cost =
                    new ceres::AutoDiffCostFunction<FixedPoseSharedFocalReprojectionResidual,
                                                    2,
                                                    3,
                                                    1>(
                        new FixedPoseSharedFocalReprojectionResidual{
                            xjw::ba::makeProjectionCamera(
                                cameras[static_cast<size_t>(observation.cameraIndex)]),
                            observation});
                problem.AddResidualBlock(cost,
                                         makeHuberLoss(options.huberDelta),
                                         pointParams[ti].data(),
                                         &sharedFocalLogPixels[
                                             static_cast<size_t>(
                                                 calibrationParameterByCamera[
                                                     static_cast<size_t>(
                                                         observation.cameraIndex)])]);
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
                auto *cost = new ceres::AutoDiffCostFunction<LaserPlaneResidual, 1, 3>(
                    new LaserPlaneResidual{constraint, options.laserPlaneWeight});
                problem.AddResidualBlock(cost, makeHuberLoss(options.laserHuberDeltaMeters), pointParams[ti].data());
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
    // 自标定采用“固定焦距预热 -> 释放共享焦距”的两阶段策略。先稳定外参和点，
    // 再释放内参可降低转台/弱基线数据中焦距吸收位姿误差的风险。
    const bool runStagedSelfCalibration =
        options.refineSharedFocalLength &&
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
        for (double &focalParameter : sharedFocalLogPixels)
        {
            problem.SetParameterBlockConstant(&focalParameter);
        }
    }

    std::unique_ptr<CeresBaIterationCallback> warmupCallback;
    std::unique_ptr<CeresBaIterationCallback> refinementCallback;
    ceres::Solver::Summary warmupSummary;
    ceres::Solver::Summary summary;
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
            for (double &focalParameter : sharedFocalLogPixels)
            {
                problem.SetParameterBlockVariable(&focalParameter);
            }
            const int refinementIterations =
                totalIterationBudget - warmupIterations;
            solverOptions.max_num_iterations = refinementIterations;
            solverOptions.callbacks.clear();
            if (options.progressCallback || options.cancelFlag)
            {
                refinementCallback =
                    std::make_unique<CeresBaIterationCallback>(
                        options,
                        result.observationCount,
                        warmupIterations,
                        refinementIterations,
                        totalIterationBudget);
                solverOptions.callbacks = {refinementCallback.get()};
            }
            ceres::Solve(solverOptions, &problem, &summary);
            result.selfCalibrationStagesRun = 2;
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
            options.refineSharedFocalLength ? 1 : 0;
    }
    const auto solveEnd = std::chrono::steady_clock::now();
    result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
    result.solveSeconds = std::chrono::duration<double>(solveEnd - setupEnd).count();
    result.totalSeconds = std::chrono::duration<double>(solveEnd - setupStart).count();
    std::string solveReport;
    if (runStagedSelfCalibration && result.selfCalibrationStagesRun == 2)
    {
        solveReport = "预热: " + warmupSummary.BriefReport() +
                      "；自标定: " + summary.BriefReport();
    }
    else
    {
        solveReport = summary.BriefReport();
    }
    const std::string solverReport =
        "求解规划: " + solverPlan.reason;
    result.backendMessage = cudaDeviceMessage.empty()
                                ? solverReport + "；" + solveReport
                                : cudaDeviceMessage + "；" +
                                      solverReport + "；" + solveReport;

    const bool cancelledByFlag =
        (warmupCallback && warmupCallback->cancelled()) ||
        (refinementCallback && refinementCallback->cancelled()) ||
        (options.cancelFlag && options.cancelFlag->load());
    const bool progressAborted =
        (warmupCallback && warmupCallback->progressAborted()) ||
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
                                result.backendMessage;
        return result;
    }

    result.solveStatus = summary.termination_type == ceres::NO_CONVERGENCE
                             ? BASolveStatus::NoConvergence
                             : BASolveStatus::Success;
    result.solutionUsable = true;

    // 阶段 5：把局部参数解码回 PlaScan Camera/BARefinedPoint。随后必须调用统一
    // 终结器重算正深度和 RMS，Ceres 的 solution usable 不能替代摄影测量质量检查。
    std::vector<double> refinedFocalPixels(sharedFocalLogPixels.size(), 1.0);
    for (size_t groupIndex = 0;
         groupIndex < sharedFocalLogPixels.size();
         ++groupIndex)
    {
        refinedFocalPixels[groupIndex] =
            std::exp(sharedFocalLogPixels[groupIndex]);
    }
    double focalScaleSum = 0.0;
    int refinedIntrinsicCount = 0;
    for (size_t ci = 0; ci < cameras.size(); ++ci)
    {
        result.refinedCameras[ci] = cameras[ci];
        result.refinedCameras[ci].applyDeltaPose(cameraDeltas[ci].data());
        if (options.refineSharedFocalLength)
        {
            const Camera::Intrinsics intrinsics = result.refinedCameras[ci].intrinsics();
            const double focalAspect =
                intrinsics.focalX > 0.0 ? intrinsics.focalY / intrinsics.focalX : 1.0;
            const size_t groupIndex =
                static_cast<size_t>(calibrationParameterByCamera[ci]);
            const double focalPixels = refinedFocalPixels[groupIndex];
            result.refinedCameras[ci].setIntrinsics(
                focalPixels,
                focalPixels * focalAspect,
                intrinsics.principalX,
                intrinsics.principalY);
            focalScaleSum +=
                focalPixels / sharedFocalReferencePixels[groupIndex];
            if (std::abs(focalPixels - intrinsics.focalX) >
                1.0e-8 * std::max(1.0, std::abs(intrinsics.focalX)))
            {
                ++refinedIntrinsicCount;
            }
        }
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
        point.iterations =
            static_cast<int>(summary.iterations.size()) +
            (result.selfCalibrationStagesRun == 2
                 ? static_cast<int>(warmupSummary.iterations.size())
                 : 0);
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
            sumAfter += point.rmsAfter;
            ++countAfter;
            ++result.optimizedTracks;
        }
    }
    result.meanRmsBefore = countBefore > 0 ? sumBefore / static_cast<double>(countBefore) : 0.0;
    result.meanRmsAfter = countAfter > 0 ? sumAfter / static_cast<double>(countAfter) : 0.0;
    result.refinedCameraCount = options.refineCameraPose ? variableCameraCount : 0;
    if (options.refineSharedFocalLength)
    {
        result.refinedSharedFocalScale =
            cameras.empty()
                ? 1.0
                : focalScaleSum / static_cast<double>(cameras.size());
        result.refinedIntrinsicCount = refinedIntrinsicCount;
        result.refinedCalibrationGroupCount =
            static_cast<int>(sharedFocalLogPixels.size());
    }

    finalizeBundleAdjustResult(cameras, tracks, options, &result);

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
