#include "BundleAdjustNativeCudaKernels.cuh"

#include "BundleAdjustNativeCudaDeviceTypes.cuh"
#include "BundleAdjustNativeCudaPointKernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace xjw::detail::native_cuda
{

namespace
{

using Clock = std::chrono::steady_clock;

static_assert(std::is_standard_layout<HostObservation>::value, "HostObservation must be standard layout");
static_assert(std::is_standard_layout<DeviceObservation>::value, "DeviceObservation must be standard layout");
static_assert(sizeof(HostObservation) == sizeof(DeviceObservation), "Host/Device observation layout size mismatch");
static_assert(offsetof(HostObservation, cameraIndex) == offsetof(DeviceObservation, cameraIndex),
              "cameraIndex layout mismatch");
static_assert(offsetof(HostObservation, pointIndex) == offsetof(DeviceObservation, pointIndex),
              "pointIndex layout mismatch");
static_assert(offsetof(HostObservation, u) == offsetof(DeviceObservation, u),
              "u layout mismatch");
static_assert(offsetof(HostObservation, v) == offsetof(DeviceObservation, v),
              "v layout mismatch");
static_assert(offsetof(HostObservation, weight) == offsetof(DeviceObservation, weight),
              "weight layout mismatch");

double elapsedSeconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

template <typename T>
bool copyToDevice(const std::vector<T> &host, T **device, KernelRunSummary *summary, const char *label)
{
    *device = nullptr;
    if (host.empty())
    {
        return true;
    }

    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(device), host.size() * sizeof(T));
    if (status != cudaSuccess)
    {
        std::snprintf(summary->message,
                      sizeof(summary->message),
                      "cudaMalloc %s 失败: %s",
                      label,
                      cudaGetErrorString(status));
        return false;
    }

    status = cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::snprintf(summary->message,
                      sizeof(summary->message),
                      "cudaMemcpy %s 失败: %s",
                      label,
                      cudaGetErrorString(status));
        cudaFree(*device);
        *device = nullptr;
        return false;
    }
    return true;
}

bool copyObservationsToDevice(const std::vector<HostObservation> &host,
                              DeviceObservation **device,
                              KernelRunSummary *summary)
{
    *device = nullptr;
    if (host.empty())
    {
        return true;
    }

    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(device), host.size() * sizeof(DeviceObservation));
    if (status != cudaSuccess)
    {
        std::snprintf(summary->message,
                      sizeof(summary->message),
                      "cudaMalloc observations 失败: %s",
                      cudaGetErrorString(status));
        return false;
    }

    status = cudaMemcpy(*device, host.data(), host.size() * sizeof(DeviceObservation), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::snprintf(summary->message,
                      sizeof(summary->message),
                      "cudaMemcpy observations 失败: %s",
                      cudaGetErrorString(status));
        cudaFree(*device);
        *device = nullptr;
        return false;
    }
    return true;
}

bool setCudaError(KernelRunSummary *summary, cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return false;
    }

    std::snprintf(summary->message,
                  sizeof(summary->message),
                  "%s 失败: %s",
                  operation,
                  cudaGetErrorString(status));
    return true;
}

bool projectHostDeviceCamera(const DeviceCamera &camera, const DevicePoint &point, double pixel[2])
{
    const double dx = point.xyz[0] - camera.cameraCenter[0];
    const double dy = point.xyz[1] - camera.cameraCenter[1];
    const double dz = point.xyz[2] - camera.cameraCenter[2];
    const double xCam = camera.cameraToWorldRotation[0] * dx +
                        camera.cameraToWorldRotation[3] * dy +
                        camera.cameraToWorldRotation[6] * dz;
    const double yCam = camera.cameraToWorldRotation[1] * dx +
                        camera.cameraToWorldRotation[4] * dy +
                        camera.cameraToWorldRotation[7] * dz;
    const double zCam = camera.cameraToWorldRotation[2] * dx +
                        camera.cameraToWorldRotation[5] * dy +
                        camera.cameraToWorldRotation[8] * dz;
    if (!(zCam > 1e-9))
    {
        return false;
    }

    const double x = xCam / zCam;
    const double y = yCam / zCam;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 +
                          camera.radialK1 * r2 +
                          camera.radialK2 * r4 +
                          camera.radialK3 * r6;
    const double xy2 = 2.0 * x * y;
    const double xd = x * radial +
                      camera.tangentialP1 * xy2 +
                      camera.tangentialP2 * (r2 + 2.0 * x * x);
    const double yd = y * radial +
                      camera.tangentialP1 * (r2 + 2.0 * y * y) +
                      camera.tangentialP2 * xy2;
    pixel[0] = static_cast<double>(camera.uAxisSign) * camera.focalX * xd + camera.principalX;
    pixel[1] = static_cast<double>(camera.vAxisSign) * camera.focalY * yd + camera.principalY;
    return std::isfinite(pixel[0]) && std::isfinite(pixel[1]);
}

double hostCost(const std::vector<DeviceCamera> &cameras,
                const std::vector<DevicePoint> &points,
                const std::vector<HostObservation> &observations)
{
    double cost = 0.0;
    for (const HostObservation &observation : observations)
    {
        const DeviceCamera &camera = cameras[static_cast<size_t>(observation.cameraIndex)];
        const DevicePoint &point = points[static_cast<size_t>(observation.pointIndex)];
        double pixel[2] = {0.0, 0.0};
        if (!projectHostDeviceCamera(camera, point, pixel))
        {
            continue;
        }

        const double du = pixel[0] - observation.u;
        const double dv = pixel[1] - observation.v;
        const double weight = std::isfinite(observation.weight) ? std::max(0.0, observation.weight) : 0.0;
        cost += weight * (du * du + dv * dv);
    }
    return cost;
}

void freeDeviceBuffers(DeviceCamera *cameras,
                       DevicePoint *points,
                       DeviceObservation *observations,
                       int *acceptedIterations,
                       KernelRunSummary *summary)
{
    const auto releaseStart = Clock::now();
    cudaFree(cameras);
    cudaFree(points);
    cudaFree(observations);
    cudaFree(acceptedIterations);
    if (summary)
    {
        summary->releaseSeconds += elapsedSeconds(releaseStart, Clock::now());
    }
}

} // namespace

KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           int maxPcgIterations,
                                           double pcgTolerance,
                                           double huberDelta,
                                           double initialDamping)
{
    KernelRunSummary summary;
    if (!workset)
    {
        std::snprintf(summary.message, sizeof(summary.message), "native_cuda workset 为空");
        return summary;
    }
    if (workset->points.empty() || workset->observations.empty() || workset->cameras.empty())
    {
        std::snprintf(summary.message, sizeof(summary.message), "native_cuda workset 缺少有效数据");
        return summary;
    }

    const auto deviceSelectStart = Clock::now();
    const cudaError_t setDeviceStatus = cudaSetDevice(deviceId);
    summary.deviceSelectSeconds += elapsedSeconds(deviceSelectStart, Clock::now());
    if (setDeviceStatus != cudaSuccess)
    {
        std::snprintf(summary.message,
                      sizeof(summary.message),
                      "cudaSetDevice 失败: %s",
                      cudaGetErrorString(setDeviceStatus));
        return summary;
    }

    const auto stagingStart = Clock::now();
    std::vector<DeviceCamera> hostCameras;
    hostCameras.reserve(workset->cameras.size());
    for (const HostCamera &camera : workset->cameras)
    {
        hostCameras.push_back(makeDeviceCamera(camera));
    }

    std::vector<DevicePoint> hostPoints;
    hostPoints.reserve(workset->points.size());
    for (const HostPoint &point : workset->points)
    {
        hostPoints.push_back(makeDevicePoint(point));
    }
    summary.stagingSeconds += elapsedSeconds(stagingStart, Clock::now());

    auto hostCostStart = Clock::now();
    summary.initialCost = hostCost(hostCameras, hostPoints, workset->observations);
    summary.hostCostSeconds += elapsedSeconds(hostCostStart, Clock::now());

    DeviceCamera *deviceCameras = nullptr;
    DevicePoint *devicePoints = nullptr;
    DeviceObservation *deviceObservations = nullptr;
    int *deviceAcceptedIterations = nullptr;
    const auto uploadStart = Clock::now();
    if (!copyToDevice(hostCameras, &deviceCameras, &summary, "cameras") ||
        !copyToDevice(hostPoints, &devicePoints, &summary, "points") ||
        !copyObservationsToDevice(workset->observations, &deviceObservations, &summary))
    {
        summary.uploadSeconds += elapsedSeconds(uploadStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }

    cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(&deviceAcceptedIterations), hostPoints.size() * sizeof(int));
    if (setCudaError(&summary, status, "cudaMalloc acceptedIterations"))
    {
        summary.uploadSeconds += elapsedSeconds(uploadStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }

    status = cudaMemset(deviceAcceptedIterations, 0, hostPoints.size() * sizeof(int));
    if (setCudaError(&summary, status, "cudaMemset acceptedIterations"))
    {
        summary.uploadSeconds += elapsedSeconds(uploadStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }
    summary.uploadSeconds += elapsedSeconds(uploadStart, Clock::now());

    const int pointCount = static_cast<int>(hostPoints.size());
    const int blockSize = 128;
    const int gridSize = (pointCount + blockSize - 1) / blockSize;
    const auto kernelStart = Clock::now();
    optimizePointsKernel<<<gridSize, blockSize>>>(deviceCameras,
                                                  devicePoints,
                                                  deviceObservations,
                                                  pointCount,
                                                  std::max(1, maxIterations),
                                                  huberDelta,
                                                  initialDamping,
                                                  deviceAcceptedIterations);
    status = cudaGetLastError();
    if (setCudaError(&summary, status, "native_cuda optimizePointsKernel"))
    {
        summary.kernelSeconds += elapsedSeconds(kernelStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }

    status = cudaDeviceSynchronize();
    if (setCudaError(&summary, status, "cudaDeviceSynchronize"))
    {
        summary.kernelSeconds += elapsedSeconds(kernelStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }
    summary.kernelSeconds += elapsedSeconds(kernelStart, Clock::now());

    const auto downloadStart = Clock::now();
    status = cudaMemcpy(hostPoints.data(),
                        devicePoints,
                        hostPoints.size() * sizeof(DevicePoint),
                        cudaMemcpyDeviceToHost);
    if (setCudaError(&summary, status, "cudaMemcpy points back"))
    {
        summary.downloadSeconds += elapsedSeconds(downloadStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }

    std::vector<int> acceptedIterations(hostPoints.size(), 0);
    status = cudaMemcpy(acceptedIterations.data(),
                        deviceAcceptedIterations,
                        acceptedIterations.size() * sizeof(int),
                        cudaMemcpyDeviceToHost);
    if (setCudaError(&summary, status, "cudaMemcpy acceptedIterations back"))
    {
        summary.downloadSeconds += elapsedSeconds(downloadStart, Clock::now());
        freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);
        return summary;
    }
    summary.downloadSeconds += elapsedSeconds(downloadStart, Clock::now());
    freeDeviceBuffers(deviceCameras, devicePoints, deviceObservations, deviceAcceptedIterations, &summary);

    for (size_t i = 0; i < hostPoints.size(); ++i)
    {
        workset->points[i].xyz = {{hostPoints[i].xyz[0], hostPoints[i].xyz[1], hostPoints[i].xyz[2]}};
        summary.acceptedSteps += acceptedIterations[i];
    }

    hostCostStart = Clock::now();
    summary.finalCost = hostCost(hostCameras, hostPoints, workset->observations);
    summary.hostCostSeconds += elapsedSeconds(hostCostStart, Clock::now());
    summary.ok = true;
    summary.activeObservations = static_cast<int>(workset->observations.size());
    summary.pcgIterations = 0;
    summary.linearResidual = summary.initialCost > 0.0
                                 ? std::sqrt(std::max(0.0, summary.finalCost / summary.initialCost))
                                 : pcgTolerance;
    summary.rejectedSteps = 0;
    std::snprintf(summary.message, sizeof(summary.message), "native_cuda CUDA 点块求解完成");
    (void)maxPcgIterations;
    return summary;
}

} // namespace xjw::detail::native_cuda
