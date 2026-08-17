// =============================================================================
// 文件: PatchMatchCUDA.cu
// 模块: MVS - GPU PatchMatch 深度图估计
// 说明:
//   实现带法向量假设的 PatchMatch 深度估计（参考 COLMAP patch_match_cuda.cu）。
//   关键改进：每像素维护 (depth, normal) 平面假设，使用平面单应 NCC，
//   彻底消除只用标量深度时产生的辐射状条纹伪影。
// =============================================================================

#include "PatchMatchCUDA.h"
#include "PatchMatchHostUtils.h"
#include "PatchMatchPhotometricCost.h"
#include "Logger.h"
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <cstdint>
#include <thread>
#include <atomic>
#include <array>
#include <memory>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        const char *_cuda_msg = cudaGetErrorString(_e); \
        LOG_ERROR("[MVS][PatchMatch][CUDA] %s:%d %s", __FILE__, __LINE__, _cuda_msg); \
        if (errorMsg) { \
            *errorMsg = std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " " + _cuda_msg; \
        } \
        return false; \
    } } while(0)

namespace xjw 
{
namespace mvs 
{

namespace
{

struct HostPinholeCamera
{
    float focalX = 0.0f;
    float focalY = 0.0f;
    float principalX = 0.0f;
    float principalY = 0.0f;
};

HostPinholeCamera makeHostPinholeCamera(const FramePinholeCamera &camera, int downsampleFactor)
{
    const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
    const float scale = 1.0f / static_cast<float>(std::max(1, downsampleFactor));

    HostPinholeCamera result;
    result.focalX = static_cast<float>(intrinsics.focalX) * scale;
    result.focalY = static_cast<float>(intrinsics.focalY) * scale;
    result.principalX = static_cast<float>(intrinsics.principalX) * scale;
    result.principalY = static_cast<float>(intrinsics.principalY) * scale;
    return result;
}

template <typename T>
class ReusableCudaBuffer
{
public:
    T *ptr = nullptr;
    std::size_t capacity = 0;

    cudaError_t reserve(std::size_t count)
    {
        if (count <= capacity)
        {
            return cudaSuccess;
        }

        T *replacement = nullptr;
        const cudaError_t allocation_error = cudaMalloc(
            reinterpret_cast<void **>(&replacement), count * sizeof(T));
        if (allocation_error != cudaSuccess)
        {
            return allocation_error;
        }
        if (ptr)
        {
            cudaFree(ptr);
        }
        ptr = replacement;
        capacity = count;
        return cudaSuccess;
    }

    void reset()
    {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
        capacity = 0;
    }
};

template <typename T>
class ReusablePinnedBuffer
{
public:
    T *ptr = nullptr;
    std::size_t capacity = 0;

    cudaError_t reserve(std::size_t count)
    {
        if (count <= capacity)
        {
            return cudaSuccess;
        }

        T *replacement = nullptr;
        const cudaError_t allocation_error = cudaHostAlloc(
            reinterpret_cast<void **>(&replacement),
            count * sizeof(T),
            cudaHostAllocPortable);
        if (allocation_error != cudaSuccess)
        {
            return allocation_error;
        }
        if (ptr)
        {
            cudaFreeHost(ptr);
        }
        ptr = replacement;
        capacity = count;
        return cudaSuccess;
    }

    void reset()
    {
        if (ptr)
        {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
        capacity = 0;
    }
};

struct PatchMatchGpuWorkspace
{
    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    cudaEvent_t uploadsReady = nullptr;
    cudaEvent_t cancellationCheckpoint = nullptr;
    cudaEvent_t computeReady = nullptr;
    cudaEvent_t downloadsReady = nullptr;
    bool initialized = false;

    ReusableCudaBuffer<float> srcData;
    ReusableCudaBuffer<float> depth;
    ReusableCudaBuffer<float> normal;
    ReusableCudaBuffer<float> confidence;
    ReusableCudaBuffer<float> hint;
    ReusableCudaBuffer<float> hintRadius;
    ReusableCudaBuffer<float *> srcPointers;
    ReusableCudaBuffer<std::uint8_t> referenceMask;
    ReusableCudaBuffer<std::uint8_t> sourceMasks;

    ReusablePinnedBuffer<std::uint8_t> sourceDataHost;
    ReusablePinnedBuffer<std::uint8_t> sourcePointersHost;
    ReusablePinnedBuffer<std::uint8_t> referenceMaskHost;
    ReusablePinnedBuffer<std::uint8_t> sourceMasksHost;
    ReusablePinnedBuffer<std::uint8_t> hintHost;
    ReusablePinnedBuffer<std::uint8_t> hintRadiusHost;
    ReusablePinnedBuffer<float> depthHost;
    ReusablePinnedBuffer<float> confidenceHost;

    cudaError_t initialize()
    {
        if (initialized)
        {
            return cudaSuccess;
        }

        cudaError_t error = cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking);
        if (error != cudaSuccess)
        {
            return error;
        }
        error = cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking);
        if (error != cudaSuccess)
        {
            reset();
            return error;
        }
        for (cudaEvent_t *event : {&uploadsReady,
                                  &cancellationCheckpoint,
                                  &computeReady,
                                  &downloadsReady})
        {
            error = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
            if (error != cudaSuccess)
            {
                reset();
                return error;
            }
        }
        initialized = true;
        return cudaSuccess;
    }

    void reset()
    {
        if (computeStream)
        {
            cudaStreamSynchronize(computeStream);
        }
        if (transferStream)
        {
            cudaStreamSynchronize(transferStream);
        }

        srcData.reset();
        depth.reset();
        normal.reset();
        confidence.reset();
        hint.reset();
        hintRadius.reset();
        srcPointers.reset();
        referenceMask.reset();
        sourceMasks.reset();
        sourceDataHost.reset();
        sourcePointersHost.reset();
        referenceMaskHost.reset();
        sourceMasksHost.reset();
        hintHost.reset();
        hintRadiusHost.reset();
        depthHost.reset();
        confidenceHost.reset();
        for (cudaEvent_t *event : {&uploadsReady,
                                  &cancellationCheckpoint,
                                  &computeReady,
                                  &downloadsReady})
        {
            if (*event)
            {
                cudaEventDestroy(*event);
                *event = nullptr;
            }
        }
        if (computeStream)
        {
            cudaStreamDestroy(computeStream);
            computeStream = nullptr;
        }
        if (transferStream)
        {
            cudaStreamDestroy(transferStream);
            transferStream = nullptr;
        }
        initialized = false;
    }
};

struct PatchMatchGpuDeviceState
{
    std::mutex executionMutex;
    std::shared_mutex cacheLifetimeMutex;
    PatchMatchGpuWorkspace workspace;
};

std::mutex g_patchMatchGpuDeviceRegistryMutex;
std::unordered_map<int, std::unique_ptr<PatchMatchGpuDeviceState>>
    g_patchMatchGpuDeviceStates;

PatchMatchGpuDeviceState &patchMatchGpuDeviceState(int deviceIndex)
{
    std::lock_guard<std::mutex> lock(g_patchMatchGpuDeviceRegistryMutex);
    auto &state = g_patchMatchGpuDeviceStates[deviceIndex];
    if (!state)
    {
        // Process-lifetime storage avoids CUDA calls during static destruction.
        // cleanupGpuImageCache() performs explicit release while CUDA is alive.
        state = std::make_unique<PatchMatchGpuDeviceState>();
    }
    return *state;
}

class CudaDeviceRestoreGuard
{
public:
    explicit CudaDeviceRestoreGuard(int previousDevice)
        : _previousDevice(previousDevice)
    {
    }

    ~CudaDeviceRestoreGuard()
    {
        if (_previousDevice >= 0)
        {
            cudaSetDevice(_previousDevice);
        }
    }

private:
    int _previousDevice = -1;
};

struct ImageUploadLane
{
    int deviceIndex = -1;
    cudaStream_t stream = nullptr;
    std::vector<std::unique_ptr<ReusablePinnedBuffer<std::uint8_t>>> hosts;

    ~ImageUploadLane()
    {
        reset();
    }

    cudaError_t beginBatch()
    {
        if (!stream)
        {
            return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }
        // The preceding compute call waited for this lane's cache-entry events,
        // so this normally completes immediately. It protects pinned slots if a
        // frame aborted before entering the CUDA execution slot.
        return cudaStreamSynchronize(stream);
    }

    ReusablePinnedBuffer<std::uint8_t> &host(std::size_t index)
    {
        while (hosts.size() <= index)
        {
            hosts.push_back(
                std::make_unique<ReusablePinnedBuffer<std::uint8_t>>());
        }
        return *hosts[index];
    }

    void reset()
    {
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        if (deviceIndex >= 0)
        {
            cudaSetDevice(deviceIndex);
        }
        if (stream)
        {
            cudaStreamSynchronize(stream);
        }
        for (const auto &buffer : hosts)
        {
            buffer->reset();
        }
        hosts.clear();
        if (stream)
        {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (previous_device >= 0)
        {
            cudaSetDevice(previous_device);
        }
    }
};

ImageUploadLane &imageUploadLaneForCurrentThread(int deviceIndex)
{
    thread_local std::unordered_map<int, std::unique_ptr<ImageUploadLane>> lanes;
    auto &lane = lanes[deviceIndex];
    if (!lane)
    {
        lane = std::make_unique<ImageUploadLane>();
        lane->deviceIndex = deviceIndex;
    }
    return *lane;
}

class PatchMatchGpuRunGuard
{
public:
    explicit PatchMatchGpuRunGuard(PatchMatchGpuWorkspace &workspace)
        : _workspace(workspace)
    {
    }

    ~PatchMatchGpuRunGuard()
    {
        if (!_completed)
        {
            cudaStreamSynchronize(_workspace.computeStream);
            cudaStreamSynchronize(_workspace.transferStream);
        }
    }

    void markCompleted()
    {
        _completed = true;
    }

private:
    PatchMatchGpuWorkspace &_workspace;
    bool _completed = false;
};

cudaError_t stageHostToDeviceCopy(ReusablePinnedBuffer<std::uint8_t> &hostBuffer,
                                  const void *source,
                                  std::size_t bytes,
                                  void *destination,
                                  cudaStream_t stream)
{
    const cudaError_t reserve_error = hostBuffer.reserve(bytes);
    if (reserve_error != cudaSuccess)
    {
        return reserve_error;
    }
    std::memcpy(hostBuffer.ptr, source, bytes);
    return cudaMemcpyAsync(destination,
                           hostBuffer.ptr,
                           bytes,
                           cudaMemcpyHostToDevice,
                           stream);
}


struct SrcImageCacheKey
{
    int deviceIndex = 0;
    uintptr_t hostData = 0;
    int rows = 0;
    int cols = 0;
    size_t step = 0;
    int ds = 1;

    bool operator==(const SrcImageCacheKey &other) const
    {
        return deviceIndex == other.deviceIndex
            && hostData == other.hostData
            && rows == other.rows
            && cols == other.cols
            && step == other.step
            && ds == other.ds;
    }
};

struct SrcImageCacheKeyHash
{
    size_t operator()(const SrcImageCacheKey &key) const
    {
        size_t hash = std::hash<int>{}(key.deviceIndex);
        hash ^= std::hash<uintptr_t>{}(key.hostData) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.rows) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.cols) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<size_t>{}(key.step) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.ds) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct SrcImageCacheEntry
{
    // Keep the OpenCV allocation alive while hostData is used as the cache key.
    // Otherwise a later cv::Mat may reuse the same address and incorrectly hit
    // an entry containing pixels from a different image.
    cv::Mat hostOwner;
    float *devicePtr = nullptr;
    cudaEvent_t readyEvent = nullptr;
    int scaledW = 0;
    int scaledH = 0;
    size_t bytes = 0;
    uint64_t lastUseTick = 0;
    int pinCount = 0;
};

std::unordered_map<SrcImageCacheKey, SrcImageCacheEntry, SrcImageCacheKeyHash> g_srcImageGpuCache;
std::mutex g_srcImageGpuCacheMutex;
uint64_t g_srcImageGpuCacheTick = 0;
size_t g_srcImageGpuCacheBytes = 0;

size_t getGrayImageGpuCacheBytes(int deviceIndex)
{
    std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);
    size_t bytes = 0;
    for (const auto &item : g_srcImageGpuCache)
    {
        if (item.first.deviceIndex == deviceIndex)
        {
            bytes += item.second.bytes;
        }
    }
    return bytes;
}

size_t getGrayImageGpuCacheLimitBytes()
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess || totalBytes == 0)
    {
        return 1024ULL * 1024ULL * 1024ULL;
    }

    const size_t oneGiB = 1024ULL * 1024ULL * 1024ULL;
    const size_t sixGiB = 6ULL * oneGiB;
    const size_t reserveBytes = 768ULL * 1024ULL * 1024ULL;

    size_t limitByTotal = totalBytes / 2;
    size_t limitByFree = freeBytes > reserveBytes ? (freeBytes - reserveBytes) : (freeBytes / 2);
    size_t limit = std::min(limitByTotal, limitByFree);
    limit = std::max(limit, oneGiB);
    limit = std::min(limit, sixGiB);
    return limit;
}

void evictSrcImageGpuCacheIfNeeded(size_t needBytes,
                                   int deviceIndex,
                                   cudaStream_t releaseStream = nullptr)
{
    const size_t cacheLimitBytes = getGrayImageGpuCacheLimitBytes();
    auto device_cache_bytes = [deviceIndex]()
    {
        size_t bytes = 0;
        for (const auto &item : g_srcImageGpuCache)
        {
            if (item.first.deviceIndex == deviceIndex)
            {
                bytes += item.second.bytes;
            }
        }
        return bytes;
    };
    while (device_cache_bytes() + needBytes > cacheLimitBytes && !g_srcImageGpuCache.empty())
    {
        auto lruIt = g_srcImageGpuCache.end();
        for (auto it = g_srcImageGpuCache.begin(); it != g_srcImageGpuCache.end(); ++it)
        {
            if (it->first.deviceIndex == deviceIndex &&
                it->second.pinCount == 0 &&
                (lruIt == g_srcImageGpuCache.end() ||
                 it->second.lastUseTick < lruIt->second.lastUseTick))
            {
                lruIt = it;
            }
        }
        if (lruIt == g_srcImageGpuCache.end())
        {
            break;
        }

        if (lruIt->second.devicePtr)
        {
            if (lruIt->second.readyEvent)
            {
                if (releaseStream)
                {
                    cudaStreamWaitEvent(releaseStream,
                                        lruIt->second.readyEvent,
                                        0);
                }
                else
                {
                    cudaEventSynchronize(lruIt->second.readyEvent);
                }
            }
            if (releaseStream)
            {
                cudaFreeAsync(lruIt->second.devicePtr, releaseStream);
            }
            else
            {
                cudaFree(lruIt->second.devicePtr);
            }
        }
        if (lruIt->second.readyEvent)
        {
            cudaEventDestroy(lruIt->second.readyEvent);
        }
        g_srcImageGpuCacheBytes -= lruIt->second.bytes;
        g_srcImageGpuCache.erase(lruIt);
    }
}

bool getOrUploadGrayImageGpu(
    const cv::Mat &srcGray,
    int deviceIndex,
    int scaledW,
    int scaledH,
    int ds,
    cudaStream_t transferStream,
    ReusablePinnedBuffer<std::uint8_t> &uploadHost,
    float **devicePtrOut,
    cudaEvent_t *readyEventOut,
    SrcImageCacheKey *cacheKeyOut,
    bool *cacheHitOut,
    std::string *errorMsg)
{
    if (devicePtrOut == nullptr || readyEventOut == nullptr || cacheKeyOut == nullptr)
    {
        if (errorMsg)
        {
            *errorMsg = "CUDA image-cache output pointer is null";
        }
        return false;
    }
    if (srcGray.empty() || srcGray.data == nullptr)
    {
        if (errorMsg)
        {
            *errorMsg = "source image is empty; cannot upload to GPU";
        }
        return false;
    }

    SrcImageCacheKey key;
    key.deviceIndex = deviceIndex;
    key.hostData = reinterpret_cast<uintptr_t>(srcGray.data);
    key.rows = srcGray.rows;
    key.cols = srcGray.cols;
    key.step = srcGray.step;
    key.ds = ds;

    {
        std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);
        auto it = g_srcImageGpuCache.find(key);
        if (it != g_srcImageGpuCache.end() && it->second.scaledW == scaledW && it->second.scaledH == scaledH)
        {
            it->second.lastUseTick = ++g_srcImageGpuCacheTick;
            ++it->second.pinCount;
            *devicePtrOut = it->second.devicePtr;
            *readyEventOut = it->second.readyEvent;
            *cacheKeyOut = key;
            if (cacheHitOut)
            {
                *cacheHitOut = true;
            }
            return true;
        }
    }

    cv::Mat srcScaled;
    cv::resize(srcGray, srcScaled, cv::Size(scaledW, scaledH), 0, 0, cv::INTER_AREA);
    cv::Mat srcFloat;
    srcScaled.convertTo(srcFloat, CV_32F, 1.f / 255.f);

    const size_t bytes = static_cast<size_t>(scaledW) * static_cast<size_t>(scaledH) * sizeof(float);
    float *newDevicePtr = nullptr;
    cudaError_t allocErr = cudaMallocAsync(
        reinterpret_cast<void **>(&newDevicePtr), bytes, transferStream);
    if (allocErr != cudaSuccess)
    {
        if (errorMsg)
        {
            *errorMsg = std::string("cudaMalloc failed: ") + cudaGetErrorString(allocErr);
        }
        return false;
    }

    const cudaError_t hostReserveError = uploadHost.reserve(bytes);
    if (hostReserveError != cudaSuccess)
    {
        cudaFreeAsync(newDevicePtr, transferStream);
        if (errorMsg)
        {
            *errorMsg = std::string("cudaHostAlloc failed: ") +
                cudaGetErrorString(hostReserveError);
        }
        return false;
    }
    std::memcpy(uploadHost.ptr, srcFloat.ptr<float>(), bytes);
    cudaError_t copyErr = cudaMemcpyAsync(newDevicePtr,
                                          uploadHost.ptr,
                                          bytes,
                                          cudaMemcpyHostToDevice,
                                          transferStream);
    if (copyErr != cudaSuccess)
    {
        cudaFreeAsync(newDevicePtr, transferStream);
        if (errorMsg)
        {
            *errorMsg = std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(copyErr);
        }
        return false;
    }

    cudaEvent_t newReadyEvent = nullptr;
    cudaError_t eventError = cudaEventCreateWithFlags(
        &newReadyEvent, cudaEventDisableTiming);
    if (eventError == cudaSuccess)
    {
        eventError = cudaEventRecord(newReadyEvent, transferStream);
    }
    if (eventError != cudaSuccess)
    {
        if (newReadyEvent)
        {
            cudaEventDestroy(newReadyEvent);
        }
        cudaFreeAsync(newDevicePtr, transferStream);
        if (errorMsg)
        {
            *errorMsg = std::string("CUDA upload event failed: ") +
                cudaGetErrorString(eventError);
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);

        auto existing = g_srcImageGpuCache.find(key);
        if (existing != g_srcImageGpuCache.end() && existing->second.scaledW == scaledW && existing->second.scaledH == scaledH)
        {
            existing->second.lastUseTick = ++g_srcImageGpuCacheTick;
            ++existing->second.pinCount;
            *devicePtrOut = existing->second.devicePtr;
            *readyEventOut = existing->second.readyEvent;
            *cacheKeyOut = key;
            if (cacheHitOut)
            {
                *cacheHitOut = true;
            }
            cudaFreeAsync(newDevicePtr, transferStream);
            cudaEventDestroy(newReadyEvent);
            return true;
        }

        evictSrcImageGpuCacheIfNeeded(bytes, deviceIndex, transferStream);

        SrcImageCacheEntry entry;
        entry.hostOwner = srcGray;
        entry.devicePtr = newDevicePtr;
        entry.readyEvent = newReadyEvent;
        entry.scaledW = scaledW;
        entry.scaledH = scaledH;
        entry.bytes = bytes;
        entry.lastUseTick = ++g_srcImageGpuCacheTick;
        entry.pinCount = 1;

        g_srcImageGpuCache[key] = entry;
        g_srcImageGpuCacheBytes += bytes;

        *devicePtrOut = newDevicePtr;
        *readyEventOut = newReadyEvent;
        *cacheKeyOut = key;
        if (cacheHitOut)
        {
            *cacheHitOut = false;
        }
    }

    return true;
}

class SrcImageCachePinGuard
{
public:
    void add(const SrcImageCacheKey &key)
    {
        _keys.push_back(key);
    }

    ~SrcImageCachePinGuard()
    {
        std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);
        for (const SrcImageCacheKey &key : _keys)
        {
            const auto entry = g_srcImageGpuCache.find(key);
            if (entry != g_srcImageGpuCache.end())
            {
                entry->second.pinCount = std::max(0, entry->second.pinCount - 1);
            }
        }
    }

private:
    std::vector<SrcImageCacheKey> _keys;
};

cv::Mat resizedBinaryMask(const cv::Mat *mask, const cv::Size &targetSize)
{
    if (mask == nullptr || mask->empty())
    {
        return cv::Mat();
    }

    cv::Mat binary;
    cv::compare(*mask, 0, binary, cv::CMP_GT);
    if (binary.size() == targetSize)
    {
        return binary.isContinuous() ? binary : binary.clone();
    }

    cv::Mat resized;
    cv::resize(binary, resized, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
    return resized.isContinuous() ? resized : resized.clone();
}

} // namespace

// =============================================================================
// GPU 常量内存：参考相机内参逆矩阵 = {1/fx, -cx/fx, 1/fy, -cy/fy}
// =============================================================================
__constant__ float c_ref_inv_K[4];

// =============================================================================
// 设备端辅助函数
// =============================================================================

__device__ inline void mat33MulVec3(const float M[9], const float v[3], float out[3]) 
{
    out[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    out[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    out[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

__device__ inline float dot3(const float a[3], const float b[3]) 
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// 双线性插值（边界外返回 -1）
__device__ float bilinear(const float *img, int W, int H, float u, float v)
{
    if (u < 0 || v < 0 || u >= W-1 || v >= H-1) return -1.f;
    int ix = (int)u, iy = (int)v;
    float fx = u - ix, fy = v - iy;
    float v00 = img[iy*W+ix],     v10 = img[iy*W+ix+1];
    float v01 = img[(iy+1)*W+ix], v11 = img[(iy+1)*W+ix+1];
    return (1-fy)*((1-fx)*v00 + fx*v10) + fy*((1-fx)*v01 + fx*v11);
}

__device__ bool bilinearMaskValid(const std::uint8_t *mask,
                                  int width,
                                  int height,
                                  float u,
                                  float v)
{
    if (mask == nullptr)
    {
        return true;
    }
    if (u < 0.0f || v < 0.0f || u >= width - 1 || v >= height - 1)
    {
        return false;
    }
    const int column = static_cast<int>(u);
    const int row = static_cast<int>(v);
    return mask[row * width + column] != 0 &&
           mask[row * width + column + 1] != 0 &&
           mask[(row + 1) * width + column] != 0 &&
           mask[(row + 1) * width + column + 1] != 0;
}

// =============================================================================
// 深度扰动（COLMAP PerturbDepth）
// =============================================================================
__device__ inline float perturbDepth(float perturbation, float depth, curandState *rs)
{
    float d_min = (1.f - perturbation) * depth;
    float d_max = (1.f + perturbation) * depth;
    return d_min + curand_uniform(rs) * (d_max - d_min);
}

__device__ inline uint32_t alignedPatchMatchHash(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__device__ bool referenceGradient(const float *image,
                                  const std::uint8_t *mask,
                                  int width,
                                  int height,
                                  int x,
                                  int y,
                                  float &gradientX,
                                  float &gradientY)
{
    if (x <= 0 || y <= 0 || x + 1 >= width || y + 1 >= height)
    {
        return false;
    }
    if (mask != nullptr
        && (mask[y * width + x - 1] == 0
            || mask[y * width + x + 1] == 0
            || mask[(y - 1) * width + x] == 0
            || mask[(y + 1) * width + x] == 0))
    {
        return false;
    }
    gradientX = 0.5f * (image[y * width + x + 1]
        - image[y * width + x - 1]);
    gradientY = 0.5f * (image[(y + 1) * width + x]
        - image[(y - 1) * width + x]);
    return true;
}

__device__ bool sourceGradient(const float *image,
                               const std::uint8_t *mask,
                               int width,
                               int height,
                               float x,
                               float y,
                               float &gradientX,
                               float &gradientY)
{
    if (!bilinearMaskValid(mask, width, height, x - 1.0f, y)
        || !bilinearMaskValid(mask, width, height, x + 1.0f, y)
        || !bilinearMaskValid(mask, width, height, x, y - 1.0f)
        || !bilinearMaskValid(mask, width, height, x, y + 1.0f))
    {
        return false;
    }
    const float left = bilinear(image, width, height, x - 1.0f, y);
    const float right = bilinear(image, width, height, x + 1.0f, y);
    const float top = bilinear(image, width, height, x, y - 1.0f);
    const float bottom = bilinear(image, width, height, x, y + 1.0f);
    if (left < 0.0f || right < 0.0f || top < 0.0f || bottom < 0.0f)
    {
        return false;
    }
    gradientX = 0.5f * (right - left);
    gradientY = 0.5f * (bottom - top);
    return true;
}

__device__ inline float alignedPatchMatchRandom(uint32_t &state)
{
    state = alignedPatchMatchHash(state + 0x9e3779b9u);
    return static_cast<float>(state & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ inline void faceNormalTowardCamera(int row,
                                              int col,
                                              float normal[3])
{
    const float lengthSquared = dot3(normal, normal);
    if (!(lengthSquared > 1.0e-12f))
    {
        normal[0] = 0.0f;
        normal[1] = 0.0f;
        normal[2] = -1.0f;
        return;
    }
    const float inverseLength = 1.0f / sqrtf(lengthSquared);
    normal[0] *= inverseLength;
    normal[1] *= inverseLength;
    normal[2] *= inverseLength;
    const float ray[3] = {
        c_ref_inv_K[0] * col + c_ref_inv_K[1],
        c_ref_inv_K[2] * row + c_ref_inv_K[3],
        1.0f};
    if (dot3(normal, ray) > 0.0f)
    {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
}

__device__ inline void alignedRandomFacingNormal(int row,
                                                  int col,
                                                  uint32_t &state,
                                                  float normal[3])
{
    const float z = alignedPatchMatchRandom(state) * 2.0f - 1.0f;
    const float angle = alignedPatchMatchRandom(state) * 6.28318530718f;
    const float radius = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    normal[0] = radius * cosf(angle);
    normal[1] = radius * sinf(angle);
    normal[2] = z;
    faceNormalTowardCamera(row, col, normal);
}

__device__ inline void alignedPerturbFacingNormal(int row,
                                                   int col,
                                                   const float normal[3],
                                                   float amount,
                                                   uint32_t &state,
                                                   float output[3])
{
    output[0] = normal[0]
        + amount * (alignedPatchMatchRandom(state) * 2.0f - 1.0f);
    output[1] = normal[1]
        + amount * (alignedPatchMatchRandom(state) * 2.0f - 1.0f);
    output[2] = normal[2]
        + amount * (alignedPatchMatchRandom(state) * 2.0f - 1.0f);
    faceNormalTowardCamera(row, col, output);
}

// =============================================================================
// 法向量处理
// =============================================================================

/// 生成随机单位法向量（Marsaglia 方法），确保朝向相机
__device__ void generateRandomNormal(int row, int col,
                                     curandState *rs, float normal[3])
{
    float v1, v2, s;
    do 
    {
        v1 = 2.f * curand_uniform(rs) - 1.f;
        v2 = 2.f * curand_uniform(rs) - 1.f;
        s  = v1*v1 + v2*v2;
    } while (s >= 1.f || s == 0.f);

    float sn = sqrtf(1.f - s);
    normal[0] = 2.f * v1 * sn;
    normal[1] = 2.f * v2 * sn;
    normal[2] = 1.f - 2.f * s;

    float ray[3] = { c_ref_inv_K[0]*col + c_ref_inv_K[1],
                     c_ref_inv_K[2]*row + c_ref_inv_K[3],
                     1.f };
    if (dot3(normal, ray) > 0) 
    {
        normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2];
    }
}

/// 扰动法向量（小角度随机旋转）
__device__ void perturbNormal(int row, int col,
                               float perturbation,
                               const float n[3], curandState *rs,
                               float out[3], int trial = 0)
{
    float a1 = (curand_uniform(rs) - 0.5f) * perturbation;
    float a2 = (curand_uniform(rs) - 0.5f) * perturbation;
    float a3 = (curand_uniform(rs) - 0.5f) * perturbation;

    float sa1=sinf(a1), ca1=cosf(a1);
    float sa2=sinf(a2), ca2=cosf(a2);
    float sa3=sinf(a3), ca3=cosf(a3);

    float R[9];
    R[0]=ca2*ca3;  R[1]=-ca2*sa3; R[2]=sa2;
    R[3]=ca1*sa3+ca3*sa1*sa2; R[4]=ca1*ca3-sa1*sa2*sa3; R[5]=-ca2*sa1;
    R[6]=sa1*sa3-ca1*ca3*sa2; R[7]=ca3*sa1+ca1*sa2*sa3; R[8]=ca1*ca2;

    mat33MulVec3(R, n, out);

    float ray[3] = { c_ref_inv_K[0]*col + c_ref_inv_K[1],
                     c_ref_inv_K[2]*row + c_ref_inv_K[3],
                     1.f };
    if (dot3(out, ray) >= 0.f) 
    {
        if (trial < 3) { perturbNormal(row, col, 0.5f*perturbation, n, rs, out, trial+1); return; }
        out[0]=n[0]; out[1]=n[1]; out[2]=n[2];
        return;
    }
    float inv = rsqrtf(dot3(out, out));
    out[0]*=inv; out[1]*=inv; out[2]*=inv;
}

// =============================================================================
// 深度沿平面传播：将 (row1,col1) 处的平面假设 (d1,n1) 传播到 (row2,col2) 射线
// =============================================================================
__device__ float propagateDepth(float d1, const float n1[3],
                                  float row1, float col1,
                                  float row2, float col2)
{
    float X1[3] = { d1*(c_ref_inv_K[0]*col1 + c_ref_inv_K[1]),
                    d1*(c_ref_inv_K[2]*row1 + c_ref_inv_K[3]),
                    d1 };
    float ray2[3] = { c_ref_inv_K[0]*col2 + c_ref_inv_K[1],
                      c_ref_inv_K[2]*row2 + c_ref_inv_K[3],
                      1.f };
    float denom = dot3(n1, ray2);
    if (fabsf(denom) < 1e-6f) return d1;
    float t = dot3(n1, X1) / denom;
    return (t > 0.f) ? t : d1;
}

// =============================================================================
// 平面单应性合成（参考 COLMAP ComposeHomography）
// srcData: {fx_s,cx_s,fy_s,cy_s, R_rel[9], T_rel[3]}  共 16 floats
// H 将参考帧像素 [col,row,1] 映射到源帧像素（齐次坐标）
// =============================================================================
__device__ void composeHomography(int row, int col,
                                   float depth, const float normal[3],
                                   const float *srcData,
                                   float H[9])
{
    float fx_s=srcData[0], cx_s=srcData[1];
    float fy_s=srcData[2], cy_s=srcData[3];
    const float *R = srcData + 4;
    const float *T = srcData + 13;

    float dist = depth * (normal[0]*(c_ref_inv_K[0]*col + c_ref_inv_K[1]) +
                          normal[1]*(c_ref_inv_K[2]*row + c_ref_inv_K[3]) +
                          normal[2]);
    if (fabsf(dist) < 1e-6f) dist = 1e-6f;
    float inv_d = 1.f / dist;

    float iN0 = inv_d*normal[0], iN1 = inv_d*normal[1], iN2 = inv_d*normal[2];

    H[0] = c_ref_inv_K[0] * (fx_s*(R[0]+iN0*T[0]) + cx_s*(R[6]+iN0*T[2]));
    H[1] = c_ref_inv_K[2] * (fx_s*(R[1]+iN1*T[0]) + cx_s*(R[7]+iN1*T[2]));
    H[2] = fx_s*(R[2]+iN2*T[0]) + cx_s*(R[8]+iN2*T[2])
         + c_ref_inv_K[1]*(fx_s*(R[0]+iN0*T[0]) + cx_s*(R[6]+iN0*T[2]))
         + c_ref_inv_K[3]*(fx_s*(R[1]+iN1*T[0]) + cx_s*(R[7]+iN1*T[2]));
    H[3] = c_ref_inv_K[0] * (fy_s*(R[3]+iN0*T[1]) + cy_s*(R[6]+iN0*T[2]));
    H[4] = c_ref_inv_K[2] * (fy_s*(R[4]+iN1*T[1]) + cy_s*(R[7]+iN1*T[2]));
    H[5] = fy_s*(R[5]+iN2*T[1]) + cy_s*(R[8]+iN2*T[2])
         + c_ref_inv_K[1]*(fy_s*(R[3]+iN0*T[1]) + cy_s*(R[6]+iN0*T[2]))
         + c_ref_inv_K[3]*(fy_s*(R[4]+iN1*T[1]) + cy_s*(R[7]+iN1*T[2]));
    H[6] = c_ref_inv_K[0] * (R[6]+iN0*T[2]);
    H[7] = c_ref_inv_K[2] * (R[7]+iN1*T[2]);
    H[8] = R[8] + c_ref_inv_K[1]*(R[6]+iN0*T[2])
                + c_ref_inv_K[3]*(R[7]+iN1*T[2]) + iN2*T[2];
}

// =============================================================================
// 平面单应 NCC：用 H 映射 patch，计算 NCC（范围 [0,1]）
// =============================================================================
__device__ float computeHomographyNCC(
    int u_r, int v_r,
    float depth, const float normal[3],
    const float *refImg, int refW, int refH,
    const float *srcImg, int srcW, int srcH,
    const std::uint8_t *refMask,
    const std::uint8_t *srcMask,
    const float *srcData,
    int patchHalf,
    float minimumMaskedSupportRatio)
{
    float H[9];
    composeHomography(v_r, u_r, depth, normal, srcData, H);

    PatchRobustPhotometricAccumulator accumulator;
    const bool maskAware = refMask != nullptr || srcMask != nullptr;
    const int r = patchHalf;
    const float centerZ = H[6] * u_r + H[7] * v_r + H[8];
    const float centerU = fabsf(centerZ) > 1.0e-6f
        ? (H[0] * u_r + H[1] * v_r + H[2]) / centerZ
        : -1.0f;
    const float centerV = fabsf(centerZ) > 1.0e-6f
        ? (H[3] * u_r + H[4] * v_r + H[5]) / centerZ
        : -1.0f;
    const bool centerValid = centerU >= 0.0f && centerV >= 0.0f
        && bilinearMaskValid(srcMask, srcW, srcH, centerU, centerV);
    const float referenceCenter = refImg[v_r * refW + u_r];
    const float sourceCenter = centerValid
        ? bilinear(srcImg, srcW, srcH, centerU, centerV)
        : -1.0f;

    for (int dv = -r; dv <= r; ++dv) 
    {
        for (int du = -r; du <= r; ++du) 
        {
            int pu = u_r + du, pv = v_r + dv;
            if (pu<0||pu>=refW||pv<0||pv>=refH) continue;

            if (refMask != nullptr && refMask[pv * refW + pu] == 0)
            {
                // Reference-mask exclusions are not candidate observations.
                // Keeping them out of the denominator preserves foreground
                // boundary support while source-mask failures still reject
                // genuinely invalid paired samples.
                continue;
            }
            float valRef = refImg[pv*refW + pu];

            float ws_c = H[0]*pu + H[1]*pv + H[2];
            float ws_r = H[3]*pu + H[4]*pv + H[5];
            float ws_z = H[6]*pu + H[7]*pv + H[8];
            if (fabsf(ws_z) < 1e-6f)
            {
                accumulator.addIntensityCandidate(false);
                accumulator.addGradientCandidate(false);
                accumulator.addCensusCandidate(false);
                continue;
            }
            float u_s = ws_c / ws_z;
            float v_s = ws_r / ws_z;

            float valSrc = bilinear(srcImg, srcW, srcH, u_s, v_s);
            if (valSrc < 0 || !bilinearMaskValid(srcMask, srcW, srcH, u_s, v_s))
            {
                accumulator.addIntensityCandidate(false);
                accumulator.addGradientCandidate(false);
                accumulator.addCensusCandidate(false);
                continue;
            }

            accumulator.addIntensityCandidate(true, valRef, valSrc);
            float referenceGradientX = 0.0f;
            float referenceGradientY = 0.0f;
            float sourceGradientX = 0.0f;
            float sourceGradientY = 0.0f;
            const bool gradientValid = referenceGradient(
                refImg,
                refMask,
                refW,
                refH,
                pu,
                pv,
                referenceGradientX,
                referenceGradientY)
                && sourceGradient(
                    srcImg,
                    srcMask,
                    srcW,
                    srcH,
                    u_s,
                    v_s,
                    sourceGradientX,
                    sourceGradientY);
            accumulator.addGradientCandidate(
                gradientValid,
                referenceGradientX,
                referenceGradientY,
                sourceGradientX,
                sourceGradientY);
            accumulator.addCensusCandidate(
                centerValid && sourceCenter >= 0.0f,
                valRef - referenceCenter,
                valSrc - sourceCenter);
        }
    }

    return accumulator.score(maskAware, minimumMaskedSupportRatio);
}

// =============================================================================
// CUDA Kernels
// =============================================================================

__device__ inline void pixelDepthSearchBounds(
    int idx,
    const float *hintDepth,
    const float *hintRadius,
    float zNear,
    float zFar,
    float &localNear,
    float &localFar)
{
    localNear = zNear;
    localFar = zFar;
    if (hintDepth == nullptr)
    {
        return;
    }

    const float center = hintDepth[idx];
    if (!isfinite(center) || center <= 0.0f || center < zNear || center > zFar)
    {
        return;
    }

    float radius = hintRadius != nullptr ? hintRadius[idx] : 0.0f;
    if (!isfinite(radius) || radius <= 0.0f)
    {
        radius = fmaxf(
            center * kDefaultPatchMatchHintRadiusRatio,
            (zFar - zNear) * 0.01f);
    }
    localNear = fmaxf(zNear, center - radius);
    localFar = fminf(zFar, center + radius);
    if (localFar <= localNear)
    {
        localNear = zNear;
        localFar = zFar;
    }
}

// =============================================================================
// 单假设多源 NCC 代价评估（代价 = 2 - 2*NCC，范围 [0,2]，越小越好）
// =============================================================================
__device__ float evalHypCost(
    int col, int row, float depth, const float normal[3],
    const float *refImg, int refW, int refH,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    int patchHalf, float minimumMaskedSupportRatio)
{
    if (depth <= 0.f) return 2.f;
    float scores[kMaxPatchMatchSourceViews] = {};
    const int source_count = numSrc < kMaxPatchMatchSourceViews
                                 ? numSrc
                                 : kMaxPatchMatchSourceViews;
    for (int si = 0; si < source_count; ++si)
    {
        const float *srcImg = srcImgs[si];
        float ncc = computeHomographyNCC(
            col, row, depth, normal,
            refImg, refW, refH,
            srcImg, srcW, srcH,
            refMask,
            srcMasks == nullptr ? nullptr : srcMasks + si * srcW * srcH,
            srcDatas + si * 16,
            patchHalf,
            minimumMaskedSupportRatio);
        scores[si] = ncc;
    }
    const float robust_ncc = robustMultiSourceNcc(scores, source_count);
    return 2.f - 2.f * robust_ncc;  // [0,2]
}

// Deterministic inverse-depth initialization shared semantically with the
// OpenCL backend. A strong fronto-parallel seed avoids backend-specific random
// basins before both implementations enter the same plane propagation stage.
__global__ void kernelInitializePlanes(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    int depthSampleCount,
    float minimumMaskedSupportRatio)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= W || row >= H)
    {
        return;
    }

    const int idx = row * W + col;
    if (refMask != nullptr && refMask[idx] == 0)
    {
        depthMap[idx] = 0.0f;
        normalMap[idx * 3] = 0.0f;
        normalMap[idx * 3 + 1] = 0.0f;
        normalMap[idx * 3 + 2] = -1.0f;
        confMap[idx] = 0.0f;
        return;
    }

    float localNear = zNear;
    float localFar = zFar;
    pixelDepthSearchBounds(idx,
                           hintDepth,
                           hintRadius,
                           zNear,
                           zFar,
                           localNear,
                           localFar);
    const int coarseSamples = max(16, min(96, depthSampleCount));
    const float inverseFar = 1.0f / localFar;
    const float inverseNear = 1.0f / localNear;
    const float inverseStep = (inverseNear - inverseFar)
        / static_cast<float>(coarseSamples - 1);
    const float normal[3] = {0.0f, 0.0f, -1.0f};
    float bestDepth = 0.0f;
    float bestCost = 2.0f;

    for (int sampleIndex = 0; sampleIndex < coarseSamples; ++sampleIndex)
    {
        const float inverseDepth = inverseFar
            + inverseStep * static_cast<float>(sampleIndex);
        const float depth = 1.0f / inverseDepth;
        const float cost = evalHypCost(col, row, depth, normal,
                                       refImg, W, H,
                                       srcImgs, srcDatas, srcW, srcH, numSrc,
                                       refMask, srcMasks,
                                       patchHalf, minimumMaskedSupportRatio);
        if (cost < bestCost)
        {
            bestCost = cost;
            bestDepth = depth;
        }
    }

    if (bestDepth > 0.0f)
    {
        const float bestInverse = 1.0f / bestDepth;
        const float refineStep = inverseStep / 6.0f;
        for (int refineIndex = -6; refineIndex <= 6; ++refineIndex)
        {
            const float inverseDepth = fmaxf(
                inverseFar,
                fminf(inverseNear,
                      bestInverse + refineStep * static_cast<float>(refineIndex)));
            const float depth = 1.0f / inverseDepth;
            const float cost = evalHypCost(col, row, depth, normal,
                                           refImg, W, H,
                                           srcImgs, srcDatas, srcW, srcH, numSrc,
                                           refMask, srcMasks,
                                           patchHalf, minimumMaskedSupportRatio);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestDepth = depth;
            }
        }
    }

    depthMap[idx] = bestDepth;
    normalMap[idx * 3] = normal[0];
    normalMap[idx * 3 + 1] = normal[1];
    normalMap[idx * 3 + 2] = normal[2];
    confMap[idx] = bestDepth > 0.0f ? 1.0f - bestCost * 0.5f : 0.0f;
}

// =============================================================================
// Sweep 核（自上而下）：一列一线程，顺序迭代各行
// 仿照 COLMAP SweepFromTopToBottom，实现无竞争的列扫描正向传播
// =============================================================================
__global__ void kernelSweepTB(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float perturbation, unsigned long long seed)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;

    curandState rs;
    curand_init(seed + (unsigned long long)col, 0, 0, &rs);

    float prevDepth = depthMap[col]; // row=0
    float prevNormal[3] = { normalMap[col*3], normalMap[col*3+1], normalMap[col*3+2] };

    for (int row = 0; row < H; ++row) 
    {
        int idx = row * W + col;

        float localNear = zNear;
        float localFar = zFar;
        pixelDepthSearchBounds(idx,
                               hintDepth,
                               hintRadius,
                               zNear,
                               zFar,
                               localNear,
                               localFar);

        float currDepth     = fmaxf(localNear, fminf(localFar, depthMap[idx]));
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        // 将上一行平面假设传播到当前行（平面-射线交点）
        float propDepth = (row > 0)
            ? propagateDepth(prevDepth, prevNormal, (float)(row-1), (float)col, (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(localNear, fminf(localFar, propDepth));

        // 随机扰动（退火）
        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(localNear, fminf(localFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        // 5 个假设（仿照 COLMAP）:
        //  0: (curr, curr)  1: (prop, prev)  2: (rand, rand)
        //  3: (curr, rand)  4: (rand, curr)
        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi)
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     refMask, srcMasks,
                                     patchHalf, minimumMaskedSupportRatio);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestIdx = hi;
            }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f; // [0,1]

        prevDepth      = dep[bestIdx];
        prevNormal[0]  = nor[bestIdx][0];
        prevNormal[1]  = nor[bestIdx][1];
        prevNormal[2]  = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自下而上）
// =============================================================================
__global__ void kernelSweepBT(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float perturbation, unsigned long long seed)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;

    curandState rs;
    curand_init(seed + (unsigned long long)col, 0, 0, &rs);

    int lastRow = H - 1;
    float prevDepth     = depthMap[lastRow * W + col];
    float prevNormal[3] = { normalMap[(lastRow*W+col)*3],
                             normalMap[(lastRow*W+col)*3+1],
                             normalMap[(lastRow*W+col)*3+2] };

    for (int row = H - 1; row >= 0; --row) 
    {
        int idx = row * W + col;

        float localNear = zNear;
        float localFar = zFar;
        pixelDepthSearchBounds(idx,
                               hintDepth,
                               hintRadius,
                               zNear,
                               zFar,
                               localNear,
                               localFar);

        float currDepth     = fmaxf(localNear, fminf(localFar, depthMap[idx]));
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (row < H - 1)
            ? propagateDepth(prevDepth, prevNormal, (float)(row+1), (float)col, (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(localNear, fminf(localFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(localNear, fminf(localFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = {

            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {            
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     refMask, srcMasks,
                                     patchHalf, minimumMaskedSupportRatio);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自左向右）：一行一线程，顺序迭代各列
// =============================================================================
__global__ void kernelSweepLR(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float perturbation, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;

    curandState rs;
    curand_init(seed + (unsigned long long)row, 0, 0, &rs);

    float prevDepth     = depthMap[row * W];
    float prevNormal[3] = { normalMap[(row*W)*3], normalMap[(row*W)*3+1], normalMap[(row*W)*3+2] };

    for (int col = 0; col < W; ++col) 
    {
        int idx = row * W + col;

        float localNear = zNear;
        float localFar = zFar;
        pixelDepthSearchBounds(idx,
                               hintDepth,
                               hintRadius,
                               zNear,
                               zFar,
                               localNear,
                               localFar);

        float currDepth     = fmaxf(localNear, fminf(localFar, depthMap[idx]));
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (col > 0)
            ? propagateDepth(prevDepth, prevNormal, (float)row, (float)(col-1), (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(localNear, fminf(localFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(localNear, fminf(localFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     refMask, srcMasks,
                                     patchHalf, minimumMaskedSupportRatio);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自右向左）
// =============================================================================
__global__ void kernelSweepRL(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float perturbation, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;

    curandState rs;
    curand_init(seed + (unsigned long long)row, 0, 0, &rs);

    int lastCol = W - 1;
    float prevDepth     = depthMap[row * W + lastCol];
    float prevNormal[3] = { normalMap[(row*W+lastCol)*3],
                             normalMap[(row*W+lastCol)*3+1],
                             normalMap[(row*W+lastCol)*3+2] };

    for (int col = W - 1; col >= 0; --col) 
    {
        int idx = row * W + col;

        float localNear = zNear;
        float localFar = zFar;
        pixelDepthSearchBounds(idx,
                               hintDepth,
                               hintRadius,
                               zNear,
                               zFar,
                               localNear,
                               localFar);

        float currDepth     = fmaxf(localNear, fminf(localFar, depthMap[idx]));
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (col < W - 1)
            ? propagateDepth(prevDepth, prevNormal, (float)row, (float)(col+1), (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(localNear, fminf(localFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(localNear, fminf(localFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     refMask, srcMasks,
                                     patchHalf, minimumMaskedSupportRatio);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

__device__ inline void considerHypothesis(
    int col, int row,
    float candidateDepth, const float candidateNormal[3],
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float *bestCost, float *bestDepth, float bestNormal[3])
{
    if (candidateDepth <= 0.f)
    {
        return;
    }

    candidateDepth = fmaxf(zNear, fminf(zFar, candidateDepth));
    if (candidateDepth == *bestDepth &&
        candidateNormal[0] == bestNormal[0] &&
        candidateNormal[1] == bestNormal[1] &&
        candidateNormal[2] == bestNormal[2])
    {
        return;
    }
    const float cost = evalHypCost(col, row, candidateDepth, candidateNormal,
                                   refImg, W, H,
                                   srcImgs, srcDatas, srcW, srcH, numSrc,
                                   refMask, srcMasks,
                                   patchHalf, minimumMaskedSupportRatio);
    if (cost < *bestCost)
    {
        *bestCost = cost;
        *bestDepth = candidateDepth;
        bestNormal[0] = candidateNormal[0];
        bestNormal[1] = candidateNormal[1];
        bestNormal[2] = candidateNormal[2];
    }
}

// =============================================================================
// 棋盘格并行传播：一像素一线程，按红黑格分两次更新，避免同色写冲突。
// 相比传统行/列 sweep，每次传播有 W*H/2 个线程并行执行，吞吐更适合现代 GPU。
// =============================================================================
__global__ void kernelCheckerboardSweep(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    const std::uint8_t *refMask, const std::uint8_t *srcMasks,
    const float *hintDepth, const float *hintRadius,
    float zNear, float zFar, int patchHalf,
    float minimumMaskedSupportRatio,
    float perturbation,
    int iteration,
    int parity)
{
    const int compactCol = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= H)
    {
        return;
    }
    const int col = compactCol * 2 + ((row + parity) & 1);
    if (col >= W)
    {
        return;
    }

    const int idx = row * W + col;
    float localNear = zNear;
    float localFar = zFar;
    pixelDepthSearchBounds(idx,
                           hintDepth,
                           hintRadius,
                           zNear,
                           zFar,
                           localNear,
                           localFar);
    float currDepth = fmaxf(localNear, fminf(localFar, depthMap[idx]));
    float currNormal[3] = {
        normalMap[idx * 3],
        normalMap[idx * 3 + 1],
        normalMap[idx * 3 + 2]
    };

    float bestDepth = currDepth;
    float bestNormal[3] = { currNormal[0], currNormal[1], currNormal[2] };
    const float currentNcc = fmaxf(0.0f, fminf(1.0f, confMap[idx]));
    float bestCost = 2.0f - 2.0f * currentNcc;

    const int dRow[4] = { -1, 1, 0, 0 };
    const int dCol[4] = { 0, 0, -1, 1 };
    for (int ni = 0; ni < 4; ++ni)
    {
        const int neighborRow = row + dRow[ni];
        const int neighborCol = col + dCol[ni];
        if (neighborRow < 0 || neighborRow >= H || neighborCol < 0 || neighborCol >= W)
        {
            continue;
        }

        const int neighborIdx = neighborRow * W + neighborCol;
        const float neighborDepth = depthMap[neighborIdx];
        if (neighborDepth <= 0.f)
        {
            continue;
        }

        float neighborNormal[3] = {
            normalMap[neighborIdx * 3],
            normalMap[neighborIdx * 3 + 1],
            normalMap[neighborIdx * 3 + 2]
        };
        const float propDepth = propagateDepth(neighborDepth,
                                               neighborNormal,
                                               static_cast<float>(neighborRow),
                                               static_cast<float>(neighborCol),
                                               static_cast<float>(row),
                                               static_cast<float>(col));
        if (propDepth >= localNear && propDepth <= localFar)
        {
            considerHypothesis(col, row, propDepth, neighborNormal,
                               refImg, W, H,
                               srcImgs, srcDatas, srcW, srcH, numSrc,
                               refMask, srcMasks,
                               localNear, localFar, patchHalf,
                               minimumMaskedSupportRatio,
                               &bestCost, &bestDepth, bestNormal);
        }
        considerHypothesis(col, row, bestDepth, neighborNormal,
                           refImg, W, H,
                           srcImgs, srcDatas, srcW, srcH, numSrc,
                           refMask, srcMasks,
                           localNear, localFar, patchHalf,
                           minimumMaskedSupportRatio,
                           &bestCost, &bestDepth, bestNormal);
    }

    const float refinedDepth = bestDepth;
    const float refinedNormal[3] = {
        bestNormal[0], bestNormal[1], bestNormal[2]};
    uint32_t rngState = alignedPatchMatchHash(
        static_cast<uint32_t>(idx)
        ^ (static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu)
        ^ (static_cast<uint32_t>(parity + 1) * 0xc2b2ae35u));
    float randNormal[3];
    if (iteration == 0)
    {
        alignedRandomFacingNormal(row, col, rngState, randNormal);
    }
    else
    {
        alignedPerturbFacingNormal(row,
                                   col,
                                   refinedNormal,
                                   fmaxf(0.02f, perturbation),
                                   rngState,
                                   randNormal);
    }
    const float randomDepth = fmaxf(
        localNear,
        fminf(localFar,
              refinedDepth + (alignedPatchMatchRandom(rngState) * 2.0f - 1.0f)
                  * (localFar - localNear) * perturbation));
    considerHypothesis(col, row, refinedDepth, randNormal,
                       refImg, W, H,
                       srcImgs, srcDatas, srcW, srcH, numSrc,
                       refMask, srcMasks,
                       localNear, localFar, patchHalf,
                       minimumMaskedSupportRatio,
                       &bestCost, &bestDepth, bestNormal);
    considerHypothesis(col, row, randomDepth, refinedNormal,
                       refImg, W, H,
                       srcImgs, srcDatas, srcW, srcH, numSrc,
                       refMask, srcMasks,
                       localNear, localFar, patchHalf,
                       minimumMaskedSupportRatio,
                       &bestCost, &bestDepth, bestNormal);
    considerHypothesis(col, row, randomDepth, randNormal,
                       refImg, W, H,
                       srcImgs, srcDatas, srcW, srcH, numSrc,
                       refMask, srcMasks,
                       localNear, localFar, patchHalf,
                       minimumMaskedSupportRatio,
                       &bestCost, &bestDepth, bestNormal);

    depthMap[idx] = bestDepth;
    normalMap[idx * 3] = bestNormal[0];
    normalMap[idx * 3 + 1] = bestNormal[1];
    normalMap[idx * 3 + 2] = bestNormal[2];
    confMap[idx] = 1.f - bestCost * 0.5f;
}

// Probe distinct depths around the converged PatchMatch hypothesis. A high
// NCC value by itself is insufficient on smooth or repetitive surfaces: when
// either neighbouring depth explains the images almost as well, the result is
// ambiguous and must enter fusion with reduced confidence.
__global__ void kernelApplyPhotometricUniqueness(
    const float *depthMap,
    const float *normalMap,
    float *confMap,
    const float *refImg,
    int W,
    int H,
    const float *const *srcImgs,
    const float *srcDatas,
    int srcW,
    int srcH,
    int numSrc,
    const std::uint8_t *refMask,
    const std::uint8_t *srcMasks,
    float zNear,
    float zFar,
    int patchHalf,
    float minimumMaskedSupportRatio,
    float relativeDepthStep,
    float minimumMargin,
    float minimumConfidenceScale)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= W || row >= H)
    {
        return;
    }

    const int idx = row * W + col;
    const float depth = depthMap[idx];
    const float bestNcc = confMap[idx];
    if (!(depth > 0.0f) || !(bestNcc > 0.0f) || !(relativeDepthStep > 0.0f))
    {
        return;
    }

    const float normal[3] = {
        normalMap[idx * 3],
        normalMap[idx * 3 + 1],
        normalMap[idx * 3 + 2]
    };
    float competingNcc = 0.0f;
    const float lowerDepth = fmaxf(zNear, depth * (1.0f - relativeDepthStep));
    const float upperDepth = fminf(zFar, depth * (1.0f + relativeDepthStep));
    const float minimumDistinctDepth = fmaxf(1e-6f, depth * relativeDepthStep * 0.25f);

    if (depth - lowerDepth >= minimumDistinctDepth)
    {
        const float cost = evalHypCost(
            col, row, lowerDepth, normal,
            refImg, W, H,
            srcImgs, srcDatas, srcW, srcH, numSrc,
            refMask, srcMasks,
            patchHalf, minimumMaskedSupportRatio);
        competingNcc = fmaxf(competingNcc, 1.0f - cost * 0.5f);
    }
    if (upperDepth - depth >= minimumDistinctDepth)
    {
        const float cost = evalHypCost(
            col, row, upperDepth, normal,
            refImg, W, H,
            srcImgs, srcDatas, srcW, srcH, numSrc,
            refMask, srcMasks,
            patchHalf, minimumMaskedSupportRatio);
        competingNcc = fmaxf(competingNcc, 1.0f - cost * 0.5f);
    }

    confMap[idx] = bestNcc * photometricUniquenessConfidenceScale(
        bestNcc,
        competingNcc,
        minimumMargin,
        minimumConfidenceScale);
}

/// 置信度阈值过滤
__global__ void kernelFinalizeDepth(
    float *depthMap, float *confMap, int W, int H, float confThresh)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;
    int idx = v * W + u;
    if (confMap[idx] < confThresh) depthMap[idx] = 0.f;
}

// =============================================================================
// GPU 实现
// =============================================================================
bool PatchMatchDepthEstimator::estimateGPU(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const FramePinholeCamera                   &refCam,
    const std::vector<FramePinholeCamera>      &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth,
    const cv::Mat                *hintRadius,
    const cv::Mat                *refValidMask,
    const std::vector<cv::Mat>   *srcValidMasks)
{
    const auto estimate_start = std::chrono::steady_clock::now();
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    const int device_index = config.cudaDeviceIndex >= 0
        ? config.cudaDeviceIndex
        : previous_device;
    if (device_index < 0 || device_index >= device_count)
    {
        if (errorMsg)
        {
            *errorMsg = "CUDA device index is outside the available device range";
        }
        return false;
    }
    CUDA_CHECK(cudaSetDevice(device_index));
    CudaDeviceRestoreGuard device_restore_guard(previous_device);
    PatchMatchGpuDeviceState &device_state = patchMatchGpuDeviceState(device_index);
    std::shared_lock<std::shared_mutex> cache_lifetime_lock(
        device_state.cacheLifetimeMutex);
    const int refW = refGray.cols, refH = refGray.rows;
    const int N    = (int)srcGrays.size();
    const int ds   = config.downsampleFactor > 0 ? config.downsampleFactor : 1;

    // ── 工作分辨率 ──────────────────────────────────────────────────
    // 参考图本身由 getOrUploadGrayImageGpu() 统一缩放/上传/缓存，避免这里先做一次重复 resize。
    cv::Mat hintScaled;
    cv::Mat hintRadiusScaled;
    const int sW = std::max(1, refW / ds);
    const int sH = std::max(1, refH / ds);
    const cv::Size scaled_size(sW, sH);
    const cv::Mat ref_mask_scaled = resizedBinaryMask(refValidMask, scaled_size);
    std::vector<cv::Mat> source_masks_scaled(static_cast<std::size_t>(N));
    bool has_source_masks = false;
    if (srcValidMasks)
    {
        for (int source_index = 0; source_index < N; ++source_index)
        {
            source_masks_scaled[static_cast<std::size_t>(source_index)] =
                resizedBinaryMask(&(*srcValidMasks)[static_cast<std::size_t>(source_index)],
                                  scaled_size);
            has_source_masks = has_source_masks ||
                !source_masks_scaled[static_cast<std::size_t>(source_index)].empty();
        }
    }

    const HostPinholeCamera refCamS = makeHostPinholeCamera(refCam, ds);

    if (hintDepth && !hintDepth->empty())
    {
        if (hintDepth->cols == sW && hintDepth->rows == sH)
        {
            hintScaled = *hintDepth;
        }
        else
        {
            cv::resize(*hintDepth, hintScaled, cv::Size(sW, sH), 0, 0, cv::INTER_NEAREST);
        }
    }
    if (!hintScaled.empty() && hintRadius && !hintRadius->empty())
    {
        if (hintRadius->cols == sW && hintRadius->rows == sH)
        {
            hintRadiusScaled = *hintRadius;
        }
        else
        {
            cv::resize(*hintRadius,
                       hintRadiusScaled,
                       cv::Size(sW, sH),
                       0,
                       0,
                       cv::INTER_NEAREST);
        }
    }

    // ── 设置常量内存 c_ref_inv_K ─────────────────────────────────
    float h_inv_K[4] = 
    {
        1.f / refCamS.focalX, -refCamS.principalX / refCamS.focalX,
        1.f / refCamS.focalY, -refCamS.principalY / refCamS.focalY
    };
    // ── 构建源图 srcData（每源 16 floats）───────────────────────
    std::vector<float>   srcDatas(N * 16, 0.f);
    int srcW = sW, srcH2 = sH;

    for (int si = 0; si < N; ++si) 
    {
        float *sd = srcDatas.data() + si * 16;
        const std::array<float, 16> source_data =
            buildPatchMatchSourceCameraData(refCam, srcCams[si], ds);
        std::copy(source_data.begin(), source_data.end(), sd);
    }

    // ── GPU 参数摘要 ──────────────────────────────────────────
    LOG_DEBUG("[MVS][PatchMatch][GPU] size=%dx%d ds=%d sources=%d requested_iterations=%d patch=%d",
              sW, sH, ds, N, config.numIterations, config.patchHalf * 2 + 1);
    for (int si = 0; si < N; ++si) 
    {
        const float *sd = srcDatas.data() + si * 16;
        const float *T_rel = sd + 13;
        float baseline = sqrtf(T_rel[0]*T_rel[0]+T_rel[1]*T_rel[1]+T_rel[2]*T_rel[2]);
        if (baseline < 1e-4f)
        {
            LOG_WARN("[MVS][PatchMatch][GPU] source=%d baseline is near zero", si);
        }
    }

    const int refPx = sW * sH;

    ImageUploadLane &image_upload_lane = imageUploadLaneForCurrentThread(device_index);
    CUDA_CHECK(image_upload_lane.beginBatch());

    // Reserve room before acquiring any cache pointers. Once this frame starts
    // collecting entries, eviction is paused until its device-to-host download
    // completes so another upload lane cannot invalidate those pointers.
    {
        std::lock_guard<std::mutex> cache_lock(g_srcImageGpuCacheMutex);
        const std::size_t conservative_batch_bytes =
            (static_cast<std::size_t>(N) + 1) *
            static_cast<std::size_t>(refPx) * sizeof(float);
        evictSrcImageGpuCacheIfNeeded(conservative_batch_bytes,
                                      device_index,
                                      image_upload_lane.stream);
    }
    // Upload cache misses before waiting for the serialized CUDA execution
    // slot. With two host frame slots, this transfer stream can feed frame N+1
    // while frame N is running PatchMatch kernels.
    float *d_ref = nullptr;
    std::vector<float *> hSrcPtrs(static_cast<std::size_t>(N), nullptr);
    std::vector<cudaEvent_t> image_ready_events(
        static_cast<std::size_t>(N) + 1, nullptr);
    bool refCacheHit = false;
    int cacheHits = 0;
    int cacheMisses = 0;
    SrcImageCachePinGuard cache_pin_guard;
    SrcImageCacheKey reference_cache_key;
    if (!getOrUploadGrayImageGpu(refGray,
                                 device_index,
                                 sW,
                                 sH,
                                 ds,
                                 image_upload_lane.stream,
                                 image_upload_lane.host(0),
                                 &d_ref,
                                 &image_ready_events[0],
                                 &reference_cache_key,
                                 &refCacheHit,
                                 errorMsg))
    {
        return false;
    }
    cache_pin_guard.add(reference_cache_key);
    for (int source_index = 0; source_index < N; ++source_index)
    {
        bool cache_hit = false;
        SrcImageCacheKey source_cache_key;
        if (!getOrUploadGrayImageGpu(
                srcGrays[static_cast<std::size_t>(source_index)],
                device_index,
                sW,
                sH,
                ds,
                image_upload_lane.stream,
                image_upload_lane.host(static_cast<std::size_t>(source_index) + 1),
                &hSrcPtrs[static_cast<std::size_t>(source_index)],
                &image_ready_events[static_cast<std::size_t>(source_index) + 1],
                &source_cache_key,
                &cache_hit,
                errorMsg))
        {
            return false;
        }
        cache_pin_guard.add(source_cache_key);
        cacheHits += cache_hit ? 1 : 0;
        cacheMisses += cache_hit ? 0 : 1;
    }

    // A single CUDA execution slot protects the shared constant-memory camera
    // parameters and persistent workspace. Two host frame workers may still
    // overlap CPU preparation/post-processing around this section.
    const auto slot_wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> gpu_execution_lock(device_state.executionMutex);
    const auto slot_acquired = std::chrono::steady_clock::now();
    PatchMatchGpuWorkspace &workspace = device_state.workspace;
    CUDA_CHECK(workspace.initialize());
    PatchMatchGpuRunGuard workspace_run_guard(workspace);
    CUDA_CHECK(workspace.srcData.reserve(static_cast<std::size_t>(N) * 16));
    CUDA_CHECK(workspace.depth.reserve(static_cast<std::size_t>(refPx)));
    CUDA_CHECK(workspace.normal.reserve(static_cast<std::size_t>(refPx) * 3));
    CUDA_CHECK(workspace.confidence.reserve(static_cast<std::size_t>(refPx)));
    CUDA_CHECK(workspace.srcPointers.reserve(static_cast<std::size_t>(N)));

    float *d_srcData = workspace.srcData.ptr;
    float *d_depth = workspace.depth.ptr;
    float *d_normal = workspace.normal.ptr;
    float *d_conf = workspace.confidence.ptr;
    float **d_srcPtrs = workspace.srcPointers.ptr;
    std::uint8_t *d_refMask = nullptr;
    std::uint8_t *d_srcMasks = nullptr;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_ref_inv_K,
                                       h_inv_K,
                                       sizeof(h_inv_K),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       workspace.computeStream));
    CUDA_CHECK(stageHostToDeviceCopy(workspace.sourceDataHost,
                                     srcDatas.data(),
                                     static_cast<std::size_t>(N) * 16 * sizeof(float),
                                     d_srcData,
                                     workspace.transferStream));
    CUDA_CHECK(cudaMemsetAsync(d_conf,
                               0,
                               static_cast<std::size_t>(refPx) * sizeof(float),
                               workspace.computeStream));

    if (!ref_mask_scaled.empty())
    {
        CUDA_CHECK(workspace.referenceMask.reserve(static_cast<std::size_t>(refPx)));
        d_refMask = workspace.referenceMask.ptr;
        CUDA_CHECK(stageHostToDeviceCopy(workspace.referenceMaskHost,
                                         ref_mask_scaled.ptr<std::uint8_t>(),
                                         static_cast<std::size_t>(refPx),
                                         d_refMask,
                                         workspace.transferStream));
    }
    if (has_source_masks)
    {
        std::vector<std::uint8_t> packed_masks(
            static_cast<std::size_t>(N) * static_cast<std::size_t>(refPx), 255);
        for (int source_index = 0; source_index < N; ++source_index)
        {
            const cv::Mat &source_mask =
                source_masks_scaled[static_cast<std::size_t>(source_index)];
            if (source_mask.empty())
            {
                continue;
            }
            std::memcpy(packed_masks.data() +
                            static_cast<std::size_t>(source_index) * static_cast<std::size_t>(refPx),
                        source_mask.ptr<std::uint8_t>(),
                        static_cast<std::size_t>(refPx));
        }
        const std::size_t source_mask_bytes = packed_masks.size() * sizeof(std::uint8_t);
        CUDA_CHECK(workspace.sourceMasks.reserve(source_mask_bytes));
        d_srcMasks = workspace.sourceMasks.ptr;
        CUDA_CHECK(stageHostToDeviceCopy(workspace.sourceMasksHost,
                                         packed_masks.data(),
                                         source_mask_bytes,
                                         d_srcMasks,
                                         workspace.transferStream));
    }

    CUDA_CHECK(stageHostToDeviceCopy(workspace.sourcePointersHost,
                                     hSrcPtrs.data(),
                                     static_cast<std::size_t>(N) * sizeof(float *),
                                     d_srcPtrs,
                                     workspace.transferStream));

    bool hasHint = !hintScaled.empty();
    float *d_hint = nullptr;
    float *d_hintRadius = nullptr;
    if (hasHint) 
    {
        cv::Mat hint_float;
        hintScaled.convertTo(hint_float, CV_32F);
        CUDA_CHECK(workspace.hint.reserve(static_cast<std::size_t>(refPx)));
        d_hint = workspace.hint.ptr;
        CUDA_CHECK(stageHostToDeviceCopy(workspace.hintHost,
                                         hint_float.ptr<float>(),
                                         static_cast<std::size_t>(refPx) * sizeof(float),
                                         d_hint,
                                         workspace.transferStream));
    }
    if (hasHint && !hintRadiusScaled.empty())
    {
        cv::Mat radius_float;
        hintRadiusScaled.convertTo(radius_float, CV_32F);
        CUDA_CHECK(workspace.hintRadius.reserve(static_cast<std::size_t>(refPx)));
        d_hintRadius = workspace.hintRadius.ptr;
        CUDA_CHECK(stageHostToDeviceCopy(workspace.hintRadiusHost,
                                         radius_float.ptr<float>(),
                                         static_cast<std::size_t>(refPx) * sizeof(float),
                                         d_hintRadius,
                                         workspace.transferStream));
    }

    CUDA_CHECK(cudaEventRecord(workspace.uploadsReady, workspace.transferStream));
    CUDA_CHECK(cudaStreamWaitEvent(workspace.computeStream, workspace.uploadsReady, 0));
    for (const cudaEvent_t ready_event : image_ready_events)
    {
        if (ready_event)
        {
            CUDA_CHECK(cudaStreamWaitEvent(workspace.computeStream,
                                           ready_event,
                                           0));
        }
    }

    // ── 初始化深度 + 法向量 ───────────────────────────────────────
    const int bW = config.cudaBlockW;
    const int bH = config.cudaBlockH;
    const int bS = config.cudaBlockSweep;
    dim3 block(bW, bH);
    dim3 grid((sW + bW - 1) / bW, (sH + bH - 1) / bH);

    const int depthSampleCount = patchMatchDepthSampleCount(config.numIterations);
    const int propagationIterations = patchMatchPropagationIterationCount(
        config.numIterations);
    const int sweepIterations = config.cudaUseParallelSweep
        ? propagationIterations
        : config.numIterations;
    kernelInitializePlanes<<<grid, block, 0, workspace.computeStream>>>(
        d_depth, d_normal, d_conf,
        d_ref, sW, sH,
        d_srcPtrs, d_srcData, srcW, srcH2, N,
        d_refMask, d_srcMasks,
        d_hint, d_hintRadius,
        zNear, zFar, config.patchHalf,
        depthSampleCount,
        config.minimumMaskedPatchSupportRatio);
    CUDA_CHECK(cudaGetLastError());

    // ── 迭代传播 + 精化 ───────────────────────────────────────
    // 默认使用棋盘格像素级并行传播；必要时可回退传统 4 向行列 sweep。
    dim3 blockSweep(bS);
    dim3 gridCols((sW + bS - 1) / bS);
    dim3 gridRows((sH + bS - 1) / bS);
    dim3 blockPixel(bW, bH);
    const int checkerboardWidth = (sW + 1) / 2;
    dim3 gridPixel((checkerboardWidth + bW - 1) / bW, (sH + bH - 1) / bH);

    float perturbation = 0.35f;
    for (int iter = 0; iter < sweepIterations; ++iter)
    {
        if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMsg) *errorMsg = "PatchMatch cancelled";
            return false;
        }

        if (config.cudaUseParallelSweep)
        {
            kernelCheckerboardSweep<<<gridPixel, blockPixel, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation, iter,
                0);
            CUDA_CHECK(cudaGetLastError());

            kernelCheckerboardSweep<<<gridPixel, blockPixel, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation, iter,
                1);
            CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            // 自上而下
            kernelSweepTB<<<gridCols, blockSweep, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation, static_cast<unsigned long long>(iter + 1) * 999983ULL);
            CUDA_CHECK(cudaGetLastError());

            // 自下而上
            kernelSweepBT<<<gridCols, blockSweep, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation,
                static_cast<unsigned long long>(iter + 1) * 999983ULL + 111111ULL);
            CUDA_CHECK(cudaGetLastError());

            // 自左向右
            kernelSweepLR<<<gridRows, blockSweep, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation,
                static_cast<unsigned long long>(iter + 1) * 999983ULL + 222222ULL);
            CUDA_CHECK(cudaGetLastError());

            // 自右向左
            kernelSweepRL<<<gridRows, blockSweep, 0, workspace.computeStream>>>(
                d_depth, d_normal, d_conf,
                d_ref, sW, sH,
                d_srcPtrs, d_srcData, srcW, srcH2, N,
                d_refMask, d_srcMasks,
                d_hint, d_hintRadius,
                zNear, zFar, config.patchHalf,
                config.minimumMaskedPatchSupportRatio,
                perturbation,
                static_cast<unsigned long long>(iter + 1) * 999983ULL + 333333ULL);
            CUDA_CHECK(cudaGetLastError());
        }

        perturbation = fmaxf(perturbation * 0.5f, 0.02f);
        constexpr int kCancellationCheckpointInterval = 4;
        const bool cancellation_checkpoint = config.cancelFlag &&
            (((iter + 1) % kCancellationCheckpointInterval) == 0 ||
             iter + 1 == sweepIterations);
        if (cancellation_checkpoint)
        {
            CUDA_CHECK(cudaEventRecord(workspace.cancellationCheckpoint,
                                       workspace.computeStream));
            CUDA_CHECK(cudaEventSynchronize(workspace.cancellationCheckpoint));
            if (config.cancelFlag->load(std::memory_order_relaxed))
            {
                if (errorMsg) *errorMsg = "PatchMatch cancelled";
                return false;
            }
        }
    }

    if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
    {
        if (errorMsg) *errorMsg = "PatchMatch cancelled";
        return false;
    }

    // ── 深度唯一性 + 置信度过滤 ──────────────────────────────────
    if (config.enablePhotometricUniqueness)
    {
        kernelApplyPhotometricUniqueness<<<grid, block, 0, workspace.computeStream>>>(
            d_depth,
            d_normal,
            d_conf,
            d_ref,
            sW,
            sH,
            d_srcPtrs,
            d_srcData,
            srcW,
            srcH2,
            N,
            d_refMask,
            d_srcMasks,
            zNear,
            zFar,
            config.patchHalf,
            config.minimumMaskedPatchSupportRatio,
            config.photometricUniquenessRelativeDepthStep,
            config.photometricUniquenessMinimumMargin,
            config.photometricUniquenessMinimumConfidenceScale);
        CUDA_CHECK(cudaGetLastError());
    }
    kernelFinalizeDepth<<<grid, block, 0, workspace.computeStream>>>(
        d_depth, d_conf, sW, sH, config.confidenceThresh);
    CUDA_CHECK(cudaGetLastError());

    // ── 独立传输流异步拷回固定页内存 ─────────────────────────────
    CUDA_CHECK(workspace.depthHost.reserve(static_cast<std::size_t>(refPx)));
    CUDA_CHECK(workspace.confidenceHost.reserve(static_cast<std::size_t>(refPx)));
    CUDA_CHECK(cudaEventRecord(workspace.computeReady, workspace.computeStream));
    CUDA_CHECK(cudaStreamWaitEvent(workspace.transferStream,
                                   workspace.computeReady,
                                   0));
    CUDA_CHECK(cudaMemcpyAsync(workspace.depthHost.ptr,
                               d_depth,
                               static_cast<std::size_t>(refPx) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               workspace.transferStream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.confidenceHost.ptr,
                               d_conf,
                               static_cast<std::size_t>(refPx) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               workspace.transferStream));
    CUDA_CHECK(cudaEventRecord(workspace.downloadsReady,
                               workspace.transferStream));
    CUDA_CHECK(cudaEventSynchronize(workspace.downloadsReady));

    cv::Mat depthS(sH, sW, CV_32F), confS(sH, sW, CV_32F);
    std::memcpy(depthS.ptr<float>(),
                workspace.depthHost.ptr,
                static_cast<std::size_t>(refPx) * sizeof(float));
    std::memcpy(confS.ptr<float>(),
                workspace.confidenceHost.ptr,
                static_cast<std::size_t>(refPx) * sizeof(float));
    workspace_run_guard.markCompleted();
    const auto gpu_slot_finished = std::chrono::steady_clock::now();
    gpu_execution_lock.unlock();
    if (!ref_mask_scaled.empty())
    {
        const cv::Mat invalid_reference = ref_mask_scaled == 0;
        depthS.setTo(cv::Scalar(0.0f), invalid_reference);
        confS.setTo(cv::Scalar(0.0f), invalid_reference);
    }

    // ── 后处理结果统计诊断 ────────────────────────────────────────
    {
        cv::Mat validMask = (depthS > 0);
        int validCnt = cv::countNonZero(validMask);
        int totalPx  = sW * sH;
        double dMin=0, dMax=0, dMean=0;
        if (validCnt > 0) 
        {
            cv::minMaxLoc(depthS, &dMin, &dMax, nullptr, nullptr, validMask);
            dMean = cv::mean(depthS, validMask)[0];
        }
        double cMean = validCnt > 0 ? cv::mean(confS, validMask)[0] : 0;
        LOG_DEBUG("[MVS][PatchMatch][GPU] result valid=%d/%d (%.1f%%) depth=[%.4f,%.4f] mean=%.4f confidence=%.4f",
                  validCnt, totalPx, 100.f * validCnt / totalPx,
                  static_cast<float>(dMin), static_cast<float>(dMax),
                  static_cast<float>(dMean), static_cast<float>(cMean));

        if (validCnt == 0) 
        {
            LOG_WARN("[MVS][PatchMatch][GPU] result contains zero valid depth pixels");
        }
    }


    // ── 后处理（在降采样分辨率上做，再 upscale → 更快 ×4+）───────
    postprocessPatchMatchDepth(depthS, config);

    // ── 上采样到原始分辨率 ───────────────────────────────────────
    cv::Mat depthFull = depthS, confFull = confS;
    if (ds > 1 && !config.returnNativeResolution)
    {
        // 深度图和置信度图必须使用最近邻插值上采样，避免在有效/无效像素
        // 边界产生错误的中间值（双线性插值会把 0 和 valid 混合出
        // 远低于 zNear 的伪深度值，导致大量点云离群点）
        cv::resize(depthS, depthFull, cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
        cv::resize(confS,  confFull,  cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
    }

    depthOut = depthFull;
    if (confOut) *confOut = confFull;

    int valid = cv::countNonZero(depthFull > 0);
    float mean = (valid>0) ? (float)cv::mean(depthFull, depthFull>0)[0] : 0.f;
    const int outputPx = depthFull.rows * depthFull.cols;
    LOG_DEBUG(
        "[MVS][PatchMatch][GPU] scaled=%dx%d output=%dx%d range=[%.2f,%.2f] valid=%d/%d mean=%.2f "
        "confidence>=%.2f hint=%d depth_samples=%d propagation_passes=%d parallel_sweep=%d",
        sW, sH, depthFull.cols, depthFull.rows, zNear, zFar, valid, outputPx, mean,
        config.confidenceThresh, hasHint ? 1 : 0, depthSampleCount,
        sweepIterations,
        config.cudaUseParallelSweep ? 1 : 0);
    LOG_DEBUG(
        "[MVS][PatchMatch][GPU] gray_cache ref_hit=%d source_hit=%d source_miss=%d usage=%.1f/%.1f MiB",
        refCacheHit ? 1 : 0,
        cacheHits,
        cacheMisses,
        static_cast<float>(getGrayImageGpuCacheBytes(device_index)) / (1024.0f * 1024.0f),
        static_cast<float>(getGrayImageGpuCacheLimitBytes()) / (1024.0f * 1024.0f));
    const auto estimate_finished = std::chrono::steady_clock::now();
    const double host_prepare_ms = std::chrono::duration<double, std::milli>(
        slot_wait_start - estimate_start).count();
    const double slot_wait_ms = std::chrono::duration<double, std::milli>(
        slot_acquired - slot_wait_start).count();
    const double gpu_slot_ms = std::chrono::duration<double, std::milli>(
        gpu_slot_finished - slot_acquired).count();
    const double postprocess_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - gpu_slot_finished).count();
    const double total_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - estimate_start).count();
    LOG_INFO("[MVS][PatchMatch][CUDA] device=%d size=%dx%d sources=%d "
             "prepare=%.1f ms wait=%.1f ms gpu_slot=%.1f ms post=%.1f ms total=%.1f ms",
             device_index,
             sW,
             sH,
             N,
             host_prepare_ms,
             slot_wait_ms,
             gpu_slot_ms,
             postprocess_ms,
             total_ms);
    return true;
}

// =============================================================================
// CPU fallback（平面扫描）
// =============================================================================

// =============================================================================
// 公有接口
// =============================================================================
bool PatchMatchDepthEstimator::isCudaAvailable()
{
    return cudaDeviceCount() > 0;
}

int PatchMatchDepthEstimator::cudaDeviceCount()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

std::string PatchMatchDepthEstimator::cudaDeviceName(int deviceIndex)
{
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, deviceIndex) == cudaSuccess
        ? std::string(properties.name)
        : std::string();
}

std::string PatchMatchDepthEstimator::cudaDeviceIdentity(int deviceIndex)
{
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, deviceIndex) != cudaSuccess)
    {
        return {};
    }
    char identity[64]{};
    std::snprintf(identity,
                  sizeof(identity),
                  "pci:%04x:%02x:%02x",
                  properties.pciDomainID,
                  properties.pciBusID,
                  properties.pciDeviceID);
    return identity;
}

bool PatchMatchDepthEstimator::reserveGpuWorkspace(
    std::size_t referencePixelCount,
    int sourceCount,
    bool reserveReferenceMask,
    bool reserveSourceMasks,
    std::string *errorMsg,
    int deviceIndex)
{
    if (referencePixelCount == 0 || sourceCount <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "CUDA workspace dimensions are invalid";
        }
        return false;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    const int device_index = deviceIndex >= 0 ? deviceIndex : previous_device;
    if (device_index < 0 || device_index >= device_count)
    {
        if (errorMsg)
        {
            *errorMsg = "CUDA workspace device index is outside the available device range";
        }
        return false;
    }
    CUDA_CHECK(cudaSetDevice(device_index));
    CudaDeviceRestoreGuard device_restore_guard(previous_device);
    PatchMatchGpuDeviceState &device_state = patchMatchGpuDeviceState(device_index);
    std::shared_lock<std::shared_mutex> lifetime_lock(
        device_state.cacheLifetimeMutex);
    std::unique_lock<std::mutex> execution_lock(
        device_state.executionMutex, std::try_to_lock);
    if (!execution_lock.owns_lock())
    {
        // Another frame is already using a workspace at least as large as its
        // own finest level. Do not block this host slot: it can prepare and
        // upload images now; estimateGPU() will grow the workspace later only
        // if this frame is actually larger.
        return true;
    }
    PatchMatchGpuWorkspace &workspace = device_state.workspace;
    CUDA_CHECK(workspace.initialize());
    CUDA_CHECK(workspace.srcData.reserve(
        static_cast<std::size_t>(sourceCount) * 16));
    CUDA_CHECK(workspace.srcPointers.reserve(
        static_cast<std::size_t>(sourceCount)));
    CUDA_CHECK(workspace.depth.reserve(referencePixelCount));
    CUDA_CHECK(workspace.normal.reserve(referencePixelCount * 3));
    CUDA_CHECK(workspace.confidence.reserve(referencePixelCount));
    CUDA_CHECK(workspace.hint.reserve(referencePixelCount));
    CUDA_CHECK(workspace.hintRadius.reserve(referencePixelCount));
    CUDA_CHECK(workspace.depthHost.reserve(referencePixelCount));
    CUDA_CHECK(workspace.confidenceHost.reserve(referencePixelCount));
    if (reserveReferenceMask)
    {
        CUDA_CHECK(workspace.referenceMask.reserve(referencePixelCount));
    }
    if (reserveSourceMasks)
    {
        CUDA_CHECK(workspace.sourceMasks.reserve(
            referencePixelCount * static_cast<std::size_t>(sourceCount)));
    }
    return true;
}

void PatchMatchDepthEstimator::cleanupGpuImageCache()
{
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    std::unique_lock<std::mutex> registry_lock(g_patchMatchGpuDeviceRegistryMutex);
    for (const auto &item : g_patchMatchGpuDeviceStates)
    {
        const int device_index = item.first;
        PatchMatchGpuDeviceState *state = item.second.get();
        cudaSetDevice(device_index);
        std::unique_lock<std::shared_mutex> lifetime_lock(state->cacheLifetimeMutex);
        std::lock_guard<std::mutex> execution_lock(state->executionMutex);
        std::lock_guard<std::mutex> cache_lock(g_srcImageGpuCacheMutex);
        for (auto it = g_srcImageGpuCache.begin(); it != g_srcImageGpuCache.end();)
        {
            if (it->first.deviceIndex != device_index)
            {
                ++it;
                continue;
            }
            if (it->second.devicePtr)
            {
                if (it->second.readyEvent)
                {
                    cudaEventSynchronize(it->second.readyEvent);
                }
                cudaFree(it->second.devicePtr);
            }
            if (it->second.readyEvent)
            {
                cudaEventDestroy(it->second.readyEvent);
            }
            g_srcImageGpuCacheBytes -= it->second.bytes;
            it = g_srcImageGpuCache.erase(it);
        }
        state->workspace.reset();
    }
    if (previous_device >= 0)
    {
        cudaSetDevice(previous_device);
    }
}


} // namespace mvs
} // namespace xjw
