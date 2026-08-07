#define CL_TARGET_OPENCL_VERSION 120

#include "PatchMatchCUDA.h"
#include "PatchMatchOpenCLKernels.h"

#include "GpuDeviceLease.h"

#include "Logger.h"

#include <CL/cl.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace xjw
{
namespace mvs
{

namespace
{

struct EnumeratedOpenClDevice
{
    OpenClDeviceInfo info;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    bool hostUnifiedMemory = false;
};

struct OpenClPciBusInfo
{
    cl_uint domain = 0;
    cl_uint bus = 0;
    cl_uint device = 0;
    cl_uint function = 0;
};

constexpr cl_device_info kOpenClDevicePciBusInfo = 0x410F;

std::string openClString(cl_device_id device, cl_device_info parameter)
{
    std::size_t size = 0;
    if (clGetDeviceInfo(device, parameter, 0, nullptr, &size) != CL_SUCCESS || size == 0)
    {
        return {};
    }
    std::string value(size, '\0');
    if (clGetDeviceInfo(device, parameter, size, value.data(), nullptr) != CL_SUCCESS)
    {
        return {};
    }
    while (!value.empty() && value.back() == '\0')
    {
        value.pop_back();
    }
    return value;
}

std::vector<EnumeratedOpenClDevice> enumerateOpenClGpuDevicesUncached()
{
    cl_uint platform_count = 0;
    const cl_int platform_error = clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_error != CL_SUCCESS || platform_count == 0)
    {
        return {};
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS)
    {
        return {};
    }

    std::vector<EnumeratedOpenClDevice> result;
    for (cl_platform_id platform : platforms)
    {
        cl_uint device_count = 0;
        const cl_int count_error = clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (count_error == CL_DEVICE_NOT_FOUND || device_count == 0)
        {
            continue;
        }
        if (count_error != CL_SUCCESS)
        {
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platform,
                           CL_DEVICE_TYPE_GPU,
                           device_count,
                           devices.data(),
                           nullptr) != CL_SUCCESS)
        {
            continue;
        }
        for (cl_device_id device : devices)
        {
            EnumeratedOpenClDevice entry;
            entry.info.index = static_cast<int>(result.size());
            entry.info.name = openClString(device, CL_DEVICE_NAME);
            entry.info.vendor = openClString(device, CL_DEVICE_VENDOR);
            entry.info.version = openClString(device, CL_DEVICE_VERSION);
            entry.info.isGpu = true;
            clGetDeviceInfo(device,
                            CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(entry.info.globalMemoryBytes),
                            &entry.info.globalMemoryBytes,
                            nullptr);
            clGetDeviceInfo(device,
                            CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(entry.info.computeUnits),
                            &entry.info.computeUnits,
                            nullptr);
            cl_bool host_unified_memory = CL_FALSE;
            clGetDeviceInfo(device,
                            CL_DEVICE_HOST_UNIFIED_MEMORY,
                            sizeof(host_unified_memory),
                            &host_unified_memory,
                            nullptr);
            entry.hostUnifiedMemory = host_unified_memory == CL_TRUE;
            OpenClPciBusInfo pci_info;
            if (clGetDeviceInfo(device,
                                kOpenClDevicePciBusInfo,
                                sizeof(pci_info),
                                &pci_info,
                                nullptr) == CL_SUCCESS)
            {
                char identity[64]{};
                std::snprintf(identity,
                              sizeof(identity),
                              "pci:%04x:%02x:%02x",
                              pci_info.domain,
                              pci_info.bus,
                              pci_info.device);
                entry.info.physicalDeviceIdentity = identity;
            }
            else
            {
                entry.info.physicalDeviceIdentity = fallbackGpuPhysicalIdentity(
                    entry.info.vendor, entry.info.name, entry.info.index);
            }
            entry.platform = platform;
            entry.device = device;
            result.push_back(std::move(entry));
        }
    }
    return result;
}

const std::vector<EnumeratedOpenClDevice> &enumerateOpenClGpuDevices()
{
    static const std::vector<EnumeratedOpenClDevice> devices =
        enumerateOpenClGpuDevicesUncached();
    return devices;
}

struct OpenClRuntime
{
    struct BufferSlot
    {
        cl_mem memory = nullptr;
        std::size_t capacity = 0;
        cl_mem_flags flags = 0;
    };

    struct ExecutionLane
    {
        cl_command_queue queue = nullptr;
        cl_kernel kernel = nullptr;
        std::array<BufferSlot, 9> buffers;
        bool busy = false;
    };

    struct CachedScaledImage
    {
        const std::uint8_t *sourceData = nullptr;
        std::size_t sourceStep = 0;
        int sourceRows = 0;
        int sourceColumns = 0;
        int sourceType = 0;
        cv::Size workingSize;
        cv::Mat sourceOwner;
        cv::Mat pixels;
        std::uint64_t lastUse = 0;
    };

    static constexpr std::size_t kExecutionLaneCount = 1;

    cl_context context = nullptr;
    cl_program program = nullptr;
    bool hostUnifiedMemory = false;
    std::array<ExecutionLane, kExecutionLaneCount> lanes;
    std::mutex laneMutex;
    std::condition_variable laneAvailable;
    std::mutex imageCacheMutex;
    std::vector<std::shared_ptr<CachedScaledImage>> imageCache;
    std::size_t imageCacheBytes = 0;
    std::size_t imageCacheLimitBytes = 512ull * 1024ull * 1024ull;
    std::uint64_t imageCacheClock = 0;

    ~OpenClRuntime()
    {
        for (ExecutionLane &lane : lanes)
        {
            for (BufferSlot &buffer : lane.buffers)
            {
                if (buffer.memory)
                {
                    clReleaseMemObject(buffer.memory);
                }
            }
            if (lane.kernel)
            {
                clReleaseKernel(lane.kernel);
            }
            if (lane.queue)
            {
                clReleaseCommandQueue(lane.queue);
            }
        }
        if (program)
        {
            clReleaseProgram(program);
        }
        if (context)
        {
            clReleaseContext(context);
        }
    }
};

std::mutex g_openClRuntimeRegistryMutex;
std::unordered_map<int, std::shared_ptr<OpenClRuntime>> g_openClRuntimes;

class OpenClExecutionLaneLease
{
public:
    OpenClExecutionLaneLease() = default;
    OpenClExecutionLaneLease(std::shared_ptr<OpenClRuntime> runtime, std::size_t laneIndex)
        : _runtime(std::move(runtime)), _laneIndex(laneIndex)
    {
    }
    OpenClExecutionLaneLease(const OpenClExecutionLaneLease &) = delete;
    OpenClExecutionLaneLease &operator=(const OpenClExecutionLaneLease &) = delete;
    OpenClExecutionLaneLease(OpenClExecutionLaneLease &&other) noexcept
        : _runtime(std::move(other._runtime)), _laneIndex(other._laneIndex)
    {
    }
    OpenClExecutionLaneLease &operator=(OpenClExecutionLaneLease &&other) noexcept
    {
        if (this != &other)
        {
            release();
            _runtime = std::move(other._runtime);
            _laneIndex = other._laneIndex;
        }
        return *this;
    }

    ~OpenClExecutionLaneLease()
    {
        release();
    }

    OpenClRuntime::ExecutionLane &lane()
    {
        return _runtime->lanes[_laneIndex];
    }

    std::size_t laneIndex() const
    {
        return _laneIndex;
    }

    void release()
    {
        if (!_runtime)
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(_runtime->laneMutex);
            _runtime->lanes[_laneIndex].busy = false;
        }
        _runtime->laneAvailable.notify_one();
        _runtime.reset();
    }

private:
    std::shared_ptr<OpenClRuntime> _runtime;
    std::size_t _laneIndex = 0;
};

std::optional<OpenClExecutionLaneLease> acquireOpenClExecutionLane(
    const std::shared_ptr<OpenClRuntime> &runtime,
    const std::atomic<bool> *cancelFlag)
{
    std::unique_lock<std::mutex> lock(runtime->laneMutex);
    runtime->laneAvailable.wait(lock, [&]()
    {
        return (cancelFlag && cancelFlag->load(std::memory_order_relaxed)) ||
            std::any_of(runtime->lanes.begin(), runtime->lanes.end(),
                        [](const OpenClRuntime::ExecutionLane &lane)
                        {
                            return !lane.busy;
                        });
    });
    if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
    {
        return std::nullopt;
    }
    for (std::size_t index = 0; index < runtime->lanes.size(); ++index)
    {
        if (!runtime->lanes[index].busy)
        {
            runtime->lanes[index].busy = true;
            return OpenClExecutionLaneLease(runtime, index);
        }
    }
    return std::nullopt;
}

std::string openClError(const char *operation, cl_int error)
{
    std::ostringstream stream;
    stream << operation << " failed with OpenCL error " << error;
    return stream.str();
}

bool reserveOpenClBuffer(cl_context context,
                         OpenClRuntime::BufferSlot &slot,
                         cl_mem_flags flags,
                         std::size_t bytes,
                         std::string *errorMsg)
{
    if (slot.memory && slot.flags == flags && slot.capacity >= bytes)
    {
        return true;
    }

    if (slot.memory)
    {
        clReleaseMemObject(slot.memory);
        slot = {};
    }

    cl_int error = CL_SUCCESS;
    slot.memory = clCreateBuffer(context, flags, bytes, nullptr, &error);
    if (error != CL_SUCCESS || !slot.memory)
    {
        slot = {};
        if (errorMsg)
        {
            *errorMsg = openClError("clCreateBuffer", error);
        }
        return false;
    }
    slot.capacity = bytes;
    slot.flags = flags;
    return true;
}

class OpenClEvent
{
public:
    OpenClEvent() = default;
    OpenClEvent(const OpenClEvent &) = delete;
    OpenClEvent &operator=(const OpenClEvent &) = delete;

    ~OpenClEvent()
    {
        if (_event)
        {
            clReleaseEvent(_event);
        }
    }

    cl_event *output()
    {
        return &_event;
    }

    cl_event get() const
    {
        return _event;
    }

private:
    cl_event _event = nullptr;
};

class OpenClQueueDrain
{
public:
    explicit OpenClQueueDrain(cl_command_queue queue) : _queue(queue)
    {
    }

    OpenClQueueDrain(const OpenClQueueDrain &) = delete;
    OpenClQueueDrain &operator=(const OpenClQueueDrain &) = delete;

    ~OpenClQueueDrain()
    {
        if (_pending)
        {
            clFinish(_queue);
        }
    }

    void markPending()
    {
        _pending = true;
    }

    void markComplete()
    {
        _pending = false;
    }

private:
    cl_command_queue _queue = nullptr;
    bool _pending = false;
};

bool eventProfileRange(cl_event first,
                       cl_profiling_info firstParameter,
                       cl_event last,
                       cl_profiling_info lastParameter,
                       cl_ulong &startNanoseconds,
                       cl_ulong &endNanoseconds)
{
    return first && last &&
        clGetEventProfilingInfo(first,
                                firstParameter,
                                sizeof(startNanoseconds),
                                &startNanoseconds,
                                nullptr) == CL_SUCCESS &&
        clGetEventProfilingInfo(last,
                                lastParameter,
                                sizeof(endNanoseconds),
                                &endNanoseconds,
                                nullptr) == CL_SUCCESS &&
        endNanoseconds >= startNanoseconds;
}

std::shared_ptr<OpenClRuntime> openClRuntimeForDevice(
    const EnumeratedOpenClDevice &device,
    std::string *errorMsg)
{
    std::lock_guard<std::mutex> registry_lock(g_openClRuntimeRegistryMutex);
    const auto existing = g_openClRuntimes.find(device.info.index);
    if (existing != g_openClRuntimes.end())
    {
        return existing->second;
    }

    auto runtime = std::make_shared<OpenClRuntime>();
    runtime->hostUnifiedMemory = device.hostUnifiedMemory;
    runtime->imageCacheLimitBytes = std::clamp<std::size_t>(
        static_cast<std::size_t>(device.info.globalMemoryBytes / 8),
        128ull * 1024ull * 1024ull,
        1024ull * 1024ull * 1024ull);
    cl_int error = CL_SUCCESS;
    const cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(device.platform),
        0};
    runtime->context = clCreateContext(
        properties, 1, &device.device, nullptr, nullptr, &error);
    if (error != CL_SUCCESS || !runtime->context)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clCreateContext", error);
        }
        return nullptr;
    }
    std::array<const char *, 2> sources = {
        detail::kPatchMatchOpenClSourcePrefix,
        detail::kPatchMatchOpenClSourceMain};
    const std::array<std::size_t, 2> source_lengths = {
        std::strlen(sources[0]),
        std::strlen(sources[1])};
    runtime->program = clCreateProgramWithSource(
        runtime->context,
        static_cast<cl_uint>(sources.size()),
        sources.data(),
        source_lengths.data(),
        &error);
    if (error != CL_SUCCESS || !runtime->program)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clCreateProgramWithSource", error);
        }
        return nullptr;
    }
    error = clBuildProgram(runtime->program,
                           1,
                           &device.device,
                           "-cl-mad-enable -cl-fast-relaxed-math",
                           nullptr,
                           nullptr);
    if (error != CL_SUCCESS)
    {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(
            runtime->program, device.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string build_log(log_size, '\0');
        if (log_size > 0)
        {
            clGetProgramBuildInfo(runtime->program,
                                  device.device,
                                  CL_PROGRAM_BUILD_LOG,
                                  log_size,
                                  build_log.data(),
                                  nullptr);
        }
        if (errorMsg)
        {
            *errorMsg = openClError("clBuildProgram", error) + ": " + build_log;
        }
        return nullptr;
    }
    for (OpenClRuntime::ExecutionLane &lane : runtime->lanes)
    {
        lane.queue = clCreateCommandQueue(
            runtime->context, device.device, CL_QUEUE_PROFILING_ENABLE, &error);
        if (error != CL_SUCCESS || !lane.queue)
        {
            if (errorMsg)
            {
                *errorMsg = openClError("clCreateCommandQueue", error);
            }
            return nullptr;
        }
        lane.kernel = clCreateKernel(runtime->program, "estimate_depth", &error);
        if (error != CL_SUCCESS || !lane.kernel)
        {
            if (errorMsg)
            {
                *errorMsg = openClError("clCreateKernel", error);
            }
            return nullptr;
        }
    }

    g_openClRuntimes.emplace(device.info.index, runtime);
    return runtime;
}

cv::Mat scaledGrayFloat(const cv::Mat &image, const cv::Size &size)
{
    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat scaled;
    if (gray.size() == size)
    {
        scaled = gray;
    }
    else
    {
        cv::resize(gray, scaled, size, 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat result;
    scaled.convertTo(result, CV_32F, 1.0 / 255.0);
    return result.isContinuous() ? result : result.clone();
}

std::shared_ptr<OpenClRuntime::CachedScaledImage> cachedScaledGrayFloat(
    const std::shared_ptr<OpenClRuntime> &runtime,
    const cv::Mat &image,
    const cv::Size &size,
    bool *cacheHit)
{
    if (cacheHit)
    {
        *cacheHit = false;
    }
    const auto matches = [&image, &size](const OpenClRuntime::CachedScaledImage &entry)
    {
        return entry.sourceData == image.data &&
            entry.sourceStep == image.step[0] &&
            entry.sourceRows == image.rows &&
            entry.sourceColumns == image.cols &&
            entry.sourceType == image.type() &&
            entry.workingSize == size;
    };
    {
        std::lock_guard<std::mutex> lock(runtime->imageCacheMutex);
        const auto found = std::find_if(
            runtime->imageCache.begin(), runtime->imageCache.end(),
            [&matches](const std::shared_ptr<OpenClRuntime::CachedScaledImage> &entry)
            {
                return matches(*entry);
            });
        if (found != runtime->imageCache.end())
        {
            (*found)->lastUse = ++runtime->imageCacheClock;
            if (cacheHit)
            {
                *cacheHit = true;
            }
            return *found;
        }
    }

    auto candidate = std::make_shared<OpenClRuntime::CachedScaledImage>();
    candidate->sourceData = image.data;
    candidate->sourceStep = image.step[0];
    candidate->sourceRows = image.rows;
    candidate->sourceColumns = image.cols;
    candidate->sourceType = image.type();
    candidate->workingSize = size;
    candidate->sourceOwner = image;
    candidate->pixels = scaledGrayFloat(image, size);
    const std::size_t candidate_bytes = candidate->pixels.total() * candidate->pixels.elemSize();

    std::lock_guard<std::mutex> lock(runtime->imageCacheMutex);
    const auto existing = std::find_if(
        runtime->imageCache.begin(), runtime->imageCache.end(),
        [&matches](const std::shared_ptr<OpenClRuntime::CachedScaledImage> &entry)
        {
            return matches(*entry);
        });
    if (existing != runtime->imageCache.end())
    {
        (*existing)->lastUse = ++runtime->imageCacheClock;
        if (cacheHit)
        {
            *cacheHit = true;
        }
        return *existing;
    }

    while (runtime->imageCacheBytes + candidate_bytes > runtime->imageCacheLimitBytes)
    {
        const auto oldest = std::min_element(
            runtime->imageCache.begin(), runtime->imageCache.end(),
            [](const std::shared_ptr<OpenClRuntime::CachedScaledImage> &left,
               const std::shared_ptr<OpenClRuntime::CachedScaledImage> &right)
            {
                const std::uint64_t left_use = left.use_count() == 1
                    ? left->lastUse : std::numeric_limits<std::uint64_t>::max();
                const std::uint64_t right_use = right.use_count() == 1
                    ? right->lastUse : std::numeric_limits<std::uint64_t>::max();
                return left_use < right_use;
            });
        if (oldest == runtime->imageCache.end() || oldest->use_count() != 1)
        {
            break;
        }
        runtime->imageCacheBytes -= (*oldest)->pixels.total() * (*oldest)->pixels.elemSize();
        runtime->imageCache.erase(oldest);
    }
    candidate->lastUse = ++runtime->imageCacheClock;
    runtime->imageCacheBytes += candidate_bytes;
    runtime->imageCache.push_back(candidate);
    return candidate;
}

cv::Mat scaledBinaryMask(const cv::Mat *mask, const cv::Size &size)
{
    if (!mask || mask->empty())
    {
        return {};
    }
    cv::Mat binary;
    cv::compare(*mask, 0, binary, cv::CMP_GT);
    if (binary.size() != size)
    {
        cv::resize(binary, binary, size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    return binary.isContinuous() ? binary : binary.clone();
}

std::vector<float> sourceCameraData(const Camera &reference,
                                    const std::vector<Camera> &sources,
                                    int downsampleFactor)
{
    const float scale = 1.0f / static_cast<float>(std::max(1, downsampleFactor));
    const Camera::Intrinsics reference_intrinsics = reference.intrinsics();
    const std::array<double, 9> reference_rotation = reference.worldToCameraRotation();
    const std::array<double, 3> reference_translation = reference.worldToCameraTranslation();
    std::vector<float> result(sources.size() * 16, 0.0f);
    for (std::size_t source_index = 0; source_index < sources.size(); ++source_index)
    {
        const Camera::Intrinsics intrinsics = sources[source_index].intrinsics();
        const std::array<double, 9> rotation = sources[source_index].worldToCameraRotation();
        const std::array<double, 3> translation = sources[source_index].worldToCameraTranslation();
        float *data = result.data() + source_index * 16;
        data[0] = static_cast<float>(intrinsics.focalX) * scale;
        data[1] = static_cast<float>(intrinsics.principalX) * scale;
        data[2] = static_cast<float>(intrinsics.focalY) * scale;
        data[3] = static_cast<float>(intrinsics.principalY) * scale;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                double value = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    value += rotation[row * 3 + k] * reference_rotation[column * 3 + k];
                }
                data[4 + row * 3 + column] = static_cast<float>(value);
            }
        }
        for (int row = 0; row < 3; ++row)
        {
            double value = translation[row];
            for (int column = 0; column < 3; ++column)
            {
                value -= static_cast<double>(data[4 + row * 3 + column])
                    * reference_translation[column];
            }
            data[13 + row] = static_cast<float>(value);
        }
    }
    return result;
}

template <typename T>
bool setKernelArgument(cl_kernel kernel,
                       cl_uint index,
                       const T &value,
                       std::string *errorMsg)
{
    const cl_int error = clSetKernelArg(kernel, index, sizeof(T), &value);
    if (error == CL_SUCCESS)
    {
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = openClError("clSetKernelArg", error);
    }
    return false;
}

} // namespace

bool PatchMatchDepthEstimator::isOpenClAvailable()
{
    return !enumerateOpenClGpuDevices().empty();
}

std::vector<OpenClDeviceInfo> PatchMatchDepthEstimator::openClDevices()
{
    const std::vector<EnumeratedOpenClDevice> &devices = enumerateOpenClGpuDevices();
    std::vector<OpenClDeviceInfo> result;
    result.reserve(devices.size());
    for (const EnumeratedOpenClDevice &device : devices)
    {
        result.push_back(device.info);
    }
    return result;
}

void PatchMatchDepthEstimator::cleanupOpenClResources()
{
    std::lock_guard<std::mutex> lock(g_openClRuntimeRegistryMutex);
    g_openClRuntimes.clear();
}

bool PatchMatchDepthEstimator::estimateOpenCL(
    const cv::Mat &refGray,
    const std::vector<cv::Mat> &srcGrays,
    const Camera &refCam,
    const std::vector<Camera> &srcCams,
    float zNear,
    float zFar,
    const PatchMatchConfig &config,
    cv::Mat &depthOut,
    cv::Mat *confOut,
    std::string *errorMsg,
    const cv::Mat *hintDepth,
    const cv::Mat *hintRadius,
    const cv::Mat *refValidMask,
    const std::vector<cv::Mat> *srcValidMasks)
{
    const auto estimate_start = std::chrono::steady_clock::now();
    const std::vector<EnumeratedOpenClDevice> &devices = enumerateOpenClGpuDevices();
    const int device_index = config.openClDeviceIndex >= 0 ? config.openClDeviceIndex : 0;
    if (device_index < 0 || device_index >= static_cast<int>(devices.size()))
    {
        if (errorMsg)
        {
            *errorMsg = "OpenCL GPU device index is outside the available device range";
        }
        return false;
    }
    if (srcGrays.empty() || srcGrays.size() > 16 || srcGrays.size() != srcCams.size())
    {
        if (errorMsg)
        {
            *errorMsg = "OpenCL PatchMatch requires 1 to 16 source images";
        }
        return false;
    }
    if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = "PatchMatch cancelled";
        }
        return false;
    }

    const std::shared_ptr<OpenClRuntime> runtime = openClRuntimeForDevice(
        devices[device_index], errorMsg);
    if (!runtime)
    {
        return false;
    }

    const int downsample_factor = std::max(1, config.downsampleFactor);
    const int width = std::max(1, refGray.cols / downsample_factor);
    const int height = std::max(1, refGray.rows / downsample_factor);
    const int pixel_count = width * height;
    const int source_count = static_cast<int>(srcGrays.size());
    const cv::Size working_size(width, height);

    bool reference_cache_hit = false;
    const std::shared_ptr<OpenClRuntime::CachedScaledImage> cached_reference =
        cachedScaledGrayFloat(runtime, refGray, working_size, &reference_cache_hit);
    const cv::Mat &reference_float = cached_reference->pixels;
    std::vector<float> packed_sources(static_cast<std::size_t>(source_count) * pixel_count);
    std::vector<std::shared_ptr<OpenClRuntime::CachedScaledImage>> cached_sources;
    cached_sources.reserve(srcGrays.size());
    int source_cache_hits = 0;
    for (int source_index = 0; source_index < source_count; ++source_index)
    {
        bool source_cache_hit = false;
        const std::shared_ptr<OpenClRuntime::CachedScaledImage> cached_source =
            cachedScaledGrayFloat(
                runtime,
                srcGrays[static_cast<std::size_t>(source_index)],
                working_size,
                &source_cache_hit);
        source_cache_hits += source_cache_hit ? 1 : 0;
        cached_sources.push_back(cached_source);
        const cv::Mat &source_float = cached_source->pixels;
        std::memcpy(packed_sources.data() + static_cast<std::size_t>(source_index) * pixel_count,
                    source_float.ptr<float>(),
                    static_cast<std::size_t>(pixel_count) * sizeof(float));
    }
    const std::vector<float> camera_data = sourceCameraData(
        refCam, srcCams, downsample_factor);

    const cv::Mat reference_mask = scaledBinaryMask(refValidMask, working_size);
    const bool has_reference_mask = !reference_mask.empty();
    std::vector<std::uint8_t> packed_reference_mask(static_cast<std::size_t>(pixel_count), 255);
    if (has_reference_mask)
    {
        std::memcpy(packed_reference_mask.data(),
                    reference_mask.ptr<std::uint8_t>(),
                    packed_reference_mask.size());
    }
    std::vector<std::uint8_t> packed_source_masks(
        static_cast<std::size_t>(source_count) * pixel_count, 255);
    bool has_source_masks = false;
    if (srcValidMasks)
    {
        for (int source_index = 0; source_index < source_count; ++source_index)
        {
            const cv::Mat source_mask = scaledBinaryMask(
                &(*srcValidMasks)[static_cast<std::size_t>(source_index)], working_size);
            if (source_mask.empty())
            {
                continue;
            }
            has_source_masks = true;
            std::memcpy(packed_source_masks.data() +
                            static_cast<std::size_t>(source_index) * pixel_count,
                        source_mask.ptr<std::uint8_t>(),
                        static_cast<std::size_t>(pixel_count));
        }
    }

    std::vector<float> packed_hint(static_cast<std::size_t>(pixel_count), 0.0f);
    std::vector<float> packed_hint_radius(static_cast<std::size_t>(pixel_count), 0.0f);
    bool has_hint = hintDepth && !hintDepth->empty();
    bool has_hint_radius = has_hint && hintRadius && !hintRadius->empty();
    auto copy_scaled_float = [&working_size](const cv::Mat &source, std::vector<float> &destination)
    {
        cv::Mat scaled;
        if (source.size() == working_size)
        {
            source.convertTo(scaled, CV_32F);
        }
        else
        {
            cv::Mat resized;
            cv::resize(source, resized, working_size, 0.0, 0.0, cv::INTER_NEAREST);
            resized.convertTo(scaled, CV_32F);
        }
        std::memcpy(destination.data(),
                    scaled.ptr<float>(),
                    destination.size() * sizeof(float));
    };
    if (has_hint)
    {
        copy_scaled_float(*hintDepth, packed_hint);
    }
    if (has_hint_radius)
    {
        copy_scaled_float(*hintRadius, packed_hint_radius);
    }

    const std::size_t image_float_bytes =
        static_cast<std::size_t>(pixel_count) * sizeof(float);
    const std::array<std::size_t, 9> buffer_sizes = {
        image_float_bytes,
        packed_sources.size() * sizeof(float),
        camera_data.size() * sizeof(float),
        packed_reference_mask.size(),
        packed_source_masks.size(),
        packed_hint.size() * sizeof(float),
        packed_hint_radius.size() * sizeof(float),
        image_float_bytes,
        image_float_bytes};
    const std::array<cl_mem_flags, 9> buffer_flags = {
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY,
        CL_MEM_WRITE_ONLY,
        CL_MEM_WRITE_ONLY};
    const std::array<const void *, 7> input_data = {
        reference_float.ptr<float>(),
        packed_sources.data(),
        camera_data.data(),
        packed_reference_mask.data(),
        packed_source_masks.data(),
        packed_hint.data(),
        packed_hint_radius.data()};

    const int patch_half = config.patchHalf;
    const int depth_sample_count = std::clamp(config.numIterations * 4, 32, 96);
    const float minimum_mask_ratio = config.minimumMaskedPatchSupportRatio;
    const float confidence_threshold = config.confidenceThresh;
    const float uniqueness_relative_step = config.enablePhotometricUniqueness
        ? config.photometricUniquenessRelativeDepthStep
        : 0.0f;
    const float uniqueness_margin = config.enablePhotometricUniqueness
        ? config.photometricUniquenessMinimumMargin
        : 0.0f;
    const float uniqueness_scale = config.photometricUniquenessMinimumConfidenceScale;
    const int reference_mask_flag = has_reference_mask ? 1 : 0;
    const int source_mask_flag = has_source_masks ? 1 : 0;
    const int hint_flag = has_hint ? 1 : 0;
    const int hint_radius_flag = has_hint_radius ? 1 : 0;
    const Camera::Intrinsics reference_intrinsics = refCam.intrinsics();
    const float scale = 1.0f / static_cast<float>(downsample_factor);
    const float inv_fx = 1.0f / (static_cast<float>(reference_intrinsics.focalX) * scale);
    const float inv_fy = 1.0f / (static_cast<float>(reference_intrinsics.focalY) * scale);
    const float cx = static_cast<float>(reference_intrinsics.principalX) * scale;
    const float cy = static_cast<float>(reference_intrinsics.principalY) * scale;

    // A single execution lane retains all input/output buffers. On integrated
    // GPUs this removes per-level clCreateBuffer/clReleaseMemObject churn while
    // the host image cache avoids repeating resize and float conversion.
    const auto slot_wait_start = std::chrono::steady_clock::now();
    std::optional<OpenClExecutionLaneLease> lane_lease =
        acquireOpenClExecutionLane(runtime, config.cancelFlag);
    if (!lane_lease)
    {
        if (errorMsg)
        {
            *errorMsg = "PatchMatch cancelled while waiting for an OpenCL execution lane";
        }
        return false;
    }
    const auto slot_acquired = std::chrono::steady_clock::now();
    OpenClRuntime::ExecutionLane &execution_lane = lane_lease->lane();
    constexpr std::size_t first_persistent_buffer = 0;
    for (std::size_t index = first_persistent_buffer;
         index < execution_lane.buffers.size();
         ++index)
    {
        if (!reserveOpenClBuffer(runtime->context,
                                 execution_lane.buffers[index],
                                 buffer_flags[index],
                                 buffer_sizes[index],
                                 errorMsg))
        {
            return false;
        }
    }

    std::array<cl_mem, 9> memory_arguments{};
    for (std::size_t index = 0; index < memory_arguments.size(); ++index)
    {
        memory_arguments[index] = execution_lane.buffers[index].memory;
    }
    std::size_t retained_buffer_bytes = 0;
    for (const OpenClRuntime::BufferSlot &buffer : execution_lane.buffers)
    {
        retained_buffer_bytes += buffer.memory ? buffer.capacity : 0;
    }
    cl_uint argument_index = 0;
    for (cl_mem argument : memory_arguments)
    {
        if (!setKernelArgument(execution_lane.kernel, argument_index++, argument, errorMsg))
        {
            return false;
        }
    }
    const auto set_argument = [&](const auto &value)
    {
        return setKernelArgument(execution_lane.kernel, argument_index++, value, errorMsg);
    };
    if (!set_argument(width)
        || !set_argument(height)
        || !set_argument(source_count)
        || !set_argument(patch_half)
        || !set_argument(depth_sample_count)
        || !set_argument(minimum_mask_ratio)
        || !set_argument(zNear)
        || !set_argument(zFar)
        || !set_argument(confidence_threshold)
        || !set_argument(uniqueness_relative_step)
        || !set_argument(uniqueness_margin)
        || !set_argument(uniqueness_scale)
        || !set_argument(reference_mask_flag)
        || !set_argument(source_mask_flag)
        || !set_argument(hint_flag)
        || !set_argument(hint_radius_flag)
        || !set_argument(inv_fx)
        || !set_argument(inv_fy)
        || !set_argument(cx)
        || !set_argument(cy))
    {
        return false;
    }

    const std::size_t local_size[2] = {16, 16};
    const std::size_t global_size[2] = {
        (static_cast<std::size_t>(width) + local_size[0] - 1) / local_size[0] * local_size[0],
        (static_cast<std::size_t>(height) + local_size[1] - 1) / local_size[1] * local_size[1]};
    cv::Mat depth_work(height, width, CV_32F);
    cv::Mat confidence_work(height, width, CV_32F);

    OpenClEvent first_write_event;
    OpenClEvent last_write_event;
    OpenClEvent kernel_event;
    OpenClEvent depth_read_event;
    OpenClEvent confidence_read_event;
    OpenClQueueDrain queue_drain(execution_lane.queue);
    const auto queue_wall_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < input_data.size(); ++index)
    {
        cl_event *event_output = nullptr;
        if (index == 0)
        {
            event_output = first_write_event.output();
        }
        else if (index + 1 == input_data.size())
        {
            event_output = last_write_event.output();
        }
        const cl_int write_error = clEnqueueWriteBuffer(execution_lane.queue,
                                                        memory_arguments[index],
                                                        CL_FALSE,
                                                        0,
                                                        buffer_sizes[index],
                                                        input_data[index],
                                                        0,
                                                        nullptr,
                                                        event_output);
        if (write_error != CL_SUCCESS)
        {
            if (errorMsg)
            {
                *errorMsg = openClError("clEnqueueWriteBuffer", write_error);
            }
            return false;
        }
        queue_drain.markPending();
    }

    cl_int error = clEnqueueNDRangeKernel(execution_lane.queue,
                                          execution_lane.kernel,
                                          2,
                                          nullptr,
                                          global_size,
                                          local_size,
                                          0,
                                          nullptr,
                                          kernel_event.output());
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clEnqueueNDRangeKernel", error);
        }
        return false;
    }
    queue_drain.markPending();

    error = clEnqueueReadBuffer(execution_lane.queue,
                                memory_arguments[7],
                                CL_FALSE,
                                0,
                                image_float_bytes,
                                depth_work.ptr<float>(),
                                0,
                                nullptr,
                                depth_read_event.output());
    if (error == CL_SUCCESS)
    {
        error = clEnqueueReadBuffer(execution_lane.queue,
                                    memory_arguments[8],
                                    CL_FALSE,
                                    0,
                                    image_float_bytes,
                                    confidence_work.ptr<float>(),
                                    0,
                                    nullptr,
                                    confidence_read_event.output());
    }
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clEnqueueReadBuffer", error);
        }
        return false;
    }

    error = clFlush(execution_lane.queue);
    if (error == CL_SUCCESS)
    {
        const cl_event final_event = confidence_read_event.get();
        error = clWaitForEvents(1, &final_event);
    }
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("OpenCL queue completion", error);
        }
        return false;
    }
    queue_drain.markComplete();
    const auto queue_wall_finished = std::chrono::steady_clock::now();

    cl_ulong write_start_nanoseconds = 0;
    cl_ulong write_end_nanoseconds = 0;
    cl_ulong kernel_start_nanoseconds = 0;
    cl_ulong kernel_end_nanoseconds = 0;
    cl_ulong read_start_nanoseconds = 0;
    cl_ulong read_end_nanoseconds = 0;
    const bool has_write_profile = eventProfileRange(
        first_write_event.get(),
        CL_PROFILING_COMMAND_START,
        last_write_event.get(),
        CL_PROFILING_COMMAND_END,
        write_start_nanoseconds,
        write_end_nanoseconds);
    const bool has_kernel_profile = eventProfileRange(
        kernel_event.get(),
        CL_PROFILING_COMMAND_START,
        kernel_event.get(),
        CL_PROFILING_COMMAND_END,
        kernel_start_nanoseconds,
        kernel_end_nanoseconds);
    const bool has_read_profile = eventProfileRange(
        depth_read_event.get(),
        CL_PROFILING_COMMAND_START,
        confidence_read_event.get(),
        CL_PROFILING_COMMAND_END,
        read_start_nanoseconds,
        read_end_nanoseconds);

    if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
    {
        if (errorMsg)
        {
            *errorMsg = "PatchMatch cancelled";
        }
        return false;
    }

    const auto read_finished = queue_wall_finished;
    const std::size_t execution_lane_index = lane_lease->laneIndex();
    lane_lease->release();
    const auto postprocess_start = read_finished;

    if (config.doMedianBlur && config.medianKernelSize > 1)
    {
        const cv::Mat valid_mask = depth_work > 0.0f;
        cv::Mat filtered;
        cv::medianBlur(depth_work, filtered, config.medianKernelSize);
        filtered.copyTo(depth_work, valid_mask);
    }
    if (config.doBilateralFilter)
    {
        const cv::Mat valid_mask = depth_work > 0.0f;
        cv::Mat filtered;
        cv::bilateralFilter(depth_work,
                            filtered,
                            config.bilateralD,
                            config.bilateralSigmaColor,
                            config.bilateralSigmaSpace);
        depth_work.setTo(cv::Scalar(0.0f));
        filtered.copyTo(depth_work, valid_mask);
    }

    if (downsample_factor > 1)
    {
        cv::resize(depth_work,
                   depthOut,
                   refGray.size(),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
        if (confOut)
        {
            cv::resize(confidence_work,
                       *confOut,
                       refGray.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
    }
    else
    {
        depthOut = depth_work;
        if (confOut)
        {
            *confOut = confidence_work;
        }
    }

    const int valid_count = cv::countNonZero(depthOut > 0.0f);
    std::size_t image_cache_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(runtime->imageCacheMutex);
        image_cache_bytes = runtime->imageCacheBytes;
    }
    const auto estimate_finished = std::chrono::steady_clock::now();
    const double host_prepare_ms = std::chrono::duration<double, std::milli>(
        slot_wait_start - estimate_start).count();
    const double slot_wait_ms = std::chrono::duration<double, std::milli>(
        slot_acquired - slot_wait_start).count();
    const double setup_ms = std::chrono::duration<double, std::milli>(
        queue_wall_start - slot_acquired).count();
    const double queue_ms = std::chrono::duration<double, std::milli>(
        queue_wall_finished - queue_wall_start).count();
    const double write_ms = has_write_profile
        ? static_cast<double>(write_end_nanoseconds - write_start_nanoseconds) / 1.0e6
        : 0.0;
    const double kernel_ms = has_kernel_profile
        ? static_cast<double>(kernel_end_nanoseconds - kernel_start_nanoseconds) / 1.0e6
        : queue_ms;
    const double read_ms = has_read_profile
        ? static_cast<double>(read_end_nanoseconds - read_start_nanoseconds) / 1.0e6
        : 0.0;
    const double driver_gap_ms = std::max(
        0.0, queue_ms - write_ms - kernel_ms - read_ms);
    const double postprocess_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - postprocess_start).count();
    const double total_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - estimate_start).count();
    LOG_INFO("[MVS][PatchMatch][OpenCL] device=%s lane=%zu unified=%d retained=%.1fMiB "
             "image_cache=%d/%d %.1fMiB "
             "size=%dx%d sources=%d sample_budget=%d valid=%d/%d "
             "prepare=%.1f ms wait=%.1f ms setup=%.1f ms queue=%.1f ms "
             "write=%.1f ms kernel=%.1f ms read=%.1f ms gap=%.1f ms "
             "post=%.1f ms total=%.1f ms",
             devices[device_index].info.name.c_str(),
             execution_lane_index,
             runtime->hostUnifiedMemory ? 1 : 0,
             static_cast<double>(retained_buffer_bytes) / (1024.0 * 1024.0),
             (reference_cache_hit ? 1 : 0) + source_cache_hits,
             source_count + 1,
             static_cast<double>(image_cache_bytes) / (1024.0 * 1024.0),
             width,
             height,
             source_count,
             depth_sample_count,
             valid_count,
             depthOut.rows * depthOut.cols,
             host_prepare_ms,
             slot_wait_ms,
             setup_ms,
             queue_ms,
             write_ms,
             kernel_ms,
             read_ms,
             driver_gap_ms,
             postprocess_ms,
             total_ms);
    return true;
}

} // namespace mvs
} // namespace xjw
