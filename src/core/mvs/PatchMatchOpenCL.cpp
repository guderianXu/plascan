#define CL_TARGET_OPENCL_VERSION 120

#include "PatchMatchCUDA.h"
#include "PatchMatchOpenCLKernels.h"

#include "Logger.h"

#include <CL/cl.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
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
};

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

std::vector<EnumeratedOpenClDevice> enumerateOpenClGpuDevices()
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
            entry.platform = platform;
            entry.device = device;
            result.push_back(std::move(entry));
        }
    }
    return result;
}

struct OpenClRuntime
{
    struct ExecutionLane
    {
        cl_command_queue queue = nullptr;
        cl_kernel kernel = nullptr;
        bool busy = false;
    };

    static constexpr std::size_t kExecutionLaneCount = 2;

    cl_context context = nullptr;
    cl_program program = nullptr;
    std::array<ExecutionLane, kExecutionLaneCount> lanes;
    std::mutex laneMutex;
    std::condition_variable laneAvailable;

    ~OpenClRuntime()
    {
        for (ExecutionLane &lane : lanes)
        {
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
    const char *source = detail::kPatchMatchOpenClSource;
    const std::size_t source_length = std::strlen(source);
    runtime->program = clCreateProgramWithSource(
        runtime->context, 1, &source, &source_length, &error);
    if (error != CL_SUCCESS || !runtime->program)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clCreateProgramWithSource", error);
        }
        return nullptr;
    }
    error = clBuildProgram(runtime->program, 1, &device.device, "-cl-mad-enable", nullptr, nullptr);
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

class OpenClBuffer
{
public:
    OpenClBuffer() = default;
    OpenClBuffer(const OpenClBuffer &) = delete;
    OpenClBuffer &operator=(const OpenClBuffer &) = delete;

    ~OpenClBuffer()
    {
        if (_buffer)
        {
            clReleaseMemObject(_buffer);
        }
    }

    bool create(cl_context context,
                cl_mem_flags flags,
                std::size_t bytes,
                const void *hostData,
                std::string *errorMsg)
    {
        cl_int error = CL_SUCCESS;
        _buffer = clCreateBuffer(
            context, flags, bytes, const_cast<void *>(hostData), &error);
        if (error != CL_SUCCESS || !_buffer)
        {
            if (errorMsg)
            {
                *errorMsg = openClError("clCreateBuffer", error);
            }
            return false;
        }
        return true;
    }

    cl_mem get() const
    {
        return _buffer;
    }

private:
    cl_mem _buffer = nullptr;
};

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
    const std::vector<EnumeratedOpenClDevice> devices = enumerateOpenClGpuDevices();
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
    const std::vector<EnumeratedOpenClDevice> devices = enumerateOpenClGpuDevices();
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

    const cv::Mat reference_float = scaledGrayFloat(refGray, working_size);
    std::vector<float> packed_sources(static_cast<std::size_t>(source_count) * pixel_count);
    for (int source_index = 0; source_index < source_count; ++source_index)
    {
        const cv::Mat source_float = scaledGrayFloat(
            srcGrays[static_cast<std::size_t>(source_index)], working_size);
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

    OpenClBuffer reference_buffer;
    OpenClBuffer sources_buffer;
    OpenClBuffer cameras_buffer;
    OpenClBuffer reference_mask_buffer;
    OpenClBuffer source_masks_buffer;
    OpenClBuffer hint_buffer;
    OpenClBuffer hint_radius_buffer;
    OpenClBuffer depth_buffer;
    OpenClBuffer confidence_buffer;
    const cl_mem_flags input_flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    if (!reference_buffer.create(runtime->context,
                                 input_flags,
                                 static_cast<std::size_t>(pixel_count) * sizeof(float),
                                 reference_float.ptr<float>(),
                                 errorMsg)
        || !sources_buffer.create(runtime->context,
                                  input_flags,
                                  packed_sources.size() * sizeof(float),
                                  packed_sources.data(),
                                  errorMsg)
        || !cameras_buffer.create(runtime->context,
                                  input_flags,
                                  camera_data.size() * sizeof(float),
                                  camera_data.data(),
                                  errorMsg)
        || !reference_mask_buffer.create(runtime->context,
                                         input_flags,
                                         packed_reference_mask.size(),
                                         packed_reference_mask.data(),
                                         errorMsg)
        || !source_masks_buffer.create(runtime->context,
                                      input_flags,
                                      packed_source_masks.size(),
                                      packed_source_masks.data(),
                                      errorMsg)
        || !hint_buffer.create(runtime->context,
                               input_flags,
                               packed_hint.size() * sizeof(float),
                               packed_hint.data(),
                               errorMsg)
        || !hint_radius_buffer.create(runtime->context,
                                      input_flags,
                                      packed_hint_radius.size() * sizeof(float),
                                      packed_hint_radius.data(),
                                      errorMsg)
        || !depth_buffer.create(runtime->context,
                                CL_MEM_WRITE_ONLY,
                                static_cast<std::size_t>(pixel_count) * sizeof(float),
                                nullptr,
                                errorMsg)
        || !confidence_buffer.create(runtime->context,
                                     CL_MEM_WRITE_ONLY,
                                     static_cast<std::size_t>(pixel_count) * sizeof(float),
                                     nullptr,
                                     errorMsg))
    {
        return false;
    }

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

    // Buffer creation and CPU image preparation are independent per host lane.
    // Each lane owns a command queue and kernel object, so two frames can be
    // queued concurrently without racing clSetKernelArg state.
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
    const std::array<cl_mem, 9> memory_arguments = {
        reference_buffer.get(),
        sources_buffer.get(),
        cameras_buffer.get(),
        reference_mask_buffer.get(),
        source_masks_buffer.get(),
        hint_buffer.get(),
        hint_radius_buffer.get(),
        depth_buffer.get(),
        confidence_buffer.get()};
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
    cl_event event = nullptr;
    const auto kernel_wall_start = std::chrono::steady_clock::now();
    cl_int error = clEnqueueNDRangeKernel(execution_lane.queue,
                                          execution_lane.kernel,
                                          2,
                                          nullptr,
                                          global_size,
                                          local_size,
                                          0,
                                          nullptr,
                                          &event);
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clEnqueueNDRangeKernel", error);
        }
        return false;
    }
    error = clWaitForEvents(1, &event);
    const auto kernel_wall_finished = std::chrono::steady_clock::now();
    cl_ulong kernel_start_nanoseconds = 0;
    cl_ulong kernel_end_nanoseconds = 0;
    const bool has_kernel_profile =
        clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(kernel_start_nanoseconds),
                                &kernel_start_nanoseconds,
                                nullptr) == CL_SUCCESS &&
        clGetEventProfilingInfo(event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(kernel_end_nanoseconds),
                                &kernel_end_nanoseconds,
                                nullptr) == CL_SUCCESS &&
        kernel_end_nanoseconds >= kernel_start_nanoseconds;
    clReleaseEvent(event);
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clWaitForEvents", error);
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

    cv::Mat depth_work(height, width, CV_32F);
    cv::Mat confidence_work(height, width, CV_32F);
    const auto read_start = std::chrono::steady_clock::now();
    error = clEnqueueReadBuffer(execution_lane.queue,
                                depth_buffer.get(),
                                CL_TRUE,
                                0,
                                static_cast<std::size_t>(pixel_count) * sizeof(float),
                                depth_work.ptr<float>(),
                                0,
                                nullptr,
                                nullptr);
    if (error == CL_SUCCESS)
    {
        error = clEnqueueReadBuffer(execution_lane.queue,
                                    confidence_buffer.get(),
                                    CL_TRUE,
                                    0,
                                    static_cast<std::size_t>(pixel_count) * sizeof(float),
                                    confidence_work.ptr<float>(),
                                    0,
                                    nullptr,
                                    nullptr);
    }
    if (error != CL_SUCCESS)
    {
        if (errorMsg)
        {
            *errorMsg = openClError("clEnqueueReadBuffer", error);
        }
        return false;
    }
    const auto read_finished = std::chrono::steady_clock::now();
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
    const auto estimate_finished = std::chrono::steady_clock::now();
    const double host_prepare_ms = std::chrono::duration<double, std::milli>(
        slot_wait_start - estimate_start).count();
    const double slot_wait_ms = std::chrono::duration<double, std::milli>(
        slot_acquired - slot_wait_start).count();
    const double queue_ms = std::chrono::duration<double, std::milli>(
        kernel_wall_finished - kernel_wall_start).count();
    const double kernel_ms = has_kernel_profile
        ? static_cast<double>(kernel_end_nanoseconds - kernel_start_nanoseconds) / 1.0e6
        : queue_ms;
    const double read_ms = std::chrono::duration<double, std::milli>(
        read_finished - read_start).count();
    const double postprocess_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - postprocess_start).count();
    const double total_ms = std::chrono::duration<double, std::milli>(
        estimate_finished - estimate_start).count();
    LOG_INFO("[MVS][PatchMatch][OpenCL] device=%s lane=%zu size=%dx%d sources=%d samples=%d valid=%d/%d "
             "prepare=%.1f ms wait=%.1f ms queue=%.1f ms kernel=%.1f ms "
             "read=%.1f ms post=%.1f ms total=%.1f ms",
             devices[device_index].info.name.c_str(),
             execution_lane_index,
             width,
             height,
             source_count,
             depth_sample_count,
             valid_count,
             depthOut.rows * depthOut.cols,
             host_prepare_ms,
             slot_wait_ms,
             queue_ms,
             kernel_ms,
             read_ms,
             postprocess_ms,
             total_ms);
    return true;
}

} // namespace mvs
} // namespace xjw
