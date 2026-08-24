#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "TerrainGpuBackend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace xjw::terrain_internal
{

    namespace
    {

        constexpr const char* kTerrainOpenClKernels = R"CLC(
#ifndef TERRAIN_MOSAIC_ONLY
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define CAMERA_VALUE_STRIDE 23
#define CAMERA_METADATA_STRIDE 7

inline int sample_dem(__global const float *elevation,
                      __global const uchar *valid,
                      int width,
                      int height,
                      double min_x,
                      double min_y,
                      double step_x,
                      double step_y,
                      double world_x,
                      double world_y,
                      double *sample)
{
    const double grid_x = (world_x - min_x) / step_x;
    const double grid_y = (world_y - min_y) / step_y;
    if (grid_x < -0.5 || grid_y < -0.5
        || grid_x > (double)width - 0.5 || grid_y > (double)height - 0.5)
    {
        return 0;
    }
    const int nearest_x = clamp((int)floor(grid_x + 0.5), 0, width - 1);
    const int nearest_y = clamp((int)floor(grid_y + 0.5), 0, height - 1);
    const int x0 = clamp((int)floor(grid_x), 0, width - 1);
    const int y0 = clamp((int)floor(grid_y), 0, height - 1);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const int index00 = y0 * width + x0;
    const int index10 = y0 * width + x1;
    const int index01 = y1 * width + x0;
    const int index11 = y1 * width + x1;
    if (valid[index00] && valid[index10] && valid[index01] && valid[index11])
    {
        const double fx = clamp(grid_x - (double)x0, 0.0, 1.0);
        const double fy = clamp(grid_y - (double)y0, 0.0, 1.0);
        *sample = (1.0 - fx) * (1.0 - fy) * elevation[index00]
            + fx * (1.0 - fy) * elevation[index10]
            + (1.0 - fx) * fy * elevation[index01]
            + fx * fy * elevation[index11];
        return isfinite(*sample);
    }
    const int nearest = nearest_y * width + nearest_x;
    if (!valid[nearest])
    {
        return 0;
    }
    *sample = elevation[nearest];
    return isfinite(*sample);
}

inline int sample_camera(__global const double *camera,
                         __global const int *metadata,
                         __global const uchar *images,
                         __global const uchar *masks,
                         double world_x,
                         double world_y,
                         double world_z,
                         float *blue,
                         float *green,
                         float *red,
                         double *weight)
{
    const double relative_x = world_x - camera[9];
    const double relative_y = world_y - camera[10];
    const double relative_z = world_z - camera[11];
    const double camera_x = camera[0] * relative_x
        + camera[3] * relative_y + camera[6] * relative_z;
    const double camera_y = camera[1] * relative_x
        + camera[4] * relative_y + camera[7] * relative_z;
    const double camera_z = camera[2] * relative_x
        + camera[5] * relative_y + camera[8] * relative_z;
    const double positive_depth = metadata[6] ? -camera_z : camera_z;
    if (!(positive_depth > 1.0e-9) || !isfinite(positive_depth))
    {
        return 0;
    }
    const double x = camera_x / camera_z;
    const double y = camera_y / camera_z;
    const double r2 = x * x + y * y;
    const double radial = 1.0 + camera[16] * r2
        + camera[17] * r2 * r2 + camera[18] * r2 * r2 * r2;
    const double distorted_x = x * radial + 2.0 * camera[19] * x * y
        + camera[20] * (r2 + 2.0 * x * x);
    const double distorted_y = y * radial + camera[19] * (r2 + 2.0 * y * y)
        + 2.0 * camera[20] * x * y;
    const double u = (double)metadata[4] * camera[12] * distorted_x + camera[14];
    const double v = (double)metadata[5] * camera[13] * distorted_y + camera[15];
    const int width = metadata[0];
    const int height = metadata[1];
    if (u < 0.0 || v < 0.0 || u >= (double)(width - 1) || v >= (double)(height - 1))
    {
        return 0;
    }
    if (metadata[3] >= 0)
    {
        const int mask_x = clamp((int)floor(u + 0.5), 0, width - 1);
        const int mask_y = clamp((int)floor(v + 0.5), 0, height - 1);
        if (masks[metadata[3] + mask_y * width + mask_x])
        {
            return 0;
        }
    }
    const int x0 = clamp((int)floor(u), 0, width - 1);
    const int y0 = clamp((int)floor(v), 0, height - 1);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float fx = (float)(u - (double)x0);
    const float fy = (float)(v - (double)y0);
    const int image_offset = metadata[2];
    float sampled[3];
    for (int channel = 0; channel < 3; ++channel)
    {
        const float c00 = images[image_offset + (y0 * width + x0) * 3 + channel];
        const float c10 = images[image_offset + (y0 * width + x1) * 3 + channel];
        const float c01 = images[image_offset + (y1 * width + x0) * 3 + channel];
        const float c11 = images[image_offset + (y1 * width + x1) * 3 + channel];
        sampled[channel] = ((1.0f - fx) * (1.0f - fy) * c00
            + fx * (1.0f - fy) * c10 + (1.0f - fx) * fy * c01
            + fx * fy * c11) * (float)camera[21];
    }
    *blue = sampled[0];
    *green = sampled[1];
    *red = sampled[2];
    const double du = (u - camera[14]) / fmax(1.0, camera[12]);
    const double dv = (v - camera[15]) / fmax(1.0, camera[13]);
    const double view_weight = 1.0 / (1.0 + du * du + dv * dv);
    const double edge_distance = fmin(
        fmin(u, (double)(width - 1) - u), fmin(v, (double)(height - 1) - v));
    const double edge_weight = clamp(edge_distance / 20.0, 0.05, 1.0);
    *weight = view_weight * edge_weight * camera[22];
    return *weight > 0.0;
}

__kernel void ortho_projection(__global const float *dem_elevation,
                               __global const uchar *dem_valid,
                               int dem_width,
                               int dem_height,
                               double dem_min_x,
                               double dem_min_y,
                               double dem_step_x,
                               double dem_step_y,
                               int output_width,
                               int output_height,
                               double output_min_edge_x,
                               double output_min_edge_y,
                               double output_step_x,
                               double output_step_y,
                               double elevation_offset,
                               __global const double *camera_values,
                               __global const int *camera_metadata,
                               int frame_count,
                               __global const uchar *images,
                               __global const uchar *masks,
                               int blend_mode,
                               __global uchar *output_image,
                               __global uchar *surface_mask,
                               __global uchar *coverage_mask,
                               __global int *contributed_frames)
{
    const int col = (int)get_global_id(0);
    const int row = (int)get_global_id(1);
    if (col >= output_width || row >= output_height)
    {
        return;
    }
    const int pixel = row * output_width + col;
    output_image[pixel * 3] = 0;
    output_image[pixel * 3 + 1] = 0;
    output_image[pixel * 3 + 2] = 0;
    surface_mask[pixel] = 0;
    coverage_mask[pixel] = 0;
    const double world_x = output_min_edge_x + ((double)col + 0.5) * output_step_x;
    const double world_y = output_min_edge_y + ((double)row + 0.5) * output_step_y;
    double elevation = 0.0;
    if (!sample_dem(dem_elevation, dem_valid, dem_width, dem_height,
                    dem_min_x, dem_min_y, dem_step_x, dem_step_y,
                    world_x, world_y, &elevation))
    {
        return;
    }
    surface_mask[pixel] = 255;
    float accumulated_blue = 0.0f;
    float accumulated_green = 0.0f;
    float accumulated_red = 0.0f;
    double total_weight = 0.0;
    double best_weight = -1.0;
    int best_frame = -1;
    int accepted = 0;
    for (int frame = 0; frame < frame_count; ++frame)
    {
        float blue = 0.0f;
        float green = 0.0f;
        float red = 0.0f;
        double weight = 0.0;
        if (!sample_camera(camera_values + frame * CAMERA_VALUE_STRIDE,
                           camera_metadata + frame * CAMERA_METADATA_STRIDE,
                           images, masks, world_x, world_y,
                           elevation + elevation_offset,
                           &blue, &green, &red, &weight))
        {
            continue;
        }
        ++accepted;
        if (blend_mode == 2)
        {
            accumulated_blue = blue;
            accumulated_green = green;
            accumulated_red = red;
            best_frame = frame;
            break;
        }
        if (blend_mode == 0)
        {
            if (weight > best_weight)
            {
                best_weight = weight;
                accumulated_blue = blue;
                accumulated_green = green;
                accumulated_red = red;
                best_frame = frame;
            }
            continue;
        }
        accumulated_blue += blue * (float)weight;
        accumulated_green += green * (float)weight;
        accumulated_red += red * (float)weight;
        total_weight += weight;
        atomic_xchg((volatile __global int *)(contributed_frames + frame), 1);
    }
    if (accepted == 0)
    {
        return;
    }
    if (blend_mode == 1)
    {
        if (!(total_weight > 0.0))
        {
            return;
        }
        accumulated_blue /= (float)total_weight;
        accumulated_green /= (float)total_weight;
        accumulated_red /= (float)total_weight;
    }
    else if (best_frame >= 0)
    {
        atomic_xchg((volatile __global int *)(contributed_frames + best_frame), 1);
    }
    output_image[pixel * 3] = (uchar)clamp(accumulated_blue, 0.0f, 255.0f);
    output_image[pixel * 3 + 1] = (uchar)clamp(accumulated_green, 0.0f, 255.0f);
    output_image[pixel * 3 + 2] = (uchar)clamp(accumulated_red, 0.0f, 255.0f);
    coverage_mask[pixel] = 255;
}
#endif

inline int mosaic_valid(__global const float *elevation,
                        __global const uchar *valid,
                        int tile,
                        int pixel,
                        int pixel_count)
{
    const int index = tile * pixel_count + pixel;
    return valid[index] && isfinite(elevation[index]);
}

inline float mosaic_kth(__global const float *elevation,
                        __global const uchar *valid,
                        int tile_count,
                        int pixel,
                        int pixel_count,
                        int kth)
{
    for (int candidate_tile = 0; candidate_tile < tile_count; ++candidate_tile)
    {
        if (!mosaic_valid(elevation, valid, candidate_tile, pixel, pixel_count))
        {
            continue;
        }
        const float candidate = elevation[candidate_tile * pixel_count + pixel];
        int less = 0;
        int equal = 0;
        for (int tile = 0; tile < tile_count; ++tile)
        {
            if (!mosaic_valid(elevation, valid, tile, pixel, pixel_count))
            {
                continue;
            }
            const float value = elevation[tile * pixel_count + pixel];
            less += value < candidate ? 1 : 0;
            equal += value == candidate ? 1 : 0;
        }
        if (less <= kth && kth < less + equal)
        {
            return candidate;
        }
    }
    return 0.0f;
}

__kernel void dem_mosaic(__global const float *input_elevation,
                         __global const uchar *input_valid,
                         __global const float *input_confidence,
                         __global const float *input_error,
                         int tile_count,
                         int pixel_count,
                         int blend_mode,
                         __global float *output_elevation,
                         __global uchar *output_valid,
                         __global int *output_count,
                         __global float *output_confidence,
                         __global float *output_error)
{
    const int pixel = (int)get_global_id(0);
    if (pixel >= pixel_count)
    {
        return;
    }
    int count = 0;
    float sum = 0.0f;
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    float confidence_sum = 0.0f;
    float error_sum = 0.0f;
    float first = 0.0f;
    float last = 0.0f;
    float minimum = INFINITY;
    float maximum = -INFINITY;
    for (int tile = 0; tile < tile_count; ++tile)
    {
        if (!mosaic_valid(input_elevation, input_valid, tile, pixel, pixel_count))
        {
            continue;
        }
        const int index = tile * pixel_count + pixel;
        const float value = input_elevation[index];
        const float confidence = fmax(0.0f, input_confidence[index]);
        const float error = fmax(0.0f, input_error[index]);
        if (count == 0)
        {
            first = value;
        }
        last = value;
        minimum = fmin(minimum, value);
        maximum = fmax(maximum, value);
        sum += value;
        confidence_sum += confidence;
        error_sum += error;
        float weight = 1.0f;
        if (blend_mode == 6)
        {
            weight = confidence;
        }
        else if (blend_mode == 7)
        {
            weight = 1.0f / fmax(error, 1.0e-6f);
        }
        weighted_sum += value * weight;
        weight_sum += weight;
        ++count;
    }
    output_elevation[pixel] = 0.0f;
    output_valid[pixel] = 0;
    output_count[pixel] = 0;
    output_confidence[pixel] = 0.0f;
    output_error[pixel] = 0.0f;
    if (count == 0)
    {
        return;
    }
    float blended = sum / (float)count;
    if (blend_mode == 0)
    {
        blended = first;
    }
    else if (blend_mode == 1)
    {
        blended = last;
    }
    else if (blend_mode == 3)
    {
        const float upper = mosaic_kth(
            input_elevation, input_valid, tile_count, pixel, pixel_count, count / 2);
        blended = count % 2 == 0
            ? 0.5f * (upper + mosaic_kth(input_elevation, input_valid,
                                         tile_count, pixel, pixel_count, count / 2 - 1))
            : upper;
    }
    else if (blend_mode == 4)
    {
        blended = minimum;
    }
    else if (blend_mode == 5)
    {
        blended = maximum;
    }
    else if (blend_mode == 6 || blend_mode == 7)
    {
        blended = weight_sum > 0.0f ? weighted_sum / weight_sum : last;
    }
    output_elevation[pixel] = blended;
    output_valid[pixel] = 255;
    output_count[pixel] = count;
    output_confidence[pixel] = confidence_sum / (float)count;
    output_error[pixel] = error_sum / (float)count;
}
)CLC";

        template <typename T, cl_int (*ReleaseFunction)(T)> class ClHandle final
        {
        public:
            ClHandle() = default;
            explicit ClHandle(T value) : _value(value)
            {
            }
            ClHandle(const ClHandle&) = delete;
            ClHandle& operator=(const ClHandle&) = delete;
            ~ClHandle()
            {
                if (_value)
                {
                    ReleaseFunction(_value);
                }
            }
            T get() const
            {
                return _value;
            }
            void reset(T value = nullptr)
            {
                if (_value)
                {
                    ReleaseFunction(_value);
                }
                _value = value;
            }

        private:
            T _value = nullptr;
        };

        using ContextHandle = ClHandle<cl_context, clReleaseContext>;
        using QueueHandle = ClHandle<cl_command_queue, clReleaseCommandQueue>;
        using ProgramHandle = ClHandle<cl_program, clReleaseProgram>;
        using KernelHandle = ClHandle<cl_kernel, clReleaseKernel>;
        using BufferHandle = ClHandle<cl_mem, clReleaseMemObject>;

        struct CachedRuntime final
        {
            ContextHandle context;
            ProgramHandle program;
        };

        struct RuntimeCacheKey final
        {
            cl_device_id device = nullptr;
            bool requireDoublePrecision = false;
        };

        struct RuntimeCacheKeyLess final
        {
            bool operator()(const RuntimeCacheKey& left, const RuntimeCacheKey& right) const
            {
                const std::less<cl_device_id> device_less;
                if (device_less(left.device, right.device))
                {
                    return true;
                }
                if (device_less(right.device, left.device))
                {
                    return false;
                }
                return left.requireDoublePrecision < right.requireDoublePrecision;
            }
        };

        using RuntimeCache = std::map<RuntimeCacheKey, std::shared_ptr<CachedRuntime>, RuntimeCacheKeyLess>;

        std::mutex& runtimeCacheMutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        RuntimeCache& runtimeCache()
        {
            static RuntimeCache cache;
            return cache;
        }

        std::string clError(const char* operation, cl_int status)
        {
            std::ostringstream stream;
            stream << "terrain OpenCL " << operation << " 失败，错误码 " << status;
            return stream.str();
        }

        std::string deviceString(cl_device_id device, cl_device_info key)
        {
            std::size_t size = 0;
            if (clGetDeviceInfo(device, key, 0, nullptr, &size) != CL_SUCCESS || size == 0)
            {
                return {};
            }
            std::string value(size, '\0');
            if (clGetDeviceInfo(device, key, size, value.data(), nullptr) != CL_SUCCESS)
            {
                return {};
            }
            while (!value.empty() && value.back() == '\0')
            {
                value.pop_back();
            }
            return value;
        }

        std::vector<cl_device_id> gpuDevices()
        {
            cl_uint platform_count = 0;
            if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0)
            {
                return {};
            }
            std::vector<cl_platform_id> platforms(platform_count);
            if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS)
            {
                return {};
            }
            std::vector<cl_device_id> devices;
            for (cl_platform_id platform : platforms)
            {
                cl_uint count = 0;
                const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
                if (status != CL_SUCCESS || count == 0)
                {
                    continue;
                }
                const std::size_t old_size = devices.size();
                devices.resize(old_size + count);
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data() + old_size, nullptr) !=
                    CL_SUCCESS)
                {
                    devices.resize(old_size);
                }
            }
            return devices;
        }

        bool supportsDoublePrecision(cl_device_id device)
        {
            const std::string extensions = deviceString(device, CL_DEVICE_EXTENSIONS);
            return extensions.find("cl_khr_fp64") != std::string::npos ||
                   extensions.find("cl_amd_fp64") != std::string::npos;
        }

        bool acquireCachedRuntime(cl_device_id device,
                                  bool requireDoublePrecision,
                                  std::shared_ptr<CachedRuntime>* runtime,
                                  std::string* errorMsg)
        {
            std::lock_guard<std::mutex> lock(runtimeCacheMutex());
            RuntimeCache& cache = runtimeCache();
            const RuntimeCacheKey key{device, requireDoublePrecision};
            const auto cached = cache.find(key);
            if (cached != cache.end())
            {
                *runtime = cached->second;
                return true;
            }

            auto candidate = std::make_shared<CachedRuntime>();
            cl_int status = CL_SUCCESS;
            candidate->context.reset(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
            if (status != CL_SUCCESS || !candidate->context.get())
            {
                if (errorMsg)
                    *errorMsg = clError("创建上下文", status);
                return false;
            }
            const std::size_t source_length = std::char_traits<char>::length(kTerrainOpenClKernels);
            const char* source = kTerrainOpenClKernels;
            candidate->program.reset(
                clCreateProgramWithSource(candidate->context.get(), 1, &source, &source_length, &status));
            if (status != CL_SUCCESS || !candidate->program.get())
            {
                if (errorMsg)
                    *errorMsg = clError("创建 kernel 程序", status);
                return false;
            }
            const char* build_options =
                requireDoublePrecision ? "-cl-std=CL1.2" : "-cl-std=CL1.2 -DTERRAIN_MOSAIC_ONLY=1";
            status = clBuildProgram(candidate->program.get(), 1, &device, build_options, nullptr, nullptr);
            if (status != CL_SUCCESS)
            {
                std::size_t log_size = 0;
                clGetProgramBuildInfo(candidate->program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log(log_size, '\0');
                if (log_size > 0)
                {
                    clGetProgramBuildInfo(
                        candidate->program.get(), device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                }
                if (errorMsg)
                {
                    *errorMsg = clError("编译 kernel", status) + ": " + log;
                }
                return false;
            }
            cache.emplace(key, candidate);
            *runtime = std::move(candidate);
            return true;
        }

        bool createInvocationRuntime(cl_device_id device,
                                     bool requireDoublePrecision,
                                     std::shared_ptr<CachedRuntime>* runtime,
                                     QueueHandle* queue,
                                     std::string* errorMsg)
        {
            if (!acquireCachedRuntime(device, requireDoublePrecision, runtime, errorMsg))
            {
                return false;
            }

            cl_int status = CL_SUCCESS;
            queue->reset(clCreateCommandQueue((*runtime)->context.get(), device, 0, &status));
            if (status != CL_SUCCESS || !queue->get())
            {
                if (errorMsg)
                    *errorMsg = clError("创建命令队列", status);
                return false;
            }
            return true;
        }

        template <typename T>
        bool
        createInputBuffer(cl_context context, const std::vector<T>& values, BufferHandle* buffer, std::string* errorMsg)
        {
            cl_int status = CL_SUCCESS;
            buffer->reset(clCreateBuffer(context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         std::max<std::size_t>(values.size(), 1) * sizeof(T),
                                         values.empty() ? nullptr : const_cast<T*>(values.data()),
                                         &status));
            if (status != CL_SUCCESS || !buffer->get())
            {
                if (errorMsg)
                    *errorMsg = clError("创建输入缓冲", status);
                return false;
            }
            return true;
        }

        template <typename T>
        bool createOutputBuffer(cl_context context,
                                std::vector<T>* values,
                                BufferHandle* buffer,
                                std::string* errorMsg,
                                bool initialize = false)
        {
            cl_int status = CL_SUCCESS;
            cl_mem_flags flags = CL_MEM_WRITE_ONLY;
            void* host = nullptr;
            if (initialize)
            {
                flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
                host = values->data();
            }
            buffer->reset(
                clCreateBuffer(context, flags, std::max<std::size_t>(values->size(), 1) * sizeof(T), host, &status));
            if (status != CL_SUCCESS || !buffer->get())
            {
                if (errorMsg)
                    *errorMsg = clError("创建输出缓冲", status);
                return false;
            }
            return true;
        }

        template <typename T>
        bool setKernelArgument(cl_kernel kernel, cl_uint index, const T& value, std::string* errorMsg)
        {
            const cl_int status = clSetKernelArg(kernel, index, sizeof(T), &value);
            if (status != CL_SUCCESS)
            {
                if (errorMsg)
                    *errorMsg = clError("设置 kernel 参数", status);
                return false;
            }
            return true;
        }

        template <typename T>
        bool
        readBuffer(cl_command_queue queue, const BufferHandle& buffer, std::vector<T>* values, std::string* errorMsg)
        {
            if (values->empty())
            {
                return true;
            }
            const cl_int status = clEnqueueReadBuffer(
                queue, buffer.get(), CL_TRUE, 0, values->size() * sizeof(T), values->data(), 0, nullptr, nullptr);
            if (status != CL_SUCCESS)
            {
                if (errorMsg)
                    *errorMsg = clError("读取输出缓冲", status);
                return false;
            }
            return true;
        }

        bool selectDevice(int deviceIndex, bool requireDoublePrecision, cl_device_id* device, TerrainDeviceInfo* info)
        {
            const std::vector<cl_device_id> devices = gpuDevices();
            if (devices.empty())
            {
                info->error = "未发现 terrain 可用的 OpenCL GPU 设备";
                return false;
            }

            int resolved = -1;
            if (deviceIndex >= 0)
            {
                if (deviceIndex >= static_cast<int>(devices.size()))
                {
                    std::ostringstream stream;
                    stream << "terrain OpenCL GPU 设备索引 " << deviceIndex << " 不可用，可用 GPU 数为 "
                           << devices.size();
                    info->error = stream.str();
                    return false;
                }
                resolved = deviceIndex;
            }
            else
            {
                for (std::size_t index = 0; index < devices.size(); ++index)
                {
                    if (!requireDoublePrecision || supportsDoublePrecision(devices[index]))
                    {
                        resolved = static_cast<int>(index);
                        break;
                    }
                }
            }

            if (resolved < 0 ||
                (requireDoublePrecision && !supportsDoublePrecision(devices[static_cast<std::size_t>(resolved)])))
            {
                if (deviceIndex >= 0)
                {
                    std::ostringstream stream;
                    stream << "terrain OpenCL GPU 设备 " << deviceIndex
                           << " 不支持正射投影所需的 cl_khr_fp64 或 cl_amd_fp64";
                    info->error = stream.str();
                }
                else
                {
                    info->error = "未发现支持 cl_khr_fp64 或 cl_amd_fp64 的 terrain OpenCL 正射投影 GPU";
                }
                return false;
            }
            const std::string vendor = deviceString(devices[static_cast<std::size_t>(resolved)], CL_DEVICE_VENDOR);
            const std::string name = deviceString(devices[static_cast<std::size_t>(resolved)], CL_DEVICE_NAME);
            info->available = true;
            info->resolvedIndex = resolved;
            info->name = "OpenCL · " + vendor + (vendor.empty() ? "" : " ") + name;
            *device = devices[static_cast<std::size_t>(resolved)];
            return true;
        }

    } // namespace

    TerrainDeviceInfo queryTerrainOpenClDevice(int deviceIndex)
    {
        TerrainDeviceInfo info;
        cl_device_id device = nullptr;
        selectDevice(deviceIndex, true, &device, &info);
        return info;
    }

    TerrainDeviceInfo queryTerrainOpenClMosaicDevice(int deviceIndex)
    {
        TerrainDeviceInfo info;
        cl_device_id device = nullptr;
        selectDevice(deviceIndex, false, &device, &info);
        return info;
    }

    bool runTerrainOpenClOrtho(const PackedOrthoProjection& input,
                               int deviceIndex,
                               PackedOrthoProjectionResult* output,
                               std::string* errorMsg)
    {
        if (!output)
        {
            if (errorMsg)
                *errorMsg = "terrain OpenCL 正射输出对象为空";
            return false;
        }
        TerrainDeviceInfo info;
        cl_device_id device = nullptr;
        if (!selectDevice(deviceIndex, true, &device, &info))
        {
            if (errorMsg)
                *errorMsg = info.error;
            return false;
        }
        std::shared_ptr<CachedRuntime> runtime;
        QueueHandle queue;
        if (!createInvocationRuntime(device, true, &runtime, &queue, errorMsg))
        {
            return false;
        }
        cl_int status = CL_SUCCESS;
        KernelHandle kernel(clCreateKernel(runtime->program.get(), "ortho_projection", &status));
        if (status != CL_SUCCESS || !kernel.get())
        {
            if (errorMsg)
                *errorMsg = clError("创建正射 kernel", status);
            return false;
        }
        BufferHandle dem_elevation;
        BufferHandle dem_valid;
        BufferHandle camera_values;
        BufferHandle camera_metadata;
        BufferHandle images;
        BufferHandle masks;
        BufferHandle output_image;
        BufferHandle surface_mask;
        BufferHandle coverage_mask;
        BufferHandle contributed_frames;
        const std::size_t pixel_count =
            static_cast<std::size_t>(input.outputWidth) * static_cast<std::size_t>(input.outputHeight);
        output->imageBgr.assign(pixel_count * 3, 0);
        output->surfaceMask.assign(pixel_count, 0);
        output->coverageMask.assign(pixel_count, 0);
        output->contributedFrames.assign(static_cast<std::size_t>(input.frameCount), 0);
        if (!createInputBuffer(runtime->context.get(), input.demElevation, &dem_elevation, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.demValid, &dem_valid, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.cameraValues, &camera_values, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.cameraMetadata, &camera_metadata, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.imageData, &images, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.maskData, &masks, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->imageBgr, &output_image, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->surfaceMask, &surface_mask, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->coverageMask, &coverage_mask, errorMsg) ||
            !createOutputBuffer(
                runtime->context.get(), &output->contributedFrames, &contributed_frames, errorMsg, true))
        {
            return false;
        }
        const cl_mem dem_elevation_mem = dem_elevation.get();
        const cl_mem dem_valid_mem = dem_valid.get();
        const cl_mem camera_values_mem = camera_values.get();
        const cl_mem camera_metadata_mem = camera_metadata.get();
        const cl_mem images_mem = images.get();
        const cl_mem masks_mem = masks.get();
        const cl_mem output_image_mem = output_image.get();
        const cl_mem surface_mask_mem = surface_mask.get();
        const cl_mem coverage_mask_mem = coverage_mask.get();
        const cl_mem contributed_frames_mem = contributed_frames.get();
        cl_uint argument = 0;
        if (!setKernelArgument(kernel.get(), argument++, dem_elevation_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, dem_valid_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demWidth, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demHeight, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demMinX, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demMinY, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demStepX, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.demStepY, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputWidth, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputHeight, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputMinEdgeX, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputMinEdgeY, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputStepX, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.outputStepY, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.elevationOffset, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, camera_values_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, camera_metadata_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.frameCount, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, images_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, masks_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.blendMode, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_image_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, surface_mask_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, coverage_mask_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, contributed_frames_mem, errorMsg))
        {
            return false;
        }
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{(static_cast<std::size_t>(input.outputWidth) + local[0] - 1) / local[0] * local[0],
                                    (static_cast<std::size_t>(input.outputHeight) + local[1] - 1) / local[1] *
                                        local[1]};
        status = clEnqueueNDRangeKernel(queue.get(), kernel.get(), 2, nullptr, global, local, 0, nullptr, nullptr);
        if (status != CL_SUCCESS)
        {
            if (errorMsg)
                *errorMsg = clError("提交正射 kernel", status);
            return false;
        }
        return readBuffer(queue.get(), output_image, &output->imageBgr, errorMsg) &&
               readBuffer(queue.get(), surface_mask, &output->surfaceMask, errorMsg) &&
               readBuffer(queue.get(), coverage_mask, &output->coverageMask, errorMsg) &&
               readBuffer(queue.get(), contributed_frames, &output->contributedFrames, errorMsg);
    }

    bool runTerrainOpenClDemMosaic(const PackedDemMosaic& input,
                                   int deviceIndex,
                                   PackedDemMosaicResult* output,
                                   std::string* errorMsg)
    {
        if (!output)
        {
            if (errorMsg)
                *errorMsg = "terrain OpenCL DEM mosaic 输出对象为空";
            return false;
        }
        TerrainDeviceInfo info;
        cl_device_id device = nullptr;
        if (!selectDevice(deviceIndex, false, &device, &info))
        {
            if (errorMsg)
                *errorMsg = info.error;
            return false;
        }
        std::shared_ptr<CachedRuntime> runtime;
        QueueHandle queue;
        if (!createInvocationRuntime(device, false, &runtime, &queue, errorMsg))
        {
            return false;
        }
        cl_int status = CL_SUCCESS;
        KernelHandle kernel(clCreateKernel(runtime->program.get(), "dem_mosaic", &status));
        if (status != CL_SUCCESS || !kernel.get())
        {
            if (errorMsg)
                *errorMsg = clError("创建 DEM mosaic kernel", status);
            return false;
        }
        BufferHandle input_elevation;
        BufferHandle input_valid;
        BufferHandle input_confidence;
        BufferHandle input_error;
        BufferHandle output_elevation;
        BufferHandle output_valid;
        BufferHandle output_count;
        BufferHandle output_confidence;
        BufferHandle output_error;
        const std::size_t pixel_count = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
        output->elevation.assign(pixel_count, 0.0f);
        output->valid.assign(pixel_count, 0);
        output->pointCount.assign(pixel_count, 0);
        output->confidence.assign(pixel_count, 0.0f);
        output->triangulationError.assign(pixel_count, 0.0f);
        if (!createInputBuffer(runtime->context.get(), input.elevation, &input_elevation, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.valid, &input_valid, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.confidence, &input_confidence, errorMsg) ||
            !createInputBuffer(runtime->context.get(), input.triangulationError, &input_error, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->elevation, &output_elevation, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->valid, &output_valid, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->pointCount, &output_count, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->confidence, &output_confidence, errorMsg) ||
            !createOutputBuffer(runtime->context.get(), &output->triangulationError, &output_error, errorMsg))
        {
            return false;
        }
        const cl_mem input_elevation_mem = input_elevation.get();
        const cl_mem input_valid_mem = input_valid.get();
        const cl_mem input_confidence_mem = input_confidence.get();
        const cl_mem input_error_mem = input_error.get();
        const cl_mem output_elevation_mem = output_elevation.get();
        const cl_mem output_valid_mem = output_valid.get();
        const cl_mem output_count_mem = output_count.get();
        const cl_mem output_confidence_mem = output_confidence.get();
        const cl_mem output_error_mem = output_error.get();
        cl_uint argument = 0;
        const int pixel_count_int = static_cast<int>(pixel_count);
        if (!setKernelArgument(kernel.get(), argument++, input_elevation_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input_valid_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input_confidence_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input_error_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.tileCount, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, pixel_count_int, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, input.blendMode, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_elevation_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_valid_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_count_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_confidence_mem, errorMsg) ||
            !setKernelArgument(kernel.get(), argument++, output_error_mem, errorMsg))
        {
            return false;
        }
        constexpr std::size_t local = 256;
        const std::size_t global = (pixel_count + local - 1) / local * local;
        status = clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (status != CL_SUCCESS)
        {
            if (errorMsg)
                *errorMsg = clError("提交 DEM mosaic kernel", status);
            return false;
        }
        return readBuffer(queue.get(), output_elevation, &output->elevation, errorMsg) &&
               readBuffer(queue.get(), output_valid, &output->valid, errorMsg) &&
               readBuffer(queue.get(), output_count, &output->pointCount, errorMsg) &&
               readBuffer(queue.get(), output_confidence, &output->confidence, errorMsg) &&
               readBuffer(queue.get(), output_error, &output->triangulationError, errorMsg);
    }

} // namespace xjw::terrain_internal
