#define CL_TARGET_OPENCL_VERSION 120

#include "VisualHullFieldBackend.h"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace xjw::mesh::detail
{
    namespace
    {

        constexpr const char* kVisualHullOpenClSource = R"CLC(
#define CAMERA_STRIDE 25
#define METADATA_STRIDE 6
#define MAX_RETAINED 64

int project_point(
    __global const float *p,
    float wx,
    float wy,
    float wz,
    __private float *pixel_x,
    __private float *pixel_y,
    __private float *positive_depth)
{
    const float x = wx - p[9];
    const float y = wy - p[10];
    const float z = wz - p[11];
    const float camera_x = p[0] * x + p[3] * y + p[6] * z;
    const float camera_y = p[1] * x + p[4] * y + p[7] * z;
    const float camera_z = p[2] * x + p[5] * y + p[8] * z;
    *positive_depth = p[23] * camera_z;
    if (!isfinite(*positive_depth) || !(*positive_depth > 1.0e-9f) ||
        fabs(camera_z) < 1.0e-9f)
    {
        return 0;
    }

    const float nx = camera_x / camera_z;
    const float ny = camera_y / camera_z;
    const float r2 = nx * nx + ny * ny;
    const float radial = 1.0f + p[16] * r2 + p[17] * r2 * r2 +
                         p[18] * r2 * r2 * r2;
    const float dx = nx * radial + 2.0f * p[19] * nx * ny +
                     p[20] * (r2 + 2.0f * nx * nx);
    const float dy = ny * radial + p[19] * (r2 + 2.0f * ny * ny) +
                     2.0f * p[20] * nx * ny;
    *pixel_x = p[21] * p[12] * dx + p[14];
    *pixel_y = p[22] * p[13] * dy + p[15];
    return isfinite(*pixel_x) && isfinite(*pixel_y);
}

int bilinear_sample(
    __global const float *samples,
    int offset,
    int width,
    int height,
    float x,
    float y,
    __private float *value)
{
    if (x < 0.0f || y < 0.0f ||
        x > (float)(width - 1) || y > (float)(height - 1))
    {
        return 0;
    }
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;
    const float top = samples[offset + y0 * width + x0] * (1.0f - tx) +
                      samples[offset + y0 * width + x1] * tx;
    const float bottom = samples[offset + y1 * width + x0] * (1.0f - tx) +
                         samples[offset + y1 * width + x1] * tx;
    *value = top * (1.0f - ty) + bottom * ty;
    return isfinite(*value);
}

int depth_violates(
    __global const float *depth,
    __global const int *metadata,
    float pixel_x,
    float pixel_y,
    float positive_depth,
    float relative_tolerance)
{
    const int offset = metadata[3];
    const int width = metadata[4];
    const int height = metadata[5];
    if (offset < 0)
    {
        return 0;
    }
    const int column = convert_int(round(pixel_x));
    const int row = convert_int(round(pixel_y));
    if (column < 0 || row < 0 || column >= width || row >= height)
    {
        return 0;
    }
    const float measured = depth[offset + row * width + column];
    if (!isfinite(measured) || !(measured > 0.0f))
    {
        return 0;
    }
    const float tolerance = fmax(1.0e-6f, measured * relative_tolerance);
    return positive_depth < measured - tolerance;
}

float evaluate_binary(
    float wx,
    float wy,
    float wz,
    __global const float *cameras,
    __global const int *metadata,
    __global const float *silhouettes,
    __global const float *depth,
    int view_count,
    int minimum_visible,
    int allowed_violations,
    int enable_depth,
    int minimum_depth_violations,
    float relative_tolerance)
{
    int visible = 0;
    int violations = 0;
    int depth_violations = 0;
    for (int view = 0; view < view_count; ++view)
    {
        __global const float *p = cameras + view * CAMERA_STRIDE;
        __global const int *m = metadata + view * METADATA_STRIDE;
        float pixel_x = 0.0f;
        float pixel_y = 0.0f;
        float positive_depth = 0.0f;
        if (!project_point(
                p, wx, wy, wz, &pixel_x, &pixel_y, &positive_depth))
        {
            continue;
        }
        const int column = convert_int(round(pixel_x));
        const int row = convert_int(round(pixel_y));
        if (column < 0 || row < 0 || column >= m[1] || row >= m[2])
        {
            continue;
        }
        ++visible;
        if (!(silhouettes[m[0] + row * m[1] + column] > 0.5f))
        {
            ++violations;
            if (violations > allowed_violations)
            {
                return 1.0f;
            }
            continue;
        }
        if (enable_depth && depth_violates(
                depth,
                m,
                pixel_x,
                pixel_y,
                positive_depth,
                relative_tolerance))
        {
            ++depth_violations;
            if (depth_violations >= minimum_depth_violations)
            {
                return 1.0f;
            }
        }
    }
    return visible >= minimum_visible && violations <= allowed_violations
        ? -1.0f
        : 1.0f;
}

void retain_smallest(
    float margin,
    int capacity,
    __private float *smallest,
    __private int *retained)
{
    if (*retained < capacity)
    {
        int position = (*retained)++;
        smallest[position] = margin;
        while (position > 0 && smallest[position] < smallest[position - 1])
        {
            const float swap = smallest[position - 1];
            smallest[position - 1] = smallest[position];
            smallest[position] = swap;
            --position;
        }
        return;
    }
    if (!(margin < smallest[capacity - 1]))
    {
        return;
    }
    int position = capacity - 1;
    smallest[position] = margin;
    while (position > 0 && smallest[position] < smallest[position - 1])
    {
        const float swap = smallest[position - 1];
        smallest[position - 1] = smallest[position];
        smallest[position] = swap;
        --position;
    }
}

float evaluate_continuous(
    float wx,
    float wy,
    float wz,
    __global const float *cameras,
    __global const int *metadata,
    __global const float *silhouettes,
    __global const float *depth,
    int view_count,
    int minimum_visible,
    int allowed_violations,
    int enable_depth,
    int minimum_depth_violations,
    float relative_tolerance)
{
    float smallest[MAX_RETAINED];
    const int clamped_allowed = max(0, allowed_violations);
    const int capacity = min(clamped_allowed + 1, MAX_RETAINED);
    int retained = 0;
    int visible = 0;
    int depth_violations = 0;
    for (int view = 0; view < view_count; ++view)
    {
        __global const float *p = cameras + view * CAMERA_STRIDE;
        __global const int *m = metadata + view * METADATA_STRIDE;
        float pixel_x = 0.0f;
        float pixel_y = 0.0f;
        float positive_depth = 0.0f;
        if (!project_point(
                p, wx, wy, wz, &pixel_x, &pixel_y, &positive_depth))
        {
            continue;
        }
        float signed_distance = 0.0f;
        if (!bilinear_sample(
                silhouettes,
                m[0],
                m[1],
                m[2],
                pixel_x,
                pixel_y,
                &signed_distance) ||
            !(p[24] > 1.0e-9f))
        {
            continue;
        }
        const float margin = signed_distance * positive_depth / p[24];
        retain_smallest(margin, capacity, smallest, &retained);
        ++visible;
        if (enable_depth && depth_violates(
                depth,
                m,
                pixel_x,
                pixel_y,
                positive_depth,
                relative_tolerance))
        {
            ++depth_violations;
        }
    }
    if (visible == 0 || visible < minimum_visible ||
        (enable_depth && depth_violations >= minimum_depth_violations))
    {
        return 1.0f;
    }
    const int target = min(clamped_allowed, visible - 1);
    const float support_margin = smallest[target];
    return isfinite(support_margin) ? -support_margin : 1.0f;
}

__kernel void evaluate_visual_hull_field(
    __global const float *cameras,
    __global const int *metadata,
    __global const float *silhouettes,
    __global const float *depth,
    __global const float *grid_coordinates,
    __global float *field,
    ulong slab_sample_count,
    int view_count,
    int minimum_visible,
    int allowed_violations,
    int enable_depth,
    int minimum_depth_violations,
    float relative_tolerance,
    int size_x,
    int size_y,
    int size_z,
    int slab_start_z,
    int close_boundary)
{
    const ulong offset = (ulong)get_global_id(0);
    if (offset >= slab_sample_count)
    {
        return;
    }
    const int x = (int)(offset % (ulong)size_x);
    const ulong yz = offset / (ulong)size_x;
    const int y = (int)(yz % (ulong)size_y);
    const int z = slab_start_z + (int)(yz / (ulong)size_y);
    if (close_boundary &&
        (x == 0 || y == 0 || z == 0 ||
         x == size_x - 1 || y == size_y - 1 || z == size_z - 1))
    {
        field[offset] = 1.0f;
        return;
    }
    const float wx = grid_coordinates[x];
    const float wy = grid_coordinates[size_x + y];
    const float wz = grid_coordinates[size_x + size_y + z];
#if CONTINUOUS_FIELD
    field[offset] = evaluate_continuous(
        wx,
        wy,
        wz,
        cameras,
        metadata,
        silhouettes,
        depth,
        view_count,
        minimum_visible,
        allowed_violations,
        enable_depth,
        minimum_depth_violations,
        relative_tolerance);
#else
    field[offset] = evaluate_binary(
        wx,
        wy,
        wz,
        cameras,
        metadata,
        silhouettes,
        depth,
        view_count,
        minimum_visible,
        allowed_violations,
        enable_depth,
        minimum_depth_violations,
        relative_tolerance);
#endif
}
)CLC";

        struct OpenClDevice
        {
            cl_platform_id platform = nullptr;
            cl_device_id device = nullptr;
        };

        std::vector<OpenClDevice> enumerateGpuDevices()
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

            std::vector<OpenClDevice> result;
            for (cl_platform_id platform : platforms)
            {
                cl_uint device_count = 0;
                const cl_int count_error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
                if (count_error == CL_DEVICE_NOT_FOUND || device_count == 0)
                {
                    continue;
                }
                if (count_error != CL_SUCCESS)
                {
                    continue;
                }
                std::vector<cl_device_id> devices(device_count);
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr) != CL_SUCCESS)
                {
                    continue;
                }
                for (cl_device_id device : devices)
                {
                    cl_bool available = CL_FALSE;
                    cl_bool compiler_available = CL_FALSE;
                    if (clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr) ==
                            CL_SUCCESS &&
                        clGetDeviceInfo(device,
                                        CL_DEVICE_COMPILER_AVAILABLE,
                                        sizeof(compiler_available),
                                        &compiler_available,
                                        nullptr) == CL_SUCCESS &&
                        available == CL_TRUE && compiler_available == CL_TRUE)
                    {
                        result.push_back({platform, device});
                    }
                }
            }
            return result;
        }

        class OpenClCallResources
        {
        public:
            ~OpenClCallResources()
            {
                for (cl_mem buffer : buffers)
                {
                    if (buffer)
                    {
                        clReleaseMemObject(buffer);
                    }
                }
                if (kernel)
                {
                    clReleaseKernel(kernel);
                }
                if (queue)
                {
                    clReleaseCommandQueue(queue);
                }
            }

            cl_command_queue queue = nullptr;
            cl_kernel kernel = nullptr;
            std::vector<cl_mem> buffers;
        };

        class OpenClProgramResources
        {
        public:
            ~OpenClProgramResources()
            {
                if (program)
                {
                    clReleaseProgram(program);
                }
                if (context)
                {
                    clReleaseContext(context);
                }
            }

            cl_context context = nullptr;
            cl_program program = nullptr;
        };

        struct OpenClProgramKey
        {
            std::uintptr_t device = 0;
            bool continuous = false;

            bool operator<(const OpenClProgramKey& other) const noexcept
            {
                return device < other.device || (device == other.device && continuous < other.continuous);
            }
        };

        std::string buildLog(cl_program program, cl_device_id device)
        {
            std::size_t size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            if (size == 0)
            {
                return {};
            }
            std::string log(size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            while (!log.empty() && log.back() == '\0')
            {
                log.pop_back();
            }
            return log;
        }

        std::shared_ptr<OpenClProgramResources>
        cachedProgram(const OpenClDevice& selected, bool continuous, std::string* errorMessage)
        {
            static std::mutex cache_mutex;
            static std::map<OpenClProgramKey, std::shared_ptr<OpenClProgramResources>> cache;
            const OpenClProgramKey key{reinterpret_cast<std::uintptr_t>(selected.device), continuous};
            std::lock_guard<std::mutex> lock(cache_mutex);
            const auto existing = cache.find(key);
            if (existing != cache.end())
            {
                return existing->second;
            }

            auto resources = std::make_shared<OpenClProgramResources>();
            cl_int error = CL_SUCCESS;
            const cl_context_properties context_properties[] = {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selected.platform), 0};
            resources->context = clCreateContext(context_properties, 1, &selected.device, nullptr, nullptr, &error);
            if (error != CL_SUCCESS || !resources->context)
            {
                if (errorMessage)
                {
                    *errorMessage = "clCreateContext failed: " + std::to_string(error);
                }
                return {};
            }

            const char* source = kVisualHullOpenClSource;
            const std::size_t source_size = std::char_traits<char>::length(source);
            resources->program = clCreateProgramWithSource(resources->context, 1, &source, &source_size, &error);
            if (error != CL_SUCCESS || !resources->program)
            {
                if (errorMessage)
                {
                    *errorMessage = "clCreateProgramWithSource failed: " + std::to_string(error);
                }
                return {};
            }
            const std::string build_options =
                std::string("-cl-std=CL1.2 -DCONTINUOUS_FIELD=") + (continuous ? "1" : "0");
            error = clBuildProgram(resources->program, 1, &selected.device, build_options.c_str(), nullptr, nullptr);
            if (error != CL_SUCCESS)
            {
                if (errorMessage)
                {
                    *errorMessage =
                        "OpenCL visual hull kernel build failed: " + buildLog(resources->program, selected.device);
                }
                return {};
            }
            cache.emplace(key, resources);
            return resources;
        }

        template <typename T>
        cl_mem createInputBuffer(OpenClCallResources* resources,
                                 cl_context context,
                                 const std::vector<T>& values,
                                 cl_int* error)
        {
            const T dummy{};
            const void* source =
                values.empty() ? static_cast<const void*>(&dummy) : static_cast<const void*>(values.data());
            const std::size_t bytes = std::max<std::size_t>(values.size(), 1) * sizeof(T);
            cl_mem buffer = clCreateBuffer(
                context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(source), error);
            resources->buffers.push_back(buffer);
            return buffer;
        }

    } // namespace

    bool openClVisualHullFieldAvailable(int deviceIndex) noexcept
    {
        const std::vector<OpenClDevice> devices = enumerateGpuDevices();
        return !devices.empty() && deviceIndex >= -1 &&
               (deviceIndex < 0 || deviceIndex < static_cast<int>(devices.size()));
    }

    bool evaluateVisualHullFieldOpenCl(const VisualHullFieldDeviceInput& input,
                                       int deviceIndex,
                                       std::vector<float>* field,
                                       int* actualDeviceIndex,
                                       std::string* errorMessage)
    {
        static_assert(sizeof(std::int32_t) == sizeof(cl_int));
        const std::size_t expected_coordinate_count = static_cast<std::size_t>(input.grid.sampleSize(0)) +
                                                      static_cast<std::size_t>(input.grid.sampleSize(1)) +
                                                      static_cast<std::size_t>(input.grid.sampleSize(2));
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = -1;
        }
        const std::vector<OpenClDevice> devices = enumerateGpuDevices();
        const int selected_index = deviceIndex < 0 ? 0 : deviceIndex;
        if (!field || !input.grid.isValid() || input.gridCoordinates.size() != expected_coordinate_count ||
            selected_index < 0 || selected_index >= static_cast<int>(devices.size()))
        {
            if (errorMessage)
            {
                *errorMessage = "OpenCL visual hull input or device is invalid";
            }
            return false;
        }
        const OpenClDevice selected = devices[static_cast<std::size_t>(selected_index)];
        if (input.isCancelled && input.isCancelled())
        {
            if (errorMessage)
            {
                *errorMessage = "OpenCL visual hull evaluation cancelled";
            }
            return false;
        }
        const std::shared_ptr<OpenClProgramResources> program =
            cachedProgram(selected, input.continuousSilhouetteField, errorMessage);
        if (!program)
        {
            return false;
        }

        OpenClCallResources resources;
        cl_int error = CL_SUCCESS;
        resources.queue = clCreateCommandQueue(program->context, selected.device, 0, &error);
        if (error != CL_SUCCESS || !resources.queue)
        {
            if (errorMessage)
            {
                *errorMessage = "clCreateCommandQueue failed: " + std::to_string(error);
            }
            return false;
        }
        resources.kernel = clCreateKernel(program->program, "evaluate_visual_hull_field", &error);
        if (error != CL_SUCCESS || !resources.kernel)
        {
            if (errorMessage)
            {
                *errorMessage = "clCreateKernel failed: " + std::to_string(error);
            }
            return false;
        }
        cl_mem camera_buffer = createInputBuffer(&resources, program->context, input.cameraParameters, &error);
        cl_mem metadata_buffer = createInputBuffer(&resources, program->context, input.viewMetadata, &error);
        cl_mem silhouette_buffer = createInputBuffer(&resources, program->context, input.silhouetteSamples, &error);
        cl_mem depth_buffer = createInputBuffer(&resources, program->context, input.depthSamples, &error);
        cl_mem grid_coordinate_buffer = createInputBuffer(&resources, program->context, input.gridCoordinates, &error);
        const std::size_t sample_count = input.grid.sampleCount();
        const std::size_t layer_sample_count =
            static_cast<std::size_t>(input.grid.sampleSize(0)) * static_cast<std::size_t>(input.grid.sampleSize(1));
        const int maximum_slab_depth = std::min(64, input.grid.sampleSize(2));
        const int slab_depth = std::clamp(input.gpuSlabDepth, 1, maximum_slab_depth);
        const std::size_t maximum_slab_sample_count = layer_sample_count * static_cast<std::size_t>(slab_depth);
        cl_mem output_buffer = clCreateBuffer(
            program->context, CL_MEM_WRITE_ONLY, maximum_slab_sample_count * sizeof(float), nullptr, &error);
        resources.buffers.push_back(output_buffer);
        if (error != CL_SUCCESS || !camera_buffer || !metadata_buffer || !silhouette_buffer || !depth_buffer ||
            !grid_coordinate_buffer || !output_buffer)
        {
            if (errorMessage)
            {
                *errorMessage = "OpenCL visual hull buffer allocation failed: " + std::to_string(error);
            }
            return false;
        }

        const cl_int view_count = input.viewCount;
        const cl_int minimum_visible = input.minimumVisibleViews;
        const cl_int allowed = input.allowedSilhouetteViolations;
        const cl_int enable_depth = input.enableDepthFreeSpaceCarving ? 1 : 0;
        const cl_int minimum_depth = input.minimumDepthFreeSpaceViolations;
        const cl_float relative_tolerance = input.relativeDepthTolerance;
        const cl_int size_x = input.grid.sampleSize(0);
        const cl_int size_y = input.grid.sampleSize(1);
        const cl_int size_z = input.grid.sampleSize(2);
        const cl_int close_boundary = input.closeVolumeBoundary ? 1 : 0;
        field->resize(sample_count);
        for (int slab_start_z = 0; slab_start_z < size_z; slab_start_z += slab_depth)
        {
            if (input.isCancelled && input.isCancelled())
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "OpenCL visual hull evaluation cancelled";
                }
                return false;
            }

            const int current_slab_depth = std::min(slab_depth, size_z - slab_start_z);
            const std::size_t slab_sample_count = layer_sample_count * static_cast<std::size_t>(current_slab_depth);
            const cl_ulong cl_slab_sample_count = static_cast<cl_ulong>(slab_sample_count);
            const cl_int cl_slab_start_z = slab_start_z;
            int argument = 0;
            error = CL_SUCCESS;
#define SET_VISUAL_HULL_ARGUMENT(value) error |= clSetKernelArg(resources.kernel, argument++, sizeof(value), &(value))
            SET_VISUAL_HULL_ARGUMENT(camera_buffer);
            SET_VISUAL_HULL_ARGUMENT(metadata_buffer);
            SET_VISUAL_HULL_ARGUMENT(silhouette_buffer);
            SET_VISUAL_HULL_ARGUMENT(depth_buffer);
            SET_VISUAL_HULL_ARGUMENT(grid_coordinate_buffer);
            SET_VISUAL_HULL_ARGUMENT(output_buffer);
            SET_VISUAL_HULL_ARGUMENT(cl_slab_sample_count);
            SET_VISUAL_HULL_ARGUMENT(view_count);
            SET_VISUAL_HULL_ARGUMENT(minimum_visible);
            SET_VISUAL_HULL_ARGUMENT(allowed);
            SET_VISUAL_HULL_ARGUMENT(enable_depth);
            SET_VISUAL_HULL_ARGUMENT(minimum_depth);
            SET_VISUAL_HULL_ARGUMENT(relative_tolerance);
            SET_VISUAL_HULL_ARGUMENT(size_x);
            SET_VISUAL_HULL_ARGUMENT(size_y);
            SET_VISUAL_HULL_ARGUMENT(size_z);
            SET_VISUAL_HULL_ARGUMENT(cl_slab_start_z);
            SET_VISUAL_HULL_ARGUMENT(close_boundary);
#undef SET_VISUAL_HULL_ARGUMENT
            if (error != CL_SUCCESS)
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "OpenCL visual hull kernel argument failed: " + std::to_string(error);
                }
                return false;
            }

            const std::size_t global_size = slab_sample_count;
            error = clEnqueueNDRangeKernel(
                resources.queue, resources.kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
            if (error == CL_SUCCESS)
            {
                error = clEnqueueReadBuffer(resources.queue,
                                            output_buffer,
                                            CL_TRUE,
                                            0,
                                            slab_sample_count * sizeof(float),
                                            field->data() + layer_sample_count * static_cast<std::size_t>(slab_start_z),
                                            0,
                                            nullptr,
                                            nullptr);
            }
            if (error != CL_SUCCESS)
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "OpenCL visual hull slab execution failed: " + std::to_string(error);
                }
                return false;
            }
            if (input.isCancelled && input.isCancelled())
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "OpenCL visual hull evaluation cancelled";
                }
                return false;
            }
        }
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = selected_index;
        }
        return true;
    }

} // namespace xjw::mesh::detail
