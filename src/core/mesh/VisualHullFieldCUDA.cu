#include "VisualHullFieldBackend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw::mesh::detail
{
    namespace
    {

        template <typename T> class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;
            DeviceBuffer(const DeviceBuffer&) = delete;
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;

            ~DeviceBuffer()
            {
                if (_data)
                {
                    cudaFree(_data);
                }
            }

            bool allocate(std::size_t count)
            {
                return cudaMalloc(reinterpret_cast<void**>(&_data), std::max<std::size_t>(count, 1) * sizeof(T)) ==
                       cudaSuccess;
            }

            bool upload(const std::vector<T>& values)
            {
                return values.empty() ||
                       cudaMemcpy(_data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) ==
                           cudaSuccess;
            }

            T* data() noexcept
            {
                return _data;
            }
            const T* data() const noexcept
            {
                return _data;
            }

        private:
            T* _data = nullptr;
        };

        std::string cudaError(const char* operation, cudaError_t error)
        {
            return std::string(operation) + ": " + cudaGetErrorString(error);
        }

        __device__ bool projectPoint(const float* parameters,
                                     float world_x,
                                     float world_y,
                                     float world_z,
                                     float* pixel_x,
                                     float* pixel_y,
                                     float* positive_depth)
        {
            const float x = world_x - parameters[9];
            const float y = world_y - parameters[10];
            const float z = world_z - parameters[11];
            const float camera_x = parameters[0] * x + parameters[3] * y + parameters[6] * z;
            const float camera_y = parameters[1] * x + parameters[4] * y + parameters[7] * z;
            const float camera_z = parameters[2] * x + parameters[5] * y + parameters[8] * z;
            *positive_depth = parameters[23] * camera_z;
            if (!isfinite(*positive_depth) || !(*positive_depth > 1.0e-9f) || fabsf(camera_z) < 1.0e-9f)
            {
                return false;
            }

            const float normalized_x = camera_x / camera_z;
            const float normalized_y = camera_y / camera_z;
            const float radius_squared = normalized_x * normalized_x + normalized_y * normalized_y;
            const float radial = 1.0f + parameters[16] * radius_squared +
                                 parameters[17] * radius_squared * radius_squared +
                                 parameters[18] * radius_squared * radius_squared * radius_squared;
            const float distorted_x = normalized_x * radial + 2.0f * parameters[19] * normalized_x * normalized_y +
                                      parameters[20] * (radius_squared + 2.0f * normalized_x * normalized_x);
            const float distorted_y = normalized_y * radial +
                                      parameters[19] * (radius_squared + 2.0f * normalized_y * normalized_y) +
                                      2.0f * parameters[20] * normalized_x * normalized_y;
            *pixel_x = parameters[21] * parameters[12] * distorted_x + parameters[14];
            *pixel_y = parameters[22] * parameters[13] * distorted_y + parameters[15];
            return isfinite(*pixel_x) && isfinite(*pixel_y);
        }

        __device__ bool
        bilinearSample(const float* samples, int offset, int width, int height, float x, float y, float* value)
        {
            if (x < 0.0f || y < 0.0f || x > static_cast<float>(width - 1) || y > static_cast<float>(height - 1))
            {
                return false;
            }
            const int x0 = static_cast<int>(floorf(x));
            const int y0 = static_cast<int>(floorf(y));
            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);
            const float top = samples[offset + y0 * width + x0] * (1.0f - tx) + samples[offset + y0 * width + x1] * tx;
            const float bottom =
                samples[offset + y1 * width + x0] * (1.0f - tx) + samples[offset + y1 * width + x1] * tx;
            *value = top * (1.0f - ty) + bottom * ty;
            return isfinite(*value);
        }

        __device__ bool depthViolatesFreeSpace(const float* depth_samples,
                                               const std::int32_t* metadata,
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
                return false;
            }
            const int column = static_cast<int>(roundf(pixel_x));
            const int row = static_cast<int>(roundf(pixel_y));
            if (column < 0 || row < 0 || column >= width || row >= height)
            {
                return false;
            }
            const float measured_depth = depth_samples[offset + row * width + column];
            if (!isfinite(measured_depth) || !(measured_depth > 0.0f))
            {
                return false;
            }
            const float tolerance = fmaxf(1.0e-6f, measured_depth * relative_tolerance);
            return positive_depth < measured_depth - tolerance;
        }

        __device__ float evaluateBinary(float world_x,
                                        float world_y,
                                        float world_z,
                                        const float* camera_parameters,
                                        const std::int32_t* view_metadata,
                                        const float* silhouette_samples,
                                        const float* depth_samples,
                                        int view_count,
                                        int minimum_visible_views,
                                        int allowed_silhouette_violations,
                                        bool enable_depth_carving,
                                        int minimum_depth_violations,
                                        float relative_depth_tolerance)
        {
            int visible_views = 0;
            int silhouette_violations = 0;
            int free_space_violations = 0;
            for (int view = 0; view < view_count; ++view)
            {
                const float* parameters = camera_parameters + view * kVisualHullCameraParameterStride;
                const std::int32_t* metadata = view_metadata + view * kVisualHullViewMetadataStride;
                float pixel_x = 0.0f;
                float pixel_y = 0.0f;
                float positive_depth = 0.0f;
                if (!projectPoint(parameters, world_x, world_y, world_z, &pixel_x, &pixel_y, &positive_depth))
                {
                    continue;
                }
                const int column = static_cast<int>(roundf(pixel_x));
                const int row = static_cast<int>(roundf(pixel_y));
                const int width = metadata[1];
                const int height = metadata[2];
                if (column < 0 || row < 0 || column >= width || row >= height)
                {
                    continue;
                }
                ++visible_views;
                if (!(silhouette_samples[metadata[0] + row * width + column] > 0.5f))
                {
                    ++silhouette_violations;
                    if (silhouette_violations > allowed_silhouette_violations)
                    {
                        return 1.0f;
                    }
                    continue;
                }
                if (enable_depth_carving &&
                    depthViolatesFreeSpace(
                        depth_samples, metadata, pixel_x, pixel_y, positive_depth, relative_depth_tolerance))
                {
                    ++free_space_violations;
                    if (free_space_violations >= minimum_depth_violations)
                    {
                        return 1.0f;
                    }
                }
            }
            return visible_views >= minimum_visible_views && silhouette_violations <= allowed_silhouette_violations
                       ? -1.0f
                       : 1.0f;
        }

        __device__ void retainSmallestMargin(float margin, int capacity, float* smallest, int* retained)
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

        __device__ float evaluateContinuous(float world_x,
                                            float world_y,
                                            float world_z,
                                            const float* camera_parameters,
                                            const std::int32_t* view_metadata,
                                            const float* silhouette_samples,
                                            const float* depth_samples,
                                            int view_count,
                                            int minimum_visible_views,
                                            int allowed_silhouette_violations,
                                            bool enable_depth_carving,
                                            int minimum_depth_violations,
                                            float relative_depth_tolerance)
        {
            float smallest[kVisualHullMaximumGpuSilhouetteViolations + 1];
            const int clamped_allowed = max(0, allowed_silhouette_violations);
            const int capacity = min(clamped_allowed + 1, kVisualHullMaximumGpuSilhouetteViolations + 1);
            int retained = 0;
            int visible_views = 0;
            int free_space_violations = 0;
            for (int view = 0; view < view_count; ++view)
            {
                const float* parameters = camera_parameters + view * kVisualHullCameraParameterStride;
                const std::int32_t* metadata = view_metadata + view * kVisualHullViewMetadataStride;
                float pixel_x = 0.0f;
                float pixel_y = 0.0f;
                float positive_depth = 0.0f;
                if (!projectPoint(parameters, world_x, world_y, world_z, &pixel_x, &pixel_y, &positive_depth))
                {
                    continue;
                }
                float signed_pixel_distance = 0.0f;
                if (!bilinearSample(silhouette_samples,
                                    metadata[0],
                                    metadata[1],
                                    metadata[2],
                                    pixel_x,
                                    pixel_y,
                                    &signed_pixel_distance) ||
                    !(parameters[24] > 1.0e-9f))
                {
                    continue;
                }
                const float margin = signed_pixel_distance * positive_depth / parameters[24];
                retainSmallestMargin(margin, capacity, smallest, &retained);
                ++visible_views;
                if (enable_depth_carving &&
                    depthViolatesFreeSpace(
                        depth_samples, metadata, pixel_x, pixel_y, positive_depth, relative_depth_tolerance))
                {
                    ++free_space_violations;
                }
            }
            if (visible_views == 0 || visible_views < minimum_visible_views ||
                (enable_depth_carving && free_space_violations >= minimum_depth_violations))
            {
                return 1.0f;
            }
            const int target = min(clamped_allowed, visible_views - 1);
            const float support_margin = smallest[target];
            return isfinite(support_margin) ? -support_margin : 1.0f;
        }

        template <bool ContinuousField>
        __global__ void evaluateVisualHullFieldKernel(const float* camera_parameters,
                                                      const std::int32_t* view_metadata,
                                                      const float* silhouette_samples,
                                                      const float* depth_samples,
                                                      const float* grid_coordinates,
                                                      float* field,
                                                      std::size_t slab_sample_count,
                                                      int view_count,
                                                      int minimum_visible_views,
                                                      int allowed_silhouette_violations,
                                                      int enable_depth_carving,
                                                      int minimum_depth_violations,
                                                      float relative_depth_tolerance,
                                                      int size_x,
                                                      int size_y,
                                                      int size_z,
                                                      int slab_start_z,
                                                      int close_boundary)
        {
            const std::size_t offset = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
            if (offset >= slab_sample_count)
            {
                return;
            }
            const int x = static_cast<int>(offset % size_x);
            const std::size_t yz = offset / size_x;
            const int y = static_cast<int>(yz % size_y);
            const int z = slab_start_z + static_cast<int>(yz / size_y);
            if (close_boundary && (x == 0 || y == 0 || z == 0 || x == size_x - 1 || y == size_y - 1 || z == size_z - 1))
            {
                field[offset] = 1.0f;
                return;
            }
            const float world_x = grid_coordinates[x];
            const float world_y = grid_coordinates[size_x + y];
            const float world_z = grid_coordinates[size_x + size_y + z];
            field[offset] = ContinuousField ? evaluateContinuous(world_x,
                                                                 world_y,
                                                                 world_z,
                                                                 camera_parameters,
                                                                 view_metadata,
                                                                 silhouette_samples,
                                                                 depth_samples,
                                                                 view_count,
                                                                 minimum_visible_views,
                                                                 allowed_silhouette_violations,
                                                                 enable_depth_carving != 0,
                                                                 minimum_depth_violations,
                                                                 relative_depth_tolerance)
                                            : evaluateBinary(world_x,
                                                             world_y,
                                                             world_z,
                                                             camera_parameters,
                                                             view_metadata,
                                                             silhouette_samples,
                                                             depth_samples,
                                                             view_count,
                                                             minimum_visible_views,
                                                             allowed_silhouette_violations,
                                                             enable_depth_carving != 0,
                                                             minimum_depth_violations,
                                                             relative_depth_tolerance);
        }

        template <bool ContinuousField>
        cudaError_t launchVisualHullFieldKernel(const VisualHullFieldDeviceInput& input,
                                                const float* cameraParameters,
                                                const std::int32_t* viewMetadata,
                                                const float* silhouetteSamples,
                                                const float* depthSamples,
                                                const float* gridCoordinates,
                                                float* output,
                                                int slabStartZ,
                                                std::size_t slabSampleCount)
        {
            constexpr int threads_per_block = 256;
            const int block_count = static_cast<int>((slabSampleCount + threads_per_block - 1) / threads_per_block);
            evaluateVisualHullFieldKernel<ContinuousField>
                <<<block_count, threads_per_block>>>(cameraParameters,
                                                     viewMetadata,
                                                     silhouetteSamples,
                                                     depthSamples,
                                                     gridCoordinates,
                                                     output,
                                                     slabSampleCount,
                                                     input.viewCount,
                                                     input.minimumVisibleViews,
                                                     input.allowedSilhouetteViolations,
                                                     input.enableDepthFreeSpaceCarving ? 1 : 0,
                                                     input.minimumDepthFreeSpaceViolations,
                                                     input.relativeDepthTolerance,
                                                     input.grid.sampleSize(0),
                                                     input.grid.sampleSize(1),
                                                     input.grid.sampleSize(2),
                                                     slabStartZ,
                                                     input.closeVolumeBoundary ? 1 : 0);
            return cudaGetLastError();
        }

    } // namespace

    bool cudaVisualHullFieldAvailable(int deviceIndex) noexcept
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        return device_count > 0 && deviceIndex >= -1 && (deviceIndex < 0 || deviceIndex < device_count);
    }

    bool evaluateVisualHullFieldCuda(const VisualHullFieldDeviceInput& input,
                                     int deviceIndex,
                                     std::vector<float>* field,
                                     int* actualDeviceIndex,
                                     std::string* errorMessage)
    {
        const std::size_t expected_coordinate_count = static_cast<std::size_t>(input.grid.sampleSize(0)) +
                                                      static_cast<std::size_t>(input.grid.sampleSize(1)) +
                                                      static_cast<std::size_t>(input.grid.sampleSize(2));
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = -1;
        }
        if (!field || !input.grid.isValid() || input.gridCoordinates.size() != expected_coordinate_count ||
            !cudaVisualHullFieldAvailable(deviceIndex))
        {
            if (errorMessage)
            {
                *errorMessage = "CUDA visual hull input or device is invalid";
            }
            return false;
        }
        int selected_device = deviceIndex;
        if (selected_device < 0)
        {
            const cudaError_t get_device_error = cudaGetDevice(&selected_device);
            if (get_device_error != cudaSuccess)
            {
                if (errorMessage)
                {
                    *errorMessage = cudaError("cudaGetDevice", get_device_error);
                }
                return false;
            }
        }
        const cudaError_t set_device_error = cudaSetDevice(selected_device);
        if (set_device_error != cudaSuccess)
        {
            if (errorMessage)
            {
                *errorMessage = cudaError("cudaSetDevice", set_device_error);
            }
            return false;
        }
        if (input.isCancelled && input.isCancelled())
        {
            if (errorMessage)
            {
                *errorMessage = "CUDA visual hull evaluation cancelled";
            }
            return false;
        }

        DeviceBuffer<float> camera_parameters;
        DeviceBuffer<std::int32_t> view_metadata;
        DeviceBuffer<float> silhouette_samples;
        DeviceBuffer<float> depth_samples;
        DeviceBuffer<float> grid_coordinates;
        DeviceBuffer<float> output;
        const std::size_t sample_count = input.grid.sampleCount();
        const std::size_t layer_sample_count =
            static_cast<std::size_t>(input.grid.sampleSize(0)) * static_cast<std::size_t>(input.grid.sampleSize(1));
        const int maximum_slab_depth = std::min(64, input.grid.sampleSize(2));
        const int slab_depth = std::clamp(input.gpuSlabDepth, 1, maximum_slab_depth);
        const std::size_t maximum_slab_sample_count = layer_sample_count * static_cast<std::size_t>(slab_depth);
        if (!camera_parameters.allocate(input.cameraParameters.size()) ||
            !view_metadata.allocate(input.viewMetadata.size()) ||
            !silhouette_samples.allocate(input.silhouetteSamples.size()) ||
            !depth_samples.allocate(input.depthSamples.size()) ||
            !grid_coordinates.allocate(input.gridCoordinates.size()) || !output.allocate(maximum_slab_sample_count) ||
            !camera_parameters.upload(input.cameraParameters) || !view_metadata.upload(input.viewMetadata) ||
            !silhouette_samples.upload(input.silhouetteSamples) || !depth_samples.upload(input.depthSamples) ||
            !grid_coordinates.upload(input.gridCoordinates))
        {
            if (errorMessage)
            {
                *errorMessage = cudaError("CUDA visual hull buffer allocation/upload", cudaGetLastError());
            }
            return false;
        }

        field->resize(sample_count);
        for (int slab_start_z = 0; slab_start_z < input.grid.sampleSize(2); slab_start_z += slab_depth)
        {
            if (input.isCancelled && input.isCancelled())
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "CUDA visual hull evaluation cancelled";
                }
                return false;
            }

            const int current_slab_depth = std::min(slab_depth, input.grid.sampleSize(2) - slab_start_z);
            const std::size_t slab_sample_count = layer_sample_count * static_cast<std::size_t>(current_slab_depth);
            cudaError_t error = input.continuousSilhouetteField
                                    ? launchVisualHullFieldKernel<true>(input,
                                                                        camera_parameters.data(),
                                                                        view_metadata.data(),
                                                                        silhouette_samples.data(),
                                                                        depth_samples.data(),
                                                                        grid_coordinates.data(),
                                                                        output.data(),
                                                                        slab_start_z,
                                                                        slab_sample_count)
                                    : launchVisualHullFieldKernel<false>(input,
                                                                         camera_parameters.data(),
                                                                         view_metadata.data(),
                                                                         silhouette_samples.data(),
                                                                         depth_samples.data(),
                                                                         grid_coordinates.data(),
                                                                         output.data(),
                                                                         slab_start_z,
                                                                         slab_sample_count);
            if (error == cudaSuccess)
            {
                error = cudaDeviceSynchronize();
            }
            if (error != cudaSuccess)
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = cudaError("CUDA visual hull slab kernel", error);
                }
                return false;
            }

            error = cudaMemcpy(field->data() + layer_sample_count * static_cast<std::size_t>(slab_start_z),
                               output.data(),
                               slab_sample_count * sizeof(float),
                               cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = cudaError("CUDA visual hull slab download", error);
                }
                return false;
            }
            if (input.isCancelled && input.isCancelled())
            {
                field->clear();
                if (errorMessage)
                {
                    *errorMessage = "CUDA visual hull evaluation cancelled";
                }
                return false;
            }
        }
        if (actualDeviceIndex)
        {
            *actualDeviceIndex = selected_device;
        }
        return true;
    }

} // namespace xjw::mesh::detail
