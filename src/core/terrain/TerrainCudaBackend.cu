#include "TerrainGpuBackend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

namespace xjw::terrain_internal
{

    namespace
    {

        constexpr int kBlendMosaic = 0;
        constexpr int kBlendWeightedAverage = 1;
        constexpr int kBlendFirstValid = 2;

        std::string cudaErrorMessage(const char* operation, cudaError_t error)
        {
            std::ostringstream stream;
            stream << "terrain CUDA " << operation << " 失败: " << cudaGetErrorString(error);
            return stream.str();
        }

        template <typename T> class DeviceBuffer final
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

            bool allocate(std::size_t count, std::string* errorMsg)
            {
                const cudaError_t status =
                    cudaMalloc(reinterpret_cast<void**>(&_data), std::max<std::size_t>(count, 1) * sizeof(T));
                if (status != cudaSuccess)
                {
                    if (errorMsg)
                    {
                        *errorMsg = cudaErrorMessage("分配设备内存", status);
                    }
                    return false;
                }
                return true;
            }

            bool upload(const std::vector<T>& values, std::string* errorMsg)
            {
                if (!allocate(values.size(), errorMsg))
                {
                    return false;
                }
                if (values.empty())
                {
                    return true;
                }
                const cudaError_t status =
                    cudaMemcpy(_data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
                if (status != cudaSuccess)
                {
                    if (errorMsg)
                    {
                        *errorMsg = cudaErrorMessage("上传输入", status);
                    }
                    return false;
                }
                return true;
            }

            bool download(std::vector<T>* values, std::string* errorMsg) const
            {
                if (values->empty())
                {
                    return true;
                }
                const cudaError_t status =
                    cudaMemcpy(values->data(), _data, values->size() * sizeof(T), cudaMemcpyDeviceToHost);
                if (status != cudaSuccess)
                {
                    if (errorMsg)
                    {
                        *errorMsg = cudaErrorMessage("下载输出", status);
                    }
                    return false;
                }
                return true;
            }

            T* data()
            {
                return _data;
            }
            const T* data() const
            {
                return _data;
            }

        private:
            T* _data = nullptr;
        };

        class CudaDeviceGuard final
        {
        public:
            bool select(int deviceIndex, std::string* errorMsg)
            {
                if (cudaGetDevice(&_previous) != cudaSuccess)
                {
                    _previous = -1;
                    cudaGetLastError();
                }
                const cudaError_t status = cudaSetDevice(deviceIndex);
                if (status != cudaSuccess)
                {
                    if (errorMsg)
                    {
                        *errorMsg = cudaErrorMessage("选择设备", status);
                    }
                    return false;
                }
                _selected = true;
                return true;
            }

            ~CudaDeviceGuard()
            {
                if (_selected && _previous >= 0)
                {
                    cudaSetDevice(_previous);
                }
            }

        private:
            int _previous = -1;
            bool _selected = false;
        };

        __device__ bool sampleDem(const float* elevation,
                                  const std::uint8_t* valid,
                                  int width,
                                  int height,
                                  double minX,
                                  double minY,
                                  double stepX,
                                  double stepY,
                                  double worldX,
                                  double worldY,
                                  double* sample)
        {
            const double grid_x = (worldX - minX) / stepX;
            const double grid_y = (worldY - minY) / stepY;
            if (grid_x < -0.5 || grid_y < -0.5 || grid_x > static_cast<double>(width) - 0.5 ||
                grid_y > static_cast<double>(height) - 0.5)
            {
                return false;
            }

            const int nearest_x = min(max(static_cast<int>(floor(grid_x + 0.5)), 0), width - 1);
            const int nearest_y = min(max(static_cast<int>(floor(grid_y + 0.5)), 0), height - 1);
            const int x0 = min(max(static_cast<int>(floor(grid_x)), 0), width - 1);
            const int y0 = min(max(static_cast<int>(floor(grid_y)), 0), height - 1);
            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);
            const int index00 = y0 * width + x0;
            const int index10 = y0 * width + x1;
            const int index01 = y1 * width + x0;
            const int index11 = y1 * width + x1;
            if (valid[index00] && valid[index10] && valid[index01] && valid[index11])
            {
                const double fx = min(max(grid_x - static_cast<double>(x0), 0.0), 1.0);
                const double fy = min(max(grid_y - static_cast<double>(y0), 0.0), 1.0);
                *sample = (1.0 - fx) * (1.0 - fy) * elevation[index00] + fx * (1.0 - fy) * elevation[index10] +
                          (1.0 - fx) * fy * elevation[index01] + fx * fy * elevation[index11];
                return isfinite(*sample);
            }
            const int nearest = nearest_y * width + nearest_x;
            if (!valid[nearest])
            {
                return false;
            }
            *sample = elevation[nearest];
            return isfinite(*sample);
        }

        __device__ bool sampleCamera(const double* camera,
                                     const int* metadata,
                                     const std::uint8_t* images,
                                     const std::uint8_t* masks,
                                     double worldX,
                                     double worldY,
                                     double worldZ,
                                     float color[3],
                                     double* weight)
        {
            const double relative_x = worldX - camera[9];
            const double relative_y = worldY - camera[10];
            const double relative_z = worldZ - camera[11];
            const double camera_x = camera[0] * relative_x + camera[3] * relative_y + camera[6] * relative_z;
            const double camera_y = camera[1] * relative_x + camera[4] * relative_y + camera[7] * relative_z;
            const double camera_z = camera[2] * relative_x + camera[5] * relative_y + camera[8] * relative_z;
            const double positive_depth = metadata[6] ? -camera_z : camera_z;
            if (!(positive_depth > 1.0e-9) || !isfinite(positive_depth))
            {
                return false;
            }

            const double x = camera_x / camera_z;
            const double y = camera_y / camera_z;
            const double r2 = x * x + y * y;
            const double radial = 1.0 + camera[16] * r2 + camera[17] * r2 * r2 + camera[18] * r2 * r2 * r2;
            const double distorted_x = x * radial + 2.0 * camera[19] * x * y + camera[20] * (r2 + 2.0 * x * x);
            const double distorted_y = y * radial + camera[19] * (r2 + 2.0 * y * y) + 2.0 * camera[20] * x * y;
            const double u = static_cast<double>(metadata[4]) * camera[12] * distorted_x + camera[14];
            const double v = static_cast<double>(metadata[5]) * camera[13] * distorted_y + camera[15];
            const int width = metadata[0];
            const int height = metadata[1];
            if (u < 0.0 || v < 0.0 || u >= static_cast<double>(width - 1) || v >= static_cast<double>(height - 1))
            {
                return false;
            }

            if (metadata[3] >= 0)
            {
                const int mask_x = min(max(static_cast<int>(floor(u + 0.5)), 0), width - 1);
                const int mask_y = min(max(static_cast<int>(floor(v + 0.5)), 0), height - 1);
                if (masks[metadata[3] + mask_y * width + mask_x])
                {
                    return false;
                }
            }

            const int x0 = min(max(static_cast<int>(floor(u)), 0), width - 1);
            const int y0 = min(max(static_cast<int>(floor(v)), 0), height - 1);
            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);
            const float fx = static_cast<float>(u - static_cast<double>(x0));
            const float fy = static_cast<float>(v - static_cast<double>(y0));
            const int image_offset = metadata[2];
            for (int channel = 0; channel < 3; ++channel)
            {
                const float c00 = images[image_offset + (y0 * width + x0) * 3 + channel];
                const float c10 = images[image_offset + (y0 * width + x1) * 3 + channel];
                const float c01 = images[image_offset + (y1 * width + x0) * 3 + channel];
                const float c11 = images[image_offset + (y1 * width + x1) * 3 + channel];
                color[channel] = ((1.0f - fx) * (1.0f - fy) * c00 + fx * (1.0f - fy) * c10 + (1.0f - fx) * fy * c01 +
                                  fx * fy * c11) *
                                 static_cast<float>(camera[21]);
            }
            const double du = (u - camera[14]) / max(1.0, camera[12]);
            const double dv = (v - camera[15]) / max(1.0, camera[13]);
            const double view_weight = 1.0 / (1.0 + du * du + dv * dv);
            const double edge_distance =
                min(min(u, static_cast<double>(width - 1) - u), min(v, static_cast<double>(height - 1) - v));
            const double edge_weight = min(max(edge_distance / 20.0, 0.05), 1.0);
            *weight = view_weight * edge_weight * camera[22];
            return *weight > 0.0;
        }

        __global__ void orthoProjectionKernel(const float* demElevation,
                                              const std::uint8_t* demValid,
                                              int demWidth,
                                              int demHeight,
                                              double demMinX,
                                              double demMinY,
                                              double demStepX,
                                              double demStepY,
                                              int outputWidth,
                                              int outputHeight,
                                              double outputMinEdgeX,
                                              double outputMinEdgeY,
                                              double outputStepX,
                                              double outputStepY,
                                              double elevationOffset,
                                              const double* cameraValues,
                                              const int* cameraMetadata,
                                              int frameCount,
                                              const std::uint8_t* images,
                                              const std::uint8_t* masks,
                                              int blendMode,
                                              std::uint8_t* outputImage,
                                              std::uint8_t* surfaceMask,
                                              std::uint8_t* coverageMask,
                                              int* contributedFrames)
        {
            const int col = blockIdx.x * blockDim.x + threadIdx.x;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            if (col >= outputWidth || row >= outputHeight)
            {
                return;
            }
            const int pixel_index = row * outputWidth + col;
            outputImage[pixel_index * 3] = 0;
            outputImage[pixel_index * 3 + 1] = 0;
            outputImage[pixel_index * 3 + 2] = 0;
            surfaceMask[pixel_index] = 0;
            coverageMask[pixel_index] = 0;
            const double world_x = outputMinEdgeX + (static_cast<double>(col) + 0.5) * outputStepX;
            const double world_y = outputMinEdgeY + (static_cast<double>(row) + 0.5) * outputStepY;
            double elevation = 0.0;
            if (!sampleDem(demElevation,
                           demValid,
                           demWidth,
                           demHeight,
                           demMinX,
                           demMinY,
                           demStepX,
                           demStepY,
                           world_x,
                           world_y,
                           &elevation))
            {
                return;
            }
            surfaceMask[pixel_index] = 255;

            float accumulated[3]{0.0f, 0.0f, 0.0f};
            double total_weight = 0.0;
            double best_weight = -1.0;
            int best_frame = -1;
            int accepted = 0;
            for (int frame = 0; frame < frameCount; ++frame)
            {
                float color[3];
                double weight = 0.0;
                if (!sampleCamera(cameraValues + frame * kTerrainCameraValueStride,
                                  cameraMetadata + frame * kTerrainCameraMetadataStride,
                                  images,
                                  masks,
                                  world_x,
                                  world_y,
                                  elevation + elevationOffset,
                                  color,
                                  &weight))
                {
                    continue;
                }
                ++accepted;
                if (blendMode == kBlendFirstValid)
                {
                    accumulated[0] = color[0];
                    accumulated[1] = color[1];
                    accumulated[2] = color[2];
                    best_frame = frame;
                    break;
                }
                if (blendMode == kBlendMosaic)
                {
                    if (weight > best_weight)
                    {
                        best_weight = weight;
                        accumulated[0] = color[0];
                        accumulated[1] = color[1];
                        accumulated[2] = color[2];
                        best_frame = frame;
                    }
                    continue;
                }
                accumulated[0] += color[0] * static_cast<float>(weight);
                accumulated[1] += color[1] * static_cast<float>(weight);
                accumulated[2] += color[2] * static_cast<float>(weight);
                total_weight += weight;
                atomicExch(contributedFrames + frame, 1);
            }
            if (accepted == 0)
            {
                return;
            }
            if (blendMode == kBlendWeightedAverage)
            {
                if (!(total_weight > 0.0))
                {
                    return;
                }
                accumulated[0] /= static_cast<float>(total_weight);
                accumulated[1] /= static_cast<float>(total_weight);
                accumulated[2] /= static_cast<float>(total_weight);
            }
            else if (best_frame >= 0)
            {
                atomicExch(contributedFrames + best_frame, 1);
            }
            for (int channel = 0; channel < 3; ++channel)
            {
                outputImage[pixel_index * 3 + channel] =
                    static_cast<std::uint8_t>(min(max(accumulated[channel], 0.0f), 255.0f));
            }
            coverageMask[pixel_index] = 255;
        }

        __device__ bool
        mosaicCellValid(const float* elevation, const std::uint8_t* valid, int tile, int pixel, int pixelCount)
        {
            const int index = tile * pixelCount + pixel;
            return valid[index] && isfinite(elevation[index]);
        }

        __device__ float mosaicKthValue(
            const float* elevation, const std::uint8_t* valid, int tileCount, int pixel, int pixelCount, int kth)
        {
            for (int candidate_tile = 0; candidate_tile < tileCount; ++candidate_tile)
            {
                if (!mosaicCellValid(elevation, valid, candidate_tile, pixel, pixelCount))
                {
                    continue;
                }
                const float candidate = elevation[candidate_tile * pixelCount + pixel];
                int less = 0;
                int equal = 0;
                for (int tile = 0; tile < tileCount; ++tile)
                {
                    if (!mosaicCellValid(elevation, valid, tile, pixel, pixelCount))
                    {
                        continue;
                    }
                    const float value = elevation[tile * pixelCount + pixel];
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

        __global__ void demMosaicKernel(const float* inputElevation,
                                        const std::uint8_t* inputValid,
                                        const float* inputConfidence,
                                        const float* inputError,
                                        int tileCount,
                                        int pixelCount,
                                        int blendMode,
                                        float* outputElevation,
                                        std::uint8_t* outputValid,
                                        int* outputPointCount,
                                        float* outputConfidence,
                                        float* outputError)
        {
            const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
            if (pixel >= pixelCount)
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
            float minimum = FLT_MAX;
            float maximum = -FLT_MAX;
            for (int tile = 0; tile < tileCount; ++tile)
            {
                if (!mosaicCellValid(inputElevation, inputValid, tile, pixel, pixelCount))
                {
                    continue;
                }
                const int index = tile * pixelCount + pixel;
                const float value = inputElevation[index];
                const float confidence = max(0.0f, inputConfidence[index]);
                const float error = max(0.0f, inputError[index]);
                if (count == 0)
                {
                    first = value;
                }
                last = value;
                minimum = min(minimum, value);
                maximum = max(maximum, value);
                sum += value;
                confidence_sum += confidence;
                error_sum += error;
                float weight = 1.0f;
                if (blendMode == 6)
                {
                    weight = confidence;
                }
                else if (blendMode == 7)
                {
                    weight = 1.0f / max(error, 1.0e-6f);
                }
                weighted_sum += value * weight;
                weight_sum += weight;
                ++count;
            }

            outputElevation[pixel] = 0.0f;
            outputValid[pixel] = 0;
            outputPointCount[pixel] = 0;
            outputConfidence[pixel] = 0.0f;
            outputError[pixel] = 0.0f;
            if (count == 0)
            {
                return;
            }
            float blended = sum / static_cast<float>(count);
            if (blendMode == 0)
            {
                blended = first;
            }
            else if (blendMode == 1)
            {
                blended = last;
            }
            else if (blendMode == 3)
            {
                const float upper = mosaicKthValue(inputElevation, inputValid, tileCount, pixel, pixelCount, count / 2);
                blended =
                    count % 2 == 0
                        ? 0.5f * (upper + mosaicKthValue(
                                              inputElevation, inputValid, tileCount, pixel, pixelCount, count / 2 - 1))
                        : upper;
            }
            else if (blendMode == 4)
            {
                blended = minimum;
            }
            else if (blendMode == 5)
            {
                blended = maximum;
            }
            else if (blendMode == 6 || blendMode == 7)
            {
                blended = weight_sum > 0.0f ? weighted_sum / weight_sum : last;
            }
            outputElevation[pixel] = blended;
            outputValid[pixel] = 255;
            outputPointCount[pixel] = count;
            outputConfidence[pixel] = confidence_sum / static_cast<float>(count);
            outputError[pixel] = error_sum / static_cast<float>(count);
        }

        bool finishKernel(std::string* errorMsg)
        {
            cudaError_t status = cudaGetLastError();
            if (status == cudaSuccess)
            {
                status = cudaDeviceSynchronize();
            }
            if (status != cudaSuccess)
            {
                if (errorMsg)
                {
                    *errorMsg = cudaErrorMessage("执行 kernel", status);
                }
                return false;
            }
            return true;
        }

    } // namespace

    TerrainDeviceInfo queryTerrainCudaDevice(int deviceIndex)
    {
        TerrainDeviceInfo info;
        int count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&count);
        if (count_status != cudaSuccess)
        {
            info.error = cudaErrorMessage("查询设备", count_status);
            cudaGetLastError();
            return info;
        }
        const int resolved = deviceIndex < 0 ? 0 : deviceIndex;
        if (resolved < 0 || resolved >= count)
        {
            std::ostringstream stream;
            stream << "terrain CUDA 设备索引 " << deviceIndex << " 不可用，可用设备数为 " << count;
            info.error = stream.str();
            return info;
        }
        cudaDeviceProp properties{};
        const cudaError_t property_status = cudaGetDeviceProperties(&properties, resolved);
        if (property_status != cudaSuccess)
        {
            info.error = cudaErrorMessage("读取设备信息", property_status);
            return info;
        }
        info.available = true;
        info.resolvedIndex = resolved;
        info.name = std::string("CUDA · ") + properties.name;
        return info;
    }

    bool runTerrainCudaOrtho(const PackedOrthoProjection& input,
                             int deviceIndex,
                             PackedOrthoProjectionResult* output,
                             std::string* errorMsg)
    {
        if (!output)
        {
            if (errorMsg)
                *errorMsg = "terrain CUDA 正射输出对象为空";
            return false;
        }
        CudaDeviceGuard guard;
        if (!guard.select(deviceIndex, errorMsg))
        {
            return false;
        }
        DeviceBuffer<float> dem_elevation;
        DeviceBuffer<std::uint8_t> dem_valid;
        DeviceBuffer<double> cameras;
        DeviceBuffer<int> metadata;
        DeviceBuffer<std::uint8_t> images;
        DeviceBuffer<std::uint8_t> masks;
        DeviceBuffer<std::uint8_t> output_image;
        DeviceBuffer<std::uint8_t> output_surface;
        DeviceBuffer<std::uint8_t> output_coverage;
        DeviceBuffer<int> contributed;
        const std::size_t pixel_count =
            static_cast<std::size_t>(input.outputWidth) * static_cast<std::size_t>(input.outputHeight);
        output->imageBgr.assign(pixel_count * 3, 0);
        output->surfaceMask.assign(pixel_count, 0);
        output->coverageMask.assign(pixel_count, 0);
        output->contributedFrames.assign(static_cast<std::size_t>(input.frameCount), 0);
        if (!dem_elevation.upload(input.demElevation, errorMsg) || !dem_valid.upload(input.demValid, errorMsg) ||
            !cameras.upload(input.cameraValues, errorMsg) || !metadata.upload(input.cameraMetadata, errorMsg) ||
            !images.upload(input.imageData, errorMsg) || !masks.upload(input.maskData, errorMsg) ||
            !output_image.allocate(output->imageBgr.size(), errorMsg) ||
            !output_surface.allocate(pixel_count, errorMsg) || !output_coverage.allocate(pixel_count, errorMsg) ||
            !contributed.allocate(output->contributedFrames.size(), errorMsg))
        {
            return false;
        }
        const cudaError_t clear_status =
            cudaMemset(contributed.data(), 0, output->contributedFrames.size() * sizeof(int));
        if (clear_status != cudaSuccess)
        {
            if (errorMsg)
                *errorMsg = cudaErrorMessage("清空相机贡献标记", clear_status);
            return false;
        }
        const dim3 block(16, 16);
        const dim3 grid((input.outputWidth + block.x - 1) / block.x, (input.outputHeight + block.y - 1) / block.y);
        orthoProjectionKernel<<<grid, block>>>(dem_elevation.data(),
                                               dem_valid.data(),
                                               input.demWidth,
                                               input.demHeight,
                                               input.demMinX,
                                               input.demMinY,
                                               input.demStepX,
                                               input.demStepY,
                                               input.outputWidth,
                                               input.outputHeight,
                                               input.outputMinEdgeX,
                                               input.outputMinEdgeY,
                                               input.outputStepX,
                                               input.outputStepY,
                                               input.elevationOffset,
                                               cameras.data(),
                                               metadata.data(),
                                               input.frameCount,
                                               images.data(),
                                               masks.data(),
                                               input.blendMode,
                                               output_image.data(),
                                               output_surface.data(),
                                               output_coverage.data(),
                                               contributed.data());
        return finishKernel(errorMsg) && output_image.download(&output->imageBgr, errorMsg) &&
               output_surface.download(&output->surfaceMask, errorMsg) &&
               output_coverage.download(&output->coverageMask, errorMsg) &&
               contributed.download(&output->contributedFrames, errorMsg);
    }

    bool runTerrainCudaDemMosaic(const PackedDemMosaic& input,
                                 int deviceIndex,
                                 PackedDemMosaicResult* output,
                                 std::string* errorMsg)
    {
        if (!output)
        {
            if (errorMsg)
                *errorMsg = "terrain CUDA DEM mosaic 输出对象为空";
            return false;
        }
        CudaDeviceGuard guard;
        if (!guard.select(deviceIndex, errorMsg))
        {
            return false;
        }
        DeviceBuffer<float> input_elevation;
        DeviceBuffer<std::uint8_t> input_valid;
        DeviceBuffer<float> input_confidence;
        DeviceBuffer<float> input_error;
        DeviceBuffer<float> output_elevation;
        DeviceBuffer<std::uint8_t> output_valid;
        DeviceBuffer<int> output_count;
        DeviceBuffer<float> output_confidence;
        DeviceBuffer<float> output_error;
        const std::size_t pixel_count = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
        output->elevation.assign(pixel_count, 0.0f);
        output->valid.assign(pixel_count, 0);
        output->pointCount.assign(pixel_count, 0);
        output->confidence.assign(pixel_count, 0.0f);
        output->triangulationError.assign(pixel_count, 0.0f);
        if (!input_elevation.upload(input.elevation, errorMsg) || !input_valid.upload(input.valid, errorMsg) ||
            !input_confidence.upload(input.confidence, errorMsg) ||
            !input_error.upload(input.triangulationError, errorMsg) ||
            !output_elevation.allocate(pixel_count, errorMsg) || !output_valid.allocate(pixel_count, errorMsg) ||
            !output_count.allocate(pixel_count, errorMsg) || !output_confidence.allocate(pixel_count, errorMsg) ||
            !output_error.allocate(pixel_count, errorMsg))
        {
            return false;
        }
        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((pixel_count + block_size - 1) / block_size);
        demMosaicKernel<<<grid_size, block_size>>>(input_elevation.data(),
                                                   input_valid.data(),
                                                   input_confidence.data(),
                                                   input_error.data(),
                                                   input.tileCount,
                                                   static_cast<int>(pixel_count),
                                                   input.blendMode,
                                                   output_elevation.data(),
                                                   output_valid.data(),
                                                   output_count.data(),
                                                   output_confidence.data(),
                                                   output_error.data());
        return finishKernel(errorMsg) && output_elevation.download(&output->elevation, errorMsg) &&
               output_valid.download(&output->valid, errorMsg) &&
               output_count.download(&output->pointCount, errorMsg) &&
               output_confidence.download(&output->confidence, errorMsg) &&
               output_error.download(&output->triangulationError, errorMsg);
    }

} // namespace xjw::terrain_internal
