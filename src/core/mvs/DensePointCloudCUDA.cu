#include "DensePointCloudCUDA.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace xjw::mvs
{
    namespace
    {

        struct CameraParameters
        {
            float focalX = 0.0f;
            float focalY = 0.0f;
            float principalX = 0.0f;
            float principalY = 0.0f;
            float cameraToWorld[9]{};
            float cameraCenter[3]{};
            int uAxisSign = 1;
            int vAxisSign = 1;
            int depthSign = 1;
        };

        __global__ void unprojectKernel(const float* depth,
                                        const std::uint8_t* mask,
                                        const std::uint8_t* color,
                                        int width,
                                        int height,
                                        int channels,
                                        float minimumDepth,
                                        float maximumDepth,
                                        int subsample,
                                        int clipAabb,
                                        float minimumX,
                                        float maximumX,
                                        float minimumY,
                                        float maximumY,
                                        float minimumZ,
                                        float maximumZ,
                                        CameraParameters camera,
                                        float* xyz,
                                        std::uint8_t* rgb,
                                        std::int32_t* valid)
        {
            const int column = blockIdx.x * blockDim.x + threadIdx.x;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            if (column >= width || row >= height)
            {
                return;
            }

            const int index = row * width + column;
            valid[index] = 0;
            if ((row % subsample) != 0 || (column % subsample) != 0 || (mask != nullptr && mask[index] == 0))
            {
                return;
            }

            const float positiveDepth = depth[index];
            if (!isfinite(positiveDepth) || positiveDepth < minimumDepth || positiveDepth > maximumDepth)
            {
                return;
            }

            const float normalizedX =
                static_cast<float>(camera.uAxisSign) * (static_cast<float>(column) - camera.principalX) / camera.focalX;
            const float normalizedY =
                static_cast<float>(camera.vAxisSign) * (static_cast<float>(row) - camera.principalY) / camera.focalY;
            const float cameraZ = static_cast<float>(camera.depthSign) * positiveDepth;
            const float cameraX = normalizedX * cameraZ;
            const float cameraY = normalizedY * cameraZ;
            const float worldX = camera.cameraCenter[0] + camera.cameraToWorld[0] * cameraX +
                                 camera.cameraToWorld[1] * cameraY + camera.cameraToWorld[2] * cameraZ;
            const float worldY = camera.cameraCenter[1] + camera.cameraToWorld[3] * cameraX +
                                 camera.cameraToWorld[4] * cameraY + camera.cameraToWorld[5] * cameraZ;
            const float worldZ = camera.cameraCenter[2] + camera.cameraToWorld[6] * cameraX +
                                 camera.cameraToWorld[7] * cameraY + camera.cameraToWorld[8] * cameraZ;
            if (!isfinite(worldX) || !isfinite(worldY) || !isfinite(worldZ) ||
                (clipAabb != 0 && (worldX < minimumX || worldX > maximumX || worldY < minimumY || worldY > maximumY ||
                                   worldZ < minimumZ || worldZ > maximumZ)))
            {
                return;
            }

            xyz[index * 3] = worldX;
            xyz[index * 3 + 1] = worldY;
            xyz[index * 3 + 2] = worldZ;
            if (channels == 3 && color != nullptr)
            {
                rgb[index * 3] = color[index * 3 + 2];
                rgb[index * 3 + 1] = color[index * 3 + 1];
                rgb[index * 3 + 2] = color[index * 3];
            }
            else if (channels == 1 && color != nullptr)
            {
                const std::uint8_t gray = color[index];
                rgb[index * 3] = gray;
                rgb[index * 3 + 1] = gray;
                rgb[index * 3 + 2] = gray;
            }
            else
            {
                rgb[index * 3] = 128;
                rgb[index * 3 + 1] = 128;
                rgb[index * 3 + 2] = 128;
            }
            valid[index] = 1;
        }

        bool hasZeroDistortion(const FramePinholeCamera& camera)
        {
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            return distortion.radialK1 == 0.0 && distortion.radialK2 == 0.0 && distortion.radialK3 == 0.0 &&
                   distortion.tangentialP1 == 0.0 && distortion.tangentialP2 == 0.0;
        }

        std::string cudaErrorMessage(const char* operation, cudaError_t status)
        {
            return std::string(operation) + " failed: " + cudaGetErrorString(status);
        }

    } // namespace

    bool DensePointCloudCUDA::isAvailable(int deviceIndex, std::string* errorMsg)
    {
        if (errorMsg)
        {
            errorMsg->clear();
        }
        int deviceCount = 0;
        const cudaError_t status = cudaGetDeviceCount(&deviceCount);
        if (status != cudaSuccess)
        {
            if (errorMsg)
            {
                *errorMsg = cudaErrorMessage("cudaGetDeviceCount", status);
            }
            (void)cudaGetLastError();
            return false;
        }
        if (deviceIndex < 0 || deviceIndex >= deviceCount)
        {
            if (errorMsg)
            {
                *errorMsg = "CUDA dense-cloud device index is unavailable";
            }
            return false;
        }
        return true;
    }

    std::string DensePointCloudCUDA::deviceName(int deviceIndex)
    {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, deviceIndex) != cudaSuccess)
        {
            return {};
        }
        return properties.name;
    }

    std::vector<DensePoint> DensePointCloudCUDA::unprojectGPU(const cv::Mat& depth,
                                                              const cv::Mat& mask,
                                                              const FramePinholeCamera& camera,
                                                              const cv::Mat& colorImage,
                                                              float minimumDepth,
                                                              float maximumDepth,
                                                              std::string* errorMsg,
                                                              const DenseCloudOptions* options)
    {
        if (errorMsg)
        {
            errorMsg->clear();
        }
        DenseCloudOptions effectiveOptions;
        if (options)
        {
            effectiveOptions = *options;
        }
        effectiveOptions.minDepth = minimumDepth;
        effectiveOptions.maxDepth = maximumDepth;

        std::string availabilityError;
        if (!isAvailable(effectiveOptions.cudaDeviceIndex, &availabilityError))
        {
            if (errorMsg)
            {
                *errorMsg = availabilityError;
            }
            return {};
        }
        if (!hasZeroDistortion(camera))
        {
            if (errorMsg)
            {
                *errorMsg = "CUDA dense-cloud unprojection requires a prepared zero-distortion camera";
            }
            return {};
        }
        if (depth.empty())
        {
            return {};
        }

        const std::int64_t elementCount64 = static_cast<std::int64_t>(depth.cols) * depth.rows;
        if (elementCount64 <= 0 || elementCount64 > std::numeric_limits<int>::max() / 3)
        {
            if (errorMsg)
            {
                *errorMsg = "CUDA dense-cloud image is too large";
            }
            return {};
        }
        const int elementCount = static_cast<int>(elementCount64);
        const int channels = colorImage.empty() ? 0 : colorImage.channels();
        const cv::Mat continuousDepth = depth.isContinuous() ? depth : depth.clone();
        const cv::Mat continuousMask = mask.empty() || mask.isContinuous() ? mask : mask.clone();
        const cv::Mat continuousColor =
            colorImage.empty() || colorImage.isContinuous() ? colorImage : colorImage.clone();

        float* deviceDepth = nullptr;
        std::uint8_t* deviceMask = nullptr;
        std::uint8_t* deviceColor = nullptr;
        float* deviceXyz = nullptr;
        std::uint8_t* deviceRgb = nullptr;
        std::int32_t* deviceValid = nullptr;
        cudaStream_t stream = nullptr;
        const auto cleanup = [&]()
        {
            if (stream)
            {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
            cudaFree(deviceDepth);
            cudaFree(deviceMask);
            cudaFree(deviceColor);
            cudaFree(deviceXyz);
            cudaFree(deviceRgb);
            cudaFree(deviceValid);
        };
        const auto check = [&](cudaError_t status, const char* operation)
        {
            if (status == cudaSuccess)
            {
                return true;
            }
            if (errorMsg)
            {
                *errorMsg = cudaErrorMessage(operation, status);
            }
            return false;
        };

        if (!check(cudaSetDevice(effectiveOptions.cudaDeviceIndex), "cudaSetDevice") ||
            !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "CUDA stream creation") ||
            !check(cudaMalloc(&deviceDepth, static_cast<std::size_t>(elementCount) * sizeof(float)),
                   "CUDA depth allocation") ||
            !check(cudaMalloc(&deviceXyz, static_cast<std::size_t>(elementCount) * 3 * sizeof(float)),
                   "CUDA point allocation") ||
            !check(cudaMalloc(&deviceRgb, static_cast<std::size_t>(elementCount) * 3 * sizeof(std::uint8_t)),
                   "CUDA color allocation") ||
            !check(cudaMalloc(&deviceValid, static_cast<std::size_t>(elementCount) * sizeof(std::int32_t)),
                   "CUDA validity allocation") ||
            !check(cudaMemcpyAsync(deviceDepth,
                                   continuousDepth.ptr<float>(),
                                   static_cast<std::size_t>(elementCount) * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "CUDA depth upload"))
        {
            cleanup();
            return {};
        }
        if (!continuousMask.empty())
        {
            if (!check(cudaMalloc(&deviceMask, static_cast<std::size_t>(elementCount) * sizeof(std::uint8_t)),
                       "CUDA mask allocation") ||
                !check(cudaMemcpyAsync(deviceMask,
                                       continuousMask.ptr<std::uint8_t>(),
                                       static_cast<std::size_t>(elementCount) * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice,
                                       stream),
                       "CUDA mask upload"))
            {
                cleanup();
                return {};
            }
        }
        if (!continuousColor.empty())
        {
            if (!check(
                    cudaMalloc(&deviceColor, static_cast<std::size_t>(elementCount) * channels * sizeof(std::uint8_t)),
                    "CUDA color allocation") ||
                !check(cudaMemcpyAsync(deviceColor,
                                       continuousColor.data,
                                       static_cast<std::size_t>(elementCount) * channels * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice,
                                       stream),
                       "CUDA color upload"))
            {
                cleanup();
                return {};
            }
        }

        CameraParameters parameters;
        const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
        const FramePinholeCamera::Pose pose = camera.pose();
        parameters.focalX = static_cast<float>(intrinsics.focalX);
        parameters.focalY = static_cast<float>(intrinsics.focalY);
        parameters.principalX = static_cast<float>(intrinsics.principalX);
        parameters.principalY = static_cast<float>(intrinsics.principalY);
        parameters.uAxisSign = intrinsics.uAxisSign;
        parameters.vAxisSign = intrinsics.vAxisSign;
        parameters.depthSign = pose.depthAxisFlipped ? -1 : 1;
        for (int index = 0; index < 9; ++index)
        {
            parameters.cameraToWorld[index] =
                static_cast<float>(pose.cameraToWorldRotation[static_cast<std::size_t>(index)]);
        }
        for (int index = 0; index < 3; ++index)
        {
            parameters.cameraCenter[index] = static_cast<float>(pose.cameraCenter[static_cast<std::size_t>(index)]);
        }

        const int subsample = std::max(1, effectiveOptions.subsample);
        const int blockWidth = std::clamp(effectiveOptions.cudaBlockW, 1, 32);
        const int blockHeight = std::clamp(effectiveOptions.cudaBlockH, 1, std::max(1, 1024 / blockWidth));
        const dim3 block(blockWidth, blockHeight);
        const dim3 grid((depth.cols + blockWidth - 1) / blockWidth, (depth.rows + blockHeight - 1) / blockHeight);
        unprojectKernel<<<grid, block, 0, stream>>>(deviceDepth,
                                                    deviceMask,
                                                    deviceColor,
                                                    depth.cols,
                                                    depth.rows,
                                                    channels,
                                                    minimumDepth,
                                                    maximumDepth,
                                                    subsample,
                                                    effectiveOptions.clipAABB ? 1 : 0,
                                                    effectiveOptions.minX,
                                                    effectiveOptions.maxX,
                                                    effectiveOptions.minY,
                                                    effectiveOptions.maxY,
                                                    effectiveOptions.minZ,
                                                    effectiveOptions.maxZ,
                                                    parameters,
                                                    deviceXyz,
                                                    deviceRgb,
                                                    deviceValid);
        if (!check(cudaGetLastError(), "CUDA dense-cloud kernel launch") ||
            !check(cudaStreamSynchronize(stream), "CUDA dense-cloud synchronization"))
        {
            cleanup();
            return {};
        }

        std::vector<float> hostXyz(static_cast<std::size_t>(elementCount) * 3);
        std::vector<std::uint8_t> hostRgb(static_cast<std::size_t>(elementCount) * 3);
        std::vector<std::int32_t> hostValid(static_cast<std::size_t>(elementCount));
        if (!check(cudaMemcpyAsync(
                       hostXyz.data(), deviceXyz, hostXyz.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "CUDA point download") ||
            !check(
                cudaMemcpyAsync(
                    hostRgb.data(), deviceRgb, hostRgb.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
                "CUDA color download") ||
            !check(cudaMemcpyAsync(hostValid.data(),
                                   deviceValid,
                                   hostValid.size() * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "CUDA validity download") ||
            !check(cudaStreamSynchronize(stream), "CUDA dense-cloud download synchronization"))
        {
            cleanup();
            return {};
        }
        cleanup();

        std::vector<DensePoint> result;
        const std::size_t sampledColumns =
            (static_cast<std::size_t>(depth.cols) + static_cast<std::size_t>(subsample) - 1) /
            static_cast<std::size_t>(subsample);
        const std::size_t sampledRows =
            (static_cast<std::size_t>(depth.rows) + static_cast<std::size_t>(subsample) - 1) /
            static_cast<std::size_t>(subsample);
        result.reserve(sampledColumns * sampledRows);
        for (int index = 0; index < elementCount; ++index)
        {
            if (hostValid[static_cast<std::size_t>(index)] == 0)
            {
                continue;
            }
            DensePoint point;
            point.x = hostXyz[static_cast<std::size_t>(index) * 3];
            point.y = hostXyz[static_cast<std::size_t>(index) * 3 + 1];
            point.z = hostXyz[static_cast<std::size_t>(index) * 3 + 2];
            point.r = hostRgb[static_cast<std::size_t>(index) * 3];
            point.g = hostRgb[static_cast<std::size_t>(index) * 3 + 1];
            point.b = hostRgb[static_cast<std::size_t>(index) * 3 + 2];
            result.push_back(point);
        }
        return result;
    }

} // namespace xjw::mvs
