#include "TensorRtCapabilities.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <memory>

namespace xjw::inference
{
    namespace
    {

        class CapabilityLogger final : public nvinfer1::ILogger
        {
        public:
            void log(Severity, const char*) noexcept override
            {
            }
        };

        template <typename T> struct TensorRtDeleter
        {
            void operator()(T* value) const noexcept
            {
                delete value;
            }
        };

    } // namespace

    TensorRtCapabilities queryTensorRtCapabilities(int cudaDevice) noexcept
    {
        TensorRtCapabilities result;
        result.compiled = true;
        result.cudaDevice = std::max(0, cudaDevice);
        result.tensorRtVersion = QStringLiteral("%1.%2.%3.%4")
                                     .arg(NV_TENSORRT_MAJOR)
                                     .arg(NV_TENSORRT_MINOR)
                                     .arg(NV_TENSORRT_PATCH)
                                     .arg(NV_TENSORRT_BUILD);

        int device_count = 0;
        cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess)
        {
            result.errorMessage =
                QStringLiteral("无法查询 CUDA 设备：%1").arg(QString::fromLatin1(cudaGetErrorString(status)));
            return result;
        }
        result.deviceCount = device_count;
        if (device_count <= 0 || result.cudaDevice >= device_count)
        {
            result.errorMessage =
                QStringLiteral("CUDA 设备 %1 不存在（共 %2 个设备）").arg(result.cudaDevice).arg(device_count);
            return result;
        }

        cudaDeviceProp properties{};
        status = cudaSetDevice(result.cudaDevice);
        if (status == cudaSuccess)
        {
            status = cudaGetDeviceProperties(&properties, result.cudaDevice);
        }
        if (status != cudaSuccess)
        {
            result.errorMessage = QStringLiteral("无法初始化 CUDA 设备 %1：%2")
                                      .arg(result.cudaDevice)
                                      .arg(QString::fromLatin1(cudaGetErrorString(status)));
            return result;
        }

        result.cudaAvailable = true;
        result.gpuName = QString::fromLocal8Bit(properties.name);
        CapabilityLogger logger;
        std::unique_ptr<nvinfer1::IBuilder, TensorRtDeleter<nvinfer1::IBuilder>> builder(
            nvinfer1::createInferBuilder(logger));
        if (!builder)
        {
            result.errorMessage = QStringLiteral("无法创建 TensorRT Builder");
            return result;
        }
        result.fastFp16 = builder->platformHasFastFp16();
        return result;
    }

} // namespace xjw::inference
