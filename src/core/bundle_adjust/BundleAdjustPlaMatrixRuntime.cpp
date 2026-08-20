#include "BundleAdjustPlaMatrixRuntime.h"

#include "BundleAdjustPlaMatrix.h"

#include <plamatrix/opencl/runtime.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <string>

namespace xjw::detail
{

plamatrix::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend)
{
    switch (backend)
    {
    case BABackend::PlaMatrixCuda:
        return plamatrix::SchurComplementLinearBackend::Cuda;
    case BABackend::PlaMatrixOpenCl:
        return plamatrix::SchurComplementLinearBackend::OpenCl;
    default:
        return plamatrix::SchurComplementLinearBackend::Cpu;
    }
}

const char* plaMatrixLinearBackendName(
    plamatrix::SchurComplementLinearBackend backend)
{
    switch (backend)
    {
    case plamatrix::SchurComplementLinearBackend::Cpu:
        return "block_jacobi_pcg_cpu";
    case plamatrix::SchurComplementLinearBackend::DenseCpu:
        return "dense_cholesky_cpu";
    case plamatrix::SchurComplementLinearBackend::Cuda:
        return "block_jacobi_pcg_cuda";
    case plamatrix::SchurComplementLinearBackend::OpenCl:
        return "block_jacobi_pcg_opencl";
    }
    return "unknown";
}

bool isPlaMatrixBackendAvailable(BABackend backend,
                                 int device_index,
                                 std::string* message)
{
    if (backend == BABackend::PlaMatrixCpu)
    {
        return true;
    }
    if (device_index < 0)
    {
        if (message)
        {
            *message = "PlaMatrix 设备索引不能为负数";
        }
        return false;
    }
    if (backend == BABackend::PlaMatrixCuda)
    {
#ifdef PLAMATRIX_WITH_CUDA
        int device_count = 0;
        const cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error == cudaSuccess && device_index < device_count)
        {
            return true;
        }
        if (message)
        {
            *message = error == cudaSuccess
                ? "PlaMatrix CUDA 设备索引超出范围"
                : std::string("PlaMatrix CUDA 运行时不可用：") +
                      cudaGetErrorString(error);
        }
#else
        if (message)
        {
            *message = "PlaMatrix 编译时未启用 CUDA";
        }
#endif
        return false;
    }
    if (backend == BABackend::PlaMatrixOpenCl)
    {
        if (!plamatrix::opencl::hasUsableOpenClDevice())
        {
            if (message)
            {
                *message = "PlaMatrix 没有可用的 OpenCL GPU 或在线编译器";
            }
            return false;
        }
        const int selected_index = plamatrix::opencl::selectedOpenClDeviceIndex();
        if (device_index != selected_index)
        {
            if (message)
            {
                *message = "PlaMatrix OpenCL 已选择设备 " +
                    std::to_string(selected_index) + "，与请求设备 " +
                    std::to_string(device_index) + " 不一致";
            }
            return false;
        }
        return true;
    }
    if (message)
    {
        *message = "不是 PlaMatrix BA 后端";
    }
    return false;
}

} // namespace xjw::detail
