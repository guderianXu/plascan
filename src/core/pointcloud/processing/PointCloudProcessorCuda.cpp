#include "processing/PointCloudProcessor.h"

#include "Logger.h"

#include <string>
#include <vector>

#if defined(POINTCLOUD_ENABLE_CUDA)
extern bool pointcloud_processor_cuda_runtime_available();
extern bool pointcloud_processor_cuda_voxel_downsample(const float *xyz,
                                                       int pointCount,
                                                       float voxelSize,
                                                       std::vector<float> *outXyz,
                                                       std::string *detail);
#endif

namespace xjw::pointcloud::detail
{

namespace
{

CudaRuntimeAvailableHook &runtimeHookStorage()
{
    static CudaRuntimeAvailableHook runtimeHook;
    return runtimeHook;
}

CudaVoxelDownsampleHook &voxelHookStorage()
{
    static CudaVoxelDownsampleHook voxelHook;
    return voxelHook;
}

} // namespace

void setCudaTestHooks(CudaRuntimeAvailableHook runtimeHook, CudaVoxelDownsampleHook voxelHook)
{
    runtimeHookStorage() = std::move(runtimeHook);
    voxelHookStorage() = std::move(voxelHook);
}

void clearCudaTestHooks()
{
    runtimeHookStorage() = {};
    voxelHookStorage() = {};
}

bool isCudaRuntimeAvailableInternal()
{
    if (runtimeHookStorage())
    {
        return runtimeHookStorage()();
    }

#if defined(POINTCLOUD_ENABLE_CUDA)
    return pointcloud_processor_cuda_runtime_available();
#else
    return false;
#endif
}

bool voxelDownsampleCuda(const PointCloud &input, double voxelSize, PointCloud *output, QString *detail)
{
    if (!output)
    {
        return false;
    }

    if (voxelHookStorage())
    {
        return voxelHookStorage()(input, voxelSize, output, detail);
    }

#if defined(POINTCLOUD_ENABLE_CUDA)
    if (input.empty() || voxelSize <= 0.0)
    {
        *output = input;
        return true;
    }

    const PointCloudCudaExport exportData = input.exportToCuda();
    std::vector<float> outXyz;
    std::string backendDetail;
    if (!pointcloud_processor_cuda_voxel_downsample(exportData.positionsXyz.data(),
                                                    static_cast<int>(input.size()),
                                                    static_cast<float>(voxelSize),
                                                    &outXyz,
                                                    &backendDetail))
    {
        if (detail)
        {
            *detail = QString::fromStdString(backendDetail);
        }
        return false;
    }

    PointCloud result;
    result.setMetadata(input.metadata());
    result.reserve(outXyz.size() / 3);
    for (std::size_t index = 0; index + 2 < outXyz.size(); index += 3)
    {
        result.addPoint(Point3f{outXyz[index], outXyz[index + 1], outXyz[index + 2]});
    }

    if (detail)
    {
        *detail = QString::fromStdString(backendDetail.empty() ? "CUDA voxel downsample completed" : backendDetail);
    }

    LOG_INFO("PointCloudProcessor::voxelDownsampleCuda: before=%zu after=%zu", input.size(), result.size());
    *output = std::move(result);
    return true;
#else
    Q_UNUSED(input);
    Q_UNUSED(voxelSize);
    if (detail)
    {
        *detail = QStringLiteral("CUDA backend not compiled");
    }
    return false;
#endif
}

} // namespace xjw::pointcloud::detail