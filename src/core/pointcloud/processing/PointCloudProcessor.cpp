#include "processing/PointCloudProcessor.h"

#include "Logger.h"

#include <algorithm>
#include <QElapsedTimer>

namespace xjw::pointcloud
{

PointCloudStats PointCloudProcessor::computeStats(const PointCloud &pointCloud)
{
    PointCloudStats stats;
    stats.count = static_cast<int>(pointCloud.size());
    stats.bounds = pointCloud.computeBounds();
    return stats;
}

bool PointCloudProcessor::runOneClickPreprocess(const PointCloud &input,
                                                PointCloud *output,
                                                PointCloudProcessResult *result,
                                                std::function<void(int, const QString &)> progressCallback)
{
    PointCloudProcessParams params;
    params.noiseMethod = PointCloudProcessParams::NoiseMethod::Statistical;
    params.intensityLevel = PointCloudProcessParams::IntensityLevel::Medium;
    params.statK = 20;
    params.statStdMul = 1.0;
    params.downsampleMethod = PointCloudProcessParams::DownsampleMethod::Voxel;
    params.voxelSize = 0.02;
    params.normalK = 32;
    params.smoothNormals = true;
    params.unifyNormalDirection = true;
    return runCustomProcess(input, params, output, result, std::move(progressCallback));
}

bool PointCloudProcessor::runCustomProcess(const PointCloud &input,
                                           const PointCloudProcessParams &params,
                                           PointCloud *output,
                                           PointCloudProcessResult *result,
                                           std::function<void(int, const QString &)> progressCallback)
{
    if (!output || !result)
    {
        return false;
    }

    auto progress = [&](int value, const QString &text) {
        if (progressCallback)
        {
            progressCallback(value, text);
        }
    };

    QElapsedTimer timer;
    timer.start();

    result->pointsBefore = static_cast<int>(input.size());
    result->pointsAfter = 0;
    result->cudaUsed = false;
    result->detail.clear();

    PointCloud working = input;
    if (params.previewOnly)
    {
        const std::size_t keepCount = static_cast<std::size_t>(std::max(
            1,
            std::min(
                static_cast<int>(working.size()),
                static_cast<int>(working.size() * std::clamp(params.previewRatio, 0.05, 1.0)))));
        working.filterInPlace([keepCount](std::size_t index,
                                          const Point3f &,
                                          const Point3f *,
                                          const ColorRGBA *,
                                          const PhotogrammetryPointAttributes *) {
            return index < keepCount;
        });
    }

    progress(10, QStringLiteral("噪声过滤"));
    if (params.noiseMethod == PointCloudProcessParams::NoiseMethod::Statistical)
    {
        working = detail::statisticalFilterMultithread(working, params.statK, params.statStdMul, params.threads);
    }
    else
    {
        working = detail::radiusFilterMultithread(working, params.radiusFilter, params.radiusMinNeighbors, params.threads);
    }

    progress(40, QStringLiteral("点云降采样"));
    if (params.downsampleMethod == PointCloudProcessParams::DownsampleMethod::Voxel)
    {
        if (params.useCuda && detail::isCudaRuntimeAvailableInternal())
        {
            PointCloud gpuOutput;
            QString gpuDetail;
            if (detail::voxelDownsampleCuda(working, params.voxelSize, &gpuOutput, &gpuDetail))
            {
                working = std::move(gpuOutput);
                result->cudaUsed = true;
                result->detail = gpuDetail;
            }
            else
            {
                working = detail::voxelDownsampleMultithread(working, params.voxelSize, params.threads);
            }
        }
        else
        {
            working = detail::voxelDownsampleMultithread(working, params.voxelSize, params.threads);
        }
    }
    else
    {
        working = detail::uniformDownsample(working, params.uniformStep);
    }

    progress(70, QStringLiteral("法向量与属性整理"));
    if (params.unifyNormalDirection && working.hasNormals())
    {
        std::vector<Point3f> normals = working.normals();
        for (Point3f &normal : normals)
        {
            if (normal.z < 0.0f)
            {
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }
        }
        working.setNormals(normals);
    }

    progress(90, QStringLiteral("结果收尾"));
    *output = std::move(working);
    result->pointsAfter = static_cast<int>(output->size());
    result->elapsedMs = timer.elapsed();
    if (result->detail.isEmpty())
    {
        result->detail = QStringLiteral("点云处理完成，耗时 %1 ms").arg(result->elapsedMs);
    }

    LOG_INFO("PointCloudProcessor::runCustomProcess: before=%d after=%d cuda=%d elapsedMs=%lld",
             result->pointsBefore,
             result->pointsAfter,
             result->cudaUsed ? 1 : 0,
             static_cast<long long>(result->elapsedMs));

    progress(100, QStringLiteral("完成"));
    return true;
}

bool PointCloudProcessor::isCudaRuntimeAvailable()
{
    return detail::isCudaRuntimeAvailableInternal();
}

} // namespace xjw::pointcloud