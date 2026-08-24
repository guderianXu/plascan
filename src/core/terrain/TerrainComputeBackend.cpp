#include "TerrainComputeBackend.h"

#include "TerrainGpuBackend.h"

namespace xjw
{

    QString terrainComputeBackendToken(TerrainComputeBackend backend)
    {
        switch (backend)
        {
        case TerrainComputeBackend::Auto:
            return QStringLiteral("auto");
        case TerrainComputeBackend::Cpu:
            return QStringLiteral("cpu");
        case TerrainComputeBackend::Cuda:
            return QStringLiteral("cuda");
        case TerrainComputeBackend::OpenCl:
            return QStringLiteral("opencl");
        }
        return QStringLiteral("auto");
    }

    QString terrainComputeBackendDisplayName(TerrainComputeBackend backend)
    {
        switch (backend)
        {
        case TerrainComputeBackend::Auto:
            return QStringLiteral("Auto");
        case TerrainComputeBackend::Cpu:
            return QStringLiteral("CPU");
        case TerrainComputeBackend::Cuda:
            return QStringLiteral("CUDA");
        case TerrainComputeBackend::OpenCl:
            return QStringLiteral("OpenCL");
        }
        return QStringLiteral("Auto");
    }

    bool parseTerrainComputeBackend(const QString& token, TerrainComputeBackend* backend, QString* errorMsg)
    {
        if (!backend)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("地形计算后端输出对象为空");
            }
            return false;
        }

        const QString normalized = token.trimmed().toLower();
        if (normalized.isEmpty() || normalized == QLatin1String("auto"))
        {
            *backend = TerrainComputeBackend::Auto;
            return true;
        }
        if (normalized == QLatin1String("cpu"))
        {
            *backend = TerrainComputeBackend::Cpu;
            return true;
        }
        if (normalized == QLatin1String("cuda"))
        {
            *backend = TerrainComputeBackend::Cuda;
            return true;
        }
        if (normalized == QLatin1String("opencl") || normalized == QLatin1String("open_cl"))
        {
            *backend = TerrainComputeBackend::OpenCl;
            return true;
        }
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("不支持的地形计算后端: %1").arg(token);
        }
        return false;
    }

    bool isTerrainComputeBackendAvailable(TerrainComputeBackend backend, int deviceIndex)
    {
        switch (backend)
        {
        case TerrainComputeBackend::Auto:
        case TerrainComputeBackend::Cpu:
            return true;
        case TerrainComputeBackend::Cuda:
            return terrain_internal::queryTerrainCudaDevice(deviceIndex).available;
        case TerrainComputeBackend::OpenCl:
            return terrain_internal::queryTerrainOpenClDevice(deviceIndex).available;
        }
        return false;
    }

    bool isTerrainDemMosaicBackendAvailable(TerrainComputeBackend backend, int deviceIndex)
    {
        switch (backend)
        {
        case TerrainComputeBackend::Auto:
        case TerrainComputeBackend::Cpu:
            return true;
        case TerrainComputeBackend::Cuda:
            return terrain_internal::queryTerrainCudaDevice(deviceIndex).available;
        case TerrainComputeBackend::OpenCl:
            return terrain_internal::queryTerrainOpenClMosaicDevice(deviceIndex).available;
        }
        return false;
    }

    QString terrainComputeBackendDeviceName(TerrainComputeBackend backend, int deviceIndex)
    {
        if (backend == TerrainComputeBackend::Cpu)
        {
            return QStringLiteral("CPU");
        }
        if (backend == TerrainComputeBackend::Auto)
        {
            return QStringLiteral("Auto");
        }
        const terrain_internal::TerrainDeviceInfo info = backend == TerrainComputeBackend::Cuda
                                                             ? terrain_internal::queryTerrainCudaDevice(deviceIndex)
                                                             : terrain_internal::queryTerrainOpenClDevice(deviceIndex);
        return info.available ? QString::fromStdString(info.name) : QString();
    }

} // namespace xjw
