#pragma once

#include <QString>

namespace xjw
{

    enum class TerrainComputeBackend
    {
        Auto,
        Cpu,
        Cuda,
        OpenCl
    };

    struct TerrainComputeOptions
    {
        TerrainComputeBackend backend = TerrainComputeBackend::Auto;
        int deviceIndex = -1;
    };

    struct TerrainComputeExecution
    {
        TerrainComputeBackend backend = TerrainComputeBackend::Cpu;
        int deviceIndex = -1;
        QString deviceName = QStringLiteral("CPU");
        QString fallbackReason;
    };

    QString terrainComputeBackendToken(TerrainComputeBackend backend);
    QString terrainComputeBackendDisplayName(TerrainComputeBackend backend);
    bool parseTerrainComputeBackend(const QString& token, TerrainComputeBackend* backend, QString* errorMsg = nullptr);
    bool isTerrainComputeBackendAvailable(TerrainComputeBackend backend, int deviceIndex = -1);
    bool isTerrainDemMosaicBackendAvailable(TerrainComputeBackend backend, int deviceIndex = -1);
    QString terrainComputeBackendDeviceName(TerrainComputeBackend backend, int deviceIndex = -1);

} // namespace xjw
