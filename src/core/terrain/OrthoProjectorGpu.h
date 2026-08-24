#pragma once

#include "OrthoProjectorInternal.h"
#include "TerrainComputeBackend.h"

namespace xjw::ortho_internal
{

    bool projectPixelsOnGpu(const DemGridData& demGrid,
                            const OrthoOutputGrid& outputGrid,
                            std::vector<LoadedFrame>* frames,
                            const OrthoGenerationOptions& options,
                            double demElevationOffset,
                            TerrainComputeBackend backend,
                            int deviceIndex,
                            OrthoProjectionResult* result,
                            qint64* surfacePixelCount,
                            int* resolvedDeviceIndex,
                            QString* deviceName,
                            QString* errorMsg);

} // namespace xjw::ortho_internal
