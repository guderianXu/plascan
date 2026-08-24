#include "TerrainGpuBackend.h"

namespace xjw::terrain_internal
{

    namespace
    {

        bool unavailable(std::string* errorMsg)
        {
            if (errorMsg)
            {
                *errorMsg = "当前构建未启用 terrain CUDA 后端";
            }
            return false;
        }

    } // namespace

    TerrainDeviceInfo queryTerrainCudaDevice(int)
    {
        TerrainDeviceInfo info;
        info.error = "当前构建未启用 terrain CUDA 后端";
        return info;
    }

    bool runTerrainCudaOrtho(const PackedOrthoProjection&, int, PackedOrthoProjectionResult*, std::string* errorMsg)
    {
        return unavailable(errorMsg);
    }

    bool runTerrainCudaDemMosaic(const PackedDemMosaic&, int, PackedDemMosaicResult*, std::string* errorMsg)
    {
        return unavailable(errorMsg);
    }

} // namespace xjw::terrain_internal
