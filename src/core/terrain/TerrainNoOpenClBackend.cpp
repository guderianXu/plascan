#include "TerrainGpuBackend.h"

namespace xjw::terrain_internal
{

    namespace
    {

        bool unavailable(std::string* errorMsg)
        {
            if (errorMsg)
            {
                *errorMsg = "当前构建未启用 terrain OpenCL 后端";
            }
            return false;
        }

    } // namespace

    TerrainDeviceInfo queryTerrainOpenClDevice(int)
    {
        TerrainDeviceInfo info;
        info.error = "当前构建未启用 terrain OpenCL 后端";
        return info;
    }

    TerrainDeviceInfo queryTerrainOpenClMosaicDevice(int)
    {
        TerrainDeviceInfo info;
        info.error = "当前构建未启用 terrain OpenCL 后端";
        return info;
    }

    bool runTerrainOpenClOrtho(const PackedOrthoProjection&, int, PackedOrthoProjectionResult*, std::string* errorMsg)
    {
        return unavailable(errorMsg);
    }

    bool runTerrainOpenClDemMosaic(const PackedDemMosaic&, int, PackedDemMosaicResult*, std::string* errorMsg)
    {
        return unavailable(errorMsg);
    }

} // namespace xjw::terrain_internal
