#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xjw::terrain_internal
{

    inline constexpr int kTerrainCameraValueStride = 23;
    inline constexpr int kTerrainCameraMetadataStride = 7;

    struct TerrainDeviceInfo
    {
        bool available = false;
        int resolvedIndex = -1;
        std::string name;
        std::string error;
    };

    struct PackedOrthoProjection
    {
        int demWidth = 0;
        int demHeight = 0;
        int outputWidth = 0;
        int outputHeight = 0;
        int frameCount = 0;
        int blendMode = 0;
        double demMinX = 0.0;
        double demMinY = 0.0;
        double demStepX = 1.0;
        double demStepY = 1.0;
        double outputMinEdgeX = 0.0;
        double outputMinEdgeY = 0.0;
        double outputStepX = 1.0;
        double outputStepY = 1.0;
        double elevationOffset = 0.0;
        std::vector<float> demElevation;
        std::vector<std::uint8_t> demValid;
        std::vector<double> cameraValues;
        std::vector<int> cameraMetadata;
        std::vector<std::uint8_t> imageData;
        std::vector<std::uint8_t> maskData;
    };

    struct PackedOrthoProjectionResult
    {
        std::vector<std::uint8_t> imageBgr;
        std::vector<std::uint8_t> surfaceMask;
        std::vector<std::uint8_t> coverageMask;
        std::vector<int> contributedFrames;
    };

    struct PackedDemMosaic
    {
        int width = 0;
        int height = 0;
        int tileCount = 0;
        int blendMode = 0;
        std::vector<float> elevation;
        std::vector<std::uint8_t> valid;
        std::vector<float> confidence;
        std::vector<float> triangulationError;
    };

    struct PackedDemMosaicResult
    {
        std::vector<float> elevation;
        std::vector<std::uint8_t> valid;
        std::vector<int> pointCount;
        std::vector<float> confidence;
        std::vector<float> triangulationError;
    };

    TerrainDeviceInfo queryTerrainCudaDevice(int deviceIndex);
    TerrainDeviceInfo queryTerrainOpenClDevice(int deviceIndex);
    TerrainDeviceInfo queryTerrainOpenClMosaicDevice(int deviceIndex);

    bool runTerrainCudaOrtho(const PackedOrthoProjection& input,
                             int deviceIndex,
                             PackedOrthoProjectionResult* output,
                             std::string* errorMsg);
    bool runTerrainOpenClOrtho(const PackedOrthoProjection& input,
                               int deviceIndex,
                               PackedOrthoProjectionResult* output,
                               std::string* errorMsg);
    bool runTerrainCudaDemMosaic(const PackedDemMosaic& input,
                                 int deviceIndex,
                                 PackedDemMosaicResult* output,
                                 std::string* errorMsg);
    bool runTerrainOpenClDemMosaic(const PackedDemMosaic& input,
                                   int deviceIndex,
                                   PackedDemMosaicResult* output,
                                   std::string* errorMsg);

} // namespace xjw::terrain_internal
