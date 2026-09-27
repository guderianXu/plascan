#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "placamera/linescan_camera.h"

namespace placamera
{
    struct PlanetaryLineScanIsdMetadata
    {
        std::string modelName;
        std::string platformName;
        std::string sensorName;
        std::string interpolationMethod;
        std::string targetName;
        std::string bodyFixedFrameName;
        int bodyFixedFrameCode = 0;
        double startingEphemerisTimeSeconds = 0.0;
        double centerEphemerisTimeSeconds = 0.0;
        double focalLengthMillimeters = 0.0;
        double detectorSampleSumming = 1.0;
        double detectorLineSumming = 1.0;
    };

    struct PlanetaryLineScanIsdImport
    {
        CameraModelPtr<LineScanModel> instance;
        PlanetaryLineScanIsdMetadata metadata;
    };

    /** Decode a USGSCSM LRO line-scan ISD document into a typed camera instance. */
    Result<PlanetaryLineScanIsdImport> parsePlanetaryLineScanIsd(std::string_view json,
                                                                 CameraDefinitionId definitionId,
                                                                 CameraInstanceId instanceId,
                                                                 ImageId imageId);

    /** Open and import a USGSCSM planetary line-scanner ISD file. */
    Result<PlanetaryLineScanIsdImport> importPlanetaryLineScanIsd(const std::filesystem::path& path,
                                                                  CameraDefinitionId definitionId,
                                                                  CameraInstanceId instanceId,
                                                                  ImageId imageId);

} // namespace placamera
