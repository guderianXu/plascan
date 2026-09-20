#pragma once

#include "camera/models/linescan/LineScanInstance.h"

#include <memory>
#include <string>

namespace xjw::camera_models::linescan
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
        std::shared_ptr<const LineScanInstance> instance;
        PlanetaryLineScanIsdMetadata metadata;
    };

    /** Import a USGSCSM planetary line-scanner ISD directly into the typed camera model. */
    bool importPlanetaryLineScanIsd(const std::string& path,
                                    camera_core::CameraDefinitionId definitionId,
                                    camera_core::CameraInstanceId instanceId,
                                    camera_core::ImageId imageId,
                                    PlanetaryLineScanIsdImport* imported,
                                    std::string* error = nullptr);

} // namespace xjw::camera_models::linescan
