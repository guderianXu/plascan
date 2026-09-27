#pragma once

#include <array>
#include <filesystem>
#include <istream>
#include <ostream>
#include <string>

#include "placamera/frame_camera.h"

namespace placamera
{

    using TsaiFramePinhole = FramePinholeGeometry;

    /** Pixel-unit fields for the versioned Tsai interchange layout. */
    struct TsaiPixelCamera
    {
        double focalX = 0.0;
        double focalY = 0.0;
        double principalX = 0.0;
        double principalY = 0.0;
        std::array<double, 5> distortion{};
        std::array<double, 3> center{};
        std::array<double, 9> cameraToWorldRotation{};
    };

    /** Read an ASP/Tsai camera without assigning an image identity or guessing its world frame. */
    Result<TsaiFramePinhole>
    readTsaiFramePinhole(std::istream& input, CameraDefinitionId definitionId, FrameId worldFrame);

    /** Open and read an ASP/Tsai camera file. */
    Result<TsaiFramePinhole>
    loadTsaiFramePinhole(const std::filesystem::path& path, CameraDefinitionId definitionId, FrameId worldFrame);

    /** Write a PlaCamera frame camera using ASP/Tsai units and field names. */
    Result<void> writeTsaiFramePinhole(std::ostream& output, const TsaiFramePinhole& camera);

    /** Validate and write an ASP/Tsai camera file. */
    Result<void> saveTsaiFramePinhole(const TsaiFramePinhole& camera, const std::filesystem::path& path);

    /** Write a versioned Tsai record with pitch one and deterministic 12-digit numeric fields. */
    Result<void> writeTsaiPixelCamera(std::ostream& output, const TsaiPixelCamera& camera);

} // namespace placamera
