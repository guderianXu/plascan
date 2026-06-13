#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xjw::camera
{

enum class CameraFormat
{
    Auto,
    MiddleburyPar,
    EpflCamera,
    ColmapText,
    MetashapeXml
};

struct CameraConversionOptions
{
    CameraFormat format = CameraFormat::Auto;
    std::filesystem::path inputPath;
    std::filesystem::path outputDir;
    std::string datasetId;
    bool overwrite = false;
};

struct CameraConversionResult
{
    bool success = false;
    std::string errorMessage;
    CameraFormat inputFormat = CameraFormat::Auto;
    std::string datasetId;
    std::filesystem::path sourceDir;
    std::filesystem::path outputDir;
    std::filesystem::path imageCameraList;
    std::filesystem::path summaryPath;
    int cameraCount = 0;
    std::vector<std::string> warnings;
    std::vector<std::filesystem::path> writtenCameraFiles;
};

std::vector<std::string> supportedFormatNames();
std::string cameraFormatName(CameraFormat format);
std::string cameraFormatDescription(CameraFormat format);
std::optional<CameraFormat> parseCameraFormat(const std::string &name);

CameraConversionResult convertCameraDataset(const CameraConversionOptions &options);

} // namespace xjw::camera
