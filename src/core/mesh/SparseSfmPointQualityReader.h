#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct SparseSfmPointQuality
{
    std::array<float, 3> point{0.0f, 0.0f, 0.0f};
    int trackLength = -1;
    float rmsReprojectionPixels = -1.0f;
    float triangulationAngleDegrees = -1.0f;
    bool hasPoint = false;
    bool hasRequiredQuality = false;
};

struct SparseSfmPointQualityReadResult
{
    std::vector<SparseSfmPointQuality> points;
    std::string error;
    std::size_t bytesRead = 0;
};

/**
 * @brief Streams the large SfM point-quality JSON without materializing observations.
 */
class SparseSfmPointQualityReader
{
public:
    static SparseSfmPointQualityReadResult read(
        const std::filesystem::path &path);
};

} // namespace xjw::mesh
