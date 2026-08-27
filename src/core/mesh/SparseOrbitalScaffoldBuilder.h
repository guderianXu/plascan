#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct SparseOrbitalScaffoldOptions
{
    std::size_t minimumPointCount = 32;
    std::size_t maximumPointCount = 80000;
    std::size_t minimumOutlierSampleCount = 16;
    float voxelSize = 0.0f;
    float voxelSizeBoundingDiagonalFraction = 0.0015f;
    float maximumRadiusMedianFactor = 4.0f;
    float radialMadMultiplier = 8.0f;
    bool enableStatisticalOutlierRemoval = true;
    int statisticalOutlierNeighborCount = 20;
    float statisticalOutlierStdDevMultiplier = 2.0f;
    int minimumTrackLength = 3;
    float maximumRmsReprojectionPixels = 1.5f;
    float minimumTriangulationAngleDegrees = 5.0f;
};

struct SparseOrbitalScaffoldStatistics
{
    std::size_t inputPointCount = 0;
    std::size_t sidecarPointCount = 0;
    std::size_t sidecarMatchedPointCount = 0;
    std::size_t sidecarUnmatchedPointCount = 0;
    bool sidecarSubsetAlignmentUsed = false;
    std::size_t nonFiniteRejectedCount = 0;
    std::size_t missingQualityRejectedCount = 0;
    std::size_t qualityRejectedCount = 0;
    std::size_t radialOutlierRejectedCount = 0;
    std::size_t statisticalOutlierRejectedCount = 0;
    std::size_t downsampledPointCount = 0;
    std::size_t outputPointCount = 0;
};

struct SparseOrbitalScaffoldResult
{
    std::filesystem::path sourcePath;
    std::filesystem::path qualitySidecarPath;
    std::vector<std::array<float, 3>> points;
    std::vector<std::array<float, 3>> normals;
    std::array<float, 3> robustCenter{0.0f, 0.0f, 0.0f};
    SparseOrbitalScaffoldStatistics statistics;
    std::string error;

    bool succeeded() const
    {
        return error.empty() && !points.empty() && points.size() == normals.size();
    }
};

/**
 * @brief Builds a sparse, outward-oriented support scaffold for closed orbital reconstruction.
 *
 * The scaffold is an independent Screened Poisson input. It is deliberately not
 * injected into TSDF samples because sparse SfM observations do not carry the same
 * free-space or measured-surface semantics as depth pixels.
 */
class SparseOrbitalScaffoldBuilder
{
public:
    static std::filesystem::path resolveSparsePly(
        const std::filesystem::path &mvsOutputPath,
        const std::filesystem::path &explicitSparsePly = {},
        std::string *error = nullptr);

    static SparseOrbitalScaffoldResult build(
        const std::filesystem::path &mvsOutputPath,
        const std::filesystem::path &explicitSparsePly = {},
        const std::filesystem::path &explicitPointsJson = {},
        const SparseOrbitalScaffoldOptions &options = {});

    static std::filesystem::path resolveSparsePointsJson(
        const std::filesystem::path &mvsOutputPath,
        const std::filesystem::path &sparsePly,
        const std::filesystem::path &explicitPointsJson = {},
        std::string *error = nullptr);
};

} // namespace xjw::mesh
