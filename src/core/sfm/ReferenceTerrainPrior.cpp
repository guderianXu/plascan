#include "ReferenceTerrainPrior.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
namespace
{

bool isValidGrid(const ReferenceTerrainGrid &grid)
{
    return grid.width > 0
        && grid.height > 0
        && std::isfinite(grid.originX)
        && std::isfinite(grid.originY)
        && std::isfinite(grid.pixelSizeX)
        && std::isfinite(grid.pixelSizeY)
        && std::abs(grid.pixelSizeX) > 1e-12
        && std::abs(grid.pixelSizeY) > 1e-12
        && grid.heights.size() == static_cast<std::size_t>(grid.width * grid.height);
}

bool isValidHeight(const ReferenceTerrainGrid &grid, double value)
{
    if (!std::isfinite(value))
    {
        return false;
    }
    return !std::isfinite(grid.nodata) || std::abs(value - grid.nodata) > 1e-12;
}

double heightAt(const ReferenceTerrainGrid &grid, int row, int col)
{
    return grid.heights[static_cast<std::size_t>(row * grid.width + col)];
}

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
    {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

} // namespace

double ReferenceTerrainPrior::sampleHeight(const ReferenceTerrainGrid &grid,
                                           double x,
                                           double y,
                                           bool *ok)
{
    if (ok)
    {
        *ok = false;
    }

    if (!isValidGrid(grid) || !std::isfinite(x) || !std::isfinite(y))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double col = (x - grid.originX) / grid.pixelSizeX;
    const double row = (y - grid.originY) / grid.pixelSizeY;
    if (col < 0.0 || row < 0.0 ||
        col > static_cast<double>(grid.width - 1) ||
        row > static_cast<double>(grid.height - 1))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int col0 = std::clamp(static_cast<int>(std::floor(col)), 0, grid.width - 1);
    const int row0 = std::clamp(static_cast<int>(std::floor(row)), 0, grid.height - 1);
    const int col1 = std::min(col0 + 1, grid.width - 1);
    const int row1 = std::min(row0 + 1, grid.height - 1);
    const double tx = col - static_cast<double>(col0);
    const double ty = row - static_cast<double>(row0);

    const double z00 = heightAt(grid, row0, col0);
    const double z10 = heightAt(grid, row0, col1);
    const double z01 = heightAt(grid, row1, col0);
    const double z11 = heightAt(grid, row1, col1);
    if (!isValidHeight(grid, z00) || !isValidHeight(grid, z10) ||
        !isValidHeight(grid, z01) || !isValidHeight(grid, z11))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double z0 = z00 * (1.0 - tx) + z10 * tx;
    const double z1 = z01 * (1.0 - tx) + z11 * tx;
    const double z = z0 * (1.0 - ty) + z1 * ty;
    if (ok)
    {
        *ok = true;
    }
    return z;
}

ReferenceTerrainPriorStats ReferenceTerrainPrior::attachHeightPlaneConstraints(
    const ReferenceTerrainGrid &grid,
    std::vector<BATrack> *tracks,
    const ReferenceTerrainPriorOptions &options)
{
    ReferenceTerrainPriorStats stats;
    if (!tracks)
    {
        return stats;
    }

    stats.inputTrackCount = static_cast<int>(tracks->size());
    if (!options.enabled || !isValidGrid(grid) || !(options.sigmaMeters > 0.0))
    {
        return stats;
    }

    std::vector<double> absoluteDistances;
    double sum2 = 0.0;
    for (BATrack &track : *tracks)
    {
        const auto &point = track.initialPoint;
        bool ok = false;
        const double height = sampleHeight(grid, point[0], point[1], &ok);
        if (!ok)
        {
            ++stats.rejectedNoHeightCount;
            continue;
        }

        const double signedDistance = point[2] - height;
        if (!std::isfinite(signedDistance))
        {
            ++stats.rejectedNoHeightCount;
            continue;
        }

        const double absDistance = std::abs(signedDistance);
        if (options.maxAssociationDistanceMeters > 0.0 &&
            absDistance > options.maxAssociationDistanceMeters)
        {
            ++stats.rejectedByDistanceCount;
            continue;
        }

        BALaserPlaneConstraint constraint;
        constraint.point = {{point[0], point[1], height}};
        constraint.normal = {{0.0, 0.0, 1.0}};
        constraint.weight = 1.0 / options.sigmaMeters;
        constraint.initialSignedDistance = signedDistance;
        track.laserPlaneConstraints.push_back(constraint);

        ++stats.associatedTrackCount;
        absoluteDistances.push_back(absDistance);
        sum2 += signedDistance * signedDistance;
    }

    if (!absoluteDistances.empty())
    {
        stats.rmsBeforeMeters = std::sqrt(sum2 / static_cast<double>(absoluteDistances.size()));
        stats.medianAbsBeforeMeters = median(absoluteDistances);
    }
    return stats;
}

BAOptions ReferenceTerrainPrior::makeBundleAdjustOptions(const ReferenceTerrainPriorOptions &options)
{
    BAOptions baOptions;
    baOptions.enableLaserPlaneConstraints = options.enabled && options.sigmaMeters > 0.0;
    baOptions.laserPlaneWeight = baOptions.enableLaserPlaneConstraints ? (1.0 / options.sigmaMeters) : 0.0;
    baOptions.laserHuberDeltaMeters = options.huberDeltaMeters > 0.0 ? options.huberDeltaMeters : baOptions.laserHuberDeltaMeters;
    baOptions.refineCameraPose = true;
    return baOptions;
}

} // namespace xjw
