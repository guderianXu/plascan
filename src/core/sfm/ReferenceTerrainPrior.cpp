/**
 * @file ReferenceTerrainPrior.cpp
 * @brief 参考高程栅格采样、轨迹关联和 BA 软约束装配实现。
 *
 * 本实现假设点坐标与栅格仿射参数处于同一坐标系，只生成法向为世界 Z 的水平面
 * 约束。它不执行 CRS 转换，也不把 DEM 当作硬约束。
 */

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

    // 该表达式同时支持正、负像元步长；边界判断在连续栅格坐标中完成。
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

    // 四个角任一为 nodata 时拒绝样本，避免跨越 DEM 空洞插值出虚假表面。
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
    // 关联只使用 BA 初始点。距离门控防止错误稀疏点被地形先验强行吸附。
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

        // 当前地形近似为采样点处水平切平面；未来支持坡度时可替换 normal。
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
