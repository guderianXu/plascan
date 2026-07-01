#include "DenseCloudQualityFilter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace xjw::mvs
{

namespace
{

struct Bounds2D
{
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    bool valid = false;
};

struct CellStats
{
    std::vector<float> zValues;
    float median = 0.0f;
    float robustSigma = 0.0f;
};

struct PlaneFit
{
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    bool valid = false;
};

double medianSorted(const std::vector<float> &values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    if ((values.size() % 2) == 1)
    {
        return values[mid];
    }
    return 0.5 * (static_cast<double>(values[mid - 1]) + static_cast<double>(values[mid]));
}

double percentileSorted(const std::vector<float> &values, double percentile)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 100.0);
    const double position = (clamped / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(values.size() - 1, lo + 1);
    const double t = position - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - t) + static_cast<double>(values[hi]) * t;
}

bool finitePoint(const DensePointCloud &cloud, std::size_t index)
{
    const plamatrix::Index row = static_cast<plamatrix::Index>(index);
    return std::isfinite(cloud.points().getValue(row, 0)) &&
           std::isfinite(cloud.points().getValue(row, 1)) &&
           std::isfinite(cloud.points().getValue(row, 2));
}

Bounds2D computeBounds(const DensePointCloud &cloud)
{
    Bounds2D bounds;
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        if (!finitePoint(cloud, i))
        {
            continue;
        }

        const plamatrix::Index row = static_cast<plamatrix::Index>(i);
        const float x = cloud.points().getValue(row, 0);
        const float y = cloud.points().getValue(row, 1);
        bounds.minX = std::min(bounds.minX, x);
        bounds.minY = std::min(bounds.minY, y);
        bounds.maxX = std::max(bounds.maxX, x);
        bounds.maxY = std::max(bounds.maxY, y);
        bounds.valid = true;
    }
    return bounds;
}

int cellIndexForPoint(const DensePointCloud &cloud,
                      std::size_t index,
                      const Bounds2D &bounds,
                      int gridResolution)
{
    const float spanX = std::max(bounds.maxX - bounds.minX, 1.0e-6f);
    const float spanY = std::max(bounds.maxY - bounds.minY, 1.0e-6f);
    const plamatrix::Index row = static_cast<plamatrix::Index>(index);
    const float x = cloud.points().getValue(row, 0);
    const float y = cloud.points().getValue(row, 1);
    const int ix = std::clamp(static_cast<int>(std::floor((x - bounds.minX) / spanX * gridResolution)),
                              0,
                              gridResolution - 1);
    const int iy = std::clamp(static_cast<int>(std::floor((y - bounds.minY) / spanY * gridResolution)),
                              0,
                              gridResolution - 1);
    return iy * gridResolution + ix;
}

std::vector<int> finiteIndices(const DensePointCloud &cloud)
{
    std::vector<int> indices;
    indices.reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        if (finitePoint(cloud, i))
        {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

DensePointCloud gatherPointCloudByIndices(const DensePointCloud &cloud, const std::vector<int> &indices)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(indices.size()), 3);
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        const plamatrix::Index src = static_cast<plamatrix::Index>(indices[i]);
        const plamatrix::Index dst = static_cast<plamatrix::Index>(i);
        points(dst, 0) = cloud.points().getValue(src, 0);
        points(dst, 1) = cloud.points().getValue(src, 1);
        points(dst, 2) = cloud.points().getValue(src, 2);
    }

    DensePointCloud output(std::move(points));

    if (cloud.hasColors())
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
            static_cast<plamatrix::Index>(indices.size()), 3);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            const plamatrix::Index src = static_cast<plamatrix::Index>(indices[i]);
            const plamatrix::Index dst = static_cast<plamatrix::Index>(i);
            colors(dst, 0) = cloud.colors()->getValue(src, 0);
            colors(dst, 1) = cloud.colors()->getValue(src, 1);
            colors(dst, 2) = cloud.colors()->getValue(src, 2);
        }
        output.setColors(std::move(colors));
    }

    if (cloud.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(
            static_cast<plamatrix::Index>(indices.size()), 3);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            const plamatrix::Index src = static_cast<plamatrix::Index>(indices[i]);
            const plamatrix::Index dst = static_cast<plamatrix::Index>(i);
            normals(dst, 0) = cloud.normals()->getValue(src, 0);
            normals(dst, 1) = cloud.normals()->getValue(src, 1);
            normals(dst, 2) = cloud.normals()->getValue(src, 2);
        }
        output.setNormals(std::move(normals));
    }

    if (cloud.hasIntensities())
    {
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(
            static_cast<plamatrix::Index>(indices.size()), 1);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            const plamatrix::Index src = static_cast<plamatrix::Index>(indices[i]);
            intensities(static_cast<plamatrix::Index>(i), 0) = cloud.intensities()->getValue(src, 0);
        }
        output.setIntensities(std::move(intensities));
    }

    if (cloud.hasScalarFields())
    {
        const int cols = cloud.scalarFields()->cols();
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> fields(
            static_cast<plamatrix::Index>(indices.size()), cols);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            const plamatrix::Index src = static_cast<plamatrix::Index>(indices[i]);
            const plamatrix::Index dst = static_cast<plamatrix::Index>(i);
            for (int col = 0; col < cols; ++col)
            {
                fields(dst, col) = cloud.scalarFields()->getValue(src, col);
            }
        }
        output.setScalarFields(cloud.scalarFieldNames(), std::move(fields));
    }

    return output;
}

void summarizeCellZRanges(const DensePointCloud &cloud,
                          const std::vector<int> &indices,
                          const Bounds2D &bounds,
                          int gridResolution,
                          int minCellPoints,
                          double *medianRange,
                          double *p95Range)
{
    std::vector<CellStats> cells(static_cast<std::size_t>(gridResolution * gridResolution));
    for (int idx : indices)
    {
        const int cell = cellIndexForPoint(cloud, static_cast<std::size_t>(idx), bounds, gridResolution);
        cells[static_cast<std::size_t>(cell)].zValues.push_back(
            cloud.points().getValue(static_cast<plamatrix::Index>(idx), 2));
    }

    std::vector<float> ranges;
    ranges.reserve(cells.size());
    for (CellStats &cell : cells)
    {
        if (static_cast<int>(cell.zValues.size()) < minCellPoints)
        {
            continue;
        }
        auto minmax = std::minmax_element(cell.zValues.begin(), cell.zValues.end());
        ranges.push_back(*minmax.second - *minmax.first);
    }

    std::sort(ranges.begin(), ranges.end());
    if (medianRange)
    {
        *medianRange = medianSorted(ranges);
    }
    if (p95Range)
    {
        *p95Range = percentileSorted(ranges, 95.0);
    }
}

void fillCellRobustStats(std::vector<CellStats> *cells)
{
    if (!cells)
    {
        return;
    }

    for (CellStats &cell : *cells)
    {
        if (cell.zValues.empty())
        {
            continue;
        }

        std::sort(cell.zValues.begin(), cell.zValues.end());
        cell.median = static_cast<float>(medianSorted(cell.zValues));

        std::vector<float> deviations;
        deviations.reserve(cell.zValues.size());
        for (float z : cell.zValues)
        {
            deviations.push_back(std::abs(z - cell.median));
        }
        std::sort(deviations.begin(), deviations.end());
        cell.robustSigma = static_cast<float>(1.4826 * medianSorted(deviations));
    }
}

bool solvePlaneFromIndices(const DensePointCloud &cloud,
                           const std::vector<int> &indices,
                           PlaneFit *plane)
{
    if (!plane || indices.size() < 3)
    {
        return false;
    }

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;
    double sxz = 0.0;
    double syz = 0.0;
    for (int idx : indices)
    {
        const plamatrix::Index row = static_cast<plamatrix::Index>(idx);
        const double x = cloud.points().getValue(row, 0);
        const double y = cloud.points().getValue(row, 1);
        const double z = cloud.points().getValue(row, 2);
        sx += x;
        sy += y;
        sz += z;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxz += x * z;
        syz += y * z;
    }

    const double n = static_cast<double>(indices.size());
    const double det = sxx * (syy * n - sy * sy) -
                       sxy * (sxy * n - sx * sy) +
                       sx * (sxy * sy - syy * sx);
    if (std::abs(det) < 1.0e-12)
    {
        return false;
    }

    const double detA = sxz * (syy * n - sy * sy) -
                        sxy * (syz * n - sy * sz) +
                        sx * (syz * sy - syy * sz);
    const double detB = sxx * (syz * n - sy * sz) -
                        sxz * (sxy * n - sx * sy) +
                        sx * (sxy * sz - syz * sx);
    const double detC = sxx * (syy * sz - syz * sy) -
                        sxy * (sxy * sz - syz * sx) +
                        sxz * (sxy * sy - syy * sx);

    plane->a = static_cast<float>(detA / det);
    plane->b = static_cast<float>(detB / det);
    plane->c = static_cast<float>(detC / det);
    plane->valid = std::isfinite(plane->a) && std::isfinite(plane->b) && std::isfinite(plane->c);
    return plane->valid;
}

float planeResidual(const DensePointCloud &cloud, int index, const PlaneFit &plane)
{
    const plamatrix::Index row = static_cast<plamatrix::Index>(index);
    const float x = cloud.points().getValue(row, 0);
    const float y = cloud.points().getValue(row, 1);
    const float z = cloud.points().getValue(row, 2);
    return std::abs(z - (plane.a * x + plane.b * y + plane.c));
}

std::vector<int> neighborCellIndices(int cellIndex, int gridResolution)
{
    std::vector<int> cells;
    cells.reserve(9);
    const int centerX = cellIndex % gridResolution;
    const int centerY = cellIndex / gridResolution;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int x = centerX + dx;
            const int y = centerY + dy;
            if (x < 0 || y < 0 || x >= gridResolution || y >= gridResolution)
            {
                continue;
            }
            cells.push_back(y * gridResolution + x);
        }
    }
    return cells;
}

std::vector<int> filterLocalPlaneResiduals(const DensePointCloud &cloud,
                                           const std::vector<int> &indices,
                                           const Bounds2D &bounds,
                                           const TerrainHeightSpikeFilterOptions &options,
                                           int gridResolution,
                                           std::size_t *removedPoints)
{
    if (removedPoints)
    {
        *removedPoints = 0;
    }
    if (!options.localPlaneFilterEnabled || indices.empty())
    {
        return indices;
    }

    const int minPlanePoints = std::max(3, options.localPlaneMinPoints);
    std::vector<std::vector<int>> cellIndices(static_cast<std::size_t>(gridResolution * gridResolution));
    for (int idx : indices)
    {
        const int cell = cellIndexForPoint(cloud, static_cast<std::size_t>(idx), bounds, gridResolution);
        cellIndices[static_cast<std::size_t>(cell)].push_back(idx);
    }

    std::vector<int> kept;
    kept.reserve(indices.size());
    for (int cell = 0; cell < gridResolution * gridResolution; ++cell)
    {
        const std::vector<int> &currentCell = cellIndices[static_cast<std::size_t>(cell)];
        if (currentCell.empty())
        {
            continue;
        }

        std::vector<int> supportIndices;
        for (int neighborCell : neighborCellIndices(cell, gridResolution))
        {
            const std::vector<int> &neighbor = cellIndices[static_cast<std::size_t>(neighborCell)];
            supportIndices.insert(supportIndices.end(), neighbor.begin(), neighbor.end());
        }
        if (static_cast<int>(supportIndices.size()) < minPlanePoints)
        {
            kept.insert(kept.end(), currentCell.begin(), currentCell.end());
            continue;
        }

        PlaneFit plane;
        if (!solvePlaneFromIndices(cloud, supportIndices, &plane))
        {
            kept.insert(kept.end(), currentCell.begin(), currentCell.end());
            continue;
        }

        std::vector<float> residuals;
        residuals.reserve(supportIndices.size());
        for (int idx : supportIndices)
        {
            residuals.push_back(planeResidual(cloud, idx, plane));
        }
        std::sort(residuals.begin(), residuals.end());
        const float medianResidual = static_cast<float>(medianSorted(residuals));

        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (float residual : residuals)
        {
            deviations.push_back(std::abs(residual - medianResidual));
        }
        std::sort(deviations.begin(), deviations.end());
        const float robustSigma = static_cast<float>(1.4826 * medianSorted(deviations));
        const float threshold = std::max(options.localPlaneMinResidualThreshold,
                                         options.localPlaneMadMultiplier * robustSigma);

        for (int idx : currentCell)
        {
            if (planeResidual(cloud, idx, plane) <= threshold)
            {
                kept.push_back(idx);
            }
        }
    }

    if (removedPoints)
    {
        *removedPoints = indices.size() - kept.size();
    }
    return kept;
}

} // namespace

DensePointCloud filterTerrainHeightSpikes(const DensePointCloud &cloud,
                                          const TerrainHeightSpikeFilterOptions &options,
                                          TerrainHeightSpikeFilterReport *report)
{
    TerrainHeightSpikeFilterReport localReport;
    localReport.inputPoints = cloud.size();

    const std::vector<int> finite = finiteIndices(cloud);
    const Bounds2D bounds = computeBounds(cloud);
    const int gridResolution = std::clamp(options.gridResolution, 1, 1024);
    const int minCellPoints = std::max(1, options.minCellPoints);

    if (!bounds.valid || !options.enabled || cloud.size() == 0)
    {
        DensePointCloud output = gatherPointCloudByIndices(cloud, finite);
        localReport.outputPoints = output.size();
        localReport.removedPoints = localReport.inputPoints - localReport.outputPoints;
        if (report)
        {
            *report = localReport;
        }
        return output;
    }

    summarizeCellZRanges(cloud,
                         finite,
                         bounds,
                         gridResolution,
                         minCellPoints,
                         &localReport.medianCellZRangeBefore,
                         &localReport.p95CellZRangeBefore);

    std::vector<CellStats> cells(static_cast<std::size_t>(gridResolution * gridResolution));
    for (int idx : finite)
    {
        const int cell = cellIndexForPoint(cloud, static_cast<std::size_t>(idx), bounds, gridResolution);
        cells[static_cast<std::size_t>(cell)].zValues.push_back(
            cloud.points().getValue(static_cast<plamatrix::Index>(idx), 2));
    }
    fillCellRobustStats(&cells);

    std::vector<int> kept;
    kept.reserve(finite.size());
    for (int idx : finite)
    {
        const int cellIndex = cellIndexForPoint(cloud, static_cast<std::size_t>(idx), bounds, gridResolution);
        const CellStats &cell = cells[static_cast<std::size_t>(cellIndex)];
        if (static_cast<int>(cell.zValues.size()) < minCellPoints)
        {
            kept.push_back(idx);
            continue;
        }

        const float z = cloud.points().getValue(static_cast<plamatrix::Index>(idx), 2);
        const float threshold = std::max(options.minHeightThreshold, options.madMultiplier * cell.robustSigma);
        if (std::abs(z - cell.median) <= threshold)
        {
            kept.push_back(idx);
        }
    }

    kept = filterLocalPlaneResiduals(cloud,
                                     kept,
                                     bounds,
                                     options,
                                     gridResolution,
                                     &localReport.localPlaneRemovedPoints);

    summarizeCellZRanges(cloud,
                         kept,
                         bounds,
                         gridResolution,
                         minCellPoints,
                         &localReport.medianCellZRangeAfter,
                         &localReport.p95CellZRangeAfter);

    DensePointCloud output = gatherPointCloudByIndices(cloud, kept);
    localReport.outputPoints = output.size();
    localReport.removedPoints = localReport.inputPoints - localReport.outputPoints;
    if (report)
    {
        *report = localReport;
    }
    return output;
}

} // namespace xjw::mvs
