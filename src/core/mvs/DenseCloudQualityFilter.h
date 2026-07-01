#pragma once

#include <cstddef>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>

namespace xjw::mvs
{

using DensePointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct TerrainHeightSpikeFilterOptions
{
    bool enabled = true;
    int gridResolution = 160;
    int minCellPoints = 8;
    float minHeightThreshold = 0.35f;
    float madMultiplier = 6.0f;
    bool localPlaneFilterEnabled = true;
    int localPlaneMinPoints = 12;
    float localPlaneMinResidualThreshold = 0.12f;
    float localPlaneMadMultiplier = 4.0f;
};

struct TerrainHeightSpikeFilterReport
{
    std::size_t inputPoints = 0;
    std::size_t outputPoints = 0;
    std::size_t removedPoints = 0;
    double medianCellZRangeBefore = 0.0;
    double p95CellZRangeBefore = 0.0;
    double medianCellZRangeAfter = 0.0;
    double p95CellZRangeAfter = 0.0;
    std::size_t localPlaneRemovedPoints = 0;
};

DensePointCloud filterTerrainHeightSpikes(const DensePointCloud &cloud,
                                          const TerrainHeightSpikeFilterOptions &options,
                                          TerrainHeightSpikeFilterReport *report = nullptr);

} // namespace xjw::mvs
