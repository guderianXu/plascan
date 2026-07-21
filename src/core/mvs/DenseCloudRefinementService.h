#pragma once

#include "DenseCloudQualityFilter.h"

#include <string>
#include <vector>

namespace xjw::mvs
{

struct DenseCloudRefinementRequest
{
    std::string inputPath;
    std::string outputPath;
    TerrainHeightSpikeFilterOptions filterOptions;
    int filterPasses = 2;
    int streamingChunkMb = 128;
};

struct DenseCloudRefinementResult
{
    std::string mode;
    TerrainHeightSpikeFilterReport report;
    std::vector<TerrainHeightSpikeFilterReport> passReports;
};

bool refineDenseCloud(const DenseCloudRefinementRequest &request,
                      DenseCloudRefinementResult *result,
                      std::string *errorMessage);

} // namespace xjw::mvs
