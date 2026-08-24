#pragma once

#include "BundleAdjustSolver.h"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace xjw::ba_benchmark
{

struct BenchmarkDataset
{
    std::vector<FramePinholeCamera> cameras;
    std::vector<BATrack> tracks;
    std::size_t observations = 0;
};

BenchmarkDataset loadRealDataset(const std::filesystem::path &datasetJson,
                                 const std::filesystem::path &cameraList);

} // namespace xjw::ba_benchmark
