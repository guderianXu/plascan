#pragma once

#include "BundleAdjust.h"
#include "graph/CovisibilityPartitioner.h"
#include "reconstruction/SfmReconstruction.h"

#include <cstddef>
#include <vector>

namespace xjw::hierarchical_ba_detail
{

struct BlockOutcome
{
    std::size_t blockIndex = 0;
    std::vector<ImageId> cameraIds;
    std::vector<Point3DId> pointIds;
    int fixedTrackCount = 0;
    BAResult result;
    bool accepted = false;
};

BlockOutcome solveBlock(std::size_t blockIndex,
                        const CovisibilityBlock &block,
                        const SfmReconstruction &reconstruction,
                        const BAOptions &baseOptions,
                        int threadsPerBlock,
                        bool useCeres);

} // namespace xjw::hierarchical_ba_detail
