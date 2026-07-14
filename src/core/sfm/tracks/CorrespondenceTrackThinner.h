#pragma once

#include <cstddef>

namespace xjw
{

class CorrespondenceGraph;
class SfmReconstruction;

struct CorrespondenceTrackThinningOptions
{
    int maxTracksPerImage = 0;
    int maxTracksPerGridCell = 0;
    int gridColumns = 8;
    int gridRows = 8;
};

struct CorrespondenceTrackThinningResult
{
    int inputTrackCount = 0;
    int retainedTrackCount = 0;
    int prunedTrackCount = 0;
    std::size_t removedMatchCount = 0;
    std::size_t retainedMatchCount = 0;
};

CorrespondenceTrackThinningResult thinCorrespondenceTracks(
    const SfmReconstruction &reconstruction,
    CorrespondenceGraph *graph,
    const CorrespondenceTrackThinningOptions &options);

} // namespace xjw
