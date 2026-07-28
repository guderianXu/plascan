#pragma once

#include "DepthTsdfSurfaceBuilder.h"

#include <QVector>

#include <cstddef>
#include <cstdint>

namespace xjw::mesh
{

struct DepthMeshFrameCompleteness
{
    int refIndex = -1;
    bool auxiliarySurfaceOnly = false;
    std::uint64_t sampledDepthPointCount = 0;
    std::uint64_t explainedDepthPointCount = 0;
    double recall = 0.0;
};

struct DepthMeshCompletenessStatistics
{
    bool available = false;
    bool gatePassed = false;
    double tolerance = 0.0;
    std::uint64_t sampledDepthPointCount = 0;
    std::uint64_t explainedDepthPointCount = 0;
    double aggregateRecall = 0.0;
    double minimumFrameRecall = 0.0;
    double p10FrameRecall = 0.0;
    double medianFrameRecall = 0.0;
    QVector<DepthMeshFrameCompleteness> frames;
};

struct DepthMeshCompletenessOptions
{
    std::size_t maximumMeshSampleCount = 400000;
    int maximumDepthSamplesPerFrame = 6000;
    double tolerance = 0.0;
    double minimumP10FrameRecall = 0.40;
    double minimumMedianFrameRecall = 0.75;
};

class DepthMeshCompleteness
{
public:
    static DepthMeshCompletenessStatistics evaluate(
        const TriMesh &mesh,
        const QVector<DepthTsdfFrame> &frames,
        const DepthMeshCompletenessOptions &options);
};

} // namespace xjw::mesh
