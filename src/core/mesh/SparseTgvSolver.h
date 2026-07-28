#pragma once

#include "AdaptiveTsdfOctree.h"

#include <cstdint>
#include <functional>

namespace xjw::mesh
{

struct SparseTgvOptions
{
    int maximumIterations = 80;
    int minimumIterations = 20;
    float firstOrderWeight = 0.08f;
    float secondOrderWeight = 0.04f;
    float dataFidelity = 0.20f;
    float primalStep = 0.12f;
    float dualStep = 0.12f;
    float extrapolation = 1.0f;
    float convergenceTolerance = 1.0e-4f;
};

struct SparseTgvStatistics
{
    bool executed = false;
    bool cancelled = false;
    int iterationCount = 0;
    std::uint64_t activeNodeCount = 0;
    double initialMeanAbsoluteCurvature = 0.0;
    double finalMeanAbsoluteCurvature = 0.0;
    double finalMeanAbsoluteUpdate = 0.0;
    std::int64_t elapsedMs = 0;
};

class SparseTgvSolver
{
public:
    static SparseTgvStatistics solve(
        const SparseTgvOptions &options,
        AdaptiveTsdfOctreeResult *octree,
        const std::function<bool()> &isCancelled = {},
        const std::function<void(int, int)> &progress = {});
};

} // namespace xjw::mesh
