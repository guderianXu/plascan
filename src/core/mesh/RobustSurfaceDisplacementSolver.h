#pragma once

#include "SurfaceReconstructor.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::mesh
{

struct RobustSurfaceDisplacementObservation
{
    int vertexIndex = -1;
    float target = 0.0f;
    float weight = 0.0f;
};

struct RobustSurfaceDisplacementOptions
{
    int irlsIterations = 4;
    int maximumPcgIterations = 120;
    float convergenceTolerance = 1.0e-5f;
    float robustScale = 1.0e-3f;
    float laplacianWeight = 0.45f;
    float hullPriorWeight = 0.02f;
    float maximumDisplacement = 0.01f;
    float minimumNormalDot = 0.50f;
};

struct RobustSurfaceDisplacementStatistics
{
    bool solved = false;
    bool converged = false;
    bool cancelled = false;
    int irlsIterationCount = 0;
    int pcgIterationCount = 0;
    std::uint64_t observationCount = 0;
    std::uint64_t regularizationEdgeCount = 0;
    std::uint64_t anchoredVertexCount = 0;
    std::uint64_t priorOnlyVertexCount = 0;
    double initialEnergy = 0.0;
    double finalEnergy = 0.0;
    double finalRelativeResidual = 0.0;
};

class RobustSurfaceDisplacementSolver
{
public:
    static RobustSurfaceDisplacementStatistics solve(
        const TriMesh &mesh,
        const std::vector<RobustSurfaceDisplacementObservation> &observations,
        const RobustSurfaceDisplacementOptions &options,
        std::vector<float> *displacement,
        const std::function<bool()> &isCancelled = {});
};

} // namespace xjw::mesh
