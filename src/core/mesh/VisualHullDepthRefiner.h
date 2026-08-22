#pragma once

#include "MeshTypes.h"

#include <QVector>

#include <cstdint>
#include <functional>

namespace xjw::mesh
{

struct DepthTsdfFrame;

struct VisualHullDepthRefineOptions
{
    float maximumEvidenceDistance = 0.0f;
    float maximumDisplacement = 0.0f;
    int minimumViewCount = 2;
    int minimumNativeViewCount = 1;
    float minimumDepthConfidence = 0.25f;
    float repairedObservationWeight = 0.35f;
    bool enableInverseDepthSpreadWeighting = false;
    float inverseDepthSpreadWeightKnee = 0.005f;
    float inverseDepthSpreadWeightZero = 0.015f;
    float minimumInverseDepthSpreadWeight = 0.05f;
    float maximumViewMedianAbsoluteDeviation = 0.0f;
    bool enableCrossViewBiasCompensation = true;
    int minimumCrossViewBiasPairSamples = 64;
    float maximumCrossViewBias = 0.0f;
    float minimumAnchorWeight = 0.05f;
    int regularizationIterations = 30;
    float regularizationWeight = 4.0f;
    float propagationDecay = 0.90f;
    float regularizationMaximumNormalAngleDegrees = 50.0f;
    bool enableGlobalRobustSolver = false;
    int globalSolverIrlsIterations = 4;
    int globalSolverMaximumPcgIterations = 120;
    float globalSolverConvergenceTolerance = 1.0e-5f;
    float globalSolverRobustScaleMultiplier = 0.5f;
    float globalSolverLaplacianWeight = 0.45f;
    float globalSolverHullPriorWeight = 0.02f;
    bool measuredDepthSamplesOnly = false;
    bool primaryFramesOnly = false;
};

struct VisualHullDepthRefineStatistics
{
    bool applied = false;
    std::uint64_t projectedObservationCount = 0;
    std::uint64_t acceptedObservationCount = 0;
    std::uint64_t rejectedAuxiliaryObservationCount = 0;
    std::uint64_t rejectedNonMeasuredObservationCount = 0;
    std::uint64_t nearestMeasuredObservationCount = 0;
    std::uint64_t spreadDownweightedObservationCount = 0;
    std::uint64_t spreadVeryWeakObservationCount = 0;
    std::uint64_t anchoredVertexCount = 0;
    std::uint64_t blendedConsensusVertexCount = 0;
    int biasCalibratedFrameCount = 0;
    std::uint64_t biasCalibrationPairCount = 0;
    float maximumAbsoluteFrameBias = 0.0f;
    std::uint64_t displacedVertexCount = 0;
    float medianSupportingViewCount = 0.0f;
    float p90SupportingViewCount = 0.0f;
    float maximumAppliedDisplacement = 0.0f;
    float medianAppliedDisplacement = 0.0f;
    float p90AppliedDisplacement = 0.0f;
    int globalSolverAttemptCount = 0;
    int globalSolverSolvedPassCount = 0;
    int globalSolverAppliedPassCount = 0;
    int globalSolverConvergedPassCount = 0;
    int globalSolverFallbackPassCount = 0;
    bool globalSolverCancelled = false;
    int globalSolverIrlsIterationCount = 0;
    int globalSolverPcgIterationCount = 0;
    std::uint64_t globalSolverObservationCount = 0;
    std::uint64_t globalSolverRegularizationEdgeCount = 0;
    std::uint64_t globalSolverAnchoredVertexCount = 0;
    std::uint64_t globalSolverPriorOnlyVertexCount = 0;
    float globalSolverEffectiveRobustScale = 0.0f;
    double globalSolverInitialEnergy = 0.0;
    double globalSolverFinalEnergy = 0.0;
    double globalSolverFinalRelativeResidual = 0.0;
};

class VisualHullDepthRefiner
{
public:
    static VisualHullDepthRefineStatistics refine(
        TriMesh *mesh,
        const QVector<DepthTsdfFrame> &frames,
        const VisualHullDepthRefineOptions &options,
        const std::function<bool()> &isCancelled = {});
};

} // namespace xjw::mesh
