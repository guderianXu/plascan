#pragma once

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct KernelRunSummary
{
    bool ok = false;
    double initialCost = 0.0;
    double finalCost = 0.0;
    int activeObservations = 0;
    int pcgIterations = 0;
    double linearResidual = 0.0;
    int acceptedSteps = 0;
    int rejectedSteps = 0;
    double uploadSeconds = 0.0;
    double kernelSeconds = 0.0;
    double downloadSeconds = 0.0;
    double hostCostSeconds = 0.0;
    double deviceSelectSeconds = 0.0;
    double stagingSeconds = 0.0;
    double releaseSeconds = 0.0;
    char message[256] = {};
};

KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           int maxPcgIterations,
                                           double pcgTolerance,
                                           double huberDelta,
                                           double initialDamping);

} // namespace xjw::detail::native_cuda
