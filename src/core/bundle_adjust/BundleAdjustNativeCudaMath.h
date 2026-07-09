#pragma once

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct ProjectionResult
{
    bool ok = false;
    double pixel[2] = {0.0, 0.0};
};

struct ObservationLinearization
{
    double residual[2] = {0.0, 0.0};
    double jc[12] = {0.0};
    double jp[6] = {0.0};
    double weightedCost = 0.0;
};

ProjectionResult projectHost(const HostCamera &camera, const std::array<double, 3> &point);

bool linearizeObservationHost(const HostCamera &camera,
                              const std::array<double, 3> &point,
                              double observedU,
                              double observedV,
                              double weight,
                              double huberDelta,
                              ObservationLinearization *out);

} // namespace xjw::detail::native_cuda
