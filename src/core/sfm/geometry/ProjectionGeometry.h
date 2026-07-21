#pragma once

#include "Camera.h"

#include <array>

namespace xjw
{

struct ProjectionResult
{
    bool success = false;
    bool usedSignedFallback = false;
    std::array<double, 2> pixel{{0.0, 0.0}};
    double positiveDepth = 0.0;
};

ProjectionResult projectForReprojection(const Camera &camera,
                                        const std::array<double, 3> &worldPoint);

double reprojectionErrorPx(const Camera &camera,
                           const std::array<double, 3> &worldPoint,
                           const std::array<double, 2> &observedPixel);

} // namespace xjw
