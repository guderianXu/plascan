#pragma once

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <limits>
#include <vector>

namespace xjw
{

struct PairIntersectionCandidate
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    double rmsReprojectionPx = std::numeric_limits<double>::infinity();
    bool valid = false;
};

double minimumTriangulationAngleDeg(const std::vector<Camera> &cameras,
                                    const BATrack &track,
                                    const std::array<double, 3> &worldPoint);

double pairRmsReprojectionErrorPx(const Camera &cameraA,
                                  const std::array<double, 2> &pixelA,
                                  const Camera &cameraB,
                                  const std::array<double, 2> &pixelB,
                                  const std::array<double, 3> &worldPoint);

PairIntersectionCandidate triangulatePairWithDirectionFallback(
    const Camera &cameraA,
    const std::array<double, 2> &pixelA,
    const Camera &cameraB,
    const std::array<double, 2> &pixelB);

} // namespace xjw
