#pragma once

#include "FramePinholeCamera.h"

#include <array>
#include <vector>

namespace xjw
{

    struct ReferenceResectionResult
    {
        bool success = false;
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        std::vector<unsigned char> inlierMask;
        int numInliers = 0;
        int selectedThresholdLevel = 0;
        int ransacIterations = 0;
    };

    ReferenceResectionResult solveReferenceResection(const std::vector<std::array<double, 3>>& worldPoints,
                                                     const std::vector<std::array<double, 2>>& imagePoints,
                                                     const FramePinholeCamera& camera,
                                                     double resectionThresholdPixels);

} // namespace xjw
