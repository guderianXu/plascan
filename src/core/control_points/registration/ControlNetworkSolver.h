#pragma once

#include "model/MarkerRoles.h"

#include <array>
#include <string>
#include <vector>

namespace xjw::control_points
{

struct SimilarityTransform3D
{
    bool valid = false;
    double scale = 1.0;
    std::array<double, 9> rotation{{1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0}};
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};

    std::array<double, 3> apply(const std::array<double, 3> &point) const;
    std::array<double, 9> rotate(const std::array<double, 9> &cameraToWorldRotation) const;
};

struct ControlNetworkPoint
{
    std::string markerId;
    MarkerRole role = MarkerRole::TieMarker;
    bool enabled = true;
    std::array<double, 3> estimatedPoint{{0.0, 0.0, 0.0}};
    std::array<double, 3> referencePoint{{0.0, 0.0, 0.0}};
    std::array<double, 3> sigma{{1.0, 1.0, 1.0}};
};

struct MarkerResidual
{
    std::string markerId;
    MarkerRole role = MarkerRole::TieMarker;
    std::array<double, 3> delta{{0.0, 0.0, 0.0}};
    double total = 0.0;
    double normalized = 0.0;
    bool inlier = false;
};

struct ControlNetworkOptions
{
    double inlierThreshold = 0.10;
    double sigmaMultiplier = 3.0;
    double minimumTriangleAreaRatio = 1.0e-6;
    int maximumHypotheses = 512;
};

struct ControlNetworkInput
{
    std::vector<ControlNetworkPoint> points;
    ControlNetworkOptions options;
};

struct ControlNetworkResult
{
    bool ok = false;
    SimilarityTransform3D transform;
    std::vector<MarkerResidual> controlResiduals;
    std::vector<MarkerResidual> checkPointResiduals;
    int controlInlierCount = 0;
    double controlInlierRms = 0.0;
    std::string error;
};

ControlNetworkResult solveControlNetwork(const ControlNetworkInput &input);

} // namespace xjw::control_points
