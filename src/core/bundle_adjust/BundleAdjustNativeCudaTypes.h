#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw::detail::native_cuda
{

struct HostCamera
{
    std::array<double, 9> cameraToWorldRotation{};
    std::array<double, 3> cameraCenter{};
    double focalX = 1.0;
    double focalY = 1.0;
    double principalX = 0.0;
    double principalY = 0.0;
    double radialK1 = 0.0;
    double radialK2 = 0.0;
    double radialK3 = 0.0;
    double tangentialP1 = 0.0;
    double tangentialP2 = 0.0;
    int uAxisSign = 1;
    int vAxisSign = 1;
    int depthAxisFlipped = 0;
    int fixed = 0;
    int originalIndex = -1;
};

struct HostPoint
{
    std::array<double, 3> xyz{};
    int originalTrackIndex = -1;
    int observationBegin = 0;
    int observationCount = 0;
};

struct HostObservation
{
    int cameraIndex = -1;
    int pointIndex = -1;
    double u = 0.0;
    double v = 0.0;
    double weight = 1.0;
};

struct Workset
{
    std::vector<HostCamera> cameras;
    std::vector<HostPoint> points;
    std::vector<HostObservation> observations;
    std::vector<int> originalTrackToPoint;
    std::string rejectionReason;
};

struct HostSolveSummary
{
    bool ok = false;
    double meanRmsBefore = 0.0;
    double meanRmsAfter = 0.0;
    int optimizedTracks = 0;
    int pcgIterations = 0;
    double linearResidual = 0.0;
    int acceptedSteps = 0;
    int rejectedSteps = 0;
    std::string message;
};

} // namespace xjw::detail::native_cuda
