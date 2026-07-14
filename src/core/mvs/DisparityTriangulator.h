#pragma once

#include "camera/Camera.h"
#include "MvsTypes.h"
#include <opencv2/core.hpp>
#include <array>
#include <string>

namespace xjw
{
namespace mvs
{

struct TriangulationConfig
{
    float maxTriangulationError = 0.01f;
    int numThreads = 0;  // 0=auto
    bool transposed = false;
};

struct TriangulationResult
{
    cv::Mat pointCloud;   // CV_64FC3: X,Y,Z per pixel
    cv::Mat errorMap;     // CV_32F: ray miss distance per pixel
    cv::Mat validMask;    // CV_8U: 255=valid
    std::array<double, 3> pointOffset = {0, 0, 0};
    int totalPixels = 0;
    int validPoints = 0;
    double medianError = 0.0;
};

class DisparityTriangulator
{
public:
    // Disparity-based triangulation (original)
    static TriangulationResult triangulate(
        const cv::Mat &disparity,
        const cv::Mat &validMask,
        const cv::Mat &H1inv,
        const cv::Mat &H2inv,
        const Camera &camL,
        const Camera &camR,
        const TriangulationConfig &cfg = {});

    // Depth-based triangulation: unproject rectified depth using camera model
    static TriangulationResult triangulateFromDepth(
        const cv::Mat &depthMap,
        const cv::Mat &validMask,
        const cv::Mat &H1inv,
        const Camera &camL,
        const Camera &camR,
        const Camera &rectCamL,
        const TriangulationConfig &cfg = {});
};

} // namespace mvs
} // namespace xjw
