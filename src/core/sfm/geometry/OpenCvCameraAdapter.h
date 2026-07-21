#pragma once

#include "Camera.h"

#include <opencv2/core.hpp>

#include <array>

namespace xjw
{

cv::Mat openCvCameraMatrix(const Camera &camera, bool positiveDepthConvention);

cv::Mat openCvCameraMatrix(double focalX,
                           double focalY,
                           double principalX,
                           double principalY,
                           int uAxisSign,
                           int vAxisSign,
                           bool depthAxisFlipped,
                           bool positiveDepthConvention);

cv::Mat openCvRvecFromCameraToWorldPose(
    const std::array<double, 9> &cameraToWorldRotation,
    bool depthAxisFlipped);

cv::Mat openCvTvecFromCameraPose(
    const std::array<double, 9> &cameraToWorldRotation,
    const std::array<double, 3> &cameraCenter,
    bool depthAxisFlipped);

cv::Mat openCvProjectionMatrix(const Camera &camera);

} // namespace xjw
