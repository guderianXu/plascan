#pragma once

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/xfeatures2d.hpp>
#define PLASCAN_OPENCV_ZERO_DISPARITY cv::STEREO_ZERO_DISPARITY
#else
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#define PLASCAN_OPENCV_ZERO_DISPARITY cv::CALIB_ZERO_DISPARITY
#endif

namespace xjw::opencv_compat
{

#if CV_VERSION_MAJOR >= 5
using AkazeFeature = cv::xfeatures2d::AKAZE;
#else
using AkazeFeature = cv::AKAZE;
#endif

inline cv::Mat findEssentialMat(cv::InputArray points1,
                                cv::InputArray points2,
                                cv::InputArray cameraMatrix,
                                int method,
                                double prob,
                                double threshold,
                                cv::OutputArray mask)
{
#if CV_VERSION_MAJOR >= 5
    return cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, 1000, mask);
#else
    return cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask);
#endif
}

} // namespace xjw::opencv_compat
