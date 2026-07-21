#include "OpenCvCameraAdapter.h"

#include <opencv2/calib3d.hpp>

namespace xjw
{
namespace
{

cv::Mat cameraToWorldMatrix(const std::array<double, 9> &rotation)
{
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            matrix.at<double>(row, column) =
                rotation[static_cast<std::size_t>(row * 3 + column)];
        }
    }
    return matrix;
}

} // namespace

cv::Mat openCvCameraMatrix(const Camera &camera, bool positiveDepthConvention)
{
    return openCvCameraMatrix(camera.focalX(),
                              camera.focalY(),
                              camera.principalX(),
                              camera.principalY(),
                              camera.uAxisSign(),
                              camera.vAxisSign(),
                              camera.depthAxisFlipped(),
                              positiveDepthConvention);
}

cv::Mat openCvCameraMatrix(double focalX,
                           double focalY,
                           double principalX,
                           double principalY,
                           int uAxisSign,
                           int vAxisSign,
                           bool depthAxisFlipped,
                           bool positiveDepthConvention)
{
    const double depthSign = positiveDepthConvention && depthAxisFlipped ? -1.0 : 1.0;
    const double fx = depthSign * (uAxisSign < 0 ? -1.0 : 1.0) * focalX;
    const double fy = depthSign * (vAxisSign < 0 ? -1.0 : 1.0) * focalY;
    return (cv::Mat_<double>(3, 3) << fx, 0.0, principalX,
                                        0.0, fy, principalY,
                                        0.0, 0.0, 1.0);
}

cv::Mat openCvRvecFromCameraToWorldPose(
    const std::array<double, 9> &cameraToWorldRotation,
    bool depthAxisFlipped)
{
    cv::Mat worldToCamera = cameraToWorldMatrix(cameraToWorldRotation).t();
    if (depthAxisFlipped)
    {
        worldToCamera.at<double>(0, 2) *= -1.0;
        worldToCamera.at<double>(1, 2) *= -1.0;
        worldToCamera.at<double>(2, 0) *= -1.0;
        worldToCamera.at<double>(2, 1) *= -1.0;
    }

    cv::Mat rotationVector;
    cv::Rodrigues(worldToCamera, rotationVector);
    return rotationVector;
}

cv::Mat openCvTvecFromCameraPose(
    const std::array<double, 9> &cameraToWorldRotation,
    const std::array<double, 3> &cameraCenter,
    bool depthAxisFlipped)
{
    const cv::Mat worldToCamera = cameraToWorldMatrix(cameraToWorldRotation).t();
    const cv::Mat center = (cv::Mat_<double>(3, 1)
        << cameraCenter[0], cameraCenter[1], cameraCenter[2]);
    cv::Mat translation = -worldToCamera * center;
    if (depthAxisFlipped)
    {
        translation.at<double>(2) *= -1.0;
    }
    return translation;
}

cv::Mat openCvProjectionMatrix(const Camera &camera)
{
    const cv::Mat cameraToWorld = cameraToWorldMatrix(camera.cameraToWorldRotation());
    const cv::Mat worldToCamera = cameraToWorld.t();
    const std::array<double, 3> cameraCenter = camera.cameraCenter();
    const cv::Mat center = (cv::Mat_<double>(3, 1)
        << cameraCenter[0], cameraCenter[1], cameraCenter[2]);
    const cv::Mat translation = -worldToCamera * center;

    cv::Mat extrinsics(3, 4, CV_64F);
    worldToCamera.copyTo(extrinsics(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(extrinsics(cv::Rect(3, 0, 1, 3)));
    return openCvCameraMatrix(camera, false) * extrinsics;
}

} // namespace xjw
