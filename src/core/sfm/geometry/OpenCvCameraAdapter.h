#pragma once

/**
 * @file OpenCvCameraAdapter.h
 * @brief PlaScan FramePinholeCamera 与 OpenCV 标定/PnP坐标约定的集中转换。
 *
 * PlaScan 保存 camera-to-world 旋转 Rcw 和相机中心 C；OpenCV 使用
 * world-to-camera 的 `Xc = Rwc * X + t`，其中 `Rwc = Rcw^T`、`t = -Rwc*C`。
 * 深度轴翻转和像素轴符号只能在本适配层处理，调用方不得再次手工取逆或翻轴。
 */

#include "FramePinholeCamera.h"

#include <opencv2/core.hpp>

#include <array>

namespace xjw
{

/**
 * @brief 由 FramePinholeCamera 构造 OpenCV 3x3 内参矩阵。
 * @param positiveDepthConvention true 时把 PlaScan 的 -Z 前向相机改写为 OpenCV +Z 前向。
 */
cv::Mat openCvCameraMatrix(const FramePinholeCamera &camera, bool positiveDepthConvention);

/// 低层内参转换重载，参数单位均为像素。
cv::Mat openCvCameraMatrix(double focalX,
                           double focalY,
                           double principalX,
                           double principalY,
                           int uAxisSign,
                           int vAxisSign,
                           bool depthAxisFlipped,
                           bool positiveDepthConvention);

/// 将 Rcw 转为 OpenCV Rodrigues world-to-camera 旋转向量。
cv::Mat openCvRvecFromCameraToWorldPose(
    const std::array<double, 9> &cameraToWorldRotation,
    bool depthAxisFlipped);

/// 由 Rcw 和世界坐标相机中心 C 计算 OpenCV 平移向量 `t=-Rwc*C`。
cv::Mat openCvTvecFromCameraPose(
    const std::array<double, 9> &cameraToWorldRotation,
    const std::array<double, 3> &cameraCenter,
    bool depthAxisFlipped);

/// 构造与 FramePinholeCamera::projectWorldPoint 同约定的 3x4 投影矩阵 `K[Rwc|t]`。
cv::Mat openCvProjectionMatrix(const FramePinholeCamera &camera);

} // namespace xjw
