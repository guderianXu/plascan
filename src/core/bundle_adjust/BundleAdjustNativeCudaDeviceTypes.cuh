#pragma once

/**
 * @file BundleAdjustNativeCudaDeviceTypes.cuh
 * @brief native CUDA 核函数使用的无 STL POD 数据结构。
 *
 * HostCamera/HostPoint 使用 std::array，不能直接假定其 ABI 可由设备代码消费。
 * 本文件显式复制每个字段，保证主机工作集与设备缓冲区之间不存在隐藏布局依赖。
 */

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

/// HostCamera 的设备侧等价布局；字段语义见 BundleAdjustNativeCudaTypes.h。
struct DeviceCamera
{
    double cameraToWorldRotation[9];
    double cameraCenter[3];
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

/// HostPoint 的设备侧等价布局；观测仍按点连续分段。
struct DevicePoint
{
    double xyz[3];
    int originalTrackIndex = -1;
    int observationBegin = 0;
    int observationCount = 0;
};

/// HostObservation 的设备侧等价布局。
struct DeviceObservation
{
    int cameraIndex = -1;
    int pointIndex = -1;
    double u = 0.0;
    double v = 0.0;
    double weight = 1.0;
};

/// 显式打包相机，避免对 std::array 进行未定义的二进制复制。
inline DeviceCamera makeDeviceCamera(const HostCamera &camera)
{
    DeviceCamera out;
    for (int i = 0; i < 9; ++i)
    {
        out.cameraToWorldRotation[i] = camera.cameraToWorldRotation[static_cast<size_t>(i)];
    }
    for (int i = 0; i < 3; ++i)
    {
        out.cameraCenter[i] = camera.cameraCenter[static_cast<size_t>(i)];
    }
    out.focalX = camera.focalX;
    out.focalY = camera.focalY;
    out.principalX = camera.principalX;
    out.principalY = camera.principalY;
    out.radialK1 = camera.radialK1;
    out.radialK2 = camera.radialK2;
    out.radialK3 = camera.radialK3;
    out.tangentialP1 = camera.tangentialP1;
    out.tangentialP2 = camera.tangentialP2;
    out.uAxisSign = camera.uAxisSign;
    out.vAxisSign = camera.vAxisSign;
    out.depthAxisFlipped = camera.depthAxisFlipped;
    out.fixed = camera.fixed;
    out.originalIndex = camera.originalIndex;
    return out;
}

/// 显式打包点坐标及其观测区间。
inline DevicePoint makeDevicePoint(const HostPoint &point)
{
    DevicePoint out;
    for (int i = 0; i < 3; ++i)
    {
        out.xyz[i] = point.xyz[static_cast<size_t>(i)];
    }
    out.originalTrackIndex = point.originalTrackIndex;
    out.observationBegin = point.observationBegin;
    out.observationCount = point.observationCount;
    return out;
}

/// 显式打包一条二维观测。
inline DeviceObservation makeDeviceObservation(const HostObservation &observation)
{
    DeviceObservation out;
    out.cameraIndex = observation.cameraIndex;
    out.pointIndex = observation.pointIndex;
    out.u = observation.u;
    out.v = observation.v;
    out.weight = observation.weight;
    return out;
}

} // namespace xjw::detail::native_cuda
