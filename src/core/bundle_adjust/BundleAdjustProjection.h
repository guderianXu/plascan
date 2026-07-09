#pragma once

#include "Camera.h"

#include <array>
#include <type_traits>

namespace xjw::ba
{

/**
 * @brief Ceres BA 使用的轻量相机参数快照。
 *
 * 这里复制 Camera 的运行态内外参，使残差函数可以用模板标量 T
 * 完成投影计算，避免固定相机场景仍走 NumericDiff。
 */
struct ProjectionCamera
{
    std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0}};
    std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
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
    bool depthAxisFlipped = false;
};

ProjectionCamera makeProjectionCamera(const Camera &camera);

template <typename T>
bool project(const ProjectionCamera &camera, const T *world, T *pixel)
{
    const T dx = world[0] - T(camera.cameraCenter[0]);
    const T dy = world[1] - T(camera.cameraCenter[1]);
    const T dz = world[2] - T(camera.cameraCenter[2]);

    const T xCam = T(camera.cameraToWorldRotation[0]) * dx +
                   T(camera.cameraToWorldRotation[3]) * dy +
                   T(camera.cameraToWorldRotation[6]) * dz;
    const T yCam = T(camera.cameraToWorldRotation[1]) * dx +
                   T(camera.cameraToWorldRotation[4]) * dy +
                   T(camera.cameraToWorldRotation[7]) * dz;
    const T zCam = T(camera.cameraToWorldRotation[2]) * dx +
                   T(camera.cameraToWorldRotation[5]) * dy +
                   T(camera.cameraToWorldRotation[8]) * dz;

    const T forwardDepth = camera.depthAxisFlipped ? -zCam : zCam;
    if constexpr (std::is_floating_point_v<T>)
    {
        if (!(forwardDepth > T(1e-9)))
        {
            return false;
        }
    }

    const T x = xCam / zCam;
    const T y = yCam / zCam;
    const T r2 = x * x + y * y;
    const T radial = T(1.0) + T(camera.radialK1) * r2 +
                     T(camera.radialK2) * r2 * r2 +
                     T(camera.radialK3) * r2 * r2 * r2;
    const T xy2 = T(2.0) * x * y;
    const T xd = x * radial + T(camera.tangentialP1) * xy2 +
                 T(camera.tangentialP2) * (r2 + T(2.0) * x * x);
    const T yd = y * radial + T(camera.tangentialP1) * (r2 + T(2.0) * y * y) +
                 T(camera.tangentialP2) * xy2;

    pixel[0] = T(camera.uAxisSign) * T(camera.focalX) * xd + T(camera.principalX);
    pixel[1] = T(camera.vAxisSign) * T(camera.focalY) * yd + T(camera.principalY);
    return true;
}

} // namespace xjw::ba
