#pragma once

/**
 * @file BundleAdjustProjection.h
 * @brief BA 残差块共享的可微投影与局部位姿参数化。
 *
 * 坐标约定：
 * - Camera 存储 camera-to-world 旋转 Rcw 和世界坐标相机中心 C；
 * - 世界点到相机坐标使用 `Rcw^T * (X - C)`；
 * - `depthAxisFlipped` 决定相机前方是局部 +Z 还是 -Z；
 * - 像素投影包含 u/v 轴符号和 Brown-Conrady 畸变。
 *
 * 模板函数必须同时支持 double 与 Ceres Jet，禁止在此处加入破坏自动微分的
 * 非模板数学调用或与 Camera::projectWorldPoint 不一致的坐标变换。
 */

#include "Camera.h"

#include <array>
#include <cmath>

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

/// 从运行态 Camera 复制一个不持有资源、可安全捕获进残差 functor 的快照。
ProjectionCamera makeProjectionCamera(const Camera &camera);

/**
 * @brief 投影一个已经位于相机坐标系的点。
 *
 * 在归一化平面应用 Brown-Conrady 畸变，再乘焦距和轴方向符号。若正深度
 * 小于阈值则返回 false，避免把相机后方点作为合法残差。
 */
template <typename T>
bool projectCameraPoint(const ProjectionCamera &camera,
                        const T &xCam,
                        const T &yCam,
                        const T &zCam,
                        const T &focalX,
                        const T &focalY,
                        T *pixel)
{
    const T forwardDepth = camera.depthAxisFlipped ? -zCam : zCam;
    if (!(forwardDepth > T(1e-9)))
    {
        return false;
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

    pixel[0] = T(camera.uAxisSign) * focalX * xd + T(camera.principalX);
    pixel[1] = T(camera.vAxisSign) * focalY * yd + T(camera.principalY);
    return true;
}

/// 使用固定外参、固定内参投影世界点，主要用于点块残差。
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
    return projectCameraPoint(camera,
                              xCam,
                              yCam,
                              zCam,
                              T(camera.focalX),
                              T(camera.focalY),
                              pixel);
}

/**
 * @brief 将 3 维轴角增量转换为旋转矩阵。
 *
 * 小角度分支使用 Rodrigues 一阶展开，避免 theta 接近零时除零，同时保持
 * Ceres Jet 的连续导数。
 */
template <typename T>
void poseDeltaRotation(const T *cameraDelta, T *deltaRotation)
{
    const T wx = cameraDelta[0];
    const T wy = cameraDelta[1];
    const T wz = cameraDelta[2];
    const T theta2 = wx * wx + wy * wy + wz * wz;

    if (theta2 < T(1e-20))
    {
        deltaRotation[0] = T(1.0);
        deltaRotation[1] = -wz;
        deltaRotation[2] = wy;
        deltaRotation[3] = wz;
        deltaRotation[4] = T(1.0);
        deltaRotation[5] = -wx;
        deltaRotation[6] = -wy;
        deltaRotation[7] = wx;
        deltaRotation[8] = T(1.0);
        return;
    }

    using std::cos;
    using std::sin;
    using std::sqrt;
    const T theta = sqrt(theta2);
    const T sinc = sin(theta) / theta;
    const T cosc = (T(1.0) - cos(theta)) / theta2;
    const T wxwy = wx * wy;
    const T wxwz = wx * wz;
    const T wywz = wy * wz;

    deltaRotation[0] = T(1.0) - cosc * (wy * wy + wz * wz);
    deltaRotation[1] = cosc * wxwy - sinc * wz;
    deltaRotation[2] = cosc * wxwz + sinc * wy;
    deltaRotation[3] = cosc * wxwy + sinc * wz;
    deltaRotation[4] = T(1.0) - cosc * (wx * wx + wz * wz);
    deltaRotation[5] = cosc * wywz - sinc * wx;
    deltaRotation[6] = cosc * wxwz - sinc * wy;
    deltaRotation[7] = cosc * wywz + sinc * wx;
    deltaRotation[8] = T(1.0) - cosc * (wx * wx + wy * wy);
}

/**
 * @brief 使用局部位姿增量和显式焦距投影世界点。
 *
 * cameraDelta 的布局为 `[wx, wy, wz, dCx, dCy, dCz]`。旋转增量左乘基准
 * camera-to-world 旋转，相机中心增量在世界坐标系中相加。
 */
template <typename T>
bool projectWithPoseDeltaAndFocal(const ProjectionCamera &camera,
                                  const T *cameraDelta,
                                  const T *world,
                                  const T &focalX,
                                  const T &focalY,
                                  T *pixel)
{
    T deltaRotation[9];
    poseDeltaRotation(cameraDelta, deltaRotation);

    T updatedRotation[9];
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            T value = T(0.0);
            for (int inner = 0; inner < 3; ++inner)
            {
                value += deltaRotation[row * 3 + inner] *
                    T(camera.cameraToWorldRotation[inner * 3 + column]);
            }
            updatedRotation[row * 3 + column] = value;
        }
    }

    const T dx = world[0] - (T(camera.cameraCenter[0]) + cameraDelta[3]);
    const T dy = world[1] - (T(camera.cameraCenter[1]) + cameraDelta[4]);
    const T dz = world[2] - (T(camera.cameraCenter[2]) + cameraDelta[5]);
    const T xCam = updatedRotation[0] * dx +
                   updatedRotation[3] * dy +
                   updatedRotation[6] * dz;
    const T yCam = updatedRotation[1] * dx +
                   updatedRotation[4] * dy +
                   updatedRotation[7] * dz;
    const T zCam = updatedRotation[2] * dx +
                   updatedRotation[5] * dy +
                   updatedRotation[8] * dz;
    return projectCameraPoint(camera, xCam, yCam, zCam, focalX, focalY, pixel);
}

/// 使用局部位姿增量和快照中的固定焦距投影。
template <typename T>
bool projectWithPoseDelta(const ProjectionCamera &camera,
                          const T *cameraDelta,
                          const T *world,
                          T *pixel)
{
    return projectWithPoseDeltaAndFocal(camera,
                                        cameraDelta,
                                        world,
                                        T(camera.focalX),
                                        T(camera.focalY),
                                        pixel);
}

/**
 * @brief 使用对数参数化的共享水平焦距投影。
 *
 * 以 log(f) 优化可天然保证焦距为正；垂直焦距按每台相机原始 fy/fx 比例恢复，
 * 因而共享的是标定组的水平尺度，而不是强制方形像素。
 */
template <typename T>
bool projectWithPoseDeltaAndSharedFocal(const ProjectionCamera &camera,
                                        const T *cameraDelta,
                                        const T *world,
                                        const T *sharedFocalLogPixels,
                                        T *pixel)
{
    using std::exp;
    const T sharedFocal = exp(sharedFocalLogPixels[0]);
    const double focalAspect =
        camera.focalX > 0.0 ? camera.focalY / camera.focalX : 1.0;
    return projectWithPoseDeltaAndFocal(camera,
                                        cameraDelta,
                                        world,
                                        sharedFocal,
                                        sharedFocal * T(focalAspect),
                                        pixel);
}

} // namespace xjw::ba
