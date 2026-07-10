#pragma once

#include "BundleAdjustNativeCudaDeviceTypes.cuh"

#include <math_constants.h>

namespace xjw::detail::native_cuda
{

__device__ inline bool projectDevice(const DeviceCamera &camera, const double point[3], double pixel[2])
{
    const double dx = point[0] - camera.cameraCenter[0];
    const double dy = point[1] - camera.cameraCenter[1];
    const double dz = point[2] - camera.cameraCenter[2];

    const double xCam = camera.cameraToWorldRotation[0] * dx +
                        camera.cameraToWorldRotation[3] * dy +
                        camera.cameraToWorldRotation[6] * dz;
    const double yCam = camera.cameraToWorldRotation[1] * dx +
                        camera.cameraToWorldRotation[4] * dy +
                        camera.cameraToWorldRotation[7] * dz;
    const double zCam = camera.cameraToWorldRotation[2] * dx +
                        camera.cameraToWorldRotation[5] * dy +
                        camera.cameraToWorldRotation[8] * dz;

    if (!(zCam > 1e-9))
    {
        return false;
    }

    const double x = xCam / zCam;
    const double y = yCam / zCam;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 +
                          camera.radialK1 * r2 +
                          camera.radialK2 * r4 +
                          camera.radialK3 * r6;
    const double xy2 = 2.0 * x * y;
    const double xd = x * radial +
                      camera.tangentialP1 * xy2 +
                      camera.tangentialP2 * (r2 + 2.0 * x * x);
    const double yd = y * radial +
                      camera.tangentialP1 * (r2 + 2.0 * y * y) +
                      camera.tangentialP2 * xy2;

    pixel[0] = static_cast<double>(camera.uAxisSign) * camera.focalX * xd + camera.principalX;
    pixel[1] = static_cast<double>(camera.vAxisSign) * camera.focalY * yd + camera.principalY;
    return isfinite(pixel[0]) && isfinite(pixel[1]);
}

__device__ inline bool pointProjectionJacobianDevice(const DeviceCamera &camera,
                                                     const double point[3],
                                                     double jacobian[6])
{
    const double dx = point[0] - camera.cameraCenter[0];
    const double dy = point[1] - camera.cameraCenter[1];
    const double dz = point[2] - camera.cameraCenter[2];

    const double xCam = camera.cameraToWorldRotation[0] * dx +
                        camera.cameraToWorldRotation[3] * dy +
                        camera.cameraToWorldRotation[6] * dz;
    const double yCam = camera.cameraToWorldRotation[1] * dx +
                        camera.cameraToWorldRotation[4] * dy +
                        camera.cameraToWorldRotation[7] * dz;
    const double zCam = camera.cameraToWorldRotation[2] * dx +
                        camera.cameraToWorldRotation[5] * dy +
                        camera.cameraToWorldRotation[8] * dz;
    if (!(zCam > 1e-9))
    {
        return false;
    }

    const double x = xCam / zCam;
    const double y = yCam / zCam;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double radial = 1.0 +
                          camera.radialK1 * r2 +
                          camera.radialK2 * r4 +
                          camera.radialK3 * r4 * r2;
    const double radialPrime = camera.radialK1 +
                               2.0 * camera.radialK2 * r2 +
                               3.0 * camera.radialK3 * r4;
    const double dradialDx = 2.0 * x * radialPrime;
    const double dradialDy = 2.0 * y * radialPrime;

    const double dxdDx = radial + x * dradialDx + 2.0 * camera.tangentialP1 * y +
                         6.0 * camera.tangentialP2 * x;
    const double dxdDy = x * dradialDy + 2.0 * camera.tangentialP1 * x +
                         2.0 * camera.tangentialP2 * y;
    const double dydDx = y * dradialDx + 2.0 * camera.tangentialP1 * x +
                         2.0 * camera.tangentialP2 * y;
    const double dydDy = radial + y * dradialDy + 6.0 * camera.tangentialP1 * y +
                         2.0 * camera.tangentialP2 * x;

    const double duDx = static_cast<double>(camera.uAxisSign) * camera.focalX * dxdDx;
    const double duDy = static_cast<double>(camera.uAxisSign) * camera.focalX * dxdDy;
    const double dvDx = static_cast<double>(camera.vAxisSign) * camera.focalY * dydDx;
    const double dvDy = static_cast<double>(camera.vAxisSign) * camera.focalY * dydDy;

    const double invZ = 1.0 / zCam;
    const double duDxc = duDx * invZ;
    const double duDyc = duDy * invZ;
    const double duDzc = -(duDx * x + duDy * y) * invZ;
    const double dvDxc = dvDx * invZ;
    const double dvDyc = dvDy * invZ;
    const double dvDzc = -(dvDx * x + dvDy * y) * invZ;

    for (int axis = 0; axis < 3; ++axis)
    {
        const double r0 = camera.cameraToWorldRotation[axis * 3 + 0];
        const double r1 = camera.cameraToWorldRotation[axis * 3 + 1];
        const double r2v = camera.cameraToWorldRotation[axis * 3 + 2];
        jacobian[axis] = duDxc * r0 + duDyc * r1 + duDzc * r2v;
        jacobian[3 + axis] = dvDxc * r0 + dvDyc * r1 + dvDzc * r2v;
        if (!isfinite(jacobian[axis]) || !isfinite(jacobian[3 + axis]))
        {
            return false;
        }
    }
    return true;
}

__device__ inline double robustScale(double residualU, double residualV, double weight, double huberDelta)
{
    if (!(weight > 0.0) || !isfinite(weight))
    {
        return 0.0;
    }

    double robustWeight = 1.0;
    const double norm = sqrt(residualU * residualU + residualV * residualV);
    if (huberDelta > 0.0 && norm > huberDelta)
    {
        robustWeight = huberDelta / fmax(norm, 1e-12);
    }
    return sqrt(weight * robustWeight);
}

__device__ inline double pointCostDevice(const DeviceCamera *cameras,
                                         const DeviceObservation *observations,
                                         const DevicePoint &point,
                                         const double xyz[3],
                                         double huberDelta)
{
    double cost = 0.0;
    for (int local = 0; local < point.observationCount; ++local)
    {
        const DeviceObservation &observation = observations[point.observationBegin + local];
        const DeviceCamera &camera = cameras[observation.cameraIndex];
        double pixel[2] = {0.0, 0.0};
        if (!projectDevice(camera, xyz, pixel))
        {
            cost += 1.0e24;
            continue;
        }

        const double du = pixel[0] - observation.u;
        const double dv = pixel[1] - observation.v;
        const double scale = robustScale(du, dv, observation.weight, huberDelta);
        cost += scale * scale * (du * du + dv * dv);
    }
    return cost;
}

__device__ inline bool solve3x3(double h[9], const double g[3], double step[3])
{
    double a[3][4] = {
        {h[0], h[1], h[2], -g[0]},
        {h[3], h[4], h[5], -g[1]},
        {h[6], h[7], h[8], -g[2]},
    };

    for (int col = 0; col < 3; ++col)
    {
        int pivot = col;
        double pivotAbs = fabs(a[col][col]);
        for (int row = col + 1; row < 3; ++row)
        {
            const double valueAbs = fabs(a[row][col]);
            if (valueAbs > pivotAbs)
            {
                pivot = row;
                pivotAbs = valueAbs;
            }
        }

        if (!(pivotAbs > 1.0e-12))
        {
            return false;
        }

        if (pivot != col)
        {
            for (int k = col; k < 4; ++k)
            {
                const double tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
        }

        const double invPivot = 1.0 / a[col][col];
        for (int k = col; k < 4; ++k)
        {
            a[col][k] *= invPivot;
        }

        for (int row = 0; row < 3; ++row)
        {
            if (row == col)
            {
                continue;
            }
            const double factor = a[row][col];
            for (int k = col; k < 4; ++k)
            {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    step[0] = a[0][3];
    step[1] = a[1][3];
    step[2] = a[2][3];
    return isfinite(step[0]) && isfinite(step[1]) && isfinite(step[2]);
}

__global__ void optimizePointsKernel(const DeviceCamera *cameras,
                                     DevicePoint *points,
                                     const DeviceObservation *observations,
                                     int pointCount,
                                     int maxIterations,
                                     double huberDelta,
                                     double damping,
                                     int *acceptedIterations)
{
    const int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIndex >= pointCount)
    {
        return;
    }

    DevicePoint &point = points[pointIndex];
    int accepted = 0;
    double xyz[3] = {point.xyz[0], point.xyz[1], point.xyz[2]};

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        double h[9] = {0.0};
        double g[3] = {0.0};
        double currentCost = 0.0;
        int validObservationCount = 0;

        for (int local = 0; local < point.observationCount; ++local)
        {
            const DeviceObservation &observation = observations[point.observationBegin + local];
            const DeviceCamera &camera = cameras[observation.cameraIndex];
            double pixel[2] = {0.0, 0.0};
            if (!projectDevice(camera, xyz, pixel))
            {
                continue;
            }

            const double du = pixel[0] - observation.u;
            const double dv = pixel[1] - observation.v;
            const double scale = robustScale(du, dv, observation.weight, huberDelta);
            if (!(scale > 0.0))
            {
                continue;
            }

            const double residual[2] = {scale * du, scale * dv};
            double jacobian[6] = {0.0};
            if (!pointProjectionJacobianDevice(camera, xyz, jacobian))
            {
                continue;
            }
            for (int i = 0; i < 6; ++i)
            {
                jacobian[i] *= scale;
            }

            currentCost += residual[0] * residual[0] + residual[1] * residual[1];
            ++validObservationCount;
            for (int row = 0; row < 3; ++row)
            {
                g[row] += jacobian[row] * residual[0] + jacobian[3 + row] * residual[1];
                for (int col = 0; col < 3; ++col)
                {
                    h[row * 3 + col] += jacobian[row] * jacobian[col] +
                                        jacobian[3 + row] * jacobian[3 + col];
                }
            }
        }

        if (validObservationCount < 2)
        {
            break;
        }

        const double lambda = fmax(damping, 1.0e-12);
        h[0] += lambda * fmax(1.0, h[0]);
        h[4] += lambda * fmax(1.0, h[4]);
        h[8] += lambda * fmax(1.0, h[8]);

        double step[3] = {0.0, 0.0, 0.0};
        if (!solve3x3(h, g, step))
        {
            break;
        }

        const double stepNorm = sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2]);
        if (!(stepNorm > 1.0e-10))
        {
            break;
        }

        double stepScale = stepNorm > 1.0 ? 1.0 / stepNorm : 1.0;
        bool stepAccepted = false;
        for (int trial = 0; trial < 8; ++trial)
        {
            const double candidate[3] = {
                xyz[0] + stepScale * step[0],
                xyz[1] + stepScale * step[1],
                xyz[2] + stepScale * step[2],
            };
            const double candidateCost =
                pointCostDevice(cameras, observations, point, candidate, huberDelta);
            if (candidateCost < currentCost && isfinite(candidateCost))
            {
                xyz[0] = candidate[0];
                xyz[1] = candidate[1];
                xyz[2] = candidate[2];
                ++accepted;
                stepAccepted = true;
                break;
            }
            stepScale *= 0.5;
        }

        if (!stepAccepted)
        {
            break;
        }
    }

    point.xyz[0] = xyz[0];
    point.xyz[1] = xyz[1];
    point.xyz[2] = xyz[2];
    acceptedIterations[pointIndex] = accepted;
}

} // namespace xjw::detail::native_cuda
