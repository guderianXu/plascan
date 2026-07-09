#include "BundleAdjustNativeCudaMath.h"

#include <algorithm>
#include <cmath>

namespace xjw::detail::native_cuda
{

ProjectionResult projectHost(const HostCamera &camera, const std::array<double, 3> &point)
{
    ProjectionResult result;

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
        return result;
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

    result.pixel[0] = static_cast<double>(camera.uAxisSign) * camera.focalX * xd + camera.principalX;
    result.pixel[1] = static_cast<double>(camera.vAxisSign) * camera.focalY * yd + camera.principalY;
    result.ok = std::isfinite(result.pixel[0]) && std::isfinite(result.pixel[1]);
    return result;
}

bool linearizeObservationHost(const HostCamera &camera,
                              const std::array<double, 3> &point,
                              double observedU,
                              double observedV,
                              double weight,
                              double huberDelta,
                              ObservationLinearization *out)
{
    if (!out || weight < 0.0 || !std::isfinite(weight))
    {
        return false;
    }

    const ProjectionResult projection = projectHost(camera, point);
    if (!projection.ok)
    {
        return false;
    }

    double residualU = projection.pixel[0] - observedU;
    double residualV = projection.pixel[1] - observedV;
    const double norm = std::sqrt(residualU * residualU + residualV * residualV);

    double robustWeight = 1.0;
    if (huberDelta > 0.0 && norm > huberDelta)
    {
        robustWeight = huberDelta / std::max(norm, 1e-12);
    }

    const double scale = std::sqrt(weight * robustWeight);
    residualU *= scale;
    residualV *= scale;

    *out = ObservationLinearization{};
    out->residual[0] = residualU;
    out->residual[1] = residualV;
    out->weightedCost = residualU * residualU + residualV * residualV;

    const double eps = 1e-6;
    for (int axis = 0; axis < 3; ++axis)
    {
        std::array<double, 3> plus = point;
        std::array<double, 3> minus = point;
        plus[axis] += eps;
        minus[axis] -= eps;

        const ProjectionResult pPlus = projectHost(camera, plus);
        const ProjectionResult pMinus = projectHost(camera, minus);
        if (!pPlus.ok || !pMinus.ok)
        {
            return false;
        }

        out->jp[axis] = scale * (pPlus.pixel[0] - pMinus.pixel[0]) / (2.0 * eps);
        out->jp[3 + axis] = scale * (pPlus.pixel[1] - pMinus.pixel[1]) / (2.0 * eps);
    }

    return true;
}

} // namespace xjw::detail::native_cuda
