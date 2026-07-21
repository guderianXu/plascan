#include "ProjectionGeometry.h"

#include <cmath>
#include <limits>

namespace xjw
{

ProjectionResult projectForReprojection(const Camera &camera,
                                        const std::array<double, 3> &worldPoint)
{
    ProjectionResult result;
    const double world[3] = {worldPoint[0], worldPoint[1], worldPoint[2]};
    double pixel[2] = {0.0, 0.0};
    if (camera.projectWorldPointWithDepth(world, pixel, result.positiveDepth))
    {
        result.success = true;
    }
    else if (camera.projectWorldPointSigned(world, pixel))
    {
        double cameraPoint[3] = {0.0, 0.0, 0.0};
        camera.worldToCamera(world, cameraPoint);
        result.positiveDepth = camera.depthAxisFlipped() ? -cameraPoint[2] : cameraPoint[2];
        result.success = true;
        result.usedSignedFallback = true;
    }

    result.pixel = {pixel[0], pixel[1]};
    return result;
}

double reprojectionErrorPx(const Camera &camera,
                           const std::array<double, 3> &worldPoint,
                           const std::array<double, 2> &observedPixel)
{
    const ProjectionResult projection = projectForReprojection(camera, worldPoint);
    if (!projection.success)
    {
        return std::numeric_limits<double>::infinity();
    }

    const double deltaU = projection.pixel[0] - observedPixel[0];
    const double deltaV = projection.pixel[1] - observedPixel[1];
    return std::sqrt(deltaU * deltaU + deltaV * deltaV);
}

} // namespace xjw
