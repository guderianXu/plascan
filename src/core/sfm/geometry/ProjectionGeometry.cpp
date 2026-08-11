#include "ProjectionGeometry.h"

#include <cmath>
#include <limits>

namespace xjw
{

ProjectionResult projectForReprojection(const FramePinholeCamera &camera,
                                        const std::array<double, 3> &worldPoint)
{
    ProjectionResult result;
    const double world[3] = {worldPoint[0], worldPoint[1], worldPoint[2]};
    double pixel[2] = {0.0, 0.0};

    // 生产几何优先要求正深度。signed fallback 只为旧相机元数据诊断保留，
    // 并通过 usedSignedFallback 暴露给调用方，不能用于掩盖负深度质量问题。
    if (camera.projectWorldPointWithDepth(world, pixel, result.positiveDepth))
    {
        result.success = true;
    }
    else if (camera.projectWorldPointSigned(world, pixel))
    {
        result.positiveDepth = camera.positiveDepth(world);
        result.success = true;
        result.usedSignedFallback = true;
    }

    result.pixel = {pixel[0], pixel[1]};
    return result;
}

double reprojectionErrorPx(const FramePinholeCamera &camera,
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
