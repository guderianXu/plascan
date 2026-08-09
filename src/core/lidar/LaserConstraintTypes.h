#pragma once

#include <array>

namespace xjw
{
namespace lidar
{

struct LaserPlaneSample
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    std::array<double, 3> normal{{0.0, 0.0, 1.0}};
    double curvature = 0.0;
    int sourceFrameIndex = -1;
};

struct LaserConstraintMapOptions
{
    double maxCurvature = 0.2;
    double voxelSizeMeters = 0.0;
    int maxSamples = 500000;
    bool useMissingNormalsAsHeightPlanes = false;
    bool sampleInputBeforeFiltering = false; ///< 快速校验模式：读取阶段均匀抽取 maxSamples 个原始顶点
};

} // namespace lidar
} // namespace xjw
