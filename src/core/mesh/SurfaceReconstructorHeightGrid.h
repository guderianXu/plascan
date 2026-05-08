#pragma once

/**
 * @file SurfaceReconstructorHeightGrid.h
 * @brief 2.5D 高程格网重建模块接口。
 *
 * 模块职责：
 * 1) 将散乱点云投影到 XY 规则格网并估计高度；
 * 2) 对空洞区域进行可控轮次的邻域补洞；
 * 3) 将高程格网三角化为 `TriMesh`。
 */

#include "MeshTypes.h"
#include "PointCloudPreprocess.h"

#include <cstdint>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace detail
{

/**
 * @brief 2.5D 高程格网数据结构。
 *
 * 该结构将散乱点云投影到 XY 规则格网，并在每个格点保存高度、有效性和补洞轮次信息。
 * 主要用于地形类/航测类点云的快速曲面重建。
 */
struct HeightGrid
{
    int nx = 0;
    int ny = 0;
    float minX = 0.0f;
    float minY = 0.0f;
    float stepX = 1.0f;
    float stepY = 1.0f;
    std::vector<float> heights;
    std::vector<float> sumW;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint16_t> fillPass;

    float &at(int ix, int iy)
    {
        return heights[static_cast<std::size_t>(iy * nx + ix)];
    }

    float at(int ix, int iy) const
    {
        return heights[static_cast<std::size_t>(iy * nx + ix)];
    }

    std::uint8_t &vd(int ix, int iy)
    {
        return valid[static_cast<std::size_t>(iy * nx + ix)];
    }

    std::uint8_t vd(int ix, int iy) const
    {
        return valid[static_cast<std::size_t>(iy * nx + ix)];
    }
};

/**
 * @brief 将散乱点云构建为 2.5D 高程格网。
 */
HeightGrid buildHeightGrid(const std::vector<PointXYZRGB> &points,
                           const ReconstructionConfig &config);

/**
 * @brief 基于邻域均值迭代填充高程格网中的空洞。
 */
void fillHoles(HeightGrid *heightGrid, int maxPasses = 8);

/**
 * @brief 将高程格网进行规则三角剖分并生成网格。
 */
void heightGridToMesh(const HeightGrid &heightGrid,
                      const std::vector<PointXYZRGB> &points,
                      const ReconstructionConfig &config,
                      TriMesh *mesh);

} // namespace detail
} // namespace mesh
} // namespace xjw
