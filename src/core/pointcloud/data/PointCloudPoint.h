#pragma once

// =============================================================================
// 文件: PointCloudPoint.h
// 功能: 定义单个点云点及其直接关联类型。
//
// 包含:
//   - PhotogrammetryPointAttributes  逐点摄影测量附加属性
//   - PointCloudFace                 三角面索引定义
//   - PointPredicate                 点选择谓词
//   - PointCloudPoint                单个点对象（位置、法向量、颜色等）
// =============================================================================

#include "math/Vec.h"

#include <array>
#include <cstddef>
#include <functional>

namespace xjw::pointcloud
{

// 将通用几何类型引入本命名空间，保持现有调用代码不变
using xjw::ColorRGBA;
using xjw::Point2f;
using xjw::Point3f;

/**
 * @brief 与摄影测量流程相关的逐点附加属性。
 *
 * 这些字段主要服务于稀疏重建、BA 结果分析、控制点筛选等流程。
 */
struct PhotogrammetryPointAttributes
{
    int pointId = -1;
    int trackLength = 0;
    float reprojectionError = 0.0f;
    float confidence = 1.0f;
    bool isControlPoint = false;
    bool isValid = true;
};

/**
 * @brief 三角面索引定义。
 *
 * OBJ 索引从 1 开始；内部统一采用 0 开始索引。
 */
struct PointCloudFace
{
    std::array<std::size_t, 3> vertexIndices{0, 0, 0};
    std::array<std::size_t, 3> textureIndices{0, 0, 0};
    bool hasTextureIndices = false;
};

/**
 * @brief 点选择谓词。
 *
 * 返回 `true` 表示保留当前点，返回 `false` 表示过滤掉当前点。
 */
using PointPredicate = std::function<bool(std::size_t index,
                                          const Point3f &position,
                                          const Point3f *normal,
                                          const ColorRGBA *color,
                                          const PhotogrammetryPointAttributes *photogrammetry)>;

/**
 * @brief 单个点云点对象。
 *
 * 承载并控制"单个点"的行为，包含位置、纹理坐标以及可选的法向量、颜色和摄影测量属性。
 * 通常用于与 PointCloud 之间的数据交换（pointAt / setPoint 等接口）。
 */
class PointCloudPoint
{
public:
    Point3f position;
    Point2f textureCoordinate;
    Point3f normal;
    ColorRGBA color;
    PhotogrammetryPointAttributes photogrammetry;
    bool hasTextureCoordinate = false;
    bool hasNormal = false;
    bool hasColor = false;
    bool hasPhotogrammetry = false;

    /** @brief 设置纹理坐标并标记为有效。 */
    void setTextureCoordinate(const Point2f &value);

    /** @brief 设置法向量并标记为有效。 */
    void setNormal(const Point3f &value);

    /** @brief 设置颜色并标记为有效。 */
    void setColor(const ColorRGBA &value);

    /** @brief 设置摄影测量属性并标记为有效。 */
    void setPhotogrammetry(const PhotogrammetryPointAttributes &value);
};

} // namespace xjw::pointcloud
