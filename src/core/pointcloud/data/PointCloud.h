#pragma once

// =============================================================================
// 文件: PointCloud.h
// 功能: 定义点云数据对象及其配套类型。
//
// 包含:
//   - PointCloudCoordinateFrame  坐标参考系枚举
//   - PointCloudMetadata         点云级元数据
//   - PointCloudBounds           轴对齐包围盒
//   - PointCloudTransform        4x4 齐次变换矩阵
//   - PointCloudCudaExport       CUDA 紧凑导出布局
//   - PointCloud                 点云数据主类
// =============================================================================

#include "PointCloudPoint.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xjw::pointcloud
{

// --------空间索引内部实现前展声明--------
struct PointCloudSpatialIndexImpl;

/**
 * @brief 点云坐标参考系。
 */
enum class PointCloudCoordinateFrame
{
    Unknown = 0,
    Camera = 1,
    World = 2,
    Local = 3
};

/**
 * @brief 点云级元数据。
 *
 * 不参与几何计算，用于描述点云来源、命名、坐标系与采集上下文。
 */
struct PointCloudMetadata
{
    std::string name;
    std::string sourcePath;
    std::string description;
    std::string coordinateSystem;
    std::string sensorName;
    std::string captureId;
    PointCloudCoordinateFrame coordinateFrame = PointCloudCoordinateFrame::Unknown;
    bool isRegistered = false;
    double timestampSeconds = 0.0;
};

/**
 * @brief 轴对齐包围盒。
 *
 * 当 `valid=false` 时表示当前包围盒尚未初始化或点云为空。
 */
struct PointCloudBounds
{
    Point3f minCorner;
    Point3f maxCorner;
    bool valid = false;
};

/**
 * @brief 4x4 齐次变换矩阵，按行存储。
 *
 * 前三行用于旋转与平移，最后一行为齐次项。
 */
struct PointCloudTransform
{
    std::array<float, 16> matrix
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    /** @brief 返回单位变换。 */
    static PointCloudTransform identity();

    /** @brief 由 3x3 旋转矩阵和平移向量构造刚体变换。 */
    static PointCloudTransform fromRotationTranslation(const std::array<float, 9> &rotation,
                                                       const Point3f &translation);
};

/**
 * @brief 面向 CUDA 计算的紧凑导出布局。
 *
 * 将点云属性拆分为线性数组，便于直接复制到 GPU 或交给 CUDA kernel 使用。
 */
struct PointCloudCudaExport
{
    std::vector<float> positionsXyz;
    std::vector<float> normalsXyz;
    std::vector<uint8_t> colorsRgba;
    std::vector<float> confidences;
    std::vector<float> reprojectionErrors;
    bool hasNormals = false;
    bool hasColors = false;
    bool hasPhotogrammetryAttributes = false;
};

/**
 * @brief 点云数据对象。
 *
 * 统一维护位置、法向量、颜色、摄影测量属性与元数据，
 * 并提供基础编辑、几何变换、筛选裁剪与 CUDA 导出能力。
 */
class PointCloud
{
public:
    PointCloud();   // 定义在 .cpp（标准 pimpl 惯用法）
    ~PointCloud();  // 同上

    // 拷贝构造/赋值：深拷贝数据，但不复制空间索引缓存（将在需要时重建）。
    PointCloud(const PointCloud &other);
    PointCloud &operator=(const PointCloud &other);

    // 移动构造/赋值：在 .cpp 定义以满足 pimpl 的完整类型要求。
    PointCloud(PointCloud &&) noexcept;
    PointCloud &operator=(PointCloud &&) noexcept;

    /** @brief 返回点数量。 */
    std::size_t size() const;

    /** @brief 判断点云是否为空。 */
    bool empty() const;

    /** @brief 判断是否携带法向量数组。 */
    bool hasNormals() const;

    /** @brief 判断是否携带颜色数组。 */
    bool hasColors() const;

    /** @brief 判断是否携带摄影测量属性数组。 */
    bool hasPhotogrammetryAttributes() const;

    /** @brief 判断是否携带纹理坐标数组。 */
    bool hasTextureCoordinates() const;

    /** @brief 判断是否携带三角面。 */
    bool hasFaces() const;

    /** @brief 判断内部属性数组是否与点数量一致。 */
    bool isConsistent() const;

    /** @brief 清空所有点和附属属性。 */
    void clear();

    /** @brief 为点及附属属性预留容量。 */
    void reserve(std::size_t pointCount);

    /** @brief 追加仅包含位置的点。 */
    void addPoint(const Point3f &position);

    /** @brief 追加带法向量的点。 */
    void addPoint(const Point3f &position, const Point3f &normal);

    /** @brief 追加带颜色的点。 */
    void addPoint(const Point3f &position, const ColorRGBA &color);

    /** @brief 追加带法向量和颜色的点。 */
    void addPoint(const Point3f &position, const Point3f &normal, const ColorRGBA &color);

    /** @brief 追加完整属性点。 */
    void addPoint(const Point3f &position,
                  const Point3f &normal,
                  const ColorRGBA &color,
                  const PhotogrammetryPointAttributes &photogrammetry);

    /** @brief 删除指定索引的点及其附属属性。 */
    bool removePoint(std::size_t index);

    /** @brief 修改指定点的位置。 */
    bool setPosition(std::size_t index, const Point3f &position);

    /** @brief 修改指定点的法向量，必要时自动补齐法向量数组。 */
    bool setNormal(std::size_t index, const Point3f &normal);

    /** @brief 修改指定点的颜色，必要时自动补齐颜色数组。 */
    bool setColor(std::size_t index, const ColorRGBA &color);

    /** @brief 修改指定点的摄影测量属性，必要时自动补齐属性数组。 */
    bool setPhotogrammetryAttributes(std::size_t index,
                                     const PhotogrammetryPointAttributes &photogrammetry);

    /** @brief 修改指定点的纹理坐标，必要时自动补齐纹理坐标数组。 */
    bool setTextureCoordinate(std::size_t index, const Point2f &textureCoordinate);

    /** @brief 按单点对象写回指定索引的全部属性。 */
    bool setPoint(std::size_t index, const PointCloudPoint &point);

    /** @brief 整体设置法向量数组。 */
    void setNormals(const std::vector<Point3f> &normals);

    /** @brief 整体设置颜色数组。 */
    void setColors(const std::vector<ColorRGBA> &colors);

    /** @brief 整体设置摄影测量属性数组。 */
    void setPhotogrammetryAttributes(const std::vector<PhotogrammetryPointAttributes> &photogrammetryAttributes);

    /** @brief 整体设置纹理坐标数组。 */
    void setTextureCoordinates(const std::vector<Point2f> &textureCoordinates);

    /** @brief 追加一个三角面。 */
    bool addFace(const PointCloudFace &face);

    /** @brief 清空所有三角面。 */
    void clearFaces();

    /** @brief 设置点云级元数据。 */
    void setMetadata(const PointCloudMetadata &metadata);

    /**
     * @brief 将另一个点云追加到当前对象末尾。
     *
     * 会尽量保留双方已有的附属属性与元数据一致性。
     */
    void append(const PointCloud &other);

    /** @brief 计算轴对齐包围盒。 */
    PointCloudBounds computeBounds() const;

    /** @brief 计算几何中心。 */
    Point3f computeCentroid() const;

    /** @brief 整体平移点云。 */
    void translate(const Point3f &offset);

    /** @brief 以原点为中心做统一缩放。 */
    void scale(float factor);

    /** @brief 以给定中心做统一缩放。 */
    void scale(float factor, const Point3f &center);

    /** @brief 以给定中心做三轴独立缩放。 */
    void scale(const Point3f &factors, const Point3f &center);

    /**
     * @brief 原地应用变换。
     *
     * 当 `transformNormals=true` 时会同步变换法向量并做归一化。
     */
    void transform(const PointCloudTransform &transform, bool transformNormals = true);

    /** @brief 返回变换后的点云副本。 */
    PointCloud transformed(const PointCloudTransform &transform, bool transformNormals = true) const;

    /** @brief 按谓词选出满足条件的点索引。 */
    std::vector<std::size_t> selectIndices(const PointPredicate &predicate) const;

    /** @brief 按谓词返回过滤后的点云副本。 */
    PointCloud filter(const PointPredicate &predicate) const;

    /** @brief 按谓词原地过滤点云。 */
    void filterInPlace(const PointPredicate &predicate);

    /** @brief 返回位于包围盒内的点云副本。 */
    PointCloud crop(const PointCloudBounds &bounds) const;

    /** @brief 原地裁剪到给定包围盒。 */
    void cropInPlace(const PointCloudBounds &bounds);

    // ── 空间索引 KD-Tree 接口 ───────────────────────────────────────────

    /**
     * @brief 显式构建或重建空间索引（惰性构建，查询时自动触发）。
     *
     * 对大规模批量查询前可主动调用以预热索引。
     */
    void buildSpatialIndex() const;

    /** @brief 返回空间索引是否已构建。 */
    bool hasSpatialIndex() const;

    /**
     * @brief 使空间索引失效。
     *
     * 点位置修改类操作（addPoint / removePoint / setPosition /
     * translate / scale / transform / append 等）会自动调用。
     */
    void invalidateSpatialIndex();

    /**
     * @brief 查询距指定点索引最近的 k 个邻居点索引（排除自身）。
     *
     * @param pointIndex 点云内部点索引（0 ~ size()-1）
     * @param k          邻居数
     * @return           按距离升序排列的邻居点索引向量
     */
    std::vector<std::size_t> kNearestIndices(std::size_t pointIndex, std::size_t k) const;

    /**
     * @brief 查询距指定空间位置最近的 k 个邻居点索引。
     */
    std::vector<std::size_t> kNearestIndices(const Point3f &position, std::size_t k) const;

    /**
     * @brief 返回位于球形邘域内的点索引列表（懒惰构建索引）。
     *
     * @param position 球心坐标
     * @param radius   球心半径（与点云坐标单位一致）
     */
    std::vector<std::size_t> radiusSearchIndices(const Point3f &position, float radius) const;

    /** @brief 导出为适合 CUDA 后端消费的紧凑数组。 */
    PointCloudCudaExport exportToCuda() const;

    /** @brief 访问点云元数据。 */
    const PointCloudMetadata &metadata() const;

    /** @brief 获取指定点的法向量指针；若不存在则返回空。 */
    const Point3f *normalAt(std::size_t index) const;

    /** @brief 获取指定点的颜色指针；若不存在则返回空。 */
    const ColorRGBA *colorAt(std::size_t index) const;

    /** @brief 获取指定点的摄影测量属性指针；若不存在则返回空。 */
    const PhotogrammetryPointAttributes *photogrammetryAttributesAt(std::size_t index) const;

    /** @brief 获取指定点的纹理坐标指针；若不存在则返回空。 */
    const Point2f *textureCoordinateAt(std::size_t index) const;

    /** @brief 读取单点对象快照。 */
    PointCloudPoint pointAt(std::size_t index) const;

    /** @brief 访问位置数组。 */
    const std::vector<Point3f> &positions() const;

    /** @brief 访问法向量数组。 */
    const std::vector<Point3f> &normals() const;

    /** @brief 访问颜色数组。 */
    const std::vector<ColorRGBA> &colors() const;

    /** @brief 访问摄影测量属性数组。 */
    const std::vector<PhotogrammetryPointAttributes> &photogrammetryAttributes() const;

    /** @brief 访问纹理坐标数组。 */
    const std::vector<Point2f> &textureCoordinates() const;

    /** @brief 访问三角面数组。 */
    const std::vector<PointCloudFace> &faces() const;

    /** @brief 设置 OBJ 材质库文件名（mtllib）。 */
    void setMaterialLibraryFile(const std::string &materialLibraryFile);

    /** @brief 获取 OBJ 材质库文件名（mtllib）。 */
    const std::string &materialLibraryFile() const;

    /** @brief 设置纹理图片路径（通常用于 MTL 的 map_Kd）。 */
    void setTextureImageFile(const std::string &textureImageFile);

    /** @brief 获取纹理图片路径。 */
    const std::string &textureImageFile() const;

private:
    void ensureNormalsForExistingPoints();
    void ensureColorsForExistingPoints();
    void ensurePhotogrammetryAttributesForExistingPoints();
    void ensureTextureCoordinatesForExistingPoints();
    void reindexFacesAfterPointRemoval(std::size_t removedIndex);
    PointCloud copyByIndices(const std::vector<std::size_t> &indices) const;

private:
    std::vector<Point3f> _positions;
    std::vector<Point3f> _normals;
    std::vector<ColorRGBA> _colors;
    std::vector<PhotogrammetryPointAttributes> _photogrammetryAttributes;
    std::vector<Point2f> _textureCoordinates;
    std::vector<PointCloudFace> _faces;
    std::string _materialLibraryFile;
    std::string _textureImageFile;
    PointCloudMetadata _metadata;
    mutable std::unique_ptr<PointCloudSpatialIndexImpl> _spatialIndex;
};

} // namespace xjw::pointcloud
