#pragma once

// ============================================================
// 文件：SfmReconstruction.h
// 功能：SfM 重建容器，管理重建过程中的所有状态。
//
// 存储内容：
//   - 所有图像数据（ImageData）及其注册状态
//   - 所有相机模型（FramePinholeCamera）
//   - 所有三维点（ScenePoint3D）
//   - 对应关系图引用
//
// 提供增删查改接口供 IncrementalSfm 使用。
//
// 参考：COLMAP 的 Reconstruction 类，简化适配 PlaScan。
// ============================================================

#include "common/SfmTypes.h"
#include "FramePinholeCamera.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace xjw {

/**
 * @brief SfM 重建容器。
 *
 * 持有重建过程中的完整状态：图像、相机、三维点。
 * IncrementalSfm 通过此对象操纵重建数据。
 */
class SfmReconstruction {
public:
    SfmReconstruction() = default;

    // ============================================================
    // 图像管理
    // ============================================================

    /**
     * @brief 添加一幅图像到重建中（初始为未注册状态）。
     * @param data  图像数据（包含路径、特征等）
     * @note data.id 用作唯一键，重复添加将覆盖。
     */
    void addImage(const ImageData &data);

    /// 获取图像数据（可修改版本）
    ImageData &image(ImageId id);

    /// 获取图像数据（只读版本）
    const ImageData &image(ImageId id) const;

    /// 判断图像是否存在
    bool hasImage(ImageId id) const;

    /// 判断图像是否已注册
    bool isRegistered(ImageId id) const;

    /// 获取所有图像 ID 列表
    std::vector<ImageId> allImageIds() const;

    /// 获取所有已注册图像 ID 列表
    std::vector<ImageId> registeredImageIds() const;

    /// 已注册图像数量
    size_t numRegisteredImages() const;

    /// 总图像数量
    size_t numImages() const
    {
        return imageDataMap.size();
    }

    /**
     * @brief 将图像标记为已注册，并关联相机。
     * @param imageId  图像 ID
     * @param camera   该图像对应的相机参数
     */
    void registerImage(ImageId imageId, const FramePinholeCamera &camera);

    /**
     * @brief 取消注册图像（标记为未注册，并移除关联相机）。
     * @param imageId  图像 ID
     */
    void deregisterImage(ImageId imageId);

    // ============================================================
    // 相机管理
    // ============================================================

    /// 获取图像对应的相机（可修改）
    FramePinholeCamera &camera(ImageId imageId);

    /// 获取图像对应的相机（只读）
    const FramePinholeCamera &camera(ImageId imageId) const;

    /// 判断图像是否有关联相机
    bool hasCamera(ImageId imageId) const;

    /// 获取所有相机（imageId → FramePinholeCamera）
    const std::unordered_map<ImageId, FramePinholeCamera> &cameras() const
    {
        return cameraMap;
    }

    // ============================================================
    // 三维点管理
    // ============================================================

    /**
     * @brief 添加一个三维点到重建中。
     * @param point  三维点数据（包含坐标和轨迹）
     * @return 分配的 Point3DId
    * @note 如果 point.id == kInvalidPoint3DId，则自动分配新 ID。
     */
    Point3DId addPoint3D(const ScenePoint3D &point);

    /**
     * @brief 添加三维点并关联到观测特征。
     *
     * 创建三维点后，同步更新 track 中每个 ImageData 的 point3DIds。
     * @param xyz    三维坐标
     * @param track  多视图观测轨迹
     * @return 分配的 Point3DId
     */
    Point3DId addPoint3DWithTrack(const std::array<double, 3> &xyz,
                                   const Track &track);

    /// 获取三维点（可修改）
    ScenePoint3D &point3D(Point3DId id);

    /// 获取三维点（只读）
    const ScenePoint3D &point3D(Point3DId id) const;

    /// 判断三维点是否存在
    bool hasPoint3D(Point3DId id) const;

    /// 删除三维点，并清理关联观测
    void deletePoint3D(Point3DId id);

    /// 从三维点轨迹中删除单个观测，并同步清理影像反向关联。
    /// 不会自动删除剩余观测不足两个的三维点，由调用方决定整点生命周期。
    bool removeObservation(Point3DId id, ImageId imageId, FeatureIdx featureIdx);

    /// 三维点总数
    size_t numPoints3D() const
    {
        return point3DMap.size();
    }

    /// 获取所有三维点
    const std::unordered_map<Point3DId, ScenePoint3D> &points3D() const
    {
        return point3DMap;
    }

    /// 获取所有三维点（可修改）
    std::unordered_map<Point3DId, ScenePoint3D> &points3DMutable()
    {
        return point3DMap;
    }

    /// 获取所有三维点 ID 列表
    std::vector<Point3DId> allPoint3DIds() const;

    // ============================================================
    // 统计
    // ============================================================

    /**
     * @brief 计算所有三维点的平均重投影误差。
     * @return 平均重投影误差（像素），无点则返回 0
     */
    double meanReprojError() const;

    /**
     * @brief 汇总重建状态描述字符串。
     */
    std::string summary() const;

private:
    /// 图像数据表 (imageId → ImageData)
    std::unordered_map<ImageId, ImageData> imageDataMap;

    /// 相机表 (imageId → FramePinholeCamera)，仅已注册图像才有
    std::unordered_map<ImageId, FramePinholeCamera> cameraMap;

    /// 三维点表 (point3DId → ScenePoint3D)
    std::unordered_map<Point3DId, ScenePoint3D> point3DMap;

    /// 下一个可分配的三维点 ID
    Point3DId nextPoint3DId = 0;
};

} // namespace xjw
