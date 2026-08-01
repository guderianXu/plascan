#pragma once

/**
 * @file SparsePointCloudWorkspace.h
 * @brief SfM 稀疏点过滤共享的 PlaPoint/PlaMatrix 工作集。
 *
 * 工作集把 ScenePoint3D 转换为连续点云并惰性构建 KD-tree。多个过滤器共享同一
 * 索引和邻域缓存，避免每个步骤重复复制点或重建搜索结构。所有索引始终对应
 * `_points`/`_pointIds` 原始顺序。
 */

#include "SparsePointCloudProcessor.h"
#include "common/SfmTypes.h"

#include <plamatrix/plamatrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/search/kdtree.h>

#include <array>
#include <memory>
#include <vector>

namespace xjw
{

class SfmReconstruction;

struct SparsePointCloudNeighbor
{
    int index = -1; ///< 工作集点索引。
    double distanceSquared = 0.0; ///< 平方欧氏距离，避免不必要开方。
};

using SparsePointCloudNeighborList = std::vector<SparsePointCloudNeighbor>;

/**
 * @brief 一次稀疏点处理会话。
 *
 * Cloud 和 KdTree 固定在 PlaMatrix CPU 设备；具体离群点过滤可通过 PlaPoint
 * ProcessingDevice 选择 CPU/CUDA 实现，但返回索引仍映射到该主机工作集。
 */
class SparsePointCloudWorkspace
{
public:
    using Cloud = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    using KdTree = plapoint::search::KdTree<double, plamatrix::Device::CPU>;

    /// 从普通点数组构建；pointIds 使用无效哨兵。
    static SparsePointCloudWorkspace fromPoints(const std::vector<SparsePointCloudPoint> &points);

    /// 从重建快照构建，并保留每个工作集点对应的 Point3DId。
    static SparsePointCloudWorkspace fromReconstruction(const SfmReconstruction &reconstruction);

    std::size_t size() const;
    bool empty() const;

    const Cloud &cloud() const;
    const std::vector<SparsePointCloudPoint> &points() const;
    const std::vector<Point3DId> &pointIds() const;

    /// 按与 size() 等长的掩码返回保留点；长度不一致会抛出参数错误。
    std::vector<SparsePointCloudPoint> filteredPoints(const std::vector<bool> &keepMask) const;

    /// 返回被掩码移除的重建 Point3DId，供 SfmReconstruction 原位删除。
    std::vector<Point3DId> removedPointIds(const std::vector<bool> &keepMask) const;

    /// 返回被移除的工作集整数索引，供纯数组调用方使用。
    std::vector<int> removedIndicesFromKeepMask(const std::vector<bool> &keepMask) const;

    /// 为全部点一次性计算 k 近邻；每项均使用工作集稳定索引。
    std::vector<SparsePointCloudNeighborList> knnCache(int k) const;

    /// 查询任意世界坐标的 k 近邻索引。
    std::vector<int> nearestKSearch(const std::array<double, 3> &query, int k) const;

    /// 查询给定欧氏半径内的点索引。
    std::vector<int> radiusSearch(const std::array<double, 3> &query, double radius) const;

    std::vector<int> statisticalOutlierIndices(int k,
                                               double stdDevMul,
                                               plapoint::ProcessingDevice processingDevice,
                                               plapoint::ProcessingReport *report = nullptr) const;

    std::vector<int> radiusOutlierIndices(double radius,
                                          int minNeighbors,
                                          plapoint::ProcessingDevice processingDevice,
                                          plapoint::ProcessingReport *report = nullptr) const;

private:
    SparsePointCloudWorkspace(std::vector<SparsePointCloudPoint> points,
                              std::vector<Point3DId> pointIds);

    /// 首次邻域查询时构建，之后在不可变工作集生命周期内复用。
    const KdTree &tree() const;
    void validateKeepMask(const std::vector<bool> &keepMask) const;

    std::vector<SparsePointCloudPoint> _points;
    std::vector<Point3DId> _pointIds;
    std::shared_ptr<Cloud> _cloud;
    mutable std::shared_ptr<KdTree> _tree;
};

} // namespace xjw
