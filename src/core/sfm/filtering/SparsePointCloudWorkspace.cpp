/**
 * @file SparsePointCloudWorkspace.cpp
 * @brief SfM 稀疏点到 PlaPoint/PlaMatrix 连续工作集的转换和邻域查询实现。
 *
 * 工作集构造后不可变，因此 KD-tree 可以惰性构建并在多个过滤阶段复用。
 * `_points`、`_pointIds` 和 PlaPoint cloud 始终保持相同索引顺序。
 */

#include "SparsePointCloudWorkspace.h"

#include "reconstruction/SfmReconstruction.h"

#include <algorithm>
#include <stdexcept>

namespace xjw
{

    namespace
    {

        SparsePointCloudPoint scenePointToSparsePoint(const ScenePoint3D& scenePoint)
        {
            SparsePointCloudPoint point;
            point.x = scenePoint.xyz[0];
            point.y = scenePoint.xyz[1];
            point.z = scenePoint.xyz[2];
            point.rmsReprojPx = scenePoint.error;
            point.trackLen = static_cast<int>(scenePoint.track.elements.size());
            point.hasColor = true;
            point.red = scenePoint.color[0];
            point.green = scenePoint.color[1];
            point.blue = scenePoint.color[2];
            return point;
        }

        std::shared_ptr<SparsePointCloudWorkspace::Cloud> buildCloud(const std::vector<SparsePointCloudPoint>& points)
        {
            // N x 3 点矩阵使用默认列优先存储，每一行对应一个稳定工作集索引。
            plamatrix::MatrixXd coordinates(static_cast<plamatrix::Index>(points.size()), 3);
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const auto row = static_cast<plamatrix::Index>(i);
                coordinates(row, 0) = points[i].x;
                coordinates(row, 1) = points[i].y;
                coordinates(row, 2) = points[i].z;
            }
            return std::make_shared<SparsePointCloudWorkspace::Cloud>(std::move(coordinates));
        }

    } // namespace

    SparsePointCloudWorkspace SparsePointCloudWorkspace::fromPoints(const std::vector<SparsePointCloudPoint>& points)
    {
        return SparsePointCloudWorkspace(points, std::vector<Point3DId>(points.size(), kInvalidPoint3DId));
    }

    SparsePointCloudWorkspace SparsePointCloudWorkspace::fromReconstruction(const SfmReconstruction& reconstruction)
    {
        // 显式排序消除 unordered 容器迭代顺序，使过滤结果和测试可重复。
        std::vector<Point3DId> pointIds = reconstruction.allPoint3DIds();
        std::sort(pointIds.begin(), pointIds.end());

        std::vector<SparsePointCloudPoint> points;
        points.reserve(pointIds.size());
        for (Point3DId pointId : pointIds)
        {
            if (!reconstruction.hasPoint3D(pointId))
            {
                continue;
            }
            points.push_back(scenePointToSparsePoint(reconstruction.point3D(pointId)));
        }

        if (points.size() != pointIds.size())
        {
            std::vector<Point3DId> compactIds;
            compactIds.reserve(points.size());
            for (Point3DId pointId : pointIds)
            {
                if (reconstruction.hasPoint3D(pointId))
                {
                    compactIds.push_back(pointId);
                }
            }
            pointIds = std::move(compactIds);
        }

        return SparsePointCloudWorkspace(std::move(points), std::move(pointIds));
    }

    SparsePointCloudWorkspace::SparsePointCloudWorkspace(std::vector<SparsePointCloudPoint> points,
                                                         std::vector<Point3DId> pointIds)
        : _points(std::move(points)), _pointIds(std::move(pointIds)), _cloud(buildCloud(_points))
    {
        if (_pointIds.size() != _points.size())
        {
            throw std::invalid_argument("SparsePointCloudWorkspace: point id count does not match point count");
        }
    }

    std::size_t SparsePointCloudWorkspace::size() const
    {
        return _points.size();
    }

    bool SparsePointCloudWorkspace::empty() const
    {
        return _points.empty();
    }

    const SparsePointCloudWorkspace::Cloud& SparsePointCloudWorkspace::cloud() const
    {
        return *_cloud;
    }

    const std::vector<SparsePointCloudPoint>& SparsePointCloudWorkspace::points() const
    {
        return _points;
    }

    const std::vector<Point3DId>& SparsePointCloudWorkspace::pointIds() const
    {
        return _pointIds;
    }

    std::vector<SparsePointCloudPoint>
    SparsePointCloudWorkspace::filteredPoints(const std::vector<bool>& keepMask) const
    {
        validateKeepMask(keepMask);

        std::vector<SparsePointCloudPoint> filtered;
        filtered.reserve(_points.size());
        for (std::size_t i = 0; i < _points.size(); ++i)
        {
            if (keepMask[i])
            {
                filtered.push_back(_points[i]);
            }
        }
        return filtered;
    }

    std::vector<Point3DId> SparsePointCloudWorkspace::removedPointIds(const std::vector<bool>& keepMask) const
    {
        validateKeepMask(keepMask);

        std::vector<Point3DId> removed;
        for (std::size_t i = 0; i < _pointIds.size(); ++i)
        {
            if (!keepMask[i] && _pointIds[i] != kInvalidPoint3DId)
            {
                removed.push_back(_pointIds[i]);
            }
        }
        return removed;
    }

    std::vector<int> SparsePointCloudWorkspace::removedIndicesFromKeepMask(const std::vector<bool>& keepMask) const
    {
        validateKeepMask(keepMask);

        std::vector<int> removed;
        for (std::size_t i = 0; i < keepMask.size(); ++i)
        {
            if (!keepMask[i])
            {
                removed.push_back(static_cast<int>(i));
            }
        }
        return removed;
    }

    std::vector<SparsePointCloudNeighborList> SparsePointCloudWorkspace::knnCache(int k) const
    {
        std::vector<SparsePointCloudNeighborList> cache(_points.size());
        if (_points.empty() || k < 1)
        {
            return cache;
        }

        // 查询时多取一个邻居，因为点自身通常以距离 0 返回，随后显式剔除。
        const int actualK = std::min<int>(k, static_cast<int>(_points.size()) - 1);
        if (actualK < 1)
        {
            return cache;
        }

        const KdTree& kdTree = tree();
        for (std::size_t i = 0; i < _points.size(); ++i)
        {
            const plapoint::PointXYZd query{_points[i].x, _points[i].y, _points[i].z};
            std::vector<int> indices;
            std::vector<float> squared_distances;
            kdTree.nearestKSearch(query, actualK + 1, indices, squared_distances);
            SparsePointCloudNeighborList neighbors;
            neighbors.reserve(static_cast<std::size_t>(actualK));
            for (int index : indices)
            {
                if (index == static_cast<int>(i))
                {
                    continue;
                }
                const auto& neighbor = _points[static_cast<std::size_t>(index)];
                const double dx = neighbor.x - query.x;
                const double dy = neighbor.y - query.y;
                const double dz = neighbor.z - query.z;
                neighbors.push_back({index, dx * dx + dy * dy + dz * dz});
            }
            cache[i] = std::move(neighbors);
        }
        return cache;
    }

    std::vector<int> SparsePointCloudWorkspace::nearestKSearch(const std::array<double, 3>& query, int k) const
    {
        if (_points.empty() || k <= 0)
        {
            return {};
        }

        const plapoint::PointXYZd query_point{query[0], query[1], query[2]};
        std::vector<int> indices;
        std::vector<float> squared_distances;
        tree().nearestKSearch(query_point, k, indices, squared_distances);
        return indices;
    }

    std::vector<int> SparsePointCloudWorkspace::radiusSearch(const std::array<double, 3>& query, double radius) const
    {
        if (_points.empty() || radius <= 0.0)
        {
            return {};
        }

        const plapoint::PointXYZd query_point{query[0], query[1], query[2]};
        std::vector<int> indices;
        std::vector<float> squared_distances;
        tree().radiusSearch(query_point, radius, indices, squared_distances);
        return indices;
    }

    std::vector<int> SparsePointCloudWorkspace::statisticalOutlierIndices(int k,
                                                                          double stdDevMul,
                                                                          plapoint::ProcessingDevice processingDevice,
                                                                          plapoint::ProcessingReport* report) const
    {
        std::vector<int> removedIndices;
        if (_points.size() <= static_cast<std::size_t>(std::max(0, k)) || k < 1)
        {
            return removedIndices;
        }

        plapoint::statisticalOutlierRemoval(*_cloud, k, stdDevMul, processingDevice, &removedIndices, report);
        return removedIndices;
    }

    std::vector<int> SparsePointCloudWorkspace::radiusOutlierIndices(double radius,
                                                                     int minNeighbors,
                                                                     plapoint::ProcessingDevice processingDevice,
                                                                     plapoint::ProcessingReport* report) const
    {
        std::vector<int> removedIndices;
        if (_points.size() < 2 || radius <= 0.0 || minNeighbors < 1)
        {
            return removedIndices;
        }

        plapoint::radiusOutlierRemoval(*_cloud, radius, minNeighbors, processingDevice, &removedIndices, report);
        return removedIndices;
    }

    const SparsePointCloudWorkspace::KdTree& SparsePointCloudWorkspace::tree() const
    {
        // 工作集不可变，首次构建后的树可安全用于后续只读查询。
        if (!_tree)
        {
            auto search_cloud = std::make_shared<plapoint::PointCloud<plapoint::PointXYZd>>();
            search_cloud->resize(_points.size());
            for (std::size_t index = 0; index < _points.size(); ++index)
            {
                const auto& point = _points[index];
                search_cloud->points[index] = plapoint::PointXYZd(point.x, point.y, point.z);
            }
            auto tree = std::make_shared<KdTree>();
            tree->setInputCloud(search_cloud);
            _tree = std::move(tree);
        }
        return *_tree;
    }

    void SparsePointCloudWorkspace::validateKeepMask(const std::vector<bool>& keepMask) const
    {
        if (keepMask.size() != _points.size())
        {
            throw std::invalid_argument("SparsePointCloudWorkspace: keep mask size does not match point count");
        }
    }

} // namespace xjw
