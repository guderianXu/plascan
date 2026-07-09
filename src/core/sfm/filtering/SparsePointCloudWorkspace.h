#pragma once

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
    int index = -1;
    double distanceSquared = 0.0;
};

using SparsePointCloudNeighborList = std::vector<SparsePointCloudNeighbor>;

class SparsePointCloudWorkspace
{
public:
    using Cloud = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    using KdTree = plapoint::search::KdTree<double, plamatrix::Device::CPU>;

    static SparsePointCloudWorkspace fromPoints(const std::vector<SparsePointCloudPoint> &points);
    static SparsePointCloudWorkspace fromReconstruction(const SfmReconstruction &reconstruction);

    std::size_t size() const;
    bool empty() const;

    const Cloud &cloud() const;
    const std::vector<SparsePointCloudPoint> &points() const;
    const std::vector<Point3DId> &pointIds() const;

    std::vector<SparsePointCloudPoint> filteredPoints(const std::vector<bool> &keepMask) const;
    std::vector<Point3DId> removedPointIds(const std::vector<bool> &keepMask) const;
    std::vector<int> removedIndicesFromKeepMask(const std::vector<bool> &keepMask) const;

    std::vector<SparsePointCloudNeighborList> knnCache(int k) const;
    std::vector<int> nearestKSearch(const std::array<double, 3> &query, int k) const;
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

    const KdTree &tree() const;
    void validateKeepMask(const std::vector<bool> &keepMask) const;

    std::vector<SparsePointCloudPoint> _points;
    std::vector<Point3DId> _pointIds;
    std::shared_ptr<Cloud> _cloud;
    mutable std::shared_ptr<KdTree> _tree;
};

} // namespace xjw
