#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cassert>
#include <vector>
#include "PhotogrammetryPointAttributes.h"

namespace xjw {
namespace sfm {

struct SparsePointCloud
{
    using GeometryType = plapoint::PointCloud<float, plamatrix::Device::CPU>;

    GeometryType geometry;
    std::vector<PhotogrammetryPointAttributes> attributes;

    size_t size() const
    {
        assert(geometry.size() == attributes.size());
        return geometry.size();
    }

    bool empty() const { return size() == 0; }

    void clear()
    {
        geometry = GeometryType();
        attributes.clear();
    }
};

} // namespace sfm
} // namespace xjw
