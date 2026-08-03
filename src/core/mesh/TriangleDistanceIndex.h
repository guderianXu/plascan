#pragma once

#include <array>
#include <memory>

namespace xjw::mesh
{

struct TriMesh;

class TriangleDistanceIndex
{
public:
    explicit TriangleDistanceIndex(const TriMesh &mesh);
    ~TriangleDistanceIndex();

    TriangleDistanceIndex(const TriangleDistanceIndex &) = delete;
    TriangleDistanceIndex &operator=(const TriangleDistanceIndex &) = delete;

    bool empty() const;
    double nearestDistanceSquared(
        const std::array<double, 3> &point) const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::mesh
