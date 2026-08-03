#include "TriangleDistanceIndex.h"

#include "DepthTsdfSurfaceBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::mesh
{
namespace
{

using Point3d = std::array<double, 3>;

Point3d subtract(const Point3d &lhs, const Point3d &rhs)
{
    return {
        lhs[0] - rhs[0],
        lhs[1] - rhs[1],
        lhs[2] - rhs[2]
    };
}

double dot(const Point3d &lhs, const Point3d &rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

double pointSegmentDistanceSquared(const Point3d &point,
                                   const Point3d &first,
                                   const Point3d &second)
{
    const Point3d segment = subtract(second, first);
    const double length_squared = dot(segment, segment);
    if (length_squared <= 1.0e-24)
    {
        const Point3d delta = subtract(point, first);
        return dot(delta, delta);
    }
    const double position = std::clamp(
        dot(subtract(point, first), segment) / length_squared,
        0.0,
        1.0);
    const Point3d closest{
        first[0] + position * segment[0],
        first[1] + position * segment[1],
        first[2] + position * segment[2]
    };
    const Point3d delta = subtract(point, closest);
    return dot(delta, delta);
}

double pointTriangleDistanceSquared(const Point3d &point,
                                    const Point3d &a,
                                    const Point3d &b,
                                    const Point3d &c)
{
    const Point3d ab = subtract(b, a);
    const Point3d ac = subtract(c, a);
    const Point3d ap = subtract(point, a);
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return dot(ap, ap);
    }

    const Point3d bp = subtract(point, b);
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        return dot(bp, bp);
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double position = d1 / (d1 - d3);
        const Point3d closest{
            a[0] + position * ab[0],
            a[1] + position * ab[1],
            a[2] + position * ab[2]
        };
        const Point3d delta = subtract(point, closest);
        return dot(delta, delta);
    }

    const Point3d cp = subtract(point, c);
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        return dot(cp, cp);
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double position = d2 / (d2 - d6);
        const Point3d closest{
            a[0] + position * ac[0],
            a[1] + position * ac[1],
            a[2] + position * ac[2]
        };
        const Point3d delta = subtract(point, closest);
        return dot(delta, delta);
    }

    const double va = d3 * d6 - d5 * d4;
    const double edge_sum = (d4 - d3) + (d5 - d6);
    if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 &&
        edge_sum > 1.0e-24)
    {
        const double position = (d4 - d3) / edge_sum;
        const Point3d bc = subtract(c, b);
        const Point3d closest{
            b[0] + position * bc[0],
            b[1] + position * bc[1],
            b[2] + position * bc[2]
        };
        const Point3d delta = subtract(point, closest);
        return dot(delta, delta);
    }

    const double denominator = va + vb + vc;
    if (std::abs(denominator) <= 1.0e-24)
    {
        return std::min({
            pointSegmentDistanceSquared(point, a, b),
            pointSegmentDistanceSquared(point, b, c),
            pointSegmentDistanceSquared(point, c, a)
        });
    }
    const double inverse_denominator = 1.0 / denominator;
    const double v = vb * inverse_denominator;
    const double w = vc * inverse_denominator;
    const Point3d closest{
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w
    };
    const Point3d delta = subtract(point, closest);
    return dot(delta, delta);
}

} // namespace

class TriangleDistanceIndex::Impl
{
public:
    explicit Impl(const TriMesh &mesh)
        : _mesh(mesh)
    {
        _triangles.reserve(mesh.faces.size());
        for (std::size_t face_index = 0;
             face_index < mesh.faces.size();
             ++face_index)
        {
            const Triangle &face = mesh.faces[face_index];
            if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 ||
                static_cast<std::size_t>(face.v[0]) >= mesh.vertices.size() ||
                static_cast<std::size_t>(face.v[1]) >= mesh.vertices.size() ||
                static_cast<std::size_t>(face.v[2]) >= mesh.vertices.size())
            {
                continue;
            }
            TriangleRecord record;
            record.faceIndex = face_index;
            record.minimum.fill(std::numeric_limits<double>::infinity());
            record.maximum.fill(-std::numeric_limits<double>::infinity());
            for (int vertex_index : face.v)
            {
                const MeshVertex &vertex =
                    mesh.vertices[static_cast<std::size_t>(vertex_index)];
                const Point3d point{vertex.x, vertex.y, vertex.z};
                for (int axis = 0; axis < 3; ++axis)
                {
                    record.minimum[axis] = std::min(
                        record.minimum[axis], point[axis]);
                    record.maximum[axis] = std::max(
                        record.maximum[axis], point[axis]);
                }
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                record.center[axis] =
                    0.5 * (record.minimum[axis] + record.maximum[axis]);
            }
            _triangles.push_back(record);
        }
        if (!_triangles.empty())
        {
            _nodes.reserve(_triangles.size() * 2);
            buildNode(0, _triangles.size());
        }
    }

    bool empty() const
    {
        return _triangles.empty();
    }

    double nearestDistanceSquared(const Point3d &point) const
    {
        double best = std::numeric_limits<double>::infinity();
        if (!_nodes.empty())
        {
            findNearest(0, point, &best);
        }
        return best;
    }

private:
    struct TriangleRecord
    {
        std::size_t faceIndex = 0;
        Point3d minimum{};
        Point3d maximum{};
        Point3d center{};
    };

    struct Node
    {
        Point3d minimum{};
        Point3d maximum{};
        std::size_t begin = 0;
        std::size_t end = 0;
        int left = -1;
        int right = -1;
    };

    int buildNode(std::size_t begin, std::size_t end)
    {
        Node node;
        node.begin = begin;
        node.end = end;
        node.minimum.fill(std::numeric_limits<double>::infinity());
        node.maximum.fill(-std::numeric_limits<double>::infinity());
        Point3d center_minimum = node.minimum;
        Point3d center_maximum = node.maximum;
        for (std::size_t index = begin; index < end; ++index)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                node.minimum[axis] = std::min(
                    node.minimum[axis], _triangles[index].minimum[axis]);
                node.maximum[axis] = std::max(
                    node.maximum[axis], _triangles[index].maximum[axis]);
                center_minimum[axis] = std::min(
                    center_minimum[axis], _triangles[index].center[axis]);
                center_maximum[axis] = std::max(
                    center_maximum[axis], _triangles[index].center[axis]);
            }
        }

        const int node_index = static_cast<int>(_nodes.size());
        _nodes.push_back(node);
        if (end - begin <= 8)
        {
            return node_index;
        }
        int split_axis = 0;
        for (int axis = 1; axis < 3; ++axis)
        {
            if (center_maximum[axis] - center_minimum[axis] >
                center_maximum[split_axis] - center_minimum[split_axis])
            {
                split_axis = axis;
            }
        }
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            _triangles.begin() + static_cast<std::ptrdiff_t>(begin),
            _triangles.begin() + static_cast<std::ptrdiff_t>(middle),
            _triangles.begin() + static_cast<std::ptrdiff_t>(end),
            [split_axis](const TriangleRecord &lhs,
                         const TriangleRecord &rhs)
            {
                return lhs.center[split_axis] < rhs.center[split_axis];
            });
        _nodes[static_cast<std::size_t>(node_index)].left =
            buildNode(begin, middle);
        _nodes[static_cast<std::size_t>(node_index)].right =
            buildNode(middle, end);
        return node_index;
    }

    static double boundsDistanceSquared(const Node &node,
                                        const Point3d &point)
    {
        double distance_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double delta = point[axis] < node.minimum[axis]
                ? node.minimum[axis] - point[axis]
                : (point[axis] > node.maximum[axis]
                       ? point[axis] - node.maximum[axis]
                       : 0.0);
            distance_squared += delta * delta;
        }
        return distance_squared;
    }

    void findNearest(int node_index,
                     const Point3d &point,
                     double *best) const
    {
        const Node &node = _nodes[static_cast<std::size_t>(node_index)];
        if (boundsDistanceSquared(node, point) >= *best)
        {
            return;
        }
        if (node.left < 0 || node.right < 0)
        {
            for (std::size_t index = node.begin; index < node.end; ++index)
            {
                const Triangle &face =
                    _mesh.faces[_triangles[index].faceIndex];
                const MeshVertex &a =
                    _mesh.vertices[static_cast<std::size_t>(face.v[0])];
                const MeshVertex &b =
                    _mesh.vertices[static_cast<std::size_t>(face.v[1])];
                const MeshVertex &c =
                    _mesh.vertices[static_cast<std::size_t>(face.v[2])];
                *best = std::min(
                    *best,
                    pointTriangleDistanceSquared(
                        point,
                        {a.x, a.y, a.z},
                        {b.x, b.y, b.z},
                        {c.x, c.y, c.z}));
            }
            return;
        }
        const double left_distance = boundsDistanceSquared(
            _nodes[static_cast<std::size_t>(node.left)], point);
        const double right_distance = boundsDistanceSquared(
            _nodes[static_cast<std::size_t>(node.right)], point);
        const int first = left_distance <= right_distance
            ? node.left
            : node.right;
        const int second = first == node.left ? node.right : node.left;
        findNearest(first, point, best);
        findNearest(second, point, best);
    }

    const TriMesh &_mesh;
    std::vector<TriangleRecord> _triangles;
    std::vector<Node> _nodes;
};

TriangleDistanceIndex::TriangleDistanceIndex(const TriMesh &mesh)
    : _impl(std::make_unique<Impl>(mesh))
{
}

TriangleDistanceIndex::~TriangleDistanceIndex() = default;

bool TriangleDistanceIndex::empty() const
{
    return _impl->empty();
}

double TriangleDistanceIndex::nearestDistanceSquared(
    const std::array<double, 3> &point) const
{
    return _impl->nearestDistanceSquared(point);
}

} // namespace xjw::mesh
