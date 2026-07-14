#include "ModelGeometryComparator.h"

#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>
#include <plapoint/search/spatial_kdtree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace xjw::qc
{
namespace
{

using GeometryKdTree = plapoint::search::SpatialKdTree<3, double>;

class DisjointSet
{
public:
    explicit DisjointSet(std::size_t size)
        : _parent(size), _rank(size, 0)
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    int find(int value)
    {
        if (_parent[static_cast<std::size_t>(value)] != value)
        {
            _parent[static_cast<std::size_t>(value)] =
                find(_parent[static_cast<std::size_t>(value)]);
        }
        return _parent[static_cast<std::size_t>(value)];
    }

    void unite(int left, int right)
    {
        left = find(left);
        right = find(right);
        if (left == right)
        {
            return;
        }
        if (_rank[static_cast<std::size_t>(left)] < _rank[static_cast<std::size_t>(right)])
        {
            std::swap(left, right);
        }
        _parent[static_cast<std::size_t>(right)] = left;
        if (_rank[static_cast<std::size_t>(left)] == _rank[static_cast<std::size_t>(right)])
        {
            ++_rank[static_cast<std::size_t>(left)];
        }
    }

private:
    std::vector<int> _parent;
    std::vector<int> _rank;
};

struct Bounds
{
    std::array<double, 3> minimum{{std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()}};
    std::array<double, 3> maximum{{-std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()}};

    void include(const xjw::mesh::MeshVertex &vertex)
    {
        const std::array<double, 3> point{{vertex.x, vertex.y, vertex.z}};
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[static_cast<std::size_t>(axis)] =
                std::min(minimum[static_cast<std::size_t>(axis)],
                         point[static_cast<std::size_t>(axis)]);
            maximum[static_cast<std::size_t>(axis)] =
                std::max(maximum[static_cast<std::size_t>(axis)],
                         point[static_cast<std::size_t>(axis)]);
        }
    }

    double diagonal() const
    {
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double difference = maximum[static_cast<std::size_t>(axis)] -
                                      minimum[static_cast<std::size_t>(axis)];
            squared += difference * difference;
        }
        return std::sqrt(std::max(0.0, squared));
    }
};

double quantile(const std::vector<double> &sorted, double fraction)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double position = std::clamp(fraction, 0.0, 1.0) *
                            static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - blend) + sorted[upper] * blend;
}

GeometryKdTree buildTree(const std::vector<Point3D> &points)
{
    std::vector<GeometryKdTree::Point> tree_points;
    tree_points.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        tree_points.push_back({{points[index].x, points[index].y, points[index].z},
                               static_cast<int>(index)});
    }
    return GeometryKdTree(tree_points);
}

std::vector<double> nearestDistances(const std::vector<Point3D> &queries,
                                     const GeometryKdTree &tree)
{
    std::vector<double> distances;
    distances.reserve(queries.size());
    for (const Point3D &query : queries)
    {
        double distance = 0.0;
        if (tree.nearest({query.x, query.y, query.z}, &distance) >= 0)
        {
            distances.push_back(distance);
        }
    }
    return distances;
}

std::vector<Point3D> sampledMeshVertices(const xjw::mesh::TriMesh &mesh,
                                         std::size_t maximum_points)
{
    std::vector<Point3D> points;
    const std::size_t stride = std::max<std::size_t>(
        1, (mesh.vertices.size() + maximum_points - 1) / maximum_points);
    points.reserve((mesh.vertices.size() + stride - 1) / stride);
    for (std::size_t index = 0; index < mesh.vertices.size(); index += stride)
    {
        const xjw::mesh::MeshVertex &vertex = mesh.vertices[index];
        points.push_back({vertex.x, vertex.y, vertex.z});
    }
    return points;
}

std::vector<Point3D> loadReferencePoints(const QString &path,
                                         std::size_t maximum_points)
{
    const auto cloud = plapoint::io::readPly<double>(
        xjw::common::io::toNativeNarrowPath(path));
    if (!cloud || cloud->size() == 0)
    {
        return {};
    }
    const std::size_t stride = std::max<std::size_t>(
        1, (cloud->size() + maximum_points - 1) / maximum_points);
    std::vector<Point3D> points;
    points.reserve((cloud->size() + stride - 1) / stride);
    for (std::size_t index = 0; index < cloud->size(); index += stride)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        points.push_back({cloud->points().getValue(row, 0),
                          cloud->points().getValue(row, 1),
                          cloud->points().getValue(row, 2)});
    }
    return points;
}

double medianPointSpacing(const std::vector<Point3D> &points)
{
    if (points.size() < 2)
    {
        return 0.0;
    }
    const GeometryKdTree tree = buildTree(points);
    std::vector<double> spacing;
    const std::size_t stride = std::max<std::size_t>(1, points.size() / 10000);
    for (std::size_t index = 0; index < points.size(); index += stride)
    {
        double distance = 0.0;
        if (tree.nearestByPointIndex(index, &distance) >= 0 && distance > 0.0)
        {
            spacing.push_back(distance);
        }
    }
    std::sort(spacing.begin(), spacing.end());
    return quantile(spacing, 0.5);
}

} // namespace

ModelGeometryQuality ModelGeometryComparator::analyzeMesh(
    const xjw::mesh::TriMesh &mesh)
{
    ModelGeometryQuality quality;
    if (mesh.faces.empty() || mesh.vertices.empty())
    {
        return quality;
    }

    DisjointSet components(mesh.faces.size());
    std::vector<int> first_face(mesh.vertices.size(), -1);
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        for (const int vertex_index : mesh.faces[face_index].v)
        {
            int &first = first_face[static_cast<std::size_t>(vertex_index)];
            if (first >= 0)
            {
                components.unite(static_cast<int>(face_index), first);
            }
            else
            {
                first = static_cast<int>(face_index);
            }
        }
    }

    std::unordered_map<int, int> face_counts;
    std::unordered_map<int, Bounds> component_bounds;
    Bounds full_bounds;
    for (const xjw::mesh::MeshVertex &vertex : mesh.vertices)
    {
        full_bounds.include(vertex);
    }
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        const int root = components.find(static_cast<int>(face_index));
        ++face_counts[root];
        for (const int vertex_index : mesh.faces[face_index].v)
        {
            component_bounds[root].include(
                mesh.vertices[static_cast<std::size_t>(vertex_index)]);
        }
    }

    quality.componentCount = static_cast<int>(face_counts.size());
    int largest_root = -1;
    int largest_count = 0;
    for (const auto &[root, count] : face_counts)
    {
        if (count > largest_count)
        {
            largest_root = root;
            largest_count = count;
        }
    }
    quality.largestComponentFaceRatio =
        static_cast<double>(largest_count) / static_cast<double>(mesh.faces.size());
    const double full_diagonal = full_bounds.diagonal();
    if (full_diagonal > 0.0)
    {
        for (const auto &[root, bounds] : component_bounds)
        {
            if (root != largest_root)
            {
                quality.largestFloatingDiagonalRatio = std::max(
                    quality.largestFloatingDiagonalRatio,
                    bounds.diagonal() / full_diagonal);
            }
        }
    }
    return quality;
}

ReferenceGeometryQuality ModelGeometryComparator::comparePointSets(
    const std::vector<Point3D> &source,
    const std::vector<Point3D> &reference)
{
    ReferenceGeometryQuality quality;
    quality.sourcePointCount = source.size();
    quality.referencePointCount = reference.size();
    if (source.empty() || reference.empty())
    {
        quality.error = QStringLiteral("待测点集和参考点集不能为空");
        return quality;
    }

    const GeometryKdTree source_tree = buildTree(source);
    const GeometryKdTree reference_tree = buildTree(reference);
    std::vector<double> source_distances = nearestDistances(source, reference_tree);
    std::vector<double> reference_distances = nearestDistances(reference, source_tree);
    std::vector<double> combined = source_distances;
    combined.insert(combined.end(), reference_distances.begin(), reference_distances.end());
    std::sort(combined.begin(), combined.end());
    if (combined.empty())
    {
        quality.error = QStringLiteral("最近邻距离计算失败");
        return quality;
    }

    double squared_sum = 0.0;
    for (const double distance : combined)
    {
        squared_sum += distance * distance;
    }
    quality.rmse = std::sqrt(squared_sum / static_cast<double>(combined.size()));
    quality.p50 = quantile(combined, 0.50);
    quality.p84 = quantile(combined, 0.84);
    quality.p95 = quantile(combined, 0.95);
    const double spacing = medianPointSpacing(reference);
    quality.distanceThreshold = spacing > 0.0 ? 3.0 * spacing : quality.p95;
    quality.sourceCoverage = static_cast<double>(std::count_if(
        source_distances.begin(), source_distances.end(),
        [&](double distance) { return distance <= quality.distanceThreshold; })) /
        static_cast<double>(source_distances.size());
    quality.referenceCoverage = static_cast<double>(std::count_if(
        reference_distances.begin(), reference_distances.end(),
        [&](double distance) { return distance <= quality.distanceThreshold; })) /
        static_cast<double>(reference_distances.size());
    quality.available = true;
    return quality;
}

ReferenceGeometryQuality ModelGeometryComparator::compareAlignedPointSets(
    const std::vector<Point3D> &source,
    const std::vector<Point3D> &reference,
    const SimilarityTransform *sourceToReference,
    bool cropReferenceToSourceBounds)
{
    std::vector<Point3D> aligned_source = source;
    if (sourceToReference)
    {
        for (Point3D &point : aligned_source)
        {
            point = PointCloudAlignment::apply(*sourceToReference, point);
        }
    }
    if (!cropReferenceToSourceBounds || aligned_source.empty())
    {
        return comparePointSets(aligned_source, reference);
    }

    std::array<double, 3> minimum{{std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()}};
    std::array<double, 3> maximum{{-std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()}};
    for (const Point3D &point : aligned_source)
    {
        const std::array<double, 3> values{{point.x, point.y, point.z}};
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[static_cast<std::size_t>(axis)] = std::min(
                minimum[static_cast<std::size_t>(axis)], values[static_cast<std::size_t>(axis)]);
            maximum[static_cast<std::size_t>(axis)] = std::max(
                maximum[static_cast<std::size_t>(axis)], values[static_cast<std::size_t>(axis)]);
        }
    }
    double diagonal_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double extent = maximum[static_cast<std::size_t>(axis)] -
                              minimum[static_cast<std::size_t>(axis)];
        diagonal_squared += extent * extent;
    }
    const double padding = std::max(1.0e-9, std::sqrt(diagonal_squared) * 0.05);
    std::vector<Point3D> cropped_reference;
    cropped_reference.reserve(reference.size());
    for (const Point3D &point : reference)
    {
        const std::array<double, 3> values{{point.x, point.y, point.z}};
        bool inside = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            const std::size_t offset = static_cast<std::size_t>(axis);
            inside = inside && values[offset] >= minimum[offset] - padding &&
                     values[offset] <= maximum[offset] + padding;
        }
        if (inside)
        {
            cropped_reference.push_back(point);
        }
    }
    if (cropped_reference.empty())
    {
        ReferenceGeometryQuality quality;
        quality.sourcePointCount = aligned_source.size();
        quality.error = QStringLiteral("参考点云在待测模型局部范围内没有点");
        return quality;
    }
    return comparePointSets(aligned_source, cropped_reference);
}

ReferenceGeometryQuality ModelGeometryComparator::compareReferenceCloud(
    const xjw::mesh::TriMesh &mesh,
    const QString &referenceCloudPath,
    bool alignReferenceCloud,
    const SimilarityTransform *sourceToReference,
    bool cropReferenceToSourceBounds)
{
    ReferenceGeometryQuality quality;
    try
    {
        std::vector<Point3D> source = sampledMeshVertices(mesh, 1000000);
        const std::vector<Point3D> reference =
            loadReferencePoints(referenceCloudPath, 1000000);
        if (sourceToReference)
        {
            return compareAlignedPointSets(source, reference, sourceToReference,
                                           cropReferenceToSourceBounds);
        }
        if (alignReferenceCloud && !source.empty() && !reference.empty())
        {
            const PointCloudAlignmentResult alignment =
                PointCloudAlignment::alignNearestNeighborTranslation(source, reference);
            if (!alignment.success)
            {
                quality.error = alignment.error;
                return quality;
            }
            for (Point3D &point : source)
            {
                point = PointCloudAlignment::apply(alignment.transform, point);
            }
        }
        return compareAlignedPointSets(source, reference, nullptr,
                                       cropReferenceToSourceBounds);
    }
    catch (const std::exception &exception)
    {
        quality.error = QString::fromUtf8(exception.what());
        return quality;
    }
}

} // namespace xjw::qc
