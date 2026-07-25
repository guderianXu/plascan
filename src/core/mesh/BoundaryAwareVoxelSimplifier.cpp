#include "BoundaryAwareVoxelSimplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

std::uint64_t edgeKey(int a, int b)
{
    const auto lower = static_cast<std::uint32_t>(std::min(a, b));
    const auto upper = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lower) << 32U) | upper;
}

int edgeVertex(std::uint64_t key, bool upper)
{
    return static_cast<int>(upper ? key & 0xffffffffU : key >> 32U);
}

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
        int &parent = _parent[static_cast<std::size_t>(value)];
        if (parent != value)
        {
            parent = find(parent);
        }
        return parent;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
        {
            return;
        }
        if (_rank[static_cast<std::size_t>(a)] < _rank[static_cast<std::size_t>(b)])
        {
            std::swap(a, b);
        }
        _parent[static_cast<std::size_t>(b)] = a;
        if (_rank[static_cast<std::size_t>(a)] == _rank[static_cast<std::size_t>(b)])
        {
            ++_rank[static_cast<std::size_t>(a)];
        }
    }

private:
    std::vector<int> _parent;
    std::vector<std::uint8_t> _rank;
};

struct ClusterKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;
    int boundaryComponent = -1;
    int normalAzimuthBin = -1;
    int normalElevationBin = -1;

    bool operator<(const ClusterKey &other) const
    {
        return std::tie(
                   x, y, z, boundaryComponent, normalAzimuthBin, normalElevationBin) <
            std::tie(other.x,
                     other.y,
                     other.z,
                     other.boundaryComponent,
                     other.normalAzimuthBin,
                     other.normalElevationBin);
    }
};

struct ClusterAccumulator
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    int count = 0;
    int outputIndex = -1;
    bool protectedBoundary = false;
};

bool validFace(const Triangle &face, std::size_t vertexCount)
{
    return face.v[0] >= 0 && face.v[1] >= 0 && face.v[2] >= 0 &&
        static_cast<std::size_t>(face.v[0]) < vertexCount &&
        static_cast<std::size_t>(face.v[1]) < vertexCount &&
        static_cast<std::size_t>(face.v[2]) < vertexCount;
}

std::array<int, 2> normalBins(const MeshVertex &vertex, float maximumAngleDegrees)
{
    if (!std::isfinite(maximumAngleDegrees) || maximumAngleDegrees >= 179.999f)
    {
        return {-1, -1};
    }

    const float length = std::sqrt(
        vertex.nx * vertex.nx + vertex.ny * vertex.ny + vertex.nz * vertex.nz);
    if (!std::isfinite(length) || length <= 1.0e-8f)
    {
        return {-1, -1};
    }

    constexpr float pi = 3.14159265358979323846f;
    const float bin_width = std::clamp(maximumAngleDegrees, 5.0f, 180.0f) *
        pi / 180.0f;
    const float inverse_length = 1.0f / length;
    const float normalized_z = std::clamp(vertex.nz * inverse_length, -1.0f, 1.0f);
    const float azimuth = std::atan2(vertex.ny, vertex.nx);
    const float elevation = std::asin(normalized_z);
    return {
        static_cast<int>(std::floor((azimuth + pi) / bin_width)),
        static_cast<int>(std::floor((elevation + pi * 0.5f) / bin_width))};
}

std::vector<std::uint8_t> featureVertices(const TriMesh &mesh, float maximumAngleDegrees)
{
    std::vector<std::uint8_t> result(mesh.vertices.size(), 0);
    if (!std::isfinite(maximumAngleDegrees) || maximumAngleDegrees >= 179.999f)
    {
        return result;
    }

    std::vector<std::array<float, 3>> face_normals(mesh.faces.size(), {0.0f, 0.0f, 0.0f});
    std::vector<std::array<double, 3>> vertex_normal_sums(
        mesh.vertices.size(), {0.0, 0.0, 0.0});
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        const Triangle &face = mesh.faces[face_index];
        if (!validFace(face, mesh.vertices.size()))
        {
            continue;
        }
        const MeshVertex &a = mesh.vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &b = mesh.vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &c = mesh.vertices[static_cast<std::size_t>(face.v[2])];
        const float ab_x = b.x - a.x;
        const float ab_y = b.y - a.y;
        const float ab_z = b.z - a.z;
        const float ac_x = c.x - a.x;
        const float ac_y = c.y - a.y;
        const float ac_z = c.z - a.z;
        std::array<float, 3> normal{
            ab_y * ac_z - ab_z * ac_y,
            ab_z * ac_x - ab_x * ac_z,
            ab_x * ac_y - ab_y * ac_x};
        const float length = std::sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (!std::isfinite(length) || length <= 1.0e-12f)
        {
            continue;
        }
        for (float &value : normal)
        {
            value /= length;
        }
        face_normals[face_index] = normal;
        for (const int vertex_index : face.v)
        {
            auto &sum = vertex_normal_sums[static_cast<std::size_t>(vertex_index)];
            sum[0] += normal[0];
            sum[1] += normal[1];
            sum[2] += normal[2];
        }
    }

    constexpr float pi = 3.14159265358979323846f;
    const float cosine_threshold = std::cos(
        std::clamp(maximumAngleDegrees, 5.0f, 180.0f) * pi / 180.0f);
    std::vector<std::array<float, 3>> average_normals(
        mesh.vertices.size(), {0.0f, 0.0f, 0.0f});
    for (std::size_t index = 0; index < vertex_normal_sums.size(); ++index)
    {
        const auto &sum = vertex_normal_sums[index];
        const double length = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        if (std::isfinite(length) && length > 1.0e-12)
        {
            average_normals[index] = {
                static_cast<float>(sum[0] / length),
                static_cast<float>(sum[1] / length),
                static_cast<float>(sum[2] / length)};
        }
    }
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        const auto &face_normal = face_normals[face_index];
        if (face_normal[0] == 0.0f && face_normal[1] == 0.0f && face_normal[2] == 0.0f)
        {
            continue;
        }
        const Triangle &face = mesh.faces[face_index];
        for (const int vertex_index : face.v)
        {
            const auto &average = average_normals[static_cast<std::size_t>(vertex_index)];
            const float dot = face_normal[0] * average[0] +
                face_normal[1] * average[1] + face_normal[2] * average[2];
            if (dot < cosine_threshold)
            {
                result[static_cast<std::size_t>(vertex_index)] = 1;
            }
        }
    }
    return result;
}

std::vector<int> boundaryComponents(const TriMesh &mesh,
                                    std::vector<int> *componentSizes,
                                    std::vector<float> *componentDiameters)
{
    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh.faces.size() * 3U);
    for (const Triangle &face : mesh.faces)
    {
        if (!validFace(face, mesh.vertices.size()))
        {
            continue;
        }
        ++edge_counts[edgeKey(face.v[0], face.v[1])];
        ++edge_counts[edgeKey(face.v[1], face.v[2])];
        ++edge_counts[edgeKey(face.v[2], face.v[0])];
    }

    DisjointSet components(mesh.vertices.size());
    std::vector<std::uint8_t> boundary(mesh.vertices.size(), 0);
    for (const auto &[key, count] : edge_counts)
    {
        if (count != 1)
        {
            continue;
        }
        const int a = edgeVertex(key, false);
        const int b = edgeVertex(key, true);
        boundary[static_cast<std::size_t>(a)] = 1;
        boundary[static_cast<std::size_t>(b)] = 1;
        components.unite(a, b);
    }

    std::vector<int> result(mesh.vertices.size(), -1);
    for (std::size_t index = 0; index < boundary.size(); ++index)
    {
        if (boundary[index] != 0)
        {
            result[index] = components.find(static_cast<int>(index));
        }
    }
    if (componentSizes)
    {
        componentSizes->assign(mesh.vertices.size(), 0);
        for (const int component : result)
        {
            if (component >= 0)
            {
                ++(*componentSizes)[static_cast<std::size_t>(component)];
            }
        }
    }
    if (componentDiameters)
    {
        const float infinity = std::numeric_limits<float>::infinity();
        std::vector<std::array<float, 3>> minimum(
            mesh.vertices.size(), {infinity, infinity, infinity});
        std::vector<std::array<float, 3>> maximum(
            mesh.vertices.size(), {-infinity, -infinity, -infinity});
        for (std::size_t index = 0; index < result.size(); ++index)
        {
            const int component = result[index];
            if (component < 0)
            {
                continue;
            }
            const MeshVertex &vertex = mesh.vertices[index];
            auto &lower = minimum[static_cast<std::size_t>(component)];
            auto &upper = maximum[static_cast<std::size_t>(component)];
            lower[0] = std::min(lower[0], vertex.x);
            lower[1] = std::min(lower[1], vertex.y);
            lower[2] = std::min(lower[2], vertex.z);
            upper[0] = std::max(upper[0], vertex.x);
            upper[1] = std::max(upper[1], vertex.y);
            upper[2] = std::max(upper[2], vertex.z);
        }
        componentDiameters->assign(mesh.vertices.size(), 0.0f);
        for (std::size_t index = 0; index < minimum.size(); ++index)
        {
            if (!std::isfinite(minimum[index][0]))
            {
                continue;
            }
            const float dx = maximum[index][0] - minimum[index][0];
            const float dy = maximum[index][1] - minimum[index][1];
            const float dz = maximum[index][2] - minimum[index][2];
            (*componentDiameters)[index] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    return result;
}

} // namespace

TriMesh voxelClusterSimplifyPreservingOpenBoundaries(const TriMesh &mesh,
                                                      float clusterSize,
                                                      int minimumProtectedBoundaryVertices,
                                                      float maximumCollapsibleBoundaryDiameter,
                                                      float maximumNormalClusterAngleDegrees,
                                                      const std::vector<std::uint8_t> *
                                                          protectedBoundaryVertices,
                                                      std::vector<std::uint8_t> *
                                                          outputProtectedBoundaryVertices)
{
    if (mesh.empty() || !std::isfinite(clusterSize) || clusterSize <= 0.0f)
    {
        return mesh;
    }

    float minimum_x = mesh.vertices.front().x;
    float minimum_y = mesh.vertices.front().y;
    float minimum_z = mesh.vertices.front().z;
    for (const MeshVertex &vertex : mesh.vertices)
    {
        minimum_x = std::min(minimum_x, vertex.x);
        minimum_y = std::min(minimum_y, vertex.y);
        minimum_z = std::min(minimum_z, vertex.z);
    }

    std::vector<int> boundary_component_sizes;
    std::vector<float> boundary_component_diameters;
    const std::vector<int> boundary_components = boundaryComponents(
        mesh, &boundary_component_sizes, &boundary_component_diameters);
    const std::vector<std::uint8_t> feature_vertices = featureVertices(
        mesh, maximumNormalClusterAngleDegrees);
    const bool has_explicit_boundary_protection =
        protectedBoundaryVertices &&
        protectedBoundaryVertices->size() == mesh.vertices.size();
    const int protection_threshold = std::max(1, minimumProtectedBoundaryVertices);
    std::map<ClusterKey, ClusterAccumulator> clusters;
    std::vector<ClusterKey> vertex_keys;
    vertex_keys.reserve(mesh.vertices.size());
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
    {
        const MeshVertex &vertex = mesh.vertices[index];
        int boundary_cluster = -1;
        const int boundary_component = boundary_components[index];
        if (boundary_component >= 0)
        {
            const std::size_t component_index =
                static_cast<std::size_t>(boundary_component);
            const bool collapsible = boundary_component_sizes[component_index] <
                    protection_threshold &&
                maximumCollapsibleBoundaryDiameter > 0.0f &&
                boundary_component_diameters[component_index] <=
                    maximumCollapsibleBoundaryDiameter;
            const bool explicitly_protected =
                has_explicit_boundary_protection &&
                (*protectedBoundaryVertices)[index] != 0;
            boundary_cluster = explicitly_protected ||
                    (!has_explicit_boundary_protection && !collapsible)
                ? static_cast<int>(index)
                : -2 - boundary_component;
        }
        const std::array<int, 2> normal_bins = feature_vertices[index] != 0
            ? normalBins(vertex, maximumNormalClusterAngleDegrees)
            : std::array<int, 2>{-1, -1};
        const ClusterKey key{
            static_cast<std::int64_t>(std::floor((vertex.x - minimum_x) / clusterSize)),
            static_cast<std::int64_t>(std::floor((vertex.y - minimum_y) / clusterSize)),
            static_cast<std::int64_t>(std::floor((vertex.z - minimum_z) / clusterSize)),
            boundary_cluster,
            normal_bins[0],
            normal_bins[1]};
        vertex_keys.push_back(key);
        ClusterAccumulator &cluster = clusters[key];
        cluster.x += vertex.x;
        cluster.y += vertex.y;
        cluster.z += vertex.z;
        cluster.nx += vertex.nx;
        cluster.ny += vertex.ny;
        cluster.nz += vertex.nz;
        cluster.r += vertex.r;
        cluster.g += vertex.g;
        cluster.b += vertex.b;
        ++cluster.count;
        cluster.protectedBoundary =
            cluster.protectedBoundary ||
            (has_explicit_boundary_protection &&
             (*protectedBoundaryVertices)[index] != 0);
    }

    TriMesh result;
    result.hasVertexColors = mesh.hasVertexColors;
    result.vertices.reserve(clusters.size());
    if (outputProtectedBoundaryVertices)
    {
        outputProtectedBoundaryVertices->clear();
        outputProtectedBoundaryVertices->reserve(clusters.size());
    }
    int output_index = 0;
    for (auto &[key, cluster] : clusters)
    {
        (void)key;
        cluster.outputIndex = output_index++;
        const double inverse_count = 1.0 / std::max(1, cluster.count);
        MeshVertex vertex;
        vertex.x = static_cast<float>(cluster.x * inverse_count);
        vertex.y = static_cast<float>(cluster.y * inverse_count);
        vertex.z = static_cast<float>(cluster.z * inverse_count);
        vertex.nx = static_cast<float>(cluster.nx * inverse_count);
        vertex.ny = static_cast<float>(cluster.ny * inverse_count);
        vertex.nz = static_cast<float>(cluster.nz * inverse_count);
        vertex.r = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(std::lround(cluster.r * inverse_count)), 0, 255));
        vertex.g = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(std::lround(cluster.g * inverse_count)), 0, 255));
        vertex.b = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(std::lround(cluster.b * inverse_count)), 0, 255));
        result.vertices.push_back(vertex);
        if (outputProtectedBoundaryVertices)
        {
            outputProtectedBoundaryVertices->push_back(
                cluster.protectedBoundary ? std::uint8_t{1} : std::uint8_t{0});
        }
    }

    result.faces.reserve(mesh.faces.size());
    for (const Triangle &face : mesh.faces)
    {
        if (!validFace(face, mesh.vertices.size()))
        {
            continue;
        }
        Triangle remapped;
        remapped.v[0] = clusters.at(vertex_keys[static_cast<std::size_t>(face.v[0])]).outputIndex;
        remapped.v[1] = clusters.at(vertex_keys[static_cast<std::size_t>(face.v[1])]).outputIndex;
        remapped.v[2] = clusters.at(vertex_keys[static_cast<std::size_t>(face.v[2])]).outputIndex;
        if (remapped.v[0] != remapped.v[1] && remapped.v[1] != remapped.v[2] &&
            remapped.v[0] != remapped.v[2])
        {
            result.faces.push_back(remapped);
        }
    }
    return result;
}

} // namespace detail
} // namespace mesh
} // namespace xjw
