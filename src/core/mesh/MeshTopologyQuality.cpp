#include "MeshTopologyQuality.h"
#include "MeshIsotropicRemesher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct EdgeFaces
{
    int firstFace = -1;
    int secondFace = -1;
    int faceCount = 0;
};

struct FlipCandidate
{
    int firstFace = -1;
    int secondFace = -1;
    Triangle firstReplacement;
    Triangle secondReplacement;
    std::array<int, 4> vertices{{-1, -1, -1, -1}};
    double improvement = 0.0;
};

class DisjointSet
{
public:
    explicit DisjointSet(std::size_t size)
        : _parent(size),
          _rank(size, 0)
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

    void unite(int first, int second)
    {
        first = find(first);
        second = find(second);
        if (first == second)
        {
            return;
        }
        if (_rank[static_cast<std::size_t>(first)] <
            _rank[static_cast<std::size_t>(second)])
        {
            std::swap(first, second);
        }
        _parent[static_cast<std::size_t>(second)] = first;
        if (_rank[static_cast<std::size_t>(first)] ==
            _rank[static_cast<std::size_t>(second)])
        {
            ++_rank[static_cast<std::size_t>(first)];
        }
    }

private:
    std::vector<int> _parent;
    std::vector<std::uint8_t> _rank;
};

std::uint64_t edgeKey(int first, int second)
{
    const auto low = static_cast<std::uint32_t>(std::min(first, second));
    const auto high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) |
        static_cast<std::uint64_t>(high);
}

bool validFace(const TriMesh &mesh, const Triangle &face)
{
    return face.v[0] >= 0 && face.v[1] >= 0 && face.v[2] >= 0 &&
        face.v[0] != face.v[1] && face.v[1] != face.v[2] &&
        face.v[2] != face.v[0] &&
        static_cast<std::size_t>(face.v[0]) < mesh.vertices.size() &&
        static_cast<std::size_t>(face.v[1]) < mesh.vertices.size() &&
        static_cast<std::size_t>(face.v[2]) < mesh.vertices.size();
}

Vec3 positionOf(const MeshVertex &vertex)
{
    return {vertex.x, vertex.y, vertex.z};
}

Vec3 subtract(const Vec3 &left, const Vec3 &right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 cross(const Vec3 &left, const Vec3 &right)
{
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x};
}

double dot(const Vec3 &left, const Vec3 &right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double length(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

Vec3 faceNormal(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    const Vec3 normal = cross(subtract(second, first), subtract(third, first));
    const double magnitude = length(normal);
    return magnitude > 1.0e-20
        ? Vec3{normal.x / magnitude, normal.y / magnitude, normal.z / magnitude}
        : Vec3{};
}

double triangleDoubleArea(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    return length(cross(subtract(second, first), subtract(third, first)));
}

double squaredDistance(const Vec3 &first, const Vec3 &second)
{
    const Vec3 difference = subtract(first, second);
    return dot(difference, difference);
}

double triangleAspectRatio(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    const double longest_squared = std::max({
        squaredDistance(first, second),
        squaredDistance(second, third),
        squaredDistance(third, first)});
    const double doubled_area = triangleDoubleArea(mesh, face);
    return doubled_area > 1.0e-20
        ? longest_squared / doubled_area
        : std::numeric_limits<double>::infinity();
}

double triangleQuality(const TriMesh &mesh, const Triangle &face)
{
    constexpr double normalized_area_factor = 3.4641016151377544;
    const Vec3 first = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third = positionOf(
        mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    const double squared_edge_sum =
        squaredDistance(first, second) +
        squaredDistance(second, third) +
        squaredDistance(third, first);
    return squared_edge_sum > 1.0e-20
        ? normalized_area_factor * triangleDoubleArea(mesh, face) /
            squared_edge_sum
        : 0.0;
}

double quantile(std::vector<double> *values, double fraction)
{
    if (!values || values->empty())
    {
        return 0.0;
    }
    std::sort(values->begin(), values->end());
    const double position = std::clamp(fraction, 0.0, 1.0) *
        static_cast<double>(values->size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return (*values)[lower] * (1.0 - blend) + (*values)[upper] * blend;
}

int oppositeVertex(const Triangle &face, int first, int second)
{
    for (const int vertex : face.v)
    {
        if (vertex != first && vertex != second)
        {
            return vertex;
        }
    }
    return -1;
}

bool hasDirectedEdge(const Triangle &face, int first, int second)
{
    for (int corner = 0; corner < 3; ++corner)
    {
        if (face.v[corner] == first &&
            face.v[(corner + 1) % 3] == second)
        {
            return true;
        }
    }
    return false;
}

void addEdge(std::unordered_map<std::uint64_t, EdgeFaces> *edges,
             int first,
             int second,
             int faceIndex,
             DisjointSet *faceComponents = nullptr)
{
    EdgeFaces &record = (*edges)[edgeKey(first, second)];
    if (record.faceCount == 0)
    {
        record.firstFace = faceIndex;
    }
    else
    {
        if (record.faceCount == 1)
        {
            record.secondFace = faceIndex;
        }
        if (faceComponents)
        {
            faceComponents->unite(record.firstFace, faceIndex);
        }
    }
    ++record.faceCount;
}

bool hasManifoldVertexLink(
    const TriMesh &mesh,
    int vertexIndex,
    const std::vector<int> &incidentFaces)
{
    std::vector<std::array<int, 2>> link_edges;
    std::vector<int> link_vertices;
    link_edges.reserve(incidentFaces.size());
    link_vertices.reserve(incidentFaces.size() * 2);
    for (const int face_index : incidentFaces)
    {
        const Triangle &face = mesh.faces[static_cast<std::size_t>(face_index)];
        std::array<int, 2> opposite{{-1, -1}};
        int opposite_count = 0;
        for (const int vertex : face.v)
        {
            if (vertex != vertexIndex && opposite_count < 2)
            {
                opposite[static_cast<std::size_t>(opposite_count++)] = vertex;
            }
        }
        if (opposite_count != 2 || opposite[0] == opposite[1])
        {
            return false;
        }
        link_edges.push_back(opposite);
        link_vertices.push_back(opposite[0]);
        link_vertices.push_back(opposite[1]);
    }
    if (link_edges.empty())
    {
        return false;
    }

    std::sort(link_vertices.begin(), link_vertices.end());
    link_vertices.erase(
        std::unique(link_vertices.begin(), link_vertices.end()),
        link_vertices.end());
    DisjointSet link_components(link_vertices.size());
    std::vector<int> link_degrees(link_vertices.size(), 0);
    for (const std::array<int, 2> &edge : link_edges)
    {
        const int first = static_cast<int>(std::lower_bound(
            link_vertices.cbegin(), link_vertices.cend(), edge[0]) -
            link_vertices.cbegin());
        const int second = static_cast<int>(std::lower_bound(
            link_vertices.cbegin(), link_vertices.cend(), edge[1]) -
            link_vertices.cbegin());
        link_components.unite(first, second);
        ++link_degrees[static_cast<std::size_t>(first)];
        ++link_degrees[static_cast<std::size_t>(second)];
    }

    const int link_root = link_components.find(0);
    int degree_one_count = 0;
    for (std::size_t index = 0; index < link_vertices.size(); ++index)
    {
        if (link_components.find(static_cast<int>(index)) != link_root)
        {
            return false;
        }
        const int degree = link_degrees[index];
        if (degree == 1)
        {
            ++degree_one_count;
        }
        else if (degree != 2)
        {
            return false;
        }
    }
    return degree_one_count == 0 || degree_one_count == 2;
}

} // namespace

bool passesMeshTopologyQualityGate(
    const MeshTopologyQualityStatistics &statistics,
    const MeshTopologyQualityThresholds &thresholds)
{
    return statistics.validFaceCount > 0 &&
        statistics.boundaryEdgeRatio <= thresholds.maximumBoundaryEdgeRatio &&
        statistics.nonManifoldEdgeCount <=
            thresholds.maximumNonManifoldEdgeCount &&
        statistics.nonManifoldVertexCount <=
            thresholds.maximumNonManifoldVertexCount &&
        statistics.componentCount <= thresholds.maximumComponentCount &&
        statistics.largestComponentFaceRatio >=
            thresholds.minimumLargestComponentFaceRatio &&
        statistics.highAspectFaceRatio <=
            thresholds.maximumHighAspectFaceRatio &&
        statistics.extremeAspectFaceRatio <=
            thresholds.maximumExtremeAspectFaceRatio &&
        statistics.topologicalComplexity <=
            thresholds.maximumTopologicalComplexity &&
        (!statistics.closedTopologyEvaluated ||
         statistics.closedGenusEstimate <= thresholds.maximumClosedGenus);
}

MeshTopologyQualityStatistics evaluateMeshTopologyQuality(
    const TriMesh &mesh,
    const MeshTopologyQualityThresholds &thresholds)
{
    MeshTopologyQualityStatistics statistics;
    if (mesh.empty())
    {
        return statistics;
    }

    std::unordered_map<std::uint64_t, EdgeFaces> edges;
    edges.reserve(mesh.faces.size() * 2);
    std::vector<int> valid_face_indices;
    valid_face_indices.reserve(mesh.faces.size());
    std::vector<Vec3> face_normals(mesh.faces.size());
    std::vector<std::uint8_t> referenced_vertices(mesh.vertices.size(), 0);
    std::vector<std::vector<int>> incident_faces(mesh.vertices.size());
    DisjointSet face_components(mesh.faces.size());
    for (std::size_t face_index = 0;
         face_index < mesh.faces.size();
         ++face_index)
    {
        const Triangle &face = mesh.faces[face_index];
        if (!validFace(mesh, face))
        {
            continue;
        }
        valid_face_indices.push_back(static_cast<int>(face_index));
        face_normals[face_index] = faceNormal(mesh, face);
        ++statistics.validFaceCount;
        addEdge(
            &edges,
            face.v[0],
            face.v[1],
            static_cast<int>(face_index),
            &face_components);
        addEdge(
            &edges,
            face.v[1],
            face.v[2],
            static_cast<int>(face_index),
            &face_components);
        addEdge(
            &edges,
            face.v[2],
            face.v[0],
            static_cast<int>(face_index),
            &face_components);
        for (const int vertex : face.v)
        {
            referenced_vertices[static_cast<std::size_t>(vertex)] = 1;
            incident_faces[static_cast<std::size_t>(vertex)].push_back(
                static_cast<int>(face_index));
        }

        const double quality = triangleQuality(mesh, face);
        const double aspect = triangleAspectRatio(mesh, face);
        if (!std::isfinite(quality) ||
            quality < thresholds.skinnyTriangleQuality)
        {
            ++statistics.skinnyFaceCount;
        }
        if (!std::isfinite(aspect) || aspect > thresholds.highAspectRatio)
        {
            ++statistics.highAspectFaceCount;
        }
        if (!std::isfinite(aspect) || aspect > thresholds.extremeAspectRatio)
        {
            ++statistics.extremeAspectFaceCount;
        }
    }

    statistics.uniqueEdgeCount = static_cast<int>(edges.size());
    std::vector<double> adjacent_normal_angles;
    adjacent_normal_angles.reserve(edges.size());
    int adjacent_over_30_count = 0;
    constexpr double radians_to_degrees =
        57.295779513082320876798154814105;
    for (const auto &[key, edge] : edges)
    {
        (void)key;
        if (edge.faceCount == 1)
        {
            ++statistics.boundaryEdgeCount;
        }
        else if (edge.faceCount > 2)
        {
            ++statistics.nonManifoldEdgeCount;
        }
        if (edge.firstFace >= 0 && edge.secondFace >= 0)
        {
            if (edge.faceCount == 2)
            {
                const Vec3 &first_normal =
                    face_normals[static_cast<std::size_t>(edge.firstFace)];
                const Vec3 &second_normal =
                    face_normals[static_cast<std::size_t>(edge.secondFace)];
                const double angle = std::acos(std::clamp(
                    dot(first_normal, second_normal), -1.0, 1.0)) *
                    radians_to_degrees;
                adjacent_normal_angles.push_back(angle);
                adjacent_over_30_count += angle > 30.0 ? 1 : 0;
            }
        }
    }
    statistics.adjacentFacePairCount =
        static_cast<int>(adjacent_normal_angles.size());
    if (!adjacent_normal_angles.empty())
    {
        statistics.adjacentNormalAngleMedianDegrees =
            quantile(&adjacent_normal_angles, 0.50);
        statistics.adjacentNormalAngleP90Degrees =
            quantile(&adjacent_normal_angles, 0.90);
        statistics.adjacentNormalAngleOver30Ratio =
            static_cast<double>(adjacent_over_30_count) /
            static_cast<double>(adjacent_normal_angles.size());
    }

    std::unordered_map<int, int> component_face_counts;
    component_face_counts.reserve(valid_face_indices.size());
    for (const int face_index : valid_face_indices)
    {
        const int count = ++component_face_counts[face_components.find(face_index)];
        statistics.largestComponentFaceCount =
            std::max(statistics.largestComponentFaceCount, count);
    }
    statistics.componentCount =
        static_cast<int>(component_face_counts.size());

    std::unordered_map<int, int> component_edge_counts;
    component_edge_counts.reserve(component_face_counts.size());
    for (const auto &[key, edge] : edges)
    {
        (void)key;
        if (edge.firstFace >= 0)
        {
            ++component_edge_counts[face_components.find(edge.firstFace)];
        }
    }
    std::unordered_map<int, std::unordered_set<int>> component_vertices;
    component_vertices.reserve(component_face_counts.size());
    for (const int face_index : valid_face_indices)
    {
        const int component = face_components.find(face_index);
        const Triangle &face = mesh.faces[static_cast<std::size_t>(face_index)];
        std::unordered_set<int> &vertices = component_vertices[component];
        vertices.insert(face.v[0]);
        vertices.insert(face.v[1]);
        vertices.insert(face.v[2]);
    }
    statistics.componentEulerCharacteristics.reserve(
        component_face_counts.size());
    for (const auto &[component, face_count] : component_face_counts)
    {
        const int vertex_count = static_cast<int>(
            component_vertices[component].size());
        const int edge_count = component_edge_counts[component];
        statistics.componentEulerCharacteristics.push_back(
            vertex_count - edge_count + face_count);
    }
    std::sort(
        statistics.componentEulerCharacteristics.begin(),
        statistics.componentEulerCharacteristics.end());

    statistics.referencedVertexCount = static_cast<int>(std::count(
        referenced_vertices.cbegin(),
        referenced_vertices.cend(),
        static_cast<std::uint8_t>(1)));
    for (std::size_t vertex_index = 0;
         vertex_index < incident_faces.size();
         ++vertex_index)
    {
        if (!incident_faces[vertex_index].empty() &&
            !hasManifoldVertexLink(
                mesh,
                static_cast<int>(vertex_index),
                incident_faces[vertex_index]))
        {
            ++statistics.nonManifoldVertexCount;
        }
    }
    statistics.eulerCharacteristic =
        statistics.referencedVertexCount -
        statistics.uniqueEdgeCount +
        statistics.validFaceCount;
    statistics.topologicalComplexity = std::max(
        0,
        2 * statistics.componentCount -
            statistics.eulerCharacteristic);
    statistics.closedTwoManifold =
        statistics.validFaceCount > 0 &&
        statistics.boundaryEdgeCount == 0 &&
        statistics.nonManifoldEdgeCount == 0 &&
        statistics.nonManifoldVertexCount == 0;
    statistics.closedTopologyEvaluated = statistics.closedTwoManifold;
    if (statistics.closedTopologyEvaluated)
    {
        statistics.closedGenusEstimate = std::max(
            0.0,
            static_cast<double>(statistics.topologicalComplexity) / 2.0);
    }
    if (statistics.uniqueEdgeCount > 0)
    {
        statistics.boundaryEdgeRatio =
            static_cast<double>(statistics.boundaryEdgeCount) /
            statistics.uniqueEdgeCount;
    }
    if (statistics.validFaceCount > 0)
    {
        const double reciprocal = 1.0 / statistics.validFaceCount;
        statistics.largestComponentFaceRatio =
            statistics.largestComponentFaceCount * reciprocal;
        statistics.skinnyFaceRatio =
            statistics.skinnyFaceCount * reciprocal;
        statistics.highAspectFaceRatio =
            statistics.highAspectFaceCount * reciprocal;
        statistics.extremeAspectFaceRatio =
            statistics.extremeAspectFaceCount * reciprocal;
    }
    statistics.strictGatePassed =
        passesMeshTopologyQualityGate(statistics, thresholds);
    return statistics;
}

MeshTopologySignature meshTopologySignature(
    const MeshTopologyQualityStatistics &statistics)
{
    MeshTopologySignature signature;
    signature.nonManifoldEdgeCount = statistics.nonManifoldEdgeCount;
    signature.nonManifoldVertexCount = statistics.nonManifoldVertexCount;
    signature.closedTwoManifold = statistics.closedTwoManifold;
    signature.componentEulerCharacteristics =
        statistics.componentEulerCharacteristics;
    return signature;
}

MeshTopologySignature evaluateMeshTopologySignature(const TriMesh &mesh)
{
    return meshTopologySignature(evaluateMeshTopologyQuality(mesh));
}

MeshTriangleOptimizationStatistics optimizeTriangleQuality(
    TriMesh *mesh,
    const MeshTriangleOptimizationOptions &options)
{
    MeshTriangleOptimizationStatistics statistics;
    if (!mesh)
    {
        return statistics;
    }
    statistics.inputFaceCount = mesh->faceCount();
    if (mesh->empty())
    {
        statistics.outputFaceCount = mesh->faceCount();
        return statistics;
    }

    constexpr double pi = 3.14159265358979323846;
    const double feature_cosine = std::cos(
        std::clamp(options.maximumFeatureAngleDegrees, 0.0, 180.0) *
        pi / 180.0);
    const double normal_cosine = std::cos(
        std::clamp(options.maximumNormalDeviationDegrees, 0.0, 180.0) *
        pi / 180.0);
    const double minimum_improvement = std::clamp(
        options.minimumWorstAspectImprovementRatio, 0.0, 0.95);
    const auto cancellationRequested = [&options]()
    {
        return options.isCancelled && options.isCancelled();
    };

    if (options.enableIsotropicRemeshing)
    {
        MeshIsotropicRemeshOptions remesh_options;
        remesh_options.maximumPasses = options.isotropicRemeshingPasses;
        remesh_options.shortEdgeRatio = options.isotropicShortEdgeRatio;
        remesh_options.longEdgeRatio = options.isotropicLongEdgeRatio;
        remesh_options.minimumAffectedAspectRatio = 10.0;
        remesh_options.minimumWorstAspectImprovementRatio =
            options.minimumWorstAspectImprovementRatio;
        remesh_options.maximumFeatureAngleDegrees =
            options.maximumFeatureAngleDegrees;
        remesh_options.maximumNormalDeviationDegrees =
            options.maximumNormalDeviationDegrees;
        remesh_options.maximumFaceGrowthRatio =
            options.isotropicMaximumFaceGrowthRatio;
        remesh_options.isCancelled = options.isCancelled;
        const MeshIsotropicRemeshStatistics remeshed =
            remeshInteriorHighAspectTriangles(mesh, remesh_options);
        statistics.isotropicRemeshingPassCount = remeshed.passCount;
        statistics.isotropicCollapsedEdgeCount = remeshed.collapsedEdgeCount;
        statistics.isotropicSplitEdgeCount = remeshed.splitEdgeCount;
        statistics.cancelled = remeshed.cancelled;
    }

    for (int pass = 0;
         !statistics.cancelled &&
             pass < std::max(1, options.maximumPasses);
         ++pass)
    {
        if (cancellationRequested())
        {
            statistics.cancelled = true;
            break;
        }

        std::unordered_map<std::uint64_t, EdgeFaces> edges;
        edges.reserve(mesh->faces.size() * 2);
        for (std::size_t face_index = 0;
             face_index < mesh->faces.size();
             ++face_index)
        {
            const Triangle &face = mesh->faces[face_index];
            if (!validFace(*mesh, face))
            {
                continue;
            }
            addEdge(&edges, face.v[0], face.v[1], static_cast<int>(face_index));
            addEdge(&edges, face.v[1], face.v[2], static_cast<int>(face_index));
            addEdge(&edges, face.v[2], face.v[0], static_cast<int>(face_index));
        }

        std::vector<FlipCandidate> candidates;
        candidates.reserve(edges.size() / 4);
        std::size_t examined_edge_count = 0;
        for (const auto &[key, edge] : edges)
        {
            if ((examined_edge_count++ & 4095U) == 0U &&
                cancellationRequested())
            {
                statistics.cancelled = true;
                break;
            }
            if (edge.faceCount != 2 ||
                edge.firstFace < 0 || edge.secondFace < 0)
            {
                continue;
            }
            const int first = static_cast<int>(key >> 32U);
            const int second = static_cast<int>(key & 0xffffffffU);
            const Triangle &first_face =
                mesh->faces[static_cast<std::size_t>(edge.firstFace)];
            const Triangle &second_face =
                mesh->faces[static_cast<std::size_t>(edge.secondFace)];
            const int first_opposite =
                oppositeVertex(first_face, first, second);
            const int second_opposite =
                oppositeVertex(second_face, first, second);
            if (first_opposite < 0 || second_opposite < 0 ||
                first_opposite == second_opposite)
            {
                continue;
            }
            if (edges.find(edgeKey(first_opposite, second_opposite)) !=
                edges.cend())
            {
                ++statistics.rejectedExistingDiagonalCount;
                continue;
            }

            const Vec3 first_normal = faceNormal(*mesh, first_face);
            const Vec3 second_normal = faceNormal(*mesh, second_face);
            if (dot(first_normal, second_normal) < feature_cosine)
            {
                ++statistics.rejectedFeatureEdgeCount;
                continue;
            }

            const bool forward =
                hasDirectedEdge(first_face, first, second);
            const int directed_first = forward ? first : second;
            const int directed_second = forward ? second : first;
            Triangle first_replacement;
            first_replacement.v[0] = first_opposite;
            first_replacement.v[1] = directed_first;
            first_replacement.v[2] = second_opposite;
            Triangle second_replacement;
            second_replacement.v[0] = first_opposite;
            second_replacement.v[1] = second_opposite;
            second_replacement.v[2] = directed_second;
            if (!validFace(*mesh, first_replacement) ||
                !validFace(*mesh, second_replacement))
            {
                continue;
            }

            const Vec3 first_new_normal =
                faceNormal(*mesh, first_replacement);
            const Vec3 second_new_normal =
                faceNormal(*mesh, second_replacement);
            if (dot(first_new_normal, first_normal) < normal_cosine ||
                dot(second_new_normal, second_normal) < normal_cosine)
            {
                ++statistics.rejectedNormalCount;
                continue;
            }

            const double old_worst_aspect = std::max(
                triangleAspectRatio(*mesh, first_face),
                triangleAspectRatio(*mesh, second_face));
            const double new_worst_aspect = std::max(
                triangleAspectRatio(*mesh, first_replacement),
                triangleAspectRatio(*mesh, second_replacement));
            const double old_aspect_sum =
                triangleAspectRatio(*mesh, first_face) +
                triangleAspectRatio(*mesh, second_face);
            const double new_aspect_sum =
                triangleAspectRatio(*mesh, first_replacement) +
                triangleAspectRatio(*mesh, second_replacement);
            if (!std::isfinite(new_worst_aspect) ||
                new_worst_aspect >= old_worst_aspect *
                    (1.0 - minimum_improvement) ||
                new_aspect_sum > old_aspect_sum)
            {
                ++statistics.rejectedQualityCount;
                continue;
            }
            candidates.push_back({
                edge.firstFace,
                edge.secondFace,
                first_replacement,
                second_replacement,
                {{first, second, first_opposite, second_opposite}},
                old_worst_aspect - new_worst_aspect});
        }
        if (statistics.cancelled)
        {
            break;
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const FlipCandidate &left, const FlipCandidate &right)
            {
                return left.improvement > right.improvement;
            });

        std::vector<std::uint8_t> used_faces(mesh->faces.size(), 0);
        std::vector<std::uint8_t> used_vertices(mesh->vertices.size(), 0);
        int pass_flip_count = 0;
        for (const FlipCandidate &candidate : candidates)
        {
            if (used_faces[static_cast<std::size_t>(candidate.firstFace)] ||
                used_faces[static_cast<std::size_t>(candidate.secondFace)])
            {
                continue;
            }
            const bool touches_used_vertex = std::any_of(
                candidate.vertices.cbegin(),
                candidate.vertices.cend(),
                [&used_vertices](int vertex)
                {
                    return used_vertices[static_cast<std::size_t>(vertex)] != 0;
                });
            if (touches_used_vertex)
            {
                continue;
            }
            mesh->faces[static_cast<std::size_t>(candidate.firstFace)] =
                candidate.firstReplacement;
            mesh->faces[static_cast<std::size_t>(candidate.secondFace)] =
                candidate.secondReplacement;
            used_faces[static_cast<std::size_t>(candidate.firstFace)] = 1;
            used_faces[static_cast<std::size_t>(candidate.secondFace)] = 1;
            for (const int vertex : candidate.vertices)
            {
                used_vertices[static_cast<std::size_t>(vertex)] = 1;
            }
            ++pass_flip_count;
        }
        if (pass_flip_count == 0)
        {
            break;
        }
        statistics.flippedEdgeCount += pass_flip_count;
        statistics.passCount = pass + 1;
    }

    if (options.enableTangentialRelaxation && !statistics.cancelled)
    {
        struct RelaxCandidate
        {
            int vertex = -1;
            MeshVertex replacement;
            double improvement = 0.0;
        };

        constexpr double minimum_length = 1.0e-12;
        const double relaxation_lambda = std::clamp(
            options.tangentialRelaxationLambda, 0.0, 1.0);
        const double maximum_displacement_ratio = std::clamp(
            options.tangentialMaximumDisplacementEdgeRatio, 0.0, 0.50);
        for (int pass = 0;
             pass < std::max(0, options.tangentialRelaxationPasses);
             ++pass)
        {
            if (cancellationRequested())
            {
                statistics.cancelled = true;
                break;
            }

            std::unordered_map<std::uint64_t, EdgeFaces> edges;
            edges.reserve(mesh->faces.size() * 2);
            std::vector<std::vector<int>> adjacent_vertices(mesh->vertices.size());
            std::vector<std::vector<int>> incident_faces(mesh->vertices.size());
            std::vector<Vec3> normals(mesh->faces.size());
            for (std::size_t face_index = 0;
                 face_index < mesh->faces.size();
                 ++face_index)
            {
                const Triangle &face = mesh->faces[face_index];
                if (!validFace(*mesh, face))
                {
                    continue;
                }
                normals[face_index] = faceNormal(*mesh, face);
                addEdge(&edges, face.v[0], face.v[1], static_cast<int>(face_index));
                addEdge(&edges, face.v[1], face.v[2], static_cast<int>(face_index));
                addEdge(&edges, face.v[2], face.v[0], static_cast<int>(face_index));
                for (int corner = 0; corner < 3; ++corner)
                {
                    const int vertex = face.v[corner];
                    incident_faces[static_cast<std::size_t>(vertex)].push_back(
                        static_cast<int>(face_index));
                    adjacent_vertices[static_cast<std::size_t>(vertex)].push_back(
                        face.v[(corner + 1) % 3]);
                    adjacent_vertices[static_cast<std::size_t>(vertex)].push_back(
                        face.v[(corner + 2) % 3]);
                }
            }
            for (auto &adjacent : adjacent_vertices)
            {
                std::sort(adjacent.begin(), adjacent.end());
                adjacent.erase(
                    std::unique(adjacent.begin(), adjacent.end()),
                    adjacent.end());
            }

            std::vector<std::uint8_t> protected_vertices(mesh->vertices.size(), 0);
            for (const auto &[key, edge] : edges)
            {
                const int first = static_cast<int>(key >> 32U);
                const int second = static_cast<int>(key & 0xffffffffU);
                const bool topology_feature = edge.faceCount != 2;
                const bool normal_feature =
                    edge.faceCount == 2 &&
                    dot(normals[static_cast<std::size_t>(edge.firstFace)],
                        normals[static_cast<std::size_t>(edge.secondFace)]) <
                        feature_cosine;
                if (topology_feature || normal_feature)
                {
                    protected_vertices[static_cast<std::size_t>(first)] = 1;
                    protected_vertices[static_cast<std::size_t>(second)] = 1;
                }
            }

            std::vector<RelaxCandidate> candidates;
            candidates.reserve(mesh->vertices.size() / 8);
            for (std::size_t vertex_index = 0;
                 vertex_index < mesh->vertices.size();
                 ++vertex_index)
            {
                if ((vertex_index & 4095U) == 0U &&
                    cancellationRequested())
                {
                    statistics.cancelled = true;
                    break;
                }
                const std::vector<int> &neighbors =
                    adjacent_vertices[vertex_index];
                const std::vector<int> &faces =
                    incident_faces[vertex_index];
                if (protected_vertices[vertex_index] ||
                    neighbors.size() < 3 || faces.size() < 3)
                {
                    continue;
                }

                const MeshVertex original = mesh->vertices[vertex_index];
                Vec3 normal_sum;
                double mean_edge_length = 0.0;
                Vec3 mean_position;
                for (const int face_index : faces)
                {
                    const Triangle &face =
                        mesh->faces[static_cast<std::size_t>(face_index)];
                    const Vec3 first = positionOf(mesh->vertices[
                        static_cast<std::size_t>(face.v[0])]);
                    const Vec3 second = positionOf(mesh->vertices[
                        static_cast<std::size_t>(face.v[1])]);
                    const Vec3 third = positionOf(mesh->vertices[
                        static_cast<std::size_t>(face.v[2])]);
                    const Vec3 weighted_normal = cross(
                        subtract(second, first), subtract(third, first));
                    normal_sum.x += weighted_normal.x;
                    normal_sum.y += weighted_normal.y;
                    normal_sum.z += weighted_normal.z;
                }
                for (const int neighbor : neighbors)
                {
                    const Vec3 position = positionOf(mesh->vertices[
                        static_cast<std::size_t>(neighbor)]);
                    mean_position.x += position.x;
                    mean_position.y += position.y;
                    mean_position.z += position.z;
                    mean_edge_length += std::sqrt(squaredDistance(
                        positionOf(original), position));
                }
                const double normal_length = length(normal_sum);
                if (normal_length <= minimum_length)
                {
                    continue;
                }
                const double inverse_neighbor_count =
                    1.0 / static_cast<double>(neighbors.size());
                mean_position.x *= inverse_neighbor_count;
                mean_position.y *= inverse_neighbor_count;
                mean_position.z *= inverse_neighbor_count;
                mean_edge_length *= inverse_neighbor_count;
                const Vec3 unit_normal{
                    normal_sum.x / normal_length,
                    normal_sum.y / normal_length,
                    normal_sum.z / normal_length};
                const Vec3 original_position = positionOf(original);
                Vec3 displacement = subtract(mean_position, original_position);
                const double normal_offset = dot(displacement, unit_normal);
                displacement.x -= normal_offset * unit_normal.x;
                displacement.y -= normal_offset * unit_normal.y;
                displacement.z -= normal_offset * unit_normal.z;
                displacement.x *= relaxation_lambda;
                displacement.y *= relaxation_lambda;
                displacement.z *= relaxation_lambda;
                const double displacement_length = length(displacement);
                const double maximum_displacement =
                    mean_edge_length * maximum_displacement_ratio;
                if (displacement_length <= minimum_length ||
                    maximum_displacement <= minimum_length)
                {
                    continue;
                }
                if (displacement_length > maximum_displacement)
                {
                    const double scale =
                        maximum_displacement / displacement_length;
                    displacement.x *= scale;
                    displacement.y *= scale;
                    displacement.z *= scale;
                }

                MeshVertex replacement = original;
                replacement.x += static_cast<float>(displacement.x);
                replacement.y += static_cast<float>(displacement.y);
                replacement.z += static_cast<float>(displacement.z);
                double old_worst_aspect = 0.0;
                double new_worst_aspect = 0.0;
                double old_aspect_sum = 0.0;
                double new_aspect_sum = 0.0;
                bool normals_are_safe = true;
                for (const int face_index : faces)
                {
                    const Triangle &face =
                        mesh->faces[static_cast<std::size_t>(face_index)];
                    const double old_aspect = triangleAspectRatio(*mesh, face);
                    old_worst_aspect = std::max(old_worst_aspect, old_aspect);
                    old_aspect_sum += old_aspect;
                }
                mesh->vertices[vertex_index] = replacement;
                for (const int face_index : faces)
                {
                    const Triangle &face =
                        mesh->faces[static_cast<std::size_t>(face_index)];
                    const double new_aspect = triangleAspectRatio(*mesh, face);
                    new_worst_aspect = std::max(new_worst_aspect, new_aspect);
                    new_aspect_sum += new_aspect;
                    normals_are_safe = normals_are_safe &&
                        dot(faceNormal(*mesh, face),
                            normals[static_cast<std::size_t>(face_index)]) >=
                            normal_cosine;
                }
                mesh->vertices[vertex_index] = original;
                if (!normals_are_safe ||
                    !std::isfinite(new_worst_aspect) ||
                    new_worst_aspect >= old_worst_aspect *
                        (1.0 - minimum_improvement) ||
                    new_aspect_sum > old_aspect_sum)
                {
                    continue;
                }
                candidates.push_back({
                    static_cast<int>(vertex_index),
                    replacement,
                    old_worst_aspect - new_worst_aspect});
            }
            if (statistics.cancelled)
            {
                break;
            }
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const RelaxCandidate &left, const RelaxCandidate &right)
                {
                    return left.improvement > right.improvement;
                });
            std::vector<std::uint8_t> used_vertices(mesh->vertices.size(), 0);
            int moved_vertex_count = 0;
            for (const RelaxCandidate &candidate : candidates)
            {
                const std::size_t vertex_index =
                    static_cast<std::size_t>(candidate.vertex);
                if (used_vertices[vertex_index])
                {
                    continue;
                }
                const bool touches_moved_neighbor = std::any_of(
                    adjacent_vertices[vertex_index].cbegin(),
                    adjacent_vertices[vertex_index].cend(),
                    [&used_vertices](int neighbor)
                    {
                        return used_vertices[
                            static_cast<std::size_t>(neighbor)] != 0;
                    });
                if (touches_moved_neighbor)
                {
                    continue;
                }
                mesh->vertices[vertex_index] = candidate.replacement;
                used_vertices[vertex_index] = 1;
                ++moved_vertex_count;
            }
            if (moved_vertex_count == 0)
            {
                break;
            }
            statistics.tangentialRelaxedVertexCount += moved_vertex_count;
            statistics.tangentialRelaxationPassCount = pass + 1;
        }
    }

    statistics.outputFaceCount = mesh->faceCount();
    return statistics;
}

} // namespace xjw::mesh
