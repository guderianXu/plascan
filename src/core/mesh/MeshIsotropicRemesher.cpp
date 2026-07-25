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

struct EdgeRecord
{
    int firstFace = -1;
    int secondFace = -1;
    int faceCount = 0;
};

struct Topology
{
    std::unordered_map<std::uint64_t, EdgeRecord> edges;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<int>> incidentFaces;
    std::vector<Vec3> faceNormals;
    std::vector<std::uint8_t> protectedVertices;
    double medianInteriorEdgeLength = 0.0;
};

struct CollapseCandidate
{
    int keep = -1;
    int remove = -1;
    MeshVertex replacement;
    std::vector<int> affectedVertices;
    double improvement = 0.0;
};

struct SplitCandidate
{
    int first = -1;
    int second = -1;
    int firstFace = -1;
    int secondFace = -1;
    double improvement = 0.0;
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

double edgeLength(const TriMesh &mesh, int first, int second)
{
    return length(subtract(
        positionOf(mesh.vertices[static_cast<std::size_t>(first)]),
        positionOf(mesh.vertices[static_cast<std::size_t>(second)])));
}

Vec3 faceNormal(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    const Vec3 normal = cross(subtract(second, first), subtract(third, first));
    const double magnitude = length(normal);
    return magnitude > 1.0e-20
        ? Vec3{normal.x / magnitude, normal.y / magnitude, normal.z / magnitude}
        : Vec3{};
}

double triangleAspectRatio(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[0])]);
    const Vec3 second =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[1])]);
    const Vec3 third =
        positionOf(mesh.vertices[static_cast<std::size_t>(face.v[2])]);
    const Vec3 first_edge = subtract(second, first);
    const Vec3 second_edge = subtract(third, second);
    const Vec3 third_edge = subtract(first, third);
    const double longest_squared = std::max({
        dot(first_edge, first_edge),
        dot(second_edge, second_edge),
        dot(third_edge, third_edge)});
    const double doubled_area = length(cross(first_edge, subtract(third, first)));
    return doubled_area > 1.0e-20
        ? longest_squared / doubled_area
        : std::numeric_limits<double>::infinity();
}

void addEdge(Topology *topology, int first, int second, int face_index)
{
    EdgeRecord &edge = topology->edges[edgeKey(first, second)];
    if (edge.faceCount == 0)
    {
        edge.firstFace = face_index;
    }
    else if (edge.faceCount == 1)
    {
        edge.secondFace = face_index;
    }
    ++edge.faceCount;
    topology->neighbors[static_cast<std::size_t>(first)].push_back(second);
    topology->neighbors[static_cast<std::size_t>(second)].push_back(first);
}

Topology buildTopology(const TriMesh &mesh, double feature_cosine)
{
    Topology topology;
    topology.edges.reserve(mesh.faces.size() * 2);
    topology.neighbors.resize(mesh.vertices.size());
    topology.incidentFaces.resize(mesh.vertices.size());
    topology.faceNormals.resize(mesh.faces.size());
    topology.protectedVertices.assign(mesh.vertices.size(), 0);
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        const Triangle &face = mesh.faces[face_index];
        if (!validFace(mesh, face))
        {
            continue;
        }
        topology.faceNormals[face_index] = faceNormal(mesh, face);
        addEdge(&topology, face.v[0], face.v[1], static_cast<int>(face_index));
        addEdge(&topology, face.v[1], face.v[2], static_cast<int>(face_index));
        addEdge(&topology, face.v[2], face.v[0], static_cast<int>(face_index));
        for (const int vertex : face.v)
        {
            topology.incidentFaces[static_cast<std::size_t>(vertex)].push_back(
                static_cast<int>(face_index));
        }
    }
    std::vector<double> interior_lengths;
    interior_lengths.reserve(topology.edges.size());
    for (const auto &[key, edge] : topology.edges)
    {
        const int first = static_cast<int>(key >> 32U);
        const int second = static_cast<int>(key & 0xffffffffU);
        const bool feature =
            edge.faceCount != 2 ||
            dot(topology.faceNormals[static_cast<std::size_t>(edge.firstFace)],
                topology.faceNormals[static_cast<std::size_t>(edge.secondFace)]) <
                feature_cosine;
        if (feature)
        {
            topology.protectedVertices[static_cast<std::size_t>(first)] = 1;
            topology.protectedVertices[static_cast<std::size_t>(second)] = 1;
        }
        if (edge.faceCount == 2)
        {
            interior_lengths.push_back(edgeLength(mesh, first, second));
        }
    }
    for (auto &neighbors : topology.neighbors)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    if (!interior_lengths.empty())
    {
        const auto middle =
            interior_lengths.begin() + interior_lengths.size() / 2;
        std::nth_element(
            interior_lengths.begin(), middle, interior_lengths.end());
        topology.medianInteriorEdgeLength = *middle;
    }
    return topology;
}

MeshVertex midpointVertex(const MeshVertex &first, const MeshVertex &second)
{
    MeshVertex result;
    result.x = 0.5f * (first.x + second.x);
    result.y = 0.5f * (first.y + second.y);
    result.z = 0.5f * (first.z + second.z);
    result.nx = first.nx + second.nx;
    result.ny = first.ny + second.ny;
    result.nz = first.nz + second.nz;
    const float normal_length = std::sqrt(
        result.nx * result.nx + result.ny * result.ny +
        result.nz * result.nz);
    if (normal_length > 1.0e-12f)
    {
        result.nx /= normal_length;
        result.ny /= normal_length;
        result.nz /= normal_length;
    }
    result.r = static_cast<std::uint8_t>(
        (static_cast<int>(first.r) + static_cast<int>(second.r)) / 2);
    result.g = static_cast<std::uint8_t>(
        (static_cast<int>(first.g) + static_cast<int>(second.g)) / 2);
    result.b = static_cast<std::uint8_t>(
        (static_cast<int>(first.b) + static_cast<int>(second.b)) / 2);
    return result;
}

int commonNeighborCount(const Topology &topology, int first, int second)
{
    const auto &first_neighbors =
        topology.neighbors[static_cast<std::size_t>(first)];
    const auto &second_neighbors =
        topology.neighbors[static_cast<std::size_t>(second)];
    int count = 0;
    auto first_iterator = first_neighbors.cbegin();
    auto second_iterator = second_neighbors.cbegin();
    while (first_iterator != first_neighbors.cend() &&
           second_iterator != second_neighbors.cend())
    {
        if (*first_iterator < *second_iterator)
        {
            ++first_iterator;
        }
        else if (*second_iterator < *first_iterator)
        {
            ++second_iterator;
        }
        else
        {
            ++count;
            ++first_iterator;
            ++second_iterator;
        }
    }
    return count;
}

std::vector<int> localFaces(const Topology &topology, int first, int second)
{
    std::vector<int> faces =
        topology.incidentFaces[static_cast<std::size_t>(first)];
    const auto &second_faces =
        topology.incidentFaces[static_cast<std::size_t>(second)];
    faces.insert(faces.end(), second_faces.cbegin(), second_faces.cend());
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    return faces;
}

bool collapseCandidateIsSafe(TriMesh *mesh,
                             const Topology &topology,
                             int keep,
                             int remove,
                             const MeshVertex &replacement,
                             double affected_aspect,
                             double normal_cosine,
                             double minimum_improvement,
                             double *improvement)
{
    const std::vector<int> faces = localFaces(topology, keep, remove);
    double old_worst = 0.0;
    double new_worst = 0.0;
    int old_affected_count = 0;
    int new_affected_count = 0;
    const MeshVertex original = mesh->vertices[static_cast<std::size_t>(keep)];
    for (const int face_index : faces)
    {
        const double aspect = triangleAspectRatio(
            *mesh, mesh->faces[static_cast<std::size_t>(face_index)]);
        old_worst = std::max(old_worst, aspect);
        old_affected_count += aspect > affected_aspect ? 1 : 0;
    }
    mesh->vertices[static_cast<std::size_t>(keep)] = replacement;
    bool safe = true;
    for (const int face_index : faces)
    {
        Triangle face = mesh->faces[static_cast<std::size_t>(face_index)];
        const bool contains_keep =
            face.v[0] == keep || face.v[1] == keep || face.v[2] == keep;
        const bool contains_remove =
            face.v[0] == remove || face.v[1] == remove || face.v[2] == remove;
        if (contains_keep && contains_remove)
        {
            continue;
        }
        for (int &vertex : face.v)
        {
            if (vertex == remove)
            {
                vertex = keep;
            }
        }
        if (!validFace(*mesh, face))
        {
            safe = false;
            break;
        }
        const double aspect = triangleAspectRatio(*mesh, face);
        new_worst = std::max(new_worst, aspect);
        new_affected_count += aspect > affected_aspect ? 1 : 0;
        if (!std::isfinite(aspect) ||
            dot(faceNormal(*mesh, face),
                topology.faceNormals[static_cast<std::size_t>(face_index)]) <
                normal_cosine)
        {
            safe = false;
            break;
        }
    }
    mesh->vertices[static_cast<std::size_t>(keep)] = original;
    if (!safe ||
        new_affected_count > old_affected_count ||
        new_worst > old_worst * (1.0 - minimum_improvement))
    {
        return false;
    }
    *improvement =
        static_cast<double>(old_affected_count - new_affected_count) * 1000.0 +
        old_worst - new_worst;
    return *improvement > 0.0;
}

void compactMesh(TriMesh *mesh)
{
    std::vector<std::uint8_t> used(mesh->vertices.size(), 0);
    for (const Triangle &face : mesh->faces)
    {
        for (const int vertex : face.v)
        {
            used[static_cast<std::size_t>(vertex)] = 1;
        }
    }
    std::vector<int> remap(mesh->vertices.size(), -1);
    std::vector<MeshVertex> vertices;
    vertices.reserve(mesh->vertices.size());
    for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
    {
        if (used[index])
        {
            remap[index] = static_cast<int>(vertices.size());
            vertices.push_back(mesh->vertices[index]);
        }
    }
    for (Triangle &face : mesh->faces)
    {
        for (int &vertex : face.v)
        {
            vertex = remap[static_cast<std::size_t>(vertex)];
        }
    }
    mesh->vertices = std::move(vertices);
}

int collapseShortEdges(TriMesh *mesh,
                       const Topology &topology,
                       const MeshIsotropicRemeshOptions &options,
                       double normal_cosine,
                       MeshIsotropicRemeshStatistics *statistics)
{
    const double threshold =
        topology.medianInteriorEdgeLength * options.shortEdgeRatio;
    std::vector<CollapseCandidate> candidates;
    for (const auto &[key, edge] : topology.edges)
    {
        if (edge.faceCount != 2)
        {
            continue;
        }
        const int first = static_cast<int>(key >> 32U);
        const int second = static_cast<int>(key & 0xffffffffU);
        if (edgeLength(*mesh, first, second) >= threshold)
        {
            continue;
        }
        const bool affected =
            triangleAspectRatio(
                *mesh, mesh->faces[static_cast<std::size_t>(edge.firstFace)]) >
                options.minimumAffectedAspectRatio ||
            triangleAspectRatio(
                *mesh, mesh->faces[static_cast<std::size_t>(edge.secondFace)]) >
                options.minimumAffectedAspectRatio;
        if (!affected)
        {
            continue;
        }
        if (topology.protectedVertices[static_cast<std::size_t>(first)] ||
            topology.protectedVertices[static_cast<std::size_t>(second)])
        {
            ++statistics->rejectedFeatureCount;
            continue;
        }
        if (commonNeighborCount(topology, first, second) != 2)
        {
            ++statistics->rejectedTopologyCount;
            continue;
        }
        const MeshVertex replacement = midpointVertex(
            mesh->vertices[static_cast<std::size_t>(first)],
            mesh->vertices[static_cast<std::size_t>(second)]);
        double improvement = 0.0;
        if (!collapseCandidateIsSafe(
                mesh,
                topology,
                first,
                second,
                replacement,
                options.minimumAffectedAspectRatio,
                normal_cosine,
                options.minimumWorstAspectImprovementRatio,
                &improvement))
        {
            ++statistics->rejectedQualityCount;
            continue;
        }
        std::vector<int> affected_vertices =
            topology.neighbors[static_cast<std::size_t>(first)];
        const auto &second_neighbors =
            topology.neighbors[static_cast<std::size_t>(second)];
        affected_vertices.insert(
            affected_vertices.end(),
            second_neighbors.cbegin(),
            second_neighbors.cend());
        affected_vertices.push_back(first);
        affected_vertices.push_back(second);
        std::sort(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.erase(
            std::unique(affected_vertices.begin(), affected_vertices.end()),
            affected_vertices.end());
        candidates.push_back(
            {first, second, replacement, std::move(affected_vertices), improvement});
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CollapseCandidate &left, const CollapseCandidate &right)
        {
            return left.improvement > right.improvement;
        });
    std::vector<std::uint8_t> used(mesh->vertices.size(), 0);
    std::vector<int> remap(mesh->vertices.size());
    std::iota(remap.begin(), remap.end(), 0);
    int accepted = 0;
    for (const CollapseCandidate &candidate : candidates)
    {
        const bool overlaps = std::any_of(
            candidate.affectedVertices.cbegin(),
            candidate.affectedVertices.cend(),
            [&used](int vertex)
            {
                return used[static_cast<std::size_t>(vertex)] != 0;
            });
        if (overlaps)
        {
            continue;
        }
        mesh->vertices[static_cast<std::size_t>(candidate.keep)] =
            candidate.replacement;
        remap[static_cast<std::size_t>(candidate.remove)] = candidate.keep;
        for (const int vertex : candidate.affectedVertices)
        {
            used[static_cast<std::size_t>(vertex)] = 1;
        }
        ++accepted;
    }
    if (accepted == 0)
    {
        return 0;
    }
    std::vector<Triangle> faces;
    faces.reserve(mesh->faces.size() - static_cast<std::size_t>(accepted * 2));
    for (Triangle face : mesh->faces)
    {
        for (int &vertex : face.v)
        {
            vertex = remap[static_cast<std::size_t>(vertex)];
        }
        if (face.v[0] != face.v[1] &&
            face.v[1] != face.v[2] &&
            face.v[2] != face.v[0])
        {
            faces.push_back(face);
        }
    }
    mesh->faces = std::move(faces);
    compactMesh(mesh);
    return accepted;
}

std::array<Triangle, 2> splitFace(const Triangle &face,
                                  int first,
                                  int second,
                                  int midpoint)
{
    int opposite = -1;
    bool forward = false;
    for (int corner = 0; corner < 3; ++corner)
    {
        if (face.v[corner] == first &&
            face.v[(corner + 1) % 3] == second)
        {
            opposite = face.v[(corner + 2) % 3];
            forward = true;
            break;
        }
        if (face.v[corner] == second &&
            face.v[(corner + 1) % 3] == first)
        {
            opposite = face.v[(corner + 2) % 3];
            break;
        }
    }
    if (forward)
    {
        std::array<Triangle, 2> children;
        children[0].v[0] = first;
        children[0].v[1] = midpoint;
        children[0].v[2] = opposite;
        children[1].v[0] = midpoint;
        children[1].v[1] = second;
        children[1].v[2] = opposite;
        return children;
    }
    std::array<Triangle, 2> children;
    children[0].v[0] = second;
    children[0].v[1] = midpoint;
    children[0].v[2] = opposite;
    children[1].v[0] = midpoint;
    children[1].v[1] = first;
    children[1].v[2] = opposite;
    return children;
}

int splitLongEdges(TriMesh *mesh,
                   const Topology &topology,
                   const MeshIsotropicRemeshOptions &options,
                   MeshIsotropicRemeshStatistics *statistics,
                   int maximum_added_faces)
{
    const double threshold =
        topology.medianInteriorEdgeLength * options.longEdgeRatio;
    std::vector<SplitCandidate> candidates;
    candidates.reserve(topology.edges.size() / 16);
    for (const auto &[key, edge] : topology.edges)
    {
        if (edge.faceCount != 2)
        {
            continue;
        }
        const int first = static_cast<int>(key >> 32U);
        const int second = static_cast<int>(key & 0xffffffffU);
        if (edgeLength(*mesh, first, second) <= threshold)
        {
            continue;
        }
        const Triangle &first_face =
            mesh->faces[static_cast<std::size_t>(edge.firstFace)];
        const Triangle &second_face =
            mesh->faces[static_cast<std::size_t>(edge.secondFace)];
        const double old_worst = std::max(
            triangleAspectRatio(*mesh, first_face),
            triangleAspectRatio(*mesh, second_face));
        if (old_worst <= options.minimumAffectedAspectRatio)
        {
            continue;
        }
        mesh->vertices.push_back(midpointVertex(
            mesh->vertices[static_cast<std::size_t>(first)],
            mesh->vertices[static_cast<std::size_t>(second)]));
        const int midpoint = static_cast<int>(mesh->vertices.size() - 1);
        const auto first_children =
            splitFace(first_face, first, second, midpoint);
        const auto second_children =
            splitFace(second_face, first, second, midpoint);
        double new_worst = 0.0;
        int old_affected_count = 0;
        int new_affected_count = 0;
        old_affected_count +=
            triangleAspectRatio(*mesh, first_face) >
                options.minimumAffectedAspectRatio ? 1 : 0;
        old_affected_count +=
            triangleAspectRatio(*mesh, second_face) >
                options.minimumAffectedAspectRatio ? 1 : 0;
        for (const Triangle &child :
             {first_children[0], first_children[1],
              second_children[0], second_children[1]})
        {
            const double aspect = triangleAspectRatio(*mesh, child);
            new_worst = std::max(new_worst, aspect);
            new_affected_count +=
                aspect > options.minimumAffectedAspectRatio ? 1 : 0;
        }
        mesh->vertices.pop_back();
        if (!std::isfinite(new_worst) ||
            new_worst >= old_worst *
                (1.0 - options.minimumWorstAspectImprovementRatio) ||
            new_affected_count > old_affected_count)
        {
            ++statistics->rejectedQualityCount;
            continue;
        }
        candidates.push_back({
            first,
            second,
            edge.firstFace,
            edge.secondFace,
            static_cast<double>(old_affected_count - new_affected_count) *
                    1000.0 +
                old_worst - new_worst});
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SplitCandidate &left, const SplitCandidate &right)
        {
            return left.improvement > right.improvement;
        });
    std::vector<std::uint8_t> used_faces(mesh->faces.size(), 0);
    std::vector<std::uint8_t> used_vertices(mesh->vertices.size(), 0);
    int accepted = 0;
    for (const SplitCandidate &candidate : candidates)
    {
        if (accepted * 2 >= maximum_added_faces ||
            used_faces[static_cast<std::size_t>(candidate.firstFace)] ||
            used_faces[static_cast<std::size_t>(candidate.secondFace)] ||
            used_vertices[static_cast<std::size_t>(candidate.first)] ||
            used_vertices[static_cast<std::size_t>(candidate.second)])
        {
            continue;
        }
        const int midpoint = static_cast<int>(mesh->vertices.size());
        mesh->vertices.push_back(midpointVertex(
            mesh->vertices[static_cast<std::size_t>(candidate.first)],
            mesh->vertices[static_cast<std::size_t>(candidate.second)]));
        const Triangle first_face =
            mesh->faces[static_cast<std::size_t>(candidate.firstFace)];
        const Triangle second_face =
            mesh->faces[static_cast<std::size_t>(candidate.secondFace)];
        const auto first_children = splitFace(
            first_face, candidate.first, candidate.second, midpoint);
        const auto second_children = splitFace(
            second_face, candidate.first, candidate.second, midpoint);
        mesh->faces[static_cast<std::size_t>(candidate.firstFace)] =
            first_children[0];
        mesh->faces[static_cast<std::size_t>(candidate.secondFace)] =
            second_children[0];
        mesh->faces.push_back(first_children[1]);
        mesh->faces.push_back(second_children[1]);
        used_faces[static_cast<std::size_t>(candidate.firstFace)] = 1;
        used_faces[static_cast<std::size_t>(candidate.secondFace)] = 1;
        used_vertices[static_cast<std::size_t>(candidate.first)] = 1;
        used_vertices[static_cast<std::size_t>(candidate.second)] = 1;
        ++accepted;
    }
    return accepted;
}

} // namespace

MeshIsotropicRemeshStatistics remeshInteriorHighAspectTriangles(
    TriMesh *mesh,
    const MeshIsotropicRemeshOptions &options)
{
    MeshIsotropicRemeshStatistics statistics;
    if (!mesh || mesh->empty())
    {
        return statistics;
    }
    constexpr double pi = 3.14159265358979323846;
    const double feature_cosine = std::cos(
        std::clamp(options.maximumFeatureAngleDegrees, 0.0, 180.0) *
        pi / 180.0);
    const double normal_cosine = std::cos(
        std::clamp(options.maximumNormalDeviationDegrees, 0.0, 180.0) *
        pi / 180.0);
    const int input_face_count = mesh->faceCount();
    const int maximum_added_faces = std::max(
        0,
        static_cast<int>(std::floor(
            input_face_count * std::max(0.0, options.maximumFaceGrowthRatio))));
    for (int pass = 0; pass < std::max(0, options.maximumPasses); ++pass)
    {
        if (options.isCancelled && options.isCancelled())
        {
            statistics.cancelled = true;
            break;
        }
        const Topology collapse_topology = buildTopology(*mesh, feature_cosine);
        if (collapse_topology.medianInteriorEdgeLength <= 1.0e-20)
        {
            break;
        }
        const int collapsed = collapseShortEdges(
            mesh,
            collapse_topology,
            options,
            normal_cosine,
            &statistics);
        const Topology split_topology = buildTopology(*mesh, feature_cosine);
        const int current_growth = mesh->faceCount() - input_face_count;
        const int split = splitLongEdges(
            mesh,
            split_topology,
            options,
            &statistics,
            std::max(0, maximum_added_faces - current_growth));
        statistics.collapsedEdgeCount += collapsed;
        statistics.splitEdgeCount += split;
        if (collapsed == 0 && split == 0)
        {
            break;
        }
        statistics.passCount = pass + 1;
    }
    return statistics;
}

} // namespace xjw::mesh
