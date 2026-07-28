#include "MeshQuadricSimplifier.h"

#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

namespace xjw::mesh
{
namespace
{

using Quadric = std::array<double, 10>;

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct EdgeRecord
{
    int first = -1;
    int second = -1;
    int faceCount = 0;
    Vec3 firstNormal;
    bool sharp = false;
};

struct CollapseCandidate
{
    int keep = -1;
    int remove = -1;
    Vec3 position;
    double cost = std::numeric_limits<double>::infinity();
};

struct FaceKey
{
    std::array<int, 3> vertices{};

    bool operator==(const FaceKey &other) const
    {
        return vertices == other.vertices;
    }
};

struct FaceKeyHash
{
    std::size_t operator()(const FaceKey &key) const
    {
        std::size_t seed = std::hash<int>{}(key.vertices[0]);
        seed ^= std::hash<int>{}(key.vertices[1]) + 0x9e3779b9u + (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<int>{}(key.vertices[2]) + 0x9e3779b9u + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
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
    return {left.y * right.z - left.z * right.y,
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

Vec3 normalized(const Vec3 &value)
{
    const double magnitude = length(value);
    if (magnitude <= 1.0e-18)
    {
        return {};
    }
    return {value.x / magnitude, value.y / magnitude, value.z / magnitude};
}

Vec3 faceNormal(const TriMesh &mesh, const Triangle &face)
{
    return normalized(cross(
        subtract(positionOf(mesh.vertices[static_cast<std::size_t>(face.v[1])]),
                 positionOf(mesh.vertices[static_cast<std::size_t>(face.v[0])])),
        subtract(positionOf(mesh.vertices[static_cast<std::size_t>(face.v[2])]),
                 positionOf(mesh.vertices[static_cast<std::size_t>(face.v[0])]))));
}

void addPlane(Quadric *quadric, const Vec3 &normal, double offset)
{
    const std::array<double, 4> plane{normal.x, normal.y, normal.z, offset};
    constexpr std::array<std::array<int, 2>, 10> entries{{
        {{0, 0}}, {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 1}},
        {{1, 2}}, {{1, 3}}, {{2, 2}}, {{2, 3}}, {{3, 3}}
    }};
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        (*quadric)[index] += plane[static_cast<std::size_t>(entries[index][0])]
                           * plane[static_cast<std::size_t>(entries[index][1])];
    }
}

Quadric addQuadrics(const Quadric &first, const Quadric &second)
{
    Quadric result{};
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        result[index] = first[index] + second[index];
    }
    return result;
}

double evaluate(const Quadric &quadric, const Vec3 &position)
{
    const double x = position.x;
    const double y = position.y;
    const double z = position.z;
    return quadric[0] * x * x + 2.0 * quadric[1] * x * y
        + 2.0 * quadric[2] * x * z + 2.0 * quadric[3] * x
        + quadric[4] * y * y + 2.0 * quadric[5] * y * z
        + 2.0 * quadric[6] * y + quadric[7] * z * z
        + 2.0 * quadric[8] * z + quadric[9];
}

bool solveOptimalPosition(const Quadric &quadric, Vec3 *position)
{
    const double a00 = quadric[0];
    const double a01 = quadric[1];
    const double a02 = quadric[2];
    const double a11 = quadric[4];
    const double a12 = quadric[5];
    const double a22 = quadric[7];
    const double b0 = -quadric[3];
    const double b1 = -quadric[6];
    const double b2 = -quadric[8];
    const double determinant = a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02);
    if (std::fabs(determinant) <= 1.0e-15)
    {
        return false;
    }
    position->x = (b0 * (a11 * a22 - a12 * a12)
                   - a01 * (b1 * a22 - a12 * b2)
                   + a02 * (b1 * a12 - a11 * b2)) / determinant;
    position->y = (a00 * (b1 * a22 - a12 * b2)
                   - b0 * (a01 * a22 - a12 * a02)
                   + a02 * (a01 * b2 - b1 * a02)) / determinant;
    position->z = (a00 * (a11 * b2 - b1 * a12)
                   - a01 * (a01 * b2 - b1 * a02)
                   + b0 * (a01 * a12 - a11 * a02)) / determinant;
    return std::isfinite(position->x) && std::isfinite(position->y)
        && std::isfinite(position->z);
}

Vec3 bestPosition(const Quadric &quadric, const Vec3 &first, const Vec3 &second)
{
    Vec3 optimal;
    if (solveOptimalPosition(quadric, &optimal))
    {
        return optimal;
    }
    const Vec3 midpoint{(first.x + second.x) * 0.5,
                        (first.y + second.y) * 0.5,
                        (first.z + second.z) * 0.5};
    std::array<Vec3, 3> candidates{first, second, midpoint};
    return *std::min_element(candidates.begin(), candidates.end(), [&](const Vec3 &left, const Vec3 &right)
    {
        return evaluate(quadric, left) < evaluate(quadric, right);
    });
}

bool collapsePreservesFaces(const TriMesh &mesh,
                            const std::vector<std::vector<int>> &vertexFaces,
                            int keep,
                            int remove,
                            const Vec3 &position,
                            double minimumNormalCosine,
                            double minimumFaceArea)
{
    std::vector<int> faces = vertexFaces[static_cast<std::size_t>(keep)];
    faces.insert(faces.end(),
                 vertexFaces[static_cast<std::size_t>(remove)].cbegin(),
                 vertexFaces[static_cast<std::size_t>(remove)].cend());
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    std::unordered_set<FaceKey, FaceKeyHash> remapped_faces;
    remapped_faces.reserve(faces.size());
    for (int faceIndex : faces)
    {
        const Triangle &face = mesh.faces[static_cast<std::size_t>(faceIndex)];
        std::array<int, 3> remapped{face.v[0], face.v[1], face.v[2]};
        for (int &index : remapped)
        {
            if (index == remove)
            {
                index = keep;
            }
        }
        if (remapped[0] == remapped[1] || remapped[1] == remapped[2]
            || remapped[2] == remapped[0])
        {
            continue;
        }
        FaceKey remapped_key{{remapped[0], remapped[1], remapped[2]}};
        std::sort(remapped_key.vertices.begin(), remapped_key.vertices.end());
        if (!remapped_faces.insert(remapped_key).second)
        {
            return false;
        }
        std::array<Vec3, 3> vertices;
        for (int corner = 0; corner < 3; ++corner)
        {
            vertices[static_cast<std::size_t>(corner)] = remapped[corner] == keep
                ? position
                : positionOf(mesh.vertices[static_cast<std::size_t>(remapped[corner])]);
        }
        const Vec3 newCross = cross(subtract(vertices[1], vertices[0]),
                                    subtract(vertices[2], vertices[0]));
        const double minimumDoubleArea =
            std::max(1.0e-18, 2.0 * minimumFaceArea);
        if (length(newCross) <= minimumDoubleArea)
        {
            return false;
        }
        if (dot(faceNormal(mesh, face), normalized(newCross)) < minimumNormalCosine)
        {
            return false;
        }
    }
    return true;
}

void applyCollapses(TriMesh *mesh, const std::vector<CollapseCandidate> &selected)
{
    std::vector<int> remap(mesh->vertices.size());
    for (std::size_t index = 0; index < remap.size(); ++index)
    {
        remap[index] = static_cast<int>(index);
    }
    for (const CollapseCandidate &candidate : selected)
    {
        MeshVertex &kept = mesh->vertices[static_cast<std::size_t>(candidate.keep)];
        const MeshVertex &removed = mesh->vertices[static_cast<std::size_t>(candidate.remove)];
        kept.x = static_cast<float>(candidate.position.x);
        kept.y = static_cast<float>(candidate.position.y);
        kept.z = static_cast<float>(candidate.position.z);
        const Vec3 normal = normalized({kept.nx + removed.nx,
                                        kept.ny + removed.ny,
                                        kept.nz + removed.nz});
        kept.nx = static_cast<float>(normal.x);
        kept.ny = static_cast<float>(normal.y);
        kept.nz = static_cast<float>(normal.z);
        kept.r = static_cast<std::uint8_t>((static_cast<int>(kept.r) + removed.r) / 2);
        kept.g = static_cast<std::uint8_t>((static_cast<int>(kept.g) + removed.g) / 2);
        kept.b = static_cast<std::uint8_t>((static_cast<int>(kept.b) + removed.b) / 2);
        remap[static_cast<std::size_t>(candidate.remove)] = candidate.keep;
    }

    std::vector<Triangle> faces;
    faces.reserve(mesh->faces.size());
    std::unordered_set<FaceKey, FaceKeyHash> uniqueFaces;
    uniqueFaces.reserve(mesh->faces.size());
    for (Triangle face : mesh->faces)
    {
        for (int &index : face.v)
        {
            index = remap[static_cast<std::size_t>(index)];
        }
        if (face.v[0] == face.v[1] || face.v[1] == face.v[2] || face.v[2] == face.v[0])
        {
            continue;
        }
        FaceKey key{{face.v[0], face.v[1], face.v[2]}};
        std::sort(key.vertices.begin(), key.vertices.end());
        if (uniqueFaces.insert(key).second)
        {
            faces.push_back(face);
        }
    }
    mesh->faces = std::move(faces);
    detail::compactReferencedVertices(mesh);
}

} // namespace

QuadricSimplifyStatistics simplifyMeshQuadric(
    TriMesh *mesh,
    const QuadricSimplifyOptions &options)
{
    QuadricSimplifyStatistics statistics;
    if (!mesh)
    {
        return statistics;
    }
    statistics.inputVertexCount = mesh->vertexCount();
    statistics.inputFaceCount = mesh->faceCount();
    const int target = std::max(4, options.targetFaceCount);
    if (mesh->empty() || options.targetFaceCount <= 0 || mesh->faceCount() <= target)
    {
        statistics.outputVertexCount = mesh->vertexCount();
        statistics.outputFaceCount = mesh->faceCount();
        statistics.reachedTarget = mesh->faceCount() <= target;
        return statistics;
    }

    constexpr double pi = 3.14159265358979323846;
    const double featureCosine = std::cos(options.featureAngleDegrees * pi / 180.0);
    const double normalCosine = std::cos(options.maximumNormalDeviationDegrees * pi / 180.0);
    const auto cancellationRequested = [&options]()
    {
        return options.isCancelled && options.isCancelled();
    };
    int stagnant_pass_count = 0;
    for (int pass = 0; pass < std::max(1, options.maximumPasses)
         && mesh->faceCount() > target; ++pass)
    {
        if (cancellationRequested())
        {
            statistics.cancelled = true;
            break;
        }
        const int input_face_count_this_pass = mesh->faceCount();
        std::vector<Quadric> quadrics(mesh->vertices.size());
        std::vector<std::vector<int>> vertexFaces(mesh->vertices.size());
        std::vector<std::vector<int>> neighbors(mesh->vertices.size());
        std::unordered_map<std::uint64_t, EdgeRecord> edges;
        edges.reserve(mesh->faces.size() * 2);
        bool pass_cancelled = false;
        for (std::size_t faceIndex = 0; faceIndex < mesh->faces.size(); ++faceIndex)
        {
            if ((faceIndex & 4095U) == 0U && cancellationRequested())
            {
                pass_cancelled = true;
                break;
            }
            const Triangle &face = mesh->faces[faceIndex];
            const Vec3 normal = faceNormal(*mesh, face);
            const Vec3 point = positionOf(mesh->vertices[static_cast<std::size_t>(face.v[0])]);
            const double offset = -dot(normal, point);
            for (int corner = 0; corner < 3; ++corner)
            {
                addPlane(&quadrics[static_cast<std::size_t>(face.v[corner])], normal, offset);
                vertexFaces[static_cast<std::size_t>(face.v[corner])].push_back(
                    static_cast<int>(faceIndex));
                const int next = face.v[(corner + 1) % 3];
                neighbors[static_cast<std::size_t>(face.v[corner])].push_back(next);
                neighbors[static_cast<std::size_t>(next)].push_back(face.v[corner]);
                const std::uint64_t key = edgeKey(face.v[corner], next);
                EdgeRecord &edge = edges[key];
                if (edge.faceCount == 0)
                {
                    edge.first = std::min(face.v[corner], next);
                    edge.second = std::max(face.v[corner], next);
                    edge.firstNormal = normal;
                }
                else if (dot(edge.firstNormal, normal) < featureCosine)
                {
                    edge.sharp = true;
                }
                ++edge.faceCount;
            }
        }
        if (pass_cancelled)
        {
            statistics.cancelled = true;
            break;
        }
        for (auto &list : neighbors)
        {
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        std::vector<std::uint8_t> boundary(mesh->vertices.size(), 0);
        std::vector<std::uint8_t> non_manifold(mesh->vertices.size(), 0);
        std::vector<int> boundary_degree(mesh->vertices.size(), 0);
        std::vector<int> sharp_edge_degree(mesh->vertices.size(), 0);
        for (const auto &item : edges)
        {
            if (item.second.faceCount == 1)
            {
                boundary[static_cast<std::size_t>(item.second.first)] = 1;
                boundary[static_cast<std::size_t>(item.second.second)] = 1;
                ++boundary_degree[static_cast<std::size_t>(item.second.first)];
                ++boundary_degree[static_cast<std::size_t>(item.second.second)];
            }
            if (item.second.sharp)
            {
                ++sharp_edge_degree[static_cast<std::size_t>(item.second.first)];
                ++sharp_edge_degree[static_cast<std::size_t>(item.second.second)];
            }
            if (item.second.faceCount > 2)
            {
                non_manifold[static_cast<std::size_t>(item.second.first)] = 1;
                non_manifold[static_cast<std::size_t>(item.second.second)] = 1;
            }
        }
        std::vector<std::uint8_t> boundary_protected = boundary;
        std::vector<std::uint8_t> topology_protected = non_manifold;
        for (std::size_t vertex_index = 0; vertex_index < boundary_degree.size(); ++vertex_index)
        {
            if (boundary_degree[vertex_index] != 0 && boundary_degree[vertex_index] != 2)
            {
                topology_protected[vertex_index] = 1;
            }
        }
        for (int protection_ring = 0; protection_ring < 2; ++protection_ring)
        {
            const std::vector<std::uint8_t> previous_protection = topology_protected;
            for (std::size_t vertex_index = 0; vertex_index < previous_protection.size();
                 ++vertex_index)
            {
                if (!previous_protection[vertex_index])
                {
                    continue;
                }
                for (const int neighbor : neighbors[vertex_index])
                {
                    topology_protected[static_cast<std::size_t>(neighbor)] = 1;
                }
            }
        }

        std::vector<EdgeRecord> edge_records;
        edge_records.reserve(edges.size());
        for (const auto &item : edges)
        {
            edge_records.push_back(item.second);
        }
        struct CandidateWorkerResult
        {
            std::vector<CollapseCandidate> candidates;
            int rejectedBoundaryEdgeCount = 0;
            int rejectedFeatureEdgeCount = 0;
            int rejectedTopologyEdgeCount = 0;
            int rejectedFlipEdgeCount = 0;
        };
        int worker_count = 1;
#ifdef MESHING_OPENMP
        worker_count = options.workerCount > 0
            ? std::clamp(options.workerCount, 1, omp_get_max_threads())
            : std::max(1, omp_get_max_threads());
#endif
        std::vector<CandidateWorkerResult> worker_results(
            static_cast<std::size_t>(worker_count));
        const std::size_t candidates_per_worker =
            edge_records.size() / static_cast<std::size_t>(worker_count) + 1;
        for (CandidateWorkerResult &worker_result : worker_results)
        {
            worker_result.candidates.reserve(candidates_per_worker / 3 + 1);
        }
        std::atomic<bool> candidate_cancelled{false};
        const auto evaluate_edge = [&](std::size_t edge_index, int worker_index)
        {
            if (candidate_cancelled.load(std::memory_order_relaxed))
            {
                return;
            }
            if ((edge_index & 8191U) == 0U && cancellationRequested())
            {
                candidate_cancelled.store(true, std::memory_order_relaxed);
                return;
            }
            CandidateWorkerResult &worker_result =
                worker_results[static_cast<std::size_t>(worker_index)];
            const EdgeRecord &edge = edge_records[edge_index];
            if (edge.faceCount > 2 ||
                topology_protected[static_cast<std::size_t>(edge.first)] ||
                topology_protected[static_cast<std::size_t>(edge.second)])
            {
                ++worker_result.rejectedTopologyEdgeCount;
                return;
            }
            const bool touches_boundary_neighborhood =
                boundary_protected[static_cast<std::size_t>(edge.first)] ||
                boundary_protected[static_cast<std::size_t>(edge.second)];
            const bool simple_boundary_edge = options.simplifySimpleOpenBoundaries &&
                edge.faceCount == 1 &&
                boundary_degree[static_cast<std::size_t>(edge.first)] == 2 &&
                boundary_degree[static_cast<std::size_t>(edge.second)] == 2;
            if (options.preserveOpenBoundaries && touches_boundary_neighborhood &&
                !simple_boundary_edge)
            {
                ++worker_result.rejectedBoundaryEdgeCount;
                return;
            }
            const int minimum_sharp_degree = std::max(
                1, options.minimumSharpEdgeEndpointDegree);
            const bool persistent_sharp_edge = edge.sharp &&
                (minimum_sharp_degree <= 1 ||
                 (sharp_edge_degree[static_cast<std::size_t>(edge.first)] >= minimum_sharp_degree &&
                  sharp_edge_degree[static_cast<std::size_t>(edge.second)] >= minimum_sharp_degree));
            if (persistent_sharp_edge)
            {
                ++worker_result.rejectedFeatureEdgeCount;
                return;
            }
            const auto &firstNeighbors = neighbors[static_cast<std::size_t>(edge.first)];
            const auto &secondNeighbors = neighbors[static_cast<std::size_t>(edge.second)];
            int commonNeighborCount = 0;
            std::array<int, 2> common_neighbors{{-1, -1}};
            std::size_t firstIndex = 0;
            std::size_t secondIndex = 0;
            while (firstIndex < firstNeighbors.size() && secondIndex < secondNeighbors.size())
            {
                if (firstNeighbors[firstIndex] == secondNeighbors[secondIndex])
                {
                    if (commonNeighborCount < static_cast<int>(common_neighbors.size()))
                    {
                        common_neighbors[static_cast<std::size_t>(commonNeighborCount)] =
                            firstNeighbors[firstIndex];
                    }
                    ++commonNeighborCount; ++firstIndex; ++secondIndex;
                }
                else if (firstNeighbors[firstIndex] < secondNeighbors[secondIndex])
                {
                    ++firstIndex;
                }
                else
                {
                    ++secondIndex;
                }
            }
            if (commonNeighborCount != edge.faceCount)
            {
                ++worker_result.rejectedTopologyEdgeCount;
                return;
            }
            if (edge.faceCount == 2 && common_neighbors[0] >= 0 &&
                common_neighbors[1] >= 0 &&
                edges.find(edgeKey(common_neighbors[0], common_neighbors[1])) != edges.cend())
            {
                ++worker_result.rejectedTopologyEdgeCount;
                return;
            }
            const Quadric quadric = addQuadrics(
                quadrics[static_cast<std::size_t>(edge.first)],
                quadrics[static_cast<std::size_t>(edge.second)]);
            const Vec3 position = bestPosition(
                quadric,
                positionOf(mesh->vertices[static_cast<std::size_t>(edge.first)]),
                positionOf(mesh->vertices[static_cast<std::size_t>(edge.second)]));
            if (!collapsePreservesFaces(*mesh, vertexFaces, edge.first, edge.second,
                                        position, normalCosine,
                                        std::max(
                                            0.0,
                                            static_cast<double>(
                                                options.minimumFaceArea))))
            {
                ++worker_result.rejectedFlipEdgeCount;
                return;
            }
            worker_result.candidates.push_back(
                {edge.first, edge.second, position, evaluate(quadric, position)});
        };
#ifdef MESHING_OPENMP
#pragma omp parallel num_threads(worker_count)
        {
            const int worker_index = omp_get_thread_num();
#pragma omp for schedule(dynamic, 512)
            for (int edge_index = 0;
                 edge_index < static_cast<int>(edge_records.size());
                 ++edge_index)
            {
                evaluate_edge(static_cast<std::size_t>(edge_index), worker_index);
            }
        }
#else
        for (std::size_t edge_index = 0; edge_index < edge_records.size(); ++edge_index)
        {
            evaluate_edge(edge_index, 0);
        }
#endif
        if (candidate_cancelled.load(std::memory_order_relaxed))
        {
            statistics.cancelled = true;
            break;
        }
        std::size_t candidate_count = 0;
        for (const CandidateWorkerResult &worker_result : worker_results)
        {
            candidate_count += worker_result.candidates.size();
            statistics.rejectedBoundaryEdgeCount +=
                worker_result.rejectedBoundaryEdgeCount;
            statistics.rejectedFeatureEdgeCount +=
                worker_result.rejectedFeatureEdgeCount;
            statistics.rejectedTopologyEdgeCount +=
                worker_result.rejectedTopologyEdgeCount;
            statistics.rejectedFlipEdgeCount +=
                worker_result.rejectedFlipEdgeCount;
        }
        std::vector<CollapseCandidate> candidates;
        candidates.reserve(candidate_count);
        for (CandidateWorkerResult &worker_result : worker_results)
        {
            candidates.insert(
                candidates.end(),
                std::make_move_iterator(worker_result.candidates.begin()),
                std::make_move_iterator(worker_result.candidates.end()));
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right)
        {
            if (left.cost != right.cost)
            {
                return left.cost < right.cost;
            }
            if (left.keep != right.keep)
            {
                return left.keep < right.keep;
            }
            return left.remove < right.remove;
        });

        const int desiredFaceReduction = mesh->faceCount() - target;
        std::vector<std::uint8_t> used(mesh->vertices.size(), 0);
        std::vector<CollapseCandidate> selected;
        selected.reserve(static_cast<std::size_t>(desiredFaceReduction / 2 + 1));
        int estimatedReduction = 0;
        std::size_t examined_candidate_count = 0;
        for (const CollapseCandidate &candidate : candidates)
        {
            if ((examined_candidate_count++ & 2047U) == 0U && cancellationRequested())
            {
                pass_cancelled = true;
                break;
            }
            bool overlaps_selected_neighborhood =
                used[static_cast<std::size_t>(candidate.keep)] ||
                used[static_cast<std::size_t>(candidate.remove)];
            for (const int neighbor : neighbors[static_cast<std::size_t>(candidate.keep)])
            {
                overlaps_selected_neighborhood = overlaps_selected_neighborhood ||
                    used[static_cast<std::size_t>(neighbor)];
            }
            for (const int neighbor : neighbors[static_cast<std::size_t>(candidate.remove)])
            {
                overlaps_selected_neighborhood = overlaps_selected_neighborhood ||
                    used[static_cast<std::size_t>(neighbor)];
            }
            if (overlaps_selected_neighborhood)
            {
                continue;
            }
            used[static_cast<std::size_t>(candidate.keep)] = 1;
            used[static_cast<std::size_t>(candidate.remove)] = 1;
            // The overlap test above already rejects any edge whose endpoints
            // share a one-ring face neighborhood with a selected endpoint.
            // Marking every neighbor as used as well expands the exclusion to
            // two rings, although those collapses cannot touch the same face.
            // Keeping only the endpoints therefore preserves independent
            // face updates while allowing substantially larger safe batches.
            selected.push_back(candidate);
            estimatedReduction += 2;
            if (estimatedReduction >= desiredFaceReduction)
            {
                break;
            }
        }
        if (pass_cancelled)
        {
            statistics.cancelled = true;
            break;
        }
        if (selected.empty())
        {
            break;
        }
        applyCollapses(mesh, selected);
        statistics.collapsedEdgeCount += static_cast<int>(selected.size());
        statistics.passCount = pass + 1;
        if (options.progress)
        {
            options.progress(statistics.passCount, mesh->faceCount());
        }
        const int face_reduction =
            input_face_count_this_pass - mesh->faceCount();
        const double reduction_ratio =
            static_cast<double>(std::max(0, face_reduction)) /
            std::max(1, input_face_count_this_pass);
        const int target_margin = std::max(16, target / 100);
        const bool remains_well_above_target =
            mesh->faceCount() > target + target_margin;
        if (remains_well_above_target &&
            options.minimumFaceReductionRatio > 0.0f &&
            reduction_ratio <
                static_cast<double>(options.minimumFaceReductionRatio))
        {
            ++stagnant_pass_count;
        }
        else
        {
            stagnant_pass_count = 0;
        }
        if (stagnant_pass_count >=
            std::max(1, options.maximumStagnantPasses))
        {
            statistics.stoppedByStagnation = true;
            break;
        }
    }

    detail::removeDegenerateFaces(
        mesh,
        std::max(0.0f, options.minimumFaceArea));
    detail::compactReferencedVertices(mesh);
    detail::recomputeNormals(mesh);
    statistics.outputVertexCount = mesh->vertexCount();
    statistics.outputFaceCount = mesh->faceCount();
    statistics.reachedTarget = mesh->faceCount() <= target;
    return statistics;
}

} // namespace xjw::mesh
