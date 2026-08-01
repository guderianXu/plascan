#include "VisibilityOccupancyCarrierFairer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
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

struct FaceReference
{
    Vec3 cross;
    double crossLength = 0.0;
};

struct MeshMetrics
{
    double area = 0.0;
    double absoluteVolume = 0.0;
};

struct LocalProjection
{
    bool valid = false;
    bool cancelled = false;
    std::uint64_t rejectedFaceCount = 0;
    std::uint64_t frozenVertexCount = 0;
};

using Edge = std::pair<int, int>;

Vec3 position(const MeshVertex &vertex)
{
    return {vertex.x, vertex.y, vertex.z};
}

Vec3 subtract(const Vec3 &first, const Vec3 &second)
{
    return {
        first.x - second.x,
        first.y - second.y,
        first.z - second.z};
}

Vec3 cross(const Vec3 &first, const Vec3 &second)
{
    return {
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x};
}

double dot(const Vec3 &first, const Vec3 &second)
{
    return first.x * second.x +
        first.y * second.y +
        first.z * second.z;
}

double length(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

Vec3 faceCross(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first = position(mesh.vertices[face.v[0]]);
    const Vec3 second = position(mesh.vertices[face.v[1]]);
    const Vec3 third = position(mesh.vertices[face.v[2]]);
    return cross(subtract(second, first), subtract(third, first));
}

bool isCancelled(const VisibilityOccupancyCarrierFairingOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

bool validOptions(
    const VisibilityOccupancyCarrierFairingOptions &options)
{
    return options.iterations >= 0 &&
        std::isfinite(options.lambda) && options.lambda > 0.0 &&
        std::isfinite(options.mu) && options.mu < 0.0 &&
        std::isfinite(options.absoluteMaximumDisplacement) &&
        options.absoluteMaximumDisplacement >= 0.0 &&
        std::isfinite(options.maximumDisplacementMeanEdgeRatio) &&
        options.maximumDisplacementMeanEdgeRatio >= 0.0 &&
        std::isfinite(options.minimumNormalDot) &&
        options.minimumNormalDot >= -1.0 &&
        options.minimumNormalDot <= 1.0 &&
        std::isfinite(options.minimumFaceAreaRatio) &&
        options.minimumFaceAreaRatio > 0.0 &&
        std::isfinite(options.minimumSurfaceAreaRatio) &&
        std::isfinite(options.maximumSurfaceAreaRatio) &&
        options.minimumSurfaceAreaRatio > 0.0 &&
        options.minimumSurfaceAreaRatio <= 1.0 &&
        options.maximumSurfaceAreaRatio >= 1.0 &&
        options.minimumSurfaceAreaRatio <=
            options.maximumSurfaceAreaRatio &&
        std::isfinite(options.minimumAbsoluteVolumeRatio) &&
        std::isfinite(options.maximumAbsoluteVolumeRatio) &&
        options.minimumAbsoluteVolumeRatio >= 0.0 &&
        options.minimumAbsoluteVolumeRatio <= 1.0 &&
        options.maximumAbsoluteVolumeRatio >= 1.0 &&
        options.minimumAbsoluteVolumeRatio <=
            options.maximumAbsoluteVolumeRatio;
}

MeshMetrics measureMesh(const TriMesh &mesh)
{
    MeshMetrics metrics;
    const Vec3 origin = position(mesh.vertices.front());
    double signed_volume_six = 0.0;
    for (const Triangle &face : mesh.faces)
    {
        const Vec3 first = subtract(
            position(mesh.vertices[face.v[0]]), origin);
        const Vec3 second = subtract(
            position(mesh.vertices[face.v[1]]), origin);
        const Vec3 third = subtract(
            position(mesh.vertices[face.v[2]]), origin);
        const Vec3 face_cross = cross(
            subtract(second, first), subtract(third, first));
        metrics.area += 0.5 * length(face_cross);
        signed_volume_six += dot(first, cross(second, third));
    }
    metrics.absoluteVolume = std::abs(signed_volume_six) / 6.0;
    return metrics;
}

bool globallySafe(
    const MeshMetrics &initial,
    const MeshMetrics &candidate,
    const VisibilityOccupancyCarrierFairingOptions &options,
    double *area_ratio,
    double *volume_ratio)
{
    *area_ratio = candidate.area / initial.area;
    *volume_ratio = candidate.absoluteVolume / initial.absoluteVolume;
    return std::isfinite(*area_ratio) &&
        std::isfinite(*volume_ratio) &&
        *area_ratio >= options.minimumSurfaceAreaRatio &&
        *area_ratio <= options.maximumSurfaceAreaRatio &&
        *volume_ratio >= options.minimumAbsoluteVolumeRatio &&
        *volume_ratio <= options.maximumAbsoluteVolumeRatio;
}

bool faceIsLocallySafe(
    const TriMesh &candidate,
    std::size_t face_index,
    const FaceReference &reference,
    const VisibilityOccupancyCarrierFairingOptions &options)
{
    const Vec3 candidate_cross = faceCross(
        candidate, candidate.faces[face_index]);
    const double candidate_length = length(candidate_cross);
    if (!std::isfinite(candidate_length) ||
        candidate_length <
            reference.crossLength * options.minimumFaceAreaRatio)
    {
        return false;
    }
    const double normal_dot = dot(reference.cross, candidate_cross) /
        (reference.crossLength * candidate_length);
    return std::isfinite(normal_dot) &&
        normal_dot >= options.minimumNormalDot;
}

bool positionChanged(
    const MeshVertex &first,
    const MeshVertex &second)
{
    return first.x != second.x ||
        first.y != second.y ||
        first.z != second.z;
}

LocalProjection projectToLocalSafety(
    const TriMesh &current,
    const std::vector<FaceReference> &references,
    const VisibilityOccupancyCarrierFairingOptions &options,
    TriMesh *candidate)
{
    LocalProjection projection;
    std::vector<std::uint8_t> rejected_faces(candidate->faces.size(), 0);
    std::vector<std::uint8_t> frozen_vertices(candidate->vertices.size(), 0);
    while (true)
    {
        std::vector<std::uint8_t> to_freeze(candidate->vertices.size(), 0);
        bool found_violation = false;
        for (std::size_t face_index = 0;
             face_index < candidate->faces.size();
             ++face_index)
        {
            if (isCancelled(options))
            {
                projection.cancelled = true;
                return projection;
            }
            if (faceIsLocallySafe(
                    *candidate, face_index, references[face_index], options))
            {
                continue;
            }
            found_violation = true;
            rejected_faces[face_index] = 1;
            for (const int vertex_index : candidate->faces[face_index].v)
            {
                if (positionChanged(
                        candidate->vertices[vertex_index],
                        current.vertices[vertex_index]))
                {
                    to_freeze[vertex_index] = 1;
                }
            }
        }
        if (!found_violation)
        {
            projection.valid = true;
            break;
        }

        bool froze_vertex = false;
        for (std::size_t index = 0; index < to_freeze.size(); ++index)
        {
            if (to_freeze[index] == 0 || frozen_vertices[index] != 0)
            {
                continue;
            }
            candidate->vertices[index].x = current.vertices[index].x;
            candidate->vertices[index].y = current.vertices[index].y;
            candidate->vertices[index].z = current.vertices[index].z;
            frozen_vertices[index] = 1;
            froze_vertex = true;
        }
        if (!froze_vertex)
        {
            break;
        }
    }
    projection.rejectedFaceCount = static_cast<std::uint64_t>(
        std::count(rejected_faces.begin(), rejected_faces.end(), 1));
    projection.frozenVertexCount = static_cast<std::uint64_t>(
        std::count(frozen_vertices.begin(), frozen_vertices.end(), 1));
    return projection;
}

VisibilityOccupancyCarrierFairingResult atomicExit(
    VisibilityOccupancyCarrierFairingResult result,
    const TriMesh &original,
    const MeshMetrics &initial,
    bool cancelled,
    std::string message)
{
    result.ok = false;
    result.cancelled = cancelled;
    result.rolledBack = true;
    result.errorMessage = std::move(message);
    result.mesh = original;
    ++result.statistics.rollbackCount;
    result.statistics.finalSurfaceArea = initial.area;
    result.statistics.finalAbsoluteVolume = initial.absoluteVolume;
    result.statistics.finalSurfaceAreaRatio = 1.0;
    result.statistics.finalAbsoluteVolumeRatio = 1.0;
    result.statistics.maximumAppliedDisplacement = 0.0;
    return result;
}

} // namespace

VisibilityOccupancyCarrierFairingResult
VisibilityOccupancyCarrierFairer::fair(
    const TriMesh &mesh,
    const VisibilityOccupancyCarrierFairingOptions &options)
{
    VisibilityOccupancyCarrierFairingResult result;
    result.mesh = mesh;
    result.statistics.inputVertexCount = mesh.vertices.size();
    result.statistics.inputFaceCount = mesh.faces.size();
    result.statistics.requestedIterationCount = options.iterations;
    if (!validOptions(options))
    {
        result.errorMessage = "invalid carrier fairing options";
        return result;
    }
    if (mesh.empty())
    {
        result.errorMessage = "carrier mesh must contain vertices and faces";
        return result;
    }
    if (isCancelled(options))
    {
        result.cancelled = true;
        result.rolledBack = true;
        result.errorMessage = "visibility occupancy carrier fairing cancelled";
        ++result.statistics.rollbackCount;
        return result;
    }

    std::vector<Edge> edges;
    edges.reserve(mesh.faces.size() * 3U);
    std::vector<FaceReference> face_references;
    face_references.reserve(mesh.faces.size());
    for (const MeshVertex &vertex : mesh.vertices)
    {
        if (!std::isfinite(vertex.x) ||
            !std::isfinite(vertex.y) ||
            !std::isfinite(vertex.z))
        {
            result.errorMessage = "carrier mesh contains a non-finite vertex";
            return result;
        }
    }
    for (const Triangle &face : mesh.faces)
    {
        if (isCancelled(options))
        {
            result.cancelled = true;
            result.rolledBack = true;
            result.errorMessage =
                "visibility occupancy carrier fairing cancelled";
            ++result.statistics.rollbackCount;
            return result;
        }
        for (const int vertex_index : face.v)
        {
            if (vertex_index < 0 ||
                static_cast<std::size_t>(vertex_index) >=
                    mesh.vertices.size())
            {
                result.errorMessage = "carrier mesh has an invalid face index";
                return result;
            }
        }
        if (face.v[0] == face.v[1] ||
            face.v[1] == face.v[2] ||
            face.v[2] == face.v[0])
        {
            result.errorMessage = "carrier mesh has a repeated face index";
            return result;
        }
        const Vec3 face_cross = faceCross(mesh, face);
        const double cross_length = length(face_cross);
        if (!(cross_length > std::numeric_limits<double>::epsilon()) ||
            !std::isfinite(cross_length))
        {
            result.errorMessage = "carrier mesh has a degenerate face";
            return result;
        }
        face_references.push_back({face_cross, cross_length});
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            edges.emplace_back(
                std::min(first, second), std::max(first, second));
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    result.statistics.uniqueEdgeCount = edges.size();

    std::vector<std::vector<int>> neighbours(mesh.vertices.size());
    double edge_length_sum = 0.0;
    for (const Edge &edge : edges)
    {
        neighbours[edge.first].push_back(edge.second);
        neighbours[edge.second].push_back(edge.first);
        edge_length_sum += length(subtract(
            position(mesh.vertices[edge.first]),
            position(mesh.vertices[edge.second])));
    }
    for (auto &vertex_neighbours : neighbours)
    {
        std::sort(vertex_neighbours.begin(), vertex_neighbours.end());
    }
    result.statistics.meanEdgeLength =
        edge_length_sum / static_cast<double>(edges.size());
    result.statistics.resolvedMaximumDisplacement =
        options.absoluteMaximumDisplacement > 0.0
        ? options.absoluteMaximumDisplacement
        : result.statistics.meanEdgeLength *
            options.maximumDisplacementMeanEdgeRatio;
    if (!(result.statistics.resolvedMaximumDisplacement > 0.0) ||
        !std::isfinite(result.statistics.resolvedMaximumDisplacement))
    {
        result.errorMessage = "carrier fairing displacement limit is invalid";
        return result;
    }

    const MeshMetrics initial = measureMesh(mesh);
    result.statistics.initialSurfaceArea = initial.area;
    result.statistics.finalSurfaceArea = initial.area;
    result.statistics.initialAbsoluteVolume = initial.absoluteVolume;
    result.statistics.finalAbsoluteVolume = initial.absoluteVolume;
    if (!(initial.area > 0.0) ||
        !(initial.absoluteVolume > 1.0e-15) ||
        !std::isfinite(initial.area) ||
        !std::isfinite(initial.absoluteVolume))
    {
        result.errorMessage =
            "carrier mesh must enclose a finite non-zero volume";
        return result;
    }

    TriMesh current = mesh;
    const std::array<double, 2> coefficients{options.lambda, options.mu};
    for (int iteration = 0; iteration < options.iterations; ++iteration)
    {
        for (const double coefficient : coefficients)
        {
            if (isCancelled(options))
            {
                return atomicExit(
                    std::move(result), mesh, initial, true,
                    "visibility occupancy carrier fairing cancelled");
            }
            ++result.statistics.attemptedHalfStepCount;
            TriMesh candidate = current;
            for (std::size_t index = 0;
                 index < current.vertices.size();
                 ++index)
            {
                if (isCancelled(options))
                {
                    return atomicExit(
                        std::move(result), mesh, initial, true,
                        "visibility occupancy carrier fairing cancelled");
                }
                const Vec3 current_position = position(current.vertices[index]);
                Vec3 weighted_sum;
                double weight_sum = 0.0;
                for (const int neighbour_index : neighbours[index])
                {
                    const Vec3 neighbour =
                        position(current.vertices[neighbour_index]);
                    const double distance =
                        length(subtract(neighbour, current_position));
                    if (!(distance > 1.0e-15) || !std::isfinite(distance))
                    {
                        continue;
                    }
                    const double weight = 1.0 / distance;
                    weighted_sum.x += weight * neighbour.x;
                    weighted_sum.y += weight * neighbour.y;
                    weighted_sum.z += weight * neighbour.z;
                    weight_sum += weight;
                }
                if (!(weight_sum > 0.0))
                {
                    continue;
                }
                Vec3 proposed{
                    current_position.x + coefficient *
                        (weighted_sum.x / weight_sum - current_position.x),
                    current_position.y + coefficient *
                        (weighted_sum.y / weight_sum - current_position.y),
                    current_position.z + coefficient *
                        (weighted_sum.z / weight_sum - current_position.z)};
                const Vec3 original_position = position(mesh.vertices[index]);
                const Vec3 original_delta =
                    subtract(proposed, original_position);
                const double displacement = length(original_delta);
                const double limit =
                    result.statistics.resolvedMaximumDisplacement;
                if (displacement > limit)
                {
                    const double scale = limit / displacement;
                    proposed = {
                        original_position.x + scale * original_delta.x,
                        original_position.y + scale * original_delta.y,
                        original_position.z + scale * original_delta.z};
                    ++result.statistics.displacementClampedVertexCount;
                }
                candidate.vertices[index].x =
                    static_cast<float>(proposed.x);
                candidate.vertices[index].y =
                    static_cast<float>(proposed.y);
                candidate.vertices[index].z =
                    static_cast<float>(proposed.z);
            }

            const LocalProjection projection = projectToLocalSafety(
                current, face_references, options, &candidate);
            if (projection.cancelled)
            {
                return atomicExit(
                    std::move(result), mesh, initial, true,
                    "visibility occupancy carrier fairing cancelled");
            }
            result.statistics.locallyRejectedFaceCount +=
                projection.rejectedFaceCount;
            result.statistics.locallyFrozenVertexCount +=
                projection.frozenVertexCount;
            if (!projection.valid)
            {
                return atomicExit(
                    std::move(result), mesh, initial, false,
                    "carrier fairing local face guard could not recover");
            }

            current = std::move(candidate);
            ++result.statistics.acceptedHalfStepCount;
            for (std::size_t index = 0;
                 index < current.vertices.size();
                 ++index)
            {
                result.statistics.maximumAppliedDisplacement = std::max(
                    result.statistics.maximumAppliedDisplacement,
                    length(subtract(
                        position(current.vertices[index]),
                        position(mesh.vertices[index]))));
            }
        }
        // Taubin fairing deliberately contracts during lambda and expands
        // during mu. Judge its global conservation after the complete pair,
        // while displacement and local face guards remain per half-step.
        const MeshMetrics iteration_metrics = measureMesh(current);
        double area_ratio = 0.0;
        double volume_ratio = 0.0;
        if (!globallySafe(
                initial,
                iteration_metrics,
                options,
                &area_ratio,
                &volume_ratio))
        {
            return atomicExit(
                std::move(result), mesh, initial, false,
                "carrier fairing global area or volume guard failed");
        }
        ++result.statistics.completedIterationCount;
    }

    const MeshMetrics final_metrics = measureMesh(current);
    result.statistics.finalSurfaceArea = final_metrics.area;
    result.statistics.finalAbsoluteVolume = final_metrics.absoluteVolume;
    result.statistics.finalSurfaceAreaRatio =
        final_metrics.area / initial.area;
    result.statistics.finalAbsoluteVolumeRatio =
        final_metrics.absoluteVolume / initial.absoluteVolume;
    result.mesh = std::move(current);
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
