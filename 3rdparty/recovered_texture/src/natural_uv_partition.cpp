#include "metashape_texture/natural_uv_partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace metashape_texture {
namespace {

using Vec3 = std::array<double, 3>;

struct FaceGeometry {
    Vec3 area_vector{};
    Vec3 normal{};
    std::array<std::int32_t, 3> neighbors{-1, -1, -1};
};

struct FaceTopology {
    std::array<std::uint32_t, 3> vertices{};
    std::array<std::int32_t, 3> neighbors{-1, -1, -1};
    std::array<std::uint32_t, 3> neighbor_corners{};
};

struct VertexRingEntry {
    std::uint32_t face{};
    std::uint32_t corner{};
    std::uint32_t next_face{};
    std::uint32_t next_corner{};
};

Vec3 scalar_float_cross(const float *first, const float *second, const float *third) {
    const float ax = second[0] - first[0];
    const float ay = second[1] - first[1];
    const float az = second[2] - first[2];
    const float bx = third[0] - first[0];
    const float by = third[1] - first[1];
    const float bz = third[2] - first[2];
    const float ay_bz = ay * bz;
    const float az_by = az * by;
    const float az_bx = az * bx;
    const float ax_bz = ax * bz;
    const float ax_by = ax * by;
    const float ay_bx = ay * bx;
    return {
        static_cast<double>(ay_bz - az_by),
        static_cast<double>(az_bx - ax_bz),
        static_cast<double>(ax_by - ay_bx),
    };
}

double squared_length(const Vec3 &value) {
    return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
}

std::vector<FaceGeometry> build_geometry(
    std::span<const float> vertices,
    std::span<const std::uint32_t> indices,
    std::span<const std::uint32_t> ordered_faces) {
    if (vertices.size() % 3U != 0 || indices.size() % 3U != 0) {
        throw std::invalid_argument("natural UV geometry arrays are not tightly packed triples");
    }
    const std::size_t vertex_count = vertices.size() / 3U;
    const std::size_t face_count = indices.size() / 3U;
    std::vector<FaceGeometry> result(ordered_faces.size());
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Edge, std::vector<std::pair<std::uint32_t, std::uint32_t>>> edge_uses;
    for (std::uint32_t local_face = 0; local_face < ordered_faces.size(); ++local_face) {
        const std::uint32_t global_face = ordered_faces[local_face];
        if (global_face >= face_count) throw std::out_of_range("natural UV face index");
        const auto *triangle = indices.data() + static_cast<std::size_t>(global_face) * 3U;
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            if (triangle[corner] >= vertex_count) throw std::out_of_range("natural UV vertex index");
            const std::uint32_t next = (corner + 1U) % 3U;
            edge_uses[std::minmax(triangle[corner], triangle[next])].emplace_back(
                local_face, corner);
        }
        Vec3 cross = scalar_float_cross(
            vertices.data() + static_cast<std::size_t>(triangle[0]) * 3U,
            vertices.data() + static_cast<std::size_t>(triangle[1]) * 3U,
            vertices.data() + static_cast<std::size_t>(triangle[2]) * 3U);
        const double length = std::sqrt(squared_length(cross));
        if (length == 0.0) throw std::runtime_error("natural UV chart contains a degenerate face");
        for (std::uint32_t axis = 0; axis < 3U; ++axis) {
            result[local_face].area_vector[axis] = cross[axis] * 0.5;
            result[local_face].normal[axis] = cross[axis] / length;
        }
    }
    for (const auto &[_, uses] : edge_uses) {
        if (uses.size() != 2U) continue;
        const auto [first_face, first_corner] = uses[0];
        const auto [second_face, second_corner] = uses[1];
        result[first_face].neighbors[first_corner] = static_cast<std::int32_t>(second_face);
        result[second_face].neighbors[second_corner] = static_cast<std::int32_t>(first_face);
    }
    return result;
}

std::vector<Vec3> update_prototypes(const std::vector<std::int32_t> &assignments,
                                    const std::vector<FaceGeometry> &faces,
                                    std::uint32_t cluster_count) {
    std::vector<Vec3> prototypes(cluster_count, {0.0, 0.0, 0.0});
    for (std::size_t face = 0; face < faces.size(); ++face) {
        const auto cluster = assignments[face];
        if (cluster < 0) continue;
        for (std::uint32_t axis = 0; axis < 3U; ++axis) {
            prototypes[static_cast<std::size_t>(cluster)][axis] += faces[face].area_vector[axis];
        }
    }
    for (Vec3 &prototype : prototypes) {
        const double length = std::sqrt(squared_length(prototype));
        if (length == 0.0) throw std::runtime_error("natural UV normal cluster is empty");
        for (double &value : prototype) value /= length;
    }
    return prototypes;
}

double face_cost(std::size_t face, std::int32_t cluster,
                 const std::vector<FaceGeometry> &faces,
                 const std::vector<Vec3> &prototypes) {
    Vec3 delta{};
    for (std::uint32_t axis = 0; axis < 3U; ++axis) {
        delta[axis] = faces[face].normal[axis] - prototypes[static_cast<std::size_t>(cluster)][axis];
    }
    return std::sqrt(squared_length(faces[face].area_vector)) * squared_length(delta);
}

double partition_objective(const std::vector<std::int32_t> &assignments,
                           const std::vector<FaceGeometry> &faces,
                           const std::vector<Vec3> &prototypes) {
    double area = 0.0;
    double error = 0.0;
    for (std::size_t face = 0; face < faces.size(); ++face) {
        area += std::sqrt(squared_length(faces[face].area_vector));
        error += face_cost(face, assignments[face], faces, prototypes);
    }
    return error / area;
}

std::int32_t select_new_seed(const std::vector<std::int32_t> &assignments,
                             const std::vector<FaceGeometry> &faces,
                             const std::vector<Vec3> &prototypes) {
    std::vector<std::uint32_t> counts(prototypes.size(), 0U);
    std::vector<double> errors(prototypes.size(), 0.0);
    for (std::size_t face = 0; face < faces.size(); ++face) {
        const auto cluster = assignments[face];
        ++counts[static_cast<std::size_t>(cluster)];
        errors[static_cast<std::size_t>(cluster)] += face_cost(face, cluster, faces, prototypes);
    }
    std::int32_t selected_cluster = -1;
    double selected_error = -1.0;
    for (std::uint32_t cluster = 0; cluster < prototypes.size(); ++cluster) {
        if (counts[cluster] > 1U && errors[cluster] > selected_error) {
            selected_cluster = static_cast<std::int32_t>(cluster);
            selected_error = errors[cluster];
        }
    }
    if (selected_cluster < 0) return -1;
    std::int32_t selected_face = -1;
    double selected_cost = -1.0;
    for (std::size_t face = 0; face < faces.size(); ++face) {
        if (assignments[face] != selected_cluster) continue;
        const double cost = face_cost(face, selected_cluster, faces, prototypes);
        if (cost > selected_cost) {
            selected_face = static_cast<std::int32_t>(face);
            selected_cost = cost;
        }
    }
    return selected_face;
}

std::vector<std::uint32_t> select_representatives(
    const std::vector<std::int32_t> &assignments,
    const std::vector<FaceGeometry> &faces,
    const std::vector<Vec3> &prototypes) {
    std::vector<std::uint32_t> seeds(prototypes.size());
    for (std::uint32_t cluster = 0; cluster < prototypes.size(); ++cluster) {
        double selected_cost = std::numeric_limits<double>::infinity();
        std::int32_t selected_face = -1;
        for (std::size_t face = 0; face < faces.size(); ++face) {
            if (assignments[face] != static_cast<std::int32_t>(cluster)) continue;
            const double cost = face_cost(face, static_cast<std::int32_t>(cluster),
                                          faces, prototypes);
            if (cost < selected_cost) {
                selected_cost = cost;
                selected_face = static_cast<std::int32_t>(face);
            }
        }
        if (selected_face < 0) throw std::runtime_error("natural UV cluster has no seed");
        seeds[cluster] = static_cast<std::uint32_t>(selected_face);
    }
    return seeds;
}

struct Frontier {
    double cost{};
    std::uint64_t insertion{};
    std::uint32_t face{};
    std::int32_t cluster{};
};

struct FrontierGreater {
    bool operator()(const Frontier &first, const Frontier &second) const {
        if (first.cost != second.cost) return first.cost > second.cost;
        return first.insertion > second.insertion;
    }
};

std::vector<std::int32_t> grow_from_seeds(
    const std::vector<std::uint32_t> &seeds,
    const std::vector<FaceGeometry> &faces,
    const std::vector<Vec3> &prototypes) {
    std::vector<std::int32_t> assignments(faces.size(), -1);
    for (std::uint32_t cluster = 0; cluster < seeds.size(); ++cluster) {
        assignments[seeds[cluster]] = static_cast<std::int32_t>(cluster);
    }
    std::priority_queue<Frontier, std::vector<Frontier>, FrontierGreater> frontier;
    std::uint64_t insertion = 0;
    auto push_neighbors = [&](std::uint32_t face, std::int32_t cluster) {
        for (const std::int32_t neighbor : faces[face].neighbors) {
            if (neighbor < 0 || assignments[static_cast<std::size_t>(neighbor)] >= 0) continue;
            frontier.push({face_cost(static_cast<std::size_t>(neighbor), cluster,
                                     faces, prototypes),
                           insertion++, static_cast<std::uint32_t>(neighbor), cluster});
        }
    };
    for (std::uint32_t cluster = 0; cluster < seeds.size(); ++cluster) {
        push_neighbors(seeds[cluster], static_cast<std::int32_t>(cluster));
    }
    while (!frontier.empty()) {
        const Frontier candidate = frontier.top();
        frontier.pop();
        if (assignments[candidate.face] >= 0) continue;
        assignments[candidate.face] = candidate.cluster;
        push_neighbors(candidate.face, candidate.cluster);
    }
    if (std::ranges::find(assignments, -1) != assignments.end()) {
        throw std::runtime_error("natural UV topology grow did not cover the chart");
    }
    return assignments;
}

std::vector<FaceTopology> build_chart_topology(
    std::span<const std::uint32_t> triangle_indices,
    std::size_t vertex_count) {
    std::vector<FaceTopology> result(triangle_indices.size() / 3U);
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Edge, std::vector<std::pair<std::uint32_t, std::uint32_t>>> edge_uses;
    for (std::uint32_t face = 0; face < result.size(); ++face) {
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t vertex = triangle_indices[3U * face + corner];
            const std::uint32_t next = triangle_indices[3U * face + (corner + 1U) % 3U];
            if (vertex >= vertex_count || next >= vertex_count) {
                throw std::out_of_range("natural UV postprocess vertex index");
            }
            result[face].vertices[corner] = vertex;
            edge_uses[std::minmax(vertex, next)].emplace_back(face, corner);
        }
    }
    for (const auto &[_, uses] : edge_uses) {
        if (uses.size() != 2U) continue;
        const auto [first_face, first_corner] = uses[0];
        const auto [second_face, second_corner] = uses[1];
        result[first_face].neighbors[first_corner] =
            static_cast<std::int32_t>(second_face);
        result[first_face].neighbor_corners[first_corner] = second_corner;
        result[second_face].neighbors[second_corner] =
            static_cast<std::int32_t>(first_face);
        result[second_face].neighbor_corners[second_corner] = first_corner;
    }
    return result;
}

std::optional<std::vector<VertexRingEntry>> vertex_ring(
    const std::vector<FaceTopology> &faces,
    std::uint32_t start_face,
    std::uint32_t start_corner) {
    std::vector<VertexRingEntry> ring;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> visited;
    std::uint32_t face = start_face;
    std::uint32_t corner = start_corner;
    while (std::ranges::find(visited, std::pair{face, corner}) == visited.end()) {
        visited.emplace_back(face, corner);
        const std::uint32_t edge = (corner + 2U) % 3U;
        const std::int32_t next_face = faces[face].neighbors[edge];
        if (next_face < 0) return std::nullopt;
        const std::uint32_t next_corner = faces[face].neighbor_corners[edge];
        ring.push_back({face, corner, static_cast<std::uint32_t>(next_face), next_corner});
        face = static_cast<std::uint32_t>(next_face);
        corner = next_corner;
    }
    if (face != start_face || corner != start_corner) return std::nullopt;
    return ring;
}

float uv_edge_length(std::span<const float> vertex_uv,
                     std::uint32_t first,
                     std::uint32_t second) {
    const float delta_u = vertex_uv[2U * first] - vertex_uv[2U * second];
    const float delta_v = vertex_uv[2U * first + 1U] - vertex_uv[2U * second + 1U];
    const float squared = delta_u * delta_u + delta_v * delta_v;
    return std::sqrt(squared);
}

std::vector<bool> partition_interior_vertices(
    const std::vector<FaceTopology> &faces,
    const std::vector<std::int32_t> &assignments,
    const std::vector<std::pair<std::int32_t, std::int32_t>> &references) {
    std::vector<bool> interior(references.size(), true);
    for (std::uint32_t vertex = 0; vertex < references.size(); ++vertex) {
        const auto [start_face, start_corner] = references[vertex];
        if (start_face < 0 || assignments[static_cast<std::size_t>(start_face)] < 0) continue;
        const auto ring = vertex_ring(
            faces, static_cast<std::uint32_t>(start_face),
            static_cast<std::uint32_t>(start_corner));
        if (!ring) {
            interior[vertex] = false;
            continue;
        }
        const std::int32_t label = assignments[static_cast<std::size_t>(start_face)];
        for (const VertexRingEntry &entry : *ring) {
            if (assignments[entry.next_face] != label) {
                interior[vertex] = false;
                break;
            }
        }
    }
    return interior;
}

std::vector<NaturalUvPartitionCandidate> partition_candidates(
    const std::vector<FaceTopology> &faces,
    std::span<const float> vertex_uv,
    const std::vector<std::int32_t> &assignments,
    const std::vector<std::pair<std::int32_t, std::int32_t>> &references,
    const std::vector<bool> &interior) {
    std::vector<NaturalUvPartitionCandidate> result;
    for (std::uint32_t vertex = 0; vertex < references.size(); ++vertex) {
        const auto [start_face, start_corner] = references[vertex];
        if (interior[vertex] || start_face < 0) continue;
        const auto ring = vertex_ring(
            faces, static_cast<std::uint32_t>(start_face),
            static_cast<std::uint32_t>(start_corner));
        if (!ring) continue;
        std::array<std::int32_t, 2> labels{-1, -1};
        std::array<double, 2> paths{};
        std::size_t label_count = 0U;
        std::uint32_t transitions = 0U;
        double boundary = 0.0;
        bool invalid = false;
        for (const VertexRingEntry &entry : *ring) {
            const std::int32_t label = assignments[entry.face];
            const std::int32_t next_label = assignments[entry.next_face];
            auto label_iterator = std::find(
                labels.begin(), labels.begin() + static_cast<std::ptrdiff_t>(label_count), label);
            if (label_iterator == labels.begin() + static_cast<std::ptrdiff_t>(label_count)) {
                if (label_count == labels.size()) {
                    invalid = true;
                    break;
                }
                labels[label_count++] = label;
                label_iterator = labels.begin() + static_cast<std::ptrdiff_t>(label_count - 1U);
            }
            const std::size_t label_index = static_cast<std::size_t>(
                label_iterator - labels.begin());
            const std::uint32_t current_outer =
                faces[entry.face].vertices[(entry.corner + 1U) % 3U];
            const std::uint32_t next_outer =
                faces[entry.next_face].vertices[(entry.next_corner + 1U) % 3U];
            paths[label_index] += uv_edge_length(vertex_uv, current_outer, next_outer);
            if (label != next_label) {
                ++transitions;
                boundary += uv_edge_length(vertex_uv, vertex, next_outer);
            }
        }
        if (invalid || label_count != 2U || transitions != 2U) continue;
        const double gain_to_first = boundary - paths[1];
        const double gain_to_second = boundary - paths[0];
        const bool choose_first = gain_to_first > gain_to_second;
        const double gain = choose_first ? gain_to_first : gain_to_second;
        if (!(gain > 0.0)) continue;
        result.push_back({vertex, labels[choose_first ? 0U : 1U], gain});
    }
    std::stable_sort(result.begin(), result.end(),
                     [](const NaturalUvPartitionCandidate &first,
                        const NaturalUvPartitionCandidate &second) {
                         return first.gain < second.gain;
                     });
    return result;
}

}  // namespace

NaturalUvNormalPartition partition_natural_uv_chart_by_normals(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const std::uint32_t> ordered_faces,
    std::uint32_t max_clusters,
    std::uint32_t refinement_iterations,
    double target_ratio) {
    if (ordered_faces.empty()) throw std::invalid_argument("cannot partition an empty chart");
    if (max_clusters == 0U || refinement_iterations == 0U || target_ratio < 0.0) {
        throw std::invalid_argument("invalid natural UV partition options");
    }
    const auto faces = build_geometry(vertex_xyz, triangle_indices, ordered_faces);
    // A tiny closed component cannot populate more normal clusters than it has
    // faces.  The production Shoe charts never hit this boundary, but clamping
    // it keeps the recovered worker well-defined for small closed meshes.
    max_clusters = std::min(
        max_clusters, static_cast<std::uint32_t>(faces.size()));
    NaturalUvNormalPartition result;
    result.assignments.assign(faces.size(), 0);
    result.prototypes = update_prototypes(result.assignments, faces, 1U);
    const double initial_objective = partition_objective(
        result.assignments, faces, result.prototypes);
    const double threshold = target_ratio * initial_objective;
    result.history.push_back({1U, initial_objective, {}});
    while (result.prototypes.size() < max_clusters) {
        const std::int32_t new_seed = select_new_seed(
            result.assignments, faces, result.prototypes);
        if (new_seed < 0) break;
        result.assignments[static_cast<std::size_t>(new_seed)] =
            static_cast<std::int32_t>(result.prototypes.size());
        result.prototypes = update_prototypes(
            result.assignments, faces,
            static_cast<std::uint32_t>(result.prototypes.size() + 1U));
        for (std::uint32_t iteration = 0; iteration < refinement_iterations; ++iteration) {
            result.seeds = select_representatives(result.assignments, faces, result.prototypes);
            auto grown = grow_from_seeds(result.seeds, faces, result.prototypes);
            std::vector<bool> populated(result.prototypes.size(), false);
            for (const std::int32_t assignment : grown) {
                if (assignment >= 0 &&
                    static_cast<std::size_t>(assignment) < populated.size()) {
                    populated[static_cast<std::size_t>(assignment)] = true;
                }
            }
            // Symmetric tiny meshes can make two representatives collapse to
            // one graph-growth region. Keep the last valid refinement instead
            // of passing an empty cluster to the centroid update. Production
            // captures never take this guard, so their arithmetic is unchanged.
            if (!std::ranges::all_of(populated, [](bool value) { return value; })) {
                break;
            }
            result.assignments = std::move(grown);
            result.prototypes = update_prototypes(
                result.assignments, faces, static_cast<std::uint32_t>(result.prototypes.size()));
        }
        const double current_objective = partition_objective(
            result.assignments, faces, result.prototypes);
        result.history.push_back({static_cast<std::uint32_t>(result.prototypes.size()),
                                  current_objective, result.seeds});
        if (result.prototypes.size() >= 2U && current_objective < threshold) break;
    }
    return result;
}

NaturalUvPartitionPostprocess postprocess_natural_uv_partition(
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    std::span<const std::int32_t> initial_assignments,
    std::uint32_t max_iterations) {
    if (triangle_indices.size() % 3U != 0U || vertex_uv.size() % 2U != 0U ||
        triangle_indices.size() / 3U != initial_assignments.size()) {
        throw std::invalid_argument("natural UV postprocess arrays have incompatible shapes");
    }
    const std::size_t vertex_count = vertex_uv.size() / 2U;
    const auto faces = build_chart_topology(triangle_indices, vertex_count);
    std::vector<std::pair<std::int32_t, std::int32_t>> references(
        vertex_count, {-1, -1});
    for (std::uint32_t face = 0; face < faces.size(); ++face) {
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            references[faces[face].vertices[corner]] = {
                static_cast<std::int32_t>(face), static_cast<std::int32_t>(corner)};
        }
    }

    NaturalUvPartitionPostprocess result;
    result.assignments.assign(initial_assignments.begin(), initial_assignments.end());
    for (std::uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::vector<bool> interior = partition_interior_vertices(
            faces, result.assignments, references);
        auto candidates = partition_candidates(
            faces, vertex_uv, result.assignments, references, interior);
        if (candidates.empty()) break;
        std::vector<bool> locked(vertex_count, false);
        std::uint32_t applied = 0U;
        for (const NaturalUvPartitionCandidate &candidate : candidates) {
            if (locked[candidate.vertex]) continue;
            const auto [start_face, start_corner] = references[candidate.vertex];
            if (start_face < 0) continue;
            const auto ring = vertex_ring(
                faces, static_cast<std::uint32_t>(start_face),
                static_cast<std::uint32_t>(start_corner));
            if (!ring) continue;
            bool valid = true;
            for (const VertexRingEntry &entry : *ring) {
                const std::int32_t label = result.assignments[entry.face];
                const std::int32_t next_label = result.assignments[entry.next_face];
                if (candidate.target == label) continue;
                const std::uint32_t side_edge = (entry.corner + 1U) % 3U;
                const std::int32_t side_face = faces[entry.face].neighbors[side_edge];
                if (side_face < 0 ||
                    result.assignments[static_cast<std::size_t>(side_face)] != label) {
                    valid = false;
                    break;
                }
                if (next_label == label) {
                    const std::uint32_t outer_vertex =
                        faces[entry.next_face].vertices[(entry.next_corner + 1U) % 3U];
                    if (!interior[outer_vertex]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid) continue;
            interior[candidate.vertex] = true;
            for (const VertexRingEntry &entry : *ring) {
                const std::uint32_t outer_vertex =
                    faces[entry.face].vertices[(entry.corner + 1U) % 3U];
                locked[outer_vertex] = true;
                if (result.assignments[entry.face] != candidate.target) {
                    interior[outer_vertex] = false;
                    result.assignments[entry.face] = candidate.target;
                }
            }
            ++applied;
        }
        result.history.push_back({std::move(candidates), applied});
        if (applied == 0U) break;
    }
    return result;
}

}  // namespace metashape_texture
