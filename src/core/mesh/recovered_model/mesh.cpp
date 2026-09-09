#include "metmodel/mesh.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace metmodel
{

    void recompute_normals(Mesh& mesh)
    {
        for (Vertex& vertex : mesh.vertices)
            vertex.normal = {};
        for (const Face& face : mesh.faces)
        {
            const metalign::Vec3 a = mesh.vertices[face.vertices[0]].position;
            const metalign::Vec3 b = mesh.vertices[face.vertices[1]].position;
            const metalign::Vec3 c = mesh.vertices[face.vertices[2]].position;
            const metalign::Vec3 weighted = metalign::cross(b - a, c - a);
            for (const std::size_t index : face.vertices)
                mesh.vertices[index].normal = mesh.vertices[index].normal + weighted;
        }
        for (Vertex& vertex : mesh.vertices)
            vertex.normal = metalign::normalized(vertex.normal);
    }

    std::size_t recovered_qem_part_face_budget(std::size_t part_faces,
                                               std::size_t total_faces,
                                               std::size_t global_target_faces,
                                               bool expanded_allowance)
    {
        if (part_faces == 0U || total_faces == 0U || global_target_faces == 0U || part_faces > total_faces)
        {
            throw std::invalid_argument("invalid recovered QEM part budget input");
        }
        const double denominator = static_cast<double>(total_faces) / static_cast<double>(global_target_faces);
        const double proportional = static_cast<double>(part_faces) / denominator + 1.0;
        if (!std::isfinite(proportional) || proportional < 0.0 ||
            proportional > static_cast<double>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("recovered QEM proportional budget overflow");
        }
        const std::size_t base = static_cast<std::size_t>(proportional);
        const double scaled = static_cast<double>(base) * (expanded_allowance ? 1.10 : 1.01);
        if (!std::isfinite(scaled) || scaled < 1.0 ||
            scaled > static_cast<double>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("recovered QEM allowance budget overflow");
        }
        return static_cast<std::size_t>(scaled);
    }

    QemDecimationStats decimate_mesh_qem_mode3(Mesh& mesh,
                                               std::size_t target_faces,
                                               std::span<const float> vertex_scale,
                                               std::vector<QemCollapseEvent>* trace,
                                               std::vector<RecoveredMeshTrimAttribute>* vertex_attributes,
                                               const std::function<bool()>& is_cancelled)
    {
        if (target_faces == 0U)
            throw std::runtime_error("QEM decimation target must be positive");
        if (vertex_attributes != nullptr && vertex_attributes->size() != mesh.vertices.size())
        {
            throw std::runtime_error("QEM trim-attribute count does not match vertices");
        }
        QemDecimationStats stats;
        stats.input_vertices = mesh.vertices.size();
        stats.input_faces = mesh.faces.size();
        if (mesh.faces.size() <= target_faces || mesh.vertices.empty())
        {
            stats.output_vertices = mesh.vertices.size();
            stats.output_faces = mesh.faces.size();
            return stats;
        }
        if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("QEM vertex count exceeds uint32 domain");
        if (!vertex_scale.empty() && vertex_scale.size() != mesh.vertices.size())
            throw std::runtime_error("QEM vertex-scale count does not match vertices");

        struct Quadric
        {
            // Symmetric 4x4 followed by the accumulated unscaled face area:
            // xx,xy,xz,xw,yy,yz,yw,zz,zw,ww,area.
            std::array<double, 11> q{};
            Quadric& operator+=(const Quadric& other) noexcept
            {
                for (std::size_t i = 0U; i != q.size(); ++i)
                    q[i] += other.q[i];
                return *this;
            }
        };
        struct Candidate
        {
            std::uint32_t a{};
            std::uint32_t b{};
            std::uint32_t generation{};
            float cost{};
            std::uint64_t serial{};
        };
        struct CandidateGreater
        {
            bool operator()(const Candidate& lhs, const Candidate& rhs) const noexcept
            {
                // Recovered push/pop assembly compares only the stored float
                // cost.  In particular, equal-cost records do not bubble during
                // push; adding an endpoint or insertion-order tie-break changes
                // the accepted collapse sequence on planar regions.
                return lhs.cost > rhs.cost;
            }
        };

        const auto as_float_point = [](const metalign::Vec3& p)
        { return metalign::Vec3{static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)}; };
        const auto evaluate = [](const Quadric& q, const metalign::Vec3& p) noexcept
        {
            const double x = p.x, y = p.y, z = p.z;
            double value = x * x * q.q[0];
            value += (x + x) * y * q.q[1];
            value += (x + x) * z * q.q[2];
            value += (x + x) * q.q[3];
            value += y * y * q.q[4];
            value += (y + y) * z * q.q[5];
            value += (y + y) * q.q[6];
            value += z * z * q.q[7];
            value += (z + z) * q.q[8];
            value += q.q[9];
            return value;
        };
        const auto sum_quadric = [](Quadric lhs, const Quadric& rhs)
        {
            lhs += rhs;
            return lhs;
        };
        const auto solve3 = [](const Quadric& q, metalign::Vec3& p) noexcept
        {
            const double q0 = q.q[0], q1 = q.q[1], q2 = q.q[2];
            const double q3 = q.q[3], q4 = q.q[4], q5 = q.q[5];
            const double q6 = q.q[6], q7 = q.q[7], q8 = q.q[8];
            // Preserve sub_2CE5AB0's exact multiply/subtract order.  The solved
            // coordinates can sit near severe cancellation; algebraically
            // equivalent cofactor expressions changed 24 of 7,155 float results.
            const double c00 = q7 * q4 - q5 * q5;
            const double determinant_middle = (q7 * q1 - q5 * q2) * q1;
            const double c02 = q5 * q1 - q2 * q4;
            const double determinant = q0 * c00 - determinant_middle + q2 * c02;
            if (!std::isfinite(determinant) || std::fabs(determinant) < 1e-12)
                return false;
            const double inverse = determinant != 0.0 ? 1.0 / determinant : 0.0;
            const double i00 = c00 * inverse;
            const double i01 = (q5 * q2 - q7 * q1) * inverse;
            const double i02 = c02 * inverse;
            const double i11 = (q7 * q0 - q2 * q2) * inverse;
            const double i12 = (q2 * q1 - q5 * q0) * inverse;
            const double i22 = (q4 * q0 - q1 * q1) * inverse;
            double x = i00 * q3;
            x += q6 * i01;
            x += q8 * i02;
            double y = i11 * q6;
            y += i01 * q3;
            y += q8 * i12;
            double z = i12 * q6;
            z += i02 * q3;
            z += q8 * i22;
            p.x = -x;
            p.y = -y;
            p.z = -z;
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        };

        std::vector<metalign::Vec3> positions(mesh.vertices.size());
        std::vector<Quadric> quadrics(mesh.vertices.size());
        std::vector<std::uint32_t> vertex_generation(mesh.vertices.size(), 0U);
        std::vector<std::uint8_t> vertex_active(mesh.vertices.size(), 1U);
        std::vector<std::uint8_t> boundary_vertex(mesh.vertices.size(), 0U);
        std::vector<std::uint8_t> face_active(mesh.faces.size(), 1U);
        std::vector<std::vector<std::uint32_t>> incident(mesh.vertices.size());
        for (std::size_t i = 0U; i != mesh.vertices.size(); ++i)
            positions[i] = as_float_point(mesh.vertices[i].position);

        std::size_t active_faces = mesh.faces.size();
        for (std::size_t fi = 0U; fi != mesh.faces.size(); ++fi)
        {
            const auto& face = mesh.faces[fi];
            for (const std::size_t vertex : face.vertices)
            {
                if (vertex >= mesh.vertices.size())
                    throw std::runtime_error("QEM face index is out of range");
                incident[vertex].push_back(static_cast<std::uint32_t>(fi));
            }
            const auto p0 = positions[face.vertices[0]];
            const auto p1 = positions[face.vertices[1]];
            const auto p2 = positions[face.vertices[2]];
            const auto edge1 = p1 - p0;
            const auto edge2 = p2 - p0;
            const metalign::Vec3 cross{edge1.y * edge2.z - edge1.z * edge2.y,
                                       edge2.x * edge1.z - edge1.x * edge2.z,
                                       edge1.x * edge2.y - edge1.y * edge2.x};
            const double squared = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
            const double length = std::sqrt(squared);
            const double inverse = length >= 1e-20 ? 1.0 / length : 0.0;
            const metalign::Vec3 n = cross * inverse;
            const double d = -(p0.y * n.y + p0.x * n.x + p0.z * n.z);
            // sub_2CE3650 recomputes the same double cross, sums x^2+y^2 and
            // then z^2, and multiplies its length by one half.  Reusing `length`
            // preserves that exact addition order.
            const double weight = length * 0.5;
            const std::array<double, 4> plane{n.x, n.y, n.z, d};
            Quadric face_quadric;
            std::size_t packed = 0U;
            for (std::size_t row = 0U; row != 4U; ++row)
            {
                for (std::size_t column = row; column != 4U; ++column)
                    face_quadric.q[packed++] = plane[row] * plane[column];
            }
            face_quadric.q[10] = weight;
            // The caller selects quadric-source mode 6.  The target sums the
            // three Marching vertex scales in float, divides by 3 and squares in
            // float, then scales q[0..9] by area / mean_scale^2 in double.  Its
            // eleventh scalar remains the unscaled face area.
            float mean_scale = 1.0F;
            if (!vertex_scale.empty())
            {
                float scale_sum = 0.0F;
                scale_sum = scale_sum + vertex_scale[face.vertices[0]];
                scale_sum = scale_sum + vertex_scale[face.vertices[1]];
                scale_sum = scale_sum + vertex_scale[face.vertices[2]];
                mean_scale = scale_sum / 3.0F;
            }
            const float squared_scale = mean_scale * mean_scale;
            const double inverse_squared_scale = 1.0 / static_cast<double>(squared_scale);
            for (std::size_t coefficient = 0U; coefficient != 10U; ++coefficient)
                face_quadric.q[coefficient] *= inverse_squared_scale;
            for (std::size_t coefficient = 0U; coefficient != 10U; ++coefficient)
                face_quadric.q[coefficient] *= weight;
            for (const std::size_t vertex : face.vertices)
                quadrics[vertex] += face_quadric;
        }

        // sub_2CE25D0 visits faces in ascending index order and prepends each
        // face to the three per-vertex intrusive lists.  Keep `incident` in the
        // resulting head-to-tail order.  Later relinks and fan splits must also
        // prepend: this order is observable when equal-cost candidates enter the
        // heap.
        for (auto& chain : incident)
            std::reverse(chain.begin(), chain.end());

        auto candidate_point = [&](std::uint32_t a, std::uint32_t b, float& stored_cost)
        {
            const Quadric combined = sum_quadric(quadrics[a], quadrics[b]);
            metalign::Vec3 point;
            if (!solve3(combined, point))
            {
                const metalign::Vec3 p0 = positions[a];
                const metalign::Vec3 p1 = positions[b];
                // sub_2CE5CC0 parameterizes from endpoint b toward endpoint a.
                const metalign::Vec3 direction = p0 - p1;
                // Exact quadratic minimizer restricted to the source-target
                // segment, matching the recovered mode>1 fallback.
                const double e0 = evaluate(combined, p0);
                const double e1 = evaluate(combined, p1);
                const metalign::Vec3 midpoint = (p0 + p1) * 0.5;
                const double em = evaluate(combined, midpoint);
                const double ax =
                    combined.q[0] * direction.x + combined.q[1] * direction.y + combined.q[2] * direction.z;
                const double ay =
                    combined.q[1] * direction.x + combined.q[4] * direction.y + combined.q[5] * direction.z;
                const double az =
                    combined.q[2] * direction.x + combined.q[5] * direction.y + combined.q[7] * direction.z;
                const double denominator = direction.x * ax + direction.y * ay + direction.z * az;
                if (std::isfinite(denominator) && std::fabs(2.0 * denominator) >= 1e-12)
                {
                    const double numerator = -(combined.q[3] * direction.x + combined.q[6] * direction.y +
                                               combined.q[8] * direction.z + p1.x * ax + p1.y * ay + p1.z * az);
                    const double t = std::clamp(numerator / denominator, 0.0, 1.0);
                    point = p1 + direction * t;
                }
                else if (e1 <= e0 && e1 <= em)
                {
                    point = p1;
                }
                else if (em < e0)
                {
                    point = midpoint;
                }
                else
                {
                    point = p0;
                }
            }
            point = as_float_point(point);
            // Source quadric mode 6 satisfies (mode & ~2) == 4, so the target
            // normalizes the candidate error by accumulated face area q[10].
            const double normalized_cost = evaluate(combined, point) / combined.q[10];
            stored_cost = static_cast<float>(std::max(normalized_cost, 0.0));
            return point;
        };

        // sub_2CE25D0 prepends faces while visiting them in ascending face order.
        // sub_2CE3D80 therefore walks incident faces in descending index order and
        // emits each face's vertices in corner order, suppressing duplicates.  The
        // resulting order is observable whenever several candidates have the same
        // float cost, so neither a sorted set nor hash iteration is equivalent.
        std::vector<std::uint32_t> neighbor_mark(mesh.vertices.size(), 0U);
        std::uint32_t neighbor_stamp = 0U;
        const auto fill_active_neighbors = [&](const std::uint32_t vertex, auto& result)
        {
            result.clear();
            if (++neighbor_stamp == 0U)
            {
                std::fill(neighbor_mark.begin(), neighbor_mark.end(), 0U);
                neighbor_stamp = 1U;
            }
            result.reserve(incident[vertex].size() * 2U);
            neighbor_mark[vertex] = neighbor_stamp;
            for (const std::uint32_t fi : incident[vertex])
            {
                if (face_active[fi] == 0U)
                    continue;
                if (std::find(mesh.faces[fi].vertices.begin(), mesh.faces[fi].vertices.end(), vertex) ==
                    mesh.faces[fi].vertices.end())
                {
                    continue;
                }
                for (const std::size_t value : mesh.faces[fi].vertices)
                {
                    const auto neighbor = static_cast<std::uint32_t>(value);
                    if (vertex_active[neighbor] == 0U || neighbor_mark[neighbor] == neighbor_stamp)
                    {
                        continue;
                    }
                    neighbor_mark[neighbor] = neighbor_stamp;
                    result.push_back(neighbor);
                }
            }
        };
        auto active_neighbors = [&](const std::uint32_t vertex)
        {
            std::vector<std::uint32_t> result;
            fill_active_neighbors(vertex, result);
            return result;
        };

        // The target keeps the candidate heap in an exposed vector because it
        // periodically removes stale records in place and rebuilds the heap.
        // Keep the recovered heap algorithms explicit: the choice of the right
        // child when two child costs are equal is observable much later in large
        // planar meshes and is not a portable std::make_heap guarantee.
        std::vector<Candidate> heap;
        const auto adjust_candidate_heap = [&](std::size_t hole, std::size_t length, Candidate value)
        {
            const std::size_t top = hole;
            std::size_t second_child = hole;
            while (second_child < (length - 1U) / 2U)
            {
                second_child = 2U * (second_child + 1U); // right child
                if (heap[second_child].cost > heap[second_child - 1U].cost)
                    --second_child;
                heap[hole] = heap[second_child];
                hole = second_child;
            }
            if ((length & 1U) == 0U && second_child == (length - 2U) / 2U)
            {
                second_child = 2U * (second_child + 1U);
                heap[hole] = heap[second_child - 1U];
                hole = second_child - 1U;
            }
            while (hole > top)
            {
                const std::size_t parent = (hole - 1U) / 2U;
                if (!(heap[parent].cost > value.cost))
                    break;
                heap[hole] = heap[parent];
                hole = parent;
            }
            heap[hole] = value;
        };
        const auto push_candidate = [&](Candidate value)
        {
            heap.push_back(value);
            std::size_t hole = heap.size() - 1U;
            while (hole != 0U)
            {
                const std::size_t parent = (hole - 1U) / 2U;
                if (!(heap[parent].cost > value.cost))
                    break;
                heap[hole] = heap[parent];
                hole = parent;
            }
            heap[hole] = value;
        };
        const auto pop_candidate = [&]()
        {
            const Candidate result = heap.front();
            if (heap.size() == 1U)
            {
                heap.pop_back();
                return result;
            }
            const Candidate value = heap.back();
            heap.pop_back();
            adjust_candidate_heap(0U, heap.size(), value);
            return result;
        };
        std::uint64_t serial = 0U;
        std::uint32_t current_generation = 0U;
        std::unordered_map<std::uint64_t, std::uint32_t> initial_edge_faces;
        initial_edge_faces.reserve(mesh.faces.size() * 2U);
        auto push_edge = [&](std::uint32_t a, std::uint32_t b)
        {
            if (a == b || vertex_active[a] == 0U || vertex_active[b] == 0U)
                return;
            float cost = 0.0F;
            (void)candidate_point(a, b, cost);
            push_candidate({a, b, current_generation, cost, serial++});
        };
        // Initial target traversal is vertex-ascending and accepts each neighbor
        // only when neighbor > vertex.  A sorted edge set preserves that order
        // independently of hash iteration.
        for (const auto& face : mesh.faces)
        {
            for (std::size_t edge = 0U; edge != 3U; ++edge)
            {
                auto a = static_cast<std::uint32_t>(face.vertices[edge]);
                auto b = static_cast<std::uint32_t>(face.vertices[(edge + 1U) % 3U]);
                if (a > b)
                    std::swap(a, b);
                const std::uint64_t key = (static_cast<std::uint64_t>(a) << 32U) | b;
                ++initial_edge_faces[key];
            }
        }
        // One candidate is emitted for every unique active edge.  Reserve the
        // exact upper bound so vector growth cannot repeatedly move the recovered
        // heap; insertion and heap-adjustment order remain unchanged.
        heap.reserve(initial_edge_faces.size());
        for (const auto& [edge, face_count] : initial_edge_faces)
        {
            if (face_count == 1U)
            {
                boundary_vertex[static_cast<std::uint32_t>(edge >> 32U)] = 1U;
                boundary_vertex[static_cast<std::uint32_t>(edge)] = 1U;
            }
        }
        std::vector<std::uint32_t> initial_neighbors;
        for (std::uint32_t vertex = 0U; vertex != mesh.vertices.size(); ++vertex)
        {
            fill_active_neighbors(vertex, initial_neighbors);
            for (const std::uint32_t neighbor : initial_neighbors)
            {
                if (neighbor > vertex)
                    push_edge(vertex, neighbor);
            }
        }
        const std::size_t heap_rebuild_threshold = heap.size() + heap.size() / 5U;
        std::size_t heap_rebuild_countdown = heap.size() / 30U;
        if (const char* path = std::getenv("METMODEL_QEM_INITIAL_HEAP_DUMP"))
        {
            struct StoredCandidate
            {
                std::uint32_t a;
                std::uint32_t b;
                std::uint32_t generation;
                float cost;
            };
            static_assert(sizeof(StoredCandidate) == 16U);
            std::FILE* stream = std::fopen(path, "wb");
            if (stream == nullptr)
                throw std::runtime_error("cannot open QEM initial-heap dump");
            bool ok = true;
            for (const Candidate& candidate : heap)
            {
                const StoredCandidate stored{candidate.a, candidate.b, candidate.generation, candidate.cost};
                ok = ok && std::fwrite(&stored, sizeof(stored), 1U, stream) == 1U;
            }
            ok = ok && std::fclose(stream) == 0;
            if (!ok)
                throw std::runtime_error("cannot write QEM initial-heap dump");
        }

        const auto candidate_is_valid = [&](const Candidate& candidate)
        {
            return vertex_active[candidate.a] != 0U && vertex_active[candidate.b] != 0U &&
                   boundary_vertex[candidate.a] == 0U && boundary_vertex[candidate.b] == 0U &&
                   candidate.generation >= vertex_generation[candidate.a] &&
                   candidate.generation >= vertex_generation[candidate.b];
        };
        const auto rebuild_candidate_heap = [&]()
        {
            if (const char* event_text = std::getenv("METMODEL_QEM_REBUILD_DUMP_EVENT"))
            {
                char* end = nullptr;
                const auto event = std::strtoull(event_text, &end, 10);
                if (end != event_text && *end == '\0' && event == stats.accepted_collapses)
                {
                    const char* directory = std::getenv("METMODEL_QEM_REBUILD_HEAP_DUMP_DIRECTORY");
                    if (directory == nullptr)
                        throw std::runtime_error("QEM rebuild dump directory is missing");
                    char path[4096];
                    const int length = std::snprintf(path,
                                                     sizeof(path),
                                                     "%s/rebuild_%zu_before_candidates16.bin",
                                                     directory,
                                                     stats.accepted_collapses);
                    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(path))
                        throw std::runtime_error("QEM rebuild dump path is too long");
                    std::FILE* stream = std::fopen(path, "wb");
                    if (stream == nullptr)
                        throw std::runtime_error("cannot open QEM rebuild-before dump");
                    bool ok = true;
                    for (const Candidate& candidate : heap)
                    {
                        const std::array<std::uint32_t, 4> stored{candidate.a,
                                                                  candidate.b,
                                                                  candidate.generation,
                                                                  std::bit_cast<std::uint32_t>(candidate.cost)};
                        ok = ok && std::fwrite(stored.data(), sizeof(stored), 1U, stream) == 1U;
                    }
                    ok = ok && std::fclose(stream) == 0;
                    if (!ok)
                        throw std::runtime_error("cannot write QEM rebuild-before dump");
                }
            }
            // sub_2CE0B60 does not preserve record order: an invalid slot is
            // replaced by the current last record and the same slot is tested
            // again.  The surviving vector is then heapified bottom-up.
            std::size_t item = 0U;
            while (item < heap.size())
            {
                if (candidate_is_valid(heap[item]))
                {
                    ++item;
                    continue;
                }
                heap[item] = heap.back();
                heap.pop_back();
            }
            if (heap.size() > 1U)
            {
                std::size_t parent = (heap.size() - 2U) / 2U;
                for (;;)
                {
                    const Candidate value = heap[parent];
                    adjust_candidate_heap(parent, heap.size(), value);
                    if (parent == 0U)
                        break;
                    --parent;
                }
            }
            if (const char* directory = std::getenv("METMODEL_QEM_REBUILD_HEAP_DUMP_DIRECTORY"))
            {
                char path[4096];
                const int length = std::snprintf(
                    path, sizeof(path), "%s/rebuild_%zu_candidates16.bin", directory, stats.accepted_collapses);
                if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(path))
                    throw std::runtime_error("QEM rebuild dump path is too long");
                std::FILE* stream = std::fopen(path, "wb");
                if (stream == nullptr)
                    throw std::runtime_error("cannot open QEM rebuild-heap dump");
                bool ok = true;
                for (const Candidate& candidate : heap)
                {
                    const std::array<std::uint32_t, 4> stored{
                        candidate.a, candidate.b, candidate.generation, std::bit_cast<std::uint32_t>(candidate.cost)};
                    ok = ok && std::fwrite(stored.data(), sizeof(stored), 1U, stream) == 1U;
                }
                ok = ok && std::fclose(stream) == 0;
                if (!ok)
                    throw std::runtime_error("cannot write QEM rebuild-heap dump");
            }
            heap_rebuild_countdown = heap.size() / 30U;
        };

        const auto fill_active_incident = [&](const std::uint32_t vertex, auto& result)
        {
            result.clear();
            result.reserve(incident[vertex].size());
            for (const std::uint32_t face : incident[vertex])
            {
                if (face_active[face] != 0U &&
                    std::find(mesh.faces[face].vertices.begin(), mesh.faces[face].vertices.end(), vertex) !=
                        mesh.faces[face].vertices.end())
                {
                    result.push_back(face);
                }
            }
        };
        auto active_incident = [&](const std::uint32_t vertex)
        {
            std::vector<std::uint32_t> result;
            fill_active_incident(vertex, result);
            return result;
        };
        auto split_disconnected_fan = [&](std::uint32_t vertex, std::uint32_t excluded)
        {
            const auto ordered = active_neighbors(vertex);
            std::vector<std::uint32_t> fan_neighbors;
            fan_neighbors.reserve(ordered.size());
            for (const std::uint32_t neighbor : ordered)
            {
                if (neighbor != excluded)
                    fan_neighbors.push_back(neighbor);
            }
            if (fan_neighbors.empty())
                return std::vector<std::uint32_t>{};

            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> graph;
            graph.reserve(fan_neighbors.size());
            for (const std::uint32_t neighbor : fan_neighbors)
                graph.try_emplace(neighbor);
            const auto chain_faces = active_incident(vertex);
            for (const std::uint32_t fi : chain_faces)
            {
                const auto& face = mesh.faces[fi].vertices;
                std::array<std::uint32_t, 2> other{};
                std::size_t count = 0U;
                for (const std::size_t value : face)
                {
                    const auto item = static_cast<std::uint32_t>(value);
                    if (item != vertex && item != excluded && count < 2U)
                        other[count++] = item;
                }
                if (count == 2U)
                {
                    graph[other[0]].push_back(other[1]);
                    graph[other[1]].push_back(other[0]);
                }
            }

            std::unordered_set<std::uint32_t> visited;
            visited.reserve(fan_neighbors.size());
            std::vector<std::vector<std::uint32_t>> components;
            for (const std::uint32_t seed : fan_neighbors)
            {
                if (!visited.insert(seed).second)
                    continue;
                components.push_back({});
                std::vector<std::uint32_t> stack{seed};
                while (!stack.empty())
                {
                    const std::uint32_t current = stack.back();
                    stack.pop_back();
                    components.back().push_back(current);
                    for (const std::uint32_t adjacent : graph[current])
                    {
                        if (visited.insert(adjacent).second)
                            stack.push_back(adjacent);
                    }
                }
            }
            std::vector<std::uint32_t> clones;
            for (std::size_t component = 1U; component != components.size(); ++component)
            {
                if (mesh.vertices.size() >= std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::runtime_error("QEM split vertex exceeds uint32 domain");
                }
                const auto clone = static_cast<std::uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(mesh.vertices[vertex]);
                positions.push_back(positions[vertex]);
                quadrics.push_back(quadrics[vertex]);
                vertex_generation.push_back(vertex_generation[vertex]);
                vertex_active.push_back(1U);
                boundary_vertex.push_back(boundary_vertex[vertex]);
                neighbor_mark.push_back(0U);
                incident.emplace_back();
                if (vertex_attributes != nullptr)
                    vertex_attributes->push_back((*vertex_attributes)[vertex]);

                std::unordered_set<std::uint32_t> members(components[component].begin(), components[component].end());
                for (const std::uint32_t fi : chain_faces)
                {
                    if (face_active[fi] == 0U)
                        continue;
                    auto& face = mesh.faces[fi].vertices;
                    if (std::find(face.begin(), face.end(), vertex) == face.end())
                        continue;
                    bool selected = false;
                    for (const std::size_t value : face)
                    {
                        selected = selected || members.contains(static_cast<std::uint32_t>(value));
                    }
                    if (!selected)
                        continue;
                    for (std::size_t& value : face)
                    {
                        if (value == vertex)
                            value = clone;
                    }
                }
                // sub_2CE3B30 clears both intrusive-list heads and visits the
                // captured pre-split face chain, prepending every surviving face
                // to the endpoint it now contains.  Consequently it reverses the
                // original endpoint's active chain as well as building the clone
                // chain.  Rebuilding only the clone leaves a later equal-cost
                // refresh in a different order (South event 293521).
                std::vector<std::uint32_t> original_chain;
                std::vector<std::uint32_t> clone_chain;
                original_chain.reserve(chain_faces.size());
                clone_chain.reserve(chain_faces.size());
                for (const std::uint32_t fi : chain_faces)
                {
                    if (face_active[fi] == 0U)
                        continue;
                    const auto& face = mesh.faces[fi].vertices;
                    if (std::find(face.begin(), face.end(), vertex) != face.end())
                        original_chain.insert(original_chain.begin(), fi);
                    if (std::find(face.begin(), face.end(), clone) != face.end())
                        clone_chain.insert(clone_chain.begin(), fi);
                }
                incident[vertex] = std::move(original_chain);
                incident[clone] = std::move(clone_chain);
                clones.push_back(clone);
            }
            return clones;
        };
        std::vector<std::uint32_t> affected;
        std::vector<std::uint32_t> other_incident;
        std::vector<std::uint32_t> repair_neighbors;
        while (active_faces > target_faces && !heap.empty())
        {
            if (is_cancelled && is_cancelled())
            {
                throw std::runtime_error("Recovered QEM cancelled");
            }
            const Candidate candidate = pop_candidate();
            if (stats.first_popped_source == std::numeric_limits<std::uint32_t>::max())
            {
                stats.first_popped_source = candidate.a;
                stats.first_popped_target = candidate.b;
                stats.first_popped_cost = candidate.cost;
            }
            const std::uint32_t a = candidate.a;
            const std::uint32_t b = candidate.b;
            if (!candidate_is_valid(candidate))
            {
                ++stats.stale_candidates;
                continue;
            }

            fill_active_incident(a, affected);
            fill_active_incident(b, other_incident);
            for (const std::uint32_t fi : other_incident)
            {
                if (std::find(affected.begin(), affected.end(), fi) == affected.end())
                    affected.push_back(fi);
            }
            std::size_t edge_faces = 0U;
            for (const std::uint32_t fi : affected)
            {
                const auto& face = mesh.faces[fi].vertices;
                const bool has_a = std::find(face.begin(), face.end(), a) != face.end();
                const bool has_b = std::find(face.begin(), face.end(), b) != face.end();
                if (has_a && has_b)
                {
                    ++edge_faces;
                }
            }
            // Work+160 is the vector of incident faces unique to one endpoint;
            // sub_2CE2AA0 accepts at most 11 of these.  It is not a vertex-valence
            // limit (the two counts coincide on many regular patches, which hid
            // this distinction in earlier replays).
            const std::size_t unique_incident_faces = affected.size() >= edge_faces ? affected.size() - edge_faces : 0U;
            // The explicit link-condition option is disabled in the captured
            // target settings.  Boundary endpoint rejection happened above; the
            // remaining recovered topology gate is the <=11 unique-face vector.
            if (unique_incident_faces > 11U)
            {
                ++stats.topology_rejections;
                continue;
            }
            float recomputed_cost = 0.0F;
            const metalign::Vec3 solved_point = candidate_point(a, b, recomputed_cost);
            // sub_2CE4550 stores the candidate as float deltas from the source
            // vertex; sub_2CE3070 and the commit path reconstruct it with float
            // additions.  This subtraction/addition round-trip is observable on
            // 24 of the 2,385 Shoe collapses and must not be optimized away.
            const auto delta_roundtrip = [](double solved, double source)
            {
                const float solved_f = static_cast<float>(solved);
                const float source_f = static_cast<float>(source);
                const float delta = solved_f - source_f;
                return static_cast<double>(source_f + delta);
            };
            const metalign::Vec3 point{delta_roundtrip(solved_point.x, positions[a].x),
                                       delta_roundtrip(solved_point.y, positions[a].y),
                                       delta_roundtrip(solved_point.z, positions[a].z)};
            bool normal_ok = true;
            for (const std::uint32_t fi : affected)
            {
                const auto& face = mesh.faces[fi].vertices;
                const bool has_a = std::find(face.begin(), face.end(), a) != face.end();
                const bool has_b = std::find(face.begin(), face.end(), b) != face.end();
                if (has_a && has_b)
                    continue;
                std::array<std::array<float, 3>, 3> old_points;
                std::array<std::array<float, 3>, 3> new_points;
                for (std::size_t corner = 0U; corner != 3U; ++corner)
                {
                    const auto vertex = static_cast<std::uint32_t>(face[corner]);
                    const auto old = positions[vertex];
                    old_points[corner] = {
                        static_cast<float>(old.x), static_cast<float>(old.y), static_cast<float>(old.z)};
                    const auto replacement = (vertex == a || vertex == b) ? point : old;
                    new_points[corner] = {static_cast<float>(replacement.x),
                                          static_cast<float>(replacement.y),
                                          static_cast<float>(replacement.z)};
                }
                const auto target_cross = [](const auto& p)
                {
                    const float e1x = p[1][0] - p[0][0];
                    const float e1y = p[1][1] - p[0][1];
                    const float e1z = p[1][2] - p[0][2];
                    const float e2x = p[2][0] - p[0][0];
                    const float e2y = p[2][1] - p[0][1];
                    const float e2z = p[2][2] - p[0][2];
                    return std::array<float, 3>{e1y * e2z - e2y * e1z, e2x * e1z - e1x * e2z, e1x * e2y - e2x * e1y};
                };
                const auto old_cross = target_cross(old_points);
                const auto new_cross = target_cross(new_points);
                const auto target_length = [](const auto& value)
                {
                    const float squared = (value[0] * value[0] + value[1] * value[1]) + value[2] * value[2];
                    return std::sqrt(squared);
                };
                const float old_length = target_length(old_cross);
                const float new_length = target_length(new_cross);
                if (old_length >= 1.0e-11F && new_length < 1.0e-12F)
                {
                    normal_ok = false;
                    break;
                }
                const float old_inverse = old_length >= 1.0e-20F ? 1.0F / old_length : 0.0F;
                const float new_inverse = new_length >= 1.0e-20F ? 1.0F / new_length : 0.0F;
                const float dot = (new_cross[0] * new_inverse) * (old_cross[0] * old_inverse) +
                                  (new_cross[1] * new_inverse) * (old_cross[1] * old_inverse) +
                                  (new_cross[2] * new_inverse) * (old_cross[2] * old_inverse);
                if (dot < -0.8F)
                {
                    normal_ok = false;
                    break;
                }
            }
            if (!normal_ok)
            {
                ++stats.normal_rejections;
                continue;
            }

            positions[a] = point;
            if (trace != nullptr)
            {
                trace->push_back(
                    {a,
                     b,
                     {static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z)},
                     recomputed_cost});
            }
            quadrics[a] += quadrics[b];
            if (vertex_attributes != nullptr)
            {
                for (std::size_t channel = 0U; channel != 4U; ++channel)
                {
                    (*vertex_attributes)[a][channel] =
                        (*vertex_attributes)[a][channel] + (*vertex_attributes)[b][channel];
                }
            }
            vertex_active[b] = 0U;
            vertex_generation[a] = current_generation;
            for (const std::uint32_t fi : affected)
            {
                if (face_active[fi] == 0U)
                    continue;
                const auto& face = mesh.faces[fi].vertices;
                bool has_a = false, has_b = false;
                for (const std::size_t value : face)
                {
                    has_a = has_a || value == a;
                    has_b = has_b || value == b;
                }
                if (has_a && has_b)
                {
                    face_active[fi] = 0U;
                    --active_faces;
                }
            }
            // Two faces whose other vertices occur in opposite order are
            // coincident with opposite winding; the target removes both rather
            // than retaining an arbitrary duplicate.  The commit has two
            // observable snapshots: one after deleting the collapsed edge faces,
            // and one after relinking the target-side faces onto the source.
            auto remove_opposite_faces = [&](std::uint32_t vertex)
            {
                std::unordered_map<std::uint64_t, std::uint32_t> directed_faces;
                directed_faces.reserve(incident[vertex].size());
                for (const std::uint32_t fi : incident[vertex])
                {
                    if (face_active[fi] == 0U)
                        continue;
                    const auto& face = mesh.faces[fi].vertices;
                    const auto position = std::find(face.begin(), face.end(), vertex);
                    if (position == face.end())
                        continue;
                    const std::size_t corner = static_cast<std::size_t>(position - face.begin());
                    const auto next = static_cast<std::uint32_t>(face[(corner + 1U) % 3U]);
                    const auto previous = static_cast<std::uint32_t>(face[(corner + 2U) % 3U]);
                    const std::uint64_t key = (static_cast<std::uint64_t>(next) << 32U) | previous;
                    const std::uint64_t reverse = (static_cast<std::uint64_t>(previous) << 32U) | next;
                    const auto opposite = directed_faces.find(reverse);
                    if (opposite == directed_faces.end())
                    {
                        directed_faces.emplace(key, fi);
                        continue;
                    }
                    if (face_active[opposite->second] != 0U)
                    {
                        face_active[opposite->second] = 0U;
                        face_active[fi] = 0U;
                        active_faces -= 2U;
                    }
                    directed_faces.erase(opposite);
                }
            };
            // Relink the target-side faces first.  sub_2CE3D80 snapshots the
            // source neighbors at this point; later opposite-face deletion does
            // not remove entries from that work vector.
            for (const std::uint32_t fi : incident[b])
            {
                if (face_active[fi] == 0U)
                    continue;
                auto& face = mesh.faces[fi].vertices;
                const bool has_b = std::find(face.begin(), face.end(), b) != face.end();
                if (!has_b)
                    continue;
                for (std::size_t& value : face)
                {
                    if (value == b)
                        value = a;
                }
                incident[a].insert(incident[a].begin(), fi);
            }
            fill_active_neighbors(a, repair_neighbors);
            if (heap.size() + repair_neighbors.size() > heap_rebuild_threshold || heap_rebuild_countdown == 0U)
            {
                rebuild_candidate_heap();
            }
            remove_opposite_faces(a);
            for (const std::uint32_t neighbor : repair_neighbors)
            {
                std::vector<std::uint32_t> edge_ring;
                for (const std::uint32_t fi : incident[a])
                {
                    if (face_active[fi] == 0U)
                        continue;
                    const auto& face = mesh.faces[fi].vertices;
                    if (std::find(face.begin(), face.end(), a) != face.end() &&
                        std::find(face.begin(), face.end(), neighbor) != face.end())
                    {
                        edge_ring.push_back(fi);
                    }
                }
                // sub_2CE0C70 splits rings with more than two active faces.  For
                // exactly two faces it compares the neighbor's cyclic offset
                // from the source in both triangles and splits only equal
                // offsets (same winding around the edge).
                bool repair = edge_ring.size() > 2U;
                if (edge_ring.size() == 2U)
                {
                    std::array<std::uint32_t, 2> orientation{};
                    for (std::size_t item = 0U; item != 2U; ++item)
                    {
                        const auto& face = mesh.faces[edge_ring[item]].vertices;
                        const auto source_position = std::find(face.begin(), face.end(), a);
                        const auto neighbor_position = std::find(face.begin(), face.end(), neighbor);
                        if (source_position == face.end() || neighbor_position == face.end())
                        {
                            throw std::runtime_error("QEM edge ring face misses endpoint");
                        }
                        const auto source_index = static_cast<std::uint32_t>(source_position - face.begin());
                        const auto neighbor_index = static_cast<std::uint32_t>(neighbor_position - face.begin());
                        orientation[item] = (neighbor_index + 3U - source_index) % 3U;
                    }
                    repair = orientation[0] == orientation[1];
                }
                if (repair)
                {
                    for (const std::uint32_t clone : split_disconnected_fan(neighbor, a))
                    {
                        for (const std::uint32_t adjacent : active_neighbors(clone))
                            push_edge(clone, adjacent);
                    }
                }
                // Target refreshes the source-neighbor work item immediately
                // after this neighbor's optional split, not in a second batch.
                push_edge(a, neighbor);
            }
            if (stats.accepted_collapses == 0U)
            {
                stats.first_source = a;
                stats.first_target = b;
                stats.first_cost = recomputed_cost;
            }
            ++stats.accepted_collapses;
            ++current_generation;
            if (heap_rebuild_countdown != 0U)
                --heap_rebuild_countdown;
        }

        // sub_2CE05A0 -> sub_2CE52D0 performs one final disconnected-fan pass
        // after the collapse loop.  It snapshots the current vertex count, skips
        // inactive/boundary vertices and calls the same splitter with the vertex
        // itself as the excluded value.  Newly appended clones are deliberately
        // outside this pass.  South creates one final clone here (vertex 18251).
        const std::size_t final_fan_vertices = mesh.vertices.size();
        for (std::size_t vertex = 0U; vertex != final_fan_vertices; ++vertex)
        {
            if (vertex_active[vertex] == 0U || boundary_vertex[vertex] != 0U)
                continue;
            (void)split_disconnected_fan(static_cast<std::uint32_t>(vertex), static_cast<std::uint32_t>(vertex));
        }

        // The target then clears the active flag of every vertex whose intrusive
        // incident-face head is -1 before compacting.  Our vectors retain stale
        // inactive entries, so the equivalent predicate is an empty active
        // incident set.  This removes isolated vertices without changing the
        // already bit-exact oriented triangle geometry.
        for (std::size_t vertex = 0U; vertex != mesh.vertices.size(); ++vertex)
        {
            if (vertex_active[vertex] != 0U && active_incident(static_cast<std::uint32_t>(vertex)).empty())
            {
                vertex_active[vertex] = 0U;
            }
        }

        std::vector<std::size_t> remap(mesh.vertices.size(), std::numeric_limits<std::size_t>::max());
        std::vector<Vertex> vertices;
        vertices.reserve(mesh.vertices.size() - stats.accepted_collapses);
        std::vector<RecoveredMeshTrimAttribute> compact_attributes;
        if (vertex_attributes != nullptr)
            compact_attributes.reserve(mesh.vertices.size() - stats.accepted_collapses);
        for (std::size_t index = 0U; index != mesh.vertices.size(); ++index)
        {
            if (vertex_active[index] == 0U)
                continue;
            remap[index] = vertices.size();
            Vertex vertex = mesh.vertices[index];
            vertex.position = positions[index];
            vertices.push_back(vertex);
            if (vertex_attributes != nullptr)
                compact_attributes.push_back((*vertex_attributes)[index]);
        }
        std::vector<Face> faces;
        faces.reserve(active_faces);
        for (std::size_t fi = 0U; fi != mesh.faces.size(); ++fi)
        {
            if (face_active[fi] == 0U)
                continue;
            Face face = mesh.faces[fi];
            for (std::size_t& vertex : face.vertices)
            {
                if (vertex >= remap.size() || remap[vertex] == std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("QEM compact remap is invalid");
                vertex = remap[vertex];
            }
            faces.push_back(face);
        }
        mesh.vertices = std::move(vertices);
        mesh.faces = std::move(faces);
        if (vertex_attributes != nullptr)
            *vertex_attributes = std::move(compact_attributes);
        recompute_normals(mesh);
        stats.output_vertices = mesh.vertices.size();
        stats.output_faces = mesh.faces.size();
        return stats;
    }

    RecoveredFixBackTrianglesStats fix_back_triangles_recovered(Mesh& mesh)
    {
        RecoveredFixBackTrianglesStats stats;
        if (mesh.vertices.empty() || mesh.faces.empty())
            return stats;
        if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
            mesh.faces.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("fix-back mesh exceeds uint32 target domain");
        }

        using Float3 = std::array<float, 3>;
        using FaceNeighbors = std::array<std::uint32_t, 3>;
        struct EdgeUse
        {
            std::uint32_t face{};
            std::uint32_t corner{};
        };
        auto edge_key = [](std::uint32_t a, std::uint32_t b)
        {
            if (a > b)
                std::swap(a, b);
            return (static_cast<std::uint64_t>(a) << 32U) | b;
        };
        auto unit_normal = [](const Float3& a, const Float3& b, const Float3& c)
        {
            const float e1x = b[0] - a[0];
            const float e1y = b[1] - a[1];
            const float e1z = b[2] - a[2];
            const float e2x = c[0] - a[0];
            const float e2y = c[1] - a[1];
            const float e2z = c[2] - a[2];
            const float nx = e1y * e2z - e1z * e2y;
            const float ny = e1z * e2x - e1x * e2z;
            const float nz = e1x * e2y - e1y * e2x;
            const float xy = nx * nx + ny * ny;
            const float length = std::sqrt(xy + nz * nz);
            float inverse = 0.0F;
            if (static_cast<double>(length) >= 1.0e-20)
                inverse = 1.0F / length;
            return Float3{nx * inverse, ny * inverse, nz * inverse};
        };
        auto dot = [](const Float3& a, const Float3& b)
        {
            const float xy = a[0] * b[0] + a[1] * b[1];
            return xy + a[2] * b[2];
        };

        std::vector<FaceNeighbors> adjacent(mesh.faces.size());
        for (std::size_t face = 0; face != adjacent.size(); ++face)
            adjacent[face].fill(static_cast<std::uint32_t>(face));
        std::unordered_map<std::uint64_t, std::vector<EdgeUse>> edge_uses;
        edge_uses.reserve(mesh.faces.size() * 2U);
        for (std::size_t face = 0; face != mesh.faces.size(); ++face)
        {
            const auto& vertices = mesh.faces[face].vertices;
            for (std::uint32_t corner = 0; corner != 3U; ++corner)
            {
                const auto a = static_cast<std::uint32_t>(vertices[(corner + 1U) % 3U]);
                const auto b = static_cast<std::uint32_t>(vertices[(corner + 2U) % 3U]);
                if (a >= mesh.vertices.size() || b >= mesh.vertices.size())
                    throw std::runtime_error("fix-back face index is invalid");
                edge_uses[edge_key(a, b)].push_back({static_cast<std::uint32_t>(face), corner});
            }
        }
        for (const auto& [key, uses] : edge_uses)
        {
            (void)key;
            if (uses.size() != 2U)
                continue;
            adjacent[uses[0].face][uses[0].corner] = uses[1].face;
            adjacent[uses[1].face][uses[1].corner] = uses[0].face;
        }

        std::vector<std::uint8_t> moved(mesh.vertices.size(), 0U);
        constexpr float opposite_threshold = -0.86F;
        for (std::size_t pass = 0; pass != 5U; ++pass)
        {
            std::vector<Float3> positions(mesh.vertices.size());
            for (std::size_t vertex = 0; vertex != mesh.vertices.size(); ++vertex)
            {
                positions[vertex] = {static_cast<float>(mesh.vertices[vertex].position.x),
                                     static_cast<float>(mesh.vertices[vertex].position.y),
                                     static_cast<float>(mesh.vertices[vertex].position.z)};
            }
            std::vector<Float3> normals(mesh.faces.size());
            std::vector<Float3> neighbor_sum(mesh.vertices.size(), Float3{});
            std::vector<std::uint32_t> neighbor_count(mesh.vertices.size(), 0U);
            for (std::size_t face = 0; face != mesh.faces.size(); ++face)
            {
                const auto& indices = mesh.faces[face].vertices;
                const auto i0 = static_cast<std::uint32_t>(indices[0]);
                const auto i1 = static_cast<std::uint32_t>(indices[1]);
                const auto i2 = static_cast<std::uint32_t>(indices[2]);
                if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
                {
                    throw std::runtime_error("fix-back face index is invalid");
                }
                normals[face] = unit_normal(positions[i0], positions[i1], positions[i2]);
                const std::array<std::uint32_t, 3> vertices{i0, i1, i2};
                for (std::size_t corner = 0; corner != 3U; ++corner)
                {
                    const auto vertex = vertices[corner];
                    neighbor_count[vertex] += 2U;
                    for (const auto other : {vertices[(corner + 1U) % 3U], vertices[(corner + 2U) % 3U]})
                    {
                        neighbor_sum[vertex][0] += positions[other][0];
                        neighbor_sum[vertex][1] += positions[other][1];
                        neighbor_sum[vertex][2] += positions[other][2];
                    }
                }
            }

            for (std::size_t face = 0; face != mesh.faces.size(); ++face)
            {
                const auto& indices = mesh.faces[face].vertices;
                bool boundary = false;
                bool protected_vertex = false;
                for (std::size_t corner = 0; corner != 3U; ++corner)
                {
                    boundary = boundary || adjacent[face][corner] == face;
                    protected_vertex = protected_vertex || moved[indices[corner]] != 0U;
                }
                if (boundary || protected_vertex)
                    continue;
                const auto& normal = normals[face];
                if (normal[0] == 0.0F && normal[1] == 0.0F && normal[2] == 0.0F)
                {
                    continue;
                }
                bool surrounded_by_opposite_faces = true;
                for (const auto neighbor : adjacent[face])
                {
                    if (dot(normal, normals[neighbor]) > opposite_threshold)
                    {
                        surrounded_by_opposite_faces = false;
                        break;
                    }
                }
                if (!surrounded_by_opposite_faces)
                    continue;

                std::array<Float3, 3> triangle{positions[indices[0]], positions[indices[1]], positions[indices[2]]};
                std::size_t best_corner = 0U;
                float best_dot = std::numeric_limits<float>::infinity();
                std::array<Float3, 3> means{};
                for (std::size_t corner = 0; corner != 3U; ++corner)
                {
                    const auto vertex = indices[corner];
                    const float count = static_cast<float>(neighbor_count[vertex]);
                    means[corner] = {neighbor_sum[vertex][0] / count,
                                     neighbor_sum[vertex][1] / count,
                                     neighbor_sum[vertex][2] / count};
                    const auto original = triangle[corner];
                    triangle[corner] = means[corner];
                    const float candidate_dot = dot(unit_normal(triangle[0], triangle[1], triangle[2]), normal);
                    triangle[corner] = original;
                    if (candidate_dot < best_dot)
                    {
                        best_dot = candidate_dot;
                        best_corner = corner;
                    }
                }
                const auto vertex = indices[best_corner];
                mesh.vertices[vertex].position = {means[best_corner][0], means[best_corner][1], means[best_corner][2]};
                moved[vertex] = 1U;
                ++stats.corrected_vertices;
            }
            ++stats.passes;
        }
        recompute_normals(mesh);
        return stats;
    }

    RecoveredMeshTrimTrace recover_mesh_trim_mask(std::span<const Face> faces,
                                                  std::span<const RecoveredMeshTrimAttribute> attributes,
                                                  const RecoveredMeshTrimParameters& parameters)
    {
        RecoveredMeshTrimTrace trace;
        const std::size_t vertex_count = attributes.size();
        trace.initial_keep.assign(vertex_count, 0U);
        if (vertex_count == 0U)
        {
            trace.after_frontier = trace.initial_keep;
            trace.before_morphology_expanded = trace.initial_keep;
            trace.final_keep = trace.initial_keep;
            return trace;
        }
        if (!std::isfinite(parameters.softness))
            throw std::invalid_argument("mesh trim softness is not finite");
        if (vertex_count > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("mesh trim vertex count exceeds uint32");

        std::vector<std::vector<std::uint32_t>> adjacency(vertex_count);
        std::vector<std::uint32_t> face_occurrences(vertex_count, 0U);
        for (const Face& face : faces)
        {
            std::array<std::uint32_t, 3> vertex{};
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                if (face.vertices[corner] >= vertex_count)
                    throw std::invalid_argument("mesh trim face index is out of range");
                vertex[corner] = static_cast<std::uint32_t>(face.vertices[corner]);
                if (face_occurrences[vertex[corner]] == std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::overflow_error("mesh trim face degree overflows");
                }
                ++face_occurrences[vertex[corner]];
            }
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                const std::uint32_t source = vertex[corner];
                for (std::size_t other = 0U; other != 3U; ++other)
                {
                    const std::uint32_t target = vertex[other];
                    if (other == corner || source == target)
                        continue;
                    auto& list = adjacency[source];
                    if (std::find(list.begin(), list.end(), target) == list.end())
                        list.push_back(target);
                }
            }
        }

        const auto average = [](float numerator, float denominator) noexcept
        { return denominator > 0.0F ? numerator / denominator : numerator; };
        float maximum_support = -1.0F;
        for (const auto& attribute : attributes)
        {
            maximum_support = std::fmax(average(attribute[1], attribute[3]), maximum_support);
        }
        float initial_threshold = 2.099999904632568359375F;
        if (maximum_support <= initial_threshold)
            initial_threshold = 0.9900000095367431640625F * maximum_support;
        trace.thresholds.initial_support = initial_threshold;

        struct WeightedLevel
        {
            float level{};
            float weight{};
        };
        std::vector<WeightedLevel> levels;
        levels.reserve(vertex_count);
        float total_weight = 0.0F;
        for (const auto& attribute : attributes)
        {
            const float denominator = attribute[3];
            const float support = average(attribute[1], denominator);
            const float level = average(attribute[2], denominator);
            if (support >= initial_threshold * 0.5F)
            {
                levels.push_back({level, denominator});
                total_weight = total_weight + denominator;
            }
        }
        std::sort(levels.begin(),
                  levels.end(),
                  [](const WeightedLevel& left, const WeightedLevel& right) { return left.level < right.level; });
        if (!levels.empty() && total_weight > 0.0F)
        {
            float cumulative = 0.0F;
            for (const WeightedLevel& item : levels)
            {
                cumulative = cumulative + item.weight;
                if (trace.thresholds.lowest_level < 0.0F && cumulative >= 0.01F * total_weight)
                    trace.thresholds.lowest_level = item.level;
                if (trace.thresholds.low_level < 0.0F && cumulative >= 0.1F * total_weight)
                    trace.thresholds.low_level = item.level;
                if (trace.thresholds.median_level < 0.0F && cumulative >= 0.5F * total_weight)
                    trace.thresholds.median_level = item.level;
                if (trace.thresholds.high_level < 0.0F && cumulative >= 0.99F * total_weight)
                    trace.thresholds.high_level = item.level;
            }
            trace.thresholds.maximum_level = levels.back().level;
            trace.thresholds.frontier_level = std::fmax(trace.thresholds.lowest_level - 0.1F, 0.0F);
        }
        if (parameters.kind == 1)
        {
            const float factor = std::clamp(1.0F - parameters.softness, 0.0F, 1.0F);
            trace.thresholds.frontier_level = factor * trace.thresholds.frontier_level;
        }

        std::vector<std::uint32_t> frontier;
        frontier.reserve(vertex_count);
        for (std::size_t vertex = 0U; vertex != vertex_count; ++vertex)
        {
            const auto& attribute = attributes[vertex];
            if (average(attribute[1], attribute[3]) >= initial_threshold)
            {
                trace.initial_keep[vertex] = 1U;
                frontier.push_back(static_cast<std::uint32_t>(vertex));
            }
        }

        std::vector<std::uint8_t> keep = trace.initial_keep;
        while (!frontier.empty())
        {
            trace.frontier_inputs.push_back(frontier);
            std::vector<std::uint32_t> next;
            for (const std::uint32_t source : frontier)
            {
                const auto& attribute = attributes[source];
                const float level = average(attribute[2], attribute[3]);
                if (parameters.kind != 2 && level < trace.thresholds.frontier_level)
                    continue;
                for (const std::uint32_t neighbor : adjacency[source])
                {
                    if (keep[neighbor] == 0U)
                        next.push_back(neighbor);
                }
            }
            std::sort(next.begin(), next.end());
            next.erase(std::unique(next.begin(), next.end()), next.end());
            trace.frontier_outputs.push_back(next);
            for (const std::uint32_t vertex : next)
                keep[vertex] = 1U;
            frontier = std::move(next);
        }
        trace.after_frontier = keep;

        if (parameters.kind != 2)
        {
            trace.morphology_auxiliary_faces.reserve(vertex_count / 32U);
            for (std::size_t vertex = 0U; vertex != vertex_count; ++vertex)
            {
                if (face_occurrences[vertex] == adjacency[vertex].size())
                    continue;
                std::size_t previous = vertex;
                for (std::size_t step = 0U; step != 4U; ++step)
                {
                    const std::size_t added = keep.size();
                    keep.push_back(0U);
                    trace.morphology_auxiliary_faces.push_back({{previous, added, added}});
                    previous = added;
                }
            }
            trace.before_morphology_expanded = keep;
            const auto morphology = [](std::span<const std::uint8_t> input,
                                       std::span<const Face> base_faces,
                                       std::span<const Face> auxiliary_faces,
                                       bool dilate)
            {
                std::vector<std::uint8_t> output(input.begin(), input.end());
                const auto apply = [&](const Face& face)
                {
                    const bool any_marked =
                        input[face.vertices[0]] != 0U || input[face.vertices[1]] != 0U || input[face.vertices[2]] != 0U;
                    const bool all_marked =
                        input[face.vertices[0]] != 0U && input[face.vertices[1]] != 0U && input[face.vertices[2]] != 0U;
                    if ((dilate && any_marked) || (!dilate && !all_marked))
                    {
                        const std::uint8_t value = dilate ? 1U : 0U;
                        for (const std::size_t vertex : face.vertices)
                            output[vertex] = value;
                    }
                };
                for (const Face& face : base_faces)
                    apply(face);
                for (const Face& face : auxiliary_faces)
                    apply(face);
                return output;
            };
            for (std::size_t iteration = 0U; iteration != 3U; ++iteration)
                keep = morphology(keep, faces, trace.morphology_auxiliary_faces, true);
            for (std::size_t iteration = 0U; iteration != 6U; ++iteration)
                keep = morphology(keep, faces, trace.morphology_auxiliary_faces, false);
            for (std::size_t iteration = 0U; iteration != 3U; ++iteration)
                keep = morphology(keep, faces, trace.morphology_auxiliary_faces, true);
            keep.resize(vertex_count);
        }
        else
        {
            trace.before_morphology_expanded = keep;
        }
        trace.final_keep = std::move(keep);
        return trace;
    }

    void compact_mesh_recovered_trim(Mesh& mesh,
                                     std::vector<RecoveredMeshTrimAttribute>& attributes,
                                     std::span<const std::uint8_t> keep)
    {
        if (keep.size() != mesh.vertices.size() || attributes.size() != mesh.vertices.size())
        {
            throw std::invalid_argument("mesh trim compaction input sizes do not match");
        }
        std::vector<std::size_t> remap(mesh.vertices.size(), std::numeric_limits<std::size_t>::max());
        std::vector<Vertex> vertices;
        std::vector<RecoveredMeshTrimAttribute> compact_attributes;
        vertices.reserve(mesh.vertices.size());
        compact_attributes.reserve(attributes.size());
        for (std::size_t vertex = 0U; vertex != mesh.vertices.size(); ++vertex)
        {
            if (keep[vertex] == 0U)
                continue;
            remap[vertex] = vertices.size();
            vertices.push_back(mesh.vertices[vertex]);
            compact_attributes.push_back(attributes[vertex]);
        }
        std::vector<Face> faces;
        faces.reserve(mesh.faces.size());
        for (Face face : mesh.faces)
        {
            bool retained = true;
            for (std::size_t& vertex : face.vertices)
            {
                retained = retained && vertex < keep.size() && keep[vertex] != 0U;
                if (retained)
                    vertex = remap[vertex];
            }
            if (retained)
                faces.push_back(face);
        }
        mesh.vertices = std::move(vertices);
        mesh.faces = std::move(faces);
        attributes = std::move(compact_attributes);
        recompute_normals(mesh);
    }

    namespace
    {

        [[nodiscard]] float color_add_f32(float left, float right) noexcept
        {
            volatile float result = left + right;
            return result;
        }

        [[nodiscard]] float color_subtract_f32(float left, float right) noexcept
        {
            volatile float result = left - right;
            return result;
        }

        [[nodiscard]] float color_multiply_f32(float left, float right) noexcept
        {
            volatile float result = left * right;
            return result;
        }

        [[nodiscard]] float color_divide_f32(float numerator, float denominator) noexcept
        {
            volatile float result = numerator / denominator;
            return result;
        }

        [[nodiscard]] double color_add_f64(double left, double right) noexcept
        {
            volatile double result = left + right;
            return result;
        }

        [[nodiscard]] double color_multiply_f64(double left, double right) noexcept
        {
            volatile double result = left * right;
            return result;
        }

        [[nodiscard]] double color_subtract_f64(double left, double right) noexcept
        {
            volatile double result = left - right;
            return result;
        }

        template <class T>
        void
        write_vertex_color_camera_value(RecoveredVertexColorVulkanCameraPayload& payload, std::size_t offset, T value)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            if (offset > payload.size() || payload.size() - offset < sizeof(T))
                throw std::logic_error("Vulkan camera payload write is out of range");
            std::memcpy(payload.data() + offset, &value, sizeof(value));
        }

        [[nodiscard]] float vertex_color_float(double value) noexcept
        {
            volatile float result = static_cast<float>(value);
            return result;
        }

        void pack_vertex_color_std140_matrix3(RecoveredVertexColorVulkanCameraPayload& payload,
                                              std::size_t offset,
                                              const std::array<double, 9>& matrix)
        {
            for (std::size_t column = 0U; column != 3U; ++column)
            {
                for (std::size_t row = 0U; row != 3U; ++row)
                {
                    write_vertex_color_camera_value(
                        payload, offset + 16U * column + 4U * row, vertex_color_float(matrix[3U * row + column]));
                }
                write_vertex_color_camera_value(payload, offset + 16U * column + 12U, 0.0F);
            }
        }

        [[nodiscard]] std::array<double, 9> invert_vertex_color_matrix3_target(const std::array<double, 9>& matrix)
        {
            const double minor_00 =
                color_subtract_f64(color_multiply_f64(matrix[4], matrix[8]), color_multiply_f64(matrix[5], matrix[7]));
            const double minor_10 =
                color_subtract_f64(color_multiply_f64(matrix[5], matrix[6]), color_multiply_f64(matrix[3], matrix[8]));
            const double minor_20 =
                color_subtract_f64(color_multiply_f64(matrix[3], matrix[7]), color_multiply_f64(matrix[4], matrix[6]));
            const double determinant = color_add_f64(
                color_subtract_f64(color_multiply_f64(matrix[0], minor_00),
                                   color_multiply_f64(matrix[1],
                                                      color_subtract_f64(color_multiply_f64(matrix[3], matrix[8]),
                                                                         color_multiply_f64(matrix[5], matrix[6])))),
                color_multiply_f64(matrix[2], minor_20));
            if (!std::isfinite(determinant) || determinant == 0.0)
                throw std::invalid_argument("Vulkan vertex-color image matrix is singular");
            const double inverse_determinant = 1.0 / determinant;
            return {
                color_multiply_f64(minor_00, inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[2], matrix[7]),
                                                      color_multiply_f64(matrix[1], matrix[8])),
                                   inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[1], matrix[5]),
                                                      color_multiply_f64(matrix[2], matrix[4])),
                                   inverse_determinant),
                color_multiply_f64(minor_10, inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[0], matrix[8]),
                                                      color_multiply_f64(matrix[2], matrix[6])),
                                   inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[2], matrix[3]),
                                                      color_multiply_f64(matrix[0], matrix[5])),
                                   inverse_determinant),
                color_multiply_f64(minor_20, inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[1], matrix[6]),
                                                      color_multiply_f64(matrix[0], matrix[7])),
                                   inverse_determinant),
                color_multiply_f64(color_subtract_f64(color_multiply_f64(matrix[0], matrix[4]),
                                                      color_multiply_f64(matrix[1], matrix[3])),
                                   inverse_determinant),
            };
        }

        [[nodiscard]] double vertex_color_maximum_radius_squared(const metalign::CameraModel& model)
        {
            // sub_2C3E930's ninth Brown value is the smallest positive root of the
            // radial mapping derivative in x=r^2:
            //   1 + 3*k1*x + 5*k2*x^2 + 7*k3*x^3 + 9*k4*x^4.
            // South's captured source value 1.132872862450893 and its packed float
            // are reproduced by this same target root implementation.
            const std::vector<double> roots = metalign::polynomial_real_roots_target(
                {1.0, 3.0 * model.k1, 5.0 * model.k2, 7.0 * model.k3, 9.0 * model.k4});
            double result = std::numeric_limits<double>::max();
            for (const double root : roots)
            {
                if (std::isfinite(root) && root > 0.0 && root < result)
                    result = root;
            }
            return result;
        }

        [[nodiscard]] std::array<float, 3> sample_recovered_uint8_mipmaps(
            std::span<const RecoveredVertexColorImageLevel> mipmaps, double x, double y, float footprint)
        {
            constexpr float first_level_threshold = std::bit_cast<float>(std::uint32_t{0x3F2D51ADU});
            constexpr double clamp_epsilon = 1.0e-9;

            const auto& base = mipmaps.front();
            const double clamped_y =
                std::fmax(std::fmin(y, static_cast<double>(base.height) - clamp_epsilon), clamp_epsilon);
            const double clamped_x =
                std::fmax(std::fmin(x, static_cast<double>(base.width) - clamp_epsilon), clamp_epsilon);

            std::size_t current_level = 0U;
            float scale = 1.0F;
            float current_level_weight = 1.0F;
            bool blend_previous = false;
            if (mipmaps.size() > 1U && footprint > first_level_threshold)
            {
                current_level = 1U;
                float threshold = first_level_threshold;
                while (true)
                {
                    threshold = color_add_f32(threshold, threshold);
                    scale = color_add_f32(scale, scale);
                    if (current_level == mipmaps.size() - 1U || footprint <= threshold)
                    {
                        break;
                    }
                    ++current_level;
                }
                const float ratio = color_divide_f32(threshold, footprint);
                current_level_weight = color_subtract_f32(1.0F, std::log2(ratio));
                blend_previous = true;
            }

            std::array<float, 3> sums{};
            float sum_weight = 0.0F;
            for (std::size_t pass = 0U; pass != (blend_previous ? 2U : 1U); ++pass)
            {
                const auto& level = mipmaps[current_level];
                const float integer_scale = static_cast<float>(static_cast<std::int32_t>(scale));
                const float sample_x = color_add_f32(static_cast<float>(clamped_x / integer_scale), 0.5F);
                const float sample_y = color_add_f32(static_cast<float>(clamped_y / integer_scale), 0.5F);
                const auto center_x = static_cast<std::size_t>(sample_x);
                const auto center_y = static_cast<std::size_t>(sample_y);
                const float level_weight =
                    pass == 0U ? current_level_weight : color_subtract_f32(1.0F, current_level_weight);

                for (std::size_t quadrant = 0U; quadrant != 4U; ++quadrant)
                {
                    const std::size_t dx = quadrant & 1U;
                    const std::size_t dy = quadrant >> 1U;
                    if (dx > center_x || dy > center_y || center_x >= dx + level.width || center_y >= dy + level.height)
                    {
                        continue;
                    }
                    const float local_x = color_subtract_f32(color_subtract_f32(sample_x, static_cast<float>(center_x)),
                                                             static_cast<float>(dx));
                    const float weighted_x = color_multiply_f32(local_x, level_weight);
                    const float local_y = color_subtract_f32(color_subtract_f32(sample_y, static_cast<float>(center_y)),
                                                             static_cast<float>(dy));
                    const float weight = std::fabs(color_multiply_f32(weighted_x, local_y));
                    const std::size_t pixel = (center_y - dy) * level.row_elements + (center_x - dx) * level.channels;
                    for (std::size_t channel = 0U; channel != 3U; ++channel)
                    {
                        const float term =
                            color_multiply_f32(static_cast<float>(level.pixels[pixel + channel]), weight);
                        sums[channel] = color_add_f32(term, sums[channel]);
                    }
                    sum_weight = color_add_f32(sum_weight, weight);
                }

                if (pass == 0U && blend_previous)
                {
                    --current_level;
                    scale = static_cast<float>(static_cast<std::int32_t>(color_multiply_f32(scale, 0.5F)));
                }
            }
            if (sum_weight != 0.0F)
            {
                for (float& sum : sums)
                    sum = color_divide_f32(sum, sum_weight);
            }
            return sums;
        }

    } // namespace

    RecoveredVertexColorVulkanCameraPayload
    pack_recovered_vertex_color_vulkan_camera(const RecoveredVertexColorVulkanCameraSource& source)
    {
        if (source.width == 0U || source.height == 0U ||
            source.width > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            source.height > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        {
            throw std::invalid_argument("Vulkan vertex-color camera dimensions are outside int32");
        }
        const auto require_finite = [](auto&& values, const char* label)
        {
            for (const double value : values)
            {
                if (!std::isfinite(value))
                    throw std::invalid_argument(std::string("Vulkan vertex-color ") + label +
                                                " contains a non-finite value");
            }
        };
        require_finite(source.camera_to_world, "camera_to_world");
        require_finite(source.auxiliary_transform, "auxiliary transform");
        require_finite(source.image_matrix, "image matrix");
        const std::array<double, 15> model_values{source.model.f,
                                                  source.model.cx_offset,
                                                  source.model.cy_offset,
                                                  source.model.b1,
                                                  source.model.b2,
                                                  source.model.k1,
                                                  source.model.k2,
                                                  source.model.k3,
                                                  source.model.k4,
                                                  source.model.p1,
                                                  source.model.p2,
                                                  source.model.p3,
                                                  source.model.p4,
                                                  source.model.cx,
                                                  source.model.cy};
        require_finite(model_values, "calibration");
        if (source.camera_to_world[15] == 0.0)
            throw std::invalid_argument("Vulkan vertex-color camera has zero homogeneous scale");

        RecoveredVertexColorVulkanCameraPayload payload{};

        std::array<double, 9> normalized_rotation{};
        double squared_norm_sum = 0.0;
        for (std::size_t column = 0U; column != 3U; ++column)
        {
            const double x = source.camera_to_world[column];
            const double y = source.camera_to_world[4U + column];
            const double z = source.camera_to_world[8U + column];
            const double squared_norm = color_add_f64(color_add_f64(color_multiply_f64(x, x), color_multiply_f64(y, y)),
                                                      color_multiply_f64(z, z));
            if (!(squared_norm > 0.0) || !std::isfinite(squared_norm))
                throw std::invalid_argument("Vulkan vertex-color camera basis is degenerate");
            const double length = std::sqrt(squared_norm);
            const double inverse_length = 1.0 / length;
            normalized_rotation[column] = color_multiply_f64(x, inverse_length);
            normalized_rotation[3U + column] = color_multiply_f64(y, inverse_length);
            normalized_rotation[6U + column] = color_multiply_f64(z, inverse_length);
            squared_norm_sum = color_add_f64(squared_norm_sum, squared_norm);
        }
        pack_vertex_color_std140_matrix3(payload, 0U, normalized_rotation);
        const double homogeneous_scale = source.camera_to_world[15];
        write_vertex_color_camera_value(
            payload, 48U, vertex_color_float(source.camera_to_world[3] / homogeneous_scale));
        write_vertex_color_camera_value(
            payload, 52U, vertex_color_float(source.camera_to_world[7] / homogeneous_scale));
        write_vertex_color_camera_value(
            payload, 56U, vertex_color_float(source.camera_to_world[11] / homogeneous_scale));
        write_vertex_color_camera_value(payload, 60U, vertex_color_float(std::sqrt(squared_norm_sum / 3.0)));

        const double maximum_radius_squared = vertex_color_maximum_radius_squared(source.model);
        const std::array<double, 9> brown{source.model.k1,
                                          source.model.k2,
                                          source.model.k3,
                                          source.model.k4,
                                          source.model.p1,
                                          source.model.p2,
                                          source.model.p3,
                                          source.model.p4,
                                          maximum_radius_squared};
        for (std::size_t index = 0U; index != brown.size(); ++index)
        {
            write_vertex_color_camera_value(payload, 64U + 4U * index, vertex_color_float(brown[index]));
        }

        // The target's inactive perspective-camera RPC array at [116, 500) is
        // copied from uninitialised stack storage.  payload{} intentionally makes
        // those bytes deterministic.  The written suffix is still reproduced.
        write_vertex_color_camera_value(payload, 500U, std::uint32_t{0});
        write_vertex_color_camera_value(payload, 504U, vertex_color_float(std::numeric_limits<double>::max()));
        write_vertex_color_camera_value(payload, 508U, vertex_color_float(std::numeric_limits<double>::max()));
        write_vertex_color_camera_value(payload, 512U, vertex_color_float(-std::numeric_limits<double>::max()));
        write_vertex_color_camera_value(payload, 516U, vertex_color_float(-std::numeric_limits<double>::max()));
        write_vertex_color_camera_value(payload, 520U, std::int32_t{1});
        write_vertex_color_camera_value(payload, 524U, static_cast<std::int32_t>(source.width));
        write_vertex_color_camera_value(payload, 528U, static_cast<std::int32_t>(source.height));
        write_vertex_color_camera_value(payload, 532U, vertex_color_float(source.model.f));
        write_vertex_color_camera_value(payload, 536U, vertex_color_float(source.model.cx_offset));
        write_vertex_color_camera_value(payload, 540U, vertex_color_float(source.model.cy_offset));
        write_vertex_color_camera_value(payload, 544U, vertex_color_float(source.model.b1));
        write_vertex_color_camera_value(payload, 548U, vertex_color_float(source.model.b2));
        write_vertex_color_camera_value(payload, 552U, std::int32_t{1});
        write_vertex_color_camera_value(payload, 556U, std::int32_t{0});

        for (std::size_t index = 0U; index != 3U; ++index)
        {
            write_vertex_color_camera_value(
                payload, 560U + 4U * index, vertex_color_float(source.auxiliary_transform[index]));
            write_vertex_color_camera_value(
                payload, 576U + 4U * index, vertex_color_float(source.auxiliary_transform[12U + index]));
        }
        write_vertex_color_camera_value(payload, 572U, 0.0F);
        write_vertex_color_camera_value(payload, 588U, 0.0F);
        pack_vertex_color_std140_matrix3(payload, 592U, source.image_matrix);
        pack_vertex_color_std140_matrix3(payload, 640U, invert_vertex_color_matrix3_target(source.image_matrix));
        return payload;
    }

    RecoveredVertexColorVulkanCameraSource
    make_recovered_vertex_color_vulkan_camera_source(const Camera& camera,
                                                     const std::array<double, 16>& auxiliary_transform,
                                                     const std::array<double, 9>& image_matrix)
    {
        if (!camera.aligned || camera.image.width == 0U || camera.image.height == 0U)
            throw std::invalid_argument("Vulkan vertex-color source camera is not an aligned image");
        RecoveredVertexColorVulkanCameraSource source;
        source.camera_to_world = {camera.pose.rotation(0, 0),
                                  camera.pose.rotation(1, 0),
                                  camera.pose.rotation(2, 0),
                                  camera.center.x,
                                  camera.pose.rotation(0, 1),
                                  camera.pose.rotation(1, 1),
                                  camera.pose.rotation(2, 1),
                                  camera.center.y,
                                  camera.pose.rotation(0, 2),
                                  camera.pose.rotation(1, 2),
                                  camera.pose.rotation(2, 2),
                                  camera.center.z,
                                  0.0,
                                  0.0,
                                  0.0,
                                  1.0};
        source.model = camera.model;
        source.width = camera.image.width;
        source.height = camera.image.height;
        source.auxiliary_transform = auxiliary_transform;
        source.image_matrix = image_matrix;
        return source;
    }

    std::vector<float>
    compute_recovered_vertex_color_geometry_denominator(std::span<const RecoveredVertexColorPosition> positions,
                                                        std::span<const RecoveredVertexColorFace> faces)
    {
        std::vector<float> denominator(positions.size(), 0.0F);
        for (const auto& face : faces)
        {
            for (const std::uint32_t index : face)
            {
                if (static_cast<std::size_t>(index) >= positions.size())
                {
                    throw std::invalid_argument("mesh color face references an out-of-range vertex");
                }
            }

            std::array<std::array<double, 3>, 3> point{};
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    point[corner][axis] = static_cast<double>(positions[face[corner]][axis]);
                }
            }
            const double x10 = point[1][0] - point[0][0];
            const double y10 = point[1][1] - point[0][1];
            const double z10 = point[1][2] - point[0][2];
            const double x20 = point[2][0] - point[0][0];
            const double y20 = point[2][1] - point[0][1];
            const double z20 = point[2][2] - point[0][2];
            const double cross0 = x20 * y10 - y20 * x10;
            const double cross1 = y20 * z10 - z20 * y10;
            const double cross2 = x10 * z20 - x20 * z10;
            const float area = static_cast<float>(std::sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2) * 0.5);
            for (const std::uint32_t index : face)
            {
                denominator[index] = denominator[index] + area;
            }
        }
        return denominator;
    }

    void project_recovered_vertex_color_camera_cpu(std::span<const RecoveredVertexColorPosition> positions,
                                                   std::span<const std::uint32_t> selected_vertices,
                                                   const RecoveredVertexColorProjectionCamera& camera,
                                                   const RecoveredVertexColorDepthImage& depth,
                                                   RecoveredVertexColorCameraScratch& scratch,
                                                   float invalid_x,
                                                   float invalid_y)
    {
        if (camera.width == 0U || camera.height == 0U || depth.width != camera.width || depth.height != camera.height ||
            depth.row_elements < depth.width ||
            depth.height > std::numeric_limits<std::size_t>::max() / depth.row_elements ||
            depth.pixels.size() != depth.height * depth.row_elements)
        {
            throw std::invalid_argument("mesh color projection camera/depth layout is invalid");
        }
        const std::size_t vertices = positions.size();
        if (scratch.geometry.size() != 3U * vertices || scratch.projected_xy.size() != 2U * vertices ||
            scratch.weighted_xy.size() != 2U * vertices)
        {
            throw std::invalid_argument("mesh color camera scratch sizes do not match positions");
        }
        for (const std::uint32_t vertex_u32 : selected_vertices)
        {
            const std::size_t vertex = vertex_u32;
            if (vertex >= vertices)
                throw std::invalid_argument("mesh color selected vertex index is invalid");
            const std::size_t coordinate_offset = 2U * vertex;
            scratch.projected_xy[coordinate_offset] = invalid_x;
            scratch.projected_xy[coordinate_offset + 1U] = invalid_y;

            const auto& position = positions[vertex];
            std::array<double, 3> local{};
            for (std::size_t row = 0U; row != 3U; ++row)
            {
                const std::size_t offset = 4U * row;
                local[row] = color_add_f64(
                    color_add_f64(color_add_f64(color_multiply_f64(camera.world_to_camera[offset], position[0]),
                                                color_multiply_f64(camera.world_to_camera[offset + 1U], position[1])),
                                  color_multiply_f64(camera.world_to_camera[offset + 2U], position[2])),
                    color_multiply_f64(camera.world_to_camera[offset + 3U], 1.0));
            }
            // Perspective type 1 in sub_27CC080 returns false for z <= 0.
            if (!(local[2] > 0.0))
                continue;
            const metalign::Vec2 projected = metalign::project_local(camera.model, {local[0], local[1], local[2]});
            if (!(projected.x >= 0.0 && projected.y >= 0.0 && projected.x <= static_cast<double>(camera.width) &&
                  projected.y <= static_cast<double>(camera.height)))
            {
                continue;
            }
            const auto pixel_x = static_cast<std::size_t>(projected.x);
            const auto pixel_y = static_cast<std::size_t>(projected.y);
            if (pixel_x >= depth.width || pixel_y >= depth.height)
                continue;
            const float visible_depth = depth.pixels[pixel_y * depth.row_elements + pixel_x];
            const double tolerance = static_cast<double>(std::fabs(visible_depth)) * 0.001;
            if (visible_depth == 0.0F || local[2] > static_cast<double>(visible_depth) + tolerance)
            {
                continue;
            }
            scratch.projected_xy[coordinate_offset] = static_cast<float>(projected.x);
            scratch.projected_xy[coordinate_offset + 1U] = static_cast<float>(projected.y);
        }
    }

    void accumulate_recovered_vertex_color_geometry_cpu(std::span<const RecoveredVertexColorPosition> positions,
                                                        std::span<const RecoveredVertexColorFace> faces,
                                                        std::span<const std::uint32_t> selected_faces,
                                                        const std::array<float, 3>& camera_center,
                                                        RecoveredVertexColorCameraScratch& scratch,
                                                        float invalid_x,
                                                        float invalid_y,
                                                        std::vector<std::uint8_t>* selected_face_acceptance)
    {
        if (positions.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::overflow_error("mesh color vertex index exceeds uint32");
        }
        const std::size_t vertices = positions.size();
        if (scratch.geometry.size() != 3U * vertices || scratch.projected_xy.size() != 2U * vertices ||
            scratch.weighted_xy.size() != 2U * vertices)
        {
            throw std::invalid_argument("mesh color camera scratch sizes do not match positions");
        }
        constexpr float half = 0.5F;
        constexpr float view_epsilon = 1.0e-20F;
        if (selected_face_acceptance != nullptr)
            selected_face_acceptance->assign(selected_faces.size(), 0U);

        for (const std::uint32_t face_index_u32 : selected_faces)
        {
            const std::size_t face_index = face_index_u32;
            if (face_index >= faces.size())
                throw std::invalid_argument("mesh color selected face index is invalid");
            for (const std::uint32_t vertex : faces[face_index])
            {
                if (vertex >= vertices)
                    throw std::invalid_argument("mesh color face vertex index is invalid");
            }
        }

#pragma omp parallel for schedule(guided, 1)
        for (std::ptrdiff_t selected_ordinal = 0; selected_ordinal < static_cast<std::ptrdiff_t>(selected_faces.size());
             ++selected_ordinal)
        {
            const std::uint32_t face_index_u32 = selected_faces[static_cast<std::size_t>(selected_ordinal)];
            const std::size_t face_index = face_index_u32;
            const auto& face = faces[face_index];

            std::array<std::array<float, 2>, 3> image{};
            std::array<std::array<float, 3>, 3> world{};
            bool visible = true;
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                const std::size_t vertex = face[corner];
                const std::size_t coordinate_offset = 2U * vertex;
                image[corner][0] = scratch.projected_xy[coordinate_offset];
                image[corner][1] = scratch.projected_xy[coordinate_offset + 1U];
                if (image[corner][0] == invalid_x && image[corner][1] == invalid_y)
                {
                    visible = false;
                    break;
                }
                world[corner][0] = positions[vertex][0];
                world[corner][1] = positions[vertex][1];
                world[corner][2] = positions[vertex][2];
            }
            if (!visible)
                continue;

            const float p2y_p0y = color_subtract_f32(world[2][1], world[0][1]);
            const float p1x_p0x = color_subtract_f32(world[1][0], world[0][0]);
            const float mixed_xy = color_multiply_f32(p2y_p0y, p1x_p0x);
            const float normal_z =
                color_multiply_f32(color_subtract_f32(color_multiply_f32(color_subtract_f32(world[2][0], world[0][0]),
                                                                         color_subtract_f32(world[1][1], world[0][1])),
                                                      mixed_xy),
                                   half);
            const float normal_y = color_multiply_f32(
                color_subtract_f32(color_multiply_f32(p1x_p0x, color_subtract_f32(world[2][2], world[0][2])),
                                   color_multiply_f32(color_subtract_f32(world[2][0], world[0][0]),
                                                      color_subtract_f32(world[1][2], world[0][2]))),
                half);
            const float normal_x = color_multiply_f32(
                color_subtract_f32(color_multiply_f32(p2y_p0y, color_subtract_f32(world[1][2], world[0][2])),
                                   color_multiply_f32(color_subtract_f32(world[2][2], world[0][2]),
                                                      color_subtract_f32(world[1][1], world[0][1]))),
                half);
            const float normal_length_squared = color_add_f32(
                color_add_f32(color_multiply_f32(normal_y, normal_y), color_multiply_f32(normal_x, normal_x)),
                color_multiply_f32(normal_z, normal_z));
            const float world_area = std::sqrt(normal_length_squared);

            const float projected_cross =
                color_subtract_f32(color_multiply_f32(color_subtract_f32(image[2][0], image[0][0]),
                                                      color_subtract_f32(image[1][1], image[0][1])),
                                   color_multiply_f32(color_subtract_f32(image[2][1], image[0][1]),
                                                      color_subtract_f32(image[1][0], image[0][0])));
            const float projected_area = std::fmax(color_multiply_f32(projected_cross, half), 0.0F);

            std::array<float, 3> projected_weights{};
            std::array<float, 3> geometry_weights{};
            bool front_facing = true;
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                const float dx = color_subtract_f32(world[corner][0], camera_center[0]);
                const float dy = color_subtract_f32(world[corner][1], camera_center[1]);
                const float dz = color_subtract_f32(world[corner][2], camera_center[2]);
                const float distance_squared = color_add_f32(
                    color_add_f32(color_multiply_f32(dx, dx), color_multiply_f32(dy, dy)), color_multiply_f32(dz, dz));
                const float distance = std::sqrt(distance_squared);
                const float inverse_distance = distance >= view_epsilon ? color_divide_f32(1.0F, distance) : 0.0F;
                const float numerator =
                    color_add_f32(color_add_f32(color_multiply_f32(normal_x, color_multiply_f32(dx, inverse_distance)),
                                                color_multiply_f32(color_multiply_f32(dy, inverse_distance), normal_y)),
                                  color_multiply_f32(color_multiply_f32(dz, inverse_distance), normal_z));
                const float cosine = color_divide_f32(numerator, world_area);
                if (!(cosine > 0.0F))
                {
                    front_facing = false;
                    break;
                }
                projected_weights[corner] = color_multiply_f32(projected_area, cosine);
                geometry_weights[corner] = color_multiply_f32(
                    std::exp(color_subtract_f32(color_multiply_f32(cosine, 3.0F), 3.0F)), world_area);
            }
            if (!front_facing)
                continue;
            if (selected_face_acceptance != nullptr)
            {
                (*selected_face_acceptance)[static_cast<std::size_t>(selected_ordinal)] = 1U;
            }

            const float common_x =
                color_multiply_f32(color_add_f32(color_add_f32(image[1][0], image[0][0]), image[2][0]), 7.0F);
            const float common_y =
                color_multiply_f32(color_add_f32(color_add_f32(image[1][1], image[0][1]), image[2][1]), 7.0F);
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                const std::size_t vertex = face[corner];
                const float scaled_weight = color_divide_f32(projected_weights[corner], 36.0F);
                const float weighted_x = color_multiply_f32(
                    color_add_f32(color_multiply_f32(image[corner][0], 15.0F), common_x), scaled_weight);
                const float weighted_y = color_multiply_f32(
                    color_add_f32(color_multiply_f32(image[corner][1], 15.0F), common_y), scaled_weight);
                const std::size_t geometry_offset = 3U * vertex;
                const std::size_t coordinate_offset = 2U * vertex;
#pragma omp atomic update
                scratch.geometry[geometry_offset] += projected_weights[corner];
#pragma omp atomic update
                scratch.geometry[geometry_offset + 1U] += geometry_weights[corner];
#pragma omp atomic update
                scratch.weighted_xy[coordinate_offset] += weighted_x;
#pragma omp atomic update
                scratch.weighted_xy[coordinate_offset + 1U] += weighted_y;
            }
        }
    }

    void accumulate_recovered_vertex_color_camera_cpu(std::span<const std::uint32_t> selected_vertices,
                                                      std::span<const RecoveredVertexColorImageLevel> mipmaps,
                                                      RecoveredVertexColorCameraScratch& scratch,
                                                      std::vector<RecoveredVertexColorAccumulator>& accumulator,
                                                      float invalid_x,
                                                      float invalid_y)
    {
        if (mipmaps.empty())
            throw std::invalid_argument("mesh color mipmap pyramid is empty");
        if (accumulator.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::overflow_error("mesh color vertex index exceeds uint32");
        }
        const std::size_t vertices = accumulator.size();
        if (scratch.geometry.size() != 3U * vertices || scratch.projected_xy.size() != 2U * vertices ||
            scratch.weighted_xy.size() != 2U * vertices)
        {
            throw std::invalid_argument("mesh color camera scratch sizes do not match vertices");
        }
        for (std::size_t level_index = 0U; level_index != mipmaps.size(); ++level_index)
        {
            const auto& level = mipmaps[level_index];
            if (level.width == 0U || level.height == 0U || level.channels != 3U ||
                level.row_elements < level.width * level.channels ||
                level.height > std::numeric_limits<std::size_t>::max() / level.row_elements ||
                level.pixels.size() != level.height * level.row_elements)
            {
                throw std::invalid_argument("mesh color uint8 mipmap layout is invalid");
            }
            if (level_index != 0U)
            {
                const auto& previous = mipmaps[level_index - 1U];
                if (level.width != (previous.width + 1U) / 2U || level.height != (previous.height + 1U) / 2U)
                {
                    throw std::invalid_argument("mesh color mipmap dimensions are not target halves");
                }
            }
        }
        constexpr float footprint_denominator = std::bit_cast<float>(std::uint32_t{0x4116CBE4U});
        constexpr float minimum_normal = std::bit_cast<float>(std::uint32_t{0x00800000U});

        for (const std::uint32_t vertex_u32 : selected_vertices)
        {
            const std::size_t vertex = vertex_u32;
            if (vertex >= vertices)
                throw std::invalid_argument("mesh color selected vertex index is invalid");
            const std::size_t geometry_offset = 3U * vertex;
            const std::size_t coordinate_offset = 2U * vertex;
            const float projected_x = scratch.projected_xy[coordinate_offset];
            const float projected_y = scratch.projected_xy[coordinate_offset + 1U];
            if (projected_x == invalid_x && projected_y == invalid_y)
            {
                scratch.geometry[geometry_offset] = 0.0F;
                scratch.geometry[geometry_offset + 1U] = 0.0F;
                continue;
            }

            const float geometry_weight = scratch.geometry[geometry_offset];
            float contribution_weight = minimum_normal;
            double sample_x = static_cast<double>(projected_x);
            double sample_y = static_cast<double>(projected_y);
            if (geometry_weight != 0.0F)
            {
                const float denominator = scratch.geometry[geometry_offset + 2U];
                contribution_weight =
                    denominator > 0.0F ? color_divide_f32(scratch.geometry[geometry_offset + 1U], denominator) : 0.0F;
                sample_x =
                    static_cast<double>(color_divide_f32(scratch.weighted_xy[coordinate_offset], geometry_weight));
                sample_y =
                    static_cast<double>(color_divide_f32(scratch.weighted_xy[coordinate_offset + 1U], geometry_weight));
            }
            const float footprint = std::sqrt(color_divide_f32(geometry_weight, footprint_denominator));
            const auto sample = sample_recovered_uint8_mipmaps(mipmaps, sample_x, sample_y, footprint);
            for (std::size_t channel = 0U; channel != 3U; ++channel)
            {
                const float term = color_multiply_f32(contribution_weight, sample[channel]);
                accumulator[vertex][channel] = color_add_f32(term, accumulator[vertex][channel]);
            }
            accumulator[vertex][3] = color_add_f32(contribution_weight, accumulator[vertex][3]);

            scratch.geometry[geometry_offset] = 0.0F;
            scratch.geometry[geometry_offset + 1U] = 0.0F;
            scratch.projected_xy[coordinate_offset] = invalid_x;
            scratch.projected_xy[coordinate_offset + 1U] = invalid_y;
            scratch.weighted_xy[coordinate_offset] = 0.0F;
            scratch.weighted_xy[coordinate_offset + 1U] = 0.0F;
        }
    }

    void assign_recovered_vertex_confidence(Mesh& mesh, std::span<const RecoveredMeshTrimAttribute> attributes)
    {
        if (attributes.size() != mesh.vertices.size())
        {
            throw std::invalid_argument("mesh confidence attribute count does not match vertices");
        }
        for (std::size_t index = 0U; index != attributes.size(); ++index)
        {
            const float numerator = attributes[index][1];
            const float denominator = attributes[index][3];
            if (!std::isfinite(numerator) || !std::isfinite(denominator))
            {
                throw std::invalid_argument("mesh confidence attribute is non-finite");
            }
            volatile float confidence = numerator;
            if (denominator > 0.0F)
                confidence = confidence / denominator;
            mesh.vertices[index].confidence = confidence;
        }
    }

    void assign_recovered_vertex_colors_from_accumulator(Mesh& mesh,
                                                         std::span<const RecoveredVertexColorAccumulator> accumulator)
    {
        if (accumulator.size() != mesh.vertices.size())
        {
            throw std::invalid_argument("mesh color accumulator count does not match vertices");
        }
        for (std::size_t index = 0U; index != accumulator.size(); ++index)
        {
            const float weight = accumulator[index][3];
            if (!std::isfinite(weight))
            {
                throw std::invalid_argument("mesh color accumulator weight is non-finite");
            }
            volatile float reciprocal = weight;
            if (weight != 0.0F)
                reciprocal = 1.0F / weight;
            for (std::size_t channel = 0U; channel != 3U; ++channel)
            {
                const float sum = accumulator[index][channel];
                if (!std::isfinite(sum))
                {
                    throw std::invalid_argument("mesh color accumulator sum is non-finite");
                }
                volatile float normalized = sum * reciprocal;
                if (!std::isfinite(normalized) || normalized < 0.0F || normalized > 255.0F)
                {
                    throw std::invalid_argument("normalized mesh color is outside uint8 range");
                }
                mesh.vertices[index].color[channel] = static_cast<std::uint8_t>(static_cast<std::int32_t>(normalized));
            }
        }
    }

    void assign_recovered_vertex_colors_from_normalized_accumulator(
        Mesh& mesh, std::span<const RecoveredVertexColorAccumulator> accumulator)
    {
        if (accumulator.size() != mesh.vertices.size())
        {
            throw std::invalid_argument("normalized mesh color accumulator count does not match vertices");
        }
        for (std::size_t index = 0U; index != accumulator.size(); ++index)
        {
            const float weight = accumulator[index][3];
            if (!std::isfinite(weight))
            {
                throw std::invalid_argument("normalized mesh color accumulator weight is non-finite");
            }
            volatile float reciprocal = weight;
            if (weight != 0.0F)
                reciprocal = 1.0F / weight;
            for (std::size_t channel = 0U; channel != 3U; ++channel)
            {
                const float sum = accumulator[index][channel];
                if (!std::isfinite(sum))
                {
                    throw std::invalid_argument("normalized mesh color accumulator sum is non-finite");
                }
                volatile float normalized = sum * reciprocal;
                volatile float scaled = normalized * 255.0F;
                if (!std::isfinite(scaled) || scaled < static_cast<float>(std::numeric_limits<std::int32_t>::min()) ||
                    scaled > static_cast<float>(std::numeric_limits<std::int32_t>::max()))
                {
                    throw std::invalid_argument("normalized Vulkan mesh color cannot convert to int32");
                }
                const std::int32_t converted = static_cast<std::int32_t>(scaled);
                if (converted < 0 || converted > 255)
                {
                    throw std::invalid_argument("normalized Vulkan mesh color is outside uint8 range");
                }
                mesh.vertices[index].color[channel] = static_cast<std::uint8_t>(converted);
            }
        }
    }

    std::vector<std::int32_t>
    extrapolate_recovered_vertex_color_accumulator(const Mesh& mesh,
                                                   std::vector<RecoveredVertexColorAccumulator>& accumulator)
    {
        if (accumulator.size() != mesh.vertices.size())
        {
            throw std::invalid_argument("mesh color accumulator count does not match vertices");
        }
        std::vector<std::vector<std::size_t>> adjacency(mesh.vertices.size());
        for (const Face& face : mesh.faces)
        {
            const auto& v = face.vertices;
            if (v[0] >= mesh.vertices.size() || v[1] >= mesh.vertices.size() || v[2] >= mesh.vertices.size())
            {
                throw std::invalid_argument("mesh color extrapolation face index is invalid");
            }
            adjacency[v[0]].push_back(v[1]);
            adjacency[v[0]].push_back(v[2]);
            adjacency[v[1]].push_back(v[0]);
            adjacency[v[1]].push_back(v[2]);
            adjacency[v[2]].push_back(v[0]);
            adjacency[v[2]].push_back(v[1]);
        }
        for (auto& neighbors : adjacency)
        {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }

        std::vector<std::int32_t> distance(mesh.vertices.size(), -1);
        std::queue<std::size_t> pending;
        for (std::size_t vertex = 0U; vertex != accumulator.size(); ++vertex)
        {
            for (const float value : accumulator[vertex])
            {
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument("mesh color accumulator is non-finite");
                }
            }
            if (accumulator[vertex][3] > 0.01F)
            {
                distance[vertex] = 0;
                pending.push(vertex);
            }
            else
            {
                accumulator[vertex].fill(0.0F);
            }
        }

        while (!pending.empty())
        {
            const std::size_t vertex = pending.front();
            pending.pop();
            const std::int32_t level = distance[vertex];
            for (const std::size_t neighbor : adjacency[vertex])
            {
                if (distance[neighbor] == -1)
                {
                    distance[neighbor] = level + 1;
                    pending.push(neighbor);
                }
                else if (level != 0 && distance[neighbor] < level)
                {
                    volatile float reciprocal = 1.0F / accumulator[neighbor][3];
                    for (std::size_t channel = 0U; channel != 3U; ++channel)
                    {
                        volatile float contribution = accumulator[neighbor][channel] * reciprocal;
                        accumulator[vertex][channel] = contribution + accumulator[vertex][channel];
                    }
                    accumulator[vertex][3] = accumulator[vertex][3] + 1.0F;
                }
            }
        }
        return distance;
    }

    RecoveredMeshClipStats clip_mesh_to_plane_recovered(std::vector<RecoveredMeshClipVertex>& vertices,
                                                        std::vector<RecoveredMeshClipFace>& faces,
                                                        const RecoveredMeshClipPlane& plane)
    {
        RecoveredMeshClipStats stats;
        stats.input_vertices = vertices.size();
        stats.input_faces = faces.size();
        if (!std::isfinite(plane.offset) || !std::isfinite(plane.tolerance) || plane.tolerance < 0.0)
        {
            throw std::invalid_argument("mesh clip plane scalar is invalid");
        }
        for (const double value : plane.normal)
        {
            if (!std::isfinite(value))
                throw std::invalid_argument("mesh clip plane normal is invalid");
        }
        if (vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::overflow_error("mesh clip vertex index exceeds uint32");
        }
        for (const auto& face : faces)
        {
            for (const std::uint32_t vertex : face)
            {
                if (vertex >= vertices.size())
                    throw std::invalid_argument("mesh clip face index is invalid");
            }
        }
        if (vertices.empty() || faces.empty())
        {
            stats.output_vertices = vertices.size();
            stats.output_faces = faces.size();
            return stats;
        }

        // Keep every rounding boundary visible.  The target performs the plane
        // equation in scalar double precision, stores its result as float, then
        // performs the interpolation entirely in scalar float precision.
        std::vector<float> distances(vertices.size());
        std::vector<std::uint8_t> classification(vertices.size());
        for (std::size_t index = 0U; index != vertices.size(); ++index)
        {
            const auto& vertex = vertices[index];
            volatile double x_product = static_cast<double>(vertex[0]) * plane.normal[0];
            volatile double xy_sum = x_product + static_cast<double>(vertex[1]) * plane.normal[1];
            volatile double xyz_sum = xy_sum + static_cast<double>(vertex[2]) * plane.normal[2];
            const double signed_distance = xyz_sum - plane.offset;
            distances[index] = static_cast<float>(signed_distance);
            classification[index] = signed_distance >= plane.tolerance
                                        ? std::uint8_t{1}
                                        : (-plane.tolerance < signed_distance ? std::uint8_t{4} : std::uint8_t{2});
        }

        using Edge = std::pair<std::uint32_t, std::uint32_t>;
        std::map<Edge, std::uint32_t> intersections;
        std::vector<std::uint8_t> removed(faces.size(), 0U);
        std::vector<RecoveredMeshClipFace> appended_faces;
        const std::size_t original_face_count = faces.size();
        for (std::size_t face_index = 0U; face_index != original_face_count; ++face_index)
        {
            auto& face = faces[face_index];
            const std::uint8_t combined = classification[face[0]] | classification[face[1]] | classification[face[2]];
            // The target requires at least one strict-inside vertex.  A triangle
            // made entirely from tolerance-band vertices is discarded.
            if ((combined & 1U) == 0U)
            {
                removed[face_index] = 1U;
                ++stats.removed_faces;
                continue;
            }
            if ((combined & 2U) == 0U)
                continue;

            std::array<std::uint32_t, 4> polygon{};
            std::size_t polygon_size = 0U;
            std::uint32_t previous = face[0];
            for (std::size_t step = 1U; step != 4U; ++step)
            {
                const std::uint32_t current = face[step % 3U];
                const std::uint8_t previous_class = classification[previous];
                const std::uint8_t current_class = classification[current];
                const std::uint8_t edge_class = previous_class | current_class;
                if ((edge_class & 1U) != 0U && (edge_class & 2U) != 0U)
                {
                    const Edge edge = previous < current ? Edge{previous, current} : Edge{current, previous};
                    auto found = intersections.find(edge);
                    std::uint32_t intersection = 0U;
                    if (found == intersections.end())
                    {
                        if (vertices.size() == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                        {
                            throw std::overflow_error("mesh clip generated too many vertices");
                        }
                        volatile float denominator = distances[previous] - distances[current];
                        volatile float weight_current = distances[previous] / denominator;
                        volatile float weight_previous = 1.0F - weight_current;
                        RecoveredMeshClipVertex created{};
                        for (std::size_t axis = 0U; axis != 3U; ++axis)
                        {
                            volatile float current_term = vertices[current][axis] * weight_current;
                            volatile float previous_term = vertices[previous][axis] * weight_previous;
                            created[axis] = current_term + previous_term;
                        }
                        // The target zero-extends the z word into the upper half
                        // of its temporary 128-bit vertex, yielding +0 here.
                        created[3] = 0.0F;
                        intersection = static_cast<std::uint32_t>(vertices.size());
                        vertices.push_back(created);
                        intersections.emplace(edge, intersection);
                        ++stats.inserted_vertices;
                    }
                    else
                    {
                        intersection = found->second;
                    }
                    polygon[polygon_size++] = intersection;
                    if (previous_class == 2U)
                        polygon[polygon_size++] = current;
                }
                else if (current_class == 4U || ((edge_class & 1U) != 0U && (edge_class & 2U) == 0U))
                {
                    polygon[polygon_size++] = current;
                }
                previous = current;
            }
            if (polygon_size == 3U)
            {
                face = {polygon[0], polygon[1], polygon[2]};
            }
            else if (polygon_size == 4U)
            {
                face = {polygon[0], polygon[1], polygon[2]};
                appended_faces.push_back({polygon[0], polygon[2], polygon[3]});
                ++stats.appended_faces;
            }
            else
            {
                throw std::runtime_error("mesh clip produced a non-triangle/non-quad polygon");
            }
        }

        std::vector<RecoveredMeshClipFace> compact_faces;
        compact_faces.reserve(faces.size() - stats.removed_faces + appended_faces.size());
        for (std::size_t index = 0U; index != faces.size(); ++index)
        {
            if (removed[index] == 0U)
                compact_faces.push_back(faces[index]);
        }
        compact_faces.insert(compact_faces.end(), appended_faces.begin(), appended_faces.end());

        std::vector<std::uint8_t> used(vertices.size(), 0U);
        for (const auto& face : compact_faces)
        {
            used[face[0]] = 1U;
            used[face[1]] = 1U;
            used[face[2]] = 1U;
        }
        std::vector<std::uint32_t> remap(vertices.size(), std::numeric_limits<std::uint32_t>::max());
        std::vector<RecoveredMeshClipVertex> compact_vertices;
        compact_vertices.reserve(vertices.size());
        for (std::size_t index = 0U; index != vertices.size(); ++index)
        {
            if (used[index] == 0U)
                continue;
            remap[index] = static_cast<std::uint32_t>(compact_vertices.size());
            compact_vertices.push_back(vertices[index]);
        }
        for (auto& face : compact_faces)
        {
            face[0] = remap[face[0]];
            face[1] = remap[face[1]];
            face[2] = remap[face[2]];
        }
        stats.removed_vertices = vertices.size() - compact_vertices.size();
        vertices = std::move(compact_vertices);
        faces = std::move(compact_faces);
        stats.output_vertices = vertices.size();
        stats.output_faces = faces.size();
        return stats;
    }

    std::array<RecoveredMeshClipPlane, 6> make_recovered_region_clip_planes(const std::array<double, 15>& region,
                                                                            double tolerance)
    {
        if (!std::isfinite(tolerance) || tolerance < 0.0)
            throw std::invalid_argument("region clip tolerance is invalid");
        for (const double value : region)
        {
            if (!std::isfinite(value))
                throw std::invalid_argument("region clip input is non-finite");
        }
        if (!(region[12] > 0.0) || !(region[13] > 0.0) || !(region[14] > 0.0))
        {
            throw std::invalid_argument("region clip size is not positive");
        }
        const std::array<double, 3> center{region[9], region[10], region[11]};
        std::array<RecoveredMeshClipPlane, 6> planes{};
        for (std::size_t axis = 0U; axis != 3U; ++axis)
        {
            for (std::size_t component = 0U; component != 3U; ++component)
            {
                planes[axis].normal[component] = -region[component * 3U + axis];
                planes[axis + 3U].normal[component] = region[component * 3U + axis];
            }
            for (std::size_t side : {axis, axis + 3U})
            {
                planes[side].offset = planes[side].normal[0] * center[0] + planes[side].normal[1] * center[1] +
                                      planes[side].normal[2] * center[2] - region[12U + axis] * 0.5;
                planes[side].tolerance = tolerance;
            }
        }
        return planes;
    }

    std::array<RecoveredMeshClipStats, 6>
    clip_mesh_to_region_recovered(Mesh& mesh, const std::array<double, 15>& region, double tolerance)
    {
        if (mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::overflow_error("region clip mesh exceeds uint32 indexing");
        }
        std::vector<RecoveredMeshClipVertex> vertices(mesh.vertices.size());
        for (std::size_t index = 0U; index != mesh.vertices.size(); ++index)
        {
            vertices[index] = {static_cast<float>(mesh.vertices[index].position.x),
                               static_cast<float>(mesh.vertices[index].position.y),
                               static_cast<float>(mesh.vertices[index].position.z),
                               mesh.vertices[index].confidence};
        }
        std::vector<RecoveredMeshClipFace> faces(mesh.faces.size());
        for (std::size_t index = 0U; index != mesh.faces.size(); ++index)
        {
            for (std::size_t corner = 0U; corner != 3U; ++corner)
            {
                const std::size_t vertex = mesh.faces[index].vertices[corner];
                if (vertex > std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("region clip face exceeds uint32 indexing");
                faces[index][corner] = static_cast<std::uint32_t>(vertex);
            }
        }
        const auto planes = make_recovered_region_clip_planes(region, tolerance);
        std::array<RecoveredMeshClipStats, 6> stats{};
        for (std::size_t plane = 0U; plane != planes.size(); ++plane)
            stats[plane] = clip_mesh_to_plane_recovered(vertices, faces, planes[plane]);

        mesh.vertices.clear();
        mesh.vertices.resize(vertices.size());
        for (std::size_t index = 0U; index != vertices.size(); ++index)
        {
            mesh.vertices[index].position = {vertices[index][0], vertices[index][1], vertices[index][2]};
            mesh.vertices[index].confidence = vertices[index][3];
        }
        mesh.faces.resize(faces.size());
        for (std::size_t index = 0U; index != faces.size(); ++index)
        {
            mesh.faces[index].vertices = {faces[index][0], faces[index][1], faces[index][2]};
        }
        recompute_normals(mesh);
        return stats;
    }

    void smooth_mesh(Mesh& mesh, double strength, bool fix_borders, bool preserve_edges)
    {
        if (strength <= 0.0 || mesh.vertices.empty())
            return;
        std::vector<std::vector<std::size_t>> adjacency(mesh.vertices.size());
        std::unordered_map<std::uint64_t, std::size_t> edge_count;
        auto edge_key = [](std::size_t a, std::size_t b)
        {
            if (a > b)
                std::swap(a, b);
            return (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
        };
        for (const Face& face : mesh.faces)
        {
            for (std::size_t edge = 0; edge < 3; ++edge)
            {
                const std::size_t a = face.vertices[edge], b = face.vertices[(edge + 1) % 3];
                adjacency[a].push_back(b);
                adjacency[b].push_back(a);
                ++edge_count[edge_key(a, b)];
            }
        }
        std::vector<std::uint8_t> border(mesh.vertices.size(), 0);
        for (const auto& [key, count] : edge_count)
            if (count == 1)
            {
                border[static_cast<std::size_t>(key >> 32U)] = 1;
                border[static_cast<std::size_t>(key & 0xFFFFFFFFU)] = 1;
            }
        const std::size_t iterations = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(strength)));
        const double lambda = std::clamp(strength / static_cast<double>(iterations) * 0.25, 0.01, 0.5);
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            std::vector<metalign::Vec3> positions(mesh.vertices.size());
            for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
            {
                const Vertex& vertex = mesh.vertices[index];
                if ((fix_borders && border[index]) || adjacency[index].empty())
                {
                    positions[index] = vertex.position;
                    continue;
                }
                metalign::Vec3 average{};
                std::size_t used = 0;
                for (const std::size_t neighbor : adjacency[index])
                {
                    if (preserve_edges && metalign::dot(vertex.normal, mesh.vertices[neighbor].normal) < 0.75)
                        continue;
                    average = average + mesh.vertices[neighbor].position;
                    ++used;
                }
                if (used == 0)
                    positions[index] = vertex.position;
                else
                    positions[index] =
                        vertex.position * (1.0 - lambda) + average * (lambda / static_cast<double>(used));
            }
            for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
                mesh.vertices[index].position = positions[index];
            recompute_normals(mesh);
        }
    }

} // namespace metmodel
