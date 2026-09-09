#include "metashape_texture/graph_partition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metashape_texture {
namespace {

using FaceList = std::vector<std::uint32_t>;
using Vec3f = std::array<float, 3>;
using Vec3d = std::array<double, 3>;
using Matrix3d = std::array<std::array<double, 3>, 3>;

struct FaceSlots {
    std::array<std::int32_t, 3> neighbor{-1, -1, -1};
};

struct PartStats {
    std::uint64_t size = 0;
    Vec3d centroid{};
    Vec3d lower{};
    Vec3d upper{};
};

struct Candidate {
    double score = 0.0;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
};

struct CandidateGreater {
    bool operator()(const Candidate &left, const Candidate &right) const {
        // Native comparator only reads score. Exact-score ordering is therefore
        // the MSVC priority_queue heap order, not an invented ID tiebreak.
        return left.score > right.score;
    }
};

std::uint64_t edge_key(std::uint32_t left, std::uint32_t right) {
    const auto lo = std::min(left, right);
    const auto hi = std::max(left, right);
    return (static_cast<std::uint64_t>(lo) << 32U) | hi;
}

std::vector<FaceSlots> build_adjacency(const PartitionMesh &mesh) {
    struct Owner {
        std::uint32_t face = 0;
        std::uint32_t edge = 0;
        bool paired = false;
    };
    std::unordered_map<std::uint64_t, Owner> owners;
    owners.reserve(mesh.faces.size() * 2U);
    std::vector<FaceSlots> result(mesh.faces.size());
    for (std::uint32_t face_id = 0; face_id < mesh.faces.size(); ++face_id) {
        const auto &face = mesh.faces[face_id];
        for (std::uint32_t edge = 0; edge < 3; ++edge) {
            const auto key = edge_key(face[edge], face[(edge + 1U) % 3U]);
            const auto [iterator, inserted] = owners.emplace(key, Owner{face_id, edge, false});
            if (inserted) continue;
            if (iterator->second.paired) {
                throw std::runtime_error("non-manifold edge cannot use Metashape's 3-slot adjacency");
            }
            const Owner other = iterator->second;
            result[face_id].neighbor[edge] = static_cast<std::int32_t>(other.face);
            result[other.face].neighbor[other.edge] = static_cast<std::int32_t>(face_id);
            iterator->second.paired = true;
        }
    }
    return result;
}

std::vector<FaceList> connected_components(const std::vector<FaceSlots> &adjacency) {
    std::vector<std::uint8_t> seen(adjacency.size(), 0);
    std::vector<FaceList> result;
    std::queue<std::uint32_t> pending;
    for (std::uint32_t start = 0; start < adjacency.size(); ++start) {
        if (seen[start]) continue;
        seen[start] = 1;
        pending.push(start);
        FaceList component;
        while (!pending.empty()) {
            const auto face = pending.front();
            pending.pop();
            component.push_back(face);
            for (const auto neighbor : adjacency[face].neighbor) {
                if (neighbor < 0 || seen[static_cast<std::size_t>(neighbor)]) continue;
                seen[static_cast<std::size_t>(neighbor)] = 1;
                pending.push(static_cast<std::uint32_t>(neighbor));
            }
        }
        std::sort(component.begin(), component.end());
        result.push_back(std::move(component));
    }
    return result;
}

std::uint32_t rng_step(std::array<std::uint32_t, 3> &state) {
    const std::uint32_t shifted = (state[0] << 16U) ^ state[0];
    std::uint32_t mixed = (shifted >> 5U) ^ shifted;
    mixed ^= 2U * mixed;
    const std::uint32_t new_first = state[2];
    const std::uint32_t new_third = state[1];
    const std::uint32_t new_second = state[1] ^ mixed ^ new_first;
    state = {new_first, new_second, new_third};
    return new_second;
}

std::vector<Vec3f> generate_samples(const FaceList &face_ids, const PartitionMesh &mesh) {
    const std::size_t requested = std::min<std::size_t>(3U * face_ids.size(), 10000U);
    std::vector<Vec3f> result;
    result.reserve(requested);
    if (requested == 3U * face_ids.size()) {
        for (const auto face_id : face_ids) {
            for (const auto vertex_id : mesh.faces[face_id]) result.push_back(mesh.vertices[vertex_id]);
        }
        return result;
    }
    std::array<std::uint32_t, 3> state{
        static_cast<std::uint32_t>(-1518166903), 521288629U, 362436069U};
    for (std::size_t ordinal = 0; ordinal < requested; ++ordinal) {
        const auto face_ordinal = rng_step(state) % static_cast<std::uint32_t>(face_ids.size());
        const auto vertex_ordinal = rng_step(state) % 3U;
        result.push_back(mesh.vertices[mesh.faces[face_ids[face_ordinal]][vertex_ordinal]]);
    }
    return result;
}

double pythag(double left, double right) {
    left = std::abs(left);
    right = std::abs(right);
    if (left > right) {
        const double ratio = right / left;
        return left * std::sqrt(1.0 + ratio * ratio);
    }
    if (right == 0.0) return 0.0;
    const double ratio = left / right;
    return right * std::sqrt(1.0 + ratio * ratio);
}

std::pair<Matrix3d, Vec3d> svd_nr(Matrix3d a) {
    constexpr int m = 3;
    constexpr int n = 3;
    Vec3d w{};
    Matrix3d v{};
    Vec3d rv1{};
    double g = 0.0;
    double scale = 0.0;
    double anorm = 0.0;
    for (int i = 0; i < n; ++i) {
        int l = i + 1;
        rv1[i] = scale * g;
        g = scale = 0.0;
        double s = 0.0;
        scale = 0.0;
        for (int k = i; k < m; ++k) scale += std::abs(a[k][i]);
        if (scale != 0.0) {
            for (int k = i; k < m; ++k) {
                a[k][i] /= scale;
                s += a[k][i] * a[k][i];
            }
            const double f = a[i][i];
            g = -std::copysign(std::sqrt(s), f);
            const double h = f * g - s;
            a[i][i] = f - g;
            if (i != n - 1) {
                for (int j = l; j < n; ++j) {
                    s = 0.0;
                    for (int k = i; k < m; ++k) s += a[k][i] * a[k][j];
                    const double factor = s / h;
                    for (int k = i; k < m; ++k) a[k][j] += factor * a[k][i];
                }
            }
            for (int k = i; k < m; ++k) a[k][i] *= scale;
        }
        w[i] = scale * g;
        g = scale = 0.0;
        s = 0.0;
        if (i != n - 1) {
            for (int k = l; k < n; ++k) scale += std::abs(a[i][k]);
            if (scale != 0.0) {
                for (int k = l; k < n; ++k) {
                    a[i][k] /= scale;
                    s += a[i][k] * a[i][k];
                }
                const double f = a[i][l];
                g = -std::copysign(std::sqrt(s), f);
                const double h = f * g - s;
                a[i][l] = f - g;
                for (int k = l; k < n; ++k) rv1[k] = a[i][k] / h;
                if (i != m - 1) {
                    for (int j = l; j < m; ++j) {
                        s = 0.0;
                        for (int k = l; k < n; ++k) s += a[j][k] * a[i][k];
                        for (int k = l; k < n; ++k) a[j][k] += s * rv1[k];
                    }
                }
                for (int k = l; k < n; ++k) a[i][k] *= scale;
            }
        }
        anorm = std::max(anorm, std::abs(w[i]) + std::abs(rv1[i]));
    }

    g = 0.0;
    int l = n;
    for (int i = n - 1; i >= 0; --i) {
        if (i < n - 1) {
            if (g != 0.0) {
                for (int j = l; j < n; ++j) v[j][i] = (a[i][j] / a[i][l]) / g;
                for (int j = l; j < n; ++j) {
                    double s = 0.0;
                    for (int k = l; k < n; ++k) s += a[i][k] * v[k][j];
                    for (int k = l; k < n; ++k) v[k][j] += s * v[k][i];
                }
            }
            for (int j = l; j < n; ++j) v[i][j] = v[j][i] = 0.0;
        }
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
    }

    for (int i = std::min(m, n) - 1; i >= 0; --i) {
        l = i + 1;
        g = w[i];
        for (int j = l; j < n; ++j) a[i][j] = 0.0;
        if (g != 0.0) {
            const double inverse = 1.0 / g;
            for (int j = l; j < n; ++j) {
                double s = 0.0;
                for (int k = l; k < m; ++k) s += a[k][i] * a[k][j];
                const double factor = (s / a[i][i]) * inverse;
                for (int k = i; k < m; ++k) a[k][j] += factor * a[k][i];
            }
            for (int j = i; j < m; ++j) a[j][i] *= inverse;
        } else {
            for (int j = i; j < m; ++j) a[j][i] = 0.0;
        }
        a[i][i] += 1.0;
    }

    for (int k = n - 1; k >= 0; --k) {
        for (int iteration = 0; iteration < 30; ++iteration) {
            bool flag = true;
            l = k;
            for (;;) {
                const int nm = l - 1;
                if (std::abs(rv1[l]) + anorm == anorm) {
                    flag = false;
                    break;
                }
                if (nm < 0 || std::abs(w[nm]) + anorm == anorm) break;
                --l;
            }
            if (flag) {
                double c = 0.0;
                double s = 1.0;
                for (int i = l; i <= k; ++i) {
                    const double f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    if (std::abs(f) + anorm == anorm) break;
                    g = w[i];
                    double h = pythag(f, g);
                    w[i] = h;
                    h = 1.0 / h;
                    c = g * h;
                    s = -f * h;
                    if (l == 0) continue;
                    const int nm = l - 1;
                    for (int j = 0; j < m; ++j) {
                        const double y = a[j][nm];
                        const double z = a[j][i];
                        a[j][nm] = y * c + z * s;
                        a[j][i] = z * c - y * s;
                    }
                }
            }
            double z = w[k];
            if (l == k) {
                if (z < 0.0) {
                    w[k] = -z;
                    for (int row = 0; row < n; ++row) v[row][k] *= -1.0;
                }
                break;
            }
            if (iteration == 29) throw std::runtime_error("recovered SVD failed to converge");
            const double x0 = w[l];
            const int nm = k - 1;
            double y = w[nm];
            g = rv1[nm];
            double h = rv1[k];
            double f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x0 - z) * (x0 + z) + h * (y / (f + std::copysign(g, f)) - h)) / x0;
            double c = 1.0;
            double s = 1.0;
            double x = x0;
            for (int j = l; j <= nm; ++j) {
                const int i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                for (int row = 0; row < n; ++row) {
                    x = v[row][j];
                    z = v[row][i];
                    v[row][j] = x * c + z * s;
                    v[row][i] = z * c - x * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z != 0.0) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = c * g + s * y;
                x = c * y - s * g;
                for (int row = 0; row < m; ++row) {
                    y = a[row][j];
                    z = a[row][i];
                    a[row][j] = y * c + z * s;
                    a[row][i] = z * c - y * s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) { return w[left] > w[right]; });
    Matrix3d sorted_a{};
    Vec3d sorted_w{};
    for (int column = 0; column < 3; ++column) {
        sorted_w[column] = w[order[column]];
        for (int row = 0; row < 3; ++row) sorted_a[row][column] = a[row][order[column]];
    }
    return {sorted_a, sorted_w};
}

Vec3f principal_axis(const std::vector<Vec3f> &samples) {
    Vec3d mean{};
    for (const auto &point : samples) {
        mean[0] += static_cast<double>(point[0]);
        mean[1] += static_cast<double>(point[1]);
        mean[2] += static_cast<double>(point[2]);
    }
    for (double &value : mean) value /= static_cast<double>(samples.size());
    Matrix3d covariance{};
    for (const auto &point : samples) {
        Vec3d delta{};
        for (int axis = 0; axis < 3; ++axis) delta[axis] = static_cast<double>(point[axis]) - mean[axis];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance[row][column] += delta[row] * delta[column];
            }
        }
    }
    const auto [u, singular] = svd_nr(covariance);
    (void)singular;
    return {static_cast<float>(u[0][0]), static_cast<float>(u[1][0]), static_cast<float>(u[2][0])};
}

Vec3f normalize_native(const Vec3f &raw) {
    const float xx = raw[0] * raw[0];
    const float yy = raw[1] * raw[1];
    const float zz = raw[2] * raw[2];
    const float xy = yy + xx;
    const float squared = xy + zz;
    const float length = static_cast<float>(std::sqrt(static_cast<double>(squared)));
    const float scale = length >= 1.0e-20F ? 1.0F / length : 0.0F;
    return {scale * raw[0], scale * raw[1], scale * raw[2]};
}

float project_native(const Vec3f &vertex, const Vec3f &origin, const Vec3f &normal) {
    const float dx = vertex[0] - origin[0];
    const float dy = vertex[1] - origin[1];
    const float dz = vertex[2] - origin[2];
    const float x = dx * normal[0];
    const float y = dy * normal[1];
    const float z = dz * normal[2];
    const float xy = x + y;
    return xy + z;
}

std::vector<FaceList> split_once(const FaceList &face_ids, const PartitionMesh &mesh,
                                 const Vec3f &raw_normal, std::size_t limit) {
    const std::size_t estimated = (face_ids.size() + limit - 1U) / limit;
    const std::size_t divisor = std::min<std::size_t>(estimated, 4U);
    const std::size_t target = (face_ids.size() + divisor - 1U) / divisor;
    const std::size_t minimum = 8U * target / 10U;
    const Vec3f origin = mesh.vertices[mesh.faces[face_ids.front()][0]];
    const Vec3f normal = normalize_native(raw_normal);
    struct Event { float projection; std::uint32_t local_id; };
    std::vector<Event> events;
    events.reserve(face_ids.size() * 2U);
    for (std::uint32_t local_id = 0; local_id < face_ids.size(); ++local_id) {
        const auto &face = mesh.faces[face_ids[local_id]];
        std::array<float, 3> projection{};
        for (int slot = 0; slot < 3; ++slot) {
            projection[slot] = project_native(mesh.vertices[face[slot]], origin, normal);
        }
        events.push_back({std::max({projection[0], projection[1], projection[2]}), local_id});
        events.push_back({std::min({projection[0], projection[1], projection[2]}), local_id});
    }
    std::sort(events.begin(), events.end(), [](const Event &left, const Event &right) {
        if (left.projection != right.projection) return left.projection < right.projection;
        return left.local_id < right.local_id;
    });
    std::vector<std::uint8_t> active(face_ids.size(), 0);
    std::size_t active_count = 0;
    FaceList encountered;
    std::vector<FaceList> chunks;
    std::size_t best_active = std::numeric_limits<std::size_t>::max();
    std::size_t best_size = 0;
    for (const auto event : events) {
        if (active[event.local_id]) {
            active[event.local_id] = 0;
            --active_count;
        } else {
            active[event.local_id] = 1;
            ++active_count;
            encountered.push_back(face_ids[event.local_id]);
        }
        if (encountered.size() >= minimum && active_count <= best_active) {
            best_size = encountered.size();
            best_active = active_count;
        }
        if (encountered.size() > target) {
            if (best_size < minimum || best_size > target) throw std::runtime_error("invalid native sweep cut");
            chunks.emplace_back(encountered.begin(), encountered.begin() + static_cast<std::ptrdiff_t>(best_size));
            encountered.erase(encountered.begin(), encountered.begin() + static_cast<std::ptrdiff_t>(best_size));
            best_size = 0;
            best_active = std::numeric_limits<std::size_t>::max();
        }
    }
    if (active_count != 0) throw std::runtime_error("native sweep ended with active faces");
    if (!encountered.empty()) {
        if (encountered.size() > target) throw std::runtime_error("native sweep tail exceeds target");
        chunks.push_back(std::move(encountered));
    }
    return chunks;
}

void recursive_split(const FaceList &part, const PartitionMesh &mesh, std::size_t limit,
                     std::vector<FaceList> &output) {
    // The top-level caller invokes the sweep even for an already-small
    // connected component.  That matters for vector order: the 22-face dino
    // component is emitted in encounter order, not its ascending input order.
    const auto normal = principal_axis(generate_samples(part, mesh));
    auto chunks = split_once(part, mesh, normal, limit);
    for (const auto &chunk : chunks) {
        if (chunk.size() > limit) recursive_split(chunk, mesh, limit, output);
        else output.push_back(chunk);
    }
}

PartStats part_stats(const FaceList &part, const PartitionMesh &mesh) {
    PartStats result;
    result.size = part.size();
    result.lower.fill(std::numeric_limits<double>::max());
    result.upper.fill(-std::numeric_limits<double>::max());
    Vec3d sum{};
    for (const auto face_id : part) {
        for (const auto vertex_id : mesh.faces[face_id]) {
            const auto &point = mesh.vertices[vertex_id];
            for (int axis = 0; axis < 3; ++axis) {
                const double value = static_cast<double>(point[axis]);
                sum[axis] += value;
                result.lower[axis] = std::min(result.lower[axis], value);
                result.upper[axis] = std::max(result.upper[axis], value);
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        result.centroid[axis] = sum[axis] / static_cast<double>(3U * part.size());
    }
    return result;
}

PartStats merge_stats(const PartStats &left, const PartStats &right) {
    PartStats result;
    result.size = left.size + right.size;
    for (int axis = 0; axis < 3; ++axis) {
        result.centroid[axis] =
            (static_cast<double>(left.size) * left.centroid[axis] +
             static_cast<double>(right.size) * right.centroid[axis]) /
            static_cast<double>(result.size);
        result.lower[axis] = std::min(left.lower[axis], right.lower[axis]);
        result.upper[axis] = std::max(left.upper[axis], right.upper[axis]);
    }
    return result;
}

double extent(const PartStats &stats) {
    const double x = stats.upper[0] - stats.lower[0];
    const double y = stats.upper[1] - stats.lower[1];
    const double z = stats.upper[2] - stats.lower[2];
    return (x + y) + z;
}

double candidate_score(const PartStats &left, const PartStats &right) {
    const double combined = extent(merge_stats(left, right));
    return std::min(combined / extent(left), combined / extent(right));
}

std::vector<std::vector<std::uint32_t>> spatial_neighbors(const std::vector<PartStats> &records) {
    std::vector<std::vector<std::uint32_t>> result(records.size());
    const std::size_t retained_count = std::min<std::size_t>(records.size(), 33U);
    for (std::uint32_t query = 0; query < records.size(); ++query) {
        std::vector<std::pair<double, std::uint32_t>> ranked;
        ranked.reserve(records.size());
        for (std::uint32_t point = 0; point < records.size(); ++point) {
            const double dx = records[point].centroid[0] - records[query].centroid[0];
            const double dy = records[point].centroid[1] - records[query].centroid[1];
            const double dz = records[point].centroid[2] - records[query].centroid[2];
            const double squared = (dx * dx + dy * dy) + dz * dz;
            ranked.emplace_back(squared, point);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &left, const auto &right) {
            if (left.first != right.first) return left.first < right.first;
            return left.second < right.second;
        });
        auto &neighbors = result[query];
        for (std::size_t ordinal = 0; ordinal < retained_count; ++ordinal) {
            if (ranked[ordinal].second != query) neighbors.push_back(ranked[ordinal].second);
        }
        std::sort(neighbors.begin(), neighbors.end());
    }
    return result;
}

std::uint32_t find_root(std::vector<std::uint32_t> &parent, std::uint32_t value) {
    if (parent[value] != value) parent[value] = find_root(parent, parent[value]);
    return parent[value];
}

std::vector<FaceList> merge_pass(const std::vector<FaceList> &parts, const PartitionMesh &mesh,
                                 std::size_t limit, std::size_t pass,
                                 std::vector<PartitionAcceptedUnion> &accepted) {
    std::vector<PartStats> records;
    records.reserve(parts.size());
    for (const auto &part : parts) records.push_back(part_stats(part, mesh));
    const auto neighbors = spatial_neighbors(records);
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> candidates;
    for (std::uint32_t left = 0; left < neighbors.size(); ++left) {
        for (const auto right : neighbors[left]) {
            candidates.push({candidate_score(records[left], records[right]), left, right});
        }
    }
    std::vector<std::uint32_t> parent(parts.size());
    std::vector<std::uint32_t> dsu_size(parts.size(), 1U);
    std::iota(parent.begin(), parent.end(), 0U);
    while (!candidates.empty()) {
        const Candidate candidate = candidates.top();
        candidates.pop();
        const auto left = find_root(parent, candidate.left);
        const auto right = find_root(parent, candidate.right);
        const PartStats combined = merge_stats(records[left], records[right]);
        if (combined.size > limit) continue;
        accepted.push_back({pass, left, right, combined.size});
        if (left == right) {
            records[left] = combined;
        } else if (dsu_size[left] >= dsu_size[right]) {
            parent[right] = left;
            dsu_size[left] += dsu_size[right];
            records[left] = combined;
        } else {
            parent[left] = right;
            dsu_size[right] += dsu_size[left];
            records[right] = combined;
        }
    }
    std::vector<FaceList> groups;
    std::unordered_map<std::uint32_t, std::size_t> group_by_root;
    for (std::uint32_t ordinal = 0; ordinal < parts.size(); ++ordinal) {
        const auto root = find_root(parent, ordinal);
        const auto [iterator, inserted] = group_by_root.emplace(root, groups.size());
        if (inserted) groups.emplace_back();
        auto &group = groups[iterator->second];
        group.insert(group.end(), parts[ordinal].begin(), parts[ordinal].end());
    }
    return groups;
}

FaceList bfs_outskirts(const FaceList &core, const std::vector<FaceSlots> &adjacency,
                       std::size_t radius) {
    std::vector<std::uint8_t> seen(adjacency.size(), 0);
    std::queue<std::pair<std::uint32_t, std::size_t>> pending;
    for (const auto face : core) {
        seen[face] = 1;
        pending.emplace(face, 0U);
    }
    FaceList output;
    while (!pending.empty()) {
        const auto [face, distance] = pending.front();
        pending.pop();
        if (distance == radius) continue;
        for (const auto neighbor : adjacency[face].neighbor) {
            if (neighbor < 0 || seen[static_cast<std::size_t>(neighbor)]) continue;
            const auto next = static_cast<std::uint32_t>(neighbor);
            seen[next] = 1;
            output.push_back(next);
            pending.emplace(next, distance + 1U);
        }
    }
    return output;
}

}  // namespace

GraphPartitionBuildResult build_graph_partitions(const PartitionMesh &mesh, std::size_t merge_limit,
                                                  std::size_t merge_pass_count,
                                                  std::size_t outskirts_radius) {
    if (mesh.faces.empty()) return {};
    if (merge_limit == 0) throw std::invalid_argument("merge_limit must be positive");
    for (const auto &face : mesh.faces) {
        for (const auto vertex : face) {
            if (vertex >= mesh.vertices.size()) throw std::out_of_range("partition mesh face vertex");
        }
    }
    GraphPartitionBuildResult result;
    const auto adjacency = build_adjacency(mesh);
    result.connected_components = connected_components(adjacency);
    const std::size_t preliminary_limit = 9U * merge_limit / 10U;
    for (const auto &component : result.connected_components) {
        recursive_split(component, mesh, preliminary_limit, result.preliminary_parts);
    }
    auto current = result.preliminary_parts;
    for (std::size_t pass = 0; pass < merge_pass_count; ++pass) {
        current = merge_pass(current, mesh, merge_limit, pass, result.accepted_unions);
        result.merge_passes.push_back(current);
    }
    std::vector<std::uint32_t> core_owners(mesh.faces.size(), 0U);
    for (const auto &core : current) {
        TextureGraphPartition partition;
        partition.core_faces = core;
        partition.outskirts_faces = bfs_outskirts(core, adjacency, outskirts_radius);
        partition.global_faces = core;
        partition.global_faces.insert(partition.global_faces.end(), partition.outskirts_faces.begin(),
                                      partition.outskirts_faces.end());
        std::sort(partition.global_faces.begin(), partition.global_faces.end());
        partition.core_mask.resize(partition.global_faces.size(), 0U);
        for (const auto face : core) {
            ++core_owners[face];
            const auto iterator = std::lower_bound(partition.global_faces.begin(),
                                                   partition.global_faces.end(), face);
            partition.core_mask[static_cast<std::size_t>(iterator - partition.global_faces.begin())] = 1U;
        }
        result.partitions.push_back(std::move(partition));
    }
    if (std::any_of(core_owners.begin(), core_owners.end(), [](std::uint32_t count) { return count != 1U; })) {
        throw std::runtime_error("recovered graph partition cores do not own every face exactly once");
    }
    return result;
}

}  // namespace metashape_texture
