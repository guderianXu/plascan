#include "metashape_texture/texture_pipeline.hpp"
#include "metashape_texture/candidate_unary.hpp"
#include "metashape_texture/focus_project_pipeline.hpp"

#include "metashape_texture/camera_model.hpp"
#include "metashape_texture/graph_partition.hpp"
#include "metashape_texture/natural_uv_worker.hpp"
#include "metashape_texture/recovered_kernels.hpp"
#if defined(MSTEXTURE_EXACT_PAGE_CODEC)
#include "metashape_texture/page_codec.hpp"
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <xatlas.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace metashape_texture {
namespace {

std::uint16_t positive_float_to_half(float value) {
    if (value == 0.0F) return 0;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffU;
    std::uint32_t rounded = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
        ++rounded;
        if (rounded == 0x400U) {
            rounded = 0;
            ++exponent;
        }
    }
    return static_cast<std::uint16_t>((exponent << 10) | static_cast<int>(rounded));
}

float positive_half_to_float(std::uint16_t value) {
    if (value == 0) return 0.0F;
    const std::uint32_t exponent = ((value >> 10) & 0x1fU) - 15U + 127U;
    const std::uint32_t mantissa = static_cast<std::uint32_t>(value & 0x3ffU) << 13;
    return std::bit_cast<float>((exponent << 23) | mantissa);
}

struct Face {
    std::array<std::uint32_t, 3> vertex{};
    std::array<std::uint32_t, 3> texcoord{};
};

struct Mesh {
    std::vector<Vec3d> vertices;
    std::vector<Vec2d> texcoords;
    std::vector<Face> faces;
};

struct Camera {
    std::string label;
    std::filesystem::path photo_path;
    CameraPose pose;
    FrameCalibration calibration;
    cv::Mat image;
    cv::Mat quality_image;
    // Optional reference/diagnostic input.  Metashape computes this 8-bit
    // 1/4-resolution photoconsistency image before reducing it per face.
    cv::Mat consistency_map;
    // Optional recovered 160x120 R8 out-of-focus quality map.  Raw nonzero
    // bytes are sampled through R8_UNORM, so byte 1 becomes float 1/255 in
    // FaceCameraOption +0x04.
    cv::Mat focus_quality_map;
    std::vector<float> focus_quality_sum;
    std::vector<float> focus_quality_count;
    std::vector<float> exact_nadirness;
    std::vector<float> exact_resolution;
    std::vector<float> exact_face_weight;
    std::vector<float> exact_edge_count;
    // Optional Metashape camera mask at the original photo resolution.
    cv::Mat mask;
    cv::Mat depth;
    cv::Mat scaled_depth;
    double depth_minimum = 0.0;
    double depth_maximum = 1.0;
    // Face-id attachment produced by the same z-tested raster pass as depth.
    // Metashape's edge-quality kernels inspect a 3x3 neighborhood in this
    // image before falling back to a projected edge-midpoint test.
    cv::Mat face_id;
};

struct ProjectedFace {
    std::array<Vec2d, 3> pixel{};
    std::array<double, 3> depth{};
    double area = 0.0;
    double view_dot = 1.0;
    bool valid = false;
};

struct FaceAdjacency {
    int first = -1;
    int second = -1;
    int first_edge = -1;
    int second_edge = -1;
};

struct GraphPartition {
    std::vector<std::uint32_t> global_faces;
    std::vector<std::uint8_t> core;
};

using DirectedEdgeCosts = std::vector<std::array<std::vector<float>, 3>>;

class DinicMaxflow {
public:
    explicit DinicMaxflow(int node_count) : graph_(node_count), level_(node_count), next_(node_count) {}

    void add_edge(int from, int to, float capacity) {
        if (capacity <= 0) return;
        Edge forward{to, static_cast<int>(graph_[to].size()), capacity};
        Edge reverse{from, static_cast<int>(graph_[from].size()), 0};
        graph_[from].push_back(forward);
        graph_[to].push_back(reverse);
    }

    float solve(int source, int sink) {
        float result = 0.0F;
        while (build_levels(source, sink)) {
            std::fill(next_.begin(), next_.end(), 0);
            while (const auto pushed = push(source, sink, std::numeric_limits<float>::max())) {
                result += pushed;
            }
        }
        return result;
    }

    std::vector<bool> source_partition(int source) const {
        std::vector<bool> reached(graph_.size(), false);
        std::queue<int> pending;
        reached[source] = true;
        pending.push(source);
        while (!pending.empty()) {
            const int node = pending.front();
            pending.pop();
            for (const Edge &edge : graph_[node]) {
                if (edge.capacity > 0 && !reached[edge.to]) {
                    reached[edge.to] = true;
                    pending.push(edge.to);
                }
            }
        }
        return reached;
    }

private:
    struct Edge { int to; int reverse; float capacity; };
    std::vector<std::vector<Edge>> graph_;
    std::vector<int> level_;
    std::vector<int> next_;

    bool build_levels(int source, int sink) {
        std::fill(level_.begin(), level_.end(), -1);
        std::queue<int> pending;
        level_[source] = 0;
        pending.push(source);
        while (!pending.empty()) {
            const int node = pending.front();
            pending.pop();
            for (const Edge &edge : graph_[node]) {
                if (edge.capacity > 0 && level_[edge.to] < 0) {
                    level_[edge.to] = level_[node] + 1;
                    pending.push(edge.to);
                }
            }
        }
        return level_[sink] >= 0;
    }

    float push(int node, int sink, float available) {
        if (node == sink) return available;
        for (int &index = next_[node]; index < static_cast<int>(graph_[node].size()); ++index) {
            Edge &edge = graph_[node][index];
            if (edge.capacity <= 0 || level_[edge.to] != level_[node] + 1) continue;
            const float sent = push(edge.to, sink, std::min(available, edge.capacity));
            if (sent == 0.0F) continue;
            edge.capacity -= sent;
            graph_[edge.to][edge.reverse].capacity += sent;
            return sent;
        }
        return 0;
    }
};

std::vector<FaceAdjacency> build_face_adjacency(const Mesh &mesh) {
    struct EdgeOwner { int face; int edge; };
    std::unordered_map<std::uint64_t, EdgeOwner> owner;
    std::vector<FaceAdjacency> adjacency;
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
        const Face &face = mesh.faces[face_index];
        for (int edge = 0; edge < 3; ++edge) {
            const std::uint32_t a = face.vertex[edge];
            const std::uint32_t b = face.vertex[(edge + 1) % 3];
            const std::uint32_t lo = std::min(a, b);
            const std::uint32_t hi = std::max(a, b);
            const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | hi;
            const auto [it, inserted] = owner.emplace(key, EdgeOwner{static_cast<int>(face_index), edge});
            if (!inserted && it->second.face != static_cast<int>(face_index)) {
                adjacency.push_back({it->second.face, static_cast<int>(face_index), it->second.edge, edge});
            }
        }
    }
    return adjacency;
}

std::vector<GraphPartition> load_graph_partitions(const std::filesystem::path &path,
                                                  std::size_t face_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open native graph partitions fixture");
    std::array<char, 4> magic{};
    std::uint32_t partition_count = 0;
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char *>(&partition_count), sizeof(partition_count));
    if (magic != std::array<char, 4>{'M', 'S', 'P', 'T'}) {
        throw std::runtime_error("native graph partitions fixture has wrong magic");
    }
    std::vector<GraphPartition> result(partition_count);
    std::vector<int> core_owners(face_count, 0);
    for (auto &partition : result) {
        std::uint32_t size = 0;
        input.read(reinterpret_cast<char *>(&size), sizeof(size));
        partition.global_faces.resize(size);
        partition.core.resize(size);
        input.read(reinterpret_cast<char *>(partition.global_faces.data()),
                   static_cast<std::streamsize>(size * sizeof(std::uint32_t)));
        input.read(reinterpret_cast<char *>(partition.core.data()),
                   static_cast<std::streamsize>(size));
        if (!input) throw std::runtime_error("native graph partitions fixture is truncated");
        for (std::size_t local = 0; local < partition.global_faces.size(); ++local) {
            const auto global = partition.global_faces[local];
            if (global >= face_count) throw std::runtime_error("partition face is out of range");
            if (partition.core[local]) ++core_owners[global];
        }
    }
    if (std::any_of(core_owners.begin(), core_owners.end(), [](int count) { return count != 1; })) {
        throw std::runtime_error("graph partitions do not have exactly one core owner per face");
    }
    return result;
}

std::vector<GraphPartition> recover_graph_partitions(const Mesh &mesh) {
    PartitionMesh partition_mesh;
    partition_mesh.vertices.reserve(mesh.vertices.size());
    for (const auto &vertex : mesh.vertices) {
        partition_mesh.vertices.push_back({static_cast<float>(vertex.x),
                                           static_cast<float>(vertex.y),
                                           static_cast<float>(vertex.z)});
    }
    partition_mesh.faces.reserve(mesh.faces.size());
    for (const auto &face : mesh.faces) partition_mesh.faces.push_back(face.vertex);
    const auto recovered = build_graph_partitions(partition_mesh);
    std::vector<GraphPartition> result;
    result.reserve(recovered.partitions.size());
    for (const auto &partition : recovered.partitions) {
        result.push_back({partition.global_faces, partition.core_mask});
    }
    return result;
}

float recovered_pairwise_cost(const FaceAdjacency &edge, int first_label, int second_label,
                              const DirectedEdgeCosts &costs) {
    if (first_label < 0 || second_label < 0 || first_label == second_label) return 0.0F;
    return std::max(
        costs[static_cast<std::size_t>(edge.first)][static_cast<std::size_t>(edge.first_edge)]
             [static_cast<std::size_t>(first_label)],
        costs[static_cast<std::size_t>(edge.second)][static_cast<std::size_t>(edge.second_edge)]
             [static_cast<std::size_t>(second_label)]);
}

float labeling_energy(const std::vector<int> &labels,
                      const std::vector<std::vector<std::int64_t>> &unary,
                      const std::vector<FaceAdjacency> &adjacency,
                      const DirectedEdgeCosts &edge_costs) {
    float result = 0.0F;
    for (std::size_t face = 0; face < labels.size(); ++face) {
        if (labels[face] >= 0) {
            result += static_cast<float>(unary[face][static_cast<std::size_t>(labels[face])]);
        }
    }
    for (const auto &edge : adjacency) {
        const int a = labels[static_cast<std::size_t>(edge.first)];
        const int b = labels[static_cast<std::size_t>(edge.second)];
        result += recovered_pairwise_cost(edge, a, b, edge_costs);
    }
    return result;
}

void alpha_expand(std::vector<int> &labels,
                  const std::vector<std::vector<std::int64_t>> &unary,
                  const std::vector<FaceAdjacency> &adjacency,
                  const DirectedEdgeCosts &edge_costs, int label_count) {
    // Metashape's optimizer is the Boykov-Kolmogorov alpha-expansion
    // construction.  In particular, an edge whose endpoints currently have
    // different non-alpha labels is represented by an auxiliary node.  Keep
    // that construction verbatim here instead of using an algebraically
    // transformed half-weight formulation: the native implementation uses
    // single-precision integral-valued costs and this also preserves its cut
    // orientation and tie behaviour.
    for (std::size_t face = 0; face < labels.size(); ++face) {
        labels[face] = static_cast<int>(std::min_element(unary[face].begin(), unary[face].end()) -
                                        unary[face].begin());
    }
    float current_energy = labeling_energy(labels, unary, adjacency, edge_costs);
    static constexpr float expansion_infinity = std::numeric_limits<float>::max();
    for (int alpha = 0; alpha < label_count; ++alpha) {
        const int source = static_cast<int>(labels.size() + adjacency.size());
        const int sink = source + 1;
        DinicMaxflow graph(sink + 1);
        auto add_terminal_weights = [&](int node, float source_capacity,
                                        float sink_capacity) {
            // BK add_tweights(node, cap_source, cap_sink).  Source-side nodes
            // pay cap_sink and retain their current label; sink-side nodes pay
            // cap_source and switch to alpha.  FUN_144188720 subtracts the
            // common minimum and stores only the signed terminal difference;
            // retaining both arcs is energy-equivalent but changes tie cuts.
            if (source_capacity > sink_capacity) {
                graph.add_edge(source, node, source_capacity - sink_capacity);
            } else if (sink_capacity > source_capacity) {
                graph.add_edge(node, sink, sink_capacity - source_capacity);
            }
        };
        auto add_bk_edge = [&](int first, int second, float forward,
                               float reverse) {
            graph.add_edge(first, second, forward);
            graph.add_edge(second, first, reverse);
        };
        for (std::size_t face = 0; face < labels.size(); ++face) {
            add_terminal_weights(static_cast<int>(face),
                                 static_cast<float>(unary[face][static_cast<std::size_t>(alpha)]),
                                 labels[face] == alpha
                                     ? expansion_infinity
                                     : static_cast<float>(
                                           unary[face][static_cast<std::size_t>(labels[face])]));
        }
        int next_auxiliary = static_cast<int>(labels.size());
        for (const auto &edge : adjacency) {
            const std::size_t i = static_cast<std::size_t>(edge.first);
            const std::size_t j = static_cast<std::size_t>(edge.second);
            const int p = labels[i];
            const int q = labels[j];
            if (p == alpha && q == alpha) continue;
            if (p == q) {
                const auto current_alpha = recovered_pairwise_cost(edge, p, alpha, edge_costs);
                const auto alpha_current = recovered_pairwise_cost(edge, alpha, q, edge_costs);
                add_bk_edge(edge.first, edge.second, current_alpha, alpha_current);
            } else if (p == alpha) {
                const auto current_alpha = recovered_pairwise_cost(edge, p, q, edge_costs);
                add_bk_edge(edge.second, edge.first, current_alpha, 0);
            } else if (q == alpha) {
                const auto current_alpha = recovered_pairwise_cost(edge, p, q, edge_costs);
                add_bk_edge(edge.first, edge.second, current_alpha, 0);
            } else {
                const auto e00 = recovered_pairwise_cost(edge, p, q, edge_costs);
                const auto e01 = recovered_pairwise_cost(edge, p, alpha, edge_costs);
                const auto e10 = recovered_pairwise_cost(edge, alpha, q, edge_costs);
                if (e01 + e10 < e00) {
                    throw std::runtime_error("non-metric alpha-expansion term");
                }
                const int auxiliary = next_auxiliary++;
                add_terminal_weights(auxiliary, 0, e00);
                // The fifth stack argument at 1422cece6/1422ced46 is loaded
                // from optimizer[0] (FLT_MAX), not zero.  It forces the
                // auxiliary node to the alpha side whenever either endpoint
                // switches, which is the exact metric construction.
                add_bk_edge(edge.first, auxiliary, e01, expansion_infinity);
                add_bk_edge(edge.second, auxiliary, e10, expansion_infinity);
            }
        }
        graph.solve(source, sink);
        const auto source_side = graph.source_partition(source);
        std::vector<int> proposal = labels;
        for (std::size_t face = 0; face < labels.size(); ++face) {
            if (!source_side[face]) proposal[face] = alpha;
        }
        const float proposal_energy = labeling_energy(proposal, unary, adjacency, edge_costs);
        if (proposal_energy != current_energy &&
            proposal_energy <= current_energy + current_energy / 1000.0F) {
            labels.swap(proposal);
            current_energy = proposal_energy;
        }
    }
    std::cout << "alpha expansion: adjacency=" << adjacency.size()
              << ", recovered directed-edge energy=" << current_energy << '\n';
}

std::vector<std::string> split(const std::string &line) {
    std::istringstream stream(line);
    std::vector<std::string> values;
    for (std::string value; stream >> value;) values.push_back(std::move(value));
    return values;
}

std::uint32_t obj_index(const std::string &value, std::size_t count) {
    const int parsed = std::stoi(value);
    const long long index = parsed > 0 ? parsed - 1LL : static_cast<long long>(count) + parsed;
    if (index < 0 || index >= static_cast<long long>(count)) throw std::runtime_error("OBJ index out of range");
    return static_cast<std::uint32_t>(index);
}

Mesh load_obj(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open OBJ: " + path.string());
    Mesh mesh;
    std::string line;
    while (std::getline(input, line)) {
        auto fields = split(line);
        if (fields.empty()) continue;
        if (fields[0] == "v" && fields.size() >= 4) {
            mesh.vertices.push_back({std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3])});
        } else if (fields[0] == "vt" && fields.size() >= 3) {
            mesh.texcoords.push_back({std::stod(fields[1]), std::stod(fields[2])});
        } else if (fields[0] == "f" && fields.size() == 4) {
            Face face;
            for (int corner = 0; corner < 3; ++corner) {
                const auto slash = fields[corner + 1].find('/');
                if (slash == std::string::npos) throw std::runtime_error("OBJ face has no UV index");
                const auto slash2 = fields[corner + 1].find('/', slash + 1);
                face.vertex[corner] = obj_index(fields[corner + 1].substr(0, slash), mesh.vertices.size());
                face.texcoord[corner] = obj_index(fields[corner + 1].substr(slash + 1, slash2 - slash - 1), mesh.texcoords.size());
            }
            mesh.faces.push_back(face);
        }
    }
    if (mesh.vertices.empty() || mesh.texcoords.empty() || mesh.faces.empty()) {
        throw std::runtime_error("OBJ does not contain a triangular UV mesh");
    }
    return mesh;
}

struct NaturalCharts {
    std::vector<std::vector<std::uint32_t>> faces;
    std::vector<std::uint32_t> id_by_face;
};

NaturalCharts build_natural_charts(const Mesh &mesh, const std::vector<int> &winners) {
    std::vector<std::array<int, 3>> neighbors(mesh.faces.size(), {-1, -1, -1});
    for (const FaceAdjacency &edge : build_face_adjacency(mesh)) {
        neighbors[static_cast<std::size_t>(edge.first)][edge.first_edge] = edge.second;
        neighbors[static_cast<std::size_t>(edge.second)][edge.second_edge] = edge.first;
    }

    // sub_143F41900: union by rank, with equal-rank ties attaching the second
    // root to the first. sub_141F17CC0 visits face edges in slots 0,1,2.
    std::vector<int> parent(mesh.faces.size(), -1);
    std::vector<int> rank(mesh.faces.size(), 0);
    auto find_root = [&parent](int value) {
        int root = value;
        while (parent[static_cast<std::size_t>(root)] != -1) {
            root = parent[static_cast<std::size_t>(root)];
        }
        while (value != root) {
            const int next = parent[static_cast<std::size_t>(value)];
            parent[static_cast<std::size_t>(value)] = root;
            value = next;
        }
        return root;
    };
    auto unite = [&parent, &rank, &find_root](int first, int second) {
        first = find_root(first);
        second = find_root(second);
        if (first == second) return;
        if (rank[static_cast<std::size_t>(first)] == rank[static_cast<std::size_t>(second)]) {
            parent[static_cast<std::size_t>(second)] = first;
            ++rank[static_cast<std::size_t>(first)];
        } else {
            if (rank[static_cast<std::size_t>(first)] < rank[static_cast<std::size_t>(second)]) {
                std::swap(first, second);
            }
            parent[static_cast<std::size_t>(second)] = first;
        }
    };
    for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
        for (int edge = 0; edge < 3; ++edge) {
            const int other = neighbors[face][edge];
            if (other < 0 || winners[face] != winners[static_cast<std::size_t>(other)]) continue;
            unite(static_cast<int>(face), other);
        }
    }

    std::unordered_map<int, std::uint32_t> group_by_root;
    for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
        if (parent[face] == -1) {
            group_by_root.emplace(static_cast<int>(face),
                                  static_cast<std::uint32_t>(group_by_root.size()));
        }
    }
    std::vector<std::vector<std::uint32_t>> winner_groups(group_by_root.size());
    for (std::uint32_t face = 0; face < mesh.faces.size(); ++face) {
        const std::uint32_t chart = group_by_root.at(find_root(static_cast<int>(face)));
        winner_groups[chart].push_back(face);
    }

    NaturalCharts result;
    result.faces = std::move(winner_groups);
    result.id_by_face.resize(mesh.faces.size());
    for (std::uint32_t chart = 0; chart < result.faces.size(); ++chart) {
        for (std::uint32_t face : result.faces[chart]) result.id_by_face[face] = chart;
    }
    return result;
}

bool triangle_contains_pixel_center(const std::array<float, 2> &a,
                                    const std::array<float, 2> &b,
                                    const std::array<float, 2> &c) {
    // Literal boolean specialization of the scanline rasterizer recovered at
    // sub_141F15B20, including its half-open left/right convention.  The
    // caller first performs a binary32 subtraction of 0.5 and, when needed,
    // an integer translation to keep all coordinates non-negative.
    std::array<std::array<double, 2>, 3> p{{
        {static_cast<double>(a[0] - 0.5F), static_cast<double>(a[1] - 0.5F)},
        {static_cast<double>(b[0] - 0.5F), static_cast<double>(b[1] - 0.5F)},
        {static_cast<double>(c[0] - 0.5F), static_cast<double>(c[1] - 0.5F)},
    }};
    const double min_x = std::min({0.0, p[0][0], p[1][0], p[2][0]});
    const double min_y = std::min({0.0, p[0][1], p[1][1], p[2][1]});
    if (min_x < 0.0 || min_y < 0.0) {
        const double dx = std::ceil(-min_x);
        const double dy = std::ceil(-min_y);
        for (auto &vertex : p) {
            vertex[0] += dx;
            vertex[1] += dy;
        }
    }
    if (p[0][1] > p[1][1]) std::swap(p[0], p[1]);
    if (p[1][1] > p[2][1]) std::swap(p[1], p[2]);
    if (p[0][1] > p[1][1]) std::swap(p[0], p[1]);

    int y = static_cast<int>(p[0][1] + 1.0);
    const int middle_y = static_cast<int>(p[1][1] + 1.0);
    const int end_y = static_cast<int>(p[2][1] + 1.0);
    if (y == end_y) return false;
    const double dy01 = p[1][1] - p[0][1];
    const double dy02 = p[2][1] - p[0][1];
    const double dy12 = p[2][1] - p[1][1];
    const double dx01 = (p[1][0] - p[0][0]) / dy01;
    const double dx02 = (p[2][0] - p[0][0]) / dy02;
    const double dx12 = (p[2][0] - p[1][0]) / dy12;
    if ((p[2][0] - p[0][0]) * dy01 - dy02 * (p[1][0] - p[0][0]) == 0.0) {
        return false;
    }
    double short_x = (static_cast<double>(y) - p[0][1]) * dx01 + p[0][0];
    double long_x = (static_cast<double>(y) - p[0][1]) * dx02 + p[0][0];
    auto scan_has_pixel = [](double first, double second) {
        if (first > second) std::swap(first, second);
        const int begin = static_cast<int>(first + 1.0);
        const int end = static_cast<int>(second + 1.0);
        return begin < end;
    };
    for (; y < middle_y; ++y, short_x += dx01, long_x += dx02) {
        if (scan_has_pixel(short_x, long_x)) return true;
    }
    short_x = (static_cast<double>(middle_y) - p[1][1]) * dx12 + p[1][0];
    for (; y < end_y; ++y, short_x += dx12, long_x += dx02) {
        if (scan_has_pixel(short_x, long_x)) return true;
    }
    return false;
}

std::size_t fix_natural_charts_without_pixel_centers(
    std::vector<std::array<float, 2>> &source_uv,
    const std::vector<std::uint32_t> &indices,
    const std::vector<std::uint32_t> &chart_ids,
    std::uint32_t chart_count) {
    std::vector<std::array<float, 2>> shifts(chart_count, {0.0F, 0.0F});
    std::vector<bool> has_pixel_center(chart_count, false);
    for (std::size_t face = 0; face < chart_ids.size(); ++face) {
        const std::uint32_t chart = chart_ids[face];
        const auto &a = source_uv[indices[face * 3U]];
        const auto &b = source_uv[indices[face * 3U + 1U]];
        const auto &c = source_uv[indices[face * 3U + 2U]];
        if (triangle_contains_pixel_center(a, b, c)) {
            has_pixel_center[chart] = true;
            continue;
        }
        auto &shift = shifts[chart];
        const bool no_candidate = shift[0] == 0.0F && shift[1] == 0.0F;
        const bool all_identical = a[0] == b[0] && a[1] == b[1]
                                && b[0] == c[0] && b[1] == c[1];
        if (no_candidate && !all_identical) {
            // Exact float evaluation order recovered from sub_141F17210.
            const float centroid_x = a[0] / 3.0F + b[0] / 3.0F + c[0] / 3.0F;
            const float centroid_y = a[1] / 3.0F + b[1] / 3.0F + c[1] / 3.0F;
            shift[0] = static_cast<float>(static_cast<int>(centroid_x)) + 0.5F - centroid_x;
            shift[1] = static_cast<float>(static_cast<int>(centroid_y)) + 0.5F - centroid_y;
        }
    }

    std::vector<bool> translated_vertex(source_uv.size(), false);
    std::vector<bool> translated_chart(chart_count, false);
    std::size_t corrected = 0;
    for (std::size_t face = 0; face < chart_ids.size(); ++face) {
        const std::uint32_t chart = chart_ids[face];
        const auto shift = shifts[chart];
        if (has_pixel_center[chart] || (shift[0] == 0.0F && shift[1] == 0.0F)) continue;
        if (!translated_chart[chart]) {
            translated_chart[chart] = true;
            ++corrected;
        }
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t vertex = indices[face * 3U + corner];
            if (translated_vertex[vertex]) continue;
            source_uv[vertex][0] += shift[0];
            source_uv[vertex][1] += shift[1];
            translated_vertex[vertex] = true;
        }
    }
    return corrected;
}

void build_natural_uv(Mesh &mesh, const std::vector<Camera> &cameras,
                      const std::vector<int> &winners, int texture_size, int downscale) {
    const auto charts = build_natural_charts(mesh, winners);
    std::vector<NaturalUvChart> primary_charts(charts.faces.size());
    std::vector<NaturalUvChart> no_camera_initial(charts.faces.size());
    std::vector<std::vector<NaturalUvChart>> rebuilt_by_component(charts.faces.size());
    std::vector<std::string> worker_errors(charts.faces.size());
    std::vector<int> worker_thread_ids(charts.faces.size(), -1);
    std::vector<std::size_t> worker_rebuilt_counts(charts.faces.size(), 0U);
    std::vector<std::size_t> completion_order;
    completion_order.reserve(charts.faces.size());
    std::vector<std::size_t> no_camera_component_indices;
    no_camera_component_indices.reserve(charts.faces.size());
    std::vector<NaturalUvChart> appended_charts;
    std::size_t no_camera_components = 0U;
    std::size_t natural_worker_steps = 0U;
    for (std::size_t component = 0; component < charts.faces.size(); ++component) {
        const std::vector<std::uint32_t> &component_faces = charts.faces[component];
        const int winner = winners[component_faces.front()];
        NaturalUvChart initial;
        std::unordered_map<std::uint32_t, std::uint32_t> vertex_map;
        for (const std::uint32_t face_index : component_faces) {
            const Face &face = mesh.faces[face_index];
            initial.source_faces.push_back(face_index);
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                const std::uint32_t source_vertex = face.vertex[corner];
                const auto [iterator, inserted] = vertex_map.emplace(
                    source_vertex,
                    static_cast<std::uint32_t>(initial.source_vertices.size()));
                if (inserted) {
                    const Vec3d &position = mesh.vertices[source_vertex];
                    initial.source_vertices.push_back(source_vertex);
                    initial.vertex_xyz.insert(
                        initial.vertex_xyz.end(),
                        {static_cast<float>(position.x),
                         static_cast<float>(position.y),
                         static_cast<float>(position.z)});
                    if (winner >= 0) {
                        const auto projection = project_frame(
                            cameras[static_cast<std::size_t>(winner)].pose,
                            cameras[static_cast<std::size_t>(winner)].calibration,
                            position, false);
                        if (!projection) {
                            throw std::runtime_error(
                                "natural UV camera projection failed");
                        }
                        initial.uv.insert(
                            initial.uv.end(),
                            {static_cast<float>(projection->x / downscale),
                             static_cast<float>(projection->y / downscale)});
                    }
                }
                initial.triangle_indices.push_back(iterator->second);
            }
        }
        if (winner >= 0) {
            primary_charts[component] = std::move(initial);
            continue;
        }

        ++no_camera_components;
        no_camera_component_indices.push_back(component);
        no_camera_initial[component] = std::move(initial);
    }

    // Native dispatches the independent no-camera components concurrently,
    // keeps each component's first chart in its fixed slot, and appends extra
    // split charts in task-completion order.  That order affects xatlas's
    // otherwise deterministic packing, so preserve the same producer shape.
    const char *serial_uv = std::getenv("METASHAPE_TEXTURE_NATURAL_UV_SERIAL");
    const bool parallel_components =
        serial_uv == nullptr || std::string(serial_uv) != "1";
#pragma omp parallel for schedule(dynamic, 1) if(parallel_components)
    for (std::int64_t no_camera_index = 0;
         no_camera_index <
             static_cast<std::int64_t>(no_camera_component_indices.size());
         ++no_camera_index) {
        const std::size_t component = no_camera_component_indices[
            static_cast<std::size_t>(no_camera_index)];
#if defined(_OPENMP)
        worker_thread_ids[component] = omp_get_thread_num();
#else
        worker_thread_ids[component] = 0;
#endif
        const std::vector<std::uint32_t> &component_faces = charts.faces[component];
        try {
            NaturalUvWorkerResult worker =
                generate_natural_uv_charts(std::move(no_camera_initial[component]));
            std::vector<NaturalUvChart> rebuilt =
                rebuild_natural_uv_charts_for_packing(component_faces, worker.charts);
            if (rebuilt.empty()) {
                throw std::runtime_error(
                    "natural UV worker produced no packing chart");
            }
            for (NaturalUvChart &chart : rebuilt) {
                scale_natural_uv_chart_for_packing(
                    chart, static_cast<std::uint32_t>(texture_size));
            }
            const std::size_t worker_step_count = worker.history.size();
            worker_rebuilt_counts[component] = rebuilt.size();
#pragma omp atomic update
            natural_worker_steps += worker_step_count;
            rebuilt_by_component[component] = std::move(rebuilt);
#pragma omp critical(metashape_natural_uv_completion)
            completion_order.push_back(component);
        } catch (const std::exception &error) {
            std::ostringstream message;
            message << "natural UV component " << component << " ("
                    << component_faces.size() << " faces, first face "
                    << component_faces.front() << "): " << error.what();
            worker_errors[component] = message.str();
        }
    }
    for (const std::string &error : worker_errors) {
        if (!error.empty()) throw std::runtime_error(error);
    }
    for (const std::size_t component : completion_order) {
        std::vector<NaturalUvChart> &rebuilt = rebuilt_by_component[component];
        primary_charts[component] = std::move(rebuilt.front());
        for (std::size_t index = 1U; index < rebuilt.size(); ++index) {
            appended_charts.push_back(std::move(rebuilt[index]));
        }
    }

    std::vector<NaturalUvChart> packing_charts;
    packing_charts.reserve(primary_charts.size() + appended_charts.size());
    for (NaturalUvChart &chart : primary_charts) {
        packing_charts.push_back(std::move(chart));
    }
    for (NaturalUvChart &chart : appended_charts) {
        packing_charts.push_back(std::move(chart));
    }
    const std::uint32_t chart_count =
        static_cast<std::uint32_t>(packing_charts.size());

    std::vector<std::array<float, 2>> source_uv;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> packed_face_order;
    std::vector<std::uint32_t> packed_chart_ids;
    indices.reserve(mesh.faces.size() * 3U);
    packed_face_order.reserve(mesh.faces.size());
    packed_chart_ids.reserve(mesh.faces.size());
    for (std::uint32_t chart = 0; chart < chart_count; ++chart) {
        const NaturalUvChart &packing_chart = packing_charts[chart];
        const std::uint32_t vertex_offset =
            static_cast<std::uint32_t>(source_uv.size());
        for (std::size_t vertex = 0; vertex < packing_chart.uv.size() / 2U; ++vertex) {
            source_uv.push_back(
                {packing_chart.uv[2U * vertex], packing_chart.uv[2U * vertex + 1U]});
        }
        for (std::size_t face = 0; face < packing_chart.source_faces.size(); ++face) {
            const std::uint32_t face_index = packing_chart.source_faces[face];
            packed_face_order.push_back(face_index);
            packed_chart_ids.push_back(chart);
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                indices.push_back(
                    vertex_offset + packing_chart.triangle_indices[3U * face + corner]);
            }
        }
    }

    const std::size_t pixel_center_corrected =
        fix_natural_charts_without_pixel_centers(
            source_uv, indices, packed_chart_ids, chart_count);
    if (const char *replay_root = std::getenv("METASHAPE_TEXTURE_XATLAS_INPUT_REPLAY")) {
        const std::filesystem::path root(replay_root);
        auto read_raw = [&root](const char *name, auto &values) {
            std::ifstream stream(root / name, std::ios::binary | std::ios::ate);
            if (!stream) throw std::runtime_error("cannot open xatlas replay input");
            const auto bytes = stream.tellg();
            if (bytes != static_cast<std::streamoff>(values.size() * sizeof(values[0]))) {
                throw std::runtime_error("xatlas replay input has the wrong size");
            }
            stream.seekg(0);
            stream.read(reinterpret_cast<char *>(values.data()), bytes);
            if (!stream) throw std::runtime_error("cannot read xatlas replay input");
        };
        read_raw("00-uv_f32.raw", source_uv);
        read_raw("00-indices_u32.raw", indices);
        read_raw("00-chart_faces_u32.raw", packed_face_order);
        std::vector<std::uint64_t> offsets(static_cast<std::size_t>(chart_count) + 1U);
        read_raw("00-chart_offsets_u64.raw", offsets);
        for (std::uint32_t chart = 0; chart < chart_count; ++chart) {
            for (std::uint64_t face = offsets[chart]; face < offsets[chart + 1U]; ++face) {
                packed_chart_ids[static_cast<std::size_t>(face)] = chart;
            }
        }
    }
    if (const char *dump_root = std::getenv("METASHAPE_TEXTURE_XATLAS_INPUT_DUMP")) {
        const std::filesystem::path root(dump_root);
        std::filesystem::create_directories(root);
        auto write_raw = [&root](const char *name, const auto &values) {
            std::ofstream stream(root / name, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot create xatlas input dump");
            stream.write(reinterpret_cast<const char *>(values.data()),
                         static_cast<std::streamsize>(values.size() * sizeof(values[0])));
            if (!stream) throw std::runtime_error("cannot write xatlas input dump");
        };
        write_raw("uv_f32.raw", source_uv);
        write_raw("indices_u32.raw", indices);
        write_raw("materials_u32.raw", packed_chart_ids);
        write_raw("face_order_u32.raw", packed_face_order);
        nlohmann::json scheduling;
        scheduling["openmp_enabled"] =
#if defined(_OPENMP)
            true;
#else
            false;
#endif
        scheduling["parallel_components"] = parallel_components;
        scheduling["schedule"] = "dynamic,1";
        scheduling["no_camera_components"] = nlohmann::json::array();
        std::vector<std::size_t> completion_rank(charts.faces.size(),
                                                 charts.faces.size());
        for (std::size_t rank = 0; rank < completion_order.size(); ++rank) {
            completion_rank[completion_order[rank]] = rank;
        }
        for (const std::size_t component : no_camera_component_indices) {
            scheduling["no_camera_components"].push_back({
                {"component", component},
                {"first_face", charts.faces[component].front()},
                {"faces", charts.faces[component].size()},
                {"thread", worker_thread_ids[component]},
                {"completion_rank", completion_rank[component]},
                {"rebuilt_charts", worker_rebuilt_counts[component]},
            });
        }
        std::ofstream scheduling_stream(root / "natural_uv_scheduling.json");
        if (!scheduling_stream) {
            throw std::runtime_error("cannot create natural UV scheduling dump");
        }
        scheduling_stream << std::setw(2) << scheduling << '\n';
    }
    if (const char *dump_only =
            std::getenv("METASHAPE_TEXTURE_XATLAS_INPUT_DUMP_ONLY")) {
        if (std::string(dump_only) == "1") {
            if (std::getenv("METASHAPE_TEXTURE_XATLAS_INPUT_DUMP") == nullptr) {
                throw std::runtime_error(
                    "METASHAPE_TEXTURE_XATLAS_INPUT_DUMP_ONLY requires "
                    "METASHAPE_TEXTURE_XATLAS_INPUT_DUMP");
            }
            return;
        }
    }

    xatlas::Atlas *atlas = xatlas::Create();
    if (atlas == nullptr) throw std::runtime_error("xatlas::Create failed");
    try {
        xatlas::UvMeshDecl declaration{};
        declaration.vertexUvData = source_uv.data();
        declaration.vertexCount = static_cast<std::uint32_t>(source_uv.size());
        declaration.vertexStride = sizeof(source_uv[0]);
        declaration.indexData = indices.data();
        declaration.indexCount = static_cast<std::uint32_t>(indices.size());
        declaration.indexFormat = xatlas::IndexFormat::UInt32;
        // Native UvMeshDecl is 56 bytes: +16 is an ordered
        // std::vector<std::vector<uint32_t>> face-group pointer and +24 is the
        // standard material pointer (null on this path).  The public
        // Korinin38 fork has the recovered PackOptions but not faceGroupData;
        // its faceMaterialData groups the same already-contiguous faces and is
        // a build-compatible fallback for diagnostics.
        std::vector<std::vector<std::uint32_t>> xatlas_face_groups(chart_count);
        for (std::uint32_t face = 0; face < packed_chart_ids.size(); ++face) {
            xatlas_face_groups[packed_chart_ids[face]].push_back(face);
        }
        [&]<typename Declaration>(Declaration &target) {
            if constexpr (requires(Declaration value) { value.faceGroupData; }) {
                static_assert(sizeof(Declaration) == 56);
                target.faceGroupData = &xatlas_face_groups;
                target.faceMaterialData = nullptr;
            } else {
                target.faceMaterialData = packed_chart_ids.data();
            }
        }(declaration);
        const auto error = xatlas::AddUvMesh(atlas, declaration);
        if (error != xatlas::AddMeshError::Success) {
            throw std::runtime_error(std::string("xatlas::AddUvMesh failed: ") +
                                     xatlas::StringForEnum(error));
        }
        xatlas::ComputeCharts(atlas);
        xatlas::PackOptions options{};
        // Native caps the packing canvas at 2047 pixels even when the target
        // texture is 4K/8K, then normalizes the packed pixel coordinates by
        // the requested texture size.  This leaves the unused high-resolution
        // area intact while preserving the source-photo texel density.
        const std::uint32_t packing_resolution = texture_size > 2048
            ? 2047U
            : static_cast<std::uint32_t>(texture_size);
        // Captured verbatim at native xatlas::PackCharts. The SysV by-value
        // PackOptions object starts at rsp+8; byte +23 is transposeCharts.
        options.maxChartSize = 0;
        options.padding = 2;
        options.texelsPerUnit = 1.0F;
        options.resolution = packing_resolution;
        options.bilinear = false;
        options.blockAlign = true;
        options.bruteForce = false;
        options.randomUseBruteForce = false;
        options.createImage = false;
        options.rotateChartsToAxis = false;
        options.rotateCharts = false;
        options.transposeCharts = false;
        options.preserveInputTexcoordsFractionalPart = true;
        options.coarseLevels = 2;
        options.skipSpeedup = true;
        options.gridSpeedup = true;
        options.usePreviousPositionOffset = 0.04F;
        options.useOpenMP = true;
        xatlas::PackCharts(atlas, options);
        if (atlas->atlasCount != 1 || atlas->meshCount != 1 ||
            atlas->width != packing_resolution || atlas->height != packing_resolution) {
            throw std::runtime_error("xatlas did not produce one requested-size atlas");
        }
        const xatlas::Mesh &output = atlas->meshes[0];
        if (output.indexCount != mesh.faces.size() * 3U) {
            throw std::runtime_error("xatlas output index count differs from mesh");
        }
        mesh.texcoords.clear();
        mesh.texcoords.reserve(output.vertexCount);
        for (std::uint32_t index = 0; index < output.vertexCount; ++index) {
            mesh.texcoords.push_back({
                output.vertexArray[index].uv[0] / static_cast<double>(texture_size),
                output.vertexArray[index].uv[1] / static_cast<double>(texture_size),
            });
        }
        for (std::size_t packed_face = 0; packed_face < mesh.faces.size(); ++packed_face) {
            const std::size_t face = packed_face_order[packed_face];
            for (std::size_t corner = 0; corner < 3; ++corner) {
                mesh.faces[face].texcoord[corner] =
                    output.indexArray[packed_face * 3U + corner];
            }
        }
        std::cout << "natural UV: " << chart_count << " charts, "
                  << source_uv.size() << " source UV vertices, "
                  << output.vertexCount << " packed UV vertices, "
                  << texture_size << 'x' << texture_size << " target, "
                  << atlas->width << 'x' << atlas->height << " packing canvas, "
                  << pixel_center_corrected << " charts moved to a pixel center, "
                  << no_camera_components << " no-camera components, "
                  << natural_worker_steps << " worker steps\n";
    } catch (...) {
        xatlas::Destroy(atlas);
        throw;
    }
    xatlas::Destroy(atlas);
}

void apply_project_float32_geometry(
    Mesh &mesh, const std::filesystem::path &project_input_directory) {
    const auto vertices_path = project_input_directory / "vertices_xyz_f32.raw";
    const auto faces_path = project_input_directory / "faces_abc_u32.raw";
    std::ifstream vertices_input(vertices_path, std::ios::binary | std::ios::ate);
    std::ifstream faces_input(faces_path, std::ios::binary | std::ios::ate);
    if (!vertices_input || !faces_input) {
        throw std::runtime_error("cannot open project float32 mesh geometry");
    }
    const auto vertex_bytes = vertices_input.tellg();
    const auto face_bytes = faces_input.tellg();
    if (vertex_bytes != static_cast<std::streamoff>(mesh.vertices.size() * 3U * sizeof(float)) ||
        face_bytes != static_cast<std::streamoff>(mesh.faces.size() * 3U * sizeof(std::uint32_t))) {
        throw std::runtime_error("project mesh geometry dimensions do not match OBJ");
    }
    vertices_input.seekg(0);
    faces_input.seekg(0);
    std::vector<float> vertices(mesh.vertices.size() * 3U);
    std::vector<std::uint32_t> faces(mesh.faces.size() * 3U);
    vertices_input.read(reinterpret_cast<char *>(vertices.data()), vertex_bytes);
    faces_input.read(reinterpret_cast<char *>(faces.data()), face_bytes);
    if (!vertices_input || !faces_input) {
        throw std::runtime_error("cannot read project float32 mesh geometry");
    }
    for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
        for (std::size_t corner = 0; corner < 3; ++corner) {
            if (faces[face * 3U + corner] != mesh.faces[face].vertex[corner]) {
                throw std::runtime_error("project mesh face order does not match OBJ");
            }
        }
    }
    for (std::size_t vertex = 0; vertex < mesh.vertices.size(); ++vertex) {
        mesh.vertices[vertex] = {
            static_cast<double>(vertices[vertex * 3U]),
            static_cast<double>(vertices[vertex * 3U + 1U]),
            static_cast<double>(vertices[vertex * 3U + 2U]),
        };
    }
    const auto texcoords_path = project_input_directory / "face_texcoord_uv_f32.raw";
    if (std::filesystem::is_regular_file(texcoords_path)) {
        std::ifstream texcoords_input(texcoords_path, std::ios::binary | std::ios::ate);
        const auto texcoord_bytes = texcoords_input.tellg();
        const auto expected_bytes = static_cast<std::streamoff>(
            mesh.faces.size() * 3U * 2U * sizeof(float));
        if (texcoord_bytes != expected_bytes) {
            throw std::runtime_error("project face texcoord dimensions do not match OBJ");
        }
        std::vector<float> texcoords(mesh.faces.size() * 3U * 2U);
        texcoords_input.seekg(0);
        texcoords_input.read(reinterpret_cast<char *>(texcoords.data()), texcoord_bytes);
        if (!texcoords_input) {
            throw std::runtime_error("cannot read project float32 face texcoords");
        }
        mesh.texcoords.clear();
        mesh.texcoords.reserve(mesh.faces.size() * 3U);
        struct ProjectUvKey {
            std::uint32_t vertex;
            std::uint32_t u;
            std::uint32_t v;
            bool operator==(const ProjectUvKey &) const = default;
        };
        struct ProjectUvKeyHash {
            std::size_t operator()(const ProjectUvKey &key) const noexcept {
                std::size_t hash = key.vertex;
                hash ^= static_cast<std::size_t>(key.u) + 0x9e3779b9U +
                        (hash << 6U) + (hash >> 2U);
                hash ^= static_cast<std::size_t>(key.v) + 0x9e3779b9U +
                        (hash << 6U) + (hash >> 2U);
                return hash;
            }
        };
        std::unordered_map<ProjectUvKey, std::uint32_t, ProjectUvKeyHash> uv_by_key;
        for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                const std::size_t source = (face * 3U + corner) * 2U;
                const ProjectUvKey key{
                    mesh.faces[face].vertex[corner],
                    std::bit_cast<std::uint32_t>(texcoords[source]),
                    std::bit_cast<std::uint32_t>(texcoords[source + 1U]),
                };
                const auto [entry, inserted] = uv_by_key.emplace(
                    key, static_cast<std::uint32_t>(mesh.texcoords.size()));
                if (inserted) {
                    mesh.texcoords.push_back({
                        static_cast<double>(texcoords[source]),
                        static_cast<double>(texcoords[source + 1U]),
                    });
                }
                mesh.faces[face].texcoord[corner] = entry->second;
            }
        }
    }
}

cv::Mat prepare_image_downscale_2x(const cv::Mat &input) {
    if (input.type() != CV_32FC3) {
        throw std::invalid_argument("prepared image source must be CV_32FC3");
    }
    const float center_weight = std::bit_cast<float>(std::uint32_t{0x3f38bf32});
    const float near_weight = std::bit_cast<float>(std::uint32_t{0x3e0d8337});
    const float far_weight = std::bit_cast<float>(std::uint32_t{0x3a7e6521});
    cv::Mat horizontal(input.size(), CV_32FC3);
    cv::Mat filtered(input.size(), CV_32FC3);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = input.at<cv::Vec3f>(y, x)[channel] * center_weight;
                value += (input.at<cv::Vec3f>(y, std::max(0, x - 1))[channel] +
                          input.at<cv::Vec3f>(y, std::min(input.cols - 1, x + 1))[channel]) *
                         near_weight;
                value += (input.at<cv::Vec3f>(y, std::max(0, x - 2))[channel] +
                          input.at<cv::Vec3f>(y, std::min(input.cols - 1, x + 2))[channel]) *
                         far_weight;
                horizontal.at<cv::Vec3f>(y, x)[channel] = value;
            }
        }
    }
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = horizontal.at<cv::Vec3f>(y, x)[channel] * center_weight;
                value += (horizontal.at<cv::Vec3f>(std::max(0, y - 1), x)[channel] +
                          horizontal.at<cv::Vec3f>(std::min(input.rows - 1, y + 1), x)[channel]) *
                         near_weight;
                value += (horizontal.at<cv::Vec3f>(std::max(0, y - 2), x)[channel] +
                          horizontal.at<cv::Vec3f>(std::min(input.rows - 1, y + 2), x)[channel]) *
                         far_weight;
                filtered.at<cv::Vec3f>(y, x)[channel] = value;
            }
        }
    }
    cv::Mat filtered_u8(input.size(), CV_8UC3);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                filtered_u8.at<cv::Vec3b>(y, x)[channel] = static_cast<std::uint8_t>(
                    std::clamp(filtered.at<cv::Vec3f>(y, x)[channel] * 255.0F + 0.5F,
                               0.0F, 255.0F));
            }
        }
    }
    cv::Mat downscaled((input.rows + 1) / 2, (input.cols + 1) / 2, CV_32FC3);
    for (int y = 0; y < downscaled.rows; ++y) {
        for (int x = 0; x < downscaled.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                int sum = 0;
                int count = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int source_x = 2 * x + dx;
                        const int source_y = 2 * y + dy;
                        if (source_x >= input.cols || source_y >= input.rows) continue;
                        sum += filtered_u8.at<cv::Vec3b>(source_y, source_x)[channel];
                        ++count;
                    }
                }
                const auto value = static_cast<std::uint8_t>(
                    static_cast<float>(sum) / static_cast<float>(count) + 0.5F);
                downscaled.at<cv::Vec3f>(y, x)[channel] =
                    static_cast<float>(value) / 255.0F;
            }
        }
    }
    return downscaled;
}

cv::Mat downscale_6tap(const cv::Mat &input, int factor) {
    cv::Mat current = input;
    static constexpr std::array<float, 6> kernel = {1, 5, 10, 10, 5, 1};
    int remaining = factor;
    while (remaining > 1) {
        if ((remaining & 1) != 0) throw std::invalid_argument("only power-of-two downscale is supported");
        const int width = (current.cols + 1) / 2;
        const int height = (current.rows + 1) / 2;
        cv::Mat output(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv::Vec3f sum(0, 0, 0);
                float normalization = 0.0F;
                for (int ky = 0; ky < 6; ++ky) {
                    const int sy = 2 * y + ky - 2;
                    if (sy < 0 || sy >= current.rows) continue;
                    for (int kx = 0; kx < 6; ++kx) {
                        const int sx = 2 * x + kx - 2;
                        if (sx < 0 || sx >= current.cols) continue;
                        const float weight = kernel[kx] * kernel[ky];
                        sum += current.at<cv::Vec3f>(sy, sx) * weight;
                        normalization += weight;
                    }
                }
                cv::Vec3f value = sum / normalization;
                for (int channel = 0; channel < 3; ++channel) {
                    value[channel] = std::nearbyint(
                        std::clamp(value[channel], 0.0F, 1.0F) * 255.0F) / 255.0F;
                }
                output.at<cv::Vec3f>(y, x) = value;
            }
        }
        current = std::move(output);
        remaining /= 2;
    }
    return current;
}

cv::Mat downscale_6tap_scalar(const cv::Mat &input, int factor) {
    cv::Mat current = input;
    static constexpr std::array<float, 6> kernel = {1, 5, 10, 10, 5, 1};
    int remaining = factor;
    while (remaining > 1) {
        if ((remaining & 1) != 0) throw std::invalid_argument("only power-of-two downscale is supported");
        const int width = (current.cols + 1) / 2;
        const int height = (current.rows + 1) / 2;
        cv::Mat output(height, width, CV_32F, cv::Scalar(0));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0F;
                float normalization = 0.0F;
                for (int ky = 0; ky < 6; ++ky) {
                    const int sy = 2 * y + ky - 2;
                    if (sy < 0 || sy >= current.rows) continue;
                    for (int kx = 0; kx < 6; ++kx) {
                        const int sx = 2 * x + kx - 2;
                        if (sx < 0 || sx >= current.cols) continue;
                        const float weight = kernel[kx] * kernel[ky];
                        sum += current.at<float>(sy, sx) * weight;
                        normalization += weight;
                    }
                }
                output.at<float>(y, x) = sum / normalization;
            }
        }
        current = std::move(output);
        remaining /= 2;
    }
    return current;
}

cv::Mat downscale_2x2_average_scalar(const cv::Mat &input, int factor) {
    cv::Mat current = input;
    int remaining = factor;
    while (remaining > 1) {
        if ((remaining & 1) != 0) {
            throw std::invalid_argument("only power-of-two downscale is supported");
        }
        const int width = (current.cols + 1) / 2;
        const int height = (current.rows + 1) / 2;
        cv::Mat output(height, width, CV_32F, cv::Scalar(0));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0F;
                int count = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int sx = 2 * x + dx;
                        const int sy = 2 * y + dy;
                        if (sx >= current.cols || sy >= current.rows) continue;
                        sum += current.at<float>(sy, sx);
                        ++count;
                    }
                }
                output.at<float>(y, x) = sum / static_cast<float>(count);
            }
        }
        current = std::move(output);
        remaining /= 2;
    }
    return current;
}

cv::Mat sharpen_image(const cv::Mat &input, float strength) {
    cv::Mat output(input.size(), CV_32FC3);
    auto load = [&](int x, int y) -> cv::Vec3f {
        return input.at<cv::Vec3f>(std::clamp(y, 0, input.rows - 1), std::clamp(x, 0, input.cols - 1));
    };
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            const auto center = load(x, y);
            const auto left = load(x - 1, y);
            const auto right = load(x + 1, y);
            const auto up = load(x, y - 1);
            const auto down = load(x, y + 1);
            const float minimum = std::min({center[1], left[1], right[1], up[1], down[1]});
            const float maximum = std::max({center[1], left[1], right[1], up[1], down[1]});
            const float adapt = std::sqrt(std::max(0.0F, std::min(1.0F - maximum, minimum)) /
                                          std::max(maximum, 1.0e-6F));
            const float coefficient = adapt * ((1.0F - strength) * -0.125F + strength * -0.2F);
            cv::Vec3f value =
                (center + coefficient * (left + right + up + down)) / (1.0F + 4.0F * coefficient);
            for (int channel = 0; channel < 3; ++channel) {
                value[channel] = std::nearbyint(
                    std::clamp(value[channel], 0.0F, 1.0F) * 255.0F) / 255.0F;
            }
            output.at<cv::Vec3f>(y, x) = value;
        }
    }
    return output;
}

std::vector<Camera> load_scene(const std::filesystem::path &path, int downscale,
                               float sharpening, bool load_images = true) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open scene JSON: " + path.string());
    nlohmann::json scene;
    input >> scene;
    std::map<std::uint32_t, FrameCalibration> sensors;
    for (const auto &sensor : scene.at("sensors")) {
        const auto &c = sensor.at("calibration");
        FrameCalibration calibration;
        calibration.width = sensor.at("width").get<int>();
        calibration.height = sensor.at("height").get<int>();
        auto number = [&](const char *key) { return c.contains(key) ? c.at(key).get<double>() : 0.0; };
        calibration.f = number("f"); calibration.cx = number("cx"); calibration.cy = number("cy");
        calibration.b1 = number("b1"); calibration.b2 = number("b2");
        calibration.k1 = number("k1"); calibration.k2 = number("k2");
        calibration.k3 = number("k3"); calibration.k4 = number("k4");
        calibration.p1 = number("p1"); calibration.p2 = number("p2");
        calibration.p3 = number("p3"); calibration.p4 = number("p4");
        sensors.emplace(sensor.at("key").get<std::uint32_t>(), calibration);
    }
    std::vector<Camera> cameras;
    const auto resolve_scene_path = [&](const std::filesystem::path &candidate) {
        return candidate.is_absolute() ? candidate : path.parent_path() / candidate;
    };
    for (const auto &entry : scene.at("cameras")) {
        if (!entry.value("enabled", true)) continue;
        Camera camera;
        camera.label = entry.at("label").get<std::string>();
        camera.photo_path = resolve_scene_path(
            std::filesystem::path(entry.at("photo_path").get<std::string>()));
        camera.calibration = sensors.at(entry.at("sensor_key").get<std::uint32_t>());
        const auto &transform = entry.at("transform");
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) camera.pose.rotation[row * 3 + column] = transform[row][column].get<double>();
        }
        camera.pose.center = {transform[0][3].get<double>(), transform[1][3].get<double>(), transform[2][3].get<double>()};
        if (!load_images) {
            cameras.push_back(std::move(camera));
            continue;
        }
        auto read_image = [](const std::filesystem::path &image_path, int flags) {
            std::ifstream stream(image_path, std::ios::binary);
            if (!stream) return cv::Mat{};
            std::vector<std::uint8_t> bytes(
                (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
            return cv::imdecode(bytes, flags);
        };
        cv::Mat encoded = read_image(camera.photo_path, cv::IMREAD_COLOR);
        if (encoded.empty()) throw std::runtime_error("cannot load photo: " + camera.photo_path.string());
        encoded.convertTo(camera.image, CV_32FC3, 1.0 / 255.0);
        if (std::getenv("METASHAPE_TEXTURE_IMAGE_PYRAMID_ONLY_DIR") == nullptr) {
            camera.quality_image = downscale_6tap(camera.image, downscale * 2);
        }
        cv::Mat downscaled_image = downscale == 2
            ? prepare_image_downscale_2x(camera.image)
            : downscale_6tap(camera.image, downscale);
        if (const char *diagnostic_root =
                std::getenv("METASHAPE_TEXTURE_IMAGE_PYRAMID_ONLY_DIR")) {
            const std::filesystem::path root(diagnostic_root);
            std::filesystem::create_directories(root);
            cv::Mat prepared_u8;
            downscaled_image.convertTo(prepared_u8, CV_8UC3, 255.0);
            std::vector<cv::Mat> bgr;
            cv::split(prepared_u8, bgr);
            std::ostringstream name;
            name << "camera_" << std::setw(3) << std::setfill('0') << cameras.size()
                 << "_unsharpened_" << downscaled_image.cols << 'x'
                 << downscaled_image.rows << "_r8_array3.raw";
            std::ofstream stream(root / name.str(), std::ios::binary);
            if (!stream) throw std::runtime_error("cannot create prepared-image dump");
            for (auto channel = bgr.rbegin(); channel != bgr.rend(); ++channel) {
                stream.write(reinterpret_cast<const char *>(channel->data),
                             static_cast<std::streamsize>(channel->total()));
            }
            if (!stream) throw std::runtime_error("cannot write prepared-image dump");
        }
        camera.image = sharpen_image(downscaled_image, sharpening);
        if (entry.contains("texture_consistency_path")) {
            const auto consistency_path = resolve_scene_path(std::filesystem::path(
                entry.at("texture_consistency_path").get<std::string>()));
            camera.consistency_map = read_image(consistency_path, cv::IMREAD_GRAYSCALE);
            if (camera.consistency_map.empty()) {
                throw std::runtime_error("cannot load texture consistency map: " +
                                         consistency_path.string());
            }
        }
        if (entry.contains("focus_quality_path")) {
            const auto focus_path = resolve_scene_path(std::filesystem::path(
                entry.at("focus_quality_path").get<std::string>()));
            camera.focus_quality_map = read_image(focus_path, cv::IMREAD_GRAYSCALE);
            if (camera.focus_quality_map.empty()) {
                throw std::runtime_error("cannot load focus quality map: " +
                                         focus_path.string());
            }
            if (camera.focus_quality_map.cols * 4 != camera.calibration.width ||
                camera.focus_quality_map.rows * 4 != camera.calibration.height) {
                throw std::runtime_error("focus quality map must be sensor dimensions / 4: " +
                                         focus_path.string());
            }
        }
        if (entry.contains("mask_path")) {
            const auto mask_path = resolve_scene_path(
                std::filesystem::path(entry.at("mask_path").get<std::string>()));
            camera.mask = read_image(mask_path, cv::IMREAD_GRAYSCALE);
            if (camera.mask.empty()) {
                throw std::runtime_error("cannot load camera mask: " + mask_path.string());
            }
            if (camera.mask.cols != camera.calibration.width ||
                camera.mask.rows != camera.calibration.height) {
                throw std::runtime_error("camera mask dimensions do not match sensor: " +
                                         mask_path.string());
            }
        }
        cameras.push_back(std::move(camera));
    }
    if (cameras.empty()) throw std::runtime_error("scene contains no enabled cameras");
    return cameras;
}

double triangle_area(Vec2d a, Vec2d b, Vec2d c) {
    return 0.5 * std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

ProjectedFace project_face(const Mesh &mesh, const Face &face, const Camera &camera) {
    ProjectedFace result;
    std::array<Vec3d, 3> local{};
    for (int corner = 0; corner < 3; ++corner) {
        const Vec3d world = mesh.vertices[face.vertex[corner]];
        local[corner] = world_to_camera(camera.pose, world);
        auto pixel = project_frame(camera.pose, camera.calibration, world, false);
        if (!pixel) return result;
        result.pixel[corner] = *pixel;
        result.depth[corner] = local[corner].z;
    }
    result.view_dot = face_view_dot(local[0], local[1], local[2]);
    result.area = triangle_area(result.pixel[0], result.pixel[1], result.pixel[2]);
    result.valid = result.view_dot <= 0.0 && result.area > 0.0;
    return result;
}

struct GpuProjectedPoint {
    Vec2d pixel{};
    double depth = 0.0;
};

// The recovered occlusion-edge shaders consume R32 vertex buffers and a
// float camera block.  Keep this path separate from the normal double
// precision camera helpers so experiments can attribute differences to the
// native GPU arithmetic alone.
std::optional<GpuProjectedPoint> project_frame_gpu_float(const Camera &camera,
                                                          Vec3d world) {
    const CameraPose &pose = camera.pose;
    const FrameCalibration &calibration = camera.calibration;
    const float wx = static_cast<float>(world.x);
    const float wy = static_cast<float>(world.y);
    const float wz = static_cast<float>(world.z);
    const float dx = wx - static_cast<float>(pose.center.x);
    const float dy = wy - static_cast<float>(pose.center.y);
    const float dz = wz - static_cast<float>(pose.center.z);
    const float scale = static_cast<float>(pose.scale);
    if (scale == 0.0F) throw std::invalid_argument("camera pose has zero scale");
    const auto r = [&](int index) { return static_cast<float>(pose.rotation[index]); };
    const bool fma_camera =
        std::getenv("METASHAPE_TEXTURE_OCCLUSION_FMA_CAMERA") != nullptr;
    const auto dot3 = [&](float a, float b, float c) {
        return fma_camera ? std::fma(a, dx, std::fma(b, dy, c * dz))
                          : a * dx + b * dy + c * dz;
    };
    const float local_x = dot3(r(0), r(3), r(6)) / scale;
    const float local_y = dot3(r(1), r(4), r(7)) / scale;
    const float local_z = dot3(r(2), r(5), r(8)) / scale;
    if (local_z <= 0.0F) return std::nullopt;

    const float x = local_x / local_z;
    const float y = local_y / local_z;
    const float maximum_radius_squared =
        static_cast<float>(calibration.maximum_radius_squared);
    const float raw_radius_squared = std::fma(x, x, y * y);
    const bool outside_radius = raw_radius_squared > maximum_radius_squared;
    const float radius_squared = outside_radius ? maximum_radius_squared : raw_radius_squared;
    const float radius_fourth = radius_squared * radius_squared;
    const float k1 = static_cast<float>(calibration.k1);
    const float k2 = static_cast<float>(calibration.k2);
    const float k3 = static_cast<float>(calibration.k3);
    const float k4 = static_cast<float>(calibration.k4);
    const float p1 = static_cast<float>(calibration.p1);
    const float p2 = static_cast<float>(calibration.p2);
    const float p3 = static_cast<float>(calibration.p3);
    const float p4 = static_cast<float>(calibration.p4);
    const float radial = std::fma(k4 * radius_fourth, radius_fourth,
                         std::fma(k3 * radius_squared, radius_fourth,
                         std::fma(k1, radius_squared, k2 * radius_fourth)));
    const float tangential_scale = std::fma(p4, radius_fourth,
                                           std::fma(p3, radius_squared, 1.0F));
    const float radius_clamp = outside_radius
        ? maximum_radius_squared / raw_radius_squared : 1.0F;
    const float tangential_x = std::fma(
        p1, std::fma(y, y, (3.0F * x) * x), ((p2 * 2.0F) * x) * y);
    const float tangential_y = std::fma(
        p2, std::fma(x, x, (3.0F * y) * y), ((p1 * 2.0F) * x) * y);
    const float distorted_x = x +
        std::fma(x, radial, tangential_x * tangential_scale) * radius_clamp;
    const float distorted_y = y +
        std::fma(y, radial, tangential_y * tangential_scale) * radius_clamp;
    const float f = static_cast<float>(calibration.f);
    const float projected_x = std::fma(
        distorted_x, f + static_cast<float>(calibration.b1),
        static_cast<float>(calibration.b2) * distorted_y) +
        std::fma(static_cast<float>(calibration.width), 0.5F,
                 static_cast<float>(calibration.cx));
    const float projected_y = f * distorted_y +
        std::fma(static_cast<float>(calibration.height), 0.5F,
                 static_cast<float>(calibration.cy));
    return GpuProjectedPoint{{static_cast<double>(projected_x),
                              static_cast<double>(projected_y)},
                             static_cast<double>(local_z)};
}

ProjectedFace project_face_gpu_float(const Mesh &mesh, const Face &face,
                                     const Camera &camera) {
    ProjectedFace result;
    std::array<Vec3d, 3> local{};
    for (int corner = 0; corner < 3; ++corner) {
        const Vec3d world = mesh.vertices[face.vertex[corner]];
        const auto projected = project_frame_gpu_float(camera, world);
        if (!projected) return result;
        result.pixel[corner] = projected->pixel;
        result.depth[corner] = projected->depth;
        local[corner] = world_to_camera(camera.pose, world);
    }
    result.view_dot = face_view_dot(local[0], local[1], local[2]);
    result.area = triangle_area(result.pixel[0], result.pixel[1], result.pixel[2]);
    result.valid = result.view_dot <= 0.0 && result.area > 0.0;
    return result;
}

template <class Function>
void raster_triangle(const std::array<Vec2d, 3> &triangle, int width, int height, Function function) {
    const double denominator = (triangle[1].y - triangle[2].y) * (triangle[0].x - triangle[2].x) +
                               (triangle[2].x - triangle[1].x) * (triangle[0].y - triangle[2].y);
    if (std::abs(denominator) < 1.0e-20) return;
    const int min_x = std::max(0, static_cast<int>(std::ceil(std::min({triangle[0].x, triangle[1].x, triangle[2].x}) - 0.5)));
    const int max_x = std::min(width - 1, static_cast<int>(std::floor(std::max({triangle[0].x, triangle[1].x, triangle[2].x}) - 0.5)));
    const int min_y = std::max(0, static_cast<int>(std::ceil(std::min({triangle[0].y, triangle[1].y, triangle[2].y}) - 0.5)));
    const int max_y = std::min(height - 1, static_cast<int>(std::floor(std::max({triangle[0].y, triangle[1].y, triangle[2].y}) - 0.5)));
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const double px = x + 0.5;
            const double py = y + 0.5;
            const double a = ((triangle[1].y - triangle[2].y) * (px - triangle[2].x) +
                              (triangle[2].x - triangle[1].x) * (py - triangle[2].y)) / denominator;
            const double b = ((triangle[2].y - triangle[0].y) * (px - triangle[2].x) +
                              (triangle[0].x - triangle[2].x) * (py - triangle[2].y)) / denominator;
            const double c = 1.0 - a - b;
            if (a >= -1.0e-8 && b >= -1.0e-8 && c >= -1.0e-8) function(x, y, std::array<double, 3>{a, b, c});
        }
    }
}

template <class Function>
void raster_triangle_fixed_top_left(const std::array<Vec2d, 3> &triangle,
                                    int width, int height, int subpixel_bits,
                                    bool alternate_top_left, Function function) {
    const std::int64_t scale = std::int64_t{1} << subpixel_bits;
    const std::int64_t half = scale / 2;
    std::array<std::array<std::int64_t, 2>, 3> vertex{};
    for (std::size_t i = 0; i < 3; ++i) {
        vertex[i][0] = std::llround(triangle[i].x * static_cast<double>(scale));
        vertex[i][1] = std::llround(triangle[i].y * static_cast<double>(scale));
    }
    const auto edge = [](const auto &a, const auto &b, std::int64_t x,
                         std::int64_t y) {
        return (b[0] - a[0]) * (y - a[1]) -
               (b[1] - a[1]) * (x - a[0]);
    };
    const std::int64_t area = edge(vertex[0], vertex[1], vertex[2][0], vertex[2][1]);
    if (area == 0) return;
    const std::int64_t orientation = area > 0 ? 1 : -1;
    const auto is_top_left = [&](std::size_t first, std::size_t second) {
        if (orientation < 0) std::swap(first, second);
        const std::int64_t dx = vertex[second][0] - vertex[first][0];
        const std::int64_t dy = vertex[second][1] - vertex[first][1];
        const bool standard = dy < 0 || (dy == 0 && dx > 0);
        return alternate_top_left ? !standard : standard;
    };
    const std::array<bool, 3> inclusive{
        is_top_left(1, 2), is_top_left(2, 0), is_top_left(0, 1)};
    const int min_x = std::max(0, static_cast<int>(std::ceil(
        std::min({triangle[0].x, triangle[1].x, triangle[2].x}) - 0.5)));
    const int max_x = std::min(width - 1, static_cast<int>(std::floor(
        std::max({triangle[0].x, triangle[1].x, triangle[2].x}) - 0.5)));
    const int min_y = std::max(0, static_cast<int>(std::ceil(
        std::min({triangle[0].y, triangle[1].y, triangle[2].y}) - 0.5)));
    const int max_y = std::min(height - 1, static_cast<int>(std::floor(
        std::max({triangle[0].y, triangle[1].y, triangle[2].y}) - 0.5)));
    const double inverse_area = 1.0 / static_cast<double>(area);
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const std::int64_t sample_x = static_cast<std::int64_t>(x) * scale + half;
            const std::int64_t sample_y = static_cast<std::int64_t>(y) * scale + half;
            const std::array<std::int64_t, 3> raw{
                edge(vertex[1], vertex[2], sample_x, sample_y),
                edge(vertex[2], vertex[0], sample_x, sample_y),
                edge(vertex[0], vertex[1], sample_x, sample_y)};
            bool inside = true;
            for (std::size_t i = 0; i < 3; ++i) {
                const std::int64_t oriented = orientation * raw[i];
                if (oriented < 0 || (oriented == 0 && !inclusive[i])) {
                    inside = false;
                    break;
                }
            }
            if (!inside) continue;
            function(x, y, std::array<double, 3>{
                static_cast<double>(raw[0]) * inverse_area,
                static_cast<double>(raw[1]) * inverse_area,
                static_cast<double>(raw[2]) * inverse_area});
        }
    }
}

template <class Function>
void raster_triangle_conservative(const std::array<Vec2d, 3> &triangle,
                                  int width, int height, double extent_scale,
                                  Function function) {
    const double denominator =
        (triangle[1].y - triangle[2].y) * (triangle[0].x - triangle[2].x) +
        (triangle[2].x - triangle[1].x) * (triangle[0].y - triangle[2].y);
    if (std::abs(denominator) < 1.0e-20) return;
    const std::array<double, 3> derivative_x = {
        (triangle[1].y - triangle[2].y) / denominator,
        (triangle[2].y - triangle[0].y) / denominator,
        (triangle[0].y - triangle[1].y) / denominator};
    const std::array<double, 3> derivative_y = {
        (triangle[2].x - triangle[1].x) / denominator,
        (triangle[0].x - triangle[2].x) / denominator,
        (triangle[1].x - triangle[0].x) / denominator};
    std::array<double, 3> half_pixel_extent{};
    for (int i = 0; i < 3; ++i) {
        half_pixel_extent[static_cast<std::size_t>(i)] = 0.5 * extent_scale *
            (std::abs(derivative_x[static_cast<std::size_t>(i)]) +
             std::abs(derivative_y[static_cast<std::size_t>(i)]));
    }
    const double triangle_min_x = std::min({triangle[0].x, triangle[1].x, triangle[2].x});
    const double triangle_max_x = std::max({triangle[0].x, triangle[1].x, triangle[2].x});
    const double triangle_min_y = std::min({triangle[0].y, triangle[1].y, triangle[2].y});
    const double triangle_max_y = std::max({triangle[0].y, triangle[1].y, triangle[2].y});
    const double lower_center_offset = 0.5 * (extent_scale + 1.0);
    const double upper_center_offset = 0.5 * (extent_scale - 1.0);
    const int min_x = std::max(0, static_cast<int>(
        std::ceil(triangle_min_x - lower_center_offset)));
    const int max_x = std::min(width - 1, static_cast<int>(
        std::floor(triangle_max_x + upper_center_offset)));
    const int min_y = std::max(0, static_cast<int>(
        std::ceil(triangle_min_y - lower_center_offset)));
    const int max_y = std::min(height - 1, static_cast<int>(
        std::floor(triangle_max_y + upper_center_offset)));
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const double px = x + 0.5;
            const double py = y + 0.5;
            const double a = ((triangle[1].y - triangle[2].y) * (px - triangle[2].x) +
                              (triangle[2].x - triangle[1].x) * (py - triangle[2].y)) /
                             denominator;
            const double b = ((triangle[2].y - triangle[0].y) * (px - triangle[2].x) +
                              (triangle[0].x - triangle[2].x) * (py - triangle[2].y)) /
                             denominator;
            const double c = 1.0 - a - b;
            const std::array<double, 3> barycentric = {a, b, c};
            bool covered = true;
            for (int i = 0; i < 3; ++i) {
                if (barycentric[static_cast<std::size_t>(i)] <
                    -half_pixel_extent[static_cast<std::size_t>(i)]) {
                    covered = false;
                    break;
                }
            }
            if (covered) function(x, y, barycentric);
        }
    }
}

template <class Function>
void raster_triangle_lines(const std::array<Vec2d, 3> &triangle,
                           const std::array<double, 3> &vertex_depth,
                           int width, int height, Function function) {
    // E-VKLINE-248: the target reports subPixelPrecisionBits=8, and the
    // observed coverage plus line varyings select nearest 1/256-pixel endpoint
    // quantization.  The fragment shader separately reconstructs its triangle
    // barycentrics from the unquantized projected positions, so this copy is
    // used only for fixed-function coverage and line-varying interpolation.
    std::array<Vec2d, 3> raster_triangle = triangle;
    constexpr double subpixel_scale = 256.0;
    for (Vec2d &point : raster_triangle) {
        point.x = std::nearbyint(point.x * subpixel_scale) / subpixel_scale;
        point.y = std::nearbyint(point.y * subpixel_scale) / subpixel_scale;
    }
    const double denominator =
        (raster_triangle[1].y - raster_triangle[2].y) *
            (raster_triangle[0].x - raster_triangle[2].x) +
        (raster_triangle[2].x - raster_triangle[1].x) *
            (raster_triangle[0].y - raster_triangle[2].y);
    if (std::abs(denominator) < 1.0e-20) return;
    // Runtime evidence E-PIPELINE-239 gives VK_POLYGON_MODE_LINE and
    // lineWidth=1 with no line-rasterization pNext state.  The target NVIDIA
    // device reports strictLines=VK_TRUE.  Vulkan therefore defines DEFAULT
    // line rasterization as RECTANGULAR: sample centers are covered by the
    // width-one rectangle extruded perpendicular to each segment, with caps
    // through the two endpoints.  Diamond-exit belongs to BRESENHAM mode and
    // was an earlier, now-rejected hypothesis.
    const auto sample_in_rectangular_line = [](Vec2d a, Vec2d b, int x, int y,
                                                double &segment_t) {
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double length_squared = dx * dx + dy * dy;
        if (length_squared < 1.0e-20) return false;
        const double px = static_cast<double>(x) + 0.5 - a.x;
        const double py = static_cast<double>(y) + 0.5 - a.y;
        const double longitudinal = px * dx + py * dy;
        if (longitudinal < 0.0 || longitudinal > length_squared) return false;
        const double cross = px * dy - py * dx;
        if (cross * cross > 0.25 * length_squared) return false;
        segment_t = longitudinal / length_squared;
        return true;
    };
    for (int edge = 0; edge < 3; ++edge) {
        const Vec2d a = raster_triangle[static_cast<std::size_t>(edge)];
        const Vec2d b = raster_triangle[static_cast<std::size_t>((edge + 1) % 3)];
        const int min_x = std::max(0, static_cast<int>(
            std::ceil(std::min(a.x, b.x) - 1.0)));
        const int max_x = std::min(width - 1, static_cast<int>(
            std::floor(std::max(a.x, b.x))));
        const int min_y = std::max(0, static_cast<int>(
            std::ceil(std::min(a.y, b.y) - 1.0)));
        const int max_y = std::min(height - 1, static_cast<int>(
            std::floor(std::max(a.y, b.y))));
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                double segment_t = 0.0;
                if (!sample_in_rectangular_line(a, b, x, y, segment_t)) continue;
                const double px = x + 0.5;
                const double py = y + 0.5;
                const double wa =
                    ((raster_triangle[1].y - raster_triangle[2].y) *
                         (px - raster_triangle[2].x) +
                     (raster_triangle[2].x - raster_triangle[1].x) *
                         (py - raster_triangle[2].y)) /
                    denominator;
                const double wb =
                    ((raster_triangle[2].y - raster_triangle[0].y) *
                         (px - raster_triangle[2].x) +
                     (raster_triangle[0].x - raster_triangle[2].x) *
                         (py - raster_triangle[2].y)) /
                    denominator;
                const int next = (edge + 1) % 3;
                double line_depth =
                    (1.0 - segment_t) * vertex_depth[static_cast<std::size_t>(edge)] +
                    segment_t * vertex_depth[static_cast<std::size_t>(next)];
                if (std::getenv("METASHAPE_TEXTURE_OCCLUSION_FLOAT_LINE_VARYING") !=
                    nullptr) {
                    const float t = static_cast<float>(segment_t);
                    const float first = static_cast<float>(
                        vertex_depth[static_cast<std::size_t>(edge)]);
                    const float second = static_cast<float>(
                        vertex_depth[static_cast<std::size_t>(next)]);
                    line_depth = static_cast<double>(std::fma(t, second - first, first));
                }
                function(x, y, std::array<double, 3>{wa, wb, 1.0 - wa - wb},
                         line_depth);
            }
        }
    }
}

void build_depth_buffers(const Mesh &mesh, std::vector<Camera> &cameras, int depth_downscale) {
    const char *depth_dump_root = std::getenv("METASHAPE_TEXTURE_DEPTH_DUMP_DIR");
    const char *cull_mode_value = std::getenv("METASHAPE_TEXTURE_DEPTH_CULL_MODE");
    // The native /10_depth_unscaled_framebuffer used by the edge-quality
    // projection kernel is captured before its later backface-mask cleanup:
    // it is a two-sided depth pass.  Vulkan rasterization exposes 8 bits of
    // subpixel precision on the recovered target GPU.
    const std::string cull_mode = cull_mode_value == nullptr ? "none" : cull_mode_value;
    int raster_subpixel_bits = 8;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_RASTER_SUBPIXEL_BITS")) {
        raster_subpixel_bits = std::stoi(value);
    }
    const char *raster_round_value = std::getenv("METASHAPE_TEXTURE_RASTER_SUBPIXEL_ROUND");
    const std::string raster_round = raster_round_value == nullptr ? "nearest" : raster_round_value;
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
        auto &camera = cameras[camera_index];
        const int width = (camera.calibration.width + depth_downscale - 1) / depth_downscale;
        const int height = (camera.calibration.height + depth_downscale - 1) / depth_downscale;
        camera.depth_minimum = std::numeric_limits<double>::infinity();
        camera.depth_maximum = -std::numeric_limits<double>::infinity();
        for (const Vec3d &vertex : mesh.vertices) {
            const double z = world_to_camera(camera.pose, vertex).z;
            camera.depth_minimum = std::min(camera.depth_minimum, z);
            camera.depth_maximum = std::max(camera.depth_maximum, z);
        }
        if (!(camera.depth_maximum > camera.depth_minimum)) {
            throw std::runtime_error("invalid camera-space model depth range");
        }
        camera.depth = cv::Mat(height, width, CV_32F,
                               cv::Scalar(std::numeric_limits<float>::max()));
        camera.scaled_depth = cv::Mat(height, width, CV_32F, cv::Scalar(1.0F));
        camera.face_id = cv::Mat(height, width, CV_32S,
                                 cv::Scalar(std::numeric_limits<std::int32_t>::max()));
        for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
            const Face &face = mesh.faces[face_index];
            auto projected = project_face(mesh, face, camera);
            if (projected.area <= 0.0) continue;
            std::array<Vec2d, 3> scaled = projected.pixel;
            for (auto &pixel : scaled) { pixel.x /= depth_downscale; pixel.y /= depth_downscale; }
            if (raster_subpixel_bits >= 0) {
                const double scale = std::ldexp(1.0, raster_subpixel_bits);
                auto quantize = [&](double coordinate) {
                    const double scaled_coordinate = coordinate * scale;
                    const double integer_coordinate = raster_round == "floor"
                        ? std::floor(scaled_coordinate)
                        : raster_round == "ceil" ? std::ceil(scaled_coordinate)
                        : std::nearbyint(scaled_coordinate);
                    return integer_coordinate / scale;
                };
                for (auto &pixel : scaled) {
                    pixel.x = quantize(pixel.x);
                    pixel.y = quantize(pixel.y);
                }
            }
            const double signed_area =
                (scaled[1].x - scaled[0].x) * (scaled[2].y - scaled[0].y) -
                (scaled[1].y - scaled[0].y) * (scaled[2].x - scaled[0].x);
            const bool culled = cull_mode == "screen-positive" ? signed_area >= 0.0
                                : cull_mode == "screen-negative" ? signed_area <= 0.0
                                : cull_mode == "view" ? projected.view_dot > 0.0
                                : false;
            if (culled) continue;
            raster_triangle(scaled, width, height, [&](int x, int y, const std::array<double, 3> &w) {
                // The native unscaled-depth vertex shader emits camera-space z
                // as a varying while gl_Position.w is 1.  The framebuffer thus
                // contains screen-space affine interpolation of vertex z, not
                // reciprocal/perspective-correct triangle depth.
                const double interpolated_depth =
                    w[0] * projected.depth[0] + w[1] * projected.depth[1] +
                    w[2] * projected.depth[2];
                if (interpolated_depth <= 0.0) return;
                float &stored = camera.depth.at<float>(y, x);
                const float candidate = static_cast<float>(interpolated_depth);
                if (candidate < stored) {
                    stored = candidate;
                    camera.scaled_depth.at<float>(y, x) = static_cast<float>(
                        (interpolated_depth - camera.depth_minimum) /
                        (camera.depth_maximum - camera.depth_minimum));
                    camera.face_id.at<std::int32_t>(y, x) = static_cast<std::int32_t>(face_index);
                }
            });
        }
        if (cull_mode == "postmask") {
            // Native order, recovered from
            // estimate_is_face_a_backface_mask.comp.spv,
            // filter_backface_speckles.comp.spv and the two remove_* shaders:
            // render both orientations into the z-buffer first, then clear a
            // pixel if its nearest face is a non-isolated backface.  This is
            // observably different from fixed-function pre-raster culling,
            // because a front face behind the cleared backface is not exposed.
            std::vector<std::uint32_t> backface_vertex_counts(mesh.vertices.size(), 0);
            std::vector<std::uint8_t> backface(mesh.faces.size(), 0);
            for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
                const auto projected = project_face(mesh, mesh.faces[face_index], camera);
                if (projected.area <= 0.0 || projected.view_dot <= 0.0) continue;
                backface[face_index] = 1;
                for (std::uint32_t vertex : mesh.faces[face_index].vertex) {
                    ++backface_vertex_counts[vertex];
                }
            }
            for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
                if (backface[face_index] == 0) continue;
                bool connected = false;
                for (std::uint32_t vertex : mesh.faces[face_index].vertex) {
                    connected = connected || backface_vertex_counts[vertex] >= 2;
                }
                backface[face_index] = connected ? 1 : 0;
            }
            const auto no_face = std::numeric_limits<std::int32_t>::max();
            const float no_depth = std::numeric_limits<float>::max();
            for (int y = 0; y < camera.face_id.rows; ++y) {
                for (int x = 0; x < camera.face_id.cols; ++x) {
                    std::int32_t &face = camera.face_id.at<std::int32_t>(y, x);
                    if (face >= 0 && face < static_cast<std::int32_t>(backface.size()) &&
                        backface[static_cast<std::size_t>(face)] != 0) {
                        face = no_face;
                        camera.depth.at<float>(y, x) = no_depth;
                        camera.scaled_depth.at<float>(y, x) = 1.0F;
                    }
                }
            }
        }
        if (!camera.mask.empty()) {
            // Exact remove_depth_wrt_mask.comp/remove_face_id_wrt_mask.comp
            // semantics: one invocation per original mask pixel; each zero
            // writes the sentinels to (source_x / scale, source_y / scale).
            // Consequently every source block must be nonzero for the
            // downscaled depth/face pixel to survive.
            const auto no_face = std::numeric_limits<std::int32_t>::max();
            const float no_depth = std::numeric_limits<float>::max();
            for (int y = 0; y < camera.depth.rows; ++y) {
                for (int x = 0; x < camera.depth.cols; ++x) {
                    bool valid = true;
                    for (int source_y = y * depth_downscale;
                         source_y < std::min((y + 1) * depth_downscale, camera.mask.rows) && valid;
                         ++source_y) {
                        const std::uint8_t *row = camera.mask.ptr<std::uint8_t>(source_y);
                        for (int source_x = x * depth_downscale;
                             source_x < std::min((x + 1) * depth_downscale, camera.mask.cols);
                             ++source_x) {
                            if (row[source_x] == 0) { valid = false; break; }
                        }
                    }
                    if (!valid) {
                        camera.face_id.at<std::int32_t>(y, x) = no_face;
                        camera.depth.at<float>(y, x) = no_depth;
                    }
                }
            }
        }
        if (const char *native_depth_root =
                std::getenv("METASHAPE_TEXTURE_NATIVE_DEPTH_DIR")) {
            const auto path = std::filesystem::path(native_depth_root) /
                              std::to_string(camera_index) /
                              "10_depth_unscaled_framebuffer.raw.f32";
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("cannot load native depth fixture: " + path.string());
            }
            stream.read(reinterpret_cast<char *>(camera.depth.data),
                        static_cast<std::streamsize>(camera.depth.total() * sizeof(float)));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("invalid native depth-fixture byte count: " +
                                         path.string());
            }
        }
        // Isolation fixture for the separate depth image consumed by
        // enblend_camera_into_atlas.frag.  Runtime push constants prove that
        // the default BuildTexture path binds a /2 image here, while the
        // candidate-scoring pass above uses /4.  Keep this gate distinct from
        // METASHAPE_TEXTURE_NATIVE_DEPTH_DIR so a 320x240 fixture cannot be
        // accidentally loaded by the earlier 160x120 scoring pass.
        if (const char *native_enblend_depth_root =
                std::getenv("METASHAPE_TEXTURE_NATIVE_ENBLEND_DEPTH_DIR")) {
            int fixture_downscale = 2;
            if (const char *value =
                    std::getenv("METASHAPE_TEXTURE_NATIVE_ENBLEND_DEPTH_DOWNSCALE")) {
                fixture_downscale = std::stoi(value);
            }
            if (depth_downscale == fixture_downscale) {
                const auto path = std::filesystem::path(native_enblend_depth_root) /
                                  std::to_string(camera_index) /
                                  "10_depth_unscaled_framebuffer.raw.f32";
                std::ifstream stream(path, std::ios::binary);
                if (!stream) {
                    throw std::runtime_error(
                        "cannot load native enblend-depth fixture: " + path.string());
                }
                stream.read(reinterpret_cast<char *>(camera.depth.data),
                            static_cast<std::streamsize>(camera.depth.total() *
                                                         sizeof(float)));
                if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                    throw std::runtime_error(
                        "invalid native enblend-depth fixture byte count: " +
                        path.string());
                }
            }
        }
        if (depth_dump_root != nullptr && *depth_dump_root != '\0') {
            const auto directory = std::filesystem::path(depth_dump_root) /
                                   std::to_string(camera_index);
            std::filesystem::create_directories(directory);
            const auto path = directory / "10_depth_unscaled_framebuffer.raw.f32";
            std::ofstream output(path, std::ios::binary);
            if (!output) {
                throw std::runtime_error("cannot create raw depth dump: " + path.string());
            }
            for (int row = 0; row < camera.depth.rows; ++row) {
                output.write(reinterpret_cast<const char *>(camera.depth.ptr<float>(row)),
                             static_cast<std::streamsize>(camera.depth.cols * sizeof(float)));
            }
            if (!output) {
                throw std::runtime_error("cannot write raw depth dump: " + path.string());
            }
            const auto scaled_path = directory / "00_depth_framebuffer.raw.f32";
            std::ofstream scaled_output(scaled_path, std::ios::binary);
            if (!scaled_output) {
                throw std::runtime_error("cannot create scaled depth dump: " +
                                         scaled_path.string());
            }
            for (int row = 0; row < camera.scaled_depth.rows; ++row) {
                scaled_output.write(
                    reinterpret_cast<const char *>(camera.scaled_depth.ptr<float>(row)),
                    static_cast<std::streamsize>(camera.scaled_depth.cols * sizeof(float)));
            }
            if (!scaled_output) {
                throw std::runtime_error("cannot write scaled depth dump: " +
                                         scaled_path.string());
            }
            const auto face_path = directory / "20_face_ids.raw.i32";
            std::ofstream face_output(face_path, std::ios::binary);
            if (!face_output) {
                throw std::runtime_error("cannot create raw face-id dump: " + face_path.string());
            }
            for (int row = 0; row < camera.face_id.rows; ++row) {
                face_output.write(
                    reinterpret_cast<const char *>(camera.face_id.ptr<std::int32_t>(row)),
                    static_cast<std::streamsize>(camera.face_id.cols * sizeof(std::int32_t)));
            }
            if (!face_output) {
                throw std::runtime_error("cannot write raw face-id dump: " + face_path.string());
            }
        }
    }
}

using PixelEdgeVisibility = std::vector<std::array<std::vector<std::uint8_t>, 3>>;

PixelEdgeVisibility build_face_id_edge_visibility(const Mesh &mesh,
                                                  const std::vector<Camera> &cameras) {
    const std::size_t camera_count = cameras.size();
    PixelEdgeVisibility result(mesh.faces.size());
    for (auto &face : result) {
        for (auto &edge : face) edge.assign(camera_count, 0);
    }

    std::vector<std::array<int, 3>> neighbors(mesh.faces.size(),
                                              std::array<int, 3>{-1, -1, -1});
    for (const auto &adjacency : build_face_adjacency(mesh)) {
        neighbors[static_cast<std::size_t>(adjacency.first)]
                 [static_cast<std::size_t>(adjacency.first_edge)] = adjacency.second;
        neighbors[static_cast<std::size_t>(adjacency.second)]
                 [static_cast<std::size_t>(adjacency.second_edge)] = adjacency.first;
    }

    const auto no_face = std::numeric_limits<std::int32_t>::max();
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
        const cv::Mat &ids = cameras[camera_index].face_id;
        for (int y = 0; y < ids.rows; ++y) {
            for (int x = 0; x < ids.cols; ++x) {
                const int face = ids.at<std::int32_t>(y, x);
                if (face == no_face || face < 0 ||
                    face >= static_cast<int>(mesh.faces.size())) continue;
                auto remaining = neighbors[static_cast<std::size_t>(face)];
                for (int dy = -1; dy <= 1; ++dy) {
                    const int ny = y + dy;
                    if (ny < 0 || ny >= ids.rows) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx;
                        if (nx < 0 || nx >= ids.cols) continue;
                        const int neighbor = ids.at<std::int32_t>(ny, nx);
                        if (neighbor == face || neighbor == no_face) continue;
                        for (int edge = 0; edge < 3; ++edge) {
                            if (remaining[static_cast<std::size_t>(edge)] == neighbor) {
                                result[static_cast<std::size_t>(face)]
                                      [static_cast<std::size_t>(edge)][camera_index] = 1;
                                remaining[static_cast<std::size_t>(edge)] = no_face;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool projected_edge_midpoint_visible(const Mesh &mesh, std::size_t face_index, int edge,
                                     const ProjectedFace &face, const Camera &camera,
                                     int depth_downscale) {
    const int next = (edge + 1) % 3;
    const Face &mesh_face = mesh.faces[face_index];
    const Vec3d &a = mesh.vertices[mesh_face.vertex[static_cast<std::size_t>(edge)]];
    // Preserve the native SPIR-V indexing exactly.  The second index is loaded
    // as indices[(3 * face + edge + 1) % 3], rather than
    // indices[3 * face + (edge + 1) % 3].  Consequently it comes from face 0
    // for every invocation.  Correcting this apparent shader bug changes the
    // visibility support by hundreds of thousands of edge/camera records.
    const Face &first_face = mesh.faces.front();
    const Vec3d &b = mesh.vertices[first_face.vertex[static_cast<std::size_t>(next)]];
    const Vec3d midpoint{(a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5};
    const Vec3d local = world_to_camera(camera.pose, midpoint);
    if (local.z <= 0.0) return false;
    const auto pixel = project_frame(camera.pose, camera.calibration, midpoint, false);
    if (!pixel || pixel->x < 0.0 || pixel->y < 0.0 ||
        pixel->x >= camera.calibration.width || pixel->y >= camera.calibration.height) {
        return false;
    }

    const int base_x = static_cast<int>(pixel->x / depth_downscale - 0.5);
    const int base_y = static_cast<int>(pixel->y / depth_downscale - 0.5);
    const float maximum = std::numeric_limits<float>::max();
    const char *midpoint_depth_mode = std::getenv("METASHAPE_TEXTURE_MIDPOINT_DEPTH_MODE");
    const double target_depth = midpoint_depth_mode != nullptr &&
                                        std::string(midpoint_depth_mode) == "euclidean"
                                    ? std::sqrt(local.x * local.x + local.y * local.y +
                                                local.z * local.z)
                                    : local.z;
    const double threshold = recovered_depth_tolerance(face.view_dot) * target_depth;
    bool occluded_in_all_samples = true;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = std::clamp(base_x + dx, 0, camera.depth.cols - 1);
            const int y = std::clamp(base_y + dy, 0, camera.depth.rows - 1);
            const float stored = camera.depth.at<float>(y, x);
            if (stored != maximum && stored >= threshold) occluded_in_all_samples = false;
        }
    }
    return !occluded_in_all_samples;
}

bool centroid_visible(const ProjectedFace &face, const Camera &camera, int depth_downscale) {
    const Vec2d centroid{(face.pixel[0].x + face.pixel[1].x + face.pixel[2].x) / 3.0,
                         (face.pixel[0].y + face.pixel[1].y + face.pixel[2].y) / 3.0};
    const double depth = 3.0 / (1.0 / face.depth[0] + 1.0 / face.depth[1] + 1.0 / face.depth[2]);
    const int base_x = static_cast<int>(centroid.x / depth_downscale - 0.5);
    const int base_y = static_cast<int>(centroid.y / depth_downscale - 0.5);
    double tolerance = recovered_depth_tolerance(face.view_dot);
    if (const char *override_value =
            std::getenv("METASHAPE_TEXTURE_DEPTH_TOLERANCE_OVERRIDE")) {
        tolerance = std::stod(override_value);
    }
    const double threshold = tolerance * depth;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = std::clamp(base_x + dx, 0, camera.depth.cols - 1);
            const int y = std::clamp(base_y + dy, 0, camera.depth.rows - 1);
            const float stored = camera.depth.at<float>(y, x);
            if (stored != std::numeric_limits<float>::max() && stored >= threshold) return true;
        }
    }
    return false;
}

bool fragment_visible(const ProjectedFace &face, const Camera &camera, int depth_downscale,
                      double pixel_x, double pixel_y, double depth) {
    if (const char *disabled = std::getenv("METASHAPE_TEXTURE_DISABLE_DEPTH")) {
        if (std::string(disabled) == "1") return true;
    }
    // Exact enblend_camera_into_atlas.frag semantics:
    //   ivec2(projected_xy / depth_downscale - vec2(0.5))
    // GLSL float-to-int conversion and C++ static_cast<int> both truncate
    // toward zero.  The former +1/-0.5 defaults were an empirical alignment
    // experiment and are retained only as explicit diagnostic overrides.
    double depth_offset_x = 0.0;
    double depth_offset_y = 0.0;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_DEPTH_SAMPLE_OFFSET_X")) {
        depth_offset_x = std::stod(value);
    }
    if (const char *value = std::getenv("METASHAPE_TEXTURE_DEPTH_SAMPLE_OFFSET_Y")) {
        depth_offset_y = std::stod(value);
    }
    const int base_x = static_cast<int>(pixel_x / depth_downscale - 0.5 + depth_offset_x);
    const int base_y = static_cast<int>(pixel_y / depth_downscale - 0.5 + depth_offset_y);
    // Runtime module embedded_0255 emits the raw camera-space z for Frame
    // cameras; the 5-level/4-channel fragment applies exactly one recovered
    // 0.999..0.9999 view-dependent factor.
    const double threshold = recovered_depth_tolerance(face.view_dot) * depth;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = std::clamp(base_x + dx, 0, camera.depth.cols - 1);
            const int y = std::clamp(base_y + dy, 0, camera.depth.rows - 1);
            const float stored = camera.depth.at<float>(y, x);
            if (stored != std::numeric_limits<float>::max() && stored >= threshold) return true;
        }
    }
    return false;
}

double recovered_face_resolution(const ProjectedFace &face, const Camera &camera,
                                 int depth_downscale) {
    if (!face.valid) return 0.0;
    int depth_invisible_vertices = 0;
    int outside_vertices = 0;
    for (int corner = 0; corner < 3; ++corner) {
        const Vec2d &pixel = face.pixel[static_cast<std::size_t>(corner)];
        const bool projected_inside = pixel.x >= 0.0 && pixel.y >= 0.0 &&
                                      pixel.x < camera.calibration.width &&
                                      pixel.y < camera.calibration.height;
        if (!projected_inside) ++outside_vertices;
        if (projected_inside &&
            !fragment_visible(face, camera, depth_downscale, pixel.x, pixel.y,
                              face.depth[static_cast<std::size_t>(corner)])) {
            ++depth_invisible_vertices;
        }
    }
    // Projection outside the calibrated image is the native zero class.  The
    // -1 marker denotes partial depth visibility only when all three projected
    // vertices are inside the image.
    if (outside_vertices != 0) return 0.0;
    if (depth_invisible_vertices == 3) return 0.0;
    if (depth_invisible_vertices != 0) return -1.0;

    // update_faces_resolution_with_pixels_count.comp evaluates this expression
    // in float32 with two fused multiply-adds.
    const float x0 = static_cast<float>(face.pixel[0].x);
    const float y0 = static_cast<float>(face.pixel[0].y);
    const float x1 = static_cast<float>(face.pixel[1].x);
    const float y1 = static_cast<float>(face.pixel[1].y);
    const float x2 = static_cast<float>(face.pixel[2].x);
    const float y2 = static_cast<float>(face.pixel[2].y);
    const float twice_area = std::fma(x0, y1 - y2,
                                     std::fma(x1, y2 - y0, x2 * (y0 - y1)));
    return static_cast<double>(std::abs(twice_area) * 0.5F);
}

bool face_has_visible_fragment(const ProjectedFace &face, const Camera &camera,
                               int depth_downscale, int raster_downscale) {
    const int width = (camera.calibration.width + raster_downscale - 1) / raster_downscale;
    const int height = (camera.calibration.height + raster_downscale - 1) / raster_downscale;
    std::array<Vec2d, 3> triangle = face.pixel;
    for (auto &point : triangle) {
        point.x /= raster_downscale;
        point.y /= raster_downscale;
    }
    bool visible = false;
    raster_triangle(triangle, width, height,
                    [&](int x, int y, const std::array<double, 3> &w) {
        if (visible) return;
        const double inverse_depth = w[0] / face.depth[0] + w[1] / face.depth[1] +
                                     w[2] / face.depth[2];
        if (inverse_depth <= 0.0) return;
        visible = fragment_visible(face, camera, depth_downscale,
                                   (x + 0.5) * raster_downscale,
                                   (y + 0.5) * raster_downscale, 1.0 / inverse_depth);
    });
    return visible;
}

bool projected_edge_has_visible_sample(const ProjectedFace &face, const Camera &camera,
                                       int depth_downscale, int edge) {
    const int next = (edge + 1) % 3;
    const Vec2d a = face.pixel[edge];
    const Vec2d b = face.pixel[next];
    const int steps = std::max(1, static_cast<int>(std::ceil(
        std::max(std::abs(b.x - a.x), std::abs(b.y - a.y)))));
    for (int index = 0; index <= steps; ++index) {
        const double t = static_cast<double>(index) / steps;
        const double x = a.x + (b.x - a.x) * t;
        const double y = a.y + (b.y - a.y) * t;
        if (x < 0.0 || y < 0.0 || x >= camera.calibration.width ||
            y >= camera.calibration.height) continue;
        const double inverse_depth = (1.0 - t) / face.depth[edge] + t / face.depth[next];
        if (inverse_depth <= 0.0) continue;
        if (fragment_visible(face, camera, depth_downscale, x, y, 1.0 / inverse_depth)) {
            return true;
        }
    }
    return false;
}

struct FaceRasterMetrics {
    double resolution = 0.0;
    double consistency = -1.0;
};

FaceRasterMetrics raster_face_metrics(const ProjectedFace &face, const Camera &camera,
                                      int depth_downscale) {
    FaceRasterMetrics result;
    // The face-id reduction is rasterized at source-image resolution.  The
    // source colour cache may be downscaled, but its per-face resolution
    // statistic is a count in the undownscaled camera grid (confirmed by the
    // native resolution.dat distribution and the face-id sum shader).
    const int resolution_downscale = 1;
    const int width = (camera.calibration.width + resolution_downscale - 1) /
                      resolution_downscale;
    const int height = (camera.calibration.height + resolution_downscale - 1) /
                       resolution_downscale;
    std::array<Vec2d, 3> triangle = face.pixel;
    for (auto &point : triangle) {
        point.x /= resolution_downscale;
        point.y /= resolution_downscale;
    }
    double quality_sum = 0.0;
    std::size_t quality_count = 0;
    raster_triangle(triangle, width, height,
                    [&](int x, int y, const std::array<double, 3> &w) {
        const double inverse_depth = w[0] / face.depth[0] + w[1] / face.depth[1] +
                                     w[2] / face.depth[2];
        if (inverse_depth <= 0.0) return;
        const double pixel_x = (x + 0.5) * resolution_downscale;
        const double pixel_y = (y + 0.5) * resolution_downscale;
        if (!fragment_visible(face, camera, depth_downscale, pixel_x, pixel_y,
                              1.0 / inverse_depth)) return;
        result.resolution += 1.0;
        if (!camera.consistency_map.empty()) {
            const int quality_x = std::clamp(
                x * camera.consistency_map.cols / std::max(width, 1),
                0, camera.consistency_map.cols - 1);
            const int quality_y = std::clamp(
                y * camera.consistency_map.rows / std::max(height, 1),
                0, camera.consistency_map.rows - 1);
            quality_sum += camera.consistency_map.at<std::uint8_t>(quality_y, quality_x) /
                           255.0;
            ++quality_count;
        }
    });
    // The native small-area path evaluates the projected triangle centroid
    // when no face-id pixel was produced.
    if (result.resolution == 0.0 && centroid_visible(face, camera, depth_downscale)) {
        result.resolution = face.area /
                            static_cast<double>(resolution_downscale * resolution_downscale);
        if (!camera.consistency_map.empty()) {
            const double x = (face.pixel[0].x + face.pixel[1].x + face.pixel[2].x) / 3.0;
            const double y = (face.pixel[0].y + face.pixel[1].y + face.pixel[2].y) / 3.0;
            const int quality_x = std::clamp(
                static_cast<int>(x * camera.consistency_map.cols /
                                 camera.calibration.width),
                0, camera.consistency_map.cols - 1);
            const int quality_y = std::clamp(
                static_cast<int>(y * camera.consistency_map.rows /
                                 camera.calibration.height),
                0, camera.consistency_map.rows - 1);
            quality_sum = camera.consistency_map.at<std::uint8_t>(quality_y, quality_x) / 255.0;
            quality_count = 1;
        }
    }
    if (quality_count != 0) result.consistency = quality_sum / quality_count;
    return result;
}

cv::Vec3f bilinear(const cv::Mat &image, double x, double y);

float rounded_luma_at_world(const Camera &camera, const ProjectedFace &face,
                            int depth_downscale, const Vec3d &world) {
    const auto pixel = project_frame(camera.pose, camera.calibration, world, false);
    if (!pixel) return -1.0F;
    const Vec3d local = world_to_camera(camera.pose, world);
    if (local.z <= 0.0 || !fragment_visible(face, camera, depth_downscale,
                                             pixel->x, pixel->y, local.z)) return -1.0F;
    const cv::Vec3f color = bilinear(camera.quality_image,
                                     pixel->x * camera.quality_image.cols /
                                         static_cast<double>(camera.calibration.width),
                                     pixel->y * camera.quality_image.rows /
                                         static_cast<double>(camera.calibration.height));
    // Exact 3x8U ZNCC shader conversion: RGB Rec.601 luma is quantized back
    // to one byte before the normalized squared residual is accumulated.
    return std::round(std::clamp(0.114F * color[0] + 0.587F * color[1] +
                                 0.299F * color[2], 0.0F, 1.0F) * 255.0F) / 255.0F;
}

std::vector<double> estimate_face_consistency(const Mesh &mesh, const Face &face,
                                               const std::vector<Camera> &cameras,
                                               const std::vector<ProjectedFace> &projected,
                                               int depth_downscale) {
    const bool captured_maps = std::any_of(
        cameras.begin(), cameras.end(),
        [](const Camera &camera) { return !camera.consistency_map.empty(); });
    if (captured_maps) {
        std::vector<double> consistency(cameras.size(), -1.0);
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            if (!projected[camera_index].valid || cameras[camera_index].consistency_map.empty()) continue;
            consistency[camera_index] = raster_face_metrics(
                projected[camera_index], cameras[camera_index], depth_downscale).consistency;
        }
        return consistency;
    }
    static constexpr std::array<std::array<double, 3>, 7> sample_weights = {{
        {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}},
        {{0.60, 0.20, 0.20}}, {{0.20, 0.60, 0.20}}, {{0.20, 0.20, 0.60}},
        {{0.50, 0.50, 0.00}}, {{0.00, 0.50, 0.50}}, {{0.50, 0.00, 0.50}},
    }};
    std::array<Vec3d, 3> vertex{};
    for (int corner = 0; corner < 3; ++corner) vertex[corner] = mesh.vertices[face.vertex[corner]];
    std::vector<std::array<float, sample_weights.size()>> samples(cameras.size());
    for (auto &entry : samples) entry.fill(-1.0F);
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
        if (!projected[camera_index].valid) continue;
        for (std::size_t sample = 0; sample < sample_weights.size(); ++sample) {
            const auto &w = sample_weights[sample];
            const Vec3d world{
                w[0] * vertex[0].x + w[1] * vertex[1].x + w[2] * vertex[2].x,
                w[0] * vertex[0].y + w[1] * vertex[1].y + w[2] * vertex[2].y,
                w[0] * vertex[0].z + w[1] * vertex[1].z + w[2] * vertex[2].z,
            };
            samples[camera_index][sample] = rounded_luma_at_world(
                cameras[camera_index], projected[camera_index], depth_downscale, world);
        }
    }
    std::vector<double> consistency(cameras.size(), -1.0);
    std::string aggregation = "mean";
    if (const char *value = std::getenv("METASHAPE_TEXTURE_CONSISTENCY_AGGREGATION")) aggregation = value;
    int neighbor_limit = 4;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_CONSISTENCY_NEIGHBORS")) {
        neighbor_limit = std::clamp(std::stoi(value), 1, static_cast<int>(cameras.size()));
    }
    for (std::size_t reference = 0; reference < cameras.size(); ++reference) {
        if (!projected[reference].valid) continue;
        std::vector<double> pair_quality;
        for (std::size_t neighbor = 0; neighbor < cameras.size(); ++neighbor) {
            if (reference == neighbor || !projected[neighbor].valid) continue;
            double squared_error = 0.0;
            int count = 0;
            for (std::size_t sample = 0; sample < sample_weights.size(); ++sample) {
                const float a = samples[reference][sample];
                const float b = samples[neighbor][sample];
                if (a < 0.0F || b < 0.0F) continue;
                const double difference = static_cast<double>(a - b);
                squared_error += difference * difference;
                ++count;
            }
            if (count * 5 < static_cast<int>(sample_weights.size()) * 3) continue;
            pair_quality.push_back(1.0 - std::min(20.0 * squared_error / count, 1.0));
        }
        if (pair_quality.empty()) continue;
        std::sort(pair_quality.begin(), pair_quality.end(), std::greater<double>());
        if (pair_quality.size() > static_cast<std::size_t>(neighbor_limit)) {
            pair_quality.resize(static_cast<std::size_t>(neighbor_limit));
        }
        if (aggregation == "min") {
            consistency[reference] = *std::min_element(pair_quality.begin(), pair_quality.end());
        } else if (aggregation == "max") {
            consistency[reference] = pair_quality.front();
        } else {
            consistency[reference] = std::accumulate(pair_quality.begin(), pair_quality.end(), 0.0) /
                                     pair_quality.size();
        }
    }
    return consistency;
}

float estimate_face_focus_quality(std::size_t face_index, const ProjectedFace &face,
                                  const Camera &camera, int depth_downscale,
                                  float resolution, float face_weight) {
    if (camera.focus_quality_map.empty()) return -128.0F;
    if (!camera.focus_quality_sum.empty() && !camera.focus_quality_count.empty()) {
        if (face_index >= camera.focus_quality_sum.size() ||
            face_index >= camera.focus_quality_count.size()) {
            throw std::runtime_error("focus face-quality rank is out of range");
        }
        const float count = camera.focus_quality_count[face_index];
        return count != 0.0F ? camera.focus_quality_sum[face_index] / count : 0.0F;
    }
    float sum = 0.0F;
    float count = 0.0F;
    const bool quality_candidate = resolution > 0.25F * face_weight;
    if (quality_candidate && !camera.face_id.empty() &&
        camera.face_id.size() == camera.focus_quality_map.size()) {
        // embedded_0214 is dispatched as 10x2 local-16x16 groups: exactly the
        // first 32 rows, not the whole 160x120 image.
        const int row_count = std::min(32, camera.focus_quality_map.rows);
        for (int y = 0; y < row_count; ++y) {
            for (int x = 0; x < camera.focus_quality_map.cols; ++x) {
                if (camera.face_id.at<std::int32_t>(y, x) !=
                    static_cast<std::int32_t>(face_index)) continue;
                const std::uint8_t sample = camera.focus_quality_map.at<std::uint8_t>(y, x);
                if (sample == 0) continue;
                sum += static_cast<float>(sample) * (1.0F / 255.0F);
                count += 1.0F;
            }
        }
    }
    if (count != 0.0F || !quality_candidate || !face.valid) {
        return count != 0.0F ? sum / count : 0.0F;
    }

    // update_face_quality_with_proj_sampling.comp fills only zero-count
    // entries by projecting the triangle centroid and depth-testing it.
    const double x = (face.pixel[0].x + face.pixel[1].x + face.pixel[2].x) / 3.0;
    const double y = (face.pixel[0].y + face.pixel[1].y + face.pixel[2].y) / 3.0;
    const double z = (face.depth[0] + face.depth[1] + face.depth[2]) / 3.0;
    if (!fragment_visible(face, camera, depth_downscale, x, y, z)) return 0.0F;
    const int sample_x = std::clamp(static_cast<int>(x / 4.0),
                                    0, camera.focus_quality_map.cols - 1);
    const int sample_y = std::clamp(static_cast<int>(y / 4.0),
                                    0, camera.focus_quality_map.rows - 1);
    const std::uint8_t sample = camera.focus_quality_map.at<std::uint8_t>(sample_y, sample_x);
    return sample == 0 ? 0.0F : static_cast<float>(sample) * (1.0F / 255.0F);
}

std::vector<int> estimate_winners(const Mesh &mesh, const std::vector<Camera> &cameras,
                                  int depth_downscale, bool ghosting_filter,
                                  bool out_of_focus_filter) {
    static constexpr std::int64_t unavailable_camera_unary = 1'000'000'000LL;
    static constexpr std::int64_t no_camera_unary = 10'000'000LL;
    const int no_camera_label = static_cast<int>(cameras.size());
    const int label_count = no_camera_label + 1;
    // With both optional quality filters disabled, Linux 2.3.2 installs a
    // smooth functor whose complete 3 * (faces * (labels + 1) + 1) byte
    // backing vector is zero.  The graph therefore contributes no pairwise
    // energy in this mode; winner selection is the exact unary argmin.
    const bool pairwise_quality_enabled = ghosting_filter || out_of_focus_filter;
    std::vector<int> winners(mesh.faces.size(), no_camera_label);
    std::vector<std::vector<std::int64_t>> unary(
        mesh.faces.size(), std::vector<std::int64_t>(label_count, unavailable_camera_unary));
    for (auto &costs : unary) costs[static_cast<std::size_t>(no_camera_label)] = no_camera_unary;
    DirectedEdgeCosts edge_costs(mesh.faces.size());
    std::vector<double> diagnostic_face_weights(mesh.faces.size(), 0.0);
    std::vector<double> diagnostic_consistency_max(mesh.faces.size(), -1.0);
    std::vector<double> diagnostic_consistency_range(mesh.faces.size(), -1.0);
    std::vector<int> diagnostic_camera_counts(mesh.faces.size(), 0);
    std::vector<float> diagnostic_camera_resolution;
    if (std::getenv("METASHAPE_TEXTURE_CAMERA_RESOLUTION_DUMP") != nullptr) {
        diagnostic_camera_resolution.resize(mesh.faces.size() * cameras.size(), -1.0F);
    }
    std::vector<float> diagnostic_nadirness;
    if (std::getenv("METASHAPE_TEXTURE_NADIRNESS_DUMP") != nullptr) {
        diagnostic_nadirness.resize(mesh.faces.size() * cameras.size(), -128.0F);
    }
    std::vector<float> native_face_weights;
    std::vector<float> native_camera_resolution;
    std::vector<std::int32_t> native_unary;
    std::vector<std::uint8_t> native_edge_bytes;
    double diagnostic_pairwise_scale = 1.0;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_PAIRWISE_SCALE")) {
        diagnostic_pairwise_scale = std::stod(value);
    }
    if (!cameras.empty() &&
        cameras.front().exact_face_weight.size() == mesh.faces.size()) {
        native_face_weights = cameras.front().exact_face_weight;
    }
    if (!cameras.empty() &&
        std::all_of(cameras.begin(), cameras.end(), [&](const Camera &camera) {
            return camera.exact_resolution.size() == mesh.faces.size();
        })) {
        native_camera_resolution.resize(mesh.faces.size() * cameras.size());
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
                native_camera_resolution[face_index * cameras.size() + camera_index] =
                    cameras[camera_index].exact_resolution[face_index];
            }
        }
    }
    if (const char *weights_path = std::getenv("METASHAPE_TEXTURE_NATIVE_RESOLUTION_PATH")) {
        std::ifstream weights(weights_path, std::ios::binary);
        if (!weights) throw std::runtime_error("cannot open native resolution fixture");
        std::array<std::uint8_t, 16> header{};
        weights.read(reinterpret_cast<char *>(header.data()), header.size());
        native_face_weights.resize(mesh.faces.size());
        weights.read(reinterpret_cast<char *>(native_face_weights.data()),
                     static_cast<std::streamsize>(native_face_weights.size() * sizeof(float)));
        if (!weights) throw std::runtime_error("native resolution fixture has wrong length");
    }
    if (const char *resolution_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_CAMERA_RESOLUTION_DIR")) {
        native_camera_resolution.resize(mesh.faces.size() * cameras.size());
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            std::ostringstream camera_name;
            camera_name << "camera_" << std::setw(3) << std::setfill('0') << camera_index;
            const auto path = std::filesystem::path(resolution_root) / camera_name.str() /
                              "binding_02.raw.f32";
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                throw std::runtime_error("cannot open native camera-resolution fixture: " +
                                         path.string());
            }
            std::vector<float> camera_values(mesh.faces.size());
            input.read(reinterpret_cast<char *>(camera_values.data()),
                       static_cast<std::streamsize>(camera_values.size() * sizeof(float)));
            if (!input) {
                throw std::runtime_error("native camera-resolution fixture has wrong length: " +
                                         path.string());
            }
            for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
                native_camera_resolution[face_index * cameras.size() + camera_index] =
                    camera_values[face_index];
            }
        }
    }
    if (const char *unary_path = std::getenv("METASHAPE_TEXTURE_NATIVE_UNARY_PATH")) {
        std::ifstream costs(unary_path, std::ios::binary);
        if (!costs) throw std::runtime_error("cannot open native unary fixture");
        native_unary.resize(mesh.faces.size() * static_cast<std::size_t>(label_count));
        costs.read(reinterpret_cast<char *>(native_unary.data()),
                   static_cast<std::streamsize>(native_unary.size() * sizeof(std::int32_t)));
        if (!costs) throw std::runtime_error("native unary fixture has wrong length");
    }
    if (const char *edge_path = std::getenv("METASHAPE_TEXTURE_NATIVE_EDGE_BYTES_PATH")) {
        std::ifstream bytes(edge_path, std::ios::binary);
        if (!bytes) throw std::runtime_error("cannot open native edge-byte fixture");
        native_edge_bytes.resize(mesh.faces.size() * 3 * static_cast<std::size_t>(label_count));
        bytes.read(reinterpret_cast<char *>(native_edge_bytes.data()),
                   static_cast<std::streamsize>(native_edge_bytes.size()));
        if (!bytes) throw std::runtime_error("native edge-byte fixture has wrong length");
    }
    for (auto &face_costs : edge_costs) {
        for (auto &values : face_costs) values.assign(static_cast<std::size_t>(label_count), 0);
    }
    const PixelEdgeVisibility pixel_edge_visibility = pairwise_quality_enabled
        ? build_face_id_edge_visibility(mesh, cameras)
        : PixelEdgeVisibility(mesh.faces.size());
    std::vector<std::uint8_t> diagnostic_edge_gate;
    if (std::getenv("METASHAPE_TEXTURE_EDGE_GATE_DUMP") != nullptr) {
        diagnostic_edge_gate.assign(mesh.faces.size() * 3 * cameras.size(), 0);
    }
    std::size_t without_winner = 0;
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
        std::vector<double> resolution(cameras.size(), -1.0);
        std::vector<ProjectedFace> projections(cameras.size());
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            if (!native_camera_resolution.empty()) {
                resolution[camera_index] = native_camera_resolution[
                    face_index * cameras.size() + camera_index];
                projections[camera_index] =
                    project_face(mesh, mesh.faces[face_index], cameras[camera_index]);
                continue;
            }
            auto projected = project_face(mesh, mesh.faces[face_index], cameras[camera_index]);
            projections[camera_index] = projected;
            if (!projected.valid) continue;
            const char *resolution_mode = std::getenv("METASHAPE_TEXTURE_RESOLUTION_MODE");
            if (resolution_mode == nullptr || std::string(resolution_mode) == "area") {
                const double recovered_resolution = recovered_face_resolution(
                    projected, cameras[camera_index], depth_downscale);
                bool visible = recovered_resolution > 0.0;
                if (const char *visibility_mode = std::getenv("METASHAPE_TEXTURE_VISIBILITY_MODE")) {
                    const std::string mode = visibility_mode;
                    if (mode == "raster") {
                        visible = raster_face_metrics(projected, cameras[camera_index],
                                                      depth_downscale).resolution > 0.0;
                    } else if (mode == "raster1-strict" || mode == "raster2-strict" ||
                               mode == "raster4-strict") {
                        const int raster_downscale = mode == "raster1-strict" ? 1 :
                                                     mode == "raster2-strict" ? 2 : 4;
                        visible = face_has_visible_fragment(projected, cameras[camera_index],
                                                            depth_downscale, raster_downscale);
                    }
                }
                if (!visible) {
                    resolution[camera_index] = recovered_resolution;
                    continue;
                }
                resolution[camera_index] = recovered_resolution;
            } else {
                const auto metrics = raster_face_metrics(projected, cameras[camera_index], depth_downscale);
                if (metrics.resolution <= 0.0) continue;
                resolution[camera_index] = metrics.resolution;
            }
        }
        double first_resolution = 0.0;
        double second_resolution = 0.0;
        for (const double value : resolution) {
            if (value >= first_resolution) {
                second_resolution = first_resolution;
                first_resolution = value;
            } else if (value >= second_resolution) {
                second_resolution = value;
            }
        }
        // update_faces_resolution_with_proj_area.comp keeps the two greatest
        // camera resolutions per face.  The candidate shader receives the
        // second-greatest buffer (resolution.dat), not the maximum.
        double maximum = native_face_weights.empty()
            ? second_resolution
            : static_cast<double>(native_face_weights[face_index]);
        if (!diagnostic_camera_resolution.empty()) {
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
                diagnostic_camera_resolution[face_index * cameras.size() + camera_index] =
                    static_cast<float>(resolution[camera_index]);
            }
        }
        auto is_edge_candidate = [&](std::size_t camera_index) {
            const bool resolution_candidate =
                resolution[camera_index] > 0.25 * maximum;
            return maximum > 0.0 && resolution_candidate;
        };
        auto is_unary_available = [&](std::size_t camera_index) {
            if (!native_unary.empty()) {
                return native_unary[face_index * static_cast<std::size_t>(label_count) +
                                    camera_index] < unavailable_camera_unary;
            }
            // Native FaceCameraOptions exist for both positive projected-area
            // values and the -1 partial-visibility marker.  Only exact zero is
            // unavailable.  The 25%-of-second-best test belongs solely to the
            // edge-quality shaders.
            return recovered::candidate_exists_from_scale4_resolution(
                static_cast<float>(resolution[camera_index]));
        };
        std::vector<double> candidates;
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            if (resolution[camera_index] > 0.0) {
                candidates.push_back(resolution[camera_index]);
            }
        }
        double range = 1.0;
        double lower = 0.0;
        if (candidates.empty()) {
            ++without_winner;
        } else {
            const auto [minimum_it, maximum_it] =
                std::minmax_element(candidates.begin(), candidates.end());
            range = std::max(*maximum_it - *minimum_it, 0.2 * *maximum_it);
            lower = std::max(*minimum_it, *maximum_it - range);
        }
        const auto consistency = ghosting_filter
            ? estimate_face_consistency(mesh, mesh.faces[face_index], cameras, projections, depth_downscale)
            : std::vector<double>(cameras.size(), -1.0);
        double consistency_minimum = std::numeric_limits<double>::infinity();
        double consistency_maximum = -std::numeric_limits<double>::infinity();
        if (ghosting_filter) {
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
                if (!is_unary_available(camera_index) || consistency[camera_index] < 0.0) continue;
                consistency_minimum = std::min(consistency_minimum, consistency[camera_index]);
                consistency_maximum = std::max(consistency_maximum, consistency[camera_index]);
            }
        }
        diagnostic_consistency_max[face_index] = consistency_maximum;
        diagnostic_consistency_range[face_index] =
            std::isfinite(consistency_minimum)
                ? consistency_maximum - consistency_minimum : -1.0;
        const double consistency_range = std::isfinite(consistency_minimum)
            ? std::max(consistency_maximum - consistency_minimum, 0.15)
            : 1.0;
        const double consistency_lower = std::isfinite(consistency_minimum)
            ? std::max(consistency_minimum, consistency_maximum - consistency_range)
            : 0.0;
        // This fallback now preserves the recovered max2 empty/single-camera
        // semantics: with fewer than two accepted inputs, the second-greatest
        // value remains zero.  The standalone scale-2 Vulkan project chain is
        // the bit-exact path; this CPU projection path remains diagnostic.
        const double face_weight = native_face_weights.empty()
            ? second_resolution
            : native_face_weights[face_index];
        diagnostic_face_weights[face_index] = face_weight;
        const double clamped_face_weight = std::clamp(face_weight, 0.1, 1000.0);

        // update_edge_quality_with_proj_sampling.comp processes the three
        // edges inside one face/camera invocation.  Pixel-stage hits already
        // present in the accumulation buffer are skipped.  The first missing
        // edge whose projected midpoint fails visibility returns from the
        // whole invocation, so no later missing edge is sampled.
        std::array<std::vector<std::uint8_t>, 3> edge_gate_flags;
        for (auto &flags : edge_gate_flags) flags.assign(cameras.size(), 0);
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            if (!pairwise_quality_enabled) continue;
            if (!is_edge_candidate(camera_index)) continue;
            bool projection_sampling_returned = false;
            for (int edge = 0; edge < 3; ++edge) {
                const bool exact_edge_gate =
                    cameras[camera_index].exact_edge_count.size() == mesh.faces.size() * 3U;
                if (exact_edge_gate) {
                    edge_gate_flags[static_cast<std::size_t>(edge)][camera_index] =
                        cameras[camera_index].exact_edge_count[
                            face_index * 3U + static_cast<std::size_t>(edge)] != 0.0F ? 1U : 0U;
                    continue;
                }
                const bool pixel_gate =
                    pixel_edge_visibility[face_index][static_cast<std::size_t>(edge)]
                                         [camera_index] != 0;
                if (pixel_gate) {
                    edge_gate_flags[static_cast<std::size_t>(edge)][camera_index] = 1;
                    continue;
                }
                if (projection_sampling_returned) continue;
                const bool midpoint_gate = projected_edge_midpoint_visible(
                    mesh, face_index, edge, projections[camera_index],
                    cameras[camera_index], depth_downscale);
                if (midpoint_gate) {
                    edge_gate_flags[static_cast<std::size_t>(edge)][camera_index] = 2;
                } else {
                    projection_sampling_returned = true;
                }
            }
        }

        const Face &mesh_face = mesh.faces[face_index];
        // FUN_141f03350 does not use the mesh's double-precision working
        // values here.  Each vertex component is first narrowed to float32;
        // the squared length is accumulated in float32, sqrt is evaluated in
        // double, and the result is narrowed back to float32.  Preserving this
        // order is necessary at the final uint8 rounding boundaries.
        std::array<float, 3> world_edge_length{};
        for (int edge = 0; edge < 3; ++edge) {
            const Vec3d a = mesh.vertices[mesh_face.vertex[edge]];
            const Vec3d b = mesh.vertices[mesh_face.vertex[(edge + 1) % 3]];
            const float ax = static_cast<float>(a.x);
            const float ay = static_cast<float>(a.y);
            const float az = static_cast<float>(a.z);
            const float bx = static_cast<float>(b.x);
            const float by = static_cast<float>(b.y);
            const float bz = static_cast<float>(b.z);
            const float dx = bx - ax;
            const float dy = by - ay;
            const float dz = bz - az;
            const float squared_xy = dx * dx + dy * dy;
            const float squared_length = squared_xy + dz * dz;
            world_edge_length[edge] =
                static_cast<float>(std::sqrt(static_cast<double>(squared_length)));
        }
        const float perimeter01 = world_edge_length[0] + world_edge_length[1];
        const float perimeter = perimeter01 + world_edge_length[2];
        const float semiperimeter = perimeter * 0.5F;
        float area_product = (semiperimeter - world_edge_length[0]) * semiperimeter;
        area_product *= semiperimeter - world_edge_length[1];
        area_product *= semiperimeter - world_edge_length[2];
        const float world_area = std::sqrt(area_product);
        const float resolution_scale =
            std::sqrt(static_cast<float>(face_weight) / world_area);
        for (int edge = 0; edge < 3; ++edge) {
            const float edge_measure = std::clamp(
                world_edge_length[edge] * resolution_scale, 0.1F, 36.5F);
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
                if (!pairwise_quality_enabled) continue;
                // Native edge records are emitted for FaceCameraOptions, i.e.
                // only after the 25%-of-best camera candidate gate.
                if (!is_edge_candidate(camera_index)) continue;
                const std::uint8_t native_gate =
                    edge_gate_flags[static_cast<std::size_t>(edge)][camera_index];
                bool emit_edge_record = native_gate != 0;
                if (!diagnostic_edge_gate.empty()) {
                    const std::size_t gate_index =
                        (face_index * 3 + static_cast<std::size_t>(edge)) * cameras.size() +
                        camera_index;
                    diagnostic_edge_gate[gate_index] = native_gate;
                }
                if (const char *edge_visibility =
                        std::getenv("METASHAPE_TEXTURE_EDGE_VISIBILITY")) {
                    const std::string mode = edge_visibility;
                    if (mode == "all") emit_edge_record = true;
                    if (mode == "line") {
                        emit_edge_record = projected_edge_has_visible_sample(
                            projections[camera_index], cameras[camera_index],
                            depth_downscale, edge);
                    }
                }
                if (!emit_edge_record) continue;
                const float quality = consistency[camera_index] >= 0.0
                    ? static_cast<float>(consistency[camera_index]) : 0.0F;
                // The native functor first rounds a float32 [0,36500] cost,
                // then performs a second float32 round after scaling to 255.
                // Default/no-quality branch: roundf((measure*1000)*0.4).
                // Optional quality branch: roundf((((1-q)*0.5)+0.025)*measure*1000).
                const float raw_36500 = ghosting_filter
                    ? (((1.0F - quality) * 0.5F + 0.025F) * edge_measure) * 1000.0F
                    : (edge_measure * 1000.0F) * 0.4F;
                const float quantized_36500 = std::round(raw_36500);
                const float scaled_255 = (quantized_36500 * 255.0F) / 36500.0F;
                const int byte = static_cast<int>(std::round(scaled_255));
                edge_costs[face_index][static_cast<std::size_t>(edge)][camera_index] =
                    static_cast<std::int64_t>(std::llround(
                        byte * 36500.0 / 255.0 * diagnostic_pairwise_scale));
            }
        }
        std::vector<std::size_t> unary_cameras;
        std::vector<recovered::FaceCameraQuality> unary_options;
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            if (!is_unary_available(camera_index)) continue;
            ++diagnostic_camera_counts[face_index];
            const auto projected = project_face(mesh, mesh.faces[face_index], cameras[camera_index]);
            const float nadirness = cameras[camera_index].exact_nadirness.empty()
                ? static_cast<float>(std::clamp(-projected.view_dot, 0.0, 1.0))
                : cameras[camera_index].exact_nadirness[face_index];
            if (!diagnostic_nadirness.empty()) {
                diagnostic_nadirness[face_index * cameras.size() + camera_index] =
                    nadirness;
            }
            const float resolution_value = static_cast<float>(resolution[camera_index]);
            const float sharpness = out_of_focus_filter
                ? estimate_face_focus_quality(face_index, projections[camera_index],
                    cameras[camera_index], depth_downscale, resolution_value,
                    static_cast<float>(face_weight))
                : -1.0F;
            const float consistency_value = ghosting_filter
                ? static_cast<float>(std::max(consistency[camera_index], 0.0))
                : -1.0F;
            unary_cameras.push_back(camera_index);
            unary_options.push_back({sharpness, consistency_value,
                                     resolution_value, nadirness});
        }
        const auto recovered_costs = recovered::build_face_camera_unary(
            unary_options, static_cast<float>(face_weight),
            {out_of_focus_filter, ghosting_filter, true, true});
        for (std::size_t option = 0; option < recovered_costs.size(); ++option) {
            unary[face_index][unary_cameras[option]] = recovered_costs[option];
        }
    }
    if (!native_unary.empty()) {
        for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
            for (int label = 0; label < label_count; ++label) {
                unary[face][static_cast<std::size_t>(label)] =
                    native_unary[face * static_cast<std::size_t>(label_count) +
                                 static_cast<std::size_t>(label)];
            }
        }
    }
    if (!native_edge_bytes.empty()) {
        // The captured byte functor is laid out as [face][mesh edge][label].
        // The mesh-edge numbering already matches build_face_adjacency(); no
        // cyclic remapping is performed by the native optimizer.
        std::array<int, 3> native_slot_for_clean_slot{0, 1, 2};
        if (const char *permutation = std::getenv("METASHAPE_TEXTURE_NATIVE_EDGE_PERM")) {
            std::istringstream values(permutation);
            char comma1 = 0, comma2 = 0;
            if (!(values >> native_slot_for_clean_slot[0] >> comma1 >>
                  native_slot_for_clean_slot[1] >> comma2 >> native_slot_for_clean_slot[2]) ||
                comma1 != ',' || comma2 != ',') {
                throw std::runtime_error("native edge permutation must be a,b,c");
            }
        }
        for (std::size_t face = 0; face < mesh.faces.size(); ++face) {
            for (int clean_slot = 0; clean_slot < 3; ++clean_slot) {
                const int native_slot = native_slot_for_clean_slot[clean_slot];
                if (native_slot < 0 || native_slot >= 3) {
                    throw std::runtime_error("native edge permutation index is out of range");
                }
                for (int label = 0; label < label_count; ++label) {
                    const std::size_t index =
                        (face * 3 + static_cast<std::size_t>(native_slot)) *
                            static_cast<std::size_t>(label_count) + static_cast<std::size_t>(label);
                    edge_costs[face][static_cast<std::size_t>(clean_slot)]
                              [static_cast<std::size_t>(label)] =
                        static_cast<std::int64_t>(std::llround(
                            native_edge_bytes[index] * 36500.0 / 255.0));
                }
            }
        }
    }
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_UNARY_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary);
        if (!dump) throw std::runtime_error("cannot create unary dump");
        for (const auto &face_costs : unary) {
            for (const auto cost : face_costs) {
                const auto value = static_cast<std::int32_t>(cost);
                dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
        }
    }
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_EDGE_BYTES_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary);
        if (!dump) throw std::runtime_error("cannot create edge-byte dump");
        for (const auto &face_costs : edge_costs) {
            for (const auto &slot_costs : face_costs) {
                for (const auto cost : slot_costs) {
                    const auto value = static_cast<std::uint8_t>(std::llround(
                        std::clamp(cost, 0.0F, 36500.0F) * 255.0F / 36500.0F));
                    dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
                }
            }
        }
    }
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_EDGE_GATE_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary);
        if (!dump) throw std::runtime_error("cannot create edge-gate dump");
        dump.write(reinterpret_cast<const char *>(diagnostic_edge_gate.data()),
                   static_cast<std::streamsize>(diagnostic_edge_gate.size()));
    }
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_CAMERA_RESOLUTION_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary);
        if (!dump) throw std::runtime_error("cannot create camera-resolution dump");
        dump.write(reinterpret_cast<const char *>(diagnostic_camera_resolution.data()),
                   static_cast<std::streamsize>(diagnostic_camera_resolution.size() *
                                                sizeof(float)));
    }
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_NADIRNESS_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary);
        if (!dump) throw std::runtime_error("cannot create nadirness dump");
        dump.write(reinterpret_cast<const char *>(diagnostic_nadirness.data()),
                   static_cast<std::streamsize>(diagnostic_nadirness.size() * sizeof(float)));
    }
    for (std::size_t face = 0; face < winners.size(); ++face) {
        winners[face] = static_cast<int>(std::min_element(unary[face].begin(), unary[face].end()) -
                                         unary[face].begin());
    }
    bool enable_graph = true;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_DISABLE_GRAPH")) {
        enable_graph = std::string(value) != "1";
    }
    if (enable_graph) {
        const auto global_adjacency = build_face_adjacency(mesh);
        const char *partition_path = std::getenv("METASHAPE_TEXTURE_NATIVE_PARTITIONS_PATH");
        const auto partitions = partition_path == nullptr
                                    ? recover_graph_partitions(mesh)
                                    : load_graph_partitions(partition_path, mesh.faces.size());
            std::vector<int> core_writes(mesh.faces.size(), 0);
            std::ofstream partition_dump;
            if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_PARTITION_LABELS_DUMP")) {
                partition_dump.open(dump_path);
                if (!partition_dump) throw std::runtime_error("cannot create partition-label dump");
            }
            for (std::size_t partition_index = 0; partition_index < partitions.size();
                 ++partition_index) {
                const auto &partition = partitions[partition_index];
                std::vector<int> global_to_local(mesh.faces.size(), -1);
                for (std::size_t local = 0; local < partition.global_faces.size(); ++local) {
                    global_to_local[partition.global_faces[local]] = static_cast<int>(local);
                }
                std::vector<int> local_labels(partition.global_faces.size(), 0);
                std::vector<std::vector<std::int64_t>> local_unary;
                local_unary.reserve(partition.global_faces.size());
                DirectedEdgeCosts local_edge_costs;
                local_edge_costs.reserve(partition.global_faces.size());
                for (const auto global : partition.global_faces) {
                    local_unary.push_back(unary[global]);
                    local_edge_costs.push_back(edge_costs[global]);
                }
                std::vector<FaceAdjacency> local_adjacency;
                for (const auto &edge : global_adjacency) {
                    const int first = global_to_local[static_cast<std::size_t>(edge.first)];
                    const int second = global_to_local[static_cast<std::size_t>(edge.second)];
                    if (first >= 0 && second >= 0) {
                        local_adjacency.push_back(
                            {first, second, edge.first_edge, edge.second_edge});
                    }
                }
                alpha_expand(local_labels, local_unary, local_adjacency,
                             local_edge_costs, label_count);
                for (std::size_t local = 0; local < partition.global_faces.size(); ++local) {
                    if (partition_dump) {
                        partition_dump << partition_index << ' ' << local << ' '
                                       << partition.global_faces[local] << ' '
                                       << local_labels[local] << ' '
                                       << static_cast<int>(partition.core[local]) << '\n';
                    }
                    if (!partition.core[local]) continue;
                    const auto global = partition.global_faces[local];
                    winners[global] = local_labels[local];
                    ++core_writes[global];
                }
            }
            if (std::any_of(core_writes.begin(), core_writes.end(),
                            [](int count) { return count != 1; })) {
                throw std::runtime_error("partitioned optimization did not write each face once");
            }
    }
    for (int &winner : winners) if (winner == no_camera_label) winner = -1;
    if (const char *dump_path = std::getenv("METASHAPE_TEXTURE_METRICS_DUMP")) {
        std::ofstream dump(dump_path);
        if (!dump) throw std::runtime_error("cannot create metrics dump");
        for (std::size_t face = 0; face < diagnostic_face_weights.size(); ++face) {
            dump << diagnostic_face_weights[face] << ' '
                 << diagnostic_consistency_max[face] << ' '
                 << diagnostic_consistency_range[face] << ' '
                 << diagnostic_camera_counts[face] << '\n';
        }
    }
    std::cout << "winner estimation: " << without_winner << '/' << mesh.faces.size() << " faces without camera\n";
    return winners;
}

cv::Vec3f bilinear(const cv::Mat &image, double x, double y) {
    x = std::clamp(x, 0.0, static_cast<double>(image.cols - 1));
    y = std::clamp(y, 0.0, static_cast<double>(image.rows - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float fx = static_cast<float>(x - x0);
    const float fy = static_cast<float>(y - y0);
    return image.at<cv::Vec3f>(y0, x0) * ((1 - fx) * (1 - fy)) +
           image.at<cv::Vec3f>(y0, x1) * (fx * (1 - fy)) +
           image.at<cv::Vec3f>(y1, x0) * ((1 - fx) * fy) +
           image.at<cv::Vec3f>(y1, x1) * (fx * fy);
}

float bilinear_scalar(const cv::Mat &image, double x, double y) {
    x = std::clamp(x, 0.0, static_cast<double>(image.cols - 1));
    y = std::clamp(y, 0.0, static_cast<double>(image.rows - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float fx = static_cast<float>(x - x0);
    const float fy = static_cast<float>(y - y0);
    return image.at<float>(y0, x0) * ((1 - fx) * (1 - fy)) +
           image.at<float>(y0, x1) * (fx * (1 - fy)) +
           image.at<float>(y1, x0) * ((1 - fx) * fy) +
           image.at<float>(y1, x1) * (fx * fy);
}

cv::Vec3f sample_linear_texture(const cv::Mat &image, double edge_x, double edge_y) {
    // Vulkan normalized linear sampling maps a coordinate of x / width to
    // texel-space x - 0.5.  The recovered enblend shader adds 0.5 only for
    // level zero; all coarser image-pyramid samples use the edge coordinate
    // directly.
    return bilinear(image, edge_x - 0.5, edge_y - 0.5);
}

float sample_linear_texture_scalar(const cv::Mat &image, double edge_x, double edge_y) {
    return bilinear_scalar(image, edge_x - 0.5, edge_y - 0.5);
}

struct CameraBlendPyramid {
    // The 640x480 RGB fixture selects the five-level, three-channel shader
    // specialization.  Each successive image level is four times coarser.
    std::array<cv::Mat, 5> image;
    std::array<cv::Mat, 5> weight;
    bool has_seed = false;
};

cv::Mat capped_distance_field(const cv::Mat &coverage, const cv::Mat &seeds,
                              float maximum_distance, bool boundary_mode) {
    // weights_distances_via_bfs.comp is a Jacobi relaxation, not a global
    // shortest-path solve.  Its ping-pong images are R16F, so every dispatch
    // rounds the result to binary16.  Reproducing both details is necessary:
    // the old float32 Dijkstra approximation differed at seam endpoints and
    // could change whether a coarse atlas fragment received any weight.
    cv::Mat current(coverage.size(), CV_32F, cv::Scalar(maximum_distance));
    // Host dispatch trace and the controlling loop both show an inclusive
    // 0..max_distance range: max=5 produces six module144 dispatches.
    const int passes = std::max(1, static_cast<int>(maximum_distance) + 1);
    for (int pass = 0; pass < passes; ++pass) {
        cv::Mat next(coverage.size(), CV_32F, cv::Scalar(maximum_distance));
        for (int y = 0; y < coverage.rows; ++y) {
            for (int x = 0; x < coverage.cols; ++x) {
                const bool border = x == 0 || y == 0 || x + 1 == coverage.cols ||
                                    y + 1 == coverage.rows;
                const bool constrained = border || coverage.at<std::uint8_t>(y, x) == 0;
                if (boundary_mode && constrained) {
                    next.at<float>(y, x) = 0.0F;
                    continue;
                }
                if (!boundary_mode && constrained) continue;
                if (!boundary_mode && seeds.at<std::uint8_t>(y, x) != 0) {
                    next.at<float>(y, x) = 0.0F;
                    continue;
                }
                float best = maximum_distance;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = x + dx;
                        const int sy = y + dy;
                        if (sx < 0 || sy < 0 || sx >= coverage.cols || sy >= coverage.rows) {
                            continue;
                        }
                        if (!boundary_mode) {
                            if (coverage.at<std::uint8_t>(sy, sx) == 0) continue;
                            if (dx != 0 && dy != 0 &&
                                coverage.at<std::uint8_t>(y, sx) == 0 &&
                                coverage.at<std::uint8_t>(sy, x) == 0) {
                                continue;
                            }
                        }
                        const float old = current.at<float>(sy, sx);
                        best = std::min(best, std::min(maximum_distance,
                            old + std::sqrt(static_cast<float>(dx * dx + dy * dy))));
                    }
                }
                next.at<float>(y, x) = best;
            }
        }
        // Round-trip once per dispatch to mirror an R16F imageStore/load.
        for (int y = 0; y < next.rows; ++y) {
            for (int x = 0; x < next.cols; ++x) {
                current.at<float>(y, x) = positive_half_to_float(
                    positive_float_to_half(next.at<float>(y, x)));
            }
        }
    }
    return current;
}

cv::Mat recovered_distance_weights(const cv::Mat &coverage, const cv::Mat &winner_value,
                                   float maximum_distance) {
    cv::Mat winner_mask(coverage.size(), CV_8U, cv::Scalar(0));
    cv::Mat nonwinner_mask(coverage.size(), CV_8U, cv::Scalar(0));
    cv::Mat unused(coverage.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < coverage.rows; ++y) {
        for (int x = 0; x < coverage.cols; ++x) {
            if (coverage.at<std::uint8_t>(y, x) == 0) continue;
            if (winner_value.at<float>(y, x) == 1.0F) {
                winner_mask.at<std::uint8_t>(y, x) = 255;
            } else {
                nonwinner_mask.at<std::uint8_t>(y, x) = 255;
            }
        }
    }
    const cv::Mat boundary_distance = capped_distance_field(coverage, unused, maximum_distance, true);
    const cv::Mat winner_distance = capped_distance_field(coverage, winner_mask, maximum_distance, false);
    const cv::Mat nonwinner_distance = capped_distance_field(coverage, nonwinner_mask, maximum_distance, false);
    cv::Mat weight(coverage.size(), CV_32F, cv::Scalar(0));
    const float half_distance = maximum_distance * 0.5F;
    for (int y = 0; y < coverage.rows; ++y) {
        for (int x = 0; x < coverage.cols; ++x) {
            const float edge = boundary_distance.at<float>(y, x);
            const float to_winner = winner_distance.at<float>(y, x);
            const float to_nonwinner = nonwinner_distance.at<float>(y, x);
            float value;
            if (to_nonwinner >= half_distance) {
                value = 1.0F;
            } else if (to_winner >= half_distance) {
                value = 0.0F;
            } else if (to_winner == 0.0F) {
                value = 0.5F + 0.5F * to_nonwinner / half_distance;
            } else {
                value = 0.5F - 0.5F * to_winner / half_distance;
            }
            if (edge < maximum_distance) value *= std::min(1.0F, edge / maximum_distance);
            // The native combination shader writes back into the current
            // winner/value image.  A previously positive value is retained as
            // epsilon even when the newly combined distance weight is zero.
            if (winner_value.at<float>(y, x) > 0.0F) value = std::max(value, 1.0e-7F);
            weight.at<float>(y, x) = value;
        }
    }
    return weight;
}

std::pair<cv::Mat, cv::Mat> downscale_low_frequency_level(const cv::Mat &value,
                                                           const cv::Mat &coverage) {
    const int width = (value.cols + 1) / 2;
    const int height = (value.rows + 1) / 2;
    cv::Mat output_value(height, width, CV_32F, cv::Scalar(0));
    cv::Mat output_coverage(height, width, CV_8U, cv::Scalar(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0F;
            int count = 0;
            bool constrained = false;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = 2 * x + dx;
                    const int sy = 2 * y + dy;
                    if (sx >= value.cols || sy >= value.rows) continue;
                    sum += value.at<float>(sy, sx);
                    ++count;
                    constrained = constrained || coverage.at<std::uint8_t>(sy, sx) == 0;
                }
            }
            output_value.at<float>(y, x) = sum / static_cast<float>(count);
            // The shader's integer image is an obstacle/constraint mask, so
            // its "any child == 255" rule means all children must be valid in
            // this complementary coverage representation.
            output_coverage.at<std::uint8_t>(y, x) = constrained ? 0 : 255;
        }
    }
    // After every module145 downscale Metashape downloads the newly produced
    // constraint image, removes 8-connected nonzero components smaller than
    // 10 pixels, and uploads it before module144 consumes the level.  This is
    // distinct from the full-resolution pre-module132 threshold of 20.
    Image<std::uint8_t> reduced_constraint(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            reduced_constraint(x, y) =
                output_coverage.at<std::uint8_t>(y, x) == 0 ? 255 : 0;
        }
    }
    reduced_constraint = recovered::remove_small_components_8(
        reduced_constraint, 10, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output_coverage.at<std::uint8_t>(y, x) =
                reduced_constraint(x, y) == 255 ? 0 : 255;
        }
    }
    return {output_value, output_coverage};
}

cv::Mat heat_diffuse_with_constraints(cv::Mat current, const cv::Mat &coverage, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        cv::Mat next(current.size(), CV_32F, cv::Scalar(0));
        for (int y = 1; y + 1 < current.rows; ++y) {
            for (int x = 1; x + 1 < current.cols; ++x) {
                if (coverage.at<std::uint8_t>(y, x) == 0) continue;
                const float old = current.at<float>(y, x);
                float value = 0.2F * (old + current.at<float>(y, x - 1) +
                                      current.at<float>(y, x + 1) +
                                      current.at<float>(y - 1, x) +
                                      current.at<float>(y + 1, x));
                if (old > 0.0F) value = std::max(value, 1.0e-7F);
                next.at<float>(y, x) = value;
            }
        }
        current = std::move(next);
    }
    return current;
}

cv::Mat square_minimum_filter(const cv::Mat &input, int radius) {
    cv::Mat output(input.size(), CV_32F);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            float value = input.at<float>(y, x);
            for (int sy = std::max(0, y - radius); sy <= std::min(input.rows - 1, y + radius); ++sy) {
                for (int sx = std::max(0, x - radius); sx <= std::min(input.cols - 1, x + radius); ++sx) {
                    value = std::min(value, input.at<float>(sy, sx));
                }
            }
            output.at<float>(y, x) = value;
        }
    }
    return output;
}

CameraBlendPyramid build_camera_blend_pyramid(const Mesh &mesh, const Camera &camera,
                                                const std::vector<int> &winners,
                                                int camera_index, int downscale,
                                                int depth_downscale) {
    CameraBlendPyramid pyramid;
    pyramid.image[0] = camera.image;
    auto fixture_path = [&](const char *root, std::string_view suffix) {
        std::ostringstream name;
        name << "camera_" << std::setw(3) << std::setfill('0') << camera_index << suffix;
        return std::filesystem::path(root) / name.str();
    };
    if (const char *native_root = std::getenv("METASHAPE_TEXTURE_NATIVE_BLEND_FIXTURE_DIR")) {
        const auto path = fixture_path(native_root, "_image.tiff");
        cv::Mat encoded = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (encoded.empty()) throw std::runtime_error("cannot load native blend image: " + path.string());
        encoded.convertTo(pyramid.image[0], CV_32FC3, 1.0 / 255.0);
        // This fixture is the native prepared image consumed by enblend.
        // Applying image_sharpening a second time measurably diverges from the
        // native texture, so preserve the captured R8 values exactly here.
    }
    const char *image_resolution_weights =
        std::getenv("METASHAPE_TEXTURE_BLEND_WEIGHT_AT_IMAGE_RES");
    const bool weight_at_image_resolution = image_resolution_weights == nullptr ||
        std::string(image_resolution_weights) != "0";
    const int weight_downscale = weight_at_image_resolution ? downscale : depth_downscale;
    const cv::Size weight_size = weight_at_image_resolution
        ? camera.image.size() : camera.depth.size();
    // Native BFS is allowed to propagate across the full camera plane.  Its
    // mask contains only explicitly rendered occlusion edges; background and
    // pixels outside the projected model are not constraints.
    cv::Mat coverage(weight_size, CV_8U, cv::Scalar(255));
    cv::Mat visible_support(weight_size, CV_8U, cv::Scalar(0));
    cv::Mat visible_face_id(weight_size, CV_32S, cv::Scalar(-1));
    cv::Mat visible_depth(weight_size, CV_32F,
                          cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat occlusion_depth(weight_size, CV_32F,
                            cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat occlusion_raster_candidate(weight_size, CV_8U, cv::Scalar(0));
    cv::Mat occlusion_depth_pass(weight_size, CV_8U, cv::Scalar(0));
    cv::Mat occlusion_fixed_depth_pass(weight_size, CV_8U, cv::Scalar(0));
    cv::Mat occlusion_edge_pass(weight_size, CV_8U, cv::Scalar(0));
    cv::Mat occlusion_visibility_ratio(
        weight_size, CV_32F,
        cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat winner_value(weight_size, CV_32F, cv::Scalar(0));
    cv::Mat low_frequency_quality(weight_size, CV_32F,
                                  cv::Scalar(std::numeric_limits<float>::max()));
    // occlusion_edges.frag samples a depth attachment at its own viewport
    // resolution.  Build that two-sided z-buffer directly instead of
    // reusing the later 1/4-resolution texture-quality depth image.
    for (const Face &face : mesh.faces) {
        const ProjectedFace projected = project_face(mesh, face, camera);
        if (projected.area <= 0.0) continue;
        std::array<Vec2d, 3> triangle = projected.pixel;
        for (Vec2d &point : triangle) {
            point.x /= weight_downscale;
            point.y /= weight_downscale;
        }
        raster_triangle(triangle, occlusion_depth.cols, occlusion_depth.rows,
                        [&](int x, int y, const std::array<double, 3> &barycentric) {
            const float depth = static_cast<float>(
                barycentric[0] * projected.depth[0] +
                barycentric[1] * projected.depth[1] +
                barycentric[2] * projected.depth[2]);
            occlusion_depth.at<float>(y, x) =
                std::min(occlusion_depth.at<float>(y, x), depth);
        });
    }
    // Diagnostic isolation only: use the exact sampled image captured from
    // occlusion_edges.frag binding 7.  This must never be enabled for an
    // independent end-to-end run.
    if (const char *native_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_OCCLUSION_DEPTH_DIR")) {
        std::ostringstream filename;
        filename << "camera_" << std::setfill('0') << std::setw(3) << camera_index
                 << "_depth_320x240x1_u4.raw";
        const auto path = std::filesystem::path(native_root) / filename.str();
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot load native occlusion depth: " +
                                     path.string());
        }
        stream.read(reinterpret_cast<char *>(occlusion_depth.data),
                    static_cast<std::streamsize>(occlusion_depth.total() * sizeof(float)));
        if (!stream || stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native occlusion-depth byte count: " +
                                     path.string());
        }
    }
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
        const auto projected = project_face(mesh, mesh.faces[face_index], camera);
        if (!projected.valid) continue;
        std::array<Vec2d, 3> triangle = projected.pixel;
        for (auto &point : triangle) { point.x /= weight_downscale; point.y /= weight_downscale; }
        const bool seed = winners[face_index] == camera_index;
        raster_triangle(triangle, coverage.cols, coverage.rows,
                        [&](int x, int y, const std::array<double, 3> &barycentric) {
                            const double inverse_depth = barycentric[0] / projected.depth[0] +
                                                         barycentric[1] / projected.depth[1] +
                                                         barycentric[2] / projected.depth[2];
                            if (inverse_depth <= 0.0) return;
                            const double depth = 1.0 / inverse_depth;
                            if (!fragment_visible(projected, camera, depth_downscale,
                                                  (x + 0.5) * weight_downscale,
                                                  (y + 0.5) * weight_downscale, depth)) return;
                            visible_support.at<std::uint8_t>(y, x) = 255;
                            if (depth < visible_depth.at<float>(y, x)) {
                                visible_depth.at<float>(y, x) = static_cast<float>(depth);
                                visible_face_id.at<std::int32_t>(y, x) =
                                    static_cast<std::int32_t>(face_index);
                            }
                            const float quality = static_cast<float>(
                                std::clamp(-projected.view_dot, 0.0, 1.0) *
                                camera.calibration.f / depth);
                            low_frequency_quality.at<float>(y, x) = std::min(
                                low_frequency_quality.at<float>(y, x), quality);
                            if (seed) winner_value.at<float>(y, x) = 1.0F;
                        });
        pyramid.has_seed = pyramid.has_seed || seed;
    }
    // occlusion_edges.frag tests only triangle-edge fragments.  For each
    // shared edge it projects the opposite vertex of the adjacent face and
    // emits 255 when that vertex and this face's opposite vertex lie on the
    // same image-space side of the edge.  This identifies folded silhouettes
    // and self-occluding contours without marking ordinary coplanar mesh
    // tessellation edges.
    const char *occlusion_mode = std::getenv("METASHAPE_TEXTURE_OCCLUSION_MASK_MODE");
    if (occlusion_mode == nullptr || std::string(occlusion_mode) != "none") {
        cv::Mat constraint(coverage.size(), CV_8U, cv::Scalar(0));
        if (occlusion_mode != nullptr && std::string(occlusion_mode) == "geometry") {
        std::vector<std::array<int, 3>> neighbor_opposite(mesh.faces.size(),
                                                          std::array<int, 3>{-1, -1, -1});
        for (const FaceAdjacency &adjacency : build_face_adjacency(mesh)) {
            neighbor_opposite[static_cast<std::size_t>(adjacency.first)]
                             [static_cast<std::size_t>(adjacency.first_edge)] =
                static_cast<int>(mesh.faces[static_cast<std::size_t>(adjacency.second)]
                                     .vertex[static_cast<std::size_t>(
                                         (adjacency.second_edge + 2) % 3)]);
            neighbor_opposite[static_cast<std::size_t>(adjacency.second)]
                             [static_cast<std::size_t>(adjacency.second_edge)] =
                static_cast<int>(mesh.faces[static_cast<std::size_t>(adjacency.first)]
                                     .vertex[static_cast<std::size_t>(
                                         (adjacency.first_edge + 2) % 3)]);
        }
        int adjacency_shift = 0;
        if (const char *value = std::getenv("METASHAPE_TEXTURE_OCCLUSION_ADJACENCY_SHIFT")) {
            adjacency_shift = ((std::stoi(value) % 3) + 3) % 3;
        }
        const bool skip_occlusion_depth =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_SKIP_DEPTH") != nullptr;
        const bool gpu_float_projection =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_FLOAT_PROJECTION") != nullptr;
        const bool gpu_float_fragment =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_FLOAT_FRAGMENT") != nullptr;
        const bool report_occlusion_stats =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_STATS") != nullptr;
        const bool conservative_raster =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_CONSERVATIVE_RASTER") != nullptr;
        const bool line_raster =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_LINE_RASTER") != nullptr;
        const bool fixed_function_depth_test =
            std::getenv("METASHAPE_TEXTURE_OCCLUSION_FIXED_DEPTH_TEST") != nullptr;
        double conservative_extent = 1.0;
        if (const char *value =
                std::getenv("METASHAPE_TEXTURE_OCCLUSION_CONSERVATIVE_EXTENT")) {
            conservative_extent = std::clamp(std::stod(value), 0.0, 2.0);
        }
        std::size_t boundary_edges = 0;
        std::size_t shared_edges = 0;
        std::size_t accepted_shared_edges = 0;
        std::size_t rasterized_fragments = 0;
        std::size_t depth_visible_fragments = 0;
        for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
            const ProjectedFace projected = gpu_float_projection
                ? project_face_gpu_float(mesh, mesh.faces[face_index], camera)
                : project_face(mesh, mesh.faces[face_index], camera);
            // The occlusion-edge draw is two-sided.  Back-facing triangles
            // are essential at folded silhouettes even though enblend later
            // rejects them with its face bit mask.
            if (projected.area <= 0.0) continue;
            std::array<Vec2d, 3> triangle = projected.pixel;
            for (Vec2d &point : triangle) {
                point.x /= weight_downscale;
                point.y /= weight_downscale;
            }
            std::array<bool, 3> occluding_edge{};
            for (int edge = 0; edge < 3; ++edge) {
                const int opposite_vertex =
                    neighbor_opposite[face_index][static_cast<std::size_t>(
                        (edge + adjacency_shift) % 3)];
                // Runtime capture shows that the native adjacency stream puts
                // this face's own opposite vertex in the slot for every open
                // boundary.  Both cross products are then identical, so the
                // GLSL sign comparison always accepts the boundary edge.
                if (opposite_vertex < 0) {
                    occluding_edge[static_cast<std::size_t>(edge)] = true;
                    ++boundary_edges;
                    continue;
                }
                ++shared_edges;
                const auto adjacent_gpu = gpu_float_projection
                    ? project_frame_gpu_float(
                          camera, mesh.vertices[static_cast<std::size_t>(opposite_vertex)])
                    : std::optional<GpuProjectedPoint>{};
                const auto adjacent_double = gpu_float_projection
                    ? std::optional<Vec2d>{}
                    : project_frame(camera.pose, camera.calibration,
                                    mesh.vertices[static_cast<std::size_t>(opposite_vertex)],
                                    false);
                const std::optional<Vec2d> adjacent = gpu_float_projection
                    ? (adjacent_gpu ? std::optional<Vec2d>{adjacent_gpu->pixel}
                                    : std::optional<Vec2d>{})
                    : adjacent_double;
                if (!adjacent) continue;
                const int next = (edge + 1) % 3;
                const int opposite = (edge + 2) % 3;
                const Vec2d a = projected.pixel[static_cast<std::size_t>(edge)];
                const Vec2d b = projected.pixel[static_cast<std::size_t>(next)];
                const Vec2d own = projected.pixel[static_cast<std::size_t>(opposite)];
                if (gpu_float_fragment) {
                    const float ax = static_cast<float>(a.x);
                    const float ay = static_cast<float>(a.y);
                    const float edge_x = static_cast<float>(b.x) - ax;
                    const float edge_y = static_cast<float>(b.y) - ay;
                    const auto side = [&](Vec2d point) {
                        const float point_x = static_cast<float>(point.x) - ax;
                        const float point_y = static_cast<float>(point.y) - ay;
                        return std::fma(edge_x, point_y, -(edge_y * point_x));
                    };
                    const auto glsl_sign = [](float value) {
                        return value > 0.0F ? 1 : value < 0.0F ? -1 : 0;
                    };
                    occluding_edge[static_cast<std::size_t>(edge)] =
                        glsl_sign(side(own)) == glsl_sign(side(*adjacent));
                } else {
                    const auto side = [&](Vec2d point) {
                        return (b.x - a.x) * (point.y - a.y) -
                               (b.y - a.y) * (point.x - a.x);
                    };
                    occluding_edge[static_cast<std::size_t>(edge)] =
                        std::signbit(side(own)) == std::signbit(side(*adjacent));
                }
                if (occluding_edge[static_cast<std::size_t>(edge)]) {
                    ++accepted_shared_edges;
                }
            }
            if (!occluding_edge[0] && !occluding_edge[1] && !occluding_edge[2]) continue;
            const double denominator =
                (triangle[1].y - triangle[2].y) * (triangle[0].x - triangle[2].x) +
                (triangle[2].x - triangle[1].x) * (triangle[0].y - triangle[2].y);
            if (std::abs(denominator) < 1.0e-20) continue;
            const std::array<double, 3> derivative_x = {
                (triangle[1].y - triangle[2].y) / denominator,
                (triangle[2].y - triangle[0].y) / denominator,
                (triangle[0].y - triangle[1].y) / denominator};
            const std::array<double, 3> derivative_y = {
                (triangle[2].x - triangle[1].x) / denominator,
                (triangle[0].x - triangle[2].x) / denominator,
                (triangle[1].x - triangle[0].x) / denominator};
            const auto shade_occlusion_fragment =
                [&](int x, int y, const std::array<double, 3> &barycentric,
                    std::optional<double> rasterized_line_depth = std::nullopt) {
                ++rasterized_fragments;
                occlusion_raster_candidate.at<std::uint8_t>(y, x) = 255;
                std::array<double, 3> fragment_barycentric = barycentric;
                std::array<double, 3> fragment_derivative_x = derivative_x;
                std::array<double, 3> fragment_derivative_y = derivative_y;
                if (gpu_float_fragment) {
                    // embedded_0230.glsl reconstructs barycentrics from the
                    // full-resolution projected vertices and
                    // gl_FragCoord*downscale, all in float32.
                    const float x0 = static_cast<float>(projected.pixel[0].x);
                    const float y0 = static_cast<float>(projected.pixel[0].y);
                    const float e10x = static_cast<float>(projected.pixel[1].x) - x0;
                    const float e10y = static_cast<float>(projected.pixel[1].y) - y0;
                    const float e20x = static_cast<float>(projected.pixel[2].x) - x0;
                    const float e20y = static_cast<float>(projected.pixel[2].y) - y0;
                    const float px = (static_cast<float>(x) + 0.5F) *
                                     static_cast<float>(weight_downscale) - x0;
                    const float py = (static_cast<float>(y) + 0.5F) *
                                     static_cast<float>(weight_downscale) - y0;
                    const float d = std::fma(e10x, e20y, -(e20x * e10y));
                    if (std::abs(d) < 1.0e-7F) return;
                    const float beta = std::fma(px, e20y, -(e20x * py)) / d;
                    const float gamma = std::fma(e10x, py, -(px * e10y)) / d;
                    const std::array<float, 3> fb = {
                        (1.0F - beta) - gamma, beta, gamma};
                    const float ds = static_cast<float>(weight_downscale);
                    const std::array<float, 3> fdx = {
                        ds * ((e10y - e20y) / d), ds * (e20y / d),
                        ds * (-e10y / d)};
                    const std::array<float, 3> fdy = {
                        ds * ((e20x - e10x) / d), ds * (-e20x / d),
                        ds * (e10x / d)};
                    for (int i = 0; i < 3; ++i) {
                        fragment_barycentric[static_cast<std::size_t>(i)] = fb[i];
                        fragment_derivative_x[static_cast<std::size_t>(i)] = fdx[i];
                        fragment_derivative_y[static_cast<std::size_t>(i)] = fdy[i];
                    }
                }
                // location 0 in the recovered vertex/fragment pair is a
                // default smooth float.  gl_Position.w is exactly 1, so for
                // polygon-mode lines Vulkan evaluates it by projecting the
                // fragment center onto the current segment and linearly
                // interpolating the segment endpoints.  The fragment shader's
                // separately reconstructed triangle barycentrics remain tied
                // to gl_FragCoord and are only used by the edge-width test.
                const double depth = rasterized_line_depth.value_or(
                    fragment_barycentric[0] * projected.depth[0] +
                    fragment_barycentric[1] * projected.depth[1] +
                    fragment_barycentric[2] * projected.depth[2]);
                // E-PIPELINE-250 shows that Metashape first fills a depth
                // attachment and then executes this line draw with depth test
                // and depth write enabled under LESS_OR_EQUAL.  This opt-in
                // diagnostic uses the recovered camera-space depth image as
                // the monotonic equivalent of that normalized attachment so
                // its explanatory power can be measured separately from the
                // fragment shader's five-sample 0.999 gate below.
                if (fixed_function_depth_test) {
                    const float fixed_stored = occlusion_depth.at<float>(y, x);
                    // Fixed-function fragment depth follows the primitive's
                    // triangle-plane gl_Position.z interpolation.  The user
                    // varying at location 0 follows the current polygon edge
                    // and is only the `depth` value consumed by the fragment
                    // shader.  E-PIPELINE-250 plus the paired fill/line shaders
                    // require keeping these two interpolants separate.
                    const float fixed_candidate = static_cast<float>(
                        fragment_barycentric[0] * projected.depth[0] +
                        fragment_barycentric[1] * projected.depth[1] +
                        fragment_barycentric[2] * projected.depth[2]);
                    if (fixed_stored == std::numeric_limits<float>::max() ||
                        fixed_candidate > fixed_stored) {
                        return;
                    }
                }
                occlusion_fixed_depth_pass.at<std::uint8_t>(y, x) = 255;
                bool edge_hit = false;
                for (int edge = 0; edge < 3; ++edge) {
                    if (!occluding_edge[static_cast<std::size_t>(edge)]) continue;
                    const int opposite = (edge + 2) % 3;
                    const double half_width = 0.5 *
                        (std::abs(fragment_derivative_x[static_cast<std::size_t>(opposite)]) +
                         std::abs(fragment_derivative_y[static_cast<std::size_t>(opposite)]));
                    if (fragment_barycentric[static_cast<std::size_t>(opposite)] <= half_width) {
                        edge_hit = true;
                        break;
                    }
                }
                bool visible = skip_occlusion_depth;
                float maximum_stored = -std::numeric_limits<float>::infinity();
                static constexpr std::array<std::array<int, 2>, 5> offsets =
                    {{{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
                for (const auto &offset : offsets) {
                    const int sx = std::clamp(x + offset[0], 0,
                                              occlusion_depth.cols - 1);
                    const int sy = std::clamp(y + offset[1], 0,
                                              occlusion_depth.rows - 1);
                    const float stored = occlusion_depth.at<float>(sy, sx);
                    if (stored == std::numeric_limits<float>::max()) continue;
                    maximum_stored = std::max(maximum_stored, stored);
                    if (stored >= 0.999F * depth) visible = true;
                }
                if (edge_hit && std::isfinite(maximum_stored) && depth > 0.0) {
                    float &ratio = occlusion_visibility_ratio.at<float>(y, x);
                    ratio = std::max(ratio,
                                     maximum_stored / static_cast<float>(depth));
                }
                if (!visible) return;
                ++depth_visible_fragments;
                occlusion_depth_pass.at<std::uint8_t>(y, x) = 255;
                if (!edge_hit) return;
                constraint.at<std::uint8_t>(y, x) = 255;
                occlusion_edge_pass.at<std::uint8_t>(y, x) = 255;
                };
            if (line_raster) {
                raster_triangle_lines(triangle, projected.depth,
                                      constraint.cols, constraint.rows,
                                      shade_occlusion_fragment);
            } else if (conservative_raster) {
                raster_triangle_conservative(triangle, constraint.cols, constraint.rows,
                                             conservative_extent,
                                             shade_occlusion_fragment);
            } else {
                raster_triangle(triangle, constraint.cols, constraint.rows,
                                shade_occlusion_fragment);
            }
        }
        if (report_occlusion_stats) {
            std::cerr << "occlusion camera " << camera_index
                      << ": boundary_edges=" << boundary_edges
                      << ", shared_edges=" << shared_edges
                      << ", accepted_shared_edges=" << accepted_shared_edges
                      << ", rasterized_fragments=" << rasterized_fragments
                      << ", depth_visible_fragments=" << depth_visible_fragments
                      << ", constrained_pixels=" << cv::countNonZero(constraint) << '\n';
        }
        }
        if (occlusion_mode == nullptr || std::string(occlusion_mode) == "depth") {
            double relative_depth_jump = 0.005;
            if (const char *value = std::getenv("METASHAPE_TEXTURE_OCCLUSION_DEPTH_JUMP")) {
                relative_depth_jump = std::stod(value);
            }
            for (int y = 0; y < visible_support.rows; ++y) {
                for (int x = 0; x < visible_support.cols; ++x) {
                    const float center_depth = occlusion_depth.at<float>(y, x);
                    if (center_depth == std::numeric_limits<float>::max()) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            if (dx != 0 && dy != 0) continue;
                            const int sx = x + dx;
                            const int sy = y + dy;
                            if (sx < 0 || sy < 0 || sx >= occlusion_depth.cols ||
                                sy >= occlusion_depth.rows) {
                                constraint.at<std::uint8_t>(y, x) = 255;
                                continue;
                            }
                            const float neighbor_depth = occlusion_depth.at<float>(sy, sx);
                            if (neighbor_depth == std::numeric_limits<float>::max() ||
                                std::abs(center_depth - neighbor_depth) >
                                    relative_depth_jump *
                                        std::min(center_depth, neighbor_depth)) {
                                constraint.at<std::uint8_t>(y, x) = 255;
                            }
                        }
                    }
                }
            }
        }
        int mask_offset_x = 0;
        int mask_offset_y = 0;
        if (const char *value = std::getenv("METASHAPE_TEXTURE_OCCLUSION_OFFSET_X")) {
            mask_offset_x = std::stoi(value);
        }
        if (const char *value = std::getenv("METASHAPE_TEXTURE_OCCLUSION_OFFSET_Y")) {
            mask_offset_y = std::stoi(value);
        }
        // Native host path:
        //   0x1422d31b0 packed uint32 -> byte image (foreground 0xff)
        //   0x141e89f10 remove 8-connected foreground components with size < 20
        //   0x1422d2b10 byte image -> packed uint32
        // The constants 20, background=0 and diagonal-connectivity=true are
        // direct call-site arguments at 0x141f2e80c..0x141f2e826.
        Image<std::uint8_t> recovered_constraint(constraint.cols, constraint.rows);
        for (int y = 0; y < constraint.rows; ++y) {
            for (int x = 0; x < constraint.cols; ++x) {
                recovered_constraint(x, y) = constraint.at<std::uint8_t>(y, x);
            }
        }
        recovered_constraint = recovered::remove_small_components_8(
            recovered_constraint, 20, 0);
        for (int y = 0; y < constraint.rows; ++y) {
            for (int x = 0; x < constraint.cols; ++x) {
                constraint.at<std::uint8_t>(y, x) = recovered_constraint(x, y);
            }
        }
        for (int y = 0; y < constraint.rows; ++y) {
            for (int x = 0; x < constraint.cols; ++x) {
                if (constraint.at<std::uint8_t>(y, x) == 0) continue;
                const int target_x = x + mask_offset_x;
                const int target_y = y + mask_offset_y;
                if (target_x >= 0 && target_y >= 0 && target_x < coverage.cols &&
                    target_y < coverage.rows) {
                    coverage.at<std::uint8_t>(target_y, target_x) = 0;
                }
            }
        }
        // uncompress_occlusions.comp explicitly writes zero on border and
        // constraint texels before distance-weight generation.
        for (int y = 0; y < coverage.rows; ++y) {
            for (int x = 0; x < coverage.cols; ++x) {
                if (coverage.at<std::uint8_t>(y, x) == 0) {
                    winner_value.at<float>(y, x) = 0.0F;
                }
            }
        }
    }
    // module133 consumes the face-ID image, winner-label buffer and the
    // filtered occlusion image.  Re-evaluate its recovered expression here
    // so overlapping faces follow the nearest face ID rather than whichever
    // triangle happened to set winner_value earlier.
    Image<std::int32_t> recovered_face_ids(coverage.cols, coverage.rows);
    Image<std::uint8_t> recovered_occlusion(coverage.cols, coverage.rows);
    std::vector<std::uint32_t> recovered_labels(winners.size());
    for (std::size_t index = 0; index < winners.size(); ++index) {
        recovered_labels[index] = winners[index] < 0
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(winners[index]);
    }
    for (int y = 0; y < coverage.rows; ++y) {
        for (int x = 0; x < coverage.cols; ++x) {
            const std::int32_t face_id = visible_face_id.at<std::int32_t>(y, x);
            recovered_face_ids(x, y) = face_id < 0
                ? std::numeric_limits<std::int32_t>::max() : face_id;
            recovered_occlusion(x, y) =
                coverage.at<std::uint8_t>(y, x) == 0 ? 255 : 0;
        }
    }
    const auto recovered_weight_init = recovered::initialize_camera_weights(
        recovered_face_ids, recovered_labels, recovered_occlusion,
        static_cast<std::uint32_t>(camera_index));
    for (int y = 0; y < coverage.rows; ++y) {
        for (int x = 0; x < coverage.cols; ++x) {
            winner_value.at<float>(y, x) = recovered_weight_init(x, y);
        }
    }
    if (const char *native_input_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_WEIGHT_INPUT_DIR")) {
        std::ostringstream name;
        name << "camera_" << std::setw(3) << std::setfill('0') << camera_index
             << "_input_0_320x240x1_u4.raw";
        const auto path = std::filesystem::path(native_input_root) / name.str();
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot load native winner-value input: " + path.string());
        if (winner_value.cols != 320 || winner_value.rows != 240 || winner_value.type() != CV_32F) {
            throw std::runtime_error("native winner-value fixture dimensions do not match");
        }
        stream.read(reinterpret_cast<char *>(winner_value.data),
                    static_cast<std::streamsize>(winner_value.total() * sizeof(float)));
        if (!stream || stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native winner-value byte count: " + path.string());
        }
        std::ostringstream mask_name;
        mask_name << "camera_" << std::setw(3) << std::setfill('0') << camera_index
                  << "_input_1_320x240x1_u4.raw";
        const auto mask_path = std::filesystem::path(native_input_root) / mask_name.str();
        std::ifstream mask_stream(mask_path, std::ios::binary);
        if (!mask_stream) {
            throw std::runtime_error("cannot load native constraint-mask input: " +
                                     mask_path.string());
        }
        cv::Mat constraint(coverage.size(), CV_32S);
        mask_stream.read(reinterpret_cast<char *>(constraint.data),
                         static_cast<std::streamsize>(constraint.total() * sizeof(std::int32_t)));
        if (!mask_stream || mask_stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native constraint-mask byte count: " +
                                     mask_path.string());
        }
        for (int y = 0; y < coverage.rows; ++y) {
            for (int x = 0; x < coverage.cols; ++x) {
                coverage.at<std::uint8_t>(y, x) =
                    constraint.at<std::int32_t>(y, x) == 255 ? 0 : 255;
            }
        }
    }

    auto load_native_stage_input = [&](int stage, cv::Mat &value, cv::Mat &validity) {
        const char *root = std::getenv("METASHAPE_TEXTURE_NATIVE_WEIGHT_STAGE_INPUT_DIR");
        if (root == nullptr) return;
        std::ostringstream prefix;
        prefix << "camera_" << std::setw(3) << std::setfill('0') << camera_index
               << "_stage_" << stage << "_input_";
        auto input_path = [&](int input) {
            std::ostringstream name;
            name << prefix.str() << input << '_' << value.cols << 'x' << value.rows
                 << "x1_u4.raw";
            return std::filesystem::path(root) / name.str();
        };
        const auto value_path = input_path(0);
        std::ifstream value_stream(value_path, std::ios::binary);
        if (!value_stream) {
            throw std::runtime_error("cannot load native stage value: " + value_path.string());
        }
        value_stream.read(reinterpret_cast<char *>(value.data),
                          static_cast<std::streamsize>(value.total() * sizeof(float)));
        if (!value_stream || value_stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native stage-value byte count: " +
                                     value_path.string());
        }
        const auto mask_path = input_path(1);
        std::ifstream mask_stream(mask_path, std::ios::binary);
        if (!mask_stream) {
            throw std::runtime_error("cannot load native stage mask: " + mask_path.string());
        }
        cv::Mat constraint(validity.size(), CV_32S);
        mask_stream.read(reinterpret_cast<char *>(constraint.data),
                         static_cast<std::streamsize>(constraint.total() * sizeof(std::int32_t)));
        if (!mask_stream || mask_stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native stage-mask byte count: " + mask_path.string());
        }
        for (int y = 0; y < validity.rows; ++y) {
            for (int x = 0; x < validity.cols; ++x) {
                validity.at<std::uint8_t>(y, x) =
                    constraint.at<std::int32_t>(y, x) == 255 ? 0 : 255;
            }
        }
    };
    load_native_stage_input(0, winner_value, coverage);

    // Native push constants: max_distance=5 at the full, half and quarter
    // resolution passes.  The coarsest pass expands it as 5,11,23,...,191.
    float maximum_distance = 5.0F;
    if (const char *override_value = std::getenv("METASHAPE_TEXTURE_MAX_DISTANCE")) {
        maximum_distance = std::clamp(std::stof(override_value), 1.0F, 4096.0F);
    }
    pyramid.weight[0] = recovered_distance_weights(coverage, winner_value, maximum_distance);
    if (const char *native_root = std::getenv("METASHAPE_TEXTURE_NATIVE_BLEND_FIXTURE_DIR")) {
        const auto path = fixture_path(native_root, "_weight.png");
        cv::Mat encoded = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (encoded.empty() || encoded.type() != CV_16U) {
            throw std::runtime_error("cannot load native blend weight: " + path.string());
        }
        encoded.convertTo(pyramid.weight[0], CV_32F, 1.0 / 65535.0);
    }

    bool use_low_frequency_quality = false;
    int quality_start_level = 1;
    if (const char *use_quality = std::getenv("METASHAPE_TEXTURE_USE_LOW_FREQUENCY_QUALITY")) {
        if (std::string(use_quality) == "1") {
            use_low_frequency_quality = true;
            if (const char *start_value = std::getenv("METASHAPE_TEXTURE_QUALITY_START_LEVEL")) {
                quality_start_level = std::clamp(std::stoi(start_value), 1, 4);
            }
            int radius = 0;
            if (const char *radius_value = std::getenv("METASHAPE_TEXTURE_QUALITY_MIN_RADIUS")) {
                radius = std::clamp(std::stoi(radius_value), 0, 64);
            }
            if (radius > 0) low_frequency_quality = square_minimum_filter(low_frequency_quality, radius);
        }
    }
    for (int y = 0; y < low_frequency_quality.rows; ++y) {
        for (int x = 0; x < low_frequency_quality.cols; ++x) {
            if (coverage.at<std::uint8_t>(y, x) == 0) low_frequency_quality.at<float>(y, x) = 0.0F;
        }
    }
    // The recovered fragment push block is
    //   image_downscale=2, weight_downscale=2, weight_scale_cap=8.
    // Native therefore prepares nine distance-weight stages but binds only
    // even stages to the five enblend levels:
    //   320@5, 80@5, 40@11, 40@47, 40@191.
    // Stages 0..3 successively halve both the winner/value image and the
    // validity mask.  weights_from_distances.comp averages each 2x2 value block and
    // marks the parent constrained when any child is constrained.  Once the
    // 40x30 cap is reached, the resolution remains fixed while BFS distance
    // limits expand as 11,23,47,95,191.
    cv::Mat stage_coverage = coverage;
    cv::Mat stage_value = winner_value;
    for (int stage = 1; stage <= 8; ++stage) {
        if (stage <= 3) {
            auto downscaled = downscale_low_frequency_level(stage_value, stage_coverage);
            stage_value = std::move(downscaled.first);
            stage_coverage = std::move(downscaled.second);
        }
        load_native_stage_input(stage, stage_value, stage_coverage);
        const float stage_maximum_distance = stage <= 3
            ? maximum_distance
            : static_cast<float>(6 * (1 << (stage - 3)) - 1);
        cv::Mat stage_weight = recovered_distance_weights(
            stage_coverage, stage_value, stage_maximum_distance);
        if ((stage & 1) == 0) pyramid.weight[stage / 2] = std::move(stage_weight);
    }
    if (const char *native_weight_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_WEIGHT_PYRAMID_DIR")) {
        for (int level = 0; level < 5; ++level) {
            const int stage = 2 * level;
            const int width = level == 0 ? 320 : level == 1 ? 80 : 40;
            const int height = level == 0 ? 240 : level == 1 ? 60 : 30;
            std::ostringstream name;
            name << "camera_" << std::setw(3) << std::setfill('0') << camera_index
                 << "_stage_" << stage << '_' << width << 'x' << height
                 << "x1_u4.raw";
            const auto path = std::filesystem::path(native_weight_root) / name.str();
            std::ifstream stream(path, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot load native weight stage: " + path.string());
            cv::Mat weight(height, width, CV_32F);
            stream.read(reinterpret_cast<char *>(weight.data),
                        static_cast<std::streamsize>(weight.total() * sizeof(float)));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("invalid native weight-stage byte count: " + path.string());
            }
            pyramid.weight[level] = std::move(weight);
        }
    }

    for (int level = 1; level < 5; ++level) {
        pyramid.image[level] = downscale_6tap(pyramid.image[level - 1], 4);
        low_frequency_quality = downscale_6tap_scalar(low_frequency_quality, 4);
        if (use_low_frequency_quality && level >= quality_start_level) {
            cv::Mat quality_for_weight;
            if (low_frequency_quality.size() == pyramid.weight[level].size()) {
                quality_for_weight = low_frequency_quality;
            } else {
                cv::resize(low_frequency_quality, quality_for_weight,
                           pyramid.weight[level].size(), 0.0, 0.0, cv::INTER_AREA);
            }
            for (int y = 0; y < pyramid.weight[level].rows; ++y) {
                for (int x = 0; x < pyramid.weight[level].cols; ++x) {
                    pyramid.weight[level].at<float>(y, x) *= quality_for_weight.at<float>(y, x);
                }
            }
        }
    }
    if (const char *native_image_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_IMAGE_PYRAMID_DIR")) {
        static constexpr std::array<int, 5> stages = {0, 2, 4, 6, 8};
        static constexpr std::array<int, 5> widths = {320, 80, 20, 5, 2};
        static constexpr std::array<int, 5> heights = {240, 60, 15, 4, 1};
        // The private GPU readback was recorded in worker-completion order.  A
        // content match against the source photos proves that captures 8 and 9
        // completed in the opposite order; all other captures are identity.
        // Keep this correction confined to the optional validation fixture.
        static constexpr std::array<int, 16> captured_camera = {
            0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15};
        const int captured_index =
            camera_index >= 0 && camera_index < static_cast<int>(captured_camera.size())
                ? captured_camera[camera_index]
                : camera_index;
        for (int level = 0; level < 5; ++level) {
            std::ostringstream name;
            name << "camera_" << std::setw(3) << std::setfill('0') << captured_index
                 << "_stage_" << stages[level] << '_' << widths[level] << 'x'
                 << heights[level] << "x3_u1.raw";
            const auto path = std::filesystem::path(native_image_root) / name.str();
            std::ifstream stream(path, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot load native image stage: " + path.string());
            cv::Mat rgb8(heights[level], widths[level], CV_8UC3);
            stream.read(reinterpret_cast<char *>(rgb8.data),
                        static_cast<std::streamsize>(rgb8.total() * rgb8.elemSize()));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("invalid native image-stage byte count: " + path.string());
            }
            cv::Mat bgr8;
            cv::cvtColor(rgb8, bgr8, cv::COLOR_RGB2BGR);
            bgr8.convertTo(pyramid.image[level], CV_32FC3, 1.0 / 255.0);
        }
    }
    if (const char *diagnostic_root = std::getenv("METASHAPE_TEXTURE_BLEND_DIAGNOSTIC_DIR")) {
        const std::filesystem::path root(diagnostic_root);
        std::filesystem::create_directories(root);
        {
            const auto coverage_path = root / ("camera_" + std::to_string(camera_index) +
                                                "_coverage_u8.raw");
            std::ofstream stream(coverage_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(coverage.data),
                         static_cast<std::streamsize>(coverage.total()));
        }
        {
            const auto winner_path = root / ("camera_" + std::to_string(camera_index) +
                                              "_winner_value_f32.raw");
            std::ofstream stream(winner_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(winner_value.data),
                         static_cast<std::streamsize>(winner_value.total() * sizeof(float)));
        }
        {
            const auto support_path = root / ("camera_" + std::to_string(camera_index) +
                                               "_visible_support_u8.raw");
            std::ofstream stream(support_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(visible_support.data),
                         static_cast<std::streamsize>(visible_support.total()));
        }
        {
            const auto depth_path = root / ("camera_" + std::to_string(camera_index) +
                                             "_visible_depth_f32.raw");
            std::ofstream stream(depth_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(visible_depth.data),
                         static_cast<std::streamsize>(visible_depth.total() * sizeof(float)));
        }
        {
            const auto face_path = root / ("camera_" + std::to_string(camera_index) +
                                            "_visible_face_id_i32.raw");
            std::ofstream stream(face_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(visible_face_id.data),
                         static_cast<std::streamsize>(visible_face_id.total() *
                                                      sizeof(std::int32_t)));
        }
        {
            const auto depth_path = root / ("camera_" + std::to_string(camera_index) +
                                             "_occlusion_depth_f32.raw");
            std::ofstream stream(depth_path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(occlusion_depth.data),
                         static_cast<std::streamsize>(occlusion_depth.total() * sizeof(float)));
        }
        const auto write_u8 = [&](const std::string &suffix, const cv::Mat &image) {
            const auto path = root / ("camera_" + std::to_string(camera_index) + suffix);
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(image.data),
                         static_cast<std::streamsize>(image.total()));
        };
        write_u8("_occlusion_raster_candidate_u8.raw", occlusion_raster_candidate);
        write_u8("_occlusion_depth_pass_u8.raw", occlusion_depth_pass);
        write_u8("_occlusion_fixed_depth_pass_u8.raw", occlusion_fixed_depth_pass);
        write_u8("_occlusion_edge_pass_u8.raw", occlusion_edge_pass);
        {
            const auto path = root / ("camera_" + std::to_string(camera_index) +
                                      "_occlusion_visibility_ratio_f32.raw");
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(occlusion_visibility_ratio.data),
                         static_cast<std::streamsize>(
                             occlusion_visibility_ratio.total() * sizeof(float)));
        }
        cv::Mat encoded_weight;
        pyramid.weight[0].convertTo(encoded_weight, CV_16U, 65535.0);
        cv::imwrite((root / ("camera_" + std::to_string(camera_index) +
                             "_weight.png")).string(), encoded_weight);
        cv::Mat encoded_image;
        pyramid.image[0].convertTo(encoded_image, CV_8UC3, 255.0);
        cv::imwrite((root / ("camera_" + std::to_string(camera_index) +
                             "_image.png")).string(), encoded_image);
        for (int level = 0; level < 5; ++level) {
            const auto path = root / ("camera_" + std::to_string(camera_index) +
                                      "_weight_level_" + std::to_string(level) +
                                      "_f32.raw");
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char *>(pyramid.weight[level].data),
                         static_cast<std::streamsize>(pyramid.weight[level].total() *
                                                      sizeof(float)));
        }
    }
    return pyramid;
}

cv::Mat render_multiband(const Mesh &mesh, const std::vector<Camera> &cameras,
                         const std::vector<int> &winners, int texture_size,
                         int downscale, int depth_downscale,
                         const std::filesystem::path &atlas_attachments = {},
                         cv::Mat *page_support = nullptr) {
    // A captured native atlas is the exact input boundary for page-rebuild
    // diagnostics.  Load it before constructing CPU camera pyramids so page
    // experiments do not repeat unrelated projection and blending work.
    const char *native_atlas_root =
        std::getenv("METASHAPE_TEXTURE_NATIVE_ATLAS_BANDS_DIR");
    if (!atlas_attachments.empty() || native_atlas_root != nullptr) {
        cv::Mat texture(texture_size, texture_size, CV_32FC3, cv::Scalar(0, 0, 0));
        if (page_support != nullptr) {
            *page_support = cv::Mat(texture_size, texture_size, CV_8U, cv::Scalar(0));
        }
        const std::size_t band_bytes = static_cast<std::size_t>(texture_size) *
            texture_size * sizeof(cv::Vec4f);
        std::ifstream combined;
        if (!atlas_attachments.empty()) {
            combined.open(atlas_attachments, std::ios::binary | std::ios::ate);
            if (!combined || combined.tellg() !=
                    static_cast<std::streamoff>(band_bytes * 5U)) {
                throw std::runtime_error(
                    "atlas attachment bundle has the wrong byte count: " +
                    atlas_attachments.string());
            }
        }
        // The save-stage image combiner visits the Laplacian bands from the
        // coarsest attachment back to the finest.  Float addition order is
        // visible at a handful of half-U8 boundaries.
        for (int band = 4; band >= 0; --band) {
            cv::Mat rgba(texture_size, texture_size, CV_32FC4);
            if (!atlas_attachments.empty()) {
                combined.seekg(static_cast<std::streamoff>(band_bytes * band));
                combined.read(reinterpret_cast<char *>(rgba.data),
                              static_cast<std::streamsize>(band_bytes));
                if (!combined) {
                    throw std::runtime_error("cannot read atlas attachment band " +
                                             std::to_string(band));
                }
            } else {
                const auto path = std::filesystem::path(native_atlas_root) /
                    ("band_" + std::to_string(band) + "_" +
                     std::to_string(texture_size) + "x" +
                     std::to_string(texture_size) + "x4_u4.raw");
                std::ifstream stream(path, std::ios::binary);
                if (!stream) {
                    throw std::runtime_error("cannot load native atlas band: " +
                                             path.string());
                }
                stream.read(reinterpret_cast<char *>(rgba.data),
                            static_cast<std::streamsize>(band_bytes));
                if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                    throw std::runtime_error(
                        "invalid native atlas-band byte count: " + path.string());
                }
            }
            for (int y = 0; y < texture_size; ++y) {
                for (int x = 0; x < texture_size; ++x) {
                    const cv::Vec4f value = rgba.at<cv::Vec4f>(y, x);
                    if (value[3] <= 0.0F) continue;
                    // The native worker uses one divss per channel.  OpenCV's
                    // Vec/scalar operator may instead form one reciprocal and
                    // multiply, which crosses four half-U8 boundaries here.
                    cv::Vec3f &destination = texture.at<cv::Vec3f>(y, x);
                    destination[0] += value[2] / value[3];
                    destination[1] += value[1] / value[3];
                    destination[2] += value[0] / value[3];
                    if (page_support != nullptr && band == 4) {
                        page_support->at<std::uint8_t>(y, x) = 255;
                    }
                }
            }
        }
        return texture;
    }
    // Diagnostic boundary for replaying the recovered 29376-byte atlas vertex
    // module without passing through OBJ's decimal UV serialization.  The
    // runtime pipeline capture establishes this exact 24-byte input ABI:
    // float3 world position, float2 normalized atlas UV, uint32 page/layer.
    // The xatlas UV-mesh output has one entry per packed UV vertex. Rebuild
    // the corresponding geometry xref from the face corners so the resulting
    // vertex/index stream preserves the native sharing and ordering.
    if (const char *dump_root =
            std::getenv("METASHAPE_TEXTURE_ATLAS_VERTEX_DUMP_DIR")) {
        struct AtlasVertex {
            float position[3];
            float uv[2];
            std::uint32_t page;
        };
        static_assert(sizeof(AtlasVertex) == 24);
        const std::filesystem::path root(dump_root);
        std::filesystem::create_directories(root);
        std::vector<AtlasVertex> vertices(mesh.texcoords.size());
        std::vector<std::uint32_t> indices;
        std::vector<std::uint32_t> geometry_by_texcoord(
            mesh.texcoords.size(), std::numeric_limits<std::uint32_t>::max());
        indices.reserve(mesh.faces.size() * 3U);
        for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
            const Face &face = mesh.faces[face_index];
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const std::uint32_t texcoord = face.texcoord[corner];
                const std::uint32_t geometry = face.vertex[corner];
                auto &mapped_geometry = geometry_by_texcoord[texcoord];
                if (mapped_geometry != std::numeric_limits<std::uint32_t>::max() &&
                    mapped_geometry != geometry) {
                    throw std::runtime_error(
                        "packed UV vertex maps to multiple geometry vertices");
                }
                mapped_geometry = geometry;
                indices.push_back(texcoord);
            }
        }
        for (std::size_t texcoord = 0; texcoord < mesh.texcoords.size(); ++texcoord) {
            const std::uint32_t geometry = geometry_by_texcoord[texcoord];
            if (geometry == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("packed UV vertex is not referenced by a face");
            }
            const Vec3d &position = mesh.vertices[geometry];
            const Vec2d &uv = mesh.texcoords[texcoord];
            vertices[texcoord] = {
                {static_cast<float>(position.x), static_cast<float>(position.y),
                 static_cast<float>(position.z)},
                {static_cast<float>(uv.x), static_cast<float>(uv.y)},
                0U,
            };
        }
        {
            std::ofstream stream(root / "atlas_vertices_stride24.raw", std::ios::binary);
            if (!stream) throw std::runtime_error("cannot create atlas vertex dump");
            stream.write(reinterpret_cast<const char *>(vertices.data()),
                         static_cast<std::streamsize>(vertices.size() * sizeof(AtlasVertex)));
        }
        {
            std::ofstream stream(root / "atlas_indices_u32.raw", std::ios::binary);
            if (!stream) throw std::runtime_error("cannot create atlas index dump");
            stream.write(reinterpret_cast<const char *>(indices.data()),
                         static_cast<std::streamsize>(indices.size() * sizeof(std::uint32_t)));
        }
        {
            std::ofstream stream(root / "atlas_draw_metadata.txt");
            if (!stream) throw std::runtime_error("cannot create atlas draw metadata");
            stream << "vertex_stride=24\n"
                   << "vertex_count=" << vertices.size() << '\n'
                   << "index_count=" << indices.size() << '\n'
                   << "face_count=" << mesh.faces.size() << '\n'
                   << "texture_size=" << texture_size << '\n'
                   << "page=0\n";
        }
        if (const char *dump_only =
                std::getenv("METASHAPE_TEXTURE_ATLAS_VERTEX_DUMP_ONLY")) {
            if (std::string(dump_only) == "1") {
                return cv::Mat(texture_size, texture_size, CV_32FC3,
                               cv::Scalar(0, 0, 0));
            }
        }
    }
    int min_x = texture_size, min_y = texture_size, max_x = -1, max_y = -1;
    for (const Face &face : mesh.faces) {
        for (std::uint32_t index : face.texcoord) {
            const Vec2d uv = mesh.texcoords[index];
            const double x = uv.x * texture_size;
            const double y = (1.0 - uv.y) * texture_size;
            min_x = std::min(min_x, static_cast<int>(std::floor(x)) - 1);
            min_y = std::min(min_y, static_cast<int>(std::floor(y)) - 1);
            max_x = std::max(max_x, static_cast<int>(std::ceil(x)) + 1);
            max_y = std::max(max_y, static_cast<int>(std::ceil(y)) + 1);
        }
    }
    min_x = std::clamp(min_x, 0, texture_size - 1);
    min_y = std::clamp(min_y, 0, texture_size - 1);
    max_x = std::clamp(max_x, 0, texture_size - 1);
    max_y = std::clamp(max_y, 0, texture_size - 1);
    const cv::Rect region(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    std::array<cv::Mat, 5> numerator;
    std::array<cv::Mat, 5> denominator;
    for (int level = 0; level < 5; ++level) {
        numerator[level] = cv::Mat(region.size(), CV_32FC3, cv::Scalar(0, 0, 0));
        denominator[level] = cv::Mat(region.size(), CV_32F, cv::Scalar(0));
    }
    const char *image_resolution_weights =
        std::getenv("METASHAPE_TEXTURE_BLEND_WEIGHT_AT_IMAGE_RES");
    const bool weight_at_image_resolution = image_resolution_weights == nullptr ||
        std::string(image_resolution_weights) != "0";
    const double weight_coordinate_scale = weight_at_image_resolution
        ? 1.0 : static_cast<double>(downscale) / depth_downscale;
    const bool skip_render_depth =
        std::getenv("METASHAPE_TEXTURE_SKIP_RENDER_DEPTH") != nullptr;
    const bool atlas_gpu_float =
        std::getenv("METASHAPE_TEXTURE_ATLAS_GPU_FLOAT") != nullptr;
    const bool reproject_fragments =
        std::getenv("METASHAPE_TEXTURE_REPROJECT_FRAGMENTS") != nullptr;
    const bool enblend_gpu_projection =
        std::getenv("METASHAPE_TEXTURE_ENBLEND_GPU_PROJECTION") != nullptr;
    const bool enblend_all_projected_faces =
        std::getenv("METASHAPE_TEXTURE_ENBLEND_ALL_PROJECTED_FACES") != nullptr;
    int atlas_subpixel_bits = -1;
    if (const char *value =
            std::getenv("METASHAPE_TEXTURE_ATLAS_RASTER_SUBPIXEL_BITS")) {
        atlas_subpixel_bits = std::stoi(value);
    }
    int atlas_top_left_mode = 0;
    if (const char *value =
            std::getenv("METASHAPE_TEXTURE_ATLAS_TOP_LEFT_RULE")) {
        atlas_top_left_mode = std::stoi(value);
    }
    int projection_dump_camera = -1;
    const char *projection_dump_root =
        std::getenv("METASHAPE_TEXTURE_CPU_ATLAS_PROJECTION_DUMP_DIR");
    if (projection_dump_root != nullptr) {
        projection_dump_camera = 0;
        if (const char *value =
                std::getenv("METASHAPE_TEXTURE_CPU_ATLAS_PROJECTION_CAMERA")) {
            projection_dump_camera = std::stoi(value);
        }
    }

    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
        if (projection_dump_root != nullptr &&
            static_cast<int>(camera_index) != projection_dump_camera) {
            continue;
        }
        const Camera &camera = cameras[camera_index];
        cv::Mat projected_rgba;
        if (projection_dump_root != nullptr) {
            projected_rgba = cv::Mat(texture_size, texture_size, CV_32FC4,
                                     cv::Scalar(std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN(), 0.0F));
        }
        auto pyramid = build_camera_blend_pyramid(mesh, camera, winners,
                                                   static_cast<int>(camera_index),
                                                   downscale, depth_downscale);
        if (!pyramid.has_seed) continue;
        const char *camera_support_dump_root =
            std::getenv("METASHAPE_TEXTURE_ATLAS_CAMERA_SUPPORT_DUMP_DIR");
        std::array<cv::Mat, 5> camera_support;
        if (camera_support_dump_root != nullptr) {
            for (auto &support : camera_support) {
                support = cv::Mat(region.size(), CV_8U, cv::Scalar(0));
            }
        }
        std::vector<std::uint32_t> native_face_discard;
        if (const char *native_bitmask_root =
                std::getenv("METASHAPE_TEXTURE_NATIVE_FACE_BITMASK_DIR")) {
            const auto path = std::filesystem::path(native_bitmask_root) /
                              std::to_string(camera_index) /
                              "face_discard_bits_u32.raw";
            const std::size_t word_count = (mesh.faces.size() + 31U) / 32U;
            native_face_discard.resize(word_count);
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("cannot load native face-bitmask fixture: " +
                                         path.string());
            }
            stream.read(reinterpret_cast<char *>(native_face_discard.data()),
                        static_cast<std::streamsize>(word_count * sizeof(std::uint32_t)));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("invalid native face-bitmask byte count: " +
                                         path.string());
            }
        }
        if (const char *bitmask_root =
                std::getenv("METASHAPE_TEXTURE_CPU_FACE_BITMASK_DUMP_DIR")) {
            std::vector<std::uint32_t> words((mesh.faces.size() + 31U) / 32U, 0U);
            std::vector<std::uint32_t> winding_positive(words.size(), 0U);
            std::vector<std::uint32_t> winding_negative(words.size(), 0U);
            std::vector<std::uint32_t> winding_positive_f32(words.size(), 0U);
            for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
                const ProjectedFace projected =
                    project_face_gpu_float(mesh, mesh.faces[face_index], camera);
                if (!projected.valid && projected.area > 0.0) {
                    words[face_index / 32U] |= 1U << (face_index % 32U);
                }
                if (projected.area > 0.0) {
                    const double signed_area =
                        (projected.pixel[1].x - projected.pixel[0].x) *
                            (projected.pixel[2].y - projected.pixel[0].y) -
                        (projected.pixel[1].y - projected.pixel[0].y) *
                            (projected.pixel[2].x - projected.pixel[0].x);
                    if (signed_area > 0.0) {
                        winding_positive[face_index / 32U] |=
                            1U << (face_index % 32U);
                    } else if (signed_area < 0.0) {
                        winding_negative[face_index / 32U] |=
                            1U << (face_index % 32U);
                    }
                    const float x0 = static_cast<float>(projected.pixel[0].x);
                    const float y0 = static_cast<float>(projected.pixel[0].y);
                    const float x1 = static_cast<float>(projected.pixel[1].x);
                    const float y1 = static_cast<float>(projected.pixel[1].y);
                    const float x2 = static_cast<float>(projected.pixel[2].x);
                    const float y2 = static_cast<float>(projected.pixel[2].y);
                    const float signed_area_f32 =
                        (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                    if (signed_area_f32 > 0.0F) {
                        winding_positive_f32[face_index / 32U] |=
                            1U << (face_index % 32U);
                    }
                }
            }
            const std::filesystem::path root(bitmask_root);
            std::filesystem::create_directories(root);
            const auto write_words = [&](const std::string &suffix,
                                         const std::vector<std::uint32_t> &values) {
                std::ofstream stream(root / ("camera_" + std::to_string(camera_index) +
                                             suffix), std::ios::binary);
                if (!stream) throw std::runtime_error("cannot create CPU face-bitmask dump");
                stream.write(reinterpret_cast<const char *>(values.data()),
                             static_cast<std::streamsize>(values.size() * sizeof(values[0])));
            };
            write_words("_discard_bits_u32.raw", words);
            write_words("_winding_positive_bits_u32.raw", winding_positive);
            write_words("_winding_negative_bits_u32.raw", winding_negative);
            write_words("_winding_positive_f32_bits_u32.raw", winding_positive_f32);
        }
        for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
            const Face &face = mesh.faces[face_index];
            // Diagnostic switch for the exact R32/FMA vertex arithmetic used
            // by the recovered 29376-byte enblend vertex module.  Keep it
            // independent of fragment reprojection: the native runtime uses
            // a plain VS+FS triangle-list pipeline and therefore interpolates
            // these three vertex results affinely.
            const auto face_projection = enblend_gpu_projection
                ? project_face_gpu_float(mesh, face, camera)
                : project_face(mesh, face, camera);
            if (!native_face_discard.empty()) {
                const bool discarded =
                    (native_face_discard[face_index / 32U] &
                     (1U << (face_index % 32U))) != 0U;
                if (discarded || face_projection.area <= 0.0) continue;
            } else if (!face_projection.valid &&
                       !(enblend_all_projected_faces && face_projection.area > 0.0)) {
                continue;
            }
            std::array<Vec2d, 3> atlas{};
            std::array<Vec2d, 3> projected{};
            for (int corner = 0; corner < 3; ++corner) {
                const Vec2d uv = mesh.texcoords[face.texcoord[corner]];
                atlas[corner] = {uv.x * texture_size - region.x,
                                 (1.0 - uv.y) * texture_size - region.y};
                if (atlas_gpu_float) {
                    atlas[corner].x = static_cast<float>(atlas[corner].x);
                    atlas[corner].y = static_cast<float>(atlas[corner].y);
                }
                if (atlas_subpixel_bits >= 0) {
                    const double subpixel_scale =
                        std::ldexp(1.0, atlas_subpixel_bits);
                    atlas[corner].x =
                        std::round(atlas[corner].x * subpixel_scale) /
                        subpixel_scale;
                    atlas[corner].y =
                        std::round(atlas[corner].y * subpixel_scale) /
                        subpixel_scale;
                }
                projected[corner] = {face_projection.pixel[corner].x / downscale,
                                     face_projection.pixel[corner].y / downscale};
            }
            auto shade_fragment =
                [&](int x, int y, const std::array<double, 3> &barycentric) {
                double px;
                double py;
                double depth;
                if (reproject_fragments) {
                    // Recovered native path: enblend_camera_into_atlas.tese
                    // interpolates the three world-space vertices at each
                    // tessellation coordinate and runs the camera projection
                    // again.  Evaluating the same projection at the atlas
                    // fragment barycentric coordinate is the continuous
                    // counterpart of that tessellated mapping and isolates it
                    // from the old affine interpolation experiment below.
                    const Vec3d &world0 = mesh.vertices[face.vertex[0]];
                    const Vec3d &world1 = mesh.vertices[face.vertex[1]];
                    const Vec3d &world2 = mesh.vertices[face.vertex[2]];
                    const Vec3d world{
                        barycentric[0] * world0.x + barycentric[1] * world1.x +
                            barycentric[2] * world2.x,
                        barycentric[0] * world0.y + barycentric[1] * world1.y +
                            barycentric[2] * world2.y,
                        barycentric[0] * world0.z + barycentric[1] * world1.z +
                            barycentric[2] * world2.z};
                    const auto projected_point = project_frame_gpu_float(camera, world);
                    if (!projected_point) return;
                    px = projected_point->pixel.x / static_cast<double>(downscale);
                    py = projected_point->pixel.y / static_cast<double>(downscale);
                    depth = projected_point->depth;
                } else if (atlas_gpu_float) {
                    const float a = static_cast<float>(barycentric[0]);
                    const float b = static_cast<float>(barycentric[1]);
                    const float c = static_cast<float>(barycentric[2]);
                    px = static_cast<float>(
                        a * static_cast<float>(face_projection.pixel[0].x) +
                        b * static_cast<float>(face_projection.pixel[1].x) +
                        c * static_cast<float>(face_projection.pixel[2].x)) /
                        static_cast<float>(downscale);
                    py = static_cast<float>(
                        a * static_cast<float>(face_projection.pixel[0].y) +
                        b * static_cast<float>(face_projection.pixel[1].y) +
                        c * static_cast<float>(face_projection.pixel[2].y)) /
                        static_cast<float>(downscale);
                    depth = static_cast<float>(
                        a * static_cast<float>(face_projection.depth[0]) +
                        b * static_cast<float>(face_projection.depth[1]) +
                        c * static_cast<float>(face_projection.depth[2]));
                } else {
                    px = (barycentric[0] * face_projection.pixel[0].x +
                          barycentric[1] * face_projection.pixel[1].x +
                          barycentric[2] * face_projection.pixel[2].x) / downscale;
                    py = (barycentric[0] * face_projection.pixel[0].y +
                          barycentric[1] * face_projection.pixel[1].y +
                          barycentric[2] * face_projection.pixel[2].y) / downscale;
                    depth = barycentric[0] * face_projection.depth[0] +
                            barycentric[1] * face_projection.depth[1] +
                            barycentric[2] * face_projection.depth[2];
                }
                if (!projected_rgba.empty()) {
                    projected_rgba.at<cv::Vec4f>(y + region.y, x + region.x) =
                        cv::Vec4f(static_cast<float>(px), static_cast<float>(py),
                                  static_cast<float>(depth),
                                  static_cast<float>(face_index + 1U));
                }
                if (!skip_render_depth &&
                    !fragment_visible(face_projection, camera, depth_downscale,
                                      px * downscale, py * downscale, depth)) return;
                double scale = 1.0;
                for (int level = 0; level < 5; ++level, scale *= 4.0) {
                    const double sx = px / scale;
                    const double sy = py / scale;
                    const double weight_level_scale = std::min(scale, 8.0);
                    const double weight_sx = px * weight_coordinate_scale / weight_level_scale;
                    const double weight_sy = py * weight_coordinate_scale / weight_level_scale;
                    float weight;
                    cv::Vec3f low;
                    if (level == 0) {
                        const int ix = std::clamp(static_cast<int>(std::floor(sx)), 0, pyramid.image[level].cols - 1);
                        const int iy = std::clamp(static_cast<int>(std::floor(sy)), 0, pyramid.image[level].rows - 1);
                        const int wx = std::clamp(static_cast<int>(std::floor(weight_sx)), 0, pyramid.weight[level].cols - 1);
                        const int wy = std::clamp(static_cast<int>(std::floor(weight_sy)), 0, pyramid.weight[level].rows - 1);
                        weight = pyramid.weight[level].at<float>(wy, wx);
                        low = pyramid.image[level].at<cv::Vec3f>(iy, ix);
                    } else {
                        weight = sample_linear_texture_scalar(
                            pyramid.weight[level], weight_sx, weight_sy);
                        low = sample_linear_texture(pyramid.image[level], sx, sy);
                    }
                    if (weight <= 0.0F) continue;
                    if (camera_support_dump_root != nullptr) {
                        camera_support[level].at<std::uint8_t>(y, x) = 1;
                    }
                    cv::Vec3f high(0, 0, 0);
                    if (level + 1 < 5) {
                        high = sample_linear_texture(
                            pyramid.image[level + 1],
                            px / (scale * 4.0), py / (scale * 4.0));
                    }
                    numerator[level].at<cv::Vec3f>(y, x) += weight * (low - high);
                    denominator[level].at<float>(y, x) += weight;
                }
                };
            if (atlas_top_left_mode != 0 && atlas_subpixel_bits >= 0) {
                raster_triangle_fixed_top_left(
                    atlas, region.width, region.height, atlas_subpixel_bits,
                    atlas_top_left_mode == 2, shade_fragment);
            } else {
                raster_triangle(atlas, region.width, region.height, shade_fragment);
            }
        }
        if (!projected_rgba.empty()) {
            const std::filesystem::path root(projection_dump_root);
            std::filesystem::create_directories(root);
            const auto path = root / ("camera_" + std::to_string(camera_index) +
                                      "_projected_rgba32f.raw");
            std::ofstream stream(path, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot create CPU atlas projection dump");
            stream.write(reinterpret_cast<const char *>(projected_rgba.data),
                         static_cast<std::streamsize>(projected_rgba.total() *
                                                      projected_rgba.elemSize()));
        }
        if (camera_support_dump_root != nullptr) {
            const std::filesystem::path root(camera_support_dump_root);
            std::filesystem::create_directories(root);
            for (int level = 4; level >= 0; --level) {
                const auto path = root / ("camera_" + std::to_string(camera_index) +
                                          "_level_" + std::to_string(level) +
                                          "_support_u8.raw");
                std::ofstream stream(path, std::ios::binary);
                if (!stream) {
                    throw std::runtime_error("cannot create per-camera atlas support dump");
                }
                stream.write(reinterpret_cast<const char *>(camera_support[level].data),
                             static_cast<std::streamsize>(camera_support[level].total()));
            }
        }
        std::cout << "enblend camera " << (camera_index + 1) << '/' << cameras.size() << '\n';
    }

    cv::Mat cropped(region.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    std::array<cv::Mat, 5> normalized_band;
    for (auto &band : normalized_band) {
        band = cv::Mat(region.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    }
    for (int y = 0; y < cropped.rows; ++y) {
        for (int x = 0; x < cropped.cols; ++x) {
            cv::Vec3f value(0, 0, 0);
            bool valid = false;
            for (int level = 0; level < 5; ++level) {
                const float weight = denominator[level].at<float>(y, x);
                if (weight > 0.0F) {
                    const cv::Vec3f source = numerator[level].at<cv::Vec3f>(y, x);
                    cv::Vec3f &normalized = normalized_band[level].at<cv::Vec3f>(y, x);
                    for (int channel = 0; channel < 3; ++channel) {
                        normalized[channel] = source[channel] / weight;
                        value[channel] += normalized[channel];
                    }
                    valid = true;
                }
            }
            if (valid) cropped.at<cv::Vec3f>(y, x) = value;
        }
    }
    if (const char *band_root =
            std::getenv("METASHAPE_TEXTURE_BAND_DIAGNOSTIC_DIR")) {
        const std::filesystem::path root(band_root);
        std::filesystem::create_directories(root);
        {
            std::ofstream metadata(root / "region.txt");
            metadata << region.x << ' ' << region.y << ' '
                     << region.width << ' ' << region.height << '\n';
        }
        for (int level = 0; level < 5; ++level) {
            const std::string prefix = "band_" + std::to_string(level);
            std::ofstream normalized_stream(root / (prefix + "_bgr_f32.raw"),
                                            std::ios::binary);
            normalized_stream.write(
                reinterpret_cast<const char *>(normalized_band[level].data),
                static_cast<std::streamsize>(normalized_band[level].total() *
                                             normalized_band[level].elemSize()));
            std::ofstream numerator_stream(root / (prefix + "_numerator_bgr_f32.raw"),
                                           std::ios::binary);
            numerator_stream.write(
                reinterpret_cast<const char *>(numerator[level].data),
                static_cast<std::streamsize>(numerator[level].total() *
                                             numerator[level].elemSize()));
            std::ofstream denominator_stream(root / (prefix + "_denominator_f32.raw"),
                                             std::ios::binary);
            denominator_stream.write(
                reinterpret_cast<const char *>(denominator[level].data),
                static_cast<std::streamsize>(denominator[level].total() *
                                             denominator[level].elemSize()));
        }
    }
    cv::Mat texture(texture_size, texture_size, CV_32FC3, cv::Scalar(0, 0, 0));
    cropped.copyTo(texture(region));
    if (page_support != nullptr) {
        *page_support = cv::Mat(texture_size, texture_size, CV_8U, cv::Scalar(0));
        cv::Mat cropped_support(region.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < cropped_support.rows; ++y) {
            for (int x = 0; x < cropped_support.cols; ++x) {
                if (denominator[4].at<float>(y, x) > 0.0F) {
                    cropped_support.at<std::uint8_t>(y, x) = 255;
                }
            }
        }
        cropped_support.copyTo((*page_support)(region));
    }
    return texture;
}

cv::Mat render_average(const Mesh &mesh, const std::vector<Camera> &cameras,
                       int texture_size, int downscale, int depth_downscale) {
    cv::Mat sum(texture_size, texture_size, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat count(texture_size, texture_size, CV_32F, cv::Scalar(0));
    for (const Camera &camera : cameras) {
        for (const Face &face : mesh.faces) {
            const auto face_projection = project_face(mesh, face, camera);
            if (!face_projection.valid) continue;
            std::array<Vec2d, 3> atlas{};
            std::array<Vec2d, 3> projected{};
            for (int corner = 0; corner < 3; ++corner) {
                const Vec2d uv = mesh.texcoords[face.texcoord[corner]];
                atlas[corner] = {uv.x * texture_size, (1.0 - uv.y) * texture_size};
                projected[corner] = {face_projection.pixel[corner].x / downscale,
                                     face_projection.pixel[corner].y / downscale};
            }
            raster_triangle(atlas, texture_size, texture_size,
                            [&](int x, int y, const std::array<double, 3> &barycentric) {
                const double px = (barycentric[0] * face_projection.pixel[0].x +
                                   barycentric[1] * face_projection.pixel[1].x +
                                   barycentric[2] * face_projection.pixel[2].x) / downscale;
                const double py = (barycentric[0] * face_projection.pixel[0].y +
                                   barycentric[1] * face_projection.pixel[1].y +
                                   barycentric[2] * face_projection.pixel[2].y) / downscale;
                const double depth = barycentric[0] * face_projection.depth[0] +
                                     barycentric[1] * face_projection.depth[1] +
                                     barycentric[2] * face_projection.depth[2];
                if (!fragment_visible(face_projection, camera, depth_downscale,
                                      px * downscale, py * downscale, depth)) return;
                sum.at<cv::Vec3f>(y, x) += bilinear(camera.image, px, py);
                count.at<float>(y, x) += 1.0F;
            });
        }
    }
    cv::Mat texture(texture_size, texture_size, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < texture.rows; ++y) {
        for (int x = 0; x < texture.cols; ++x) {
            const float n = count.at<float>(y, x);
            if (n > 0.0F) texture.at<cv::Vec3f>(y, x) = sum.at<cv::Vec3f>(y, x) / n;
        }
    }
    return texture;
}

cv::Mat render_mosaic(const Mesh &mesh, const std::vector<Camera> &cameras,
                      const std::vector<int> &winners, int texture_size, int downscale) {
    cv::Mat texture(texture_size, texture_size, CV_32FC3, cv::Scalar(0, 0, 0));
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
        const int winner = winners[face_index];
        if (winner < 0) continue;
        const Face &face = mesh.faces[face_index];
        const Camera &camera = cameras[static_cast<std::size_t>(winner)];
        std::array<Vec2d, 3> atlas{};
        std::array<Vec2d, 3> projected{};
        bool valid = true;
        for (int corner = 0; corner < 3; ++corner) {
            const Vec2d uv = mesh.texcoords[face.texcoord[corner]];
            atlas[corner] = {uv.x * texture_size, (1.0 - uv.y) * texture_size};
            auto pixel = project_frame(camera.pose, camera.calibration, mesh.vertices[face.vertex[corner]], false);
            if (!pixel) { valid = false; break; }
            projected[corner] = {pixel->x / downscale, pixel->y / downscale};
        }
        if (!valid) continue;
        raster_triangle(atlas, texture.cols, texture.rows, [&](int x, int y, const std::array<double, 3> &w) {
            const double px = w[0] * projected[0].x + w[1] * projected[1].x + w[2] * projected[2].x;
            const double py = w[0] * projected[0].y + w[1] * projected[1].y + w[2] * projected[2].y;
            texture.at<cv::Vec3f>(y, x) = bilinear(camera.image, px, py);
        });
    }
    return texture;
}

bool has_texture_sample(const cv::Vec3f &value) {
    return value[0] != 0.0F || value[1] != 0.0F || value[2] != 0.0F;
}

cv::Mat interpolate_texture_page(const Mesh &mesh, const cv::Mat &input,
                                 const cv::Mat &initial_mask = cv::Mat());

// Recovered from FUN_141f34510/FUN_141f34c20/FUN_141f357c0/FUN_141f34fb0.
// Metashape fills missed surface texels in mesh-topology space: it samples the
// atlas at geometric vertices, propagates those samples over missing mesh
// vertices, and rasterizes the resulting vertex field back into UV triangles.
cv::Mat interpolate_holes_on_mesh(const Mesh &mesh, const cv::Mat &input,
                                  cv::Mat &page_support) {
    struct VertexField {
        cv::Vec3f sum{0, 0, 0};
        float denominator = 0.0F;
        bool processed = false;
    };

    cv::Mat valid = page_support.clone();
    if (valid.empty()) {
        valid = cv::Mat(input.rows, input.cols, CV_8U, cv::Scalar(0));
        for (int y = 0; y < input.rows; ++y) {
            for (int x = 0; x < input.cols; ++x) {
                if (has_texture_sample(input.at<cv::Vec3f>(y, x))) valid.at<std::uint8_t>(y, x) = 255;
            }
        }
    }
    const char *prefill_vertex_colors = std::getenv("METASHAPE_TEXTURE_VALIDATE_PREFILL_VERTEX_COLORS");
    const cv::Mat vertex_image = prefill_vertex_colors != nullptr &&
                                         std::string(prefill_vertex_colors) == "1"
        ? interpolate_texture_page(mesh, input)
        : input;

    std::vector<VertexField> field(mesh.vertices.size());
    for (const Face &face : mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            const Vec2d uv = mesh.texcoords[face.texcoord[corner]];
            const double sample_x = uv.x * input.cols - 0.5;
            const double sample_y = (1.0 - uv.y) * input.rows - 0.5;
            const int x0 = static_cast<int>(std::floor(sample_x));
            const int y0 = static_cast<int>(std::floor(sample_y));
            bool any_valid = false;
            cv::Vec3f masked_sum(0, 0, 0);
            double masked_normalization = 0.0;
            for (int dy = 0; dy != 2; ++dy) {
                for (int dx = 0; dx != 2; ++dx) {
                    const int x = std::clamp(x0 + dx, 0, input.cols - 1);
                    const int y = std::clamp(y0 + dy, 0, input.rows - 1);
                    if (valid.at<std::uint8_t>(y, x) != 0) {
                        any_valid = true;
                        const double fx = sample_x - std::floor(sample_x);
                        const double fy = sample_y - std::floor(sample_y);
                        const double weight = (dx == 0 ? 1.0 - fx : fx) * (dy == 0 ? 1.0 - fy : fy);
                        masked_sum += vertex_image.at<cv::Vec3f>(y, x) * static_cast<float>(weight);
                        masked_normalization += weight;
                    }
                }
            }
            if (!any_valid || masked_normalization <= 0.0) continue;
            VertexField &vertex = field[face.vertex[corner]];
            // Only supported texels participate in a seam footprint.  Mixing
            // unsupported black background into a vertex seed permanently
            // darkens the topology fill.
            vertex.sum += masked_sum / static_cast<float>(masked_normalization);
            vertex.denominator += 1.0F;
        }
    }

    std::vector<std::vector<std::uint32_t>> adjacency(mesh.vertices.size());
    auto add_unique = [&](std::uint32_t from, std::uint32_t to) {
        auto &neighbors = adjacency[from];
        if (std::find(neighbors.begin(), neighbors.end(), to) == neighbors.end()) neighbors.push_back(to);
    };
    for (const Face &face : mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            const std::uint32_t a = face.vertex[corner];
            const std::uint32_t b = face.vertex[(corner + 1) % 3];
            if (field[a].denominator == 0.0F || field[b].denominator == 0.0F) {
                add_unique(a, b);
                add_unique(b, a);
            }
        }
    }

    std::queue<std::uint32_t> wavefront;
    for (std::uint32_t vertex = 0; vertex < field.size(); ++vertex) {
        if (field[vertex].denominator > 0.0F && !adjacency[vertex].empty()) wavefront.push(vertex);
    }
    while (!wavefront.empty()) {
        const std::uint32_t vertex = wavefront.front();
        wavefront.pop();
        VertexField &source = field[vertex];
        if (source.processed || source.denominator <= 0.0F) continue;
        source.sum /= source.denominator;
        source.processed = true;
        for (const std::uint32_t neighbor_index : adjacency[vertex]) {
            VertexField &neighbor = field[neighbor_index];
            if (neighbor.processed) continue;
            if (neighbor.denominator == 0.0F) wavefront.push(neighbor_index);
            neighbor.sum += source.sum;
            neighbor.denominator += 1.0F;
        }
    }

    cv::Mat output = input.clone();
    for (const Face &face : mesh.faces) {
        std::array<Vec2d, 3> atlas{};
        for (int corner = 0; corner < 3; ++corner) {
            const Vec2d uv = mesh.texcoords[face.texcoord[corner]];
            // FUN_141f34fb0 shifts UV-space pixel centers by -0.5 before the
            // scanline rasterizer and its per-sample 2x2 conservative store.
            atlas[corner] = {uv.x * input.cols - 0.5, (1.0 - uv.y) * input.rows - 0.5};
        }

        auto store_sample = [&](int sample_x, int sample_y) {
            const double determinant =
                (atlas[1].y - atlas[2].y) * (atlas[0].x - atlas[2].x) +
                (atlas[2].x - atlas[1].x) * (atlas[0].y - atlas[2].y);
            if (std::abs(determinant) < 1.0e-20) return;
            const double a = ((atlas[1].y - atlas[2].y) * (sample_x - atlas[2].x) +
                              (atlas[2].x - atlas[1].x) * (sample_y - atlas[2].y)) / determinant;
            const double b = ((atlas[2].y - atlas[0].y) * (sample_x - atlas[2].x) +
                              (atlas[0].x - atlas[2].x) * (sample_y - atlas[2].y)) / determinant;
            const std::array<double, 3> barycentric{a, b, 1.0 - a - b};
            cv::Vec3f numerator(0, 0, 0);
            double denominator = 0.0;
            for (int corner = 0; corner < 3; ++corner) {
                const VertexField &vertex = field[face.vertex[corner]];
                if (vertex.denominator <= 0.0F) continue;
                const float weight = vertex.processed ? 1.0F : vertex.denominator;
                numerator += vertex.sum * static_cast<float>(barycentric[corner]);
                denominator += weight * barycentric[corner];
            }
            if (denominator <= 0.0) return;
            const cv::Vec3f color = numerator / static_cast<float>(denominator);
            for (int dy = -1; dy <= 0; ++dy) {
                for (int dx = -1; dx <= 0; ++dx) {
                    const int x = sample_x + dx;
                    const int y = sample_y + dy;
                    if (x < 0 || y < 0 || x >= output.cols || y >= output.rows) continue;
                    if (valid.at<std::uint8_t>(y, x) != 0) continue;
                    output.at<cv::Vec3f>(y, x) = color;
                    valid.at<std::uint8_t>(y, x) = 255;
                }
            }
        };

        std::array<Vec2d, 3> sorted = atlas;
        std::sort(sorted.begin(), sorted.end(), [](const Vec2d &left, const Vec2d &right) {
            return left.y < right.y;
        });
        // The call site passes the inclusive-edge flag (the hidden ninth
        // argument) as true; the scanline itself is not shifted back here.
        const int first_y = static_cast<int>(sorted[0].y + 0.5);
        const int middle_y = static_cast<int>(sorted[1].y + 0.5);
        const int last_y = static_cast<int>(sorted[2].y + 0.5);
        auto edge_x = [](const Vec2d &from, const Vec2d &to, int y) {
            if (to.y == from.y) return from.x;
            return from.x + (static_cast<double>(y) - from.y) * (to.x - from.x) / (to.y - from.y);
        };
        for (int y = first_y; y < last_y; ++y) {
            const double long_edge = edge_x(sorted[0], sorted[2], y);
            const double short_edge = y < middle_y
                ? edge_x(sorted[0], sorted[1], y)
                : edge_x(sorted[1], sorted[2], y);
            const double left = std::min(long_edge, short_edge);
            const double right = std::max(long_edge, short_edge);
            const int first_x = static_cast<int>(left + 0.5);
            const int last_x = static_cast<int>(right + 0.5);
            for (int x = first_x; x < last_x; ++x) store_sample(x, y);
        }
    }
    page_support = std::move(valid);
    return output;
}

// Recovered from FUN_1424bec50/FUN_1424bc680 (one-pixel seed expansion) and
// FUN_1424b9bb0/FUN_1424a9970/FUN_1424b2360 (recursive masked pyramid).
// This page interpolation is a common save-stage operation and is performed
// independently of BuildTexture's fill_holes topology option.
void interpolate_masked_pyramid_u8(cv::Mat &image, cv::Mat &mask,
                                   bool expand_seed = true) {
    double sample_offset = -0.25;
    if (const char *value = std::getenv("METASHAPE_TEXTURE_PAGE_SAMPLE_OFFSET")) {
        sample_offset = std::stod(value);
    }
    if (expand_seed) {
        cv::Mat expanded_mask = mask.clone();
        cv::Mat expanded_image = image.clone();
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                if (mask.at<std::uint8_t>(y, x) != 0) continue;
                int best_distance = std::numeric_limits<int>::max();
                int best_x = -1;
                int best_y = -1;
                // FUN_1424bc680 runs both signed offsets from -1 while offset < 2:
                // the seed expansion is a full 3x3 search, not a 2x2 quadrant.
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = x + dx;
                        const int sy = y + dy;
                        if (sx < 0 || sy < 0 || sx >= image.cols || sy >= image.rows) continue;
                        if (mask.at<std::uint8_t>(sy, sx) == 0) continue;
                        const int distance = dx * dx + dy * dy;
                        if (distance <= best_distance) {
                            best_distance = distance;
                            best_x = sx;
                            best_y = sy;
                        }
                    }
                }
                if (best_x >= 0) {
                    expanded_image.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(best_y, best_x);
                    expanded_mask.at<std::uint8_t>(y, x) = 255;
                }
            }
        }
        image = std::move(expanded_image);
        mask = std::move(expanded_mask);
        if (const char *diagnostic_root =
                std::getenv("METASHAPE_TEXTURE_PAGE_DIAGNOSTIC_DIR")) {
            const std::filesystem::path root(diagnostic_root);
            std::filesystem::create_directories(root);
            cv::Mat rgb;
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
            std::ofstream image_stream(root / "expanded_image_rgb_u8.raw",
                                       std::ios::binary);
            image_stream.write(reinterpret_cast<const char *>(rgb.data),
                               static_cast<std::streamsize>(rgb.total() *
                                                            rgb.elemSize()));
            std::ofstream mask_stream(root / "expanded_mask_u8.raw",
                                      std::ios::binary);
            mask_stream.write(reinterpret_cast<const char *>(mask.data),
                              static_cast<std::streamsize>(mask.total() *
                                                           mask.elemSize()));
        }
    }

    if (image.cols <= 1 && image.rows <= 1) return;
    const int half_width = (image.cols + 1) / 2;
    const int half_height = (image.rows + 1) / 2;
    cv::Mat half_image(half_height, half_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat half_mask(half_height, half_width, CV_8U, cv::Scalar(0));
    for (int y = 0; y < half_height; ++y) {
        for (int x = 0; x < half_width; ++x) {
            std::array<std::uint64_t, 3> sum{};
            std::uint64_t normalization = 0;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = 2 * x + dx;
                    const int sy = 2 * y + dy;
                    if (sx >= image.cols || sy >= image.rows) continue;
                    const std::uint8_t weight = mask.at<std::uint8_t>(sy, sx);
                    if (weight == 0) continue;
                    const cv::Vec3b value = image.at<cv::Vec3b>(sy, sx);
                    normalization += weight;
                    for (int channel = 0; channel < 3; ++channel) {
                        sum[channel] += static_cast<std::uint64_t>(value[channel]) * weight;
                    }
                }
            }
            if (normalization != 0) {
                cv::Vec3b &value = half_image.at<cv::Vec3b>(y, x);
                // Linux 2.3.2 worker 0x2b6b350 converts the integer mask sum
                // to double, computes a reciprocal with divsd, multiplies the
                // integer color sum by that reciprocal, then truncates with
                // cvttsd2si.  This is observably different from integer
                // division at exact-integer boundaries: the rounded double
                // reciprocal can place the product one ULP below the integer.
                const double inverse_normalization =
                    1.0 / static_cast<double>(normalization);
                for (int channel = 0; channel < 3; ++channel) {
                    value[channel] = static_cast<std::uint8_t>(
                        static_cast<double>(sum[channel]) * inverse_normalization);
                }
                half_mask.at<std::uint8_t>(y, x) = 1;
            }
        }
    }

    interpolate_masked_pyramid_u8(half_image, half_mask, false);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (mask.at<std::uint8_t>(y, x) != 0) continue;
            const double source_x = std::max(0.0, 0.5 * x + sample_offset);
            const double source_y = std::max(0.0, 0.5 * y + sample_offset);
            const int x0 = static_cast<int>(source_x);
            const int y0 = static_cast<int>(source_y);
            const double fx = source_x - x0;
            const double fy = source_y - y0;
            cv::Vec3d sum(0, 0, 0);
            double normalization = 0.0;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = x0 + dx;
                    const int sy = y0 + dy;
                    if (sx >= half_image.cols || sy >= half_image.rows) continue;
                    const double weight = (dx == 0 ? 1.0 - fx : fx) * (dy == 0 ? 1.0 - fy : fy);
                    sum += cv::Vec3d(half_image.at<cv::Vec3b>(sy, sx)) * weight;
                    normalization += weight;
                }
            }
            if (normalization != 0.0) {
                cv::Vec3b &destination = image.at<cv::Vec3b>(y, x);
                for (int channel = 0; channel < 3; ++channel) {
                    destination[channel] =
                        static_cast<std::uint8_t>(sum[channel] / normalization);
                }
            }
        }
    }
    if (expand_seed) {
        if (const char *diagnostic_root =
                std::getenv("METASHAPE_TEXTURE_PAGE_DIAGNOSTIC_DIR")) {
            const std::filesystem::path root(diagnostic_root);
            std::filesystem::create_directories(root);
            cv::Mat rgb;
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
            std::ofstream stream(root / "interpolated_image_rgb_u8.raw",
                                 std::ios::binary);
            stream.write(reinterpret_cast<const char *>(rgb.data),
                         static_cast<std::streamsize>(rgb.total() * rgb.elemSize()));
        }
    }
}

cv::Mat interpolate_texture_page(const Mesh &mesh, const cv::Mat &input,
                                 const cv::Mat &initial_mask) {
    double minimum_x = static_cast<double>(input.cols);
    double minimum_y = static_cast<double>(input.rows);
    double maximum_x = 0.0;
    double maximum_y = 0.0;
    for (const Face &face : mesh.faces) {
        for (const std::uint32_t texcoord : face.texcoord) {
            const Vec2d uv = mesh.texcoords[texcoord];
            const double x = uv.x * input.cols;
            const double y = (1.0 - uv.y) * input.rows;
            minimum_x = std::min(minimum_x, x);
            minimum_y = std::min(minimum_y, y);
            maximum_x = std::max(maximum_x, x);
            maximum_y = std::max(maximum_y, y);
        }
    }
    const int min_x = std::clamp(static_cast<int>(std::floor(minimum_x)), 0, input.cols - 1);
    const int min_y = std::clamp(static_cast<int>(std::floor(minimum_y)), 0, input.rows - 1);
    const int max_x = std::clamp(static_cast<int>(std::ceil(maximum_x)) + 1, 0, input.cols - 1);
    const int max_y = std::clamp(static_cast<int>(std::ceil(maximum_y)) + 1, 0, input.rows - 1);
    const cv::Rect region(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);

    cv::Mat encoded;
    cv::Mat mask;
    bool already_expanded = false;
    if (const char *native_page_root =
            std::getenv("METASHAPE_TEXTURE_NATIVE_PAGE_INPUT_DIR")) {
        const std::filesystem::path root(native_page_root);
        cv::Mat native_rgb(input.rows, input.cols, CV_8UC3);
        mask = cv::Mat(input.rows, input.cols, CV_8U);
        const auto read_exact = [](const std::filesystem::path &path,
                                   cv::Mat &destination) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("cannot load native page buffer: " +
                                         path.string());
            }
            const auto size = static_cast<std::streamsize>(
                destination.total() * destination.elemSize());
            stream.read(reinterpret_cast<char *>(destination.data), size);
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("invalid native page-buffer size: " +
                                         path.string());
            }
        };
        read_exact(root / "input_image_2048x2048x3_u8.raw", native_rgb);
        read_exact(root / "input_mask_2048x2048_u8.raw", mask);
        cv::cvtColor(native_rgb, encoded, cv::COLOR_RGB2BGR);
        already_expanded = true;
    } else {
    // The observed FUN_1424b9bb0 recursion receives 2048x2048 at the top
    // level.  Metashape builds the pyramid over the complete atlas, then
    // publishes only the UV page rectangle into the exported texture.
        encoded = cv::Mat(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < input.rows; ++y) {
            for (int x = 0; x < input.cols; ++x) {
                const cv::Vec3f source = input.at<cv::Vec3f>(y, x);
                cv::Vec3b &destination = encoded.at<cv::Vec3b>(y, x);
                for (int channel = 0; channel < 3; ++channel) {
                    const float scaled = source[channel] * 255.0F + 0.5F;
                    destination[channel] = scaled < 0.0F
                        ? 0
                        : scaled > 255.0F
                            ? 255
                            : static_cast<std::uint8_t>(scaled);
                }
            }
        }
        if (!initial_mask.empty()) {
            if (initial_mask.type() != CV_8U || initial_mask.size() != encoded.size()) {
                throw std::invalid_argument("page support mask does not match texture");
            }
            mask = initial_mask.clone();
        } else {
            mask = cv::Mat(encoded.rows, encoded.cols, CV_8U, cv::Scalar(0));
        }
        for (int y = 0; y < encoded.rows; ++y) {
            for (int x = 0; x < encoded.cols; ++x) {
                if (initial_mask.empty() &&
                    has_texture_sample(input.at<cv::Vec3f>(y, x))) {
                    mask.at<std::uint8_t>(y, x) = 255;
                }
            }
        }
    }
    if (const char *diagnostic_root =
            std::getenv("METASHAPE_TEXTURE_PAGE_DIAGNOSTIC_DIR")) {
        const std::filesystem::path root(diagnostic_root);
        std::filesystem::create_directories(root);
        cv::Mat rgb;
        cv::cvtColor(encoded, rgb, cv::COLOR_BGR2RGB);
        std::ofstream image_stream(root / "seed_image_rgb_u8.raw",
                                   std::ios::binary);
        image_stream.write(reinterpret_cast<const char *>(rgb.data),
                           static_cast<std::streamsize>(rgb.total() *
                                                        rgb.elemSize()));
        std::ofstream mask_stream(root / "seed_mask_u8.raw", std::ios::binary);
        mask_stream.write(reinterpret_cast<const char *>(mask.data),
                          static_cast<std::streamsize>(mask.total() *
                                                       mask.elemSize()));
    }
    interpolate_masked_pyramid_u8(encoded, mask, !already_expanded);
    if (const char *diagnostic_root =
            std::getenv("METASHAPE_TEXTURE_PAGE_DIAGNOSTIC_DIR")) {
        const std::filesystem::path root(diagnostic_root);
        std::filesystem::create_directories(root);
        cv::Mat rgb;
        cv::cvtColor(encoded, rgb, cv::COLOR_BGR2RGB);
        std::ofstream stream(root / "interpolated_image_rgb_u8.raw",
                             std::ios::binary);
        stream.write(reinterpret_cast<const char *>(rgb.data),
                     static_cast<std::streamsize>(rgb.total() * rgb.elemSize()));
    }
    cv::Mat interpolated;
    encoded.convertTo(interpolated, CV_32FC3, 1.0 / 255.0);
    return interpolated;
}

void save_packed_obj(const PipelineInput& input, const Mesh& mesh)
{
    std::ofstream output(input.output_directory / "model.obj");
    if (!output)
        throw std::runtime_error("cannot create packed OBJ output");
    output << "mtllib model.mtl\n" << std::fixed << std::setprecision(input.precise_output_uv ? 17 : 6);
    for (const Vec3d& vertex : mesh.vertices)
    {
        output << "v " << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
    }
    for (const Vec2d& uv : mesh.texcoords)
        output << "vt " << uv.x << ' ' << uv.y << '\n';
    output << "usemtl material_0\n";
    for (const Face &face : mesh.faces) {
        output << "f";
        for (int corner = 0; corner < 3; ++corner) {
            output << ' ' << face.vertex[corner] + 1U << '/' << face.texcoord[corner] + 1U;
        }
        output << '\n';
    }
}

void save_outputs(const PipelineInput &input, const Mesh &mesh, const cv::Mat &texture) {
    std::filesystem::create_directories(input.output_directory);
    const auto texture_path = input.output_directory / "texture.tif";
    cv::Mat encoded;
    texture.convertTo(encoded, CV_8UC3, 255.0);
#if defined(MSTEXTURE_EXACT_PAGE_CODEC)
    // The native save path does not feed the interpolated RGB directly to its
    // final lossless TIFF.  It first crosses a tiled quality-100 JPEG-in-TIFF
    // boundary, then decodes that page and merges it into the final image.
    cv::Mat rgb;
    cv::cvtColor(encoded, rgb, cv::COLOR_BGR2RGB);
    const auto decoded_rgb = page_codec::jpeg_tiff_roundtrip_rgb(
        std::span<const std::uint8_t>(rgb.ptr<std::uint8_t>(),
                                      rgb.total() * rgb.elemSize()),
        rgb.cols, rgb.rows, input.output_directory / ".texture_page.jpeg.tiff");
    page_codec::write_lzw_tiff_rgb(decoded_rgb, rgb.cols, rgb.rows, texture_path);
#else
    if (!cv::imwrite(texture_path.string(), encoded)) {
        throw std::runtime_error("cannot write TIFF output");
    }
#endif
    save_packed_obj(input, mesh);
    std::ofstream material(input.output_directory / "model.mtl");
    material << "newmtl material_0\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd texture.tif\n";
    std::cout << "BuildTexture output: " << texture_path << '\n';
}

void save_winners(const std::filesystem::path &directory, const std::vector<int> &winners) {
    std::filesystem::create_directories(directory);
    std::ofstream output(directory / "winners.txt");
    std::ofstream binary(directory / "winner_labels_u32.raw", std::ios::binary);
    if (!output || !binary) throw std::runtime_error("cannot create winner outputs");
    for (int winner : winners) {
        output << winner << '\n';
        const std::uint32_t encoded = winner < 0
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(winner);
        binary.write(reinterpret_cast<const char *>(&encoded), sizeof(encoded));
    }
}

std::vector<int> load_winners(const std::filesystem::path &path, std::size_t face_count) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open winner labels: " + path.string());
    std::vector<int> winners;
    for (int value; input >> value;) winners.push_back(value);
    if (winners.size() != face_count) throw std::runtime_error("winner label count does not match faces");
    return winners;
}

std::vector<float> read_exact_float_array(const std::filesystem::path &path,
                                          std::size_t expected_count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open winner quality input: " + path.string());
    const auto bytes = input.tellg();
    if (bytes != static_cast<std::streamoff>(expected_count * sizeof(float))) {
        throw std::runtime_error("winner quality input has wrong length: " + path.string());
    }
    std::vector<float> values(expected_count);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) throw std::runtime_error("cannot read winner quality input: " + path.string());
    return values;
}

void apply_winner_quality_inputs(const std::filesystem::path &root,
                                 std::size_t face_count,
                                 std::vector<Camera> &cameras) {
    if (root.empty()) return;
    const auto face_weight = read_exact_float_array(root / "face_weight.raw", face_count);
    for (std::size_t camera = 0; camera < cameras.size(); ++camera) {
        std::ostringstream name;
        name << "camera_" << std::setw(3) << std::setfill('0') << camera;
        const auto directory = root / name.str();
        cameras[camera].exact_resolution =
            read_exact_float_array(directory / "resolution.raw", face_count);
        cameras[camera].exact_nadirness =
            read_exact_float_array(directory / "nadirness.raw", face_count);
        cameras[camera].exact_face_weight = face_weight;
        const auto edge_count = directory / "edge_count.raw";
        if (std::filesystem::is_regular_file(edge_count)) {
            cameras[camera].exact_edge_count =
                read_exact_float_array(edge_count, face_count * 3U);
        }
    }
}

}  // namespace

void build_texture_compat(const PipelineInput &input) {
    if (input.parameters.texture_size <= 0 || input.parameters.downscale <= 0) {
        throw std::invalid_argument("texture_size and downscale must be positive");
    }
    std::cout << "BuildTexture: blending_mode=NaturalBlending, texture_size="
              << input.parameters.texture_size << ", downscale=" << input.parameters.downscale
              << ", sharpening=" << input.parameters.sharpening << '\n';
    Mesh mesh = load_obj(input.model_obj);
    // The OBJ export is rounded to six decimal places.  Metashape's winner
    // and edge host stages consume the project's float32 mesh instead; the
    // difference is small geometrically but crosses uint8 edge-cost bins.
    const std::filesystem::path &project_geometry =
        input.project_input_directory.empty()
            ? input.focus_project_input_directory
            : input.project_input_directory;
    if (!project_geometry.empty()) {
        apply_project_float32_geometry(mesh, project_geometry);
    }
    const char *native_atlas_root =
        std::getenv("METASHAPE_TEXTURE_NATIVE_ATLAS_BANDS_DIR");
    const char *keep_input_uv_environment =
        std::getenv("METASHAPE_TEXTURE_KEEP_INPUT_UV");
    const bool keep_input_uv = input.keep_input_uv ||
        (keep_input_uv_environment != nullptr &&
         std::string(keep_input_uv_environment) == "1");
    if ((!input.atlas_attachments.empty() || native_atlas_root != nullptr) &&
        keep_input_uv) {
        std::vector<int> winners = input.winner_labels.empty()
            ? std::vector<int>(mesh.faces.size(), -1)
            : load_winners(input.winner_labels, mesh.faces.size());
        save_winners(input.output_directory, winners);
        const std::vector<Camera> no_cameras;
        cv::Mat page_support;
        cv::Mat texture = render_multiband(
            mesh, no_cameras, winners, input.parameters.texture_size,
            input.parameters.downscale, input.parameters.downscale,
            input.atlas_attachments, &page_support);
        if (input.parameters.fill_holes) {
            texture = interpolate_holes_on_mesh(mesh, texture, page_support);
        }
        const char *disable_page_interpolation =
            std::getenv("METASHAPE_TEXTURE_DISABLE_PAGE_INTERPOLATION");
        if (disable_page_interpolation == nullptr ||
            std::string(disable_page_interpolation) != "1") {
            texture = interpolate_texture_page(mesh, texture, page_support);
        }
        save_outputs(input, mesh, texture);
        return;
    }
    const char *atlas_dump_only_environment =
        std::getenv("METASHAPE_TEXTURE_ATLAS_VERTEX_DUMP_ONLY");
    const char *winners_only_environment =
        std::getenv("METASHAPE_TEXTURE_WINNERS_ONLY");
    const bool winners_only = winners_only_environment != nullptr &&
        std::string(winners_only_environment) == "1";
    const bool exact_winner_only = winners_only && input.winner_labels.empty() &&
        !input.winner_quality_input_directory.empty() &&
        !input.parameters.ghosting_filter && !input.parameters.out_of_focus_filter;
    const bool metadata_only_scene = exact_winner_only || (!input.winner_labels.empty() &&
        atlas_dump_only_environment != nullptr &&
        std::string(atlas_dump_only_environment) == "1");
    std::vector<Camera> cameras = load_scene(
        input.scene_json, input.parameters.downscale, input.parameters.sharpening,
        !metadata_only_scene);
    apply_winner_quality_inputs(input.winner_quality_input_directory,
                                mesh.faces.size(), cameras);
    if (const char *image_pyramid_root =
            std::getenv("METASHAPE_TEXTURE_IMAGE_PYRAMID_ONLY_DIR")) {
        const std::filesystem::path root(image_pyramid_root);
        std::filesystem::create_directories(root);
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            cv::Mat level = cameras[camera_index].image;
            for (int level_index = 0; level_index < 5; ++level_index) {
                cv::Mat encoded;
                level.convertTo(encoded, CV_8UC3, 255.0);
                std::vector<cv::Mat> bgr;
                cv::split(encoded, bgr);
                std::ostringstream name;
                name << "camera_" << std::setw(3) << std::setfill('0') << camera_index
                     << "_level_" << level_index << '_' << level.cols << 'x'
                     << level.rows << "_r8_array3.raw";
                std::ofstream stream(root / name.str(), std::ios::binary);
                if (!stream) throw std::runtime_error("cannot create image pyramid dump");
                for (auto channel = bgr.rbegin(); channel != bgr.rend(); ++channel) {
                    stream.write(reinterpret_cast<const char *>(channel->data),
                                 static_cast<std::streamsize>(channel->total()));
                }
                if (!stream) throw std::runtime_error("cannot write image pyramid dump");
                if (level_index + 1 < 5) level = downscale_6tap(level, 4);
            }
        }
        std::cout << "image pyramid diagnostic output: " << root << '\n';
        return;
    }
    if (input.parameters.out_of_focus_filter &&
        std::any_of(cameras.begin(), cameras.end(),
                    [](const Camera &camera) { return camera.focus_quality_map.empty(); })) {
        if (input.focus_project_input_directory.empty() || input.focus_shader_directory.empty()) {
            throw std::invalid_argument(
                "out-of-focus filtering without focus_quality_path requires "
                "--focus-project-input and --focus-shaders; --focus-zrange is optional");
        }
        recovered::FocusProjectPipelineInput focus_input{
            input.scene_json,
            input.focus_project_input_directory,
            input.focus_zrange_directory,
            input.focus_shader_directory,
            input.focus_quality_graphics_directory,
            input.focus_quality_resolution_directory,
            input.focus_face_weight_path,
            input.focus_nadirness_shader_path,
            input.focus_resolution_shader_path,
            input.parameters.ghosting_filter,
            input.focus_work_directory,
        };
        const auto generated = recovered::build_focus_quality_images(focus_input);
        if (generated.size() != cameras.size()) {
            throw std::runtime_error("generated focus image count does not match cameras");
        }
        for (std::size_t index = 0; index < cameras.size(); ++index) {
            const auto &source = generated[index];
            if (source.width * 4 != cameras[index].calibration.width ||
                source.height * 4 != cameras[index].calibration.height ||
                source.pixels.size() !=
                    static_cast<std::size_t>(source.width * source.height)) {
                throw std::runtime_error("generated focus image dimensions do not match sensor");
            }
            cv::Mat view(source.height, source.width, CV_8UC1,
                         const_cast<std::uint8_t *>(source.pixels.data()));
            cameras[index].focus_quality_map = view.clone();
            cameras[index].focus_quality_sum = source.face_sum;
            cameras[index].focus_quality_count = source.face_count;
            cameras[index].exact_nadirness = source.face_nadirness;
            cameras[index].exact_resolution = source.face_resolution;
            cameras[index].exact_face_weight = source.face_weight;
            cameras[index].exact_edge_count = source.edge_count;
        }
    }
    if (const char *focus_only = std::getenv("METASHAPE_TEXTURE_FOCUS_ONLY")) {
        if (std::string(focus_only) == "1") return;
    }
    const int depth_downscale = input.parameters.downscale * 2;
    std::vector<int> winners;
    if (input.winner_labels.empty()) {
        if (!exact_winner_only) {
            build_depth_buffers(mesh, cameras, depth_downscale);
        }
        winners = estimate_winners(mesh, cameras, depth_downscale,
                                   input.parameters.ghosting_filter,
                                   input.parameters.out_of_focus_filter);
    } else {
        winners = load_winners(input.winner_labels, mesh.faces.size());
    }
    save_winners(input.output_directory, winners);
    if (winners_only) return;
    bool natural_uv_dump_only = false;
    if (!keep_input_uv) {
        build_natural_uv(mesh, cameras, winners, input.parameters.texture_size,
                         input.parameters.downscale);
        if (const char *dump_only =
                std::getenv("METASHAPE_TEXTURE_XATLAS_INPUT_DUMP_ONLY")) {
            natural_uv_dump_only = std::string(dump_only) == "1";
        }
    }
    if (natural_uv_dump_only) return;
    if (const char *natural_uv_only =
            std::getenv("METASHAPE_TEXTURE_NATURAL_UV_ONLY")) {
        if (std::string(natural_uv_only) == "1") {
            save_packed_obj(input, mesh);
            return;
        }
    }
    const char *validation_mode = std::getenv("METASHAPE_TEXTURE_VALIDATION_MODE");
    cv::Mat texture;
    cv::Mat page_support;
    if (validation_mode != nullptr && std::string(validation_mode) == "average") {
        texture = render_average(mesh, cameras, input.parameters.texture_size,
                                 input.parameters.downscale, depth_downscale);
    } else if (validation_mode != nullptr && std::string(validation_mode) == "mosaic") {
        texture = render_mosaic(mesh, cameras, winners, input.parameters.texture_size,
                                input.parameters.downscale);
    } else {
        // Native enblend push constants captured at the 2048 atlas draw are
        // [image_downscale=2, depth/weight_downscale=2, scale_cap=8].  The
        // 1/4-resolution depth image above belongs to candidate scoring; the
        // atlas fragment shader binds a separate 1/2-resolution depth pass.
        // Rebuild that attachment after winner selection and pass its actual
        // scale through every enblend visibility lookup.
        const int enblend_depth_downscale = input.parameters.downscale;
        const char *atlas_dump_only =
            std::getenv("METASHAPE_TEXTURE_ATLAS_VERTEX_DUMP_ONLY");
        const bool dumping_atlas_geometry = atlas_dump_only != nullptr &&
            std::string(atlas_dump_only) == "1";
        if (std::getenv("METASHAPE_TEXTURE_NATIVE_ATLAS_BANDS_DIR") == nullptr &&
            !dumping_atlas_geometry) {
            build_depth_buffers(mesh, cameras, enblend_depth_downscale);
        }
        texture = render_multiband(mesh, cameras, winners, input.parameters.texture_size,
                                   input.parameters.downscale, enblend_depth_downscale,
                                   input.atlas_attachments, &page_support);
        if (dumping_atlas_geometry) {
            save_packed_obj(input, mesh);
            return;
        }
    }
    if (input.parameters.fill_holes) {
        texture = interpolate_holes_on_mesh(mesh, texture, page_support);
    }
    const char *disable_page_interpolation = std::getenv("METASHAPE_TEXTURE_DISABLE_PAGE_INTERPOLATION");
    if (disable_page_interpolation == nullptr || std::string(disable_page_interpolation) != "1") {
        texture = interpolate_texture_page(mesh, texture, page_support);
    }
    save_outputs(input, mesh, texture);
}

}  // namespace metashape_texture
