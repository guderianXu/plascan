#include "metalign/matching.hpp"
#include "metalign/gpu.hpp"
#include "metalign/math.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

#if defined(METALIGN_HAS_OPENMP)
#include <omp.h>
#endif

namespace metalign {
namespace {

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
__attribute__((target("popcnt"), noinline)) std::uint32_t
hctree_hamming_popcnt(const Descriptor& left, const Descriptor& right) {
    std::uint32_t distance = 0;
#pragma GCC unroll 8
    for (std::size_t lane = 0; lane < 8; ++lane) {
        std::uint64_t left_word = 0;
        std::uint64_t right_word = 0;
        std::memcpy(&left_word, left.data() + lane * sizeof(left_word), sizeof(left_word));
        std::memcpy(&right_word, right.data() + lane * sizeof(right_word), sizeof(right_word));
        distance += static_cast<std::uint32_t>(
            __builtin_popcountll(left_word ^ right_word));
    }
    return distance;
}
#endif

class DescriptorHCTree {
public:
    // sub_26F3950 writes this exact five-dword result record: two signed row
    // indices, two unsigned Hamming distances, then the examined-row count.
    // Keeping the target ABI avoids widening every hot-query result to two
    // double/size_t pairs in the GOMP output buffer.
    struct NearestTwoResult {
        std::int32_t best_index = -1;
        std::int32_t second_index = -1;
        std::uint32_t best_distance = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t second_distance = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t examined = 0;
    };

    DescriptorHCTree(const FeatureSet& features,
                     const std::vector<std::size_t>& rows) {
        // The target build/query ABI passes a standalone descriptor matrix
        // with stride=64.  Reading descriptors through the replica's much
        // larger Keypoint records doubles the candidate-stream footprint in
        // the 1,000-check hot loop. Preserve row identity while materializing
        // the recovered compact matrix layout.
        descriptors_.reserve(rows.size());
        for (const std::size_t row : rows)
            descriptors_.push_back(features.keypoints[row].descriptor);
        static std::atomic<std::size_t> next_debug_build{0};
        debug_build_index_ = next_debug_build.fetch_add(1, std::memory_order_relaxed);
        // 0x26F2350 first appends all eight root records.  It then fills one
        // independent index slice per tree and recursively splits that root.
        // Keeping this layout matters to exact node ids observed by 0x26F3950
        // (the first root's children begin at node 8, not node 1).
        indices_.resize(rows.size() * kTreeCount);
        roots_.reserve(kTreeCount);
        for (int tree = 0; tree < kTreeCount; ++tree)
            roots_.push_back(make_node(0, rows.size() * static_cast<std::size_t>(tree),
                                       rows.size()));
        for (int tree = 0; tree < kTreeCount; ++tree) {
            const std::size_t begin = rows.size() * static_cast<std::size_t>(tree);
            std::iota(indices_.begin() + static_cast<std::ptrdiff_t>(begin),
                      indices_.begin() + static_cast<std::ptrdiff_t>(begin + rows.size()),
                      std::uint32_t{0});
            split(roots_[static_cast<std::size_t>(tree)]);
            if (tree == 0 && std::getenv("METALIGN_TRACE_HCTREE_ROOT") != nullptr) {
                const Node& root = nodes_[roots_.front()];
                std::cerr << "HCTREE_ROOT rows=" << rows.size()
                          << " first_child=" << root.first_child
                          << " child_count=" << root.child_count << '\n';
                for (std::size_t child = 0; child < root.child_count; ++child) {
                    const Node& node = nodes_[root.first_child + child];
                    std::cerr << "ROOT_CHILD_" << child
                              << " pivot=" << node.center
                              << " begin=" << node.begin
                              << " size=" << node.count
                              << " child=" << node.first_child
                              << " nchild=" << node.child_count << '\n';
                }
            }
        }
        dump_root_centers();
        if (const char* dump_prefix = std::getenv("METALIGN_DUMP_HCTREE_TREE")) {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                std::ofstream index_stream(std::string(dump_prefix) + ".indices.bin",
                                           std::ios::binary);
                index_stream.write(reinterpret_cast<const char*>(indices_.data()),
                                   static_cast<std::streamsize>(indices_.size() *
                                                                sizeof(indices_.front())));
                std::ofstream node_stream(std::string(dump_prefix) + ".nodes.bin",
                                          std::ios::binary);
                node_stream.write(reinterpret_cast<const char*>(nodes_.data()),
                                  static_cast<std::streamsize>(nodes_.size() *
                                                               sizeof(nodes_.front())));
            }
        }
    }

    NearestTwoResult nearest_two(const Descriptor& descriptor) const {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
        // sub_26F3950 checks CPU feature bit 0x40 once per query and dispatches
        // to sub_26FD3D0. Its stride-64 specialization uses eight hardware
        // POPCNT instructions per descriptor. Keep the scalar SWAR replay as
        // the fallback for hosts without that feature.
        if (__builtin_cpu_supports("popcnt"))
            return nearest_two_impl<true>(descriptor);
#endif
        return nearest_two_impl<false>(descriptor);
    }

private:
    template <bool UsePopcnt>
    NearestTwoResult nearest_two_impl(const Descriptor& descriptor) const {
        NearestTwoResult result;
        struct Branch {
            std::uint32_t distance = 0;
            std::size_t node = 0;
        };
        // 0x26F3950 uses a hand-written, 1-based binary min-heap.  Its tie
        // rules are observable when the 1000-check budget cuts through many
        // equal Hamming-distance branches: push only rises on strict '<'; pop
        // chooses the left child on equal keys and sinks the last entry on
        // equality.  std::priority_queue is not equivalent for those ties.
        // 0x26F39BD increments a generation field at search-context +0x18.
        // Reuse the target-style visited set and priority-queue storage between
        // queries instead of allocating/clearing target_count bytes each time.
        thread_local std::vector<Branch> queue;
        thread_local std::vector<std::uint32_t> visited;
        thread_local std::uint32_t visited_generation = 0;
        // Both serial 0x26F53F0 and GOMP worker 0x26F5670 begin each search
        // context with malloc(0x2000): 512 sixteen-byte heap records. A six-
        // photo target trace hit neither of the two growth reallocations.
        if (queue.capacity() < 512) queue.reserve(512);
        queue.clear();
        queue.push_back({});  // preserve the target's 1-based heap layout
        if (visited.size() < descriptors_.size()) visited.resize(descriptors_.size(), 0);
        ++visited_generation;
        if (visited_generation == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            visited_generation = 1;
        }
        auto queue_push = [&](Branch branch) {
            queue.push_back(branch);
            std::size_t hole = queue.size() - 1;
            while (hole > 1) {
                const std::size_t parent = hole / 2;
                if (branch.distance >= queue[parent].distance) break;
                queue[hole] = queue[parent];
                hole = parent;
            }
            queue[hole] = branch;
        };
        auto queue_pop = [&]() {
            const Branch result = queue[1];
            const Branch last = queue.back();
            queue.pop_back();
            const std::size_t size = queue.size() - 1;
            if (size != 0) {
                std::size_t hole = 1;
                std::size_t child = 2;
                while (child <= size) {
                    if (child < size &&
                        queue[child + 1].distance < queue[child].distance)
                        ++child;
                    if (last.distance < queue[child].distance) break;
                    queue[hole] = queue[child];
                    hole = child;
                    child *= 2;
                }
                queue[hole] = last;
            }
            return result;
        };
        std::size_t examined = 0;
        auto descend = [&](std::size_t node_index) {
            while (nodes_[node_index].child_count != 0) {
                const Node& node = nodes_[node_index];
                std::size_t nearest = node.first_child;
                std::array<std::uint32_t, kBranching> distances{};
                std::uint32_t nearest_distance = std::numeric_limits<std::uint32_t>::max();
                std::size_t nearest_offset = 0;
                for (std::size_t offset = 0; offset < node.child_count; ++offset) {
                    const std::size_t child = node.first_child + offset;
                    const std::uint32_t distance = query_hamming<UsePopcnt>(
                        descriptor, descriptors_[nodes_[child].center]);
                    distances[offset] = distance;
                    if (distance < nearest_distance) {
                        nearest_distance = distance;
                        nearest = child;
                        nearest_offset = offset;
                    }
                }
                // 0x26F3950 computes all child distances first, then inserts
                // every non-winning child in ascending child-index order.
                // Inserting a displaced provisional winner immediately gives
                // a different heap shape for equal Hamming keys.
                for (std::size_t offset = 0; offset < node.child_count; ++offset) {
                    if (offset == nearest_offset) continue;
                    queue_push({distances[offset], node.first_child + offset});
                }
                node_index = nearest;
            }
            const Node& leaf = nodes_[node_index];
            for (std::size_t offset = 0; offset < leaf.count; ++offset) {
                const std::size_t point = indices_[leaf.begin + offset];
                if (visited[point] == visited_generation) continue;
                visited[point] = visited_generation;
                ++examined;
                update_best(result, query_hamming<UsePopcnt>(
                    descriptor, descriptors_[point]), point);
            }
        };
        for (std::size_t root : roots_) {
            if (examined >= kMaxChecks) break;
            descend(root);
        }
        while (examined < kMaxChecks && queue.size() > 1) {
            const std::size_t node = queue_pop().node;
            descend(node);
        }
        result.examined = static_cast<std::uint32_t>(examined);
        return result;
    }

    struct Node {
        std::size_t center = 0;
        std::size_t begin = 0;
        std::size_t count = 0;
        std::size_t first_child = 0;
        std::size_t child_count = 0;
    };

    static constexpr int kBranching = 4;
    static constexpr int kTreeCount = 8;
    static constexpr std::size_t kLeafSize = 200;
    static constexpr std::size_t kMaxChecks = 1000;

    std::size_t make_node(std::size_t center, std::size_t begin, std::size_t count) {
        nodes_.push_back(Node{center, begin, count, 0, 0});
        return nodes_.size() - 1;
    }

    bool choose_centers(const Node& node, std::array<std::size_t, kBranching>& centers) const {
        for (int center = 0; center < kBranching; ++center) {
            bool accepted = false;
            for (int attempt = 0; attempt < 100 && !accepted; ++attempt) {
                const std::size_t sampled = static_cast<std::size_t>(
                    static_cast<double>(std::rand()) * 4.656612873077393e-10 *
                    static_cast<double>(node.count));
                centers[static_cast<std::size_t>(center)] =
                    indices_[node.begin + std::min(sampled, node.count - 1)];
                accepted = true;
                for (int previous = 0; previous < center; ++previous) {
                    if (descriptors_[centers[static_cast<std::size_t>(center)]] ==
                        descriptors_[centers[static_cast<std::size_t>(previous)]]) {
                        accepted = false;
                        break;
                    }
                }
            }
            if (!accepted) return false;
        }
        return true;
    }

    void split(std::size_t node_index) {
        const Node snapshot = nodes_[node_index];
        if (snapshot.count < kLeafSize || snapshot.count == 0) return;
        std::array<std::size_t, kBranching> centers{};
        if (!choose_centers(snapshot, centers)) return;
        if (snapshot.count == descriptors_.size() && !descriptors_.empty() &&
            snapshot.begin % descriptors_.size() == 0 &&
            snapshot.begin / descriptors_.size() < static_cast<std::size_t>(kTreeCount)) {
            debug_root_centers_.push_back(centers);
        }
        std::vector<unsigned> assignment(snapshot.count, 0);
        for (std::size_t offset = 0; offset < snapshot.count; ++offset) {
            const std::size_t point = indices_[snapshot.begin + offset];
            unsigned best_center = 0;
            std::uint32_t best_distance = descriptor_hamming_distance(
                descriptors_[point], descriptors_[centers[0]]);
            for (unsigned center = 1; center < kBranching; ++center) {
                const std::uint32_t distance = descriptor_hamming_distance(
                    descriptors_[point], descriptors_[centers[center]]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_center = center;
                }
            }
            assignment[offset] = best_center;
        }
        const std::size_t first_child = nodes_.size();
        nodes_[node_index].first_child = first_child;
        nodes_[node_index].child_count = kBranching;
        for (unsigned center = 0; center < kBranching; ++center)
            make_node(centers[center], 0, 0);
        // 0x26F2350/0x26F4B70 do FLANN's original in-place label
        // partition.  It is deliberately not a stable gather: every accepted
        // item is swapped with the current partition end, and its label is
        // swapped at the same time.  The resulting descriptor order affects
        // all recursively sampled centres and therefore the approximate-NN
        // result set.
        std::size_t partition_begin = 0;
        for (unsigned center = 0; center < kBranching; ++center) {
            std::size_t partition_end = partition_begin;
            for (std::size_t offset = 0; offset < snapshot.count; ++offset) {
                if (assignment[offset] != center) continue;
                std::swap(indices_[snapshot.begin + offset],
                          indices_[snapshot.begin + partition_end]);
                std::swap(assignment[offset], assignment[partition_end]);
                ++partition_end;
            }
            Node& child = nodes_[first_child + center];
            child.begin = snapshot.begin + partition_begin;
            child.count = partition_end - partition_begin;
            // 0x26F4B70 allocates all four child records first, but recurses
            // into each child immediately after partitioning that child and
            // before partitioning the next sibling.  This order is observable:
            // center selection uses the process-global rand() stream.
            split(first_child + center);
            partition_begin = partition_end;
        }
    }

    static void update_best(NearestTwoResult& result,
                            std::uint32_t distance, std::size_t index) {
        const std::int32_t row = static_cast<std::int32_t>(index);
        if (distance < result.best_distance) {
            result.second_index = result.best_index;
            result.second_distance = result.best_distance;
            result.best_index = row;
            result.best_distance = distance;
        } else if (distance < result.second_distance && row != result.best_index) {
            result.second_index = row;
            result.second_distance = distance;
        }
    }

    template <bool UsePopcnt>
    static inline std::uint32_t query_hamming(
        const Descriptor& left, const Descriptor& right) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
        if constexpr (UsePopcnt) return hctree_hamming_popcnt(left, right);
#endif
        return descriptor_hamming_distance(left, right);
    }

    void dump_root_centers() const {
        const char* directory = std::getenv("METALIGN_DUMP_HCTREE_ROOT_CENTERS");
        if (directory == nullptr || *directory == '\0') return;
        const std::filesystem::path root(directory);
        std::filesystem::create_directories(root);
        const std::filesystem::path path = root /
            ("build_" + std::to_string(debug_build_index_) + "_rows_" +
             std::to_string(descriptors_.size()) + ".centers32.bin");
        std::ofstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("cannot write HCTree root centers: " + path.string());
        const std::array<char, 8> magic{'H', 'C', 'R', 'O', 'O', 'T', '1', '\0'};
        const std::uint32_t build = static_cast<std::uint32_t>(debug_build_index_);
        const std::uint32_t rows = static_cast<std::uint32_t>(descriptors_.size());
        const std::uint32_t roots = static_cast<std::uint32_t>(debug_root_centers_.size());
        const std::uint32_t branches = kBranching;
        stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        stream.write(reinterpret_cast<const char*>(&build), sizeof(build));
        stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        stream.write(reinterpret_cast<const char*>(&roots), sizeof(roots));
        stream.write(reinterpret_cast<const char*>(&branches), sizeof(branches));
        for (const auto& centers : debug_root_centers_) {
            for (std::size_t center : centers) {
                const std::uint32_t value = static_cast<std::uint32_t>(center);
                stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
        if (!stream)
            throw std::runtime_error("failed to write HCTree root centers: " + path.string());
    }

    std::vector<Descriptor> descriptors_;
    // 0x26F3950 loads leaf permutation entries with
    // `movsxd ..., dword ptr [indices + 4*offset]`. Node fields remain qwords,
    // but the eight tree permutations are a compact signed-int32 stream.
    std::vector<std::uint32_t> indices_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> roots_;
    std::size_t debug_build_index_ = 0;
    std::vector<std::array<std::size_t, kBranching>> debug_root_centers_;
};

std::string basename(const FeatureSet& feature) { return feature.path.filename().string(); }

bool target_gomp_hctree_enabled() {
    const char* value = std::getenv("METALIGN_TARGET_GOMP_HCTREE");
    return value == nullptr || (*value != '\0' && std::string_view(value) != "0");
}

bool target_pair_local_hctree_enabled() {
    const char* value = std::getenv("METALIGN_TARGET_PAIR_LOCAL_HCTREE");
    return value != nullptr && *value != '\0' && std::string_view(value) != "0";
}

bool target_focal_hctree_enabled() {
    if (target_pair_local_hctree_enabled()) return false;
    const char* value = std::getenv("METALIGN_TARGET_FOCAL_HCTREE");
    return value == nullptr || (*value != '\0' && std::string_view(value) != "0");
}

bool local_decision_dump_enabled_for_pair(const ImagePair* pair) {
    const char* directory = std::getenv("METALIGN_DUMP_LOCAL_DECISIONS");
    if (directory == nullptr || *directory == '\0') return false;
    const char* requested = std::getenv("METALIGN_DUMP_LOCAL_DECISIONS_PAIRS");
    if (requested == nullptr || *requested == '\0') return true;
    if (pair == nullptr) return false;

    // Analysis-only pair selector.  It deliberately accepts a semicolon or
    // whitespace separated list of FIRST,SECOND records, so a full matching
    // run can expose one target-probed local-consistency worker without
    // materializing hundreds of unrelated decision tables.
    std::istringstream stream(requested);
    while (stream) {
        std::size_t first = 0;
        std::size_t second = 0;
        char comma = '\0';
        if (!(stream >> first >> comma >> second) || comma != ',') break;
        if (pair->ordered() == ImagePair{first, second}.ordered()) return true;
        stream >> std::ws;
        if (stream.peek() == ';') stream.get();
    }
    return false;
}

std::optional<std::size_t> find_image_index(const std::vector<FeatureSet>& features,
                                            const std::string& name) {
    for (std::size_t i = 0; i < features.size(); ++i) {
        if (features[i].path.filename() == name || features[i].path.string() == name)
            return i;
    }
    return std::nullopt;
}

double reference_distance(const ReferencePosition& left, const ReferencePosition& right) {
    return std::hypot(std::hypot(left.x - right.x, left.y - right.y), left.z - right.z);
}

void consolidate_orientation_matches(std::vector<FeatureMatch>& matches,
                                     const FeatureSet& first, const FeatureSet& second,
                                     std::size_t limit) {
    // 0x13A6900 collapses direction variants by detector-point identity, but
    // its emitted records are ordered by the canonical direction-row source
    // ids.  Both identities are observable and are therefore kept separate.
    std::map<std::pair<std::size_t, std::size_t>, FeatureMatch> unique;
    for (const FeatureMatch& match : matches) {
        // 0x13A6900 consolidates the direction-expanded descriptor matches by
        // their underlying detector-point pair.  Track construction later
        // consumes the distinct direction-row source ids, so these identities
        // must not be conflated.
        const auto key = std::make_pair(first.keypoints[match.first].detector_id,
                                        second.keypoints[match.second].detector_id);
        // sub_1D6C080 maps every surviving direction row to the first row of
        // its contiguous orientation block.  sub_13A6900 stores and orders
        // that canonical source-row pair, not whichever orientation variant
        // happened to win the HCTree query.  Keeping `match` here made the
        // physical x/y pairs exact but split one detector observation into
        // different temporary tracks across camera pairs.
        unique.emplace(key, FeatureMatch{
            first.keypoints[match.first].source_id,
            second.keypoints[match.second].source_id,
            match.distance});
    }
    std::vector<FeatureMatch> consolidated;
    consolidated.reserve(unique.size());
    for (const auto& [key, match] : unique) {
        static_cast<void>(key);
        consolidated.push_back(match);
    }
    std::sort(consolidated.begin(), consolidated.end(),
              [&](const FeatureMatch& left, const FeatureMatch& right) {
                  return std::make_pair(first.keypoints[left.first].source_id,
                                        second.keypoints[left.second].source_id) <
                         std::make_pair(first.keypoints[right.first].source_id,
                                        second.keypoints[right.second].source_id);
              });
    if (limit != 0 && consolidated.size() > limit) consolidated.resize(limit);
    matches = std::move(consolidated);
}

struct Neighbor {
    float squared_distance = 0.0F;
    std::size_t match = 0;
};

class SpatialNeighborIndex {
public:
    explicit SpatialNeighborIndex(const std::vector<Vec2>& points) : points_(points) {
        indices_.resize(points.size());
        std::iota(indices_.begin(), indices_.end(), std::size_t{0});
        if (!indices_.empty()) {
            root_ = make_node(0, indices_.size());
            build();
        }
    }

    std::vector<Neighbor> nearest(std::size_t query, std::size_t count) const {
        std::vector<Neighbor> result;
        result.reserve(std::min(count, points_.size()));
        if (root_ == invalid_node || count == 0) return result;
        struct Branch {
            float squared_distance = 0.0F;
            std::size_t node = 0;
        };
        struct BranchCompare {
            bool operator()(const Branch& left, const Branch& right) const {
                if (left.squared_distance != right.squared_distance)
                    return left.squared_distance > right.squared_distance;
                // sub_13A1680's float/index heap comparison gives the larger
                // node id priority when two queued boxes have equal distance.
                return left.node < right.node;
            }
        };
        std::priority_queue<Branch, std::vector<Branch>, BranchCompare> pending;
        pending.push({0.0F, root_});
        const float query_x = static_cast<float>(points_[query].x);
        const float query_y = static_cast<float>(points_[query].y);
        while (!pending.empty()) {
            Branch branch = pending.top();
            pending.pop();
            if (result.size() == count &&
                branch.squared_distance >= result.back().squared_distance)
                break;
            std::size_t node_index = branch.node;
            while (nodes_[node_index].left != invalid_node) {
                const Node& node = nodes_[node_index];
                const float left_distance = box_distance(
                    nodes_[node.left], query_x, query_y);
                const float right_distance = box_distance(
                    nodes_[node.right], query_x, query_y);
                std::size_t near_node = node.right;
                std::size_t far_node = node.left;
                float far_distance = left_distance;
                if (left_distance < right_distance) {
                    near_node = node.left;
                    far_node = node.right;
                    far_distance = right_distance;
                }
                if (result.size() != count ||
                    far_distance < result.back().squared_distance)
                    pending.push({far_distance, far_node});
                node_index = near_node;
            }
            const Node& leaf = nodes_[node_index];
            const float leaf_distance = box_distance(leaf, query_x, query_y);
            if (result.size() == count &&
                leaf_distance >= result.back().squared_distance)
                continue;
            for (std::size_t offset = leaf.begin; offset < leaf.end; ++offset)
                insert_candidate(indices_[offset], count, query_x, query_y, result);
        }
        return result;
    }

private:
    struct Node {
        float minimum_x = 0.0F;
        float minimum_y = 0.0F;
        float maximum_x = 0.0F;
        float maximum_y = 0.0F;
        std::size_t begin = 0;
        std::size_t end = 0;
        std::size_t left = invalid_node;
        std::size_t right = invalid_node;
    };

    static constexpr std::size_t invalid_node =
        std::numeric_limits<std::size_t>::max();
    // 0x1394282/0x1394287 pass {64, 16} to 0x13A0E50. In the System V ABI
    // these arrive as max_depth=64 and leaf_size=16. The builder partitions
    // at the midpoint of the widest bounding-box axis; it is not a median
    // nth_element tree.
    static constexpr std::size_t leaf_size = 16;
    static constexpr std::size_t max_depth = 64;

    static bool neighbor_less(const Neighbor& left, const Neighbor& right) {
        if (left.squared_distance != right.squared_distance)
            return left.squared_distance < right.squared_distance;
        return left.match < right.match;
    }

    float coordinate(std::size_t index, std::size_t axis) const {
        return axis == 0 ? static_cast<float>(points_[index].x)
                         : static_cast<float>(points_[index].y);
    }

    std::size_t make_node(std::size_t begin, std::size_t end) {
        Node node;
        node.begin = begin;
        node.end = end;
        node.minimum_x = node.minimum_y = std::numeric_limits<float>::max();
        node.maximum_x = node.maximum_y = std::numeric_limits<float>::lowest();
        for (std::size_t offset = begin; offset < end; ++offset) {
            const std::size_t index = indices_[offset];
            const float x = static_cast<float>(points_[index].x);
            const float y = static_cast<float>(points_[index].y);
            node.minimum_x = std::min(node.minimum_x, x);
            node.minimum_y = std::min(node.minimum_y, y);
            node.maximum_x = std::max(node.maximum_x, x);
            node.maximum_y = std::max(node.maximum_y, y);
        }
        const std::size_t node_index = nodes_.size();
        nodes_.push_back(node);
        return node_index;
    }

    void build() {
        struct PendingNode {
            std::size_t node = 0;
            std::size_t depth = 0;
        };
        std::vector<PendingNode> pending{{root_, 0}};
        while (!pending.empty()) {
            const PendingNode work = pending.back();
            pending.pop_back();
            const Node snapshot = nodes_[work.node];
            if (work.depth >= max_depth || snapshot.end - snapshot.begin <= leaf_size)
                continue;
            const float width = snapshot.maximum_x - snapshot.minimum_x;
            const float height = snapshot.maximum_y - snapshot.minimum_y;
            const std::size_t axis = height > width ? 1 : 0;
            const float split = (axis == 0
                ? snapshot.maximum_x + snapshot.minimum_x
                : snapshot.maximum_y + snapshot.minimum_y) * 0.5F;

            std::size_t left_cursor = snapshot.begin;
            std::size_t right_cursor = snapshot.end - 1;
            while (left_cursor < right_cursor) {
                while (left_cursor < snapshot.end &&
                       coordinate(indices_[left_cursor], axis) < split)
                    ++left_cursor;
                while (right_cursor >= snapshot.begin &&
                       coordinate(indices_[right_cursor], axis) >= split) {
                    if (right_cursor == 0) break;
                    --right_cursor;
                }
                if (right_cursor < left_cursor) break;
                std::swap(indices_[left_cursor], indices_[right_cursor]);
                ++left_cursor;
                if (right_cursor == 0) break;
                --right_cursor;
            }
            if (left_cursor < snapshot.end &&
                coordinate(indices_[left_cursor], axis) < split)
                ++left_cursor;
            if (left_cursor <= snapshot.begin || left_cursor >= snapshot.end) continue;

            const std::size_t left = make_node(snapshot.begin, left_cursor);
            const std::size_t right = make_node(left_cursor, snapshot.end);
            nodes_[work.node].left = left;
            nodes_[work.node].right = right;
            if (work.depth + 1 < max_depth) {
                if (nodes_[left].end - nodes_[left].begin > leaf_size)
                    pending.push_back({left, work.depth + 1});
                if (nodes_[right].end - nodes_[right].begin > leaf_size)
                    pending.push_back({right, work.depth + 1});
            }
        }
    }

    float box_distance(const Node& node, float x, float y) const {
        float dx = 0.0F;
        float dy = 0.0F;
        if (x < node.minimum_x) dx = node.minimum_x - x;
        else if (x > node.maximum_x) dx = x - node.maximum_x;
        if (y < node.minimum_y) dy = node.minimum_y - y;
        else if (y > node.maximum_y) dy = y - node.maximum_y;
        return dx * dx + dy * dy;
    }

    void insert_candidate(std::size_t index, std::size_t count,
                          float query_x, float query_y,
                          std::vector<Neighbor>& result) const {
        const float dx = static_cast<float>(points_[index].x) - query_x;
        const float dy = static_cast<float>(points_[index].y) - query_y;
        const Neighbor candidate{dx * dx + dy * dy, index};
        // sub_13A22F0 admits a candidate only when its distance is strictly
        // smaller than the current max-heap root. Equal-distance candidates
        // encountered after the heap reaches k entries never replace an
        // earlier point merely because their integer index is smaller.
        if (result.size() == count &&
            candidate.squared_distance >= result.back().squared_distance)
            return;
        const auto position = std::lower_bound(
            result.begin(), result.end(), candidate, neighbor_less);
        result.insert(position, candidate);
        if (result.size() > count) result.pop_back();
    }

    const std::vector<Vec2>& points_;
    std::vector<std::size_t> indices_;
    std::vector<Node> nodes_;
    std::size_t root_ = invalid_node;
};

float median_fifth_neighbor_distance(const SpatialNeighborIndex& index,
                                     const std::vector<Vec2>& points) {
    std::vector<float> distances;
    distances.reserve(points.size());
    for (std::size_t query = 0; query < points.size(); ++query) {
        const std::vector<Neighbor> nearest = index.nearest(query, 5);
        distances.push_back(nearest.front().squared_distance);
        for (const Neighbor& neighbor : nearest)
            distances.back() = std::max(distances.back(), neighbor.squared_distance);
    }
    const auto middle = distances.begin() +
        static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), middle, distances.end());
    return *middle;
}

std::vector<std::size_t> accepted_neighbors(const SpatialNeighborIndex& index,
                                            std::size_t query,
                                            float local_threshold,
                                            float global_threshold) {
    const std::vector<Neighbor> nearest = index.nearest(query, 10);
    std::vector<std::size_t> accepted;
    accepted.reserve(nearest.size());
    for (std::size_t rank = 0; rank < nearest.size(); ++rank) {
        const float threshold = rank < 5 ? local_threshold : global_threshold;
        if (nearest[rank].squared_distance <= threshold)
            accepted.push_back(nearest[rank].match);
    }
    std::sort(accepted.begin(), accepted.end());
    accepted.erase(std::unique(accepted.begin(), accepted.end()), accepted.end());
    return accepted;
}

}  // namespace

struct CpuDescriptorIndex::Impl {
    struct Group {
        std::vector<std::size_t> original_indices;
        std::unique_ptr<DescriptorHCTree> tree;
    };

    std::array<Group, 2> groups;
};

CpuDescriptorIndex::CpuDescriptorIndex(const FeatureSet& features)
    : impl_(std::make_unique<Impl>()) {
    // 0x26F53C0 is invoked exactly twice per indexed image, in -LoG then +LoG
    // order. The target schedules images lazily between directed query batches;
    // construction itself consumes rand(), whereas querying does not, so this
    // caller's eager lifetime does not change the controlled tree sequence.
    for (std::size_t group_index = 0; group_index < impl_->groups.size(); ++group_index) {
        const int sign = group_index == 0 ? -1 : 1;
        Impl::Group& group = impl_->groups[group_index];
        for (std::size_t index = 0; index < features.keypoints.size(); ++index) {
            if (features.keypoints[index].laplacian_sign != sign) continue;
            group.original_indices.push_back(index);
        }
        if (group.original_indices.size() >= 2)
            group.tree = std::make_unique<DescriptorHCTree>(
                features, group.original_indices);
    }
}

CpuDescriptorIndex::~CpuDescriptorIndex() = default;
CpuDescriptorIndex::CpuDescriptorIndex(CpuDescriptorIndex&&) noexcept = default;
CpuDescriptorIndex& CpuDescriptorIndex::operator=(CpuDescriptorIndex&&) noexcept = default;

namespace {

struct CoarsePairMatches {
    ImagePair pair;
    std::vector<FeatureMatch> matches;
    std::size_t directional_count = 0;
    std::size_t combined_count = 0;
    std::size_t filtered_count = 0;
};

std::vector<CoarsePairMatches> read_coarse_correspondence_fixture(
    const std::filesystem::path& path,
    const std::vector<FeatureSet>& coarse) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error(
        "cannot open coarse-correspondence fixture: " + path.string());
    const auto read_exact = [&](void* data, std::size_t size) {
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        if (!stream) throw std::runtime_error(
            "truncated coarse-correspondence fixture: " + path.string());
    };
    std::array<char, 8> magic{};
    std::uint64_t pair_count = 0;
    std::uint64_t total_matches = 0;
    std::uint64_t chunk_count = 0;
    read_exact(magic.data(), magic.size());
    read_exact(&pair_count, sizeof(pair_count));
    read_exact(&total_matches, sizeof(total_matches));
    read_exact(&chunk_count, sizeof(chunk_count));
    const std::array<char, 8> expected{'M', 'S', 'C', 'R', 'S', '1', '\0', '\0'};
    if (magic != expected || pair_count > 1'000'000 || total_matches > 1'000'000'000)
        throw std::runtime_error("invalid coarse-correspondence fixture header");

    std::vector<CoarsePairMatches> result;
    result.reserve(static_cast<std::size_t>(pair_count));
    std::uint64_t observed_matches = 0;
    for (std::uint64_t pair_index = 0; pair_index < pair_count; ++pair_index) {
        std::uint32_t first = 0;
        std::uint32_t second = 0;
        std::uint64_t count = 0;
        read_exact(&first, sizeof(first));
        read_exact(&second, sizeof(second));
        read_exact(&count, sizeof(count));
        if (first >= coarse.size() || second >= coarse.size() || first == second ||
            count > 1'000'000)
            throw std::runtime_error("invalid pair in coarse-correspondence fixture");
        CoarsePairMatches pair;
        pair.pair = ImagePair{first, second}.ordered();
        pair.matches.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t match_index = 0; match_index < count; ++match_index) {
            std::int32_t left = -1;
            std::int32_t right = -1;
            read_exact(&left, sizeof(left));
            read_exact(&right, sizeof(right));
            if (left < 0 || right < 0 ||
                static_cast<std::size_t>(left) >= coarse[first].keypoints.size() ||
                static_cast<std::size_t>(right) >= coarse[second].keypoints.size())
                throw std::runtime_error(
                    "invalid feature index in coarse-correspondence fixture");
            pair.matches.push_back({static_cast<std::size_t>(left),
                                    static_cast<std::size_t>(right), 0.0});
        }
        pair.combined_count = pair.matches.size();
        result.push_back(std::move(pair));
        observed_matches += count;
    }
    if (observed_matches != total_matches || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("coarse-correspondence fixture size mismatch");
    std::cout << "  replayed target coarse fixture: " << result.size()
              << " pairs, " << observed_matches << " correspondences, "
              << chunk_count << " source chunks\n";
    return result;
}

void write_coarse_correspondence_fixture(
    const std::filesystem::path& path,
    const std::vector<CoarsePairMatches>& pairs) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot write coarse-correspondence dump: " +
                                 path.string());
    const std::array<char, 8> magic{'M', 'S', 'C', 'R', 'S', '1', '\0', '\0'};
    const std::uint64_t pair_count = pairs.size();
    const std::uint64_t total_matches = std::accumulate(
        pairs.begin(), pairs.end(), std::uint64_t{0},
        [](std::uint64_t total, const CoarsePairMatches& pair) {
            return total + static_cast<std::uint64_t>(pair.matches.size());
        });
    // Native generic preselection stores all evaluated pairs in one chunk at
    // the recovered sub_138E5F0 boundary.
    const std::uint64_t chunk_count = 1;
    stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    stream.write(reinterpret_cast<const char*>(&pair_count), sizeof(pair_count));
    stream.write(reinterpret_cast<const char*>(&total_matches), sizeof(total_matches));
    stream.write(reinterpret_cast<const char*>(&chunk_count), sizeof(chunk_count));
    for (const CoarsePairMatches& pair : pairs) {
        const std::uint32_t first = static_cast<std::uint32_t>(pair.pair.first);
        const std::uint32_t second = static_cast<std::uint32_t>(pair.pair.second);
        const std::uint64_t count = pair.matches.size();
        stream.write(reinterpret_cast<const char*>(&first), sizeof(first));
        stream.write(reinterpret_cast<const char*>(&second), sizeof(second));
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const FeatureMatch& match : pair.matches) {
            const std::int32_t left = static_cast<std::int32_t>(match.first);
            const std::int32_t right = static_cast<std::int32_t>(match.second);
            stream.write(reinterpret_cast<const char*>(&left), sizeof(left));
            stream.write(reinterpret_cast<const char*>(&right), sizeof(right));
        }
    }
    if (!stream)
        throw std::runtime_error("failed to write coarse-correspondence dump: " +
                                 path.string());
    std::cout << "  dumped coarse correspondences: " << pair_count << " pairs, "
              << total_matches << " rows to " << path << '\n';
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }

    std::size_t find(std::size_t value) {
        std::size_t root = value;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[value] != value) {
            const std::size_t next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void unite(std::size_t left, std::size_t right) {
        left = find(left);
        right = find(right);
        if (left == right) return;
        if (rank_[left] < rank_[right]) std::swap(left, right);
        parent_[right] = left;
        if (rank_[left] == rank_[right]) ++rank_[left];
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned char> rank_;
};

std::set<ImagePair> recovered_generic_preselection(
    const std::vector<FeatureSet>& features,
    const MatchPhotosOptions& options,
    std::size_t thread_count,
    DescriptorAccelerator* accelerator,
    const std::vector<ImagePair>* candidate_override = nullptr,
    const std::vector<const CpuDescriptorIndex*>* index_override = nullptr,
    bool inputs_are_coarse = false) {
    const std::size_t camera_count = features.size();
    std::vector<FeatureSet> coarse_storage;
    if (!inputs_are_coarse) coarse_storage.resize(camera_count);
    const std::vector<FeatureSet>& coarse = inputs_are_coarse ? features : coarse_storage;
    std::vector<std::size_t> offsets(camera_count + 1, 0);
    for (std::size_t camera = 0; camera < camera_count; ++camera) {
        if (!inputs_are_coarse) {
            coarse_storage[camera].path = features[camera].path;
            coarse_storage[camera].image_width = features[camera].image_width;
            coarse_storage[camera].image_height = features[camera].image_height;
            coarse_storage[camera].focal_length_pixels = features[camera].focal_length_pixels;
            coarse_storage[camera].keypoints = features[camera].coarse_keypoints;
            coarse_storage[camera].source_keypoint_count = coarse_storage[camera].keypoints.size();
        }
        offsets[camera + 1] = offsets[camera] + coarse[camera].keypoints.size();
    }

    // At sub_1388FB0 the generic-only (--no-reference-preselection) South
    // Building run receives all C(128,2)=8,128 unique pairs in lexicographic
    // order.  With reference preselection enabled, the captured input is the
    // exact ring graph i<->i+/-[1,24]: 3,072 edges and degree 48.  The earlier
    // unconditional ring implementation therefore silently changed the
    // public no-reference option into a reference-window search.
    std::set<ImagePair> candidate_set;
    if (candidate_override) {
        for (const ImagePair pair : *candidate_override) {
            const ImagePair ordered = pair.ordered();
            if (ordered.first < camera_count && ordered.second < camera_count &&
                ordered.first != ordered.second)
                candidate_set.insert(ordered);
        }
    } else if (!options.reference_preselection) {
        for (std::size_t first = 0; first < camera_count; ++first)
            for (std::size_t second = first + 1; second < camera_count; ++second)
                candidate_set.insert({first, second});
    } else {
        for (std::size_t camera = 0; camera < camera_count; ++camera) {
            for (std::size_t delta = 1; delta <= 24 && delta < camera_count; ++delta) {
                const std::size_t other = (camera + delta) % camera_count;
                if (camera != other)
                    candidate_set.insert(ImagePair{camera, other}.ordered());
            }
        }
    }
    std::vector<ImagePair> candidates(candidate_set.begin(), candidate_set.end());

    std::vector<std::unique_ptr<CpuDescriptorIndex>> indices;
    std::vector<const CpuDescriptorIndex*> index_pointers;
    if (!accelerator && !target_pair_local_hctree_enabled() &&
        !target_focal_hctree_enabled()) {
        if (index_override && index_override->size() == camera_count) {
            index_pointers = *index_override;
        } else {
            indices.reserve(camera_count);
            index_pointers.reserve(camera_count);
            for (const FeatureSet& feature : coarse) {
                indices.push_back(std::make_unique<CpuDescriptorIndex>(feature));
                index_pointers.push_back(indices.back().get());
            }
        }
    }

    std::vector<CoarsePairMatches> evaluated(candidates.size());
    if (accelerator) {
        const std::vector<PairMatches> batched =
            match_feature_pairs_accelerated_batches(
                coarse, candidates, options, *accelerator);
        std::atomic<std::size_t> next_filtered_pair{0};
        const std::size_t filter_workers = std::min(
            std::max<std::size_t>(1, thread_count), candidates.size());
        std::vector<std::future<void>> filter_tasks;
        filter_tasks.reserve(filter_workers);
        for (std::size_t worker = 0; worker < filter_workers; ++worker) {
            filter_tasks.push_back(std::async(std::launch::async, [&] {
                while (true) {
                    const std::size_t pair_index = next_filtered_pair.fetch_add(
                        1, std::memory_order_relaxed);
                    if (pair_index >= candidates.size()) return;
                    CoarsePairMatches& result = evaluated[pair_index];
                    result.pair = candidates[pair_index];
                    result.matches = batched[pair_index].matches;
                    result.directional_count = batched[pair_index].directional_count;
                    result.combined_count = result.matches.size();
                    const std::vector<std::size_t> inliers = local_consistency_inliers(
                        coarse[result.pair.first], coarse[result.pair.second],
                        result.matches, &result.pair);
                    result.filtered_count = result.matches.size() - inliers.size();
                    std::vector<FeatureMatch> filtered;
                    filtered.reserve(inliers.size());
                    for (std::size_t index : inliers)
                        filtered.push_back(result.matches[index]);
                    result.matches = std::move(filtered);
                }
            }));
        }
        for (std::future<void>& task : filter_tasks) task.get();
    } else if (target_focal_hctree_enabled()) {
        const std::vector<PairMatches> batched =
            match_feature_pairs_focal_batches(coarse, candidates, options);
        for (std::size_t pair_index = 0; pair_index < candidates.size(); ++pair_index) {
            CoarsePairMatches& result = evaluated[pair_index];
            result.pair = candidates[pair_index];
            result.matches = batched[pair_index].matches;
            result.directional_count = batched[pair_index].directional_count;
            result.combined_count = result.matches.size();
            const std::vector<std::size_t> inliers = local_consistency_inliers(
                coarse[result.pair.first], coarse[result.pair.second], result.matches,
                &result.pair);
            result.filtered_count = result.matches.size() - inliers.size();
            std::vector<FeatureMatch> filtered;
            filtered.reserve(inliers.size());
            for (std::size_t index : inliers) filtered.push_back(result.matches[index]);
            result.matches = std::move(filtered);
        }
    } else {
    std::atomic<std::size_t> next_pair{0};
    // sub_26DAC30 always passes parallel=1 to the HCTree query vtable.  Its
    // 0x26F5670 worker owns the per-thread heap/visited state and dynamically
    // claims query rows in chunks of ten.  When replaying that path, do not
    // also run image pairs concurrently: the target's parallelism lives
    // inside each directional query, not around several HCTree databases.
    const std::size_t worker_count = target_gomp_hctree_enabled() ? 1 : std::min(
        std::max<std::size_t>(1, thread_count), candidates.size());
    std::vector<std::future<void>> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            while (true) {
                const std::size_t pair_index = next_pair.fetch_add(
                    1, std::memory_order_relaxed);
                if (pair_index >= candidates.size()) return;
                const ImagePair pair = candidates[pair_index];
                CoarsePairMatches& result = evaluated[pair_index];
                result.pair = pair;
                result.matches = match_feature_sets(
                    coarse[pair.first], coarse[pair.second], options, accelerator,
                    index_pointers.empty() ? nullptr : index_pointers[pair.first],
                    index_pointers.empty() ? nullptr : index_pointers[pair.second],
                    &result.directional_count);
                result.combined_count = result.matches.size();
                const std::vector<std::size_t> inliers = local_consistency_inliers(
                    coarse[pair.first], coarse[pair.second], result.matches, &pair);
                result.filtered_count = result.matches.size() - inliers.size();
                std::vector<FeatureMatch> filtered;
                filtered.reserve(inliers.size());
                for (std::size_t index : inliers) filtered.push_back(result.matches[index]);
                result.matches = std::move(filtered);
            }
        }));
    }
    for (std::future<void>& worker : workers) worker.get();
    }

    std::size_t directional_total = 0;
    std::size_t combined_total = 0;
    std::size_t filtered_total = 0;
    std::vector<CoarsePairMatches> retained;
    for (CoarsePairMatches& pair : evaluated) {
        directional_total += pair.directional_count;
        combined_total += pair.combined_count;
        filtered_total += pair.filtered_count;
        if (pair.matches.size() >= 10) retained.push_back(std::move(pair));
    }
    std::cout << "  coarse candidate pairs: " << candidates.size() << '\n'
              << "  coarse directional matches: " << directional_total << '\n'
              << "  coarse combined matches: " << combined_total << '\n'
              << "  coarse filtered: " << filtered_total << " out of "
              << combined_total << '\n'
              << "  coarse threshold pairs: " << retained.size() << '\n';

    if (const char* dump = std::getenv("METALIGN_DUMP_COARSE_CORRESPONDENCES");
        dump != nullptr && *dump != '\0') {
        write_coarse_correspondence_fixture(dump, retained);
    }

    // Analysis-only identical-input boundary.  The fixture is emitted directly
    // from sub_138E5F0's 16-byte chunk / 48-byte pair containers and is never
    // consulted unless explicitly requested.  It allows the production
    // skeletal state machine to be tested independently of detector/HCTree row
    // differences; it is not a pair-list shortcut for ordinary alignment.
    if (const char* fixture = std::getenv("METALIGN_REPLAY_TARGET_COARSE");
        fixture != nullptr && *fixture != '\0') {
        retained = read_coarse_correspondence_fixture(fixture, coarse);
    }

    // sub_138E5F0: at most twenty rounds.  Rebuild temporary feature tracks
    // from all accumulated skeletal pairs, score every remaining pair by the
    // number of links joining different tracks, gate at >=10 and >= the
    // already-shared count, then take a strict-greater maximum spanning
    // forest over cameras.
    std::vector<unsigned char> accepted(retained.size(), 0);
    std::size_t accepted_count = 0;
    for (std::size_t round = 0; round < 20; ++round) {
        DisjointSet tracks(offsets.back());
        for (std::size_t pair_index = 0; pair_index < retained.size(); ++pair_index) {
            if (accepted[pair_index] == 0) continue;
            const CoarsePairMatches& pair = retained[pair_index];
            for (const FeatureMatch& match : pair.matches) {
                tracks.unite(offsets[pair.pair.first] + match.first,
                             offsets[pair.pair.second] + match.second);
            }
        }

        struct EdgeScore {
            std::size_t pair_index = 0;
            std::size_t weight = 0;
        };
        std::vector<EdgeScore> scores;
        std::vector<std::vector<std::size_t>> edge_at(
            camera_count, std::vector<std::size_t>(camera_count,
                std::numeric_limits<std::size_t>::max()));
        for (std::size_t pair_index = 0; pair_index < retained.size(); ++pair_index) {
            if (accepted[pair_index] != 0) continue;
            const CoarsePairMatches& pair = retained[pair_index];
            std::size_t shared = 0;
            std::size_t novel = 0;
            for (const FeatureMatch& match : pair.matches) {
                if (tracks.find(offsets[pair.pair.first] + match.first) ==
                    tracks.find(offsets[pair.pair.second] + match.second))
                    ++shared;
                else
                    ++novel;
            }
            if (novel < 10 || novel < shared) continue;
            const std::size_t score_index = scores.size();
            scores.push_back({pair_index, novel});
            edge_at[pair.pair.first][pair.pair.second] = score_index;
            edge_at[pair.pair.second][pair.pair.first] = score_index;
        }
        if (scores.empty()) break;

        std::vector<unsigned char> visited(camera_count, 0);
        std::vector<std::size_t> best(camera_count, 0);
        std::vector<std::size_t> parent_edge(
            camera_count, std::numeric_limits<std::size_t>::max());
        std::vector<std::size_t> forest_scores;
        for (std::size_t root = 0; root < camera_count; ++root) {
            if (visited[root] != 0) continue;
            std::size_t current = root;
            while (true) {
                visited[current] = 1;
                if (parent_edge[current] != std::numeric_limits<std::size_t>::max())
                    forest_scores.push_back(parent_edge[current]);
                for (std::size_t other = 0; other < camera_count; ++other) {
                    const std::size_t score_index = edge_at[current][other];
                    if (visited[other] != 0 ||
                        score_index == std::numeric_limits<std::size_t>::max())
                        continue;
                    if (scores[score_index].weight > best[other]) {
                        best[other] = scores[score_index].weight;
                        parent_edge[other] = score_index;
                    }
                }
                std::size_t strongest = 0;
                std::size_t selected = camera_count;
                for (std::size_t camera = 0; camera < camera_count; ++camera) {
                    if (visited[camera] == 0 && best[camera] > strongest) {
                        strongest = best[camera];
                        selected = camera;
                    }
                }
                if (selected == camera_count) break;
                current = selected;
            }
        }
        if (forest_scores.empty()) break;
        if (const char* dump_directory = std::getenv("METALIGN_DUMP_SKELETON_ROUNDS")) {
            const std::filesystem::path directory(dump_directory);
            std::filesystem::create_directories(directory);
            std::ofstream stream(directory /
                ("round_" + std::to_string(round + 1) + "_forest.csv"));
            stream << "weight,first,second,pair_index\n";
            for (std::size_t score_index : forest_scores) {
                const EdgeScore& score = scores[score_index];
                const ImagePair pair = retained[score.pair_index].pair;
                stream << score.weight << ',' << pair.first << ',' << pair.second
                       << ',' << score.pair_index << '\n';
            }
        }
        for (std::size_t score_index : forest_scores) {
            const std::size_t pair_index = scores[score_index].pair_index;
            if (accepted[pair_index] == 0) {
                accepted[pair_index] = 1;
                ++accepted_count;
            }
        }
        std::cout << "  coarse skeleton round " << (round + 1) << ": +"
                  << forest_scores.size() << ", total " << accepted_count << '\n';
    }

    std::set<ImagePair> result;
    for (std::size_t index = 0; index < retained.size(); ++index)
        if (accepted[index] != 0) result.insert(retained[index].pair);
    // Diagnostic-only hand-off record. The target's full-resolution matcher
    // receives this set after sequential-reference augmentation; exposing the
    // upstream generic set separates pair-identity drift from later matching
    // and track-building differences without changing selection.
    if (const char* dump_path = std::getenv("METALIGN_DUMP_GENERIC_SKELETAL_PAIRS");
        dump_path != nullptr && *dump_path != '\0') {
        std::ofstream stream(dump_path);
        if (!stream) throw std::runtime_error(
            std::string("cannot write generic skeletal-pair dump: ") + dump_path);
        for (const ImagePair pair : result)
            stream << pair.first << ' ' << pair.second << '\n';
    }
    std::cout << "  coarse skeletal pairs: " << result.size() << '\n';
    return result;
}

}  // namespace

std::set<ImagePair> select_generic_image_pairs_from_coarse(
    const std::vector<FeatureSet>& coarse_features,
    const std::vector<ImagePair>& candidates,
    const MatchPhotosOptions& options,
    std::size_t thread_count,
    DescriptorAccelerator* accelerator,
    const std::vector<const CpuDescriptorIndex*>* indices) {
    return recovered_generic_preselection(coarse_features, options, thread_count,
                                          accelerator, &candidates, indices, true);
}

std::set<ImagePair> select_generic_image_pairs_from_coarse(
    const std::vector<FeatureSet>& coarse_features,
    const MatchPhotosOptions& options,
    std::size_t thread_count,
    DescriptorAccelerator* accelerator,
    const std::vector<const CpuDescriptorIndex*>* indices) {
    return recovered_generic_preselection(coarse_features, options, thread_count,
                                          accelerator, nullptr, indices, true);
}

std::map<std::string, ReferencePosition> read_reference_csv(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open reference CSV: " + path.string());
    std::map<std::string, ReferencePosition> result;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::replace(line.begin(), line.end(), ';', ',');
        std::stringstream parser(line);
        std::string name;
        std::string x;
        std::string y;
        std::string z;
        if (!std::getline(parser, name, ',') || !std::getline(parser, x, ',') ||
            !std::getline(parser, y, ',') || !std::getline(parser, z, ',')) continue;
        try {
            result[name] = {std::stod(x), std::stod(y), std::stod(z)};
        } catch (const std::exception&) {
            if (result.empty()) continue;  // permit one header row
            throw std::runtime_error("invalid reference CSV row: " + line);
        }
    }
    return result;
}

std::set<ImagePair> select_image_pairs(
    const std::vector<FeatureSet>& features, const ProgramOptions& options,
    const std::map<std::string, ReferencePosition>& reference,
    std::size_t thread_count,
    DescriptorAccelerator* accelerator) {
    std::set<ImagePair> pairs;
    const std::size_t count = features.size();
    if (options.pairs_file) {
        std::ifstream stream(*options.pairs_file);
        if (!stream) throw std::runtime_error("cannot open pairs file");
        std::string left;
        std::string right;
        while (stream >> left >> right) {
            const auto first = find_image_index(features, left);
            const auto second = find_image_index(features, right);
            if (!first || !second || *first == *second)
                throw std::runtime_error("pairs file references an unknown/duplicate image");
            pairs.insert(ImagePair{*first, *second}.ordered());
        }
        return pairs;
    }

    if (options.match.generic_preselection) {
        pairs = recovered_generic_preselection(
            features, options.match, thread_count, accelerator);
    }

    if (options.match.reference_preselection) {
        const std::size_t neighbors = std::max<std::size_t>(
            1, options.match.reference_preselection_neighbors);
        if (options.match.reference_preselection_mode ==
            ReferencePreselectionMode::Sequential) {
            // The image-directory adapter has no Metashape sequence metadata.
            // In the natural South MatchPhotos call, sub_138AB20 resolved all
            // 128 parent ids to 0 but assigned distinct group keys 0..127.
            // Every group was therefore a singleton and the function executed
            // zero adjacent-candidate and zero insertion paths.  Do not
            // replace that observed metadata state with a guessed all-images
            // sequence: it incorrectly adds (17,18), (84,85) and (110,111).
            //
            // Multi-camera sequence metadata has not yet been recovered into
            // this CLI's FeatureSet, so the sequential augmentation is an
            // evidence-backed no-op for the plain image-directory contract.
        } else if (reference.empty()) {
            // Source/Estimated modes cannot be evaluated without reference
            // coordinates.  Keep the documented fallback separate from the
            // recovered Sequential state machine.
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t delta = 1; delta <= neighbors && i + delta < count; ++delta)
                    pairs.insert({i, i + delta});
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                const auto found = reference.find(basename(features[i]));
                if (found == reference.end()) continue;
                std::vector<std::pair<double, std::size_t>> scored;
                for (std::size_t j = 0; j < count; ++j) {
                    if (i == j) continue;
                    const auto other = reference.find(basename(features[j]));
                    if (other != reference.end())
                        scored.emplace_back(reference_distance(found->second, other->second), j);
                }
                const std::size_t keep = std::min(neighbors, scored.size());
                std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(keep),
                                  scored.end());
                for (std::size_t k = 0; k < keep; ++k)
                    pairs.insert(ImagePair{i, scored[k].second}.ordered());
            }
        }
    }

    if (pairs.empty()) {
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t j = i + 1; j < count; ++j) pairs.insert({i, j});
    }
    if (const char* dump_path = std::getenv("METALIGN_DUMP_SELECTED_PAIRS");
        dump_path != nullptr && *dump_path != '\0') {
        std::ofstream stream(dump_path);
        if (!stream) throw std::runtime_error(
            std::string("cannot write selected-pair dump: ") + dump_path);
        for (const ImagePair pair : pairs)
            stream << pair.first << ' ' << pair.second << '\n';
    }
    return pairs;
}

std::vector<FeatureMatch> match_feature_sets(const FeatureSet& first,
                                             const FeatureSet& second,
                                             const MatchPhotosOptions& options,
                                             DescriptorAccelerator* accelerator,
                                             const CpuDescriptorIndex* first_index,
                                             const CpuDescriptorIndex* second_index,
                                             std::size_t* raw_directional_count) {
    static_cast<void>(options);
    if (first.keypoints.empty() || second.keypoints.size() < 2) return {};
    std::vector<FeatureMatch> matches;
    matches.reserve(std::min(first.keypoints.size(), second.keypoints.size()));
    // Runtime object at 0x26F53F0 stores ratio bits 0x3f4ccccd.  Keep both
    // conversion and multiplication in binary32: promoting 0.8 to double can
    // change the strict boundary for integer Hamming distances (for example
    // second_distance == 5).
    constexpr float ratio = 0.8F;
    const auto passes_ratio = [ratio](double best, double second) {
        return static_cast<float>(best) <
               ratio * static_cast<float>(second);
    };
    std::unique_ptr<CpuDescriptorIndex> owned_first_index;
    std::unique_ptr<CpuDescriptorIndex> owned_second_index;
    if (!accelerator && (!first_index || !second_index)) {
        if (target_pair_local_hctree_enabled()) {
            // The runtime build/query sequence is target-database first:
            // build second image -LoG/+LoG, query every first-image row, then
            // rebuild the two objects for the first image and query in reverse.
            // Queries consume no rand() state, so materializing both pair-local
            // indices in this recovered build order preserves the same tree
            // sequence while bounding residency to one image pair.
            owned_second_index = std::make_unique<CpuDescriptorIndex>(second);
            owned_first_index = std::make_unique<CpuDescriptorIndex>(first);
        } else {
            owned_first_index = std::make_unique<CpuDescriptorIndex>(first);
            owned_second_index = std::make_unique<CpuDescriptorIndex>(second);
        }
        first_index = owned_first_index.get();
        second_index = owned_second_index.get();
    }
    struct DirectionCandidate {
        std::size_t query = 0;
        std::size_t target = 0;
        double distance = 0.0;
    };
    auto append_unique_targets = [&](const std::vector<DirectionCandidate>& candidates,
                                     std::size_t target_count,
                                     bool forward,
                                     const std::vector<std::size_t>& query_indices,
                                     const std::vector<std::size_t>& target_indices) {
        std::vector<std::uint32_t> selected_count(target_count, 0);
        for (const DirectionCandidate& candidate : candidates)
            ++selected_count[candidate.target];
        for (const DirectionCandidate& candidate : candidates) {
            // 0x1D6C080 counts every returned target orientation-row index
            // and emits a match only when that exact target row was selected
            // once in this direction.  Only afterwards are orientation rows
            // mapped back to their detector/source point identities.
            if (selected_count[candidate.target] != 1) continue;
            if (forward) {
                matches.push_back({query_indices[candidate.query],
                                   target_indices[candidate.target],
                                   candidate.distance});
            } else {
                matches.push_back({target_indices[candidate.target],
                                   query_indices[candidate.query],
                                   candidate.distance});
            }
        }
    };
    for (std::size_t group_index = 0; group_index < 2; ++group_index) {
        const int sign = group_index == 0 ? -1 : 1;
        std::vector<Keypoint> first_group_storage;
        std::vector<Keypoint> second_group_storage;
        std::vector<std::size_t> first_indices_storage;
        std::vector<std::size_t> second_indices_storage;
        const std::vector<Keypoint>* first_group_pointer = nullptr;
        const std::vector<Keypoint>* second_group_pointer = nullptr;
        const std::vector<std::size_t>* first_indices_pointer = nullptr;
        const std::vector<std::size_t>* second_indices_pointer = nullptr;
        if (accelerator) {
            for (std::size_t index = 0; index < first.keypoints.size(); ++index) {
                if (first.keypoints[index].laplacian_sign == sign) {
                    first_group_storage.push_back(first.keypoints[index]);
                    first_indices_storage.push_back(index);
                }
            }
            for (std::size_t index = 0; index < second.keypoints.size(); ++index) {
                if (second.keypoints[index].laplacian_sign == sign) {
                    second_group_storage.push_back(second.keypoints[index]);
                    second_indices_storage.push_back(index);
                }
            }
            first_group_pointer = &first_group_storage;
            second_group_pointer = &second_group_storage;
            first_indices_pointer = &first_indices_storage;
            second_indices_pointer = &second_indices_storage;
        } else {
            const CpuDescriptorIndex::Impl::Group& first_cached =
                first_index->impl_->groups[group_index];
            const CpuDescriptorIndex::Impl::Group& second_cached =
                second_index->impl_->groups[group_index];
            first_indices_pointer = &first_cached.original_indices;
            second_indices_pointer = &second_cached.original_indices;
        }
        const std::vector<std::size_t>& first_indices = *first_indices_pointer;
        const std::vector<std::size_t>& second_indices = *second_indices_pointer;
        const std::size_t first_group_size = first_indices.size();
        const std::size_t second_group_size = second_indices.size();
        if (first_group_size < 2 || second_group_size < 2) continue;
        if (accelerator) {
            const std::vector<Keypoint>& first_group = *first_group_pointer;
            const std::vector<Keypoint>& second_group = *second_group_pointer;
            // The recovered CUDA/OpenCL matcher performs the binary32 ratio
            // test in the device kernel and returns either the accepted row
            // index or -1.  Do not repeat the ratio test in host precision.
            const auto forward = accelerator->ratio_matches(
                first_group, second_group, ratio);
            const auto reverse = accelerator->ratio_matches(
                second_group, first_group, ratio);
            std::vector<DirectionCandidate> forward_candidates;
            std::vector<DirectionCandidate> reverse_candidates;
            for (std::size_t query = 0; query < forward.size(); ++query) {
                if (forward[query].target >= 0) {
                    forward_candidates.push_back(
                        {query, static_cast<std::size_t>(forward[query].target),
                         forward[query].distance});
                }
            }
            for (std::size_t query = 0; query < reverse.size(); ++query) {
                if (reverse[query].target >= 0) {
                    reverse_candidates.push_back(
                        {query, static_cast<std::size_t>(reverse[query].target),
                         reverse[query].distance});
                }
            }
            append_unique_targets(forward_candidates, second_group_size, true,
                                  first_indices, second_indices);
            append_unique_targets(reverse_candidates, first_group_size, false,
                                  second_indices, first_indices);
        } else {
            const DescriptorHCTree& first_tree =
                *first_index->impl_->groups[group_index].tree;
            const DescriptorHCTree& second_tree =
                *second_index->impl_->groups[group_index].tree;
            std::vector<DirectionCandidate> forward_candidates;
            std::vector<DirectionCandidate> reverse_candidates;
            struct QueryResultDump {
                std::uint32_t best_index;
                std::uint32_t second_index;
                std::uint32_t best_distance;
                std::uint32_t second_distance;
                std::uint32_t checks;
            };
            const bool dump_results = std::getenv("METALIGN_DUMP_HCTREE_RESULTS") != nullptr;
            std::vector<QueryResultDump> forward_results;
            std::vector<QueryResultDump> reverse_results;
            if (dump_results) {
                forward_results.reserve(first_group_size);
                reverse_results.reserve(second_group_size);
            }
            using CpuQueryResult = DescriptorHCTree::NearestTwoResult;
            std::vector<std::int32_t> forward_output(first_group_size, -1);
            std::vector<std::int32_t> reverse_output(second_group_size, -1);
            std::vector<CpuQueryResult> forward_query_results;
            std::vector<CpuQueryResult> reverse_query_results;
            if (dump_results) {
                forward_query_results.resize(first_group_size);
                reverse_query_results.resize(second_group_size);
            }
            const bool parallel_queries = target_gomp_hctree_enabled();
#if defined(METALIGN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 10) if(parallel_queries)
#endif
            for (std::ptrdiff_t query = 0;
                 query < static_cast<std::ptrdiff_t>(first_group_size); ++query) {
                const std::size_t row = static_cast<std::size_t>(query);
                const CpuQueryResult nearest = second_tree.nearest_two(
                    first.keypoints[first_indices[row]].descriptor);
                if (dump_results) forward_query_results[row] = nearest;
                if (nearest.second_index >= 0 &&
                    passes_ratio(nearest.best_distance, nearest.second_distance))
                    forward_output[row] = nearest.best_index;
            }
            for (std::size_t query = 0; query < first_group_size; ++query) {
                if (dump_results) {
                    const CpuQueryResult& nearest = forward_query_results[query];
                    forward_results.push_back({
                        static_cast<std::uint32_t>(nearest.best_index),
                        static_cast<std::uint32_t>(nearest.second_index),
                        nearest.best_distance,
                        nearest.second_distance,
                        nearest.examined});
                }
                if (forward_output[query] >= 0) {
                    const double distance = dump_results
                        ? static_cast<double>(forward_query_results[query].best_distance)
                        : 0.0;
                    forward_candidates.push_back(
                        {query, static_cast<std::size_t>(forward_output[query]), distance});
                }
            }
#if defined(METALIGN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 10) if(parallel_queries)
#endif
            for (std::ptrdiff_t query = 0;
                 query < static_cast<std::ptrdiff_t>(second_group_size); ++query) {
                const std::size_t row = static_cast<std::size_t>(query);
                const CpuQueryResult nearest = first_tree.nearest_two(
                    second.keypoints[second_indices[row]].descriptor);
                if (dump_results) reverse_query_results[row] = nearest;
                if (nearest.second_index >= 0 &&
                    passes_ratio(nearest.best_distance, nearest.second_distance))
                    reverse_output[row] = nearest.best_index;
            }
            for (std::size_t query = 0; query < second_group_size; ++query) {
                if (dump_results) {
                    const CpuQueryResult& nearest = reverse_query_results[query];
                    reverse_results.push_back({
                        static_cast<std::uint32_t>(nearest.best_index),
                        static_cast<std::uint32_t>(nearest.second_index),
                        nearest.best_distance,
                        nearest.second_distance,
                        nearest.examined});
                }
                if (reverse_output[query] >= 0) {
                    const double distance = dump_results
                        ? static_cast<double>(reverse_query_results[query].best_distance)
                        : 0.0;
                    reverse_candidates.push_back(
                        {query, static_cast<std::size_t>(reverse_output[query]), distance});
                }
            }
            if (dump_results) {
                const std::filesystem::path dump_dir =
                    std::getenv("METALIGN_DUMP_HCTREE_RESULTS");
                auto dump_query_results = [&](const std::vector<QueryResultDump>& output,
                                              std::size_t query_count,
                                              std::size_t target_count) {
                    const std::filesystem::path path =
                        dump_dir / ("target_" + std::to_string(target_count) +
                                    "_query_" + std::to_string(query_count) + ".bin");
                    std::ofstream stream(path, std::ios::binary);
                    stream.write(reinterpret_cast<const char*>(output.data()),
                                 static_cast<std::streamsize>(output.size() *
                                                              sizeof(output.front())));
                };
                dump_query_results(forward_results, first_group_size, second_group_size);
                dump_query_results(reverse_results, second_group_size, first_group_size);
            }
            if (std::getenv("METALIGN_TRACE_HCTREE_PASSES") != nullptr) {
                static std::size_t pass_call = 0;
                std::cerr << "QUERY_PASS call=" << ++pass_call
                          << " rows=" << first_group_size
                          << " passed=" << forward_candidates.size() << '\n';
                std::cerr << "QUERY_PASS call=" << ++pass_call
                          << " rows=" << second_group_size
                          << " passed=" << reverse_candidates.size() << '\n';
            }
            if (const char* dump_dir = std::getenv("METALIGN_DUMP_HCTREE_OUTPUTS")) {
                auto dump_candidates = [&](const std::vector<std::int32_t>& output,
                                           std::size_t target_count) {
                    const std::filesystem::path path =
                        std::filesystem::path(dump_dir) /
                        ("target_" + std::to_string(target_count) + "_query_" +
                         std::to_string(output.size()) + ".bin");
                    std::ofstream stream(path, std::ios::binary);
                    stream.write(reinterpret_cast<const char*>(output.data()),
                                 static_cast<std::streamsize>(output.size() *
                                                              sizeof(output.front())));
                };
                dump_candidates(forward_output, second_group_size);
                dump_candidates(reverse_output, first_group_size);
            }
            append_unique_targets(forward_candidates, second_group_size, true,
                                  first_indices, second_indices);
            append_unique_targets(reverse_candidates, first_group_size, false,
                                  second_indices, first_indices);
        }
    }
    if (raw_directional_count) *raw_directional_count = matches.size();
    consolidate_orientation_matches(matches, first, second, 0);
    return matches;
}

std::vector<PairMatches> match_feature_pairs_accelerated_batches(
    std::span<const FeatureSet* const> features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options,
    DescriptorAccelerator& accelerator,
    const std::function<bool()>& should_cancel,
    const std::function<void(std::size_t, std::size_t)>& progress_callback) {
    if (should_cancel && should_cancel())
        throw std::runtime_error("accelerated descriptor batch cancelled");
    struct DescriptorGroup {
        std::vector<Keypoint> keypoints;
        std::vector<std::size_t> source_indices;
    };
    std::vector<std::array<DescriptorGroup, 2>> groups(features.size());
    for (std::size_t image = 0; image < features.size(); ++image) {
        for (std::size_t index = 0; index < features[image]->keypoints.size(); ++index) {
            const Keypoint& keypoint = features[image]->keypoints[index];
            const std::size_t group = keypoint.laplacian_sign < 0 ? 0U : 1U;
            groups[image][group].keypoints.push_back(keypoint);
            groups[image][group].source_indices.push_back(index);
        }
    }

    std::vector<PairMatches> result(pairs.size());
    for (std::size_t index = 0; index < pairs.size(); ++index)
        result[index].pair = pairs[index];
    constexpr float ratio = 0.8F;
    std::ofstream ratio_dump;
    if (const char* dump_path = std::getenv("METALIGN_DUMP_RATIO_SOURCE_ROWS")) {
        ratio_dump.open(dump_path, std::ios::binary);
        if (!ratio_dump) throw std::runtime_error(
            std::string("cannot write ratio source-row dump: ") + dump_path);
        ratio_dump.write("MTRATIO1", 8);
    }
    const std::size_t pair_batch_size =
        std::max<std::size_t>(1, options.workitem_size_pairs);
    for (std::size_t begin = 0; begin < pairs.size(); begin += pair_batch_size) {
        if (should_cancel && should_cancel())
            throw std::runtime_error("accelerated descriptor batch cancelled");
        const std::size_t end = std::min(pairs.size(), begin + pair_batch_size);
        struct Direction {
            std::size_t pair_index = 0;
            const DescriptorGroup* query = nullptr;
            const DescriptorGroup* target = nullptr;
            bool forward = true;
        };
        std::vector<Direction> directions;
        std::vector<RatioMatchBatch> batches;
        directions.reserve((end - begin) * 4U);
        batches.reserve((end - begin) * 4U);
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
            const ImagePair pair = pairs[pair_index];
            for (std::size_t group = 0; group < 2; ++group) {
                const DescriptorGroup& first = groups[pair.first][group];
                const DescriptorGroup& second = groups[pair.second][group];
                if (first.keypoints.size() < 2 || second.keypoints.size() < 2) continue;
                directions.push_back({pair_index, &first, &second, true});
                batches.push_back({&first.keypoints, &second.keypoints});
                directions.push_back({pair_index, &second, &first, false});
                batches.push_back({&second.keypoints, &first.keypoints});
            }
        }
        const auto matched = accelerator.ratio_match_batches(batches, ratio);
        if (matched.size() != directions.size())
            throw std::runtime_error("accelerated descriptor batch count mismatch");
        for (std::size_t direction_index = 0;
             direction_index < directions.size(); ++direction_index) {
            const Direction& direction = directions[direction_index];
            const std::vector<RatioMatchResult>& rows = matched[direction_index];
            if (rows.size() != direction.query->keypoints.size())
                throw std::runtime_error("accelerated descriptor row count mismatch");
            if (ratio_dump) {
                const std::size_t query_image = direction.forward
                    ? result[direction.pair_index].pair.first
                    : result[direction.pair_index].pair.second;
                const std::size_t target_image = direction.forward
                    ? result[direction.pair_index].pair.second
                    : result[direction.pair_index].pair.first;
                const std::uint64_t accepted = static_cast<std::uint64_t>(
                    std::count_if(rows.begin(), rows.end(),
                                  [](const RatioMatchResult& row) {
                                      return row.target >= 0;
                                  }));
                const std::uint32_t query_image_u32 =
                    static_cast<std::uint32_t>(query_image);
                const std::uint32_t target_image_u32 =
                    static_cast<std::uint32_t>(target_image);
                ratio_dump.write(reinterpret_cast<const char*>(&query_image_u32),
                                 sizeof(query_image_u32));
                ratio_dump.write(reinterpret_cast<const char*>(&target_image_u32),
                                 sizeof(target_image_u32));
                ratio_dump.write(reinterpret_cast<const char*>(&accepted), sizeof(accepted));
                for (std::size_t query = 0; query < rows.size(); ++query) {
                    if (rows[query].target < 0) continue;
                    const std::uint32_t query_source = static_cast<std::uint32_t>(
                        direction.query->keypoints[query].source_id);
                    const std::uint32_t target_source = static_cast<std::uint32_t>(
                        direction.target->keypoints[
                            static_cast<std::size_t>(rows[query].target)].source_id);
                    ratio_dump.write(reinterpret_cast<const char*>(&query_source),
                                     sizeof(query_source));
                    ratio_dump.write(reinterpret_cast<const char*>(&target_source),
                                     sizeof(target_source));
                }
            }
            std::vector<std::uint32_t> selected_count(
                direction.target->keypoints.size(), 0U);
            for (const RatioMatchResult& row : rows) {
                if (row.target >= 0)
                    ++selected_count[static_cast<std::size_t>(row.target)];
            }
            std::vector<FeatureMatch>& output = result[direction.pair_index].matches;
            for (std::size_t query = 0; query < rows.size(); ++query) {
                const RatioMatchResult& row = rows[query];
                if (row.target < 0) continue;
                const std::size_t target = static_cast<std::size_t>(row.target);
                if (selected_count[target] != 1U) continue;
                if (direction.forward) {
                    output.push_back({direction.query->source_indices[query],
                                      direction.target->source_indices[target],
                                      row.distance});
                } else {
                    output.push_back({direction.target->source_indices[target],
                                      direction.query->source_indices[query],
                                      row.distance});
                }
            }
        }
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
            PairMatches& pair = result[pair_index];
            pair.directional_count = pair.matches.size();
            consolidate_orientation_matches(
                pair.matches, *features[pair.pair.first], *features[pair.pair.second], 0);
        }
        if (progress_callback) progress_callback(end, pairs.size());
    }
    if (ratio_dump.is_open() && !ratio_dump)
        throw std::runtime_error("failed while writing ratio source-row dump");
    return result;
}

std::vector<PairMatches> match_feature_pairs_accelerated_batches(
    const std::vector<FeatureSet>& features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options,
    DescriptorAccelerator& accelerator,
    const std::function<bool()>& should_cancel,
    const std::function<void(std::size_t, std::size_t)>& progress_callback) {
    std::vector<const FeatureSet*> pointers;
    pointers.reserve(features.size());
    for (const FeatureSet& feature : features) pointers.push_back(&feature);
    return match_feature_pairs_accelerated_batches(
        std::span<const FeatureSet* const>(pointers), pairs, options, accelerator,
        should_cancel, progress_callback);
}

std::vector<PairMatches> match_feature_pairs_focal_batches(
    const std::vector<FeatureSet>& features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options) {
    static_cast<void>(options);
    struct DirectionCandidate {
        std::size_t query = 0;
        std::size_t target = 0;
        double distance = 0.0;
    };
    struct PairState {
        std::array<std::vector<DirectionCandidate>, 2> forward;
        std::array<std::vector<DirectionCandidate>, 2> reverse;
    };
    struct Adjacency {
        std::size_t pair = 0;
        bool database_is_first = false;
    };

    std::vector<std::array<std::vector<std::size_t>, 2>> rows(features.size());
    for (std::size_t image = 0; image < features.size(); ++image) {
        for (std::size_t row = 0; row < features[image].keypoints.size(); ++row) {
            const std::size_t group =
                features[image].keypoints[row].laplacian_sign < 0 ? 0 : 1;
            rows[image][group].push_back(row);
        }
    }
    std::vector<std::vector<Adjacency>> adjacency(features.size());
    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        const ImagePair pair = pairs[pair_index].ordered();
        if (pair.first >= features.size() || pair.second >= features.size() ||
            pair.first == pair.second)
            throw std::runtime_error("invalid image pair in focal HCTree batch");
        adjacency[pair.first].push_back({pair_index, true});
        adjacency[pair.second].push_back({pair_index, false});
    }

    std::vector<PairState> states(pairs.size());
    constexpr float ratio = 0.8F;
    const bool parallel_queries = target_gomp_hctree_enabled();
    // sub_1D6C080 trains the two sign-group matchers for one focal image,
    // queries every adjacent feature set through sub_26DB230, then returns the
    // matcher to the worker for the next focal image.  This is deliberately
    // neither an all-images-resident cache nor a per-pair rebuild.
    for (std::size_t database = 0; database < features.size(); ++database) {
        if (adjacency[database].empty()) continue;
        CpuDescriptorIndex database_index(features[database]);
        for (const Adjacency edge : adjacency[database]) {
            const ImagePair pair = pairs[edge.pair].ordered();
            const std::size_t query = edge.database_is_first
                ? pair.second : pair.first;
            for (std::size_t group = 0; group < 2; ++group) {
                const CpuDescriptorIndex::Impl::Group& database_group =
                    database_index.impl_->groups[group];
                const std::vector<std::size_t>& query_rows = rows[query][group];
                if (!database_group.tree || query_rows.size() < 2) continue;
                // sub_26F5670 keeps one five-dword nearest-two record on each
                // worker's stack, applies the binary32 ratio gate immediately,
                // and writes only an int32 row id (or -1) to the shared output.
                std::vector<std::int32_t> query_output(query_rows.size(), -1);
#if defined(METALIGN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 10) if(parallel_queries)
#endif
                for (std::ptrdiff_t query_local = 0;
                     query_local < static_cast<std::ptrdiff_t>(query_rows.size());
                     ++query_local) {
                    const std::size_t row = static_cast<std::size_t>(query_local);
                    const DescriptorHCTree::NearestTwoResult nearest =
                        database_group.tree->nearest_two(
                            features[query].keypoints[query_rows[row]].descriptor);
                    if (nearest.second_index >= 0 &&
                        static_cast<float>(nearest.best_distance) <
                            ratio * static_cast<float>(nearest.second_distance))
                        query_output[row] = nearest.best_index;
                }
                std::vector<DirectionCandidate>& candidates = edge.database_is_first
                    ? states[edge.pair].reverse[group]
                    : states[edge.pair].forward[group];
                candidates.reserve(query_rows.size());
                for (std::size_t query_local = 0;
                     query_local < query_rows.size(); ++query_local) {
                    if (query_output[query_local] >= 0) {
                        candidates.push_back(
                            {query_local,
                             static_cast<std::size_t>(query_output[query_local]), 0.0});
                    }
                }
            }
        }
    }

    std::vector<PairMatches> output(pairs.size());
    auto append_unique_targets = [](
        const std::vector<DirectionCandidate>& candidates,
        const std::vector<std::size_t>& query_rows,
        const std::vector<std::size_t>& target_rows,
        bool forward,
        std::vector<FeatureMatch>& matches) {
        std::vector<std::uint32_t> selected_count(target_rows.size(), 0);
        for (const DirectionCandidate& candidate : candidates)
            ++selected_count[candidate.target];
        for (const DirectionCandidate& candidate : candidates) {
            if (selected_count[candidate.target] != 1) continue;
            if (forward) {
                matches.push_back({query_rows[candidate.query],
                                   target_rows[candidate.target],
                                   candidate.distance});
            } else {
                matches.push_back({target_rows[candidate.target],
                                   query_rows[candidate.query],
                                   candidate.distance});
            }
        }
    };
    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        PairMatches& result = output[pair_index];
        result.pair = pairs[pair_index].ordered();
        const PairState& state = states[pair_index];
        for (std::size_t group = 0; group < 2; ++group) {
            append_unique_targets(state.forward[group], rows[result.pair.first][group],
                                  rows[result.pair.second][group], true,
                                  result.matches);
            append_unique_targets(state.reverse[group], rows[result.pair.second][group],
                                  rows[result.pair.first][group], false,
                                  result.matches);
        }
        result.directional_count = result.matches.size();
        consolidate_orientation_matches(
            result.matches, features[result.pair.first], features[result.pair.second], 0);
    }
    return output;
}

std::vector<std::size_t> local_consistency_inliers(
    const FeatureSet& first, const FeatureSet& second,
    const std::vector<FeatureMatch>& matches, const ImagePair* pair) {
    // Metashape 2.3.0: 0x137BAA0 initializes {5, 10, 3, 2.0}; the OpenMP
    // worker at 0x1393BA0 applies the rule below to every type-6 image pair.
    if (matches.size() <= 7) return {};
    std::vector<Vec2> first_points(matches.size());
    std::vector<Vec2> second_points(matches.size());
    for (std::size_t index = 0; index < matches.size(); ++index) {
        const FeatureMatch& match = matches[index];
        first_points[index] = {first.keypoints[match.first].x,
                               first.keypoints[match.first].y};
        second_points[index] = {second.keypoints[match.second].x,
                                second.keypoints[match.second].y};
    }
    // 0x13A0E50 builds one 2-D AABB/k-d index for each image's match
    // coordinates. 0x13A22F0 performs exact bounded nearest-neighbour queries
    // for both the five-neighbour median and the ten-neighbour consistency pass.
    const SpatialNeighborIndex first_index(first_points);
    const SpatialNeighborIndex second_index(second_points);
    const float first_local =
        median_fifth_neighbor_distance(first_index, first_points) * 4.0F;
    const float second_local =
        median_fifth_neighbor_distance(second_index, second_points) * 4.0F;
    const float first_global_radius = static_cast<float>(
        std::max(first.image_width, first.image_height)) * 0.050000001F;
    const float second_global_radius = static_cast<float>(
        std::max(second.image_width, second.image_height)) * 0.050000001F;
    const float first_global = first_global_radius * first_global_radius;
    const float second_global = second_global_radius * second_global_radius;

    std::vector<std::size_t> inliers;
    std::vector<std::uint32_t> common_counts;
    const bool dump_decisions = local_decision_dump_enabled_for_pair(pair);
    if (dump_decisions) common_counts.resize(matches.size());
    inliers.reserve(matches.size());
    for (std::size_t query = 0; query < matches.size(); ++query) {
        const std::vector<std::size_t> first_neighbors = accepted_neighbors(
            first_index, query, first_local, first_global);
        const std::vector<std::size_t> second_neighbors = accepted_neighbors(
            second_index, query, second_local, second_global);
        std::size_t left = 0;
        std::size_t right = 0;
        std::size_t common = 0;
        while (left < first_neighbors.size() && right < second_neighbors.size()) {
            if (first_neighbors[left] < second_neighbors[right]) {
                ++left;
            } else if (second_neighbors[right] < first_neighbors[left]) {
                ++right;
            } else {
                ++common;
                ++left;
                ++right;
            }
        }
        if (dump_decisions) common_counts[query] = static_cast<std::uint32_t>(common);
        if (common >= 3) inliers.push_back(query);
    }
    if (dump_decisions) {
        static std::atomic<std::size_t> dump_call{0};
        const std::filesystem::path directory =
            std::getenv("METALIGN_DUMP_LOCAL_DECISIONS");
        std::filesystem::create_directories(directory);
        const std::size_t call = dump_call.fetch_add(1, std::memory_order_relaxed);
        std::ofstream stream(directory / ("call_" + std::to_string(call) + ".csv"));
        stream << std::setprecision(9)
               << "# pair_first=" << (pair == nullptr ? -1 : static_cast<long long>(pair->first))
               << ",pair_second=" << (pair == nullptr ? -1 : static_cast<long long>(pair->second))
               << '\n'
               << "# first_local=" << first_local
               << ",second_local=" << second_local
               << ",first_global=" << first_global
               << ",second_global=" << second_global << '\n'
               << "match_row,first_feature,second_feature,first_source_id,second_source_id,"
                  "x1,y1,x2,y2,common,accepted\n";
        for (std::size_t index = 0; index < matches.size(); ++index) {
            const FeatureMatch& match = matches[index];
            stream << index << ',' << match.first << ',' << match.second << ','
                   << first.keypoints[match.first].source_id << ','
                   << second.keypoints[match.second].source_id << ','
                   << static_cast<float>(first_points[index].x) << ','
                   << static_cast<float>(first_points[index].y) << ','
                   << static_cast<float>(second_points[index].x) << ','
                   << static_cast<float>(second_points[index].y) << ','
                   << common_counts[index] << ',' << (common_counts[index] >= 3) << '\n';
        }
    }
    return inliers;
}

}  // namespace metalign
