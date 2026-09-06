#include "metmodel/patchmatch.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace metmodel {
namespace {

std::atomic<std::uint64_t> recovered_patchmatch_atlas_generation{0U};

int solve_recovered_quadratic_monic(double linear, double constant,
                                    double* roots) {
    const double half_linear = 0.5 * linear;
    const double discriminant = half_linear * half_linear - constant;
    if (discriminant > 0.0) {
        const double square_root = std::sqrt(discriminant);
        roots[0] = square_root - half_linear;
        roots[1] = -half_linear - square_root;
        return 2;
    }
    if (discriminant < 0.0) return 0;
    roots[0] = -half_linear;
    return 1;
}

int solve_recovered_quadratic(const std::array<double, 3>& coefficients,
                              double* roots) {
    const double constant = coefficients[0];
    const double linear = coefficients[1];
    const double quadratic = coefficients[2];
    if (quadratic == 0.0) {
        if (linear == 0.0) return constant == 0.0 ? -1 : 0;
        roots[0] = -constant / linear;
        return 1;
    }

    double discriminant = linear * linear;
    double four_quadratic = 4.0 * quadratic;
    four_quadratic *= constant;
    discriminant -= four_quadratic;
    if (discriminant < 0.0) return 0;

    const double square_root = std::sqrt(discriminant);
    double signed_square_root = linear < 0.0 ? -square_root : square_root;
    signed_square_root -= linear;
    signed_square_root *= 0.5;
    roots[0] = signed_square_root / quadratic;
    if (!(square_root > 0.0)) return 1;
    roots[1] = constant / signed_square_root;
    return 2;
}

int solve_recovered_cubic(const std::array<double, 4>& coefficients,
                          double* roots) {
    const double leading = coefficients[3];
    if (leading == 0.0) {
        return solve_recovered_quadratic(
            {coefficients[0], coefficients[1], coefficients[2]}, roots);
    }

    const double inverse_leading = 1.0 / leading;
    const double constant = coefficients[0] * inverse_leading;
    const double linear = coefficients[1] * inverse_leading;
    const double quadratic = coefficients[2] * inverse_leading;

    double p = quadratic * quadratic;
    double three_linear = 3.0 * linear;
    p -= three_linear;
    p *= 0.1111111111111111;

    double nine_quadratic = 9.0 * quadratic;
    double nine_quadratic_linear = linear * nine_quadratic;
    double two_quadratic_cubed = quadratic + quadratic;
    two_quadratic_cubed *= quadratic;
    two_quadratic_cubed *= quadratic;
    two_quadratic_cubed -= nine_quadratic_linear;
    double twenty_seven_constant = constant * 27.0;
    twenty_seven_constant += two_quadratic_cubed;
    double q = twenty_seven_constant * 0.018518518518518517;

    double p_cubed = p * p;
    p_cubed *= p;
    double q_squared = q * q;
    const double discriminant = p_cubed - q_squared;
    const double one_third = 0.3333333333333333;

    if (discriminant >= 0.0) {
        const double p_root = std::sqrt(p_cubed);
        const double angle = std::acos(q / p_root);
        const double radius = 2.0 * std::sqrt(p);
        const double reduced_angle = angle * one_third;
        const double shift = quadratic * one_third;
        roots[0] = radius * std::cos(reduced_angle) - shift;
        roots[1] = radius * std::cos(reduced_angle + 2.0943951023931953) - shift;
        roots[2] = radius * std::cos(reduced_angle + 4.1887902047863905) - shift;
        return 3;
    }

    const double discriminant_root = std::sqrt(-discriminant);
    double magnitude = std::abs(q) + discriminant_root;
    // This deliberately uses the target's truncated exponent constant rather
    // than std::cbrt or the exactly rounded value 1/3.
    magnitude = std::pow(magnitude, 0.333333333333);
    if (q > 0.0) magnitude = -magnitude;
    double root = p / magnitude;
    root += magnitude;
    root -= quadratic * one_third;
    roots[0] = root;
    return 1;
}

int solve_recovered_quartic(const std::array<double, 5>& coefficients,
                            double* roots) {
    const double leading = coefficients[4];
    if (leading == 0.0) {
        return solve_recovered_cubic(
            {coefficients[0], coefficients[1], coefficients[2],
             coefficients[3]},
            roots);
    }

    const double constant = coefficients[0] / leading;
    const double linear = coefficients[1] / leading;
    const double quadratic = coefficients[2] / leading;
    const double cubic = coefficients[3] / leading;
    const double shift = cubic * 0.25;

    double shift_squared = shift * shift;
    double p = 6.0 * shift_squared;
    p = quadratic - p;

    double q = 8.0 * shift_squared;
    double twice_quadratic = quadratic + quadratic;
    q -= twice_quadratic;
    q *= shift;
    q += linear;

    double r_base = quadratic - 3.0 * shift_squared;
    double r = r_base * shift;
    r -= linear;
    r *= shift;
    r += constant;

    if (q == 0.0) {
        const double discriminant = p * p - 4.0 * r;
        if (discriminant < 0.0) return 0;
        const double discriminant_root = std::sqrt(discriminant);
        const double first_squared = 0.5 * (discriminant_root - p);
        int count = 0;
        if (first_squared >= 0.0) {
            const double first = std::sqrt(first_squared);
            roots[count++] = first - shift;
            roots[count++] = -first - shift;
        }
        if (!(discriminant_root > 0.0)) return count;
        const double second_squared = -0.5 * (p + discriminant_root);
        if (second_squared >= 0.0) {
            const double second = std::sqrt(second_squared);
            roots[count++] = second - shift;
            roots[count++] = -second - shift;
        }
        return count;
    }

    if (r == 0.0) {
        const int cubic_count = solve_recovered_cubic({q, p, 0.0, 1.0}, roots);
        if (cubic_count <= 0) {
            roots[0] = -shift;
            return 1;
        }
        for (int index = 0; index < cubic_count; ++index) roots[index] -= shift;
        roots[cubic_count] = -shift;
        return cubic_count + 1;
    }

    double q_squared = q * q;
    double resolvent_constant = 0.5 * r;
    resolvent_constant *= p;
    q_squared *= 0.125;
    resolvent_constant -= q_squared;
    std::array<double, 3> resolvent_roots{};
    const int resolvent_count = solve_recovered_cubic(
        {resolvent_constant, -r, -0.5 * p, 1.0},
        resolvent_roots.data());
    const double z = resolvent_count == -1 ? 0.0 : resolvent_roots[0];

    const double z_discriminant = z * z - r;
    const double linear_discriminant = z + z - p;
    if (z_discriminant < 0.0 || linear_discriminant < 0.0) return 0;
    const double z_root = std::sqrt(z_discriminant);
    const double linear_root = std::sqrt(linear_discriminant);
    double first_linear = linear_root;
    double second_linear = -linear_root;
    if (q > 0.0) std::swap(first_linear, second_linear);

    int count = solve_recovered_quadratic_monic(
        first_linear, z - z_root, roots);
    if (count < 0) count = 0;
    int second_count = solve_recovered_quadratic_monic(
        second_linear, z + z_root, roots + count);
    if (second_count < 0) second_count = 0;
    count += second_count;
    for (int index = 0; index < count; ++index) roots[index] -= shift;
    return count;
}

std::array<float, 16> identity4() {
    return {1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F};
}

std::array<double, 16> target_inverse_double_from_float(
    const std::array<float, 16>& source) {
    std::array<double, 16> m{};
    for (std::size_t index = 0; index < m.size(); ++index)
        m[index] = static_cast<double>(source[index]);
    const double a0 = m[0] * m[5] - m[1] * m[4];
    const double a1 = m[0] * m[6] - m[2] * m[4];
    const double a2 = m[0] * m[7] - m[3] * m[4];
    const double a3 = m[1] * m[6] - m[2] * m[5];
    const double a4 = m[1] * m[7] - m[3] * m[5];
    const double a5 = m[2] * m[7] - m[3] * m[6];
    const double b0 = m[8] * m[13] - m[9] * m[12];
    const double b1 = m[8] * m[14] - m[10] * m[12];
    const double b2 = m[8] * m[15] - m[11] * m[12];
    const double b3 = m[9] * m[14] - m[10] * m[13];
    const double b4 = m[9] * m[15] - m[11] * m[13];
    const double b5 = m[10] * m[15] - m[11] * m[14];
    double determinant = a3 * b2 + a0 * b5 - a1 * b4 +
                         a2 * b3 - a4 * b1 + a5 * b0;
    if (determinant != 0.0) determinant = 1.0 / determinant;
    std::array<double, 16> result{};
    result[0] = (m[5] * b5 - m[6] * b4 + m[7] * b3) * determinant;
    result[1] = (m[2] * b4 - m[1] * b5 - m[3] * b3) * determinant;
    result[2] = (m[13] * a5 - m[14] * a4 + m[15] * a3) * determinant;
    result[3] = (m[10] * a4 - m[9] * a5 - m[11] * a3) * determinant;
    result[4] = (m[6] * b2 - m[4] * b5 - m[7] * b1) * determinant;
    result[5] = (m[0] * b5 - m[2] * b2 + m[3] * b1) * determinant;
    result[6] = (m[14] * a2 - m[12] * a5 - m[15] * a1) * determinant;
    result[7] = (m[8] * a5 - m[10] * a2 + m[11] * a1) * determinant;
    result[8] = (m[4] * b4 - m[5] * b2 + m[7] * b0) * determinant;
    result[9] = (m[1] * b2 - m[0] * b4 - m[3] * b0) * determinant;
    result[10] = (m[12] * a4 - m[13] * a2 + m[15] * a0) * determinant;
    result[11] = (m[9] * a2 - m[8] * a4 - m[11] * a0) * determinant;
    result[12] = (m[5] * b1 - m[4] * b3 - m[6] * b0) * determinant;
    result[13] = (m[0] * b3 - m[1] * b1 + m[2] * b0) * determinant;
    result[14] = (m[13] * a1 - m[12] * a3 - m[14] * a0) * determinant;
    result[15] = (m[8] * a3 - m[9] * a1 + m[10] * a0) * determinant;
    return result;
}

template <class T>
void store_opaque(std::array<std::byte, 120>& destination,
                  std::size_t offset, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset + sizeof(T) > destination.size())
        throw std::out_of_range("opaque calibration write exceeds record");
    std::memcpy(destination.data() + offset, &value, sizeof(T));
}

std::uint32_t ceil_div(std::uint32_t numerator, std::uint32_t denominator) {
    return numerator / denominator + static_cast<std::uint32_t>(numerator % denominator != 0);
}

std::vector<std::uint32_t> batch_offsets(std::uint32_t items) {
    if (items == 0) return {};
    const std::uint32_t span = patchmatch_balanced_batch_span(items);
    std::vector<std::uint32_t> result;
    for (std::uint32_t offset = 0; offset < items;) {
        result.push_back(offset);
        if (offset > std::numeric_limits<std::uint32_t>::max() - span)
            throw std::overflow_error("PatchMatch batch offset overflow");
        offset += span;
    }
    return result;
}

void append_wta(std::vector<PatchMatchScheduleEvent>& result,
                std::uint32_t downscale, std::uint32_t hypotheses,
                std::uint32_t is_checkboard, std::uint32_t checkboard_step,
                std::uint32_t only_fourth, std::uint32_t offset) {
    result.push_back({PatchMatchScheduleOperation::Wta, downscale, 0, hypotheses,
                      is_checkboard, checkboard_step, only_fourth, offset});
}

void append_iteration(std::vector<PatchMatchScheduleEvent>& result,
                      std::uint32_t downscale, std::uint32_t iteration,
                      std::uint32_t only_fourth,
                      const std::vector<std::uint32_t>& refinement_offsets,
                      const std::vector<std::uint32_t>& propagation_offsets) {
    for (const std::uint32_t offset : refinement_offsets) {
        result.push_back({PatchMatchScheduleOperation::Refinement, downscale,
                          iteration, 0, 0, 0, only_fourth, offset});
        append_wta(result, downscale, 8, 0, 0, only_fourth, offset);
    }
    for (std::uint32_t step = 0; step < 2; ++step) {
        for (const std::uint32_t offset : propagation_offsets) {
            result.push_back({PatchMatchScheduleOperation::Propagation,
                              downscale, 0, 0, 1, step, only_fourth, offset});
            append_wta(result, downscale, 8, 1, step, only_fourth, offset);
        }
    }
}

void append_filters(std::vector<PatchMatchScheduleEvent>& result,
                    std::uint32_t downscale,
                    const std::vector<std::uint32_t>& full_offsets) {
    const auto append_all = [&](PatchMatchScheduleOperation operation) {
        for (const std::uint32_t offset : full_offsets)
            result.push_back({operation, downscale, 0, 0, 0, 0, 0, offset});
    };
    append_all(PatchMatchScheduleOperation::FilterCheckCost);
    append_all(PatchMatchScheduleOperation::FilterCheckNeighbours);
    append_all(PatchMatchScheduleOperation::FilterClearDepth);
    append_all(PatchMatchScheduleOperation::FilterNormals);
    append_all(PatchMatchScheduleOperation::FilterClearDepth);
    append_all(PatchMatchScheduleOperation::FilterSpecklesEdges);
}

}  // namespace

std::uint32_t recovered_depth_component_threshold(FilterMode mode) {
    switch (mode) {
        case FilterMode::None:
        case FilterMode::Mild:
            return 6;
        case FilterMode::Moderate:
            return 24;
        case FilterMode::Aggressive:
            return 96;
    }
    throw std::invalid_argument("unsupported depth filter mode");
}

std::uint32_t patchmatch_balanced_batch_span(
    std::uint32_t items, std::uint32_t alignment, std::uint32_t capacity) {
    if (alignment == 0 || capacity == 0)
        throw std::invalid_argument("PatchMatch batch alignment/capacity must be non-zero");
    if (items == 0) return alignment;
    const std::uint32_t batches = ceil_div(items, capacity);
    const std::uint32_t unaligned = ceil_div(items, batches);
    const std::uint64_t aligned =
        static_cast<std::uint64_t>(ceil_div(unaligned, alignment)) * alignment;
    if (aligned > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("PatchMatch balanced batch span overflow");
    return static_cast<std::uint32_t>(aligned);
}

std::vector<PatchMatchScheduleEvent> make_recovered_patchmatch_level_schedule(
    std::uint32_t width_original, std::uint32_t height_original,
    std::uint32_t target_downscale) {
    if (width_original == 0 || height_original == 0)
        throw std::invalid_argument("PatchMatch schedule requires non-zero image dimensions");
    if (target_downscale < 2 || target_downscale > 32 ||
        (target_downscale & (target_downscale - 1)) != 0)
        throw std::invalid_argument(
            "recovered level schedule is validated for power-of-two downscale 2..32");

    std::vector<PatchMatchScheduleEvent> result;
    for (std::uint32_t downscale = 32;; downscale /= 2) {
        const std::uint32_t width = ceil_div(width_original, downscale);
        const std::uint32_t height = ceil_div(height_original, downscale);
        const std::uint32_t full_items = width * height;
        const std::uint32_t checker_items = ceil_div(width, 2) * height;
        const std::uint32_t fourth_items = ceil_div(width, 2) * ceil_div(height, 2);
        const std::uint32_t fourth_checker_items =
            ceil_div(width, 4) * ceil_div(height, 2);
        const auto full_offsets = batch_offsets(full_items);
        const auto checker_offsets = batch_offsets(checker_items);
        const auto fourth_offsets = batch_offsets(fourth_items);
        const auto fourth_checker_offsets = batch_offsets(fourth_checker_items);
        const bool coarsest = downscale == 32;
        const bool target = downscale == target_downscale;

        if (coarsest) {
            for (std::uint32_t iteration = 0; iteration < 6; ++iteration)
                append_iteration(result, downscale, iteration, 0,
                                 full_offsets, checker_offsets);
        } else {
            const std::uint32_t inherited_iterations = target ? 1U : 6U;
            for (std::uint32_t iteration = 0; iteration < inherited_iterations;
                 ++iteration) {
                append_iteration(result, downscale, iteration, 1,
                                 fourth_offsets, fourth_checker_offsets);
            }

            result.push_back({PatchMatchScheduleOperation::CostPipeline,
                              downscale});
            for (const std::uint32_t offset : full_offsets) {
                result.push_back({PatchMatchScheduleOperation::CoarseToPrecise,
                                  downscale, 0, 0, 0, 0, 0, offset});
                result.push_back({PatchMatchScheduleOperation::CostBatch,
                                  downscale, 0, 0, 0, 0, 0, offset});
                append_wta(result, downscale, 2, 0, 0, 0, offset);
            }

            if (target) {
                append_iteration(result, downscale, 2, 0,
                                 full_offsets, checker_offsets);
            } else {
                append_iteration(result, downscale, 7, 0,
                                 full_offsets, checker_offsets);
                append_iteration(result, downscale, 8, 0,
                                 full_offsets, checker_offsets);
            }
        }

        // Three finest requested pyramid levels use the deterministic final
        // refinement.  This boundary is identical in d2, d4 and d8 traces.
        if (downscale <= target_downscale * 4U) {
            for (const std::uint32_t offset : full_offsets) {
                result.push_back({PatchMatchScheduleOperation::FinalRefinement,
                                  downscale, 0, 0, 0, 0, 0, offset});
                append_wta(result, downscale, 8, 0, 0, 0, offset);
            }
        }
        append_filters(result, downscale, full_offsets);
        if (target) break;
    }
    return result;
}

const char* patchmatch_schedule_operation_name(PatchMatchScheduleOperation operation) {
    switch (operation) {
        case PatchMatchScheduleOperation::Refinement: return "refinement";
        case PatchMatchScheduleOperation::Wta: return "wta";
        case PatchMatchScheduleOperation::Propagation: return "propagation";
        case PatchMatchScheduleOperation::CoarseToPrecise:
            return "coarse_to_precise";
        case PatchMatchScheduleOperation::CostPipeline: return "cost_pipeline";
        case PatchMatchScheduleOperation::CostBatch: return "cost_batch";
        case PatchMatchScheduleOperation::FinalRefinement: return "final_refinement";
        case PatchMatchScheduleOperation::FilterCheckCost: return "filter_check_cost";
        case PatchMatchScheduleOperation::FilterCheckNeighbours:
            return "filter_check_neighbours";
        case PatchMatchScheduleOperation::FilterClearDepth: return "filter_clear_depth";
        case PatchMatchScheduleOperation::FilterNormals: return "filter_normals";
        case PatchMatchScheduleOperation::FilterSpecklesEdges:
            return "filter_speckles_edges";
    }
    return "unknown";
}

std::vector<std::uint32_t> make_recovered_patchmatch_neighbor_subset(
    std::uint32_t neighbor_count,
    std::uint32_t iteration,
    bool all_neighbors_state) {
    std::vector<std::uint32_t> result;
    result.reserve(neighbor_count);
    for (std::uint32_t index = 0; index < neighbor_count; ++index) {
        if (all_neighbors_state || index <= 2U ||
            ((((index - 3U) ^ iteration) & 3U) == 0U)) {
            result.push_back(index);
        }
    }
    return result;
}

PatchMatchDepthRangeInput make_recovered_patchmatch_depth_range_input(
    const Scene& scene, std::size_t camera_index) {
    if (camera_index >= scene.cameras.size())
        throw std::out_of_range("PatchMatch depth-range camera index is out of range");
    if (!scene.region.specified)
        throw std::invalid_argument(
            "recovered PatchMatch depth range requires the reconstruction region");

    std::unordered_map<std::uint32_t, const SparsePoint*> points_by_track;
    points_by_track.reserve(scene.sparse_points.size());
    for (const SparsePoint& point : scene.sparse_points) {
        if (point.track_id == std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "recovered PatchMatch depth range requires sparse track IDs");
        if (point.homogeneous_w != 1.0F)
            throw std::invalid_argument(
                "recovered PatchMatch depth range is validated only for w=1 sparse points");
        if (!points_by_track.emplace(point.track_id, &point).second)
            throw std::invalid_argument("duplicate sparse track ID");
    }

    const Camera& camera = scene.cameras[camera_index];
    const ReconstructionRegion& region = scene.region;
    const bool region_disabled =
        region.size.x == 0.0 && region.size.y == 0.0 && region.size.z == 0.0;
    PatchMatchDepthRangeInput result;
    result.track_ids.reserve(camera.track_ids.size());
    result.camera_depths.reserve(camera.track_ids.size());

    for (const std::uint32_t track_id : camera.track_ids) {
        const auto found = points_by_track.find(track_id);
        if (found == points_by_track.end()) continue;
        const SparsePoint& point = *found->second;

        // The target's working point record stores float3 even though both the
        // region and camera transforms are double precision.
        const float point_x = static_cast<float>(point.position.x);
        const float point_y = static_cast<float>(point.position.y);
        const float point_z = static_cast<float>(point.position.z);
        const double dx = static_cast<double>(point_x) - region.center.x;
        const double dy = static_cast<double>(point_y) - region.center.y;
        const double dz = static_cast<double>(point_z) - region.center.z;
        if (!region_disabled) {
            const double axis0 =
                region.rotation[0] * dx + region.rotation[3] * dy +
                region.rotation[6] * dz;
            if (axis0 < -0.5 * region.size.x || axis0 > 0.5 * region.size.x)
                continue;
            const double axis1 =
                region.rotation[1] * dx + region.rotation[4] * dy +
                region.rotation[7] * dz;
            if (axis1 < -0.5 * region.size.y || axis1 > 0.5 * region.size.y)
                continue;
            const double axis2 =
                dx * region.rotation[2] + dy * region.rotation[5] +
                dz * region.rotation[8];
            if (axis2 < -0.5 * region.size.z || axis2 > 0.5 * region.size.z)
                continue;
        }

        const auto& rotation = camera.pose.rotation.v;
        const double depth_double =
            (rotation[6] * static_cast<double>(point_x) +
             rotation[7] * static_cast<double>(point_y)) +
            rotation[8] * static_cast<double>(point_z) + camera.pose.translation.z;
        const float depth = static_cast<float>(depth_double);
        if (!std::isfinite(depth) || depth <= 0.0F)
            throw std::invalid_argument(
                "perspective sparse point did not produce a positive finite camera depth");
        result.track_ids.push_back(track_id);
        result.camera_depths.push_back(depth);
    }
    return result;
}

PatchMatchDepthRange reduce_recovered_patchmatch_depth_range(
    const std::vector<float>& camera_depths) {
    float minimum = std::numeric_limits<float>::max();
    float maximum = -std::numeric_limits<float>::max();
    for (const float depth : camera_depths) {
        if (std::isnan(depth))
            throw std::invalid_argument("NaN in PatchMatch camera-depth samples");
        if (depth < minimum) minimum = depth;
        if (depth > maximum) maximum = depth;
    }
    if (camera_depths.empty()) {
        // Exact target empty-vector branch.  The caller normally rejects this
        // state using the simultaneously returned valid-projection count.
        minimum = std::bit_cast<float>(0x7EFFFFFFU);
        maximum *= 1.5F;
        return {minimum, maximum};
    }
    if (camera_depths.size() == 1U)
        return {minimum * 0.5F, maximum * 1.5F};
    if (maximum < minimum)
        throw std::runtime_error("invalid PatchMatch camera-depth range");
    const float span_margin = (maximum - minimum) * 0.25F;
    const float scale_margin = maximum * 0.01F;
    const float margin = std::max(scale_margin, span_margin);
    return {minimum - 1.5F * margin, maximum + margin};
}

PatchMatchDepthRange make_recovered_patchmatch_depth_range(
    const Scene& scene, std::size_t camera_index) {
    return reduce_recovered_patchmatch_depth_range(
        make_recovered_patchmatch_depth_range_input(scene, camera_index)
            .camera_depths);
}

float reduce_recovered_patchmatch_deviation_multiplier(
    std::span<const std::uint8_t> prepared_image) {
    if (prepared_image.empty())
        throw std::invalid_argument(
            "recovered PatchMatch deviation multiplier requires a non-empty uint8 image");
    const auto [minimum, maximum] =
        std::minmax_element(prepared_image.begin(), prepared_image.end());

    // sub_1CC1EC0 asks the uint8 descriptor for its full [0,255] interval,
    // asks sub_1CC0E20 for the actual image min/max, adds one to both inclusive
    // interval lengths, divides in double, then sub_1CEE400 converts to float.
    const double actual_span =
        static_cast<double>(*maximum) - static_cast<double>(*minimum) + 1.0;
    constexpr double full_span = 256.0;
    return static_cast<float>(actual_span / full_span);
}

float compose_recovered_patchmatch_deviation_multiplier(
    std::span<const std::uint8_t> prepared_image,
    float owner_multiplier) {
    // 0x1CF393C performs mulss, so preserve the float multiplication after the
    // double-ratio-to-float conversion.
    return reduce_recovered_patchmatch_deviation_multiplier(prepared_image) *
           owner_multiplier;
}

std::vector<std::uint8_t> make_recovered_patchmatch_source_u8(
    const metalign::Image& image) {
    const std::size_t pixels = image.width * image.height;
    if (image.width == 0 || image.height == 0)
        throw std::invalid_argument(
            "recovered PatchMatch source conversion requires a non-empty image");
    std::vector<std::uint8_t> result(pixels);
    if (image.rgb.empty()) {
        if (image.gray.size() != pixels)
            throw std::invalid_argument(
                "recovered PatchMatch source conversion requires RGB8 or canonical grayscale");
        for (std::size_t index = 0; index < pixels; ++index) {
            const float value = image.gray[index];
            if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
                throw std::invalid_argument(
                    "recovered PatchMatch canonical grayscale is outside [0,1]");
            const auto code = static_cast<std::uint8_t>(std::clamp(
                std::lround(static_cast<double>(value) * 255.0), 0L, 255L));
            if (value != static_cast<float>(static_cast<double>(code) / 255.0))
                throw std::invalid_argument(
                    "recovered PatchMatch grayscale is not an exact uint8 code plane");
            result[index] = code;
        }
        return result;
    }
    if (image.rgb.size() != pixels * 3U)
        throw std::invalid_argument(
            "recovered PatchMatch source conversion has an invalid RGB8 plane");
    for (std::size_t index = 0; index < pixels; ++index) {
        const double red = static_cast<double>(image.rgb[index * 3U]);
        const double green = static_cast<double>(image.rgb[index * 3U + 1U]);
        const double blue = static_cast<double>(image.rgb[index * 3U + 2U]);
        // sub_407B2D0: double BT.601 arithmetic followed by truncation to the
        // uint8 code.  This is deliberately not rounded.
        const double luminance = red * 0.299 + green * 0.587 + blue * 0.114;
        result[index] = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(luminance), 0, 255));
    }
    return result;
}

double recovered_calibration_maximum_radius_squared(
    const metalign::CameraModel& model) {
    // Calibration setter 0x1597F0B calls 0x2C69CC0 after every radial
    // coefficient update.  That wrapper constructs this derivative
    // polynomial in ascending order, calls the real-root solver at
    // 0x413AB70, and applies a strict-positive minimum reduction.
    const std::array<double, 5> coefficients{
        1.0,
        3.0 * model.k1,
        5.0 * model.k2,
        7.0 * model.k3,
        9.0 * model.k4,
    };
    std::array<double, 4> roots{};
    const int count = solve_recovered_quartic(coefficients, roots.data());
    double result = 1.0e9;
    for (int index = 0; index < count; ++index) {
        if (roots[index] > 0.0) result = std::min(result, roots[index]);
    }
    return result;
}

namespace {

void store_recovered_calibration_common(
    DepthVotingCalibrationCu& result,
    const Camera& camera,
    std::uint32_t type,
    float maximum_radius_squared) {
    if (camera.image.width == 0 || camera.image.height == 0 ||
        camera.image.width > std::numeric_limits<std::uint32_t>::max() ||
        camera.image.height > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "PatchMatch image preparation requires uint32 image dimensions");
    const auto width = static_cast<std::uint32_t>(camera.image.width);
    const auto height = static_cast<std::uint32_t>(camera.image.height);
    store_opaque(result.bytes, 32U, maximum_radius_squared);
    store_opaque(result.bytes, 64U, std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 68U, std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 72U, -std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 76U, -std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 80U, type);
    store_opaque(result.bytes, 84U, width);
    store_opaque(result.bytes, 88U, height);
    store_opaque(result.bytes, 92U, static_cast<float>(camera.model.f));
    store_opaque(result.bytes, 96U, static_cast<float>(
        camera.model.cx - static_cast<double>(width) * 0.5));
    store_opaque(result.bytes, 100U, static_cast<float>(
        camera.model.cy - static_cast<double>(height) * 0.5));
    // The low byte at +112 is set in both captured ordinary-camera records.
    // CUDA type 0 does not evaluate Brown correction despite this flag.
    store_opaque(result.bytes, 112U, std::uint8_t{1});
}

}  // namespace

DepthVotingCalibrationCu make_recovered_patchmatch_source_calibration(
    const Camera& camera) {
    return make_recovered_patchmatch_source_calibration(
        camera,
        static_cast<float>(
            recovered_calibration_maximum_radius_squared(camera.model)));
}

DepthVotingCalibrationCu make_recovered_patchmatch_source_calibration(
    const Camera& camera,
    float maximum_radius_squared) {
    if (!(maximum_radius_squared > 0.0F) ||
        !std::isfinite(maximum_radius_squared))
        throw std::invalid_argument(
            "PatchMatch source maximum radius squared must be finite and positive");
    DepthVotingCalibrationCu result{};
    store_opaque(result.bytes, 0U, static_cast<float>(camera.model.k1));
    store_opaque(result.bytes, 4U, static_cast<float>(camera.model.k2));
    store_opaque(result.bytes, 8U, static_cast<float>(camera.model.k3));
    store_opaque(result.bytes, 12U, static_cast<float>(camera.model.k4));
    store_opaque(result.bytes, 16U, static_cast<float>(camera.model.p1));
    store_opaque(result.bytes, 20U, static_cast<float>(camera.model.p2));
    store_opaque(result.bytes, 24U, static_cast<float>(camera.model.p3));
    store_opaque(result.bytes, 28U, static_cast<float>(camera.model.p4));
    store_recovered_calibration_common(
        result, camera, std::uint32_t{1}, maximum_radius_squared);
    store_opaque(result.bytes, 104U, static_cast<float>(camera.model.b1));
    store_opaque(result.bytes, 108U, static_cast<float>(camera.model.b2));
    return result;
}

DepthVotingCalibrationCu make_recovered_patchmatch_undistorted_calibration(
    const Camera& camera) {
    DepthVotingCalibrationCu result{};
    store_recovered_calibration_common(result, camera, std::uint32_t{0}, 1.0e9F);
    return result;
}

RecoveredPatchMatchImageU8 reduce_recovered_patchmatch_image_half(
    const RecoveredPatchMatchImageU8& source) {
    const std::size_t pixels =
        static_cast<std::size_t>(source.width) * source.height;
    if (source.width == 0 || source.height == 0 || source.image.size() != pixels ||
        (!source.rejection_mask.empty() &&
         source.rejection_mask.size() != pixels))
        throw std::invalid_argument(
            "PatchMatch uint8 half reducer input dimensions do not match buffers");
    RecoveredPatchMatchImageU8 result;
    result.width = (source.width + 1U) / 2U;
    result.height = (source.height + 1U) / 2U;
    result.image.resize(static_cast<std::size_t>(result.width) * result.height);
    for (std::uint32_t y = 0; y < result.height; ++y) {
        const std::uint32_t y0 = y * 2U;
        const std::uint32_t y1 = std::min(y0 + 1U, source.height - 1U);
        for (std::uint32_t x = 0; x < result.width; ++x) {
            const std::uint32_t x0 = x * 2U;
            const std::uint32_t x1 = std::min(x0 + 1U, source.width - 1U);
            const auto index = [width = source.width](std::uint32_t px,
                                                       std::uint32_t py) {
                return static_cast<std::size_t>(py) * width + px;
            };
            const unsigned int sum =
                static_cast<unsigned int>(source.image[index(x0, y0)]) +
                static_cast<unsigned int>(source.image[index(x1, y0)]) +
                static_cast<unsigned int>(source.image[index(x0, y1)]) +
                static_cast<unsigned int>(source.image[index(x1, y1)]);
            const std::size_t destination =
                static_cast<std::size_t>(y) * result.width + x;
            result.image[destination] = static_cast<std::uint8_t>((sum + 2U) >> 2U);
        }
    }
    return result;
}

std::array<float, 16> patchmatch_reference_to_neighbor_transform(
    const Camera& reference, const Camera& neighbor) {
    // 0x25A1070 does not compose the original double poses directly.  It
    // rounds the neighbor world->camera matrix and the reference
    // camera->world matrix (R transpose plus the stored camera center) to
    // float independently, promotes both matrices to double for the explicit
    // row-major product, and rounds the product back to float.  Those two
    // input rounding boundaries are required for byte-exact South records.
    const auto packed_world_to_camera = [](const Camera& camera) {
        return std::array<float, 16>{
            static_cast<float>(camera.pose.rotation(0, 0)),
            static_cast<float>(camera.pose.rotation(0, 1)),
            static_cast<float>(camera.pose.rotation(0, 2)),
            static_cast<float>(camera.pose.translation.x),
            static_cast<float>(camera.pose.rotation(1, 0)),
            static_cast<float>(camera.pose.rotation(1, 1)),
            static_cast<float>(camera.pose.rotation(1, 2)),
            static_cast<float>(camera.pose.translation.y),
            static_cast<float>(camera.pose.rotation(2, 0)),
            static_cast<float>(camera.pose.rotation(2, 1)),
            static_cast<float>(camera.pose.rotation(2, 2)),
            static_cast<float>(camera.pose.translation.z),
            0.0F, 0.0F, 0.0F, 1.0F};
    };
    const std::array<float, 16> neighbor_base =
        packed_world_to_camera(neighbor);
    const std::array<float, 16> reference_inverse{
        static_cast<float>(reference.pose.rotation(0, 0)),
        static_cast<float>(reference.pose.rotation(1, 0)),
        static_cast<float>(reference.pose.rotation(2, 0)),
        static_cast<float>(reference.center.x),
        static_cast<float>(reference.pose.rotation(0, 1)),
        static_cast<float>(reference.pose.rotation(1, 1)),
        static_cast<float>(reference.pose.rotation(2, 1)),
        static_cast<float>(reference.center.y),
        static_cast<float>(reference.pose.rotation(0, 2)),
        static_cast<float>(reference.pose.rotation(1, 2)),
        static_cast<float>(reference.pose.rotation(2, 2)),
        static_cast<float>(reference.center.z),
        0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 16> result{};
    for (std::size_t row = 0; row < 4U; ++row) {
        for (std::size_t column = 0; column < 4U; ++column) {
            const double value =
                static_cast<double>(neighbor_base[row * 4U]) *
                    static_cast<double>(reference_inverse[column]) +
                static_cast<double>(neighbor_base[row * 4U + 1U]) *
                    static_cast<double>(reference_inverse[4U + column]) +
                static_cast<double>(neighbor_base[row * 4U + 2U]) *
                    static_cast<double>(reference_inverse[8U + column]) +
                static_cast<double>(neighbor_base[row * 4U + 3U]) *
                    static_cast<double>(reference_inverse[12U + column]);
            result[row * 4U + column] = static_cast<float>(value);
        }
    }
    return result;
}

PatchMatchCamera make_patchmatch_perspective_camera(const Camera& camera,
                                                    int downscale) {
    if (downscale < 2 || (downscale & (downscale - 1)) != 0)
        throw std::runtime_error(
            "recovered PatchMatch packed-scale contract requires power-of-two downscale >= 2");
    PatchMatchCamera result;
    result.f = static_cast<float>(camera.model.f);
    result.cx = static_cast<float>(camera.model.cx -
                                   static_cast<double>(camera.image.width) * 0.5);
    result.cy = static_cast<float>(camera.model.cy -
                                   static_cast<double>(camera.image.height) * 0.5);
    result.width_original = static_cast<std::uint32_t>(camera.image.width);
    result.height_original = static_cast<std::uint32_t>(camera.image.height);
    result.pyramid_level0_downscale = static_cast<std::uint32_t>(downscale / 2);
    result.b1 = static_cast<float>(camera.model.b1);
    result.b2 = static_cast<float>(camera.model.b2);
    result.type = 0;
    result.transform = identity4();
    return result;
}

PatchMatchCamera make_patchmatch_reference_camera(const Camera& camera,
                                                  int downscale) {
    PatchMatchCamera result =
        make_patchmatch_perspective_camera(camera, downscale);
    result.transform = {
        static_cast<float>(camera.pose.rotation(0, 0)),
        static_cast<float>(camera.pose.rotation(1, 0)),
        static_cast<float>(camera.pose.rotation(2, 0)),
        static_cast<float>(camera.center.x),
        static_cast<float>(camera.pose.rotation(0, 1)),
        static_cast<float>(camera.pose.rotation(1, 1)),
        static_cast<float>(camera.pose.rotation(2, 1)),
        static_cast<float>(camera.center.y),
        static_cast<float>(camera.pose.rotation(0, 2)),
        static_cast<float>(camera.pose.rotation(1, 2)),
        static_cast<float>(camera.pose.rotation(2, 2)),
        static_cast<float>(camera.center.z),
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    return result;
}

PatchMatchNormalRotationCameras make_patchmatch_normal_rotation_cameras(
    const PatchMatchCamera& reference_camera) {
    // 0x25A1070 first promotes the reference camera's already-rounded float
    // transform to double and runs its explicit 4x4 cofactor inverse.  It
    // normalizes the first three inverse columns in double before constructing a
    // transform-only float camera.  The inverse camera is then produced by a
    // second execution of the same cofactor routine on that float matrix.
    std::array<double, 16> inverse =
        target_inverse_double_from_float(reference_camera.transform);
    for (std::size_t column = 0; column < 3U; ++column) {
        const double squared =
            inverse[column] * inverse[column] +
            inverse[4U + column] * inverse[4U + column] +
            inverse[8U + column] * inverse[8U + column];
        if (squared > 0.0) {
            const double reciprocal = 1.0 / std::sqrt(squared);
            inverse[column] *= reciprocal;
            inverse[4U + column] *= reciprocal;
            inverse[8U + column] *= reciprocal;
        }
    }

    PatchMatchNormalRotationCameras result;
    result.before.transform = {
        static_cast<float>(inverse[0]),
        static_cast<float>(inverse[1]),
        static_cast<float>(inverse[2]), 0.0F,
        static_cast<float>(inverse[4]),
        static_cast<float>(inverse[5]),
        static_cast<float>(inverse[6]), 0.0F,
        static_cast<float>(inverse[8]),
        static_cast<float>(inverse[9]),
        static_cast<float>(inverse[10]), 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    const std::array<double, 16> restored =
        target_inverse_double_from_float(result.before.transform);
    for (std::size_t index = 0; index < restored.size(); ++index)
        result.after.transform[index] = static_cast<float>(restored[index]);
    return result;
}

std::array<float, 12> make_patchmatch_propagation_rotation(
    const PatchMatchCamera& reference_camera) {
    const PatchMatchNormalRotationCameras rotations =
        make_patchmatch_normal_rotation_cameras(reference_camera);
    std::array<float, 12> result{};
    std::copy_n(rotations.before.transform.begin(), result.size(),
                result.begin());
    return result;
}

DepthVotingCalibrationCu make_depth_voting_perspective_calibration(
    const Camera& camera,
    std::uint32_t downscale,
    std::uint32_t pyramid_level) {
    if (camera.image.width == 0 || camera.image.height == 0)
        throw std::invalid_argument(
            "depth-voting calibration requires non-zero image dimensions");
    if (downscale < 2 || (downscale & (downscale - 1U)) != 0U)
        throw std::invalid_argument(
            "recovered depth-voting calibration requires a power-of-two downscale >= 2");
    if (pyramid_level >= 3U)
        throw std::invalid_argument(
            "recovered depth-voting calibration has exactly three voting levels");
    if (downscale > (std::numeric_limits<std::uint32_t>::max() >> pyramid_level))
        throw std::overflow_error("depth-voting calibration scale overflow");
    const std::uint32_t scale = downscale << pyramid_level;
    if (camera.image.width % scale != 0U || camera.image.height % scale != 0U)
        throw std::invalid_argument(
            "odd/remainder depth-voting calibration dimensions are not yet dynamically verified");

    const std::uint32_t width =
        static_cast<std::uint32_t>(camera.image.width / scale);
    const std::uint32_t height =
        static_cast<std::uint32_t>(camera.image.height / scale);
    DepthVotingCalibrationCu result{};

    // The target's rectified perspective calibration has no distortion.  The
    // wrapper copies max_radius_squared and the valid projection bounds from
    // the high-level calibration at +0xB0 and +0xB8..+0xD0 respectively.
    // South d4 records establish the exact constants below at all three levels.
    store_opaque(result.bytes, 32U, 1.0e9F);
    store_opaque(result.bytes, 64U, std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 68U, std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 72U, -std::numeric_limits<float>::infinity());
    store_opaque(result.bytes, 76U, -std::numeric_limits<float>::infinity());

    store_opaque(result.bytes, 80U, std::uint32_t{1});
    store_opaque(result.bytes, 84U, width);
    store_opaque(result.bytes, 88U, height);
    const double scale_double = static_cast<double>(scale);
    store_opaque(result.bytes, 92U,
                 static_cast<float>(camera.model.f / scale_double));
    store_opaque(result.bytes, 96U,
                 static_cast<float>(camera.model.cx / scale_double -
                                    static_cast<double>(width) * 0.5));
    store_opaque(result.bytes, 100U,
                 static_cast<float>(camera.model.cy / scale_double -
                                    static_cast<double>(height) * 0.5));
    store_opaque(result.bytes, 104U, 0.0F);
    store_opaque(result.bytes, 108U, 0.0F);
    store_opaque(result.bytes, 112U, std::uint8_t{1});
    store_opaque(result.bytes, 113U, std::uint8_t{0});
    return result;
}

DepthVotingMatrix4x4f make_depth_voting_camera_to_world_transform(
    const Camera& camera) {
    DepthVotingMatrix4x4f result;
    result.values = {
        static_cast<float>(camera.pose.rotation(0, 0)),
        static_cast<float>(camera.pose.rotation(1, 0)),
        static_cast<float>(camera.pose.rotation(2, 0)),
        static_cast<float>(camera.center.x),
        static_cast<float>(camera.pose.rotation(0, 1)),
        static_cast<float>(camera.pose.rotation(1, 1)),
        static_cast<float>(camera.pose.rotation(2, 1)),
        static_cast<float>(camera.center.y),
        static_cast<float>(camera.pose.rotation(0, 2)),
        static_cast<float>(camera.pose.rotation(1, 2)),
        static_cast<float>(camera.pose.rotation(2, 2)),
        static_cast<float>(camera.center.z),
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    return result;
}

DepthVotingMatrix4x4f make_depth_voting_camera_transform(
    const Camera& from,
    const Camera& to) {
    // The depth-voting host path has a different precision boundary from the
    // compact PatchMatch-camera builder: it composes the original double
    // poses first and casts the finished rigid transform to float.  South
    // direct/occlusion launch records distinguish this path by one or more
    // ULPs, so do not share patchmatch_reference_to_neighbor_transform here.
    const metalign::Mat3 rotation =
        to.pose.rotation * metalign::transpose(from.pose.rotation);
    const metalign::Vec3 translation =
        to.pose.translation - rotation * from.pose.translation;
    DepthVotingMatrix4x4f result;
    result.values = {
        static_cast<float>(rotation(0, 0)),
        static_cast<float>(rotation(0, 1)),
        static_cast<float>(rotation(0, 2)),
        static_cast<float>(translation.x),
        static_cast<float>(rotation(1, 0)),
        static_cast<float>(rotation(1, 1)),
        static_cast<float>(rotation(1, 2)),
        static_cast<float>(translation.y),
        static_cast<float>(rotation(2, 0)),
        static_cast<float>(rotation(2, 1)),
        static_cast<float>(rotation(2, 2)),
        static_cast<float>(translation.z),
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    return result;
}

int depth_vote_weight(bool inlier, DepthVoteRelation relation) {
    if (relation == DepthVoteRelation::Occludes)
        return inlier ? (-50 / 4) : (-16 / 4);
    if (inlier)
        return relation == DepthVoteRelation::Supports ? 100 : -50;
    switch (relation) {
        case DepthVoteRelation::Supports: return 15;
        case DepthVoteRelation::Intersects: return -15;
        case DepthVoteRelation::DoesNotReach: return -5;
        case DepthVoteRelation::NoDepth: return 0;
        case DepthVoteRelation::Occludes: break;
    }
    return 0;
}

DepthVotingClass classify_depth_voting(float depth, int vote_sum) {
    if (depth == 0.0F) return DepthVotingClass::Empty;
    if (vote_sum <= 0) return DepthVotingClass::Bad;
    if (vote_sum <= 199) return DepthVotingClass::Normal;
    return DepthVotingClass::Good;
}

std::size_t filter_recovered_small_depth_components(
    std::vector<float>& depth,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t component_size_threshold) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (width == 0 || height == 0 || depth.size() != pixels)
        throw std::invalid_argument("invalid recovered depth-component image");
    if (component_size_threshold == 0) return 0;

    std::vector<std::uint32_t> parent(pixels);
    std::vector<std::uint32_t> rank(pixels, 0);
    std::vector<std::uint32_t> size(pixels, 1);
    std::iota(parent.begin(), parent.end(), std::uint32_t{0});
    const auto active = [&](std::size_t index) { return depth[index] != 0.0F; };
    const auto find_root = [&](std::uint32_t value, const auto& self) -> std::uint32_t {
        const std::uint32_t next = parent[value];
        if (next == value) return value;
        parent[value] = self(next, self);
        return parent[value];
    };
    const auto join = [&](std::uint32_t a, std::uint32_t b) {
        a = find_root(a, find_root);
        b = find_root(b, find_root);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        if (rank[a] == rank[b]) ++rank[a];
    };

    // The target worker slides a four-neighbour predecessor window containing
    // left, upper-left, upper and upper-right.  This is exactly 8-connectivity
    // without duplicate unions.
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            if (!active(index)) continue;
            const auto current = static_cast<std::uint32_t>(index);
            if (x != 0 && active(index - 1))
                join(current, current - 1);
            if (y == 0) continue;
            const std::size_t upper = index - width;
            if (x != 0 && active(upper - 1))
                join(current, static_cast<std::uint32_t>(upper - 1));
            if (active(upper))
                join(current, static_cast<std::uint32_t>(upper));
            if (x + 1 < width && active(upper + 1))
                join(current, static_cast<std::uint32_t>(upper + 1));
        }
    }

    std::size_t cleared = 0;
    for (std::size_t index = 0; index < pixels; ++index) {
        if (!active(index)) continue;
        const auto root = find_root(static_cast<std::uint32_t>(index), find_root);
        if (size[root] <= component_size_threshold) {
            depth[index] = 0.0F;
            ++cleared;
        }
    }
    return cleared;
}

#ifdef METMODEL_HAS_CUDA
bool begin_recovered_cuda_module_session_impl(
    std::size_t device_index,
    std::string& error);
bool end_recovered_cuda_module_session_impl(
    RecoveredCudaModuleSessionStats& stats,
    std::string& error);
bool run_recovered_patchmatch_undistort_u8_cuda_impl(
    const PatchMatchUndistortU8Input& input,
    RecoveredPatchMatchImageU8& output,
    std::string& error);
bool run_recovered_patchmatch_refinement_cuda_impl(
    PatchMatchRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_final_refinement_cuda_impl(
    PatchMatchFinalRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::vector<float>* updated_cost,
    std::string& error);
bool run_recovered_patchmatch_wta_cuda_impl(
    PatchMatchWtaInput& input,
    PatchMatchWtaOutput& output,
    std::string& error);
bool run_recovered_patchmatch_copy_inlier_masks_cuda_impl(
    PatchMatchCopyInlierMasksInput& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error);
bool run_recovered_patchmatch_propagation_cuda_impl(
    PatchMatchPropagationInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_coarse_to_precise_cuda_impl(
    PatchMatchCoarseToPreciseInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_cost_cuda_impl(
    PatchMatchCostInput& input,
    PatchMatchCostOutput& output,
    std::string& error);
bool run_recovered_patchmatch_bilateral_u8_cuda_impl(
    const PatchMatchBilateralU8Input& input,
    PatchMatchBilateralU8Output& output,
    std::string& error);
bool run_recovered_patchmatch_filter_check_cost_cuda_impl(
    const PatchMatchFilterCheckCostInput& input,
    PatchMatchFilterCheckCostOutput& output,
    std::string& error);
bool run_recovered_patchmatch_filter_check_neighbours_cuda_impl(
    const PatchMatchFilterCheckNeighboursInput& input,
    PatchMatchFilterCheckNeighboursOutput& output,
    std::string& error);
bool run_recovered_patchmatch_filter_clear_depth_cuda_impl(
    const PatchMatchFilterClearDepthInput& input,
    PatchMatchFilterClearDepthOutput& output,
    std::string& error);
bool run_recovered_patchmatch_filter_normals_cuda_impl(
    const PatchMatchFilterNormalsInput& input,
    PatchMatchFilterNormalsOutput& output,
    std::string& error);
bool run_recovered_patchmatch_filter_speckles_edges_cuda_impl(
    const PatchMatchFilterSpecklesEdgesInput& input,
    PatchMatchFilterSpecklesEdgesOutput& output,
    std::string& error);
bool run_recovered_patchmatch_filter_chain_cuda_impl(
    const PatchMatchFilterChainInput& input,
    PatchMatchFilterChainOutput& output,
    std::string& error);
bool run_recovered_depth_voting_finalize_cuda_impl(
    const DepthVotingFinalizeInput& input,
    DepthVotingFinalizeOutput& output,
    std::string& error);
bool run_recovered_depth_voting_chain_cuda_impl(
    const DepthVotingChainInput& input,
    DepthVotingChainOutput& output,
    std::string& error);
bool prime_recovered_cuda_voting_depth_cache_impl(
    std::span<const std::span<const float>> depth_levels,
    std::size_t device_index,
    std::string& error);
bool clear_recovered_cuda_voting_depth_cache_impl(
    std::size_t device_index,
    std::string& error);
bool run_recovered_depth_radius_estimate_cuda_impl(
    const DepthRadiusEstimateInput& input,
    std::vector<float>& radius_output,
    std::string& error);
bool run_recovered_depth_neighbor_votes_cuda_impl(
    const DepthNeighborVotesInput& input,
    DepthNeighborVotesOutput& output,
    std::string& error);
bool run_recovered_depth_neighbor_occlusion_votes_cuda_impl(
    const DepthNeighborOcclusionVotesInput& input,
    DepthNeighborOcclusionVotesOutput& output,
    std::string& error);
#endif

bool begin_recovered_cuda_module_session(
    std::size_t device_index,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return begin_recovered_cuda_module_session_impl(device_index, error);
#else
    (void)device_index;
    error = "CUDA support was not compiled";
    return false;
#endif
}

bool prime_recovered_cuda_voting_depth_cache(
    std::span<const std::span<const float>> depth_levels,
    std::size_t device_index,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return prime_recovered_cuda_voting_depth_cache_impl(
        depth_levels, device_index, error);
#else
    (void)depth_levels;
    (void)device_index;
    error = "CUDA support was not compiled";
    return false;
#endif
}

bool clear_recovered_cuda_voting_depth_cache(
    std::size_t device_index,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return clear_recovered_cuda_voting_depth_cache_impl(device_index, error);
#else
    (void)device_index;
    error = "CUDA support was not compiled";
    return false;
#endif
}

bool end_recovered_cuda_module_session(
    RecoveredCudaModuleSessionStats& stats,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return end_recovered_cuda_module_session_impl(stats, error);
#else
    (void)stats;
    error = "CUDA support was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_undistort_u8_cuda(
    const PatchMatchUndistortU8Input& input,
    RecoveredPatchMatchImageU8& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_undistort_u8_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "CUDA support was not compiled";
    return false;
#endif
}

bool make_recovered_patchmatch_unmasked_camera_preparation(
    const Scene& scene,
    std::size_t camera_index,
    std::uint32_t target_downscale,
    std::size_t device_index,
    RecoveredPatchMatchPreparedCamera& output,
    std::string& error,
    std::uint32_t minimum_retained_downscale) {
    if (camera_index >= scene.cameras.size()) {
        error = "PatchMatch preparation camera index is out of range";
        return false;
    }
    if (target_downscale < 2U || target_downscale > 32U ||
        (target_downscale & (target_downscale - 1U)) != 0U) {
        error = "PatchMatch preparation requires power-of-two downscale 2..32";
        return false;
    }
    const std::uint32_t required_base_downscale = target_downscale / 2U;
    if (minimum_retained_downscale == 0U ||
        minimum_retained_downscale > required_base_downscale ||
        (minimum_retained_downscale &
         (minimum_retained_downscale - 1U)) != 0U) {
        error = "PatchMatch preparation retained downscale must be a power of two no larger than the target atlas base";
        return false;
    }
    const Camera& camera = scene.cameras[camera_index];
    if (!camera.aligned) {
        error = "PatchMatch preparation requires an aligned camera";
        return false;
    }
    const std::size_t camera_pixels = camera.image.width * camera.image.height;
    const bool has_rgb = camera.image.rgb.size() == camera_pixels * 3U;
    const bool has_gray = camera.image.gray.size() == camera_pixels;
    const bool metadata_only = camera.image.rgb.empty() &&
                               camera.image.gray.empty() &&
                               std::filesystem::is_regular_file(camera.path);
    if (camera.image.width == 0U || camera.image.height == 0U ||
        (!has_rgb && !has_gray && !metadata_only)) {
        error = "PatchMatch preparation requires RGB8, canonical grayscale, or lazy image metadata";
        return false;
    }
    if (camera.image.width > std::numeric_limits<std::uint32_t>::max() ||
        camera.image.height > std::numeric_limits<std::uint32_t>::max()) {
        error = "PatchMatch preparation image dimensions exceed uint32";
        return false;
    }

    RecoveredPatchMatchPreparedCamera prepared;
    prepared.camera_index = camera_index;
    prepared.target_downscale = target_downscale;
    prepared.device_index = device_index;
    try {
        prepared.depth_range = make_recovered_patchmatch_depth_range(
            scene, camera_index);
        prepared.source_calibration =
            make_recovered_patchmatch_source_calibration(camera);
        prepared.undistorted_calibration =
            make_recovered_patchmatch_undistorted_calibration(camera);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }

    std::optional<metalign::Image> decoded_image;
    const metalign::Image* source_image = &camera.image;
    if (metadata_only) {
        try {
            decoded_image = metalign::load_gray_image(camera.path);
        } catch (const std::exception& exception) {
            error = "lazy PatchMatch image decode failed: " +
                    std::string(exception.what());
            return false;
        }
        if (decoded_image->width != camera.image.width ||
            decoded_image->height != camera.image.height) {
            error = "lazy PatchMatch image dimensions changed after scene load";
            return false;
        }
        source_image = &*decoded_image;
    }

    PatchMatchUndistortU8Input undistort;
    undistort.width = static_cast<std::uint32_t>(source_image->width);
    undistort.height = static_cast<std::uint32_t>(source_image->height);
    try {
        undistort.source_image = make_recovered_patchmatch_source_u8(*source_image);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    // South and the currently proven frame path have no explicit source mask.
    // This must remain an empty allocation: the target ABI passes with_mask=0
    // and a null source-mask pointer.  A zero-filled allocation would instead
    // mean with_mask=1 and reject every pixel whose mask byte is zero.
    undistort.source_calibration = prepared.source_calibration;
    undistort.target_calibration = prepared.undistorted_calibration;
    undistort.device_index = device_index;

    RecoveredPatchMatchImageU8 level;
    if (!run_recovered_patchmatch_undistort_u8_cuda(undistort, level, error))
        return false;
    try {
        for (std::uint32_t downscale = 1U;; downscale *= 2U) {
            if (downscale >= minimum_retained_downscale) {
                RecoveredPatchMatchPreparedLevel entry;
                entry.downscale = downscale;
                entry.deviation_ratio =
                    reduce_recovered_patchmatch_deviation_multiplier(
                        level.image);
                entry.data = level;
                prepared.image_levels.push_back(std::move(entry));
            } else if (std::any_of(
                           level.rejection_mask.begin(),
                           level.rejection_mask.end(),
                           [](std::uint8_t value) { return value != 0U; })) {
                error = "PatchMatch preparation cannot discard a pre-base level with a non-zero rejection mask";
                return false;
            }
            if (downscale == 32U) break;
            level = reduce_recovered_patchmatch_image_half(level);
        }
        prepared.camera = make_patchmatch_perspective_camera(
            camera, static_cast<int>(target_downscale));
        prepared.valid = true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    output = std::move(prepared);
    error.clear();
    return true;
}

namespace {

struct CrossLevelPoint3f {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

std::array<float, 3> decode_recovered_patchmatch_normal(
    const std::uint8_t* encoded) {
    std::array<float, 3> result{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const float component = static_cast<float>(encoded[axis]);
        result[axis] = (component + component) / 255.0F - 1.0F;
    }
    float squared = result[0] * result[0] + result[1] * result[1];
    squared = squared + result[2] * result[2];
    const float length = std::sqrt(squared);
    const float inverse = length >= 1.0e-20F ? 1.0F / length : 0.0F;
    for (float& component : result) component *= inverse;
    return result;
}

std::array<std::uint8_t, 3> encode_recovered_patchmatch_normal(
    const std::array<float, 3>& normal) {
    std::array<std::uint8_t, 3> result{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const float component = std::clamp(normal[axis], -1.0F, 1.0F);
        result[axis] = static_cast<std::uint8_t>(
            (component + 1.0F) * 255.0F * 0.5F + 0.5F);
    }
    return result;
}

bool cross_level_point_invalid(const CrossLevelPoint3f& point) {
    return point.x == 0.0F && point.y == 0.0F && point.z == 0.0F;
}

float cross_level_squared_distance(const CrossLevelPoint3f& first,
                                   const CrossLevelPoint3f& second,
                                   bool half_weight) {
    const float dx = second.x - first.x;
    const float dy = second.y - first.y;
    const float dz = second.z - first.z;
    float squared = dx * dx + dy * dy;
    squared = squared + dz * dz;
    if (half_weight) squared *= 0.5F;
    return squared;
}

bool make_recovered_patchmatch_host_geometry(
    const Camera& camera,
    std::uint32_t depth_downscale,
    std::uint32_t width,
    std::uint32_t height,
    std::span<const float> depth,
    std::vector<CrossLevelPoint3f>& points,
    std::vector<float>& radius,
    std::string& error) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (!camera.aligned || width == 0U || height == 0U ||
        depth_downscale == 0U || depth.size() != pixels ||
        camera.model.f == 0.0 ||
        camera.model.f + camera.model.b1 == 0.0) {
        error = "PatchMatch host geometry input is invalid";
        return false;
    }

    points.assign(pixels, {});
    const double downscale = static_cast<double>(depth_downscale);
    const float downscale_float = static_cast<float>(depth_downscale);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * width + x;
            const float current_depth = depth[index];
            if (current_depth == 0.0F) continue;

            const double pixel_x = (static_cast<double>(x) + 0.5) * downscale;
            const double pixel_y = (static_cast<double>(y) + 0.5) * downscale;
            const double local_y =
                (pixel_y - camera.model.cy) / camera.model.f;
            const double local_x =
                (pixel_x - camera.model.cx - camera.model.b2 * local_y) /
                (camera.model.f + camera.model.b1);
            const double local_z =
                static_cast<double>(current_depth / downscale_float) * downscale;
            const double camera_x = local_x * local_z;
            const double camera_y = local_y * local_z;

            const double world_x =
                camera.pose.rotation(0, 0) * camera_x +
                camera.pose.rotation(1, 0) * camera_y +
                camera.pose.rotation(2, 0) * local_z + camera.center.x;
            const double world_y =
                camera.pose.rotation(0, 1) * camera_x +
                camera.pose.rotation(1, 1) * camera_y +
                camera.pose.rotation(2, 1) * local_z + camera.center.y;
            const double world_z =
                camera.pose.rotation(0, 2) * camera_x +
                camera.pose.rotation(1, 2) * camera_y +
                camera.pose.rotation(2, 2) * local_z + camera.center.z;
            points[index] = {
                static_cast<float>(world_x), static_cast<float>(world_y),
                static_cast<float>(world_z)};
        }
    }

    radius.assign(pixels, 0.0F);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t center_index =
                static_cast<std::size_t>(y) * width + x;
            if (cross_level_point_invalid(points[center_index])) continue;

            std::array<float, 42> distances{};
            std::size_t count = 0;
            const std::uint32_t first_x = x == 0U ? 0U : x - 1U;
            const std::uint32_t last_x =
                std::min<std::uint32_t>(width - 1U, x + 1U);
            const std::uint32_t first_y = y == 0U ? 0U : y - 1U;
            const std::uint32_t last_y =
                std::min<std::uint32_t>(height - 1U, y + 1U);
            for (std::uint32_t row = first_y; row <= last_y; ++row) {
                for (std::uint32_t column = first_x; column <= last_x;
                     ++column) {
                    const CrossLevelPoint3f& point = points[
                        static_cast<std::size_t>(row) * width + column];
                    if (cross_level_point_invalid(point)) continue;
                    if (column < last_x) {
                        const CrossLevelPoint3f& right = points[
                            static_cast<std::size_t>(row) * width + column + 1U];
                        if (!cross_level_point_invalid(right))
                            distances[count++] = cross_level_squared_distance(
                                point, right, false);
                    }
                    if (row < last_y) {
                        const CrossLevelPoint3f& below = points[
                            static_cast<std::size_t>(row + 1U) * width + column];
                        if (!cross_level_point_invalid(below))
                            distances[count++] = cross_level_squared_distance(
                                point, below, false);
                        if (column < last_x) {
                            const CrossLevelPoint3f& diagonal = points[
                                static_cast<std::size_t>(row + 1U) * width +
                                column + 1U];
                            if (!cross_level_point_invalid(diagonal))
                                distances[count++] = cross_level_squared_distance(
                                    point, diagonal, true);
                        }
                    }
                }
            }
            if (count > 1U) {
                const std::size_t selected = 3U * count / 10U;
                std::nth_element(distances.begin(),
                                 distances.begin() +
                                     static_cast<std::ptrdiff_t>(selected),
                                 distances.begin() +
                                     static_cast<std::ptrdiff_t>(count));
                radius[center_index] = std::sqrt(distances[selected]);
            }
        }
    }
    error.clear();
    return true;
}

}  // namespace

bool filter_recovered_patchmatch_speckle_components(
    const Camera& camera,
    std::uint32_t depth_downscale,
    std::vector<float>& depth,
    std::uint32_t component_size_threshold,
    std::string& error) {
    if (camera.image.width == 0U || camera.image.height == 0U ||
        camera.image.width > std::numeric_limits<std::uint32_t>::max() ||
        camera.image.height > std::numeric_limits<std::uint32_t>::max() ||
        depth_downscale == 0U) {
        error = "PatchMatch speckle component camera grid is invalid";
        return false;
    }
    const auto width = ceil_div(
        static_cast<std::uint32_t>(camera.image.width), depth_downscale);
    const auto height = ceil_div(
        static_cast<std::uint32_t>(camera.image.height), depth_downscale);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (depth.size() < pixels) {
        error = "PatchMatch speckle component depth allocation is too small";
        return false;
    }
    if (component_size_threshold <= 1U) {
        error.clear();
        return true;
    }

    std::vector<CrossLevelPoint3f> points;
    std::vector<float> radius;
    if (!make_recovered_patchmatch_host_geometry(
            camera, depth_downscale, width, height,
            std::span<const float>(depth.data(), pixels),
            points, radius, error))
        return false;

    std::vector<std::uint32_t> parent(pixels);
    std::vector<std::uint32_t> rank(pixels, 0U);
    std::vector<std::uint32_t> component_size(pixels, 1U);
    std::iota(parent.begin(), parent.end(), std::uint32_t{0});
    const auto find_root = [&](std::uint32_t value,
                               const auto& self) -> std::uint32_t {
        const std::uint32_t next = parent[value];
        if (next == value) return value;
        parent[value] = self(next, self);
        return parent[value];
    };
    const auto join = [&](std::uint32_t first, std::uint32_t second) {
        first = find_root(first, find_root);
        second = find_root(second, find_root);
        if (first == second) return;
        if (rank[first] < rank[second]) std::swap(first, second);
        parent[second] = first;
        component_size[first] += component_size[second];
        if (rank[first] == rank[second]) ++rank[first];
    };

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * width + x;
            if (depth[index] == 0.0F) continue;
            const CrossLevelPoint3f& point = points[index];
            const float threshold = radius[index] + radius[index];
            const float threshold_squared = threshold * threshold;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::int64_t neighbor_y =
                    static_cast<std::int64_t>(y) + dy;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const std::int64_t neighbor_x =
                        static_cast<std::int64_t>(x) + dx;
                    if (neighbor_x < 0 || neighbor_y < 0 ||
                        neighbor_x >= static_cast<std::int64_t>(width) ||
                        neighbor_y >= static_cast<std::int64_t>(height))
                        continue;
                    const std::size_t neighbor_index =
                        static_cast<std::size_t>(neighbor_y) * width +
                        static_cast<std::size_t>(neighbor_x);
                    if (depth[neighbor_index] == 0.0F) continue;
                    const float neighbor_threshold =
                        radius[neighbor_index] + radius[neighbor_index];
                    if (!(threshold <= neighbor_threshold)) continue;
                    const CrossLevelPoint3f& neighbor = points[neighbor_index];
                    const float px = neighbor.x - point.x;
                    const float py = neighbor.y - point.y;
                    const float pz = neighbor.z - point.z;
                    float distance_squared = px * px + py * py;
                    distance_squared = distance_squared + pz * pz;
                    const float limit = dx == 0 || dy == 0
                        ? threshold_squared
                        : threshold_squared + threshold_squared;
                    if (!(distance_squared < limit)) continue;
                    join(static_cast<std::uint32_t>(index),
                         static_cast<std::uint32_t>(neighbor_index));
                }
            }
        }
    }

    for (std::size_t index = 0; index < pixels; ++index) {
        if (depth[index] == 0.0F) continue;
        const std::uint32_t root =
            find_root(static_cast<std::uint32_t>(index), find_root);
        if (component_size[root] <= component_size_threshold)
            depth[index] = 0.0F;
    }
    error.clear();
    return true;
}

bool filter_recovered_patchmatch_cuda_speckle_components(
    std::uint32_t width,
    std::uint32_t height,
    std::span<const std::uint8_t> filtered_mask,
    std::vector<float>& depth,
    std::uint32_t component_size_threshold,
    std::string& error) {
    if (width == 0U || height == 0U) {
        error = "PatchMatch CUDA speckle component grid is empty";
        return false;
    }
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (filtered_mask.size() < pixels || depth.size() < pixels) {
        error = "PatchMatch CUDA speckle component allocation is too small";
        return false;
    }
    if (component_size_threshold <= 1U) {
        error.clear();
        return true;
    }

    std::vector<std::uint32_t> parent(pixels);
    std::vector<std::uint32_t> rank(pixels, 0U);
    std::vector<std::uint32_t> component_size(pixels, 1U);
    std::iota(parent.begin(), parent.end(), std::uint32_t{0});
    const auto find_root = [&](std::uint32_t value,
                               const auto& self) -> std::uint32_t {
        const std::uint32_t next = parent[value];
        if (next == value) return value;
        parent[value] = self(next, self);
        return parent[value];
    };
    const auto join = [&](std::uint32_t first, std::uint32_t second) {
        first = find_root(first, find_root);
        second = find_root(second, find_root);
        if (first == second) return;
        if (rank[first] < rank[second]) std::swap(first, second);
        parent[second] = first;
        component_size[first] += component_size[second];
        if (rank[first] == rank[second]) ++rank[first];
    };

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            const std::uint8_t edges = filtered_mask[index];
            if (edges == 0U) continue;
            unsigned int bit = 0U;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::int64_t neighbor_y =
                    static_cast<std::int64_t>(y) + dy;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const std::int64_t neighbor_x =
                        static_cast<std::int64_t>(x) + dx;
                    if (neighbor_x < 0 || neighbor_y < 0 ||
                        neighbor_x >= static_cast<std::int64_t>(width) ||
                        neighbor_y >= static_cast<std::int64_t>(height))
                        continue;
                    if ((edges & static_cast<std::uint8_t>(1U << bit)) != 0U) {
                        const std::size_t neighbor_index =
                            static_cast<std::size_t>(neighbor_y) * width +
                            static_cast<std::size_t>(neighbor_x);
                        join(static_cast<std::uint32_t>(index),
                             static_cast<std::uint32_t>(neighbor_index));
                    }
                    ++bit;
                }
            }
        }
    }

    for (std::size_t index = 0; index < pixels; ++index) {
        if (depth[index] == 0.0F) continue;
        const std::uint32_t root =
            find_root(static_cast<std::uint32_t>(index), find_root);
        if (component_size[root] <= component_size_threshold)
            depth[index] = 0.0F;
    }
    error.clear();
    return true;
}

bool make_recovered_patchmatch_cross_level_state(
    const Camera& camera,
    std::uint32_t depth_downscale,
    std::uint32_t previous_width,
    std::uint32_t previous_height,
    std::span<const float> previous_depth,
    std::span<const std::uint8_t> previous_normal,
    RecoveredPatchMatchCrossLevelState& output,
    std::string& error) {
    if (!camera.aligned || camera.image.width == 0U ||
        camera.image.height == 0U) {
        error = "PatchMatch cross-level state requires an aligned non-empty camera";
        return false;
    }
    if (camera.image.width > std::numeric_limits<std::uint32_t>::max() ||
        camera.image.height > std::numeric_limits<std::uint32_t>::max()) {
        error = "PatchMatch cross-level image dimensions exceed uint32";
        return false;
    }
    if (depth_downscale < 2U || depth_downscale > 16U ||
        (depth_downscale & (depth_downscale - 1U)) != 0U) {
        error = "PatchMatch cross-level state requires power-of-two downscale 2..16";
        return false;
    }
    if (camera.model.f == 0.0 ||
        camera.model.f + camera.model.b1 == 0.0) {
        error = "PatchMatch cross-level perspective calibration is singular";
        return false;
    }

    const auto original_width = static_cast<std::uint32_t>(camera.image.width);
    const auto original_height = static_cast<std::uint32_t>(camera.image.height);
    const std::uint32_t previous_downscale = depth_downscale * 2U;
    const std::uint32_t expected_previous_width =
        ceil_div(original_width, previous_downscale);
    const std::uint32_t expected_previous_height =
        ceil_div(original_height, previous_downscale);
    if (previous_width != expected_previous_width ||
        previous_height != expected_previous_height) {
        error = "PatchMatch cross-level previous dimensions do not match the camera pyramid";
        return false;
    }
    const std::size_t previous_pixels =
        static_cast<std::size_t>(previous_width) * previous_height;
    if (previous_depth.size() != previous_pixels ||
        previous_normal.size() != previous_pixels * 3U) {
        error = "PatchMatch cross-level previous buffer size mismatch";
        return false;
    }

    RecoveredPatchMatchCrossLevelState state;
    state.width = ceil_div(original_width, depth_downscale);
    state.height = ceil_div(original_height, depth_downscale);
    const std::size_t pixels =
        static_cast<std::size_t>(state.width) * state.height;
    state.depth.assign(pixels, 0.0F);
    state.normal.resize(pixels * 3U);
    state.cost.assign(pixels, -1.0F);

    for (std::uint32_t y = 0; y < state.height; ++y) {
        const double source_y =
            (static_cast<double>(y) + 0.5) * 0.5 - 0.5;
        const auto floor_y = static_cast<std::int64_t>(std::floor(source_y));
        const float fraction_y =
            static_cast<float>(source_y - static_cast<double>(floor_y));
        const std::int64_t y0 = std::clamp<std::int64_t>(
            floor_y, 0, static_cast<std::int64_t>(previous_height) - 1);
        const std::int64_t y1 = std::clamp<std::int64_t>(
            floor_y + 1, 0, static_cast<std::int64_t>(previous_height) - 1);
        for (std::uint32_t x = 0; x < state.width; ++x) {
            const double source_x =
                (static_cast<double>(x) + 0.5) * 0.5 - 0.5;
            const auto floor_x = static_cast<std::int64_t>(std::floor(source_x));
            const float fraction_x =
                static_cast<float>(source_x - static_cast<double>(floor_x));
            const std::int64_t x0 = std::clamp<std::int64_t>(
                floor_x, 0, static_cast<std::int64_t>(previous_width) - 1);
            const std::int64_t x1 = std::clamp<std::int64_t>(
                floor_x + 1, 0, static_cast<std::int64_t>(previous_width) - 1);

            const std::array<std::size_t, 4> source_indices{
                static_cast<std::size_t>(y0) * previous_width +
                    static_cast<std::size_t>(x0),
                static_cast<std::size_t>(y0) * previous_width +
                    static_cast<std::size_t>(x1),
                static_cast<std::size_t>(y1) * previous_width +
                    static_cast<std::size_t>(x0),
                static_cast<std::size_t>(y1) * previous_width +
                    static_cast<std::size_t>(x1),
            };
            const float one_minus_x = 1.0F - fraction_x;
            const float one_minus_y = 1.0F - fraction_y;
            const std::array<float, 4> weights{
                one_minus_x * one_minus_y,
                one_minus_y * fraction_x,
                one_minus_x * fraction_y,
                fraction_x * fraction_y,
            };

            float weight_sum = 0.0F;
            float depth_sum = 0.0F;
            float best_weight = -1.0F;
            std::array<float, 3> selected_normal{};
            for (std::size_t sample = 0; sample < 4U; ++sample) {
                const std::size_t source_index = source_indices[sample];
                const float source_depth = previous_depth[source_index];
                if (source_depth == 0.0F) continue;
                const float weight = weights[sample];
                weight_sum = weight_sum + weight;
                depth_sum = depth_sum + source_depth * weight;
                if (weight > best_weight) {
                    best_weight = weight;
                    selected_normal = decode_recovered_patchmatch_normal(
                        previous_normal.data() + source_index * 3U);
                }
            }

            const std::size_t index =
                static_cast<std::size_t>(y) * state.width + x;
            const float depth =
                weight_sum > 0.0F ? depth_sum / weight_sum : 0.0F;
            state.depth[index] = depth;
            state.cost[index] = depth != 0.0F ? 0.5F : -1.0F;
            if (depth == 0.0F) selected_normal = {};
            const auto encoded =
                encode_recovered_patchmatch_normal(selected_normal);
            std::copy(encoded.begin(), encoded.end(),
                      state.normal.begin() + static_cast<std::ptrdiff_t>(index * 3U));
        }
    }

    state.coarse_depth = state.depth;
    std::vector<CrossLevelPoint3f> points;
    if (!make_recovered_patchmatch_host_geometry(
            camera, depth_downscale, state.width, state.height,
            state.depth, points, state.coarse_radius, error))
        return false;

    output = std::move(state);
    error.clear();
    return true;
}

bool make_recovered_patchmatch_neighbor_crop_descriptor(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::size_t neighbor_camera_index,
    std::uint32_t target_downscale,
    RecoveredPatchMatchCropDescriptor& output,
    std::string& error) {
    if (reference_camera_index >= scene.cameras.size() ||
        neighbor_camera_index >= scene.cameras.size()) {
        error = "PatchMatch crop camera index is out of range";
        return false;
    }
    if (reference_camera_index == neighbor_camera_index) {
        error = "PatchMatch crop requires distinct cameras";
        return false;
    }
    if (target_downscale < 2U || target_downscale > 32U ||
        (target_downscale & (target_downscale - 1U)) != 0U) {
        error = "PatchMatch crop requires power-of-two downscale 2..32";
        return false;
    }
    const Camera& reference = scene.cameras[reference_camera_index];
    const Camera& neighbor = scene.cameras[neighbor_camera_index];
    if (!reference.aligned || !neighbor.aligned ||
        reference.image.width == 0U || reference.image.height == 0U ||
        neighbor.image.width == 0U || neighbor.image.height == 0U ||
        reference.image.width > std::numeric_limits<std::uint32_t>::max() ||
        reference.image.height > std::numeric_limits<std::uint32_t>::max() ||
        neighbor.image.width > std::numeric_limits<std::uint32_t>::max() ||
        neighbor.image.height > std::numeric_limits<std::uint32_t>::max()) {
        error = "PatchMatch crop requires aligned uint32-sized cameras";
        return false;
    }
    if (reference.track_ids.empty() || neighbor.track_ids.empty()) {
        error = "PatchMatch crop requires sparse track observations";
        return false;
    }
    metalign::CameraModel reference_model = reference.model;
    reference_model.k1 = 0.0; reference_model.k2 = 0.0;
    reference_model.k3 = 0.0; reference_model.k4 = 0.0;
    reference_model.p1 = 0.0; reference_model.p2 = 0.0;
    reference_model.p3 = 0.0; reference_model.p4 = 0.0;
    metalign::CameraModel neighbor_model = neighbor.model;
    neighbor_model.k1 = 0.0; neighbor_model.k2 = 0.0;
    neighbor_model.k3 = 0.0; neighbor_model.k4 = 0.0;
    neighbor_model.p1 = 0.0; neighbor_model.p2 = 0.0;
    neighbor_model.p3 = 0.0; neighbor_model.p4 = 0.0;

    std::unordered_map<std::uint32_t, const SparsePoint*> points;
    points.reserve(scene.sparse_points.size());
    for (const SparsePoint& point : scene.sparse_points) {
        if (point.track_id != std::numeric_limits<std::uint32_t>::max())
            points.emplace(point.track_id, &point);
    }

    const double scale = static_cast<double>(target_downscale);
    const std::uint64_t reference_width = ceil_div(
        static_cast<std::uint32_t>(reference.image.width), target_downscale);
    const std::uint64_t reference_height = ceil_div(
        static_cast<std::uint32_t>(reference.image.height), target_downscale);
    const double descriptor_center_x =
        (static_cast<double>(reference_width) + 0.5) * scale * 0.5;
    const double descriptor_center_y =
        (static_cast<double>(reference_height) + 0.5) * scale * 0.5;

    struct Sample { double x; double y; double depth; };
    std::vector<const SparsePoint*> common_points;
    std::vector<Sample> samples;
    std::size_t first = 0U;
    std::size_t second = 0U;
    while (first < reference.track_ids.size() &&
           second < neighbor.track_ids.size()) {
        const std::uint32_t first_id = reference.track_ids[first];
        const std::uint32_t second_id = neighbor.track_ids[second];
        if (first_id < second_id) { ++first; continue; }
        if (second_id < first_id) { ++second; continue; }
        ++first;
        ++second;
        const auto found = points.find(first_id);
        if (found == points.end()) continue;
        const SparsePoint& point = *found->second;
        common_points.push_back(&point);
        const metalign::Vec3 local =
            reference.pose.rotation * point.position + reference.pose.translation;
        if (!(local.z > 0.0) || !std::isfinite(local.z)) continue;
        const metalign::Vec2 projected =
            metalign::project(reference_model, reference.pose, point.position);
        if (!std::isfinite(projected.x) || !std::isfinite(projected.y) ||
            projected.x < 0.0 || projected.y < 0.0 ||
            projected.x >= static_cast<double>(reference.image.width) ||
            projected.y >= static_cast<double>(reference.image.height))
            continue;
        if (!((0.5 * scale) < projected.x &&
              projected.x < (static_cast<double>(reference_width) + 0.5) * scale &&
              (0.5 * scale) < projected.y &&
              projected.y < (static_cast<double>(reference_height) + 0.5) * scale))
            continue;
        samples.push_back({projected.x, projected.y, local.z});
    }

    const auto upper_median = [](std::vector<double> values) {
        auto middle = values.begin() +
            static_cast<std::ptrdiff_t>(values.size() / 2U);
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    };
    double center_x = descriptor_center_x;
    double center_y = descriptor_center_y;
    double center_depth = 0.0;
    if (samples.size() > 2U) {
        std::vector<double> depths;
        depths.reserve(samples.size());
        for (const Sample& sample : samples) {
            depths.push_back(sample.depth);
        }
        center_depth = upper_median(std::move(depths));
        // 0x1CFDA70 deliberately keeps the descriptor center for 3..24
        // in-window projections.  It replaces x/y by their independent upper
        // medians only when the strict-window population exceeds 24.
        if (samples.size() > 24U) {
            std::vector<double> xs, ys;
            xs.reserve(samples.size());
            ys.reserve(samples.size());
            for (const Sample& sample : samples) {
                xs.push_back(sample.x);
                ys.push_back(sample.y);
            }
            center_x = upper_median(std::move(xs));
            center_y = upper_median(std::move(ys));
        }
    } else {
        struct NearbyDepth { double squared_distance; double depth; };
        std::vector<NearbyDepth> nearby;
        nearby.reserve(common_points.size());
        for (const SparsePoint* point : common_points) {
            const metalign::Vec3 local = reference.pose.rotation *
                point->position + reference.pose.translation;
            if (!(local.z > 0.0) || !std::isfinite(local.z)) continue;
            const metalign::Vec2 projected = metalign::project(
                reference_model, reference.pose, point->position);
            if (!std::isfinite(projected.x) || !std::isfinite(projected.y) ||
                projected.x < 0.0 || projected.y < 0.0 ||
                projected.x >= static_cast<double>(reference.image.width) ||
                projected.y >= static_cast<double>(reference.image.height))
                continue;
            const double x = projected.x - center_x;
            const double y = projected.y - center_y;
            nearby.push_back({x * x + y * y, local.z});
        }
        if (nearby.size() <= 4U) {
            error = "PatchMatch crop fallback requires five valid common projections";
            return false;
        }
        std::sort(nearby.begin(), nearby.end(),
                  [](const NearbyDepth& left, const NearbyDepth& right) {
                      return left.squared_distance < right.squared_distance;
                  });
        std::vector<double> depths;
        depths.reserve(5U);
        for (std::size_t index = 0; index < 5U; ++index)
            depths.push_back(nearby[index].depth);
        center_depth = upper_median(std::move(depths));
    }

    const metalign::Vec3 ray = metalign::bearing(
        reference_model, {center_x, center_y});
    if (!(std::abs(ray.z) > 1.0e-15) || !std::isfinite(ray.z)) {
        error = "PatchMatch crop center cannot be unprojected";
        return false;
    }
    const metalign::Vec3 reference_local = ray * (center_depth / ray.z);
    const metalign::Vec3 world = metalign::transpose(reference.pose.rotation) *
        (reference_local - reference.pose.translation);
    const metalign::Vec3 neighbor_local =
        neighbor.pose.rotation * world + neighbor.pose.translation;
    if (!(neighbor_local.z > 0.0) || !std::isfinite(neighbor_local.z)) {
        error = "PatchMatch crop center is behind the neighbor camera";
        return false;
    }
    const metalign::Vec2 neighbor_center =
        metalign::project(neighbor_model, neighbor.pose, world);
    if (!std::isfinite(neighbor_center.x) || !std::isfinite(neighbor_center.y)) {
        error = "PatchMatch crop neighbor projection is non-finite";
        return false;
    }

    // 0x1CFDA70 receives the reference working width as its extent argument.
    const std::uint64_t expanded_extent = static_cast<std::uint64_t>(
        static_cast<double>(reference_width) * 1.5);
    const double acceptance_margin =
        static_cast<double>(expanded_extent) * 0.25;
    if (neighbor_center.x < -acceptance_margin ||
        neighbor_center.y < -acceptance_margin ||
        neighbor_center.x > static_cast<double>(neighbor.image.width) + acceptance_margin ||
        neighbor_center.y > static_cast<double>(neighbor.image.height) + acceptance_margin) {
        error = "PatchMatch crop center is outside the recovered acceptance margin";
        return false;
    }

    const std::uint64_t full_width = ceil_div(
        static_cast<std::uint32_t>(neighbor.image.width), target_downscale);
    const std::uint64_t full_height = ceil_div(
        static_cast<std::uint32_t>(neighbor.image.height), target_downscale);
    const double half_extent = static_cast<double>(expanded_extent >> 1U);
    const double scaled_x = neighbor_center.x / scale;
    const double scaled_y = neighbor_center.y / scale;
    const auto lower_bound = [](double value) {
        const std::int64_t converted = static_cast<std::int64_t>(value);
        return converted < 0 ? std::uint64_t{0} :
                               static_cast<std::uint64_t>(converted);
    };
    const auto upper_bound = [](double value, std::uint64_t limit) {
        const std::int64_t converted = static_cast<std::int64_t>(value);
        if (converted < 0) return std::uint64_t{0};
        return std::min(static_cast<std::uint64_t>(converted), limit);
    };
    std::uint64_t left = lower_bound(scaled_x - half_extent);
    std::uint64_t right = upper_bound(scaled_x + half_extent, full_width);
    std::uint64_t top = lower_bound(scaled_y - half_extent);
    std::uint64_t bottom = upper_bound(scaled_y + half_extent, full_height);
    if (right <= left || bottom <= top) {
        error = "PatchMatch crop produced an empty area";
        return false;
    }

    const std::uint64_t alignment = 32U / target_downscale;
    left &= ~(alignment - 1U);
    top &= ~(alignment - 1U);
    right = std::min(full_width,
                     (right + alignment - 1U) & ~(alignment - 1U));
    bottom = std::min(full_height,
                      (bottom + alignment - 1U) & ~(alignment - 1U));

    RecoveredPatchMatchCropDescriptor result;
    result.camera_index = static_cast<std::int32_t>(neighbor.index);
    result.left = left;
    result.right = right;
    result.top = top;
    result.bottom = bottom;
    result.width = right - left;
    result.height = bottom - top;
    result.full_width = full_width;
    result.full_height = full_height;
    output = result;
    error.clear();
    return true;
}

bool apply_recovered_patchmatch_neighbor_crop(
    const Camera& source_camera,
    std::uint32_t target_downscale,
    const RecoveredPatchMatchCropDescriptor& crop,
    RecoveredPatchMatchPreparedCamera& prepared,
    std::string& error) {
    if (target_downscale < 2U || target_downscale > 32U ||
        (target_downscale & (target_downscale - 1U)) != 0U ||
        crop.width == 0U || crop.height == 0U ||
        crop.right != crop.left + crop.width ||
        crop.bottom != crop.top + crop.height ||
        crop.right > crop.full_width || crop.bottom > crop.full_height) {
        error = "invalid recovered PatchMatch crop descriptor";
        return false;
    }
    for (RecoveredPatchMatchPreparedLevel& level : prepared.image_levels) {
        std::uint64_t left = crop.left;
        std::uint64_t right = crop.right;
        std::uint64_t top = crop.top;
        std::uint64_t bottom = crop.bottom;
        if (level.downscale < target_downscale) {
            if (target_downscale % level.downscale != 0U) {
                error = "PatchMatch crop and image pyramid scales are incompatible";
                return false;
            }
            const std::uint64_t multiplier = target_downscale / level.downscale;
            left *= multiplier; right *= multiplier;
            top *= multiplier; bottom *= multiplier;
        } else if (level.downscale > target_downscale) {
            if (level.downscale % target_downscale != 0U) {
                error = "PatchMatch crop and image pyramid scales are incompatible";
                return false;
            }
            const std::uint64_t divisor = level.downscale / target_downscale;
            if ((left % divisor) != 0U || (right % divisor) != 0U ||
                (top % divisor) != 0U || (bottom % divisor) != 0U) {
                error = "PatchMatch crop is not aligned for a prepared mip";
                return false;
            }
            left /= divisor; right /= divisor;
            top /= divisor; bottom /= divisor;
        }
        if (right > level.data.width || bottom > level.data.height ||
            right <= left || bottom <= top) {
            error = "PatchMatch crop exceeds a prepared image level";
            return false;
        }
        RecoveredPatchMatchImageU8 cropped;
        cropped.width = static_cast<std::uint32_t>(right - left);
        cropped.height = static_cast<std::uint32_t>(bottom - top);
        cropped.image.resize(static_cast<std::size_t>(cropped.width) * cropped.height);
        const bool has_mask = !level.data.rejection_mask.empty();
        if (has_mask) cropped.rejection_mask.resize(cropped.image.size());
        for (std::uint32_t row = 0; row < cropped.height; ++row) {
            const std::size_t source_offset =
                static_cast<std::size_t>(top + row) * level.data.width + left;
            const std::size_t destination_offset =
                static_cast<std::size_t>(row) * cropped.width;
            std::copy_n(level.data.image.begin() +
                            static_cast<std::ptrdiff_t>(source_offset),
                        cropped.width,
                        cropped.image.begin() +
                            static_cast<std::ptrdiff_t>(destination_offset));
            if (has_mask) {
                std::copy_n(level.data.rejection_mask.begin() +
                                static_cast<std::ptrdiff_t>(source_offset),
                            cropped.width,
                            cropped.rejection_mask.begin() +
                                static_cast<std::ptrdiff_t>(destination_offset));
            }
        }
        level.data = std::move(cropped);
    }

    const std::uint64_t full_left = crop.left * target_downscale;
    const std::uint64_t full_top = crop.top * target_downscale;
    const std::uint64_t full_width = crop.width * target_downscale;
    const std::uint64_t full_height = crop.height * target_downscale;
    if (full_width > std::numeric_limits<std::uint32_t>::max() ||
        full_height > std::numeric_limits<std::uint32_t>::max()) {
        error = "PatchMatch cropped camera dimensions exceed uint32";
        return false;
    }
    PatchMatchCamera camera = make_patchmatch_perspective_camera(
        source_camera, static_cast<int>(target_downscale));
    camera.width_original = static_cast<std::uint32_t>(full_width);
    camera.height_original = static_cast<std::uint32_t>(full_height);
    camera.cx = static_cast<float>(
        source_camera.model.cx - static_cast<double>(full_left) -
        static_cast<double>(full_width) * 0.5);
    camera.cy = static_cast<float>(
        source_camera.model.cy - static_cast<double>(full_top) -
        static_cast<double>(full_height) * 0.5);
    prepared.camera = camera;
    error.clear();
    return true;
}

static bool make_recovered_patchmatch_unmasked_neighbor_resources_impl(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const RecoveredPatchMatchPreparedCamera* const> ranked_neighbors,
    std::uint32_t target_downscale,
    RecoveredPatchMatchNeighborResources& output,
    std::string& error,
    bool materialize_full_atlases) {
    if (reference_camera_index >= scene.cameras.size()) {
        error = "PatchMatch neighbour-resource reference camera is out of range";
        return false;
    }
    if (ranked_neighbors.empty()) {
        error = "PatchMatch neighbour-resource list is empty";
        return false;
    }
    if (target_downscale < 2U || target_downscale > 32U ||
        (target_downscale & (target_downscale - 1U)) != 0U) {
        error = "PatchMatch neighbour resources require power-of-two downscale 2..32";
        return false;
    }
    const std::uint32_t base_downscale = target_downscale / 2U;

    struct CroppedLevelView {
        const RecoveredPatchMatchPreparedLevel* source = nullptr;
        std::uint32_t left = 0;
        std::uint32_t top = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };
    struct Layout {
        const RecoveredPatchMatchPreparedCamera* prepared = nullptr;
        PatchMatchCamera camera;
        RecoveredPatchMatchCropDescriptor crop;
        std::vector<CroppedLevelView> levels;
        std::uint32_t bottom_width = 0;
        std::uint32_t atlas_height = 0;
    };
    std::vector<Layout> layouts;
    layouts.reserve(ranked_neighbors.size());
    std::uint32_t atlas_width = 0;
    std::uint32_t atlas_height = 0;
    for (const RecoveredPatchMatchPreparedCamera* source_prepared_pointer :
         ranked_neighbors) {
        if (source_prepared_pointer == nullptr) {
            error = "PatchMatch prepared neighbour pointer is null";
            return false;
        }
        const RecoveredPatchMatchPreparedCamera& source_prepared =
            *source_prepared_pointer;
        if (source_prepared.camera_index >= scene.cameras.size()) {
            error = "PatchMatch prepared neighbour camera is out of range";
            return false;
        }
        Layout layout;
        layout.prepared = &source_prepared;
        if (!make_recovered_patchmatch_neighbor_crop_descriptor(
                scene, reference_camera_index, source_prepared.camera_index,
                target_downscale, layout.crop, error))
            return false;
        const auto find_level = [&](std::uint32_t downscale) {
            return std::find_if(
                source_prepared.image_levels.begin(),
                source_prepared.image_levels.end(),
                [downscale](const RecoveredPatchMatchPreparedLevel& level) {
                    return level.downscale == downscale;
                });
        };
        auto base = find_level(base_downscale);
        auto level_one = find_level(base_downscale * 2U);
        if (base == source_prepared.image_levels.end() ||
            level_one == source_prepared.image_levels.end()) {
            error = "PatchMatch prepared neighbour lacks atlas base/level-one image";
            return false;
        }

        const auto make_level_view = [&](
            const RecoveredPatchMatchPreparedLevel& level,
            CroppedLevelView& view) {
            std::uint64_t left = layout.crop.left;
            std::uint64_t right = layout.crop.right;
            std::uint64_t top = layout.crop.top;
            std::uint64_t bottom = layout.crop.bottom;
            if (level.downscale < target_downscale) {
                if (target_downscale % level.downscale != 0U) {
                    error = "PatchMatch crop and image pyramid scales are incompatible";
                    return false;
                }
                const std::uint64_t multiplier =
                    target_downscale / level.downscale;
                left *= multiplier;
                right *= multiplier;
                top *= multiplier;
                bottom *= multiplier;
            } else if (level.downscale > target_downscale) {
                if (level.downscale % target_downscale != 0U) {
                    error = "PatchMatch crop and image pyramid scales are incompatible";
                    return false;
                }
                const std::uint64_t divisor =
                    level.downscale / target_downscale;
                if ((left % divisor) != 0U || (right % divisor) != 0U ||
                    (top % divisor) != 0U || (bottom % divisor) != 0U) {
                    error = "PatchMatch crop is not aligned for a prepared mip";
                    return false;
                }
                left /= divisor;
                right /= divisor;
                top /= divisor;
                bottom /= divisor;
            }
            const std::size_t source_pixels =
                static_cast<std::size_t>(level.data.width) * level.data.height;
            if (level.data.image.size() != source_pixels ||
                right > level.data.width || bottom > level.data.height ||
                right <= left || bottom <= top ||
                right - left > std::numeric_limits<std::uint32_t>::max() ||
                bottom - top > std::numeric_limits<std::uint32_t>::max()) {
                error = "PatchMatch crop exceeds a prepared image level";
                return false;
            }
            view.source = &level;
            view.left = static_cast<std::uint32_t>(left);
            view.top = static_cast<std::uint32_t>(top);
            view.width = static_cast<std::uint32_t>(right - left);
            view.height = static_cast<std::uint32_t>(bottom - top);
            return true;
        };
        const auto append_level = [&](
            const RecoveredPatchMatchPreparedLevel& level) {
            CroppedLevelView view;
            if (!make_level_view(level, view)) return false;
            layout.levels.push_back(view);
            return true;
        };
        if (!append_level(*base) || !append_level(*level_one)) return false;

        // The target always copies level one.  It then copies the child of
        // every current parent, stopping only after the child of the first
        // parent whose width or height is <=252 has been copied.
        auto parent = level_one;
        for (;;) {
            if (parent->downscale > std::numeric_limits<std::uint32_t>::max() / 2U) {
                error = "PatchMatch neighbour mip downscale overflow";
                return false;
            }
            const auto child = find_level(parent->downscale * 2U);
            if (child == source_prepared.image_levels.end()) {
                error = "PatchMatch prepared neighbour mip chain ends before target atlas rule";
                return false;
            }
            if (!append_level(*child)) return false;
            const CroppedLevelView& cropped_parent =
                layout.levels[layout.levels.size() - 2U];
            if (cropped_parent.width <= 252U ||
                cropped_parent.height <= 252U)
                break;
            parent = child;
        }

        const CroppedLevelView& base_image = layout.levels[0];
        const CroppedLevelView& level_one_image = layout.levels[1];
        std::uint64_t bottom_width = 0;
        for (std::size_t index = 1; index < layout.levels.size(); ++index) {
            bottom_width += layout.levels[index].width;
        }
        if (base_image.width == 0U || base_image.height == 0U ||
            bottom_width > std::numeric_limits<std::uint32_t>::max() ||
            static_cast<std::uint64_t>(base_image.height) +
                    level_one_image.height >
                std::numeric_limits<std::uint32_t>::max()) {
            error = "PatchMatch neighbour atlas dimensions are invalid";
            return false;
        }
        layout.bottom_width = static_cast<std::uint32_t>(bottom_width);
        layout.atlas_height = base_image.height + level_one_image.height;
        const Camera& source_camera =
            scene.cameras[source_prepared.camera_index];
        const std::uint64_t full_left = layout.crop.left * target_downscale;
        const std::uint64_t full_top = layout.crop.top * target_downscale;
        const std::uint64_t full_width = layout.crop.width * target_downscale;
        const std::uint64_t full_height = layout.crop.height * target_downscale;
        if (full_width > std::numeric_limits<std::uint32_t>::max() ||
            full_height > std::numeric_limits<std::uint32_t>::max()) {
            error = "PatchMatch cropped camera dimensions exceed uint32";
            return false;
        }
        layout.camera = make_patchmatch_perspective_camera(
            source_camera, static_cast<int>(target_downscale));
        layout.camera.width_original = static_cast<std::uint32_t>(full_width);
        layout.camera.height_original = static_cast<std::uint32_t>(full_height);
        layout.camera.cx = static_cast<float>(
            source_camera.model.cx - static_cast<double>(full_left) -
            static_cast<double>(full_width) * 0.5);
        layout.camera.cy = static_cast<float>(
            source_camera.model.cy - static_cast<double>(full_top) -
            static_cast<double>(full_height) * 0.5);
        atlas_width = std::max(
            atlas_width, std::max(base_image.width, layout.bottom_width));
        atlas_height = std::max(atlas_height, layout.atlas_height);
        layouts.push_back(std::move(layout));
    }
    if (static_cast<std::uint64_t>(atlas_width) * atlas_height >
        std::numeric_limits<std::size_t>::max()) {
        error = "PatchMatch neighbour atlas allocation overflows size_t";
        return false;
    }

    RecoveredPatchMatchNeighborResources result;
    result.base_downscale = base_downscale;
    result.texture_width = atlas_width;
    result.texture_height = atlas_height;
    result.resource_groups.resize((layouts.size() + 9U) / 10U);
    for (PatchMatchCostResourceGroup& group : result.resource_groups) {
        group.image_offsets.assign(10U, 0U);
        group.level_offsets.assign(10U, 0U);
    }
    if (!materialize_full_atlases) {
        for (std::size_t rank = 0; rank < layouts.size(); ++rank) {
            std::size_t bytes = 0U;
            for (const CroppedLevelView& level : layouts[rank].levels)
                bytes += static_cast<std::size_t>(level.width) * level.height;
            result.resource_groups[rank / 10U].image.reserve(
                result.resource_groups[rank / 10U].image.capacity() + bytes);
        }
    }
    result.ranked_neighbors.reserve(layouts.size());

    for (std::size_t rank = 0; rank < layouts.size(); ++rank) {
        const Layout& layout = layouts[rank];
        const RecoveredPatchMatchPreparedCamera& prepared = *layout.prepared;
        const std::size_t group_index = rank / 10U;
        const std::size_t resource_index = rank % 10U;
        PatchMatchCostResourceGroup& group = result.resource_groups[group_index];
        group.image_offsets[resource_index] = group.image.size();
        group.level_offsets[resource_index] = group.mask.size();

        const CroppedLevelView& base = layout.levels.front();
        const std::size_t base_pixels =
            static_cast<std::size_t>(base.width) * base.height;
        const std::size_t mask_bytes = (base_pixels + 7U) / 8U;
        const std::size_t old_mask_size = group.mask.size();
        group.mask.resize(old_mask_size + mask_bytes, 0U);
        if (!base.source->data.rejection_mask.empty()) {
            const std::size_t source_pixels =
                static_cast<std::size_t>(base.source->data.width) *
                base.source->data.height;
            if (base.source->data.rejection_mask.size() != source_pixels) {
                error = "PatchMatch base rejection mask dimensions are invalid";
                return false;
            }
            for (std::uint32_t row = 0; row < base.height; ++row) {
                for (std::uint32_t column = 0; column < base.width; ++column) {
                    const std::size_t source_pixel =
                        static_cast<std::size_t>(base.top + row) *
                            base.source->data.width +
                        base.left + column;
                    const std::size_t pixel =
                        static_cast<std::size_t>(row) * base.width + column;
                    if (base.source->data.rejection_mask[source_pixel] != 0U)
                        group.mask[old_mask_size + (pixel >> 3U)] |=
                            static_cast<std::uint8_t>(1U << (pixel & 7U));
                }
            }
        } else {
            // Current ceil-half image reducer deliberately does not invent a
            // mask transition.  An all-valid directly undistorted source
            // proves all-valid descendants; any non-zero source rejection
            // requires the still-unrecovered transition and must fail closed.
            for (const RecoveredPatchMatchPreparedLevel& level :
                 prepared.image_levels) {
                if (level.downscale >= base_downscale) break;
                if (std::any_of(level.data.rejection_mask.begin(),
                                level.data.rejection_mask.end(),
                                [](std::uint8_t value) { return value != 0U; })) {
                    error = "non-zero PatchMatch rejection-mask scale transition is not recovered";
                    return false;
                }
            }
        }

        PatchMatchCostNeighbor neighbor;
        neighbor.camera = layout.camera;
        neighbor.camera.transform = patchmatch_reference_to_neighbor_transform(
            scene.cameras[reference_camera_index],
            scene.cameras[prepared.camera_index]);
        neighbor.resource_group = static_cast<std::uint32_t>(group_index);
        neighbor.resource_index = static_cast<std::uint32_t>(resource_index);
        neighbor.output_index = static_cast<std::uint32_t>(rank);
        neighbor.texture_source_grouped = !materialize_full_atlases;
        if (materialize_full_atlases) {
            neighbor.texture.assign(
                static_cast<std::size_t>(atlas_width) * atlas_height, 0U);
            neighbor.texture_write_mask.assign(neighbor.texture.size(), 0U);
        }
        std::uint32_t bottom_x = 0U;
        for (std::size_t mip = 0; mip < layout.levels.size(); ++mip) {
            const CroppedLevelView& image = layout.levels[mip];
            const std::uint32_t destination_x = mip < 2U ? 0U : bottom_x;
            const std::uint32_t destination_y = mip == 0U ? 0U : base.height;
            std::vector<std::uint8_t>& compact_source =
                materialize_full_atlases ? neighbor.texture_source : group.image;
            const std::uint64_t source_offset = compact_source.size();
            compact_source.resize(
                compact_source.size() +
                static_cast<std::size_t>(image.width) * image.height);
            neighbor.texture_copy_regions.push_back({
                source_offset, image.width, destination_x, destination_y,
                image.width, image.height});
            for (std::uint32_t y = 0; y < image.height; ++y) {
                const std::size_t source_row =
                    static_cast<std::size_t>(image.top + y) *
                        image.source->data.width +
                    image.left;
                const std::size_t compact_offset =
                    static_cast<std::size_t>(source_offset) +
                    static_cast<std::size_t>(y) * image.width;
                std::copy_n(
                    image.source->data.image.begin() +
                        static_cast<std::ptrdiff_t>(source_row),
                    image.width,
                    compact_source.begin() +
                        static_cast<std::ptrdiff_t>(compact_offset));
                if (!materialize_full_atlases) continue;
                const std::size_t destination_offset =
                    static_cast<std::size_t>(destination_y + y) * atlas_width +
                    destination_x;
                std::copy_n(compact_source.begin() +
                                static_cast<std::ptrdiff_t>(compact_offset),
                            image.width,
                            neighbor.texture.begin() +
                                static_cast<std::ptrdiff_t>(destination_offset));
                std::fill_n(neighbor.texture_write_mask.begin() +
                                static_cast<std::ptrdiff_t>(destination_offset),
                            image.width, std::uint8_t{1});
            }
            if (mip >= 1U) bottom_x += image.width;
        }
        result.ranked_neighbors.push_back(std::move(neighbor));
    }
    output = std::move(result);
    error.clear();
    return true;
}

bool make_recovered_patchmatch_unmasked_neighbor_resources(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const RecoveredPatchMatchPreparedCamera> ranked_neighbors,
    std::uint32_t target_downscale,
    RecoveredPatchMatchNeighborResources& output,
    std::string& error) {
    std::vector<const RecoveredPatchMatchPreparedCamera*> neighbor_pointers;
    neighbor_pointers.reserve(ranked_neighbors.size());
    for (const RecoveredPatchMatchPreparedCamera& neighbor : ranked_neighbors)
        neighbor_pointers.push_back(&neighbor);
    return make_recovered_patchmatch_unmasked_neighbor_resources_impl(
        scene, reference_camera_index, neighbor_pointers, target_downscale,
        output, error, true);
}

bool make_recovered_patchmatch_unmasked_neighbor_resources(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::uint32_t target_downscale,
    RecoveredPatchMatchNeighborResources& output,
    std::string& error) {
    if (prepared_camera_cache.size() != scene.cameras.size()) {
        error = "PatchMatch prepared-camera cache size does not match the scene";
        return false;
    }
    if (reference_camera_index >= prepared_camera_cache.size() ||
        !prepared_camera_cache[reference_camera_index].valid ||
        prepared_camera_cache[reference_camera_index].camera_index !=
            reference_camera_index ||
        prepared_camera_cache[reference_camera_index].target_downscale !=
            target_downscale) {
        error = "PatchMatch prepared-camera cache reference identity is invalid";
        return false;
    }
    const std::size_t expected_device_index =
        prepared_camera_cache[reference_camera_index].device_index;
    std::vector<const RecoveredPatchMatchPreparedCamera*> neighbor_pointers;
    neighbor_pointers.reserve(ranked_neighbor_camera_indices.size());
    for (const std::size_t camera_index : ranked_neighbor_camera_indices) {
        if (camera_index >= prepared_camera_cache.size()) {
            error = "PatchMatch prepared-camera cache index is out of range";
            return false;
        }
        const auto& cached = prepared_camera_cache[camera_index];
        if (!cached.valid || cached.camera_index != camera_index ||
            cached.target_downscale != target_downscale ||
            cached.device_index != expected_device_index) {
            error = "PatchMatch prepared-camera cache entry identity is invalid";
            return false;
        }
        neighbor_pointers.push_back(&cached);
    }
    return make_recovered_patchmatch_unmasked_neighbor_resources_impl(
        scene, reference_camera_index, neighbor_pointers, target_downscale,
        output, error, false);
}

bool make_recovered_patchmatch_unmasked_host_preparation(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::uint32_t target_downscale,
    std::size_t device_index,
    RecoveredPatchMatchHostPreparation& output,
    std::string& error,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache,
    bool materialize_prepared_neighbors) {
    if (reference_camera_index >= scene.cameras.size()) {
        error = "PatchMatch host-preparation reference camera is out of range";
        return false;
    }
    if (ranked_neighbor_camera_indices.empty()) {
        error = "PatchMatch host preparation requires ranked neighbors";
        return false;
    }
    if (target_downscale < 2U || target_downscale > 32U ||
        (target_downscale & (target_downscale - 1U)) != 0U) {
        error = "PatchMatch host preparation requires power-of-two downscale 2..32";
        return false;
    }

    RecoveredPatchMatchHostPreparation result;
    result.reference_camera_index = reference_camera_index;
    result.target_downscale = target_downscale;
    result.device_index = device_index;
    result.ranked_neighbor_camera_indices.assign(
        ranked_neighbor_camera_indices.begin(),
        ranked_neighbor_camera_indices.end());
    const auto prepare_camera = [&](std::size_t camera_index,
                                    RecoveredPatchMatchPreparedCamera& prepared) {
        if (camera_index >= scene.cameras.size()) {
            error = "PatchMatch prepared-camera cache index is out of range";
            return false;
        }
        if (prepared_camera_cache.empty())
            return make_recovered_patchmatch_unmasked_camera_preparation(
                scene, camera_index, target_downscale, device_index,
                prepared, error);
        if (prepared_camera_cache.size() != scene.cameras.size()) {
            error = "PatchMatch prepared-camera cache size does not match the scene";
            return false;
        }
        const auto& cached = prepared_camera_cache[camera_index];
        if (!cached.valid || cached.camera_index != camera_index ||
            cached.target_downscale != target_downscale ||
            cached.device_index != device_index) {
            error = "PatchMatch prepared-camera cache entry identity is invalid";
            return false;
        }
        prepared = cached;
        return true;
    };
    if (!prepare_camera(reference_camera_index, result.reference))
        return false;
    for (const std::size_t camera_index : ranked_neighbor_camera_indices) {
        if (camera_index == reference_camera_index) {
            error = "PatchMatch ranked neighbors contain the reference camera";
            return false;
        }
        if (camera_index >= scene.cameras.size()) {
            error = "PatchMatch ranked neighbor camera is out of range";
            return false;
        }
    }
    if (!materialize_prepared_neighbors && prepared_camera_cache.empty()) {
        error = "non-materialized PatchMatch host preparation requires a prepared-camera cache";
        return false;
    }
    if (materialize_prepared_neighbors) {
        result.ranked_neighbors.reserve(ranked_neighbor_camera_indices.size());
        for (const std::size_t camera_index : ranked_neighbor_camera_indices) {
            RecoveredPatchMatchPreparedCamera prepared;
            if (!prepare_camera(camera_index, prepared))
                return false;
            result.ranked_neighbors.push_back(std::move(prepared));
        }
    }
    if (materialize_prepared_neighbors) {
        if (!make_recovered_patchmatch_unmasked_neighbor_resources(
                scene, reference_camera_index, result.ranked_neighbors,
                target_downscale, result.neighbor_resources, error))
            return false;
    } else if (!make_recovered_patchmatch_unmasked_neighbor_resources(
                   scene, reference_camera_index, prepared_camera_cache,
                   ranked_neighbor_camera_indices, target_downscale,
                   result.neighbor_resources, error)) {
        return false;
    }

    const std::uint32_t base_downscale = target_downscale / 2U;
    const auto base = std::find_if(
        result.reference.image_levels.begin(),
        result.reference.image_levels.end(),
        [base_downscale](const RecoveredPatchMatchPreparedLevel& level) {
            return level.downscale == base_downscale;
        });
    if (base == result.reference.image_levels.end()) {
        error = "PatchMatch reference preparation lacks target atlas base image";
        return false;
    }
    result.deviation_threshold_multiplier = base->deviation_ratio;
    result.reference_camera = make_patchmatch_reference_camera(
        scene.cameras[reference_camera_index],
        static_cast<int>(target_downscale));
    result.normal_rotations =
        make_patchmatch_normal_rotation_cameras(result.reference_camera);
    result.propagation_rotation =
        make_patchmatch_propagation_rotation(result.reference_camera);
    output = std::move(result);
    error.clear();
    return true;
}

std::vector<std::uint8_t>
make_recovered_patchmatch_reference_image_allocation(
    const RecoveredPatchMatchPreparedCamera& reference,
    std::uint32_t target_downscale,
    std::uint32_t depth_downscale) {
    const auto power_of_two = [](std::uint32_t value) {
        return value != 0U && (value & (value - 1U)) == 0U;
    };
    if (target_downscale < 2U || target_downscale > 32U ||
        !power_of_two(target_downscale))
        throw std::invalid_argument(
            "PatchMatch reference allocation requires power-of-two target downscale 2..32");
    if (depth_downscale < target_downscale || depth_downscale > 32U ||
        !power_of_two(depth_downscale))
        throw std::invalid_argument(
            "PatchMatch reference allocation depth downscale is outside the recovered target..32 schedule");

    const std::uint32_t allocation_level = target_downscale / 2U;
    const std::uint32_t image_level = depth_downscale / 2U;
    const auto find_level = [&reference](std::uint32_t downscale) {
        return std::find_if(
            reference.image_levels.begin(), reference.image_levels.end(),
            [downscale](const RecoveredPatchMatchPreparedLevel& level) {
                return level.downscale == downscale;
            });
    };
    const auto allocation = find_level(allocation_level);
    const auto image = find_level(image_level);
    if (allocation == reference.image_levels.end() ||
        image == reference.image_levels.end())
        throw std::invalid_argument(
            "PatchMatch reference preparation lacks a required allocation mip");
    const std::size_t allocation_pixels =
        static_cast<std::size_t>(allocation->data.width) *
        allocation->data.height;
    const std::size_t image_pixels =
        static_cast<std::size_t>(image->data.width) * image->data.height;
    if (allocation->data.image.size() != allocation_pixels ||
        image->data.image.size() != image_pixels ||
        image_pixels > allocation_pixels)
        throw std::invalid_argument(
            "PatchMatch reference preparation has inconsistent mip dimensions");
    std::vector<std::uint8_t> result(allocation_pixels, std::uint8_t{0});
    std::copy(image->data.image.begin(), image->data.image.end(), result.begin());
    return result;
}

RecoveredPatchMatchCostAtlasState
make_recovered_patchmatch_cost_atlas_state(
    const RecoveredPatchMatchNeighborResources& resources) {
    const std::uint64_t texture_bytes =
        static_cast<std::uint64_t>(resources.texture_width) *
        resources.texture_height;
    if (texture_bytes > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(
            "PatchMatch camera atlas dimensions overflow size_t");
    RecoveredPatchMatchCostAtlasState result;
    result.generation = recovered_patchmatch_atlas_generation.fetch_add(
                            1U, std::memory_order_relaxed) +
                        1U;
    if (result.generation == 0U)
        throw std::overflow_error("PatchMatch camera atlas generation overflow");
    result.texture_width = resources.texture_width;
    result.texture_height = resources.texture_height;
    result.texture.assign(static_cast<std::size_t>(texture_bytes), 0U);
    return result;
}

RecoveredPatchMatchCostBinding make_recovered_patchmatch_cost_binding(
    const RecoveredPatchMatchHostPreparation& preparation,
    RecoveredPatchMatchCostAtlasState& camera_atlas_state,
    std::uint32_t depth_downscale,
    std::uint32_t iteration,
    bool all_neighbors_state,
    bool materialize_cumulative_atlas) {
    const auto selectors = make_recovered_patchmatch_neighbor_subset(
        static_cast<std::uint32_t>(
            preparation.neighbor_resources.ranked_neighbors.size()),
        iteration, all_neighbors_state);
    RecoveredPatchMatchCostBinding result;
    result.reference_camera = preparation.reference_camera;
    result.normal_rotations = preparation.normal_rotations;
    if (materialize_cumulative_atlas)
        result.reference_image_allocation =
            make_recovered_patchmatch_reference_image_allocation(
                preparation.reference, preparation.target_downscale,
                depth_downscale);
    result.reference_image_level_downscale = depth_downscale / 2U;
    result.deviation_threshold_multiplier =
        preparation.deviation_threshold_multiplier;
    result.neighbor_texture_width =
        preparation.neighbor_resources.texture_width;
    result.neighbor_texture_height =
        preparation.neighbor_resources.texture_height;
    if (!materialize_cumulative_atlas)
        result.initial_neighbor_texture_view = camera_atlas_state.texture;
    if (materialize_cumulative_atlas)
        result.resource_groups = preparation.neighbor_resources.resource_groups;
    else
        result.resource_groups_view =
            preparation.neighbor_resources.resource_groups;
    result.neighbor_batch = make_recovered_patchmatch_cost_batch(
        preparation.neighbor_resources, selectors, camera_atlas_state,
        materialize_cumulative_atlas);
    result.neighbor_count =
        static_cast<std::uint32_t>(result.neighbor_batch.size());
    result.neighbor_cost_capacity = static_cast<std::uint32_t>(
        preparation.neighbor_resources.ranked_neighbors.size());
    result.camera_resource_generation = camera_atlas_state.generation;
    return result;
}

RecoveredPatchMatchCostBinding make_recovered_patchmatch_cost_binding(
    const RecoveredPatchMatchHostPreparation& preparation,
    std::uint32_t depth_downscale,
    std::uint32_t iteration,
    bool all_neighbors_state) {
    auto camera_atlas_state = make_recovered_patchmatch_cost_atlas_state(
        preparation.neighbor_resources);
    return make_recovered_patchmatch_cost_binding(
        preparation, camera_atlas_state, depth_downscale, iteration,
        all_neighbors_state, true);
}

std::vector<PatchMatchCostNeighbor> make_recovered_patchmatch_cost_batch(
    const RecoveredPatchMatchNeighborResources& resources,
    std::span<const std::uint32_t> ranked_selectors,
    RecoveredPatchMatchCostAtlasState& camera_atlas_state,
    bool materialize_cumulative_atlas) {
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(resources.texture_width) *
        resources.texture_height;
    if (expected_bytes > std::numeric_limits<std::size_t>::max() ||
        camera_atlas_state.texture_width != resources.texture_width ||
        camera_atlas_state.texture_height != resources.texture_height ||
        camera_atlas_state.texture.size() !=
            static_cast<std::size_t>(expected_bytes))
        throw std::invalid_argument(
            "PatchMatch camera atlas state dimensions differ from neighbor resources");

    std::vector<PatchMatchCostNeighbor> result;
    result.reserve(ranked_selectors.size());
    for (std::size_t output_index = 0; output_index < ranked_selectors.size();
         ++output_index) {
        const std::uint32_t selector = ranked_selectors[output_index];
        if (selector >= resources.ranked_neighbors.size())
            throw std::out_of_range(
                "PatchMatch cost-batch ranked selector is out of range");
        const PatchMatchCostNeighbor& source =
            resources.ranked_neighbors[selector];
        const bool sparse_view = !materialize_cumulative_atlas &&
            !source.texture_copy_regions.empty();
        if (sparse_view) {
            const bool grouped_source_available =
                source.texture_source_grouped &&
                source.resource_group < resources.resource_groups.size() &&
                !resources.resource_groups[source.resource_group].image.empty();
            if (source.texture_source.empty() && !grouped_source_available)
                throw std::invalid_argument(
                    "PatchMatch ranked neighbor compact texture source is empty");
        } else if (source.texture.size() != camera_atlas_state.texture.size() ||
                   source.texture_write_mask.size() !=
                       camera_atlas_state.texture.size()) {
            throw std::invalid_argument(
                "PatchMatch ranked neighbor atlas/write-mask dimensions differ");
        }
        PatchMatchCostNeighbor neighbor;
        neighbor.camera = source.camera;
        neighbor.resource_group = source.resource_group;
        neighbor.resource_index = source.resource_index;
        neighbor.output_index = static_cast<std::uint32_t>(output_index);
        neighbor.texture_copy_regions = source.texture_copy_regions;
        neighbor.texture_source_grouped = source.texture_source_grouped;
        if (sparse_view) {
            if (source.texture_source_grouped) {
                if (source.resource_group >= resources.resource_groups.size())
                    throw std::invalid_argument(
                        "PatchMatch grouped texture source selector is invalid");
                neighbor.texture_view =
                    resources.resource_groups[source.resource_group].image;
            } else {
                neighbor.texture_view = source.texture_source;
            }
        } else {
            neighbor.texture = source.texture;
            neighbor.texture_write_mask = source.texture_write_mask;
            for (std::size_t index = 0;
                 index < camera_atlas_state.texture.size(); ++index) {
                if (neighbor.texture_write_mask[index] != 0U)
                    camera_atlas_state.texture[index] = neighbor.texture[index];
            }
            neighbor.texture = camera_atlas_state.texture;
        }
        result.push_back(std::move(neighbor));
        if (camera_atlas_state.prepare_count ==
            std::numeric_limits<std::size_t>::max())
            throw std::overflow_error(
                "PatchMatch camera atlas prepare count overflow");
        ++camera_atlas_state.prepare_count;
    }
    return result;
}

std::vector<PatchMatchCostNeighbor> make_recovered_patchmatch_cost_batch(
    const RecoveredPatchMatchNeighborResources& resources,
    std::span<const std::uint32_t> ranked_selectors) {
    auto camera_atlas_state =
        make_recovered_patchmatch_cost_atlas_state(resources);
    return make_recovered_patchmatch_cost_batch(
        resources, ranked_selectors, camera_atlas_state);
}

bool run_recovered_depth_radius_estimate_cuda(
    const DepthRadiusEstimateInput& input,
    std::vector<float>& radius_output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_depth_radius_estimate_cuda_impl(
        input, radius_output, error);
#else
    (void)input;
    (void)radius_output;
    error = "recovered CUDA depth-radius backend was not compiled";
    return false;
#endif
}

bool run_recovered_depth_neighbor_votes_cuda(
    const DepthNeighborVotesInput& input,
    DepthNeighborVotesOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_depth_neighbor_votes_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA direct depth-voting backend was not compiled";
    return false;
#endif
}

bool run_recovered_depth_neighbor_occlusion_votes_cuda(
    const DepthNeighborOcclusionVotesInput& input,
    DepthNeighborOcclusionVotesOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_depth_neighbor_occlusion_votes_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA occlusion depth-voting backend was not compiled";
    return false;
#endif
}

bool run_recovered_depth_voting_finalize_cuda(
    const DepthVotingFinalizeInput& input,
    DepthVotingFinalizeOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_depth_voting_finalize_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA depth-voting backend was not compiled";
    return false;
#endif
}

DepthVotingChainInput make_recovered_depth_voting_chain_input(
    const Scene& scene,
    const DepthVotingPreparedReference& reference,
    const std::vector<DepthVotingPreparedNeighbor>& neighbors,
    std::uint32_t downscale,
    FilterMode filter_mode,
    std::size_t device_index) {
    if (reference.camera_index >= scene.cameras.size())
        throw std::out_of_range("depth-voting reference camera is out of range");
    const Camera& reference_camera = scene.cameras[reference.camera_index];
    DepthVotingChainInput chain;
    chain.component_size_threshold =
        recovered_depth_component_threshold(filter_mode);
    chain.device_index = device_index;

    std::array<std::size_t, 3> level_pixels{};
    std::array<std::uint32_t, 3> level_offsets{};
    std::size_t pyramid_pixels = 0;
    for (std::uint32_t level = 0; level < 3U; ++level) {
        const auto calibration = make_depth_voting_perspective_calibration(
            reference_camera, downscale, level);
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::memcpy(&width, calibration.bytes.data() + 84U, sizeof(width));
        std::memcpy(&height, calibration.bytes.data() + 88U, sizeof(height));
        level_pixels[level] =
            static_cast<std::size_t>(width) * height;
        const std::span<const float> reference_depth =
            reference.depth_level_views[level].empty()
                ? std::span<const float>(reference.depth_levels[level])
                : reference.depth_level_views[level];
        if (reference_depth.size() != level_pixels[level])
            throw std::invalid_argument(
                "depth-voting reference pyramid level size mismatch");
        if (pyramid_pixels > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("depth-voting pyramid offset overflow");
        level_offsets[level] = static_cast<std::uint32_t>(pyramid_pixels);
        pyramid_pixels += level_pixels[level];

        auto& output = chain.reference[level];
        output.width = width;
        output.height = height;
        if (reference.depth_level_views[level].empty())
            output.depth = reference.depth_levels[level];
        else
            output.depth_view = reference_depth;
        output.initial_votes.assign(level_pixels[level], 0);
        output.radius.calibration = calibration;
        output.radius.transform =
            make_depth_voting_camera_to_world_transform(reference_camera);
    }

    chain.neighbors.reserve(neighbors.size());
    for (const auto& prepared : neighbors) {
        if (prepared.camera_index >= scene.cameras.size())
            throw std::out_of_range(
                "depth-voting neighbor camera is out of range");
        const Camera& neighbor_camera = scene.cameras[prepared.camera_index];
        if (neighbor_camera.image.width != reference_camera.image.width ||
            neighbor_camera.image.height != reference_camera.image.height)
            throw std::invalid_argument(
                "mixed voting image dimensions are not yet dynamically verified");

        DepthVotingChainNeighborInput output;
        bool use_depth_views = true;
        for (std::uint32_t level = 0; level < 3U; ++level)
            use_depth_views = use_depth_views &&
                !prepared.depth_level_views[level].empty();
        if (!use_depth_views) output.depth_levels.reserve(pyramid_pixels);
        for (std::uint32_t level = 0; level < 3U; ++level) {
            const std::span<const float> depth =
                prepared.depth_level_views[level].empty()
                    ? std::span<const float>(prepared.depth_levels[level])
                    : prepared.depth_level_views[level];
            const std::span<const std::uint8_t> mask =
                prepared.inlier_mask_views[level].empty()
                    ? std::span<const std::uint8_t>(
                          prepared.inlier_masks[level])
                    : prepared.inlier_mask_views[level];
            if (depth.size() != level_pixels[level] ||
                mask.size() != level_pixels[level])
                throw std::invalid_argument(
                    "depth-voting neighbor pyramid or mask level size mismatch");
            if (use_depth_views)
                output.depth_level_views[level] = depth;
            else
                output.depth_levels.insert(
                    output.depth_levels.end(), depth.begin(), depth.end());
            if (prepared.inlier_mask_views[level].empty())
                output.inlier_masks[level] = prepared.inlier_masks[level];
            else
                output.inlier_mask_views[level] = mask;
            output.radius[level].calibration =
                make_depth_voting_perspective_calibration(
                    neighbor_camera, downscale, level);
            output.radius[level].transform =
                make_depth_voting_camera_to_world_transform(neighbor_camera);
            output.radius[level].level_offset = level_offsets[level];
        }

        const auto reference_to_neighbor =
            make_depth_voting_camera_transform(
                reference_camera, neighbor_camera);
        const auto neighbor_to_reference =
            make_depth_voting_camera_transform(
                neighbor_camera, reference_camera);
        const auto neighbor_base_calibration =
            make_depth_voting_perspective_calibration(
                neighbor_camera, downscale, 0U);
        for (std::uint32_t reference_level = 0;
             reference_level < 3U; ++reference_level) {
            const auto reference_calibration =
                make_depth_voting_perspective_calibration(
                    reference_camera, downscale, reference_level);
            auto& direct = output.direct[reference_level];
            direct.reference_calibration = reference_calibration;
            direct.reference_to_neighbor = reference_to_neighbor;
            direct.neighbor_calibration = neighbor_base_calibration;
            for (auto& occlusion : output.occlusion[reference_level]) {
                occlusion.reference_calibration = reference_calibration;
                occlusion.neighbor_to_reference = neighbor_to_reference;
                occlusion.neighbor_calibration = neighbor_base_calibration;
            }
        }
        chain.neighbors.push_back(std::move(output));
    }
    return chain;
}

bool run_recovered_depth_voting_chain_cuda(
    const DepthVotingChainInput& input,
    DepthVotingChainOutput& output,
    std::string& error) {
    constexpr std::size_t level_count = 3;
#ifdef METMODEL_HAS_CUDA
    if (!run_recovered_depth_voting_chain_cuda_impl(input, output, error))
        return false;
    for (std::size_t level = 0; level < level_count; ++level) {
        output.depth_after_components[level] =
            output.depth_before_components[level];
        output.components_cleared[level] =
            filter_recovered_small_depth_components(
                output.depth_after_components[level],
                input.reference[level].width,
                input.reference[level].height,
                input.component_size_threshold);
    }
    return true;
#else
    auto calibration_u32 = [](const DepthVotingCalibrationCu& calibration,
                              std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, calibration.bytes.data() + offset, sizeof(value));
        return value;
    };
    std::array<std::vector<float>, level_count> reference_radius;
    output = {};
    output.direct_counters = input.direct_counters;
    output.occlusion_counters = input.occlusion_counters;
    output.final_counters = input.final_counters;

    for (std::size_t level = 0; level < level_count; ++level) {
        const auto& reference = input.reference[level];
        const std::size_t pixels =
            static_cast<std::size_t>(reference.width) * reference.height;
        if (reference.width == 0 || reference.height == 0 ||
            reference.depth.size() != pixels ||
            (!reference.initial_votes.empty() &&
             reference.initial_votes.size() != pixels)) {
            error = "depth-voting chain reference resource size is invalid";
            return false;
        }
        output.votes[level] = reference.initial_votes;
        if (output.votes[level].empty()) output.votes[level].assign(pixels, 0);
        reference_radius[level].assign(pixels, 0.0F);
        DepthRadiusEstimateInput radius;
        radius.calibration = reference.radius.calibration;
        radius.transform = reference.radius.transform;
        radius.depth_allocation = reference.depth;
        radius.radius_allocation = reference_radius[level];
        radius.level_offset = reference.radius.level_offset;
        radius.kernel_offset = reference.radius.kernel_offset;
        radius.global_work_items = reference.radius.global_work_items;
        radius.device_index = input.device_index;
        if (!run_recovered_depth_radius_estimate_cuda(
                radius, reference_radius[level], error)) {
            error = "reference radius level " + std::to_string(level) +
                    ": " + error;
            return false;
        }
    }

    for (std::size_t neighbor_index = 0;
         neighbor_index < input.neighbors.size(); ++neighbor_index) {
        const auto& neighbor = input.neighbors[neighbor_index];
        std::vector<float> neighbor_radius(neighbor.depth_levels.size(), 0.0F);
        for (std::size_t level = 0; level < level_count; ++level) {
            DepthRadiusEstimateInput radius;
            radius.calibration = neighbor.radius[level].calibration;
            radius.transform = neighbor.radius[level].transform;
            radius.depth_allocation = neighbor.depth_levels;
            radius.radius_allocation = neighbor_radius;
            radius.level_offset = neighbor.radius[level].level_offset;
            radius.kernel_offset = neighbor.radius[level].kernel_offset;
            radius.global_work_items = neighbor.radius[level].global_work_items;
            radius.device_index = input.device_index;
            if (!run_recovered_depth_radius_estimate_cuda(
                    radius, neighbor_radius, error)) {
                error = "neighbor " + std::to_string(neighbor_index) +
                        " radius level " + std::to_string(level) +
                        ": " + error;
                return false;
            }
        }

        for (std::size_t reference_level = 0;
             reference_level < level_count; ++reference_level) {
            const auto& reference = input.reference[reference_level];
            const auto& launch = neighbor.direct[reference_level];
            DepthNeighborVotesInput direct;
            direct.reference_calibration = launch.reference_calibration;
            direct.reference_to_neighbor = launch.reference_to_neighbor;
            direct.neighbor_calibration = launch.neighbor_calibration;
            direct.votes = output.votes[reference_level];
            direct.reference_depth = reference.depth;
            direct.reference_radius = reference_radius[reference_level];
            direct.neighbor_inlier_mask =
                neighbor.inlier_masks[reference_level];
            direct.neighbor_depth_levels = neighbor.depth_levels;
            direct.neighbor_radius_levels = neighbor_radius;
            direct.reference_level =
                static_cast<std::uint32_t>(reference_level);
            direct.neighbor_levels = 3;
            direct.counters = output.direct_counters;
            direct.kernel_offset = launch.kernel_offset;
            direct.global_work_items = launch.global_work_items;
            direct.device_index = input.device_index;
            DepthNeighborVotesOutput direct_output;
            if (!run_recovered_depth_neighbor_votes_cuda(
                    direct, direct_output, error)) {
                error = "neighbor " + std::to_string(neighbor_index) +
                        " direct level " +
                        std::to_string(reference_level) + ": " + error;
                return false;
            }
            output.votes[reference_level] = std::move(direct_output.votes);
            output.direct_counters = direct_output.counters;

            for (std::size_t neighbor_level = 0;
                 neighbor_level < level_count; ++neighbor_level) {
                const auto& occlusion_launch =
                    neighbor.occlusion[reference_level][neighbor_level];
                const std::uint32_t neighbor_width = calibration_u32(
                    occlusion_launch.neighbor_calibration, 84);
                const std::uint32_t neighbor_height = calibration_u32(
                    occlusion_launch.neighbor_calibration, 88);
                std::size_t prefix_pixels = 0;
                for (std::size_t level = 0; level <= neighbor_level; ++level) {
                    const std::size_t divisor = std::size_t{1} << level;
                    prefix_pixels +=
                        ((neighbor_width + divisor - 1) / divisor) *
                        ((neighbor_height + divisor - 1) / divisor);
                }
                if (prefix_pixels > neighbor.depth_levels.size()) {
                    error = "occlusion neighbor pyramid prefix is invalid";
                    return false;
                }
                DepthNeighborOcclusionVotesInput occlusion;
                occlusion.reference_calibration =
                    occlusion_launch.reference_calibration;
                occlusion.neighbor_to_reference =
                    occlusion_launch.neighbor_to_reference;
                occlusion.neighbor_calibration =
                    occlusion_launch.neighbor_calibration;
                occlusion.votes = output.votes[reference_level];
                occlusion.reference_depth = reference.depth;
                occlusion.reference_radius = reference_radius[reference_level];
                occlusion.neighbor_inlier_mask =
                    neighbor.inlier_masks[reference_level];
                occlusion.neighbor_depth_levels.assign(
                    neighbor.depth_levels.begin(),
                    neighbor.depth_levels.begin() +
                        static_cast<std::ptrdiff_t>(prefix_pixels));
                occlusion.neighbor_radius_levels.assign(
                    neighbor_radius.begin(),
                    neighbor_radius.begin() +
                        static_cast<std::ptrdiff_t>(prefix_pixels));
                occlusion.neighbor_level =
                    static_cast<std::uint32_t>(neighbor_level);
                occlusion.reference_level =
                    static_cast<std::uint32_t>(reference_level);
                occlusion.counters = output.occlusion_counters;
                occlusion.kernel_offset = occlusion_launch.kernel_offset;
                occlusion.global_work_items =
                    occlusion_launch.global_work_items;
                occlusion.device_index = input.device_index;
                DepthNeighborOcclusionVotesOutput occlusion_output;
                if (!run_recovered_depth_neighbor_occlusion_votes_cuda(
                        occlusion, occlusion_output, error)) {
                    error = "neighbor " + std::to_string(neighbor_index) +
                            " occlusion levels " +
                            std::to_string(reference_level) + "/" +
                            std::to_string(neighbor_level) + ": " + error;
                    return false;
                }
                output.votes[reference_level] =
                    std::move(occlusion_output.votes);
                output.occlusion_counters = occlusion_output.counters;
            }
        }
    }

    for (std::size_t level = 0; level < level_count; ++level) {
        DepthVotingFinalizeInput final;
        final.width = input.reference[level].width;
        final.height = input.reference[level].height;
        final.depth_allocation = input.reference[level].depth;
        final.votes_allocation = output.votes[level];
        final.counter_empty = output.final_counters[0];
        final.counter_bad = output.final_counters[1];
        final.counter_normal = output.final_counters[2];
        final.counter_good = output.final_counters[3];
        final.device_index = input.device_index;
        DepthVotingFinalizeOutput final_output;
        if (!run_recovered_depth_voting_finalize_cuda(
                final, final_output, error)) {
            error = "final voting level " + std::to_string(level) +
                    ": " + error;
            return false;
        }
        output.final_counters = {
            final_output.counter_empty, final_output.counter_bad,
            final_output.counter_normal, final_output.counter_good};
        output.depth_before_components[level] =
            std::move(final_output.depth_allocation);
        output.depth_after_components[level] =
            output.depth_before_components[level];
        output.components_cleared[level] =
            filter_recovered_small_depth_components(
                output.depth_after_components[level], final.width, final.height,
                input.component_size_threshold);
    }
    return true;
#endif
}

bool run_recovered_patchmatch_refinement_cuda(
    const PatchMatchRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchRefinementInput replay_safe_input = input;
    return run_recovered_patchmatch_refinement_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_refinement_cuda_movable(
    PatchMatchRefinementInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_refinement_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_final_refinement_cuda(
    const PatchMatchFinalRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchFinalRefinementInput replay_safe_input = input;
    return run_recovered_patchmatch_final_refinement_cuda_impl(
        replay_safe_input, output, nullptr, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_final_refinement_state_cuda(
    const PatchMatchFinalRefinementInput& input,
    PatchMatchFinalRefinementOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchFinalRefinementInput replay_safe_input = input;
    return run_recovered_patchmatch_final_refinement_cuda_impl(
        replay_safe_input, output.candidates, &output.cost, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_final_refinement_state_cuda_movable(
    PatchMatchFinalRefinementInput&& input,
    PatchMatchFinalRefinementOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_final_refinement_cuda_impl(
        input, output.candidates, &output.cost, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_wta_cuda(
    const PatchMatchWtaInput& input,
    PatchMatchWtaOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchWtaInput replay_safe_input = input;
    return run_recovered_patchmatch_wta_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_wta_cuda_movable(
    PatchMatchWtaInput&& input,
    PatchMatchWtaOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_wta_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_copy_inlier_masks_cuda(
    const PatchMatchCopyInlierMasksInput& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchCopyInlierMasksInput replay_safe_input = input;
    return run_recovered_patchmatch_copy_inlier_masks_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch inlier-mask backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_copy_inlier_masks_cuda_movable(
    PatchMatchCopyInlierMasksInput&& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_copy_inlier_masks_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch inlier-mask backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_propagation_cuda(
    const PatchMatchPropagationInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchPropagationInput replay_safe_input = input;
    return run_recovered_patchmatch_propagation_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_propagation_cuda_movable(
    PatchMatchPropagationInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_propagation_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_coarse_to_precise_cuda(
    const PatchMatchCoarseToPreciseInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchCoarseToPreciseInput replay_safe_input = input;
    return run_recovered_patchmatch_coarse_to_precise_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_coarse_to_precise_cuda_movable(
    PatchMatchCoarseToPreciseInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_coarse_to_precise_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_cost_cuda(
    const PatchMatchCostInput& input,
    PatchMatchCostOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    PatchMatchCostInput replay_safe_input = input;
    return run_recovered_patchmatch_cost_cuda_impl(
        replay_safe_input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_cost_cuda_movable(
    PatchMatchCostInput&& input,
    PatchMatchCostOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_cost_cuda_impl(input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_bilateral_u8_cuda(
    const PatchMatchBilateralU8Input& input,
    PatchMatchBilateralU8Output& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_bilateral_u8_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch bilateral backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_check_cost_cuda(
    const PatchMatchFilterCheckCostInput& input,
    PatchMatchFilterCheckCostOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_check_cost_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_check_neighbours_cuda(
    const PatchMatchFilterCheckNeighboursInput& input,
    PatchMatchFilterCheckNeighboursOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_check_neighbours_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_clear_depth_cuda(
    const PatchMatchFilterClearDepthInput& input,
    PatchMatchFilterClearDepthOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_clear_depth_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_normals_cuda(
    const PatchMatchFilterNormalsInput& input,
    PatchMatchFilterNormalsOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_normals_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_speckles_edges_cuda(
    const PatchMatchFilterSpecklesEdgesInput& input,
    PatchMatchFilterSpecklesEdgesOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_speckles_edges_cuda_impl(
        input, output, error);
#else
    (void)input;
    (void)output;
    error = "recovered CUDA PatchMatch backend was not compiled";
    return false;
#endif
}

bool run_recovered_patchmatch_filter_chain_cuda(
    const PatchMatchFilterChainInput& input,
    PatchMatchFilterChainOutput& output,
    std::string& error) {
#ifdef METMODEL_HAS_CUDA
    return run_recovered_patchmatch_filter_chain_cuda_impl(
        input, output, error);
#else
    try {
        if (input.depth_downscale == 0U ||
            input.camera.width_original == 0U ||
            input.camera.height_original == 0U) {
            error = "PatchMatch filter chain camera grid is invalid";
            return false;
        }
        const std::uint32_t width =
            ceil_div(input.camera.width_original, input.depth_downscale);
        const std::uint32_t height =
            ceil_div(input.camera.height_original, input.depth_downscale);
        const std::uint64_t pixels64 =
            static_cast<std::uint64_t>(width) * height;
        if (pixels64 == 0U ||
            pixels64 > std::numeric_limits<std::uint32_t>::max()) {
            error = "PatchMatch filter chain active grid is unsupported";
            return false;
        }
        const auto pixels = static_cast<std::uint32_t>(pixels64);
        const std::uint32_t span = patchmatch_balanced_batch_span(pixels);
        const auto for_each_launch = [&](auto&& callback) {
            for (std::uint32_t offset = 0; offset < pixels;) {
                const std::uint32_t work_items =
                    std::min(span, pixels - offset);
                if (!callback(offset, work_items)) return false;
                offset += work_items;
            }
            return true;
        };

        PatchMatchFilterCheckCostInput check_cost;
        check_cost.camera = input.camera;
        check_cost.depth_downscale = input.depth_downscale;
        check_cost.depth_allocation = input.depth_allocation;
        check_cost.cost_allocation = input.cost_allocation;
        check_cost.counter_no_cost_samples = input.counter_no_cost_samples;
        check_cost.counter_big_cost_samples = input.counter_big_cost_samples;
        check_cost.device_index = input.device_index;
        PatchMatchFilterCheckCostOutput check_cost_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                check_cost.pixel_offset = offset;
                check_cost.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_check_cost_cuda(
                        check_cost, check_cost_output, error))
                    return false;
                check_cost.depth_allocation =
                    check_cost_output.depth_allocation;
                check_cost.counter_no_cost_samples =
                    check_cost_output.counter_no_cost_samples;
                check_cost.counter_big_cost_samples =
                    check_cost_output.counter_big_cost_samples;
                return true;
            }))
            return false;

        PatchMatchFilterCheckNeighboursInput check_neighbours;
        check_neighbours.camera = input.camera;
        check_neighbours.depth_downscale = input.depth_downscale;
        check_neighbours.depth_min = input.depth_min;
        check_neighbours.depth_max = input.depth_max;
        check_neighbours.depth_allocation =
            check_cost_output.depth_allocation;
        check_neighbours.filtered_mask_allocation =
            input.filtered_mask_allocation;
        check_neighbours.counter_no_neighbours =
            input.counter_no_neighbours;
        check_neighbours.counter_no_close_neighbours =
            input.counter_no_close_neighbours;
        check_neighbours.device_index = input.device_index;
        PatchMatchFilterCheckNeighboursOutput check_neighbours_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                check_neighbours.pixel_offset = offset;
                check_neighbours.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_check_neighbours_cuda(
                        check_neighbours, check_neighbours_output, error))
                    return false;
                check_neighbours.filtered_mask_allocation =
                    check_neighbours_output.filtered_mask_allocation;
                check_neighbours.counter_no_neighbours =
                    check_neighbours_output.counter_no_neighbours;
                check_neighbours.counter_no_close_neighbours =
                    check_neighbours_output.counter_no_close_neighbours;
                return true;
            }))
            return false;

        PatchMatchFilterClearDepthInput first_clear;
        first_clear.camera = input.camera;
        first_clear.depth_downscale = input.depth_downscale;
        first_clear.depth_allocation = check_cost_output.depth_allocation;
        first_clear.filtered_mask_allocation =
            check_neighbours_output.filtered_mask_allocation;
        first_clear.counter_not_empty = input.first_counter_not_empty;
        first_clear.device_index = input.device_index;
        PatchMatchFilterClearDepthOutput first_clear_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                first_clear.pixel_offset = offset;
                first_clear.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_clear_depth_cuda(
                        first_clear, first_clear_output, error))
                    return false;
                first_clear.depth_allocation =
                    first_clear_output.depth_allocation;
                first_clear.counter_not_empty =
                    first_clear_output.counter_not_empty;
                return true;
            }))
            return false;

        PatchMatchFilterNormalsInput normals;
        normals.camera = input.camera;
        normals.depth_downscale = input.depth_downscale;
        normals.depth_allocation = first_clear_output.depth_allocation;
        normals.normal_allocation = input.normal_allocation;
        normals.estimated_normal_allocation =
            input.estimated_normal_allocation;
        normals.estimate_normal_map = input.estimate_normal_map;
        normals.filtered_mask_allocation =
            check_neighbours_output.filtered_mask_allocation;
        normals.counter_inconsistent_normal =
            input.counter_inconsistent_normal;
        normals.counter_bad_view_angle_estimated_normal =
            input.counter_bad_view_angle_estimated_normal;
        normals.counter_bad_view_angle_found_normal =
            input.counter_bad_view_angle_found_normal;
        normals.counter_cos_sum = input.counter_cos_sum;
        normals.counter_ncos_sum = input.counter_ncos_sum;
        normals.device_index = input.device_index;
        PatchMatchFilterNormalsOutput normals_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                normals.pixel_offset = offset;
                normals.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_normals_cuda(
                        normals, normals_output, error))
                    return false;
                normals.estimated_normal_allocation =
                    normals_output.estimated_normal_allocation;
                normals.filtered_mask_allocation =
                    normals_output.filtered_mask_allocation;
                normals.counter_inconsistent_normal =
                    normals_output.counter_inconsistent_normal;
                normals.counter_bad_view_angle_estimated_normal =
                    normals_output.counter_bad_view_angle_estimated_normal;
                normals.counter_bad_view_angle_found_normal =
                    normals_output.counter_bad_view_angle_found_normal;
                normals.counter_cos_sum = normals_output.counter_cos_sum;
                normals.counter_ncos_sum = normals_output.counter_ncos_sum;
                return true;
            }))
            return false;

        PatchMatchFilterClearDepthInput second_clear;
        second_clear.camera = input.camera;
        second_clear.depth_downscale = input.depth_downscale;
        second_clear.depth_allocation = first_clear_output.depth_allocation;
        second_clear.filtered_mask_allocation =
            normals_output.filtered_mask_allocation;
        second_clear.counter_not_empty = input.second_counter_not_empty;
        second_clear.device_index = input.device_index;
        PatchMatchFilterClearDepthOutput second_clear_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                second_clear.pixel_offset = offset;
                second_clear.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_clear_depth_cuda(
                        second_clear, second_clear_output, error))
                    return false;
                second_clear.depth_allocation =
                    second_clear_output.depth_allocation;
                second_clear.counter_not_empty =
                    second_clear_output.counter_not_empty;
                return true;
            }))
            return false;

        PatchMatchFilterSpecklesEdgesInput speckles;
        speckles.camera = input.camera;
        speckles.depth_downscale = input.depth_downscale;
        speckles.depth_allocation = second_clear_output.depth_allocation;
        speckles.filtered_mask_allocation =
            normals_output.filtered_mask_allocation;
        speckles.device_index = input.device_index;
        PatchMatchFilterSpecklesEdgesOutput speckles_output;
        if (!for_each_launch([&](std::uint32_t offset,
                                 std::uint32_t work_items) {
                speckles.pixel_offset = offset;
                speckles.global_work_items = work_items;
                if (!run_recovered_patchmatch_filter_speckles_edges_cuda(
                        speckles, speckles_output, error))
                    return false;
                speckles.filtered_mask_allocation =
                    speckles_output.filtered_mask_allocation;
                return true;
            }))
            return false;

        output.depth_allocation = std::move(second_clear_output.depth_allocation);
        output.estimated_normal_allocation =
            std::move(normals_output.estimated_normal_allocation);
        output.filtered_mask_allocation =
            std::move(speckles_output.filtered_mask_allocation);
        output.counter_no_cost_samples =
            check_cost_output.counter_no_cost_samples;
        output.counter_big_cost_samples =
            check_cost_output.counter_big_cost_samples;
        output.counter_no_neighbours =
            check_neighbours_output.counter_no_neighbours;
        output.counter_no_close_neighbours =
            check_neighbours_output.counter_no_close_neighbours;
        output.first_counter_not_empty = first_clear_output.counter_not_empty;
        output.counter_inconsistent_normal =
            normals_output.counter_inconsistent_normal;
        output.counter_bad_view_angle_estimated_normal =
            normals_output.counter_bad_view_angle_estimated_normal;
        output.counter_bad_view_angle_found_normal =
            normals_output.counter_bad_view_angle_found_normal;
        output.counter_cos_sum = normals_output.counter_cos_sum;
        output.counter_ncos_sum = normals_output.counter_ncos_sum;
        output.second_counter_not_empty = second_clear_output.counter_not_empty;
        return true;
    } catch (const std::exception& exception) {
        error = std::string("PatchMatch filter chain assembly failed: ") +
                exception.what();
        return false;
    }
#endif
}

bool run_recovered_patchmatch_level_boundary_cuda(
    const PatchMatchLevelBoundaryInput& input,
    PatchMatchLevelBoundaryOutput& output,
    std::string& error) {
    PatchMatchFilterChainOutput filtered;
    if (!run_recovered_patchmatch_filter_chain_cuda(
            input.filter, filtered, error))
        return false;

    if (input.filter.depth_downscale == 0U) {
        error = "PatchMatch level boundary downscale is invalid";
        return false;
    }
    const std::uint32_t width = ceil_div(
        input.filter.camera.width_original, input.filter.depth_downscale);
    const std::uint32_t height = ceil_div(
        input.filter.camera.height_original, input.filter.depth_downscale);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (filtered.depth_allocation.size() < pixels ||
        filtered.estimated_normal_allocation.size() < pixels * 3U ||
        input.bilateral_image.size() != pixels) {
        error = "PatchMatch level boundary active allocations are invalid";
        return false;
    }
    if (!filter_recovered_patchmatch_cuda_speckle_components(
            width, height, filtered.filtered_mask_allocation,
            filtered.depth_allocation,
            input.speckle_component_size_threshold, error))
        return false;

    PatchMatchBilateralU8Input bilateral;
    bilateral.width = width;
    bilateral.height = height;
    bilateral.depth.assign(filtered.depth_allocation.begin(),
                           filtered.depth_allocation.begin() + pixels);
    bilateral.normal.assign(filtered.estimated_normal_allocation.begin(),
                            filtered.estimated_normal_allocation.begin() +
                                pixels * 3U);
    bilateral.image = input.bilateral_image;
    bilateral.sigma_d = input.bilateral_sigma_d;
    bilateral.sigma_r = input.bilateral_sigma_r;
    bilateral.device_index = input.filter.device_index;

    PatchMatchBilateralU8Output bilateral_output;
    if (!run_recovered_patchmatch_bilateral_u8_cuda(
            bilateral, bilateral_output, error))
        return false;
    output.filter = std::move(filtered);
    output.bilateral = std::move(bilateral_output);
    return true;
}

static_assert(sizeof(PatchMatchCamera) == 272);
static_assert(offsetof(PatchMatchCamera, f) == 0);
static_assert(offsetof(PatchMatchCamera, pyramid_level0_downscale) == 20);
static_assert(offsetof(PatchMatchCamera, type) == 32);
static_assert(offsetof(PatchMatchCamera, line_model_num) == 48);
static_assert(offsetof(PatchMatchCamera, inv_line_model_num) == 112);
static_assert(offsetof(PatchMatchCamera, transform) == 208);

}  // namespace metmodel
