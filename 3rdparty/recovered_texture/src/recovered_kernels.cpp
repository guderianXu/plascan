#include "metashape_texture/recovered_kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace metashape_texture::recovered {
namespace {

void require_single_channel(const Image<float> &image) {
    if (image.channels() != 1) {
        throw std::invalid_argument("kernel requires a single-channel image");
    }
}

float clamped_load(const Image<float> &image, int x, int y, int channel) {
    x = std::clamp(x, 0, image.width() - 1);
    y = std::clamp(y, 0, image.height() - 1);
    return image(x, y, channel);
}

std::uint16_t positive_float_to_half(float value) {
    if (value == 0.0F) return 0;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    std::uint32_t rounded = (bits & 0x7fffffU) >> 13;
    const std::uint32_t remainder = bits & 0x1fffU;
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

}  // namespace

void update_face_resolution_max2(
    const std::vector<float> &input,
    std::vector<float> &maximum,
    std::vector<float> &second_greatest) {
    if (maximum.size() != input.size() || second_greatest.size() != input.size()) {
        throw std::invalid_argument("face-resolution max2 buffer sizes do not match");
    }
    for (std::size_t face = 0; face < input.size(); ++face) {
        const float value = input[face];
        if (value == 0.0F || value == -1.0F) continue;
        if (value >= maximum[face]) {
            second_greatest[face] = maximum[face];
            maximum[face] = value;
        } else if (value >= second_greatest[face]) {
            second_greatest[face] = value;
        }
    }
}

std::vector<float> second_greatest_face_resolution(
    const std::vector<std::vector<float>> &camera_resolution) {
    if (camera_resolution.empty()) return {};
    const std::size_t face_count = camera_resolution.front().size();
    std::vector<float> maximum(face_count, 0.0F);
    std::vector<float> second(face_count, 0.0F);
    for (const auto &input : camera_resolution) {
        if (input.size() != face_count) {
            throw std::invalid_argument("camera face-resolution sizes do not match");
        }
        update_face_resolution_max2(input, maximum, second);
    }
    return second;
}

bool candidate_exists_from_scale4_resolution(float resolution) noexcept {
    return resolution != 0.0F;
}

Image<float> heat_diffusion_step(const Image<float> &input,
                                 const Image<std::uint8_t> &constraint_mask) {
    require_single_channel(input);
    if (constraint_mask.width() != input.width() || constraint_mask.height() != input.height() ||
        constraint_mask.channels() != 1) {
        throw std::invalid_argument("heat-diffusion mask dimensions do not match");
    }
    Image<float> output(input.width(), input.height());
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            const bool boundary = x == 0 || y == 0 || x + 1 == input.width() ||
                                  y + 1 == input.height();
            float value = 0.0F;
            if (!boundary && constraint_mask(x, y) == 0) {
                value = 0.2F * (input(x, y) + input(x - 1, y) + input(x + 1, y) +
                                input(x, y - 1) + input(x, y + 1));
            }
            if (input(x, y) > 0.0F) {
                value = std::max(value, 1.0e-7F);
            }
            output(x, y) = value;
        }
    }
    return output;
}

Image<float> chamfer_distance_step(const Image<float> &distance,
                                   const Image<std::uint8_t> &mask,
                                   float maximum_distance) {
    require_single_channel(distance);
    if (mask.width() != distance.width() || mask.height() != distance.height() ||
        mask.channels() != 1 || maximum_distance < 0.0F) {
        throw std::invalid_argument("invalid chamfer-distance arguments");
    }
    Image<float> output(distance.width(), distance.height(), 1, maximum_distance);
    for (int y = 0; y < distance.height(); ++y) {
        for (int x = 0; x < distance.width(); ++x) {
            if (mask(x, y) == 255) {
                output(x, y) = 0.0F;
                continue;
            }
            float best = std::min(distance(x, y), maximum_distance);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= distance.width() || ny >= distance.height()) continue;
                    if (mask(nx, ny) == 255) continue;
                    if (dx != 0 && dy != 0) {
                        // The recovered shader forbids a diagonal step through either masked
                        // orthogonal cross-neighbour.
                        if (mask(x + dx, y) == 255 || mask(x, y + dy) == 255) continue;
                    }
                    const float step = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                    best = std::min(best, std::min(maximum_distance, distance(nx, ny) + step));
                }
            }
            output(x, y) = best;
        }
    }
    return output;
}

Image<float> downscale_distance_2x(const Image<float> &input,
                                   const Image<std::uint8_t> &mask,
                                   Image<std::uint8_t> *output_mask) {
    require_single_channel(input);
    if (mask.width() != input.width() || mask.height() != input.height() || mask.channels() != 1) {
        throw std::invalid_argument("distance-downscale mask dimensions do not match");
    }
    const int width = (input.width() + 1) / 2;
    const int height = (input.height() + 1) / 2;
    Image<float> output(width, height);
    Image<std::uint8_t> reduced_mask(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0F;
            int samples = 0;
            bool masked = false;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = 2 * x + dx;
                    const int sy = 2 * y + dy;
                    if (sx >= input.width() || sy >= input.height()) continue;
                    sum += input(sx, sy);
                    ++samples;
                    masked = masked || mask(sx, sy) == 255;
                }
            }
            output(x, y) = samples == 0 ? 0.0F : sum / static_cast<float>(samples);
            reduced_mask(x, y) = masked ? 255 : 0;
        }
    }
    if (output_mask != nullptr) *output_mask = std::move(reduced_mask);
    return output;
}

Image<float> minimum_filter_square(const Image<float> &input, int radius) {
    require_single_channel(input);
    if (radius < 0) throw std::invalid_argument("negative minimum-filter radius");
    Image<float> output(input.width(), input.height());
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            float value = std::numeric_limits<float>::infinity();
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = std::clamp(x + dx, 0, input.width() - 1);
                    const int sy = std::clamp(y + dy, 0, input.height() - 1);
                    value = std::min(value, input(sx, sy));
                }
            }
            output(x, y) = value;
        }
    }
    return output;
}

Image<float> downscale_pyramid_6tap(const Image<float> &input) {
    if (input.empty()) return {};
    static constexpr std::array<float, 6> kernel = {1.0F, 5.0F, 10.0F, 10.0F, 5.0F, 1.0F};
    const int width = (input.width() + 1) / 2;
    const int height = (input.height() + 1) / 2;
    Image<float> output(width, height, input.channels());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < input.channels(); ++channel) {
                float sum = 0.0F;
                float normalization = 0.0F;
                for (int ky = 0; ky < 6; ++ky) {
                    const int sy = 2 * y + ky - 2;
                    if (sy < 0 || sy >= input.height()) continue;
                    for (int kx = 0; kx < 6; ++kx) {
                        const int sx = 2 * x + kx - 2;
                        if (sx < 0 || sx >= input.width()) continue;
                        const float weight = kernel[kx] * kernel[ky];
                        sum += weight * input(sx, sy, channel);
                        normalization += weight;
                    }
                }
                output(x, y, channel) = normalization == 0.0F ? 0.0F : sum / normalization;
            }
        }
    }
    return output;
}

Image<float> sharpen_cross(const Image<float> &input, float sharpening) {
    if (input.channels() < 2) {
        throw std::invalid_argument("sharpening kernel requires a green channel");
    }
    if (sharpening < 0.0F || sharpening > 1.0F) {
        throw std::invalid_argument("sharpening must be in [0, 1]");
    }
    Image<float> output(input.width(), input.height(), input.channels());
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            const std::array<float, 5> green = {
                input(x, y, 1), clamped_load(input, x - 1, y, 1),
                clamped_load(input, x + 1, y, 1), clamped_load(input, x, y - 1, 1),
                clamped_load(input, x, y + 1, 1)};
            const auto [minimum, maximum] = std::minmax_element(green.begin(), green.end());
            const float adapt = std::sqrt(std::min(1.0F - *maximum, *minimum) /
                                          std::max(*maximum, 1.0e-6F));
            const float coefficient = adapt * ((1.0F - sharpening) * -0.125F +
                                                sharpening * -0.2F);
            const float denominator = 1.0F + 4.0F * coefficient;
            for (int channel = 0; channel < input.channels(); ++channel) {
                const float neighbours = clamped_load(input, x - 1, y, channel) +
                                         clamped_load(input, x + 1, y, channel) +
                                         clamped_load(input, x, y - 1, channel) +
                                         clamped_load(input, x, y + 1, channel);
                output(x, y, channel) =
                    (input(x, y, channel) + coefficient * neighbours) / denominator;
            }
        }
    }
    return output;
}

Image<std::uint8_t> remove_small_components_8(
    const Image<std::uint8_t> &input,
    std::size_t minimum_size,
    std::uint8_t background) {
    if (input.channels() != 1) {
        throw std::invalid_argument("component filter requires a single-channel image");
    }
    if (minimum_size == 0 || input.empty()) return input;

    // The native implementation at 0x143f411f0/0x143f41900 uses three
    // int32 arrays: parent (-1 roots), rank (0) and component size (1).  This
    // translation keeps the same union-by-rank/size semantics while using a
    // conventional self-parent root representation.
    const std::size_t pixel_count = input.pixels().size();
    std::vector<std::size_t> parent(pixel_count);
    std::vector<std::uint8_t> rank(pixel_count, 0);
    std::vector<std::size_t> component_size(pixel_count, 1);
    std::iota(parent.begin(), parent.end(), std::size_t{0});

    const auto find_root = [&](std::size_t value, const auto &self) -> std::size_t {
        if (parent[value] != value) parent[value] = self(parent[value], self);
        return parent[value];
    };
    const auto unite = [&](std::size_t first, std::size_t second) {
        std::size_t first_root = find_root(first, find_root);
        std::size_t second_root = find_root(second, find_root);
        if (first_root == second_root) return;
        if (rank[first_root] < rank[second_root]) std::swap(first_root, second_root);
        parent[second_root] = first_root;
        component_size[first_root] += component_size[second_root];
        if (rank[first_root] == rank[second_root]) ++rank[first_root];
    };

    const int width = input.width();
    const int height = input.height();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (input(x, y) == background) continue;
            const std::size_t current = static_cast<std::size_t>(y) * width + x;
            // One half of the 8-neighbourhood is sufficient for union.  The
            // native parallel worker visits the equivalent neighbour pairs.
            for (const auto [dx, dy] :
                 std::array<std::array<int, 2>, 4>{{{{-1, 0}}, {{-1, -1}},
                                                     {{0, -1}}, {{1, -1}}}}) {
                const int neighbor_x = x + dx;
                const int neighbor_y = y + dy;
                if (neighbor_x < 0 || neighbor_y < 0 || neighbor_x >= width ||
                    neighbor_y >= height || input(neighbor_x, neighbor_y) == background) {
                    continue;
                }
                const std::size_t neighbor =
                    static_cast<std::size_t>(neighbor_y) * width + neighbor_x;
                unite(current, neighbor);
            }
        }
    }

    Image<std::uint8_t> output = input;
    for (std::size_t index = 0; index < pixel_count; ++index) {
        if (input.pixels()[index] == background) continue;
        const std::size_t root = find_root(index, find_root);
        if (component_size[root] < minimum_size) output.pixels()[index] = background;
    }
    return output;
}

Image<float> initialize_camera_weights(
    const Image<std::int32_t> &face_ids,
    const std::vector<std::uint32_t> &face_labels,
    const Image<std::uint8_t> &occlusion_mask,
    std::uint32_t camera_label) {
    if (face_ids.channels() != 1 || occlusion_mask.channels() != 1 ||
        face_ids.width() != occlusion_mask.width() ||
        face_ids.height() != occlusion_mask.height()) {
        throw std::invalid_argument("weight-init input dimensions do not match");
    }
    Image<float> output(face_ids.width(), face_ids.height());
    for (int y = 0; y < face_ids.height(); ++y) {
        for (int x = 0; x < face_ids.width(); ++x) {
            const bool boundary = x == 0 || y == 0 || x + 1 == face_ids.width() ||
                                  y + 1 == face_ids.height();
            if (boundary || occlusion_mask(x, y) == 255) {
                output(x, y) = 0.0F;
                continue;
            }
            const std::int32_t face_id = face_ids(x, y);
            std::uint32_t label = std::numeric_limits<std::uint32_t>::max();
            if (face_id != std::numeric_limits<std::int32_t>::max()) {
                if (face_id < 0 || static_cast<std::size_t>(face_id) >= face_labels.size()) {
                    throw std::out_of_range("module133 face id is outside the label buffer");
                }
                label = face_labels[static_cast<std::size_t>(face_id)];
            }
            output(x, y) = label == camera_label ? 1.0F : 0.0F;
        }
    }
    return output;
}

Image<float> distance_via_bfs(
    const Image<std::int32_t> &constraint,
    const Image<float> &camera_value,
    int mode,
    float maximum_distance) {
    if (constraint.channels() != 1 || camera_value.channels() != 1 ||
        constraint.width() != camera_value.width() ||
        constraint.height() != camera_value.height() ||
        (mode != 1 && mode != 2 && mode != 3) || maximum_distance < 0.0F) {
        throw std::invalid_argument("invalid distance-via-bfs arguments");
    }
    const int width = constraint.width();
    const int height = constraint.height();
    Image<float> current(width, height, 1, maximum_distance);
    const int passes = static_cast<int>(maximum_distance) + 1;
    for (int pass = 0; pass < passes; ++pass) {
        Image<float> next(width, height, 1, maximum_distance);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const bool border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
                const bool constrained = border || constraint(x, y) != 0;
                if (mode == 1 && constrained) {
                    next(x, y) = 0.0F;
                    continue;
                }
                if (mode != 1 && constrained) continue;
                if ((mode == 2 && camera_value(x, y) == 1.0F) ||
                    (mode == 3 && camera_value(x, y) != 1.0F)) {
                    next(x, y) = 0.0F;
                    continue;
                }
                float best = maximum_distance;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int source_x = x + dx;
                        const int source_y = y + dy;
                        if (source_x < 0 || source_y < 0 || source_x >= width ||
                            source_y >= height) {
                            continue;
                        }
                        if (mode != 1) {
                            if (constraint(source_x, source_y) != 0) continue;
                            // This is an AND in module144: diagonal movement is
                            // blocked only when both orthogonal crossings are constrained.
                            if (dx != 0 && dy != 0 &&
                                constraint(x, source_y) != 0 &&
                                constraint(source_x, y) != 0) {
                                continue;
                            }
                        }
                        const float step = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                        const float candidate = std::min(
                            current(source_x, source_y) + step, maximum_distance);
                        if (candidate < best) best = candidate;
                    }
                }
                next(x, y) = best;
            }
        }
        for (float &value : next.pixels()) {
            value = positive_half_to_float(positive_float_to_half(value));
        }
        current = std::move(next);
    }
    return current;
}

Image<float> distances_to_weights(
    const Image<float> &boundary_distance,
    const Image<float> &winner_distance,
    const Image<float> &nonwinner_distance,
    const Image<float> &previous_value,
    float maximum_distance) {
    require_single_channel(boundary_distance);
    require_single_channel(winner_distance);
    require_single_channel(nonwinner_distance);
    require_single_channel(previous_value);
    const int width = previous_value.width();
    const int height = previous_value.height();
    const auto same_size = [width, height](const Image<float> &image) {
        return image.width() == width && image.height() == height;
    };
    if (!same_size(boundary_distance) || !same_size(winner_distance) ||
        !same_size(nonwinner_distance) || maximum_distance <= 0.0F) {
        throw std::invalid_argument("distance-to-weight dimensions do not match");
    }
    Image<float> output(width, height);
    const float half_distance = maximum_distance * 0.5F;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float edge = boundary_distance(x, y);
            const float to_winner = winner_distance(x, y);       // shader binding 1
            const float to_nonwinner = nonwinner_distance(x, y); // shader binding 2
            float value;
            if (to_nonwinner >= half_distance) {
                value = 1.0F;
            } else if (to_winner >= half_distance) {
                value = 0.0F;
            } else if (to_winner == 0.0F) {
                value = 0.5F + (0.5F * to_nonwinner) / half_distance;
            } else {
                value = 0.5F - (0.5F * to_winner) / half_distance;
            }
            if (edge < maximum_distance) {
                value *= std::min(1.0F, edge / maximum_distance);
            }
            if (previous_value(x, y) > 0.0F) value = std::max(value, 1.0e-7F);
            output(x, y) = value;
        }
    }
    return output;
}

Image<float> build_distance_weights(
    const Image<std::int32_t> &constraint,
    const Image<float> &camera_value,
    float maximum_distance) {
    const Image<float> boundary = distance_via_bfs(constraint, camera_value, 1, maximum_distance);
    const Image<float> winner = distance_via_bfs(constraint, camera_value, 2, maximum_distance);
    const Image<float> nonwinner = distance_via_bfs(constraint, camera_value, 3, maximum_distance);
    return distances_to_weights(boundary, winner, nonwinner, camera_value, maximum_distance);
}

std::uint16_t quantize_weight(float weight) {
    if (weight == 0.0F) return 0;
    const float normalized = std::clamp(std::max(weight, 1.0F / 65535.0F), 0.0F, 1.0F);
    return static_cast<std::uint16_t>(std::lround(normalized * 65535.0F));
}

}  // namespace metashape_texture::recovered
