#include "metashape_texture/natural_uv_parameterization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace metashape_texture {

NaturalUvSvd3x3 decompose_natural_uv_covariance_exact(
    const std::array<double, 9> &matrix);

namespace {

struct IncidentAngles {
    std::uint32_t center{};
    std::uint32_t next{};
    std::uint32_t previous{};
};

struct LscmEntry {
    std::size_t variable{};
    double value{};
};

double dot(const std::array<double, 3> &first, const std::array<double, 3> &second) {
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

double corner_angle(const float *center, const float *first, const float *second) {
    // Native subtracts float32 coordinates before promoting the edge vectors
    // to double for the dot products, square root, and acos.
    const std::array<double, 3> edge_first{
        static_cast<double>(first[0] - center[0]),
        static_cast<double>(first[1] - center[1]),
        static_cast<double>(first[2] - center[2]),
    };
    const std::array<double, 3> edge_second{
        static_cast<double>(second[0] - center[0]),
        static_cast<double>(second[1] - center[1]),
        static_cast<double>(second[2] - center[2]),
    };
    const double denominator = std::sqrt(dot(edge_first, edge_first) *
                                         dot(edge_second, edge_second));
    if (denominator == 0.0) {
        throw std::runtime_error("natural UV chart contains a degenerate corner");
    }
    return std::acos(std::clamp(dot(edge_first, edge_second) / denominator, -1.0, 1.0));
}

std::vector<double> solve_positive_definite(std::vector<double> matrix,
                                            std::vector<double> right_hand_side) {
    const std::size_t size = right_hand_side.size();
    if (matrix.size() != size * size) {
        throw std::invalid_argument("natural UV projection matrix shape mismatch");
    }

    // In-place lower Cholesky factorization.  The recovered constraint matrix
    // has full row rank, so J*J^T is positive definite for valid disk charts.
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = matrix[row * size + column];
            for (std::size_t inner = 0; inner < column; ++inner) {
                value -= matrix[row * size + inner] * matrix[column * size + inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("natural UV angle constraints are rank deficient");
                }
                matrix[row * size + column] = std::sqrt(value);
            } else {
                matrix[row * size + column] = value / matrix[column * size + column];
            }
        }
    }

    for (std::size_t row = 0; row < size; ++row) {
        double value = right_hand_side[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -= matrix[row * size + column] * right_hand_side[column];
        }
        right_hand_side[row] = value / matrix[row * size + row];
    }
    for (std::size_t reverse = size; reverse-- > 0;) {
        double value = right_hand_side[reverse];
        for (std::size_t row = reverse + 1; row < size; ++row) {
            value -= matrix[row * size + reverse] * right_hand_side[row];
        }
        right_hand_side[reverse] = value / matrix[reverse * size + reverse];
    }
    return right_hand_side;
}

double triangle_sqrt_area(const float *first,
                          const float *second,
                          const float *third) {
    // Preserve every float32 operation and, especially, the native reduction
    // order (x*x + y*y) + z*z.  Reassociating this sum changes some final UVs
    // by one float32 ULP on the 119-face Shoe oracle chart.
    const float first_x = third[0] - first[0];
    const float first_y = third[1] - first[1];
    const float first_z = third[2] - first[2];
    const float second_x = second[0] - first[0];
    const float second_y = second[1] - first[1];
    const float second_z = second[2] - first[2];
    const float cross_x = first_y * second_z - first_z * second_y;
    const float cross_y = first_z * second_x - first_x * second_z;
    const float cross_z = first_x * second_y - first_y * second_x;
    const float cross_xy_squared = cross_x * cross_x + cross_y * cross_y;
    const float cross_squared = cross_xy_squared + cross_z * cross_z;
    const float area = std::sqrt(cross_squared) * 0.5F;
    return std::sqrt(static_cast<double>(area));
}
}  // namespace

NaturalUvSvd3x3 decompose_natural_uv_covariance(
    const std::array<double, 9> &matrix) {
    return decompose_natural_uv_covariance_exact(matrix);
}

std::array<std::uint32_t, 2> select_natural_uv_anchors(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices) {
    if (vertex_xyz.size() % 3U != 0U || triangle_indices.size() % 3U != 0U) {
        throw std::invalid_argument("natural UV chart arrays are not tightly packed triples");
    }
    const std::size_t vertex_count = vertex_xyz.size() / 3U;
    if (vertex_count < 2U) {
        throw std::invalid_argument("natural UV chart needs two anchor vertices");
    }

    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    struct OrientedEdge {
        std::uint32_t first{};
        std::uint32_t second{};
    };
    std::map<Edge, std::vector<OrientedEdge>> edge_uses;
    for (std::size_t face = 0; face < triangle_indices.size() / 3U; ++face) {
        for (std::size_t edge = 0; edge < 3U; ++edge) {
            const std::uint32_t first = triangle_indices[3U * face + edge];
            const std::uint32_t second = triangle_indices[3U * face + (edge + 1U) % 3U];
            if (first >= vertex_count || second >= vertex_count) {
                throw std::out_of_range("natural UV local vertex index");
            }
            edge_uses[std::minmax(first, second)].push_back({first, second});
        }
    }
    constexpr std::int32_t missing = -1;
    std::vector<std::int32_t> next(vertex_count, missing);
    for (const auto &[edge, uses] : edge_uses) {
        if (uses.size() != 1U) continue;
        next[uses.front().first] = static_cast<std::int32_t>(uses.front().second);
    }

    std::vector<std::int32_t> working = next;
    std::int32_t longest_start = missing;
    std::size_t longest_length = 0U;
    for (std::size_t start = 0; start < vertex_count; ++start) {
        if (working[start] == missing) continue;
        std::size_t length = 0U;
        std::int32_t vertex = static_cast<std::int32_t>(start);
        while (vertex >= 0 && working[static_cast<std::size_t>(vertex)] != missing) {
            const std::int32_t successor = working[static_cast<std::size_t>(vertex)];
            working[static_cast<std::size_t>(vertex)] = missing;
            vertex = successor;
            ++length;
        }
        if (length > longest_length) {
            longest_start = static_cast<std::int32_t>(start);
            longest_length = length;
        }
    }
    if (longest_start == missing || longest_length < 2U) {
        throw std::runtime_error("natural UV chart has no usable boundary loop");
    }

    std::vector<std::uint32_t> loop;
    loop.reserve(longest_length);
    std::int32_t current = longest_start;
    do {
        if (current < 0 || loop.size() > vertex_count) {
            throw std::runtime_error("natural UV chart boundary is not a closed loop");
        }
        loop.push_back(static_cast<std::uint32_t>(current));
        current = next[static_cast<std::size_t>(current)];
    } while (current != longest_start);

    std::array<double, 3> center{};
    for (const std::uint32_t vertex : loop) {
        const float *xyz = vertex_xyz.data() + static_cast<std::size_t>(vertex) * 3U;
        center[0] += static_cast<double>(xyz[0]);
        center[1] += static_cast<double>(xyz[1]);
        center[2] += static_cast<double>(xyz[2]);
    }
    for (double &coordinate : center) coordinate /= static_cast<double>(loop.size());

    std::array<double, 9> covariance{};
    for (const std::uint32_t vertex : loop) {
        const float *xyz = vertex_xyz.data() + static_cast<std::size_t>(vertex) * 3U;
        const std::array<double, 3> offset{
            static_cast<double>(xyz[0]) - center[0],
            static_cast<double>(xyz[1]) - center[1],
            static_cast<double>(xyz[2]) - center[2],
        };
        for (std::size_t row = 0; row < 3U; ++row) {
            for (std::size_t column = 0; column < 3U; ++column) {
                covariance[3U * row + column] += offset[row] * offset[column];
            }
        }
    }
    const NaturalUvSvd3x3 decomposition =
        decompose_natural_uv_covariance(covariance);
    const std::size_t dominant = decomposition.dominant_index;
    const std::array<double, 3> direction{
        decomposition.left_columns[dominant],
        decomposition.left_columns[3U + dominant],
        decomposition.left_columns[6U + dominant],
    };

    double minimum_projection = std::numeric_limits<double>::infinity();
    double maximum_projection = -std::numeric_limits<double>::infinity();
    std::uint32_t minimum_vertex = loop.front();
    std::uint32_t maximum_vertex = loop.front();
    for (const std::uint32_t vertex : loop) {
        const float *xyz = vertex_xyz.data() + static_cast<std::size_t>(vertex) * 3U;
        const double projection =
            (static_cast<double>(xyz[0]) - center[0]) * direction[0] +
            (static_cast<double>(xyz[1]) - center[1]) * direction[1] +
            (static_cast<double>(xyz[2]) - center[2]) * direction[2];
        if (projection < minimum_projection) {
            minimum_projection = projection;
            minimum_vertex = vertex;
        }
        if (projection > maximum_projection) {
            maximum_projection = projection;
            maximum_vertex = vertex;
        }
    }
    if (minimum_vertex == maximum_vertex) {
        throw std::runtime_error("natural UV chart boundary anchors coincide");
    }
    return {minimum_vertex, maximum_vertex};
}

NaturalUvAngleProjection project_natural_uv_angles(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices) {
    if (vertex_xyz.size() % 3U != 0U || triangle_indices.size() % 3U != 0U) {
        throw std::invalid_argument("natural UV chart arrays are not tightly packed triples");
    }
    const std::size_t vertex_count = vertex_xyz.size() / 3U;
    const std::size_t face_count = triangle_indices.size() / 3U;
    if (face_count == 0U) throw std::invalid_argument("cannot parameterize an empty chart");

    NaturalUvAngleProjection result;
    result.angles.resize(face_count * 3U);
    result.boundary_vertices.assign(vertex_count, false);
    std::vector<std::vector<IncidentAngles>> incident(vertex_count);
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Edge, std::vector<std::pair<std::uint32_t, std::uint32_t>>> edge_uses;

    for (std::uint32_t face = 0; face < face_count; ++face) {
        const auto *vertices = triangle_indices.data() + static_cast<std::size_t>(face) * 3U;
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            if (vertices[corner] >= vertex_count) {
                throw std::out_of_range("natural UV local vertex index");
            }
            const std::uint32_t next_corner = (corner + 1U) % 3U;
            const std::uint32_t previous_corner = (corner + 2U) % 3U;
            const auto *center = vertex_xyz.data() + static_cast<std::size_t>(vertices[corner]) * 3U;
            const auto *next = vertex_xyz.data() +
                               static_cast<std::size_t>(vertices[next_corner]) * 3U;
            const auto *previous = vertex_xyz.data() +
                                   static_cast<std::size_t>(vertices[previous_corner]) * 3U;
            result.angles[static_cast<std::size_t>(face) * 3U + corner] =
                corner_angle(center, next, previous);
            incident[vertices[corner]].push_back({
                face * 3U + corner,
                face * 3U + next_corner,
                face * 3U + previous_corner,
            });
            edge_uses[std::minmax(vertices[corner], vertices[next_corner])].emplace_back(
                face, corner);
        }
    }
    for (const auto &[edge, uses] : edge_uses) {
        if (uses.size() == 2U) continue;
        result.boundary_vertices[edge.first] = true;
        result.boundary_vertices[edge.second] = true;
    }

    std::vector<std::uint32_t> interior_vertices;
    for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (!result.boundary_vertices[vertex]) interior_vertices.push_back(vertex);
    }

    constexpr double minimum_angle = 2.0 * std::numbers::pi_v<double> / 180.0;
    constexpr double maximum_angle = 178.0 * std::numbers::pi_v<double> / 180.0;
    for (const std::uint32_t vertex : interior_vertices) {
        double sum = 0.0;
        for (const IncidentAngles &angles : incident[vertex]) {
            sum += result.angles[angles.center];
        }
        const double scale = 2.0 * std::numbers::pi_v<double> / sum;
        for (const IncidentAngles &angles : incident[vertex]) {
            result.angles[angles.center] = std::clamp(
                result.angles[angles.center] * scale, minimum_angle, maximum_angle);
        }
    }

    const std::size_t constraint_count = face_count + 2U * interior_vertices.size();
    const std::size_t angle_count = result.angles.size();
    result.constraint_count = static_cast<std::uint32_t>(constraint_count);
    std::vector<double> jacobian(constraint_count * angle_count, 0.0);
    std::vector<double> residual(constraint_count, 0.0);
    std::size_t row = 0;
    for (const std::uint32_t vertex : interior_vertices) {
        double sum = 0.0;
        for (const IncidentAngles &angles : incident[vertex]) {
            jacobian[row * angle_count + angles.center] = result.angles[angles.center];
            sum += result.angles[angles.center];
        }
        residual[row++] = 2.0 * std::numbers::pi_v<double> - sum;
    }
    for (std::uint32_t face = 0; face < face_count; ++face) {
        double sum = 0.0;
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t angle = face * 3U + corner;
            jacobian[row * angle_count + angle] = result.angles[angle];
            sum += result.angles[angle];
        }
        residual[row++] = std::numbers::pi_v<double> - sum;
    }
    for (const std::uint32_t vertex : interior_vertices) {
        double value = 0.0;
        for (const IncidentAngles &angles : incident[vertex]) {
            const double next = result.angles[angles.next];
            const double previous = result.angles[angles.previous];
            jacobian[row * angle_count + angles.next] += next / std::tan(next);
            jacobian[row * angle_count + angles.previous] -= previous / std::tan(previous);
            value += std::log(std::sin(previous)) - std::log(std::sin(next));
        }
        residual[row++] = value;
    }

    std::vector<double> normal(constraint_count * constraint_count, 0.0);
    for (std::size_t first = 0; first < constraint_count; ++first) {
        for (std::size_t second = 0; second <= first; ++second) {
            double value = 0.0;
            for (std::size_t angle = 0; angle < angle_count; ++angle) {
                value += jacobian[first * angle_count + angle] *
                         jacobian[second * angle_count + angle];
            }
            normal[first * constraint_count + second] = value;
            normal[second * constraint_count + first] = value;
        }
    }
    const std::vector<double> multipliers = solve_positive_definite(
        std::move(normal), std::move(residual));
    for (std::size_t angle = 0; angle < angle_count; ++angle) {
        double relative_delta = 0.0;
        for (std::size_t constraint = 0; constraint < constraint_count; ++constraint) {
            relative_delta += jacobian[constraint * angle_count + angle] *
                              multipliers[constraint];
        }
        result.angles[angle] *= 1.0 + relative_delta;
    }
    return result;
}

NaturalUvParameterization parameterize_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::uint32_t anchor_first,
    std::uint32_t anchor_second) {
    if (vertex_xyz.size() % 3U != 0U || triangle_indices.size() % 3U != 0U) {
        throw std::invalid_argument("natural UV chart arrays are not tightly packed triples");
    }
    const std::size_t vertex_count = vertex_xyz.size() / 3U;
    const std::size_t face_count = triangle_indices.size() / 3U;
    if (anchor_first >= vertex_count || anchor_second >= vertex_count ||
        anchor_first == anchor_second) {
        throw std::out_of_range("natural UV chart anchors");
    }

    NaturalUvParameterization result;
    result.anchor_first = anchor_first;
    result.anchor_second = anchor_second;
    result.angle_projection = project_natural_uv_angles(vertex_xyz, triangle_indices);

    const std::array<std::size_t, 4> fixed_variables{
        2U * anchor_first,
        2U * anchor_first + 1U,
        2U * anchor_second,
        2U * anchor_second + 1U,
    };
    constexpr std::array<double, 4> fixed_values{0.0, 0.0, 0.0, 1.0};
    std::vector<std::size_t> free_variable(vertex_count * 2U, 0U);
    std::vector<bool> fixed(vertex_count * 2U, false);
    for (const std::size_t variable : fixed_variables) fixed[variable] = true;
    std::size_t free_count = 0U;
    for (std::size_t variable = 0; variable < fixed.size(); ++variable) {
        if (!fixed[variable]) free_variable[variable] = free_count++;
    }

    std::vector<std::vector<LscmEntry>> rows(face_count * 2U);
    std::vector<double> right_hand_side(face_count * 2U, 0.0);
    auto add_entry = [&](std::size_t row, std::size_t variable, double value) {
        if (!fixed[variable]) {
            rows[row].push_back({free_variable[variable], value});
            return;
        }
        const auto iterator = std::find(
            fixed_variables.begin(), fixed_variables.end(), variable);
        const std::size_t fixed_index = static_cast<std::size_t>(
            iterator - fixed_variables.begin());
        right_hand_side[row] -= value * fixed_values[fixed_index];
    };

    for (std::size_t face = 0; face < face_count; ++face) {
        const std::uint32_t first_vertex = triangle_indices[3U * face];
        const std::uint32_t second_vertex = triangle_indices[3U * face + 1U];
        const std::uint32_t third_vertex = triangle_indices[3U * face + 2U];
        if (first_vertex >= vertex_count || second_vertex >= vertex_count ||
            third_vertex >= vertex_count) {
            throw std::out_of_range("natural UV local vertex index");
        }
        if (first_vertex == second_vertex || second_vertex == third_vertex ||
            first_vertex == third_vertex) {
            continue;
        }
        const double weight = triangle_sqrt_area(
            vertex_xyz.data() + static_cast<std::size_t>(first_vertex) * 3U,
            vertex_xyz.data() + static_cast<std::size_t>(second_vertex) * 3U,
            vertex_xyz.data() + static_cast<std::size_t>(third_vertex) * 3U);
        const double first_angle = result.angle_projection.angles[3U * face];
        const double second_angle = result.angle_projection.angles[3U * face + 1U];
        const double third_angle = result.angle_projection.angles[3U * face + 2U];
        const double ratio = std::sin(second_angle) / std::sin(third_angle);
        const double cosine = std::cos(first_angle) * ratio;
        const double sine = std::sin(first_angle) * ratio;

        const std::size_t first_u = 2U * first_vertex;
        const std::size_t first_v = first_u + 1U;
        const std::size_t second_u = 2U * second_vertex;
        const std::size_t second_v = second_u + 1U;
        const std::size_t third_u = 2U * third_vertex;
        const std::size_t third_v = third_u + 1U;
        const std::size_t real_row = 2U * face;
        const std::size_t imaginary_row = real_row + 1U;

        add_entry(real_row, first_u, (1.0 - cosine) * weight);
        add_entry(real_row, first_v, sine * weight);
        add_entry(real_row, second_u, cosine * weight);
        add_entry(real_row, second_v, -sine * weight);
        add_entry(real_row, third_u, -weight);

        add_entry(imaginary_row, first_u, -sine * weight);
        add_entry(imaginary_row, first_v, (1.0 - cosine) * weight);
        add_entry(imaginary_row, second_u, sine * weight);
        add_entry(imaginary_row, second_v, cosine * weight);
        add_entry(imaginary_row, third_v, -weight);
    }

    std::vector<double> normal(free_count * free_count, 0.0);
    std::vector<double> normal_right_hand_side(free_count, 0.0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        for (const LscmEntry &first : rows[row]) {
            normal_right_hand_side[first.variable] +=
                first.value * right_hand_side[row];
            for (const LscmEntry &second : rows[row]) {
                normal[first.variable * free_count + second.variable] +=
                    first.value * second.value;
            }
        }
    }
    const std::vector<double> free_solution = solve_positive_definite(
        std::move(normal), std::move(normal_right_hand_side));

    result.uv.resize(vertex_count * 2U);
    for (std::size_t variable = 0; variable < result.uv.size(); ++variable) {
        double value = 0.0;
        if (fixed[variable]) {
            const auto iterator = std::find(
                fixed_variables.begin(), fixed_variables.end(), variable);
            value = fixed_values[static_cast<std::size_t>(
                iterator - fixed_variables.begin())];
        } else {
            value = free_solution[free_variable[variable]];
        }
        result.uv[variable] = static_cast<float>(value);
    }
    return result;
}

NaturalUvParameterization parameterize_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices) {
    const auto anchors = select_natural_uv_anchors(vertex_xyz, triangle_indices);
    return parameterize_natural_uv_chart(
        vertex_xyz, triangle_indices, anchors[0], anchors[1]);
}

double measure_natural_uv_stretch(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv) {
    if (vertex_xyz.size() % 3U != 0U || vertex_uv.size() % 2U != 0U ||
        triangle_indices.size() % 3U != 0U ||
        vertex_xyz.size() / 3U != vertex_uv.size() / 2U) {
        throw std::invalid_argument("natural UV validation arrays have incompatible shapes");
    }
    const std::size_t vertex_count = vertex_uv.size() / 2U;
    struct Areas {
        double uv{};
        double xyz{};
    };
    std::vector<Areas> areas;
    areas.reserve(triangle_indices.size() / 3U);
    double maximum_uv_area = 0.0;
    double maximum_xyz_area = 0.0;
    for (std::size_t face = 0; face < triangle_indices.size() / 3U; ++face) {
        const std::uint32_t first_index = triangle_indices[3U * face];
        const std::uint32_t second_index = triangle_indices[3U * face + 1U];
        const std::uint32_t third_index = triangle_indices[3U * face + 2U];
        if (first_index >= vertex_count || second_index >= vertex_count ||
            third_index >= vertex_count) {
            throw std::out_of_range("natural UV validation vertex index");
        }
        const float *first_uv = vertex_uv.data() + 2U * first_index;
        const float *second_uv = vertex_uv.data() + 2U * second_index;
        const float *third_uv = vertex_uv.data() + 2U * third_index;
        const float uv_first_v = third_uv[1] - first_uv[1];
        const float uv_second_u = second_uv[0] - first_uv[0];
        const float uv_first_u = third_uv[0] - first_uv[0];
        const float uv_second_v = second_uv[1] - first_uv[1];
        const float uv_positive = uv_first_v * uv_second_u;
        const float uv_negative = uv_first_u * uv_second_v;
        const float uv_area = std::abs((uv_positive - uv_negative) * 0.5F);

        const float *first_xyz = vertex_xyz.data() + 3U * first_index;
        const float *second_xyz = vertex_xyz.data() + 3U * second_index;
        const float *third_xyz = vertex_xyz.data() + 3U * third_index;
        const float first_x = third_xyz[0] - first_xyz[0];
        const float first_y = third_xyz[1] - first_xyz[1];
        const float first_z = third_xyz[2] - first_xyz[2];
        const float second_x = second_xyz[0] - first_xyz[0];
        const float second_y = second_xyz[1] - first_xyz[1];
        const float second_z = second_xyz[2] - first_xyz[2];
        const float cross_x = first_y * second_z - first_z * second_y;
        const float cross_y = first_z * second_x - first_x * second_z;
        const float cross_z = first_x * second_y - first_y * second_x;
        const float cross_xy_squared = cross_x * cross_x + cross_y * cross_y;
        const float cross_squared = cross_xy_squared + cross_z * cross_z;
        const float xyz_area = std::sqrt(cross_squared) * 0.5F;

        areas.push_back({static_cast<double>(uv_area), static_cast<double>(xyz_area)});
        maximum_uv_area = std::max(maximum_uv_area, static_cast<double>(uv_area));
        maximum_xyz_area = std::max(maximum_xyz_area, static_cast<double>(xyz_area));
    }

    std::vector<double> ratios;
    ratios.reserve(areas.size());
    for (const Areas &area : areas) {
        if (area.uv >= 0.001 * maximum_uv_area ||
            area.xyz >= 0.001 * maximum_xyz_area) {
            ratios.push_back(area.uv / area.xyz);
        }
    }
    if (ratios.empty()) return 0.0;
    std::sort(ratios.begin(), ratios.end());
    return ratios.back() / ratios.front();
}

NaturalUvRasterMetrics measure_natural_uv_raster_quality(
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    std::uint32_t maximum_dimension) {
    if (vertex_uv.size() % 2U != 0U || triangle_indices.size() % 3U != 0U) {
        throw std::invalid_argument("natural UV raster arrays are not tightly packed");
    }
    NaturalUvRasterMetrics result;
    if (vertex_uv.empty() || maximum_dimension == 0U) return result;
    const std::size_t vertex_count = vertex_uv.size() / 2U;
    float minimum_u = vertex_uv[0];
    float maximum_u = vertex_uv[0];
    float minimum_v = vertex_uv[1];
    float maximum_v = vertex_uv[1];
    for (std::size_t vertex = 1U; vertex < vertex_count; ++vertex) {
        minimum_u = std::min(minimum_u, vertex_uv[2U * vertex]);
        maximum_u = std::max(maximum_u, vertex_uv[2U * vertex]);
        minimum_v = std::min(minimum_v, vertex_uv[2U * vertex + 1U]);
        maximum_v = std::max(maximum_v, vertex_uv[2U * vertex + 1U]);
    }
    const float extent_u = maximum_u - minimum_u;
    const float extent_v = maximum_v - minimum_v;
    if (!(extent_u > 0.0F) || !(extent_v > 0.0F)) return result;
    if (extent_u > extent_v) {
        result.width = maximum_dimension;
        const float scaled = static_cast<float>(maximum_dimension) * extent_v / extent_u;
        result.height = std::min(maximum_dimension, static_cast<std::uint32_t>(scaled));
    } else {
        result.height = maximum_dimension;
        const float scaled = static_cast<float>(maximum_dimension) * extent_u / extent_v;
        result.width = std::min(maximum_dimension, static_cast<std::uint32_t>(scaled));
    }
    if (result.width == 0U || result.height == 0U) return result;

    std::vector<std::array<double, 2>> normalized(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        normalized[vertex] = {
            (static_cast<double>(vertex_uv[2U * vertex]) - minimum_u) * result.width /
                static_cast<double>(extent_u) - 0.5,
            (static_cast<double>(vertex_uv[2U * vertex + 1U]) - minimum_v) * result.height /
                static_cast<double>(extent_v) - 0.5,
        };
    }
    std::vector<std::uint8_t> raster(
        static_cast<std::size_t>(result.width) * result.height, 0U);
    for (std::size_t face = 0; face < triangle_indices.size() / 3U; ++face) {
        std::array<std::array<double, 2>, 3> points{};
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t vertex = triangle_indices[3U * face + corner];
            if (vertex >= vertex_count) {
                throw std::out_of_range("natural UV raster vertex index");
            }
            points[corner] = normalized[vertex];
        }
        const double minimum_x = std::min({points[0][0], points[1][0], points[2][0]});
        const double maximum_x = std::max({points[0][0], points[1][0], points[2][0]});
        const double minimum_y = std::min({points[0][1], points[1][1], points[2][1]});
        const double maximum_y = std::max({points[0][1], points[1][1], points[2][1]});
        const std::int64_t first_x = std::max<std::int64_t>(
            0, static_cast<std::int64_t>(std::ceil(minimum_x)));
        const std::int64_t last_x = std::min<std::int64_t>(
            static_cast<std::int64_t>(result.width) - 1,
            static_cast<std::int64_t>(std::floor(maximum_x)));
        const std::int64_t first_y = std::max<std::int64_t>(
            0, static_cast<std::int64_t>(std::ceil(minimum_y)));
        const std::int64_t last_y = std::min<std::int64_t>(
            static_cast<std::int64_t>(result.height) - 1,
            static_cast<std::int64_t>(std::floor(maximum_y)));
        if (last_x < first_x || last_y < first_y) continue;
        for (std::int64_t y = first_y; y <= last_y; ++y) {
            for (std::int64_t x = first_x; x <= last_x; ++x) {
                std::array<double, 3> edges{};
                for (std::size_t edge = 0; edge < 3U; ++edge) {
                    const auto &first = points[edge];
                    const auto &second = points[(edge + 1U) % 3U];
                    edges[edge] =
                        (static_cast<double>(x) - first[0]) * (second[1] - first[1]) -
                        (static_cast<double>(y) - first[1]) * (second[0] - first[0]);
                }
                const bool inside_positive =
                    edges[0] >= 0.0 && edges[1] >= 0.0 && edges[2] >= 0.0;
                const bool inside_negative =
                    edges[0] <= 0.0 && edges[1] <= 0.0 && edges[2] <= 0.0;
                if (!inside_positive && !inside_negative) continue;
                std::uint8_t &value = raster[
                    static_cast<std::size_t>(y) * result.width +
                    static_cast<std::size_t>(x)];
                if (value != std::numeric_limits<std::uint8_t>::max()) ++value;
            }
        }
    }
    for (const std::uint8_t value : raster) {
        if (value != 0U) ++result.occupied_pixels;
        if (value >= 2U) ++result.overlap_pixels;
    }
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(result.width) * result.height;
    result.area_ratio = static_cast<double>(result.occupied_pixels) / pixels;
    result.conformal_error = static_cast<double>(result.overlap_pixels) / pixels;
    return result;
}

NaturalUvValidationResult validate_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    const NaturalUvValidationOptions &options) {
    NaturalUvValidationResult result;
    if (vertex_xyz.size() % 3U != 0U || vertex_uv.size() % 2U != 0U ||
        triangle_indices.size() % 3U != 0U ||
        vertex_xyz.size() / 3U != vertex_uv.size() / 2U ||
        triangle_indices.empty()) {
        return result;
    }
    for (const float value : vertex_uv) {
        if (!std::isfinite(value)) return result;
    }
    const std::size_t face_count = triangle_indices.size() / 3U;
    if (face_count <= options.small_chart_fast_accept_max_faces) {
        result.accepted = true;
        result.decision = NaturalUvValidationDecision::small_chart_fast_accept;
        return result;
    }
    result.stretch_evaluated = true;
    result.stretch_ratio = measure_natural_uv_stretch(
        vertex_xyz, triangle_indices, vertex_uv);
    if (result.stretch_ratio > options.max_stretch_ratio) {
        result.decision = NaturalUvValidationDecision::reject_stretch;
        return result;
    }
    result.raster_evaluated = true;
    result.raster = measure_natural_uv_raster_quality(
        triangle_indices, vertex_uv, options.raster_maximum_dimension);
    if (result.raster.area_ratio < options.min_area_ratio) {
        result.decision = NaturalUvValidationDecision::reject_area_ratio;
        return result;
    }
    if (result.raster.conformal_error > options.max_conformal_error) {
        result.decision = NaturalUvValidationDecision::reject_conformal_error;
        return result;
    }
    result.accepted = true;
    result.decision = NaturalUvValidationDecision::accept;
    return result;
}

}  // namespace metashape_texture
