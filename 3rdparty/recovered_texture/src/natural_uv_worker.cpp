#include "metashape_texture/natural_uv_worker.hpp"

#include "metashape_texture/natural_uv_partition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace metashape_texture {
namespace {

void validate_chart_shape(const NaturalUvChart &chart) {
    if (chart.vertex_xyz.size() % 3U != 0U ||
        chart.triangle_indices.size() % 3U != 0U ||
        chart.source_vertices.size() != chart.vertex_xyz.size() / 3U ||
        chart.source_faces.size() != chart.triangle_indices.size() / 3U ||
        chart.vertex_xyz.empty() || chart.triangle_indices.empty()) {
        throw std::invalid_argument("natural UV worker chart has incompatible arrays");
    }
    if (!chart.uv.empty() && chart.uv.size() != chart.vertex_xyz.size() / 3U * 2U) {
        throw std::invalid_argument("natural UV worker chart has incompatible UV values");
    }
    const std::size_t vertex_count = chart.vertex_xyz.size() / 3U;
    for (const std::uint32_t vertex : chart.triangle_indices) {
        if (vertex >= vertex_count) {
            throw std::out_of_range("natural UV worker chart vertex index");
        }
    }
}

float binary32_face_area(const NaturalUvChart &chart, std::size_t face) {
    const std::uint32_t first = chart.triangle_indices[3U * face];
    const std::uint32_t second = chart.triangle_indices[3U * face + 1U];
    const std::uint32_t third = chart.triangle_indices[3U * face + 2U];
    const float *a = chart.vertex_xyz.data() + 3U * first;
    const float *b = chart.vertex_xyz.data() + 3U * second;
    const float *c = chart.vertex_xyz.data() + 3U * third;
    const float ab_x = b[0] - a[0];
    const float ab_y = b[1] - a[1];
    const float ab_z = b[2] - a[2];
    const float ac_x = c[0] - a[0];
    const float ac_y = c[1] - a[1];
    const float ac_z = c[2] - a[2];
    const float cross_x = ab_y * ac_z - ab_z * ac_y;
    const float cross_y = ab_z * ac_x - ab_x * ac_z;
    const float cross_z = ab_x * ac_y - ab_y * ac_x;
    const float cross_xy_squared = cross_x * cross_x + cross_y * cross_y;
    const float cross_squared = cross_xy_squared + cross_z * cross_z;
    return std::sqrt(cross_squared);
}

double chart_surface_area(const NaturalUvChart &chart) {
    double area = 0.0;
    for (std::size_t face = 0; face < chart.source_faces.size(); ++face) {
        const std::uint32_t first = chart.triangle_indices[3U * face];
        const std::uint32_t second = chart.triangle_indices[3U * face + 1U];
        const std::uint32_t third = chart.triangle_indices[3U * face + 2U];
        const float *a = chart.vertex_xyz.data() + 3U * first;
        const float *b = chart.vertex_xyz.data() + 3U * second;
        const float *c = chart.vertex_xyz.data() + 3U * third;
        const float ab_x = b[0] - a[0];
        const float ab_y = b[1] - a[1];
        const float ab_z = b[2] - a[2];
        const float ac_x = c[0] - a[0];
        const float ac_y = c[1] - a[1];
        const float ac_z = c[2] - a[2];
        const float cross_x = ab_z * ac_y - ab_y * ac_z;
        const float cross_y = ab_x * ac_z - ab_z * ac_x;
        const float cross_z = ab_y * ac_x - ab_x * ac_y;
        const float cross_xy_squared = cross_x * cross_x + cross_y * cross_y;
        const float squared = cross_z * cross_z + cross_xy_squared;
        area += std::sqrt(squared) * 0.5F;
    }
    return area;
}

double chart_uv_double_area(const NaturalUvChart &chart) {
    double area = 0.0;
    for (std::size_t face = 0; face < chart.source_faces.size(); ++face) {
        const std::uint32_t first = chart.triangle_indices[3U * face];
        const std::uint32_t second = chart.triangle_indices[3U * face + 1U];
        const std::uint32_t third = chart.triangle_indices[3U * face + 2U];
        const float *a = chart.uv.data() + 2U * first;
        const float *b = chart.uv.data() + 2U * second;
        const float *c = chart.uv.data() + 2U * third;
        const float first_product = (b[0] - a[0]) * (c[1] - a[1]);
        const float second_product = (c[0] - a[0]) * (b[1] - a[1]);
        area += std::abs(first_product - second_product) * 0.5F;
    }
    return area;
}

using Point2 = std::array<float, 2>;

float point_cross(const Point2 &origin, const Point2 &first,
                  const Point2 &second) {
    const float first_product =
        (first[0] - origin[0]) * (second[1] - origin[1]);
    const float second_product =
        (first[1] - origin[1]) * (second[0] - origin[0]);
    return first_product - second_product;
}

std::vector<Point2> convex_hull(std::span<const float> uv) {
    const std::size_t count = uv.size() / 2U;
    std::vector<Point2> points(count);
    for (std::size_t index = 0; index < count; ++index) {
        points[index] = {uv[2U * index], uv[2U * index + 1U]};
    }
    if (count <= 3U) {
        if (count == 3U && point_cross(points[0], points[1], points[2]) > 0.0F) {
            std::swap(points[0], points[2]);
        }
        return points;
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() <= 3U) {
        if (points.size() == 3U &&
            point_cross(points[0], points[1], points[2]) > 0.0F) {
            std::swap(points[0], points[2]);
        }
        return points;
    }
    std::vector<Point2> hull;
    hull.reserve(points.size() * 2U);
    for (const Point2 &point : points) {
        while (hull.size() >= 2U &&
               point_cross(hull[hull.size() - 2U], hull.back(), point) <= 0.0F) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    const std::size_t lower_size = hull.size();
    for (std::size_t index = points.size() - 1U; index-- > 0U;) {
        const Point2 &point = points[index];
        while (hull.size() > lower_size &&
               point_cross(hull[hull.size() - 2U], hull.back(), point) <= 0.0F) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    if (!hull.empty()) hull.pop_back();
    if (hull.size() >= 3U && point_cross(hull[0], hull[1], hull[2]) > 0.0F) {
        std::reverse(hull.begin(), hull.end());
    }
    return hull;
}

float preferred_chart_angle(const NaturalUvChart &chart) {
    const std::vector<Point2> hull = convex_hull(chart.uv);
    if (hull.empty()) return 0.0F;
    const std::size_t count = hull.size();
    std::vector<Point2> edges(count);
    std::vector<float> inverse_lengths(count);
    std::size_t maximum_x = 0U;
    std::size_t maximum_y = 0U;
    std::size_t minimum_y = 0U;
    std::size_t minimum_x = 0U;
    float max_x = hull[0][0];
    float max_y = hull[0][1];
    float min_x = hull[0][0];
    float min_y = hull[0][1];
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t next = index + 1U == count ? 0U : index + 1U;
        if (hull[index][0] > max_x) {
            max_x = hull[index][0];
            maximum_x = index;
        }
        if (hull[index][1] > max_y) {
            max_y = hull[index][1];
            maximum_y = index;
        }
        if (min_y > hull[index][1]) {
            min_y = hull[index][1];
            minimum_y = index;
        }
        if (min_x > hull[next][0]) {
            min_x = hull[next][0];
            minimum_x = next;
        }
        edges[index] = {
            hull[next][0] - hull[index][0],
            hull[next][1] - hull[index][1]};
        const float squared = edges[index][1] * edges[index][1] +
                              edges[index][0] * edges[index][0];
        inverse_lengths[index] = 1.0F / std::sqrt(squared);
    }
    const float orientation_cross =
        edges.back()[0] * edges[0][1] - edges.back()[1] * edges[0][0];
    float axis_x = orientation_cross > 0.0F ? 1.0F : -1.0F;
    float axis_y = 0.0F;
    float best_area = std::numeric_limits<float>::max();
    float best_width = 0.0F;
    float best_height = 0.0F;
    float best_axis_x = 0.0F;
    float best_axis_y = 0.0F;
    for (std::size_t iteration = 0; iteration < count; ++iteration) {
        const Point2 &edge_min_y = edges[minimum_y];
        const Point2 &edge_max_x = edges[maximum_x];
        const Point2 &edge_max_y = edges[maximum_y];
        const Point2 &edge_min_x = edges[minimum_x];
        const float candidate_min_y =
            (edge_min_y[0] * axis_x + edge_min_y[1] * axis_y) *
            inverse_lengths[minimum_y];
        const float candidate_max_x =
            (edge_max_x[0] * -axis_y + edge_max_x[1] * axis_x) *
            inverse_lengths[maximum_x];
        const float candidate_max_y =
            (edge_max_y[1] * -axis_y + edge_max_y[0] * -axis_x) *
            inverse_lengths[maximum_y];
        const float candidate_min_x =
            (edge_min_x[1] * -axis_x + edge_min_x[0] * axis_y) *
            inverse_lengths[minimum_x];
        std::size_t selected = 0U;
        float selected_value = candidate_min_y;
        if (candidate_max_x > selected_value) {
            selected_value = candidate_max_x;
            selected = 1U;
        }
        if (candidate_max_y > selected_value) {
            selected_value = candidate_max_y;
            selected = 2U;
        }
        if (candidate_min_x > selected_value) selected = 3U;

        float perpendicular_x = 0.0F;
        if (selected == 0U) {
            const float inverse = inverse_lengths[minimum_y];
            axis_y = edges[minimum_y][1] * inverse;
            axis_x = edges[minimum_y][0] * inverse;
            perpendicular_x = -axis_y;
            minimum_y = minimum_y + 1U == count ? 0U : minimum_y + 1U;
        } else if (selected == 1U) {
            const float inverse = inverse_lengths[maximum_x];
            axis_x = edges[maximum_x][1] * inverse;
            axis_y = -(edges[maximum_x][0] * inverse);
            perpendicular_x = -axis_y;
            maximum_x = maximum_x + 1U == count ? 0U : maximum_x + 1U;
        } else if (selected == 2U) {
            const float inverse = inverse_lengths[maximum_y];
            perpendicular_x = edges[maximum_y][1] * inverse;
            axis_y = -perpendicular_x;
            axis_x = -(edges[maximum_y][0] * inverse);
            maximum_y = maximum_y + 1U == count ? 0U : maximum_y + 1U;
        } else {
            const float inverse = inverse_lengths[minimum_x];
            axis_y = edges[minimum_x][0] * inverse;
            axis_x = -(edges[minimum_x][1] * inverse);
            perpendicular_x = -axis_y;
            minimum_x = minimum_x + 1U == count ? 0U : minimum_x + 1U;
        }
        const float width =
            (hull[maximum_y][0] - hull[minimum_y][0]) * perpendicular_x +
            (hull[maximum_y][1] - hull[minimum_y][1]) * axis_x;
        const float height =
            (hull[maximum_x][1] - hull[minimum_x][1]) * axis_y +
            (hull[maximum_x][0] - hull[minimum_x][0]) * axis_x;
        const float area = width * height;
        if (best_area >= area) {
            best_area = area;
            best_width = width;
            best_height = height;
            best_axis_x = axis_x;
            best_axis_y = axis_y;
        }
    }
    (void)best_width;
    const float direction_x = best_axis_x * best_height;
    const float direction_y = best_axis_y * best_height;
    const float direction_squared =
        direction_x * direction_x + direction_y * direction_y;
    const float direction_length = std::sqrt(direction_squared);
    const float inverse_direction =
        direction_length >= 1.0e-20F ? 1.0F / direction_length : 0.0F;
    const Point2 selected_direction{
        direction_x * inverse_direction, direction_y * inverse_direction};
    float angle = static_cast<float>(
        std::atan2(static_cast<double>(selected_direction[1]),
                   static_cast<double>(selected_direction[0])));
    constexpr float half_pi = 1.5707964F;
    while (angle > 0.7853981633974483) angle -= half_pi;
    while (angle < -0.7853981633974483) angle += half_pi;

    const float sine = static_cast<float>(std::sin(static_cast<double>(angle)));
    const float cosine = static_cast<float>(std::cos(static_cast<double>(angle)));
    float rotated_min_x = std::numeric_limits<float>::max();
    float rotated_max_x = -std::numeric_limits<float>::max();
    float rotated_min_y = std::numeric_limits<float>::max();
    float rotated_max_y = -std::numeric_limits<float>::max();
    for (std::size_t vertex = 0; vertex < chart.uv.size() / 2U; ++vertex) {
        const float x = cosine * chart.uv[2U * vertex] +
                        sine * chart.uv[2U * vertex + 1U];
        const float y = -sine * chart.uv[2U * vertex] +
                        cosine * chart.uv[2U * vertex + 1U];
        rotated_min_x = std::min(rotated_min_x, x);
        rotated_max_x = std::max(rotated_max_x, x);
        rotated_min_y = std::min(rotated_min_y, y);
        rotated_max_y = std::max(rotated_max_y, y);
    }
    if (rotated_max_x - rotated_min_x > rotated_max_y - rotated_min_y) {
        angle += half_pi;
    }
    return angle;
}

void reorder_faces_by_area(NaturalUvChart &chart) {
    const std::size_t face_count = chart.source_faces.size();
    std::vector<std::uint32_t> order(face_count);
    std::iota(order.begin(), order.end(), 0U);
    std::vector<float> areas(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        areas[face] = binary32_face_area(chart, face);
    }
    std::stable_sort(order.begin(), order.end(), [&areas](std::uint32_t first,
                                                          std::uint32_t second) {
        return areas[first] > areas[second];
    });
    std::vector<std::uint32_t> reordered_faces;
    std::vector<std::uint32_t> reordered_indices;
    reordered_faces.reserve(face_count);
    reordered_indices.reserve(face_count * 3U);
    for (const std::uint32_t face : order) {
        reordered_faces.push_back(chart.source_faces[face]);
        reordered_indices.insert(
            reordered_indices.end(), chart.triangle_indices.begin() + 3U * face,
            chart.triangle_indices.begin() + 3U * face + 3U);
    }
    chart.source_faces = std::move(reordered_faces);
    chart.triangle_indices = std::move(reordered_indices);
}

void split_disconnected_vertex_fans(NaturalUvChart &chart) {
    const std::uint32_t original_vertex_count =
        static_cast<std::uint32_t>(chart.source_vertices.size());
    const std::uint32_t face_count =
        static_cast<std::uint32_t>(chart.source_faces.size());
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Edge, std::vector<std::uint32_t>> edge_faces;
    std::vector<std::vector<std::uint32_t>> incident_faces(original_vertex_count);
    for (std::uint32_t face = 0; face < face_count; ++face) {
        const std::uint32_t *triangle = chart.triangle_indices.data() + 3U * face;
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            incident_faces[triangle[corner]].push_back(face);
            const std::uint32_t next = (corner + 1U) % 3U;
            edge_faces[std::minmax(triangle[corner], triangle[next])].push_back(face);
        }
    }

    std::vector<std::vector<std::uint32_t>> fan_neighbors(face_count);
    for (const auto &[edge, faces] : edge_faces) {
        (void)edge;
        for (std::size_t first = 0; first < faces.size(); ++first) {
            for (std::size_t second = first + 1U; second < faces.size(); ++second) {
                fan_neighbors[faces[first]].push_back(faces[second]);
                fan_neighbors[faces[second]].push_back(faces[first]);
            }
        }
    }

    std::vector<std::uint32_t> visit_generation(face_count, 0U);
    std::uint32_t generation = 0U;
    for (std::uint32_t vertex = 0; vertex < original_vertex_count; ++vertex) {
        std::vector<std::vector<std::uint32_t>> components;
        for (const std::uint32_t seed : incident_faces[vertex]) {
            if (visit_generation[seed] == generation + 1U) continue;
            std::vector<std::uint32_t> component;
            std::deque<std::uint32_t> frontier{seed};
            visit_generation[seed] = generation + 1U;
            while (!frontier.empty()) {
                const std::uint32_t face = frontier.front();
                frontier.pop_front();
                component.push_back(face);
                for (const std::uint32_t neighbor : fan_neighbors[face]) {
                    const std::uint32_t *triangle =
                        chart.triangle_indices.data() + 3U * neighbor;
                    if (triangle[0] != vertex && triangle[1] != vertex &&
                        triangle[2] != vertex) {
                        continue;
                    }
                    if (visit_generation[neighbor] == generation + 1U) continue;
                    visit_generation[neighbor] = generation + 1U;
                    frontier.push_back(neighbor);
                }
            }
            std::sort(component.begin(), component.end());
            components.push_back(std::move(component));
        }
        ++generation;
        std::sort(components.begin(), components.end(), [](const auto &first,
                                                           const auto &second) {
            return first.front() < second.front();
        });
        for (std::size_t component_index = 1U;
             component_index < components.size(); ++component_index) {
            const std::uint32_t duplicate =
                static_cast<std::uint32_t>(chart.source_vertices.size());
            chart.source_vertices.push_back(chart.source_vertices[vertex]);
            const std::array<float, 3> duplicate_xyz{
                chart.vertex_xyz[3U * vertex], chart.vertex_xyz[3U * vertex + 1U],
                chart.vertex_xyz[3U * vertex + 2U]};
            chart.vertex_xyz.insert(chart.vertex_xyz.end(), duplicate_xyz.begin(),
                                    duplicate_xyz.end());
            if (!chart.uv.empty()) {
                const std::array<float, 2> duplicate_uv{
                    chart.uv[2U * vertex], chart.uv[2U * vertex + 1U]};
                chart.uv.insert(chart.uv.end(), duplicate_uv.begin(), duplicate_uv.end());
            }
            for (const std::uint32_t face : components[component_index]) {
                for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                    std::uint32_t &current = chart.triangle_indices[3U * face + corner];
                    if (current == vertex) current = duplicate;
                }
            }
        }
    }
}

NaturalUvChart build_child_chart(const NaturalUvChart &parent,
                                 const std::vector<std::uint32_t> &selected_faces) {
    const std::size_t parent_vertex_count = parent.vertex_xyz.size() / 3U;
    std::vector<bool> used(parent_vertex_count, false);
    for (const std::uint32_t face : selected_faces) {
        if (face >= parent.source_faces.size()) {
            throw std::out_of_range("natural UV child face index");
        }
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            used[parent.triangle_indices[3U * face + corner]] = true;
        }
    }

    NaturalUvChart child;
    std::vector<std::uint32_t> remap(parent_vertex_count, 0U);
    for (std::uint32_t parent_vertex = 0; parent_vertex < parent_vertex_count;
         ++parent_vertex) {
        if (!used[parent_vertex]) continue;
        remap[parent_vertex] = static_cast<std::uint32_t>(child.source_vertices.size());
        child.source_vertices.push_back(parent.source_vertices[parent_vertex]);
        child.vertex_xyz.insert(
            child.vertex_xyz.end(),
            parent.vertex_xyz.begin() + static_cast<std::ptrdiff_t>(3U * parent_vertex),
            parent.vertex_xyz.begin() + static_cast<std::ptrdiff_t>(3U * parent_vertex + 3U));
    }
    child.source_faces.reserve(selected_faces.size());
    child.triangle_indices.reserve(selected_faces.size() * 3U);
    for (const std::uint32_t face : selected_faces) {
        child.source_faces.push_back(parent.source_faces[face]);
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            child.triangle_indices.push_back(
                remap[parent.triangle_indices[3U * face + corner]]);
        }
    }
    validate_chart_shape(child);
    return child;
}

bool has_boundary_edge(const NaturalUvChart &chart) {
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Edge, std::uint32_t> uses;
    for (std::size_t face = 0; face < chart.source_faces.size(); ++face) {
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t first = chart.triangle_indices[3U * face + corner];
            const std::uint32_t second =
                chart.triangle_indices[3U * face + (corner + 1U) % 3U];
            ++uses[std::minmax(first, second)];
        }
    }
    return std::ranges::any_of(uses, [](const auto &entry) {
        return entry.second == 1U;
    });
}

float chart_uv_maximum_span(const NaturalUvChart &chart) {
    float minimum_x = std::numeric_limits<float>::max();
    float maximum_x = -std::numeric_limits<float>::max();
    float minimum_y = std::numeric_limits<float>::max();
    float maximum_y = -std::numeric_limits<float>::max();
    for (std::size_t vertex = 0; vertex < chart.uv.size() / 2U; ++vertex) {
        minimum_x = std::min(minimum_x, chart.uv[2U * vertex]);
        maximum_x = std::max(maximum_x, chart.uv[2U * vertex]);
        minimum_y = std::min(minimum_y, chart.uv[2U * vertex + 1U]);
        maximum_y = std::max(maximum_y, chart.uv[2U * vertex + 1U]);
    }
    return std::max(maximum_x - minimum_x, maximum_y - minimum_y);
}

NaturalUvChart build_connected_child_chart(
    const NaturalUvChart &parent,
    const std::vector<std::uint32_t> &selected_faces) {
    const std::size_t parent_vertex_count = parent.vertex_xyz.size() / 3U;
    NaturalUvChart child;
    std::vector<std::uint32_t> remap(
        parent_vertex_count, std::numeric_limits<std::uint32_t>::max());
    child.source_faces.reserve(selected_faces.size());
    child.triangle_indices.reserve(selected_faces.size() * 3U);
    for (const std::uint32_t face : selected_faces) {
        child.source_faces.push_back(parent.source_faces[face]);
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t parent_vertex =
                parent.triangle_indices[3U * face + corner];
            std::uint32_t &child_vertex = remap[parent_vertex];
            if (child_vertex == std::numeric_limits<std::uint32_t>::max()) {
                child_vertex = static_cast<std::uint32_t>(child.source_vertices.size());
                child.source_vertices.push_back(parent.source_vertices[parent_vertex]);
                child.vertex_xyz.insert(
                    child.vertex_xyz.end(),
                    parent.vertex_xyz.begin() +
                        static_cast<std::ptrdiff_t>(3U * parent_vertex),
                    parent.vertex_xyz.begin() +
                        static_cast<std::ptrdiff_t>(3U * parent_vertex + 3U));
                if (!parent.uv.empty()) {
                    child.uv.insert(
                        child.uv.end(),
                        parent.uv.begin() +
                            static_cast<std::ptrdiff_t>(2U * parent_vertex),
                        parent.uv.begin() +
                            static_cast<std::ptrdiff_t>(2U * parent_vertex + 2U));
                }
            }
            child.triangle_indices.push_back(child_vertex);
        }
    }
    validate_chart_shape(child);
    return child;
}

std::vector<NaturalUvChart> extract_edge_connected_charts(
    const NaturalUvChart &parent) {
    using Edge = std::pair<std::uint32_t, std::uint32_t>;
    const std::uint32_t face_count =
        static_cast<std::uint32_t>(parent.source_faces.size());
    std::map<Edge, std::vector<std::uint32_t>> edge_faces;
    for (std::uint32_t face = 0; face < face_count; ++face) {
        for (std::uint32_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t first =
                parent.triangle_indices[3U * face + corner];
            const std::uint32_t second =
                parent.triangle_indices[3U * face + (corner + 1U) % 3U];
            edge_faces[std::minmax(first, second)].push_back(face);
        }
    }
    std::vector<std::vector<std::uint32_t>> neighbors(face_count);
    for (const auto &[edge, uses] : edge_faces) {
        (void)edge;
        if (uses.size() != 2U) continue;
        neighbors[uses[0]].push_back(uses[1]);
        neighbors[uses[1]].push_back(uses[0]);
    }

    std::vector<bool> visited(face_count, false);
    std::vector<NaturalUvChart> result;
    for (std::uint32_t seed = 0; seed < face_count; ++seed) {
        if (visited[seed]) continue;
        std::deque<std::uint32_t> frontier{seed};
        visited[seed] = true;
        std::vector<std::uint32_t> component;
        while (!frontier.empty()) {
            const std::uint32_t face = frontier.front();
            frontier.pop_front();
            component.push_back(face);
            for (const std::uint32_t neighbor : neighbors[face]) {
                if (visited[neighbor]) continue;
                visited[neighbor] = true;
                frontier.push_back(neighbor);
            }
        }
        std::sort(component.begin(), component.end());
        result.push_back(build_connected_child_chart(parent, component));
    }
    return result;
}

}  // namespace

NaturalUvChart normalize_natural_uv_root_chart(NaturalUvChart chart) {
    validate_chart_shape(chart);
    reorder_faces_by_area(chart);
    split_disconnected_vertex_fans(chart);
    validate_chart_shape(chart);
    return chart;
}

void transform_accepted_natural_uv_chart(NaturalUvChart &chart,
                                         double scale_argument) {
    validate_chart_shape(chart);
    if (chart.uv.empty()) {
        throw std::invalid_argument("accepted natural UV chart has no UV values");
    }
    const float angle = preferred_chart_angle(chart);
    const double surface_area = chart_surface_area(chart);
    const double uv_area = chart_uv_double_area(chart);
    float sine = static_cast<float>(std::sin(static_cast<double>(angle)));
    float cosine = static_cast<float>(std::cos(static_cast<double>(angle)));
    if (std::abs(uv_area) > 1.0e-10) {
        const float scale = static_cast<float>(
            std::sqrt(surface_area / uv_area) * scale_argument);
        sine *= scale;
        cosine *= scale;
    }
    const float origin_x = chart.uv[0];
    const float origin_y = chart.uv[1];
    for (std::size_t vertex = 0; vertex < chart.uv.size() / 2U; ++vertex) {
        const float x = chart.uv[2U * vertex] - origin_x;
        const float y = chart.uv[2U * vertex + 1U] - origin_y;
        chart.uv[2U * vertex] = cosine * x + sine * y;
        chart.uv[2U * vertex + 1U] = (-sine) * x + cosine * y;
    }
}

std::vector<NaturalUvChart> rebuild_natural_uv_charts_for_packing(
    std::span<const std::uint32_t> source_face_order,
    std::span<const NaturalUvChart> accepted_charts) {
    if (source_face_order.empty() || accepted_charts.empty()) {
        throw std::invalid_argument("empty natural UV chart rebuild input");
    }

    NaturalUvChart flattened;
    std::map<std::uint32_t, std::array<std::uint32_t, 3>> indices_by_source_face;
    for (const NaturalUvChart &accepted : accepted_charts) {
        validate_chart_shape(accepted);
        if (accepted.uv.empty()) {
            throw std::invalid_argument("natural UV accepted chart has no UV values");
        }
        const std::uint32_t vertex_offset =
            static_cast<std::uint32_t>(flattened.source_vertices.size());
        for (std::size_t vertex = 0; vertex < accepted.source_vertices.size(); ++vertex) {
            const float u = accepted.uv[2U * vertex];
            const float v = accepted.uv[2U * vertex + 1U];
            flattened.source_vertices.push_back(vertex_offset +
                                                  static_cast<std::uint32_t>(vertex));
            flattened.vertex_xyz.insert(flattened.vertex_xyz.end(), {u, v, 0.0F});
            flattened.uv.insert(flattened.uv.end(), {u, v});
        }
        for (std::size_t face = 0; face < accepted.source_faces.size(); ++face) {
            const std::uint32_t source_face = accepted.source_faces[face];
            const std::array<std::uint32_t, 3> indices{
                vertex_offset + accepted.triangle_indices[3U * face],
                vertex_offset + accepted.triangle_indices[3U * face + 1U],
                vertex_offset + accepted.triangle_indices[3U * face + 2U]};
            if (!indices_by_source_face.emplace(source_face, indices).second) {
                throw std::invalid_argument("duplicate natural UV source face");
            }
        }
    }
    flattened.source_faces.assign(source_face_order.begin(), source_face_order.end());
    flattened.triangle_indices.reserve(source_face_order.size() * 3U);
    for (const std::uint32_t source_face : source_face_order) {
        const auto iterator = indices_by_source_face.find(source_face);
        if (iterator == indices_by_source_face.end()) {
            throw std::invalid_argument("missing natural UV source face");
        }
        flattened.triangle_indices.insert(flattened.triangle_indices.end(),
                                          iterator->second.begin(),
                                          iterator->second.end());
    }
    if (indices_by_source_face.size() != source_face_order.size()) {
        throw std::invalid_argument("natural UV source face count mismatch");
    }
    flattened = normalize_natural_uv_root_chart(std::move(flattened));
    return extract_edge_connected_charts(flattened);
}

void scale_natural_uv_chart_for_packing(NaturalUvChart &chart,
                                        std::uint32_t texture_size) {
    validate_chart_shape(chart);
    if (chart.uv.empty() || texture_size <= 5U) {
        throw std::invalid_argument("invalid natural UV packing scale input");
    }
    const float area = static_cast<float>(chart_uv_double_area(chart));
    float scale = 0.0F;
    if (area > 0.0F) {
        const float face_count = static_cast<float>(chart.source_faces.size());
        const double density = std::sqrt(
            static_cast<double>(face_count + face_count) / static_cast<double>(area));
        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        for (std::size_t vertex = 0; vertex < chart.uv.size() / 2U; ++vertex) {
            min_x = std::min(min_x, chart.uv[2U * vertex]);
            max_x = std::max(max_x, chart.uv[2U * vertex]);
            min_y = std::min(min_y, chart.uv[2U * vertex + 1U]);
            max_y = std::max(max_y, chart.uv[2U * vertex + 1U]);
        }
        const double maximum_span = static_cast<double>(
            std::max(max_x - min_x, max_y - min_y));
        const double cap = maximum_span > 0.0
            ? static_cast<double>(texture_size - 5U) / maximum_span - 1.0e-9
            : density;
        scale = static_cast<float>(std::min(density, cap));
    }
    for (float &coordinate : chart.vertex_xyz) coordinate *= scale;
    for (float &coordinate : chart.uv) coordinate *= scale;
}

NaturalUvWorkerResult generate_natural_uv_charts(
    NaturalUvChart initial_chart,
    const NaturalUvWorkerOptions &options) {
    initial_chart = normalize_natural_uv_root_chart(std::move(initial_chart));
    if (options.partition_max_clusters == 0U ||
        options.partition_refinement_iterations == 0U ||
        options.partition_postprocess_iterations == 0U ||
        options.max_processed_charts == 0U) {
        throw std::invalid_argument("invalid natural UV worker options");
    }

    const double root_area = chart_surface_area(initial_chart);
    if (!(root_area > 0.0)) {
        throw std::runtime_error("natural UV root chart has no surface area");
    }
    const double accepted_scale = std::sqrt(1.0 / (2.0 * root_area));
    NaturalUvWorkerResult result;
    std::deque<NaturalUvChart> frontier;
    // Native ChartQueue::add normalizes the incoming root, immediately
    // extracts its edge-connected pieces, and enqueues those pieces in
    // extraction order.  Treating a vertex-only-connected root as one chart
    // changes its vertex-zero anchor and leaves a translation/rotation gauge
    // difference even when every downstream numerical kernel is exact.
    for (NaturalUvChart &connected : extract_edge_connected_charts(initial_chart)) {
        frontier.push_back(std::move(connected));
    }
    while (!frontier.empty()) {
        if (result.history.size() >= options.max_processed_charts) {
            throw std::runtime_error("natural UV worker chart limit exceeded");
        }
        NaturalUvChart chart = std::move(frontier.front());
        frontier.pop_front();
        if (!has_boundary_edge(chart)) {
            if (chart.source_faces.size() <= 1U) {
                throw std::runtime_error("natural UV chart has no usable boundary loop");
            }
            std::vector<std::uint32_t> ordered_faces(chart.source_faces.size());
            std::iota(ordered_faces.begin(), ordered_faces.end(), 0U);
            const NaturalUvNormalPartition partition =
                partition_natural_uv_chart_by_normals(
                    chart.vertex_xyz, chart.triangle_indices, ordered_faces,
                    options.partition_max_clusters,
                    options.partition_refinement_iterations,
                    options.partition_target_ratio);
            const auto maximum_label =
                std::ranges::max_element(partition.assignments);
            if (maximum_label == partition.assignments.end() ||
                *maximum_label < 1) {
                throw std::runtime_error(
                    "closed natural UV chart did not split by normals");
            }
            const std::uint32_t cluster_count =
                static_cast<std::uint32_t>(*maximum_label + 1);
            std::vector<std::vector<std::uint32_t>> cluster_faces(cluster_count);
            for (std::uint32_t face = 0; face < partition.assignments.size(); ++face) {
                const std::int32_t cluster = partition.assignments[face];
                if (cluster < 0 ||
                    static_cast<std::uint32_t>(cluster) >= cluster_count) {
                    throw std::runtime_error(
                        "closed natural UV partition contains an invalid label");
                }
                cluster_faces[static_cast<std::size_t>(cluster)].push_back(face);
            }
            NaturalUvWorkerStep step;
            step.source_faces = chart.source_faces;
            for (const auto &faces : cluster_faces) {
                if (faces.empty()) {
                    throw std::runtime_error(
                        "closed natural UV partition contains an empty chart");
                }
                NaturalUvChart child = build_child_chart(chart, faces);
                step.child_source_faces.push_back(child.source_faces);
                frontier.push_back(std::move(child));
            }
            result.history.push_back(std::move(step));
            continue;
        }
        const NaturalUvParameterization parameterized =
            parameterize_natural_uv_chart(chart.vertex_xyz, chart.triangle_indices);
        chart.uv = parameterized.uv;
        const NaturalUvValidationResult validation = validate_natural_uv_chart(
            chart.vertex_xyz, chart.triangle_indices, chart.uv, options.validation);

        NaturalUvWorkerStep step;
        step.source_faces = chart.source_faces;
        step.anchor_first = parameterized.anchor_first;
        step.anchor_second = parameterized.anchor_second;
        step.uv = chart.uv;
        step.validation = validation;

        if (validation.accepted) {
            transform_accepted_natural_uv_chart(chart, accepted_scale);
            const double maximum_span =
                static_cast<double>(chart_uv_maximum_span(chart));
            if (maximum_span <= 0.8 || chart.source_faces.size() <= 1U) {
                result.charts.push_back(std::move(chart));
                result.history.push_back(std::move(step));
                continue;
            }
        }
        if (validation.decision == NaturalUvValidationDecision::invalid_input) {
            throw std::runtime_error("natural UV parameterization produced invalid UV values");
        }
        std::vector<std::uint32_t> ordered_faces(chart.source_faces.size());
        std::iota(ordered_faces.begin(), ordered_faces.end(), 0U);
        const NaturalUvNormalPartition partition = partition_natural_uv_chart_by_normals(
            chart.vertex_xyz, chart.triangle_indices, ordered_faces,
            options.partition_max_clusters,
            options.partition_refinement_iterations,
            options.partition_target_ratio);
        std::vector<std::int32_t> assignments = postprocess_natural_uv_partition(
            chart.triangle_indices, chart.uv, partition.assignments,
            options.partition_postprocess_iterations).assignments;
        const auto maximum_label = std::ranges::max_element(assignments);
        if (maximum_label == assignments.end() || *maximum_label < 1) {
            std::ostringstream message;
            message << "natural UV rejected chart did not split (faces="
                    << chart.source_faces.size() << ')';
            throw std::runtime_error(message.str());
        }
        const std::uint32_t cluster_count = static_cast<std::uint32_t>(*maximum_label + 1);
        std::vector<std::vector<std::uint32_t>> cluster_faces(cluster_count);
        for (std::uint32_t face = 0; face < assignments.size(); ++face) {
            const std::int32_t cluster = assignments[face];
            if (cluster < 0 || static_cast<std::uint32_t>(cluster) >= cluster_count) {
                throw std::runtime_error("natural UV partition contains an invalid label");
            }
            cluster_faces[static_cast<std::size_t>(cluster)].push_back(face);
        }
        for (const auto &faces : cluster_faces) {
            if (faces.empty()) throw std::runtime_error("natural UV partition contains an empty chart");
            NaturalUvChart child = build_child_chart(chart, faces);
            step.child_source_faces.push_back(child.source_faces);
            frontier.push_back(std::move(child));
        }
        result.history.push_back(std::move(step));
    }
    return result;
}

}  // namespace metashape_texture
