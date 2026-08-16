#include "ModelMeshRenderer.h"

#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::qc
{
namespace
{

constexpr int kTileSize = 32;
// Keep clipped vertices safely beyond FramePinholeCamera's projection singularity.
constexpr double kNearPlaneDepth = 1.0e-6;

struct ClippedVertex
{
    std::array<double, 3> world = {};
    double positiveDepth = 0.0;
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
};

struct ProjectedVertex
{
    double x = 0.0;
    double y = 0.0;
    double depth = 0.0;
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
};

struct ProjectedTriangle
{
    std::array<ProjectedVertex, 3> vertices;
    double area = 0.0;
    int minimumX = 0;
    int maximumX = -1;
    int minimumY = 0;
    int maximumY = -1;
};

double edge(double ax, double ay, double bx, double by, double px, double py)
{
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

bool sameWorldPosition(const ClippedVertex &left, const ClippedVertex &right)
{
    for (std::size_t axis = 0; axis < left.world.size(); ++axis)
    {
        const double scale = std::max({1.0,
                                       std::abs(left.world[axis]),
                                       std::abs(right.world[axis])});
        if (std::abs(left.world[axis] - right.world[axis]) > 1.0e-12 * scale)
        {
            return false;
        }
    }
    return true;
}

bool appendClippedVertex(const ClippedVertex &vertex,
                         std::array<ClippedVertex, 4> *polygon,
                         int *vertex_count)
{
    if (!polygon || !vertex_count)
    {
        return false;
    }
    if (*vertex_count > 0 &&
        sameWorldPosition((*polygon)[static_cast<std::size_t>(*vertex_count - 1)], vertex))
    {
        return true;
    }
    if (*vertex_count >= static_cast<int>(polygon->size()))
    {
        return false;
    }
    (*polygon)[static_cast<std::size_t>(*vertex_count)] = vertex;
    ++(*vertex_count);
    return true;
}

bool intersectNearPlane(const ClippedVertex &start,
                        const ClippedVertex &end,
                        ClippedVertex *intersection)
{
    if (!intersection)
    {
        return false;
    }
    const double depth_delta = end.positiveDepth - start.positiveDepth;
    if (!std::isfinite(depth_delta) || std::abs(depth_delta) <= 1.0e-15)
    {
        return false;
    }

    const double interpolation = std::clamp(
        (kNearPlaneDepth - start.positiveDepth) / depth_delta, 0.0, 1.0);
    for (std::size_t axis = 0; axis < intersection->world.size(); ++axis)
    {
        intersection->world[axis] = start.world[axis] +
            interpolation * (end.world[axis] - start.world[axis]);
    }
    intersection->positiveDepth = kNearPlaneDepth;
    intersection->red = start.red + interpolation * (end.red - start.red);
    intersection->green = start.green + interpolation * (end.green - start.green);
    intersection->blue = start.blue + interpolation * (end.blue - start.blue);
    return std::all_of(intersection->world.begin(), intersection->world.end(), [](double value) {
        return std::isfinite(value);
    });
}

int clipTriangleToPositiveDepth(const xjw::mesh::TriMesh &mesh,
                                const xjw::mesh::Triangle &face,
                                const xjw::FramePinholeCamera &camera,
                                std::array<ClippedVertex, 4> *polygon)
{
    if (!polygon)
    {
        return 0;
    }

    std::array<ClippedVertex, 3> source_vertices;
    for (int corner = 0; corner < 3; ++corner)
    {
        const xjw::mesh::MeshVertex &source =
            mesh.vertices[static_cast<std::size_t>(face.v[corner])];
        ClippedVertex &target = source_vertices[static_cast<std::size_t>(corner)];
        target.world = {source.x, source.y, source.z};
        target.positiveDepth = camera.positiveDepth(target.world.data());
        if (!std::isfinite(target.positiveDepth) ||
            !std::all_of(target.world.begin(), target.world.end(), [](double value) {
                return std::isfinite(value);
            }))
        {
            return 0;
        }
        target.red = static_cast<double>(source.r);
        target.green = static_cast<double>(source.g);
        target.blue = static_cast<double>(source.b);
    }

    int clipped_vertex_count = 0;
    ClippedVertex previous = source_vertices.back();
    bool previous_inside = previous.positiveDepth >= kNearPlaneDepth;
    for (const ClippedVertex &current : source_vertices)
    {
        const bool current_inside = current.positiveDepth >= kNearPlaneDepth;
        if (current_inside != previous_inside)
        {
            ClippedVertex intersection;
            if (!intersectNearPlane(previous, current, &intersection) ||
                !appendClippedVertex(intersection, polygon, &clipped_vertex_count))
            {
                return 0;
            }
        }
        if (current_inside &&
            !appendClippedVertex(current, polygon, &clipped_vertex_count))
        {
            return 0;
        }
        previous = current;
        previous_inside = current_inside;
    }

    if (clipped_vertex_count > 1 &&
        sameWorldPosition((*polygon)[0],
                          (*polygon)[static_cast<std::size_t>(clipped_vertex_count - 1)]))
    {
        --clipped_vertex_count;
    }
    return clipped_vertex_count;
}

bool projectTriangle(const std::array<ClippedVertex, 3> &source_vertices,
                     const xjw::FramePinholeCamera &camera,
                     const cv::Size &image_size,
                     ProjectedTriangle *projected)
{
    if (!projected)
    {
        return false;
    }

    for (int corner = 0; corner < 3; ++corner)
    {
        const ClippedVertex &source = source_vertices[static_cast<std::size_t>(corner)];
        ProjectedVertex &target = projected->vertices[static_cast<std::size_t>(corner)];
        double pixel[2] = {};
        double positiveDepth = 0.0;
        if (!camera.projectWorldPointWithDepth(source.world.data(), pixel, positiveDepth) ||
            !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
            !std::isfinite(positiveDepth) || positiveDepth <= 0.0)
        {
            return false;
        }
        target.x = pixel[0];
        target.y = pixel[1];
        target.depth = positiveDepth;
        if (!std::isfinite(target.x) || !std::isfinite(target.y) ||
            !std::isfinite(target.depth) || target.depth <= 0.0)
        {
            return false;
        }
        target.red = source.red;
        target.green = source.green;
        target.blue = source.blue;
    }

    projected->area = edge(projected->vertices[0].x, projected->vertices[0].y,
                           projected->vertices[1].x, projected->vertices[1].y,
                           projected->vertices[2].x, projected->vertices[2].y);
    if (!std::isfinite(projected->area) || std::abs(projected->area) < 1.0e-6)
    {
        return false;
    }

    const double minimum_x = std::min({projected->vertices[0].x,
                                       projected->vertices[1].x,
                                       projected->vertices[2].x});
    const double maximum_x = std::max({projected->vertices[0].x,
                                       projected->vertices[1].x,
                                       projected->vertices[2].x});
    const double minimum_y = std::min({projected->vertices[0].y,
                                       projected->vertices[1].y,
                                       projected->vertices[2].y});
    const double maximum_y = std::max({projected->vertices[0].y,
                                       projected->vertices[1].y,
                                       projected->vertices[2].y});
    if (!std::isfinite(minimum_x) || !std::isfinite(maximum_x) ||
        !std::isfinite(minimum_y) || !std::isfinite(maximum_y))
    {
        return false;
    }

    const double minimum_sample_x = 0.5;
    const double maximum_sample_x = static_cast<double>(image_size.width) - 0.5;
    const double minimum_sample_y = 0.5;
    const double maximum_sample_y = static_cast<double>(image_size.height) - 0.5;
    if (maximum_x < minimum_sample_x || minimum_x > maximum_sample_x ||
        maximum_y < minimum_sample_y || minimum_y > maximum_sample_y)
    {
        return false;
    }

    const double maximum_index_x = static_cast<double>(image_size.width - 1);
    const double maximum_index_y = static_cast<double>(image_size.height - 1);
    const double bounded_minimum_x = std::clamp(minimum_x, 0.0, maximum_index_x);
    const double bounded_maximum_x = std::clamp(maximum_x, 0.0, maximum_index_x);
    const double bounded_minimum_y = std::clamp(minimum_y, 0.0, maximum_index_y);
    const double bounded_maximum_y = std::clamp(maximum_y, 0.0, maximum_index_y);
    projected->minimumX = static_cast<int>(std::floor(bounded_minimum_x));
    projected->maximumX = static_cast<int>(std::ceil(bounded_maximum_x));
    projected->minimumY = static_cast<int>(std::floor(bounded_minimum_y));
    projected->maximumY = static_cast<int>(std::ceil(bounded_maximum_y));
    return projected->minimumX <= projected->maximumX &&
           projected->minimumY <= projected->maximumY;
}

std::uint8_t colorByte(double value)
{
    return static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0) + 0.5);
}

} // namespace

ModelRenderResult ModelMeshRenderer::render(
    const xjw::mesh::TriMesh &mesh,
    const xjw::FramePinholeCamera &camera,
    const cv::Size &imageSize) const
{
    ModelRenderResult result;
    const auto start = std::chrono::steady_clock::now();
    if (mesh.empty())
    {
        result.error = QStringLiteral("模型没有可渲染三角面");
        return result;
    }
    if (!camera.isValid())
    {
        result.error = QStringLiteral("相机模型无效");
        return result;
    }
    if (imageSize.width <= 0 || imageSize.height <= 0)
    {
        result.error = QStringLiteral("渲染尺寸无效");
        return result;
    }

    result.color = cv::Mat::zeros(imageSize, CV_8UC3);
    result.validMask = cv::Mat::zeros(imageSize, CV_8UC1);
    result.depth = cv::Mat(imageSize, CV_32FC1,
                           cv::Scalar(std::numeric_limits<float>::infinity()));

    std::vector<ProjectedTriangle> triangles;
    triangles.reserve(mesh.faces.size() * 2);
    int visible_triangle_count = 0;
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        const std::size_t projected_triangle_count_before_face = triangles.size();
        std::array<ClippedVertex, 4> clipped_polygon;
        const int clipped_vertex_count =
            clipTriangleToPositiveDepth(mesh, face, camera, &clipped_polygon);
        for (int corner = 1; corner + 1 < clipped_vertex_count; ++corner)
        {
            const std::array<ClippedVertex, 3> clipped_triangle = {
                clipped_polygon[0],
                clipped_polygon[static_cast<std::size_t>(corner)],
                clipped_polygon[static_cast<std::size_t>(corner + 1)]};
            ProjectedTriangle projected;
            if (projectTriangle(clipped_triangle, camera, imageSize, &projected))
            {
                triangles.push_back(projected);
            }
        }
        if (triangles.size() > projected_triangle_count_before_face)
        {
            ++visible_triangle_count;
        }
    }
    if (triangles.empty())
    {
        result.error = QStringLiteral("模型完全不在相机有效视野内");
        return result;
    }

    const int tile_columns = (imageSize.width + kTileSize - 1) / kTileSize;
    const int tile_rows = (imageSize.height + kTileSize - 1) / kTileSize;
    std::vector<std::vector<int>> tile_triangles(
        static_cast<std::size_t>(tile_columns * tile_rows));
    for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index)
    {
        const ProjectedTriangle &triangle = triangles[triangle_index];
        const int first_tile_x = triangle.minimumX / kTileSize;
        const int last_tile_x = triangle.maximumX / kTileSize;
        const int first_tile_y = triangle.minimumY / kTileSize;
        const int last_tile_y = triangle.maximumY / kTileSize;
        for (int tile_y = first_tile_y; tile_y <= last_tile_y; ++tile_y)
        {
            for (int tile_x = first_tile_x; tile_x <= last_tile_x; ++tile_x)
            {
                tile_triangles[static_cast<std::size_t>(tile_y * tile_columns + tile_x)]
                    .push_back(static_cast<int>(triangle_index));
            }
        }
    }

    cv::parallel_for_(cv::Range(0, tile_columns * tile_rows), [&](const cv::Range &range)
    {
        for (int tile_index = range.start; tile_index < range.end; ++tile_index)
        {
            const int tile_x = tile_index % tile_columns;
            const int tile_y = tile_index / tile_columns;
            const int tile_minimum_x = tile_x * kTileSize;
            const int tile_maximum_x = std::min(imageSize.width - 1,
                                                tile_minimum_x + kTileSize - 1);
            const int tile_minimum_y = tile_y * kTileSize;
            const int tile_maximum_y = std::min(imageSize.height - 1,
                                                tile_minimum_y + kTileSize - 1);
            for (const int triangle_index : tile_triangles[static_cast<std::size_t>(tile_index)])
            {
                const ProjectedTriangle &triangle =
                    triangles[static_cast<std::size_t>(triangle_index)];
                const int minimum_x = std::max(tile_minimum_x, triangle.minimumX);
                const int maximum_x = std::min(tile_maximum_x, triangle.maximumX);
                const int minimum_y = std::max(tile_minimum_y, triangle.minimumY);
                const int maximum_y = std::min(tile_maximum_y, triangle.maximumY);
                for (int y = minimum_y; y <= maximum_y; ++y)
                {
                    for (int x = minimum_x; x <= maximum_x; ++x)
                    {
                        const double pixel_x = static_cast<double>(x) + 0.5;
                        const double pixel_y = static_cast<double>(y) + 0.5;
                        const double weight_0 = edge(
                            triangle.vertices[1].x, triangle.vertices[1].y,
                            triangle.vertices[2].x, triangle.vertices[2].y,
                            pixel_x, pixel_y) / triangle.area;
                        const double weight_1 = edge(
                            triangle.vertices[2].x, triangle.vertices[2].y,
                            triangle.vertices[0].x, triangle.vertices[0].y,
                            pixel_x, pixel_y) / triangle.area;
                        const double weight_2 = 1.0 - weight_0 - weight_1;
                        constexpr double tolerance = -1.0e-5;
                        if (weight_0 < tolerance || weight_1 < tolerance ||
                            weight_2 < tolerance)
                        {
                            continue;
                        }

                        const double reciprocal_depth =
                            weight_0 / triangle.vertices[0].depth +
                            weight_1 / triangle.vertices[1].depth +
                            weight_2 / triangle.vertices[2].depth;
                        if (!std::isfinite(reciprocal_depth) || reciprocal_depth <= 0.0)
                        {
                            continue;
                        }
                        const double depth = 1.0 / reciprocal_depth;
                        if (!std::isfinite(depth) || depth <= 0.0 ||
                            depth > static_cast<double>(std::numeric_limits<float>::max()))
                        {
                            continue;
                        }
                        float &stored_depth = result.depth.at<float>(y, x);
                        if (depth >= stored_depth)
                        {
                            continue;
                        }
                        stored_depth = static_cast<float>(depth);
                        result.validMask.at<std::uint8_t>(y, x) = 255;
                        cv::Vec3b &color = result.color.at<cv::Vec3b>(y, x);
                        const double perspective_weight_0 =
                            (weight_0 / triangle.vertices[0].depth) * depth;
                        const double perspective_weight_1 =
                            (weight_1 / triangle.vertices[1].depth) * depth;
                        const double perspective_weight_2 =
                            (weight_2 / triangle.vertices[2].depth) * depth;
                        color[0] = colorByte(
                            perspective_weight_0 * triangle.vertices[0].blue +
                            perspective_weight_1 * triangle.vertices[1].blue +
                            perspective_weight_2 * triangle.vertices[2].blue);
                        color[1] = colorByte(
                            perspective_weight_0 * triangle.vertices[0].green +
                            perspective_weight_1 * triangle.vertices[1].green +
                            perspective_weight_2 * triangle.vertices[2].green);
                        color[2] = colorByte(
                            perspective_weight_0 * triangle.vertices[0].red +
                            perspective_weight_1 * triangle.vertices[1].red +
                            perspective_weight_2 * triangle.vertices[2].red);
                    }
                }
            }
        }
    });

    result.visibleTriangleCount = visible_triangle_count;
    result.ok = cv::countNonZero(result.validMask) > 0;
    if (!result.ok)
    {
        result.error = QStringLiteral("模型投影没有产生有效像素");
    }
    result.elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace xjw::qc
