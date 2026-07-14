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

struct ProjectedVertex
{
    float x = 0.0f;
    float y = 0.0f;
    float depth = 0.0f;
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
};

struct ProjectedTriangle
{
    std::array<ProjectedVertex, 3> vertices;
    float area = 0.0f;
    int minimumX = 0;
    int maximumX = -1;
    int minimumY = 0;
    int maximumY = -1;
};

float edge(float ax, float ay, float bx, float by, float px, float py)
{
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

bool projectTriangle(const xjw::mesh::TriMesh &mesh,
                     const xjw::mesh::Triangle &face,
                     const xjw::Camera &camera,
                     const cv::Size &image_size,
                     ProjectedTriangle *projected)
{
    if (!projected)
    {
        return false;
    }

    for (int corner = 0; corner < 3; ++corner)
    {
        const xjw::mesh::MeshVertex &source =
            mesh.vertices[static_cast<std::size_t>(face.v[corner])];
        ProjectedVertex &target = projected->vertices[static_cast<std::size_t>(corner)];
        const double world[3] = {source.x, source.y, source.z};
        double pixel[2] = {};
        double positiveDepth = 0.0;
        if (!camera.projectWorldPointWithDepth(world, pixel, positiveDepth) ||
            !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
            !std::isfinite(positiveDepth) || positiveDepth <= 0.0)
        {
            return false;
        }
        target.x = static_cast<float>(pixel[0]);
        target.y = static_cast<float>(pixel[1]);
        target.depth = static_cast<float>(positiveDepth);
        if (!std::isfinite(target.x) || !std::isfinite(target.y) ||
            !std::isfinite(target.depth) || target.depth <= 0.0f)
        {
            return false;
        }
        target.red = static_cast<float>(source.r);
        target.green = static_cast<float>(source.g);
        target.blue = static_cast<float>(source.b);
    }

    projected->area = edge(projected->vertices[0].x, projected->vertices[0].y,
                           projected->vertices[1].x, projected->vertices[1].y,
                           projected->vertices[2].x, projected->vertices[2].y);
    if (std::abs(projected->area) < 1.0e-6f)
    {
        return false;
    }

    const float minimum_x = std::min({projected->vertices[0].x,
                                      projected->vertices[1].x,
                                      projected->vertices[2].x});
    const float maximum_x = std::max({projected->vertices[0].x,
                                      projected->vertices[1].x,
                                      projected->vertices[2].x});
    const float minimum_y = std::min({projected->vertices[0].y,
                                      projected->vertices[1].y,
                                      projected->vertices[2].y});
    const float maximum_y = std::max({projected->vertices[0].y,
                                      projected->vertices[1].y,
                                      projected->vertices[2].y});

    projected->minimumX = std::max(0, static_cast<int>(std::floor(minimum_x)));
    projected->maximumX = std::min(image_size.width - 1,
                                   static_cast<int>(std::ceil(maximum_x)));
    projected->minimumY = std::max(0, static_cast<int>(std::floor(minimum_y)));
    projected->maximumY = std::min(image_size.height - 1,
                                   static_cast<int>(std::ceil(maximum_y)));
    return projected->minimumX <= projected->maximumX &&
           projected->minimumY <= projected->maximumY;
}

std::uint8_t colorByte(float value)
{
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 255.0f) + 0.5f);
}

} // namespace

ModelRenderResult ModelMeshRenderer::render(
    const xjw::mesh::TriMesh &mesh,
    const xjw::Camera &camera,
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
    triangles.reserve(mesh.faces.size());
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        ProjectedTriangle projected;
        if (projectTriangle(mesh, face, camera, imageSize, &projected))
        {
            triangles.push_back(projected);
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
                        const float pixel_x = static_cast<float>(x) + 0.5f;
                        const float pixel_y = static_cast<float>(y) + 0.5f;
                        const float weight_0 = edge(
                            triangle.vertices[1].x, triangle.vertices[1].y,
                            triangle.vertices[2].x, triangle.vertices[2].y,
                            pixel_x, pixel_y) / triangle.area;
                        const float weight_1 = edge(
                            triangle.vertices[2].x, triangle.vertices[2].y,
                            triangle.vertices[0].x, triangle.vertices[0].y,
                            pixel_x, pixel_y) / triangle.area;
                        const float weight_2 = 1.0f - weight_0 - weight_1;
                        constexpr float tolerance = -1.0e-5f;
                        if (weight_0 < tolerance || weight_1 < tolerance ||
                            weight_2 < tolerance)
                        {
                            continue;
                        }

                        const float depth = weight_0 * triangle.vertices[0].depth +
                                            weight_1 * triangle.vertices[1].depth +
                                            weight_2 * triangle.vertices[2].depth;
                        float &stored_depth = result.depth.at<float>(y, x);
                        if (depth >= stored_depth)
                        {
                            continue;
                        }
                        stored_depth = depth;
                        result.validMask.at<std::uint8_t>(y, x) = 255;
                        cv::Vec3b &color = result.color.at<cv::Vec3b>(y, x);
                        color[0] = colorByte(weight_0 * triangle.vertices[0].blue +
                                             weight_1 * triangle.vertices[1].blue +
                                             weight_2 * triangle.vertices[2].blue);
                        color[1] = colorByte(weight_0 * triangle.vertices[0].green +
                                             weight_1 * triangle.vertices[1].green +
                                             weight_2 * triangle.vertices[2].green);
                        color[2] = colorByte(weight_0 * triangle.vertices[0].red +
                                             weight_1 * triangle.vertices[1].red +
                                             weight_2 * triangle.vertices[2].red);
                    }
                }
            }
        }
    });

    result.visibleTriangleCount = static_cast<int>(triangles.size());
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
