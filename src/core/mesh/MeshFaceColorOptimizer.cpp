#include "MeshFaceColorOptimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct FacePoint
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

FacePoint subtract(const MeshVertex &left, const MeshVertex &right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

FacePoint normalizedCross(const FacePoint &left, const FacePoint &right)
{
    FacePoint result{
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x};
    const float magnitude = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
    if (magnitude <= 1.0e-12f)
    {
        return {};
    }
    result.x /= magnitude;
    result.y /= magnitude;
    result.z /= magnitude;
    return result;
}

cv::Vec3f bilinearColor(const cv::Mat &image, double x, double y)
{
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.rows - 1);
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float tx = static_cast<float>(x - x0);
    const float ty = static_cast<float>(y - y0);
    const cv::Vec3f top = cv::Vec3f(image.at<cv::Vec3b>(y0, x0)) * (1.0f - tx)
        + cv::Vec3f(image.at<cv::Vec3b>(y0, x1)) * tx;
    const cv::Vec3f bottom = cv::Vec3f(image.at<cv::Vec3b>(y1, x0)) * (1.0f - tx)
        + cv::Vec3f(image.at<cv::Vec3b>(y1, x1)) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

bool isBilinearMaskSampleValid(const cv::Mat &mask, double x, double y)
{
    if (mask.empty())
    {
        return true;
    }
    if (mask.type() != CV_8UC1 ||
        x < 0.0 || y < 0.0 || x > mask.cols - 1.0 || y > mask.rows - 1.0)
    {
        return false;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, mask.cols - 1);
    const int y1 = std::min(y0 + 1, mask.rows - 1);
    return mask.at<std::uint8_t>(y0, x0) != 0 &&
           mask.at<std::uint8_t>(y0, x1) != 0 &&
           mask.at<std::uint8_t>(y1, x0) != 0 &&
           mask.at<std::uint8_t>(y1, x1) != 0;
}

bool projectColorPoint(const MeshColorView &view,
                       const double world[3],
                       double pixel[2])
{
    double depth = 0.0;
    return view.colorCamera.projectWorldPointWithDepth(world, pixel, depth) &&
           pixel[0] >= 0.0 && pixel[1] >= 0.0 &&
           pixel[0] <= view.colorBgr.cols - 1.0 &&
           pixel[1] <= view.colorBgr.rows - 1.0 &&
           isBilinearMaskSampleValid(view.colorForegroundMask, pixel[0], pixel[1]);
}

float scoreFaceView(const TriMesh &mesh,
                    const Triangle &face,
                    const MeshColorView &view,
                    const MeshColorOptions &options)
{
    if (view.colorBgr.type() != CV_8UC3 || view.depth.type() != CV_32FC1 ||
        view.confidence.type() != CV_32FC1 || view.depthValidMask.type() != CV_8UC1 ||
        view.supportMask.type() != CV_8UC1)
    {
        return -1.0f;
    }
    const MeshVertex &first = mesh.vertices[static_cast<std::size_t>(face.v[0])];
    const MeshVertex &second = mesh.vertices[static_cast<std::size_t>(face.v[1])];
    const MeshVertex &third = mesh.vertices[static_cast<std::size_t>(face.v[2])];
    const FacePoint centroid{(first.x + second.x + third.x) / 3.0f,
                             (first.y + second.y + third.y) / 3.0f,
                             (first.z + second.z + third.z) / 3.0f};
    const FacePoint normal = normalizedCross(subtract(second, first), subtract(third, first));
    const double world[3] = {centroid.x, centroid.y, centroid.z};
    double pixel[2]{};
    double camera_depth = 0.0;
    if (!view.camera.projectWorldPointWithDepth(world, pixel, camera_depth))
    {
        return -1.0f;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 || row >= view.depth.rows || column >= view.depth.cols ||
        view.supportMask.at<std::uint8_t>(row, column) == 0 ||
        view.depthValidMask.at<std::uint8_t>(row, column) == 0)
    {
        return -1.0f;
    }
    double color_pixel[2]{};
    if (!projectColorPoint(view, world, color_pixel))
    {
        return -1.0f;
    }
    const float observedDepth = view.depth.at<float>(row, column);
    const float confidence = view.confidence.at<float>(row, column);
    if (!std::isfinite(observedDepth) || observedDepth <= 0.0f ||
        !std::isfinite(confidence) || confidence < options.minimumConfidence)
    {
        return -1.0f;
    }
    const float voxelSize = std::max(options.maximumVoxelSize, 1.0e-8f);
    const float tolerance = std::max(
        options.depthToleranceVoxels * voxelSize,
        options.relativeDepthTolerance * std::fabs(static_cast<float>(camera_depth)));
    const float residual = std::fabs(observedDepth - static_cast<float>(camera_depth));
    if (residual > tolerance)
    {
        return -1.0f;
    }
    const std::array<double, 3> center = view.camera.cameraCenter();
    FacePoint direction{static_cast<float>(center[0]) - centroid.x,
                        static_cast<float>(center[1]) - centroid.y,
                        static_cast<float>(center[2]) - centroid.z};
    const float direction_length = std::sqrt(
        direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    if (direction_length <= 1.0e-8f)
    {
        return -1.0f;
    }
    direction.x /= direction_length;
    direction.y /= direction_length;
    direction.z /= direction_length;
    const float view_cosine = std::fabs(
        normal.x * direction.x + normal.y * direction.y + normal.z * direction.z);
    if (view_cosine < options.minimumViewCosine)
    {
        return -1.0f;
    }
    const float residual_score = 1.0f /
        std::pow(1.0f + residual / std::max(tolerance, 1.0e-8f), 2.0f);
    return confidence * std::max(0.0f, view.qualityWeight)
        * std::pow(view_cosine, 4.0f) * residual_score;
}

bool sampleVertex(const MeshVertex &vertex,
                  const MeshColorView &view,
                  const MeshColorOptions &options,
                  cv::Vec3f *color)
{
    const double world[3] = {vertex.x, vertex.y, vertex.z};
    double pixel[2]{};
    double camera_depth = 0.0;
    if (!view.camera.projectWorldPointWithDepth(world, pixel, camera_depth))
    {
        return false;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 || row >= view.depth.rows || column >= view.depth.cols ||
        view.supportMask.at<std::uint8_t>(row, column) == 0 ||
        view.depthValidMask.at<std::uint8_t>(row, column) == 0)
    {
        return false;
    }
    const float observedDepth = view.depth.at<float>(row, column);
    const float voxel_size = std::max(options.maximumVoxelSize, 1.0e-8f);
    const float tolerance = std::max(7.5f * voxel_size,
        0.008f * std::fabs(static_cast<float>(camera_depth)));
    if (!std::isfinite(observedDepth) || observedDepth <= 0.0f ||
        std::fabs(observedDepth - static_cast<float>(camera_depth)) > tolerance)
    {
        return false;
    }
    double color_pixel[2]{};
    if (!projectColorPoint(view, world, color_pixel))
    {
        return false;
    }
    *color = bilinearColor(view.colorBgr, color_pixel[0], color_pixel[1]);
    return true;
}

} // namespace

FaceColorCoherenceStatistics applyFaceCoherentPrimaryViews(
    TriMesh *mesh,
    const QVector<MeshColorView> &views,
    const MeshColorOptions &options)
{
    FaceColorCoherenceStatistics statistics;
    if (!mesh || mesh->empty() || views.empty())
    {
        return statistics;
    }
    const int viewCount = std::min(static_cast<int>(views.size()), 32);
    std::vector<float> votes(mesh->vertices.size() * static_cast<std::size_t>(viewCount), 0.0f);
    for (const Triangle &face : mesh->faces)
    {
        int bestView = -1;
        float bestScore = 0.0f;
        for (int viewIndex = 0; viewIndex < viewCount; ++viewIndex)
        {
            const float score = scoreFaceView(*mesh, face, views[viewIndex], options);
            if (score > bestScore)
            {
                bestScore = score;
                bestView = viewIndex;
            }
        }
        if (bestView < 0)
        {
            continue;
        }
        ++statistics.assignedFaceCount;
        for (int vertexIndex : face.v)
        {
            votes[static_cast<std::size_t>(vertexIndex) * viewCount + bestView] += bestScore;
        }
    }
    for (std::size_t vertexIndex = 0; vertexIndex < mesh->vertices.size(); ++vertexIndex)
    {
        const auto begin = votes.cbegin() + static_cast<std::ptrdiff_t>(vertexIndex * viewCount);
        const auto end = begin + viewCount;
        const auto best = std::max_element(begin, end);
        if (best == end || *best <= 0.0f)
        {
            continue;
        }
        const int viewIndex = static_cast<int>(std::distance(begin, best));
        cv::Vec3f color;
        if (!sampleVertex(mesh->vertices[vertexIndex], views[viewIndex], options, &color))
        {
            continue;
        }
        MeshVertex &vertex = mesh->vertices[vertexIndex];
        vertex.b = static_cast<std::uint8_t>(std::clamp(std::lround(color[0]), 0l, 255l));
        vertex.g = static_cast<std::uint8_t>(std::clamp(std::lround(color[1]), 0l, 255l));
        vertex.r = static_cast<std::uint8_t>(std::clamp(std::lround(color[2]), 0l, 255l));
        ++statistics.recoloredVertexCount;
    }
    return statistics;
}

} // namespace xjw::mesh
