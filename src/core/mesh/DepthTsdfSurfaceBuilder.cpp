#include "DepthTsdfSurfaceBuilder.h"

#include "DepthFrameUtils.h"
#include "MeshColorizer.h"
#include "MeshQuadricSimplifier.h"
#include "MeshTopologyQuality.h"
#include "SurfaceReconstructorPostprocess.h"
#include "io/PathIO.h"

#include <QJsonArray>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <plapoint/mesh/marching_cubes.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <atomic>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace xjw::mesh
{

namespace
{

constexpr std::uint64_t kBaseBytesPerSample =
    sizeof(float) * 3 + sizeof(std::uint16_t) * 2;
constexpr std::uint64_t kColorBytesPerSample = sizeof(float) * 4;

bool checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *result)
{
    if (!result || (rhs > 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs))
    {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

std::uint64_t availablePhysicalMemoryBytes()
{
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
    {
        return static_cast<std::uint64_t>(status.ullAvailPhys);
    }
#endif
    return 0;
}

struct MeshBoundaryTopology
{
    int boundaryEdgeCount = 0;
    int danglingBoundaryVertexCount = 0;
    int nonManifoldEdgeCount = 0;
};

MeshBoundaryTopology boundaryTopology(const TriMesh &mesh)
{
    const auto edge_key = [](int first, int second)
    {
        const auto low = static_cast<std::uint32_t>(std::min(first, second));
        const auto high = static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U)
            | static_cast<std::uint64_t>(high);
    };
    std::unordered_map<std::uint64_t, int> counts;
    counts.reserve(mesh.faces.size() * 3);
    for (const Triangle &face : mesh.faces)
    {
        ++counts[edge_key(face.v[0], face.v[1])];
        ++counts[edge_key(face.v[1], face.v[2])];
        ++counts[edge_key(face.v[2], face.v[0])];
    }
    MeshBoundaryTopology topology;
    std::vector<int> boundary_degree(mesh.vertices.size(), 0);
    for (const auto &entry : counts)
    {
        if (entry.second == 1)
        {
            ++topology.boundaryEdgeCount;
            const int first = static_cast<int>(entry.first >> 32U);
            const int second = static_cast<int>(entry.first & 0xffffffffU);
            ++boundary_degree[static_cast<std::size_t>(first)];
            ++boundary_degree[static_cast<std::size_t>(second)];
        }
        else if (entry.second > 2)
        {
            ++topology.nonManifoldEdgeCount;
        }
    }
    topology.danglingBoundaryVertexCount = static_cast<int>(std::count(
        boundary_degree.cbegin(), boundary_degree.cend(), 1));
    return topology;
}

int boundaryEdgeCount(const TriMesh &mesh)
{
    return boundaryTopology(mesh).boundaryEdgeCount;
}

struct TriangleQualitySummary
{
    int validFaceCount = 0;
    int sliverFaceCount = 0;
    double sliverRatio = 0.0;
};

TriangleQualitySummary triangleQualitySummary(const TriMesh &mesh)
{
    constexpr double sliver_quality_threshold = 0.05;
    constexpr double normalized_area_factor = 3.4641016151377544;
    TriangleQualitySummary summary;
    for (const Triangle &face : mesh.faces)
    {
        if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 ||
            static_cast<std::size_t>(face.v[0]) >= mesh.vertices.size() ||
            static_cast<std::size_t>(face.v[1]) >= mesh.vertices.size() ||
            static_cast<std::size_t>(face.v[2]) >= mesh.vertices.size())
        {
            continue;
        }
        const MeshVertex &a = mesh.vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &b = mesh.vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &c = mesh.vertices[static_cast<std::size_t>(face.v[2])];
        const double ab_x = b.x - a.x;
        const double ab_y = b.y - a.y;
        const double ab_z = b.z - a.z;
        const double ac_x = c.x - a.x;
        const double ac_y = c.y - a.y;
        const double ac_z = c.z - a.z;
        const double bc_x = c.x - b.x;
        const double bc_y = c.y - b.y;
        const double bc_z = c.z - b.z;
        const double cross_x = ab_y * ac_z - ab_z * ac_y;
        const double cross_y = ab_z * ac_x - ab_x * ac_z;
        const double cross_z = ab_x * ac_y - ab_y * ac_x;
        const double doubled_area = std::sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        const double squared_edge_sum =
            ab_x * ab_x + ab_y * ab_y + ab_z * ab_z +
            ac_x * ac_x + ac_y * ac_y + ac_z * ac_z +
            bc_x * bc_x + bc_y * bc_y + bc_z * bc_z;
        const double quality = squared_edge_sum > 1.0e-20
            ? normalized_area_factor * doubled_area / squared_edge_sum
            : 0.0;
        ++summary.validFaceCount;
        if (!std::isfinite(quality) || quality < sliver_quality_threshold)
        {
            ++summary.sliverFaceCount;
        }
    }
    if (summary.validFaceCount > 0)
    {
        summary.sliverRatio = static_cast<double>(summary.sliverFaceCount) /
            summary.validFaceCount;
    }
    return summary;
}

std::vector<std::uint8_t> boundaryVertexMask(const TriMesh &mesh)
{
    const auto edge_key = [](int first, int second)
    {
        const auto low = static_cast<std::uint32_t>(std::min(first, second));
        const auto high = static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U) |
            static_cast<std::uint64_t>(high);
    };
    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh.faces.size() * 3);
    for (const Triangle &face : mesh.faces)
    {
        ++edge_counts[edge_key(face.v[0], face.v[1])];
        ++edge_counts[edge_key(face.v[1], face.v[2])];
        ++edge_counts[edge_key(face.v[2], face.v[0])];
    }
    std::vector<std::uint8_t> boundary_vertices(mesh.vertices.size(), 0);
    for (const auto &[key, count] : edge_counts)
    {
        if (count != 1)
        {
            continue;
        }
        boundary_vertices[static_cast<std::size_t>(key >> 32U)] = 1;
        boundary_vertices[static_cast<std::size_t>(key & 0xffffffffU)] = 1;
    }
    return boundary_vertices;
}

std::vector<std::uint8_t> multiViewSilhouetteBoundaryVertices(
    const TriMesh &mesh,
    const QVector<DepthTsdfFrame> &frames,
    const QVector<cv::Mat> &effectiveDepthValidMasks,
    int minimumViews,
    int bandPixels,
    float depthToleranceVoxels,
    float maximumVoxelSize,
    float minimumConfidence)
{
    std::vector<std::uint8_t> protected_vertices(mesh.vertices.size(), 0);
    if (mesh.empty() || frames.empty())
    {
        return protected_vertices;
    }

    const std::vector<std::uint8_t> boundary_vertices = boundaryVertexMask(mesh);
    QVector<cv::Mat> silhouette_bands;
    silhouette_bands.reserve(frames.size());
    const int radius = std::clamp(bandPixels, 1, 8);
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
    for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
    {
        const DepthTsdfFrame &frame = frames[frame_index];
        cv::Mat band;
        cv::morphologyEx(frame.supportMask, band, cv::MORPH_GRADIENT, kernel);
        cv::bitwise_and(band, frame.supportMask, band);
        const cv::Mat &valid_mask =
            effectiveDepthValidMasks.size() == frames.size()
            ? effectiveDepthValidMasks[frame_index]
            : frame.depthValidMask;
        cv::bitwise_and(band, valid_mask, band);
        silhouette_bands.push_back(std::move(band));
    }

    const int required_views = std::clamp(
        minimumViews, 1, static_cast<int>(frames.size()));
    const float absolute_tolerance =
        std::max(1.0f, depthToleranceVoxels) *
        std::max(maximumVoxelSize, 1.0e-8f);
    for (std::size_t vertex_index = 0;
         vertex_index < mesh.vertices.size();
         ++vertex_index)
    {
        if (boundary_vertices[vertex_index] == 0)
        {
            continue;
        }
        const MeshVertex &vertex = mesh.vertices[vertex_index];
        const double world[3] = {vertex.x, vertex.y, vertex.z};
        int agreeing_views = 0;
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame &frame = frames[frame_index];
            double pixel[2]{};
            double camera_depth = 0.0;
            if (!frame.camera.projectWorldPointWithDepth(
                    world, pixel, camera_depth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
            if (row < 0 || column < 0 ||
                row >= silhouette_bands[frame_index].rows ||
                column >= silhouette_bands[frame_index].cols ||
                silhouette_bands[frame_index].at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            const float observed_depth = frame.depth.at<float>(row, column);
            const float confidence = frame.confidence.at<float>(row, column);
            const float tolerance = std::max(
                absolute_tolerance,
                0.008f * std::fabs(static_cast<float>(camera_depth)));
            if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
                !std::isfinite(confidence) || confidence < minimumConfidence ||
                std::fabs(observed_depth - static_cast<float>(camera_depth)) >
                    tolerance)
            {
                continue;
            }
            if (++agreeing_views >= required_views)
            {
                protected_vertices[vertex_index] = 1;
                break;
            }
        }
    }
    return protected_vertices;
}

struct VisibilityHoleProtectionResult
{
    std::vector<std::uint8_t> protectedVertices;
    int consideredLoopCount = 0;
    int releasedLoopCount = 0;
    int rejectedSupportLoopCount = 0;
    int rejectedConflictLoopCount = 0;
};

VisibilityHoleProtectionResult visibilityConstrainedHoleProtection(
    const TriMesh &mesh,
    const QVector<DepthTsdfFrame> &frames,
    const QVector<cv::Mat> &effectiveDepthValidMasks,
    const std::vector<std::uint8_t> &silhouetteProtectedVertices,
    int maximumBoundaryEdges,
    int minimumSupportingViews,
    int maximumConflictViews,
    float depthToleranceVoxels,
    float strongSilhouetteRatio,
    float maximumVoxelSize,
    float minimumConfidence)
{
    VisibilityHoleProtectionResult result;
    result.protectedVertices = silhouetteProtectedVertices;
    if (mesh.empty() || frames.empty() ||
        silhouetteProtectedVertices.size() != mesh.vertices.size())
    {
        return result;
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh.faces.size() * 3);
    const auto add_edge = [&edge_counts](int first, int second)
    {
        const auto low =
            static_cast<std::uint32_t>(std::min(first, second));
        const auto high =
            static_cast<std::uint32_t>(std::max(first, second));
        const std::uint64_t key =
            (static_cast<std::uint64_t>(low) << 32U) |
            static_cast<std::uint64_t>(high);
        ++edge_counts[key];
        return key;
    };
    for (const Triangle &face : mesh.faces)
    {
        add_edge(face.v[0], face.v[1]);
        add_edge(face.v[1], face.v[2]);
        add_edge(face.v[2], face.v[0]);
    }
    const auto edge_key = [](int first, int second)
    {
        const auto low =
            static_cast<std::uint32_t>(std::min(first, second));
        const auto high =
            static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U) |
            static_cast<std::uint64_t>(high);
    };
    std::vector<std::vector<int>> boundary_neighbors(mesh.vertices.size());
    for (const Triangle &face : mesh.faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            if (edge_counts[edge_key(edge[0], edge[1])] != 1)
            {
                continue;
            }
            boundary_neighbors[static_cast<std::size_t>(edge[0])].push_back(
                edge[1]);
            boundary_neighbors[static_cast<std::size_t>(edge[1])].push_back(
                edge[0]);
        }
    }

    std::unordered_set<std::uint64_t> visited_edges;
    visited_edges.reserve(edge_counts.size());
    const int required_supporting_views = std::clamp(
        minimumSupportingViews, 1, static_cast<int>(frames.size()));
    const int allowed_conflict_views = std::max(0, maximumConflictViews);
    const float absolute_tolerance =
        std::max(1.0f, depthToleranceVoxels) *
        std::max(maximumVoxelSize, 1.0e-8f);
    const float strong_ratio = std::clamp(
        strongSilhouetteRatio, 0.0f, 1.0f);
    for (int start = 0;
         start < static_cast<int>(boundary_neighbors.size());
         ++start)
    {
        if (boundary_neighbors[static_cast<std::size_t>(start)].size() != 2)
        {
            continue;
        }
        for (const int first_neighbor :
             boundary_neighbors[static_cast<std::size_t>(start)])
        {
            if (visited_edges.find(edge_key(start, first_neighbor)) !=
                visited_edges.cend())
            {
                continue;
            }
            std::vector<int> loop{start};
            int previous = start;
            int current = first_neighbor;
            bool closed = false;
            visited_edges.insert(edge_key(previous, current));
            while (static_cast<int>(loop.size()) <= maximumBoundaryEdges)
            {
                if (current == start)
                {
                    closed = true;
                    break;
                }
                loop.push_back(current);
                const std::vector<int> &neighbors =
                    boundary_neighbors[static_cast<std::size_t>(current)];
                if (neighbors.size() != 2)
                {
                    break;
                }
                const int next =
                    neighbors[0] == previous ? neighbors[1] : neighbors[0];
                const std::uint64_t next_key = edge_key(current, next);
                if (next != start &&
                    visited_edges.find(next_key) != visited_edges.cend())
                {
                    break;
                }
                previous = current;
                current = next;
                visited_edges.insert(next_key);
            }
            if (!closed || loop.size() < 3 ||
                static_cast<int>(loop.size()) > maximumBoundaryEdges)
            {
                continue;
            }
            const int silhouette_vertex_count = static_cast<int>(std::count_if(
                loop.cbegin(), loop.cend(),
                [&silhouetteProtectedVertices](int vertex)
                {
                    return silhouetteProtectedVertices[
                        static_cast<std::size_t>(vertex)] != 0;
                }));
            if (silhouette_vertex_count == 0)
            {
                continue;
            }
            ++result.consideredLoopCount;

            MeshVertex center;
            for (const int vertex_index : loop)
            {
                const MeshVertex &vertex =
                    mesh.vertices[static_cast<std::size_t>(vertex_index)];
                center.x += vertex.x;
                center.y += vertex.y;
                center.z += vertex.z;
            }
            const float inverse_count =
                1.0f / static_cast<float>(loop.size());
            center.x *= inverse_count;
            center.y *= inverse_count;
            center.z *= inverse_count;
            std::array<MeshVertex, 5> samples{};
            samples[0] = center;
            for (int sample_index = 1; sample_index < 5; ++sample_index)
            {
                const std::size_t loop_index =
                    (loop.size() * static_cast<std::size_t>(sample_index - 1)) /
                    4;
                const MeshVertex &boundary = mesh.vertices[
                    static_cast<std::size_t>(loop[loop_index])];
                samples[static_cast<std::size_t>(sample_index)].x =
                    0.5f * (center.x + boundary.x);
                samples[static_cast<std::size_t>(sample_index)].y =
                    0.5f * (center.y + boundary.y);
                samples[static_cast<std::size_t>(sample_index)].z =
                    0.5f * (center.z + boundary.z);
            }

            int supporting_views = 0;
            int conflict_views = 0;
            for (int frame_index = 0;
                 frame_index < frames.size();
                 ++frame_index)
            {
                const DepthTsdfFrame &frame = frames[frame_index];
                const cv::Mat &valid_mask =
                    effectiveDepthValidMasks.size() == frames.size()
                    ? effectiveDepthValidMasks[frame_index]
                    : frame.depthValidMask;
                int projected_sample_count = 0;
                int consistent_sample_count = 0;
                int outside_support_sample_count = 0;
                for (const MeshVertex &sample : samples)
                {
                    const double world[3] = {
                        sample.x, sample.y, sample.z};
                    double pixel[2]{};
                    double camera_depth = 0.0;
                    if (!frame.camera.projectWorldPointWithDepth(
                            world, pixel, camera_depth))
                    {
                        continue;
                    }
                    const int column =
                        static_cast<int>(std::lround(pixel[0]));
                    const int row =
                        static_cast<int>(std::lround(pixel[1]));
                    if (row < 0 || column < 0 ||
                        row >= frame.supportMask.rows ||
                        column >= frame.supportMask.cols)
                    {
                        continue;
                    }
                    ++projected_sample_count;
                    if (frame.supportMask.at<std::uint8_t>(
                            row, column) == 0)
                    {
                        ++outside_support_sample_count;
                        continue;
                    }
                    if (valid_mask.at<std::uint8_t>(row, column) == 0)
                    {
                        continue;
                    }
                    const float observed_depth =
                        frame.depth.at<float>(row, column);
                    const float confidence =
                        frame.confidence.at<float>(row, column);
                    const float tolerance = std::max(
                        absolute_tolerance,
                        0.008f * std::abs(
                            static_cast<float>(camera_depth)));
                    if (std::isfinite(observed_depth) &&
                        observed_depth > 0.0f &&
                        std::isfinite(confidence) &&
                        confidence >= minimumConfidence &&
                        std::abs(
                            observed_depth -
                            static_cast<float>(camera_depth)) <= tolerance)
                    {
                        ++consistent_sample_count;
                    }
                }
                if (projected_sample_count >= 3 &&
                    consistent_sample_count * 2 >=
                        projected_sample_count)
                {
                    ++supporting_views;
                }
                if (projected_sample_count >= 3 &&
                    outside_support_sample_count * 2 >=
                        projected_sample_count)
                {
                    ++conflict_views;
                }
            }

            const float silhouette_ratio =
                static_cast<float>(silhouette_vertex_count) /
                static_cast<float>(loop.size());
            const int extra_strong_support =
                silhouette_ratio >= strong_ratio ? 1 : 0;
            const bool enough_support =
                supporting_views >=
                    required_supporting_views + extra_strong_support;
            const bool has_conflict =
                conflict_views > allowed_conflict_views;
            if (enough_support && !has_conflict)
            {
                for (const int vertex : loop)
                {
                    result.protectedVertices[
                        static_cast<std::size_t>(vertex)] = 0;
                }
                ++result.releasedLoopCount;
            }
            else if (has_conflict)
            {
                ++result.rejectedConflictLoopCount;
            }
            else
            {
                ++result.rejectedSupportLoopCount;
            }
        }
    }
    return result;
}

QString frameArtifactError(const DepthFrameArtifact &artifact, const QString &reason)
{
    return QStringLiteral(
        "Invalid TSDF frame ref_index=%1 depth=%2 confidence=%3 geometry_support=%4 "
        "depth_valid_mask=%5 support_mask=%6: %7")
        .arg(artifact.refIndex)
        .arg(artifact.depthPath,
             artifact.confidencePath,
             artifact.geometrySupportPath,
             artifact.validMaskPath,
             artifact.supportMaskPath,
             reason);
}

bool loadFloatMatrix(const QString &path, cv::Mat *matrix, QString *reason)
{
    if (!matrix)
    {
        return false;
    }
    const xjw::common::OperationResult status =
        xjw::core::project::loadDepthMatStorage(path, matrix);
    if (!status.ok || matrix->empty())
    {
        if (reason)
        {
            *reason = status.errorMessage.isEmpty()
                ? QStringLiteral("matrix is empty")
                : status.errorMessage;
        }
        return false;
    }
    if (matrix->type() != CV_32FC1)
    {
        if (reason)
        {
            *reason = QStringLiteral("expected CV_32FC1, got type=%1").arg(matrix->type());
        }
        return false;
    }
    return true;
}

bool loadUnsignedShortMatrix(const QString &path, cv::Mat *matrix, QString *reason)
{
    if (!matrix)
    {
        return false;
    }
    const xjw::common::OperationResult status =
        xjw::core::project::loadDepthMatStorage(path, matrix);
    if (!status.ok || matrix->empty())
    {
        if (reason)
        {
            *reason = status.errorMessage.isEmpty()
                ? QStringLiteral("matrix is empty")
                : status.errorMessage;
        }
        return false;
    }
    if (matrix->type() != CV_16UC1)
    {
        if (reason)
        {
            *reason = QStringLiteral("expected CV_16UC1, got type=%1").arg(matrix->type());
        }
        return false;
    }
    return true;
}

bool loadMask(const QString &path,
              const cv::Size &size,
              cv::Mat *mask,
              QString *reason)
{
    if (!mask)
    {
        return false;
    }
    *mask = xjw::common::io::readImage(xjw::common::io::toUtf8Path(path), cv::IMREAD_GRAYSCALE);
    if (mask->empty())
    {
        if (reason)
        {
            *reason = QStringLiteral("mask cannot be read");
        }
        return false;
    }
    if (mask->type() != CV_8UC1 || mask->size() != size)
    {
        if (reason)
        {
            *reason = QStringLiteral("expected CV_8UC1 %1x%2, got type=%3 %4x%5")
                          .arg(size.width)
                          .arg(size.height)
                          .arg(mask->type())
                          .arg(mask->cols)
                          .arg(mask->rows);
        }
        return false;
    }
    cv::threshold(*mask, *mask, 0.0, 255.0, cv::THRESH_BINARY);
    return true;
}

void integrateWeighted(float *value, float *weight, float observation, float observationWeight)
{
    if (!value || !weight || observationWeight <= 0.0f)
    {
        return;
    }
    const float updatedWeight = *weight + observationWeight;
    *value = (*value * *weight + observation * observationWeight) / updatedWeight;
    *weight = updatedWeight;
}

std::size_t sampleIndex(const DepthTsdfLayout &layout, int x, int y, int z)
{
    const std::size_t rowSize = static_cast<std::size_t>(layout.cells[0] + 1);
    const std::size_t layerSize = rowSize * static_cast<std::size_t>(layout.cells[1] + 1);
    return static_cast<std::size_t>(z) * layerSize +
           static_cast<std::size_t>(y) * rowSize + static_cast<std::size_t>(x);
}

int bitCount(std::uint16_t value)
{
    int count = 0;
    while (value != 0)
    {
        value = static_cast<std::uint16_t>(value & (value - 1));
        ++count;
    }
    return count;
}

std::uint16_t globalGeometrySourceMask(const DepthTsdfFrame &frame,
                                       std::uint16_t local_mask)
{
    std::uint16_t global_mask = 0;
    const int source_count = std::min(
        16, static_cast<int>(frame.sourceIndices.size()));
    for (int ordinal = 0; ordinal < source_count; ++ordinal)
    {
        if ((local_mask & (static_cast<std::uint16_t>(1U) << ordinal)) == 0)
        {
            continue;
        }
        const int source_index = frame.sourceIndices[ordinal];
        if (source_index >= 0 && source_index < 16)
        {
            global_mask = static_cast<std::uint16_t>(
                global_mask | (static_cast<std::uint16_t>(1U) << source_index));
        }
    }
    return global_mask;
}

bool volumeNormalAt(const DepthTsdfLayout &layout,
                    const std::vector<float> &tsdf,
                    int x,
                    int y,
                    int z,
                    cv::Vec3f *normal)
{
    if (!normal || x <= 0 || x >= layout.cells[0] ||
        y <= 0 || y >= layout.cells[1] ||
        z <= 0 || z >= layout.cells[2])
    {
        return false;
    }
    cv::Vec3f gradient(
        tsdf[sampleIndex(layout, x + 1, y, z)] -
            tsdf[sampleIndex(layout, x - 1, y, z)],
        tsdf[sampleIndex(layout, x, y + 1, z)] -
            tsdf[sampleIndex(layout, x, y - 1, z)],
        tsdf[sampleIndex(layout, x, y, z + 1)] -
            tsdf[sampleIndex(layout, x, y, z - 1)]);
    const float length = std::sqrt(gradient.dot(gradient));
    if (!std::isfinite(length) || length <= 1.0e-6f)
    {
        return false;
    }
    *normal = gradient / length;
    return true;
}

struct WeakBoundaryTipResult
{
    int weakVertexCount = 0;
    int candidateFaceCount = 0;
    int trimmedFaceCount = 0;
};

WeakBoundaryTipResult trimWeakBoundaryTips(TriMesh *mesh,
                                           const DepthTsdfLayout &layout,
                                           const std::vector<std::uint16_t> &support,
                                           int minimum_support,
                                           int passes,
                                           bool enabled)
{
    WeakBoundaryTipResult result;
    if (!mesh || mesh->faces.empty() || support.empty())
    {
        return result;
    }
    const auto edge_key = [](int first, int second)
    {
        const auto low = static_cast<std::uint32_t>(std::min(first, second));
        const auto high = static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U) | high;
    };
    std::vector<std::uint8_t> counted_weak_vertices(mesh->vertices.size(), 0);
    const int maximum_passes = enabled ? std::clamp(passes, 1, 4) : 1;
    for (int pass = 0; pass < maximum_passes; ++pass)
    {
        std::unordered_map<std::uint64_t, int> edge_counts;
        edge_counts.reserve(mesh->faces.size() * 3);
        for (const Triangle &face : mesh->faces)
        {
            ++edge_counts[edge_key(face.v[0], face.v[1])];
            ++edge_counts[edge_key(face.v[1], face.v[2])];
            ++edge_counts[edge_key(face.v[2], face.v[0])];
        }
        std::vector<std::uint8_t> boundary_vertex(mesh->vertices.size(), 0);
        for (const Triangle &face : mesh->faces)
        {
            const std::array<std::array<int, 2>, 3> edges{{
                {{face.v[0], face.v[1]}},
                {{face.v[1], face.v[2]}},
                {{face.v[2], face.v[0]}}
            }};
            for (const auto &edge : edges)
            {
                if (edge_counts[edge_key(edge[0], edge[1])] == 1)
                {
                    boundary_vertex[static_cast<std::size_t>(edge[0])] = 1;
                    boundary_vertex[static_cast<std::size_t>(edge[1])] = 1;
                }
            }
        }
        std::vector<std::uint8_t> weak_vertex(mesh->vertices.size(), 0);
        for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
        {
            if (!boundary_vertex[index])
            {
                continue;
            }
            const MeshVertex &vertex = mesh->vertices[index];
            const int x = std::clamp(static_cast<int>(std::lround(
                                         (vertex.x - layout.boundsMin[0]) / layout.voxelSize[0])),
                                     0, layout.cells[0]);
            const int y = std::clamp(static_cast<int>(std::lround(
                                         (vertex.y - layout.boundsMin[1]) / layout.voxelSize[1])),
                                     0, layout.cells[1]);
            const int z = std::clamp(static_cast<int>(std::lround(
                                         (vertex.z - layout.boundsMin[2]) / layout.voxelSize[2])),
                                     0, layout.cells[2]);
            if (support[sampleIndex(layout, x, y, z)] <= minimum_support)
            {
                weak_vertex[index] = 1;
                if (!counted_weak_vertices[index])
                {
                    counted_weak_vertices[index] = 1;
                    ++result.weakVertexCount;
                }
            }
        }

        std::vector<std::uint8_t> remove_face(mesh->faces.size(), 0);
        int candidates = 0;
        for (std::size_t index = 0; index < mesh->faces.size(); ++index)
        {
            const Triangle &face = mesh->faces[index];
            const std::array<std::array<int, 2>, 3> edges{{
                {{face.v[0], face.v[1]}},
                {{face.v[1], face.v[2]}},
                {{face.v[2], face.v[0]}}
            }};
            int boundary_edge_count = 0;
            for (const auto &edge : edges)
            {
                boundary_edge_count += edge_counts[edge_key(edge[0], edge[1])] == 1;
            }
            const int weak_vertex_count =
                static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[0])]) +
                static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[1])]) +
                static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[2])]);
            const bool candidate = DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(
                boundary_edge_count, weak_vertex_count);
            if (candidate)
            {
                remove_face[index] = 1;
                ++candidates;
            }
        }
        result.candidateFaceCount += candidates;
        if (!enabled || candidates == 0)
        {
            break;
        }
        std::vector<Triangle> kept_faces;
        kept_faces.reserve(mesh->faces.size() - static_cast<std::size_t>(candidates));
        for (std::size_t index = 0; index < mesh->faces.size(); ++index)
        {
            if (!remove_face[index])
            {
                kept_faces.push_back(mesh->faces[index]);
            }
        }
        mesh->faces = std::move(kept_faces);
        result.trimmedFaceCount += candidates;
    }
    return result;
}

} // namespace

bool DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(int boundaryEdgeCount,
                                                          int weakVertexCount)
{
    // Remove ordinary boundary ears, and also peel a one-edge boundary face
    // only when every vertex comes from weak camera support.  The latter is
    // the common topology of the narrow triangular ribbons visible along
    // Temple's column silhouettes; requiring all three weak vertices avoids
    // eroding a well-supported surface merely because it reaches an opening.
    return (boundaryEdgeCount >= 2 && weakVertexCount >= 1) ||
           (boundaryEdgeCount == 1 && weakVertexCount == 3);
}

bool DepthTsdfSurfaceBuilder::shouldAcceptQuadricSimplification(
    int inputFaceCount,
    int outputFaceCount,
    int boundaryEdgeCountBefore,
    int boundaryEdgeCountAfter,
    float maximumBoundaryEdgeGrowthRatio)
{
    if (inputFaceCount <= 0 || outputFaceCount >= inputFaceCount ||
        boundaryEdgeCountBefore < 0 || boundaryEdgeCountAfter < 0)
    {
        return false;
    }
    if (boundaryEdgeCountBefore == 0)
    {
        return boundaryEdgeCountAfter == 0;
    }
    const int allowed_growth = static_cast<int>(std::ceil(
        boundaryEdgeCountBefore * std::clamp(maximumBoundaryEdgeGrowthRatio, 0.0f, 1.0f)));
    return boundaryEdgeCountAfter <= boundaryEdgeCountBefore + allowed_growth;
}

DepthTsdfLayout DepthTsdfSurfaceBuilder::makeLayout(const std::array<float, 3> &boundsMin,
                                                    const std::array<float, 3> &boundsMax,
                                                    int resolution,
                                                    bool includeColor)
{
    DepthTsdfLayout layout;
    layout.boundsMin = boundsMin;
    layout.boundsMax = boundsMax;
    if (resolution < 8)
    {
        return layout;
    }

    std::array<float, 3> extents{};
    float longestExtent = 0.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(boundsMin[axis]) || !std::isfinite(boundsMax[axis]) ||
            boundsMax[axis] <= boundsMin[axis])
        {
            return layout;
        }
        extents[axis] = boundsMax[axis] - boundsMin[axis];
        longestExtent = std::max(longestExtent, extents[axis]);
    }

    for (int axis = 0; axis < 3; ++axis)
    {
        layout.cells[axis] = std::max(
            1,
            static_cast<int>(std::lround(static_cast<double>(resolution) *
                                         static_cast<double>(extents[axis]) /
                                         static_cast<double>(longestExtent))));
        layout.voxelSize[axis] = extents[axis] / static_cast<float>(layout.cells[axis]);
    }

    std::uint64_t sampleCount = 1;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!checkedMultiply(sampleCount,
                             static_cast<std::uint64_t>(layout.cells[axis]) + 1u,
                             &sampleCount))
        {
            return layout;
        }
    }

    const std::uint64_t bytesPerSample =
        kBaseBytesPerSample + (includeColor ? kColorBytesPerSample : 0u);
    std::uint64_t requiredBytes = 0;
    if (!checkedMultiply(sampleCount, bytesPerSample, &requiredBytes))
    {
        return layout;
    }

    layout.sampleCount = sampleCount;
    layout.requiredBytes = requiredBytes;
    layout.ok = true;
    return layout;
}

DepthTsdfResult DepthTsdfSurfaceBuilder::validateAllocation(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const DepthTsdfOptions &options)
{
    DepthTsdfResult result;
    result.layout = makeLayout(boundsMin,
                               boundsMax,
                               options.resolution,
                               options.calculateVertexColors);
    if (!result.layout.ok)
    {
        result.errorMessage = QStringLiteral("Invalid TSDF bounds or resolution=%1")
                                  .arg(options.resolution);
        return result;
    }
    if (options.enableSurfacePatchSupport ||
        options.enableContourBandZeroCrossingSupport)
    {
        std::uint64_t evidence_bytes = 0;
        const std::size_t float_field_count =
            options.enableContourBandZeroCrossingSupport ? 4u : 3u;
        if (!checkedMultiply(result.layout.sampleCount,
                             sizeof(std::uint16_t) * 2u +
                                 sizeof(float) * float_field_count,
                             &evidence_bytes) ||
            result.layout.requiredBytes >
                std::numeric_limits<std::uint64_t>::max() - evidence_bytes)
        {
            result.layout.ok = false;
            result.errorMessage = QStringLiteral(
                "TSDF surface-patch evidence allocation overflow");
            return result;
        }
        result.layout.requiredBytes += evidence_bytes;
    }

    const std::uint64_t available = options.availableMemoryBytes > 0
        ? options.availableMemoryBytes
        : availablePhysicalMemoryBytes();
    const std::uint64_t budget = available > 0 ? available * 3u / 4u : 0u;
    if (budget > 0 && result.layout.requiredBytes > budget)
    {
        result.errorMessage = QStringLiteral(
            "TSDF allocation rejected: resolution=%1 cells=%2x%3x%4 required=%5 bytes available=%6 bytes")
                                  .arg(options.resolution)
                                  .arg(result.layout.cells[0])
                                  .arg(result.layout.cells[1])
                                  .arg(result.layout.cells[2])
                                  .arg(result.layout.requiredBytes)
                                  .arg(available);
        return result;
    }

    result.ok = true;
    return result;
}

DepthTsdfFrameLoadResult DepthTsdfSurfaceBuilder::loadFrames(
    const QVector<DepthFrameArtifact> &artifacts)
{
    DepthTsdfFrameLoadResult result;
    for (const DepthFrameArtifact &artifact : artifacts)
    {
        if ((!artifact.status.isEmpty() && artifact.status != QStringLiteral("completed")) ||
            !artifact.fusionEligible ||
            artifact.acceptance == QStringLiteral("rejected") ||
            artifact.acceptance == QStringLiteral("validation_only"))
        {
            continue;
        }
        if (!artifact.hasCameraModel || !artifact.cameraModel.isValid())
        {
            result.errorMessage = frameArtifactError(artifact, QStringLiteral("camera is invalid"));
            return result;
        }
        if (artifact.depthPath.isEmpty())
        {
            result.errorMessage = frameArtifactError(artifact, QStringLiteral("raw depth path is empty"));
            return result;
        }

        DepthTsdfFrame frame;
        frame.refIndex = artifact.refIndex;
        frame.refImage = artifact.refImage;
        frame.camera = artifact.cameraModel;
        frame.sourceIndices = artifact.sourceIndices;
        QString reason;
        if (!loadFloatMatrix(artifact.depthPath, &frame.depth, &reason))
        {
            result.errorMessage = frameArtifactError(artifact, QStringLiteral("depth: %1").arg(reason));
            return result;
        }

        if (artifact.confidencePath.isEmpty())
        {
            frame.confidence = cv::Mat(frame.depth.size(), CV_32FC1, cv::Scalar(1.0f));
        }
        else if (!loadFloatMatrix(artifact.confidencePath, &frame.confidence, &reason) ||
                 frame.confidence.size() != frame.depth.size())
        {
            if (reason.isEmpty())
            {
                reason = QStringLiteral("confidence dimensions do not match depth");
            }
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("confidence: %1").arg(reason));
            return result;
        }

        if (artifact.geometrySupportPath.isEmpty())
        {
            frame.geometrySupportCount = cv::Mat(
                frame.depth.size(), CV_16UC1, cv::Scalar(0));
        }
        else if (!loadUnsignedShortMatrix(
                     artifact.geometrySupportPath,
                     &frame.geometrySupportCount,
                     &reason) ||
                 frame.geometrySupportCount.size() != frame.depth.size())
        {
            if (reason.isEmpty())
            {
                reason = QStringLiteral("geometry support dimensions do not match depth");
            }
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("geometry support: %1").arg(reason));
            return result;
        }

        if (artifact.geometrySourceMaskPath.isEmpty())
        {
            frame.geometrySourceMask = cv::Mat(
                frame.depth.size(), CV_16UC1, cv::Scalar(0));
        }
        else if (!loadUnsignedShortMatrix(
                     artifact.geometrySourceMaskPath,
                     &frame.geometrySourceMask,
                     &reason) ||
                 frame.geometrySourceMask.size() != frame.depth.size())
        {
            if (reason.isEmpty())
            {
                reason = QStringLiteral("geometry source mask dimensions do not match depth");
            }
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("geometry source mask: %1").arg(reason));
            return result;
        }

        auto load_optional_float_evidence = [&](const QString &path,
                                                cv::Mat *destination,
                                                const QString &label)
        {
            if (path.isEmpty())
            {
                *destination = cv::Mat(frame.depth.size(), CV_32FC1, cv::Scalar(0.0f));
                return true;
            }
            if (!loadFloatMatrix(path, destination, &reason) ||
                destination->size() != frame.depth.size())
            {
                if (reason.isEmpty())
                {
                    reason = QStringLiteral("dimensions do not match depth");
                }
                result.errorMessage = frameArtifactError(
                    artifact, QStringLiteral("%1: %2").arg(label, reason));
                return false;
            }
            return true;
        };
        if (!load_optional_float_evidence(
                artifact.inverseDepthMeanPath,
                &frame.inverseDepthMean,
                QStringLiteral("inverse depth mean")) ||
            !load_optional_float_evidence(
                artifact.inverseDepthSpreadPath,
                &frame.inverseDepthRelativeSpread,
                QStringLiteral("inverse depth spread")))
        {
            return result;
        }

        if (artifact.crossViewRepairedMaskPath.isEmpty())
        {
            frame.crossViewRepairedMask = cv::Mat(
                frame.depth.size(), CV_8UC1, cv::Scalar(0));
        }
        else if (!loadMask(artifact.crossViewRepairedMaskPath,
                           frame.depth.size(),
                           &frame.crossViewRepairedMask,
                           &reason))
        {
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("cross-view repaired mask: %1").arg(reason));
            return result;
        }

        if (artifact.validMaskPath.isEmpty())
        {
            frame.depthValidMask = frame.depth > 0.0f;
        }
        else if (!loadMask(artifact.validMaskPath,
                           frame.depth.size(),
                           &frame.depthValidMask,
                           &reason))
        {
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("depth-valid mask: %1").arg(reason));
            return result;
        }

        if (artifact.supportMaskPath.isEmpty())
        {
            frame.supportMask = cv::Mat(frame.depth.size(), CV_8UC1, cv::Scalar(255));
        }
        else if (!loadMask(artifact.supportMaskPath,
                           frame.depth.size(),
                           &frame.supportMask,
                           &reason))
        {
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("support mask: %1").arg(reason));
            return result;
        }

        if (!artifact.refImage.isEmpty() && QFileInfo::exists(artifact.refImage))
        {
            frame.colorBgr = xjw::common::io::readImage(
                xjw::common::io::toUtf8Path(artifact.refImage), cv::IMREAD_COLOR);
            if (!frame.colorBgr.empty() && frame.colorBgr.size() != frame.depth.size())
            {
                cv::resize(frame.colorBgr,
                           frame.colorBgr,
                           frame.depth.size(),
                           0.0,
                           0.0,
                           cv::INTER_AREA);
            }
        }
        frame.frameQualityWeight = artifact.meanConfidence >= 0.0
            ? static_cast<float>(std::clamp(artifact.meanConfidence, 0.05, 1.0))
            : 1.0f;
        result.frames.push_back(std::move(frame));
    }

    if (result.frames.size() < 3)
    {
        result.errorMessage = QStringLiteral("TSDF requires at least 3 accepted depth frames; loaded=%1")
                                  .arg(result.frames.size());
        return result;
    }
    result.ok = true;
    return result;
}

DepthTsdfBoundsResult DepthTsdfSurfaceBuilder::estimateBounds(
    const QVector<DepthTsdfFrame> &frames)
{
    DepthTsdfBoundsResult result;
    if (frames.size() < 3)
    {
        result.errorMessage = QStringLiteral("TSDF bounds require at least 3 accepted depth frames");
        return result;
    }

    std::array<std::vector<float>, 3> coordinates;
    for (const DepthTsdfFrame &frame : frames)
    {
        if (!frame.camera.isValid() || frame.depth.type() != CV_32FC1 ||
            frame.depthValidMask.type() != CV_8UC1 ||
            frame.depthValidMask.size() != frame.depth.size() ||
            frame.supportMask.type() != CV_8UC1 ||
            frame.supportMask.size() != frame.depth.size())
        {
            continue;
        }
        const int stride = std::max(
            1,
            static_cast<int>(std::ceil(std::sqrt(
                static_cast<double>(frame.depth.total()) / 6000.0))));
        for (int row = 0; row < frame.depth.rows; row += stride)
        {
            for (int column = 0; column < frame.depth.cols; column += stride)
            {
                if (frame.depthValidMask.at<std::uint8_t>(row, column) == 0 ||
                    frame.supportMask.at<std::uint8_t>(row, column) == 0)
                {
                    continue;
                }
                const float depth = frame.depth.at<float>(row, column);
                if (!std::isfinite(depth) || depth <= 0.0f)
                {
                    continue;
                }
                const double pixel[2] = {column + 0.5, row + 0.5};
                double world[3] = {};
                if (!frame.camera.unprojectPixel(pixel, depth, world) ||
                    !std::isfinite(world[0]) || !std::isfinite(world[1]) ||
                    !std::isfinite(world[2]))
                {
                    continue;
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    coordinates[axis].push_back(static_cast<float>(world[axis]));
                }
            }
        }
    }

    result.sampleCount = coordinates[0].size();
    if (result.sampleCount < 500)
    {
        result.errorMessage = QStringLiteral("Insufficient finite TSDF bound samples: %1")
                                  .arg(result.sampleCount);
        return result;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        std::sort(coordinates[axis].begin(), coordinates[axis].end());
        const std::size_t last = coordinates[axis].size() - 1;
        const float low = coordinates[axis][static_cast<std::size_t>(last * 0.01)];
        const float high = coordinates[axis][static_cast<std::size_t>(last * 0.99)];
        const float padding = std::max((high - low) * 0.08f, 1.0e-5f);
        result.minimum[axis] = low - padding;
        result.maximum[axis] = high + padding;
    }
    result.ok = true;
    return result;
}

DepthTsdfObservationSample DepthTsdfSurfaceBuilder::sampleObservation(
    const DepthTsdfFrame &frame,
    const cv::Mat &effectiveDepthValidMask,
    const cv::Point2d &pixel,
    float minimumConfidence,
    bool discontinuityAware,
    float maximumRelativeDepthSpread,
    float maximumObservationInverseDepthSpread,
    bool allowInvalidNearestPixelRecovery,
    float maximumInvalidNearestPixelRecoveryInverseDepthSpread,
    bool enableCrossViewConsensusDepth,
    float maximumCrossViewConsensusInverseDepthSpread,
    const cv::Mat &crossViewConsensusMask)
{
    DepthTsdfObservationSample result;
    const int nearest_column = static_cast<int>(std::lround(pixel.x));
    const int nearest_row = static_cast<int>(std::lround(pixel.y));
    if (nearest_row < 0 || nearest_row >= frame.depth.rows ||
        nearest_column < 0 || nearest_column >= frame.depth.cols)
    {
        return result;
    }

    auto consensus_depth = [&](int row,
                               int column,
                               float raw_depth,
                               std::uint16_t geometry_support,
                               float inverse_depth_spread,
                               bool *used)
    {
        if (used)
        {
            *used = false;
        }
        if (!enableCrossViewConsensusDepth || geometry_support < 2 ||
            (!crossViewConsensusMask.empty() &&
             crossViewConsensusMask.at<std::uint8_t>(row, column) == 0) ||
            !std::isfinite(inverse_depth_spread) || inverse_depth_spread < 0.0f ||
            inverse_depth_spread > maximumCrossViewConsensusInverseDepthSpread ||
            frame.inverseDepthMean.type() != CV_32FC1 ||
            frame.inverseDepthMean.size() != frame.depth.size())
        {
            return raw_depth;
        }
        const float inverse_depth_mean = frame.inverseDepthMean.at<float>(row, column);
        if (!std::isfinite(inverse_depth_mean) || inverse_depth_mean <= 1.0e-12f)
        {
            return raw_depth;
        }
        const float depth = 1.0f / inverse_depth_mean;
        if (!std::isfinite(depth) || depth <= 0.0f)
        {
            return raw_depth;
        }
        if (used)
        {
            *used = true;
        }
        return depth;
    };

    auto classify_pixel = [&](int row, int column, DepthTsdfObservationSample *sample)
    {
        if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
        {
            sample->failure = DepthTsdfObservationFailure::SupportMask;
            return false;
        }
        if (effectiveDepthValidMask.at<std::uint8_t>(row, column) == 0)
        {
            sample->failure = DepthTsdfObservationFailure::DepthValid;
            return false;
        }
        const float depth = frame.depth.at<float>(row, column);
        if (!std::isfinite(depth) || depth <= 0.0f)
        {
            sample->failure = DepthTsdfObservationFailure::Depth;
            return false;
        }
        const float confidence = frame.confidence.at<float>(row, column);
        if (!std::isfinite(confidence) || confidence < minimumConfidence)
        {
            sample->failure = DepthTsdfObservationFailure::Confidence;
            return false;
        }
        sample->geometrySupportCount =
            frame.geometrySupportCount.at<std::uint16_t>(row, column);
        sample->geometrySourceMask = frame.geometrySourceMask.type() == CV_16UC1 &&
                frame.geometrySourceMask.size() == frame.depth.size()
            ? frame.geometrySourceMask.at<std::uint16_t>(row, column) : 0;
        sample->inverseDepthRelativeSpread =
            frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
                frame.inverseDepthRelativeSpread.size() == frame.depth.size()
            ? frame.inverseDepthRelativeSpread.at<float>(row, column) : 0.0f;
        if (maximumObservationInverseDepthSpread > 0.0f &&
            std::isfinite(sample->inverseDepthRelativeSpread) &&
            sample->inverseDepthRelativeSpread > maximumObservationInverseDepthSpread)
        {
            sample->valid = false;
            sample->failure = DepthTsdfObservationFailure::GeometryConsistency;
            return false;
        }
        sample->valid = true;
        sample->depth = consensus_depth(row,
                                        column,
                                        depth,
                                        sample->geometrySupportCount,
                                        sample->inverseDepthRelativeSpread,
                                        &sample->usedCrossViewConsensusDepth);
        sample->confidence = confidence;
        sample->contributingPixelCount = 1;
        sample->failure = DepthTsdfObservationFailure::None;
        return true;
    };

    if (!discontinuityAware)
    {
        classify_pixel(nearest_row, nearest_column, &result);
        return result;
    }

    struct Candidate
    {
        int row = 0;
        int column = 0;
        float depth = 0.0f;
        float confidence = 0.0f;
        float spatialWeight = 0.0f;
        std::uint16_t geometrySupportCount = 0;
        std::uint16_t geometrySourceMask = 0;
        float inverseDepthRelativeSpread = 0.0f;
        bool nearest = false;
        bool usedCrossViewConsensusDepth = false;
    };
    std::array<Candidate, 4> candidates{};
    int candidate_count = 0;
    bool passed_support = false;
    bool passed_depth_valid = false;
    bool passed_depth = false;
    bool passed_confidence = false;
    const int floor_column = static_cast<int>(std::floor(pixel.x));
    const int floor_row = static_cast<int>(std::floor(pixel.y));
    for (int delta_row = 0; delta_row <= 1; ++delta_row)
    {
        for (int delta_column = 0; delta_column <= 1; ++delta_column)
        {
            const int row = std::clamp(floor_row + delta_row, 0, frame.depth.rows - 1);
            const int column = std::clamp(
                floor_column + delta_column, 0, frame.depth.cols - 1);
            bool duplicate = false;
            for (int index = 0; index < candidate_count; ++index)
            {
                duplicate = duplicate ||
                    (candidates[index].row == row && candidates[index].column == column);
            }
            if (duplicate)
            {
                continue;
            }
            if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            passed_support = true;
            if (effectiveDepthValidMask.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            passed_depth_valid = true;
            const float depth = frame.depth.at<float>(row, column);
            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                continue;
            }
            passed_depth = true;
            const float confidence = frame.confidence.at<float>(row, column);
            if (!std::isfinite(confidence) || confidence < minimumConfidence)
            {
                continue;
            }
            passed_confidence = true;
            const float inverse_depth_relative_spread =
                frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
                    frame.inverseDepthRelativeSpread.size() == frame.depth.size()
                ? frame.inverseDepthRelativeSpread.at<float>(row, column) : 0.0f;
            if (maximumObservationInverseDepthSpread > 0.0f &&
                std::isfinite(inverse_depth_relative_spread) &&
                inverse_depth_relative_spread > maximumObservationInverseDepthSpread)
            {
                continue;
            }
            Candidate &candidate = candidates[candidate_count++];
            candidate.row = row;
            candidate.column = column;
            candidate.confidence = confidence;
            const float weight_x = std::max(
                0.05f, 1.0f - std::fabs(static_cast<float>(pixel.x) - column));
            const float weight_y = std::max(
                0.05f, 1.0f - std::fabs(static_cast<float>(pixel.y) - row));
            candidate.spatialWeight = weight_x * weight_y;
            candidate.geometrySupportCount =
                frame.geometrySupportCount.at<std::uint16_t>(row, column);
            candidate.geometrySourceMask =
                frame.geometrySourceMask.type() == CV_16UC1 &&
                    frame.geometrySourceMask.size() == frame.depth.size()
                ? frame.geometrySourceMask.at<std::uint16_t>(row, column) : 0;
            candidate.inverseDepthRelativeSpread = inverse_depth_relative_spread;
            candidate.depth = consensus_depth(row,
                                              column,
                                              depth,
                                              candidate.geometrySupportCount,
                                              candidate.inverseDepthRelativeSpread,
                                              &candidate.usedCrossViewConsensusDepth);
            candidate.nearest = row == nearest_row && column == nearest_column;
        }
    }

    if (candidate_count == 0)
    {
        result.failure = !passed_support
            ? DepthTsdfObservationFailure::SupportMask
            : (!passed_depth_valid
                   ? DepthTsdfObservationFailure::DepthValid
                   : (!passed_depth
                          ? DepthTsdfObservationFailure::Depth
                          : (!passed_confidence
                                 ? DepthTsdfObservationFailure::Confidence
                                 : DepthTsdfObservationFailure::GeometryConsistency)));
        return result;
    }

    const bool has_nearest_candidate = std::any_of(
        candidates.cbegin(),
        candidates.cbegin() + candidate_count,
        [](const Candidate &candidate)
        {
            return candidate.nearest;
        });
    if (!allowInvalidNearestPixelRecovery && !has_nearest_candidate)
    {
        result.failure = DepthTsdfObservationFailure::DepthValid;
        result.rejectedInvalidNearestPixelRecovery = true;
        return result;
    }

    int anchor_index = -1;
    float best_anchor_score = -1.0f;
    for (int index = 0; index < candidate_count; ++index)
    {
        if (candidates[index].nearest)
        {
            anchor_index = index;
            break;
        }
        const float score = candidates[index].spatialWeight * candidates[index].confidence;
        if (score > best_anchor_score)
        {
            best_anchor_score = score;
            anchor_index = index;
        }
    }

    const float anchor_depth = candidates[anchor_index].depth;
    const float relative_threshold = std::max(0.0f, maximumRelativeDepthSpread);
    float depth_weight_sum = 0.0f;
    float confidence_weight_sum = 0.0f;
    float spatial_weight_sum = 0.0f;
    bool first_evidence = true;
    for (int index = 0; index < candidate_count; ++index)
    {
        const Candidate &candidate = candidates[index];
        const float relative_error = std::fabs(candidate.depth - anchor_depth) /
                                     std::max(anchor_depth, 1.0e-6f);
        if (relative_error > relative_threshold)
        {
            ++result.discontinuityRejectedPixelCount;
            continue;
        }
        const float depth_weight = candidate.spatialWeight * candidate.confidence;
        result.depth += candidate.depth * depth_weight;
        depth_weight_sum += depth_weight;
        result.confidence += candidate.confidence * candidate.spatialWeight;
        spatial_weight_sum += candidate.spatialWeight;
        confidence_weight_sum += candidate.confidence;
        result.geometrySupportCount = std::max(
            result.geometrySupportCount, candidate.geometrySupportCount);
        result.geometrySourceMask = first_evidence
            ? candidate.geometrySourceMask
            : static_cast<std::uint16_t>(result.geometrySourceMask &
                                         candidate.geometrySourceMask);
        result.inverseDepthRelativeSpread = std::max(
            result.inverseDepthRelativeSpread,
            candidate.inverseDepthRelativeSpread);
        first_evidence = false;
        result.usedCrossViewConsensusDepth = result.usedCrossViewConsensusDepth ||
            candidate.usedCrossViewConsensusDepth;
        ++result.contributingPixelCount;
    }
    if (depth_weight_sum <= 0.0f || spatial_weight_sum <= 0.0f ||
        confidence_weight_sum <= 0.0f)
    {
        result.failure = DepthTsdfObservationFailure::Depth;
        return result;
    }

    result.depth /= depth_weight_sum;
    result.confidence /= spatial_weight_sum;
    result.valid = true;
    result.failure = DepthTsdfObservationFailure::None;
    result.recoveredFromInvalidNearestPixel = !has_nearest_candidate;
    if (result.recoveredFromInvalidNearestPixel &&
        maximumInvalidNearestPixelRecoveryInverseDepthSpread > 0.0f &&
        result.inverseDepthRelativeSpread >
            maximumInvalidNearestPixelRecoveryInverseDepthSpread)
    {
        result.valid = false;
        result.failure = DepthTsdfObservationFailure::GeometryConsistency;
        result.rejectedInvalidNearestPixelRecovery = true;
    }
    return result;
}

bool DepthTsdfSurfaceBuilder::isSampleSupported(
    float accumulatedWeight,
    int distinctSupportCount,
    float maximumObservationWeight,
    const DepthTsdfOptions &options,
    bool *singleView,
    bool *multiView,
    int maximumGeometrySupportCount,
    bool *geometryVerifiedSingleView)
{
    const bool multi_view_supported = distinctSupportCount >= std::max(
                                          2, options.minimumDistinctCameraSupport)
        && accumulatedWeight >= options.minimumVoxelWeight;
    const bool legacy_single_view_supported = options.minimumDistinctCameraSupport <= 1
        && distinctSupportCount == 1
        && maximumObservationWeight >= options.minimumSingleObservationWeight;
    const bool geometry_verified_single_view_supported =
        options.allowGeometryVerifiedSingleObservation &&
        distinctSupportCount == 1 &&
        maximumObservationWeight >= options.minimumGeometryVerifiedObservationWeight &&
        maximumGeometrySupportCount >= options.minimumGeometrySupportCount;
    const bool single_view_supported = legacy_single_view_supported ||
                                       geometry_verified_single_view_supported;
    if (singleView)
    {
        *singleView = single_view_supported;
    }
    if (multiView)
    {
        *multiView = multi_view_supported;
    }
    if (geometryVerifiedSingleView)
    {
        *geometryVerifiedSingleView = geometry_verified_single_view_supported;
    }
    return multi_view_supported || single_view_supported;
}

int DepthTsdfSurfaceBuilder::growGeometryVerifiedSingleViewSamples(
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<std::size_t> &candidateIndices,
    int minimumNeighborCount,
    int passes,
    float maximumTsdfDelta,
    std::vector<std::uint8_t> *supported)
{
    if (!supported || supported->size() != tsdf.size() ||
        supported->size() != static_cast<std::size_t>(layout.sampleCount) ||
        candidateIndices.empty())
    {
        return 0;
    }
    const int samples_x = layout.cells[0] + 1;
    const int samples_y = layout.cells[1] + 1;
    const int samples_z = layout.cells[2] + 1;
    if (samples_x <= 0 || samples_y <= 0 || samples_z <= 0)
    {
        return 0;
    }
    const std::size_t slice_size = static_cast<std::size_t>(samples_x) * samples_y;
    const int required_neighbors = std::clamp(minimumNeighborCount, 1, 26);
    const int maximum_passes = std::clamp(passes, 1, 6);
    const float maximum_delta = std::max(0.01f, maximumTsdfDelta);
    std::vector<std::size_t> pending = candidateIndices;
    std::vector<std::size_t> next_pending;
    std::vector<std::size_t> accepted;
    int accepted_count = 0;
    for (int pass = 0; pass < maximum_passes && !pending.empty(); ++pass)
    {
        accepted.clear();
        next_pending.clear();
        accepted.reserve(pending.size());
        next_pending.reserve(pending.size());
        for (const std::size_t index : pending)
        {
            if (index >= supported->size() || (*supported)[index])
            {
                continue;
            }
            const int z = static_cast<int>(index / slice_size);
            const std::size_t within_slice = index % slice_size;
            const int y = static_cast<int>(within_slice / samples_x);
            const int x = static_cast<int>(within_slice % samples_x);
            int neighbor_count = 0;
            for (int dz = -1; dz <= 1 && neighbor_count < required_neighbors; ++dz)
            {
                const int neighbor_z = z + dz;
                if (neighbor_z < 0 || neighbor_z >= samples_z)
                {
                    continue;
                }
                for (int dy = -1; dy <= 1 && neighbor_count < required_neighbors; ++dy)
                {
                    const int neighbor_y = y + dy;
                    if (neighbor_y < 0 || neighbor_y >= samples_y)
                    {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int neighbor_x = x + dx;
                        if ((dx == 0 && dy == 0 && dz == 0) ||
                            neighbor_x < 0 || neighbor_x >= samples_x)
                        {
                            continue;
                        }
                        const std::size_t neighbor_index = sampleIndex(
                            layout, neighbor_x, neighbor_y, neighbor_z);
                        if (!(*supported)[neighbor_index] ||
                            std::fabs(tsdf[neighbor_index] - tsdf[index]) > maximum_delta)
                        {
                            continue;
                        }
                        ++neighbor_count;
                        if (neighbor_count >= required_neighbors)
                        {
                            break;
                        }
                    }
                }
            }
            if (neighbor_count >= required_neighbors)
            {
                accepted.push_back(index);
            }
            else
            {
                next_pending.push_back(index);
            }
        }
        if (accepted.empty())
        {
            break;
        }
        for (const std::size_t index : accepted)
        {
            (*supported)[index] = 1;
        }
        accepted_count += static_cast<int>(accepted.size());
        pending.swap(next_pending);
    }
    return accepted_count;
}

DepthTsdfZeroCrossingStatistics DepthTsdfSurfaceBuilder::analyzeZeroCrossings(
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<std::uint8_t> &supported)
{
    DepthTsdfZeroCrossingStatistics statistics;
    const std::size_t expected_size = static_cast<std::size_t>(layout.sampleCount);
    if (!layout.ok || tsdf.size() != expected_size || weight.size() != expected_size ||
        supported.size() != expected_size)
    {
        return statistics;
    }

    unsigned long long observed_cell_count = 0;
    unsigned long long raw_candidate_cell_count = 0;
    unsigned long long extractable_cell_count = 0;
    unsigned long long suppressed_by_support_cell_count = 0;
    unsigned long long positive_only_supported_cell_count = 0;
    unsigned long long negative_only_supported_cell_count = 0;
    unsigned long long partially_supported_cell_count = 0;
    unsigned long long fully_unsupported_observed_cell_count = 0;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) \
    reduction(+:observed_cell_count,raw_candidate_cell_count,extractable_cell_count) \
    reduction(+:suppressed_by_support_cell_count,positive_only_supported_cell_count) \
    reduction(+:negative_only_supported_cell_count,partially_supported_cell_count) \
    reduction(+:fully_unsupported_observed_cell_count)
#endif
    for (int z = 0; z < layout.cells[2]; ++z)
    {
        for (int y = 0; y < layout.cells[1]; ++y)
        {
            for (int x = 0; x < layout.cells[0]; ++x)
            {
                int observed_count = 0;
                int supported_count = 0;
                bool observed_positive = false;
                bool observed_negative = false;
                bool supported_positive = false;
                bool supported_negative = false;
                for (int dz = 0; dz <= 1; ++dz)
                {
                    for (int dy = 0; dy <= 1; ++dy)
                    {
                        for (int dx = 0; dx <= 1; ++dx)
                        {
                            const std::size_t index = sampleIndex(
                                layout, x + dx, y + dy, z + dz);
                            if (weight[index] > 0.0f)
                            {
                                ++observed_count;
                                observed_positive = observed_positive || tsdf[index] >= 0.0f;
                                observed_negative = observed_negative || tsdf[index] < 0.0f;
                            }
                            if (supported[index] != 0)
                            {
                                ++supported_count;
                                supported_positive = supported_positive || tsdf[index] >= 0.0f;
                                supported_negative = supported_negative || tsdf[index] < 0.0f;
                            }
                        }
                    }
                }
                if (observed_count == 0)
                {
                    continue;
                }
                ++observed_cell_count;
                const bool raw_candidate = observed_positive && observed_negative;
                const bool extractable = supported_positive && supported_negative;
                raw_candidate_cell_count += raw_candidate;
                extractable_cell_count += extractable;
                suppressed_by_support_cell_count += raw_candidate && !extractable;
                positive_only_supported_cell_count +=
                    supported_positive && !supported_negative;
                negative_only_supported_cell_count +=
                    supported_negative && !supported_positive;
                partially_supported_cell_count +=
                    supported_count > 0 && supported_count < 8;
                fully_unsupported_observed_cell_count += supported_count == 0;
            }
        }
    }
    statistics.observedCellCount = observed_cell_count;
    statistics.rawCandidateCellCount = raw_candidate_cell_count;
    statistics.extractableCellCount = extractable_cell_count;
    statistics.suppressedBySupportCellCount = suppressed_by_support_cell_count;
    statistics.positiveOnlySupportedCellCount = positive_only_supported_cell_count;
    statistics.negativeOnlySupportedCellCount = negative_only_supported_cell_count;
    statistics.partiallySupportedCellCount = partially_supported_cell_count;
    statistics.fullyUnsupportedObservedCellCount =
        fully_unsupported_observed_cell_count;
    return statistics;
}

DepthTsdfZeroCrossingRecoveryStatistics
DepthTsdfSurfaceBuilder::recoverGeometryVerifiedZeroCrossingSamples(
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint8_t> &eligible,
    int minimumSupportedCorners,
    int minimumCellVotes,
    std::vector<std::uint8_t> *supported)
{
    DepthTsdfZeroCrossingRecoveryStatistics statistics;
    const std::size_t expected_size =
        static_cast<std::size_t>(layout.sampleCount);
    if (!supported || !layout.ok ||
        tsdf.size() != expected_size ||
        weight.size() != expected_size ||
        geometrySourceMask.size() != expected_size ||
        eligible.size() != expected_size ||
        supported->size() != expected_size)
    {
        return statistics;
    }

    const int required_supported_corners =
        std::clamp(minimumSupportedCorners, 1, 7);
    const int required_votes = std::clamp(minimumCellVotes, 1, 8);
    const std::vector<std::uint8_t> core_supported = *supported;
    std::vector<std::uint8_t> votes(expected_size, 0);
    for (int z = 0; z < layout.cells[2]; ++z)
    {
        for (int y = 0; y < layout.cells[1]; ++y)
        {
            for (int x = 0; x < layout.cells[0]; ++x)
            {
                std::array<std::size_t, 8> corners{};
                int corner_count = 0;
                int supported_count = 0;
                bool observed_positive = false;
                bool observed_negative = false;
                bool supported_positive = false;
                bool supported_negative = false;
                for (int dz = 0; dz <= 1; ++dz)
                {
                    for (int dy = 0; dy <= 1; ++dy)
                    {
                        for (int dx = 0; dx <= 1; ++dx)
                        {
                            const std::size_t index = sampleIndex(
                                layout, x + dx, y + dy, z + dz);
                            corners[static_cast<std::size_t>(corner_count++)] =
                                index;
                            if (weight[index] > 0.0f)
                            {
                                observed_positive =
                                    observed_positive || tsdf[index] >= 0.0f;
                                observed_negative =
                                    observed_negative || tsdf[index] < 0.0f;
                            }
                            if (core_supported[index] != 0)
                            {
                                ++supported_count;
                                supported_positive =
                                    supported_positive || tsdf[index] >= 0.0f;
                                supported_negative =
                                    supported_negative || tsdf[index] < 0.0f;
                            }
                        }
                    }
                }
                if (!observed_positive || !observed_negative ||
                    supported_count < required_supported_corners ||
                    supported_positive == supported_negative)
                {
                    continue;
                }
                const bool missing_negative = supported_positive;
                for (const std::size_t candidate : corners)
                {
                    if (core_supported[candidate] != 0 ||
                        eligible[candidate] == 0 ||
                        weight[candidate] <= 0.0f ||
                        ((tsdf[candidate] < 0.0f) != missing_negative))
                    {
                        continue;
                    }
                    bool shares_source = false;
                    for (const std::size_t neighbor : corners)
                    {
                        if (core_supported[neighbor] != 0 &&
                            (geometrySourceMask[candidate] &
                             geometrySourceMask[neighbor]) != 0)
                        {
                            shares_source = true;
                            break;
                        }
                    }
                    bool connected_to_surface = true;
                    if (supported_count == 1)
                    {
                        const int samples_x = layout.cells[0] + 1;
                        const int samples_y = layout.cells[1] + 1;
                        const std::size_t samples_per_slice =
                            static_cast<std::size_t>(samples_x) *
                            static_cast<std::size_t>(samples_y);
                        const int candidate_z = static_cast<int>(
                            candidate / samples_per_slice);
                        const std::size_t slice_offset =
                            candidate % samples_per_slice;
                        const int candidate_y = static_cast<int>(
                            slice_offset / static_cast<std::size_t>(samples_x));
                        const int candidate_x = static_cast<int>(
                            slice_offset % static_cast<std::size_t>(samples_x));
                        int source_connected_neighbor_count = 0;
                        bool has_candidate_sign_neighbor = false;
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            const int neighbor_z = candidate_z + dz;
                            if (neighbor_z < 0 || neighbor_z > layout.cells[2])
                            {
                                continue;
                            }
                            for (int dy = -1; dy <= 1; ++dy)
                            {
                                const int neighbor_y = candidate_y + dy;
                                if (neighbor_y < 0 || neighbor_y > layout.cells[1])
                                {
                                    continue;
                                }
                                for (int dx = -1; dx <= 1; ++dx)
                                {
                                    const int neighbor_x = candidate_x + dx;
                                    if ((dx == 0 && dy == 0 && dz == 0) ||
                                        neighbor_x < 0 ||
                                        neighbor_x > layout.cells[0])
                                    {
                                        continue;
                                    }
                                    const std::size_t neighbor = sampleIndex(
                                        layout,
                                        neighbor_x,
                                        neighbor_y,
                                        neighbor_z);
                                    if (core_supported[neighbor] == 0 ||
                                        (geometrySourceMask[candidate] &
                                         geometrySourceMask[neighbor]) == 0)
                                    {
                                        continue;
                                    }
                                    ++source_connected_neighbor_count;
                                    has_candidate_sign_neighbor =
                                        has_candidate_sign_neighbor ||
                                        ((tsdf[neighbor] < 0.0f) ==
                                         missing_negative);
                                }
                            }
                        }
                        connected_to_surface =
                            source_connected_neighbor_count >= 2 &&
                            has_candidate_sign_neighbor;
                    }
                    if (shares_source && connected_to_surface)
                    {
                        votes[candidate] = static_cast<std::uint8_t>(
                            std::min(255, static_cast<int>(votes[candidate]) + 1));
                    }
                }
            }
        }
    }
    for (std::size_t index = 0; index < expected_size; ++index)
    {
        if (votes[index] == 0)
        {
            continue;
        }
        ++statistics.candidateSampleCount;
        if (votes[index] >= required_votes)
        {
            (*supported)[index] = 1;
            ++statistics.recoveredSampleCount;
        }
    }
    return statistics;
}

QVector<float> DepthTsdfSurfaceBuilder::robustFrameQualityWeights(
    const QVector<float> &rawWeights,
    float minimumMultiplier,
    float madFloor,
    float penaltyOnset,
    float penaltyStrength,
    float *median,
    float *scale)
{
    QVector<float> result;
    result.reserve(rawWeights.size());
    if (rawWeights.isEmpty())
    {
        if (median)
        {
            *median = 0.0f;
        }
        if (scale)
        {
            *scale = 0.0f;
        }
        return result;
    }

    QVector<float> sorted_weights;
    sorted_weights.reserve(rawWeights.size());
    for (const float weight : rawWeights)
    {
        sorted_weights.push_back(std::clamp(
            std::isfinite(weight) ? weight : 0.0f, 0.0f, 1.0f));
    }
    std::sort(sorted_weights.begin(), sorted_weights.end());
    const auto medianOfSorted = [](const QVector<float> &values)
    {
        const int middle = values.size() / 2;
        return values.size() % 2 == 0
            ? 0.5f * (values[middle - 1] + values[middle])
            : values[middle];
    };
    const float robust_median = medianOfSorted(sorted_weights);
    QVector<float> deviations;
    deviations.reserve(sorted_weights.size());
    for (const float weight : sorted_weights)
    {
        deviations.push_back(std::fabs(weight - robust_median));
    }
    std::sort(deviations.begin(), deviations.end());
    const float robust_scale = std::max(
        std::max(0.0f, madFloor),
        1.4826f * medianOfSorted(deviations));
    const float bounded_minimum_multiplier =
        std::clamp(minimumMultiplier, 0.0f, 1.0f);
    const float bounded_onset = std::max(0.0f, penaltyOnset);
    const float bounded_strength = std::max(0.0f, penaltyStrength);
    for (const float raw_weight : rawWeights)
    {
        const float weight = std::clamp(
            std::isfinite(raw_weight) ? raw_weight : 0.0f, 0.0f, 1.0f);
        const float low_tail_distance = robust_scale > 0.0f
            ? std::max(
                0.0f,
                (robust_median - weight) / robust_scale - bounded_onset)
            : 0.0f;
        const float multiplier = std::max(
            bounded_minimum_multiplier,
            std::exp(
                -bounded_strength * low_tail_distance * low_tail_distance));
        result.push_back(weight * multiplier);
    }
    if (median)
    {
        *median = robust_median;
    }
    if (scale)
    {
        *scale = robust_scale;
    }
    return result;
}

DepthTsdfResult DepthTsdfSurfaceBuilder::build(const QVector<DepthTsdfFrame> &frames,
                                               const DepthTsdfOptions &options)
{
    DepthTsdfResult result;
    const bool requested_robust_frame_quality_weighting =
        options.enableRobustFrameQualityWeighting;
    const float requested_robust_frame_quality_minimum_multiplier =
        options.robustFrameQualityMinimumMultiplier;
    const float requested_robust_frame_quality_mad_floor =
        options.robustFrameQualityMadFloor;
    const float requested_robust_frame_quality_penalty_onset =
        options.robustFrameQualityPenaltyOnset;
    const float requested_robust_frame_quality_penalty_strength =
        options.robustFrameQualityPenaltyStrength;
    result.statistics.inputFrameCount = frames.size();
    if (frames.size() < options.minimumInputFrames)
    {
        result.errorMessage = QStringLiteral("TSDF requires at least %1 input frames; received=%2")
                                  .arg(options.minimumInputFrames)
                                  .arg(frames.size());
        return result;
    }
    for (const DepthTsdfFrame &frame : frames)
    {
        if (!frame.camera.isValid() || frame.depth.type() != CV_32FC1 ||
            frame.confidence.type() != CV_32FC1 ||
            frame.geometrySupportCount.type() != CV_16UC1 ||
            frame.depthValidMask.type() != CV_8UC1 ||
            frame.supportMask.type() != CV_8UC1 ||
            frame.confidence.size() != frame.depth.size() ||
            frame.geometrySupportCount.size() != frame.depth.size() ||
            (!frame.geometrySourceMask.empty() &&
             (frame.geometrySourceMask.type() != CV_16UC1 ||
              frame.geometrySourceMask.size() != frame.depth.size())) ||
            (!frame.inverseDepthRelativeSpread.empty() &&
             (frame.inverseDepthRelativeSpread.type() != CV_32FC1 ||
              frame.inverseDepthRelativeSpread.size() != frame.depth.size())) ||
            frame.depthValidMask.size() != frame.depth.size() ||
            frame.supportMask.size() != frame.depth.size())
        {
            result.errorMessage = QStringLiteral("TSDF input frame %1 has invalid camera or matrix layout")
                                      .arg(frame.refIndex);
            return result;
        }
    }
    result.statistics.acceptedFrameCount = frames.size();
    QVector<float> raw_frame_quality_weights;
    raw_frame_quality_weights.reserve(frames.size());
    for (const DepthTsdfFrame &frame : frames)
    {
        raw_frame_quality_weights.push_back(std::clamp(
            frame.frameQualityWeight, 0.0f, 1.0f));
    }
    QVector<float> effective_frame_quality_weights =
        raw_frame_quality_weights;
    const bool robust_frame_quality_weighting_enabled =
        requested_robust_frame_quality_weighting;
    float robust_frame_quality_median = 0.0f;
    float robust_frame_quality_scale = 0.0f;
    float robust_frame_quality_minimum_effective_weight = 1.0f;
    int robust_frame_quality_downweighted_frame_count = 0;
    result.statistics.effectiveRobustFrameQualityWeighting =
        robust_frame_quality_weighting_enabled;
    if (robust_frame_quality_weighting_enabled)
    {
        effective_frame_quality_weights = robustFrameQualityWeights(
            raw_frame_quality_weights,
            requested_robust_frame_quality_minimum_multiplier,
            requested_robust_frame_quality_mad_floor,
            requested_robust_frame_quality_penalty_onset,
            requested_robust_frame_quality_penalty_strength,
            &robust_frame_quality_median,
            &robust_frame_quality_scale);
    }
    for (int frame_index = 0;
         frame_index < effective_frame_quality_weights.size();
         ++frame_index)
    {
        robust_frame_quality_minimum_effective_weight = std::min(
            robust_frame_quality_minimum_effective_weight,
            effective_frame_quality_weights[frame_index]);
        if (effective_frame_quality_weights[frame_index] + 1.0e-6f <
            raw_frame_quality_weights[frame_index])
        {
            ++robust_frame_quality_downweighted_frame_count;
        }
    }
    result.statistics.robustFrameQualityDownweightedFrameCount =
        robust_frame_quality_downweighted_frame_count;
    result.statistics.robustFrameQualityMedian =
        robust_frame_quality_median;
    result.statistics.robustFrameQualityScale =
        robust_frame_quality_scale;
    result.statistics.robustFrameQualityMinimumEffectiveWeight =
        robust_frame_quality_minimum_effective_weight;

    QVector<cv::Mat> effective_depth_valid_masks;
    effective_depth_valid_masks.reserve(frames.size());
    const int erosion_pixels = std::clamp(
        options.depthValidBoundaryErosionPixels, 0, 4);
    std::uint64_t boundary_recovered_depth_valid_pixel_count = 0;
    if (erosion_pixels > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(erosion_pixels * 2 + 1, erosion_pixels * 2 + 1));
        for (const DepthTsdfFrame &frame : frames)
        {
            cv::Mat eroded;
            cv::erode(frame.depthValidMask, eroded, kernel);
            if (options.enableGeometryVerifiedBoundaryRecovery && erosion_pixels > 1 &&
                !frame.geometrySupportCount.empty() &&
                !frame.inverseDepthRelativeSpread.empty())
            {
                const int recovery_erosion_pixels = erosion_pixels - 1;
                const cv::Mat recovery_kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(recovery_erosion_pixels * 2 + 1,
                             recovery_erosion_pixels * 2 + 1));
                cv::Mat less_eroded;
                cv::erode(frame.depthValidMask, less_eroded, recovery_kernel);
                cv::Mat recovery_ring;
                cv::subtract(less_eroded, eroded, recovery_ring);
                cv::Mat geometry_supported;
                cv::compare(
                    frame.geometrySupportCount,
                    options.minimumBoundaryRecoveryGeometrySupport,
                    geometry_supported,
                    cv::CMP_GE);
                cv::Mat spread_supported;
                cv::compare(
                    frame.inverseDepthRelativeSpread,
                    options.maximumBoundaryRecoveryInverseDepthSpread,
                    spread_supported,
                    cv::CMP_LE);
                cv::bitwise_and(recovery_ring, geometry_supported, recovery_ring);
                cv::bitwise_and(recovery_ring, spread_supported, recovery_ring);
                cv::bitwise_and(recovery_ring, frame.supportMask, recovery_ring);
                boundary_recovered_depth_valid_pixel_count +=
                    static_cast<std::uint64_t>(cv::countNonZero(recovery_ring));
                cv::bitwise_or(eroded, recovery_ring, eroded);
            }
            effective_depth_valid_masks.push_back(std::move(eroded));
        }
    }

    QVector<cv::Mat> contour_band_masks;
    std::uint64_t cross_view_consensus_contour_band_pixel_count = 0;
    if ((options.enableCrossViewConsensusDepth &&
         options.crossViewConsensusContourBandOnly) ||
        options.enableContourBandZeroCrossingSupport)
    {
        contour_band_masks.reserve(frames.size());
        const cv::Mat contour_kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame &frame = frames[frame_index];
            const cv::Mat &valid_mask = erosion_pixels > 0
                ? effective_depth_valid_masks[frame_index]
                : frame.depthValidMask;
            cv::Mat inner_mask;
            cv::erode(valid_mask, inner_mask, contour_kernel);
            cv::Mat contour_band;
            cv::subtract(valid_mask, inner_mask, contour_band);
            cv::Mat geometry_supported;
            cv::compare(
                frame.geometrySupportCount,
                options.minimumBoundaryRecoveryGeometrySupport,
                geometry_supported,
                cv::CMP_GE);
            cv::Mat spread_supported(
                frame.depth.size(), CV_8UC1, cv::Scalar(0));
            if (!frame.inverseDepthRelativeSpread.empty())
            {
                cv::compare(
                    frame.inverseDepthRelativeSpread,
                    std::min(options.maximumCrossViewConsensusInverseDepthSpread,
                             options.maximumBoundaryRecoveryInverseDepthSpread),
                    spread_supported,
                    cv::CMP_LE);
            }
            cv::bitwise_and(contour_band, frame.supportMask, contour_band);
            cv::bitwise_and(contour_band, geometry_supported, contour_band);
            cv::bitwise_and(contour_band, spread_supported, contour_band);
            if (frame.geometrySourceMask.empty())
            {
                contour_band.setTo(cv::Scalar(0));
            }
            else
            {
                for (int row = 0; row < contour_band.rows; ++row)
                {
                    std::uint8_t *band_row = contour_band.ptr<std::uint8_t>(row);
                    const std::uint16_t *source_row =
                        frame.geometrySourceMask.ptr<std::uint16_t>(row);
                    for (int column = 0; column < contour_band.cols; ++column)
                    {
                        if (band_row[column] != 0 && bitCount(source_row[column]) < 2)
                        {
                            band_row[column] = 0;
                        }
                    }
                }
            }
            cross_view_consensus_contour_band_pixel_count +=
                static_cast<std::uint64_t>(cv::countNonZero(contour_band));
            contour_band_masks.push_back(std::move(contour_band));
        }
    }

    const DepthTsdfBoundsResult bounds = estimateBounds(frames);
    if (!bounds.ok)
    {
        result.errorMessage = bounds.errorMessage;
        return result;
    }
    result = validateAllocation(bounds.minimum, bounds.maximum, options);
    result.statistics.inputFrameCount = frames.size();
    result.statistics.acceptedFrameCount = frames.size();
    result.statistics.effectiveRobustFrameQualityWeighting =
        robust_frame_quality_weighting_enabled;
    result.statistics.robustFrameQualityDownweightedFrameCount =
        robust_frame_quality_downweighted_frame_count;
    result.statistics.robustFrameQualityMedian =
        robust_frame_quality_median;
    result.statistics.robustFrameQualityScale =
        robust_frame_quality_scale;
    result.statistics.robustFrameQualityMinimumEffectiveWeight =
        robust_frame_quality_minimum_effective_weight;
    if (!result.ok)
    {
        return result;
    }
    result.ok = false;
    if (options.progress)
    {
        options.progress(QStringLiteral("正在融合置信度加权 TSDF..."), 5);
    }

    std::vector<float> tsdf;
    std::vector<float> weight;
    std::vector<float> maximumObservationWeight;
    std::vector<std::uint16_t> maximumGeometrySupportCount;
    std::vector<std::uint16_t> support;
    std::vector<std::uint16_t> geometrySourceMask;
    std::vector<std::uint16_t> minimumInverseDepthSpread;
    std::vector<float> surfaceTsdfWeightedSum;
    std::vector<float> surfaceObservationWeight;
    std::vector<float> contourBandObservationWeight;
    try
    {
        tsdf.assign(static_cast<std::size_t>(result.layout.sampleCount), 1.0f);
        weight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumObservationWeight.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumGeometrySupportCount.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0);
        support.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
        if (options.enableSurfacePatchSupport ||
            options.enableContourBandZeroCrossingSupport)
        {
            geometrySourceMask.assign(
                static_cast<std::size_t>(result.layout.sampleCount), 0);
            minimumInverseDepthSpread.assign(
                static_cast<std::size_t>(result.layout.sampleCount),
                std::numeric_limits<std::uint16_t>::max());
            surfaceTsdfWeightedSum.assign(
                static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            surfaceObservationWeight.assign(
                static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            if (options.enableContourBandZeroCrossingSupport)
            {
                contourBandObservationWeight.assign(
                    static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            }
        }
    }
    catch (const std::bad_alloc &)
    {
        result.errorMessage = QStringLiteral(
            "TSDF allocation failed: resolution=%1 required=%2 bytes")
                                  .arg(options.resolution)
                                  .arg(result.layout.requiredBytes);
        return result;
    }

    const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                               result.layout.voxelSize[1],
                                               result.layout.voxelSize[2]});
    const float effective_truncation_voxels = std::max(1.0f, options.truncationVoxels);
    const float effective_surface_support_band_voxels =
        options.surfaceSupportBandVoxels > 0.0f
        ? std::clamp(options.surfaceSupportBandVoxels, 0.5f, effective_truncation_voxels)
        : effective_truncation_voxels;
    const float truncation = maximum_voxel_size * effective_truncation_voxels;
    const float surface_support_distance =
        maximum_voxel_size * effective_surface_support_band_voxels;
    const float maximum_free_space_distance =
        options.maximumFreeSpaceVoxels > 0.0f
        ? std::max({result.layout.voxelSize[0],
                    result.layout.voxelSize[1],
                    result.layout.voxelSize[2]}) *
              std::max(options.truncationVoxels, options.maximumFreeSpaceVoxels)
        : std::numeric_limits<float>::infinity();
    unsigned long long integratedVoxelUpdates = 0;
    unsigned long long rejectedProjectionCount = 0;
    unsigned long long rejectedSupportMaskCount = 0;
    unsigned long long supportMaskFreeSpaceUpdateCount = 0;
    unsigned long long rejectedDepthValidCount = 0;
    unsigned long long rejectedDepthCount = 0;
    unsigned long long rejectedConfidenceCount = 0;
    unsigned long long subpixelObservationCount = 0;
    unsigned long long recoveredNeighborObservationCount = 0;
    unsigned long long discontinuityRejectedCandidateCount = 0;
    unsigned long long rejectedGeometryConsistencyCount = 0;
    unsigned long long rejectedInvalidNearestPixelRecoveryCount = 0;
    unsigned long long crossViewConsensusDepthObservationCount = 0;
    std::atomic_bool cancelled{false};
    std::atomic<int> completed_z_slices{0};
    std::atomic<int> last_progress_percent{5};
    const int zSamples = result.layout.cells[2] + 1;
#ifdef MESHING_OPENMP
    const int workerCount = options.workerCount > 0 ? options.workerCount : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(workerCount) \
    reduction(+:integratedVoxelUpdates,rejectedProjectionCount,rejectedSupportMaskCount,supportMaskFreeSpaceUpdateCount,rejectedDepthValidCount,rejectedDepthCount,rejectedConfidenceCount,subpixelObservationCount,recoveredNeighborObservationCount,discontinuityRejectedCandidateCount,rejectedGeometryConsistencyCount,rejectedInvalidNearestPixelRecoveryCount,crossViewConsensusDepthObservationCount)
#endif
    for (int z = 0; z < zSamples; ++z)
    {
        if (cancelled.load(std::memory_order_relaxed) ||
            (options.isCancelled && options.isCancelled()))
        {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
        }
        const double worldZ = result.layout.boundsMin[2] +
                              result.layout.voxelSize[2] * static_cast<float>(z);
        for (int y = 0; y <= result.layout.cells[1]; ++y)
        {
            const double worldY = result.layout.boundsMin[1] +
                                  result.layout.voxelSize[1] * static_cast<float>(y);
            for (int x = 0; x <= result.layout.cells[0]; ++x)
            {
                const double world[3] = {
                    result.layout.boundsMin[0] +
                        result.layout.voxelSize[0] * static_cast<float>(x),
                    worldY,
                    worldZ
                };
                const std::size_t index = sampleIndex(result.layout, x, y, z);
                int support_mask_free_space_votes = 0;
                for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
                {
                    const DepthTsdfFrame &frame = frames[frame_index];
                    double pixel[2]{};
                    double voxelDepth = 0.0;
                    if (!frame.camera.projectWorldPointWithDepth(world, pixel, voxelDepth))
                    {
                        ++rejectedProjectionCount;
                        continue;
                    }
                    const cv::Mat &depth_valid_mask = erosion_pixels > 0
                        ? effective_depth_valid_masks[frame_index]
                        : frame.depthValidMask;
                    const DepthTsdfObservationSample observation = sampleObservation(
                        frame,
                        depth_valid_mask,
                        cv::Point2d(pixel[0], pixel[1]),
                        options.minimumConfidence,
                        options.enableDiscontinuityAwareSampling,
                        options.maximumInterpolationRelativeDepthSpread,
                        options.maximumObservationInverseDepthSpread,
                        options.allowInvalidNearestPixelRecovery,
                        options.maximumInvalidNearestPixelRecoveryInverseDepthSpread,
                        options.enableCrossViewConsensusDepth,
                        options.maximumCrossViewConsensusInverseDepthSpread,
                        options.crossViewConsensusContourBandOnly
                            ? contour_band_masks[frame_index]
                            : cv::Mat());
                    if (!observation.valid)
                    {
                        switch (observation.failure)
                        {
                        case DepthTsdfObservationFailure::SupportMask:
                            if (options.enableSupportMaskFreeSpaceCarving)
                            {
                                ++support_mask_free_space_votes;
                            }
                            ++rejectedSupportMaskCount;
                            break;
                        case DepthTsdfObservationFailure::DepthValid:
                            ++rejectedDepthValidCount;
                            break;
                        case DepthTsdfObservationFailure::Depth:
                            ++rejectedDepthCount;
                            break;
                        case DepthTsdfObservationFailure::Confidence:
                            ++rejectedConfidenceCount;
                            break;
                        case DepthTsdfObservationFailure::GeometryConsistency:
                            ++rejectedGeometryConsistencyCount;
                            break;
                        case DepthTsdfObservationFailure::Projection:
                        case DepthTsdfObservationFailure::None:
                        default:
                            ++rejectedProjectionCount;
                            break;
                        }
                        rejectedInvalidNearestPixelRecoveryCount +=
                            observation.rejectedInvalidNearestPixelRecovery;
                        continue;
                    }
                    subpixelObservationCount += observation.contributingPixelCount > 1;
                    recoveredNeighborObservationCount +=
                        observation.recoveredFromInvalidNearestPixel;
                    discontinuityRejectedCandidateCount +=
                        observation.discontinuityRejectedPixelCount;
                    crossViewConsensusDepthObservationCount +=
                        observation.usedCrossViewConsensusDepth;
                    bool contour_band_evidence = false;
                    if (options.enableContourBandZeroCrossingSupport &&
                        !contour_band_masks.empty())
                    {
                        const int contour_column = static_cast<int>(std::lround(pixel[0]));
                        const int contour_row = static_cast<int>(std::lround(pixel[1]));
                        contour_band_evidence = contour_row >= 0 &&
                            contour_row < contour_band_masks[frame_index].rows &&
                            contour_column >= 0 &&
                            contour_column < contour_band_masks[frame_index].cols &&
                            contour_band_masks[frame_index].at<std::uint8_t>(
                                contour_row, contour_column) != 0;
                    }
                    const float observedDepth = observation.depth;
                    const float confidence = observation.confidence;
                    const float signedDistance =
                        observedDepth - static_cast<float>(voxelDepth);
                    if (signedDistance < -truncation)
                    {
                        continue;
                    }
                    if (signedDistance > maximum_free_space_distance)
                    {
                        continue;
                    }
                    const float normalized = std::clamp(
                        signedDistance / truncation, -1.0f, 1.0f);
                    const float observationWeight =
                        confidence * effective_frame_quality_weights[frame_index];
                    integrateWeighted(&tsdf[index],
                                      &weight[index],
                                      normalized,
                                      observationWeight);
                    maximumObservationWeight[index] = std::max(
                        maximumObservationWeight[index], observationWeight);
                    ++integratedVoxelUpdates;
                    if (std::fabs(signedDistance) <= surface_support_distance)
                    {
                        maximumGeometrySupportCount[index] = std::max(
                            maximumGeometrySupportCount[index],
                            observation.geometrySupportCount);
                        if (options.enableSurfacePatchSupport ||
                            options.enableContourBandZeroCrossingSupport)
                        {
                            geometrySourceMask[index] = static_cast<std::uint16_t>(
                                geometrySourceMask[index] |
                                globalGeometrySourceMask(
                                    frame, observation.geometrySourceMask));
                            if (observation.geometrySourceMask != 0 &&
                                std::isfinite(observation.inverseDepthRelativeSpread) &&
                                observation.inverseDepthRelativeSpread >= 0.0f)
                            {
                                const int quantized_spread = static_cast<int>(std::lround(
                                    observation.inverseDepthRelativeSpread * 100000.0f));
                                minimumInverseDepthSpread[index] = std::min(
                                    minimumInverseDepthSpread[index],
                                    static_cast<std::uint16_t>(std::clamp(
                                        quantized_spread, 0, 65535)));
                            }
                            if (bitCount(observation.geometrySourceMask) >=
                                    std::max(2, options.minimumSurfacePatchSourceCount) &&
                                std::isfinite(observation.inverseDepthRelativeSpread) &&
                                observation.inverseDepthRelativeSpread <=
                                    options.maximumSurfacePatchInverseDepthSpread)
                            {
                                surfaceTsdfWeightedSum[index] +=
                                    normalized * observationWeight;
                                surfaceObservationWeight[index] += observationWeight;
                            }
                            if (contour_band_evidence &&
                                !contourBandObservationWeight.empty())
                            {
                                contourBandObservationWeight[index] += observationWeight;
                            }
                        }
                        support[index] = static_cast<std::uint16_t>(std::min<int>(
                            std::numeric_limits<std::uint16_t>::max(),
                            static_cast<int>(support[index]) + 1));
                    }
                }
                const int minimum_free_space_views = std::clamp(
                    options.minimumSupportMaskFreeSpaceViews, 1, 16);
                if (options.enableSupportMaskFreeSpaceCarving &&
                    support_mask_free_space_votes >= minimum_free_space_views)
                {
                    integrateWeighted(
                        &tsdf[index],
                        &weight[index],
                        1.0f,
                        std::max(0.0f, options.supportMaskFreeSpaceWeight) *
                            support_mask_free_space_votes);
                    ++integratedVoxelUpdates;
                    supportMaskFreeSpaceUpdateCount += support_mask_free_space_votes;
                }
            }
        }
        const int completed = completed_z_slices.fetch_add(1, std::memory_order_relaxed) + 1;
        const int progress_percent = 5 + completed * 65 / std::max(1, zSamples);
        int previous_progress = last_progress_percent.load(std::memory_order_relaxed);
        while (options.progress && progress_percent >= previous_progress + 5 &&
               !last_progress_percent.compare_exchange_weak(
                   previous_progress,
                   progress_percent,
                   std::memory_order_relaxed))
        {
        }
        if (options.progress && progress_percent >= previous_progress + 5)
        {
            options.progress(QStringLiteral("正在融合置信度加权 TSDF..."), progress_percent);
        }
    }

    if (cancelled.load(std::memory_order_relaxed))
    {
        result.errorMessage = QStringLiteral("TSDF integration cancelled");
        return result;
    }
    result.statistics.integratedVoxelUpdates = integratedVoxelUpdates;
    result.statistics.rejectedProjectionCount = rejectedProjectionCount;
    result.statistics.rejectedSupportMaskCount = rejectedSupportMaskCount;
    result.statistics.supportMaskFreeSpaceUpdateCount = supportMaskFreeSpaceUpdateCount;
    result.statistics.rejectedDepthValidCount = rejectedDepthValidCount;
    result.statistics.rejectedDepthCount = rejectedDepthCount;
    result.statistics.rejectedConfidenceCount = rejectedConfidenceCount;
    result.statistics.subpixelObservationCount = subpixelObservationCount;
    result.statistics.recoveredNeighborObservationCount =
        recoveredNeighborObservationCount;
    result.statistics.discontinuityRejectedCandidateCount =
        discontinuityRejectedCandidateCount;
    result.statistics.rejectedGeometryConsistencyCount =
        rejectedGeometryConsistencyCount;
    result.statistics.rejectedInvalidNearestPixelRecoveryCount =
        rejectedInvalidNearestPixelRecoveryCount;
    result.statistics.crossViewConsensusDepthObservationCount =
        crossViewConsensusDepthObservationCount;
    result.statistics.crossViewConsensusContourBandPixelCount =
        cross_view_consensus_contour_band_pixel_count;
    result.statistics.effectiveMinimumVoxelWeight = options.minimumVoxelWeight;
    result.statistics.effectiveMinimumSingleObservationWeight =
        options.minimumSingleObservationWeight;
    result.statistics.effectiveMinimumGeometryVerifiedObservationWeight =
        options.minimumGeometryVerifiedObservationWeight;
    result.statistics.effectiveMinimumGeometrySupportCount =
        options.minimumGeometrySupportCount;
    result.statistics.effectiveAllowGeometryVerifiedSingleObservation =
        options.allowGeometryVerifiedSingleObservation;
    result.statistics.effectiveGeometrySingleViewNeighborhoodGuard =
        options.enableGeometrySingleViewNeighborhoodGuard;
    result.statistics.effectiveMinimumGeometrySingleViewNeighborCount =
        options.minimumGeometrySingleViewNeighborCount;
    result.statistics.effectiveGeometrySingleViewGrowthPasses =
        options.geometrySingleViewGrowthPasses;
    result.statistics.effectiveMaximumGeometrySingleViewNeighborTsdfDelta =
        options.maximumGeometrySingleViewNeighborTsdfDelta;
    result.statistics.effectiveDiscontinuityAwareSampling =
        options.enableDiscontinuityAwareSampling;
    result.statistics.effectiveMaximumInterpolationRelativeDepthSpread =
        options.maximumInterpolationRelativeDepthSpread;
    result.statistics.effectiveMaximumObservationInverseDepthSpread =
        options.maximumObservationInverseDepthSpread;
    result.statistics.effectiveAllowInvalidNearestPixelRecovery =
        options.allowInvalidNearestPixelRecovery;
    result.statistics.effectiveMaximumInvalidNearestPixelRecoveryInverseDepthSpread =
        options.maximumInvalidNearestPixelRecoveryInverseDepthSpread;
    result.statistics.effectiveCrossViewConsensusDepth =
        options.enableCrossViewConsensusDepth;
    result.statistics.effectiveMaximumCrossViewConsensusInverseDepthSpread =
        options.maximumCrossViewConsensusInverseDepthSpread;
    result.statistics.effectiveCrossViewConsensusContourBandOnly =
        options.crossViewConsensusContourBandOnly;
    result.statistics.effectiveSurfacePatchSupport =
        options.enableSurfacePatchSupport;
    result.statistics.effectiveContourBandZeroCrossingSupport =
        options.enableContourBandZeroCrossingSupport;
    result.statistics.effectiveGeometryZeroCrossingRecovery =
        options.enableGeometryZeroCrossingRecovery;
    result.statistics.effectiveMinimumSurfacePatchObservationWeight =
        options.minimumSurfacePatchObservationWeight;
    result.statistics.effectiveMinimumSurfacePatchSourceCount =
        options.minimumSurfacePatchSourceCount;
    result.statistics.effectiveMinimumSurfacePatchCoreNeighborCount =
        options.minimumSurfacePatchCoreNeighborCount;
    result.statistics.effectiveSurfacePatchGrowthPasses =
        options.surfacePatchGrowthPasses;
    result.statistics.effectiveMaximumSurfacePatchInverseDepthSpread =
        options.maximumSurfacePatchInverseDepthSpread;
    result.statistics.effectiveMaximumSurfacePatchNormalAngleDegrees =
        options.maximumSurfacePatchNormalAngleDegrees;
    result.statistics.effectiveMaximumSurfacePatchAbsoluteTsdf =
        options.maximumSurfacePatchAbsoluteTsdf;
    result.statistics.effectiveMinimumSurfacePatchWeightRatio =
        options.minimumSurfacePatchWeightRatio;
    result.statistics.effectiveMinimumDistinctCameraSupport =
        options.minimumDistinctCameraSupport;
    result.statistics.effectiveTruncationVoxels =
        effective_truncation_voxels;
    result.statistics.effectiveSurfaceSupportBandVoxels =
        effective_surface_support_band_voxels;
    result.statistics.effectiveMaximumFreeSpaceVoxels =
        options.maximumFreeSpaceVoxels > 0.0f
        ? std::max(options.truncationVoxels, options.maximumFreeSpaceVoxels)
        : 0.0f;
    result.statistics.effectiveMinimumSupportMaskFreeSpaceViews = std::clamp(
        options.minimumSupportMaskFreeSpaceViews, 1, 16);
    result.statistics.effectiveDepthValidBoundaryErosionPixels = erosion_pixels;
    result.statistics.effectiveGeometryVerifiedBoundaryRecovery =
        options.enableGeometryVerifiedBoundaryRecovery;
    result.statistics.effectiveMinimumBoundaryRecoveryGeometrySupport =
        options.minimumBoundaryRecoveryGeometrySupport;
    result.statistics.effectiveMaximumBoundaryRecoveryInverseDepthSpread =
        options.maximumBoundaryRecoveryInverseDepthSpread;
    result.statistics.boundaryRecoveredDepthValidPixelCount =
        boundary_recovered_depth_valid_pixel_count;

    std::vector<std::uint8_t> supported(static_cast<std::size_t>(result.layout.sampleCount), 0);
    std::vector<std::size_t> guarded_geometry_single_view_candidates;
    if (options.enableGeometrySingleViewNeighborhoodGuard)
    {
        guarded_geometry_single_view_candidates.reserve(
            static_cast<std::size_t>(result.layout.sampleCount / 32));
    }
    for (std::size_t index = 0; index < supported.size(); ++index)
    {
        bool single_view_supported = false;
        bool multi_view_supported = false;
        bool geometry_verified_single_view_supported = false;
        const bool sample_supported = isSampleSupported(
            weight[index],
            support[index],
            maximumObservationWeight[index],
            options,
            &single_view_supported,
            &multi_view_supported,
            maximumGeometrySupportCount[index],
            &geometry_verified_single_view_supported);
        if (multi_view_supported)
        {
            supported[index] = 1;
            ++result.statistics.multiViewSupportedSampleCount;
        }
        else if (single_view_supported)
        {
            if (geometry_verified_single_view_supported &&
                options.enableGeometrySingleViewNeighborhoodGuard)
            {
                guarded_geometry_single_view_candidates.push_back(index);
            }
            else
            {
                supported[index] = 1;
                ++result.statistics.singleViewSupportedSampleCount;
                if (geometry_verified_single_view_supported)
                {
                    ++result.statistics.geometryVerifiedSingleViewSupportedSampleCount;
                }
            }
        }
        else if (!sample_supported && support[index] == 1)
        {
            ++result.statistics.rejectedSingleObservationWeightCount;
        }
        else if (!sample_supported && support[index] >= options.minimumDistinctCameraSupport)
        {
            ++result.statistics.rejectedAccumulatedWeightCount;
        }
        result.statistics.supportedSampleCount += supported[index] != 0;
    }
    if (options.enableGeometrySingleViewNeighborhoodGuard &&
        !guarded_geometry_single_view_candidates.empty())
    {
        const int accepted_count = growGeometryVerifiedSingleViewSamples(
            result.layout,
            tsdf,
            guarded_geometry_single_view_candidates,
            options.minimumGeometrySingleViewNeighborCount,
            options.geometrySingleViewGrowthPasses,
            options.maximumGeometrySingleViewNeighborTsdfDelta,
            &supported);
        result.statistics.geometrySingleViewNeighborhoodCandidateCount =
            guarded_geometry_single_view_candidates.size();
        result.statistics.geometrySingleViewNeighborhoodAcceptedCount = accepted_count;
        result.statistics.geometrySingleViewNeighborhoodRejectedCount =
            guarded_geometry_single_view_candidates.size() -
            static_cast<std::size_t>(accepted_count);
        result.statistics.singleViewSupportedSampleCount += accepted_count;
        result.statistics.geometryVerifiedSingleViewSupportedSampleCount += accepted_count;
        result.statistics.supportedSampleCount += accepted_count;
    }
    if ((options.enableSurfacePatchSupport ||
         options.enableContourBandZeroCrossingSupport ||
         options.enableGeometryZeroCrossingRecovery) &&
        !geometrySourceMask.empty())
    {
        const int minimum_source_count = std::clamp(
            options.minimumSurfacePatchSourceCount, 2, 8);
        const int minimum_core_neighbor_count = std::clamp(
            options.minimumSurfacePatchCoreNeighborCount, 1, 26);
        const int growth_passes = std::clamp(
            options.surfacePatchGrowthPasses, 1, 6);
        const float maximum_spread = std::clamp(
            options.maximumSurfacePatchInverseDepthSpread, 0.001f, 0.05f);
        const float maximum_normal_angle = std::clamp(
            options.maximumSurfacePatchNormalAngleDegrees, 5.0f, 45.0f);
        const float maximum_absolute_tsdf = std::clamp(
            options.maximumSurfacePatchAbsoluteTsdf, 0.05f, 0.95f);
        const float minimum_surface_weight_ratio = std::clamp(
            options.minimumSurfacePatchWeightRatio, 0.01f, 1.0f);
        std::vector<float> surface_candidate_tsdf = tsdf;
        for (std::size_t index = 0; index < surface_candidate_tsdf.size(); ++index)
        {
            if (surfaceObservationWeight[index] > 1.0e-6f)
            {
                surface_candidate_tsdf[index] =
                    surfaceTsdfWeightedSum[index] / surfaceObservationWeight[index];
            }
        }
        std::vector<std::array<int, 3>> neighbor_offsets;
        neighbor_offsets.reserve(26);
        for (int delta_z = -1; delta_z <= 1; ++delta_z)
        {
            for (int delta_y = -1; delta_y <= 1; ++delta_y)
            {
                for (int delta_x = -1; delta_x <= 1; ++delta_x)
                {
                    if (delta_x != 0 || delta_y != 0 || delta_z != 0)
                    {
                        neighbor_offsets.push_back({delta_x, delta_y, delta_z});
                    }
                }
            }
        }
        for (int growth_pass = 0; growth_pass < growth_passes; ++growth_pass)
        {
            const std::vector<std::uint8_t> core_supported = supported;
            int recovered_this_pass = 0;
            for (int z = 1; z < result.layout.cells[2]; ++z)
            {
                for (int y = 1; y < result.layout.cells[1]; ++y)
                {
                    for (int x = 1; x < result.layout.cells[0]; ++x)
                    {
                        const std::size_t index = sampleIndex(
                            result.layout, x, y, z);
                        if (supported[index] != 0 || support[index] == 0)
                        {
                            continue;
                        }
                        ++result.statistics.surfacePatchConsideredSampleCount;
                        if (options.enableContourBandZeroCrossingSupport)
                        {
                            ++result.statistics
                                  .contourBandZeroCrossingConsideredSampleCount;
                            if (contourBandObservationWeight[index] <= 1.0e-6f)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedNoContourCount;
                            }
                        }
                        if (maximumObservationWeight[index] <
                            options.minimumSurfacePatchObservationWeight)
                        {
                            ++result.statistics.surfacePatchRejectedWeightCount;
                            continue;
                        }
                        if (bitCount(geometrySourceMask[index]) <
                            minimum_source_count)
                        {
                            ++result.statistics
                                  .surfacePatchRejectedSourceOverlapCount;
                            continue;
                        }
                        const std::uint16_t spread_value =
                            minimumInverseDepthSpread[index];
                        if (spread_value ==
                                std::numeric_limits<std::uint16_t>::max() ||
                            static_cast<float>(spread_value) / 100000.0f >
                                maximum_spread)
                        {
                            ++result.statistics.surfacePatchRejectedDepthSpreadCount;
                            continue;
                        }
                        if (surfaceObservationWeight[index] <= 1.0e-6f ||
                            weight[index] <= 1.0e-6f ||
                            surfaceObservationWeight[index] / weight[index] <
                                minimum_surface_weight_ratio ||
                            std::fabs(surface_candidate_tsdf[index]) >
                                maximum_absolute_tsdf)
                        {
                            ++result.statistics.surfacePatchRejectedFreeSpaceCount;
                            continue;
                        }

                        bool has_core_neighbor = false;
                        bool has_source_overlap = false;
                        bool has_normal_agreement = false;
                        int agreeing_core_neighbor_count = 0;
                        int same_sign_core_neighbor_count = 0;
                        int opposite_sign_core_neighbor_count = 0;
                        cv::Vec3f candidate_normal;
                        const bool candidate_normal_valid = volumeNormalAt(
                            result.layout,
                            surface_candidate_tsdf,
                            x,
                            y,
                            z,
                            &candidate_normal);
                        for (const auto &offset : neighbor_offsets)
                        {
                            const int neighbor_x = x + offset[0];
                            const int neighbor_y = y + offset[1];
                            const int neighbor_z = z + offset[2];
                            const std::size_t neighbor_index = sampleIndex(
                                result.layout,
                                neighbor_x,
                                neighbor_y,
                                neighbor_z);
                            if (core_supported[neighbor_index] == 0)
                            {
                                continue;
                            }
                            has_core_neighbor = true;
                            const bool candidate_negative =
                                surface_candidate_tsdf[index] < 0.0f;
                            const bool neighbor_negative =
                                tsdf[neighbor_index] < 0.0f;
                            if (candidate_negative == neighbor_negative)
                            {
                                ++same_sign_core_neighbor_count;
                            }
                            else
                            {
                                ++opposite_sign_core_neighbor_count;
                            }
                            if ((geometrySourceMask[index] &
                                 geometrySourceMask[neighbor_index]) == 0)
                            {
                                continue;
                            }
                            has_source_overlap = true;
                            cv::Vec3f neighbor_normal;
                            if (!candidate_normal_valid ||
                                !volumeNormalAt(result.layout,
                                                surface_candidate_tsdf,
                                                neighbor_x,
                                                neighbor_y,
                                                neighbor_z,
                                                &neighbor_normal))
                            {
                                continue;
                            }
                            const float cosine = std::clamp(
                                std::fabs(candidate_normal.dot(neighbor_normal)),
                                0.0f,
                                1.0f);
                            const float angle = std::acos(cosine) * 180.0f /
                                static_cast<float>(CV_PI);
                            if (angle <= maximum_normal_angle)
                            {
                                has_normal_agreement = true;
                                ++agreeing_core_neighbor_count;
                            }
                        }
                        if (!has_core_neighbor || !has_source_overlap)
                        {
                            ++result.statistics
                                  .surfacePatchRejectedSourceOverlapCount;
                            continue;
                        }
                        const bool normal_patch_supported =
                            options.enableSurfacePatchSupport &&
                            has_normal_agreement &&
                            agreeing_core_neighbor_count >=
                                minimum_core_neighbor_count;
                        const bool zero_crossing_supported =
                            options.enableContourBandZeroCrossingSupport &&
                            contourBandObservationWeight[index] > 1.0e-6f &&
                            maximumGeometrySupportCount[index] >=
                                options.minimumBoundaryRecoveryGeometrySupport &&
                            same_sign_core_neighbor_count >= 1 &&
                            opposite_sign_core_neighbor_count >= 1;
                        if (!normal_patch_supported &&
                            !zero_crossing_supported)
                        {
                            ++result.statistics.surfacePatchRejectedNormalCount;
                            if (options.enableContourBandZeroCrossingSupport &&
                                contourBandObservationWeight[index] > 1.0e-6f)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedNoSignPairCount;
                            }
                            continue;
                        }
                        supported[index] = 1;
                        tsdf[index] = surface_candidate_tsdf[index];
                        ++recovered_this_pass;
                        ++result.statistics.surfacePatchRecoveredSampleCount;
                        ++result.statistics.supportedSampleCount;
                        if (zero_crossing_supported &&
                            !normal_patch_supported)
                        {
                            ++result.statistics
                                  .contourBandZeroCrossingRecoveredSampleCount;
                        }
                    }
                }
            }
            ++result.statistics.surfacePatchExecutedGrowthPassCount;
            if (recovered_this_pass == 0)
            {
                break;
            }
        }
        if (options.enableGeometryZeroCrossingRecovery)
        {
            std::vector<std::uint8_t> eligible(supported.size(), 0);
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                const std::uint16_t spread_value =
                    minimumInverseDepthSpread[index];
                eligible[index] =
                    supported[index] == 0 &&
                    maximumObservationWeight[index] >=
                        options.minimumSurfacePatchObservationWeight &&
                    bitCount(geometrySourceMask[index]) >=
                        minimum_source_count &&
                    spread_value != std::numeric_limits<std::uint16_t>::max() &&
                    static_cast<float>(spread_value) / 100000.0f <=
                        maximum_spread &&
                    surfaceObservationWeight[index] > 1.0e-6f &&
                    weight[index] > 1.0e-6f &&
                    surfaceObservationWeight[index] / weight[index] >=
                        minimum_surface_weight_ratio &&
                    std::fabs(surface_candidate_tsdf[index]) <=
                        maximum_absolute_tsdf;
            }
            const std::vector<std::uint8_t> supported_before_recovery =
                supported;
            const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                recoverGeometryVerifiedZeroCrossingSamples(
                    result.layout,
                    surface_candidate_tsdf,
                    weight,
                    geometrySourceMask,
                    eligible,
                    options.geometryZeroCrossingMinimumSupportedCorners,
                    options.geometryZeroCrossingMinimumCellVotes,
                    &supported);
            result.statistics.geometryZeroCrossingCandidateSampleCount =
                recovery.candidateSampleCount;
            result.statistics.geometryZeroCrossingRecoveredSampleCount =
                recovery.recoveredSampleCount;
            result.statistics.supportedSampleCount +=
                recovery.recoveredSampleCount;
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                if (supported_before_recovery[index] == 0 &&
                    supported[index] != 0)
                {
                    tsdf[index] = surface_candidate_tsdf[index];
                }
            }
        }
        result.statistics.surfacePatchCreatedComponentCount = 0;
    }
    if (result.statistics.supportedSampleCount == 0)
    {
        result.errorMessage = QStringLiteral("TSDF integration produced no multi-camera supported samples");
        return result;
    }

    result.statistics.effectiveZeroCrossingDiagnostics =
        options.collectZeroCrossingDiagnostics;
    if (options.collectZeroCrossingDiagnostics)
    {
        const DepthTsdfZeroCrossingStatistics zero_crossings = analyzeZeroCrossings(
            result.layout, tsdf, weight, supported);
        result.statistics.zeroCrossingObservedCellCount = zero_crossings.observedCellCount;
        result.statistics.zeroCrossingRawCandidateCellCount =
            zero_crossings.rawCandidateCellCount;
        result.statistics.zeroCrossingExtractableCellCount =
            zero_crossings.extractableCellCount;
        result.statistics.zeroCrossingSuppressedBySupportCellCount =
            zero_crossings.suppressedBySupportCellCount;
        result.statistics.zeroCrossingPositiveOnlySupportedCellCount =
            zero_crossings.positiveOnlySupportedCellCount;
        result.statistics.zeroCrossingNegativeOnlySupportedCellCount =
            zero_crossings.negativeOnlySupportedCellCount;
        result.statistics.zeroCrossingPartiallySupportedCellCount =
            zero_crossings.partiallySupportedCellCount;
        result.statistics.zeroCrossingFullyUnsupportedObservedCellCount =
            zero_crossings.fullyUnsupportedObservedCellCount;
    }

    if (options.progress)
    {
        options.progress(QStringLiteral("正在提取 TSDF 零等值面..."), 75);
    }
    using PostprocessClock = std::chrono::steady_clock;
    const auto elapsedMilliseconds = [](const PostprocessClock::time_point &start)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   PostprocessClock::now() - start)
            .count();
    };
    const auto reportProgress = [&options](const QString &stage, int percent)
    {
        if (options.progress)
        {
            options.progress(stage, percent);
        }
    };
    const auto postprocessCancelled = [&options, &result]()
    {
        if (!options.isCancelled || !options.isCancelled())
        {
            return false;
        }
        result.errorMessage = QStringLiteral("TSDF 后处理已取消");
        return true;
    };
    const PostprocessClock::time_point postprocess_start = PostprocessClock::now();
    const PostprocessClock::time_point marching_cubes_start = PostprocessClock::now();
    try
    {
        plapoint::mesh::MarchingCubes<float> marchingCubes;
        marchingCubes.setBounds(
            {result.layout.boundsMin[0], result.layout.boundsMin[1], result.layout.boundsMin[2]},
            {result.layout.boundsMax[0], result.layout.boundsMax[1], result.layout.boundsMax[2]});
        marchingCubes.setResolution(result.layout.cells[0],
                                    result.layout.cells[1],
                                    result.layout.cells[2]);
        marchingCubes.setIsoLevel(0.0f);
        auto [vertices, faces] = marchingCubes.extract(
            [&](float x, float y, float z)
            {
                const int ix = std::clamp(static_cast<int>(std::lround(
                                              (x - result.layout.boundsMin[0]) /
                                              result.layout.voxelSize[0])),
                                          0,
                                          result.layout.cells[0]);
                const int iy = std::clamp(static_cast<int>(std::lround(
                                              (y - result.layout.boundsMin[1]) /
                                              result.layout.voxelSize[1])),
                                          0,
                                          result.layout.cells[1]);
                const int iz = std::clamp(static_cast<int>(std::lround(
                                              (z - result.layout.boundsMin[2]) /
                                              result.layout.voxelSize[2])),
                                          0,
                                          result.layout.cells[2]);
                const std::size_t index = sampleIndex(result.layout, ix, iy, iz);
                return supported[index] != 0 ? tsdf[index] : 1.0f;
            });
        result.mesh.vertices.resize(static_cast<std::size_t>(vertices.rows()));
        for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
        {
            MeshVertex &vertex = result.mesh.vertices[static_cast<std::size_t>(row)];
            vertex.x = vertices(row, 0);
            vertex.y = vertices(row, 1);
            vertex.z = vertices(row, 2);
        }
        result.mesh.faces.resize(static_cast<std::size_t>(faces.rows()));
        for (plamatrix::Index row = 0; row < faces.rows(); ++row)
        {
            Triangle &face = result.mesh.faces[static_cast<std::size_t>(row)];
            face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
            face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
            face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
        }
        result.statistics.marchingCubesVertexCount = result.mesh.vertexCount();
        result.statistics.marchingCubesFaceCount = result.mesh.faceCount();
        result.statistics.marchingCubesElapsedMs =
            elapsedMilliseconds(marching_cubes_start);
    }
    catch (const std::exception &exception)
    {
        result.errorMessage = QStringLiteral("TSDF Marching Cubes failed: %1")
                                  .arg(QString::fromUtf8(exception.what()));
        return result;
    }
    if (result.mesh.empty())
    {
        result.errorMessage = QStringLiteral("TSDF extraction produced an empty mesh");
        return result;
    }
    reportProgress(QStringLiteral("TSDF 零等值面提取完成，正在清理网格..."), 82);
    if (postprocessCancelled())
    {
        return result;
    }

    const PostprocessClock::time_point cleanup_start = PostprocessClock::now();
    detail::removeDegenerateFaces(&result.mesh);
    detail::weldCoincidentVertices(&result.mesh, 1.0e-6f);
    detail::removeSmallConnectedComponents(
        &result.mesh,
        std::max(2, options.minimumComponentFaces),
        options.minimumComponentFaceRatio);
    result.statistics.componentFilteredFaceCount = result.mesh.faceCount();
    const WeakBoundaryTipResult weak_tips = trimWeakBoundaryTips(
        &result.mesh,
        result.layout,
        support,
        std::max(2, options.minimumDistinctCameraSupport),
        options.weakBoundaryTipTrimPasses,
        options.trimWeakBoundaryTips);
    result.statistics.weakBoundaryTipVertexCount = weak_tips.weakVertexCount;
    result.statistics.candidateWeakBoundaryTipFaceCount = weak_tips.candidateFaceCount;
    result.statistics.trimmedWeakBoundaryTipFaceCount = weak_tips.trimmedFaceCount;
    if (weak_tips.trimmedFaceCount > 0)
    {
        detail::removeSmallConnectedComponents(
            &result.mesh,
            std::max(2, options.minimumComponentFaces),
            options.minimumComponentFaceRatio);
    }
    reportProgress(QStringLiteral("正在清理 TSDF 网格拓扑..."), 84);
    if (postprocessCancelled())
    {
        return result;
    }
    if (options.fillSmallBoundaryHoles || options.enableQuadricSimplification)
    {
        result.statistics.removedDuplicateFaceCount =
            detail::removeDuplicateFaces(&result.mesh);
        result.statistics.removedNonManifoldFaceCount =
            detail::removeNonManifoldFaces(&result.mesh);
        detail::removeDegenerateFaces(&result.mesh);
    }
    result.statistics.boundaryEdgeCountBefore = boundaryEdgeCount(result.mesh);
    if (options.fillSmallBoundaryHoles)
    {
        if (options.splitPinchedBoundaryVertices)
        {
            result.statistics.splitPinchedBoundaryVertexCount =
                detail::splitPinchedBoundaryVertices(&result.mesh);
        }
        const int faces_before = result.mesh.faceCount();
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        result.statistics.filledBoundaryHoleCount = detail::fillSmallBoundaryHoles(
            &result.mesh,
            std::max(3, options.maximumHoleBoundaryEdges),
            std::max(0.0f, options.maximumHoleDiameterVoxels) * maximum_voxel_size);
        result.statistics.addedHoleFillFaceCount = std::max(
            0, result.mesh.faceCount() - faces_before);
        detail::removeDegenerateFaces(&result.mesh);
    }
    if (options.boundarySmoothingIterations > 0)
    {
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        result.statistics.smoothedBoundaryVertexCount = detail::smoothOpenBoundaryVertices(
            &result.mesh,
            options.boundarySmoothingIterations,
            options.boundarySmoothingLambda,
            options.maximumBoundarySmoothingDisplacementVoxels * maximum_voxel_size);
    }
    result.statistics.effectiveSurfaceDenoisingIterations =
        options.surfaceDenoisingIterations;
    result.statistics.effectiveSurfaceDenoisingLambda =
        options.surfaceDenoisingLambda;
    result.statistics.effectiveMaximumSurfaceDenoisingDisplacementVoxels =
        options.maximumSurfaceDenoisingDisplacementVoxels;
    result.statistics.effectiveMaximumSurfaceDenoisingNormalAngleDegrees =
        options.maximumSurfaceDenoisingNormalAngleDegrees;
    result.statistics.effectiveSurfaceDenoisingBoundaryProtectionRings =
        options.surfaceDenoisingBoundaryProtectionRings;
    if (options.surfaceDenoisingIterations > 0)
    {
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        result.statistics.smoothedSurfaceVertexCount =
            detail::smoothSurfaceVerticesNormalAware(
                &result.mesh,
                options.surfaceDenoisingIterations,
                options.surfaceDenoisingLambda,
                options.maximumSurfaceDenoisingDisplacementVoxels * maximum_voxel_size,
                options.maximumSurfaceDenoisingNormalAngleDegrees,
                options.surfaceDenoisingBoundaryProtectionRings);
    }
    detail::removeDegenerateFaces(&result.mesh);
    if (options.fillSmallBoundaryHoles)
    {
        result.statistics.splitPinchedBoundaryVertexCount +=
            detail::splitPinchedBoundaryVertices(&result.mesh);
        const int faces_before_refill = result.mesh.faceCount();
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        result.statistics.filledBoundaryHoleCount += detail::fillSmallBoundaryHoles(
            &result.mesh,
            std::max(3, options.maximumHoleBoundaryEdges),
            std::max(0.0f, options.maximumHoleDiameterVoxels) * maximum_voxel_size);
        result.statistics.addedHoleFillFaceCount += std::max(
            0, result.mesh.faceCount() - faces_before_refill);
        detail::removeDegenerateFaces(&result.mesh);
    }
    reportProgress(QStringLiteral("正在修补和整理网格边界..."), 86);
    if (postprocessCancelled())
    {
        return result;
    }
    result.statistics.compactedUnusedVertexCount =
        detail::compactReferencedVertices(&result.mesh);
    result.statistics.preSimplificationFaceCount = result.mesh.faceCount();
    result.statistics.meshCleanupElapsedMs = elapsedMilliseconds(cleanup_start);
    result.statistics.effectiveVoxelFallbackSimplification =
        options.enableVoxelFallbackSimplification;
    result.statistics.effectiveVoxelFallbackQemPolish =
        options.enableVoxelFallbackQemPolish;
    result.statistics.effectiveVoxelFallbackMinimumProtectedBoundaryVertices =
        options.voxelFallbackMinimumProtectedBoundaryVertices;
    result.statistics.effectiveVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels =
        options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels;
    result.statistics.effectiveVoxelFallbackMaximumNormalClusterAngleDegrees =
        options.voxelFallbackMaximumNormalClusterAngleDegrees;
    result.statistics.effectiveVoxelFallbackInitialClusterFactor =
        options.voxelFallbackInitialClusterFactor;
    result.statistics.effectiveVoxelFallbackMultiViewSilhouetteProtection =
        options.enableVoxelFallbackMultiViewSilhouetteProtection;
    result.statistics.effectiveVoxelFallbackMinimumSilhouetteViews =
        options.voxelFallbackMinimumSilhouetteViews;
    result.statistics.effectiveVoxelFallbackSilhouetteBandPixels =
        options.voxelFallbackSilhouetteBandPixels;
    result.statistics.effectiveVoxelFallbackSilhouetteDepthToleranceVoxels =
        options.voxelFallbackSilhouetteDepthToleranceVoxels;
    const PostprocessClock::time_point simplification_start = PostprocessClock::now();
    if (options.enableQuadricSimplification && options.simplifyTargetFaces > 0 &&
        result.mesh.faceCount() > options.simplifyTargetFaces)
    {
        reportProgress(QStringLiteral("正在进行保拓扑 QEM 简化..."), 87);
        TriMesh unsimplified_mesh = result.mesh;
        const int input_face_count = result.mesh.faceCount();
        const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
        const int simplification_boundary_edge_count_before = topology_before.boundaryEdgeCount;
        QuadricSimplifyOptions simplify_options;
        simplify_options.targetFaceCount = options.simplifyTargetFaces;
        simplify_options.maximumPasses = options.simplificationMaximumPasses;
        simplify_options.workerCount = options.workerCount;
        simplify_options.featureAngleDegrees = options.simplificationFeatureAngleDegrees;
        simplify_options.maximumNormalDeviationDegrees =
            options.simplificationMaximumNormalDeviationDegrees;
        simplify_options.minimumSharpEdgeEndpointDegree =
            options.simplificationMinimumSharpEdgeEndpointDegree;
        simplify_options.simplifySimpleOpenBoundaries =
            options.simplifySimpleOpenBoundaries;
        simplify_options.isCancelled = options.isCancelled;
        simplify_options.progress = [&reportProgress](int pass, int face_count)
        {
            reportProgress(
                QStringLiteral("正在进行保拓扑 QEM 简化（第 %1 轮，%2 面）...")
                    .arg(pass)
                    .arg(face_count),
                std::min(89, 87 + pass / 8));
        };
        const QuadricSimplifyStatistics simplify_statistics = simplifyMeshQuadric(
            &result.mesh, simplify_options);
        const MeshBoundaryTopology topology_after = boundaryTopology(result.mesh);
        const int simplified_boundary_edge_count = topology_after.boundaryEdgeCount;
        const auto topology_growth_is_safe = [](int before, int after)
        {
            return after <= before + std::max(8, static_cast<int>(std::ceil(before * 0.10f)));
        };
        const bool accept_simplification = shouldAcceptQuadricSimplification(
            input_face_count,
            result.mesh.faceCount(),
            simplification_boundary_edge_count_before,
            simplified_boundary_edge_count,
            options.maximumSimplificationBoundaryEdgeGrowthRatio) &&
            topology_growth_is_safe(topology_before.danglingBoundaryVertexCount,
                                    topology_after.danglingBoundaryVertexCount) &&
            topology_growth_is_safe(topology_before.nonManifoldEdgeCount,
                                    topology_after.nonManifoldEdgeCount);
        result.statistics.effectiveQuadricSimplification = accept_simplification;
        result.statistics.quadricSimplificationAccepted = accept_simplification;
        result.statistics.quadricSimplificationBoundarySafetyRejected =
            !accept_simplification && result.mesh.faceCount() < input_face_count;
        result.statistics.quadricBoundaryEdgeCountBefore =
            simplification_boundary_edge_count_before;
        result.statistics.quadricBoundaryEdgeCountAfter = simplified_boundary_edge_count;
        result.statistics.quadricDanglingBoundaryVertexCountBefore =
            topology_before.danglingBoundaryVertexCount;
        result.statistics.quadricDanglingBoundaryVertexCountAfter =
            topology_after.danglingBoundaryVertexCount;
        result.statistics.quadricNonManifoldEdgeCountBefore = topology_before.nonManifoldEdgeCount;
        result.statistics.quadricNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;
        result.statistics.requestedSimplifyTargetFaces = options.simplifyTargetFaces;
        result.statistics.quadricCollapsedEdgeCount = simplify_statistics.collapsedEdgeCount;
        result.statistics.quadricRejectedBoundaryEdgeCount =
            simplify_statistics.rejectedBoundaryEdgeCount;
        result.statistics.quadricRejectedFeatureEdgeCount =
            simplify_statistics.rejectedFeatureEdgeCount;
        result.statistics.quadricRejectedTopologyEdgeCount =
            simplify_statistics.rejectedTopologyEdgeCount;
        result.statistics.quadricRejectedFlipEdgeCount =
            simplify_statistics.rejectedFlipEdgeCount;
        result.statistics.quadricSimplifyPassCount = simplify_statistics.passCount;
        result.statistics.quadricSimplifyReachedTarget = simplify_statistics.reachedTarget;
        result.statistics.quadricSimplifyStoppedByStagnation =
            simplify_statistics.stoppedByStagnation;
        if (!accept_simplification)
        {
            result.mesh = std::move(unsimplified_mesh);
            result.statistics.quadricSimplifyReachedTarget = false;
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    if (options.enableQuadricSimplification &&
        options.enableVoxelFallbackSimplification &&
        options.simplifyTargetFaces > 0 &&
        !result.statistics.quadricSimplifyReachedTarget &&
        result.mesh.faceCount() > options.simplifyTargetFaces * 2)
    {
        reportProgress(QStringLiteral("正在进行边界感知后备简化..."), 90);
        result.statistics.voxelFallbackAttempted = true;
        result.statistics.voxelFallbackPreservedOpenBoundaries = true;
        result.statistics.voxelFallbackInputFaceCount = result.mesh.faceCount();
        const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
        result.statistics.voxelFallbackBoundaryEdgeCountBefore =
            topology_before.boundaryEdgeCount;
        const TriangleQualitySummary quality_before =
            triangleQualitySummary(result.mesh);
        result.statistics.voxelFallbackSliverFaceCountBefore =
            quality_before.sliverFaceCount;
        result.statistics.voxelFallbackSliverRatioBefore =
            quality_before.sliverRatio;

        TriMesh candidate = result.mesh;
        ReconstructionConfig fallback_config;
        fallback_config.resolution = options.resolution;
        fallback_config.simplifyTargetFaces = std::max(
            options.simplifyTargetFaces, 120000);
        fallback_config.voxelSimplifyFactor = options.voxelFallbackInitialClusterFactor;
        fallback_config.minComponentFaces = options.minimumComponentFaces;
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        std::vector<std::uint8_t> protected_silhouette_vertices;
        if (options.enableVoxelFallbackMultiViewSilhouetteProtection)
        {
            protected_silhouette_vertices = multiViewSilhouetteBoundaryVertices(
                candidate,
                frames,
                effective_depth_valid_masks,
                options.voxelFallbackMinimumSilhouetteViews,
                options.voxelFallbackSilhouetteBandPixels,
                options.voxelFallbackSilhouetteDepthToleranceVoxels,
                maximum_voxel_size,
                options.minimumConfidence);
            result.statistics.voxelFallbackProtectedSilhouetteVertexCount =
                static_cast<int>(std::count(
                    protected_silhouette_vertices.cbegin(),
                    protected_silhouette_vertices.cend(),
                    std::uint8_t{1}));
        }
        const std::vector<std::uint8_t> *boundary_protection =
            result.statistics.voxelFallbackProtectedSilhouetteVertexCount > 0
            ? &protected_silhouette_vertices
            : nullptr;
        detail::simplifyVoxelMeshAdaptive(
            &candidate,
            fallback_config,
            maximum_voxel_size * 0.5f,
            true,
            options.voxelFallbackMinimumProtectedBoundaryVertices,
            options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels *
                maximum_voxel_size,
            options.voxelFallbackMaximumNormalClusterAngleDegrees,
            boundary_protection);
        detail::removeDuplicateFaces(&candidate);
        detail::removeNonManifoldFaces(&candidate);
        detail::removeDegenerateFaces(&candidate);
        detail::removeSmallConnectedComponents(
            &candidate,
            std::max(2, options.minimumComponentFaces),
            options.minimumComponentFaceRatio);
        if (options.fillSmallBoundaryHoles)
        {
            result.statistics.splitPinchedBoundaryVertexCount +=
                detail::splitPinchedBoundaryVertices(&candidate);
            const int faces_before_fallback_fill = candidate.faceCount();
            result.statistics.filledBoundaryHoleCount +=
                detail::fillSmallBoundaryHoles(
                    &candidate,
                    std::max(3, options.maximumHoleBoundaryEdges),
                    std::max(0.0f, options.maximumHoleDiameterVoxels) *
                        maximum_voxel_size);
            result.statistics.addedHoleFillFaceCount += std::max(
                0, candidate.faceCount() - faces_before_fallback_fill);
            detail::removeDegenerateFaces(&candidate);
        }
        detail::compactReferencedVertices(&candidate);

        const int polish_target_faces = std::max(
            options.simplifyTargetFaces, 120000);
        if (options.enableVoxelFallbackQemPolish &&
            candidate.faceCount() > polish_target_faces)
        {
            result.statistics.voxelFallbackQemPolishAttempted = true;
            result.statistics.voxelFallbackQemPolishInputFaceCount =
                candidate.faceCount();
            TriMesh unpolished_candidate = candidate;
            const MeshBoundaryTopology polish_topology_before =
                boundaryTopology(candidate);
            QuadricSimplifyOptions polish_options;
            polish_options.targetFaceCount = polish_target_faces;
            polish_options.maximumPasses = options.simplificationMaximumPasses;
            polish_options.workerCount = options.workerCount;
            polish_options.featureAngleDegrees = std::max(
                options.simplificationFeatureAngleDegrees, 80.0f);
            polish_options.maximumNormalDeviationDegrees = std::max(
                options.simplificationMaximumNormalDeviationDegrees, 80.0f);
            polish_options.minimumSharpEdgeEndpointDegree = std::max(
                options.simplificationMinimumSharpEdgeEndpointDegree, 4);
            polish_options.simplifySimpleOpenBoundaries = true;
            polish_options.isCancelled = options.isCancelled;
            polish_options.progress = [&reportProgress](int pass, int face_count)
            {
                reportProgress(
                    QStringLiteral("正在抛光后备网格（第 %1 轮，%2 面）...")
                        .arg(pass)
                        .arg(face_count),
                    std::min(92, 90 + pass / 8));
            };
            const QuadricSimplifyStatistics polish_statistics = simplifyMeshQuadric(
                &candidate, polish_options);
            const MeshBoundaryTopology polish_topology_after =
                boundaryTopology(candidate);
            result.statistics.voxelFallbackQemPolishOutputFaceCount =
                candidate.faceCount();
            result.statistics.voxelFallbackQemPolishCollapsedEdgeCount =
                polish_statistics.collapsedEdgeCount;
            const bool accept_polish = shouldAcceptQuadricSimplification(
                result.statistics.voxelFallbackQemPolishInputFaceCount,
                candidate.faceCount(),
                polish_topology_before.boundaryEdgeCount,
                polish_topology_after.boundaryEdgeCount,
                options.maximumSimplificationBoundaryEdgeGrowthRatio) &&
                polish_topology_after.danglingBoundaryVertexCount <=
                    polish_topology_before.danglingBoundaryVertexCount +
                        std::max(8, static_cast<int>(std::ceil(
                            polish_topology_before.danglingBoundaryVertexCount * 0.10f))) &&
                polish_topology_after.nonManifoldEdgeCount <=
                    polish_topology_before.nonManifoldEdgeCount;
            result.statistics.voxelFallbackQemPolishAccepted = accept_polish;
            if (!accept_polish)
            {
                candidate = std::move(unpolished_candidate);
            }
        }

        const MeshBoundaryTopology topology_after = boundaryTopology(candidate);
        const TriangleQualitySummary quality_after =
            triangleQualitySummary(candidate);
        result.statistics.voxelFallbackOutputFaceCount = candidate.faceCount();
        result.statistics.voxelFallbackBoundaryEdgeCountAfter =
            topology_after.boundaryEdgeCount;
        result.statistics.voxelFallbackNonManifoldEdgeCountAfter =
            topology_after.nonManifoldEdgeCount;
        result.statistics.voxelFallbackSliverFaceCountAfter =
            quality_after.sliverFaceCount;
        result.statistics.voxelFallbackSliverRatioAfter =
            quality_after.sliverRatio;
        const auto topology_growth_is_safe = [](int before, int after)
        {
            return after <= before + std::max(
                8, static_cast<int>(std::ceil(before * 0.10f)));
        };
        const bool triangle_quality_is_safe =
            quality_after.sliverRatio <=
                std::max(0.0f, options.voxelFallbackMaximumSliverRatio) ||
            quality_after.sliverRatio <= quality_before.sliverRatio * 0.75;
        result.statistics.voxelFallbackTriangleQualityRejected =
            !triangle_quality_is_safe;
        const bool accept_fallback = !candidate.empty() &&
            candidate.faceCount() < result.mesh.faceCount() &&
            triangle_quality_is_safe &&
            topology_growth_is_safe(topology_before.boundaryEdgeCount,
                                    topology_after.boundaryEdgeCount) &&
            topology_growth_is_safe(topology_before.danglingBoundaryVertexCount,
                                    topology_after.danglingBoundaryVertexCount) &&
            topology_growth_is_safe(topology_before.nonManifoldEdgeCount,
                                    topology_after.nonManifoldEdgeCount);
        result.statistics.voxelFallbackAccepted = accept_fallback;
        if (accept_fallback)
        {
            result.mesh = std::move(candidate);
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    result.statistics.effectiveSilhouetteAwareFinalHoleFill =
        options.enableSilhouetteAwareFinalHoleFill;
    result.statistics.effectiveVisibilityConstrainedFinalHoleFill =
        options.enableVisibilityConstrainedFinalHoleFill;
    if (options.enableSilhouetteAwareFinalHoleFill && options.fillSmallBoundaryHoles)
    {
        reportProgress(QStringLiteral("正在按多视图轮廓修补最终网格孔洞..."), 92);
        result.statistics.finalHoleFillAttempted = true;
        if (options.splitPinchedBoundaryVertices)
        {
            result.statistics.splitPinchedBoundaryVertexCount +=
                detail::splitPinchedBoundaryVertices(&result.mesh);
        }
        const float maximum_voxel_size = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        std::vector<std::uint8_t> protected_silhouette_vertices =
            multiViewSilhouetteBoundaryVertices(
                result.mesh,
                frames,
                effective_depth_valid_masks,
                options.voxelFallbackMinimumSilhouetteViews,
                options.voxelFallbackSilhouetteBandPixels,
                options.voxelFallbackSilhouetteDepthToleranceVoxels,
                maximum_voxel_size,
                options.minimumConfidence);
        if (options.enableVisibilityConstrainedFinalHoleFill)
        {
            const VisibilityHoleProtectionResult visibility =
                visibilityConstrainedHoleProtection(
                    result.mesh,
                    frames,
                    effective_depth_valid_masks,
                    protected_silhouette_vertices,
                    std::max(
                        3, options.finalHoleFillMaximumBoundaryEdges),
                    options.visibilityHoleFillMinimumSupportingViews,
                    options.visibilityHoleFillMaximumConflictViews,
                    options.visibilityHoleFillDepthToleranceVoxels,
                    options.visibilityHoleFillStrongSilhouetteRatio,
                    maximum_voxel_size,
                    options.minimumConfidence);
            protected_silhouette_vertices =
                visibility.protectedVertices;
            result.statistics.visibilityHoleFillConsideredLoopCount =
                visibility.consideredLoopCount;
            result.statistics.visibilityHoleFillReleasedLoopCount =
                visibility.releasedLoopCount;
            result.statistics.visibilityHoleFillRejectedSupportLoopCount =
                visibility.rejectedSupportLoopCount;
            result.statistics.visibilityHoleFillRejectedConflictLoopCount =
                visibility.rejectedConflictLoopCount;
        }
        result.statistics.finalHoleFillProtectedSilhouetteVertexCount =
            static_cast<int>(std::count(
                protected_silhouette_vertices.cbegin(),
                protected_silhouette_vertices.cend(),
                std::uint8_t{1}));

        const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
        const TriangleQualitySummary quality_before =
            triangleQualitySummary(result.mesh);
        result.statistics.finalHoleFillBoundaryEdgeCountBefore =
            topology_before.boundaryEdgeCount;
        result.statistics.finalHoleFillNonManifoldEdgeCountBefore =
            topology_before.nonManifoldEdgeCount;
        result.statistics.finalHoleFillSliverRatioBefore =
            quality_before.sliverRatio;

        if (result.statistics.finalHoleFillProtectedSilhouetteVertexCount > 0 ||
            options.enableVisibilityConstrainedFinalHoleFill)
        {
            TriMesh candidate = result.mesh;
            const int face_count_before = candidate.faceCount();
            int protected_hole_count = 0;
            const int filled_hole_count = detail::fillSmallBoundaryHoles(
                &candidate,
                std::max(3, options.finalHoleFillMaximumBoundaryEdges),
                std::max(0.0f, options.finalHoleFillMaximumDiameterVoxels) *
                    maximum_voxel_size,
                &protected_silhouette_vertices,
                &protected_hole_count,
                true);
            detail::removeDegenerateFaces(&candidate);
            detail::compactReferencedVertices(&candidate);

            const MeshBoundaryTopology topology_after =
                boundaryTopology(candidate);
            const TriangleQualitySummary quality_after =
                triangleQualitySummary(candidate);
            const int added_face_count = std::max(
                0, candidate.faceCount() - face_count_before);
            const int maximum_added_faces = std::max(
                8,
                static_cast<int>(std::ceil(
                    face_count_before *
                    std::max(0.0f, options.finalHoleFillMaximumFaceGrowthRatio))));
            const bool triangle_quality_is_safe =
                quality_after.sliverRatio <=
                    std::max(0.0f, options.finalHoleFillMaximumSliverRatio) ||
                quality_after.sliverRatio <= quality_before.sliverRatio;
            const bool accept_final_fill =
                filled_hole_count > 0 &&
                topology_after.boundaryEdgeCount < topology_before.boundaryEdgeCount &&
                topology_after.danglingBoundaryVertexCount <=
                    topology_before.danglingBoundaryVertexCount &&
                topology_after.nonManifoldEdgeCount <=
                    topology_before.nonManifoldEdgeCount &&
                added_face_count <= maximum_added_faces &&
                triangle_quality_is_safe;

            if (accept_final_fill &&
                options.enableQuadricSimplification &&
                options.simplifyTargetFaces > 0 &&
                candidate.faceCount() > options.simplifyTargetFaces)
            {
                result.statistics.finalHoleFillPostSimplificationAttempted = true;
                result.statistics.finalHoleFillPostSimplificationInputFaceCount =
                    candidate.faceCount();
                result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountBefore =
                    topology_after.boundaryEdgeCount;

                TriMesh filled_candidate = candidate;
                QuadricSimplifyOptions post_fill_options;
                post_fill_options.targetFaceCount = options.simplifyTargetFaces;
                post_fill_options.maximumPasses = options.simplificationMaximumPasses;
                post_fill_options.workerCount = options.workerCount;
                post_fill_options.featureAngleDegrees =
                    options.simplificationFeatureAngleDegrees;
                post_fill_options.maximumNormalDeviationDegrees =
                    options.simplificationMaximumNormalDeviationDegrees;
                post_fill_options.minimumSharpEdgeEndpointDegree =
                    options.simplificationMinimumSharpEdgeEndpointDegree;
                post_fill_options.simplifySimpleOpenBoundaries = true;
                post_fill_options.isCancelled = options.isCancelled;
                post_fill_options.progress = [&reportProgress](int pass, int face_count)
                {
                    reportProgress(
                        QStringLiteral("正在整理补洞网格（第 %1 轮，%2 面）...")
                            .arg(pass)
                            .arg(face_count),
                        std::min(96, 94 + pass / 8));
                };
                const QuadricSimplifyStatistics post_fill_simplification =
                    simplifyMeshQuadric(&candidate, post_fill_options);
                const MeshBoundaryTopology post_fill_topology =
                    boundaryTopology(candidate);
                const TriangleQualitySummary post_fill_quality =
                    triangleQualitySummary(candidate);
                const int permitted_boundary_edges =
                    topology_after.boundaryEdgeCount +
                    std::max(8, static_cast<int>(std::ceil(
                        topology_after.boundaryEdgeCount * 0.10f)));
                const bool accept_post_fill_simplification =
                    !candidate.empty() &&
                    candidate.faceCount() < filled_candidate.faceCount() &&
                    post_fill_topology.boundaryEdgeCount <=
                        permitted_boundary_edges &&
                    post_fill_topology.danglingBoundaryVertexCount <=
                        topology_after.danglingBoundaryVertexCount &&
                    post_fill_topology.nonManifoldEdgeCount <=
                        topology_after.nonManifoldEdgeCount &&
                    post_fill_quality.sliverRatio <=
                        std::max(
                            static_cast<double>(
                                options.finalHoleFillMaximumSliverRatio),
                            quality_after.sliverRatio + 1.0e-12);
                result.statistics.finalHoleFillPostSimplificationAccepted =
                    accept_post_fill_simplification;
                result.statistics.finalHoleFillPostSimplificationCollapsedEdgeCount =
                    post_fill_simplification.collapsedEdgeCount;
                result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountAfter =
                    post_fill_topology.boundaryEdgeCount;
                if (!accept_post_fill_simplification)
                {
                    candidate = std::move(filled_candidate);
                    result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountAfter =
                        topology_after.boundaryEdgeCount;
                }
                result.statistics.finalHoleFillPostSimplificationOutputFaceCount =
                    candidate.faceCount();
            }

            result.statistics.finalHoleFillProtectedHoleCount =
                protected_hole_count;
            result.statistics.finalHoleFillFilledHoleCount =
                filled_hole_count;
            result.statistics.finalHoleFillAddedFaceCount =
                added_face_count;
            result.statistics.finalHoleFillBoundaryEdgeCountAfter =
                topology_after.boundaryEdgeCount;
            result.statistics.finalHoleFillNonManifoldEdgeCountAfter =
                topology_after.nonManifoldEdgeCount;
            result.statistics.finalHoleFillSliverRatioAfter =
                quality_after.sliverRatio;
            result.statistics.finalHoleFillTriangleQualityRejected =
                !triangle_quality_is_safe;
            result.statistics.finalHoleFillAccepted = accept_final_fill;
            if (accept_final_fill)
            {
                result.mesh = std::move(candidate);
                result.statistics.filledBoundaryHoleCount +=
                    filled_hole_count;
                result.statistics.addedHoleFillFaceCount +=
                    added_face_count;
            }
        }
        else
        {
            result.statistics.finalHoleFillBoundaryEdgeCountAfter =
                topology_before.boundaryEdgeCount;
            result.statistics.finalHoleFillNonManifoldEdgeCountAfter =
                topology_before.nonManifoldEdgeCount;
            result.statistics.finalHoleFillSliverRatioAfter =
                quality_before.sliverRatio;
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    result.statistics.effectiveTinyBoundaryLoopCollapse =
        options.enableTinyBoundaryLoopCollapse;
    if (options.enableTinyBoundaryLoopCollapse)
    {
        reportProgress(
            QStringLiteral("正在收缩退化微孔边界..."), 92);
        result.statistics.tinyBoundaryLoopCollapseAttempted = true;
        MeshTopologyQualityThresholds quality_thresholds;
        quality_thresholds.maximumBoundaryEdgeRatio =
            std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
        quality_thresholds.maximumHighAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
        quality_thresholds.maximumExtremeAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
        MeshTopologyQualityStatistics current_quality =
            evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
        result.statistics.tinyBoundaryLoopBoundaryEdgeCountBefore =
            current_quality.boundaryEdgeCount;
        result.statistics.tinyBoundaryLoopNonManifoldEdgeCountBefore =
            current_quality.nonManifoldEdgeCount;
        result.statistics.tinyBoundaryLoopHighAspectRatioBefore =
            current_quality.highAspectFaceRatio;

        const float maximum_voxel_size = std::max({
            result.layout.voxelSize[0],
            result.layout.voxelSize[1],
            result.layout.voxelSize[2]});
        TriMesh accepted_mesh = result.mesh;
        constexpr double quality_epsilon = 1.0e-12;
        for (int pass = 0;
             pass < options.tinyBoundaryLoopCollapseMaximumPasses;
             ++pass)
        {
            TriMesh candidate = accepted_mesh;
            const int collapsed_edge_count =
                detail::collapseTinyBoundaryLoops(
                    &candidate,
                    std::max(
                        3,
                        options.tinyBoundaryLoopCollapseMaximumEdges),
                    std::max(
                        0.0f,
                        options.tinyBoundaryLoopCollapseMaximumDiameterVoxels) *
                        maximum_voxel_size,
                    std::max(
                        0.0f,
                        options.tinyBoundaryLoopCollapseMaximumEdgeVoxels) *
                        maximum_voxel_size);
            if (collapsed_edge_count <= 0)
            {
                break;
            }

            const MeshTopologyQualityStatistics candidate_quality =
                evaluateMeshTopologyQuality(candidate, quality_thresholds);
            const bool topology_is_safe =
                candidate_quality.boundaryEdgeCount <
                    current_quality.boundaryEdgeCount &&
                candidate_quality.nonManifoldEdgeCount <=
                    current_quality.nonManifoldEdgeCount &&
                candidate_quality.componentCount <=
                    current_quality.componentCount &&
                candidate_quality.largestComponentFaceRatio + quality_epsilon >=
                    current_quality.largestComponentFaceRatio;
            const bool triangle_quality_is_safe =
                candidate_quality.highAspectFaceRatio <=
                    current_quality.highAspectFaceRatio + quality_epsilon &&
                candidate_quality.extremeAspectFaceRatio <=
                    current_quality.extremeAspectFaceRatio + quality_epsilon;
            if (!topology_is_safe || !triangle_quality_is_safe)
            {
                break;
            }

            accepted_mesh = std::move(candidate);
            current_quality = candidate_quality;
            ++result.statistics.tinyBoundaryLoopCollapsePassCount;
            result.statistics.tinyBoundaryLoopCollapsedEdgeCount +=
                collapsed_edge_count;
        }
        result.statistics.tinyBoundaryLoopCollapseAccepted =
            result.statistics.tinyBoundaryLoopCollapsePassCount > 0;
        if (result.statistics.tinyBoundaryLoopCollapseAccepted)
        {
            result.mesh = std::move(accepted_mesh);
        }
        result.statistics.tinyBoundaryLoopBoundaryEdgeCountAfter =
            current_quality.boundaryEdgeCount;
        result.statistics.tinyBoundaryLoopNonManifoldEdgeCountAfter =
            current_quality.nonManifoldEdgeCount;
        result.statistics.tinyBoundaryLoopHighAspectRatioAfter =
            current_quality.highAspectFaceRatio;
        if (postprocessCancelled())
        {
            return result;
        }
    }
    result.statistics.effectiveTriangleQualityOptimization =
        options.enableTriangleQualityOptimization;
    if (options.enableTriangleQualityOptimization)
    {
        reportProgress(QStringLiteral("正在优化三角网格质量..."), 92);
        result.statistics.triangleQualityOptimizationAttempted = true;
        result.statistics.triangleQualityOptimizationInputFaceCount =
            result.mesh.faceCount();
        MeshTopologyQualityThresholds quality_thresholds;
        quality_thresholds.maximumBoundaryEdgeRatio =
            std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
        quality_thresholds.maximumHighAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
        quality_thresholds.maximumExtremeAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
        const MeshTopologyQualityStatistics quality_before =
            evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
        result.statistics.triangleQualityHighAspectFaceRatioBefore =
            quality_before.highAspectFaceRatio;
        result.statistics.triangleQualityExtremeAspectFaceRatioBefore =
            quality_before.extremeAspectFaceRatio;

        TriMesh candidate = result.mesh;
        MeshTriangleOptimizationOptions optimization_options;
        optimization_options.maximumPasses =
            options.triangleQualityOptimizationMaximumPasses;
        optimization_options.minimumWorstAspectImprovementRatio =
            options.triangleQualityMinimumAspectImprovementRatio;
        optimization_options.maximumFeatureAngleDegrees =
            options.triangleQualityMaximumFeatureAngleDegrees;
        optimization_options.maximumNormalDeviationDegrees =
            options.triangleQualityMaximumNormalDeviationDegrees;
        optimization_options.enableTangentialRelaxation =
            options.enableTriangleQualityTangentialRelaxation;
        optimization_options.tangentialRelaxationPasses =
            options.triangleQualityTangentialRelaxationPasses;
        optimization_options.tangentialRelaxationLambda =
            options.triangleQualityTangentialRelaxationLambda;
        optimization_options.tangentialMaximumDisplacementEdgeRatio =
            options.triangleQualityTangentialMaximumDisplacementEdgeRatio;
        optimization_options.enableIsotropicRemeshing =
            options.enableTriangleQualityIsotropicRemeshing;
        optimization_options.isotropicRemeshingPasses =
            options.triangleQualityIsotropicRemeshingPasses;
        optimization_options.isotropicShortEdgeRatio =
            options.triangleQualityIsotropicShortEdgeRatio;
        optimization_options.isotropicLongEdgeRatio =
            options.triangleQualityIsotropicLongEdgeRatio;
        optimization_options.isotropicMaximumFaceGrowthRatio =
            options.triangleQualityIsotropicMaximumFaceGrowthRatio;
        optimization_options.isCancelled = options.isCancelled;
        const MeshTriangleOptimizationStatistics optimization =
            optimizeTriangleQuality(&candidate, optimization_options);
        const MeshTopologyQualityStatistics quality_after =
            evaluateMeshTopologyQuality(candidate, quality_thresholds);
        result.statistics.triangleQualityOptimizationPassCount =
            optimization.passCount;
        result.statistics.triangleQualityOptimizationFlippedEdgeCount =
            optimization.flippedEdgeCount;
        result.statistics.triangleQualityTangentialRelaxationPassCount =
            optimization.tangentialRelaxationPassCount;
        result.statistics.triangleQualityTangentialRelaxedVertexCount =
            optimization.tangentialRelaxedVertexCount;
        result.statistics.triangleQualityIsotropicRemeshingPassCount =
            optimization.isotropicRemeshingPassCount;
        result.statistics.triangleQualityIsotropicCollapsedEdgeCount =
            optimization.isotropicCollapsedEdgeCount;
        result.statistics.triangleQualityIsotropicSplitEdgeCount =
            optimization.isotropicSplitEdgeCount;
        result.statistics.triangleQualityOptimizationOutputFaceCount =
            candidate.faceCount();
        result.statistics.triangleQualityHighAspectFaceRatioAfter =
            quality_after.highAspectFaceRatio;
        result.statistics.triangleQualityExtremeAspectFaceRatioAfter =
            quality_after.extremeAspectFaceRatio;

        constexpr double ratio_epsilon = 1.0e-12;
        const bool topology_preserved =
            quality_after.boundaryEdgeCount == quality_before.boundaryEdgeCount &&
            quality_after.nonManifoldEdgeCount ==
                quality_before.nonManifoldEdgeCount &&
            quality_after.componentCount == quality_before.componentCount &&
            quality_after.largestComponentFaceRatio + ratio_epsilon >=
                quality_before.largestComponentFaceRatio;
        const bool triangle_quality_improved =
            quality_after.highAspectFaceRatio <=
                quality_before.highAspectFaceRatio + ratio_epsilon &&
            quality_after.extremeAspectFaceRatio <=
                quality_before.extremeAspectFaceRatio + ratio_epsilon &&
            quality_after.skinnyFaceRatio <=
                quality_before.skinnyFaceRatio + ratio_epsilon &&
            (quality_after.highAspectFaceRatio + ratio_epsilon <
                 quality_before.highAspectFaceRatio ||
             quality_after.extremeAspectFaceRatio + ratio_epsilon <
                 quality_before.extremeAspectFaceRatio ||
             quality_after.skinnyFaceRatio + ratio_epsilon <
                 quality_before.skinnyFaceRatio);
        const bool accept_optimization =
            !optimization.cancelled &&
            (optimization.flippedEdgeCount > 0 ||
             optimization.tangentialRelaxedVertexCount > 0 ||
             optimization.isotropicCollapsedEdgeCount > 0 ||
             optimization.isotropicSplitEdgeCount > 0) &&
            topology_preserved &&
            triangle_quality_improved;
        result.statistics.triangleQualityOptimizationAccepted =
            accept_optimization;
        if (accept_optimization)
        {
            result.mesh = std::move(candidate);
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    if (options.enableSilhouetteAwareFinalHoleFill &&
        options.fillSmallBoundaryHoles)
    {
        reportProgress(QStringLiteral("正在修补简化后残余微孔..."), 92);
        result.statistics.residualMicroHoleFillAttempted = true;
        const float maximum_voxel_size = std::max({
            result.layout.voxelSize[0],
            result.layout.voxelSize[1],
            result.layout.voxelSize[2]});
        const int maximum_boundary_edges = std::min(
            16,
            std::max(3, options.finalHoleFillMaximumBoundaryEdges));
        std::vector<std::uint8_t> protected_silhouette_vertices =
            multiViewSilhouetteBoundaryVertices(
                result.mesh,
                frames,
                effective_depth_valid_masks,
                options.voxelFallbackMinimumSilhouetteViews,
                options.voxelFallbackSilhouetteBandPixels,
                options.voxelFallbackSilhouetteDepthToleranceVoxels,
                maximum_voxel_size,
                options.minimumConfidence);
        if (options.enableVisibilityConstrainedFinalHoleFill)
        {
            protected_silhouette_vertices =
                visibilityConstrainedHoleProtection(
                    result.mesh,
                    frames,
                    effective_depth_valid_masks,
                    protected_silhouette_vertices,
                    maximum_boundary_edges,
                    options.visibilityHoleFillMinimumSupportingViews,
                    options.visibilityHoleFillMaximumConflictViews,
                    options.visibilityHoleFillDepthToleranceVoxels,
                    options.visibilityHoleFillStrongSilhouetteRatio,
                    maximum_voxel_size,
                    options.minimumConfidence)
                    .protectedVertices;
        }

        const MeshBoundaryTopology topology_before =
            boundaryTopology(result.mesh);
        const TriangleQualitySummary quality_before =
            triangleQualitySummary(result.mesh);
        result.statistics.residualMicroHoleFillBoundaryEdgeCountBefore =
            topology_before.boundaryEdgeCount;
        result.statistics.residualMicroHoleFillNonManifoldEdgeCountBefore =
            topology_before.nonManifoldEdgeCount;
        result.statistics.residualMicroHoleFillSliverRatioBefore =
            quality_before.sliverRatio;

        TriMesh candidate = result.mesh;
        const int face_count_before = candidate.faceCount();
        int protected_hole_count = 0;
        const int filled_hole_count = detail::fillSmallBoundaryHoles(
            &candidate,
            maximum_boundary_edges,
            std::min(
                4.0f,
                std::max(
                    0.0f,
                    options.finalHoleFillMaximumDiameterVoxels)) *
                maximum_voxel_size,
            &protected_silhouette_vertices,
            &protected_hole_count,
            true);
        detail::removeDegenerateFaces(&candidate);
        detail::compactReferencedVertices(&candidate);

        const MeshBoundaryTopology topology_after =
            boundaryTopology(candidate);
        const TriangleQualitySummary quality_after =
            triangleQualitySummary(candidate);
        const int added_face_count = std::max(
            0, candidate.faceCount() - face_count_before);
        const int maximum_added_faces = std::max(
            8,
            static_cast<int>(std::ceil(face_count_before * 0.05f)));
        const bool triangle_quality_is_safe =
            quality_after.sliverRatio <=
                std::max(
                    static_cast<double>(
                        options.finalHoleFillMaximumSliverRatio),
                    quality_before.sliverRatio + 1.0e-12);
        const bool accept_residual_fill =
            filled_hole_count > 0 &&
            topology_after.boundaryEdgeCount <
                topology_before.boundaryEdgeCount &&
            topology_after.danglingBoundaryVertexCount <=
                topology_before.danglingBoundaryVertexCount &&
            topology_after.nonManifoldEdgeCount <=
                topology_before.nonManifoldEdgeCount &&
            added_face_count <= maximum_added_faces &&
            triangle_quality_is_safe;
        result.statistics.residualMicroHoleFillProtectedHoleCount =
            protected_hole_count;
        result.statistics.residualMicroHoleFillFilledHoleCount =
            filled_hole_count;
        result.statistics.residualMicroHoleFillAddedFaceCount =
            added_face_count;
        result.statistics.residualMicroHoleFillBoundaryEdgeCountAfter =
            topology_after.boundaryEdgeCount;
        result.statistics.residualMicroHoleFillNonManifoldEdgeCountAfter =
            topology_after.nonManifoldEdgeCount;
        result.statistics.residualMicroHoleFillSliverRatioAfter =
            quality_after.sliverRatio;
        result.statistics.residualMicroHoleFillAccepted =
            accept_residual_fill;
        if (accept_residual_fill)
        {
            result.mesh = std::move(candidate);
            result.statistics.filledBoundaryHoleCount +=
                filled_hole_count;
            result.statistics.addedHoleFillFaceCount +=
                added_face_count;
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    result.statistics.meshSimplificationElapsedMs =
        elapsedMilliseconds(simplification_start);
    result.statistics.boundaryEdgeCountAfter = boundaryEdgeCount(result.mesh);
    result.statistics.postSimplificationFaceCount = result.mesh.faceCount();
    reportProgress(QStringLiteral("网格简化完成，正在重算法线..."), 93);
    detail::recomputeNormals(&result.mesh);
    if (result.mesh.empty())
    {
        result.errorMessage = QStringLiteral("TSDF cleanup removed all mesh components");
        return result;
    }
    if (postprocessCancelled())
    {
        return result;
    }

    const PostprocessClock::time_point colorization_start = PostprocessClock::now();
    if (options.calculateVertexColors)
    {
        reportProgress(QStringLiteral("正在计算网格顶点颜色..."), 94);
        QVector<MeshColorView> color_views;
        color_views.reserve(frames.size());
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame &frame = frames[frame_index];
            MeshColorView view;
            view.camera = frame.camera;
            view.colorBgr = frame.colorBgr;
            view.depth = frame.depth;
            view.confidence = frame.confidence;
            view.depthValidMask = erosion_pixels > 0
                ? effective_depth_valid_masks[frame_index]
                : frame.depthValidMask;
            view.supportMask = frame.supportMask;
            view.qualityWeight = effective_frame_quality_weights[frame_index];
            color_views.push_back(std::move(view));
        }
        MeshColorOptions color_options;
        color_options.maximumVoxelSize = std::max({result.layout.voxelSize[0],
                                                   result.layout.voxelSize[1],
                                                   result.layout.voxelSize[2]});
        color_options.minimumConfidence = options.minimumConfidence;
        color_options.compensateExposure = options.compensateColorExposure;
        color_options.coherentFacePrimaryViews = options.coherentFacePrimaryViewColors;
        result.statistics.effectiveColorExposureCompensation =
            options.compensateColorExposure;
        result.statistics.effectiveCoherentFacePrimaryViewColors =
            options.coherentFacePrimaryViewColors;
        const MeshColorStatistics color_statistics = MeshColorizer::colorize(
            &result.mesh, color_views, color_options);
        result.statistics.colorCandidateObservationCount =
            color_statistics.candidateObservationCount;
        result.statistics.colorRejectedProjectionCount =
            color_statistics.rejectedProjectionCount;
        result.statistics.colorRejectedMaskCount = color_statistics.rejectedMaskCount;
        result.statistics.colorRejectedDepthCount = color_statistics.rejectedDepthCount;
        result.statistics.colorRejectedVisibilityCount =
            color_statistics.rejectedVisibilityCount;
        result.statistics.colorRejectedViewAngleCount =
            color_statistics.rejectedViewAngleCount;
        result.statistics.colorRejectedOutlierCount =
            color_statistics.rejectedColorOutlierCount;
        result.statistics.reliablyColoredVertexCount =
            color_statistics.reliablyColoredVertexCount;
        result.statistics.bestViewFallbackColorVertexCount =
            color_statistics.bestViewFallbackVertexCount;
        result.statistics.propagatedColorVertexCount =
            color_statistics.propagatedVertexCount;
        result.statistics.fallbackColorVertexCount = color_statistics.fallbackVertexCount;
        result.statistics.cleanedColorSpeckleVertexCount =
            color_statistics.cleanedSpeckleVertexCount;
        result.statistics.coherentPrimaryViewFaceCount =
            color_statistics.coherentPrimaryViewFaceCount;
        result.statistics.coherentPrimaryViewVertexCount =
            color_statistics.coherentPrimaryViewVertexCount;
    }
    else
    {
        result.mesh.hasVertexColors = false;
    }
    result.statistics.meshColorizationElapsedMs =
        elapsedMilliseconds(colorization_start);
    if (postprocessCancelled())
    {
        return result;
    }
    reportProgress(QStringLiteral("正在汇总网格质量统计..."), 98);
    MeshTopologyQualityThresholds topology_quality_thresholds;
    topology_quality_thresholds.maximumBoundaryEdgeRatio =
        std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
    topology_quality_thresholds.maximumHighAspectFaceRatio =
        std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
    topology_quality_thresholds.maximumExtremeAspectFaceRatio =
        std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
    const MeshTopologyQualityStatistics topology_quality =
        evaluateMeshTopologyQuality(result.mesh, topology_quality_thresholds);
    result.statistics.topologyQualityUniqueEdgeCount =
        topology_quality.uniqueEdgeCount;
    result.statistics.topologyQualityBoundaryEdgeCount =
        topology_quality.boundaryEdgeCount;
    result.statistics.topologyQualityNonManifoldEdgeCount =
        topology_quality.nonManifoldEdgeCount;
    result.statistics.topologyQualityComponentCount =
        topology_quality.componentCount;
    result.statistics.topologyQualityHighAspectFaceCount =
        topology_quality.highAspectFaceCount;
    result.statistics.topologyQualityExtremeAspectFaceCount =
        topology_quality.extremeAspectFaceCount;
    result.statistics.topologyQualityBoundaryEdgeRatio =
        topology_quality.boundaryEdgeRatio;
    result.statistics.topologyQualityLargestComponentFaceRatio =
        topology_quality.largestComponentFaceRatio;
    result.statistics.topologyQualityHighAspectFaceRatio =
        topology_quality.highAspectFaceRatio;
    result.statistics.topologyQualityExtremeAspectFaceRatio =
        topology_quality.extremeAspectFaceRatio;
    result.statistics.topologyQualityStrictGatePassed =
        topology_quality.strictGatePassed;
    const MeshConnectivityStats connectivity =
        VisualHullReconstructor::analyzeConnectivity(result.mesh);
    result.statistics.componentCount = connectivity.componentCount;
    result.statistics.largestComponentFaceRatio = connectivity.largestComponentFaceRatio;
    result.statistics.componentFaceCounts = connectivity.componentFaceCounts;
    result.statistics.components = connectivity.components;
    result.statistics.vertexCount = result.mesh.vertexCount();
    result.statistics.faceCount = result.mesh.faceCount();
    result.statistics.postIntegrationElapsedMs =
        elapsedMilliseconds(postprocess_start);

    result.ok = true;
    result.errorMessage.clear();
    if (options.progress)
    {
        options.progress(QStringLiteral("TSDF 表面重建完成"), 100);
    }
    return result;
}

QJsonObject DepthTsdfSurfaceBuilder::statisticsToJson(const DepthTsdfResult &result)
{
    const DepthTsdfStatistics &statistics = result.statistics;
    QJsonArray component_face_counts;
    for (const std::size_t face_count : statistics.componentFaceCounts)
    {
        component_face_counts.append(static_cast<qint64>(face_count));
    }
    QJsonArray components;
    for (const MeshConnectivityStats::Component &component : statistics.components)
    {
        QJsonArray bounds_min;
        QJsonArray bounds_max;
        for (int axis = 0; axis < 3; ++axis)
        {
            bounds_min.append(component.boundsMin[axis]);
            bounds_max.append(component.boundsMax[axis]);
        }
        components.append(QJsonObject{
            {QStringLiteral("face_count"), static_cast<qint64>(component.faceCount)},
            {QStringLiteral("bounds_min"), bounds_min},
            {QStringLiteral("bounds_max"), bounds_max},
            {QStringLiteral("diagonal"), component.diagonal}});
    }
    QJsonObject object{
        {QStringLiteral("input_frame_count"), statistics.inputFrameCount},
        {QStringLiteral("accepted_frame_count"), statistics.acceptedFrameCount},
        {QStringLiteral("integrated_voxel_updates"),
         static_cast<double>(statistics.integratedVoxelUpdates)},
        {QStringLiteral("supported_sample_count"),
         static_cast<double>(statistics.supportedSampleCount)},
        {QStringLiteral("single_view_supported_sample_count"),
         static_cast<double>(statistics.singleViewSupportedSampleCount)},
        {QStringLiteral("geometry_verified_single_view_supported_sample_count"),
         static_cast<double>(statistics.geometryVerifiedSingleViewSupportedSampleCount)},
        {QStringLiteral("geometry_single_view_neighborhood_candidate_count"),
         static_cast<double>(statistics.geometrySingleViewNeighborhoodCandidateCount)},
        {QStringLiteral("geometry_single_view_neighborhood_accepted_count"),
         static_cast<double>(statistics.geometrySingleViewNeighborhoodAcceptedCount)},
        {QStringLiteral("geometry_single_view_neighborhood_rejected_count"),
         static_cast<double>(statistics.geometrySingleViewNeighborhoodRejectedCount)},
        {QStringLiteral("multi_view_supported_sample_count"),
         static_cast<double>(statistics.multiViewSupportedSampleCount)},
        {QStringLiteral("rejected_projection_count"),
         static_cast<double>(statistics.rejectedProjectionCount)},
        {QStringLiteral("rejected_support_mask_count"),
         static_cast<double>(statistics.rejectedSupportMaskCount)},
        {QStringLiteral("support_mask_free_space_update_count"),
         static_cast<double>(statistics.supportMaskFreeSpaceUpdateCount)},
        {QStringLiteral("rejected_depth_valid_count"),
         static_cast<double>(statistics.rejectedDepthValidCount)},
        {QStringLiteral("rejected_depth_count"),
         static_cast<double>(statistics.rejectedDepthCount)},
        {QStringLiteral("rejected_confidence_count"),
         static_cast<double>(statistics.rejectedConfidenceCount)},
        {QStringLiteral("subpixel_observation_count"),
         static_cast<double>(statistics.subpixelObservationCount)},
        {QStringLiteral("recovered_neighbor_observation_count"),
         static_cast<double>(statistics.recoveredNeighborObservationCount)},
        {QStringLiteral("discontinuity_rejected_candidate_count"),
         static_cast<double>(statistics.discontinuityRejectedCandidateCount)},
        {QStringLiteral("rejected_geometry_consistency_count"),
         static_cast<double>(statistics.rejectedGeometryConsistencyCount)},
        {QStringLiteral("rejected_invalid_nearest_pixel_recovery_count"),
         static_cast<double>(statistics.rejectedInvalidNearestPixelRecoveryCount)},
        {QStringLiteral("cross_view_consensus_depth_observation_count"),
         static_cast<double>(statistics.crossViewConsensusDepthObservationCount)},
        {QStringLiteral("cross_view_consensus_contour_band_pixel_count"),
         static_cast<double>(statistics.crossViewConsensusContourBandPixelCount)},
        {QStringLiteral("rejected_accumulated_weight_count"),
         static_cast<double>(statistics.rejectedAccumulatedWeightCount)},
        {QStringLiteral("rejected_single_observation_weight_count"),
         static_cast<double>(statistics.rejectedSingleObservationWeightCount)},
        {QStringLiteral("surface_patch_recovered_sample_count"),
         static_cast<double>(statistics.surfacePatchRecoveredSampleCount)},
        {QStringLiteral("surface_patch_executed_growth_pass_count"),
         statistics.surfacePatchExecutedGrowthPassCount},
        {QStringLiteral("surface_patch_considered_sample_count"),
         static_cast<double>(statistics.surfacePatchConsideredSampleCount)},
        {QStringLiteral("surface_patch_rejected_weight_count"),
         static_cast<double>(statistics.surfacePatchRejectedWeightCount)},
        {QStringLiteral("surface_patch_rejected_normal_count"),
         static_cast<double>(statistics.surfacePatchRejectedNormalCount)},
        {QStringLiteral("surface_patch_rejected_source_overlap_count"),
         static_cast<double>(statistics.surfacePatchRejectedSourceOverlapCount)},
        {QStringLiteral("surface_patch_rejected_depth_spread_count"),
         static_cast<double>(statistics.surfacePatchRejectedDepthSpreadCount)},
        {QStringLiteral("surface_patch_rejected_free_space_count"),
         static_cast<double>(statistics.surfacePatchRejectedFreeSpaceCount)},
        {QStringLiteral("surface_patch_created_component_count"),
         statistics.surfacePatchCreatedComponentCount},
        {QStringLiteral("effective_contour_band_zero_crossing_support"),
         statistics.effectiveContourBandZeroCrossingSupport},
        {QStringLiteral("contour_band_zero_crossing_considered_sample_count"),
         static_cast<double>(statistics.contourBandZeroCrossingConsideredSampleCount)},
        {QStringLiteral("contour_band_zero_crossing_recovered_sample_count"),
         static_cast<double>(statistics.contourBandZeroCrossingRecoveredSampleCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_no_contour_count"),
         static_cast<double>(statistics.contourBandZeroCrossingRejectedNoContourCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_no_sign_pair_count"),
         static_cast<double>(statistics.contourBandZeroCrossingRejectedNoSignPairCount)},
        {QStringLiteral("effective_zero_crossing_diagnostics"),
         statistics.effectiveZeroCrossingDiagnostics},
        {QStringLiteral("effective_geometry_zero_crossing_recovery"),
         statistics.effectiveGeometryZeroCrossingRecovery},
        {QStringLiteral("geometry_zero_crossing_candidate_sample_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingCandidateSampleCount)},
        {QStringLiteral("geometry_zero_crossing_recovered_sample_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingRecoveredSampleCount)},
        {QStringLiteral("zero_crossing_observed_cell_count"),
         static_cast<double>(statistics.zeroCrossingObservedCellCount)},
        {QStringLiteral("zero_crossing_raw_candidate_cell_count"),
         static_cast<double>(statistics.zeroCrossingRawCandidateCellCount)},
        {QStringLiteral("zero_crossing_extractable_cell_count"),
         static_cast<double>(statistics.zeroCrossingExtractableCellCount)},
        {QStringLiteral("zero_crossing_suppressed_by_support_cell_count"),
         static_cast<double>(statistics.zeroCrossingSuppressedBySupportCellCount)},
        {QStringLiteral("zero_crossing_positive_only_supported_cell_count"),
         static_cast<double>(statistics.zeroCrossingPositiveOnlySupportedCellCount)},
        {QStringLiteral("zero_crossing_negative_only_supported_cell_count"),
         static_cast<double>(statistics.zeroCrossingNegativeOnlySupportedCellCount)},
        {QStringLiteral("zero_crossing_partially_supported_cell_count"),
         static_cast<double>(statistics.zeroCrossingPartiallySupportedCellCount)},
        {QStringLiteral("zero_crossing_fully_unsupported_observed_cell_count"),
         static_cast<double>(statistics.zeroCrossingFullyUnsupportedObservedCellCount)},
        {QStringLiteral("effective_minimum_voxel_weight"),
         statistics.effectiveMinimumVoxelWeight},
        {QStringLiteral("effective_minimum_single_observation_weight"),
         statistics.effectiveMinimumSingleObservationWeight},
        {QStringLiteral("effective_minimum_geometry_verified_observation_weight"),
         statistics.effectiveMinimumGeometryVerifiedObservationWeight},
        {QStringLiteral("effective_minimum_geometry_support_count"),
         statistics.effectiveMinimumGeometrySupportCount},
        {QStringLiteral("effective_allow_geometry_verified_single_observation"),
         statistics.effectiveAllowGeometryVerifiedSingleObservation},
        {QStringLiteral("effective_geometry_single_view_neighborhood_guard"),
         statistics.effectiveGeometrySingleViewNeighborhoodGuard},
        {QStringLiteral("effective_minimum_geometry_single_view_neighbor_count"),
         statistics.effectiveMinimumGeometrySingleViewNeighborCount},
        {QStringLiteral("effective_geometry_single_view_growth_passes"),
         statistics.effectiveGeometrySingleViewGrowthPasses},
        {QStringLiteral("effective_maximum_geometry_single_view_neighbor_tsdf_delta"),
         statistics.effectiveMaximumGeometrySingleViewNeighborTsdfDelta},
        {QStringLiteral("effective_discontinuity_aware_sampling"),
         statistics.effectiveDiscontinuityAwareSampling},
        {QStringLiteral("effective_maximum_interpolation_relative_depth_spread"),
         statistics.effectiveMaximumInterpolationRelativeDepthSpread},
        {QStringLiteral("effective_maximum_observation_inverse_depth_spread"),
         statistics.effectiveMaximumObservationInverseDepthSpread},
        {QStringLiteral("effective_allow_invalid_nearest_pixel_recovery"),
         statistics.effectiveAllowInvalidNearestPixelRecovery},
        {QStringLiteral(
             "effective_maximum_invalid_nearest_pixel_recovery_inverse_depth_spread"),
         statistics.effectiveMaximumInvalidNearestPixelRecoveryInverseDepthSpread},
        {QStringLiteral("effective_cross_view_consensus_depth"),
         statistics.effectiveCrossViewConsensusDepth},
        {QStringLiteral("effective_maximum_cross_view_consensus_inverse_depth_spread"),
         statistics.effectiveMaximumCrossViewConsensusInverseDepthSpread},
        {QStringLiteral("effective_cross_view_consensus_contour_band_only"),
         statistics.effectiveCrossViewConsensusContourBandOnly},
        {QStringLiteral("effective_robust_frame_quality_weighting"),
         statistics.effectiveRobustFrameQualityWeighting},
        {QStringLiteral("robust_frame_quality_downweighted_frame_count"),
         statistics.robustFrameQualityDownweightedFrameCount},
        {QStringLiteral("robust_frame_quality_median"),
         statistics.robustFrameQualityMedian},
        {QStringLiteral("robust_frame_quality_scale"),
         statistics.robustFrameQualityScale},
        {QStringLiteral("robust_frame_quality_minimum_effective_weight"),
         statistics.robustFrameQualityMinimumEffectiveWeight},
        {QStringLiteral("effective_surface_patch_support"),
         statistics.effectiveSurfacePatchSupport},
        {QStringLiteral("effective_minimum_surface_patch_observation_weight"),
         statistics.effectiveMinimumSurfacePatchObservationWeight},
        {QStringLiteral("effective_minimum_surface_patch_source_count"),
         statistics.effectiveMinimumSurfacePatchSourceCount},
        {QStringLiteral("effective_minimum_surface_patch_core_neighbor_count"),
         statistics.effectiveMinimumSurfacePatchCoreNeighborCount},
        {QStringLiteral("effective_surface_patch_growth_passes"),
         statistics.effectiveSurfacePatchGrowthPasses},
        {QStringLiteral("effective_maximum_surface_patch_inverse_depth_spread"),
         statistics.effectiveMaximumSurfacePatchInverseDepthSpread},
        {QStringLiteral("effective_maximum_surface_patch_normal_angle_degrees"),
         statistics.effectiveMaximumSurfacePatchNormalAngleDegrees},
        {QStringLiteral("effective_maximum_surface_patch_absolute_tsdf"),
         statistics.effectiveMaximumSurfacePatchAbsoluteTsdf},
        {QStringLiteral("effective_minimum_surface_patch_weight_ratio"),
         statistics.effectiveMinimumSurfacePatchWeightRatio},
        {QStringLiteral("effective_minimum_distinct_camera_support"),
         statistics.effectiveMinimumDistinctCameraSupport},
        {QStringLiteral("effective_truncation_voxels"),
         statistics.effectiveTruncationVoxels},
        {QStringLiteral("effective_surface_support_band_voxels"),
         statistics.effectiveSurfaceSupportBandVoxels},
        {QStringLiteral("effective_maximum_free_space_voxels"),
         statistics.effectiveMaximumFreeSpaceVoxels},
        {QStringLiteral("effective_minimum_support_mask_free_space_views"),
         statistics.effectiveMinimumSupportMaskFreeSpaceViews},
        {QStringLiteral("effective_depth_valid_boundary_erosion_pixels"),
         statistics.effectiveDepthValidBoundaryErosionPixels},
        {QStringLiteral("effective_geometry_verified_boundary_recovery"),
         statistics.effectiveGeometryVerifiedBoundaryRecovery},
        {QStringLiteral("effective_minimum_boundary_recovery_geometry_support"),
         statistics.effectiveMinimumBoundaryRecoveryGeometrySupport},
        {QStringLiteral("effective_maximum_boundary_recovery_inverse_depth_spread"),
         statistics.effectiveMaximumBoundaryRecoveryInverseDepthSpread},
        {QStringLiteral("boundary_recovered_depth_valid_pixel_count"),
         static_cast<double>(statistics.boundaryRecoveredDepthValidPixelCount)},
        {QStringLiteral("boundary_edge_count_before"),
         statistics.boundaryEdgeCountBefore},
        {QStringLiteral("boundary_edge_count_after"),
         statistics.boundaryEdgeCountAfter},
        {QStringLiteral("marching_cubes_vertex_count"),
         statistics.marchingCubesVertexCount},
        {QStringLiteral("marching_cubes_face_count"),
         statistics.marchingCubesFaceCount},
        {QStringLiteral("marching_cubes_elapsed_ms"),
         static_cast<qint64>(statistics.marchingCubesElapsedMs)},
        {QStringLiteral("mesh_cleanup_elapsed_ms"),
         static_cast<qint64>(statistics.meshCleanupElapsedMs)},
        {QStringLiteral("mesh_simplification_elapsed_ms"),
         static_cast<qint64>(statistics.meshSimplificationElapsedMs)},
        {QStringLiteral("mesh_colorization_elapsed_ms"),
         static_cast<qint64>(statistics.meshColorizationElapsedMs)},
        {QStringLiteral("post_integration_elapsed_ms"),
         static_cast<qint64>(statistics.postIntegrationElapsedMs)},
        {QStringLiteral("component_filtered_face_count"),
         statistics.componentFilteredFaceCount},
        {QStringLiteral("pre_simplification_face_count"),
         statistics.preSimplificationFaceCount},
        {QStringLiteral("post_simplification_face_count"),
         statistics.postSimplificationFaceCount},
        {QStringLiteral("compacted_unused_vertex_count"),
         statistics.compactedUnusedVertexCount},
        {QStringLiteral("removed_duplicate_face_count"),
         statistics.removedDuplicateFaceCount},
        {QStringLiteral("removed_non_manifold_face_count"),
         statistics.removedNonManifoldFaceCount},
        {QStringLiteral("split_pinched_boundary_vertex_count"),
         statistics.splitPinchedBoundaryVertexCount},
        {QStringLiteral("filled_boundary_hole_count"),
         statistics.filledBoundaryHoleCount},
        {QStringLiteral("added_hole_fill_face_count"),
         statistics.addedHoleFillFaceCount},
        {QStringLiteral("effective_silhouette_aware_final_hole_fill"),
         statistics.effectiveSilhouetteAwareFinalHoleFill},
        {QStringLiteral(
             "effective_visibility_constrained_final_hole_fill"),
         statistics.effectiveVisibilityConstrainedFinalHoleFill},
        {QStringLiteral("visibility_hole_fill_considered_loop_count"),
         statistics.visibilityHoleFillConsideredLoopCount},
        {QStringLiteral("visibility_hole_fill_released_loop_count"),
         statistics.visibilityHoleFillReleasedLoopCount},
        {QStringLiteral(
             "visibility_hole_fill_rejected_support_loop_count"),
         statistics.visibilityHoleFillRejectedSupportLoopCount},
        {QStringLiteral(
             "visibility_hole_fill_rejected_conflict_loop_count"),
         statistics.visibilityHoleFillRejectedConflictLoopCount},
        {QStringLiteral("effective_tiny_boundary_loop_collapse"),
         statistics.effectiveTinyBoundaryLoopCollapse},
        {QStringLiteral("tiny_boundary_loop_collapse_attempted"),
         statistics.tinyBoundaryLoopCollapseAttempted},
        {QStringLiteral("tiny_boundary_loop_collapse_accepted"),
         statistics.tinyBoundaryLoopCollapseAccepted},
        {QStringLiteral("tiny_boundary_loop_collapse_pass_count"),
         statistics.tinyBoundaryLoopCollapsePassCount},
        {QStringLiteral("tiny_boundary_loop_collapsed_edge_count"),
         statistics.tinyBoundaryLoopCollapsedEdgeCount},
        {QStringLiteral("tiny_boundary_loop_boundary_edge_count_before"),
         statistics.tinyBoundaryLoopBoundaryEdgeCountBefore},
        {QStringLiteral("tiny_boundary_loop_boundary_edge_count_after"),
         statistics.tinyBoundaryLoopBoundaryEdgeCountAfter},
        {QStringLiteral(
             "tiny_boundary_loop_non_manifold_edge_count_before"),
         statistics.tinyBoundaryLoopNonManifoldEdgeCountBefore},
        {QStringLiteral(
             "tiny_boundary_loop_non_manifold_edge_count_after"),
         statistics.tinyBoundaryLoopNonManifoldEdgeCountAfter},
        {QStringLiteral("tiny_boundary_loop_high_aspect_ratio_before"),
         statistics.tinyBoundaryLoopHighAspectRatioBefore},
        {QStringLiteral("tiny_boundary_loop_high_aspect_ratio_after"),
         statistics.tinyBoundaryLoopHighAspectRatioAfter},
        {QStringLiteral("final_hole_fill_attempted"),
         statistics.finalHoleFillAttempted},
        {QStringLiteral("final_hole_fill_accepted"),
         statistics.finalHoleFillAccepted},
        {QStringLiteral("final_hole_fill_triangle_quality_rejected"),
         statistics.finalHoleFillTriangleQualityRejected},
        {QStringLiteral("final_hole_fill_post_simplification_attempted"),
         statistics.finalHoleFillPostSimplificationAttempted},
        {QStringLiteral("final_hole_fill_post_simplification_accepted"),
         statistics.finalHoleFillPostSimplificationAccepted},
        {QStringLiteral("final_hole_fill_post_simplification_input_face_count"),
         statistics.finalHoleFillPostSimplificationInputFaceCount},
        {QStringLiteral("final_hole_fill_post_simplification_output_face_count"),
         statistics.finalHoleFillPostSimplificationOutputFaceCount},
        {QStringLiteral("final_hole_fill_post_simplification_collapsed_edge_count"),
         statistics.finalHoleFillPostSimplificationCollapsedEdgeCount},
        {QStringLiteral(
             "final_hole_fill_post_simplification_boundary_edge_count_before"),
         statistics.finalHoleFillPostSimplificationBoundaryEdgeCountBefore},
        {QStringLiteral(
             "final_hole_fill_post_simplification_boundary_edge_count_after"),
         statistics.finalHoleFillPostSimplificationBoundaryEdgeCountAfter},
        {QStringLiteral("final_hole_fill_protected_silhouette_vertex_count"),
         statistics.finalHoleFillProtectedSilhouetteVertexCount},
        {QStringLiteral("final_hole_fill_protected_hole_count"),
         statistics.finalHoleFillProtectedHoleCount},
        {QStringLiteral("final_hole_fill_filled_hole_count"),
         statistics.finalHoleFillFilledHoleCount},
        {QStringLiteral("final_hole_fill_added_face_count"),
         statistics.finalHoleFillAddedFaceCount},
        {QStringLiteral("final_hole_fill_boundary_edge_count_before"),
         statistics.finalHoleFillBoundaryEdgeCountBefore},
        {QStringLiteral("final_hole_fill_boundary_edge_count_after"),
         statistics.finalHoleFillBoundaryEdgeCountAfter},
        {QStringLiteral("final_hole_fill_non_manifold_edge_count_before"),
         statistics.finalHoleFillNonManifoldEdgeCountBefore},
        {QStringLiteral("final_hole_fill_non_manifold_edge_count_after"),
         statistics.finalHoleFillNonManifoldEdgeCountAfter},
        {QStringLiteral("final_hole_fill_sliver_ratio_before"),
         statistics.finalHoleFillSliverRatioBefore},
        {QStringLiteral("final_hole_fill_sliver_ratio_after"),
         statistics.finalHoleFillSliverRatioAfter},
        {QStringLiteral("residual_micro_hole_fill_attempted"),
         statistics.residualMicroHoleFillAttempted},
        {QStringLiteral("residual_micro_hole_fill_accepted"),
         statistics.residualMicroHoleFillAccepted},
        {QStringLiteral("residual_micro_hole_fill_protected_hole_count"),
         statistics.residualMicroHoleFillProtectedHoleCount},
        {QStringLiteral("residual_micro_hole_fill_filled_hole_count"),
         statistics.residualMicroHoleFillFilledHoleCount},
        {QStringLiteral("residual_micro_hole_fill_added_face_count"),
         statistics.residualMicroHoleFillAddedFaceCount},
        {QStringLiteral("residual_micro_hole_fill_boundary_edge_count_before"),
         statistics.residualMicroHoleFillBoundaryEdgeCountBefore},
        {QStringLiteral("residual_micro_hole_fill_boundary_edge_count_after"),
         statistics.residualMicroHoleFillBoundaryEdgeCountAfter},
        {QStringLiteral(
             "residual_micro_hole_fill_non_manifold_edge_count_before"),
         statistics.residualMicroHoleFillNonManifoldEdgeCountBefore},
        {QStringLiteral(
             "residual_micro_hole_fill_non_manifold_edge_count_after"),
         statistics.residualMicroHoleFillNonManifoldEdgeCountAfter},
        {QStringLiteral("residual_micro_hole_fill_sliver_ratio_before"),
         statistics.residualMicroHoleFillSliverRatioBefore},
        {QStringLiteral("residual_micro_hole_fill_sliver_ratio_after"),
         statistics.residualMicroHoleFillSliverRatioAfter},
        {QStringLiteral("smoothed_boundary_vertex_count"),
         statistics.smoothedBoundaryVertexCount},
        {QStringLiteral("smoothed_surface_vertex_count"),
         statistics.smoothedSurfaceVertexCount},
        {QStringLiteral("effective_surface_denoising_iterations"),
         statistics.effectiveSurfaceDenoisingIterations},
        {QStringLiteral("effective_surface_denoising_lambda"),
         statistics.effectiveSurfaceDenoisingLambda},
        {QStringLiteral("effective_maximum_surface_denoising_displacement_voxels"),
         statistics.effectiveMaximumSurfaceDenoisingDisplacementVoxels},
        {QStringLiteral("effective_maximum_surface_denoising_normal_angle_degrees"),
         statistics.effectiveMaximumSurfaceDenoisingNormalAngleDegrees},
        {QStringLiteral("effective_surface_denoising_boundary_protection_rings"),
         statistics.effectiveSurfaceDenoisingBoundaryProtectionRings},
        {QStringLiteral("weak_boundary_tip_vertex_count"),
         statistics.weakBoundaryTipVertexCount},
        {QStringLiteral("candidate_weak_boundary_tip_face_count"),
         statistics.candidateWeakBoundaryTipFaceCount},
        {QStringLiteral("trimmed_weak_boundary_tip_face_count"),
         statistics.trimmedWeakBoundaryTipFaceCount},
        {QStringLiteral("color_candidate_observation_count"),
         static_cast<double>(statistics.colorCandidateObservationCount)},
        {QStringLiteral("color_rejected_projection_count"),
         static_cast<double>(statistics.colorRejectedProjectionCount)},
        {QStringLiteral("color_rejected_mask_count"),
         static_cast<double>(statistics.colorRejectedMaskCount)},
        {QStringLiteral("color_rejected_depth_count"),
         static_cast<double>(statistics.colorRejectedDepthCount)},
        {QStringLiteral("color_rejected_visibility_count"),
         static_cast<double>(statistics.colorRejectedVisibilityCount)},
        {QStringLiteral("color_rejected_view_angle_count"),
         static_cast<double>(statistics.colorRejectedViewAngleCount)},
        {QStringLiteral("color_rejected_outlier_count"),
         static_cast<double>(statistics.colorRejectedOutlierCount)},
        {QStringLiteral("reliably_colored_vertex_count"),
         statistics.reliablyColoredVertexCount},
        {QStringLiteral("best_view_fallback_color_vertex_count"),
         statistics.bestViewFallbackColorVertexCount},
        {QStringLiteral("propagated_color_vertex_count"),
         statistics.propagatedColorVertexCount},
        {QStringLiteral("fallback_color_vertex_count"),
         statistics.fallbackColorVertexCount},
        {QStringLiteral("cleaned_color_speckle_vertex_count"),
         statistics.cleanedColorSpeckleVertexCount},
        {QStringLiteral("effective_color_exposure_compensation"),
         statistics.effectiveColorExposureCompensation},
        {QStringLiteral("effective_coherent_face_primary_view_colors"),
         statistics.effectiveCoherentFacePrimaryViewColors},
        {QStringLiteral("coherent_primary_view_face_count"),
         statistics.coherentPrimaryViewFaceCount},
        {QStringLiteral("coherent_primary_view_vertex_count"),
         statistics.coherentPrimaryViewVertexCount},
        {QStringLiteral("effective_quadric_simplification"),
         statistics.effectiveQuadricSimplification},
        {QStringLiteral("quadric_simplification_accepted"),
         statistics.quadricSimplificationAccepted},
        {QStringLiteral("quadric_simplification_boundary_safety_rejected"),
         statistics.quadricSimplificationBoundarySafetyRejected},
        {QStringLiteral("quadric_boundary_edge_count_before"),
         statistics.quadricBoundaryEdgeCountBefore},
        {QStringLiteral("quadric_boundary_edge_count_after"),
         statistics.quadricBoundaryEdgeCountAfter},
        {QStringLiteral("quadric_dangling_boundary_vertex_count_before"),
         statistics.quadricDanglingBoundaryVertexCountBefore},
        {QStringLiteral("quadric_dangling_boundary_vertex_count_after"),
         statistics.quadricDanglingBoundaryVertexCountAfter},
        {QStringLiteral("quadric_non_manifold_edge_count_before"),
         statistics.quadricNonManifoldEdgeCountBefore},
        {QStringLiteral("quadric_non_manifold_edge_count_after"),
         statistics.quadricNonManifoldEdgeCountAfter},
        {QStringLiteral("requested_simplify_target_faces"),
         statistics.requestedSimplifyTargetFaces},
        {QStringLiteral("quadric_collapsed_edge_count"),
         statistics.quadricCollapsedEdgeCount},
        {QStringLiteral("quadric_rejected_boundary_edge_count"),
         statistics.quadricRejectedBoundaryEdgeCount},
        {QStringLiteral("quadric_rejected_feature_edge_count"),
         statistics.quadricRejectedFeatureEdgeCount},
        {QStringLiteral("quadric_rejected_topology_edge_count"),
         statistics.quadricRejectedTopologyEdgeCount},
        {QStringLiteral("quadric_rejected_flip_edge_count"),
         statistics.quadricRejectedFlipEdgeCount},
        {QStringLiteral("quadric_simplify_pass_count"),
         statistics.quadricSimplifyPassCount},
        {QStringLiteral("quadric_simplify_reached_target"),
         statistics.quadricSimplifyReachedTarget},
        {QStringLiteral("quadric_simplify_stopped_by_stagnation"),
         statistics.quadricSimplifyStoppedByStagnation},
        {QStringLiteral("voxel_fallback_attempted"),
         statistics.voxelFallbackAttempted},
        {QStringLiteral("effective_voxel_fallback_simplification"),
         statistics.effectiveVoxelFallbackSimplification},
        {QStringLiteral("effective_voxel_fallback_qem_polish"),
         statistics.effectiveVoxelFallbackQemPolish},
        {QStringLiteral("effective_voxel_fallback_minimum_protected_boundary_vertices"),
         statistics.effectiveVoxelFallbackMinimumProtectedBoundaryVertices},
        {QStringLiteral(
             "effective_voxel_fallback_maximum_collapsible_boundary_diameter_voxels"),
         statistics.effectiveVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels},
        {QStringLiteral(
             "effective_voxel_fallback_maximum_normal_cluster_angle_degrees"),
         statistics.effectiveVoxelFallbackMaximumNormalClusterAngleDegrees},
        {QStringLiteral("effective_voxel_fallback_initial_cluster_factor"),
         statistics.effectiveVoxelFallbackInitialClusterFactor},
        {QStringLiteral(
             "effective_voxel_fallback_multi_view_silhouette_protection"),
         statistics.effectiveVoxelFallbackMultiViewSilhouetteProtection},
        {QStringLiteral("effective_voxel_fallback_minimum_silhouette_views"),
         statistics.effectiveVoxelFallbackMinimumSilhouetteViews},
        {QStringLiteral("effective_voxel_fallback_silhouette_band_pixels"),
         statistics.effectiveVoxelFallbackSilhouetteBandPixels},
        {QStringLiteral(
             "effective_voxel_fallback_silhouette_depth_tolerance_voxels"),
         statistics.effectiveVoxelFallbackSilhouetteDepthToleranceVoxels},
        {QStringLiteral("voxel_fallback_protected_silhouette_vertex_count"),
         statistics.voxelFallbackProtectedSilhouetteVertexCount},
        {QStringLiteral("voxel_fallback_sliver_face_count_before"),
         statistics.voxelFallbackSliverFaceCountBefore},
        {QStringLiteral("voxel_fallback_sliver_face_count_after"),
         statistics.voxelFallbackSliverFaceCountAfter},
        {QStringLiteral("voxel_fallback_sliver_ratio_before"),
         statistics.voxelFallbackSliverRatioBefore},
        {QStringLiteral("voxel_fallback_sliver_ratio_after"),
         statistics.voxelFallbackSliverRatioAfter},
        {QStringLiteral("voxel_fallback_triangle_quality_rejected"),
         statistics.voxelFallbackTriangleQualityRejected},
        {QStringLiteral("voxel_fallback_qem_polish_attempted"),
         statistics.voxelFallbackQemPolishAttempted},
        {QStringLiteral("voxel_fallback_qem_polish_accepted"),
         statistics.voxelFallbackQemPolishAccepted},
        {QStringLiteral("voxel_fallback_qem_polish_input_face_count"),
         statistics.voxelFallbackQemPolishInputFaceCount},
        {QStringLiteral("voxel_fallback_qem_polish_output_face_count"),
         statistics.voxelFallbackQemPolishOutputFaceCount},
        {QStringLiteral("voxel_fallback_qem_polish_collapsed_edge_count"),
         statistics.voxelFallbackQemPolishCollapsedEdgeCount},
        {QStringLiteral("voxel_fallback_accepted"),
         statistics.voxelFallbackAccepted},
        {QStringLiteral("voxel_fallback_preserved_open_boundaries"),
         statistics.voxelFallbackPreservedOpenBoundaries},
        {QStringLiteral("voxel_fallback_input_face_count"),
         statistics.voxelFallbackInputFaceCount},
        {QStringLiteral("voxel_fallback_output_face_count"),
         statistics.voxelFallbackOutputFaceCount},
        {QStringLiteral("voxel_fallback_boundary_edge_count_before"),
         statistics.voxelFallbackBoundaryEdgeCountBefore},
        {QStringLiteral("voxel_fallback_boundary_edge_count_after"),
         statistics.voxelFallbackBoundaryEdgeCountAfter},
        {QStringLiteral("voxel_fallback_non_manifold_edge_count_after"),
         statistics.voxelFallbackNonManifoldEdgeCountAfter},
        {QStringLiteral("effective_triangle_quality_optimization"),
         statistics.effectiveTriangleQualityOptimization},
        {QStringLiteral("triangle_quality_optimization_attempted"),
         statistics.triangleQualityOptimizationAttempted},
        {QStringLiteral("triangle_quality_optimization_accepted"),
         statistics.triangleQualityOptimizationAccepted},
        {QStringLiteral("triangle_quality_optimization_pass_count"),
         statistics.triangleQualityOptimizationPassCount},
        {QStringLiteral("triangle_quality_optimization_flipped_edge_count"),
         statistics.triangleQualityOptimizationFlippedEdgeCount},
        {QStringLiteral(
             "triangle_quality_tangential_relaxation_pass_count"),
         statistics.triangleQualityTangentialRelaxationPassCount},
        {QStringLiteral(
             "triangle_quality_tangential_relaxed_vertex_count"),
         statistics.triangleQualityTangentialRelaxedVertexCount},
        {QStringLiteral(
             "triangle_quality_isotropic_remeshing_pass_count"),
         statistics.triangleQualityIsotropicRemeshingPassCount},
        {QStringLiteral(
             "triangle_quality_isotropic_collapsed_edge_count"),
         statistics.triangleQualityIsotropicCollapsedEdgeCount},
        {QStringLiteral(
             "triangle_quality_isotropic_split_edge_count"),
         statistics.triangleQualityIsotropicSplitEdgeCount},
        {QStringLiteral("triangle_quality_optimization_input_face_count"),
         statistics.triangleQualityOptimizationInputFaceCount},
        {QStringLiteral("triangle_quality_optimization_output_face_count"),
         statistics.triangleQualityOptimizationOutputFaceCount},
        {QStringLiteral("triangle_quality_high_aspect_face_ratio_before"),
         statistics.triangleQualityHighAspectFaceRatioBefore},
        {QStringLiteral("triangle_quality_high_aspect_face_ratio_after"),
         statistics.triangleQualityHighAspectFaceRatioAfter},
        {QStringLiteral("triangle_quality_extreme_aspect_face_ratio_before"),
         statistics.triangleQualityExtremeAspectFaceRatioBefore},
        {QStringLiteral("triangle_quality_extreme_aspect_face_ratio_after"),
         statistics.triangleQualityExtremeAspectFaceRatioAfter},
        {QStringLiteral("topology_quality_unique_edge_count"),
         statistics.topologyQualityUniqueEdgeCount},
        {QStringLiteral("topology_quality_boundary_edge_count"),
         statistics.topologyQualityBoundaryEdgeCount},
        {QStringLiteral("topology_quality_non_manifold_edge_count"),
         statistics.topologyQualityNonManifoldEdgeCount},
        {QStringLiteral("topology_quality_component_count"),
         statistics.topologyQualityComponentCount},
        {QStringLiteral("topology_quality_high_aspect_face_count"),
         statistics.topologyQualityHighAspectFaceCount},
        {QStringLiteral("topology_quality_extreme_aspect_face_count"),
         statistics.topologyQualityExtremeAspectFaceCount},
        {QStringLiteral("topology_quality_boundary_edge_ratio"),
         statistics.topologyQualityBoundaryEdgeRatio},
        {QStringLiteral("topology_quality_largest_component_face_ratio"),
         statistics.topologyQualityLargestComponentFaceRatio},
        {QStringLiteral("topology_quality_high_aspect_face_ratio"),
         statistics.topologyQualityHighAspectFaceRatio},
        {QStringLiteral("topology_quality_extreme_aspect_face_ratio"),
         statistics.topologyQualityExtremeAspectFaceRatio},
        {QStringLiteral("topology_quality_strict_gate_passed"),
         statistics.topologyQualityStrictGatePassed},
        {QStringLiteral("vertex_count"), statistics.vertexCount},
        {QStringLiteral("face_count"), statistics.faceCount},
        {QStringLiteral("component_count"), statistics.componentCount},
        {QStringLiteral("largest_component_face_ratio"), statistics.largestComponentFaceRatio},
        {QStringLiteral("component_face_counts"), component_face_counts},
        {QStringLiteral("components"), components}
    };
    return object;
}

} // namespace xjw::mesh
