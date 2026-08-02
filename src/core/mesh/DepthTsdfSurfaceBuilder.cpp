#include "DepthTsdfSurfaceBuilder.h"

#include "DepthTsdfNarrowBandActivation.h"
#include "AdaptiveTsdfOctree.h"
#include "ConsistentIsoSurfaceExtractor.h"
#include "DepthConsensusBiasPolicy.h"
#include "DepthFrameUtils.h"
#include "DepthFusionFramePolicy.h"
#include "DepthMeshCompleteness.h"
#include "DepthImplicitFieldRegularizer.h"
#include "DepthTsdfCellSheetRecovery.h"
#include "DepthVisibilityHistogram.h"
#include "MeshBoundaryAttribution.h"
#include "MeshColorizer.h"
#include "MeshFaceOrientation.h"
#include "MeshQuadricSimplifier.h"
#include "OpenMeshSimplifier.h"
#include "MeshTopologyQuality.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "SparseTgvSolver.h"
#include "SurfaceReconstructorPostprocess.h"
#include "VisibilityOccupancySurfaceBuilder.h"
#include "VisibilityOccupancyBoundaryExtractor.h"
#include "VisibilityOccupancyDistanceField.h"
#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyTsdfCompletion.h"
#include "io/PathIO.h"

#include <QJsonArray>
#include <QFileInfo>
#include <QSet>

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
#include <utility>
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
    sizeof(float) * 5 + sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t);
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

double meshSurfaceArea(const TriMesh &mesh)
{
    double area = 0.0;
    for (const Triangle &face : mesh.faces)
    {
        if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 ||
            face.v[0] >= mesh.vertexCount() ||
            face.v[1] >= mesh.vertexCount() ||
            face.v[2] >= mesh.vertexCount())
        {
            continue;
        }
        const MeshVertex &first =
            mesh.vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &second =
            mesh.vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &third =
            mesh.vertices[static_cast<std::size_t>(face.v[2])];
        const double ab_x = static_cast<double>(second.x) - first.x;
        const double ab_y = static_cast<double>(second.y) - first.y;
        const double ab_z = static_cast<double>(second.z) - first.z;
        const double ac_x = static_cast<double>(third.x) - first.x;
        const double ac_y = static_cast<double>(third.y) - first.y;
        const double ac_z = static_cast<double>(third.z) - first.z;
        const double cross_x = ab_y * ac_z - ab_z * ac_y;
        const double cross_y = ab_z * ac_x - ab_x * ac_z;
        const double cross_z = ab_x * ac_y - ab_y * ac_x;
        const double twice_area = std::sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        if (std::isfinite(twice_area))
        {
            area += 0.5 * twice_area;
        }
    }
    return area;
}

double meshBoundsDiagonal(const TriMesh &mesh)
{
    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    bool has_finite_vertex = false;
    for (const MeshVertex &vertex : mesh.vertices)
    {
        const std::array<double, 3> position{vertex.x, vertex.y, vertex.z};
        if (!std::isfinite(position[0]) || !std::isfinite(position[1]) ||
            !std::isfinite(position[2]))
        {
            continue;
        }
        has_finite_vertex = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }
    if (!has_finite_vertex)
    {
        return 0.0;
    }
    const double dx = maximum[0] - minimum[0];
    const double dy = maximum[1] - minimum[1];
    const double dz = maximum[2] - minimum[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct ComparableIsoSurfaceExtraction
{
    bool ok = false;
    QString errorMessage;
    TriMesh mesh;
    std::uint64_t mc33SupportMaskedSampleCount = 0;
    std::uint64_t mc33RejectedUnsupportedCellFaceCount = 0;
    std::uint64_t isoSurfaceAmbiguousFaceCount = 0;
    std::uint64_t isoSurfaceTopologyAdjustedCellCount = 0;
    std::uint64_t isoSurfaceDeciderTieCount = 0;
    std::uint64_t isoSurfaceMultipleLoopCellCount = 0;
    std::uint64_t isoSurfaceEdgeVertexCacheHitCount = 0;
    std::uint64_t isoSurfaceEdgeVertexCacheMissCount = 0;
    std::uint64_t isoSurfaceInteriorLoopVertexCount = 0;
    std::uint64_t isoSurfaceRejectedDegenerateFaceCount = 0;
    std::uint64_t isoSurfaceUnresolvedCellCount = 0;
};

ComparableIsoSurfaceExtraction extractComparableIsoSurface(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &support,
    bool useMc33,
    bool requireSupportedSignChange,
    bool useConsistentExtractor,
    const std::function<bool()> &isCancelled)
{
    ComparableIsoSurfaceExtraction result;
    if (useMc33)
    {
        Mc33IsoSurfaceOptions options;
        options.isoLevel = 0.0f;
        options.requireSupportedSignChange = requireSupportedSignChange;
        options.isCancelled = isCancelled;
        Mc33IsoSurfaceResult extracted = Mc33IsoSurfaceExtractor::extract(
            boundsMin, boundsMax, cells, field, support, options);
        result.ok = extracted.ok;
        result.errorMessage = QString::fromStdString(extracted.errorMessage);
        result.mesh = std::move(extracted.mesh);
        result.mc33SupportMaskedSampleCount =
            extracted.statistics.supportMaskedSampleCount;
        result.mc33RejectedUnsupportedCellFaceCount =
            extracted.statistics.rejectedUnsupportedCellFaceCount;
        return result;
    }
    if (useConsistentExtractor)
    {
        ConsistentIsoSurfaceOptions options;
        options.isoLevel = 0.0f;
        options.isCancelled = isCancelled;
        ConsistentIsoSurfaceResult extracted =
            ConsistentIsoSurfaceExtractor::extract(
                boundsMin, boundsMax, cells, field, support, options);
        result.ok = extracted.ok;
        result.errorMessage = QString::fromStdString(extracted.errorMessage);
        result.mesh = std::move(extracted.mesh);
        result.isoSurfaceAmbiguousFaceCount =
            extracted.statistics.uniqueAmbiguousFaceCount;
        result.isoSurfaceTopologyAdjustedCellCount =
            extracted.statistics.topologyAdjustedCellCount;
        result.isoSurfaceDeciderTieCount = extracted.statistics.deciderTieCount;
        result.isoSurfaceMultipleLoopCellCount =
            extracted.statistics.multipleLoopCellCount;
        result.isoSurfaceEdgeVertexCacheHitCount =
            extracted.statistics.edgeVertexCacheHitCount;
        result.isoSurfaceEdgeVertexCacheMissCount =
            extracted.statistics.edgeVertexCacheMissCount;
        result.isoSurfaceInteriorLoopVertexCount =
            extracted.statistics.interiorLoopVertexCount;
        result.isoSurfaceRejectedDegenerateFaceCount =
            extracted.statistics.rejectedDegenerateFaceCount;
        result.isoSurfaceUnresolvedCellCount =
            extracted.statistics.unresolvedCellCount;
        return result;
    }

    plapoint::mesh::MarchingCubes<float> marching_cubes;
    marching_cubes.setBounds(
        {boundsMin[0], boundsMin[1], boundsMin[2]},
        {boundsMax[0], boundsMax[1], boundsMax[2]});
    marching_cubes.setResolution(cells[0], cells[1], cells[2]);
    marching_cubes.setIsoLevel(0.0f);
    auto [vertices, faces] = marching_cubes.extract(
        [&](float x, float y, float z)
        {
            const int ix = std::clamp(
                static_cast<int>(std::lround(
                    (x - boundsMin[0]) * cells[0] /
                    (boundsMax[0] - boundsMin[0]))),
                0,
                cells[0]);
            const int iy = std::clamp(
                static_cast<int>(std::lround(
                    (y - boundsMin[1]) * cells[1] /
                    (boundsMax[1] - boundsMin[1]))),
                0,
                cells[1]);
            const int iz = std::clamp(
                static_cast<int>(std::lround(
                    (z - boundsMin[2]) * cells[2] /
                    (boundsMax[2] - boundsMin[2]))),
                0,
                cells[2]);
            const std::size_t index =
                (static_cast<std::size_t>(iz) *
                     static_cast<std::size_t>(cells[1] + 1) +
                 static_cast<std::size_t>(iy)) *
                    static_cast<std::size_t>(cells[0] + 1) +
                static_cast<std::size_t>(ix);
            return support[index] != 0 ? field[index] : 1.0f;
        });
    result.mesh.vertices.resize(static_cast<std::size_t>(vertices.rows()));
    for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
    {
        MeshVertex &vertex =
            result.mesh.vertices[static_cast<std::size_t>(row)];
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
    result.ok = true;
    return result;
}

bool meshVerticesInsidePaddedLayout(const TriMesh &mesh,
                                    const DepthTsdfLayout &layout,
                                    float paddingVoxels = 2.0f)
{
    const float maximum_voxel_size = std::max(
        {layout.voxelSize[0], layout.voxelSize[1], layout.voxelSize[2]});
    const float padding = std::max(0.0f, paddingVoxels) * maximum_voxel_size;
    for (const MeshVertex &vertex : mesh.vertices)
    {
        const std::array<float, 3> position{{vertex.x, vertex.y, vertex.z}};
        for (int axis = 0; axis < 3; ++axis)
        {
            const float value = position[static_cast<std::size_t>(axis)];
            if (!std::isfinite(value) ||
                value < layout.boundsMin[axis] - padding ||
                value > layout.boundsMax[axis] + padding)
            {
                return false;
            }
        }
    }
    return true;
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

struct HoleFillPatchEvidenceResult
{
    bool attempted = false;
    bool accepted = false;
    int vertexSampleCount = 0;
    int faceCenterSampleCount = 0;
    int acceptedSampleCount = 0;
    int rejectedSupportSampleCount = 0;
    int rejectedConflictSampleCount = 0;
    int minimumSupportingViewCount = 0;
    int maximumConflictViewCount = 0;
};

std::pair<int, int> holeFillPatchSampleViewEvidence(
    const MeshVertex &sample,
    const QVector<DepthTsdfFrame> &frames,
    const QVector<cv::Mat> &effectiveDepthValidMasks,
    float absoluteDepthTolerance,
    float minimumConfidence)
{
    int supporting_views = 0;
    int conflict_views = 0;
    const double world[3] = {sample.x, sample.y, sample.z};
    for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
    {
        const DepthTsdfFrame &frame = frames[frame_index];
        if (frame.depth.empty() || frame.depth.type() != CV_32FC1 ||
            frame.confidence.empty() || frame.confidence.type() != CV_32FC1 ||
            frame.supportMask.empty() || frame.supportMask.type() != CV_8UC1 ||
            frame.depth.size() != frame.confidence.size() ||
            frame.depth.size() != frame.supportMask.size())
        {
            continue;
        }
        const cv::Mat &valid_mask =
            effectiveDepthValidMasks.size() == frames.size()
            ? effectiveDepthValidMasks[frame_index]
            : frame.depthValidMask;
        if (valid_mask.empty() || valid_mask.type() != CV_8UC1 ||
            valid_mask.size() != frame.depth.size())
        {
            continue;
        }

        double pixel[2]{};
        double camera_depth = 0.0;
        if (!frame.camera.projectWorldPointWithDepth(
                world, pixel, camera_depth) ||
            !std::isfinite(camera_depth) || camera_depth <= 0.0)
        {
            continue;
        }
        const int column = static_cast<int>(std::lround(pixel[0]));
        const int row = static_cast<int>(std::lround(pixel[1]));
        if (row < 0 || column < 0 || row >= frame.depth.rows ||
            column >= frame.depth.cols)
        {
            continue;
        }
        if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
        {
            ++conflict_views;
            continue;
        }
        if (valid_mask.at<std::uint8_t>(row, column) == 0)
        {
            continue;
        }

        const float observed_depth = frame.depth.at<float>(row, column);
        const float confidence = frame.confidence.at<float>(row, column);
        if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
            !std::isfinite(confidence) || confidence < minimumConfidence)
        {
            continue;
        }
        const float projected_depth = static_cast<float>(camera_depth);
        const float tolerance = std::max(
            absoluteDepthTolerance, 0.008f * std::abs(projected_depth));
        if (std::abs(observed_depth - projected_depth) <= tolerance)
        {
            ++supporting_views;
        }
        else if (projected_depth < observed_depth - tolerance)
        {
            // The proposed patch lies in front of an observed surface and
            // therefore occupies free space in this view.  A sample behind
            // the observed surface is merely occluded and is not a conflict.
            ++conflict_views;
        }
    }
    return {supporting_views, conflict_views};
}

HoleFillPatchEvidenceResult validateAddedHoleFillPatchEvidence(
    const TriMesh &candidate,
    std::size_t baselineVertexCount,
    std::size_t baselineFaceCount,
    const QVector<DepthTsdfFrame> &frames,
    const QVector<cv::Mat> &effectiveDepthValidMasks,
    int minimumSupportingViews,
    int maximumConflictViews,
    float depthToleranceVoxels,
    float maximumVoxelSize,
    float minimumConfidence)
{
    HoleFillPatchEvidenceResult result;
    if (baselineVertexCount > candidate.vertices.size() ||
        baselineFaceCount > candidate.faces.size())
    {
        return result;
    }

    result.vertexSampleCount = static_cast<int>(
        candidate.vertices.size() - baselineVertexCount);
    result.faceCenterSampleCount = static_cast<int>(
        candidate.faces.size() - baselineFaceCount);
    const int sample_count =
        result.vertexSampleCount + result.faceCenterSampleCount;
    result.attempted = sample_count > 0;
    if (!result.attempted)
    {
        return result;
    }

    const float absolute_tolerance =
        std::max(1.0f, depthToleranceVoxels) *
        std::max(maximumVoxelSize, 1.0e-8f);
    result.minimumSupportingViewCount = std::numeric_limits<int>::max();
    const auto validate_sample = [&](const MeshVertex &sample)
    {
        const auto [supporting_views, conflict_views] =
            holeFillPatchSampleViewEvidence(
                sample,
                frames,
                effectiveDepthValidMasks,
                absolute_tolerance,
                minimumConfidence);
        result.minimumSupportingViewCount = std::min(
            result.minimumSupportingViewCount, supporting_views);
        result.maximumConflictViewCount = std::max(
            result.maximumConflictViewCount, conflict_views);
        if (supporting_views < std::max(2, minimumSupportingViews))
        {
            ++result.rejectedSupportSampleCount;
        }
        if (conflict_views > std::max(0, maximumConflictViews))
        {
            ++result.rejectedConflictSampleCount;
        }
        if (DepthTsdfSurfaceBuilder::shouldAcceptFinalHoleFillPatchSample(
                supporting_views,
                conflict_views,
                minimumSupportingViews,
                maximumConflictViews))
        {
            ++result.acceptedSampleCount;
        }
    };

    for (std::size_t vertex_index = baselineVertexCount;
         vertex_index < candidate.vertices.size();
         ++vertex_index)
    {
        validate_sample(candidate.vertices[vertex_index]);
    }
    for (std::size_t face_index = baselineFaceCount;
         face_index < candidate.faces.size();
         ++face_index)
    {
        const Triangle &face = candidate.faces[face_index];
        const MeshVertex &first = candidate.vertices[
            static_cast<std::size_t>(face.v[0])];
        const MeshVertex &second = candidate.vertices[
            static_cast<std::size_t>(face.v[1])];
        const MeshVertex &third = candidate.vertices[
            static_cast<std::size_t>(face.v[2])];
        MeshVertex center;
        center.x = (first.x + second.x + third.x) / 3.0f;
        center.y = (first.y + second.y + third.y) / 3.0f;
        center.z = (first.z + second.z + third.z) / 3.0f;
        validate_sample(center);
    }

    result.accepted = result.acceptedSampleCount == sample_count;
    if (result.minimumSupportingViewCount == std::numeric_limits<int>::max())
    {
        result.minimumSupportingViewCount = 0;
    }
    return result;
}

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
            // Visibility-constrained filling is fail-closed for every eligible
            // boundary loop.  Missing a silhouette tag is not positive evidence
            // that an opening is safe to cap.
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

            const bool release_hole =
                DepthTsdfSurfaceBuilder::shouldReleaseVisibilityConstrainedHole(
                    static_cast<int>(loop.size()),
                    silhouette_vertex_count,
                    supporting_views,
                    conflict_views,
                    required_supporting_views,
                    allowed_conflict_views,
                    strong_ratio);
            if (release_hole)
            {
                for (const int vertex : loop)
                {
                    result.protectedVertices[
                        static_cast<std::size_t>(vertex)] = 0;
                }
                ++result.releasedLoopCount;
            }
            else
            {
                for (const int vertex : loop)
                {
                    result.protectedVertices[
                        static_cast<std::size_t>(vertex)] = 1;
                }
                if (conflict_views > allowed_conflict_views)
                {
                    ++result.rejectedConflictLoopCount;
                }
                else
                {
                    ++result.rejectedSupportLoopCount;
                }
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

bool hasOnlyRecoverableOrbitalQualityReasons(const DepthFrameArtifact &artifact)
{
    if (artifact.qualityReasons.isEmpty())
    {
        return false;
    }

    static const QSet<QString> recoverableReasons{
        QStringLiteral("insufficient_mask_normalized_coverage"),
        QStringLiteral("depth_consistency_collapse"),
        QStringLiteral("depth_consistency_coverage_loss"),
        QStringLiteral("weak_multiview_consistency")
    };
    return std::all_of(
        artifact.qualityReasons.cbegin(),
        artifact.qualityReasons.cend(),
        [](const QString &reason)
        {
            return recoverableReasons.contains(reason);
        });
}

bool canUseRejectedOrbitalFrameAsAuxiliary(const DepthFrameArtifact &artifact)
{
    if (artifact.acceptance != QStringLiteral("rejected")
        || artifact.sceneProfile != QStringLiteral("orbital_object")
        || artifact.validCoverage < 0.02
        || artifact.validWithinMaskRatio < 0.10
        || artifact.consistencyRetentionRatio < 0.10
        || artifact.largestComponentRatio < 0.15
        || artifact.meanConfidence < 0.45
        || artifact.sourceViewCount < 2)
    {
        return false;
    }

    return hasOnlyRecoverableOrbitalQualityReasons(artifact);
}

bool canPromoteHealthyOrbitalValidationFrame(const DepthFrameArtifact &artifact)
{
    return artifact.acceptance == QStringLiteral("validation_only")
        && artifact.sceneProfile == QStringLiteral("orbital_object")
        && artifact.validCoverage >= 0.02
        && artifact.validWithinMaskRatio >= 0.90
        && artifact.consistencyRetentionRatio >= 0.90
        && artifact.largestComponentRatio >= 0.15
        && artifact.meanConfidence >= 0.45
        && artifact.sourceViewCount >= 2
        && hasOnlyRecoverableOrbitalQualityReasons(artifact);
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
              QString *reason,
              bool allow_resize = false)
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
    if (mask->type() == CV_8UC1 && mask->size() != size && allow_resize)
    {
        cv::resize(*mask, *mask, size, 0.0, 0.0, cv::INTER_NEAREST);
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

struct DepthUncertaintyBandEstimate
{
    std::uint64_t sampleCount = 0;
    float p90Voxels = 0.0f;
};

DepthUncertaintyBandEstimate estimateDepthUncertaintyBand(
    const QVector<DepthTsdfFrame> &frames,
    const DepthTsdfOptions &options,
    float maximum_voxel_size)
{
    DepthUncertaintyBandEstimate estimate;
    if (!options.enableUncertaintyAdaptiveTruncation ||
        !std::isfinite(maximum_voxel_size) ||
        maximum_voxel_size <= std::numeric_limits<float>::epsilon())
    {
        return estimate;
    }

    const int maximum_samples_per_frame =
        std::max(256, options.uncertaintyAdaptiveMaximumSamplesPerFrame);
    std::vector<float> uncertainty_voxels;
    uncertainty_voxels.reserve(
        static_cast<std::size_t>(frames.size()) *
        static_cast<std::size_t>(maximum_samples_per_frame));
    for (const DepthTsdfFrame &frame : frames)
    {
        if (frame.depth.type() != CV_32FC1 ||
            frame.inverseDepthRelativeSpread.type() != CV_32FC1 ||
            frame.depth.size() != frame.inverseDepthRelativeSpread.size())
        {
            continue;
        }
        const bool use_confidence =
            frame.confidence.type() == CV_32FC1 &&
            frame.confidence.size() == frame.depth.size();
        const bool use_depth_valid =
            frame.depthValidMask.type() == CV_8UC1 &&
            frame.depthValidMask.size() == frame.depth.size();
        const bool use_support =
            frame.supportMask.type() == CV_8UC1 &&
            frame.supportMask.size() == frame.depth.size();
        const int pixel_count = frame.depth.rows * frame.depth.cols;
        const int sampling_step = std::max(
            1,
            static_cast<int>(std::ceil(std::sqrt(
                static_cast<double>(pixel_count) /
                static_cast<double>(maximum_samples_per_frame)))));
        const int sampling_offset = sampling_step / 2;
        for (int row = sampling_offset; row < frame.depth.rows;
             row += sampling_step)
        {
            const float *depth_row = frame.depth.ptr<float>(row);
            const float *spread_row =
                frame.inverseDepthRelativeSpread.ptr<float>(row);
            const float *confidence_row =
                use_confidence ? frame.confidence.ptr<float>(row) : nullptr;
            const std::uint8_t *depth_valid_row =
                use_depth_valid ? frame.depthValidMask.ptr<std::uint8_t>(row)
                                : nullptr;
            const std::uint8_t *support_row =
                use_support ? frame.supportMask.ptr<std::uint8_t>(row)
                            : nullptr;
            for (int column = sampling_offset; column < frame.depth.cols;
                 column += sampling_step)
            {
                if ((depth_valid_row && depth_valid_row[column] == 0) ||
                    (support_row && support_row[column] == 0) ||
                    (confidence_row &&
                     confidence_row[column] < options.minimumConfidence))
                {
                    continue;
                }
                const float depth = depth_row[column];
                const float relative_spread = spread_row[column];
                if (!std::isfinite(depth) || depth <= 0.0f ||
                    !std::isfinite(relative_spread) ||
                    relative_spread <= 0.0f)
                {
                    continue;
                }
                if (options.maximumObservationInverseDepthSpread > 0.0f &&
                    relative_spread >
                        options.maximumObservationInverseDepthSpread)
                {
                    continue;
                }
                const float uncertainty =
                    depth * relative_spread / maximum_voxel_size;
                if (std::isfinite(uncertainty) && uncertainty > 0.0f)
                {
                    uncertainty_voxels.push_back(uncertainty);
                }
            }
        }
    }
    estimate.sampleCount = uncertainty_voxels.size();
    if (estimate.sampleCount <
        static_cast<std::uint64_t>(
            std::max(64, options.uncertaintyAdaptiveMinimumSampleCount)))
    {
        return estimate;
    }
    const std::size_t p90_index = static_cast<std::size_t>(
        std::floor(0.90 * static_cast<double>(
            uncertainty_voxels.size() - 1)));
    std::nth_element(
        uncertainty_voxels.begin(),
        uncertainty_voxels.begin() + static_cast<std::ptrdiff_t>(p90_index),
        uncertainty_voxels.end());
    estimate.p90Voxels = uncertainty_voxels[p90_index];
    return estimate;
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
                                           const std::vector<std::uint8_t> &strongAdaptiveSupport,
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
            const std::size_t sample_index = sampleIndex(layout, x, y, z);
            const bool has_strong_adaptive_support =
                sample_index < strongAdaptiveSupport.size() &&
                strongAdaptiveSupport[sample_index] != 0;
            if (support[sample_index] < minimum_support &&
                !has_strong_adaptive_support)
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

bool DepthTsdfSurfaceBuilder::shouldReleaseVisibilityConstrainedHole(
    int loopVertexCount,
    int silhouetteProtectedVertexCount,
    int supportingViewCount,
    int conflictViewCount,
    int minimumSupportingViews,
    int maximumConflictViews,
    float strongSilhouetteRatio)
{
    if (loopVertexCount < 3)
    {
        return false;
    }

    const int silhouette_vertex_count = std::clamp(
        silhouetteProtectedVertexCount, 0, loopVertexCount);
    const float silhouette_ratio =
        static_cast<float>(silhouette_vertex_count) /
        static_cast<float>(loopVertexCount);
    const int extra_strong_support =
        silhouette_vertex_count > 0 &&
        silhouette_ratio >= std::clamp(strongSilhouetteRatio, 0.0f, 1.0f)
        ? 1
        : 0;
    const int required_supporting_views =
        std::max(1, minimumSupportingViews) + extra_strong_support;
    return supportingViewCount >= required_supporting_views &&
           conflictViewCount <= std::max(0, maximumConflictViews);
}

bool DepthTsdfSurfaceBuilder::shouldAcceptFinalHoleFillPatchSample(
    int supportingViewCount,
    int conflictViewCount,
    int minimumSupportingViews,
    int maximumConflictViews)
{
    return supportingViewCount >= std::max(2, minimumSupportingViews) &&
           conflictViewCount <= std::max(0, maximumConflictViews);
}

bool DepthTsdfSurfaceBuilder::shouldApplyVisibilityOccupancyCut(
    bool cutOk,
    bool emptyCut,
    std::uint64_t sampleCount,
    std::uint64_t fullSampleCount,
    float minimumFullFraction)
{
    if (!cutOk || emptyCut || sampleCount == 0)
    {
        return false;
    }
    const double full_fraction =
        static_cast<double>(fullSampleCount) /
        static_cast<double>(sampleCount);
    return full_fraction >= static_cast<double>(std::clamp(
        minimumFullFraction, 0.0f, 0.25f));
}

DepthTsdfVisualHullTopologyGuardEvaluation
DepthTsdfSurfaceBuilder::evaluateVisualHullCompletionTopologyGuard(
    const TriMesh &baseline,
    const TriMesh &candidate,
    int maximumTopologicalComplexityIncrease,
    double maximumSurfaceAreaRatio,
    double maximumBoundsDiagonalRatio)
{
    DepthTsdfVisualHullTopologyGuardEvaluation evaluation;
    evaluation.baselineFaceCount = baseline.faceCount();
    evaluation.candidateFaceCount = candidate.faceCount();
    if (baseline.empty() || candidate.empty())
    {
        evaluation.rejectionFlags |= VisualHullTopologyGuardEmptyMesh;
        return evaluation;
    }

    const MeshTopologyQualityStatistics baseline_quality =
        evaluateMeshTopologyQuality(baseline);
    const MeshTopologyQualityStatistics candidate_quality =
        evaluateMeshTopologyQuality(candidate);
    evaluation.baselineBoundaryEdgeCount = baseline_quality.boundaryEdgeCount;
    evaluation.candidateBoundaryEdgeCount = candidate_quality.boundaryEdgeCount;
    evaluation.baselineNonManifoldEdgeCount =
        baseline_quality.nonManifoldEdgeCount;
    evaluation.candidateNonManifoldEdgeCount =
        candidate_quality.nonManifoldEdgeCount;
    evaluation.baselineComponentCount = baseline_quality.componentCount;
    evaluation.candidateComponentCount = candidate_quality.componentCount;
    evaluation.baselineEulerCharacteristic =
        baseline_quality.eulerCharacteristic;
    evaluation.candidateEulerCharacteristic =
        candidate_quality.eulerCharacteristic;
    evaluation.baselineTopologicalComplexity =
        baseline_quality.topologicalComplexity;
    evaluation.candidateTopologicalComplexity =
        candidate_quality.topologicalComplexity;

    if (candidate_quality.boundaryEdgeCount > baseline_quality.boundaryEdgeCount)
    {
        evaluation.rejectionFlags |= VisualHullTopologyGuardBoundaryEdgeGrowth;
    }
    if (candidate_quality.nonManifoldEdgeCount >
        baseline_quality.nonManifoldEdgeCount)
    {
        evaluation.rejectionFlags |=
            VisualHullTopologyGuardNonManifoldEdgeGrowth;
    }
    if (candidate_quality.nonManifoldVertexCount >
        baseline_quality.nonManifoldVertexCount)
    {
        evaluation.rejectionFlags |=
            VisualHullTopologyGuardNonManifoldVertexGrowth;
    }
    if (candidate_quality.componentCount > baseline_quality.componentCount)
    {
        evaluation.rejectionFlags |= VisualHullTopologyGuardComponentGrowth;
    }
    if (candidate_quality.topologicalComplexity >
        baseline_quality.topologicalComplexity +
            std::max(0, maximumTopologicalComplexityIncrease))
    {
        evaluation.rejectionFlags |=
            VisualHullTopologyGuardTopologicalComplexityGrowth;
    }

    const double baseline_area = meshSurfaceArea(baseline);
    const double candidate_area = meshSurfaceArea(candidate);
    evaluation.surfaceAreaRatio = baseline_area > 1.0e-18
        ? candidate_area / baseline_area
        : std::numeric_limits<double>::infinity();
    if (!std::isfinite(evaluation.surfaceAreaRatio) ||
        evaluation.surfaceAreaRatio > std::max(1.0, maximumSurfaceAreaRatio))
    {
        evaluation.rejectionFlags |= VisualHullTopologyGuardSurfaceAreaGrowth;
    }

    const double baseline_diagonal = meshBoundsDiagonal(baseline);
    const double candidate_diagonal = meshBoundsDiagonal(candidate);
    evaluation.boundsDiagonalRatio = baseline_diagonal > 1.0e-18
        ? candidate_diagonal / baseline_diagonal
        : std::numeric_limits<double>::infinity();
    if (!std::isfinite(evaluation.boundsDiagonalRatio) ||
        evaluation.boundsDiagonalRatio >
            std::max(1.0, maximumBoundsDiagonalRatio))
    {
        evaluation.rejectionFlags |=
            VisualHullTopologyGuardBoundsDiagonalGrowth;
    }

    evaluation.accepted =
        evaluation.rejectionFlags == VisualHullTopologyGuardAccepted;
    return evaluation;
}

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
                                                    bool includeColor,
                                                    int nestedResolution)
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

    const bool use_nested_resolution = nestedResolution >= 4 &&
        resolution >= nestedResolution &&
        resolution % nestedResolution == 0;
    const int nested_scale = use_nested_resolution
        ? resolution / nestedResolution
        : 1;
    for (int axis = 0; axis < 3; ++axis)
    {
        const int unaligned_cells = std::max(
            1,
            static_cast<int>(std::lround(static_cast<double>(resolution) *
                                         static_cast<double>(extents[axis]) /
                                         static_cast<double>(longestExtent))));
        if (use_nested_resolution)
        {
            const int nested_cells = std::max(
                1,
                static_cast<int>(std::lround(
                    static_cast<double>(nestedResolution) *
                    static_cast<double>(extents[axis]) /
                    static_cast<double>(longestExtent))));
            layout.cells[axis] = nested_cells * nested_scale;
        }
        else
        {
            layout.cells[axis] = unaligned_cells;
        }
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
    int nested_occupancy_resolution = 0;
    const int occupancy_resolution = std::clamp(
        options.visibilityOccupancyResolution, 24, 128);
    if (options.enableVisibilityOccupancyCompletion &&
        options.visibilityOccupancyAlignCarrierGrid &&
        options.resolution >= occupancy_resolution &&
        options.resolution % occupancy_resolution == 0)
    {
        nested_occupancy_resolution = occupancy_resolution;
    }
    result.layout = makeLayout(boundsMin,
                               boundsMax,
                               options.resolution,
                               options.calculateVertexColors,
                               nested_occupancy_resolution);
    if (!result.layout.ok)
    {
        result.errorMessage = QStringLiteral("Invalid TSDF bounds or resolution=%1")
                                  .arg(options.resolution);
        return result;
    }
    if (options.enableSurfacePatchSupport ||
        options.enableContourBandZeroCrossingSupport ||
        options.enableCrossViewAnchoredSurfaceRecovery ||
        options.enableGlobalImplicitRegularization ||
        options.enableAdaptiveTgvRegularization)
    {
        std::uint64_t evidence_bytes = 0;
        const std::size_t float_field_count =
            options.enableContourBandZeroCrossingSupport ? 4u : 3u;
        if (!checkedMultiply(result.layout.sampleCount,
                             sizeof(std::uint16_t) * 2u +
                                 sizeof(float) * float_field_count +
                                 (options.enableCrossViewAnchoredSurfaceRecovery
                                      ? sizeof(std::uint8_t)
                                      : 0u),
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
    if (options.enableAdaptiveTgvRegularization)
    {
        constexpr std::uint64_t kSparseTgvBytesPerActiveSample =
            sizeof(AdaptiveTsdfOctreeNode) + sizeof(float) * 23u + 32u;
        std::uint64_t adaptive_bytes = 0;
        if (!checkedMultiply(
                result.layout.sampleCount,
                sizeof(DepthVisibilityHistogram) +
                    kSparseTgvBytesPerActiveSample,
                &adaptive_bytes) ||
            result.layout.requiredBytes >
                std::numeric_limits<std::uint64_t>::max() - adaptive_bytes)
        {
            result.layout.ok = false;
            result.errorMessage = QStringLiteral(
                "TSDF adaptive TGV allocation overflow");
            return result;
        }
        result.layout.requiredBytes += adaptive_bytes;
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
        const bool recovered_rejected_frame =
            canUseRejectedOrbitalFrameAsAuxiliary(artifact);
        const bool promoted_validation_frame =
            canPromoteHealthyOrbitalValidationFrame(artifact);
        const bool validation_only =
            (artifact.acceptance == QStringLiteral("validation_only")
             && !promoted_validation_frame)
            || recovered_rejected_frame;
        const bool quality_override =
            validation_only || promoted_validation_frame;
        if ((!artifact.status.isEmpty() && artifact.status != QStringLiteral("completed")) ||
            (!artifact.fusionEligible && !quality_override) ||
            (artifact.acceptance == QStringLiteral("rejected")
             && !recovered_rejected_frame) ||
            artifact.acceptance == QStringLiteral("failed"))
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
        const bool requires_current_orbital_evidence =
            artifact.algorithmRevision >= 11 &&
            artifact.sceneProfile == QStringLiteral("orbital_object");
        const bool requires_adaptive_orbital_evidence =
            requires_current_orbital_evidence &&
            artifact.algorithmRevision >= 13;
        const bool requires_conflict_ratio =
            requires_adaptive_orbital_evidence &&
            artifact.algorithmRevision >= 14;
        if (requires_current_orbital_evidence &&
            (artifact.geometrySupportPath.isEmpty() ||
             artifact.geometrySourceMaskPath.isEmpty() ||
             artifact.inverseDepthMeanPath.isEmpty() ||
             artifact.inverseDepthSpreadPath.isEmpty() ||
             artifact.crossViewRepairedMaskPath.isEmpty()))
        {
            result.errorMessage = frameArtifactError(
                artifact,
                QStringLiteral(
                    "current orbital depth evidence is incomplete; "
                    "regenerate depth maps so geometry support, source mask, "
                    "inverse-depth statistics, and repair provenance are all present"));
            return result;
        }
        if (requires_adaptive_orbital_evidence &&
            (artifact.adaptiveGeometrySupportWeightPath.isEmpty() ||
             artifact.adaptiveGeometryEffectiveViewCountPath.isEmpty() ||
             (requires_conflict_ratio &&
              artifact.adaptiveGeometryConflictRatioPath.isEmpty())))
        {
            result.errorMessage = frameArtifactError(
                artifact,
                requires_conflict_ratio
                    ? QStringLiteral(
                          "current orbital adaptive geometry evidence is incomplete; "
                          "regenerate depth maps so support weight, effective view count, "
                          "and conflict ratio are all present")
                    : QStringLiteral(
                          "current orbital adaptive geometry evidence is incomplete; "
                          "regenerate depth maps so support weight and effective view count "
                          "are both present"));
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
                     &reason))
        {
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("geometry support: %1").arg(reason));
            return result;
        }
        else if (frame.geometrySupportCount.size() != frame.depth.size())
        {
            if (!artifact.pyramidFallback)
            {
                result.errorMessage = frameArtifactError(
                    artifact,
                    QStringLiteral("geometry support dimensions do not match depth"));
                return result;
            }
            cv::resize(frame.geometrySupportCount,
                       frame.geometrySupportCount,
                       frame.depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }

        if (artifact.geometrySourceMaskPath.isEmpty())
        {
            frame.geometrySourceMask = cv::Mat(
                frame.depth.size(), CV_16UC1, cv::Scalar(0));
        }
        else if (!loadUnsignedShortMatrix(
                     artifact.geometrySourceMaskPath,
                     &frame.geometrySourceMask,
                     &reason))
        {
            result.errorMessage = frameArtifactError(
                artifact, QStringLiteral("geometry source mask: %1").arg(reason));
            return result;
        }
        else if (frame.geometrySourceMask.size() != frame.depth.size())
        {
            if (!artifact.pyramidFallback)
            {
                result.errorMessage = frameArtifactError(
                    artifact,
                    QStringLiteral("geometry source mask dimensions do not match depth"));
                return result;
            }
            cv::resize(frame.geometrySourceMask,
                       frame.geometrySourceMask,
                       frame.depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
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
            if (!loadFloatMatrix(path, destination, &reason))
            {
                result.errorMessage = frameArtifactError(
                    artifact, QStringLiteral("%1: %2").arg(label, reason));
                return false;
            }
            if (destination->size() != frame.depth.size())
            {
                if (!artifact.pyramidFallback)
                {
                    result.errorMessage = frameArtifactError(
                        artifact, QStringLiteral("%1: dimensions do not match depth").arg(label));
                    return false;
                }
                cv::resize(*destination,
                           *destination,
                           frame.depth.size(),
                           0.0,
                           0.0,
                           cv::INTER_AREA);
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

        auto load_optional_adaptive_evidence = [&](const QString &path,
                                                    cv::Mat *destination,
                                                    const QString &label)
        {
            destination->release();
            if (path.isEmpty())
            {
                return true;
            }
            reason.clear();
            if (!loadFloatMatrix(path, destination, &reason))
            {
                result.errorMessage = frameArtifactError(
                    artifact, QStringLiteral("%1: %2").arg(label, reason));
                return false;
            }
            if (destination->size() != frame.depth.size())
            {
                if (!artifact.pyramidFallback)
                {
                    result.errorMessage = frameArtifactError(
                        artifact, QStringLiteral("%1: dimensions do not match depth").arg(label));
                    return false;
                }
                cv::resize(*destination,
                           *destination,
                           frame.depth.size(),
                           0.0,
                           0.0,
                           cv::INTER_AREA);
            }
            return true;
        };
        if (!load_optional_adaptive_evidence(
                artifact.adaptiveGeometrySupportWeightPath,
                &frame.adaptiveGeometrySupportWeight,
                QStringLiteral("adaptive geometry support weight")) ||
            !load_optional_adaptive_evidence(
                artifact.adaptiveGeometryEffectiveViewCountPath,
                &frame.adaptiveGeometryEffectiveViewCount,
                QStringLiteral("adaptive geometry effective view count")) ||
            !load_optional_adaptive_evidence(
                artifact.adaptiveGeometryConflictRatioPath,
                &frame.adaptiveGeometryConflictRatio,
                QStringLiteral("adaptive geometry conflict ratio")))
        {
            return result;
        }

        auto validate_adaptive_evidence = [&](const cv::Mat &matrix,
                                              const QString &label,
                                              bool bounded_probability,
                                              bool effective_view_count)
        {
            if (matrix.empty())
            {
                return true;
            }
            for (int row = 0; row < matrix.rows; ++row)
            {
                const float *values = matrix.ptr<float>(row);
                for (int column = 0; column < matrix.cols; ++column)
                {
                    const float value = values[column];
                    if (!std::isfinite(value) || value < 0.0f ||
                        (bounded_probability && value > 1.0f))
                    {
                        result.errorMessage = frameArtifactError(
                            artifact,
                            QStringLiteral(
                                "%1 contains an invalid value at row=%2 column=%3: value=%4")
                                .arg(label)
                                .arg(row)
                                .arg(column)
                                .arg(value));
                        return false;
                    }
                    if (effective_view_count && value > 0.0f && value < 1.0f)
                    {
                        result.errorMessage = frameArtifactError(
                            artifact,
                            QStringLiteral(
                                "adaptive geometry effective view count must be zero or "
                                "at least one at row=%1 column=%2; value=%3")
                                .arg(row)
                                .arg(column)
                                .arg(value));
                        return false;
                    }
                }
            }
            return true;
        };
        if (!validate_adaptive_evidence(
                frame.adaptiveGeometrySupportWeight,
                QStringLiteral("adaptive geometry support weight"),
                true,
                false) ||
            !validate_adaptive_evidence(
                frame.adaptiveGeometryEffectiveViewCount,
                QStringLiteral("adaptive geometry effective view count"),
                false,
                true) ||
            !validate_adaptive_evidence(
                frame.adaptiveGeometryConflictRatio,
                QStringLiteral("adaptive geometry conflict ratio"),
                true,
                false))
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
                           &reason,
                           artifact.pyramidFallback))
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
                           &reason,
                           artifact.pyramidFallback))
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
                           &reason,
                           artifact.pyramidFallback))
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
        frame.auxiliarySurfaceOnly = validation_only;
        frame.useAdaptiveGeometryEvidence = requires_conflict_ratio &&
            !frame.adaptiveGeometrySupportWeight.empty() &&
            !frame.adaptiveGeometryEffectiveViewCount.empty() &&
            !frame.adaptiveGeometryConflictRatio.empty();
        result.frames.push_back(std::move(frame));
    }

    if (result.frames.size() < 3)
    {
        result.errorMessage = QStringLiteral(
            "TSDF requires at least 3 usable primary or auxiliary depth frames; loaded=%1")
                                  .arg(result.frames.size());
        return result;
    }
    result.ok = true;
    return result;
}

DepthTsdfBoundsResult DepthTsdfSurfaceBuilder::estimateBounds(
    const QVector<DepthTsdfFrame> &frames)
{
    constexpr std::uint64_t minimum_trusted_sample_count = 500;
    constexpr double minimum_trusted_sample_ratio = 0.10;

    DepthTsdfBoundsResult result;
    if (frames.size() < 3)
    {
        result.errorMessage = QStringLiteral("TSDF bounds require at least 3 accepted depth frames");
        return result;
    }

    std::array<std::vector<float>, 3> candidate_coordinates;
    std::array<std::vector<float>, 3> trusted_coordinates;
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
        const bool has_geometry_evidence =
            frame.geometrySupportCount.type() == CV_16UC1 &&
            frame.geometrySupportCount.size() == frame.depth.size();
        const bool has_adaptive_evidence = frame.useAdaptiveGeometryEvidence &&
            frame.adaptiveGeometryEffectiveViewCount.type() == CV_32FC1 &&
            frame.adaptiveGeometryEffectiveViewCount.size() == frame.depth.size();
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
                const bool trusted = has_geometry_evidence &&
                    frame.geometrySupportCount.at<std::uint16_t>(row, column) >= 2 &&
                    (!has_adaptive_evidence ||
                     frame.adaptiveGeometryEffectiveViewCount.at<float>(row, column) >= 2.0f);
                for (int axis = 0; axis < 3; ++axis)
                {
                    const float coordinate = static_cast<float>(world[axis]);
                    candidate_coordinates[axis].push_back(coordinate);
                    if (trusted)
                    {
                        trusted_coordinates[axis].push_back(coordinate);
                    }
                }
            }
        }
    }

    result.candidateSampleCount = candidate_coordinates[0].size();
    result.trustedSampleCount = trusted_coordinates[0].size();
    if (result.candidateSampleCount < 500)
    {
        result.selectionReason = QStringLiteral("insufficient_candidate_samples");
        result.errorMessage = QStringLiteral("Insufficient finite TSDF bound samples: %1")
                                  .arg(result.candidateSampleCount);
        return result;
    }
    const bool sufficient_trusted_samples =
        result.trustedSampleCount >= minimum_trusted_sample_count &&
        static_cast<double>(result.trustedSampleCount) >=
            static_cast<double>(result.candidateSampleCount) * minimum_trusted_sample_ratio;
    result.usedEvidenceAwareSamples = sufficient_trusted_samples;
    result.fellBackToCandidateSamples = !sufficient_trusted_samples;
    result.selectionReason = sufficient_trusted_samples
        ? QStringLiteral("trusted_multiview_evidence")
        : (result.trustedSampleCount < minimum_trusted_sample_count
               ? QStringLiteral("insufficient_trusted_sample_count")
               : QStringLiteral("insufficient_trusted_sample_ratio"));
    std::array<std::vector<float>, 3> &coordinates = sufficient_trusted_samples
        ? trusted_coordinates
        : candidate_coordinates;
    result.sampleCount = coordinates[0].size();
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
    const cv::Mat &crossViewConsensusMask,
    const cv::Mat &referenceAnchoredConsensusDepth)
{
    DepthTsdfObservationSample result;
    result.useAdaptiveGeometryEvidence = frame.useAdaptiveGeometryEvidence;
    const int nearest_column = static_cast<int>(std::lround(pixel.x));
    const int nearest_row = static_cast<int>(std::lround(pixel.y));
    if (nearest_row < 0 || nearest_row >= frame.depth.rows ||
        nearest_column < 0 || nearest_column >= frame.depth.cols)
    {
        return result;
    }

    auto optional_float_evidence = [&](const cv::Mat &matrix,
                                       int row,
                                       int column)
    {
        return matrix.type() == CV_32FC1 && matrix.size() == frame.depth.size()
            ? matrix.at<float>(row, column)
            : 0.0f;
    };

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
        if (!enableCrossViewConsensusDepth)
        {
            return raw_depth;
        }
        // The precomputed map already encodes geometry, confidence, contour,
        // repair and bias gates. Legacy per-sample gates apply only without it.
        if (referenceAnchoredConsensusDepth.type() == CV_32FC1 &&
            referenceAnchoredConsensusDepth.size() == frame.depth.size())
        {
            const float depth = referenceAnchoredConsensusDepth.at<float>(row, column);
            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                return raw_depth;
            }
            if (used)
            {
                *used = std::fabs(depth - raw_depth) > 1.0e-8f;
            }
            return depth;
        }
        if (geometry_support < 2 ||
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
        sample->adaptiveGeometrySupportWeight = optional_float_evidence(
            frame.adaptiveGeometrySupportWeight, row, column);
        sample->adaptiveGeometryEffectiveViewCount = optional_float_evidence(
            frame.adaptiveGeometryEffectiveViewCount, row, column);
        sample->adaptiveGeometryConflictRatio = optional_float_evidence(
            frame.adaptiveGeometryConflictRatio, row, column);
        sample->inverseDepthRelativeSpread =
            frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
                frame.inverseDepthRelativeSpread.size() == frame.depth.size()
            ? frame.inverseDepthRelativeSpread.at<float>(row, column) : 0.0f;
        if (!frame.useAdaptiveGeometryEvidence &&
            maximumObservationInverseDepthSpread > 0.0f &&
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
        sample->usedCrossViewRepairedDepth =
            frame.crossViewRepairedMask.type() == CV_8UC1 &&
            frame.crossViewRepairedMask.size() == frame.depth.size() &&
            frame.crossViewRepairedMask.at<std::uint8_t>(row, column) != 0;
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
        float adaptiveGeometrySupportWeight = 0.0f;
        float adaptiveGeometryEffectiveViewCount = 0.0f;
        float adaptiveGeometryConflictRatio = 0.0f;
        float inverseDepthRelativeSpread = 0.0f;
        bool nearest = false;
        bool usedCrossViewConsensusDepth = false;
        bool usedCrossViewRepairedDepth = false;
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
            if (!frame.useAdaptiveGeometryEvidence &&
                maximumObservationInverseDepthSpread > 0.0f &&
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
            candidate.adaptiveGeometrySupportWeight = optional_float_evidence(
                frame.adaptiveGeometrySupportWeight, row, column);
            candidate.adaptiveGeometryEffectiveViewCount = optional_float_evidence(
                frame.adaptiveGeometryEffectiveViewCount, row, column);
            candidate.adaptiveGeometryConflictRatio = optional_float_evidence(
                frame.adaptiveGeometryConflictRatio, row, column);
            candidate.inverseDepthRelativeSpread = inverse_depth_relative_spread;
            candidate.depth = consensus_depth(row,
                                              column,
                                              depth,
                                              candidate.geometrySupportCount,
                                              candidate.inverseDepthRelativeSpread,
                                              &candidate.usedCrossViewConsensusDepth);
            candidate.usedCrossViewRepairedDepth =
                frame.crossViewRepairedMask.type() == CV_8UC1 &&
                frame.crossViewRepairedMask.size() == frame.depth.size() &&
                frame.crossViewRepairedMask.at<std::uint8_t>(row, column) != 0;
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
        result.adaptiveGeometrySupportWeight +=
            candidate.adaptiveGeometrySupportWeight * candidate.spatialWeight;
        result.adaptiveGeometryEffectiveViewCount +=
            candidate.adaptiveGeometryEffectiveViewCount * candidate.spatialWeight;
        result.adaptiveGeometryConflictRatio +=
            candidate.adaptiveGeometryConflictRatio * candidate.spatialWeight;
        result.inverseDepthRelativeSpread = std::max(
            result.inverseDepthRelativeSpread,
            candidate.inverseDepthRelativeSpread);
        first_evidence = false;
        result.usedCrossViewConsensusDepth = result.usedCrossViewConsensusDepth ||
            candidate.usedCrossViewConsensusDepth;
        result.usedCrossViewRepairedDepth = result.usedCrossViewRepairedDepth ||
            candidate.usedCrossViewRepairedDepth;
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
    result.adaptiveGeometrySupportWeight /= spatial_weight_sum;
    result.adaptiveGeometryEffectiveViewCount /= spatial_weight_sum;
    result.adaptiveGeometryConflictRatio /= spatial_weight_sum;
    result.valid = true;
    result.failure = DepthTsdfObservationFailure::None;
    result.recoveredFromInvalidNearestPixel = !has_nearest_candidate;
    if (!frame.useAdaptiveGeometryEvidence &&
        result.recoveredFromInvalidNearestPixel &&
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
    bool *geometryVerifiedSingleView,
    bool hasStrongAdaptiveSurfaceObservation)
{
    const bool multi_view_supported = distinctSupportCount >= std::max(
                                          2, options.minimumDistinctCameraSupport)
        && accumulatedWeight >= options.minimumVoxelWeight;
    const bool legacy_single_view_supported = options.minimumDistinctCameraSupport <= 1
        && distinctSupportCount == 1
        && maximumObservationWeight >= options.minimumSingleObservationWeight;
    const bool legacy_geometry_verified_single_view_supported =
        distinctSupportCount == 1 &&
        maximumObservationWeight >= options.minimumGeometryVerifiedObservationWeight &&
        options.allowGeometryVerifiedSingleObservation &&
        maximumGeometrySupportCount >= options.minimumGeometrySupportCount;
    // A strong adaptive observation already represents continuous agreement
    // from multiple source views.  Its TSDF weight is intentionally reduced by
    // confidence, frame quality and spread factors, so applying the legacy
    // 0.85 gate again can make the guarded path unreachable.  It remains only
    // a candidate here and must still pass neighborhood growth below.
    const bool strong_adaptive_single_view_supported =
        distinctSupportCount == 1 &&
        options.enableGeometrySingleViewNeighborhoodGuard &&
        hasStrongAdaptiveSurfaceObservation;
    const bool geometry_verified_single_view_supported =
        legacy_geometry_verified_single_view_supported ||
        strong_adaptive_single_view_supported;
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

std::vector<std::uint8_t> buildVisualHullOccupancy(
    const DepthTsdfLayout &layout,
    const QVector<DepthTsdfFrame> &frames,
    int minimumVisibleViews,
    int allowedSilhouetteViolations,
    const std::function<bool()> &isCancelled)
{
    std::vector<std::uint8_t> occupied(
        static_cast<std::size_t>(layout.sampleCount), 0);
    if (!layout.ok || frames.size() < 2)
    {
        return occupied;
    }

    const int minimum_visible_views = std::clamp(
        minimumVisibleViews, 2, static_cast<int>(frames.size()));
    const int allowed_violations = std::clamp(
        allowedSilhouetteViolations,
        0,
        static_cast<int>(frames.size()) - 1);
    std::atomic_bool cancelled{false};
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int z = 0; z <= layout.cells[2]; ++z)
    {
        if (cancelled.load(std::memory_order_relaxed) ||
            (isCancelled && isCancelled()))
        {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
        }
        const double world_z =
            layout.boundsMin[2] + layout.voxelSize[2] * z;
        for (int y = 0; y <= layout.cells[1]; ++y)
        {
            const double world_y =
                layout.boundsMin[1] + layout.voxelSize[1] * y;
            for (int x = 0; x <= layout.cells[0]; ++x)
            {
                const double world[3] = {
                    layout.boundsMin[0] + layout.voxelSize[0] * x,
                    world_y,
                    world_z};
                int visible_views = 0;
                int silhouette_violations = 0;
                for (const DepthTsdfFrame &frame : frames)
                {
                    if (!frame.camera.isValid() ||
                        frame.supportMask.empty() ||
                        frame.supportMask.type() != CV_8UC1)
                    {
                        continue;
                    }
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
                    ++visible_views;
                    if (frame.supportMask.at<std::uint8_t>(
                            row, column) == 0)
                    {
                        ++silhouette_violations;
                        if (silhouette_violations > allowed_violations)
                        {
                            break;
                        }
                    }
                }
                if (visible_views >= minimum_visible_views &&
                    silhouette_violations <= allowed_violations)
                {
                    occupied[sampleIndex(layout, x, y, z)] = 1;
                }
            }
        }
    }
    if (cancelled.load(std::memory_order_relaxed))
    {
        occupied.clear();
    }
    return occupied;
}

std::uint64_t relaxVisualHullCompletionField(
    const DepthTsdfLayout &layout,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::uint8_t> &completionMask,
    const std::vector<std::uint8_t> &supported,
    int iterations,
    float lambda,
    float maximumUpdate,
    std::vector<float> *tsdf)
{
    if (!tsdf || !layout.ok || occupied.size() != tsdf->size() ||
        completionMask.size() != tsdf->size() ||
        supported.size() != tsdf->size())
    {
        return 0;
    }
    std::vector<std::size_t> completion_indices;
    completion_indices.reserve(static_cast<std::size_t>(
        std::count(completionMask.cbegin(),
                   completionMask.cend(),
                   std::uint8_t{1})));
    for (std::size_t index = 0; index < completionMask.size(); ++index)
    {
        if (completionMask[index] != 0)
        {
            completion_indices.push_back(index);
        }
    }
    if (completion_indices.empty())
    {
        return 0;
    }

    const int relaxation_iterations = std::clamp(iterations, 0, 24);
    const float relaxation_lambda = std::clamp(lambda, 0.0f, 0.49f);
    const float maximum_update = std::clamp(maximumUpdate, 0.01f, 1.0f);
    std::vector<float> relaxed(completion_indices.size(), 0.0f);
    const int row_size = layout.cells[0] + 1;
    const int layer_size = row_size * (layout.cells[1] + 1);
    for (int iteration = 0; iteration < relaxation_iterations; ++iteration)
    {
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t candidate = 0;
             candidate < static_cast<std::int64_t>(
                             completion_indices.size());
             ++candidate)
        {
            const std::size_t index =
                completion_indices[static_cast<std::size_t>(candidate)];
            const int z = static_cast<int>(index / layer_size);
            const int remainder =
                static_cast<int>(index % layer_size);
            const int y = remainder / row_size;
            const int x = remainder % row_size;
            float sum = 0.0f;
            int count = 0;
            constexpr std::array<std::array<int, 3>, 6> kOffsets = {{
                {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
                {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
            for (const auto &offset : kOffsets)
            {
                const int neighbor_x = x + offset[0];
                const int neighbor_y = y + offset[1];
                const int neighbor_z = z + offset[2];
                if (neighbor_x < 0 || neighbor_y < 0 ||
                    neighbor_z < 0 ||
                    neighbor_x > layout.cells[0] ||
                    neighbor_y > layout.cells[1] ||
                    neighbor_z > layout.cells[2])
                {
                    continue;
                }
                const std::size_t neighbor = sampleIndex(
                    layout, neighbor_x, neighbor_y, neighbor_z);
                if (supported[neighbor] == 0)
                {
                    continue;
                }
                sum += (*tsdf)[neighbor];
                ++count;
            }
            float value = (*tsdf)[index];
            if (count >= 2)
            {
                const float target = sum / static_cast<float>(count);
                const float update = std::clamp(
                    (target - value) * relaxation_lambda,
                    -maximum_update,
                    maximum_update);
                value += update;
            }
            const bool inside = occupied[index] != 0;
            if (inside)
            {
                value = std::min(value, -1.0e-4f);
            }
            else
            {
                value = std::max(value, 1.0e-4f);
            }
            relaxed[static_cast<std::size_t>(candidate)] = value;
        }
        for (std::size_t candidate = 0;
             candidate < completion_indices.size();
             ++candidate)
        {
            (*tsdf)[completion_indices[candidate]] = relaxed[candidate];
        }
    }
    return static_cast<std::uint64_t>(completion_indices.size());
}

float DepthTsdfSurfaceBuilder::observationEvidenceWeightMultiplier(
    const DepthTsdfObservationSample &observation,
    const DepthTsdfOptions &options)
{
    if (observation.useAdaptiveGeometryEvidence)
    {
        if (observation.usedCrossViewRepairedDepth)
        {
            return std::clamp(
                options.repairedObservationMultiplier, 0.05f, 1.0f);
        }
        const float support_weight = std::isfinite(
                observation.adaptiveGeometrySupportWeight)
            ? std::clamp(
                  observation.adaptiveGeometrySupportWeight, 0.0f, 1.0f)
            : 0.0f;
        return std::max(
            std::clamp(
                options.adaptiveGeometryMinimumObservationMultiplier,
                0.05f,
                1.0f),
            support_weight);
    }
    if (!options.enablePixelEvidenceWeighting)
    {
        return 1.0f;
    }
    if (observation.usedCrossViewRepairedDepth)
    {
        return std::clamp(options.repairedObservationMultiplier, 0.05f, 1.0f);
    }
    if (observation.geometrySupportCount == 0)
    {
        return std::clamp(
            options.unconfirmedNativeObservationMultiplier, 0.05f, 1.0f);
    }
    if (observation.geometrySupportCount == 1)
    {
        return std::clamp(
            options.weakNativeObservationMultiplier, 0.05f, 1.0f);
    }
    return 1.0f;
}

float DepthTsdfSurfaceBuilder::observationInverseDepthSpreadWeightMultiplier(
    const DepthTsdfObservationSample &observation,
    const DepthTsdfOptions &options)
{
    if (!options.enableInverseDepthSpreadWeighting ||
        !std::isfinite(observation.inverseDepthRelativeSpread) ||
        observation.inverseDepthRelativeSpread <= 0.0f)
    {
        return 1.0f;
    }

    const float knee = std::clamp(
        options.inverseDepthSpreadWeightKnee, 0.0f, 0.099f);
    const float zero = std::clamp(
        options.inverseDepthSpreadWeightZero, knee + 1.0e-6f, 0.10f);
    const float minimum_multiplier = std::clamp(
        options.minimumInverseDepthSpreadWeightMultiplier, 0.0f, 1.0f);
    if (observation.inverseDepthRelativeSpread <= knee)
    {
        return 1.0f;
    }
    if (observation.inverseDepthRelativeSpread >= zero)
    {
        return minimum_multiplier;
    }

    const float normalized =
        (observation.inverseDepthRelativeSpread - knee) / (zero - knee);
    const float smooth = normalized * normalized * (3.0f - 2.0f * normalized);
    return 1.0f - smooth * (1.0f - minimum_multiplier);
}

float DepthTsdfSurfaceBuilder::
    observationInverseDepthSpreadSupportWeightMultiplier(
        const DepthTsdfObservationSample &observation,
        const DepthTsdfOptions &options)
{
    const float field_multiplier =
        observationInverseDepthSpreadWeightMultiplier(observation, options);
    if (!options.enableInverseDepthSpreadSupportWeightDecoupling ||
        !options.enableInverseDepthSpreadWeighting)
    {
        return field_multiplier;
    }
    if (field_multiplier <= 0.0f)
    {
        return 0.0f;
    }
    return std::pow(
        field_multiplier,
        std::clamp(
            options.inverseDepthSpreadSupportWeightExponent, 0.05f, 1.0f));
}

float DepthTsdfSurfaceBuilder::observationEvidenceSupportWeightMultiplier(
    const DepthTsdfObservationSample &observation,
    const DepthTsdfOptions &options)
{
    const float field_multiplier =
        observationEvidenceWeightMultiplier(observation, options);
    if (!options.enableEvidenceSupportWeightDecoupling)
    {
        return field_multiplier;
    }
    return std::pow(
        field_multiplier,
        std::clamp(options.evidenceSupportWeightExponent, 0.0f, 1.0f));
}

bool DepthTsdfSurfaceBuilder::observationHasStrongAdaptiveGeometryEvidence(
    const DepthTsdfObservationSample &observation,
    const DepthTsdfOptions &options)
{
    if (!observation.useAdaptiveGeometryEvidence ||
        observation.usedCrossViewRepairedDepth ||
        !std::isfinite(observation.adaptiveGeometrySupportWeight) ||
        !std::isfinite(observation.adaptiveGeometryEffectiveViewCount) ||
        !std::isfinite(observation.adaptiveGeometryConflictRatio))
    {
        return false;
    }
    return observation.adaptiveGeometrySupportWeight >= std::clamp(
               options.adaptiveGeometryFullIntegrationMinimumSupportWeight,
               0.0f,
               1.0f) &&
        observation.adaptiveGeometryEffectiveViewCount >= std::max(
               1.0f,
               options.adaptiveGeometryFullIntegrationMinimumEffectiveViewCount) &&
        observation.adaptiveGeometryConflictRatio <= std::clamp(
               options.adaptiveGeometryFullIntegrationMaximumConflictRatio,
               0.0f,
               1.0f);
}

bool DepthTsdfSurfaceBuilder::observationUsesSurfaceOnlyIntegration(
    const DepthTsdfObservationSample &observation,
    const DepthTsdfOptions &options)
{
    if (observation.useAdaptiveGeometryEvidence)
    {
        return !observationHasStrongAdaptiveGeometryEvidence(
            observation, options);
    }
    return options.enableWeakEvidenceSurfaceOnlyIntegration &&
        (observation.usedCrossViewRepairedDepth ||
         observation.geometrySupportCount <= 1);
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

DepthTsdfVisualHullCompletionStatistics
DepthTsdfSurfaceBuilder::completeUnsupportedSamplesWithVisualHullSignedDistance(
    const DepthTsdfLayout &layout,
    const std::vector<std::uint8_t> &occupied,
    float bandVoxels,
    std::vector<float> *tsdf,
    std::vector<std::uint8_t> *supported,
    const std::vector<std::uint8_t> *immutableVeto)
{
    DepthTsdfVisualHullCompletionStatistics statistics;
    const std::size_t expected_size =
        static_cast<std::size_t>(layout.sampleCount);
    if (!tsdf || !supported || !layout.ok ||
        occupied.size() != expected_size ||
        tsdf->size() != expected_size ||
        supported->size() != expected_size ||
        (immutableVeto && immutableVeto->size() != expected_size) ||
        !std::isfinite(bandVoxels) || bandVoxels <= 0.0f)
    {
        return statistics;
    }

    statistics.occupiedSampleCount = static_cast<std::uint64_t>(
        std::count(occupied.cbegin(), occupied.cend(), std::uint8_t{1}));
    if (statistics.occupiedSampleCount == 0 ||
        statistics.occupiedSampleCount == expected_size)
    {
        return statistics;
    }

    // Keep the original provenance fixed for the complete operation.  Values
    // synthesized below can never become new zero-crossing anchors.
    const std::vector<std::uint8_t> core_supported = *supported;
    std::vector<std::uint8_t> veto_neighborhood;
    if (immutableVeto)
    {
        veto_neighborhood.assign(expected_size, 0);
        for (int z = 0; z <= layout.cells[2]; ++z)
        {
            for (int y = 0; y <= layout.cells[1]; ++y)
            {
                for (int x = 0; x <= layout.cells[0]; ++x)
                {
                    const std::size_t index = sampleIndex(layout, x, y, z);
                    if ((*immutableVeto)[index] == 0)
                    {
                        continue;
                    }
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                const int neighbor_x = x + dx;
                                const int neighbor_y = y + dy;
                                const int neighbor_z = z + dz;
                                if (neighbor_x < 0 || neighbor_y < 0 ||
                                    neighbor_z < 0 ||
                                    neighbor_x > layout.cells[0] ||
                                    neighbor_y > layout.cells[1] ||
                                    neighbor_z > layout.cells[2])
                                {
                                    continue;
                                }
                                veto_neighborhood[sampleIndex(
                                    layout,
                                    neighbor_x,
                                    neighbor_y,
                                    neighbor_z)] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    const auto is_vetoed = [&veto_neighborhood](std::size_t index)
    {
        return !veto_neighborhood.empty() &&
               veto_neighborhood[index] != 0;
    };
    std::vector<std::uint8_t> requested(expected_size, 0);
    const std::size_t cell_count =
        static_cast<std::size_t>(layout.cells[0]) *
        static_cast<std::size_t>(layout.cells[1]) *
        static_cast<std::size_t>(layout.cells[2]);
    std::vector<std::uint8_t> frontier_cells(cell_count, 0);
    const auto cell_index = [&](int x, int y, int z)
    {
        return (static_cast<std::size_t>(z) *
                    static_cast<std::size_t>(layout.cells[1]) +
                static_cast<std::size_t>(y)) *
                   static_cast<std::size_t>(layout.cells[0]) +
               static_cast<std::size_t>(x);
    };
    const auto inspect_core_signs = [&](int cell_x,
                                        int cell_y,
                                        int cell_z,
                                        bool *has_positive,
                                        bool *has_negative)
    {
        *has_positive = false;
        *has_negative = false;
        for (int dz = 0; dz <= 1; ++dz)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                for (int dx = 0; dx <= 1; ++dx)
                {
                    const std::size_t corner = sampleIndex(
                        layout, cell_x + dx, cell_y + dy, cell_z + dz);
                    if (core_supported[corner] == 0)
                    {
                        continue;
                    }
                    if ((*tsdf)[corner] < 0.0f)
                    {
                        *has_negative = true;
                    }
                    else
                    {
                        *has_positive = true;
                    }
                }
            }
        }
    };
    for (int z = 0; z < layout.cells[2]; ++z)
    {
        for (int y = 0; y < layout.cells[1]; ++y)
        {
            for (int x = 0; x < layout.cells[0]; ++x)
            {
                bool anchor_positive = false;
                bool anchor_negative = false;
                inspect_core_signs(
                    x, y, z, &anchor_positive, &anchor_negative);
                if (!anchor_positive || !anchor_negative)
                {
                    continue;
                }
                ++statistics.anchorCellCount;

                for (int neighbor_z = std::max(0, z - 1);
                     neighbor_z <= std::min(layout.cells[2] - 1, z + 1);
                     ++neighbor_z)
                {
                    for (int neighbor_y = std::max(0, y - 1);
                         neighbor_y <= std::min(layout.cells[1] - 1, y + 1);
                         ++neighbor_y)
                    {
                        for (int neighbor_x = std::max(0, x - 1);
                             neighbor_x <= std::min(layout.cells[0] - 1, x + 1);
                             ++neighbor_x)
                        {
                            bool core_positive = false;
                            bool core_negative = false;
                            inspect_core_signs(
                                neighbor_x,
                                neighbor_y,
                                neighbor_z,
                                &core_positive,
                                &core_negative);
                            if (core_positive == core_negative)
                            {
                                continue;
                            }

                            bool requested_in_cell = false;
                            for (int dz = 0; dz <= 1; ++dz)
                            {
                                for (int dy = 0; dy <= 1; ++dy)
                                {
                                    for (int dx = 0; dx <= 1; ++dx)
                                    {
                                        const std::size_t corner = sampleIndex(
                                            layout,
                                            neighbor_x + dx,
                                            neighbor_y + dy,
                                            neighbor_z + dz);
                                        if (core_supported[corner] != 0)
                                        {
                                            continue;
                                        }
                                        // Preserve a one-sample clearance
                                        // around trusted free-space evidence;
                                        // otherwise neighboring hull samples
                                        // can still cap a real through-hole.
                                        if (is_vetoed(corner))
                                        {
                                            continue;
                                        }
                                        const bool hull_negative =
                                            occupied[corner] != 0;
                                        const bool supplies_missing_sign =
                                            (core_positive && hull_negative) ||
                                            (core_negative && !hull_negative);
                                        if (!supplies_missing_sign)
                                        {
                                            continue;
                                        }
                                        requested[corner] = 1;
                                        requested_in_cell = true;
                                    }
                                }
                            }
                            if (requested_in_cell)
                            {
                                frontier_cells[cell_index(
                                    neighbor_x, neighbor_y, neighbor_z)] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    statistics.frontierCellCount = static_cast<std::uint64_t>(
        std::count(
            frontier_cells.cbegin(), frontier_cells.cend(), std::uint8_t{1}));
    if (statistics.anchorCellCount == 0 ||
        statistics.frontierCellCount == 0)
    {
        return statistics;
    }

    constexpr std::uint16_t kInfinity =
        std::numeric_limits<std::uint16_t>::max();
    const int maximum_distance = std::clamp(
        static_cast<int>(std::ceil(bandVoxels * 3.0f)), 3, 192);
    std::vector<std::uint16_t> distance(expected_size, kInfinity);
    constexpr std::array<std::array<int, 3>, 6> kAxisOffsets = {{
        {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
        {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
    for (int z = 0; z <= layout.cells[2]; ++z)
    {
        for (int y = 0; y <= layout.cells[1]; ++y)
        {
            for (int x = 0; x <= layout.cells[0]; ++x)
            {
                const std::size_t index = sampleIndex(layout, x, y, z);
                for (const auto &offset : kAxisOffsets)
                {
                    const int neighbor_x = x + offset[0];
                    const int neighbor_y = y + offset[1];
                    const int neighbor_z = z + offset[2];
                    if (neighbor_x < 0 || neighbor_y < 0 ||
                        neighbor_z < 0 ||
                        neighbor_x > layout.cells[0] ||
                        neighbor_y > layout.cells[1] ||
                        neighbor_z > layout.cells[2])
                    {
                        continue;
                    }
                    const std::size_t neighbor = sampleIndex(
                        layout, neighbor_x, neighbor_y, neighbor_z);
                    if (occupied[neighbor] != occupied[index])
                    {
                        distance[index] = 1;
                        ++statistics.boundarySampleCount;
                        break;
                    }
                }
            }
        }
    }

    const auto relax = [&](int x,
                           int y,
                           int z,
                           int dx,
                           int dy,
                           int dz,
                           std::uint16_t cost)
    {
        const int neighbor_x = x + dx;
        const int neighbor_y = y + dy;
        const int neighbor_z = z + dz;
        if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 ||
            neighbor_x > layout.cells[0] ||
            neighbor_y > layout.cells[1] ||
            neighbor_z > layout.cells[2])
        {
            return;
        }
        const std::size_t index = sampleIndex(layout, x, y, z);
        const std::uint16_t neighbor_distance =
            distance[sampleIndex(
                layout, neighbor_x, neighbor_y, neighbor_z)];
        if (neighbor_distance == kInfinity)
        {
            return;
        }
        const int candidate =
            static_cast<int>(neighbor_distance) + cost;
        if (candidate < distance[index] &&
            candidate <= maximum_distance)
        {
            distance[index] = static_cast<std::uint16_t>(candidate);
        }
    };

    for (int pass = 0; pass < 2; ++pass)
    {
        for (int z = 0; z <= layout.cells[2]; ++z)
        {
            for (int y = 0; y <= layout.cells[1]; ++y)
            {
                for (int x = 0; x <= layout.cells[0]; ++x)
                {
                    for (int dz = -1; dz <= 0; ++dz)
                    {
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                if ((dz == 0 && dy > 0) ||
                                    (dz == 0 && dy == 0 && dx >= 0))
                                {
                                    continue;
                                }
                                const int axes =
                                    (dx != 0) + (dy != 0) + (dz != 0);
                                relax(x, y, z, dx, dy, dz,
                                      axes == 1 ? 3 : (axes == 2 ? 4 : 5));
                            }
                        }
                    }
                }
            }
        }
        for (int z = layout.cells[2]; z >= 0; --z)
        {
            for (int y = layout.cells[1]; y >= 0; --y)
            {
                for (int x = layout.cells[0]; x >= 0; --x)
                {
                    for (int dz = 0; dz <= 1; ++dz)
                    {
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                if ((dz == 0 && dy < 0) ||
                                    (dz == 0 && dy == 0 && dx <= 0))
                                {
                                    continue;
                                }
                                const int axes =
                                    (dx != 0) + (dy != 0) + (dz != 0);
                                relax(x, y, z, dx, dy, dz,
                                      axes == 1 ? 3 : (axes == 2 ? 4 : 5));
                            }
                        }
                    }
                }
            }
        }
    }

    const float inverse_band =
        1.0f / std::max(1.0f, bandVoxels * 3.0f);
    for (std::size_t index = 0; index < expected_size; ++index)
    {
        if (requested[index] == 0 ||
            (*supported)[index] != 0 ||
            is_vetoed(index) ||
            distance[index] == kInfinity ||
            distance[index] > maximum_distance)
        {
            continue;
        }
        const float magnitude = std::clamp(
            (static_cast<float>(distance[index]) - 0.5f) *
                inverse_band,
            1.0e-4f,
            1.0f);
        (*tsdf)[index] = occupied[index] != 0
            ? -magnitude
            : magnitude;
        (*supported)[index] = 1;
        ++statistics.recoveredSampleCount;
    }
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
    const bool requested_robust_frame_quality_rejection =
        options.enableRobustFrameQualityRejection;
    const float requested_robust_frame_quality_rejection_sigma =
        options.robustFrameQualityRejectionSigma;
    const float requested_robust_frame_quality_maximum_rejected_ratio =
        options.robustFrameQualityMaximumRejectedRatio;
    const int requested_robust_frame_quality_minimum_retained_frames =
        options.robustFrameQualityMinimumRetainedFrames;
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
    std::vector<DepthFusionView> fusion_views;
    fusion_views.reserve(static_cast<std::size_t>(frames.size()));
    QVector<int> auxiliary_surface_only_ref_indices;
    for (const DepthTsdfFrame &frame : frames)
    {
        raw_frame_quality_weights.push_back(std::clamp(
            frame.frameQualityWeight, 0.0f, 1.0f));
        DepthFusionView view;
        view.frameIndex = static_cast<int>(fusion_views.size());
        view.refIndex = frame.refIndex;
        view.cameraCenter = frame.camera.cameraCenter();
        fusion_views.push_back(view);
        if (frame.auxiliarySurfaceOnly)
        {
            auxiliary_surface_only_ref_indices.push_back(frame.refIndex);
        }
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
    for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
    {
        if (frames[frame_index].auxiliarySurfaceOnly)
        {
            effective_frame_quality_weights[frame_index] *= std::clamp(
                options.validationOnlyFrameWeightMultiplier, 0.05f, 1.0f);
        }
    }
    QVector<int> rejected_frame_indices;
    QVector<int> rejected_frame_ref_indices;
    QVector<int> coverage_protected_ref_indices;
    const bool robust_frame_quality_rejection_enabled =
        requested_robust_frame_quality_rejection &&
        robust_frame_quality_scale > 0.0f;
    if (robust_frame_quality_rejection_enabled)
    {
        QVector<int> low_tail_candidates;
        for (int frame_index = 0;
             frame_index < raw_frame_quality_weights.size();
             ++frame_index)
        {
            const float low_tail_sigma =
                (robust_frame_quality_median -
                 raw_frame_quality_weights[frame_index]) /
                robust_frame_quality_scale;
            if (low_tail_sigma >
                std::max(0.0f,
                         requested_robust_frame_quality_rejection_sigma))
            {
                low_tail_candidates.push_back(frame_index);
            }
        }
        std::sort(
            low_tail_candidates.begin(),
            low_tail_candidates.end(),
            [&raw_frame_quality_weights](int left, int right)
            {
                return raw_frame_quality_weights[left] <
                    raw_frame_quality_weights[right];
            });
        const int frame_count = static_cast<int>(frames.size());
        const int minimum_retained_frames = std::clamp(
            std::max(options.minimumInputFrames,
                     requested_robust_frame_quality_minimum_retained_frames),
            1,
            frame_count);
        const int ratio_limit = static_cast<int>(std::floor(
            frame_count *
            std::clamp(
                requested_robust_frame_quality_maximum_rejected_ratio,
                0.0f,
                0.50f)));
        const int rejection_limit = std::max(
            0,
            std::min(ratio_limit,
                     frame_count - minimum_retained_frames));
        for (int candidate_index = 0;
             candidate_index < low_tail_candidates.size() &&
             candidate_index < rejection_limit;
             ++candidate_index)
        {
            const int frame_index = low_tail_candidates[candidate_index];
            if (options.enableOrbitalFrameCoverageProtection)
            {
                const std::vector<float> trial_weights(
                    effective_frame_quality_weights.cbegin(),
                    effective_frame_quality_weights.cend());
                if (!DepthFusionFramePolicy::canRejectWithoutCoverageGap(
                        fusion_views,
                        trial_weights,
                        frame_index,
                        options.maximumOrbitalAngularGapRatio,
                        minimum_retained_frames))
                {
                    effective_frame_quality_weights[frame_index] = std::max(
                        effective_frame_quality_weights[frame_index],
                        raw_frame_quality_weights[frame_index] *
                            std::clamp(
                                options.coverageProtectedFrameMinimumMultiplier,
                                0.05f,
                                1.0f));
                    coverage_protected_ref_indices.push_back(
                        frames[frame_index].refIndex);
                    continue;
                }
            }
            effective_frame_quality_weights[frame_index] = 0.0f;
            rejected_frame_indices.push_back(frame_index);
            rejected_frame_ref_indices.push_back(frames[frame_index].refIndex);
        }
    }
    std::vector<float> final_frame_quality_weights(
        effective_frame_quality_weights.cbegin(),
        effective_frame_quality_weights.cend());
    OrbitalCoverageStatistics orbital_coverage =
        options.enableOrbitalFrameCoverageProtection
        ? DepthFusionFramePolicy::evaluateOrbitalCoverage(
              fusion_views, final_frame_quality_weights)
        : OrbitalCoverageStatistics{};
    QVector<int> gap_quality_floor_ref_indices;
    if (options.enableOrbitalGapBoundaryRecovery &&
        orbital_coverage.significantGap)
    {
        for (const OrbitalFrameRoleAssignment &assignment :
             orbital_coverage.frameRoles)
        {
            if (assignment.frameIndex < 0 ||
                assignment.frameIndex >= effective_frame_quality_weights.size() ||
                effective_frame_quality_weights[assignment.frameIndex] <= 0.0f)
            {
                continue;
            }
            float minimum_multiplier = 0.0f;
            if (assignment.role == OrbitalFrameRole::GapBoundary)
            {
                minimum_multiplier = std::clamp(
                    options.orbitalGapBoundaryMinimumQualityMultiplier,
                    0.05f,
                    1.0f);
            }
            else if (assignment.role == OrbitalFrameRole::GapOpposite)
            {
                minimum_multiplier = std::clamp(
                    options.orbitalGapOppositeMinimumQualityMultiplier,
                    0.05f,
                    1.0f);
            }
            if (minimum_multiplier <= 0.0f)
            {
                continue;
            }
            const float minimum_weight =
                raw_frame_quality_weights[assignment.frameIndex] *
                minimum_multiplier;
            if (effective_frame_quality_weights[assignment.frameIndex] + 1.0e-6f <
                minimum_weight)
            {
                effective_frame_quality_weights[assignment.frameIndex] =
                    minimum_weight;
                gap_quality_floor_ref_indices.push_back(assignment.refIndex);
            }
        }
        final_frame_quality_weights.assign(
            effective_frame_quality_weights.cbegin(),
            effective_frame_quality_weights.cend());
        orbital_coverage = DepthFusionFramePolicy::evaluateOrbitalCoverage(
            fusion_views, final_frame_quality_weights);
    }
    QJsonArray orbital_frame_roles;
    for (const OrbitalFrameRoleAssignment &assignment :
         orbital_coverage.frameRoles)
    {
        orbital_frame_roles.append(QJsonObject{
            {QStringLiteral("frame_index"), assignment.frameIndex},
            {QStringLiteral("ref_index"), assignment.refIndex},
            {QStringLiteral("azimuth_degrees"), assignment.azimuthDegrees},
            {QStringLiteral("role"),
             QString::fromLatin1(orbitalFrameRoleId(assignment.role))}
        });
    }
    result.statistics.effectiveRobustFrameQualityRejection =
        robust_frame_quality_rejection_enabled;
    result.statistics.robustFrameQualityRejectedFrameCount =
        rejected_frame_indices.size();
    result.statistics.acceptedFrameCount = static_cast<int>(std::count_if(
        effective_frame_quality_weights.cbegin(),
        effective_frame_quality_weights.cend(),
        [](float weight)
        {
            return weight > 0.0f;
        }));
    result.statistics.auxiliarySurfaceOnlyFrameCount =
        auxiliary_surface_only_ref_indices.size();
    result.statistics.auxiliarySurfaceOnlyRefIndices =
        auxiliary_surface_only_ref_indices;
    result.statistics.effectiveOrbitalFrameCoverageProtection =
        options.enableOrbitalFrameCoverageProtection;
    result.statistics.orbitalCoverageProtectedFrameCount =
        coverage_protected_ref_indices.size();
    result.statistics.orbitalCoverageProtectedRefIndices =
        coverage_protected_ref_indices;
    result.statistics.orbitalMedianAngularSpacingDegrees =
        orbital_coverage.medianAngularSpacingDegrees;
    result.statistics.orbitalMaximumAngularGapDegrees =
        orbital_coverage.maximumAngularGapDegrees;
    result.statistics.orbitalMaximumAngularGapRatio =
        orbital_coverage.maximumAngularGapRatio;
    result.statistics.orbitalSignificantAngularGap =
        orbital_coverage.significantGap;
    result.statistics.orbitalGapStartRefIndex =
        orbital_coverage.gapStartRefIndex;
    result.statistics.orbitalGapEndRefIndex =
        orbital_coverage.gapEndRefIndex;
    result.statistics.orbitalGapOppositeRefIndex =
        orbital_coverage.gapOppositeRefIndex;
    result.statistics.orbitalFrameRoles = orbital_frame_roles;
    result.statistics.effectiveOrbitalGapBoundaryRecovery =
        options.enableOrbitalGapBoundaryRecovery &&
        orbital_coverage.significantGap;
    result.statistics.orbitalGapQualityFloorFrameCount =
        gap_quality_floor_ref_indices.size();
    result.statistics.orbitalGapQualityFloorRefIndices =
        gap_quality_floor_ref_indices;
    result.statistics.effectiveOrbitalGapBoundaryMinimumObservationWeight =
        options.orbitalGapBoundaryMinimumObservationWeight;
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
            if (frame.useAdaptiveGeometryEvidence)
            {
                effective_depth_valid_masks.push_back(
                    frame.depthValidMask.clone());
                continue;
            }
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

    QVector<cv::Mat> reference_anchored_consensus_depths;
    if (options.enableCrossViewConsensusDepth)
    {
        reference_anchored_consensus_depths.reserve(frames.size());
        xjw::mvs::ReferenceAnchoredDepthConsensusOptions consensus_options;
        consensus_options.maximumInverseDepthSpread =
            options.maximumCrossViewConsensusInverseDepthSpread;
        consensus_options.minimumConfidence = std::max(
            consensus_options.minimumConfidence, options.minimumConfidence);
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame &frame = frames[frame_index];
            const cv::Mat &valid_mask = erosion_pixels > 0
                ? effective_depth_valid_masks[frame_index]
                : frame.depthValidMask;
            xjw::mvs::ReferenceAnchoredDepthConsensusResult consensus =
                xjw::mvs::makeReferenceAnchoredDepthConsensus(
                    frame.depth,
                    frame.inverseDepthMean,
                    frame.geometrySupportCount,
                    frame.inverseDepthRelativeSpread,
                    frame.confidence,
                    frame.crossViewRepairedMask,
                    frame.supportMask,
                    valid_mask,
                    consensus_options);
            reference_anchored_consensus_depths.push_back(
                consensus.depth.empty() ? frame.depth.clone() : std::move(consensus.depth));
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

    QVector<DepthTsdfFrame> retained_frames;
    retained_frames.reserve(frames.size() - rejected_frame_indices.size());
    for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
    {
        if (effective_frame_quality_weights[frame_index] > 0.0f)
        {
            retained_frames.push_back(frames[frame_index]);
        }
    }
    const DepthTsdfBoundsResult bounds = estimateBounds(retained_frames);
    if (!bounds.ok)
    {
        result.errorMessage = bounds.errorMessage;
        return result;
    }
    result = validateAllocation(bounds.minimum, bounds.maximum, options);
    result.statistics.boundsCandidateSampleCount = bounds.candidateSampleCount;
    result.statistics.boundsTrustedSampleCount = bounds.trustedSampleCount;
    result.statistics.boundsSelectedSampleCount = bounds.sampleCount;
    result.statistics.boundsUsedEvidenceAwareSamples = bounds.usedEvidenceAwareSamples;
    result.statistics.boundsFellBackToCandidateSamples = bounds.fellBackToCandidateSamples;
    result.statistics.boundsSelectionReason = bounds.selectionReason;
    result.statistics.inputFrameCount = frames.size();
    result.statistics.acceptedFrameCount = retained_frames.size();
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
    result.statistics.effectiveRobustFrameQualityRejection =
        robust_frame_quality_rejection_enabled;
    result.statistics.robustFrameQualityRejectedFrameCount =
        rejected_frame_indices.size();
    result.statistics.robustFrameQualityRejectedRefIndices =
        rejected_frame_ref_indices;
    result.statistics.auxiliarySurfaceOnlyFrameCount =
        auxiliary_surface_only_ref_indices.size();
    result.statistics.auxiliarySurfaceOnlyRefIndices =
        auxiliary_surface_only_ref_indices;
    result.statistics.effectiveOrbitalFrameCoverageProtection =
        options.enableOrbitalFrameCoverageProtection;
    result.statistics.orbitalCoverageProtectedFrameCount =
        coverage_protected_ref_indices.size();
    result.statistics.orbitalCoverageProtectedRefIndices =
        coverage_protected_ref_indices;
    result.statistics.orbitalMedianAngularSpacingDegrees =
        orbital_coverage.medianAngularSpacingDegrees;
    result.statistics.orbitalMaximumAngularGapDegrees =
        orbital_coverage.maximumAngularGapDegrees;
    result.statistics.orbitalMaximumAngularGapRatio =
        orbital_coverage.maximumAngularGapRatio;
    result.statistics.orbitalSignificantAngularGap =
        orbital_coverage.significantGap;
    result.statistics.orbitalGapStartRefIndex =
        orbital_coverage.gapStartRefIndex;
    result.statistics.orbitalGapEndRefIndex =
        orbital_coverage.gapEndRefIndex;
    result.statistics.orbitalGapOppositeRefIndex =
        orbital_coverage.gapOppositeRefIndex;
    result.statistics.orbitalFrameRoles = orbital_frame_roles;
    result.statistics.effectiveOrbitalGapBoundaryRecovery =
        options.enableOrbitalGapBoundaryRecovery &&
        orbital_coverage.significantGap;
    result.statistics.orbitalGapQualityFloorFrameCount =
        gap_quality_floor_ref_indices.size();
    result.statistics.orbitalGapQualityFloorRefIndices =
        gap_quality_floor_ref_indices;
    result.statistics.effectiveOrbitalGapBoundaryMinimumObservationWeight =
        options.orbitalGapBoundaryMinimumObservationWeight;
    if (!result.ok)
    {
        return result;
    }
    result.ok = false;
    std::vector<std::uint8_t> orbital_gap_boundary_frames(
        static_cast<std::size_t>(frames.size()), 0);
    if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
    {
        for (const OrbitalFrameRoleAssignment &assignment :
             orbital_coverage.frameRoles)
        {
            if (assignment.role == OrbitalFrameRole::GapBoundary &&
                assignment.frameIndex >= 0 &&
                assignment.frameIndex < frames.size())
            {
                orbital_gap_boundary_frames[static_cast<std::size_t>(
                    assignment.frameIndex)] = 1;
            }
        }
    }
    if (options.progress)
    {
        options.progress(QStringLiteral("正在融合置信度加权 TSDF..."), 5);
    }

    std::vector<float> tsdf;
    std::vector<float> weight;
    std::vector<float> evidenceSupportWeight;
    std::vector<float> maximumObservationWeight;
    std::vector<float> maximumEvidenceSupportObservationWeight;
    std::vector<std::uint16_t> maximumGeometrySupportCount;
    std::vector<std::uint8_t> strongAdaptiveSurfaceObservation;
    std::vector<std::uint16_t> support;
    std::vector<std::uint16_t> geometrySourceMask;
    std::vector<std::uint16_t> minimumInverseDepthSpread;
    std::vector<float> surfaceTsdfWeightedSum;
    std::vector<float> surfaceObservationWeight;
    std::vector<std::uint8_t> crossViewRepairedSurfaceWeight;
    std::vector<float> orbitalGapBoundaryObservationWeight;
    std::vector<float> contourBandObservationWeight;
    std::vector<DepthVisibilityHistogram> visibilityHistograms;
    try
    {
        tsdf.assign(static_cast<std::size_t>(result.layout.sampleCount), 1.0f);
        weight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        evidenceSupportWeight.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumObservationWeight.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumEvidenceSupportObservationWeight.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumGeometrySupportCount.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0);
        strongAdaptiveSurfaceObservation.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0);
        support.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
        if (options.enableSurfacePatchSupport ||
            options.enableContourBandZeroCrossingSupport ||
            options.enableCrossViewAnchoredSurfaceRecovery ||
            options.enableGlobalImplicitRegularization ||
            options.enableAdaptiveTgvRegularization ||
            result.statistics.effectiveOrbitalGapBoundaryRecovery)
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
            if (options.enableCrossViewAnchoredSurfaceRecovery)
            {
                crossViewRepairedSurfaceWeight.assign(
                    static_cast<std::size_t>(result.layout.sampleCount), 0);
            }
            if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
            {
                orbitalGapBoundaryObservationWeight.assign(
                    static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            }
            if (options.enableContourBandZeroCrossingSupport)
            {
                contourBandObservationWeight.assign(
                    static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            }
        }
        if (options.enableAdaptiveTgvRegularization)
        {
            visibilityHistograms.resize(
                static_cast<std::size_t>(result.layout.sampleCount));
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
    const float base_truncation_voxels =
        std::max(1.0f, options.truncationVoxels);
    const DepthUncertaintyBandEstimate uncertainty_band =
        estimateDepthUncertaintyBand(
            retained_frames, options, maximum_voxel_size);
    const bool uncertainty_adaptation_available =
        options.enableUncertaintyAdaptiveTruncation &&
        uncertainty_band.p90Voxels >
            base_truncation_voxels *
                std::max(
                    1.0f, options.uncertaintyAdaptiveActivationRatio);
    const bool orbital_gap_adaptive_truncation =
        uncertainty_adaptation_available &&
        options.enableOrbitalGapAdaptiveTruncation &&
        orbital_coverage.significantGap;
    const float effective_uncertainty_adaptive_scale =
        orbital_gap_adaptive_truncation
        ? std::max(
              options.uncertaintyAdaptiveScale,
              options.orbitalGapAdaptiveTruncationScale)
        : options.uncertaintyAdaptiveScale;
    const float adaptive_maximum_truncation_voxels = std::max(
        {base_truncation_voxels,
         options.uncertaintyAdaptiveMaximumTruncationVoxels,
         orbital_gap_adaptive_truncation
             ? options.orbitalGapAdaptiveMaximumTruncationVoxels
             : base_truncation_voxels});
    const float uncertainty_adaptive_added_voxels =
        uncertainty_adaptation_available
        ? std::max(0.0f, effective_uncertainty_adaptive_scale) *
              uncertainty_band.p90Voxels
        : 0.0f;
    const float effective_truncation_voxels = std::clamp(
        base_truncation_voxels + uncertainty_adaptive_added_voxels,
        base_truncation_voxels,
        adaptive_maximum_truncation_voxels);
    const float base_surface_support_band_voxels =
        options.surfaceSupportBandVoxels > 0.0f
        ? std::clamp(
              options.surfaceSupportBandVoxels,
              0.5f,
              base_truncation_voxels)
        : base_truncation_voxels;
    const float effective_surface_support_band_voxels =
        std::clamp(
            base_surface_support_band_voxels +
                uncertainty_adaptive_added_voxels,
            base_surface_support_band_voxels,
            effective_truncation_voxels);
    const float truncation = maximum_voxel_size * effective_truncation_voxels;
    const float surface_support_distance =
        maximum_voxel_size * effective_surface_support_band_voxels;
    const float weak_evidence_surface_band_voxels =
        options.weakEvidenceSurfaceBandVoxels > 0.0f
        ? std::clamp(
              options.weakEvidenceSurfaceBandVoxels,
              0.5f,
              effective_truncation_voxels)
        : effective_surface_support_band_voxels;
    const float weak_evidence_surface_distance =
        maximum_voxel_size * weak_evidence_surface_band_voxels;
    const float maximum_free_space_distance =
        options.maximumFreeSpaceVoxels > 0.0f
        ? std::max({result.layout.voxelSize[0],
                    result.layout.voxelSize[1],
                    result.layout.voxelSize[2]}) *
              std::max(
                  effective_truncation_voxels,
                  options.maximumFreeSpaceVoxels)
        : std::numeric_limits<float>::infinity();
    DepthTsdfNarrowBandActivation narrow_band_activation;
    if (options.enableNarrowBandActivation)
    {
        if (options.progress)
        {
            options.progress(
                QStringLiteral("正在激活有效深度附近的 TSDF 窄带..."), 4);
        }
        std::vector<DepthTsdfNarrowBandFrameView> activation_frames;
        activation_frames.reserve(static_cast<std::size_t>(frames.size()));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (effective_frame_quality_weights[frame_index] <= 0.0f)
            {
                continue;
            }
            const DepthTsdfFrame &frame = frames[frame_index];
            DepthTsdfNarrowBandFrameView view;
            view.camera = &frame.camera;
            view.depth = &frame.depth;
            view.depthValidMask = erosion_pixels > 0
                ? &effective_depth_valid_masks[frame_index]
                : &frame.depthValidMask;
            view.supportMask = &frame.supportMask;
            activation_frames.push_back(view);
        }

        DepthTsdfNarrowBandActivationOptions activation_options;
        activation_options.blockSizeSamples = std::clamp(
            options.narrowBandActivationBlockSizeSamples, 2, 32);
        activation_options.depthStride = std::clamp(
            options.narrowBandActivationDepthStride, 1, 16);
        activation_options.truncationDistance = truncation;
        activation_options.rayStepVoxels = std::clamp(
            options.narrowBandActivationRayStepVoxels, 0.25f, 4.0f);
        activation_options.haloBlocks = std::clamp(
            options.narrowBandActivationHaloBlocks, 0, 3);
        activation_options.isCancelled = options.isCancelled;
        if (!narrow_band_activation.build(
                result.layout, activation_frames, activation_options))
        {
            result.errorMessage = narrow_band_activation.wasCancelled()
                ? QStringLiteral("TSDF 窄带激活已取消")
                : QStringLiteral("TSDF 窄带激活失败：输入布局或参数无效");
            return result;
        }
        const DepthTsdfNarrowBandActivationStatistics &activation_statistics =
            narrow_band_activation.statistics();
        if (activation_statistics.activeBlocks == 0)
        {
            result.errorMessage = QStringLiteral(
                "TSDF 窄带激活没有找到可用深度样本；"
                "无效深度和蒙版外区域保持为 unknown");
            return result;
        }
        result.statistics.narrowBandActivationTotalBlockCount =
            activation_statistics.totalBlocks;
        result.statistics.narrowBandActivationActiveBlockCount =
            activation_statistics.activeBlocks;
        result.statistics.narrowBandActivationValidSourceSampleCount =
            activation_statistics.validSourceSamples;
        result.statistics.narrowBandActivationMarkedRaySampleCount =
            activation_statistics.markedRaySamples;
    }
    unsigned long long integratedVoxelUpdates = 0;
    unsigned long long narrowBandActivationSkippedSampleCount = 0;
    unsigned long long rejectedProjectionCount = 0;
    unsigned long long rejectedSupportMaskCount = 0;
    unsigned long long supportMaskFreeSpaceUpdateCount = 0;
    unsigned long long supportMaskFreeSpaceSurfaceVetoCount = 0;
    unsigned long long auxiliaryOutsideSurfaceBandRejectedCount = 0;
    unsigned long long rejectedDepthValidCount = 0;
    unsigned long long rejectedDepthCount = 0;
    unsigned long long rejectedConfidenceCount = 0;
    unsigned long long subpixelObservationCount = 0;
    unsigned long long recoveredNeighborObservationCount = 0;
    unsigned long long discontinuityRejectedCandidateCount = 0;
    unsigned long long rejectedGeometryConsistencyCount = 0;
    unsigned long long rejectedInvalidNearestPixelRecoveryCount = 0;
    unsigned long long crossViewConsensusDepthObservationCount = 0;
    unsigned long long unconfirmedNativeObservationCount = 0;
    unsigned long long weakNativeObservationCount = 0;
    unsigned long long repairedObservationCount = 0;
    unsigned long long strongNativeObservationCount = 0;
    unsigned long long inverseDepthSpreadDownweightedObservationCount = 0;
    unsigned long long inverseDepthSpreadVeryWeakObservationCount = 0;
    unsigned long long inverseDepthSpreadSupportLiftedObservationCount = 0;
    unsigned long long weakEvidenceOutsideSurfaceBandRejectedCount = 0;
    std::atomic_bool cancelled{false};
    std::atomic<int> completed_z_slices{0};
    std::atomic<int> last_progress_percent{5};
    const int zSamples = result.layout.cells[2] + 1;
#ifdef MESHING_OPENMP
    const int workerCount = options.workerCount > 0 ? options.workerCount : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(workerCount) \
    reduction(+:integratedVoxelUpdates,narrowBandActivationSkippedSampleCount,rejectedProjectionCount,rejectedSupportMaskCount,supportMaskFreeSpaceUpdateCount,supportMaskFreeSpaceSurfaceVetoCount,auxiliaryOutsideSurfaceBandRejectedCount,rejectedDepthValidCount,rejectedDepthCount,rejectedConfidenceCount,subpixelObservationCount,recoveredNeighborObservationCount,discontinuityRejectedCandidateCount,rejectedGeometryConsistencyCount,rejectedInvalidNearestPixelRecoveryCount,crossViewConsensusDepthObservationCount,unconfirmedNativeObservationCount,weakNativeObservationCount,repairedObservationCount,strongNativeObservationCount,inverseDepthSpreadDownweightedObservationCount,inverseDepthSpreadVeryWeakObservationCount,inverseDepthSpreadSupportLiftedObservationCount,weakEvidenceOutsideSurfaceBandRejectedCount)
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
                if (options.enableNarrowBandActivation &&
                    !narrow_band_activation.isSampleActive(x, y, z))
                {
                    ++narrowBandActivationSkippedSampleCount;
                    continue;
                }
                int support_mask_free_space_votes = 0;
                for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
                {
                    if (effective_frame_quality_weights[frame_index] <= 0.0f)
                    {
                        continue;
                    }
                    const DepthTsdfFrame &frame = frames[frame_index];
                    double pixel[2]{};
                    double voxelDepth = 0.0;
                    if (!frame.camera.projectWorldPointWithDepth(world, pixel, voxelDepth))
                    {
                        ++rejectedProjectionCount;
                        continue;
                    }
                    const cv::Mat &depth_valid_mask =
                        !frame.useAdaptiveGeometryEvidence && erosion_pixels > 0
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
                            : cv::Mat(),
                        options.enableCrossViewConsensusDepth
                            ? reference_anchored_consensus_depths[frame_index]
                            : cv::Mat());
                    if (!observation.valid)
                    {
                        switch (observation.failure)
                        {
                        case DepthTsdfObservationFailure::SupportMask:
                            if (options.enableSupportMaskFreeSpaceCarving &&
                                !frame.auxiliarySurfaceOnly &&
                                !frame.useAdaptiveGeometryEvidence)
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
                    if (frame.auxiliarySurfaceOnly &&
                        std::fabs(signedDistance) > surface_support_distance)
                    {
                        ++auxiliaryOutsideSurfaceBandRejectedCount;
                        continue;
                    }
                    if (observationUsesSurfaceOnlyIntegration(
                            observation, options) &&
                        std::fabs(signedDistance) >
                            weak_evidence_surface_distance)
                    {
                        ++weakEvidenceOutsideSurfaceBandRejectedCount;
                        continue;
                    }
                    const float evidence_weight_multiplier =
                        observationEvidenceWeightMultiplier(observation, options);
                    const float evidence_support_weight_multiplier =
                        observationEvidenceSupportWeightMultiplier(
                            observation, options);
                    const float spread_weight_multiplier =
                        observationInverseDepthSpreadWeightMultiplier(
                            observation, options);
                    const float spread_support_weight_multiplier =
                        observationInverseDepthSpreadSupportWeightMultiplier(
                            observation, options);
                    const float observationWeight =
                        confidence *
                        effective_frame_quality_weights[frame_index] *
                        evidence_weight_multiplier *
                        spread_weight_multiplier;
                    const float evidenceSupportObservationWeight =
                        confidence *
                        effective_frame_quality_weights[frame_index] *
                        evidence_support_weight_multiplier *
                        spread_support_weight_multiplier;
                    if (spread_weight_multiplier < 1.0f)
                    {
                        ++inverseDepthSpreadDownweightedObservationCount;
                    }
                    if (options.enableInverseDepthSpreadWeighting &&
                        spread_weight_multiplier <=
                        options.minimumInverseDepthSpreadWeightMultiplier +
                            1.0e-6f)
                    {
                        ++inverseDepthSpreadVeryWeakObservationCount;
                    }
                    if (spread_support_weight_multiplier >
                        spread_weight_multiplier + 1.0e-6f)
                    {
                        ++inverseDepthSpreadSupportLiftedObservationCount;
                    }
                    if (observation.usedCrossViewRepairedDepth)
                    {
                        ++repairedObservationCount;
                    }
                    else if (observation.geometrySupportCount == 0)
                    {
                        ++unconfirmedNativeObservationCount;
                    }
                    else if (observation.geometrySupportCount == 1)
                    {
                        ++weakNativeObservationCount;
                    }
                    else
                    {
                        ++strongNativeObservationCount;
                    }
                    if (!visibilityHistograms.empty() &&
                        options.adaptiveTgvUseGlobalVisibilityField)
                    {
                        visibilityHistograms[index].add(
                            std::clamp(
                                signedDistance / truncation,
                                -1.0f,
                                1.0f),
                            observationWeight);
                    }
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
                    if (!visibilityHistograms.empty() &&
                        !options.adaptiveTgvUseGlobalVisibilityField)
                    {
                        visibilityHistograms[index].add(
                            normalized, observationWeight);
                    }
                    integrateWeighted(&tsdf[index],
                                      &weight[index],
                                      normalized,
                                      observationWeight);
                    maximumObservationWeight[index] = std::max(
                        maximumObservationWeight[index], observationWeight);
                    evidenceSupportWeight[index] +=
                        evidenceSupportObservationWeight;
                    maximumEvidenceSupportObservationWeight[index] = std::max(
                        maximumEvidenceSupportObservationWeight[index],
                        evidenceSupportObservationWeight);
                    ++integratedVoxelUpdates;
                    if (std::fabs(signedDistance) <= surface_support_distance)
                    {
                        maximumGeometrySupportCount[index] = std::max(
                            maximumGeometrySupportCount[index],
                            observation.geometrySupportCount);
                        if (!frame.auxiliarySurfaceOnly &&
                            observationHasStrongAdaptiveGeometryEvidence(
                                observation, options))
                        {
                            strongAdaptiveSurfaceObservation[index] = 1;
                        }
                        if (options.enableSurfacePatchSupport ||
                            options.enableContourBandZeroCrossingSupport ||
                            options.enableCrossViewAnchoredSurfaceRecovery ||
                            options.enableGlobalImplicitRegularization ||
                            options.enableAdaptiveTgvRegularization ||
                            result.statistics.effectiveOrbitalGapBoundaryRecovery)
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
                            if (!crossViewRepairedSurfaceWeight.empty() &&
                                observation.usedCrossViewRepairedDepth)
                            {
                                const int quantized_weight =
                                    static_cast<int>(std::lround(
                                        std::clamp(
                                            observationWeight, 0.0f, 1.0f) *
                                        255.0f));
                                crossViewRepairedSurfaceWeight[index] =
                                    static_cast<std::uint8_t>(std::max(
                                        static_cast<int>(
                                            crossViewRepairedSurfaceWeight[index]),
                                        quantized_weight));
                            }
                            if (!orbitalGapBoundaryObservationWeight.empty() &&
                                orbital_gap_boundary_frames[
                                    static_cast<std::size_t>(frame_index)] != 0)
                            {
                                orbitalGapBoundaryObservationWeight[index] =
                                    std::max(
                                        orbitalGapBoundaryObservationWeight[index],
                                        observationWeight);
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
                    const bool has_surface_evidence =
                        support[index] > 0 ||
                        (!surfaceObservationWeight.empty() &&
                         surfaceObservationWeight[index] > 0.0f);
                    if (options.enableSurfaceEvidenceFreeSpaceVeto &&
                        has_surface_evidence)
                    {
                        supportMaskFreeSpaceSurfaceVetoCount +=
                            support_mask_free_space_votes;
                        continue;
                    }
                    const float carving_weight =
                        std::max(0.0f, options.supportMaskFreeSpaceWeight) *
                        support_mask_free_space_votes;
                    integrateWeighted(
                        &tsdf[index],
                        &weight[index],
                        1.0f,
                        carving_weight);
                    if (!visibilityHistograms.empty())
                    {
                        visibilityHistograms[index].add(1.0f, carving_weight);
                    }
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
    result.statistics.narrowBandActivationSkippedSampleCount =
        narrowBandActivationSkippedSampleCount;
    result.statistics.rejectedProjectionCount = rejectedProjectionCount;
    result.statistics.rejectedSupportMaskCount = rejectedSupportMaskCount;
    result.statistics.supportMaskFreeSpaceUpdateCount = supportMaskFreeSpaceUpdateCount;
    result.statistics.supportMaskFreeSpaceSurfaceVetoCount =
        supportMaskFreeSpaceSurfaceVetoCount;
    result.statistics.auxiliaryOutsideSurfaceBandRejectedCount =
        auxiliaryOutsideSurfaceBandRejectedCount;
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
    result.statistics.unconfirmedNativeObservationCount =
        unconfirmedNativeObservationCount;
    result.statistics.weakNativeObservationCount =
        weakNativeObservationCount;
    result.statistics.repairedObservationCount =
        repairedObservationCount;
    result.statistics.strongNativeObservationCount =
        strongNativeObservationCount;
    result.statistics.inverseDepthSpreadDownweightedObservationCount =
        inverseDepthSpreadDownweightedObservationCount;
    result.statistics.inverseDepthSpreadVeryWeakObservationCount =
        inverseDepthSpreadVeryWeakObservationCount;
    result.statistics.inverseDepthSpreadSupportLiftedObservationCount =
        inverseDepthSpreadSupportLiftedObservationCount;
    result.statistics.weakEvidenceOutsideSurfaceBandRejectedCount =
        weakEvidenceOutsideSurfaceBandRejectedCount;
    result.statistics.effectivePixelEvidenceWeighting =
        options.enablePixelEvidenceWeighting;
    result.statistics.effectiveUnconfirmedNativeObservationMultiplier =
        options.unconfirmedNativeObservationMultiplier;
    result.statistics.effectiveWeakNativeObservationMultiplier =
        options.weakNativeObservationMultiplier;
    result.statistics.effectiveRepairedObservationMultiplier =
        options.repairedObservationMultiplier;
    result.statistics.effectiveInverseDepthSpreadWeighting =
        options.enableInverseDepthSpreadWeighting;
    result.statistics.effectiveInverseDepthSpreadWeightKnee =
        options.inverseDepthSpreadWeightKnee;
    result.statistics.effectiveInverseDepthSpreadWeightZero =
        options.inverseDepthSpreadWeightZero;
    result.statistics.effectiveMinimumInverseDepthSpreadWeightMultiplier =
        options.minimumInverseDepthSpreadWeightMultiplier;
    result.statistics.effectiveInverseDepthSpreadSupportWeightDecoupling =
        options.enableInverseDepthSpreadSupportWeightDecoupling;
    result.statistics.effectiveInverseDepthSpreadSupportWeightExponent =
        std::clamp(
            options.inverseDepthSpreadSupportWeightExponent, 0.05f, 1.0f);
    result.statistics.effectiveEvidenceSupportWeightDecoupling =
        options.enableEvidenceSupportWeightDecoupling;
    result.statistics.effectiveEvidenceSupportWeightExponent =
        options.evidenceSupportWeightExponent;
    result.statistics.effectiveWeakEvidenceSurfaceOnlyIntegration =
        options.enableWeakEvidenceSurfaceOnlyIntegration;
    result.statistics.effectiveWeakEvidenceSurfaceBandVoxels =
        weak_evidence_surface_band_voxels;
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
    result.statistics.effectiveCrossViewAnchoredSurfaceRecovery =
        options.enableCrossViewAnchoredSurfaceRecovery;
    result.statistics.effectiveCrossViewAnchoredMinimumObservationWeight =
        options.crossViewAnchoredMinimumObservationWeight;
    result.statistics.effectiveCrossViewAnchoredMinimumSupportedCorners =
        options.crossViewAnchoredMinimumSupportedCorners;
    result.statistics.effectiveCrossViewAnchoredMinimumCellVotes =
        options.crossViewAnchoredMinimumCellVotes;
    result.statistics.effectiveCrossViewAnchoredGrowthPasses =
        options.crossViewAnchoredGrowthPasses;
    result.statistics.effectiveGeometryZeroCrossingCellSheets =
        options.enableGeometryZeroCrossingCellSheets;
    result.statistics
        .effectiveMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf =
        options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf;
    result.statistics.effectiveGlobalImplicitRegularization =
        options.enableGlobalImplicitRegularization;
    result.statistics.effectiveAdaptiveTgvRegularization =
        options.enableAdaptiveTgvRegularization;
    result.statistics.effectiveAdaptiveTgvGlobalVisibilityField =
        options.enableAdaptiveTgvRegularization &&
        options.adaptiveTgvUseGlobalVisibilityField;
    result.statistics.effectiveImplicitRegularizationLevels =
        options.implicitRegularizationLevels;
    result.statistics.effectiveImplicitRegularizationPassesPerLevel =
        options.implicitRegularizationPassesPerLevel;
    result.statistics.effectiveImplicitRegularizationSmoothness =
        options.implicitRegularizationSmoothness;
    result.statistics.effectiveImplicitRegularizationDataFidelity =
        options.implicitRegularizationDataFidelity;
    result.statistics.effectiveImplicitRegularizationMaximumUpdate =
        options.implicitRegularizationMaximumUpdate;
    result.statistics.effectiveImplicitRegularizationEdgeThreshold =
        options.implicitRegularizationEdgeThreshold;
    result.statistics.effectiveImplicitRegularizationRecoverAxialGaps =
        options.implicitRegularizationRecoverAxialGaps;
    result.statistics.effectiveImplicitRegularizationMinimumBridgeAxes =
        options.implicitRegularizationMinimumBridgeAxes;
    result.statistics
        .effectiveImplicitRegularizationMaximumBridgePredictionDelta =
        options.implicitRegularizationMaximumBridgePredictionDelta;
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
    result.statistics.effectiveUncertaintyAdaptiveTruncation =
        uncertainty_adaptation_available;
    result.statistics.uncertaintyAdaptiveSampleCount =
        uncertainty_band.sampleCount;
    result.statistics.uncertaintyAdaptiveP90Voxels =
        uncertainty_band.p90Voxels;
    result.statistics.uncertaintyAdaptiveAddedVoxels =
        effective_truncation_voxels - base_truncation_voxels;
    result.statistics.effectiveUncertaintyAdaptiveScale =
        effective_uncertainty_adaptive_scale;
    result.statistics.effectiveUncertaintyAdaptiveActivationRatio =
        options.uncertaintyAdaptiveActivationRatio;
    result.statistics.effectiveUncertaintyAdaptiveMaximumTruncationVoxels =
        adaptive_maximum_truncation_voxels;
    result.statistics.effectiveOrbitalGapAdaptiveTruncation =
        orbital_gap_adaptive_truncation;
    result.statistics.effectiveOrbitalGapAdaptiveTruncationScale =
        options.orbitalGapAdaptiveTruncationScale;
    result.statistics.effectiveOrbitalGapAdaptiveMaximumTruncationVoxels =
        options.orbitalGapAdaptiveMaximumTruncationVoxels;
    result.statistics.effectiveTruncationVoxels =
        effective_truncation_voxels;
    result.statistics.effectiveSurfaceSupportBandVoxels =
        effective_surface_support_band_voxels;
    result.statistics.effectiveMaximumFreeSpaceVoxels =
        options.maximumFreeSpaceVoxels > 0.0f
        ? std::max(
              effective_truncation_voxels, options.maximumFreeSpaceVoxels)
        : 0.0f;
    result.statistics.effectiveMinimumSupportMaskFreeSpaceViews = std::clamp(
        options.minimumSupportMaskFreeSpaceViews, 1, 16);
    result.statistics.effectiveSurfaceEvidenceFreeSpaceVeto =
        options.enableSurfaceEvidenceFreeSpaceVeto;
    result.statistics.effectiveNarrowBandActivation =
        options.enableNarrowBandActivation;
    result.statistics.effectiveNarrowBandActivationBlockSizeSamples =
        options.enableNarrowBandActivation
        ? std::clamp(options.narrowBandActivationBlockSizeSamples, 2, 32)
        : 0;
    result.statistics.effectiveNarrowBandActivationDepthStride =
        options.enableNarrowBandActivation
        ? std::clamp(options.narrowBandActivationDepthStride, 1, 16)
        : 0;
    result.statistics.effectiveNarrowBandActivationRayStepVoxels =
        options.enableNarrowBandActivation
        ? std::clamp(
              options.narrowBandActivationRayStepVoxels, 0.25f, 4.0f)
        : 0.0f;
    result.statistics.effectiveNarrowBandActivationHaloBlocks =
        options.enableNarrowBandActivation
        ? std::clamp(options.narrowBandActivationHaloBlocks, 0, 3)
        : 0;
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
    std::vector<std::uint8_t> adaptiveTgvExtractionSupport;
    std::vector<std::size_t> guarded_geometry_single_view_candidates;
    std::vector<std::size_t> orbital_gap_boundary_recovery_candidates;
    if (options.enableGeometrySingleViewNeighborhoodGuard)
    {
        guarded_geometry_single_view_candidates.reserve(
            static_cast<std::size_t>(result.layout.sampleCount / 32));
    }
    if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
    {
        orbital_gap_boundary_recovery_candidates.reserve(
            static_cast<std::size_t>(result.layout.sampleCount / 64));
    }
    for (std::size_t index = 0; index < supported.size(); ++index)
    {
        bool single_view_supported = false;
        bool multi_view_supported = false;
        bool geometry_verified_single_view_supported = false;
        const bool sample_supported = isSampleSupported(
            evidenceSupportWeight[index],
            support[index],
            maximumEvidenceSupportObservationWeight[index],
            options,
            &single_view_supported,
            &multi_view_supported,
            maximumGeometrySupportCount[index],
            &geometry_verified_single_view_supported,
            strongAdaptiveSurfaceObservation[index] != 0);
        const bool field_weight_supported = isSampleSupported(
            weight[index],
            support[index],
            maximumObservationWeight[index],
            options,
            nullptr,
            nullptr,
            maximumGeometrySupportCount[index],
            nullptr);
        if (sample_supported && !field_weight_supported)
        {
            ++result.statistics.evidenceSupportRecoveredSampleCount;
        }
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
        if (!sample_supported &&
            result.statistics.effectiveOrbitalGapBoundaryRecovery &&
            support[index] == 1 &&
            !orbitalGapBoundaryObservationWeight.empty() &&
            orbitalGapBoundaryObservationWeight[index] > 1.0e-6f)
        {
            ++result.statistics.orbitalGapBoundarySingleObservationCount;
            const float maximum_spread = std::min(
                options.maximumBoundaryRecoveryInverseDepthSpread,
                options.maximumCrossViewConsensusInverseDepthSpread);
            if (orbitalGapBoundaryObservationWeight[index] <
                options.orbitalGapBoundaryMinimumObservationWeight)
            {
                ++result.statistics.orbitalGapBoundaryRejectedWeightCount;
            }
            else if (maximumGeometrySupportCount[index] <
                     options.minimumBoundaryRecoveryGeometrySupport)
            {
                ++result.statistics
                      .orbitalGapBoundaryRejectedGeometrySupportCount;
            }
            else if (bitCount(geometrySourceMask[index]) <
                     std::max(2, options.minimumSurfacePatchSourceCount))
            {
                ++result.statistics.orbitalGapBoundaryRejectedSourceCount;
            }
            else if (minimumInverseDepthSpread[index] ==
                         std::numeric_limits<std::uint16_t>::max() ||
                     maximum_spread <= 0.0f ||
                     static_cast<float>(minimumInverseDepthSpread[index]) /
                             100000.0f >
                         maximum_spread)
            {
                ++result.statistics.orbitalGapBoundaryRejectedSpreadCount;
            }
            else if (surfaceObservationWeight[index] <= 1.0e-6f ||
                     weight[index] <= 1.0e-6f ||
                     surfaceObservationWeight[index] / weight[index] <
                         options.minimumSurfacePatchWeightRatio ||
                     std::fabs(tsdf[index]) >
                         options.maximumSurfacePatchAbsoluteTsdf)
            {
                ++result.statistics.orbitalGapBoundaryRejectedFieldCount;
            }
            else
            {
                orbital_gap_boundary_recovery_candidates.push_back(index);
            }
        }
        result.statistics.supportedSampleCount += supported[index] != 0;
    }
    if (result.statistics.effectiveOrbitalGapBoundaryRecovery &&
        !orbital_gap_boundary_recovery_candidates.empty())
    {
        const int accepted_count = growGeometryVerifiedSingleViewSamples(
            result.layout,
            tsdf,
            orbital_gap_boundary_recovery_candidates,
            options.minimumGeometrySingleViewNeighborCount,
            options.geometrySingleViewGrowthPasses,
            options.maximumGeometrySingleViewNeighborTsdfDelta,
            &supported);
        result.statistics.orbitalGapBoundaryRecoveryCandidateCount =
            orbital_gap_boundary_recovery_candidates.size();
        result.statistics.orbitalGapBoundaryRecoveryAcceptedCount =
            accepted_count;
        result.statistics.orbitalGapBoundaryRecoveryRejectedCount =
            orbital_gap_boundary_recovery_candidates.size() -
            static_cast<std::size_t>(accepted_count);
        result.statistics.singleViewSupportedSampleCount += accepted_count;
        result.statistics.geometryVerifiedSingleViewSupportedSampleCount +=
            accepted_count;
        result.statistics.supportedSampleCount += accepted_count;
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
         options.enableGeometryZeroCrossingRecovery ||
         options.enableCrossViewAnchoredSurfaceRecovery ||
         options.enableGlobalImplicitRegularization ||
         options.enableAdaptiveTgvRegularization) &&
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
        const float maximum_contour_band_absolute_tsdf = std::clamp(
            options.maximumContourBandAbsoluteTsdf,
            maximum_absolute_tsdf,
            0.95f);
        result.statistics.effectiveMaximumContourBandAbsoluteTsdf =
            maximum_contour_band_absolute_tsdf;
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
                        const bool has_contour_band_evidence =
                            options.enableContourBandZeroCrossingSupport &&
                            contourBandObservationWeight[index] > 1.0e-6f;
                        if (options.enableContourBandZeroCrossingSupport)
                        {
                            ++result.statistics
                                  .contourBandZeroCrossingConsideredSampleCount;
                            if (!has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedNoContourCount;
                            }
                        }
                        if (maximumObservationWeight[index] <
                            options.minimumSurfacePatchObservationWeight)
                        {
                            ++result.statistics.surfacePatchRejectedWeightCount;
                            if (has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedWeightCount;
                            }
                            continue;
                        }
                        if (bitCount(geometrySourceMask[index]) <
                            minimum_source_count)
                        {
                            ++result.statistics
                                  .surfacePatchRejectedSourceOverlapCount;
                            if (has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedSourceOverlapCount;
                            }
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
                            if (has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedDepthSpreadCount;
                            }
                            continue;
                        }
                        const bool insufficient_surface_weight_ratio =
                            surfaceObservationWeight[index] <= 1.0e-6f ||
                            weight[index] <= 1.0e-6f ||
                            surfaceObservationWeight[index] / weight[index] <
                                minimum_surface_weight_ratio;
                        const bool excessive_absolute_tsdf =
                            std::fabs(surface_candidate_tsdf[index]) >
                            (has_contour_band_evidence
                                 ? maximum_contour_band_absolute_tsdf
                                 : maximum_absolute_tsdf);
                        if (insufficient_surface_weight_ratio ||
                            excessive_absolute_tsdf)
                        {
                            ++result.statistics.surfacePatchRejectedFreeSpaceCount;
                            if (insufficient_surface_weight_ratio)
                            {
                                ++result.statistics
                                      .surfacePatchRejectedSurfaceWeightRatioCount;
                            }
                            if (excessive_absolute_tsdf)
                            {
                                ++result.statistics
                                      .surfacePatchRejectedAbsoluteTsdfCount;
                            }
                            if (has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedFreeSpaceCount;
                                if (insufficient_surface_weight_ratio)
                                {
                                    ++result.statistics
                                          .contourBandZeroCrossingRejectedSurfaceWeightRatioCount;
                                }
                                if (excessive_absolute_tsdf)
                                {
                                    ++result.statistics
                                          .contourBandZeroCrossingRejectedAbsoluteTsdfCount;
                                }
                            }
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
                            if (has_contour_band_evidence)
                            {
                                ++result.statistics
                                      .contourBandZeroCrossingRejectedNeighborhoodCount;
                            }
                            continue;
                        }
                        const bool normal_patch_supported =
                            options.enableSurfacePatchSupport &&
                            has_normal_agreement &&
                            agreeing_core_neighbor_count >=
                                minimum_core_neighbor_count;
                        const bool has_geometry_support =
                            maximumGeometrySupportCount[index] >=
                            options.minimumBoundaryRecoveryGeometrySupport;
                        const bool has_sign_pair =
                            same_sign_core_neighbor_count >= 1 &&
                            opposite_sign_core_neighbor_count >= 1;
                        const bool zero_crossing_supported =
                            has_contour_band_evidence &&
                            has_geometry_support &&
                            has_sign_pair;
                        if (!normal_patch_supported &&
                            !zero_crossing_supported)
                        {
                            ++result.statistics.surfacePatchRejectedNormalCount;
                            if (has_contour_band_evidence)
                            {
                                if (!has_geometry_support)
                                {
                                    ++result.statistics
                                          .contourBandZeroCrossingRejectedGeometrySupportCount;
                                }
                                else if (!has_sign_pair)
                                {
                                    ++result.statistics
                                          .contourBandZeroCrossingRejectedNoSignPairCount;
                                }
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
        if (options.enableCrossViewAnchoredSurfaceRecovery &&
            !crossViewRepairedSurfaceWeight.empty())
        {
            std::vector<std::uint8_t> repaired_eligible(supported.size(), 0);
            const int minimum_repaired_weight = std::clamp(
                static_cast<int>(std::ceil(
                    std::clamp(
                        options.crossViewAnchoredMinimumObservationWeight,
                        0.0f,
                        1.0f) *
                    255.0f)),
                1,
                255);
            const int minimum_geometry_support = std::max(
                2, options.minimumBoundaryRecoveryGeometrySupport);
            const float maximum_repaired_spread = std::min(
                options.maximumBoundaryRecoveryInverseDepthSpread,
                options.maximumSurfacePatchInverseDepthSpread);
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                if (crossViewRepairedSurfaceWeight[index] == 0)
                {
                    continue;
                }
                ++result.statistics.crossViewAnchoredObservedSampleCount;
                const std::uint16_t spread_value =
                    minimumInverseDepthSpread[index];
                const bool eligible =
                    supported[index] == 0 &&
                    support[index] > 0 &&
                    crossViewRepairedSurfaceWeight[index] >=
                        minimum_repaired_weight &&
                    maximumGeometrySupportCount[index] >=
                        minimum_geometry_support &&
                    bitCount(geometrySourceMask[index]) >=
                        minimum_source_count &&
                    maximum_repaired_spread > 0.0f &&
                    spread_value != std::numeric_limits<std::uint16_t>::max() &&
                    static_cast<float>(spread_value) / 100000.0f <=
                        maximum_repaired_spread &&
                    surfaceObservationWeight[index] > 1.0e-6f &&
                    weight[index] > 1.0e-6f &&
                    surfaceObservationWeight[index] / weight[index] >=
                        minimum_surface_weight_ratio &&
                    std::fabs(surface_candidate_tsdf[index]) <=
                        maximum_absolute_tsdf;
                repaired_eligible[index] = eligible;
                result.statistics.crossViewAnchoredEligibleSampleCount +=
                    eligible;
            }

            const int growth_passes = std::clamp(
                options.crossViewAnchoredGrowthPasses, 1, 4);
            for (int pass = 0; pass < growth_passes; ++pass)
            {
                const std::vector<std::uint8_t> supported_before_recovery =
                    supported;
                const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                    recoverGeometryVerifiedZeroCrossingSamples(
                        result.layout,
                        surface_candidate_tsdf,
                        weight,
                        geometrySourceMask,
                        repaired_eligible,
                        options.crossViewAnchoredMinimumSupportedCorners,
                        options.crossViewAnchoredMinimumCellVotes,
                        &supported);
                result.statistics.crossViewAnchoredCandidateSampleCount +=
                    recovery.candidateSampleCount;
                result.statistics.crossViewAnchoredRecoveredSampleCount +=
                    recovery.recoveredSampleCount;
                ++result.statistics.crossViewAnchoredExecutedGrowthPassCount;
                if (recovery.recoveredSampleCount == 0)
                {
                    break;
                }
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
        }
        std::vector<std::uint8_t> eligible;
        if (options.enableGeometryZeroCrossingRecovery ||
            options.enableGlobalImplicitRegularization ||
            options.enableAdaptiveTgvRegularization)
        {
            eligible.assign(supported.size(), 0);
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                const std::uint16_t spread_value =
                    minimumInverseDepthSpread[index];
                eligible[index] =
                    supported[index] == 0 &&
                    maximumEvidenceSupportObservationWeight[index] >=
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
        }
        if (options.enableGeometryZeroCrossingRecovery)
        {
            const std::vector<std::uint8_t> supported_before_recovery =
                supported;
            const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                options.enableGeometryZeroCrossingCellSheets
                ? recoverGeometryVerifiedZeroCrossingCellSheets(
                      result.layout,
                      surface_candidate_tsdf,
                      weight,
                      geometrySourceMask,
                      eligible,
                      options.geometryZeroCrossingMinimumSupportedCorners,
                      options.minimumGeometryZeroCrossingSheetCells,
                      options.minimumGeometryZeroCrossingSheetAnchorCells,
                      options
                          .maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf,
                      &supported)
                : recoverGeometryVerifiedZeroCrossingSamples(
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
            result.statistics.geometryZeroCrossingSheetCandidateCellCount =
                recovery.candidateCellCount;
            result.statistics.geometryZeroCrossingSheetAcceptedCellCount =
                recovery.acceptedCellCount;
            result.statistics.geometryZeroCrossingSheetComponentCount =
                recovery.componentCount;
            result.statistics.geometryZeroCrossingSheetAcceptedComponentCount =
                recovery.acceptedComponentCount;
            result.statistics
                .geometryZeroCrossingSheetRejectedSmallComponentCount =
                recovery.rejectedSmallComponentCount;
            result.statistics
                .geometryZeroCrossingSheetRejectedAnchorComponentCount =
                recovery.rejectedAnchorComponentCount;
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
        if (options.enableGlobalImplicitRegularization)
        {
            if (options.progress)
            {
                options.progress(
                    QStringLiteral("正在进行多尺度隐式场正则化..."), 73);
            }
            DepthImplicitFieldRegularizationOptions regularization_options;
            regularization_options.coarseToFineLevels =
                options.implicitRegularizationLevels;
            regularization_options.passesPerLevel =
                options.implicitRegularizationPassesPerLevel;
            regularization_options.smoothness =
                options.implicitRegularizationSmoothness;
            regularization_options.dataFidelity =
                options.implicitRegularizationDataFidelity;
            regularization_options.maximumUpdate =
                options.implicitRegularizationMaximumUpdate;
            regularization_options.edgeThreshold =
                options.implicitRegularizationEdgeThreshold;
            regularization_options.recoverAxialGaps =
                options.implicitRegularizationRecoverAxialGaps;
            regularization_options.minimumBridgeAxes =
                options.implicitRegularizationMinimumBridgeAxes;
            regularization_options.maximumBridgePredictionDelta =
                options.implicitRegularizationMaximumBridgePredictionDelta;
            const DepthImplicitFieldRegularizationStatistics regularization =
                DepthImplicitFieldRegularizer::regularize(
                    {result.layout.cells[0] + 1,
                     result.layout.cells[1] + 1,
                     result.layout.cells[2] + 1},
                    surface_candidate_tsdf,
                    surfaceObservationWeight,
                    geometrySourceMask,
                    eligible,
                    regularization_options,
                    &tsdf,
                    &supported,
                    options.isCancelled);
            result.statistics.implicitRegularizationBridgeCandidateCount =
                regularization.bridgeCandidateCount;
            result.statistics.implicitRegularizationRecoveredSampleCount =
                regularization.recoveredSampleCount;
            result.statistics.implicitRegularizationUpdateOperationCount =
                regularization.updateOperationCount;
            result.statistics.implicitRegularizationMeanAbsoluteUpdate =
                regularization.meanAbsoluteUpdate;
            result.statistics.implicitRegularizationMaximumAbsoluteUpdate =
                regularization.maximumAbsoluteUpdate;
            result.statistics.implicitRegularizationElapsedMs =
                regularization.elapsedMs;
            result.statistics.supportedSampleCount +=
                regularization.recoveredSampleCount;
            if (regularization.cancelled)
            {
                result.errorMessage =
                    QStringLiteral("TSDF 隐式场正则化已取消");
                return result;
            }
        }
        if (options.enableAdaptiveTgvRegularization)
        {
            if (options.progress)
            {
                options.progress(
                    QStringLiteral("正在构建 2:1 平衡可见性八叉树..."), 72);
            }

            std::vector<std::uint8_t> active(supported.size(), 0);
            std::vector<float> adaptive_field = tsdf;
            constexpr float kHalfHistogramBinWidth =
                1.0f / static_cast<float>(
                    kDepthVisibilityHistogramBinCount);
            result.statistics.effectiveAdaptiveTgvMaximumActiveAbsoluteField =
                options.adaptiveTgvMaximumActiveAbsoluteField;
            for (std::size_t index = 0; index < active.size(); ++index)
            {
                if (visibilityHistograms[index].empty())
                {
                    continue;
                }
                const bool unsupported_sample = weight[index] <= 1.0e-6f;
                if (unsupported_sample &&
                    (!options.adaptiveTgvUseGlobalVisibilityField ||
                     !options.adaptiveTgvRecoverUnsupportedSamples))
                {
                    continue;
                }
                const DepthVisibilityHistogramSummary histogram =
                    visibilityHistograms[index].summary();
                const float median = histogram.weightedMedian();
                float candidate_value = median;
                if (weight[index] <= 1.0e-6f)
                {
                    candidate_value = median;
                }
                else
                {
                    candidate_value = std::clamp(
                        tsdf[index],
                        median - kHalfHistogramBinWidth,
                        median + kHalfHistogramBinWidth);
                }
                if (std::fabs(candidate_value) >
                    options.adaptiveTgvMaximumActiveAbsoluteField)
                {
                    continue;
                }
                adaptive_field[index] = candidate_value;
                active[index] = 1;
                ++result.statistics.adaptiveTgvHistogramSampleCount;
                if (weight[index] <= 1.0e-6f)
                {
                    ++result.statistics
                          .adaptiveTgvGlobalVisibilitySampleCount;
                }
            }

            AdaptiveTsdfOctreeOptions octree_options;
            octree_options.maximumMergeLevel =
                options.adaptiveTgvMaximumMergeLevel;
            octree_options.minimumMergeAbsoluteField =
                options.adaptiveTgvMinimumMergeAbsoluteField;
            octree_options.maximumMergeFieldRange =
                options.adaptiveTgvMaximumMergeFieldRange;

            AdaptiveTsdfOctreeResult octree;
            const auto octree_start = std::chrono::steady_clock::now();
            try
            {
                octree = AdaptiveTsdfOctree::build(
                    {result.layout.cells[0] + 1,
                     result.layout.cells[1] + 1,
                     result.layout.cells[2] + 1},
                    adaptive_field,
                    weight,
                    geometrySourceMask,
                    active,
                    supported,
                    visibilityHistograms,
                    octree_options);
            }
            catch (const std::bad_alloc &)
            {
                result.errorMessage = QStringLiteral(
                    "自适应 TGV 八叉树内存分配失败");
                return result;
            }
            catch (const std::exception &error)
            {
                result.errorMessage = QStringLiteral(
                    "自适应 TGV 八叉树构建失败: %1")
                                          .arg(QString::fromUtf8(error.what()));
                return result;
            }
            result.statistics.adaptiveTgvOctreeElapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - octree_start)
                    .count();
            result.statistics.adaptiveTgvInputActiveSampleCount =
                octree.statistics.inputActiveSampleCount;
            result.statistics.adaptiveTgvLeafCount = octree.leaves.size();
            result.statistics.adaptiveTgvMergedNodeCount =
                octree.statistics.mergedNodeCount;
            result.statistics.adaptiveTgvBalanceSplitCount =
                octree.statistics.balanceSplitCount;
            result.statistics.adaptiveTgvTwoToOneBalanced =
                octree.statistics.twoToOneBalanced;
            if (octree.leaves.empty())
            {
                result.errorMessage = QStringLiteral(
                    "自适应 TGV 八叉树没有可求解节点");
                return result;
            }
            if (options.isCancelled && options.isCancelled())
            {
                result.errorMessage = QStringLiteral(
                    "自适应 TGV 八叉树构建已取消");
                return result;
            }

            SparseTgvOptions tgv_options;
            tgv_options.maximumIterations =
                options.adaptiveTgvMaximumIterations;
            tgv_options.minimumIterations =
                options.adaptiveTgvMinimumIterations;
            tgv_options.firstOrderWeight =
                options.adaptiveTgvFirstOrderWeight;
            tgv_options.secondOrderWeight =
                options.adaptiveTgvSecondOrderWeight;
            tgv_options.dataFidelity =
                options.adaptiveTgvDataFidelity;
            tgv_options.primalStep =
                options.adaptiveTgvPrimalStep;
            tgv_options.dualStep =
                options.adaptiveTgvDualStep;
            tgv_options.convergenceTolerance =
                options.adaptiveTgvConvergenceTolerance;
            const SparseTgvStatistics tgv = SparseTgvSolver::solve(
                tgv_options,
                &octree,
                options.isCancelled,
                [&](int iteration, int maximum_iterations)
                {
                    if (options.progress &&
                        (iteration == 1 || iteration % 5 == 0 ||
                         iteration == maximum_iterations))
                    {
                        options.progress(
                            QStringLiteral(
                                "正在进行稀疏 TGV 全局优化（%1/%2）...")
                                .arg(iteration)
                                .arg(maximum_iterations),
                            73 + iteration * 3 /
                                std::max(1, maximum_iterations));
                    }
                });
            result.statistics.adaptiveTgvIterationCount =
                tgv.iterationCount;
            result.statistics.adaptiveTgvInitialMeanAbsoluteCurvature =
                tgv.initialMeanAbsoluteCurvature;
            result.statistics.adaptiveTgvFinalMeanAbsoluteCurvature =
                tgv.finalMeanAbsoluteCurvature;
            result.statistics.adaptiveTgvFinalMeanAbsoluteUpdate =
                tgv.finalMeanAbsoluteUpdate;
            result.statistics.adaptiveTgvSolverElapsedMs =
                tgv.elapsedMs;
            if (tgv.cancelled)
            {
                result.errorMessage = QStringLiteral(
                    "稀疏 TGV 全局优化已取消");
                return result;
            }

            for (const AdaptiveTsdfOctreeNode &leaf : octree.leaves)
            {
                const int end_x = std::min(
                    result.layout.cells[0] + 1,
                    leaf.origin[0] + leaf.size);
                const int end_y = std::min(
                    result.layout.cells[1] + 1,
                    leaf.origin[1] + leaf.size);
                const int end_z = std::min(
                    result.layout.cells[2] + 1,
                    leaf.origin[2] + leaf.size);
                for (int z = leaf.origin[2]; z < end_z; ++z)
                {
                    for (int y = leaf.origin[1]; y < end_y; ++y)
                    {
                        for (int x = leaf.origin[0]; x < end_x; ++x)
                        {
                            const std::size_t index =
                                sampleIndex(result.layout, x, y, z);
                            if (active[index] != 0)
                            {
                                tsdf[index] = leaf.value;
                            }
                        }
                    }
                }
            }
            if (options.adaptiveTgvUseGlobalVisibilityField)
            {
                adaptiveTgvExtractionSupport = active;
                if (!options.adaptiveTgvRecoverUnsupportedSamples)
                {
                    for (std::size_t index = 0;
                         index < adaptiveTgvExtractionSupport.size();
                         ++index)
                    {
                        adaptiveTgvExtractionSupport[index] =
                            adaptiveTgvExtractionSupport[index] != 0 &&
                            supported[index] != 0
                            ? 1
                            : 0;
                    }
                }
            }
            else if (options.adaptiveTgvRecoverUnsupportedSamples)
            {
                std::vector<std::uint8_t> robust_eligible = eligible;
                const float maximum_conflict_ratio = std::clamp(
                    options.adaptiveTgvMaximumRecoveryConflictRatio,
                    0.0f,
                    0.5f);
                for (std::size_t index = 0;
                     index < robust_eligible.size();
                     ++index)
                {
                    result.statistics.adaptiveTgvRecoveryEligibleSampleCount +=
                        robust_eligible[index] != 0;
                    if (robust_eligible[index] != 0 &&
                        visibilityHistograms[index]
                                .summary()
                                .conflictingSignRatio() >
                            maximum_conflict_ratio)
                    {
                        robust_eligible[index] = 0;
                        ++result.statistics
                              .adaptiveTgvRecoveryConflictRejectedSampleCount;
                    }
                }
                const int recovery_passes = std::clamp(
                    options.adaptiveTgvRecoveryPasses, 1, 6);
                for (int pass = 0; pass < recovery_passes; ++pass)
                {
                    const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                        options.enableGeometryZeroCrossingCellSheets
                        ? recoverGeometryVerifiedZeroCrossingCellSheets(
                              result.layout,
                              tsdf,
                              weight,
                              geometrySourceMask,
                              robust_eligible,
                              options.adaptiveTgvMinimumRecoveryNeighbors,
                              options.minimumGeometryZeroCrossingSheetCells,
                              options.minimumGeometryZeroCrossingSheetAnchorCells,
                              options
                                  .maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf,
                              &supported)
                        : recoverGeometryVerifiedZeroCrossingSamples(
                              result.layout,
                              tsdf,
                              weight,
                              geometrySourceMask,
                              robust_eligible,
                              options.adaptiveTgvMinimumRecoveryNeighbors,
                              options.geometryZeroCrossingMinimumCellVotes,
                              &supported);
                    result.statistics.adaptiveTgvRecoveredSampleCount +=
                        recovery.recoveredSampleCount;
                    result.statistics.geometryZeroCrossingSheetCandidateCellCount +=
                        recovery.candidateCellCount;
                    result.statistics.geometryZeroCrossingSheetAcceptedCellCount +=
                        recovery.acceptedCellCount;
                    result.statistics.geometryZeroCrossingSheetComponentCount +=
                        recovery.componentCount;
                    result.statistics.geometryZeroCrossingSheetAcceptedComponentCount +=
                        recovery.acceptedComponentCount;
                    result.statistics
                        .geometryZeroCrossingSheetRejectedSmallComponentCount +=
                        recovery.rejectedSmallComponentCount;
                    result.statistics
                        .geometryZeroCrossingSheetRejectedAnchorComponentCount +=
                        recovery.rejectedAnchorComponentCount;
                    result.statistics.supportedSampleCount +=
                        recovery.recoveredSampleCount;
                    if (recovery.recoveredSampleCount == 0)
                    {
                        break;
                    }
                }
            }
        }
        result.statistics.surfacePatchCreatedComponentCount = 0;
    }
    const bool use_visual_hull_completion =
        options.enableVisualHullSignedDistanceCompletion &&
        !options.enableVisibilityOccupancyCompletion;
    result.statistics.effectiveVisualHullSignedDistanceCompletion =
        use_visual_hull_completion;
    result.statistics.effectiveVisualHullCompletionTopologyGuard =
        use_visual_hull_completion &&
        options.enableVisualHullCompletionTopologyGuard;
    result.statistics.effectiveVisualHullCompletionBandVoxels =
        use_visual_hull_completion
        ? std::clamp(options.visualHullCompletionBandVoxels, 1.0f, 24.0f)
        : 0.0f;
    std::vector<float> visual_hull_completion_tsdf;
    std::vector<std::uint8_t> visual_hull_completion_support;
    if (use_visual_hull_completion)
    {
        if (options.progress)
        {
            options.progress(
                QStringLiteral("正在构建轮廓约束的局部有符号距离先验..."),
                74);
        }
        const std::vector<std::uint8_t> occupied =
            buildVisualHullOccupancy(
                result.layout,
                retained_frames,
                options.visualHullCompletionMinimumVisibleViews,
                options.visualHullCompletionAllowedSilhouetteViolations,
                options.isCancelled);
        if (options.isCancelled && options.isCancelled())
        {
            result.errorMessage =
                QStringLiteral("TSDF 轮廓距离先验构建已取消");
            return result;
        }
        visual_hull_completion_tsdf = tsdf;
        visual_hull_completion_support = supported;
        std::vector<std::uint8_t> observed_conflict_veto;
        if (options.visualHullCompletionPreserveObservedTsdf &&
            occupied.size() == visual_hull_completion_support.size())
        {
            observed_conflict_veto.assign(
                visual_hull_completion_support.size(), 0);
            const float maximum_absolute_tsdf = std::clamp(
                options.visualHullCompletionMaximumObservedAbsoluteTsdf,
                0.05f,
                1.0f);
            const int minimum_geometry_support = std::max(
                1, options.visualHullCompletionMinimumGeometrySupport);
            for (std::size_t index = 0;
                 index < visual_hull_completion_support.size();
                 ++index)
            {
                if (weight[index] <= 1.0e-6f ||
                    maximumGeometrySupportCount[index] <
                        minimum_geometry_support ||
                    bitCount(geometrySourceMask[index]) < 2 ||
                    std::fabs(visual_hull_completion_tsdf[index]) >
                        maximum_absolute_tsdf)
                {
                    continue;
                }
                const bool hull_inside = occupied[index] != 0;
                const bool tsdf_inside =
                    visual_hull_completion_tsdf[index] < 0.0f;
                if (hull_inside != tsdf_inside)
                {
                    observed_conflict_veto[index] = 1;
                    ++result.statistics
                          .visualHullCompletionObservedConflictVetoSampleCount;
                }
                if (visual_hull_completion_support[index] == 0)
                {
                    visual_hull_completion_support[index] = 1;
                    ++result.statistics
                          .visualHullCompletionPreservedObservedSampleCount;
                }
            }
        }
        const std::vector<std::uint8_t> supported_before_hull_completion =
            visual_hull_completion_support;
        const DepthTsdfVisualHullCompletionStatistics completion =
            completeUnsupportedSamplesWithVisualHullSignedDistance(
                result.layout,
                occupied,
                result.statistics.effectiveVisualHullCompletionBandVoxels,
                &visual_hull_completion_tsdf,
                &visual_hull_completion_support,
                observed_conflict_veto.empty()
                    ? nullptr
                    : &observed_conflict_veto);
        result.statistics.visualHullCompletionOccupiedSampleCount =
            completion.occupiedSampleCount;
        result.statistics.visualHullCompletionBoundarySampleCount =
            completion.boundarySampleCount;
        result.statistics.visualHullCompletionAnchorCellCount =
            completion.anchorCellCount;
        result.statistics.visualHullCompletionFrontierCellCount =
            completion.frontierCellCount;
        result.statistics.visualHullCompletionRecoveredSampleCount =
            completion.recoveredSampleCount;
        if (completion.recoveredSampleCount > 0 &&
            options.visualHullCompletionRelaxationIterations > 0)
        {
            std::vector<std::uint8_t> completion_mask(
                visual_hull_completion_support.size(), 0);
            for (std::size_t index = 0;
                 index < completion_mask.size();
                 ++index)
            {
                completion_mask[index] =
                    supported_before_hull_completion[index] == 0 &&
                    visual_hull_completion_support[index] != 0
                    ? 1
                    : 0;
            }
            result.statistics.visualHullCompletionRelaxedSampleCount =
                relaxVisualHullCompletionField(
                    result.layout,
                    occupied,
                    completion_mask,
                    visual_hull_completion_support,
                    options.visualHullCompletionRelaxationIterations,
                    options.visualHullCompletionRelaxationLambda,
                    options.visualHullCompletionMaximumUpdate,
                    &visual_hull_completion_tsdf);
        }
    }
    result.statistics.effectiveVisibilityOccupancyCompletion =
        options.enableVisibilityOccupancyCompletion;
    result.statistics.effectiveVisibilityOccupancyResolution =
        options.enableVisibilityOccupancyCompletion
        ? std::clamp(options.visibilityOccupancyResolution, 24, 128)
        : 0;
    std::array<float, 3> native_carrier_bounds_min{};
    std::array<float, 3> native_carrier_bounds_max{};
    std::array<int, 3> native_carrier_dimensions{};
    std::array<int, 3> native_carrier_cells{};
    std::vector<std::uint8_t> native_carrier_occupied;
    std::vector<float> native_carrier_field;
    if (options.enableVisibilityOccupancyCompletion)
    {
        if (options.progress)
        {
            options.progress(
                QStringLiteral("正在求解可见性约束的全局空实占据场..."),
                74);
        }
        std::vector<VisibilityOccupancyFrameView> occupancy_frames;
        occupancy_frames.reserve(static_cast<std::size_t>(frames.size()));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (effective_frame_quality_weights[frame_index] <= 0.0f)
            {
                continue;
            }
            const DepthTsdfFrame &frame = frames[frame_index];
            VisibilityOccupancyFrameView view;
            view.camera = &frame.camera;
            view.depth = &frame.depth;
            view.confidence = &frame.confidence;
            view.depthValidMask =
                &effective_depth_valid_masks[frame_index];
            view.supportMask = &frame.supportMask;
            view.frameWeight =
                effective_frame_quality_weights[frame_index];
            occupancy_frames.push_back(view);
        }

        VisibilityOccupancyOptions occupancy_options;
        occupancy_options.resolution =
            result.statistics.effectiveVisibilityOccupancyResolution;
        if (options.visibilityOccupancyAlignCarrierGrid &&
            options.resolution >= occupancy_options.resolution &&
            options.resolution % occupancy_options.resolution == 0)
        {
            const int grid_scale =
                options.resolution / occupancy_options.resolution;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (grid_scale > 1 &&
                    result.layout.cells[axis] % grid_scale == 0)
                {
                    occupancy_options.sampleDimensions[axis] =
                        result.layout.cells[axis] / grid_scale + 1;
                }
            }
        }
        occupancy_options.minimumVisibleViews = std::clamp(
            options.visibilityOccupancyMinimumVisibleViews, 1, 16);
        occupancy_options.minimumSilhouetteViews = std::clamp(
            options.visibilityOccupancyMinimumSilhouetteViews, 1, 16);
        occupancy_options.minimumDepthFullViewsForSilhouettePrior = std::clamp(
            options.visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior,
            0,
            16);
        occupancy_options.allowedSilhouetteViolations = std::clamp(
            options.visibilityOccupancyAllowedSilhouetteViolations, 0, 8);
        occupancy_options.frontTolerancePixelFootprints = std::clamp(
            options.visibilityOccupancyFrontTolerancePixelFootprints,
            0.5f,
            12.0f);
        occupancy_options.behindSurfaceBandPixelFootprints = std::clamp(
            options.visibilityOccupancyBehindSurfaceBandPixelFootprints,
            1.0f,
            16.0f);
        occupancy_options.depthEmptyCapacity = std::max(
            0, options.visibilityOccupancyDepthEmptyCapacity);
        occupancy_options.depthFullCapacity = std::max(
            0, options.visibilityOccupancyDepthFullCapacity);
        occupancy_options.silhouetteEmptyCapacity = std::max(
            0, options.visibilityOccupancySilhouetteEmptyCapacity);
        occupancy_options.silhouetteFullPriorCapacity = std::max(
            0, options.visibilityOccupancySilhouetteFullPriorCapacity);
        occupancy_options.pairwiseCapacity = std::max(
            0, options.visibilityOccupancyPairwiseCapacity);
        result.statistics.effectiveVisibilityOccupancyPairwiseCapacity =
            occupancy_options.pairwiseCapacity;
        occupancy_options.closingIterations = std::clamp(
            options.visibilityOccupancyClosingIterations, 0, 8);
        result.statistics.effectiveVisibilityOccupancyClosingIterations =
            occupancy_options.closingIterations;
        occupancy_options.maximumHandleRepairPasses = std::clamp(
            options.visibilityOccupancyMaximumHandleRepairPasses, 1, 16);
        occupancy_options.maximumHandleRepairAcceptedCandidateCount =
            std::clamp(
                options
                    .visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount,
                0,
                512);
        occupancy_options.maximumHandleRepairCandidateSampleCount =
            std::max<std::size_t>(
                1,
                options
                    .visibilityOccupancyMaximumHandleRepairCandidateSampleCount);
        occupancy_options.maximumHandleRepairSubsetSampleCount =
            std::clamp<std::size_t>(
                options
                    .visibilityOccupancyMaximumHandleRepairSubsetSampleCount,
                1,
                1024);
        occupancy_options.maximumHandleRepairSubsetSeedCount = std::clamp(
            options.visibilityOccupancyMaximumHandleRepairSubsetSeedCount,
            0,
            4096);
        occupancy_options.repairNonManifoldConfigurations =
            options.visibilityOccupancyCellBoundaryExtraction;
        occupancy_options.closingMinimumDepthEmptyViewsToProtect = std::clamp(
            options
                .visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect,
            1,
            16);
        occupancy_options.closingMinimumSilhouetteOutsideViewsToProtect =
            std::clamp(
                options
                    .visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect,
                1,
                16);
        occupancy_options.buildSignedDistanceSamples = false;
        occupancy_options.isCancelled = options.isCancelled;
        const float minimum_full_fraction = std::clamp(
            options.visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction,
            0.0f,
            0.25f);
        VisibilityOccupancyResult occupancy;
        int depth_support_fallback_count = 0;
        while (true)
        {
            occupancy = VisibilityOccupancySurfaceBuilder::build(
                result.layout.boundsMin,
                result.layout.boundsMax,
                occupancy_frames,
                occupancy_options);
            const bool empty_cut =
                occupancy.error ==
                "visibility occupancy cut contains no full samples";
            const bool accepted_cut = shouldApplyVisibilityOccupancyCut(
                occupancy.ok,
                empty_cut,
                occupancy.statistics.sampleCount,
                occupancy.statistics.fullSampleCountAfterCleanup,
                minimum_full_fraction);
            if (occupancy.cancelled || accepted_cut ||
                occupancy_options
                        .minimumDepthFullViewsForSilhouettePrior <= 0)
            {
                break;
            }
            const int current_threshold = occupancy_options
                .minimumDepthFullViewsForSilhouettePrior;
            occupancy_options.minimumDepthFullViewsForSilhouettePrior =
                current_threshold > 1 ? 1 : 0;
            ++depth_support_fallback_count;
        }
        const bool rejected_empty_cut =
            occupancy.error ==
            "visibility occupancy cut contains no full samples";
        const bool accepted_cut = shouldApplyVisibilityOccupancyCut(
            occupancy.ok,
            rejected_empty_cut,
            occupancy.statistics.sampleCount,
            occupancy.statistics.fullSampleCountAfterCleanup,
            minimum_full_fraction);
        const bool rejected_collapsed_cut = occupancy.ok &&
            !rejected_empty_cut && !accepted_cut;
        result.statistics.visibilityOccupancyRejectedEmptyCut =
            rejected_empty_cut;
        result.statistics.visibilityOccupancyRejectedCollapsedCut =
            rejected_collapsed_cut;
        result.statistics.effectiveVisibilityOccupancyCompletion =
            !rejected_empty_cut && !rejected_collapsed_cut;
        if (occupancy.cancelled)
        {
            result.errorMessage = QStringLiteral(
                "Visibility occupancy solve was cancelled");
            return result;
        }
        if (!occupancy.ok && !rejected_empty_cut)
        {
            result.errorMessage = occupancy.cancelled
                ? QStringLiteral("可见性占据场求解已取消")
                : QStringLiteral("可见性占据场求解失败: %1")
                      .arg(QString::fromStdString(occupancy.error));
            return result;
        }
        VisibilityOccupancyDistanceFieldResult distance_field;
        if (result.statistics.effectiveVisibilityOccupancyCompletion)
        {
            distance_field = VisibilityOccupancyDistanceField::build(
                occupancy.sampleDimensions,
                occupancy.boundsMin,
                occupancy.boundsMax,
                occupancy.occupied);
        }
        if (result.statistics.effectiveVisibilityOccupancyCompletion &&
            !distance_field.ok)
        {
            result.errorMessage = QStringLiteral(
                "可见性占据场距离场构建失败: %1")
                                      .arg(QString::fromStdString(
                                          distance_field.error));
            return result;
        }
        if (result.statistics.effectiveVisibilityOccupancyCompletion)
        {
            occupancy.signedDistanceSamples =
                std::move(distance_field.signedWorldDistance);
            occupancy.signedDistanceSamplesAreWorldUnits = true;
        }
        if (result.statistics.effectiveVisibilityOccupancyCompletion &&
            (options.visibilityOccupancyNativeCarrierExtraction ||
             options.visibilityOccupancyCellBoundaryExtraction))
        {
            native_carrier_bounds_min = occupancy.boundsMin;
            native_carrier_bounds_max = occupancy.boundsMax;
            native_carrier_dimensions = occupancy.sampleDimensions;
            for (int axis = 0; axis < 3; ++axis)
            {
                native_carrier_cells[axis] =
                    occupancy.sampleDimensions[axis] - 1;
            }
            native_carrier_occupied = occupancy.occupied;
            native_carrier_field =
                std::move(occupancy.signedDistanceSamples);
        }
        result.statistics.visibilityOccupancySampleCount =
            occupancy.statistics.sampleCount;
        result.statistics
            .effectiveVisibilityOccupancyMinimumDepthFullViewsForSilhouettePrior =
            occupancy_options.minimumDepthFullViewsForSilhouettePrior;
        result.statistics.visibilityOccupancyDepthSupportFallbackCount =
            depth_support_fallback_count;
        result.statistics.visibilityOccupancyDepthEmptyVoteCount =
            occupancy.statistics.depthEmptyVoteCount;
        result.statistics.visibilityOccupancyDepthFullVoteCount =
            occupancy.statistics.depthFullVoteCount;
        result.statistics
            .visibilityOccupancySilhouetteFullPriorCandidateSampleCount =
            occupancy.statistics.silhouetteFullPriorCandidateSampleCount;
        result.statistics.visibilityOccupancySilhouetteFullPriorSampleCount =
            occupancy.statistics.silhouetteFullPriorSampleCount;
        result.statistics
            .visibilityOccupancySilhouetteFullPriorRejectedWithoutDepthSupportSampleCount =
            occupancy.statistics
                .silhouetteFullPriorRejectedWithoutDepthSupportSampleCount;
        result.statistics.visibilityOccupancySilhouetteFullPriorCapacityTotal =
            occupancy.statistics.silhouetteFullPriorCapacityTotal;
        result.statistics.visibilityOccupancyFullSampleCount =
            occupancy.statistics.fullSampleCountAfterCleanup;
        result.statistics.visibilityOccupancyFilledBubbleSampleCount =
            occupancy.statistics.filledInteriorEmptySampleCount;
        result.statistics.visibilityOccupancyRemovedDustSampleCount =
            occupancy.statistics.removedFullDustSampleCount;
        result.statistics.visibilityOccupancyClosingChangedSampleCount =
            occupancy.statistics.closingChangedSampleCount;
        result.statistics.visibilityOccupancyClosingProposalAddedSampleCount =
            occupancy.statistics.closingProposalAddedSampleCount;
        result.statistics
            .visibilityOccupancyClosingProposalDepthEmptyAtLeastTwoSampleCount =
            occupancy.statistics
                .closingProposalDepthEmptyAtLeastTwoSampleCount;
        result.statistics
            .visibilityOccupancyClosingProposalDepthEmptyAtLeastThreeSampleCount =
            occupancy.statistics
                .closingProposalDepthEmptyAtLeastThreeSampleCount;
        result.statistics
            .visibilityOccupancyClosingProposalDepthEmptyAtLeastFourSampleCount =
            occupancy.statistics
                .closingProposalDepthEmptyAtLeastFourSampleCount;
        result.statistics
            .visibilityOccupancyClosingProposalDepthFullSampleCount =
            occupancy.statistics.closingProposalDepthFullSampleCount;
        result.statistics
            .visibilityOccupancyClosingProposalSilhouetteOutsideAtLeastTwoSampleCount =
            occupancy.statistics
                .closingProposalSilhouetteOutsideAtLeastTwoSampleCount;
        result.statistics.visibilityOccupancyClosingProtectedEmptySampleCount =
            occupancy.statistics.closingProtectedEmptySampleCount;
        result.statistics
            .visibilityOccupancyClosingDepthEmptyProtectedSampleCount =
            occupancy.statistics.closingDepthEmptyProtectedSampleCount;
        result.statistics
            .visibilityOccupancyClosingSilhouetteEmptyProtectedSampleCount =
            occupancy.statistics.closingSilhouetteEmptyProtectedSampleCount;
        result.statistics
            .visibilityOccupancyHandleRepairCandidateComponentCount =
            occupancy.statistics.handleRepairCandidateComponentCount;
        result.statistics
            .visibilityOccupancyHandleRepairAcceptedCandidateCount =
            occupancy.statistics.handleRepairAcceptedCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairAcceptedSubsetCandidateCount =
            occupancy.statistics.handleRepairAcceptedSubsetCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairAcceptedPlateauSubsetCandidateCount =
            occupancy.statistics
                .handleRepairAcceptedPlateauSubsetCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairAttemptedSubsetSeedCount =
            occupancy.statistics.handleRepairAttemptedSubsetSeedCount;
        result.statistics
            .visibilityOccupancyHandleRepairRejectedProtectedCandidateCount =
            occupancy.statistics.handleRepairRejectedProtectedCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairRejectedOversizedCandidateCount =
            occupancy.statistics.handleRepairRejectedOversizedCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairRejectedTopologyCandidateCount =
            occupancy.statistics.handleRepairRejectedTopologyCandidateCount;
        result.statistics
            .visibilityOccupancyHandleRepairRejectedProtectedReachabilityCandidateCount =
            occupancy.statistics
                .handleRepairRejectedProtectedReachabilityCandidateCount;
        result.statistics.visibilityOccupancyHandleRepairBodyEulerBefore =
            occupancy.statistics.handleRepairBodyEulerBefore;
        result.statistics.visibilityOccupancyHandleRepairBodyEulerAfter =
            occupancy.statistics.handleRepairBodyEulerAfter;
        result.statistics
            .visibilityOccupancyWellComposedRepairFilledSampleCount =
            occupancy.statistics.wellComposedRepairFilledSampleCount;
        result.statistics
            .visibilityOccupancyWellComposedRepairAcceptedPassCount =
            occupancy.statistics.wellComposedRepairAcceptedPassCount;
        result.statistics
            .visibilityOccupancyWellComposedRepairBodyEulerBefore =
            occupancy.statistics.wellComposedRepairBodyEulerBefore;
        result.statistics
            .visibilityOccupancyWellComposedRepairBodyEulerAfter =
            occupancy.statistics.wellComposedRepairBodyEulerAfter;
        result.statistics
            .visibilityOccupancyWellComposedRepairRemainingEdgeCheckerboardCount =
            occupancy.statistics
                .wellComposedRepairRemainingEdgeCheckerboardCount;
        result.statistics
            .visibilityOccupancyWellComposedRepairRemainingVertexOccupiedDefectCount =
            occupancy.statistics
                .wellComposedRepairRemainingVertexOccupiedComponentDefectCount;
        result.statistics
            .visibilityOccupancyWellComposedRepairRemainingVertexEmptyDefectCount =
            occupancy.statistics
                .wellComposedRepairRemainingVertexEmptyComponentDefectCount;
        result.statistics.visibilityOccupancyCutEnergy =
            occupancy.statistics.minCut.cutEnergy;

        std::vector<float> *completion_tsdf =
            visual_hull_completion_tsdf.empty()
            ? &tsdf
            : &visual_hull_completion_tsdf;
        std::vector<std::uint8_t> *completion_support =
            visual_hull_completion_support.empty()
            ? &supported
            : &visual_hull_completion_support;
        VisibilityOccupancyTsdfCompletionOptions completion_options;
        completion_options.enableTopologyLockedResidualBlend =
            options.visibilityOccupancyTopologyLockedResidualBlend;
        completion_options.truncationDistanceWorld = truncation;
        completion_options.observedBand = std::clamp(
            options.visibilityOccupancyObservedBand, 0.01f, 1.0f);
        completion_options.carrierBand = std::clamp(
            options.visibilityOccupancyCarrierBand, 0.01f, 1.0f);
        completion_options.maximumResidual = std::clamp(
            options.visibilityOccupancyMaximumResidual, 0.0f, 1.0f);
        completion_options.detailBlend = std::clamp(
            options.visibilityOccupancyDetailBlend, 0.0f, 1.0f);
        completion_options.preserveAllObservedSamples =
            options.visibilityOccupancyPreserveAllObservedSamples;
        completion_options.preserveObservedNearSurface =
            options.visibilityOccupancyPreserveObservedNearSurface;
        completion_options.requireOccupancySignAgreement =
            options.visibilityOccupancyRequireSignAgreement;
        completion_options.maximumPreservedAbsoluteTsdf = std::clamp(
            options.visibilityOccupancyMaximumPreservedAbsoluteTsdf,
            0.0f,
            1.0f);
        completion_options.signedDistanceNormalizationSamples = std::clamp(
            options.visibilityOccupancySignedDistanceNormalizationSamples,
            0.5f,
            16.0f);
        VisibilityOccupancyTsdfCompletionStatistics completion;
        if (result.statistics.effectiveVisibilityOccupancyCompletion)
        {
            completion = VisibilityOccupancyTsdfCompletion::apply(
                result.layout,
                occupancy,
                completion_options,
                completion_tsdf,
                completion_support);
        }
        result.statistics.visibilityOccupancyRecoveredUnsupportedSampleCount =
            completion.recoveredUnsupportedSampleCount;
        result.statistics.visibilityOccupancyPreservedObservedSampleCount =
            completion.preservedObservedSampleCount;
        result.statistics.visibilityOccupancyOverriddenObservedSampleCount =
            completion.overriddenObservedSampleCount;
        result.statistics.visibilityOccupancyForcedBoundarySampleCount =
            completion.forcedExteriorBoundarySampleCount;
        result.statistics
            .visibilityOccupancyAdjustedExactIsoValueSampleCount =
            completion.adjustedExactIsoValueSampleCount;
        result.statistics.visibilityOccupancyTrustedObservationSampleCount =
            completion.trustedObservationSampleCount;
        result.statistics
            .visibilityOccupancyIgnoredSignConflictObservationCount =
            completion.ignoredSignConflictObservationCount;
        result.statistics.visibilityOccupancyBlendedSampleCount =
            completion.blendedSampleCount;
        result.statistics.visibilityOccupancyClippedResidualSampleCount =
            completion.clippedResidualSampleCount;
        result.statistics.visibilityOccupancyCarrierSignMismatchSampleCount =
            completion.carrierSignMismatchSampleCount;
        result.statistics.visibilityOccupancyMaximumAppliedResidual =
            completion.maximumAppliedResidual;
    }
    if (result.statistics.supportedSampleCount == 0)
    {
        result.errorMessage = QStringLiteral("TSDF integration produced no multi-camera supported samples");
        return result;
    }

    result.statistics.effectiveZeroCrossingDiagnostics =
        options.collectZeroCrossingDiagnostics;
    result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction =
        result.statistics.effectiveVisibilityOccupancyCompletion &&
        options.visibilityOccupancyCellBoundaryExtraction;
    result.statistics.effectiveConsistentIsoSurfaceExtraction =
        options.enableConsistentIsoSurfaceExtraction &&
        !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction;
    result.statistics.effectiveMc33IsoSurfaceExtraction =
        options.enableMc33IsoSurfaceExtraction &&
        !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction;
    result.statistics.effectiveMc33RequireSupportedSignChange =
        result.statistics.effectiveMc33IsoSurfaceExtraction &&
        options.mc33RequireSupportedSignChange;
    if (options.enableConsistentIsoSurfaceExtraction &&
        options.enableMc33IsoSurfaceExtraction &&
        !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction)
    {
        result.errorMessage = QStringLiteral(
            "TSDF consistent and MC33 iso-surface extractors cannot both be enabled");
        return result;
    }
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
    const bool use_native_carrier =
        options.visibilityOccupancyNativeCarrierExtraction &&
        !native_carrier_field.empty() &&
        (options.enableMc33IsoSurfaceExtraction ||
         options.enableConsistentIsoSurfaceExtraction);
    const std::array<float, 3> &extraction_bounds_min =
        use_native_carrier
        ? native_carrier_bounds_min
        : result.layout.boundsMin;
    const std::array<float, 3> &extraction_bounds_max =
        use_native_carrier
        ? native_carrier_bounds_max
        : result.layout.boundsMax;
    const std::array<int, 3> &extraction_cells =
        use_native_carrier
        ? native_carrier_cells
        : result.layout.cells;
    const std::vector<float> &extraction_tsdf = use_native_carrier
        ? native_carrier_field
        : (!visual_hull_completion_tsdf.empty()
        ? visual_hull_completion_tsdf
        : tsdf);
    const std::vector<std::uint8_t> no_native_support;
    const std::vector<std::uint8_t> &extraction_support = use_native_carrier
        ? no_native_support
        : (!visual_hull_completion_support.empty()
        ? visual_hull_completion_support
        : (result.statistics.effectiveVisibilityOccupancyCompletion
               ? supported
               : (!adaptiveTgvExtractionSupport.empty()
               ? adaptiveTgvExtractionSupport
               : supported)));
    try
    {
        if (result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction)
        {
            VisibilityOccupancyBoundaryOptions extraction_options;
            extraction_options.isCancelled = options.isCancelled;
            VisibilityOccupancyBoundaryResult extraction =
                VisibilityOccupancyBoundaryExtractor::extract(
                    native_carrier_bounds_min,
                    native_carrier_bounds_max,
                    native_carrier_dimensions,
                    native_carrier_occupied,
                    extraction_options);
            if (!extraction.ok)
            {
                result.errorMessage = QStringLiteral(
                    "可见性占据体边界提取失败: %1")
                    .arg(QString::fromStdString(extraction.errorMessage));
                return result;
            }
            result.mesh = std::move(extraction.mesh);
            result.statistics.visibilityOccupancyBoundaryOccupiedCellCount =
                extraction.statistics.occupiedCellCount;
            result.statistics.visibilityOccupancyBoundaryExposedQuadCount =
                extraction.statistics.exposedQuadCount;
            result.statistics
                .visibilityOccupancyBoundaryNonManifoldEdgeCount =
                extraction.statistics.nonManifoldEdgeCount;
            result.statistics
                .visibilityOccupancyBoundaryNonManifoldVertexCount =
                extraction.statistics.nonManifoldVertexCount;
            const int occupancy_body_euler =
                VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
                    native_carrier_dimensions,
                    native_carrier_occupied);
            result.statistics
                .visibilityOccupancyBoundaryBodyEulerCharacteristic =
                occupancy_body_euler;
            result.statistics
                .visibilityOccupancyBoundarySurfaceEulerCharacteristic =
                extraction.statistics.eulerCharacteristic;
            const bool topology_consistent =
                extraction.statistics.closedTwoManifold &&
                extraction.statistics.eulerCharacteristic ==
                    2 * occupancy_body_euler;
            result.statistics
                .visibilityOccupancyBoundaryTopologyConsistent =
                topology_consistent;
            if (!topology_consistent)
            {
                result.errorMessage = QStringLiteral(
                    "可见性占据体不是良构闭合流形: 边界边=%1, "
                    "非流形边=%2, 非流形顶点=%3, 体欧拉特征=%4, "
                    "曲面欧拉特征=%5（期望 %6）")
                    .arg(extraction.statistics.boundaryEdgeCount)
                    .arg(extraction.statistics.nonManifoldEdgeCount)
                    .arg(extraction.statistics.nonManifoldVertexCount)
                    .arg(occupancy_body_euler)
                    .arg(extraction.statistics.eulerCharacteristic)
                    .arg(2 * occupancy_body_euler);
                return result;
            }
        }
        else if (options.enableMc33IsoSurfaceExtraction)
        {
            Mc33IsoSurfaceOptions extraction_options;
            extraction_options.isoLevel = 0.0f;
            extraction_options.requireSupportedSignChange =
                options.mc33RequireSupportedSignChange;
            extraction_options.isCancelled = options.isCancelled;
            Mc33IsoSurfaceResult extraction =
                Mc33IsoSurfaceExtractor::extract(
                    extraction_bounds_min,
                    extraction_bounds_max,
                    extraction_cells,
                    extraction_tsdf,
                    extraction_support,
                    extraction_options);
            if (!extraction.ok)
            {
                result.errorMessage =
                    QStringLiteral("TSDF MC33 iso-surface extraction failed: %1")
                        .arg(QString::fromStdString(extraction.errorMessage));
                return result;
            }
            result.mesh = std::move(extraction.mesh);
            result.statistics.mc33SupportMaskedSampleCount =
                extraction.statistics.supportMaskedSampleCount;
            result.statistics.mc33RejectedUnsupportedCellFaceCount =
                extraction.statistics.rejectedUnsupportedCellFaceCount;
        }
        else if (options.enableConsistentIsoSurfaceExtraction)
        {
            ConsistentIsoSurfaceOptions extraction_options;
            extraction_options.isoLevel = 0.0f;
            extraction_options.isCancelled = options.isCancelled;
            ConsistentIsoSurfaceResult extraction =
                ConsistentIsoSurfaceExtractor::extract(
                    extraction_bounds_min,
                    extraction_bounds_max,
                    extraction_cells,
                    extraction_tsdf,
                    extraction_support,
                    extraction_options);
            if (!extraction.ok)
            {
                result.errorMessage =
                    QStringLiteral("TSDF consistent iso-surface extraction failed: %1")
                        .arg(QString::fromStdString(extraction.errorMessage));
                return result;
            }
            result.mesh = std::move(extraction.mesh);
            result.statistics.isoSurfaceAmbiguousFaceCount =
                extraction.statistics.uniqueAmbiguousFaceCount;
            result.statistics.isoSurfaceTopologyAdjustedCellCount =
                extraction.statistics.topologyAdjustedCellCount;
            result.statistics.isoSurfaceDeciderTieCount =
                extraction.statistics.deciderTieCount;
            result.statistics.isoSurfaceMultipleLoopCellCount =
                extraction.statistics.multipleLoopCellCount;
            result.statistics.isoSurfaceEdgeVertexCacheHitCount =
                extraction.statistics.edgeVertexCacheHitCount;
            result.statistics.isoSurfaceEdgeVertexCacheMissCount =
                extraction.statistics.edgeVertexCacheMissCount;
            result.statistics.isoSurfaceInteriorLoopVertexCount =
                extraction.statistics.interiorLoopVertexCount;
            result.statistics.isoSurfaceRejectedDegenerateFaceCount =
                extraction.statistics.rejectedDegenerateFaceCount;
            result.statistics.isoSurfaceUnresolvedCellCount =
                extraction.statistics.unresolvedCellCount;
        }
        else
        {
            plapoint::mesh::MarchingCubes<float> marchingCubes;
            marchingCubes.setBounds(
                {result.layout.boundsMin[0],
                 result.layout.boundsMin[1],
                 result.layout.boundsMin[2]},
                {result.layout.boundsMax[0],
                 result.layout.boundsMax[1],
                 result.layout.boundsMax[2]});
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
                    const std::size_t index =
                        sampleIndex(result.layout, ix, iy, iz);
                    return extraction_support[index] != 0
                        ? extraction_tsdf[index]
                        : 1.0f;
                });
            result.mesh.vertices.resize(
                static_cast<std::size_t>(vertices.rows()));
            for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
            {
                MeshVertex &vertex =
                    result.mesh.vertices[static_cast<std::size_t>(row)];
                vertex.x = vertices(row, 0);
                vertex.y = vertices(row, 1);
                vertex.z = vertices(row, 2);
            }
            result.mesh.faces.resize(static_cast<std::size_t>(faces.rows()));
            for (plamatrix::Index row = 0; row < faces.rows(); ++row)
            {
                Triangle &face =
                    result.mesh.faces[static_cast<std::size_t>(row)];
                face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
                face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
                face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
            }
        }
        if (result.statistics.effectiveVisualHullCompletionTopologyGuard &&
            result.statistics.visualHullCompletionRecoveredSampleCount > 0)
        {
            const std::vector<std::uint8_t> &baseline_support =
                !adaptiveTgvExtractionSupport.empty()
                ? adaptiveTgvExtractionSupport
                : supported;
            ComparableIsoSurfaceExtraction baseline_extraction =
                extractComparableIsoSurface(
                    result.layout.boundsMin,
                    result.layout.boundsMax,
                    result.layout.cells,
                    tsdf,
                    baseline_support,
                    options.enableMc33IsoSurfaceExtraction,
                    options.mc33RequireSupportedSignChange,
                    options.enableConsistentIsoSurfaceExtraction,
                    options.isCancelled);
            if (!baseline_extraction.ok || baseline_extraction.mesh.empty())
            {
                result.errorMessage = QStringLiteral(
                    "TSDF visual-hull topology guard could not extract the "
                    "uncompleted baseline: %1")
                    .arg(baseline_extraction.errorMessage);
                return result;
            }

            const DepthTsdfVisualHullTopologyGuardEvaluation evaluation =
                evaluateVisualHullCompletionTopologyGuard(
                    baseline_extraction.mesh,
                    result.mesh,
                    options
                        .visualHullCompletionMaximumTopologicalComplexityIncrease,
                    options.visualHullCompletionMaximumSurfaceAreaRatio,
                    options.visualHullCompletionMaximumBoundsDiagonalRatio);
            result.statistics.visualHullCompletionTopologyGuardEvaluated = true;
            result.statistics.visualHullCompletionTopologyGuardAccepted =
                evaluation.accepted;
            result.statistics.visualHullCompletionTopologyGuardRejectionFlags =
                evaluation.rejectionFlags;
            result.statistics.visualHullCompletionTopologyGuardBaselineFaceCount =
                evaluation.baselineFaceCount;
            result.statistics.visualHullCompletionTopologyGuardCandidateFaceCount =
                evaluation.candidateFaceCount;
            result.statistics
                .visualHullCompletionTopologyGuardBaselineBoundaryEdgeCount =
                evaluation.baselineBoundaryEdgeCount;
            result.statistics
                .visualHullCompletionTopologyGuardCandidateBoundaryEdgeCount =
                evaluation.candidateBoundaryEdgeCount;
            result.statistics
                .visualHullCompletionTopologyGuardBaselineNonManifoldEdgeCount =
                evaluation.baselineNonManifoldEdgeCount;
            result.statistics
                .visualHullCompletionTopologyGuardCandidateNonManifoldEdgeCount =
                evaluation.candidateNonManifoldEdgeCount;
            result.statistics.visualHullCompletionTopologyGuardBaselineComponentCount =
                evaluation.baselineComponentCount;
            result.statistics.visualHullCompletionTopologyGuardCandidateComponentCount =
                evaluation.candidateComponentCount;
            result.statistics
                .visualHullCompletionTopologyGuardBaselineEulerCharacteristic =
                evaluation.baselineEulerCharacteristic;
            result.statistics
                .visualHullCompletionTopologyGuardCandidateEulerCharacteristic =
                evaluation.candidateEulerCharacteristic;
            result.statistics
                .visualHullCompletionTopologyGuardBaselineTopologicalComplexity =
                evaluation.baselineTopologicalComplexity;
            result.statistics
                .visualHullCompletionTopologyGuardCandidateTopologicalComplexity =
                evaluation.candidateTopologicalComplexity;
            result.statistics.visualHullCompletionTopologyGuardSurfaceAreaRatio =
                evaluation.surfaceAreaRatio;
            result.statistics.visualHullCompletionTopologyGuardBoundsDiagonalRatio =
                evaluation.boundsDiagonalRatio;

            if (!evaluation.accepted)
            {
                result.mesh = std::move(baseline_extraction.mesh);
                result.statistics.mc33SupportMaskedSampleCount =
                    baseline_extraction.mc33SupportMaskedSampleCount;
                result.statistics.mc33RejectedUnsupportedCellFaceCount =
                    baseline_extraction.mc33RejectedUnsupportedCellFaceCount;
                result.statistics.isoSurfaceAmbiguousFaceCount =
                    baseline_extraction.isoSurfaceAmbiguousFaceCount;
                result.statistics.isoSurfaceTopologyAdjustedCellCount =
                    baseline_extraction.isoSurfaceTopologyAdjustedCellCount;
                result.statistics.isoSurfaceDeciderTieCount =
                    baseline_extraction.isoSurfaceDeciderTieCount;
                result.statistics.isoSurfaceMultipleLoopCellCount =
                    baseline_extraction.isoSurfaceMultipleLoopCellCount;
                result.statistics.isoSurfaceEdgeVertexCacheHitCount =
                    baseline_extraction.isoSurfaceEdgeVertexCacheHitCount;
                result.statistics.isoSurfaceEdgeVertexCacheMissCount =
                    baseline_extraction.isoSurfaceEdgeVertexCacheMissCount;
                result.statistics.isoSurfaceInteriorLoopVertexCount =
                    baseline_extraction.isoSurfaceInteriorLoopVertexCount;
                result.statistics.isoSurfaceRejectedDegenerateFaceCount =
                    baseline_extraction.isoSurfaceRejectedDegenerateFaceCount;
                result.statistics.isoSurfaceUnresolvedCellCount =
                    baseline_extraction.isoSurfaceUnresolvedCellCount;
            }
        }
        result.statistics.marchingCubesVertexCount = result.mesh.vertexCount();
        result.statistics.marchingCubesFaceCount = result.mesh.faceCount();
        result.statistics.marchingCubesBoundaryEdgeCount =
            boundaryEdgeCount(result.mesh);
        const MeshTopologyQualityStatistics extraction_topology =
            evaluateMeshTopologyQuality(result.mesh);
        result.statistics.isoSurfaceExtractionNonManifoldEdgeCount =
            extraction_topology.nonManifoldEdgeCount;
        result.statistics.isoSurfaceExtractionComponentCount =
            extraction_topology.componentCount;
        result.statistics.isoSurfaceExtractionEulerCharacteristic =
            extraction_topology.eulerCharacteristic;
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
    reportProgress(
        QStringLiteral("TSDF 零等值面提取完成：%1 点、%2 面，正在清理网格...")
            .arg(result.mesh.vertexCount())
            .arg(result.mesh.faceCount()),
        82);
    if (postprocessCancelled())
    {
        return result;
    }

    const PostprocessClock::time_point cleanup_start = PostprocessClock::now();
    const float minimum_degenerate_face_area =
        options.visibilityOccupancyCellBoundaryExtraction ||
            options.enableConsistentIsoSurfaceExtraction ||
            options.enableMc33IsoSurfaceExtraction
        ? 0.0f
        : 5.0e-9f;
    const int faces_before_degenerate_cleanup = result.mesh.faceCount();
    reportProgress(QStringLiteral("正在移除退化三角面..."), 82);
    detail::removeDegenerateFaces(
        &result.mesh,
        minimum_degenerate_face_area);
    result.statistics.initialDegenerateRemovedFaceCount = std::max(
        0,
        faces_before_degenerate_cleanup - result.mesh.faceCount());
    if (!options.visibilityOccupancyCellBoundaryExtraction &&
        !options.enableConsistentIsoSurfaceExtraction &&
        !options.enableMc33IsoSurfaceExtraction)
    {
        detail::weldCoincidentVertices(&result.mesh, 1.0e-6f);
    }
    const int faces_before_component_filter = result.mesh.faceCount();
    reportProgress(
        QStringLiteral("正在过滤孤立网格组件（%1 面）...")
            .arg(faces_before_component_filter),
        83);
    detail::removeSmallConnectedComponents(
        &result.mesh,
        std::max(2, options.minimumComponentFaces),
        options.minimumComponentFaceRatio);
    result.statistics.componentFilterRemovedFaceCount = std::max(
        0,
        faces_before_component_filter - result.mesh.faceCount());
    result.statistics.componentFilteredFaceCount = result.mesh.faceCount();
    reportProgress(
        QStringLiteral("正在统计清理后边界（%1 面）...")
            .arg(result.mesh.faceCount()),
        83);
    result.statistics.componentFilteredBoundaryEdgeCount =
        boundaryEdgeCount(result.mesh);
    const WeakBoundaryTipResult weak_tips = trimWeakBoundaryTips(
        &result.mesh,
        result.layout,
        support,
        strongAdaptiveSurfaceObservation,
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
    result.statistics.weakBoundaryTrimmedBoundaryEdgeCount =
        boundaryEdgeCount(result.mesh);
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
        detail::removeDegenerateFaces(
            &result.mesh,
            minimum_degenerate_face_area);
    }
    result.statistics.topologyCleanedBoundaryEdgeCount =
        boundaryEdgeCount(result.mesh);
    if (options.collectZeroCrossingDiagnostics)
    {
        MeshBoundaryAttributionOptions attribution_options;
        attribution_options.minimumSourceCount =
            options.minimumSurfacePatchSourceCount;
        attribution_options.maximumInverseDepthSpread =
            options.maximumSurfacePatchInverseDepthSpread;
        attribution_options.minimumSurfaceWeightRatio =
            options.minimumSurfacePatchWeightRatio;
        attribution_options.maximumAbsoluteTsdf =
            options.maximumContourBandAbsoluteTsdf;
        const MeshBoundaryAttributionStatistics attribution =
            attributeMeshBoundaryEdges(
                result.mesh,
                result.layout,
                tsdf,
                weight,
                surfaceObservationWeight,
                geometrySourceMask,
                minimumInverseDepthSpread,
                supported,
                attribution_options);
        result.statistics.topologyCleanedAttributionEdgeCount =
            attribution.boundaryEdgeCount;
        result.statistics.topologyCleanedAttributionNoObservationEdgeCount =
            attribution.noObservationEdgeCount;
        result.statistics.topologyCleanedAttributionInsufficientSourceEdgeCount =
            attribution.insufficientSourceEdgeCount;
        result.statistics.topologyCleanedAttributionDepthSpreadRejectedEdgeCount =
            attribution.depthSpreadRejectedEdgeCount;
        result.statistics.topologyCleanedAttributionSurfaceWeightRejectedEdgeCount =
            attribution.surfaceWeightRejectedEdgeCount;
        result.statistics.topologyCleanedAttributionAbsoluteTsdfRejectedEdgeCount =
            attribution.absoluteTsdfRejectedEdgeCount;
        result.statistics.topologyCleanedAttributionSupportGateRejectedEdgeCount =
            attribution.supportGateRejectedEdgeCount;
        result.statistics
            .topologyCleanedAttributionExtractionOrPostprocessEdgeCount =
            attribution.extractionOrPostprocessEdgeCount;
        result.statistics.topologyCleanedAttributionUnclassifiedEdgeCount =
            attribution.unclassifiedEdgeCount;
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
        detail::removeDegenerateFaces(
            &result.mesh,
            minimum_degenerate_face_area);
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
    const auto repair_face_orientation_safely = [](TriMesh *mesh)
    {
        MeshFaceOrientationStatistics statistics;
        if (!mesh || mesh->faces.empty())
        {
            return std::make_pair(statistics, false);
        }
        TriMesh candidate = *mesh;
        statistics = repairMeshFaceOrientation(&candidate);
        const bool accepted =
            statistics.succeeded &&
            statistics.removedContradictoryFaceCount == 0 &&
            statistics.nonManifoldEdgeCount == 0 &&
            candidate.faceCount() == mesh->faceCount();
        if (accepted)
        {
            *mesh = std::move(candidate);
        }
        return std::make_pair(statistics, accepted);
    };
    const auto pre_denoising_orientation =
        repair_face_orientation_safely(&result.mesh);
    result.statistics.preDenoisingFaceOrientationRepairAccepted =
        pre_denoising_orientation.second;
    result.statistics
        .preDenoisingFaceOrientationInconsistentSharedEdgeCountBefore =
        pre_denoising_orientation.first.inconsistentSharedEdgeCountBefore;
    result.statistics.preDenoisingFaceOrientationFlippedFaceCount =
        pre_denoising_orientation.first.flippedFaceCount;
    result.statistics.effectiveSurfaceDenoisingIterations =
        options.surfaceDenoisingIterations;
    result.statistics.effectiveSurfaceDenoisingLambda =
        options.surfaceDenoisingLambda;
    result.statistics.effectiveSurfaceDenoisingMu =
        options.surfaceDenoisingMu;
    result.statistics.effectiveMaximumSurfaceDenoisingDisplacementVoxels =
        options.maximumSurfaceDenoisingDisplacementVoxels;
    result.statistics.effectiveMaximumSurfaceDenoisingNormalAngleDegrees =
        options.maximumSurfaceDenoisingNormalAngleDegrees;
    result.statistics.effectiveSurfaceDenoisingBoundaryProtectionRings =
        options.surfaceDenoisingBoundaryProtectionRings;
    result.statistics.effectiveProtectedTaubinSurfaceDenoising =
        options.enableProtectedTaubinSurfaceDenoising;
    result.statistics.effectivePostSimplificationSurfaceDenoising =
        options.enablePostSimplificationSurfaceDenoising &&
        options.surfaceDenoisingIterations > 0;
    // Protected Taubin denoising is applied only after simplification, where
    // its topology and normal-quality guard can reject an unsafe candidate.
    if (options.surfaceDenoisingIterations > 0 &&
        !options.enableProtectedTaubinSurfaceDenoising)
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
    detail::removeDegenerateFaces(
        &result.mesh,
        minimum_degenerate_face_area);
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
        detail::removeDegenerateFaces(
            &result.mesh,
            minimum_degenerate_face_area);
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
    if (options.enableOpenMeshSimplification &&
        options.simplifyTargetFaces > 0 &&
        result.mesh.faceCount() > options.simplifyTargetFaces)
    {
        reportProgress(
            QStringLiteral("正在进行 OpenMesh 拓扑约束简化..."), 87);
        result.statistics.openMeshSimplificationAttempted = true;
        result.statistics.openMeshSimplificationInputVertexCount =
            result.mesh.vertexCount();
        result.statistics.openMeshSimplificationInputFaceCount =
            result.mesh.faceCount();

        TriMesh candidate = result.mesh;
        const MeshBoundaryTopology topology_before =
            boundaryTopology(result.mesh);
        const MeshTopologyQualityStatistics quality_before =
            evaluateMeshTopologyQuality(result.mesh);
        result.statistics.openMeshBoundaryEdgeCountBefore =
            topology_before.boundaryEdgeCount;
        result.statistics.openMeshNonManifoldEdgeCountBefore =
            topology_before.nonManifoldEdgeCount;

        OpenMeshSimplifyOptions simplify_options;
        simplify_options.targetFaceCount = options.simplifyTargetFaces;
        simplify_options.maximumNormalDeviationDegrees =
            options.openMeshMaximumNormalDeviationDegrees;
        simplify_options.maximumNormalFlippingDegrees =
            options.openMeshMaximumNormalFlippingDegrees;
        simplify_options.smoothingIterations =
            options.openMeshSmoothingIterations;
        simplify_options.smoothingMaximumDisplacement =
            options.openMeshSmoothingMaximumDisplacementVoxels *
            std::max({
                result.layout.voxelSize[0],
                result.layout.voxelSize[1],
                result.layout.voxelSize[2]});
        simplify_options.smoothingFeatureAngleDegrees =
            options.openMeshSmoothingFeatureAngleDegrees;
        simplify_options.notificationInterval =
            options.openMeshNotificationInterval;
        simplify_options.isCancelled = options.isCancelled;
        simplify_options.progress = [&reportProgress](
            int collapsed_vertex_count,
            int estimated_face_count)
        {
            reportProgress(
                QStringLiteral(
                    "正在进行 OpenMesh 拓扑约束简化（已折叠 %1 点，约 %2 面）...")
                    .arg(collapsed_vertex_count)
                    .arg(estimated_face_count),
                88);
        };

        const OpenMeshSimplifyStatistics simplify_statistics =
            simplifyMeshWithOpenMesh(&candidate, simplify_options);
        const MeshBoundaryTopology topology_after =
            boundaryTopology(candidate);
        const MeshTopologyQualityStatistics quality_after =
            evaluateMeshTopologyQuality(candidate);
        result.statistics.openMeshSimplificationOutputVertexCount =
            candidate.vertexCount();
        result.statistics.openMeshSimplificationOutputFaceCount =
            candidate.faceCount();
        result.statistics.openMeshSimplificationCollapsedVertexCount =
            simplify_statistics.collapsedVertexCount;
        result.statistics.openMeshSimplificationRejectedInputFaceCount =
            simplify_statistics.rejectedInputFaceCount;
        result.statistics.openMeshInconsistentSharedEdgeCountBefore =
            simplify_statistics.inconsistentSharedEdgeCountBefore;
        result.statistics.openMeshReorientedInputFaceCount =
            simplify_statistics.reorientedInputFaceCount;
        result.statistics.openMeshRemovedContradictoryFaceCount =
            simplify_statistics.removedContradictoryFaceCount;
        result.statistics.openMeshOrientationConflictCount =
            simplify_statistics.orientationConflictCount;
        result.statistics.openMeshSmoothingApplied =
            simplify_statistics.smoothingApplied;
        result.statistics.openMeshSimplificationReachedTarget =
            simplify_statistics.reachedTarget;
        result.statistics.openMeshSimplificationCancelled =
            simplify_statistics.cancelled;
        result.statistics.openMeshSimplificationError =
            QString::fromStdString(simplify_statistics.error);
        result.statistics.openMeshBoundaryEdgeCountAfter =
            topology_after.boundaryEdgeCount;
        result.statistics.openMeshNonManifoldEdgeCountAfter =
            topology_after.nonManifoldEdgeCount;

        const auto topology_growth_is_safe = [](int before, int after)
        {
            return after <= before + std::max(
                8, static_cast<int>(std::ceil(before * 0.10f)));
        };
        const double maximum_accepted_normal_median = std::max(
            quality_before.adjacentNormalAngleMedianDegrees + 8.0,
            static_cast<double>(
                options.openMeshMaximumNormalDeviationDegrees));
        const double maximum_accepted_high_aspect_ratio = std::max(
            quality_before.highAspectFaceRatio * 1.5, 0.10);
        const bool accept_simplification =
            simplify_statistics.succeeded &&
            candidate.faceCount() < result.mesh.faceCount() &&
            meshVerticesInsidePaddedLayout(candidate, result.layout) &&
            topology_growth_is_safe(
                topology_before.boundaryEdgeCount,
                topology_after.boundaryEdgeCount) &&
            topology_growth_is_safe(
                topology_before.danglingBoundaryVertexCount,
                topology_after.danglingBoundaryVertexCount) &&
            topology_after.nonManifoldEdgeCount <=
                topology_before.nonManifoldEdgeCount &&
            quality_after.adjacentNormalAngleMedianDegrees <=
                maximum_accepted_normal_median &&
            quality_after.highAspectFaceRatio <=
                maximum_accepted_high_aspect_ratio;
        result.statistics.openMeshSimplificationAccepted =
            accept_simplification;
        result.statistics.effectiveOpenMeshSimplification =
            accept_simplification;
        if (accept_simplification)
        {
            result.mesh = std::move(candidate);
        }
        if (postprocessCancelled())
        {
            return result;
        }
    }
    if (!result.statistics.openMeshSimplificationAccepted &&
        options.enableQuadricSimplification &&
        options.simplifyTargetFaces > 0 &&
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
        simplify_options.minimumFaceArea = minimum_degenerate_face_area;
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
            meshVerticesInsidePaddedLayout(result.mesh, result.layout) &&
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
        !result.statistics.openMeshSimplificationAccepted &&
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
        detail::removeDegenerateFaces(
            &candidate,
            minimum_degenerate_face_area);
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
            detail::removeDegenerateFaces(
                &candidate,
                minimum_degenerate_face_area);
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
            polish_options.minimumFaceArea = minimum_degenerate_face_area;
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
                meshVerticesInsidePaddedLayout(candidate, result.layout) &&
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
            meshVerticesInsidePaddedLayout(candidate, result.layout) &&
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
    const int faces_before_post_simplification_component_filter =
        result.mesh.faceCount();
    detail::removeSmallConnectedComponents(
        &result.mesh,
        std::max(2, options.minimumComponentFaces),
        options.minimumComponentFaceRatio);
    result.statistics.postSimplificationComponentFilterRemovedFaceCount =
        std::max(
            0,
            faces_before_post_simplification_component_filter -
                result.mesh.faceCount());
    if (result.statistics.postSimplificationComponentFilterRemovedFaceCount > 0)
    {
        detail::compactReferencedVertices(&result.mesh);
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
            const std::size_t vertex_count_before = candidate.vertices.size();
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
            HoleFillPatchEvidenceResult patch_evidence;
            if (options.enableVisibilityConstrainedFinalHoleFill &&
                filled_hole_count > 0)
            {
                patch_evidence = validateAddedHoleFillPatchEvidence(
                    candidate,
                    vertex_count_before,
                    static_cast<std::size_t>(face_count_before),
                    frames,
                    effective_depth_valid_masks,
                    options.visibilityHoleFillMinimumSupportingViews,
                    options.visibilityHoleFillMaximumConflictViews,
                    options.visibilityHoleFillDepthToleranceVoxels,
                    maximum_voxel_size,
                    options.minimumConfidence);
                result.statistics.finalHoleFillPatchEvidenceValidationAttempted =
                    patch_evidence.attempted;
                result.statistics.finalHoleFillPatchEvidenceValidationAccepted =
                    patch_evidence.accepted;
                result.statistics.finalHoleFillPatchEvidenceVertexSampleCount =
                    patch_evidence.vertexSampleCount;
                result.statistics.finalHoleFillPatchEvidenceFaceCenterSampleCount =
                    patch_evidence.faceCenterSampleCount;
                result.statistics.finalHoleFillPatchEvidenceAcceptedSampleCount =
                    patch_evidence.acceptedSampleCount;
                result.statistics
                    .finalHoleFillPatchEvidenceRejectedSupportSampleCount =
                    patch_evidence.rejectedSupportSampleCount;
                result.statistics
                    .finalHoleFillPatchEvidenceRejectedConflictSampleCount =
                    patch_evidence.rejectedConflictSampleCount;
                result.statistics
                    .finalHoleFillPatchEvidenceMinimumSupportingViewCount =
                    patch_evidence.minimumSupportingViewCount;
                result.statistics
                    .finalHoleFillPatchEvidenceMaximumConflictViewCount =
                    patch_evidence.maximumConflictViewCount;
            }
            detail::removeDegenerateFaces(
                &candidate,
                minimum_degenerate_face_area);
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
            const bool patch_evidence_is_safe =
                !options.enableVisibilityConstrainedFinalHoleFill ||
                (patch_evidence.attempted && patch_evidence.accepted);
            const bool accept_final_fill =
                filled_hole_count > 0 &&
                patch_evidence_is_safe &&
                meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                topology_after.boundaryEdgeCount < topology_before.boundaryEdgeCount &&
                topology_after.danglingBoundaryVertexCount <=
                    topology_before.danglingBoundaryVertexCount &&
                topology_after.nonManifoldEdgeCount <=
                    topology_before.nonManifoldEdgeCount &&
                added_face_count <= maximum_added_faces &&
                triangle_quality_is_safe;

            if (accept_final_fill &&
                !result.statistics.openMeshSimplificationAccepted &&
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
                post_fill_options.minimumFaceArea =
                    minimum_degenerate_face_area;
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
                    meshVerticesInsidePaddedLayout(candidate, result.layout) &&
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
        detail::removeDegenerateFaces(
            &candidate,
            minimum_degenerate_face_area);
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
            meshVerticesInsidePaddedLayout(candidate, result.layout) &&
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
    reportProgress(QStringLiteral("正在统一最终网格面方向..."), 93);
    result.statistics.finalFaceOrientationRepairAttempted = true;
    const auto final_orientation =
        repair_face_orientation_safely(&result.mesh);
    result.statistics.finalFaceOrientationInconsistentSharedEdgeCountBefore =
        final_orientation.first.inconsistentSharedEdgeCountBefore;
    result.statistics.finalFaceOrientationInconsistentSharedEdgeCountAfter =
        final_orientation.first.inconsistentSharedEdgeCountAfter;
    result.statistics.finalFaceOrientationFlippedFaceCount =
        final_orientation.first.flippedFaceCount;
    result.statistics.finalFaceOrientationRemovedFaceCount =
        final_orientation.first.removedContradictoryFaceCount;
    result.statistics.finalFaceOrientationNonManifoldEdgeCount =
        final_orientation.first.nonManifoldEdgeCount;
    result.statistics.finalFaceOrientationRepairAccepted =
        final_orientation.second;
    if (postprocessCancelled())
    {
        return result;
    }
    if (result.statistics.effectivePostSimplificationSurfaceDenoising)
    {
        reportProgress(
            QStringLiteral("正在进行轮廓保护的最终表面降噪..."), 93);
        result.statistics.postSimplificationSurfaceDenoisingAttempted = true;
        MeshTopologyQualityThresholds quality_thresholds;
        quality_thresholds.maximumBoundaryEdgeRatio =
            std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
        quality_thresholds.maximumHighAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
        quality_thresholds.maximumExtremeAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
        const MeshTopologyQualityStatistics quality_before =
            evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
        result.statistics.postSimplificationNormalAngleMedianBefore =
            quality_before.adjacentNormalAngleMedianDegrees;
        result.statistics.postSimplificationNormalAngleP90Before =
            quality_before.adjacentNormalAngleP90Degrees;
        result.statistics.postSimplificationNormalAngleOver30RatioBefore =
            quality_before.adjacentNormalAngleOver30Ratio;
        result.statistics.postSimplificationHighAspectFaceRatioBefore =
            quality_before.highAspectFaceRatio;
        result.statistics.postSimplificationExtremeAspectFaceRatioBefore =
            quality_before.extremeAspectFaceRatio;

        TriMesh candidate = result.mesh;
        const float maximum_voxel_size = std::max({
            result.layout.voxelSize[0],
            result.layout.voxelSize[1],
            result.layout.voxelSize[2]});
        const int moved_vertex_count = options.enableProtectedTaubinSurfaceDenoising
            ? detail::smoothSurfaceVerticesTaubinProtected(
                  &candidate,
                  options.surfaceDenoisingIterations,
                  options.surfaceDenoisingLambda,
                  options.surfaceDenoisingMu,
                  options.maximumSurfaceDenoisingDisplacementVoxels *
                      maximum_voxel_size,
                  options.maximumSurfaceDenoisingNormalAngleDegrees,
                  options.surfaceDenoisingBoundaryProtectionRings)
            : detail::smoothSurfaceVerticesNormalAware(
                  &candidate,
                  options.surfaceDenoisingIterations,
                  options.surfaceDenoisingLambda,
                  options.maximumSurfaceDenoisingDisplacementVoxels *
                      maximum_voxel_size,
                  options.maximumSurfaceDenoisingNormalAngleDegrees,
                  options.surfaceDenoisingBoundaryProtectionRings);
        MeshTriangleOptimizationOptions relaxation_options;
        relaxation_options.maximumPasses = 0;
        relaxation_options.maximumFeatureAngleDegrees =
            options.triangleQualityMaximumFeatureAngleDegrees;
        relaxation_options.maximumNormalDeviationDegrees =
            options.triangleQualityMaximumNormalDeviationDegrees;
        relaxation_options.enableTangentialRelaxation = true;
        relaxation_options.tangentialRelaxationPasses = 1;
        relaxation_options.tangentialRelaxationLambda = std::min(
            0.30f, options.triangleQualityTangentialRelaxationLambda);
        relaxation_options.tangentialMaximumDisplacementEdgeRatio = std::min(
            0.10f,
            options.triangleQualityTangentialMaximumDisplacementEdgeRatio);
        relaxation_options.enableIsotropicRemeshing = false;
        relaxation_options.isCancelled = options.isCancelled;
        const MeshTriangleOptimizationStatistics relaxation =
            optimizeTriangleQuality(&candidate, relaxation_options);
        result.statistics.postSimplificationTangentialRelaxedVertexCount =
            relaxation.tangentialRelaxedVertexCount;
        detail::recomputeNormals(&candidate);
        const MeshTopologyQualityStatistics quality_after =
            evaluateMeshTopologyQuality(candidate, quality_thresholds);
        result.statistics.postSimplificationSmoothedSurfaceVertexCount =
            moved_vertex_count;
        result.statistics.postSimplificationNormalAngleMedianAfter =
            quality_after.adjacentNormalAngleMedianDegrees;
        result.statistics.postSimplificationNormalAngleP90After =
            quality_after.adjacentNormalAngleP90Degrees;
        result.statistics.postSimplificationNormalAngleOver30RatioAfter =
            quality_after.adjacentNormalAngleOver30Ratio;
        result.statistics.postSimplificationHighAspectFaceRatioAfter =
            quality_after.highAspectFaceRatio;
        result.statistics.postSimplificationExtremeAspectFaceRatioAfter =
            quality_after.extremeAspectFaceRatio;

        constexpr double ratio_epsilon = 1.0e-12;
        const bool topology_preserved =
            quality_after.validFaceCount == quality_before.validFaceCount &&
            quality_after.boundaryEdgeCount == quality_before.boundaryEdgeCount &&
            quality_after.nonManifoldEdgeCount ==
                quality_before.nonManifoldEdgeCount &&
            quality_after.componentCount == quality_before.componentCount &&
            quality_after.largestComponentFaceRatio + ratio_epsilon >=
                quality_before.largestComponentFaceRatio;
        const bool triangle_quality_is_safe =
            quality_after.highAspectFaceRatio <=
                std::max(
                    quality_before.highAspectFaceRatio * 1.02,
                    quality_before.highAspectFaceRatio + 0.002) &&
            quality_after.extremeAspectFaceRatio <=
                std::max(
                    quality_before.extremeAspectFaceRatio * 1.02,
                    quality_before.extremeAspectFaceRatio + 0.001);
        const bool normal_quality_is_safe =
            quality_after.adjacentNormalAngleMedianDegrees <=
                quality_before.adjacentNormalAngleMedianDegrees + 0.25 &&
            quality_after.adjacentNormalAngleP90Degrees <=
                quality_before.adjacentNormalAngleP90Degrees + 1.0 &&
            quality_after.adjacentNormalAngleOver30Ratio <=
                quality_before.adjacentNormalAngleOver30Ratio + 0.0025;
        const bool normal_quality_improved =
            quality_after.adjacentNormalAngleMedianDegrees + 0.10 <
                quality_before.adjacentNormalAngleMedianDegrees ||
            quality_after.adjacentNormalAngleP90Degrees + 0.50 <
                quality_before.adjacentNormalAngleP90Degrees ||
            quality_after.adjacentNormalAngleOver30Ratio + 0.001 <
                quality_before.adjacentNormalAngleOver30Ratio;
        const bool accept_denoising =
            moved_vertex_count > 0 &&
            meshVerticesInsidePaddedLayout(candidate, result.layout) &&
            topology_preserved &&
            triangle_quality_is_safe &&
            normal_quality_is_safe &&
            normal_quality_improved;
        result.statistics.postSimplificationSurfaceDenoisingAccepted =
            accept_denoising;
        if (accept_denoising)
        {
            result.mesh = std::move(candidate);
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
    topology_quality_thresholds.maximumClosedGenus =
        std::max(0.0f, options.topologyQualityMaximumClosedGenus);
    topology_quality_thresholds.maximumTopologicalComplexity =
        std::max(0, options.topologyQualityMaximumTopologicalComplexity);
    const MeshTopologyQualityStatistics topology_quality =
        evaluateMeshTopologyQuality(result.mesh, topology_quality_thresholds);
    result.statistics.topologyQualityUniqueEdgeCount =
        topology_quality.uniqueEdgeCount;
    result.statistics.topologyQualityReferencedVertexCount =
        topology_quality.referencedVertexCount;
    result.statistics.topologyQualityEulerCharacteristic =
        topology_quality.eulerCharacteristic;
    result.statistics.topologyQualityTopologicalComplexity =
        topology_quality.topologicalComplexity;
    result.statistics.topologyQualityClosedGenusEstimate =
        topology_quality.closedGenusEstimate;
    result.statistics.topologyQualityClosedTopologyEvaluated =
        topology_quality.closedTopologyEvaluated;
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
    result.statistics.topologyQualityAdjacentFacePairCount =
        topology_quality.adjacentFacePairCount;
    result.statistics.topologyQualityAdjacentNormalAngleMedianDegrees =
        topology_quality.adjacentNormalAngleMedianDegrees;
    result.statistics.topologyQualityAdjacentNormalAngleP90Degrees =
        topology_quality.adjacentNormalAngleP90Degrees;
    result.statistics.topologyQualityAdjacentNormalAngleOver30Ratio =
        topology_quality.adjacentNormalAngleOver30Ratio;
    if (options.collectZeroCrossingDiagnostics)
    {
        MeshBoundaryAttributionOptions attribution_options;
        attribution_options.minimumSourceCount =
            options.minimumSurfacePatchSourceCount;
        attribution_options.maximumInverseDepthSpread =
            options.maximumSurfacePatchInverseDepthSpread;
        attribution_options.minimumSurfaceWeightRatio =
            options.minimumSurfacePatchWeightRatio;
        attribution_options.maximumAbsoluteTsdf =
            options.maximumContourBandAbsoluteTsdf;
        std::vector<MeshBoundaryAttributionReason> vertex_reasons;
        const MeshBoundaryAttributionStatistics attribution =
            attributeMeshBoundaryEdges(
                result.mesh,
                result.layout,
                tsdf,
                weight,
                surfaceObservationWeight,
                geometrySourceMask,
                minimumInverseDepthSpread,
                supported,
                attribution_options,
                &vertex_reasons);
        result.boundaryAttributionDebugMesh = result.mesh;
        applyMeshBoundaryAttributionColors(
            &result.boundaryAttributionDebugMesh,
            vertex_reasons);
        result.statistics.boundaryAttributionEdgeCount =
            attribution.boundaryEdgeCount;
        result.statistics.boundaryAttributionNoObservationEdgeCount =
            attribution.noObservationEdgeCount;
        result.statistics.boundaryAttributionInsufficientSourceEdgeCount =
            attribution.insufficientSourceEdgeCount;
        result.statistics.boundaryAttributionDepthSpreadRejectedEdgeCount =
            attribution.depthSpreadRejectedEdgeCount;
        result.statistics.boundaryAttributionSurfaceWeightRejectedEdgeCount =
            attribution.surfaceWeightRejectedEdgeCount;
        result.statistics.boundaryAttributionAbsoluteTsdfRejectedEdgeCount =
            attribution.absoluteTsdfRejectedEdgeCount;
        result.statistics.boundaryAttributionSupportGateRejectedEdgeCount =
            attribution.supportGateRejectedEdgeCount;
        result.statistics
            .boundaryAttributionExtractionOrPostprocessEdgeCount =
            attribution.extractionOrPostprocessEdgeCount;
        result.statistics.boundaryAttributionUnclassifiedEdgeCount =
            attribution.unclassifiedEdgeCount;
    }
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
    result.statistics.effectiveDepthCompletenessDiagnostics =
        options.enableDepthCompletenessDiagnostics;
    result.statistics.effectiveDepthCompletenessGateEnforcement =
        options.enforceDepthCompletenessGate;
    if (options.enableDepthCompletenessDiagnostics)
    {
        if (options.progress)
        {
            options.progress(QStringLiteral("正在检查各视角模型完整性..."), 99);
        }
        DepthMeshCompletenessOptions completeness_options;
        completeness_options.maximumDepthSamplesPerFrame =
            options.depthCompletenessMaximumSamplesPerFrame;
        completeness_options.tolerance =
            maximum_voxel_size *
            std::max(1.0f, options.depthCompletenessToleranceVoxels);
        completeness_options.minimumP10FrameRecall =
            options.minimumDepthCompletenessP10Recall;
        completeness_options.minimumMedianFrameRecall =
            options.minimumDepthCompletenessMedianRecall;
        const DepthMeshCompletenessStatistics completeness =
            DepthMeshCompleteness::evaluate(
                result.mesh, frames, completeness_options);
        result.statistics.depthCompletenessAvailable = completeness.available;
        result.statistics.depthCompletenessTolerance = completeness.tolerance;
        result.statistics.depthCompletenessSampledPointCount =
            completeness.sampledDepthPointCount;
        result.statistics.depthCompletenessExplainedPointCount =
            completeness.explainedDepthPointCount;
        result.statistics.depthCompletenessAggregateRecall =
            completeness.aggregateRecall;
        result.statistics.depthCompletenessMinimumFrameRecall =
            completeness.minimumFrameRecall;
        result.statistics.depthCompletenessP10FrameRecall =
            completeness.p10FrameRecall;
        result.statistics.depthCompletenessMedianFrameRecall =
            completeness.medianFrameRecall;
        for (const DepthMeshFrameCompleteness &frame : completeness.frames)
        {
            result.statistics.depthCompletenessRefIndices.push_back(
                frame.refIndex);
            result.statistics.depthCompletenessFrameRecalls.push_back(
                frame.recall);
        }
        std::unordered_map<int, QString> orbital_role_by_ref_index;
        for (const QJsonValue &value : result.statistics.orbitalFrameRoles)
        {
            const QJsonObject role_object = value.toObject();
            orbital_role_by_ref_index.emplace(
                role_object.value(QStringLiteral("ref_index")).toInt(-1),
                role_object.value(QStringLiteral("role")).toString());
        }
        double gap_boundary_minimum_recall = 1.0;
        for (const DepthMeshFrameCompleteness &frame : completeness.frames)
        {
            const auto role_it = orbital_role_by_ref_index.find(frame.refIndex);
            if (role_it == orbital_role_by_ref_index.end() ||
                role_it->second != QStringLiteral("gap_boundary"))
            {
                continue;
            }
            result.statistics.depthCompletenessGapBoundaryAvailable = true;
            result.statistics.depthCompletenessGapBoundaryRefIndices.push_back(
                frame.refIndex);
            result.statistics.depthCompletenessGapBoundaryFrameRecalls.push_back(
                frame.recall);
            gap_boundary_minimum_recall = std::min(
                gap_boundary_minimum_recall, frame.recall);
        }
        if (result.statistics.depthCompletenessGapBoundaryAvailable)
        {
            result.statistics.depthCompletenessGapBoundaryMinimumRecall =
                gap_boundary_minimum_recall;
            result.statistics.depthCompletenessGapBoundaryGatePassed =
                gap_boundary_minimum_recall + 1.0e-9 >=
                options.minimumDepthCompletenessP10Recall;
        }
        result.statistics.depthCompletenessGatePassed =
            completeness.gatePassed &&
            result.statistics.depthCompletenessGapBoundaryGatePassed;
        if (options.enforceDepthCompletenessGate &&
            (!completeness.available ||
             !result.statistics.depthCompletenessGatePassed))
        {
            std::vector<DepthMeshFrameCompleteness> worst_frames(
                completeness.frames.cbegin(),
                completeness.frames.cend());
            std::sort(
                worst_frames.begin(),
                worst_frames.end(),
                [](const DepthMeshFrameCompleteness &lhs,
                   const DepthMeshFrameCompleteness &rhs)
                {
                    return lhs.recall < rhs.recall;
                });
            QStringList worst_labels;
            const int worst_count = std::min(
                3, static_cast<int>(worst_frames.size()));
            for (int index = 0; index < worst_count; ++index)
            {
                const DepthMeshFrameCompleteness &frame =
                    worst_frames[static_cast<std::size_t>(index)];
                QString role;
                const auto role_it =
                    orbital_role_by_ref_index.find(frame.refIndex);
                if (role_it != orbital_role_by_ref_index.end())
                {
                    role = role_it->second;
                }
                worst_labels.push_back(
                    role.isEmpty()
                        ? QStringLiteral("%1=%2%")
                              .arg(frame.refIndex)
                              .arg(100.0 * frame.recall, 0, 'f', 1)
                        : QStringLiteral("%1=%2%[%3]")
                              .arg(frame.refIndex)
                              .arg(100.0 * frame.recall, 0, 'f', 1)
                              .arg(role));
            }
            result.errorMessage = QStringLiteral(
                "TSDF 深度观测完整性质量门未通过：中位召回率=%1，"
                "P10 召回率=%2，最低召回率=%3；最差视角=%4。"
                "模型可能存在整侧缺失或大面积空洞。")
                                      .arg(completeness.medianFrameRecall, 0, 'f', 4)
                                      .arg(completeness.p10FrameRecall, 0, 'f', 4)
                                      .arg(completeness.minimumFrameRecall, 0, 'f', 4)
                                      .arg(worst_labels.join(
                                          QStringLiteral(", ")));
            return result;
        }
    }
    if (options.visibilityOccupancyCellBoundaryExtraction &&
        !native_carrier_field.empty())
    {
        result.visibilityOccupancyCarrierField.sampleDimensions =
            native_carrier_dimensions;
        result.visibilityOccupancyCarrierField.boundsMin =
            native_carrier_bounds_min;
        result.visibilityOccupancyCarrierField.boundsMax =
            native_carrier_bounds_max;
        result.visibilityOccupancyCarrierField.signedWorldDistance =
            std::move(native_carrier_field);
    }
    if (options.visibilityOccupancyCellBoundaryExtraction)
    {
        const std::vector<float> &implicit_tsdf =
            visual_hull_completion_tsdf.empty()
            ? tsdf
            : visual_hull_completion_tsdf;
        if (implicit_tsdf.size() ==
            static_cast<std::size_t>(result.layout.sampleCount))
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                result.depthImplicitField.sampleDimensions[axis] =
                    result.layout.cells[axis] + 1;
            }
            result.depthImplicitField.boundsMin = result.layout.boundsMin;
            result.depthImplicitField.boundsMax = result.layout.boundsMax;
            result.depthImplicitField.signedWorldDistance.resize(
                implicit_tsdf.size());
            std::transform(
                implicit_tsdf.cbegin(),
                implicit_tsdf.cend(),
                result.depthImplicitField.signedWorldDistance.begin(),
                [truncation](float value)
                {
                    return std::clamp(value, -1.0f, 1.0f) * truncation;
                });
        }
    }
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
    QJsonArray rejected_frame_ref_indices;
    for (const int ref_index :
         statistics.robustFrameQualityRejectedRefIndices)
    {
        rejected_frame_ref_indices.append(ref_index);
    }
    QJsonArray auxiliary_surface_only_ref_indices;
    for (const int ref_index : statistics.auxiliarySurfaceOnlyRefIndices)
    {
        auxiliary_surface_only_ref_indices.append(ref_index);
    }
    QJsonArray orbital_coverage_protected_ref_indices;
    for (const int ref_index : statistics.orbitalCoverageProtectedRefIndices)
    {
        orbital_coverage_protected_ref_indices.append(ref_index);
    }
    QJsonArray orbital_gap_quality_floor_ref_indices;
    for (const int ref_index : statistics.orbitalGapQualityFloorRefIndices)
    {
        orbital_gap_quality_floor_ref_indices.append(ref_index);
    }
    QJsonArray depth_completeness_frames;
    std::unordered_map<int, QString> orbital_role_by_ref_index;
    for (const QJsonValue &value : statistics.orbitalFrameRoles)
    {
        const QJsonObject role_object = value.toObject();
        orbital_role_by_ref_index.emplace(
            role_object.value(QStringLiteral("ref_index")).toInt(-1),
            role_object.value(QStringLiteral("role")).toString());
    }
    const int completeness_frame_count = std::min(
        statistics.depthCompletenessRefIndices.size(),
        statistics.depthCompletenessFrameRecalls.size());
    for (int index = 0; index < completeness_frame_count; ++index)
    {
        QJsonObject frame_object{
            {QStringLiteral("ref_index"),
             statistics.depthCompletenessRefIndices[index]},
            {QStringLiteral("recall"),
             statistics.depthCompletenessFrameRecalls[index]}
        };
        const auto role_it = orbital_role_by_ref_index.find(
            statistics.depthCompletenessRefIndices[index]);
        if (role_it != orbital_role_by_ref_index.end())
        {
            frame_object.insert(QStringLiteral("orbital_role"), role_it->second);
        }
        depth_completeness_frames.append(frame_object);
    }
    QJsonArray depth_completeness_gap_boundary_frames;
    const int gap_boundary_frame_count = std::min(
        statistics.depthCompletenessGapBoundaryRefIndices.size(),
        statistics.depthCompletenessGapBoundaryFrameRecalls.size());
    for (int index = 0; index < gap_boundary_frame_count; ++index)
    {
        depth_completeness_gap_boundary_frames.append(QJsonObject{
            {QStringLiteral("ref_index"),
             statistics.depthCompletenessGapBoundaryRefIndices[index]},
            {QStringLiteral("recall"),
             statistics.depthCompletenessGapBoundaryFrameRecalls[index]}
        });
    }
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
        {QStringLiteral("bounds_candidate_sample_count"),
         static_cast<double>(statistics.boundsCandidateSampleCount)},
        {QStringLiteral("bounds_trusted_sample_count"),
         static_cast<double>(statistics.boundsTrustedSampleCount)},
        {QStringLiteral("bounds_selected_sample_count"),
         static_cast<double>(statistics.boundsSelectedSampleCount)},
        {QStringLiteral("bounds_used_evidence_aware_samples"),
         statistics.boundsUsedEvidenceAwareSamples},
        {QStringLiteral("bounds_fell_back_to_candidate_samples"),
         statistics.boundsFellBackToCandidateSamples},
        {QStringLiteral("bounds_selection_reason"),
         statistics.boundsSelectionReason},
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
        {QStringLiteral("support_mask_free_space_surface_veto_count"),
         static_cast<double>(statistics.supportMaskFreeSpaceSurfaceVetoCount)},
        {QStringLiteral("narrow_band_activation_total_block_count"),
         static_cast<double>(
             statistics.narrowBandActivationTotalBlockCount)},
        {QStringLiteral("narrow_band_activation_active_block_count"),
         static_cast<double>(
             statistics.narrowBandActivationActiveBlockCount)},
        {QStringLiteral("narrow_band_activation_valid_source_sample_count"),
         static_cast<double>(
             statistics.narrowBandActivationValidSourceSampleCount)},
        {QStringLiteral("narrow_band_activation_marked_ray_sample_count"),
         static_cast<double>(
             statistics.narrowBandActivationMarkedRaySampleCount)},
        {QStringLiteral("narrow_band_activation_skipped_sample_count"),
         static_cast<double>(
             statistics.narrowBandActivationSkippedSampleCount)},
        {QStringLiteral("auxiliary_outside_surface_band_rejected_count"),
         static_cast<double>(statistics.auxiliaryOutsideSurfaceBandRejectedCount)},
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
        {QStringLiteral("unconfirmed_native_observation_count"),
         static_cast<double>(statistics.unconfirmedNativeObservationCount)},
        {QStringLiteral("weak_native_observation_count"),
         static_cast<double>(statistics.weakNativeObservationCount)},
        {QStringLiteral("repaired_observation_count"),
         static_cast<double>(statistics.repairedObservationCount)},
        {QStringLiteral("strong_native_observation_count"),
         static_cast<double>(statistics.strongNativeObservationCount)},
        {QStringLiteral("inverse_depth_spread_downweighted_observation_count"),
         static_cast<double>(
             statistics.inverseDepthSpreadDownweightedObservationCount)},
        {QStringLiteral("inverse_depth_spread_very_weak_observation_count"),
         static_cast<double>(
             statistics.inverseDepthSpreadVeryWeakObservationCount)},
        {QStringLiteral(
             "inverse_depth_spread_support_lifted_observation_count"),
         static_cast<double>(
             statistics.inverseDepthSpreadSupportLiftedObservationCount)},
        {QStringLiteral("weak_evidence_outside_surface_band_rejected_count"),
         static_cast<double>(
             statistics.weakEvidenceOutsideSurfaceBandRejectedCount)},
        {QStringLiteral("effective_pixel_evidence_weighting"),
         statistics.effectivePixelEvidenceWeighting},
        {QStringLiteral("effective_unconfirmed_native_observation_multiplier"),
         statistics.effectiveUnconfirmedNativeObservationMultiplier},
        {QStringLiteral("effective_weak_native_observation_multiplier"),
         statistics.effectiveWeakNativeObservationMultiplier},
        {QStringLiteral("effective_repaired_observation_multiplier"),
         statistics.effectiveRepairedObservationMultiplier},
        {QStringLiteral("effective_inverse_depth_spread_weighting"),
         statistics.effectiveInverseDepthSpreadWeighting},
        {QStringLiteral("effective_inverse_depth_spread_weight_knee"),
         statistics.effectiveInverseDepthSpreadWeightKnee},
        {QStringLiteral("effective_inverse_depth_spread_weight_zero"),
         statistics.effectiveInverseDepthSpreadWeightZero},
        {QStringLiteral(
             "effective_minimum_inverse_depth_spread_weight_multiplier"),
         statistics.effectiveMinimumInverseDepthSpreadWeightMultiplier},
        {QStringLiteral(
             "effective_inverse_depth_spread_support_weight_decoupling"),
         statistics.effectiveInverseDepthSpreadSupportWeightDecoupling},
        {QStringLiteral(
             "effective_inverse_depth_spread_support_weight_exponent"),
         statistics.effectiveInverseDepthSpreadSupportWeightExponent},
        {QStringLiteral("effective_evidence_support_weight_decoupling"),
         statistics.effectiveEvidenceSupportWeightDecoupling},
        {QStringLiteral("effective_evidence_support_weight_exponent"),
         statistics.effectiveEvidenceSupportWeightExponent},
        {QStringLiteral("effective_weak_evidence_surface_only_integration"),
         statistics.effectiveWeakEvidenceSurfaceOnlyIntegration},
        {QStringLiteral("effective_weak_evidence_surface_band_voxels"),
         statistics.effectiveWeakEvidenceSurfaceBandVoxels},
        {QStringLiteral("evidence_support_recovered_sample_count"),
         static_cast<double>(statistics.evidenceSupportRecoveredSampleCount)},
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
        {QStringLiteral("surface_patch_rejected_surface_weight_ratio_count"),
         static_cast<double>(
             statistics.surfacePatchRejectedSurfaceWeightRatioCount)},
        {QStringLiteral("surface_patch_rejected_absolute_tsdf_count"),
         static_cast<double>(statistics.surfacePatchRejectedAbsoluteTsdfCount)},
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
        {QStringLiteral("contour_band_zero_crossing_rejected_weight_count"),
         static_cast<double>(statistics.contourBandZeroCrossingRejectedWeightCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_source_overlap_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedSourceOverlapCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_depth_spread_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedDepthSpreadCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_free_space_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedFreeSpaceCount)},
        {QStringLiteral(
             "contour_band_zero_crossing_rejected_surface_weight_ratio_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedSurfaceWeightRatioCount)},
        {QStringLiteral(
             "contour_band_zero_crossing_rejected_absolute_tsdf_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedAbsoluteTsdfCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_neighborhood_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedNeighborhoodCount)},
        {QStringLiteral(
             "contour_band_zero_crossing_rejected_geometry_support_count"),
         static_cast<double>(
             statistics.contourBandZeroCrossingRejectedGeometrySupportCount)},
        {QStringLiteral("contour_band_zero_crossing_rejected_no_sign_pair_count"),
         static_cast<double>(statistics.contourBandZeroCrossingRejectedNoSignPairCount)},
        {QStringLiteral("effective_maximum_contour_band_absolute_tsdf"),
         statistics.effectiveMaximumContourBandAbsoluteTsdf},
        {QStringLiteral("effective_zero_crossing_diagnostics"),
         statistics.effectiveZeroCrossingDiagnostics},
        {QStringLiteral("effective_consistent_iso_surface_extraction"),
         statistics.effectiveConsistentIsoSurfaceExtraction},
        {QStringLiteral("effective_mc33_iso_surface_extraction"),
         statistics.effectiveMc33IsoSurfaceExtraction},
        {QStringLiteral("effective_mc33_require_supported_sign_change"),
         statistics.effectiveMc33RequireSupportedSignChange},
        {QStringLiteral("mc33_support_masked_sample_count"),
         static_cast<double>(statistics.mc33SupportMaskedSampleCount)},
        {QStringLiteral("mc33_rejected_unsupported_cell_face_count"),
         static_cast<double>(
             statistics.mc33RejectedUnsupportedCellFaceCount)},
        {QStringLiteral("iso_surface_extraction_non_manifold_edge_count"),
         statistics.isoSurfaceExtractionNonManifoldEdgeCount},
        {QStringLiteral("iso_surface_extraction_component_count"),
         statistics.isoSurfaceExtractionComponentCount},
        {QStringLiteral("iso_surface_extraction_euler_characteristic"),
         statistics.isoSurfaceExtractionEulerCharacteristic},
        {QStringLiteral("initial_degenerate_removed_face_count"),
         statistics.initialDegenerateRemovedFaceCount},
        {QStringLiteral("component_filter_removed_face_count"),
         statistics.componentFilterRemovedFaceCount},
        {QStringLiteral("effective_geometry_zero_crossing_recovery"),
         statistics.effectiveGeometryZeroCrossingRecovery},
        {QStringLiteral("geometry_zero_crossing_candidate_sample_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingCandidateSampleCount)},
        {QStringLiteral("geometry_zero_crossing_recovered_sample_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingRecoveredSampleCount)},
        {QStringLiteral("effective_cross_view_anchored_surface_recovery"),
         statistics.effectiveCrossViewAnchoredSurfaceRecovery},
        {QStringLiteral("cross_view_anchored_observed_sample_count"),
         static_cast<double>(
             statistics.crossViewAnchoredObservedSampleCount)},
        {QStringLiteral("cross_view_anchored_eligible_sample_count"),
         static_cast<double>(
             statistics.crossViewAnchoredEligibleSampleCount)},
        {QStringLiteral("cross_view_anchored_candidate_sample_count"),
         static_cast<double>(
             statistics.crossViewAnchoredCandidateSampleCount)},
        {QStringLiteral("cross_view_anchored_recovered_sample_count"),
         static_cast<double>(
             statistics.crossViewAnchoredRecoveredSampleCount)},
        {QStringLiteral("cross_view_anchored_executed_growth_pass_count"),
         statistics.crossViewAnchoredExecutedGrowthPassCount},
        {QStringLiteral(
             "effective_cross_view_anchored_minimum_observation_weight"),
         statistics.effectiveCrossViewAnchoredMinimumObservationWeight},
        {QStringLiteral(
             "effective_cross_view_anchored_minimum_supported_corners"),
         statistics.effectiveCrossViewAnchoredMinimumSupportedCorners},
        {QStringLiteral("effective_cross_view_anchored_minimum_cell_votes"),
         statistics.effectiveCrossViewAnchoredMinimumCellVotes},
        {QStringLiteral("effective_cross_view_anchored_growth_passes"),
         statistics.effectiveCrossViewAnchoredGrowthPasses},
        {QStringLiteral("effective_geometry_zero_crossing_cell_sheets"),
         statistics.effectiveGeometryZeroCrossingCellSheets},
        {QStringLiteral(
             "geometry_zero_crossing_sheet_candidate_cell_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingSheetCandidateCellCount)},
        {QStringLiteral(
             "geometry_zero_crossing_sheet_accepted_cell_count"),
         static_cast<double>(
             statistics.geometryZeroCrossingSheetAcceptedCellCount)},
        {QStringLiteral("geometry_zero_crossing_sheet_component_count"),
         statistics.geometryZeroCrossingSheetComponentCount},
        {QStringLiteral(
             "geometry_zero_crossing_sheet_accepted_component_count"),
         statistics.geometryZeroCrossingSheetAcceptedComponentCount},
        {QStringLiteral(
             "geometry_zero_crossing_sheet_rejected_small_component_count"),
         statistics.geometryZeroCrossingSheetRejectedSmallComponentCount},
        {QStringLiteral(
             "geometry_zero_crossing_sheet_rejected_anchor_component_count"),
         statistics.geometryZeroCrossingSheetRejectedAnchorComponentCount},
        {QStringLiteral(
             "effective_maximum_geometry_zero_crossing_sheet_single_vote_absolute_tsdf"),
         statistics
             .effectiveMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf},
        {QStringLiteral("effective_global_implicit_regularization"),
         statistics.effectiveGlobalImplicitRegularization},
        {QStringLiteral("effective_implicit_regularization_levels"),
         statistics.effectiveImplicitRegularizationLevels},
        {QStringLiteral(
             "effective_implicit_regularization_passes_per_level"),
         statistics.effectiveImplicitRegularizationPassesPerLevel},
        {QStringLiteral("effective_implicit_regularization_smoothness"),
         statistics.effectiveImplicitRegularizationSmoothness},
        {QStringLiteral("effective_implicit_regularization_data_fidelity"),
         statistics.effectiveImplicitRegularizationDataFidelity},
        {QStringLiteral("effective_implicit_regularization_maximum_update"),
         statistics.effectiveImplicitRegularizationMaximumUpdate},
        {QStringLiteral("effective_implicit_regularization_edge_threshold"),
         statistics.effectiveImplicitRegularizationEdgeThreshold},
        {QStringLiteral(
             "effective_implicit_regularization_recover_axial_gaps"),
         statistics.effectiveImplicitRegularizationRecoverAxialGaps},
        {QStringLiteral(
             "effective_implicit_regularization_minimum_bridge_axes"),
         statistics.effectiveImplicitRegularizationMinimumBridgeAxes},
        {QStringLiteral(
             "effective_implicit_regularization_maximum_bridge_prediction_delta"),
         statistics
             .effectiveImplicitRegularizationMaximumBridgePredictionDelta},
        {QStringLiteral(
             "implicit_regularization_bridge_candidate_count"),
         static_cast<double>(
             statistics.implicitRegularizationBridgeCandidateCount)},
        {QStringLiteral("implicit_regularization_recovered_sample_count"),
         static_cast<double>(
             statistics.implicitRegularizationRecoveredSampleCount)},
        {QStringLiteral("implicit_regularization_update_operation_count"),
         static_cast<double>(
             statistics.implicitRegularizationUpdateOperationCount)},
        {QStringLiteral("implicit_regularization_mean_absolute_update"),
         statistics.implicitRegularizationMeanAbsoluteUpdate},
        {QStringLiteral("implicit_regularization_maximum_absolute_update"),
         statistics.implicitRegularizationMaximumAbsoluteUpdate},
        {QStringLiteral("implicit_regularization_elapsed_ms"),
         static_cast<double>(statistics.implicitRegularizationElapsedMs)},
        {QStringLiteral("effective_adaptive_tgv_regularization"),
         statistics.effectiveAdaptiveTgvRegularization},
        {QStringLiteral(
             "effective_adaptive_tgv_global_visibility_field"),
         statistics.effectiveAdaptiveTgvGlobalVisibilityField},
        {QStringLiteral(
             "effective_adaptive_tgv_maximum_active_absolute_field"),
         statistics.effectiveAdaptiveTgvMaximumActiveAbsoluteField},
        {QStringLiteral("adaptive_tgv_histogram_sample_count"),
         static_cast<double>(statistics.adaptiveTgvHistogramSampleCount)},
        {QStringLiteral("adaptive_tgv_input_active_sample_count"),
         static_cast<double>(statistics.adaptiveTgvInputActiveSampleCount)},
        {QStringLiteral("adaptive_tgv_leaf_count"),
         static_cast<double>(statistics.adaptiveTgvLeafCount)},
        {QStringLiteral("adaptive_tgv_merged_node_count"),
         static_cast<double>(statistics.adaptiveTgvMergedNodeCount)},
        {QStringLiteral("adaptive_tgv_balance_split_count"),
         static_cast<double>(statistics.adaptiveTgvBalanceSplitCount)},
        {QStringLiteral("adaptive_tgv_global_visibility_sample_count"),
         static_cast<double>(
             statistics.adaptiveTgvGlobalVisibilitySampleCount)},
        {QStringLiteral("adaptive_tgv_recovery_eligible_sample_count"),
         static_cast<double>(
             statistics.adaptiveTgvRecoveryEligibleSampleCount)},
        {QStringLiteral(
             "adaptive_tgv_recovery_conflict_rejected_sample_count"),
         static_cast<double>(
             statistics.adaptiveTgvRecoveryConflictRejectedSampleCount)},
        {QStringLiteral("adaptive_tgv_recovered_sample_count"),
         static_cast<double>(statistics.adaptiveTgvRecoveredSampleCount)},
        {QStringLiteral(
             "effective_visual_hull_signed_distance_completion"),
         statistics.effectiveVisualHullSignedDistanceCompletion},
        {QStringLiteral(
             "effective_visual_hull_completion_band_voxels"),
         statistics.effectiveVisualHullCompletionBandVoxels},
        {QStringLiteral(
             "visual_hull_completion_occupied_sample_count"),
         static_cast<double>(
             statistics.visualHullCompletionOccupiedSampleCount)},
        {QStringLiteral(
             "visual_hull_completion_boundary_sample_count"),
         static_cast<double>(
             statistics.visualHullCompletionBoundarySampleCount)},
        {QStringLiteral(
             "visual_hull_completion_anchor_cell_count"),
         static_cast<double>(
             statistics.visualHullCompletionAnchorCellCount)},
        {QStringLiteral(
             "visual_hull_completion_frontier_cell_count"),
         static_cast<double>(
             statistics.visualHullCompletionFrontierCellCount)},
        {QStringLiteral(
             "visual_hull_completion_preserved_observed_sample_count"),
         static_cast<double>(
             statistics
                  .visualHullCompletionPreservedObservedSampleCount)},
        {QStringLiteral(
             "visual_hull_completion_observed_conflict_veto_sample_count"),
         static_cast<double>(
             statistics
                 .visualHullCompletionObservedConflictVetoSampleCount)},
        {QStringLiteral(
             "visual_hull_completion_recovered_sample_count"),
         static_cast<double>(
             statistics.visualHullCompletionRecoveredSampleCount)},
        {QStringLiteral(
             "visual_hull_completion_relaxed_sample_count"),
         static_cast<double>(
             statistics.visualHullCompletionRelaxedSampleCount)},
        {QStringLiteral(
             "effective_visual_hull_completion_topology_guard"),
         statistics.effectiveVisualHullCompletionTopologyGuard},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_evaluated"),
         statistics.visualHullCompletionTopologyGuardEvaluated},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_accepted"),
         statistics.visualHullCompletionTopologyGuardAccepted},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_rejection_flags"),
         static_cast<double>(
             statistics.visualHullCompletionTopologyGuardRejectionFlags)},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_face_count"),
         statistics.visualHullCompletionTopologyGuardBaselineFaceCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_face_count"),
         statistics.visualHullCompletionTopologyGuardCandidateFaceCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_boundary_edge_count"),
         statistics
             .visualHullCompletionTopologyGuardBaselineBoundaryEdgeCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_boundary_edge_count"),
         statistics
             .visualHullCompletionTopologyGuardCandidateBoundaryEdgeCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_non_manifold_edge_count"),
         statistics
             .visualHullCompletionTopologyGuardBaselineNonManifoldEdgeCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_non_manifold_edge_count"),
         statistics
             .visualHullCompletionTopologyGuardCandidateNonManifoldEdgeCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_component_count"),
         statistics.visualHullCompletionTopologyGuardBaselineComponentCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_component_count"),
         statistics.visualHullCompletionTopologyGuardCandidateComponentCount},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_euler_characteristic"),
         statistics
             .visualHullCompletionTopologyGuardBaselineEulerCharacteristic},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_euler_characteristic"),
         statistics
             .visualHullCompletionTopologyGuardCandidateEulerCharacteristic},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_baseline_topological_complexity"),
         statistics
             .visualHullCompletionTopologyGuardBaselineTopologicalComplexity},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_candidate_topological_complexity"),
         statistics
             .visualHullCompletionTopologyGuardCandidateTopologicalComplexity},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_surface_area_ratio"),
         statistics.visualHullCompletionTopologyGuardSurfaceAreaRatio},
        {QStringLiteral(
             "visual_hull_completion_topology_guard_bounds_diagonal_ratio"),
         statistics.visualHullCompletionTopologyGuardBoundsDiagonalRatio},
        {QStringLiteral("effective_visibility_occupancy_completion"),
         statistics.effectiveVisibilityOccupancyCompletion},
        {QStringLiteral("visibility_occupancy_rejected_empty_cut"),
         statistics.visibilityOccupancyRejectedEmptyCut},
        {QStringLiteral("visibility_occupancy_rejected_collapsed_cut"),
         statistics.visibilityOccupancyRejectedCollapsedCut},
        {QStringLiteral(
             "effective_visibility_occupancy_cell_boundary_extraction"),
         statistics.effectiveVisibilityOccupancyCellBoundaryExtraction},
        {QStringLiteral("effective_visibility_occupancy_resolution"),
         statistics.effectiveVisibilityOccupancyResolution},
        {QStringLiteral(
             "effective_visibility_occupancy_pairwise_capacity"),
         statistics.effectiveVisibilityOccupancyPairwiseCapacity},
        {QStringLiteral(
             "effective_visibility_occupancy_closing_iterations"),
         statistics.effectiveVisibilityOccupancyClosingIterations},
        {QStringLiteral(
             "effective_visibility_occupancy_minimum_depth_full_views_for_silhouette_prior"),
         statistics
             .effectiveVisibilityOccupancyMinimumDepthFullViewsForSilhouettePrior},
        {QStringLiteral(
             "visibility_occupancy_depth_support_fallback_count"),
         statistics.visibilityOccupancyDepthSupportFallbackCount},
        {QStringLiteral("visibility_occupancy_sample_count"),
         static_cast<double>(statistics.visibilityOccupancySampleCount)},
        {QStringLiteral(
             "visibility_occupancy_boundary_occupied_cell_count"),
         static_cast<double>(
             statistics.visibilityOccupancyBoundaryOccupiedCellCount)},
        {QStringLiteral(
             "visibility_occupancy_boundary_exposed_quad_count"),
         static_cast<double>(
             statistics.visibilityOccupancyBoundaryExposedQuadCount)},
        {QStringLiteral(
             "visibility_occupancy_boundary_non_manifold_edge_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyBoundaryNonManifoldEdgeCount)},
        {QStringLiteral(
             "visibility_occupancy_boundary_non_manifold_vertex_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyBoundaryNonManifoldVertexCount)},
        {QStringLiteral(
             "visibility_occupancy_boundary_body_euler_characteristic"),
         statistics.visibilityOccupancyBoundaryBodyEulerCharacteristic},
        {QStringLiteral(
             "visibility_occupancy_boundary_surface_euler_characteristic"),
         statistics.visibilityOccupancyBoundarySurfaceEulerCharacteristic},
        {QStringLiteral(
             "visibility_occupancy_boundary_topology_consistent"),
         statistics.visibilityOccupancyBoundaryTopologyConsistent},
        {QStringLiteral("visibility_occupancy_depth_empty_vote_count"),
         static_cast<double>(
             statistics.visibilityOccupancyDepthEmptyVoteCount)},
        {QStringLiteral("visibility_occupancy_depth_full_vote_count"),
         static_cast<double>(
             statistics.visibilityOccupancyDepthFullVoteCount)},
        {QStringLiteral(
             "visibility_occupancy_silhouette_full_prior_candidate_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancySilhouetteFullPriorCandidateSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_silhouette_full_prior_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancySilhouetteFullPriorSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_silhouette_full_prior_rejected_without_depth_support_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancySilhouetteFullPriorRejectedWithoutDepthSupportSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_silhouette_full_prior_capacity_total"),
         static_cast<double>(
             statistics
                 .visibilityOccupancySilhouetteFullPriorCapacityTotal)},
        {QStringLiteral("visibility_occupancy_full_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyFullSampleCount)},
        {QStringLiteral("visibility_occupancy_filled_bubble_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyFilledBubbleSampleCount)},
        {QStringLiteral("visibility_occupancy_removed_dust_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyRemovedDustSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_changed_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyClosingChangedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_added_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyClosingProposalAddedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_depth_empty_at_least_two_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyClosingProposalDepthEmptyAtLeastTwoSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_depth_empty_at_least_three_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyClosingProposalDepthEmptyAtLeastThreeSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_depth_empty_at_least_four_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyClosingProposalDepthEmptyAtLeastFourSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_depth_full_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyClosingProposalDepthFullSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_proposal_silhouette_outside_at_least_two_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyClosingProposalSilhouetteOutsideAtLeastTwoSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_protected_empty_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyClosingProtectedEmptySampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_depth_empty_protected_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyClosingDepthEmptyProtectedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_closing_silhouette_empty_protected_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyClosingSilhouetteEmptyProtectedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_candidate_component_count"),
         statistics.visibilityOccupancyHandleRepairCandidateComponentCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_accepted_candidate_count"),
         statistics.visibilityOccupancyHandleRepairAcceptedCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_accepted_subset_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairAcceptedSubsetCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_accepted_plateau_subset_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairAcceptedPlateauSubsetCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_attempted_subset_seed_count"),
         statistics
             .visibilityOccupancyHandleRepairAttemptedSubsetSeedCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_rejected_protected_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairRejectedProtectedCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_rejected_oversized_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairRejectedOversizedCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_rejected_topology_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairRejectedTopologyCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_rejected_protected_reachability_candidate_count"),
         statistics
             .visibilityOccupancyHandleRepairRejectedProtectedReachabilityCandidateCount},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_body_euler_before"),
         statistics.visibilityOccupancyHandleRepairBodyEulerBefore},
        {QStringLiteral(
             "visibility_occupancy_handle_repair_body_euler_after"),
         statistics.visibilityOccupancyHandleRepairBodyEulerAfter},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_filled_sample_count"),
         static_cast<double>(statistics
                                 .visibilityOccupancyWellComposedRepairFilledSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_accepted_pass_count"),
         statistics.visibilityOccupancyWellComposedRepairAcceptedPassCount},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_body_euler_before"),
         statistics.visibilityOccupancyWellComposedRepairBodyEulerBefore},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_body_euler_after"),
         statistics.visibilityOccupancyWellComposedRepairBodyEulerAfter},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_remaining_edge_checkerboard_count"),
         static_cast<double>(statistics
             .visibilityOccupancyWellComposedRepairRemainingEdgeCheckerboardCount)},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_remaining_vertex_occupied_defect_count"),
         static_cast<double>(statistics
             .visibilityOccupancyWellComposedRepairRemainingVertexOccupiedDefectCount)},
        {QStringLiteral(
             "visibility_occupancy_well_composed_repair_remaining_vertex_empty_defect_count"),
         static_cast<double>(statistics
             .visibilityOccupancyWellComposedRepairRemainingVertexEmptyDefectCount)},
        {QStringLiteral(
             "visibility_occupancy_recovered_unsupported_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyRecoveredUnsupportedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_preserved_observed_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyPreservedObservedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_overridden_observed_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyOverriddenObservedSampleCount)},
        {QStringLiteral("visibility_occupancy_forced_boundary_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyForcedBoundarySampleCount)},
        {QStringLiteral(
             "visibility_occupancy_adjusted_exact_iso_value_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyAdjustedExactIsoValueSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_trusted_observation_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyTrustedObservationSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_ignored_sign_conflict_observation_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyIgnoredSignConflictObservationCount)},
        {QStringLiteral("visibility_occupancy_blended_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyBlendedSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_clipped_residual_sample_count"),
         static_cast<double>(
             statistics.visibilityOccupancyClippedResidualSampleCount)},
        {QStringLiteral(
             "visibility_occupancy_carrier_sign_mismatch_sample_count"),
         static_cast<double>(
             statistics
                 .visibilityOccupancyCarrierSignMismatchSampleCount)},
        {QStringLiteral("visibility_occupancy_maximum_applied_residual"),
         statistics.visibilityOccupancyMaximumAppliedResidual},
        {QStringLiteral("visibility_occupancy_cut_energy"),
         static_cast<double>(statistics.visibilityOccupancyCutEnergy)},
        {QStringLiteral("adaptive_tgv_two_to_one_balanced"),
         statistics.adaptiveTgvTwoToOneBalanced},
        {QStringLiteral("adaptive_tgv_iteration_count"),
         statistics.adaptiveTgvIterationCount},
        {QStringLiteral("adaptive_tgv_initial_mean_absolute_curvature"),
         statistics.adaptiveTgvInitialMeanAbsoluteCurvature},
        {QStringLiteral("adaptive_tgv_final_mean_absolute_curvature"),
         statistics.adaptiveTgvFinalMeanAbsoluteCurvature},
        {QStringLiteral("adaptive_tgv_final_mean_absolute_update"),
         statistics.adaptiveTgvFinalMeanAbsoluteUpdate},
        {QStringLiteral("adaptive_tgv_octree_elapsed_ms"),
         static_cast<double>(statistics.adaptiveTgvOctreeElapsedMs)},
        {QStringLiteral("adaptive_tgv_solver_elapsed_ms"),
         static_cast<double>(statistics.adaptiveTgvSolverElapsedMs)},
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
        {QStringLiteral("effective_robust_frame_quality_rejection"),
         statistics.effectiveRobustFrameQualityRejection},
        {QStringLiteral("robust_frame_quality_rejected_frame_count"),
         statistics.robustFrameQualityRejectedFrameCount},
        {QStringLiteral("robust_frame_quality_rejected_ref_indices"),
         rejected_frame_ref_indices},
        {QStringLiteral("auxiliary_surface_only_frame_count"),
         statistics.auxiliarySurfaceOnlyFrameCount},
        {QStringLiteral("auxiliary_surface_only_ref_indices"),
         auxiliary_surface_only_ref_indices},
        {QStringLiteral("effective_orbital_frame_coverage_protection"),
         statistics.effectiveOrbitalFrameCoverageProtection},
        {QStringLiteral("orbital_coverage_protected_frame_count"),
         statistics.orbitalCoverageProtectedFrameCount},
        {QStringLiteral("orbital_coverage_protected_ref_indices"),
         orbital_coverage_protected_ref_indices},
        {QStringLiteral("orbital_median_angular_spacing_degrees"),
         statistics.orbitalMedianAngularSpacingDegrees},
        {QStringLiteral("orbital_maximum_angular_gap_degrees"),
         statistics.orbitalMaximumAngularGapDegrees},
        {QStringLiteral("orbital_maximum_angular_gap_ratio"),
         statistics.orbitalMaximumAngularGapRatio},
        {QStringLiteral("orbital_significant_angular_gap"),
         statistics.orbitalSignificantAngularGap},
        {QStringLiteral("orbital_gap_start_ref_index"),
         statistics.orbitalGapStartRefIndex},
        {QStringLiteral("orbital_gap_end_ref_index"),
         statistics.orbitalGapEndRefIndex},
        {QStringLiteral("orbital_gap_opposite_ref_index"),
         statistics.orbitalGapOppositeRefIndex},
        {QStringLiteral("orbital_frame_roles"),
         statistics.orbitalFrameRoles},
        {QStringLiteral("effective_orbital_gap_boundary_recovery"),
         statistics.effectiveOrbitalGapBoundaryRecovery},
        {QStringLiteral("effective_orbital_gap_adaptive_truncation"),
         statistics.effectiveOrbitalGapAdaptiveTruncation},
        {QStringLiteral(
             "effective_orbital_gap_adaptive_truncation_scale"),
         statistics.effectiveOrbitalGapAdaptiveTruncationScale},
        {QStringLiteral(
             "effective_orbital_gap_adaptive_maximum_truncation_voxels"),
         statistics.effectiveOrbitalGapAdaptiveMaximumTruncationVoxels},
        {QStringLiteral("orbital_gap_quality_floor_frame_count"),
         statistics.orbitalGapQualityFloorFrameCount},
        {QStringLiteral("orbital_gap_quality_floor_ref_indices"),
         orbital_gap_quality_floor_ref_indices},
        {QStringLiteral("orbital_gap_boundary_recovery_candidate_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRecoveryCandidateCount)},
        {QStringLiteral("orbital_gap_boundary_recovery_accepted_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRecoveryAcceptedCount)},
        {QStringLiteral("orbital_gap_boundary_recovery_rejected_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRecoveryRejectedCount)},
        {QStringLiteral("orbital_gap_boundary_single_observation_count"),
         static_cast<double>(
             statistics.orbitalGapBoundarySingleObservationCount)},
        {QStringLiteral("orbital_gap_boundary_rejected_weight_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRejectedWeightCount)},
        {QStringLiteral(
             "orbital_gap_boundary_rejected_geometry_support_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRejectedGeometrySupportCount)},
        {QStringLiteral("orbital_gap_boundary_rejected_source_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRejectedSourceCount)},
        {QStringLiteral("orbital_gap_boundary_rejected_spread_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRejectedSpreadCount)},
        {QStringLiteral("orbital_gap_boundary_rejected_field_count"),
         static_cast<double>(
             statistics.orbitalGapBoundaryRejectedFieldCount)},
        {QStringLiteral(
             "effective_orbital_gap_boundary_minimum_observation_weight"),
         statistics.effectiveOrbitalGapBoundaryMinimumObservationWeight},
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
        {QStringLiteral("effective_uncertainty_adaptive_truncation"),
         statistics.effectiveUncertaintyAdaptiveTruncation},
        {QStringLiteral("uncertainty_adaptive_sample_count"),
         static_cast<double>(statistics.uncertaintyAdaptiveSampleCount)},
        {QStringLiteral("uncertainty_adaptive_p90_voxels"),
         statistics.uncertaintyAdaptiveP90Voxels},
        {QStringLiteral("uncertainty_adaptive_added_voxels"),
         statistics.uncertaintyAdaptiveAddedVoxels},
        {QStringLiteral("effective_uncertainty_adaptive_scale"),
         statistics.effectiveUncertaintyAdaptiveScale},
        {QStringLiteral("effective_uncertainty_adaptive_activation_ratio"),
         statistics.effectiveUncertaintyAdaptiveActivationRatio},
        {QStringLiteral(
             "effective_uncertainty_adaptive_maximum_truncation_voxels"),
         statistics.effectiveUncertaintyAdaptiveMaximumTruncationVoxels},
        {QStringLiteral("effective_truncation_voxels"),
         statistics.effectiveTruncationVoxels},
        {QStringLiteral("effective_surface_support_band_voxels"),
         statistics.effectiveSurfaceSupportBandVoxels},
        {QStringLiteral("effective_maximum_free_space_voxels"),
         statistics.effectiveMaximumFreeSpaceVoxels},
        {QStringLiteral("effective_minimum_support_mask_free_space_views"),
         statistics.effectiveMinimumSupportMaskFreeSpaceViews},
        {QStringLiteral("effective_surface_evidence_free_space_veto"),
         statistics.effectiveSurfaceEvidenceFreeSpaceVeto},
        {QStringLiteral("effective_narrow_band_activation"),
         statistics.effectiveNarrowBandActivation},
        {QStringLiteral(
             "effective_narrow_band_activation_block_size_samples"),
         statistics.effectiveNarrowBandActivationBlockSizeSamples},
        {QStringLiteral("effective_narrow_band_activation_depth_stride"),
         statistics.effectiveNarrowBandActivationDepthStride},
        {QStringLiteral(
             "effective_narrow_band_activation_ray_step_voxels"),
         statistics.effectiveNarrowBandActivationRayStepVoxels},
        {QStringLiteral("effective_narrow_band_activation_halo_blocks"),
         statistics.effectiveNarrowBandActivationHaloBlocks},
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
        {QStringLiteral("marching_cubes_boundary_edge_count"),
         statistics.marchingCubesBoundaryEdgeCount},
        {QStringLiteral("iso_surface_ambiguous_face_count"),
         static_cast<double>(statistics.isoSurfaceAmbiguousFaceCount)},
        {QStringLiteral("iso_surface_topology_adjusted_cell_count"),
         static_cast<double>(statistics.isoSurfaceTopologyAdjustedCellCount)},
        {QStringLiteral("iso_surface_decider_tie_count"),
         static_cast<double>(statistics.isoSurfaceDeciderTieCount)},
        {QStringLiteral("iso_surface_multiple_loop_cell_count"),
         static_cast<double>(statistics.isoSurfaceMultipleLoopCellCount)},
        {QStringLiteral("iso_surface_edge_vertex_cache_hit_count"),
         static_cast<double>(statistics.isoSurfaceEdgeVertexCacheHitCount)},
        {QStringLiteral("iso_surface_edge_vertex_cache_miss_count"),
         static_cast<double>(statistics.isoSurfaceEdgeVertexCacheMissCount)},
        {QStringLiteral("iso_surface_interior_loop_vertex_count"),
         static_cast<double>(statistics.isoSurfaceInteriorLoopVertexCount)},
        {QStringLiteral("iso_surface_rejected_degenerate_face_count"),
         static_cast<double>(statistics.isoSurfaceRejectedDegenerateFaceCount)},
        {QStringLiteral("iso_surface_unresolved_cell_count"),
         static_cast<double>(statistics.isoSurfaceUnresolvedCellCount)},
        {QStringLiteral("component_filtered_boundary_edge_count"),
         statistics.componentFilteredBoundaryEdgeCount},
        {QStringLiteral("weak_boundary_trimmed_boundary_edge_count"),
         statistics.weakBoundaryTrimmedBoundaryEdgeCount},
        {QStringLiteral("topology_cleaned_boundary_edge_count"),
         statistics.topologyCleanedBoundaryEdgeCount},
        {QStringLiteral("topology_cleaned_attribution_edge_count"),
         static_cast<double>(statistics.topologyCleanedAttributionEdgeCount)},
        {QStringLiteral("topology_cleaned_attribution_no_observation_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionNoObservationEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_insufficient_source_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionInsufficientSourceEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_depth_spread_rejected_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionDepthSpreadRejectedEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_surface_weight_rejected_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionSurfaceWeightRejectedEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_absolute_tsdf_rejected_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionAbsoluteTsdfRejectedEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_support_gate_rejected_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionSupportGateRejectedEdgeCount)},
        {QStringLiteral(
             "topology_cleaned_attribution_extraction_or_postprocess_edge_count"),
         static_cast<double>(
             statistics
                 .topologyCleanedAttributionExtractionOrPostprocessEdgeCount)},
        {QStringLiteral("topology_cleaned_attribution_unclassified_edge_count"),
         static_cast<double>(
             statistics.topologyCleanedAttributionUnclassifiedEdgeCount)},
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
        {QStringLiteral(
             "post_simplification_component_filter_removed_face_count"),
         statistics.postSimplificationComponentFilterRemovedFaceCount},
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
        {QStringLiteral(
             "final_hole_fill_patch_evidence_validation_attempted"),
         statistics.finalHoleFillPatchEvidenceValidationAttempted},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_validation_accepted"),
         statistics.finalHoleFillPatchEvidenceValidationAccepted},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_vertex_sample_count"),
         statistics.finalHoleFillPatchEvidenceVertexSampleCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_face_center_sample_count"),
         statistics.finalHoleFillPatchEvidenceFaceCenterSampleCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_accepted_sample_count"),
         statistics.finalHoleFillPatchEvidenceAcceptedSampleCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_rejected_support_sample_count"),
         statistics.finalHoleFillPatchEvidenceRejectedSupportSampleCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_rejected_conflict_sample_count"),
         statistics.finalHoleFillPatchEvidenceRejectedConflictSampleCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_minimum_supporting_view_count"),
         statistics.finalHoleFillPatchEvidenceMinimumSupportingViewCount},
        {QStringLiteral(
             "final_hole_fill_patch_evidence_maximum_conflict_view_count"),
         statistics.finalHoleFillPatchEvidenceMaximumConflictViewCount},
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
        {QStringLiteral("effective_surface_denoising_mu"),
         statistics.effectiveSurfaceDenoisingMu},
        {QStringLiteral("effective_maximum_surface_denoising_displacement_voxels"),
         statistics.effectiveMaximumSurfaceDenoisingDisplacementVoxels},
        {QStringLiteral("effective_maximum_surface_denoising_normal_angle_degrees"),
         statistics.effectiveMaximumSurfaceDenoisingNormalAngleDegrees},
        {QStringLiteral("effective_surface_denoising_boundary_protection_rings"),
         statistics.effectiveSurfaceDenoisingBoundaryProtectionRings},
        {QStringLiteral("effective_protected_taubin_surface_denoising"),
         statistics.effectiveProtectedTaubinSurfaceDenoising},
        {QStringLiteral("effective_post_simplification_surface_denoising"),
         statistics.effectivePostSimplificationSurfaceDenoising},
        {QStringLiteral("post_simplification_surface_denoising_attempted"),
         statistics.postSimplificationSurfaceDenoisingAttempted},
        {QStringLiteral("post_simplification_surface_denoising_accepted"),
         statistics.postSimplificationSurfaceDenoisingAccepted},
        {QStringLiteral(
             "post_simplification_smoothed_surface_vertex_count"),
         statistics.postSimplificationSmoothedSurfaceVertexCount},
        {QStringLiteral(
             "post_simplification_tangential_relaxed_vertex_count"),
         statistics.postSimplificationTangentialRelaxedVertexCount},
        {QStringLiteral(
             "post_simplification_high_aspect_face_ratio_before"),
         statistics.postSimplificationHighAspectFaceRatioBefore},
        {QStringLiteral(
             "post_simplification_high_aspect_face_ratio_after"),
         statistics.postSimplificationHighAspectFaceRatioAfter},
        {QStringLiteral(
             "post_simplification_extreme_aspect_face_ratio_before"),
         statistics.postSimplificationExtremeAspectFaceRatioBefore},
        {QStringLiteral(
             "post_simplification_extreme_aspect_face_ratio_after"),
         statistics.postSimplificationExtremeAspectFaceRatioAfter},
        {QStringLiteral(
             "post_simplification_normal_angle_median_before"),
         statistics.postSimplificationNormalAngleMedianBefore},
        {QStringLiteral(
             "post_simplification_normal_angle_median_after"),
         statistics.postSimplificationNormalAngleMedianAfter},
        {QStringLiteral("post_simplification_normal_angle_p90_before"),
         statistics.postSimplificationNormalAngleP90Before},
        {QStringLiteral("post_simplification_normal_angle_p90_after"),
         statistics.postSimplificationNormalAngleP90After},
        {QStringLiteral(
             "post_simplification_normal_angle_over_30_ratio_before"),
         statistics.postSimplificationNormalAngleOver30RatioBefore},
        {QStringLiteral(
             "post_simplification_normal_angle_over_30_ratio_after"),
         statistics.postSimplificationNormalAngleOver30RatioAfter},
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
        {QStringLiteral("pre_denoising_face_orientation_repair_accepted"),
         statistics.preDenoisingFaceOrientationRepairAccepted},
        {QStringLiteral(
             "pre_denoising_face_orientation_inconsistent_shared_edge_count_before"),
         statistics
             .preDenoisingFaceOrientationInconsistentSharedEdgeCountBefore},
        {QStringLiteral("pre_denoising_face_orientation_flipped_face_count"),
         statistics.preDenoisingFaceOrientationFlippedFaceCount},
        {QStringLiteral("openmesh_simplification_attempted"),
         statistics.openMeshSimplificationAttempted},
        {QStringLiteral("effective_openmesh_simplification"),
         statistics.effectiveOpenMeshSimplification},
        {QStringLiteral("openmesh_simplification_accepted"),
         statistics.openMeshSimplificationAccepted},
        {QStringLiteral("openmesh_simplification_reached_target"),
         statistics.openMeshSimplificationReachedTarget},
        {QStringLiteral("openmesh_simplification_cancelled"),
         statistics.openMeshSimplificationCancelled},
        {QStringLiteral("openmesh_simplification_input_vertex_count"),
         statistics.openMeshSimplificationInputVertexCount},
        {QStringLiteral("openmesh_simplification_input_face_count"),
         statistics.openMeshSimplificationInputFaceCount},
        {QStringLiteral("openmesh_simplification_output_vertex_count"),
         statistics.openMeshSimplificationOutputVertexCount},
        {QStringLiteral("openmesh_simplification_output_face_count"),
         statistics.openMeshSimplificationOutputFaceCount},
        {QStringLiteral("openmesh_simplification_collapsed_vertex_count"),
         statistics.openMeshSimplificationCollapsedVertexCount},
        {QStringLiteral("openmesh_simplification_rejected_input_face_count"),
         statistics.openMeshSimplificationRejectedInputFaceCount},
        {QStringLiteral("openmesh_inconsistent_shared_edge_count_before"),
         statistics.openMeshInconsistentSharedEdgeCountBefore},
        {QStringLiteral("openmesh_reoriented_input_face_count"),
         statistics.openMeshReorientedInputFaceCount},
        {QStringLiteral("openmesh_removed_contradictory_face_count"),
         statistics.openMeshRemovedContradictoryFaceCount},
        {QStringLiteral("openmesh_orientation_conflict_count"),
         statistics.openMeshOrientationConflictCount},
        {QStringLiteral("openmesh_smoothing_applied"),
         statistics.openMeshSmoothingApplied},
        {QStringLiteral("openmesh_boundary_edge_count_before"),
         statistics.openMeshBoundaryEdgeCountBefore},
        {QStringLiteral("openmesh_boundary_edge_count_after"),
         statistics.openMeshBoundaryEdgeCountAfter},
        {QStringLiteral("openmesh_non_manifold_edge_count_before"),
         statistics.openMeshNonManifoldEdgeCountBefore},
        {QStringLiteral("openmesh_non_manifold_edge_count_after"),
         statistics.openMeshNonManifoldEdgeCountAfter},
        {QStringLiteral("openmesh_simplification_error"),
         statistics.openMeshSimplificationError},
        {QStringLiteral("final_face_orientation_repair_attempted"),
         statistics.finalFaceOrientationRepairAttempted},
        {QStringLiteral("final_face_orientation_repair_accepted"),
         statistics.finalFaceOrientationRepairAccepted},
        {QStringLiteral(
             "final_face_orientation_inconsistent_shared_edge_count_before"),
         statistics.finalFaceOrientationInconsistentSharedEdgeCountBefore},
        {QStringLiteral(
             "final_face_orientation_inconsistent_shared_edge_count_after"),
         statistics.finalFaceOrientationInconsistentSharedEdgeCountAfter},
        {QStringLiteral("final_face_orientation_flipped_face_count"),
         statistics.finalFaceOrientationFlippedFaceCount},
        {QStringLiteral("final_face_orientation_removed_face_count"),
         statistics.finalFaceOrientationRemovedFaceCount},
        {QStringLiteral("final_face_orientation_non_manifold_edge_count"),
         statistics.finalFaceOrientationNonManifoldEdgeCount},
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
        {QStringLiteral("topology_quality_referenced_vertex_count"),
         statistics.topologyQualityReferencedVertexCount},
        {QStringLiteral("topology_quality_euler_characteristic"),
         statistics.topologyQualityEulerCharacteristic},
        {QStringLiteral("topology_quality_topological_complexity"),
         statistics.topologyQualityTopologicalComplexity},
        {QStringLiteral("topology_quality_closed_genus_estimate"),
         statistics.topologyQualityClosedGenusEstimate},
        {QStringLiteral("topology_quality_closed_topology_evaluated"),
         statistics.topologyQualityClosedTopologyEvaluated},
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
        {QStringLiteral("topology_quality_adjacent_face_pair_count"),
         statistics.topologyQualityAdjacentFacePairCount},
        {QStringLiteral(
             "topology_quality_adjacent_normal_angle_median_degrees"),
         statistics.topologyQualityAdjacentNormalAngleMedianDegrees},
        {QStringLiteral(
             "topology_quality_adjacent_normal_angle_p90_degrees"),
         statistics.topologyQualityAdjacentNormalAngleP90Degrees},
        {QStringLiteral(
             "topology_quality_adjacent_normal_angle_over_30_ratio"),
         statistics.topologyQualityAdjacentNormalAngleOver30Ratio},
        {QStringLiteral("topology_quality_strict_gate_passed"),
         statistics.topologyQualityStrictGatePassed},
        {QStringLiteral("effective_depth_completeness_diagnostics"),
         statistics.effectiveDepthCompletenessDiagnostics},
        {QStringLiteral("effective_depth_completeness_gate_enforcement"),
         statistics.effectiveDepthCompletenessGateEnforcement},
        {QStringLiteral("depth_completeness_available"),
         statistics.depthCompletenessAvailable},
        {QStringLiteral("depth_completeness_gate_passed"),
         statistics.depthCompletenessGatePassed},
        {QStringLiteral("depth_completeness_tolerance"),
         statistics.depthCompletenessTolerance},
        {QStringLiteral("depth_completeness_sampled_point_count"),
         static_cast<double>(statistics.depthCompletenessSampledPointCount)},
        {QStringLiteral("depth_completeness_explained_point_count"),
         static_cast<double>(statistics.depthCompletenessExplainedPointCount)},
        {QStringLiteral("depth_completeness_aggregate_recall"),
         statistics.depthCompletenessAggregateRecall},
        {QStringLiteral("depth_completeness_minimum_frame_recall"),
         statistics.depthCompletenessMinimumFrameRecall},
        {QStringLiteral("depth_completeness_p10_frame_recall"),
         statistics.depthCompletenessP10FrameRecall},
        {QStringLiteral("depth_completeness_median_frame_recall"),
         statistics.depthCompletenessMedianFrameRecall},
        {QStringLiteral("depth_completeness_frames"),
         depth_completeness_frames},
        {QStringLiteral("depth_completeness_gap_boundary_available"),
         statistics.depthCompletenessGapBoundaryAvailable},
        {QStringLiteral("depth_completeness_gap_boundary_gate_passed"),
         statistics.depthCompletenessGapBoundaryGatePassed},
        {QStringLiteral("depth_completeness_gap_boundary_minimum_recall"),
         statistics.depthCompletenessGapBoundaryMinimumRecall},
        {QStringLiteral("depth_completeness_gap_boundary_frames"),
         depth_completeness_gap_boundary_frames},
        {QStringLiteral("boundary_attribution_edge_count"),
         static_cast<double>(statistics.boundaryAttributionEdgeCount)},
        {QStringLiteral("boundary_attribution_no_observation_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionNoObservationEdgeCount)},
        {QStringLiteral("boundary_attribution_insufficient_source_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionInsufficientSourceEdgeCount)},
        {QStringLiteral(
             "boundary_attribution_depth_spread_rejected_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionDepthSpreadRejectedEdgeCount)},
        {QStringLiteral(
             "boundary_attribution_surface_weight_rejected_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionSurfaceWeightRejectedEdgeCount)},
        {QStringLiteral(
             "boundary_attribution_absolute_tsdf_rejected_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionAbsoluteTsdfRejectedEdgeCount)},
        {QStringLiteral(
             "boundary_attribution_support_gate_rejected_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionSupportGateRejectedEdgeCount)},
        {QStringLiteral(
             "boundary_attribution_extraction_or_postprocess_edge_count"),
         static_cast<double>(
             statistics
                 .boundaryAttributionExtractionOrPostprocessEdgeCount)},
        {QStringLiteral("boundary_attribution_unclassified_edge_count"),
         static_cast<double>(
             statistics.boundaryAttributionUnclassifiedEdgeCount)},
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
