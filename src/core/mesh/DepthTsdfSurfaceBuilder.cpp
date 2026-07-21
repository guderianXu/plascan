#include "DepthTsdfSurfaceBuilder.h"

#include "DepthFrameUtils.h"
#include "MeshColorizer.h"
#include "MeshQuadricSimplifier.h"
#include "SurfaceReconstructorPostprocess.h"
#include "io/PathIO.h"

#include <QJsonArray>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <plapoint/mesh/marching_cubes.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <atomic>
#include <new>
#include <unordered_map>
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

int boundaryEdgeCount(const TriMesh &mesh)
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
    return static_cast<int>(std::count_if(
        counts.cbegin(), counts.cend(), [](const auto &entry) { return entry.second == 1; }));
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
            const bool candidate = weak_vertex[static_cast<std::size_t>(face.v[0])] &&
                weak_vertex[static_cast<std::size_t>(face.v[1])] &&
                weak_vertex[static_cast<std::size_t>(face.v[2])];
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
    if (options.enableSurfacePatchSupport)
    {
        std::uint64_t evidence_bytes = 0;
        if (!checkedMultiply(result.layout.sampleCount,
                             sizeof(std::uint16_t) * 2u + sizeof(float) * 3u,
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
    bool allowInvalidNearestPixelRecovery)
{
    DepthTsdfObservationSample result;
    const int nearest_column = static_cast<int>(std::lround(pixel.x));
    const int nearest_row = static_cast<int>(std::lround(pixel.y));
    if (nearest_row < 0 || nearest_row >= frame.depth.rows ||
        nearest_column < 0 || nearest_column >= frame.depth.cols)
    {
        return result;
    }

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
        sample->valid = true;
        sample->depth = depth;
        sample->confidence = confidence;
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
            candidate.depth = depth;
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

DepthTsdfResult DepthTsdfSurfaceBuilder::build(const QVector<DepthTsdfFrame> &frames,
                                               const DepthTsdfOptions &options)
{
    DepthTsdfResult result;
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

    QVector<cv::Mat> effective_depth_valid_masks;
    effective_depth_valid_masks.reserve(frames.size());
    const int erosion_pixels = std::clamp(
        options.depthValidBoundaryErosionPixels, 0, 4);
    if (erosion_pixels > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(erosion_pixels * 2 + 1, erosion_pixels * 2 + 1));
        for (const DepthTsdfFrame &frame : frames)
        {
            cv::Mat eroded;
            cv::erode(frame.depthValidMask, eroded, kernel);
            effective_depth_valid_masks.push_back(std::move(eroded));
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
    try
    {
        tsdf.assign(static_cast<std::size_t>(result.layout.sampleCount), 1.0f);
        weight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumObservationWeight.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
        maximumGeometrySupportCount.assign(
            static_cast<std::size_t>(result.layout.sampleCount), 0);
        support.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
        if (options.enableSurfacePatchSupport)
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
    std::atomic_bool cancelled{false};
    std::atomic<int> completed_z_slices{0};
    std::atomic<int> last_progress_percent{5};
    const int zSamples = result.layout.cells[2] + 1;
#ifdef MESHING_OPENMP
    const int workerCount = options.workerCount > 0 ? options.workerCount : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(workerCount) \
    reduction(+:integratedVoxelUpdates,rejectedProjectionCount,rejectedSupportMaskCount,supportMaskFreeSpaceUpdateCount,rejectedDepthValidCount,rejectedDepthCount,rejectedConfidenceCount,subpixelObservationCount,recoveredNeighborObservationCount,discontinuityRejectedCandidateCount,rejectedGeometryConsistencyCount,rejectedInvalidNearestPixelRecoveryCount)
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
                        options.allowInvalidNearestPixelRecovery);
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
                        confidence * std::max(0.0f, frame.frameQualityWeight);
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
                        if (options.enableSurfacePatchSupport)
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
    result.statistics.effectiveMinimumVoxelWeight = options.minimumVoxelWeight;
    result.statistics.effectiveMinimumSingleObservationWeight =
        options.minimumSingleObservationWeight;
    result.statistics.effectiveMinimumGeometryVerifiedObservationWeight =
        options.minimumGeometryVerifiedObservationWeight;
    result.statistics.effectiveMinimumGeometrySupportCount =
        options.minimumGeometrySupportCount;
    result.statistics.effectiveAllowGeometryVerifiedSingleObservation =
        options.allowGeometryVerifiedSingleObservation;
    result.statistics.effectiveDiscontinuityAwareSampling =
        options.enableDiscontinuityAwareSampling;
    result.statistics.effectiveMaximumInterpolationRelativeDepthSpread =
        options.maximumInterpolationRelativeDepthSpread;
    result.statistics.effectiveMaximumObservationInverseDepthSpread =
        options.maximumObservationInverseDepthSpread;
    result.statistics.effectiveAllowInvalidNearestPixelRecovery =
        options.allowInvalidNearestPixelRecovery;
    result.statistics.effectiveSurfacePatchSupport =
        options.enableSurfacePatchSupport;
    result.statistics.effectiveMinimumSurfacePatchSourceCount =
        options.minimumSurfacePatchSourceCount;
    result.statistics.effectiveMinimumSurfacePatchCoreNeighborCount =
        options.minimumSurfacePatchCoreNeighborCount;
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

    std::vector<std::uint8_t> supported(static_cast<std::size_t>(result.layout.sampleCount), 0);
    for (std::size_t index = 0; index < supported.size(); ++index)
    {
        bool single_view_supported = false;
        bool multi_view_supported = false;
        bool geometry_verified_single_view_supported = false;
        supported[index] = isSampleSupported(
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
            ++result.statistics.multiViewSupportedSampleCount;
        }
        else if (single_view_supported)
        {
            ++result.statistics.singleViewSupportedSampleCount;
            if (geometry_verified_single_view_supported)
            {
                ++result.statistics.geometryVerifiedSingleViewSupportedSampleCount;
            }
        }
        else if (support[index] == 1)
        {
            ++result.statistics.rejectedSingleObservationWeightCount;
        }
        else if (support[index] >= options.minimumDistinctCameraSupport)
        {
            ++result.statistics.rejectedAccumulatedWeightCount;
        }
        result.statistics.supportedSampleCount += supported[index] != 0;
    }
    if (options.enableSurfacePatchSupport && !geometrySourceMask.empty())
    {
        const std::vector<std::uint8_t> core_supported = supported;
        const int minimum_source_count = std::clamp(
            options.minimumSurfacePatchSourceCount, 2, 8);
        const int minimum_core_neighbor_count = std::clamp(
            options.minimumSurfacePatchCoreNeighborCount, 1, 26);
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
        for (int z = 1; z < result.layout.cells[2]; ++z)
        {
            for (int y = 1; y < result.layout.cells[1]; ++y)
            {
                for (int x = 1; x < result.layout.cells[0]; ++x)
                {
                    const std::size_t index = sampleIndex(result.layout, x, y, z);
                    if (supported[index] != 0 || support[index] == 0)
                    {
                        continue;
                    }
                    ++result.statistics.surfacePatchConsideredSampleCount;
                    if (maximumObservationWeight[index] <
                        options.minimumGeometryVerifiedObservationWeight)
                    {
                        ++result.statistics.surfacePatchRejectedWeightCount;
                        continue;
                    }
                    if (bitCount(geometrySourceMask[index]) < minimum_source_count)
                    {
                        ++result.statistics.surfacePatchRejectedSourceOverlapCount;
                        continue;
                    }
                    const std::uint16_t spread_value = minimumInverseDepthSpread[index];
                    if (spread_value == std::numeric_limits<std::uint16_t>::max() ||
                        static_cast<float>(spread_value) / 100000.0f > maximum_spread)
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
                            result.layout, neighbor_x, neighbor_y, neighbor_z);
                        if (core_supported[neighbor_index] == 0)
                        {
                            continue;
                        }
                        has_core_neighbor = true;
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
                        ++result.statistics.surfacePatchRejectedSourceOverlapCount;
                        continue;
                    }
                    if (!has_normal_agreement ||
                        agreeing_core_neighbor_count < minimum_core_neighbor_count)
                    {
                        ++result.statistics.surfacePatchRejectedNormalCount;
                        continue;
                    }
                    supported[index] = 1;
                    tsdf[index] = surface_candidate_tsdf[index];
                    ++result.statistics.surfacePatchRecoveredSampleCount;
                    ++result.statistics.supportedSampleCount;
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

    if (options.progress)
    {
        options.progress(QStringLiteral("正在提取 TSDF 零等值面..."), 75);
    }
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

    detail::removeDegenerateFaces(&result.mesh);
    detail::weldCoincidentVertices(&result.mesh, 1.0e-6f);
    detail::removeSmallConnectedComponents(
        &result.mesh,
        std::max(2, options.minimumComponentFaces),
        options.minimumComponentFaceRatio);
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
    result.statistics.boundaryEdgeCountBefore = boundaryEdgeCount(result.mesh);
    if (options.fillSmallBoundaryHoles)
    {
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
    result.statistics.compactedUnusedVertexCount =
        detail::compactReferencedVertices(&result.mesh);
    if (options.enableQuadricSimplification && options.simplifyTargetFaces > 0 &&
        result.mesh.faceCount() > options.simplifyTargetFaces)
    {
        QuadricSimplifyOptions simplify_options;
        simplify_options.targetFaceCount = options.simplifyTargetFaces;
        const QuadricSimplifyStatistics simplify_statistics = simplifyMeshQuadric(
            &result.mesh, simplify_options);
        result.statistics.effectiveQuadricSimplification = true;
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
    }
    result.statistics.boundaryEdgeCountAfter = boundaryEdgeCount(result.mesh);
    detail::recomputeNormals(&result.mesh);
    if (result.mesh.empty())
    {
        result.errorMessage = QStringLiteral("TSDF cleanup removed all mesh components");
        return result;
    }

    if (options.calculateVertexColors)
    {
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
            view.qualityWeight = frame.frameQualityWeight;
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
    const MeshConnectivityStats connectivity =
        VisualHullReconstructor::analyzeConnectivity(result.mesh);
    result.statistics.componentCount = connectivity.componentCount;
    result.statistics.largestComponentFaceRatio = connectivity.largestComponentFaceRatio;
    result.statistics.componentFaceCounts = connectivity.componentFaceCounts;
    result.statistics.components = connectivity.components;
    result.statistics.vertexCount = result.mesh.vertexCount();
    result.statistics.faceCount = result.mesh.faceCount();

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
        {QStringLiteral("rejected_accumulated_weight_count"),
         static_cast<double>(statistics.rejectedAccumulatedWeightCount)},
        {QStringLiteral("rejected_single_observation_weight_count"),
         static_cast<double>(statistics.rejectedSingleObservationWeightCount)},
        {QStringLiteral("surface_patch_recovered_sample_count"),
         static_cast<double>(statistics.surfacePatchRecoveredSampleCount)},
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
        {QStringLiteral("effective_discontinuity_aware_sampling"),
         statistics.effectiveDiscontinuityAwareSampling},
        {QStringLiteral("effective_maximum_interpolation_relative_depth_spread"),
         statistics.effectiveMaximumInterpolationRelativeDepthSpread},
        {QStringLiteral("effective_maximum_observation_inverse_depth_spread"),
         statistics.effectiveMaximumObservationInverseDepthSpread},
        {QStringLiteral("effective_allow_invalid_nearest_pixel_recovery"),
         statistics.effectiveAllowInvalidNearestPixelRecovery},
        {QStringLiteral("effective_surface_patch_support"),
         statistics.effectiveSurfacePatchSupport},
        {QStringLiteral("effective_minimum_surface_patch_source_count"),
         statistics.effectiveMinimumSurfacePatchSourceCount},
        {QStringLiteral("effective_minimum_surface_patch_core_neighbor_count"),
         statistics.effectiveMinimumSurfacePatchCoreNeighborCount},
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
        {QStringLiteral("boundary_edge_count_before"),
         statistics.boundaryEdgeCountBefore},
        {QStringLiteral("boundary_edge_count_after"),
         statistics.boundaryEdgeCountAfter},
        {QStringLiteral("compacted_unused_vertex_count"),
         statistics.compactedUnusedVertexCount},
        {QStringLiteral("filled_boundary_hole_count"),
         statistics.filledBoundaryHoleCount},
        {QStringLiteral("added_hole_fill_face_count"),
         statistics.addedHoleFillFaceCount},
        {QStringLiteral("smoothed_boundary_vertex_count"),
         statistics.smoothedBoundaryVertexCount},
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
