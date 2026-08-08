#include "SceneGeometryPreparation.h"

#include "CameraSceneViewMath.h"

#include <QRectF>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{

constexpr std::size_t kMaximumP95Samples = 250'000;
constexpr std::size_t kMaximumAxisSamples = 100'000;
constexpr std::size_t kCancellationCheckInterval = 4'096;
constexpr std::size_t kMaximumInitialSelectionReserve = 262'144;

bool isCancellationRequested(const std::atomic_bool *flag)
{
    return flag && flag->load(std::memory_order_relaxed);
}

bool isFinitePoint(const QVector3D &point)
{
    return std::isfinite(point.x())
        && std::isfinite(point.y())
        && std::isfinite(point.z());
}

QVector3D cloudPoint(const SceneRenderCloud &cloud, std::size_t index)
{
    const auto row = static_cast<plamatrix::Index>(index);
    return QVector3D(cloud.points()(row, 0),
                     cloud.points()(row, 1),
                     cloud.points()(row, 2));
}

} // namespace

CloudSpatialSummary prepareCloudSpatialSummary(
    const SceneRenderCloud &cloud,
    const std::atomic_bool *cancellationFlag)
{
    CloudSpatialSummary result;
    if (cloud.size() == 0 || isCancellationRequested(cancellationFlag))
    {
        return result;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    QVector<QVector3D> axis_samples;
    const bool needs_oriented_box = !cloud.hasFaces();
    if (needs_oriented_box)
    {
        axis_samples.reserve(static_cast<qsizetype>(
            std::min(cloud.size(), kMaximumAxisSamples + 1)));
    }

    std::size_t valid_count = 0;
    std::size_t axis_stride = 1;
    double minimum_elevation = std::numeric_limits<double>::infinity();
    double maximum_elevation = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const QVector3D point = cloudPoint(cloud, index);
        if (!isFinitePoint(point))
        {
            continue;
        }
        if (valid_count == 0)
        {
            result.aabbMinimum = point;
            result.aabbMaximum = point;
        }
        else
        {
            result.aabbMinimum.setX(std::min(result.aabbMinimum.x(), point.x()));
            result.aabbMinimum.setY(std::min(result.aabbMinimum.y(), point.y()));
            result.aabbMinimum.setZ(std::min(result.aabbMinimum.z(), point.z()));
            result.aabbMaximum.setX(std::max(result.aabbMaximum.x(), point.x()));
            result.aabbMaximum.setY(std::max(result.aabbMaximum.y(), point.y()));
            result.aabbMaximum.setZ(std::max(result.aabbMaximum.z(), point.z()));
        }
        sum_x += static_cast<double>(point.x());
        sum_y += static_cast<double>(point.y());
        sum_z += static_cast<double>(point.z());
        const std::size_t valid_index = valid_count++;
        minimum_elevation = std::min(minimum_elevation, double(point.z()));
        maximum_elevation = std::max(maximum_elevation, double(point.z()));
        if (needs_oriented_box && valid_index % axis_stride == 0)
        {
            axis_samples.push_back(point);
            if (axis_samples.size() > static_cast<qsizetype>(kMaximumAxisSamples))
            {
                qsizetype write_index = 0;
                for (qsizetype read_index = 0;
                     read_index < axis_samples.size();
                     read_index += 2)
                {
                    axis_samples[write_index++] = axis_samples.at(read_index);
                }
                axis_samples.resize(write_index);
                axis_stride *= 2;
            }
        }
    }
    if (valid_count == 0)
    {
        return result;
    }

    const double inverse_count = 1.0 / static_cast<double>(valid_count);
    const double center_x = sum_x * inverse_count;
    const double center_y = sum_y * inverse_count;
    const double center_z = sum_z * inverse_count;
    constexpr double maximum_float = static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(center_z)
        || std::abs(center_x) > maximum_float
        || std::abs(center_y) > maximum_float
        || std::abs(center_z) > maximum_float)
    {
        return result;
    }
    result.center = QVector3D(static_cast<float>(center_x),
                              static_cast<float>(center_y),
                              static_cast<float>(center_z));
    result.sourcePointCount = cloud.size();
    result.validPointCount = valid_count;
    result.elevationRange = {minimum_elevation, maximum_elevation};
    const std::size_t distance_stride = std::max<std::size_t>(
        1, (valid_count + kMaximumP95Samples - 1) / kMaximumP95Samples);
    std::vector<float> distances;
    distances.reserve(std::min(cloud.size(), kMaximumP95Samples));

    QVector3D local_min(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
    QVector3D local_max(std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest());
    const auto axes = needs_oriented_box
        ? xjw::gui::camera_scene::pointCloudPrincipalAxes(axis_samples)
        : xjw::gui::camera_scene::PointCloudPrincipalAxes{};
    std::size_t valid_index = 0;
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const QVector3D point = cloudPoint(cloud, index);
        if (!isFinitePoint(point))
        {
            continue;
        }
        if (valid_index++ % distance_stride == 0)
        {
            distances.push_back((point - result.center).length());
        }
        if (axes.valid)
        {
            const QVector3D offset = point - axes.center;
            const QVector3D local(QVector3D::dotProduct(offset, axes.first),
                                  QVector3D::dotProduct(offset, axes.second),
                                  QVector3D::dotProduct(offset, axes.third));
            local_min.setX(std::min(local_min.x(), local.x()));
            local_min.setY(std::min(local_min.y(), local.y()));
            local_min.setZ(std::min(local_min.z(), local.z()));
            local_max.setX(std::max(local_max.x(), local.x()));
            local_max.setY(std::max(local_max.y(), local.y()));
            local_max.setZ(std::max(local_max.z(), local.z()));
        }
    }

    if (!distances.empty())
    {
        const std::size_t p95 = std::min(
            distances.size() - 1,
            static_cast<std::size_t>(static_cast<double>(distances.size()) * 0.95));
        std::nth_element(distances.begin(), distances.begin() + p95, distances.end());
        result.p95Radius = std::max(1.0e-4f, distances[p95] * 1.15f);
    }
    if (axes.valid)
    {
        result.boxLineVertices = xjw::gui::camera_scene::orientedBoundingBoxLineVertices(
            axes, local_min, local_max);
    }
    if (result.boxLineVertices.isEmpty())
    {
        result.boxLineVertices = xjw::gui::camera_scene::axisAlignedBoundingBoxLineVertices(
            result.aabbMinimum, result.aabbMaximum);
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.valid = true;
    return result;
}

QByteArray preparePointScalarData(std::size_t pointCount,
                                  const QVector<int> &imageCounts,
                                  xjw::gui::tie_points::ScalarRange *range,
                                  const std::atomic_bool *cancellationFlag)
{
    if (range)
    {
        *range = {};
    }
    if (pointCount == 0
        || pointCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(float)
        || isCancellationRequested(cancellationFlag))
    {
        return {};
    }

    QByteArray data(static_cast<int>(pointCount * sizeof(float)), Qt::Uninitialized);
    float *output = reinterpret_cast<float *>(data.data());
    const bool valid_counts = imageCounts.size() == static_cast<qsizetype>(pointCount);
    int minimum_count = std::numeric_limits<int>::max();
    int maximum_count = std::numeric_limits<int>::lowest();
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        if (index % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const int count = valid_counts
            ? imageCounts.at(static_cast<qsizetype>(index))
            : 0;
        output[index] = static_cast<float>(count);
        minimum_count = std::min(minimum_count, count);
        maximum_count = std::max(maximum_count, count);
    }
    if (range && valid_counts)
    {
        *range = {double(minimum_count), double(maximum_count)};
    }
    return data;
}

PointRenderPreparation preparePointRenderData(
    const SceneRenderCloud &cloud,
    const QVector<int> &imageCounts,
    const std::atomic_bool *cancellationFlag)
{
    PointRenderPreparation result;
    if (cloud.size() == 0 || cloud.hasFaces())
    {
        return result;
    }
    result.spatialSummary = prepareCloudSpatialSummary(cloud, cancellationFlag);
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    if (cloud.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max() / (9 * int(sizeof(float)))))
    {
        return result;
    }

    constexpr int stride_floats = 9;
    constexpr float color_scale = 1.0f / 255.0f;
    result.vertexData.resize(static_cast<qsizetype>(cloud.size())
                             * stride_floats * static_cast<qsizetype>(sizeof(float)));
    result.scalarData.resize(static_cast<qsizetype>(cloud.size())
                             * static_cast<qsizetype>(sizeof(float)));
    float *output = reinterpret_cast<float *>(result.vertexData.data());
    float *scalar_output = reinterpret_cast<float *>(result.scalarData.data());
    const bool has_colors = cloud.hasColors();
    const bool has_normals = cloud.hasNormals();
    const bool has_image_counts = imageCounts.size()
        == static_cast<qsizetype>(cloud.size());
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const auto row = static_cast<plamatrix::Index>(index);
        float *vertex = output + index * stride_floats;
        vertex[0] = cloud.points()(row, 0);
        vertex[1] = cloud.points()(row, 1);
        vertex[2] = cloud.points()(row, 2);
        vertex[3] = has_normals ? cloud.normals()->getValue(row, 0) : 0.0f;
        vertex[4] = has_normals ? cloud.normals()->getValue(row, 1) : 0.0f;
        vertex[5] = has_normals ? cloud.normals()->getValue(row, 2) : 0.0f;
        vertex[6] = has_colors ? cloud.colors()->getValue(row, 0) * color_scale : 0.45f;
        vertex[7] = has_colors ? cloud.colors()->getValue(row, 1) * color_scale : 0.45f;
        vertex[8] = has_colors ? cloud.colors()->getValue(row, 2) * color_scale : 0.50f;
        scalar_output[index] = has_image_counts
            ? static_cast<float>(imageCounts.at(static_cast<qsizetype>(index)))
            : 0.0f;
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.pointCount = static_cast<int>(cloud.size());
    return result;
}

std::vector<PointVertexIndex> selectPointVertexIndices(
    const QByteArray &vertexData,
    int strideBytes,
    const QRect &screenRect,
    const QMatrix4x4 &clipMatrix,
    const QSize &viewportSize,
    const QPointF &sceneOffset,
    const std::atomic_bool *cancellationFlag)
{
    std::vector<PointVertexIndex> indices;
    const QRect rect = screenRect.normalized();
    if (strideBytes < 3 * int(sizeof(float))
        || strideBytes % int(sizeof(float)) != 0
        || vertexData.isEmpty()
        || vertexData.size() % strideBytes != 0
        || rect.width() < 3 || rect.height() < 3
        || viewportSize.width() <= 0 || viewportSize.height() <= 0
        || isCancellationRequested(cancellationFlag))
    {
        return indices;
    }

    const QRectF continuous_rect(rect.x(), rect.y(), rect.width(), rect.height());
    const std::size_t point_count = static_cast<std::size_t>(vertexData.size() / strideBytes);
    if (point_count > static_cast<std::size_t>(
            std::numeric_limits<PointVertexIndex>::max()))
    {
        return indices;
    }
    const int stride_floats = strideBytes / int(sizeof(float));
    const float *vertices = reinterpret_cast<const float *>(vertexData.constData());
    indices.reserve(std::min(point_count / 8, kMaximumInitialSelectionReserve));
    for (std::size_t index = 0; index < point_count; ++index)
    {
        if (index % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const float *vertex = vertices + index * static_cast<std::size_t>(stride_floats);
        const QVector4D clip = clipMatrix * QVector4D(vertex[0], vertex[1], vertex[2], 1.0f);
        if (!std::isfinite(clip.x())
            || !std::isfinite(clip.y())
            || !std::isfinite(clip.z())
            || !std::isfinite(clip.w())
            || clip.w() <= 1.0e-6f
            || clip.z() < -clip.w()
            || clip.z() > clip.w())
        {
            continue;
        }
        const QPointF screen_point(
            (clip.x() / clip.w() * 0.5f + 0.5f) * viewportSize.width() + sceneOffset.x(),
            (1.0f - (clip.y() / clip.w() * 0.5f + 0.5f)) * viewportSize.height()
                + sceneOffset.y());
        if (continuous_rect.contains(screen_point))
        {
            indices.push_back(static_cast<PointVertexIndex>(index));
        }
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    return indices;
}

PointSelectionPreparation preparePointSelection(
    const QByteArray &vertexData,
    int strideBytes,
    const QByteArray &scalarData,
    const QRect &screenRect,
    const QMatrix4x4 &clipMatrix,
    const QSize &viewportSize,
    const QPointF &sceneOffset,
    std::size_t maximumCompactPointCount,
    const std::atomic_bool *cancellationFlag)
{
    PointSelectionPreparation result;
    result.indices = selectPointVertexIndices(
        vertexData,
        strideBytes,
        screenRect,
        clipMatrix,
        viewportSize,
        sceneOffset,
        cancellationFlag);
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    const std::size_t compact_count = result.indices.size();
    const std::size_t maximum_byte_array_size = static_cast<std::size_t>(
        std::numeric_limits<qsizetype>::max());
    if (compact_count == 0
        || compact_count > maximumCompactPointCount
        || compact_count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || compact_count > maximum_byte_array_size / static_cast<std::size_t>(strideBytes)
        || compact_count > maximum_byte_array_size / sizeof(float))
    {
        return result;
    }

    const std::size_t point_count = static_cast<std::size_t>(vertexData.size() / strideBytes);
    const bool has_scalars = scalarData.size()
        == static_cast<qsizetype>(point_count * sizeof(float));
    result.pointCount = static_cast<int>(result.indices.size());
    result.vertexData.resize(static_cast<qsizetype>(result.pointCount) * strideBytes);
    result.scalarData.resize(static_cast<qsizetype>(result.pointCount)
                             * static_cast<qsizetype>(sizeof(float)));

    char *compact_vertices = result.vertexData.data();
    float *compact_scalars = reinterpret_cast<float *>(result.scalarData.data());
    const char *source_vertices = vertexData.constData();
    const float *source_scalars = has_scalars
        ? reinterpret_cast<const float *>(scalarData.constData())
        : nullptr;
    for (int compact_index = 0; compact_index < result.pointCount; ++compact_index)
    {
        if (static_cast<std::size_t>(compact_index)
                % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const std::size_t source_index = static_cast<std::size_t>(
            result.indices[static_cast<std::size_t>(compact_index)]);
        std::memcpy(compact_vertices + static_cast<qsizetype>(compact_index) * strideBytes,
                    source_vertices + static_cast<qsizetype>(source_index) * strideBytes,
                    static_cast<std::size_t>(strideBytes));
        compact_scalars[compact_index] = source_scalars ? source_scalars[source_index] : 0.0f;
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    return result;
}
