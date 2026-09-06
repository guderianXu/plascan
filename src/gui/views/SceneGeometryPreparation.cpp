#include "SceneGeometryPreparation.h"

#include "CameraSceneViewMath.h"

#include <QRectF>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace
{

    constexpr std::size_t kMaximumP95Samples = 250'000;
    constexpr std::size_t kMaximumAxisSamples = 100'000;
    constexpr std::size_t kCancellationCheckInterval = 4'096;
    constexpr std::size_t kMaximumInitialSelectionReserve = 262'144;

    bool isCancellationRequested(const std::atomic_bool* flag)
    {
        return flag && flag->load(std::memory_order_relaxed);
    }

    bool isFinitePoint(const QVector3D& point)
    {
        return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
    }

    QVector3D cloudPoint(const SceneRenderCloud& cloud, std::size_t index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        return QVector3D(cloud.points()(row, 0), cloud.points()(row, 1), cloud.points()(row, 2));
    }

    int pointChunkCellCoordinate(float value, float minimum, float maximum, int gridDimension)
    {
        const float extent = maximum - minimum;
        if (!std::isfinite(value) || !std::isfinite(extent) || extent <= 1.0e-20f)
        {
            return 0;
        }
        const float normalized = std::clamp((value - minimum) / extent, 0.0f, 1.0f);
        return std::min(gridDimension - 1, static_cast<int>(normalized * gridDimension));
    }

    bool chunkIsVisible(const PointRenderChunkPreparation& chunk,
                        const QMatrix4x4& clipMatrix,
                        const QSize& viewportSize,
                        double* projectedAreaPixels)
    {
        if (projectedAreaPixels)
        {
            *projectedAreaPixels = 0.0;
        }
        if (chunk.pointCount <= 0 || chunk.sourceIndices.size() != chunk.pointCount || viewportSize.isEmpty())
        {
            return false;
        }

        const QVector3D minimum = chunk.aabbMinimum;
        const QVector3D maximum = chunk.aabbMaximum;
        const QVector<QVector3D> corners = {{minimum.x(), minimum.y(), minimum.z()},
                                            {maximum.x(), minimum.y(), minimum.z()},
                                            {minimum.x(), maximum.y(), minimum.z()},
                                            {maximum.x(), maximum.y(), minimum.z()},
                                            {minimum.x(), minimum.y(), maximum.z()},
                                            {maximum.x(), minimum.y(), maximum.z()},
                                            {minimum.x(), maximum.y(), maximum.z()},
                                            {maximum.x(), maximum.y(), maximum.z()}};

        QVector<QVector4D> clipCorners;
        clipCorners.reserve(corners.size());
        for (const QVector3D& corner : corners)
        {
            clipCorners.push_back(clipMatrix * QVector4D(corner, 1.0f));
        }

        const auto allOutside = [&clipCorners](const auto& predicate)
        { return std::all_of(clipCorners.cbegin(), clipCorners.cend(), predicate); };
        constexpr float epsilon = 1.0e-6f;
        if (allOutside([epsilon](const QVector4D& value) { return value.w() <= epsilon; }) ||
            allOutside([](const QVector4D& value) { return value.x() < -value.w(); }) ||
            allOutside([](const QVector4D& value) { return value.x() > value.w(); }) ||
            allOutside([](const QVector4D& value) { return value.y() < -value.w(); }) ||
            allOutside([](const QVector4D& value) { return value.y() > value.w(); }) ||
            allOutside([](const QVector4D& value) { return value.z() > value.w(); }))
        {
            return false;
        }

        float minimumNdcX = 1.0f;
        float minimumNdcY = 1.0f;
        float maximumNdcX = -1.0f;
        float maximumNdcY = -1.0f;
        bool hasProjectedCorner = false;
        for (const QVector4D& corner : std::as_const(clipCorners))
        {
            if (!std::isfinite(corner.w()) || corner.w() <= epsilon)
            {
                continue;
            }
            const float x = std::clamp(corner.x() / corner.w(), -1.0f, 1.0f);
            const float y = std::clamp(corner.y() / corner.w(), -1.0f, 1.0f);
            minimumNdcX = std::min(minimumNdcX, x);
            minimumNdcY = std::min(minimumNdcY, y);
            maximumNdcX = std::max(maximumNdcX, x);
            maximumNdcY = std::max(maximumNdcY, y);
            hasProjectedCorner = true;
        }
        if (projectedAreaPixels)
        {
            if (!hasProjectedCorner)
            {
                *projectedAreaPixels =
                    static_cast<double>(viewportSize.width()) * static_cast<double>(viewportSize.height());
            }
            else
            {
                const double widthPixels =
                    std::max(1.0, static_cast<double>(maximumNdcX - minimumNdcX) * 0.5 * viewportSize.width());
                const double heightPixels =
                    std::max(1.0, static_cast<double>(maximumNdcY - minimumNdcY) * 0.5 * viewportSize.height());
                *projectedAreaPixels = widthPixels * heightPixels;
            }
        }
        return true;
    }

} // namespace

CloudSpatialSummary prepareCloudSpatialSummary(const SceneRenderCloud& cloud, const std::atomic_bool* cancellationFlag)
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
        axis_samples.reserve(static_cast<qsizetype>(std::min(cloud.size(), kMaximumAxisSamples + 1)));
    }

    std::size_t valid_count = 0;
    std::size_t axis_stride = 1;
    double minimum_elevation = std::numeric_limits<double>::infinity();
    double maximum_elevation = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
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
                for (qsizetype read_index = 0; read_index < axis_samples.size(); read_index += 2)
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
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(center_z) ||
        std::abs(center_x) > maximum_float || std::abs(center_y) > maximum_float || std::abs(center_z) > maximum_float)
    {
        return result;
    }
    result.center = QVector3D(static_cast<float>(center_x), static_cast<float>(center_y), static_cast<float>(center_z));
    result.sourcePointCount = cloud.size();
    result.validPointCount = valid_count;
    result.elevationRange = {minimum_elevation, maximum_elevation};
    const std::size_t distance_stride =
        std::max<std::size_t>(1, (valid_count + kMaximumP95Samples - 1) / kMaximumP95Samples);
    std::vector<float> distances;
    distances.reserve(std::min(cloud.size(), kMaximumP95Samples));

    QVector3D local_min(
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    QVector3D local_max(std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest());
    const auto axes = needs_oriented_box ? xjw::gui::camera_scene::pointCloudPrincipalAxes(axis_samples)
                                         : xjw::gui::camera_scene::PointCloudPrincipalAxes{};
    std::size_t valid_index = 0;
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
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
        const std::size_t p95 =
            std::min(distances.size() - 1, static_cast<std::size_t>(static_cast<double>(distances.size()) * 0.95));
        std::nth_element(distances.begin(), distances.begin() + p95, distances.end());
        result.p95Radius = std::max(1.0e-4f, distances[p95] * 1.15f);
    }
    if (axes.valid)
    {
        result.boxLineVertices = xjw::gui::camera_scene::orientedBoundingBoxLineVertices(axes, local_min, local_max);
    }
    if (result.boxLineVertices.isEmpty())
    {
        result.boxLineVertices =
            xjw::gui::camera_scene::axisAlignedBoundingBoxLineVertices(result.aabbMinimum, result.aabbMaximum);
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.valid = true;
    return result;
}

QByteArray preparePointScalarData(std::size_t pointCount,
                                  const QVector<int>& imageCounts,
                                  xjw::gui::tie_points::ScalarRange* range,
                                  const std::atomic_bool* cancellationFlag)
{
    if (range)
    {
        *range = {};
    }
    if (pointCount == 0 || pointCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(float) ||
        isCancellationRequested(cancellationFlag))
    {
        return {};
    }

    QByteArray data(static_cast<int>(pointCount * sizeof(float)), Qt::Uninitialized);
    float* output = reinterpret_cast<float*>(data.data());
    const bool valid_counts = imageCounts.size() == static_cast<qsizetype>(pointCount);
    int minimum_count = std::numeric_limits<int>::max();
    int maximum_count = std::numeric_limits<int>::lowest();
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const int count = valid_counts ? imageCounts.at(static_cast<qsizetype>(index)) : 0;
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

PointRenderPreparation preparePointRenderData(const SceneRenderCloud& cloud,
                                              const QVector<int>& imageCounts,
                                              const std::atomic_bool* cancellationFlag)
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
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / (9 * int(sizeof(float)))))
    {
        return result;
    }

    constexpr int stride_floats = 9;
    constexpr float color_scale = 1.0f / 255.0f;
    result.vertexData.resize(static_cast<qsizetype>(cloud.size()) * stride_floats *
                             static_cast<qsizetype>(sizeof(float)));
    result.scalarData.resize(static_cast<qsizetype>(cloud.size()) * static_cast<qsizetype>(sizeof(float)));
    float* output = reinterpret_cast<float*>(result.vertexData.data());
    float* scalar_output = reinterpret_cast<float*>(result.scalarData.data());
    const bool has_colors = cloud.hasColors();
    const bool has_normals = cloud.hasNormals();
    const bool has_image_counts = imageCounts.size() == static_cast<qsizetype>(cloud.size());
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const auto row = static_cast<plamatrix::Index>(index);
        float* vertex = output + index * stride_floats;
        vertex[0] = cloud.points()(row, 0);
        vertex[1] = cloud.points()(row, 1);
        vertex[2] = cloud.points()(row, 2);
        vertex[3] = has_normals ? cloud.normals()->getValue(row, 0) : 0.0f;
        vertex[4] = has_normals ? cloud.normals()->getValue(row, 1) : 0.0f;
        vertex[5] = has_normals ? cloud.normals()->getValue(row, 2) : 0.0f;
        vertex[6] = has_colors ? cloud.colors()->getValue(row, 0) * color_scale : 0.45f;
        vertex[7] = has_colors ? cloud.colors()->getValue(row, 1) * color_scale : 0.45f;
        vertex[8] = has_colors ? cloud.colors()->getValue(row, 2) * color_scale : 0.50f;
        scalar_output[index] =
            has_image_counts ? static_cast<float>(imageCounts.at(static_cast<qsizetype>(index))) : 0.0f;
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.pointCount = static_cast<int>(cloud.size());
    result.chunks = preparePointRenderChunks(result.vertexData,
                                             stride_floats * int(sizeof(float)),
                                             result.scalarData,
                                             result.spatialSummary,
                                             262'144,
                                             524'288,
                                             cancellationFlag);
    return result;
}

QVector<PointRenderChunkPreparation> preparePointRenderChunks(const QByteArray& vertexData,
                                                              int strideBytes,
                                                              const QByteArray& scalarData,
                                                              const CloudSpatialSummary& spatialSummary,
                                                              int maximumPointsPerChunk,
                                                              int minimumPointCountForChunking,
                                                              const std::atomic_bool* cancellationFlag)
{
    QVector<PointRenderChunkPreparation> chunks;
    if (!spatialSummary.valid || strideBytes < 3 * int(sizeof(float)) || strideBytes % int(sizeof(float)) != 0 ||
        vertexData.isEmpty() || vertexData.size() % strideBytes != 0 || maximumPointsPerChunk <= 0 ||
        minimumPointCountForChunking <= 0 || isCancellationRequested(cancellationFlag))
    {
        return chunks;
    }

    const qsizetype pointCount = vertexData.size() / strideBytes;
    if (pointCount < minimumPointCountForChunking ||
        pointCount > static_cast<qsizetype>(std::numeric_limits<PointVertexIndex>::max()))
    {
        return chunks;
    }

    const int desiredChunkCount =
        std::max(1, static_cast<int>((pointCount + maximumPointsPerChunk - 1) / maximumPointsPerChunk));
    const int gridDimension =
        std::max(1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(desiredChunkCount)))));
    const int cellCount = gridDimension * gridDimension * gridDimension;
    std::vector<std::vector<PointVertexIndex>> cellIndices(static_cast<std::size_t>(cellCount));
    const float* vertices = reinterpret_cast<const float*>(vertexData.constData());
    const int strideFloats = strideBytes / int(sizeof(float));
    for (qsizetype index = 0; index < pointCount; ++index)
    {
        if (static_cast<std::size_t>(index) % kCancellationCheckInterval == 0 &&
            isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const float* point = vertices + index * strideFloats;
        const int x = pointChunkCellCoordinate(
            point[0], spatialSummary.aabbMinimum.x(), spatialSummary.aabbMaximum.x(), gridDimension);
        const int y = pointChunkCellCoordinate(
            point[1], spatialSummary.aabbMinimum.y(), spatialSummary.aabbMaximum.y(), gridDimension);
        const int z = pointChunkCellCoordinate(
            point[2], spatialSummary.aabbMinimum.z(), spatialSummary.aabbMaximum.z(), gridDimension);
        const int cellIndex = x + gridDimension * (y + gridDimension * z);
        cellIndices[static_cast<std::size_t>(cellIndex)].push_back(static_cast<PointVertexIndex>(index));
    }

    const bool hasScalars = scalarData.size() == pointCount * int(sizeof(float));
    const float* scalars = hasScalars ? reinterpret_cast<const float*>(scalarData.constData()) : nullptr;
    for (int cellIndex = 0; cellIndex < cellCount; ++cellIndex)
    {
        std::vector<PointVertexIndex>& indices = cellIndices[static_cast<std::size_t>(cellIndex)];
        if (indices.empty())
        {
            continue;
        }
        if (isCancellationRequested(cancellationFlag))
        {
            return {};
        }

        // 用与元素个数互质的步长重排。这样 LOD 绘制连续前缀时仍能覆盖
        // 整个空间块，而不是只显示文件中相邻的一小段点。
        const std::size_t count = indices.size();
        std::size_t step = count > 1 ? count / 2 + 1 : 1;
        while (count > 1 && std::gcd(step, count) != 1)
        {
            ++step;
        }
        const std::size_t offset = count > 0 ? (static_cast<std::size_t>(cellIndex) * 2'654'435'761ULL) % count : 0;

        for (std::size_t groupStart = 0; groupStart < count;
             groupStart += static_cast<std::size_t>(maximumPointsPerChunk))
        {
            const std::size_t groupCount =
                std::min(static_cast<std::size_t>(maximumPointsPerChunk), count - groupStart);
            PointRenderChunkPreparation chunk;
            chunk.pointCount = static_cast<int>(groupCount);
            chunk.vertexData.resize(static_cast<qsizetype>(groupCount) * strideBytes);
            chunk.scalarData.resize(static_cast<qsizetype>(groupCount) * int(sizeof(float)));
            chunk.sourceIndices.resize(static_cast<qsizetype>(groupCount));
            char* chunkVertices = chunk.vertexData.data();
            float* chunkScalars = reinterpret_cast<float*>(chunk.scalarData.data());
            for (std::size_t destinationIndex = 0; destinationIndex < groupCount; ++destinationIndex)
            {
                const std::size_t sequenceIndex = groupStart + destinationIndex;
                const std::size_t permuted = (offset + sequenceIndex * step) % count;
                const PointVertexIndex sourceIndex = indices[permuted];
                chunk.sourceIndices[static_cast<qsizetype>(destinationIndex)] = sourceIndex;
                std::memcpy(chunkVertices + static_cast<qsizetype>(destinationIndex) * strideBytes,
                            vertexData.constData() + static_cast<qsizetype>(sourceIndex) * strideBytes,
                            static_cast<std::size_t>(strideBytes));
                chunkScalars[destinationIndex] = scalars ? scalars[sourceIndex] : 0.0f;

                const float* point = vertices + static_cast<std::size_t>(sourceIndex) * strideFloats;
                const QVector3D position(point[0], point[1], point[2]);
                if (destinationIndex == 0)
                {
                    chunk.aabbMinimum = position;
                    chunk.aabbMaximum = position;
                }
                else
                {
                    chunk.aabbMinimum.setX(std::min(chunk.aabbMinimum.x(), position.x()));
                    chunk.aabbMinimum.setY(std::min(chunk.aabbMinimum.y(), position.y()));
                    chunk.aabbMinimum.setZ(std::min(chunk.aabbMinimum.z(), position.z()));
                    chunk.aabbMaximum.setX(std::max(chunk.aabbMaximum.x(), position.x()));
                    chunk.aabbMaximum.setY(std::max(chunk.aabbMaximum.y(), position.y()));
                    chunk.aabbMaximum.setZ(std::max(chunk.aabbMaximum.z(), position.z()));
                }
            }
            chunk.center = (chunk.aabbMinimum + chunk.aabbMaximum) * 0.5f;
            chunk.radius = (chunk.aabbMaximum - chunk.center).length();
            chunks.push_back(std::move(chunk));
        }
    }
    return chunks;
}

PointRenderPlan planPointRenderChunks(const QVector<PointRenderChunkPreparation>& chunks,
                                      const QMatrix4x4& clipMatrix,
                                      const QSize& viewportSize,
                                      float pointDiameterPixels,
                                      qint64 maximumVisiblePoints)
{
    PointRenderPlan plan;
    plan.drawCounts.resize(chunks.size());
    if (chunks.isEmpty() || viewportSize.isEmpty() || maximumVisiblePoints <= 0)
    {
        return plan;
    }

    const double pointArea =
        std::max(1.0, static_cast<double>(pointDiameterPixels) * static_cast<double>(pointDiameterPixels));
    qint64 fullyVisiblePoints = 0;
    QVector<double> projectedAreas(chunks.size(), 0.0);
    QVector<bool> visibleChunks(chunks.size(), false);
    for (qsizetype index = 0; index < chunks.size(); ++index)
    {
        const PointRenderChunkPreparation& chunk = chunks.at(index);
        if (!chunkIsVisible(chunk, clipMatrix, viewportSize, &projectedAreas[index]))
        {
            continue;
        }
        visibleChunks[index] = true;
        fullyVisiblePoints += chunk.pointCount;
        ++plan.visibleChunkCount;
    }

    // 稀疏点云在 GPU 预算以内时必须完整绘制。此前即使总点数远低于
    // maximumVisiblePoints，也会按投影面积抽取每个空间块的前缀；斜视角下
    // 低密度结构可能因此整片消失。只有真正超过预算时才启用 LOD。
    if (fullyVisiblePoints <= maximumVisiblePoints)
    {
        for (qsizetype index = 0; index < chunks.size(); ++index)
        {
            if (visibleChunks.at(index))
            {
                plan.drawCounts[index] = chunks.at(index).pointCount;
            }
        }
        plan.visiblePointCount = fullyVisiblePoints;
        return plan;
    }

    qint64 requestedPoints = 0;
    for (qsizetype index = 0; index < chunks.size(); ++index)
    {
        const PointRenderChunkPreparation& chunk = chunks.at(index);
        if (!visibleChunks.at(index))
        {
            continue;
        }
        const double minimumSampleCount = std::min(64.0, double(chunk.pointCount));
        const int areaDrivenCount = static_cast<int>(std::clamp(std::ceil(projectedAreas.at(index) / pointArea),
                                                                minimumSampleCount,
                                                                static_cast<double>(chunk.pointCount)));
        plan.drawCounts[index] = areaDrivenCount;
        requestedPoints += areaDrivenCount;
    }

    if (requestedPoints > maximumVisiblePoints)
    {
        const double scale = static_cast<double>(maximumVisiblePoints) / static_cast<double>(requestedPoints);
        requestedPoints = 0;
        for (int& drawCount : plan.drawCounts)
        {
            if (drawCount <= 0)
            {
                continue;
            }
            drawCount = std::max(1, static_cast<int>(std::floor(drawCount * scale)));
            requestedPoints += drawCount;
        }
    }
    for (qsizetype index = 0; requestedPoints > maximumVisiblePoints && index < plan.drawCounts.size(); ++index)
    {
        int& drawCount = plan.drawCounts[index];
        const qint64 removable = std::min<qint64>(std::max(0, drawCount - 1), requestedPoints - maximumVisiblePoints);
        drawCount -= static_cast<int>(removable);
        requestedPoints -= removable;
    }
    plan.visiblePointCount = requestedPoints;
    return plan;
}

std::vector<PointVertexIndex> selectPointVertexIndices(const QByteArray& vertexData,
                                                       int strideBytes,
                                                       const QRect& screenRect,
                                                       const QMatrix4x4& clipMatrix,
                                                       const QSize& viewportSize,
                                                       const QPointF& sceneOffset,
                                                       const std::atomic_bool* cancellationFlag)
{
    ScreenSelectionRegion region;
    region.shape = ScreenSelectionShape::Rectangle;
    region.bounds = QRectF(screenRect.normalized());
    return selectPointVertexIndices(
        vertexData, strideBytes, region, clipMatrix, viewportSize, sceneOffset, cancellationFlag);
}

std::vector<PointVertexIndex> selectPointVertexIndices(const QByteArray& vertexData,
                                                       int strideBytes,
                                                       const ScreenSelectionRegion& region,
                                                       const QMatrix4x4& clipMatrix,
                                                       const QSize& viewportSize,
                                                       const QPointF& sceneOffset,
                                                       const std::atomic_bool* cancellationFlag)
{
    std::vector<PointVertexIndex> indices;
    const QRectF bounds = region.bounds.normalized();
    const bool polygon_valid = region.shape != ScreenSelectionShape::Polygon || region.polygon.size() >= 3;
    if (strideBytes < 3 * int(sizeof(float)) || strideBytes % int(sizeof(float)) != 0 || vertexData.isEmpty() ||
        vertexData.size() % strideBytes != 0 || bounds.width() < 3.0 || bounds.height() < 3.0 || !polygon_valid ||
        viewportSize.width() <= 0 || viewportSize.height() <= 0 || isCancellationRequested(cancellationFlag))
    {
        return indices;
    }

    const std::size_t point_count = static_cast<std::size_t>(vertexData.size() / strideBytes);
    if (point_count > static_cast<std::size_t>(std::numeric_limits<PointVertexIndex>::max()))
    {
        return indices;
    }
    const int stride_floats = strideBytes / int(sizeof(float));
    const float* vertices = reinterpret_cast<const float*>(vertexData.constData());
    indices.reserve(std::min(point_count / 8, kMaximumInitialSelectionReserve));
    for (std::size_t index = 0; index < point_count; ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const float* vertex = vertices + index * static_cast<std::size_t>(stride_floats);
        const QVector4D clip = clipMatrix * QVector4D(vertex[0], vertex[1], vertex[2], 1.0f);
        if (!std::isfinite(clip.x()) || !std::isfinite(clip.y()) || !std::isfinite(clip.z()) ||
            !std::isfinite(clip.w()) || clip.w() <= 1.0e-6f || clip.z() < -clip.w() || clip.z() > clip.w())
        {
            continue;
        }
        const QPointF screen_point((clip.x() / clip.w() * 0.5f + 0.5f) * viewportSize.width() + sceneOffset.x(),
                                   (1.0f - (clip.y() / clip.w() * 0.5f + 0.5f)) * viewportSize.height() +
                                       sceneOffset.y());
        bool selected = bounds.contains(screen_point);
        if (selected && region.shape == ScreenSelectionShape::Ellipse)
        {
            const qreal radius_x = bounds.width() * 0.5;
            const qreal radius_y = bounds.height() * 0.5;
            const qreal normalized_x = (screen_point.x() - bounds.center().x()) / radius_x;
            const qreal normalized_y = (screen_point.y() - bounds.center().y()) / radius_y;
            selected = normalized_x * normalized_x + normalized_y * normalized_y <= 1.0;
        }
        else if (selected && region.shape == ScreenSelectionShape::Polygon)
        {
            selected = region.polygon.containsPoint(screen_point, Qt::OddEvenFill);
        }
        if (selected)
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

PointSelectionPreparation preparePointSelection(const QByteArray& vertexData,
                                                int strideBytes,
                                                const QByteArray& scalarData,
                                                const QRect& screenRect,
                                                const QMatrix4x4& clipMatrix,
                                                const QSize& viewportSize,
                                                const QPointF& sceneOffset,
                                                std::size_t maximumCompactPointCount,
                                                const std::atomic_bool* cancellationFlag)
{
    ScreenSelectionRegion region;
    region.shape = ScreenSelectionShape::Rectangle;
    region.bounds = QRectF(screenRect.normalized());
    return preparePointSelection(vertexData,
                                 strideBytes,
                                 scalarData,
                                 region,
                                 clipMatrix,
                                 viewportSize,
                                 sceneOffset,
                                 maximumCompactPointCount,
                                 cancellationFlag);
}

PointSelectionPreparation preparePointSelection(const QByteArray& vertexData,
                                                int strideBytes,
                                                const QByteArray& scalarData,
                                                const ScreenSelectionRegion& region,
                                                const QMatrix4x4& clipMatrix,
                                                const QSize& viewportSize,
                                                const QPointF& sceneOffset,
                                                std::size_t maximumCompactPointCount,
                                                const std::atomic_bool* cancellationFlag)
{
    PointSelectionPreparation result;
    result.indices = selectPointVertexIndices(
        vertexData, strideBytes, region, clipMatrix, viewportSize, sceneOffset, cancellationFlag);
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    const std::size_t compact_count = result.indices.size();
    const std::size_t maximum_byte_array_size = static_cast<std::size_t>(std::numeric_limits<qsizetype>::max());
    if (compact_count == 0 || compact_count > maximumCompactPointCount ||
        compact_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        compact_count > maximum_byte_array_size / static_cast<std::size_t>(strideBytes) ||
        compact_count > maximum_byte_array_size / sizeof(float))
    {
        return result;
    }

    const std::size_t point_count = static_cast<std::size_t>(vertexData.size() / strideBytes);
    const bool has_scalars = scalarData.size() == static_cast<qsizetype>(point_count * sizeof(float));
    result.pointCount = static_cast<int>(result.indices.size());
    result.vertexData.resize(static_cast<qsizetype>(result.pointCount) * strideBytes);
    result.scalarData.resize(static_cast<qsizetype>(result.pointCount) * static_cast<qsizetype>(sizeof(float)));

    char* compact_vertices = result.vertexData.data();
    float* compact_scalars = reinterpret_cast<float*>(result.scalarData.data());
    const char* source_vertices = vertexData.constData();
    const float* source_scalars = has_scalars ? reinterpret_cast<const float*>(scalarData.constData()) : nullptr;
    for (int compact_index = 0; compact_index < result.pointCount; ++compact_index)
    {
        if (static_cast<std::size_t>(compact_index) % kCancellationCheckInterval == 0 &&
            isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const std::size_t source_index =
            static_cast<std::size_t>(result.indices[static_cast<std::size_t>(compact_index)]);
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

PointSelectionPreparation prepareIndexedPointSelection(const QByteArray& vertexData,
                                                       int strideBytes,
                                                       const QByteArray& scalarData,
                                                       const std::vector<PointVertexIndex>& indices,
                                                       const std::atomic_bool* cancellationFlag)
{
    PointSelectionPreparation result;
    if (indices.empty())
    {
        return result;
    }
    if (strideBytes < 3 * int(sizeof(float)) || strideBytes % int(sizeof(float)) != 0 || vertexData.isEmpty() ||
        vertexData.size() % strideBytes != 0 ||
        indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        isCancellationRequested(cancellationFlag))
    {
        return result;
    }

    const std::size_t point_count = static_cast<std::size_t>(vertexData.size() / strideBytes);
    const std::size_t compact_count = indices.size();
    const std::size_t maximum_byte_array_size = static_cast<std::size_t>(std::numeric_limits<qsizetype>::max());
    if (compact_count > maximum_byte_array_size / static_cast<std::size_t>(strideBytes) ||
        compact_count > maximum_byte_array_size / sizeof(float))
    {
        return result;
    }
    for (std::size_t index = 0; index < compact_count; ++index)
    {
        if (index % kCancellationCheckInterval == 0 && isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        if (indices[index] >= point_count || (index > 0 && indices[index - 1] >= indices[index]))
        {
            return {};
        }
    }

    const bool has_scalars = scalarData.size() == static_cast<qsizetype>(point_count * sizeof(float));
    result.indices = indices;
    result.pointCount = static_cast<int>(compact_count);
    result.vertexData.resize(static_cast<qsizetype>(result.pointCount) * strideBytes);
    result.scalarData.resize(static_cast<qsizetype>(result.pointCount) * int(sizeof(float)));
    char* compact_vertices = result.vertexData.data();
    float* compact_scalars = reinterpret_cast<float*>(result.scalarData.data());
    const char* source_vertices = vertexData.constData();
    const float* source_scalars = has_scalars ? reinterpret_cast<const float*>(scalarData.constData()) : nullptr;
    for (int compact_index = 0; compact_index < result.pointCount; ++compact_index)
    {
        if (static_cast<std::size_t>(compact_index) % kCancellationCheckInterval == 0 &&
            isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const std::size_t source_index = static_cast<std::size_t>(indices[static_cast<std::size_t>(compact_index)]);
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
