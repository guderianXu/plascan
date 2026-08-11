#include "ObjPointPreviewPreparation.h"

#include <QVector3D>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace
{

constexpr std::size_t kCancellationCheckInterval = 4'096;

bool isCancellationRequested(const std::atomic_bool *flag)
{
    return flag && flag->load(std::memory_order_relaxed);
}

bool shouldCancel(const std::atomic_bool *flag, std::size_t index)
{
    return index % kCancellationCheckInterval == 0
        && isCancellationRequested(flag);
}

QVector3D normalizedVector(const QVector3D &vector)
{
    const float lengthSquared = vector.lengthSquared();
    if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-30f)
    {
        return QVector3D(0.0f, 0.0f, 1.0f);
    }
    return vector / std::sqrt(lengthSquared);
}

} // namespace

ObjRenderPreparation prepareObjPointPreview(
    const ObjRenderCloud &cloud,
    const std::atomic_bool *cancellationFlag,
    const ObjPrepareProgressCallback &progress,
    const ObjRenderPreparationLimits &limits)
{
    ObjRenderPreparation result;
    result.isPointPreview = true;
    result.hasVertexColors = cloud.hasColors();
    result.sourceVertexCount = cloud.size();
    result.sourceFaceCount = cloud.hasFaces()
        ? static_cast<std::size_t>(cloud.faces()->rows())
        : 0;
    const std::size_t previewCount = std::min(
        cloud.size(), limits.maximumPreviewPoints);
    result.vertexCount = static_cast<int>(previewCount);
    if (previewCount == 0)
    {
        return {};
    }

    double minimumElevation = std::numeric_limits<double>::infinity();
    double maximumElevation = -std::numeric_limits<double>::infinity();
    std::size_t written = 0;
    while (written < previewCount)
    {
        if (isCancellationRequested(cancellationFlag))
        {
            return {};
        }
        const std::size_t chunkCount = std::min(
            limits.previewPointsPerChunk, previewCount - written);
        ObjPointPreviewChunk chunk;
        chunk.vertexCount = static_cast<int>(chunkCount);
        chunk.vertexData.resize(
            static_cast<qsizetype>(chunkCount) * result.strideBytes);
        float *vertices = reinterpret_cast<float *>(chunk.vertexData.data());
        for (std::size_t localIndex = 0; localIndex < chunkCount; ++localIndex)
        {
            const std::size_t previewIndex = written + localIndex;
            if (shouldCancel(cancellationFlag, previewIndex))
            {
                return {};
            }
            const std::size_t sourceIndex = previewIndex * cloud.size() / previewCount;
            const auto row = static_cast<plamatrix::Index>(sourceIndex);
            float *vertex = vertices + localIndex * 9;
            vertex[0] = cloud.points()(row, 0);
            vertex[1] = cloud.points()(row, 1);
            vertex[2] = cloud.points()(row, 2);
            if (cloud.hasNormals())
            {
                const QVector3D normal = normalizedVector(QVector3D(
                    cloud.normals()->getValue(row, 0),
                    cloud.normals()->getValue(row, 1),
                    cloud.normals()->getValue(row, 2)));
                vertex[3] = normal.x();
                vertex[4] = normal.y();
                vertex[5] = normal.z();
            }
            else
            {
                vertex[3] = 0.0f;
                vertex[4] = 0.0f;
                vertex[5] = 1.0f;
            }
            if (cloud.hasColors())
            {
                vertex[6] = cloud.colors()->getValue(row, 0) / 255.0f;
                vertex[7] = cloud.colors()->getValue(row, 1) / 255.0f;
                vertex[8] = cloud.colors()->getValue(row, 2) / 255.0f;
            }
            else
            {
                vertex[6] = 239.0f / 255.0f;
                vertex[7] = 236.0f / 255.0f;
                vertex[8] = 224.0f / 255.0f;
            }
            if (std::isfinite(vertex[2]))
            {
                minimumElevation = std::min(minimumElevation, double(vertex[2]));
                maximumElevation = std::max(maximumElevation, double(vertex[2]));
            }
        }
        result.pointPreviewChunks.push_back(std::move(chunk));
        written += chunkCount;
        if (progress)
        {
            const int percent = 88 + static_cast<int>(
                10 * written / std::max<std::size_t>(1, previewCount));
            progress(percent, QStringLiteral("正在生成超大模型显示代理..."));
        }
    }
    result.elevationRange = {minimumElevation, maximumElevation};
    return result;
}
