#pragma once

#include "SceneGeometryPreparation.h"

#include <QVector>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

struct PointCloudEditDelta
{
    std::shared_ptr<SceneRenderCloud> removedCloud;
    std::vector<PointVertexIndex> removedIndices;
    QVector<int> removedImageCounts;
    std::size_t originalPointCount = 0;

    bool isValid() const
    {
        if (!removedCloud || removedIndices.empty()
            || removedCloud->size() != removedIndices.size()
            || originalPointCount < removedIndices.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < removedIndices.size(); ++index)
        {
            if (removedIndices[index] >= originalPointCount
                || (index > 0 && removedIndices[index - 1] >= removedIndices[index]))
            {
                return false;
            }
        }
        return true;
    }
};

struct PointCloudEditResult
{
    std::shared_ptr<SceneRenderCloud> cloud;
    QVector<int> imageCounts;
    PointRenderPreparation renderPreparation;
    PointCloudEditDelta undo;

    bool isValid() const
    {
        return cloud && renderPreparation.pointCount == static_cast<int>(cloud->size());
    }
};

PointCloudEditResult filterPointCloudWithDelta(
    std::shared_ptr<SceneRenderCloud> source,
    std::vector<PointVertexIndex> removedIndices,
    const QVector<int> &imageCounts = {},
    const std::atomic_bool *cancellationFlag = nullptr);

PointCloudEditResult restorePointCloudFromDelta(
    std::shared_ptr<SceneRenderCloud> filtered,
    PointCloudEditDelta delta,
    const QVector<int> &filteredImageCounts = {},
    const std::atomic_bool *cancellationFlag = nullptr);
