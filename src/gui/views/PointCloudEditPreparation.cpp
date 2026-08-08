#include "PointCloudEditPreparation.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace
{

using Index = plamatrix::Index;

bool isCancelled(const std::atomic_bool *cancellationFlag)
{
    return cancellationFlag
        && cancellationFlag->load(std::memory_order_relaxed);
}

std::shared_ptr<SceneRenderCloud> copyRows(
    const SceneRenderCloud &source,
    const std::vector<Index> &rows,
    const std::atomic_bool *cancellationFlag)
{
    if (isCancelled(cancellationFlag))
    {
        return {};
    }
    auto output = std::make_shared<SceneRenderCloud>(rows.size());
    for (std::size_t target = 0; target < rows.size(); ++target)
    {
        if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
        {
            return {};
        }
        for (int column = 0; column < 3; ++column)
        {
            output->points()(static_cast<Index>(target), column) =
                source.points()(rows[target], column);
        }
    }
    if (source.hasColors())
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> values(rows.size(), 3);
        for (std::size_t target = 0; target < rows.size(); ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            for (int column = 0; column < 3; ++column)
            {
                values(static_cast<Index>(target), column) =
                    source.colors()->getValue(rows[target], column);
            }
        }
        output->setColors(std::move(values));
    }
    if (source.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> values(rows.size(), 3);
        for (std::size_t target = 0; target < rows.size(); ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            for (int column = 0; column < 3; ++column)
            {
                values(static_cast<Index>(target), column) =
                    source.normals()->getValue(rows[target], column);
            }
        }
        output->setNormals(std::move(values));
    }
    if (source.hasIntensities())
    {
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> values(rows.size(), 1);
        for (std::size_t target = 0; target < rows.size(); ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            values(static_cast<Index>(target), 0) =
                source.intensities()->getValue(rows[target], 0);
        }
        output->setIntensities(std::move(values));
    }
    if (source.hasScalarFields())
    {
        const int columns = source.scalarFields()->cols();
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> values(rows.size(), columns);
        for (std::size_t target = 0; target < rows.size(); ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            for (int column = 0; column < columns; ++column)
            {
                values(static_cast<Index>(target), column) =
                    source.scalarFields()->getValue(rows[target], column);
            }
        }
        output->setScalarFields(source.scalarFieldNames(), std::move(values));
    }
    if (source.hasPointAlignedTextureCoords())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> values(rows.size(), 2);
        for (std::size_t target = 0; target < rows.size(); ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            for (int column = 0; column < 2; ++column)
            {
                values(static_cast<Index>(target), column) =
                    source.textureCoords()->getValue(rows[target], column);
            }
        }
        output->setTextureCoords(std::move(values));
    }
    output->setMaterialLibraryFile(source.materialLibraryFile());
    output->setTextureImageFile(source.textureImageFile());
    return output;
}

QVector<int> copyCounts(const QVector<int> &source,
                        const std::vector<Index> &rows,
                        std::size_t expectedSize,
                        const std::atomic_bool *cancellationFlag)
{
    if (source.size() != static_cast<qsizetype>(expectedSize))
    {
        return {};
    }
    QVector<int> output;
    output.reserve(static_cast<qsizetype>(rows.size()));
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        if ((index & 4095U) == 0U && isCancelled(cancellationFlag))
        {
            return {};
        }
        output.push_back(source.at(static_cast<qsizetype>(rows[index])));
    }
    return output;
}

} // namespace

PointCloudEditResult filterPointCloudWithDelta(
    std::shared_ptr<SceneRenderCloud> source,
    std::vector<PointVertexIndex> removedIndices,
    const QVector<int> &imageCounts,
    const std::atomic_bool *cancellationFlag)
{
    PointCloudEditResult result;
    if (isCancelled(cancellationFlag)
        || !source || source->hasFaces() || source->size() == 0
        || source->size() > static_cast<std::size_t>(
            std::numeric_limits<PointVertexIndex>::max()))
    {
        return result;
    }

    removedIndices.erase(
        std::remove_if(removedIndices.begin(), removedIndices.end(),
            [point_count = source->size()](PointVertexIndex index)
            {
                return index >= point_count;
            }),
        removedIndices.end());
    std::sort(removedIndices.begin(), removedIndices.end());
    removedIndices.erase(std::unique(removedIndices.begin(), removedIndices.end()),
                         removedIndices.end());
    if (removedIndices.empty() || isCancelled(cancellationFlag))
    {
        return result;
    }

    std::vector<Index> removed_rows;
    removed_rows.reserve(removedIndices.size());
    std::vector<Index> kept_rows;
    kept_rows.reserve(source->size() - removedIndices.size());
    std::size_t removed_cursor = 0;
    for (std::size_t index = 0; index < source->size(); ++index)
    {
        if ((index & 4095U) == 0U && isCancelled(cancellationFlag))
        {
            return {};
        }
        if (removed_cursor < removedIndices.size()
            && removedIndices[removed_cursor] == index)
        {
            removed_rows.push_back(static_cast<Index>(index));
            ++removed_cursor;
        }
        else
        {
            kept_rows.push_back(static_cast<Index>(index));
        }
    }

    auto removed_cloud = copyRows(*source, removed_rows, cancellationFlag);
    auto filtered_cloud = copyRows(*source, kept_rows, cancellationFlag);
    if (!removed_cloud || !filtered_cloud || isCancelled(cancellationFlag))
    {
        return {};
    }
    QVector<int> removed_counts = copyCounts(
        imageCounts, removed_rows, source->size(), cancellationFlag);
    QVector<int> filtered_counts = copyCounts(
        imageCounts, kept_rows, source->size(), cancellationFlag);
    if (isCancelled(cancellationFlag))
    {
        return {};
    }
    PointRenderPreparation render_preparation = preparePointRenderData(
        *filtered_cloud, filtered_counts, cancellationFlag);
    if ((filtered_cloud->size() > 0 && !render_preparation.isValid())
        || isCancelled(cancellationFlag))
    {
        return {};
    }

    result.undo.originalPointCount = source->size();
    result.undo.removedIndices = std::move(removedIndices);
    result.undo.removedCloud = std::move(removed_cloud);
    result.undo.removedImageCounts = std::move(removed_counts);
    result.cloud = std::move(filtered_cloud);
    result.imageCounts = std::move(filtered_counts);
    result.renderPreparation = std::move(render_preparation);
    return result;
}

PointCloudEditResult restorePointCloudFromDelta(
    std::shared_ptr<SceneRenderCloud> filtered,
    PointCloudEditDelta delta,
    const QVector<int> &filteredImageCounts,
    const std::atomic_bool *cancellationFlag)
{
    PointCloudEditResult result;
    if (isCancelled(cancellationFlag)
        || !filtered || !delta.isValid()
        || filtered->size() + delta.removedIndices.size() != delta.originalPointCount)
    {
        return result;
    }

    auto restored = std::make_shared<SceneRenderCloud>(delta.originalPointCount);
    const bool has_colors = filtered->hasColors() && delta.removedCloud->hasColors();
    const bool has_normals = filtered->hasNormals() && delta.removedCloud->hasNormals();
    const bool has_intensities = filtered->hasIntensities() && delta.removedCloud->hasIntensities();
    const bool has_scalars = filtered->hasScalarFields() && delta.removedCloud->hasScalarFields()
        && filtered->scalarFieldNames() == delta.removedCloud->scalarFieldNames();
    const bool has_texture = filtered->hasPointAlignedTextureCoords()
        && delta.removedCloud->hasPointAlignedTextureCoords();
    const int scalar_columns = has_scalars ? filtered->scalarFields()->cols() : 0;

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        delta.originalPointCount, has_colors ? 3 : 0);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(
        delta.originalPointCount, has_normals ? 3 : 0);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(
        delta.originalPointCount, has_intensities ? 1 : 0);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> scalars(
        delta.originalPointCount, scalar_columns);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture(
        delta.originalPointCount, has_texture ? 2 : 0);

    std::size_t filtered_cursor = 0;
    std::size_t removed_cursor = 0;
    for (std::size_t target = 0; target < delta.originalPointCount; ++target)
    {
        if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
        {
            return {};
        }
        const bool from_removed = removed_cursor < delta.removedIndices.size()
            && delta.removedIndices[removed_cursor] == target;
        const SceneRenderCloud &source = from_removed ? *delta.removedCloud : *filtered;
        const Index source_row = static_cast<Index>(
            from_removed ? removed_cursor++ : filtered_cursor++);
        const Index target_row = static_cast<Index>(target);
        for (int column = 0; column < 3; ++column)
        {
            restored->points()(target_row, column) = source.points()(source_row, column);
            if (has_colors)
            {
                colors(target_row, column) = source.colors()->getValue(source_row, column);
            }
            if (has_normals)
            {
                normals(target_row, column) = source.normals()->getValue(source_row, column);
            }
        }
        if (has_intensities)
        {
            intensities(target_row, 0) = source.intensities()->getValue(source_row, 0);
        }
        for (int column = 0; column < scalar_columns; ++column)
        {
            scalars(target_row, column) = source.scalarFields()->getValue(source_row, column);
        }
        if (has_texture)
        {
            texture(target_row, 0) = source.textureCoords()->getValue(source_row, 0);
            texture(target_row, 1) = source.textureCoords()->getValue(source_row, 1);
        }
    }
    if (has_colors) restored->setColors(std::move(colors));
    if (has_normals) restored->setNormals(std::move(normals));
    if (has_intensities) restored->setIntensities(std::move(intensities));
    if (has_scalars) restored->setScalarFields(filtered->scalarFieldNames(), std::move(scalars));
    if (has_texture) restored->setTextureCoords(std::move(texture));
    restored->setMaterialLibraryFile(filtered->materialLibraryFile());
    restored->setTextureImageFile(filtered->textureImageFile());

    const bool has_counts = filteredImageCounts.size() == static_cast<qsizetype>(filtered->size())
        && delta.removedImageCounts.size()
            == static_cast<qsizetype>(delta.removedIndices.size());
    if (has_counts)
    {
        result.imageCounts.resize(static_cast<qsizetype>(delta.originalPointCount));
        filtered_cursor = 0;
        removed_cursor = 0;
        for (std::size_t target = 0; target < delta.originalPointCount; ++target)
        {
            if ((target & 4095U) == 0U && isCancelled(cancellationFlag))
            {
                return {};
            }
            const bool from_removed = removed_cursor < delta.removedIndices.size()
                && delta.removedIndices[removed_cursor] == target;
            result.imageCounts[static_cast<qsizetype>(target)] = from_removed
                ? delta.removedImageCounts.at(static_cast<qsizetype>(removed_cursor++))
                : filteredImageCounts.at(static_cast<qsizetype>(filtered_cursor++));
        }
    }
    PointRenderPreparation render_preparation = preparePointRenderData(
        *restored, result.imageCounts, cancellationFlag);
    if (!render_preparation.isValid() || isCancelled(cancellationFlag))
    {
        return {};
    }
    result.cloud = std::move(restored);
    result.renderPreparation = std::move(render_preparation);
    return result;
}
