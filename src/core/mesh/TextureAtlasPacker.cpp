#include "TextureAtlasPacker.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace xjw::mesh
{
namespace
{

struct Shelf
{
    int y = 0;
    int height = 0;
    int nextX = 0;
    int remainingWidth = 0;
};

enum class TryPackStatus
{
    Packed,
    DoesNotFit,
    Cancelled
};

TryPackStatus tryPack(const QVector<TextureAtlasItem> &source,
                      int atlas_size,
                      int reserved_left,
                      float scale,
                      const std::function<bool()> &is_cancelled,
                      QVector<TextureAtlasItem> *packed)
{
    QVector<TextureAtlasItem> ordered = source;
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right)
    {
        if (left.requestedSize.height() != right.requestedSize.height())
        {
            return left.requestedSize.height() > right.requestedSize.height();
        }
        if (left.requestedSize.width() != right.requestedSize.width())
        {
            return left.requestedSize.width() > right.requestedSize.width();
        }
        const qint64 left_area =
            static_cast<qint64>(left.requestedSize.width()) * left.requestedSize.height();
        const qint64 right_area =
            static_cast<qint64>(right.requestedSize.width()) * right.requestedSize.height();
        return left_area != right_area ? left_area > right_area : left.id < right.id;
    });

    const int available_width = atlas_size - reserved_left;
    QVector<Shelf> shelves;
    shelves.reserve(std::min<qsizetype>(ordered.size(), atlas_size));
    std::set<std::pair<int, int>> shelves_by_remaining_width;
    int used_height = 0;
    for (int item_index = 0; item_index < ordered.size(); ++item_index)
    {
        if ((item_index % 1024 == 0) && is_cancelled && is_cancelled())
        {
            return TryPackStatus::Cancelled;
        }
        TextureAtlasItem &item = ordered[item_index];
        const int width = std::max(1, static_cast<int>(
            std::ceil(item.requestedSize.width() * scale)));
        const int height = std::max(1, static_cast<int>(
            std::ceil(item.requestedSize.height() * scale)));
        if (width > available_width || height > atlas_size)
        {
            return TryPackStatus::DoesNotFit;
        }

        const auto best = shelves_by_remaining_width.lower_bound({width, -1});
        int shelf_index = -1;
        if (best != shelves_by_remaining_width.end())
        {
            shelf_index = best->second;
            shelves_by_remaining_width.erase(best);
        }
        else
        {
            if (used_height + height > atlas_size)
            {
                return TryPackStatus::DoesNotFit;
            }
            shelf_index = shelves.size();
            shelves.push_back({used_height, height, reserved_left, available_width});
            used_height += height;
        }

        Shelf &shelf = shelves[shelf_index];
        item.packedRect = QRect(shelf.nextX, shelf.y, width, height);
        shelf.nextX += width;
        shelf.remainingWidth -= width;
        if (shelf.remainingWidth > 0)
        {
            shelves_by_remaining_width.emplace(
                shelf.remainingWidth, shelf_index);
        }
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right)
    {
        return left.id < right.id;
    });
    *packed = std::move(ordered);
    return TryPackStatus::Packed;
}

} // namespace

TextureAtlasPackingResult TextureAtlasPacker::pack(const QVector<TextureAtlasItem> &items,
                                                   int atlasSize,
                                                   int reservedLeft,
                                                   const std::function<bool()> &isCancelled,
                                                   const std::function<void(int)> &progressFn)
{
    TextureAtlasPackingResult result;
    if (items.isEmpty() || atlasSize <= 0 || reservedLeft < 0 || reservedLeft >= atlasSize)
    {
        return result;
    }

    qint64 requested_area = 0;
    for (const TextureAtlasItem &item : items)
    {
        if (item.id < 0 || item.requestedSize.isEmpty())
        {
            return result;
        }
        requested_area +=
            static_cast<qint64>(item.requestedSize.width()) * item.requestedSize.height();
    }
    const qint64 available_area =
        static_cast<qint64>(atlasSize - reservedLeft) * atlasSize;
    float high = std::min(1.0f, static_cast<float>(
        std::sqrt(static_cast<double>(available_area) /
                  std::max<qint64>(requested_area, 1)) * 0.96));
    float low = 0.01f;
    QVector<TextureAtlasItem> packed;
    constexpr int search_iterations = 14;
    constexpr int maximum_attempts = search_iterations + 3;
    int completed_attempts = 0;
    const auto try_pack = [&](float scale, QVector<TextureAtlasItem> *output)
    {
        if (progressFn)
        {
            progressFn(completed_attempts * 100 / maximum_attempts);
        }
        const TryPackStatus status = tryPack(
            items, atlasSize, reservedLeft, scale, isCancelled, output);
        ++completed_attempts;
        return status;
    };

    TryPackStatus status = try_pack(high, &packed);
    if (status == TryPackStatus::Cancelled)
    {
        result.cancelled = true;
        return result;
    }
    if (status == TryPackStatus::Packed)
    {
        low = high;
    }
    else
    {
        QVector<TextureAtlasItem> minimum_packed;
        status = try_pack(low, &minimum_packed);
        if (status == TryPackStatus::Cancelled)
        {
            result.cancelled = true;
            return result;
        }
        if (status != TryPackStatus::Packed)
        {
            return result;
        }
        packed = std::move(minimum_packed);
        for (int iteration = 0; iteration < search_iterations; ++iteration)
        {
            const float candidate = (low + high) * 0.5f;
            QVector<TextureAtlasItem> trial;
            status = try_pack(candidate, &trial);
            if (status == TryPackStatus::Cancelled)
            {
                result.cancelled = true;
                return result;
            }
            if (status == TryPackStatus::Packed)
            {
                low = candidate;
                packed = std::move(trial);
            }
            else
            {
                high = candidate;
            }
        }
    }
    if (packed.isEmpty())
    {
        return result;
    }

    result.ok = true;
    result.scale = low;
    QVector<TextureAtlasItem> final_packed;
    status = try_pack(low, &final_packed);
    if (status == TryPackStatus::Cancelled)
    {
        result.ok = false;
        result.cancelled = true;
        return result;
    }
    if (status == TryPackStatus::Packed)
    {
        packed = std::move(final_packed);
    }
    qint64 packed_area = 0;
    for (const TextureAtlasItem &item : packed)
    {
        packed_area += static_cast<qint64>(
            item.packedRect.width()) * item.packedRect.height();
    }
    result.items = std::move(packed);
    result.occupancy = static_cast<double>(packed_area) /
        static_cast<double>(std::max<qint64>(available_area, 1));
    if (progressFn)
    {
        progressFn(100);
    }
    return result;
}

} // namespace xjw::mesh
