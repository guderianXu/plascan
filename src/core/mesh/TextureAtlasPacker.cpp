#include "TextureAtlasPacker.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

struct Placement
{
    int freeIndex = -1;
    int shortSide = std::numeric_limits<int>::max();
    int longSide = std::numeric_limits<int>::max();
};

bool intersectsArea(const QRect &left, const QRect &right)
{
    return left.left() < right.right() + 1 &&
           left.right() + 1 > right.left() &&
           left.top() < right.bottom() + 1 &&
           left.bottom() + 1 > right.top();
}

void splitFreeRectangles(QVector<QRect> *free_rectangles, const QRect &used)
{
    QVector<QRect> next;
    next.reserve(free_rectangles->size() * 2);
    for (const QRect &free_rect : *free_rectangles)
    {
        if (!intersectsArea(free_rect, used))
        {
            next.push_back(free_rect);
            continue;
        }

        if (used.left() > free_rect.left())
        {
            next.push_back(QRect(free_rect.left(),
                                 free_rect.top(),
                                 used.left() - free_rect.left(),
                                 free_rect.height()));
        }
        if (used.right() < free_rect.right())
        {
            next.push_back(QRect(used.right() + 1,
                                 free_rect.top(),
                                 free_rect.right() - used.right(),
                                 free_rect.height()));
        }
        if (used.top() > free_rect.top())
        {
            next.push_back(QRect(free_rect.left(),
                                 free_rect.top(),
                                 free_rect.width(),
                                 used.top() - free_rect.top()));
        }
        if (used.bottom() < free_rect.bottom())
        {
            next.push_back(QRect(free_rect.left(),
                                 used.bottom() + 1,
                                 free_rect.width(),
                                 free_rect.bottom() - used.bottom()));
        }
    }

    for (int first = 0; first < next.size(); ++first)
    {
        if (next[first].isEmpty())
        {
            continue;
        }
        for (int second = 0; second < next.size(); ++second)
        {
            if (first == second || next[second].isEmpty())
            {
                continue;
            }
            if (next[second].contains(next[first]))
            {
                next[first] = QRect();
                break;
            }
        }
    }
    next.erase(std::remove_if(next.begin(), next.end(), [](const QRect &rect)
    {
        return rect.isEmpty();
    }), next.end());
    *free_rectangles = std::move(next);
}

bool tryPack(const QVector<TextureAtlasItem> &source,
             int atlas_size,
             int reserved_left,
             float scale,
             QVector<TextureAtlasItem> *packed)
{
    QVector<TextureAtlasItem> ordered = source;
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right)
    {
        const int left_max = std::max(left.requestedSize.width(), left.requestedSize.height());
        const int right_max = std::max(right.requestedSize.width(), right.requestedSize.height());
        if (left_max != right_max)
        {
            return left_max > right_max;
        }
        const qint64 left_area =
            static_cast<qint64>(left.requestedSize.width()) * left.requestedSize.height();
        const qint64 right_area =
            static_cast<qint64>(right.requestedSize.width()) * right.requestedSize.height();
        return left_area != right_area ? left_area > right_area : left.id < right.id;
    });

    QVector<QRect> free_rectangles{
        QRect(reserved_left, 0, atlas_size - reserved_left, atlas_size)
    };
    for (TextureAtlasItem &item : ordered)
    {
        const int width = std::max(1, static_cast<int>(
            std::ceil(item.requestedSize.width() * scale)));
        const int height = std::max(1, static_cast<int>(
            std::ceil(item.requestedSize.height() * scale)));
        Placement best;
        for (int index = 0; index < free_rectangles.size(); ++index)
        {
            const QRect &free_rect = free_rectangles[index];
            if (width > free_rect.width() || height > free_rect.height())
            {
                continue;
            }
            const int remaining_width = free_rect.width() - width;
            const int remaining_height = free_rect.height() - height;
            const int short_side = std::min(remaining_width, remaining_height);
            const int long_side = std::max(remaining_width, remaining_height);
            if (short_side < best.shortSide ||
                (short_side == best.shortSide && long_side < best.longSide))
            {
                best = {index, short_side, long_side};
            }
        }
        if (best.freeIndex < 0)
        {
            return false;
        }

        const QRect &selected = free_rectangles[best.freeIndex];
        item.packedRect = QRect(selected.topLeft(), QSize(width, height));
        splitFreeRectangles(&free_rectangles, item.packedRect);
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right)
    {
        return left.id < right.id;
    });
    *packed = std::move(ordered);
    return true;
}

} // namespace

TextureAtlasPackingResult TextureAtlasPacker::pack(const QVector<TextureAtlasItem> &items,
                                                   int atlasSize,
                                                   int reservedLeft)
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
    if (tryPack(items, atlasSize, reservedLeft, high, &packed))
    {
        low = high;
    }
    else
    {
        QVector<TextureAtlasItem> minimum_packed;
        if (!tryPack(items, atlasSize, reservedLeft, low, &minimum_packed))
        {
            return result;
        }
        packed = std::move(minimum_packed);
        for (int iteration = 0; iteration < 18; ++iteration)
        {
            const float candidate = (low + high) * 0.5f;
            QVector<TextureAtlasItem> trial;
            if (tryPack(items, atlasSize, reservedLeft, candidate, &trial))
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
    if (tryPack(items, atlasSize, reservedLeft, low, &final_packed))
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
    return result;
}

} // namespace xjw::mesh
