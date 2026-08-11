#include "TextureAtlasPackingInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace xjw::mesh::detail
{
namespace
{

constexpr int kMaximumFreeRectangles = 4096;
constexpr int kMaximumSplitRectangles = kMaximumFreeRectangles * 2;
constexpr std::uint64_t kMaximumOperations = 4'000'000;
constexpr std::uint64_t kCancellationCheckMask = 1023;

struct PackingRect
{
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    int right() const
    {
        return x + width;
    }

    int bottom() const
    {
        return y + height;
    }
};

struct Placement
{
    int freeIndex = -1;
    std::array<int, 7> score{};
};

qint64 itemArea(const TextureAtlasItem &item)
{
    return static_cast<qint64>(item.requestedSize.width()) *
        item.requestedSize.height();
}

bool intersects(const PackingRect &left, const PackingRect &right)
{
    return left.x < right.right() && left.right() > right.x &&
        left.y < right.bottom() && left.bottom() > right.y;
}

bool contains(const PackingRect &outer, const PackingRect &inner)
{
    return outer.x <= inner.x && outer.y <= inner.y &&
        outer.right() >= inner.right() && outer.bottom() >= inner.bottom();
}

TextureAtlasTryPackStatus accountWork(
    std::uint64_t *operations,
    std::uint64_t *remaining_operations,
    const std::function<bool()> &is_cancelled)
{
    if (*remaining_operations == 0)
    {
        return TextureAtlasTryPackStatus::TooComplex;
    }
    ++(*operations);
    --(*remaining_operations);
    if (((*operations) & kCancellationCheckMask) == 0 &&
        is_cancelled && is_cancelled())
    {
        return TextureAtlasTryPackStatus::Cancelled;
    }
    return *operations > kMaximumOperations
        ? TextureAtlasTryPackStatus::TooComplex
        : TextureAtlasTryPackStatus::Packed;
}

void appendIfValid(QVector<PackingRect> *rectangles, const PackingRect &rect)
{
    if (rect.width > 0 && rect.height > 0)
    {
        rectangles->push_back(rect);
    }
}

TextureAtlasTryPackStatus splitFreeRectangles(
    QVector<PackingRect> *free_rectangles,
    const PackingRect &used,
    std::uint64_t *operations,
    std::uint64_t *remaining_operations,
    const std::function<bool()> &is_cancelled)
{
    QVector<PackingRect> next;
    next.reserve(std::min(
        kMaximumSplitRectangles,
        static_cast<int>(free_rectangles->size()) * 2));
    for (const PackingRect &free_rect : *free_rectangles)
    {
        const TextureAtlasTryPackStatus work_status = accountWork(
            operations, remaining_operations, is_cancelled);
        if (work_status != TextureAtlasTryPackStatus::Packed)
        {
            return work_status;
        }
        if (!intersects(free_rect, used))
        {
            next.push_back(free_rect);
            continue;
        }

        appendIfValid(&next,
                      {free_rect.x,
                       free_rect.y,
                       used.x - free_rect.x,
                       free_rect.height});
        appendIfValid(&next,
                      {used.right(),
                       free_rect.y,
                       free_rect.right() - used.right(),
                       free_rect.height});
        appendIfValid(&next,
                      {free_rect.x,
                       free_rect.y,
                       free_rect.width,
                       used.y - free_rect.y});
        appendIfValid(&next,
                      {free_rect.x,
                       used.bottom(),
                       free_rect.width,
                       free_rect.bottom() - used.bottom()});
        if (next.size() > kMaximumSplitRectangles)
        {
            return TextureAtlasTryPackStatus::TooComplex;
        }
    }

    QVector<bool> removed(next.size(), false);
    for (int first = 0; first < next.size(); ++first)
    {
        if (removed[first])
        {
            continue;
        }
        for (int second = 0; second < next.size(); ++second)
        {
            if (first == second || removed[second])
            {
                continue;
            }
            const TextureAtlasTryPackStatus work_status = accountWork(
                operations, remaining_operations, is_cancelled);
            if (work_status != TextureAtlasTryPackStatus::Packed)
            {
                return work_status;
            }
            if (contains(next[second], next[first]))
            {
                removed[first] = true;
                break;
            }
        }
    }

    QVector<PackingRect> pruned;
    pruned.reserve(next.size());
    for (int index = 0; index < next.size(); ++index)
    {
        if (!removed[index])
        {
            pruned.push_back(next[index]);
        }
    }
    if (pruned.size() > kMaximumFreeRectangles)
    {
        return TextureAtlasTryPackStatus::TooComplex;
    }
    *free_rectangles = std::move(pruned);
    return TextureAtlasTryPackStatus::Packed;
}

} // namespace

QVector<TextureAtlasItem> orderForMaxRects(
    const QVector<TextureAtlasItem> &source)
{
    QVector<TextureAtlasItem> ordered = source;
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right)
    {
        const int left_max = std::max(
            left.requestedSize.width(), left.requestedSize.height());
        const int right_max = std::max(
            right.requestedSize.width(), right.requestedSize.height());
        if (left_max != right_max)
        {
            return left_max > right_max;
        }
        const qint64 left_area = itemArea(left);
        const qint64 right_area = itemArea(right);
        if (left_area != right_area)
        {
            return left_area > right_area;
        }
        if (left.requestedSize.height() != right.requestedSize.height())
        {
            return left.requestedSize.height() > right.requestedSize.height();
        }
        if (left.requestedSize.width() != right.requestedSize.width())
        {
            return left.requestedSize.width() > right.requestedSize.width();
        }
        return left.id < right.id;
    });
    return ordered;
}

TextureAtlasTryPackStatus tryPackMaxRects(
    const QVector<TextureAtlasItem> &source,
    int atlasSize,
    int reservedLeft,
    float scale,
    const std::function<bool()> &isCancelled,
    std::uint64_t *remainingOperations,
    QVector<TextureAtlasItem> *packed)
{
    if (isCancelled && isCancelled())
    {
        return TextureAtlasTryPackStatus::Cancelled;
    }
    QVector<TextureAtlasItem> ordered = source;
    QVector<PackingRect> free_rectangles{
        {reservedLeft, 0, atlasSize - reservedLeft, atlasSize}
    };
    std::uint64_t operations = 0;
    for (TextureAtlasItem &item : ordered)
    {
        const int width = scaledAtlasItemDimension(
            item.requestedSize.width(), item.fixedPadding, scale);
        const int height = scaledAtlasItemDimension(
            item.requestedSize.height(), item.fixedPadding, scale);
        Placement best;
        for (int index = 0; index < free_rectangles.size(); ++index)
        {
            const TextureAtlasTryPackStatus work_status = accountWork(
                &operations, remainingOperations, isCancelled);
            if (work_status != TextureAtlasTryPackStatus::Packed)
            {
                return work_status;
            }
            const PackingRect &free_rect = free_rectangles[index];
            if (width > free_rect.width || height > free_rect.height)
            {
                continue;
            }
            const int remaining_width = free_rect.width - width;
            const int remaining_height = free_rect.height - height;
            const std::array<int, 7> score{
                std::min(remaining_width, remaining_height),
                std::max(remaining_width, remaining_height),
                free_rect.y,
                free_rect.x,
                free_rect.height,
                free_rect.width,
                index};
            if (best.freeIndex < 0 || score < best.score)
            {
                best = {index, score};
            }
        }
        if (best.freeIndex < 0)
        {
            return TextureAtlasTryPackStatus::DoesNotFit;
        }

        const PackingRect &selected = free_rectangles[best.freeIndex];
        const PackingRect used{selected.x, selected.y, width, height};
        item.packedRect = QRect(used.x, used.y, used.width, used.height);
        const TextureAtlasTryPackStatus split_status = splitFreeRectangles(
            &free_rectangles,
            used,
            &operations,
            remainingOperations,
            isCancelled);
        if (split_status != TextureAtlasTryPackStatus::Packed)
        {
            return split_status;
        }
    }
    *packed = std::move(ordered);
    return TextureAtlasTryPackStatus::Packed;
}

} // namespace xjw::mesh::detail
