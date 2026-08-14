#include "TextureAtlasPacker.h"

#include "TextureAtlasPackingInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace xjw::mesh
{
namespace
{

using detail::TextureAtlasTryPackStatus;

struct Shelf
{
    int y = 0;
    int height = 0;
    int nextX = 0;
    int remainingWidth = 0;
};

qint64 itemArea(const TextureAtlasItem &item)
{
    return static_cast<qint64>(item.requestedSize.width()) *
        item.requestedSize.height();
}

QVector<TextureAtlasItem> orderForShelves(
    const QVector<TextureAtlasItem> &source)
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
        return left.id < right.id;
    });
    return ordered;
}

float nextLowerPackingScale(
    const QVector<TextureAtlasItem> &items,
    float current_scale)
{
    double next_breakpoint = 0.0;
    const auto consider_dimension = [&](int dimension)
    {
        const int current_pixels = std::max(
            1,
            static_cast<int>(std::ceil(
                static_cast<double>(dimension) * current_scale)));
        if (current_pixels > 1)
        {
            next_breakpoint = std::max(
                next_breakpoint,
                static_cast<double>(current_pixels - 1) / dimension);
        }
    };
    for (const TextureAtlasItem &item : items)
    {
        consider_dimension(item.requestedSize.width());
        consider_dimension(item.requestedSize.height());
    }
    if (next_breakpoint <= 0.0)
    {
        return 0.0f;
    }

    float next_scale = std::nextafter(
        static_cast<float>(next_breakpoint), 0.0f);
    if (next_scale >= current_scale)
    {
        next_scale = std::nextafter(current_scale, 0.0f);
    }
    return std::max(next_scale, 0.0f);
}

TextureAtlasTryPackStatus tryPackShelves(
    const QVector<TextureAtlasItem> &source,
    int atlas_size,
    int reserved_left,
    float scale,
    const std::function<bool()> &is_cancelled,
    QVector<TextureAtlasItem> *packed)
{
    QVector<TextureAtlasItem> ordered = source;
    const int available_width = atlas_size - reserved_left;
    QVector<Shelf> shelves;
    shelves.reserve(std::min<qsizetype>(ordered.size(), atlas_size));
    std::set<std::pair<int, int>> shelves_by_remaining_width;
    int used_height = 0;
    for (int item_index = 0; item_index < ordered.size(); ++item_index)
    {
        if ((item_index % 1024 == 0) && is_cancelled && is_cancelled())
        {
            return TextureAtlasTryPackStatus::Cancelled;
        }
        TextureAtlasItem &item = ordered[item_index];
        const int width = detail::scaledAtlasItemDimension(
            item.requestedSize.width(), item.fixedPadding, scale);
        const int height = detail::scaledAtlasItemDimension(
            item.requestedSize.height(), item.fixedPadding, scale);
        if (width > available_width || height > atlas_size)
        {
            return TextureAtlasTryPackStatus::DoesNotFit;
        }

        auto best = shelves_by_remaining_width.lower_bound({width, -1});
        while (best != shelves_by_remaining_width.end() &&
               shelves[best->second].height < height)
        {
            ++best;
        }
        int shelf_index = -1;
        if (best != shelves_by_remaining_width.end())
        {
            shelf_index = best->second;
            shelves_by_remaining_width.erase(best);
        }
        else
        {
            if (height > atlas_size - used_height)
            {
                return TextureAtlasTryPackStatus::DoesNotFit;
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
    *packed = std::move(ordered);
    return TextureAtlasTryPackStatus::Packed;
}

TextureAtlasTryPackStatus tryPack(
    const QVector<TextureAtlasItem> &shelf_ordered,
    const QVector<TextureAtlasItem> &max_rects_ordered,
    int atlas_size,
    int reserved_left,
    float scale,
    const std::function<bool()> &is_cancelled,
    bool *max_rects_enabled,
    std::uint64_t *remaining_max_rects_operations,
    QVector<TextureAtlasItem> *packed)
{
    if (*max_rects_enabled)
    {
        const TextureAtlasTryPackStatus max_rects_status =
            detail::tryPackMaxRects(
                max_rects_ordered,
                atlas_size,
                reserved_left,
                scale,
                is_cancelled,
                remaining_max_rects_operations,
                packed);
        if (max_rects_status == TextureAtlasTryPackStatus::Packed ||
            max_rects_status == TextureAtlasTryPackStatus::Cancelled)
        {
            return max_rects_status;
        }
        if (max_rects_status == TextureAtlasTryPackStatus::TooComplex)
        {
            *max_rects_enabled = false;
        }
    }
    return tryPackShelves(
        shelf_ordered,
        atlas_size,
        reserved_left,
        scale,
        is_cancelled,
        packed);
}

} // namespace

TextureAtlasPackingResult TextureAtlasPacker::pack(
    const QVector<TextureAtlasItem> &items,
    int atlasSize,
    int reservedLeft,
    const std::function<bool()> &isCancelled,
    const std::function<void(int)> &progressFn,
    float maximumScale)
{
    TextureAtlasPackingResult result;
    if (items.isEmpty() || atlasSize <= 0 ||
        reservedLeft < 0 || reservedLeft >= atlasSize ||
        !std::isfinite(maximumScale) || maximumScale <= 0.0f)
    {
        return result;
    }
    if (isCancelled && isCancelled())
    {
        result.cancelled = true;
        return result;
    }

    long double requested_area = 0.0L;
    long double minimum_packed_area = 0.0L;
    double dimension_scale_bound = maximumScale;
    const int available_width = atlasSize - reservedLeft;
    for (const TextureAtlasItem &item : items)
    {
        if (item.id < 0 || item.requestedSize.isEmpty() ||
            item.fixedPadding < 0)
        {
            return result;
        }
        if (item.fixedPadding > (available_width - 1) / 2 ||
            item.fixedPadding > (atlasSize - 1) / 2)
        {
            return result;
        }
        const int horizontal_content_space =
            available_width - item.fixedPadding * 2;
        const int vertical_content_space =
            atlasSize - item.fixedPadding * 2;
        if (horizontal_content_space < 1 || vertical_content_space < 1)
        {
            return result;
        }
        requested_area += static_cast<long double>(itemArea(item));
        const long double minimum_width = 1.0L + item.fixedPadding * 2.0L;
        minimum_packed_area += minimum_width * minimum_width;
        dimension_scale_bound = std::min(
            dimension_scale_bound,
            static_cast<double>(horizontal_content_space) /
                item.requestedSize.width());
        dimension_scale_bound = std::min(
            dimension_scale_bound,
            static_cast<double>(vertical_content_space) /
                item.requestedSize.height());
    }
    const qint64 available_area =
        static_cast<qint64>(atlasSize - reservedLeft) * atlasSize;
    if (static_cast<qint64>(items.size()) > available_area)
    {
        return result;
    }
    if (minimum_packed_area > static_cast<long double>(available_area))
    {
        return result;
    }
    const long double area_scale_bound = std::sqrt(
        static_cast<long double>(available_area) /
        std::max(requested_area, 1.0L));
    float high = static_cast<float>(std::min(
        {static_cast<long double>(maximumScale),
         area_scale_bound,
         static_cast<long double>(dimension_scale_bound)}));
    // At this scale every content dimension is rounded up to one pixel.  The
    // fixed atlas padding remains intact and can still make the set infeasible.
    float low = std::numeric_limits<float>::denorm_min();
    const QVector<TextureAtlasItem> shelf_ordered = orderForShelves(items);
    const bool use_max_rects =
        items.size() <= detail::kMaximumMaxRectsItems;
    const QVector<TextureAtlasItem> max_rects_ordered = use_max_rects
        ? detail::orderForMaxRects(items)
        : QVector<TextureAtlasItem>();
    bool max_rects_enabled = use_max_rects;
    std::uint64_t remaining_max_rects_operations =
        detail::kMaximumMaxRectsTotalOperations;
    QVector<TextureAtlasItem> packed;
    constexpr int search_iterations = 14;
    constexpr int maximum_max_rects_scale_states = 128;
    const int maximum_attempts = use_max_rects
        ? maximum_max_rects_scale_states + search_iterations + 2
        : search_iterations + 2;
    int completed_attempts = 0;
    const auto try_pack = [&](float scale, QVector<TextureAtlasItem> *output)
    {
        if (progressFn)
        {
            progressFn(completed_attempts * 100 / maximum_attempts);
        }
        const TextureAtlasTryPackStatus status = tryPack(
            shelf_ordered,
            max_rects_ordered,
            atlasSize,
            reservedLeft,
            scale,
            isCancelled,
            &max_rects_enabled,
            &remaining_max_rects_operations,
            output);
        ++completed_attempts;
        return status;
    };

    TextureAtlasTryPackStatus status = try_pack(high, &packed);
    if (status == TextureAtlasTryPackStatus::Cancelled)
    {
        result.cancelled = true;
        return result;
    }
    if (status == TextureAtlasTryPackStatus::Packed)
    {
        low = high;
    }
    else
    {
        float candidate = high;
        for (int state = 0;
             max_rects_enabled &&
             state < maximum_max_rects_scale_states;
             ++state)
        {
            candidate = nextLowerPackingScale(items, candidate);
            if (candidate <= 0.0f)
            {
                break;
            }
            QVector<TextureAtlasItem> trial;
            status = try_pack(candidate, &trial);
            if (status == TextureAtlasTryPackStatus::Cancelled)
            {
                result.cancelled = true;
                return result;
            }
            if (status == TextureAtlasTryPackStatus::Packed)
            {
                low = candidate;
                packed = std::move(trial);
                break;
            }
        }
        if (!packed.isEmpty())
        {
            status = TextureAtlasTryPackStatus::Packed;
        }
    }
    if (status != TextureAtlasTryPackStatus::Packed)
    {
        max_rects_enabled = false;
        QVector<TextureAtlasItem> minimum_packed;
        status = try_pack(low, &minimum_packed);
        if (status == TextureAtlasTryPackStatus::Cancelled)
        {
            result.cancelled = true;
            return result;
        }
        if (status != TextureAtlasTryPackStatus::Packed)
        {
            return result;
        }
        packed = std::move(minimum_packed);
        for (int iteration = 0; iteration < search_iterations; ++iteration)
        {
            const float candidate = (low + high) * 0.5f;
            QVector<TextureAtlasItem> trial;
            status = try_pack(candidate, &trial);
            if (status == TextureAtlasTryPackStatus::Cancelled)
            {
                result.cancelled = true;
                return result;
            }
            if (status == TextureAtlasTryPackStatus::Packed)
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

    std::sort(packed.begin(), packed.end(), [](const auto &left, const auto &right)
    {
        return left.id < right.id;
    });
    qint64 packed_area = 0;
    for (const TextureAtlasItem &item : packed)
    {
        packed_area += static_cast<qint64>(
            item.packedRect.width()) * item.packedRect.height();
    }
    result.ok = true;
    result.scale = low;
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
