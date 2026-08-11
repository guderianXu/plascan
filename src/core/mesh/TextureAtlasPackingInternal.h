#pragma once

#include "TextureAtlasPacker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace xjw::mesh::detail
{

inline constexpr int kMaximumMaxRectsItems = 4096;
inline constexpr std::uint64_t kMaximumMaxRectsTotalOperations = 12'000'000;

inline int scaledAtlasItemDimension(int requestedDimension,
                                    int fixedPadding,
                                    float scale)
{
    const double content = std::max(
        1.0,
        std::ceil(static_cast<double>(requestedDimension) * scale));
    const double total = content + 2.0 * fixedPadding;
    return total >= std::numeric_limits<int>::max()
        ? std::numeric_limits<int>::max()
        : static_cast<int>(total);
}

enum class TextureAtlasTryPackStatus
{
    Packed,
    DoesNotFit,
    TooComplex,
    Cancelled
};

QVector<TextureAtlasItem> orderForMaxRects(
    const QVector<TextureAtlasItem> &source);

TextureAtlasTryPackStatus tryPackMaxRects(
    const QVector<TextureAtlasItem> &source,
    int atlasSize,
    int reservedLeft,
    float scale,
    const std::function<bool()> &isCancelled,
    std::uint64_t *remainingOperations,
    QVector<TextureAtlasItem> *packed);

} // namespace xjw::mesh::detail
