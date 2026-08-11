#pragma once

#include "TextureAtlasPacker.h"

#include <cstdint>
#include <functional>

namespace xjw::mesh::detail
{

inline constexpr int kMaximumMaxRectsItems = 4096;
inline constexpr std::uint64_t kMaximumMaxRectsTotalOperations = 12'000'000;

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
