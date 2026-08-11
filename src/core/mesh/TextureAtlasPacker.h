#pragma once

#include <QRect>
#include <QSize>
#include <QVector>

#include <functional>

namespace xjw::mesh
{

struct TextureAtlasItem
{
    int id = -1;
    QSize requestedSize;
    QRect packedRect;
};

struct TextureAtlasPackingResult
{
    bool ok = false;
    bool cancelled = false;
    float scale = 1.0f;
    double occupancy = 0.0;
    QVector<TextureAtlasItem> items;
};

class TextureAtlasPacker
{
public:
    /**
     * @brief Pack unrotated charts with bounded MaxRects and a large-input shelf fallback.
     */
    static TextureAtlasPackingResult pack(const QVector<TextureAtlasItem> &items,
                                          int atlasSize,
                                          int reservedLeft = 0,
                                          const std::function<bool()> &isCancelled = {},
                                          const std::function<void(int)> &progressFn = {});
};

} // namespace xjw::mesh
