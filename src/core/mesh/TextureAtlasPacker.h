#pragma once

#include <QRect>
#include <QSize>
#include <QVector>

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
    float scale = 1.0f;
    double occupancy = 0.0;
    QVector<TextureAtlasItem> items;
};

class TextureAtlasPacker
{
public:
    static TextureAtlasPackingResult pack(const QVector<TextureAtlasItem> &items,
                                          int atlasSize,
                                          int reservedLeft = 0);
};

} // namespace xjw::mesh
