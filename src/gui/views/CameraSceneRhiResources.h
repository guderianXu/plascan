#pragma once

#include <QByteArray>
#include <QHash>
#include <QScopedPointer>
#include <QSet>
#include <QSharedPointer>
#include <QSize>
#include <QString>
#include <QVector>
#include <QVector3D>

#include <cstdint>

class QRhiBuffer;
class QRhiGraphicsPipeline;
class QRhiSampler;
class QRhiShaderResourceBindings;
class QRhiTexture;

namespace xjw::gui::camera_scene
{

struct RhiBufferSet
{
    QScopedPointer<QRhiBuffer> vertexBuffer;
    QByteArray vertexData;
    int vertexCount = 0;
    int strideBytes = 0;
    bool dirty = true;
};

struct RhiPipelineSet
{
    QScopedPointer<QRhiBuffer> uniformBuffer;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    QScopedPointer<QRhiGraphicsPipeline> pipeline;
    QString vertexShaderPath;
    QString fragmentShaderPath;
};

struct RhiIndexBufferSet
{
    QScopedPointer<QRhiBuffer> indexBuffer;
    QByteArray indexData;
    int indexCount = 0;
    bool dirty = true;
};

struct RhiPointChunk
{
    RhiBufferSet points;
    RhiBufferSet scalars;
    QVector<std::uint32_t> sourceIndices;
    QVector3D aabbMinimum;
    QVector3D aabbMaximum;
    QVector3D center;
    float radius = 0.0f;
    int pointCount = 0;
};

struct RhiMeshPointPreviewChunk
{
    RhiBufferSet points;
};

struct RhiImagePipelineSet
{
    QScopedPointer<QRhiBuffer> vertexBuffer;
    QScopedPointer<QRhiBuffer> uniformBuffer;
    QScopedPointer<QRhiTexture> texture;
    QScopedPointer<QRhiSampler> sampler;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    QScopedPointer<QRhiGraphicsPipeline> pipeline;
    QSize textureSize;
    QString uploadedImageKey;
    QString uploadedGeometryKey;
    bool pipelineDirty = true;
};

struct RhiTexturedMeshPipelineSet
{
    QScopedPointer<QRhiBuffer> uniformBuffer;
    QScopedPointer<QRhiTexture> texture;
    QScopedPointer<QRhiSampler> sampler;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    QScopedPointer<QRhiGraphicsPipeline> pipeline;
    QSize textureSize;
    QString uploadedTexturePath;
    QString vertexShaderPath;
    QString fragmentShaderPath;
};

struct RhiCameraThumbnailResource
{
    QScopedPointer<QRhiBuffer> instanceBuffer;
    QScopedPointer<QRhiTexture> texture;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    int instanceCapacity = 0;
    int instanceCount = 0;
};

struct RhiCameraThumbnailAtlasPage
{
    QScopedPointer<QRhiBuffer> instanceBuffer;
    QScopedPointer<QRhiTexture> texture;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    QSet<int> uploadedPoseIndices;
    QHash<int, QSize> imageSizes;
    int instanceCapacity = 0;
    int instanceCount = 0;
};

struct RhiCameraThumbnailPipelineSet
{
    QScopedPointer<QRhiBuffer> uniformBuffer;
    QScopedPointer<QRhiSampler> sampler;
    QScopedPointer<QRhiGraphicsPipeline> pipeline;
    QScopedPointer<QRhiGraphicsPipeline> leaderPipeline;
    QScopedPointer<QRhiShaderResourceBindings> leaderBindings;
    QVector<QSharedPointer<RhiCameraThumbnailAtlasPage>> atlasPages;
    QSharedPointer<RhiCameraThumbnailResource> solidResource;
    QSharedPointer<RhiCameraThumbnailResource> highlightedSolidResource;
    QScopedPointer<QRhiBuffer> leaderInstanceBuffer;
    int leaderInstanceCapacity = 0;
    int leaderInstanceCount = 0;
    int segmentInstanceCount = 0;
    int atlasSize = 0;
    bool resourcesDirty = true;
    bool instancesDirty = true;
    bool pipelinesDirty = true;
};

} // namespace xjw::gui::camera_scene
