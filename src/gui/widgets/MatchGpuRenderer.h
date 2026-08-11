#pragma once

#include <QColor>
#include <QByteArray>
#include <QLineF>
#include <QScopedPointer>
#include <QSize>
#include <QVector>
#include <QString>

class QRhi;
class QRhiBuffer;
class QRhiCommandBuffer;
class QRhiGraphicsPipeline;
class QRhiRenderTarget;
class QRhiShaderResourceBindings;

struct MatchGpuLineBatch
{
    const QVector<QLineF> *lines = nullptr;
    QColor color;
};

class MatchGpuRenderer
{
public:
    MatchGpuRenderer();
    ~MatchGpuRenderer();

    bool initialize(QRhi *rhi,
                    QRhiRenderTarget *renderTarget,
                    int sampleCount,
                    qreal lineWidth,
                    QString *errorMessage);
    void release();

    bool render(QRhi *rhi,
                QRhiRenderTarget *renderTarget,
                QRhiCommandBuffer *commandBuffer,
                const QSize &logicalSize,
                const QVector<MatchGpuLineBatch> &batches,
                bool showEndpoints,
                qreal opacity,
                quint64 generation,
                qreal lineWidth,
                QString *errorMessage);

private:
    struct Vertex
    {
        float x = 0.0f;
        float y = 0.0f;
        float red = 1.0f;
        float green = 1.0f;
        float blue = 1.0f;
        float alpha = 1.0f;
    };

    bool createPipelines(QRhi *rhi,
                         QRhiRenderTarget *renderTarget,
                         int sampleCount,
                         qreal lineWidth,
                         QString *errorMessage);
    bool uploadVertices(QRhi *rhi,
                        const QVector<MatchGpuLineBatch> &batches,
                        bool showEndpoints,
                        qreal opacity,
                        quint64 generation,
                        QString *errorMessage);

    QScopedPointer<QRhiBuffer> _vertexBuffer;
    QScopedPointer<QRhiBuffer> _uniformBuffer;
    QScopedPointer<QRhiShaderResourceBindings> _bindings;
    QScopedPointer<QRhiGraphicsPipeline> _linePipeline;
    QScopedPointer<QRhiGraphicsPipeline> _pointPipeline;
    QByteArray _pendingVertexData;
    quint64 _uploadedGeneration = 0;
    quint32 _vertexCapacityBytes = 0;
    quint32 _lineVertexCount = 0;
    quint32 _pointVertexCount = 0;
    qreal _pipelineLineWidth = -1.0;
    int _sampleCount = 1;
};
