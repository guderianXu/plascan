#include "MatchGpuRenderer.h"

#include <QFile>
#include <QMatrix4x4>
#include <rhi/qrhi.h>
#include <rhi/qshader.h>

#include <algorithm>
#include <cstring>

namespace
{

QShader loadShader(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 着色器资源缺失：%1").arg(path);
        }
        return {};
    }
    const QShader shader = QShader::fromSerialized(file.readAll());
    if (!shader.isValid() && errorMessage)
    {
        *errorMessage = QStringLiteral("匹配查看 GPU 着色器加载失败：%1").arg(path);
    }
    return shader;
}

} // namespace

MatchGpuRenderer::MatchGpuRenderer() = default;

MatchGpuRenderer::~MatchGpuRenderer()
{
    release();
}

bool MatchGpuRenderer::initialize(QRhi *rhi,
                                  QRhiRenderTarget *renderTarget,
                                  int sampleCount,
                                  qreal lineWidth,
                                  QString *errorMessage)
{
    release();
    if (!rhi || !renderTarget)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 渲染目标不可用。");
        }
        return false;
    }
    _sampleCount = std::max(1, sampleCount);
    return createPipelines(rhi, renderTarget, _sampleCount, lineWidth, errorMessage);
}

void MatchGpuRenderer::release()
{
    _pointPipeline.reset();
    _linePipeline.reset();
    _bindings.reset();
    _uniformBuffer.reset();
    _vertexBuffer.reset();
    _pendingVertexData.clear();
    _uploadedGeneration = 0;
    _vertexCapacityBytes = 0;
    _lineVertexCount = 0;
    _pointVertexCount = 0;
    _pipelineLineWidth = -1.0;
}

bool MatchGpuRenderer::createPipelines(QRhi *rhi,
                                       QRhiRenderTarget *renderTarget,
                                       int sampleCount,
                                       qreal lineWidth,
                                       QString *errorMessage)
{
    const QShader vertex_shader = loadShader(
        QStringLiteral(":/shaders/match_overlay.vert.qsb"), errorMessage);
    const QShader fragment_shader = loadShader(
        QStringLiteral(":/shaders/match_overlay.frag.qsb"), errorMessage);
    if (!vertex_shader.isValid() || !fragment_shader.isValid())
    {
        return false;
    }

    _uniformBuffer.reset(rhi->newBuffer(
        QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 16 * sizeof(float)));
    if (!_uniformBuffer->create())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU uniform 缓冲创建失败。");
        }
        return false;
    }

    _bindings.reset(rhi->newShaderResourceBindings());
    _bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0, QRhiShaderResourceBinding::VertexStage, _uniformBuffer.data())
    });
    if (!_bindings->create())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 资源绑定创建失败。");
        }
        return false;
    }

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({QRhiVertexInputBinding(sizeof(Vertex))});
    input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float4, 2 * sizeof(float))
    });

    const auto create_pipeline = [&](QRhiGraphicsPipeline::Topology topology,
                                     QScopedPointer<QRhiGraphicsPipeline> *pipeline)
    {
        pipeline->reset(rhi->newGraphicsPipeline());
        (*pipeline)->setTopology(topology);
        (*pipeline)->setShaderStages({
            QRhiShaderStage(QRhiShaderStage::Vertex, vertex_shader),
            QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader)
        });
        (*pipeline)->setVertexInputLayout(input_layout);
        (*pipeline)->setShaderResourceBindings(_bindings.data());
        (*pipeline)->setRenderPassDescriptor(renderTarget->renderPassDescriptor());
        (*pipeline)->setSampleCount(sampleCount);
        (*pipeline)->setDepthTest(false);
        (*pipeline)->setDepthWrite(false);
        (*pipeline)->setCullMode(QRhiGraphicsPipeline::None);
        QRhiGraphicsPipeline::TargetBlend blend;
        blend.enable = true;
        blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
        blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        (*pipeline)->setTargetBlends({blend});
        if (topology == QRhiGraphicsPipeline::Lines)
        {
            (*pipeline)->setLineWidth(static_cast<float>(lineWidth));
        }
        return (*pipeline)->create();
    };

    if (!create_pipeline(QRhiGraphicsPipeline::Lines, &_linePipeline)
        || !create_pipeline(QRhiGraphicsPipeline::Points, &_pointPipeline))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 图形管线创建失败。");
        }
        return false;
    }
    _pipelineLineWidth = lineWidth;
    return true;
}

bool MatchGpuRenderer::uploadVertices(QRhi *rhi,
                                      const QVector<MatchGpuLineBatch> &batches,
                                      bool showEndpoints,
                                      qreal opacity,
                                      quint64 generation,
                                      QString *errorMessage)
{
    QVector<Vertex> line_vertices;
    QVector<Vertex> point_vertices;
    qsizetype line_count = 0;
    for (const MatchGpuLineBatch &batch : batches)
    {
        if (batch.lines)
        {
            line_count += batch.lines->size();
        }
    }
    line_vertices.reserve(line_count * 2);
    if (showEndpoints)
    {
        point_vertices.reserve(line_count * 2);
    }

    for (const MatchGpuLineBatch &batch : batches)
    {
        if (!batch.lines || batch.lines->isEmpty())
        {
            continue;
        }
        const QColor color = batch.color;
        const auto make_vertex = [&color, opacity](const QPointF &point)
        {
            return Vertex{static_cast<float>(point.x()),
                          static_cast<float>(point.y()),
                          static_cast<float>(color.redF()),
                          static_cast<float>(color.greenF()),
                          static_cast<float>(color.blueF()),
                          static_cast<float>(opacity)};
        };
        for (const QLineF &line : *batch.lines)
        {
            const Vertex first = make_vertex(line.p1());
            const Vertex second = make_vertex(line.p2());
            line_vertices.append(first);
            line_vertices.append(second);
            if (showEndpoints)
            {
                point_vertices.append(first);
                point_vertices.append(second);
            }
        }
    }

    _lineVertexCount = static_cast<quint32>(line_vertices.size());
    _pointVertexCount = static_cast<quint32>(point_vertices.size());
    const quint32 byte_count = static_cast<quint32>(
        (line_vertices.size() + point_vertices.size()) * sizeof(Vertex));
    _pendingVertexData.resize(static_cast<qsizetype>(byte_count));
    if (!line_vertices.isEmpty())
    {
        std::memcpy(_pendingVertexData.data(),
                    line_vertices.constData(),
                    line_vertices.size() * sizeof(Vertex));
    }
    if (!point_vertices.isEmpty())
    {
        std::memcpy(_pendingVertexData.data() + line_vertices.size() * sizeof(Vertex),
                    point_vertices.constData(),
                    point_vertices.size() * sizeof(Vertex));
    }

    if (byte_count > _vertexCapacityBytes)
    {
        _vertexBuffer.reset(rhi->newBuffer(
            QRhiBuffer::Dynamic,
            QRhiBuffer::VertexBuffer,
            std::max<quint32>(byte_count, sizeof(Vertex))));
        if (!_vertexBuffer->create())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("匹配查看 GPU 顶点缓冲创建失败。");
            }
            return false;
        }
        _vertexCapacityBytes = std::max<quint32>(byte_count, sizeof(Vertex));
    }
    _uploadedGeneration = generation;
    return true;
}

bool MatchGpuRenderer::render(QRhi *rhi,
                              QRhiRenderTarget *renderTarget,
                              QRhiCommandBuffer *commandBuffer,
                              const QSize &logicalSize,
                              const QVector<MatchGpuLineBatch> &batches,
                              bool showEndpoints,
                              qreal opacity,
                              quint64 generation,
                              qreal lineWidth,
                              QString *errorMessage)
{
    if (!rhi || !renderTarget || !commandBuffer)
    {
        return false;
    }
    if (!_linePipeline || !_pointPipeline || !qFuzzyCompare(_pipelineLineWidth, lineWidth))
    {
        _linePipeline.reset();
        _pointPipeline.reset();
        if (!createPipelines(rhi, renderTarget, _sampleCount, lineWidth, errorMessage))
        {
            return false;
        }
    }
    if (_uploadedGeneration != generation
        && !uploadVertices(rhi, batches, showEndpoints, opacity, generation, errorMessage))
    {
        return false;
    }

    QRhiResourceUpdateBatch *updates = rhi->nextResourceUpdateBatch();
    if (!_pendingVertexData.isEmpty())
    {
        updates->updateDynamicBuffer(
            _vertexBuffer.data(), 0, _pendingVertexData.size(), _pendingVertexData.constData());
        _pendingVertexData.clear();
    }
    QMatrix4x4 projection;
    projection.ortho(0.0f,
                     static_cast<float>(std::max(1, logicalSize.width())),
                     static_cast<float>(std::max(1, logicalSize.height())),
                     0.0f,
                     -1.0f,
                     1.0f);
    const QMatrix4x4 matrix = rhi->clipSpaceCorrMatrix() * projection;
    updates->updateDynamicBuffer(
        _uniformBuffer.data(), 0, 16 * sizeof(float), matrix.constData());

    commandBuffer->beginPass(renderTarget,
                             QColor::fromRgbF(0.0, 0.0, 0.0, 0.0),
                             QRhiDepthStencilClearValue(1.0f, 0),
                             updates);
    const QSize pixel_size = renderTarget->pixelSize();
    commandBuffer->setViewport(QRhiViewport(
        0.0f, 0.0f, static_cast<float>(pixel_size.width()), static_cast<float>(pixel_size.height())));

    if (_vertexBuffer && _lineVertexCount > 0)
    {
        commandBuffer->setGraphicsPipeline(_linePipeline.data());
        commandBuffer->setShaderResources(_bindings.data());
        const QRhiCommandBuffer::VertexInput input(_vertexBuffer.data(), 0);
        commandBuffer->setVertexInput(0, 1, &input);
        commandBuffer->draw(_lineVertexCount);
    }
    if (_vertexBuffer && _pointVertexCount > 0)
    {
        commandBuffer->setGraphicsPipeline(_pointPipeline.data());
        commandBuffer->setShaderResources(_bindings.data());
        const quint32 point_offset = _lineVertexCount * sizeof(Vertex);
        const QRhiCommandBuffer::VertexInput input(_vertexBuffer.data(), point_offset);
        commandBuffer->setVertexInput(0, 1, &input);
        commandBuffer->draw(_pointVertexCount);
    }
    commandBuffer->endPass();
    return true;
}
