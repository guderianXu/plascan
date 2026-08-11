#include "MatchGpuRenderer.h"

#include <QFile>
#include <QMatrix4x4>
#include <rhi/qrhi.h>
#include <rhi/qshader.h>

#include <algorithm>
#include <array>
#include <cmath>
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
    _endpointMeshBuffer.reset();
    _vertexBuffer.reset();
    _pendingVertexData.clear();
    _pendingEndpointMeshData.clear();
    _uploadedGeneration = 0;
    _vertexCapacityBytes = 0;
    _lineVertexCount = 0;
    _endpointInstanceCount = 0;
    _endpointMeshVertexCount = 0;
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
    const QShader endpoint_vertex_shader = loadShader(
        QStringLiteral(":/shaders/match_overlay_endpoint.vert.qsb"), errorMessage);
    if (!vertex_shader.isValid() || !fragment_shader.isValid()
        || !endpoint_vertex_shader.isValid())
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

    const auto configure_pipeline = [&](QRhiGraphicsPipeline *pipeline,
                                        QRhiGraphicsPipeline::Topology topology,
                                        const QShader &pipeline_vertex_shader,
                                        const QRhiVertexInputLayout &pipeline_input_layout)
    {
        pipeline->setTopology(topology);
        pipeline->setShaderStages({
            QRhiShaderStage(QRhiShaderStage::Vertex, pipeline_vertex_shader),
            QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader)
        });
        pipeline->setVertexInputLayout(pipeline_input_layout);
        pipeline->setShaderResourceBindings(_bindings.data());
        pipeline->setRenderPassDescriptor(renderTarget->renderPassDescriptor());
        pipeline->setSampleCount(sampleCount);
        pipeline->setDepthTest(false);
        pipeline->setDepthWrite(false);
        pipeline->setCullMode(QRhiGraphicsPipeline::None);
        QRhiGraphicsPipeline::TargetBlend blend;
        blend.enable = true;
        blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
        blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        pipeline->setTargetBlends({blend});
        if (topology == QRhiGraphicsPipeline::Lines)
        {
            pipeline->setLineWidth(static_cast<float>(lineWidth));
        }
    };

    _linePipeline.reset(rhi->newGraphicsPipeline());
    configure_pipeline(
        _linePipeline.data(), QRhiGraphicsPipeline::Lines, vertex_shader, input_layout);

    QRhiVertexInputLayout endpoint_input_layout;
    endpoint_input_layout.setBindings({
        QRhiVertexInputBinding(2 * sizeof(float)),
        QRhiVertexInputBinding(sizeof(Vertex), QRhiVertexInputBinding::PerInstance)
    });
    endpoint_input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0),
        QRhiVertexInputAttribute(1, 1, QRhiVertexInputAttribute::Float2, 0),
        QRhiVertexInputAttribute(1, 2, QRhiVertexInputAttribute::Float4, 2 * sizeof(float))
    });
    _pointPipeline.reset(rhi->newGraphicsPipeline());
    configure_pipeline(_pointPipeline.data(),
                       QRhiGraphicsPipeline::Triangles,
                       endpoint_vertex_shader,
                       endpoint_input_layout);

    if (!_linePipeline->create() || !_pointPipeline->create())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 图形管线创建失败。");
        }
        return false;
    }

    constexpr int endpoint_segments = 12;
    constexpr float pi = 3.14159265358979323846f;
    QVector<std::array<float, 2>> endpoint_vertices;
    endpoint_vertices.reserve(endpoint_segments * 3);
    for (int segment = 0; segment < endpoint_segments; ++segment)
    {
        const float first_angle = float(segment) * 2.0f * pi / endpoint_segments;
        const float second_angle = float(segment + 1) * 2.0f * pi / endpoint_segments;
        endpoint_vertices.append({0.0f, 0.0f});
        endpoint_vertices.append({std::cos(first_angle), std::sin(first_angle)});
        endpoint_vertices.append({std::cos(second_angle), std::sin(second_angle)});
    }
    _endpointMeshVertexCount = static_cast<quint32>(endpoint_vertices.size());
    _pendingEndpointMeshData = QByteArray(
        reinterpret_cast<const char *>(endpoint_vertices.constData()),
        endpoint_vertices.size() * qsizetype(sizeof(endpoint_vertices.front())));
    _endpointMeshBuffer.reset(rhi->newBuffer(
        QRhiBuffer::Immutable,
        QRhiBuffer::VertexBuffer,
        static_cast<quint32>(_pendingEndpointMeshData.size())));
    if (!_endpointMeshBuffer->create())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配查看 GPU 端点圆片缓冲创建失败。");
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
    QVector<Vertex> endpoint_instances;
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
        endpoint_instances.reserve(line_count * 2);
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
                endpoint_instances.append(first);
                endpoint_instances.append(second);
            }
        }
    }

    _lineVertexCount = static_cast<quint32>(line_vertices.size());
    _endpointInstanceCount = static_cast<quint32>(endpoint_instances.size());
    const quint32 byte_count = static_cast<quint32>(
        (line_vertices.size() + endpoint_instances.size()) * sizeof(Vertex));
    _pendingVertexData.resize(static_cast<qsizetype>(byte_count));
    if (!line_vertices.isEmpty())
    {
        std::memcpy(_pendingVertexData.data(),
                    line_vertices.constData(),
                    line_vertices.size() * sizeof(Vertex));
    }
    if (!endpoint_instances.isEmpty())
    {
        std::memcpy(_pendingVertexData.data() + line_vertices.size() * sizeof(Vertex),
                    endpoint_instances.constData(),
                    endpoint_instances.size() * sizeof(Vertex));
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
    if (!_pendingEndpointMeshData.isEmpty())
    {
        updates->uploadStaticBuffer(
            _endpointMeshBuffer.data(), _pendingEndpointMeshData.constData());
        _pendingEndpointMeshData.clear();
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
    if (_vertexBuffer && _endpointMeshBuffer && _endpointInstanceCount > 0)
    {
        commandBuffer->setGraphicsPipeline(_pointPipeline.data());
        commandBuffer->setShaderResources(_bindings.data());
        const quint32 endpoint_offset = _lineVertexCount * sizeof(Vertex);
        const QRhiCommandBuffer::VertexInput inputs[] = {
            QRhiCommandBuffer::VertexInput(_endpointMeshBuffer.data(), 0),
            QRhiCommandBuffer::VertexInput(_vertexBuffer.data(), endpoint_offset)
        };
        commandBuffer->setVertexInput(0, 2, inputs);
        commandBuffer->draw(_endpointMeshVertexCount, _endpointInstanceCount);
    }
    commandBuffer->endPass();
    return true;
}
