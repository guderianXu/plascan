#include <QColor>
#include <QFile>
#include <QGuiApplication>
#include <QImage>
#include <QMatrix4x4>
#include <QVulkanInstance>

#include <rhi/qrhi.h>
#include <rhi/qshader.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>

namespace
{

struct alignas(16) SceneUniforms
{
    std::array<float, 16> mvp{};
    std::array<float, 16> modelView{};
    std::array<float, 16> normalMatrix{};
    std::array<float, 4> lightDirPointSize{};
    std::array<float, 4> viewportSize{};
    std::array<float, 4> renderModeFlags{};
    std::array<float, 4> scalarRange{};
};

static_assert(sizeof(SceneUniforms) == 64 * sizeof(float));

QShader loadShader(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return {};
    }
    return QShader::fromSerialized(file.readAll());
}

QColor readCenterPixel(const QRhiReadbackResult &result)
{
    if (result.pixelSize.isEmpty()
        || result.data.size() < result.pixelSize.width()
            * result.pixelSize.height() * 4)
    {
        return {};
    }
    const int x = result.pixelSize.width() / 2;
    const int y = result.pixelSize.height() / 2;
    const qsizetype offset = (static_cast<qsizetype>(y) * result.pixelSize.width() + x) * 4;
    const uchar *pixel = reinterpret_cast<const uchar *>(result.data.constData() + offset);
    return QColor(pixel[0], pixel[1], pixel[2], pixel[3]);
}

} // namespace

TEST(CameraSceneRhiSmokeTest, VulkanPreservesVertexColorAcrossModeSwitches)
{
    QVulkanInstance instance;
    instance.setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
    ASSERT_TRUE(instance.create())
        << "A real Vulkan instance is required for the GUI rendering smoke test.";

    QRhiVulkanInitParams initParams;
    initParams.inst = &instance;
    std::unique_ptr<QRhi> rhi(QRhi::create(QRhi::Vulkan, &initParams));
    ASSERT_NE(rhi, nullptr)
        << "Qt could not create its Vulkan QRhi backend.";

    constexpr int width = 96;
    constexpr int height = 96;
    std::unique_ptr<QRhiTexture> colorTexture(rhi->newTexture(
        QRhiTexture::RGBA8,
        QSize(width, height),
        1,
        QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
    ASSERT_TRUE(colorTexture->create());
    std::unique_ptr<QRhiTextureRenderTarget> renderTarget(
        rhi->newTextureRenderTarget({colorTexture.get()}));
    std::unique_ptr<QRhiRenderPassDescriptor> renderPass(
        renderTarget->newCompatibleRenderPassDescriptor());
    renderTarget->setRenderPassDescriptor(renderPass.get());
    ASSERT_TRUE(renderTarget->create());

    constexpr float vertices[] = {
        -0.85f, -0.80f, 0.0f, 0.0f, 0.0f, 1.0f, 0.90f, 0.08f, 0.04f,
         0.85f, -0.80f, 0.0f, 0.0f, 0.0f, 1.0f, 0.90f, 0.08f, 0.04f,
         0.00f,  0.85f, 0.0f, 0.0f, 0.0f, 1.0f, 0.90f, 0.08f, 0.04f};
    std::unique_ptr<QRhiBuffer> vertexBuffer(rhi->newBuffer(
        QRhiBuffer::Immutable,
        QRhiBuffer::VertexBuffer,
        sizeof(vertices)));
    ASSERT_TRUE(vertexBuffer->create());
    std::unique_ptr<QRhiBuffer> uniformBuffer(rhi->newBuffer(
        QRhiBuffer::Dynamic,
        QRhiBuffer::UniformBuffer,
        sizeof(SceneUniforms)));
    ASSERT_TRUE(uniformBuffer->create());

    std::unique_ptr<QRhiShaderResourceBindings> bindings(
        rhi->newShaderResourceBindings());
    bindings->setBindings({QRhiShaderResourceBinding::uniformBuffer(
        0,
        QRhiShaderResourceBinding::VertexStage
            | QRhiShaderResourceBinding::FragmentStage,
        uniformBuffer.get())});
    ASSERT_TRUE(bindings->create());

    const QShader vertexShader = loadShader(
        QStringLiteral(":/shaders/camera_scene_mesh.vert.qsb"));
    const QShader fragmentShader = loadShader(
        QStringLiteral(":/shaders/camera_scene_mesh.frag.qsb"));
    ASSERT_TRUE(vertexShader.isValid());
    ASSERT_TRUE(fragmentShader.isValid());

    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({QRhiVertexInputBinding(9 * sizeof(float))});
    inputLayout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(
            0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        QRhiVertexInputAttribute(
            0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float))});
    std::unique_ptr<QRhiGraphicsPipeline> pipeline(rhi->newGraphicsPipeline());
    pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertexShader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragmentShader)});
    pipeline->setVertexInputLayout(inputLayout);
    pipeline->setShaderResourceBindings(bindings.get());
    pipeline->setRenderPassDescriptor(renderPass.get());
    pipeline->setCullMode(QRhiGraphicsPipeline::None);
    ASSERT_TRUE(pipeline->create());

    bool vertexUploaded = false;
    auto renderMode = [&](int mode)
    {
        QRhiCommandBuffer *commandBuffer = nullptr;
        if (rhi->beginOffscreenFrame(&commandBuffer) != QRhi::FrameOpSuccess
            || !commandBuffer)
        {
            return QColor();
        }

        SceneUniforms uniforms;
        QMatrix4x4 identity;
        identity.setToIdentity();
        QMatrix4x4 modelView;
        modelView.setToIdentity();
        modelView.translate(0.0f, 0.0f, -2.0f);
        std::copy_n(identity.constData(), 16, uniforms.mvp.begin());
        std::copy_n(modelView.constData(), 16, uniforms.modelView.begin());
        std::copy_n(identity.constData(), 16, uniforms.normalMatrix.begin());
        uniforms.lightDirPointSize = {0.0f, 0.0f, 1.0f, 1.0f};
        uniforms.viewportSize = {float(width), float(height), 0.0f, 0.0f};
        uniforms.renderModeFlags = {float(mode), 1.0f, 0.0f, 0.0f};

        QRhiResourceUpdateBatch *updates = rhi->nextResourceUpdateBatch();
        if (!vertexUploaded)
        {
            updates->uploadStaticBuffer(vertexBuffer.get(), vertices);
            vertexUploaded = true;
        }
        updates->updateDynamicBuffer(
            uniformBuffer.get(), 0, sizeof(SceneUniforms), &uniforms);
        commandBuffer->beginPass(
            renderTarget.get(),
            QColor(Qt::black),
            QRhiDepthStencilClearValue(1.0f, 0),
            updates);
        commandBuffer->setGraphicsPipeline(pipeline.get());
        commandBuffer->setViewport(QRhiViewport(0, 0, width, height));
        commandBuffer->setShaderResources(bindings.get());
        const QRhiCommandBuffer::VertexInput vertexInput(vertexBuffer.get(), 0);
        commandBuffer->setVertexInput(0, 1, &vertexInput);
        commandBuffer->draw(3);

        QRhiReadbackResult readback;
        QRhiResourceUpdateBatch *readbackBatch = rhi->nextResourceUpdateBatch();
        readbackBatch->readBackTexture(
            QRhiReadbackDescription(colorTexture.get()), &readback);
        commandBuffer->endPass(readbackBatch);
        if (rhi->endOffscreenFrame() != QRhi::FrameOpSuccess)
        {
            return QColor();
        }
        return readCenterPixel(readback);
    };

    const QColor firstShaded = renderMode(1);
    const QColor solid = renderMode(2);
    const QColor secondShaded = renderMode(1);

    ASSERT_TRUE(firstShaded.isValid());
    ASSERT_TRUE(solid.isValid());
    ASSERT_TRUE(secondShaded.isValid());
    EXPECT_GT(firstShaded.red(), firstShaded.green() * 2);
    EXPECT_GT(firstShaded.red(), firstShaded.blue() * 2);
    EXPECT_GT(solid.blue(), solid.red());
    EXPECT_GT(solid.blue(), solid.green());
    EXPECT_NEAR(firstShaded.red(), secondShaded.red(), 3);
    EXPECT_NEAR(firstShaded.green(), secondShaded.green(), 3);
    EXPECT_NEAR(firstShaded.blue(), secondShaded.blue(), 3);
}

int main(int argc, char **argv)
{
    QGuiApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
