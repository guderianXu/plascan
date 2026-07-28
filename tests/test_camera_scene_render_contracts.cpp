#include <QDir>
#include <QFile>
#include <QString>

#include <gtest/gtest.h>

namespace
{

QString readProjectFile(const QString &relativePath)
{
    QFile file(QDir(QString::fromUtf8(PLASCAN_SOURCE_DIR)).filePath(relativePath));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

} // namespace

TEST(CameraSceneRenderContractTest, RegistersQrhiCameraImageShaders)
{
    const QString cmake = readProjectFile(QStringLiteral("src/gui/CMakeLists.txt"));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_image.vert")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_image.frag")));
}

TEST(CameraSceneRenderContractTest, DrawsBackgroundBeforeGeometryAndForegroundAfterGeometry)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const qsizetype background = source.indexOf(
        QStringLiteral("_cameraImageDisplayLayer == CameraImageDisplayLayer::Background"));
    const qsizetype first_image_draw = source.indexOf(QStringLiteral("drawActiveCameraImage(cb"), background);
    const qsizetype geometry_draw = source.indexOf(QStringLiteral("drawSceneGeometry(cb"), first_image_draw);
    const qsizetype foreground = source.indexOf(
        QStringLiteral("_cameraImageDisplayLayer == CameraImageDisplayLayer::Foreground"), geometry_draw);
    const qsizetype second_image_draw = source.indexOf(QStringLiteral("drawActiveCameraImage(cb"), foreground);

    EXPECT_GE(background, 0);
    EXPECT_GT(first_image_draw, background);
    EXPECT_GT(geometry_draw, first_image_draw);
    EXPECT_GT(foreground, geometry_draw);
    EXPECT_GT(second_image_draw, foreground);
}

TEST(CameraSceneRenderContractTest, SortsThumbnailPlanesFarToNearBeforeDrawingLabels)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const qsizetype order = source.indexOf(QStringLiteral("farToNearCameraIndices"));
    const qsizetype plane_loop = source.indexOf(QStringLiteral("camera_draw_order"), order);
    const qsizetype label_loop = source.indexOf(QStringLiteral("camera_label_order"), plane_loop);

    EXPECT_GE(order, 0);
    EXPECT_GT(plane_loop, order);
    EXPECT_GT(label_loop, plane_loop);
}

TEST(CameraSceneRenderContractTest, ThumbnailPlanesUseTheSceneDepthBuffer)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), ensureStart);
    const qsizetype renderStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::render(QRhiCommandBuffer *cb)"), drawStart);
    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"), renderStart);

    ASSERT_GE(ensureStart, 0);
    ASSERT_GT(drawStart, ensureStart);
    ASSERT_GT(renderStart, drawStart);
    ASSERT_GT(overlayStart, renderStart);

    const QString ensureBlock = source.mid(ensureStart, drawStart - ensureStart);
    const QString renderBlock = source.mid(renderStart, overlayStart - renderStart);
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("setDepthTest(true)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("setDepthWrite(true)")));
    EXPECT_LT(renderBlock.indexOf(QStringLiteral("drawCameraThumbnails(cb")),
              renderBlock.indexOf(QStringLiteral("drawSceneGeometry(cb")));

    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);
    ASSERT_GT(overlayEnd, overlayStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawImageOnCameraPlane")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("painter.drawPolygon(imagePlane)")));
}

TEST(CameraSceneRenderContractTest, CachesImageLoadFailuresToPreventRetryStorm)
{
    const QString header = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.h"));
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("_cameraImageLoadFailures")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.contains(key)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.insert(key)")));
}

TEST(CameraSceneRenderContractTest, ImageModeKeepsTheFreeOrbitViewMatrix)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const qsizetype start = source.indexOf(QStringLiteral("CameraSceneWidget::SceneMatrices CameraSceneWidget::sceneMatrices() const"));
    const qsizetype end = source.indexOf(QStringLiteral("QPointF CameraSceneWidget::projectToScreen"), start);

    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(block.contains(QStringLiteral("model.rotate(_viewRot)")));
    EXPECT_FALSE(block.contains(QStringLiteral("displayedCameraImagePoseIndex")));
    EXPECT_FALSE(block.contains(QStringLiteral("_showCameraImage")));
}

TEST(CameraSceneRenderContractTest, CameraImageShaderProjectsAWorldSpacePlane)
{
    const QString vertexShader = readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_image.vert"));
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(vertexShader.contains(QStringLiteral("layout(location = 0) in vec3 position")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("uniform ImagePlaneUniforms")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("uMVP * vec4(position, 1.0)")));
    EXPECT_TRUE(source.contains(QStringLiteral("calibratedImagePlaneCorners")));
    EXPECT_FALSE(source.contains(QStringLiteral("const float image_half_extent")));
    EXPECT_FALSE(source.contains(QStringLiteral("thumbnail_half_extent * 5.2f")));
    EXPECT_FALSE(source.contains(QStringLiteral("activeCameraImageViewportScale")));
}

TEST(CameraSceneRenderContractTest, AutomaticImageModeHasNoFirstPhotoFallback)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const qsizetype updateStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::updateActiveCameraForView()"));
    const qsizetype displayedStart = source.indexOf(
        QStringLiteral("int CameraSceneWidget::displayedCameraImagePoseIndex() const"), updateStart);
    const qsizetype refreshStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::refreshLockedCameraImage()"), displayedStart);

    ASSERT_GE(updateStart, 0);
    ASSERT_GT(displayedStart, updateStart);
    ASSERT_GT(refreshStart, displayedStart);

    const QString updateBlock = source.mid(updateStart, displayedStart - updateStart);
    const QString displayedBlock = source.mid(displayedStart, refreshStart - displayedStart);
    const qsizetype selectionCall = updateBlock.indexOf(QStringLiteral("selectCameraForView("));
    ASSERT_GE(selectionCall, 0);
    EXPECT_EQ(updateBlock.indexOf(QStringLiteral("_poses.at(index).imagePath"), selectionCall), -1);
    EXPECT_FALSE(displayedBlock.contains(QStringLiteral("for (qsizetype index")));
}

TEST(CameraSceneRenderContractTest, MainWorkspaceCopiesCompleteCameraDisplayPose)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("camera.intrinsics()")));
    EXPECT_TRUE(source.contains(QStringLiteral("pose.focalX")));
    EXPECT_TRUE(source.contains(QStringLiteral("pose.imageWidth")));
    EXPECT_TRUE(source.contains(QStringLiteral("pose.uAxisSign")));
    EXPECT_TRUE(source.contains(QStringLiteral("pose.depthAxisFlipped = camera.depthAxisFlipped()")));
}
