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
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
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
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const qsizetype order = source.indexOf(QStringLiteral("farToNearCameraIndices"));
    const qsizetype plane_loop = source.indexOf(QStringLiteral("camera_draw_order"), order);
    const qsizetype label_loop = source.indexOf(QStringLiteral("camera_label_order"), plane_loop);

    EXPECT_GE(order, 0);
    EXPECT_GT(plane_loop, order);
    EXPECT_GT(label_loop, plane_loop);
}

TEST(CameraSceneRenderContractTest, ThumbnailPlanesUseTheSceneDepthBuffer)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
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
    EXPECT_GT(renderBlock.indexOf(QStringLiteral("drawCameraThumbnails(cb")),
              renderBlock.indexOf(QStringLiteral("drawSceneGeometry(cb")));

    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);
    ASSERT_GT(overlayEnd, overlayStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawImageOnCameraPlane")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("painter.drawPolygon(imagePlane)")));
}

TEST(CameraSceneRenderContractTest, SolidCameraCardsUseDepthTestedGpuResourcesWhenThumbnailsAreDisabled)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), ensureStart);
    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"), drawStart);
    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);

    ASSERT_GE(ensureStart, 0);
    ASSERT_GT(drawStart, ensureStart);
    ASSERT_GE(overlayStart, 0);
    ASSERT_GT(overlayEnd, overlayStart);
    const QString ensureBlock = source.mid(ensureStart, drawStart - ensureStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("CameraImagePlaneMode::Solid")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("QImage(QSize(1, 1), QImage::Format_RGBA8888)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("image.fill(highlighted ? QColor")));
    EXPECT_FALSE(ensureBlock.contains(QStringLiteral(".rgba()")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("QPolygonF cameraCard")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("painter.drawPolygon(cameraCard)")));
}

TEST(CameraSceneRenderContractTest, SelectedCameraCardUsesRedHighlight)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), ensureStart);

    ASSERT_GE(ensureStart, 0);
    ASSERT_GT(drawStart, ensureStart);
    const QString ensureBlock = source.mid(ensureStart, drawStart - ensureStart);
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("isCameraHighlighted(pose)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("QColor(57, 112, 173, 220)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("QColor(205, 60, 70, 230)")));
}

TEST(CameraSceneRenderContractTest, CameraCardsUseScreenSizeAndExternalBlackDirectionLeader)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mathSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneViewMath.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("QLineF cameraDirectionLeaderLine")));
    EXPECT_FALSE(header.contains(QStringLiteral("drawCameraDirectionArrow")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraPlaneHalfExtentForScreenSize(")));
    EXPECT_FALSE(source.contains(QStringLiteral("cameraPlaneHalfExtentForViewDepth(")));
    EXPECT_FALSE(source.contains(QStringLiteral("qMax(0.1f, _zoomScale)")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::pow(1.10, wheel_steps)")));
    EXPECT_TRUE(mathSource.contains(
        QStringLiteral("CameraCardZoomResponseExponent = 0.25")));

    const qsizetype extentStart = source.indexOf(
        QStringLiteral("float CameraSceneWidget::cameraImagePlaneHalfExtent"));
    const qsizetype highlightedStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::isCameraHighlighted"), extentStart);
    ASSERT_GE(extentStart, 0);
    ASSERT_GT(highlightedStart, extentStart);
    const QString extentBlock = source.mid(extentStart, highlightedStart - extentStart);
    EXPECT_TRUE(extentBlock.contains(QStringLiteral("pose.center,")));
    EXPECT_TRUE(extentBlock.contains(QStringLiteral("worldToView,")));
    EXPECT_TRUE(extentBlock.contains(QStringLiteral("_zoomScale,")));
    EXPECT_TRUE(extentBlock.contains(QStringLiteral("return screenScaledExtent;")));
    EXPECT_FALSE(extentBlock.contains(QStringLiteral("18.0f")));
    EXPECT_FALSE(extentBlock.contains(QStringLiteral("72.0f")));

    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"));
    const qsizetype imageStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawActiveCameraImage"), drawStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(imageStart, drawStart);
    const QString drawBlock = source.mid(drawStart, imageStart - drawStart);
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("pose, matrices.modelView")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "fullDynamicBufferUpdateForCurrentFrame(")));

    const qsizetype leaderStart = source.indexOf(
        QStringLiteral("QLineF CameraSceneWidget::cameraDirectionLeaderLine"));
    const qsizetype manipStart = source.indexOf(
        QStringLiteral("QVector3D CameraSceneWidget::manipCenterWorld"), leaderStart);
    ASSERT_GE(leaderStart, 0);
    ASSERT_GT(manipStart, leaderStart);
    const QString leaderBlock = source.mid(leaderStart, manipStart - leaderStart);
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("cameraForwardDirection(")));
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("pose.center - forward")));
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("cameraPlaneScreenHalfExtentPixels(")));
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("cameraPlaneLeaderLine(")));
    EXPECT_FALSE(leaderBlock.contains(QStringLiteral("projectedCorners")));
    EXPECT_FALSE(leaderBlock.contains(QStringLiteral("headLength")));

    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"));
    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);
    ASSERT_GE(overlayStart, 0);
    ASSERT_GT(overlayEnd, overlayStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_TRUE(overlayBlock.contains(QStringLiteral("cameraDirectionLeaderLine(")));
    EXPECT_TRUE(overlayBlock.contains(QStringLiteral("pose, thumbnailHalfExtent")));
    EXPECT_TRUE(overlayBlock.contains(QStringLiteral("QColor(25, 25, 25")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawCameraDirectionArrow")));
}

TEST(CameraSceneRenderContractTest, CachesImageLoadFailuresToPreventRetryStorm)
{
    const QString header = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("_cameraImageLoadFailures")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.contains(key)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.insert(key)")));
}

TEST(CameraSceneRenderContractTest, ImageModeKeepsTheFreeOrbitViewMatrix)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
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
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

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
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
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

TEST(CameraSceneRenderContractTest, ModelMenuProvidesExclusiveTiePointColorModes)
{
    const QString menuSource = readProjectFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));

    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionGroupTiePointViewMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionTiePointColorMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionTiePointElevationMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionTiePointImageCountMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("tiePointModeGroup->setExclusive(true)")));
}

TEST(CameraSceneRenderContractTest, TiePointModesUseMetadataAndDrawLegend)
{
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mainWindowSource =
        readProjectFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));

    EXPECT_TRUE(sceneSource.contains(QStringLiteral("loadTiePointCloudFromFile")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("loadImageCountMetadata")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("TiePointColorMode::Elevation")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("TiePointColorMode::ImageCount")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("drawTiePointLegend(painter)")));
    EXPECT_TRUE(mainWindowSource.contains(
        QStringLiteral("sparse_cloud_points_json")));
    EXPECT_TRUE(mainWindowSource.contains(
        QStringLiteral("showTiePointCloudFile(path, sidecarPath)")));
}

TEST(CameraSceneRenderContractTest, TiePointOverlayHonorsCameraPlaneDepth)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("drawTiePointCloudOverlay")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_tiePointPipeline")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::Elevation")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::ImageCount")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("pointIsBehindProjectedQuad")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("CameraImagePlaneAxes")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("drawTiePointCloudOverlay(painter);")));

    const qsizetype drawStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::drawSceneGeometry"));
    const qsizetype renderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::render("), drawStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(renderStart, drawStart);
    const QString drawBlock = sceneSource.mid(drawStart, renderStart - drawStart);
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("if (!_isTiePointCloud)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "drawRhiBuffer(cb, &_pointBuffer, &_colorPointPipeline, uniforms);")));

    const qsizetype thumbnailPipelineStart = sceneSource.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype thumbnailDrawStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"),
        thumbnailPipelineStart);
    ASSERT_GE(thumbnailPipelineStart, 0);
    ASSERT_GT(thumbnailDrawStart, thumbnailPipelineStart);
    const QString thumbnailPipelineBlock = sceneSource.mid(
        thumbnailPipelineStart,
        thumbnailDrawStart - thumbnailPipelineStart);
    EXPECT_TRUE(thumbnailPipelineBlock.contains(QStringLiteral("setDepthTest(true)")));
    EXPECT_TRUE(thumbnailPipelineBlock.contains(QStringLiteral("setDepthWrite(true)")));
    EXPECT_TRUE(thumbnailPipelineBlock.contains(QStringLiteral("Format_RGBX8888")));

    const QString renderBlock = sceneSource.mid(renderStart);
    const qsizetype geometryCall = renderBlock.indexOf(
        QStringLiteral("drawSceneGeometry(cb, uniforms);"));
    const qsizetype thumbnailsCall = renderBlock.indexOf(
        QStringLiteral("drawCameraThumbnails(cb, mvp);"));
    ASSERT_GE(geometryCall, 0);
    ASSERT_GT(thumbnailsCall, geometryCall);
}

TEST(CameraSceneRenderContractTest, TiePointLoadFitsViewToLoadedGeometry)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("void fitViewToLoadedGeometry();")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("bool _fitViewAfterLoad = false;")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("bool _hasFocusedGeometryBounds = false;")));

    const qsizetype tiePointLoadStart =
        sceneSource.indexOf(QStringLiteral("void CameraSceneWidget::loadTiePointCloudFromFile"));
    const qsizetype fitViewStart =
        sceneSource.indexOf(QStringLiteral("void CameraSceneWidget::fitViewToLoadedGeometry"));
    ASSERT_GE(tiePointLoadStart, 0);
    ASSERT_GT(fitViewStart, tiePointLoadStart);
    const QString tiePointLoadBlock =
        sceneSource.mid(tiePointLoadStart, fitViewStart - tiePointLoadStart);
    EXPECT_TRUE(tiePointLoadBlock.contains(
        QStringLiteral("loadModelFromPlyInternal(pointCloudPath, true, true);")));
    EXPECT_TRUE(tiePointLoadBlock.contains(
        QStringLiteral("loadModelFromObjInternal(pointCloudPath, true, true);")));
    EXPECT_TRUE(tiePointLoadBlock.contains(
        QStringLiteral("loadPointCloudFromXyzInternal(pointCloudPath, true, true);")));

    const qsizetype plyLoadStart =
        sceneSource.indexOf(QStringLiteral("void CameraSceneWidget::loadModelFromPlyInternal"));
    const qsizetype plyFutureStart =
        sceneSource.indexOf(QStringLiteral("watcher->setFuture"), plyLoadStart);
    const qsizetype plyTiePointState =
        sceneSource.indexOf(QStringLiteral("_isTiePointCloud = tiePointCloud;"), plyLoadStart);
    const qsizetype plyFitState =
        sceneSource.indexOf(QStringLiteral("_fitViewAfterLoad = fitAfterLoad;"), plyLoadStart);
    ASSERT_GE(plyLoadStart, 0);
    ASSERT_GT(plyFutureStart, plyLoadStart);
    EXPECT_GT(plyTiePointState, plyLoadStart);
    EXPECT_LT(plyTiePointState, plyFutureStart);
    EXPECT_GT(plyFitState, plyLoadStart);
    EXPECT_LT(plyFitState, plyFutureStart);

    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "if (self->_fitViewAfterLoad)\n"
        "                {\n"
        "                    self->fitViewToLoadedGeometry();")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "if (_hasFocusedGeometryBounds)\n"
        "    {\n"
        "        return _focusedGeometryCenter;")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "if (_hasFocusedGeometryBounds)\n"
        "    {\n"
        "        return _focusedGeometryRadius;")));
}

TEST(CameraSceneRenderContractTest, PointCloudUsesOwnBoundsAndPixelSizedFloorPivot)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("bool       _hasCloudBounds = false;")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "axisAlignedBoundingBoxLineVertices(\n"
        "                _cachedCloudAABBMin, _cachedCloudAABBMax)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "constexpr qreal half_size_pixels = 7.0;")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral(
        "qMax(0.25f, _cachedRadius * 0.045f)")));
}

TEST(CameraSceneRenderContractTest, ModelMenuGatesUnsupportedModesAndDefaultsToShaded)
{
    const QString menuSource = readProjectFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString mainWindowSource =
        readProjectFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));

    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionGroupModelSurfaceViewMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelTextureMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelShadedMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelSolidMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelWireframeMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelElevationMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelConfidenceMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("actionModelAssignedImageMode")));
    EXPECT_TRUE(menuSource.contains(QStringLiteral("modelModeGroup->setExclusive(true)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelTextureModeAct->setEnabled(false)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelConfidenceModeAct->setEnabled(false)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelAssignedImageModeAct->setEnabled(false)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelShadedModeAct->setChecked(true)")));
    EXPECT_TRUE(sceneHeader.contains(
        QStringLiteral("_modelColorMode = ModelColorMode::Shaded")));
    EXPECT_TRUE(mainWindowSource.contains(QStringLiteral("setModelColorMode")));
}

TEST(CameraSceneRenderContractTest, ModelModesRebuildGeometryAndDrawLegends)
{
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString modelSource =
        readProjectFile(QStringLiteral("src/gui/views/ModelVisualization.cpp"));
    const QString vertexShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_mesh.vert"));
    const QString fragmentShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_mesh.frag"));

    EXPECT_TRUE(sceneSource.contains(QStringLiteral("_preparedObjVertexData")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::Shaded")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::Solid")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("_pipelinesDirty = true;")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::Wireframe")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::Elevation")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::Confidence")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("ModelColorMode::AssignedImage")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("_modelWireframeBuffer")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("evaluateFaceSupport")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("drawModelLegend(painter)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("_modelVisualization.buildGeometry")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("geometryInput.vertexNormals")));
    EXPECT_TRUE(modelSource.contains(QStringLiteral("ModelVisualizationManager::buildGeometry")));
    EXPECT_FALSE(modelSource.contains(QStringLiteral("evaluateFaceSupport")));
    EXPECT_TRUE(modelSource.contains(QStringLiteral("displayVertexNormals")));
    EXPECT_TRUE(modelSource.contains(QStringLiteral("generatedVertexNormals")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("vViewPosition")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("dot(n, viewDir) < 0.0")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("0.18 + 0.72 * diffuse")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("fillDiffuse")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("specular")));
}
