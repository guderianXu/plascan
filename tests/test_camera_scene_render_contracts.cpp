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

TEST(CameraSceneRenderContractTest, AvoidsPerFrameSortingForOpaqueDepthWrittenThumbnails)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"));
    const qsizetype nextFunction = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawActiveCameraImage"), drawStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(nextFunction, drawStart);
    const QString drawBlock = source.mid(drawStart, nextFunction - drawStart);

    EXPECT_TRUE(drawBlock.contains(QStringLiteral("已加载照片按纹理")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("farToNearCameraIndices")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("camera_draw_order")));
}

TEST(CameraSceneRenderContractTest, LargeCameraImagesUseContinuousBoundedPrefetch)
{
    const QString header = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString imageLoader = readProjectFile(QStringLiteral("src/gui/views/LayerImageLoader.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(imageLoader.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QThreadPool _cameraImageLoadPool")));
    EXPECT_TRUE(header.contains(QStringLiteral("QQueue<CameraPlaneImageRequest> _cameraImageLoadQueue")));
    EXPECT_TRUE(source.contains(QStringLiteral("pumpCameraPlaneImageLoads();")));
    EXPECT_TRUE(source.contains(QStringLiteral("&_cameraImageLoadPool")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::clamp((ideal_threads + 1) / 2, 4, 12)")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cameraImageLoadsInFlight.size() >= 6")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cameraImageCache.size() > 512")));
    EXPECT_TRUE(source.contains(QStringLiteral("imagePath, QString(), targetSize, &source_size")));
    EXPECT_TRUE(imageLoader.contains(QStringLiteral("reader.setScaledSize(scaled_size)")));
    EXPECT_TRUE(header.contains(QStringLiteral("drawCameraThumbnailProgressOverlay")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在加载相机影像 %1/%2")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiCameraThumbnailAtlasPage")));
    EXPECT_TRUE(source.contains(QStringLiteral("subresource.setDestinationTopLeft")));
    EXPECT_TRUE(source.contains(QStringLiteral("atlas_vertices")));
}

TEST(CameraSceneRenderContractTest, ProjectCameraPosesAreParsedOffTheGuiThread)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("QtConcurrent::run([images]()")));
    EXPECT_TRUE(source.contains(QStringLiteral("cameraPosesFromImages(images)")));
    EXPECT_TRUE(source.contains(QStringLiteral("generation == self->_cameraPoseGeneration")));
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

TEST(CameraSceneRenderContractTest, ForegroundImageOccludesGpuAndOverlaySceneContent)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(header.contains(
        QStringLiteral("QVector<QVector3D> displayedCameraImagePlaneCorners() const")));
    EXPECT_TRUE(header.contains(
        QStringLiteral("QPainterPath foregroundCameraImageOcclusionPath() const")));
    EXPECT_TRUE(source.contains(QStringLiteral("Format_RGBX8888")));

    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"));
    const qsizetype floorCross = source.indexOf(
        QStringLiteral("drawFloorPivotCross(painter);"), overlayStart);
    ASSERT_GE(overlayStart, 0);
    ASSERT_GT(floorCross, overlayStart);
    const QString overlayBlock = source.mid(overlayStart, floorCross - overlayStart);
    EXPECT_TRUE(overlayBlock.contains(
        QStringLiteral("foregroundCameraImageOcclusionPath()")));
    EXPECT_TRUE(overlayBlock.contains(
        QStringLiteral("visibleScene.subtracted(foregroundImageOcclusion)")));
    EXPECT_TRUE(overlayBlock.contains(
        QStringLiteral("painter.setClipPath(visibleScene, Qt::IntersectClip)")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawPointCloudOverlay(painter);")));
}

TEST(CameraSceneRenderContractTest, SolidCameraCardsUseBatchedDepthTestedGpuResources)
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
    const QString drawBlock = source.mid(drawStart, overlayStart - drawStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("ensureSolidCameraBatchResource(")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("setDepthTest(true)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("setDepthWrite(true)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("draw_batch(")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("solid_vertices")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("atlas_vertices")));
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
    const qsizetype drawEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawActiveCameraImage"), drawStart);
    ASSERT_GT(drawEnd, drawStart);
    const QString drawBlock = source.mid(drawStart, drawEnd - drawStart);
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("QColor(57, 112, 173)")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("QColor(205, 60, 70)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("isCameraHighlighted(pose)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("highlightedSolidResource")));
}

TEST(CameraSceneRenderContractTest, CameraCardsAndDirectionLeadersShareVulkanGeometry)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString mathSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneViewMath.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("bool cameraDirectionLeaderSegment")));
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
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("_cameraLeaderPipeline.pipeline")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("leader_vertices")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("draw_batch")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("atlas_vertices")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("_leftDragging")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("_middleDragging")));

    const qsizetype leaderStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::cameraDirectionLeaderSegment"));
    const qsizetype manipStart = source.indexOf(
        QStringLiteral("QVector3D CameraSceneWidget::manipCenterWorld"), leaderStart);
    ASSERT_GE(leaderStart, 0);
    ASSERT_GT(manipStart, leaderStart);
    const QString leaderBlock = source.mid(leaderStart, manipStart - leaderStart);
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("cameraForwardDirection(")));
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("*start = pose.center")));
    EXPECT_TRUE(leaderBlock.contains(QStringLiteral("pose.center - forward.normalized()")));
    EXPECT_FALSE(leaderBlock.contains(QStringLiteral("projectToScreen(")));

    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay(QPainter &painter)"));
    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);
    ASSERT_GE(overlayStart, 0);
    ASSERT_GT(overlayEnd, overlayStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_TRUE(overlayBlock.contains(QStringLiteral("cameraDirectionLeaderSegment(")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("cameraPlaneOcclusionPath")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("cameraLeaderClip")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawCameraDirectionArrow")));
}

TEST(CameraSceneRenderContractTest, DeduplicatesCameraPosesByImageIdentity)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const qsizetype start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setCameraPoses"));
    const qsizetype end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setShowGizmo"), start);

    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(block.contains(QStringLiteral("QSet<QString> cameraKeys")));
    EXPECT_TRUE(block.contains(QStringLiteral("cameraKeys.contains(cameraKey)")));
    EXPECT_TRUE(block.contains(QStringLiteral("_poses.push_back(pose)")));
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
        readProjectFile(QStringLiteral(
            "src/gui/main_window/MainWindowProjectBindings.cpp"));

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

TEST(CameraSceneRenderContractTest, TiePointsUseGpuCameraPlaneDepth)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("drawPointCloudOverlay")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_tiePointPipeline")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::Elevation")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::ImageCount")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("pointIsBehindProjectedQuad")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("CameraImagePlaneAxes")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("drawPointCloudOverlay(painter);")));

    const qsizetype drawStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::drawSceneGeometry"));
    const qsizetype renderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::render("), drawStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(renderStart, drawStart);
    const QString drawBlock = sceneSource.mid(drawStart, renderStart - drawStart);
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("if (!_isTiePointCloud)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("drawPointCloud(cb, uniforms);")));

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
        QStringLiteral("drawCameraThumbnails(cb, mvp, uniforms);"));
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
        QStringLiteral("loadModelFromPlyInternal(pointCloudPath, true, true, true);")));
    EXPECT_TRUE(tiePointLoadBlock.contains(
        QStringLiteral("loadModelFromObjInternal(pointCloudPath, true, true, true);")));
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

TEST(CameraSceneRenderContractTest, ImportedPointCloudsUsePointCloudLoadersAndFitView)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString workspaceSource =
        readProjectFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("void loadPointCloudFromPly(const QString &plyPath);")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("void loadPointCloudFromObj(const QString &objPath);")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "loadModelFromPlyInternal(plyPath, false, true, true);")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "loadModelFromObjInternal(objPath, false, true, true);")));

    const qsizetype pointCloudStart = workspaceSource.indexOf(
        QStringLiteral("void WorkspaceCenterWidget::showPointCloudFile"));
    const qsizetype tiePointStart = workspaceSource.indexOf(
        QStringLiteral("void WorkspaceCenterWidget::showTiePointCloudFile"), pointCloudStart);
    ASSERT_GE(pointCloudStart, 0);
    ASSERT_GT(tiePointStart, pointCloudStart);
    const QString pointCloudBlock = workspaceSource.mid(
        pointCloudStart,
        tiePointStart - pointCloudStart);
    EXPECT_TRUE(pointCloudBlock.contains(QStringLiteral("loadPointCloudFromPly(pointCloudPath)")));
    EXPECT_TRUE(pointCloudBlock.contains(QStringLiteral("loadPointCloudFromObj(pointCloudPath)")));
    EXPECT_TRUE(pointCloudBlock.contains(QStringLiteral("loadPointCloudFromXyz(pointCloudPath)")));
    EXPECT_TRUE(pointCloudBlock.contains(QStringLiteral("showModelView()")));
    EXPECT_FALSE(pointCloudBlock.contains(QStringLiteral("showModelFile(pointCloudPath)")));

    const qsizetype objLoaderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::loadModelFromObjInternal"));
    const qsizetype tiePointLoaderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::loadTiePointCloudFromFile"), objLoaderStart);
    ASSERT_GE(objLoaderStart, 0);
    ASSERT_GT(tiePointLoaderStart, objLoaderStart);
    const QString objLoaderBlock = sceneSource.mid(
        objLoaderStart,
        tiePointLoaderStart - objLoaderStart);
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral("if (pointCloudResource)")));
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral("plapoint::io::readObj<float>")));
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral(
        "if (!pointCloudResource && !result.textureWarning.isEmpty())")));

    const qsizetype uploadStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::uploadGpuData"));
    const qsizetype pipelineStart = sceneSource.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensurePipeline"), uploadStart);
    ASSERT_GE(uploadStart, 0);
    ASSERT_GT(pipelineStart, uploadStart);
    const QString uploadBlock = sceneSource.mid(uploadStart, pipelineStart - uploadStart);
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral(
        "if (!(_cloud.size() == 0) && !_cloud.hasFaces())")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral(
        "!_cloud.hasFaces() && !_cloud.hasNormals()")));
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral(
        "constexpr float pointColorScale = 1.0f / 255.0f")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("maximumColorComponent")));
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral(
        "assignBuffer(_pointBuffer, data, int(_cloud.size()), 9)")));
    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("_modelPointBuffer")));
    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("_modelPointPipeline")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_preferModelPointRender")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("drawPointCloud(cb, uniforms);")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("drawPointCloudOverlay(painter);")));
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

TEST(CameraSceneRenderContractTest, ModelMenuEnablesTextureAndGatesUnsupportedModes)
{
    const QString menuSource = readProjectFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/dialogs/camera/CameraModel3DDialog.cpp"));
    const QString menuBindingsSource =
        readProjectFile(QStringLiteral(
            "src/gui/main_window/MainWindowMenuBindings.cpp"));

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
        QStringLiteral("_modelTextureModeAct->setEnabled(true)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelConfidenceModeAct->setEnabled(false)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelAssignedImageModeAct->setEnabled(false)")));
    EXPECT_TRUE(menuSource.contains(
        QStringLiteral("_modelShadedModeAct->setChecked(true)")));
    EXPECT_TRUE(sceneHeader.contains(
        QStringLiteral("_modelColorMode = ModelColorMode::Shaded")));
    EXPECT_TRUE(menuBindingsSource.contains(QStringLiteral("setModelColorMode")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "self->setModelColorMode(ModelColorMode::Texture)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "self->setModelColorMode(ModelColorMode::Solid)")));
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
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("isNeutralSurface")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("neutralShape")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("dot(n, viewDir) < 0.0")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "0.86 + 0.10 * keyDiffuse + 0.04 * headDiffuse")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "Photograph-derived vertex colours")));
    EXPECT_FALSE(fragmentShader.contains(QStringLiteral("0.18 + 0.72 * diffuse")));
}
