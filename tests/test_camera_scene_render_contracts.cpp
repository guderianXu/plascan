#include <QDir>
#include <QFile>
#include <QString>

#include <gtest/gtest.h>

namespace
{

QString readProjectFile(const QString &relativePath)
{
    const QDir projectRoot(QString::fromUtf8(PLASCAN_SOURCE_DIR));
    const auto readOne = [&projectRoot](const QString &path)
    {
        QFile file(projectRoot.filePath(path));
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            return QString();
        }
        return QString::fromUtf8(file.readAll());
    };

    QString source = readOne(relativePath);
    if (relativePath == QStringLiteral("src/gui/views/CameraSceneWidget.cpp"))
    {
        source += QLatin1Char('\n')
            + readOne(QStringLiteral("src/gui/views/CameraSceneWidgetOverlay.cpp"));
        source += QLatin1Char('\n')
            + readOne(QStringLiteral("src/gui/views/CameraSceneWidgetLegends.cpp"));
    }
    return source;
}

} // namespace

TEST(CameraSceneRenderContractTest, RegistersQrhiCameraImageShaders)
{
    const QString cmake = readProjectFile(QStringLiteral("src/gui/CMakeLists.txt"));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_image.vert")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_image.frag")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_camera.vert")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("shaders/camera_scene_camera_leader.vert")));
}

TEST(CameraSceneRenderContractTest, DrawsBackgroundBeforeGeometryAndForegroundAfterGeometry)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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

TEST(CameraSceneInteractionContractTest, UsesMetashapeStyleNavigationOutsideTheGizmo)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString viewMath =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneViewMath.cpp"));
    const QString taskStatus =
        readProjectFile(QStringLiteral("src/gui/main_window/MainWindowTaskStatus.cpp"));
    const qsizetype pressStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::mousePressEvent"));
    const qsizetype moveStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::mouseMoveEvent"), pressStart);
    const qsizetype releaseStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::mouseReleaseEvent"), moveStart);
    ASSERT_GE(pressStart, 0);
    ASSERT_GT(moveStart, pressStart);
    ASSERT_GT(releaseStart, moveStart);
    const QString pressBlock = source.mid(pressStart, moveStart - pressStart);
    const QString moveBlock = source.mid(moveStart, releaseStart - moveStart);

    EXPECT_TRUE(header.contains(QStringLiteral("enum class LeftDragMode")));
    EXPECT_TRUE(header.contains(QStringLiteral("Pan,")));
    EXPECT_TRUE(header.contains(QStringLiteral("Orbit,")));
    EXPECT_TRUE(header.contains(QStringLiteral("GizmoOrbit,")));
    EXPECT_TRUE(pressBlock.contains(QStringLiteral(
        "_leftDragMode = LeftDragMode::Pan")));
    EXPECT_TRUE(pressBlock.contains(QStringLiteral(
        "_leftDragMode = LeftDragMode::Orbit")));
    EXPECT_TRUE(pressBlock.contains(QStringLiteral("_rightDragging = true")));
    EXPECT_TRUE(pressBlock.contains(QStringLiteral("isInsideRotationGizmo")));
    EXPECT_TRUE(moveBlock.contains(QStringLiteral(
        "_manualSelecting && (event->buttons() & Qt::LeftButton)")));
    EXPECT_TRUE(moveBlock.contains(QStringLiteral("applyOrbitDrag(delta)")));
    EXPECT_TRUE(viewMath.contains(QStringLiteral("orbitSceneViewRotation")));
    EXPECT_TRUE(taskStatus.contains(QStringLiteral("鼠标左键拖拽框选")));
    EXPECT_TRUE(taskStatus.contains(QStringLiteral("右键拖拽可环绕查看")));
}

TEST(CameraSceneRenderContractTest, AvoidsPerFrameSortingForOpaqueDepthWrittenThumbnails)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"));
    const qsizetype nextFunction = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawActiveCameraImage"), drawStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(nextFunction, drawStart);
    const QString drawBlock = source.mid(drawStart, nextFunction - drawStart);

    EXPECT_TRUE(drawBlock.contains(QStringLiteral("draw_instances")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("farToNearCameraIndices")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("camera_draw_order")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("for (qsizetype pose_index")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("cameraImagePlaneHalfExtent")));
}

TEST(CameraSceneRenderContractTest, LargeCameraImagesUseContinuousBoundedPrefetch)
{
    const QString header = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString imageLoader = readProjectFile(QStringLiteral("src/gui/views/LayerImageLoader.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(imageLoader.isEmpty());

    EXPECT_FALSE(header.contains(QStringLiteral("QThreadPool _cameraImageLoadPool")));
    EXPECT_TRUE(header.contains(QStringLiteral("int _maximumCameraImageLoads = 4")));
    EXPECT_TRUE(header.contains(QStringLiteral("QQueue<CameraPlaneImageRequest> _cameraImageLoadQueue")));
    EXPECT_TRUE(source.contains(QStringLiteral("pumpCameraPlaneImageLoads();")));
    EXPECT_TRUE(source.contains(QStringLiteral("watcher->setFuture(QtConcurrent::run(")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cameraImageLoadPool.waitForDone()")));
    EXPECT_TRUE(source.contains(QStringLiteral("std::clamp((ideal_threads + 1) / 2, 4, 12)")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cameraImageLoadsInFlight.size() >= 6")));
    EXPECT_FALSE(source.contains(QStringLiteral("_cameraImageCache.size() > 512")));
    EXPECT_TRUE(source.contains(QStringLiteral("imagePath, QString(), targetSize, &source_size")));
    EXPECT_TRUE(imageLoader.contains(QStringLiteral("reader.setScaledSize(scaled_size)")));
    EXPECT_TRUE(header.contains(QStringLiteral("drawCameraThumbnailProgressOverlay")));
    EXPECT_TRUE(source.contains(QStringLiteral("正在加载相机影像 %1/%2")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiCameraThumbnailAtlasPage")));
    EXPECT_TRUE(source.contains(QStringLiteral("subresource.setDestinationTopLeft")));
    EXPECT_TRUE(source.contains(QStringLiteral("atlas_instances")));
    EXPECT_TRUE(source.contains(QStringLiteral("QRhiVertexInputBinding::PerInstance")));
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
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

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
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("draw_instances")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("instanceBuffer")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("cb->draw(6")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("QPolygonF cameraCard")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("painter.drawPolygon(cameraCard)")));
}

TEST(CameraSceneRenderContractTest, SelectedCameraCardUsesRedHighlight)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("isCameraHighlighted(pose)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("highlightedSolidResource")));
}

TEST(CameraSceneRenderContractTest, CameraCardsDirectionLeadersAndLocalAxesUseBatchedGpuGeometry)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString mathSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneViewMath.cpp"));
    const QString leaderShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_camera_leader.vert"));

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
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "fullDynamicBufferUpdateForCurrentFrame(")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("leaderPipeline")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("draw_instances")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("cb->draw(2")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("segmentInstanceCount")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("leader_vertices")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("atlas_vertices")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("!isNavigationDragging()")));

    const qsizetype leaderStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::cameraDirectionLeaderSegment"));
    const qsizetype manipStart = source.indexOf(
        QStringLiteral("QPointF CameraSceneWidget::manipCenterScreen"), leaderStart);
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
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("cameraLocalAxes(")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawWorldSegment")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("cameraPlaneOcclusionPath")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("cameraLeaderClip")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("drawCameraDirectionArrow")));

    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    ASSERT_GE(ensureStart, 0);
    ASSERT_GT(drawStart, ensureStart);
    const QString ensureBlock = source.mid(ensureStart, drawStart - ensureStart);
    EXPECT_TRUE(header.contains(QStringLiteral("int segmentInstanceCount = 0")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("segment_instances")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("cameraLocalAxes(")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("append_segment(")));
    EXPECT_TRUE(leaderShader.contains(QStringLiteral("normalize(instanceForward)")));
    EXPECT_TRUE(leaderShader.contains(QStringLiteral("instanceUvRect.xyz")));
    EXPECT_TRUE(leaderShader.contains(QStringLiteral("instanceUvRect.w")));
}

TEST(CameraSceneRenderContractTest, LargePointCloudsPrepareGpuVerticesOffTheGuiThread)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString preparationSource =
        readProjectFile(QStringLiteral("src/gui/views/SceneGeometryPreparation.cpp"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(preparationSource.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("SceneLoadTaskResult prepareSceneLoad")));
    EXPECT_FALSE(source.contains(QStringLiteral("cloneRenderCloud")));
    EXPECT_FALSE(source.contains(QStringLiteral("CameraSceneWidget::setPointCloud")));
    EXPECT_TRUE(source.contains(QStringLiteral("prepareSceneLoad(std::move(cloud),")));
    EXPECT_TRUE(source.contains(QStringLiteral("_preparedPointVertexData")));
    EXPECT_TRUE(source.contains(QStringLiteral("use_prepared_point_buffer")));
    EXPECT_TRUE(source.contains(QStringLiteral("*result.cloud, {}, cancellationFlag")));
    EXPECT_TRUE(preparationSource.contains(QStringLiteral("prepareCloudSpatialSummary")));
    EXPECT_TRUE(preparationSource.contains(QStringLiteral(
        "scalar_output[index] = has_image_counts")));
    EXPECT_FALSE(preparationSource.contains(QStringLiteral(
        "cloud.size(), imageCounts, nullptr, cancellationFlag")));
    EXPECT_TRUE(preparationSource.contains(QStringLiteral("std::nth_element(")));
    EXPECT_FALSE(preparationSource.contains(QStringLiteral("std::sort(distances.begin(), distances.end())")));

    const qsizetype uploadStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::uploadGpuData"));
    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureRhiBuffer"), uploadStart);
    ASSERT_GE(uploadStart, 0);
    ASSERT_GT(ensureStart, uploadStart);
    const QString uploadBlock = source.mid(uploadStart, ensureStart - uploadStart);
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("for (std::size_t")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("prepareObjRenderData(")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("preparePointScalarData(")));
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral("不会在 GUI 线程回退重建")));
}

TEST(CameraSceneRenderContractTest, UploadedPointChunksRemainReusableAfterCpuCopiesAreReleased)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype ensure_start = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureRhiBuffer"));
    const qsizetype ensure_end = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureRhiIndexBuffer"), ensure_start);
    const qsizetype commit_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::commitResourceUpdateState"));
    const qsizetype commit_end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::resizeEvent"), commit_start);
    ASSERT_GE(ensure_start, 0);
    ASSERT_GT(ensure_end, ensure_start);
    ASSERT_GE(commit_start, 0);
    ASSERT_GT(commit_end, commit_start);

    const QString ensure_block = source.mid(ensure_start, ensure_end - ensure_start);
    const QString commit_block = source.mid(commit_start, commit_end - commit_start);
    EXPECT_TRUE(ensure_block.contains(QStringLiteral("if (buffer->vertexCount <= 0)")));
    EXPECT_TRUE(ensure_block.contains(QStringLiteral("if (buffer->vertexData.isEmpty())")));
    EXPECT_TRUE(ensure_block.contains(QStringLiteral(
        "return buffer->vertexBuffer && !buffer->dirty;")));
    EXPECT_TRUE(commit_block.contains(QStringLiteral("chunk->points.vertexData.clear()")));
    EXPECT_TRUE(commit_block.contains(QStringLiteral("chunk->scalars.vertexData.clear()")));
}

TEST(CameraSceneRenderContractTest, OversizedMeshTexturesArePreparedOffTheRenderThread)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype prepare_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::prepareMeshTextureUploadImage"));
    const qsizetype ensure_start = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureTexturedMeshPipeline"), prepare_start);
    const qsizetype ensure_end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawTexturedMesh"), ensure_start);
    ASSERT_GE(prepare_start, 0);
    ASSERT_GT(ensure_start, prepare_start);
    ASSERT_GT(ensure_end, ensure_start);

    const QString prepare_block = source.mid(prepare_start, ensure_start - prepare_start);
    const QString ensure_block = source.mid(ensure_start, ensure_end - ensure_start);
    EXPECT_TRUE(prepare_block.contains(QStringLiteral("runGuardedWithOutcome(")));
    EXPECT_TRUE(prepare_block.contains(QStringLiteral("source_image.scaled(")));
    EXPECT_FALSE(ensure_block.contains(QStringLiteral("_meshTextureImage.scaled(")));
    EXPECT_TRUE(ensure_block.contains(QStringLiteral(
        "prepareMeshTextureUploadImage(maximumTextureSize)")));
}

TEST(CameraSceneRenderContractTest, SceneLoadsAreSingleFlightAndLatestOnly)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype pump_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::pumpSceneLoad"));
    const qsizetype pump_end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::loadTiePointCloudFromFile"), pump_start);
    ASSERT_GE(pump_start, 0);
    ASSERT_GT(pump_end, pump_start);
    const QString pump_block = source.mid(pump_start, pump_end - pump_start);

    EXPECT_TRUE(header.contains(QStringLiteral("std::optional<SceneLoadRequest> _pendingSceneLoad")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _sceneLoadWorkerActive = false")));
    EXPECT_TRUE(pump_block.contains(QStringLiteral(
        "if (_sceneLoadWorkerActive || !_pendingSceneLoad)")));
    EXPECT_TRUE(pump_block.contains(QStringLiteral("_pendingSceneLoad.reset()")));
    EXPECT_TRUE(pump_block.contains(QStringLiteral("runGuardedWithOutcome(")));
    EXPECT_TRUE(pump_block.contains(QStringLiteral("cancellation->load")));
    EXPECT_TRUE(pump_block.contains(QStringLiteral("self->pumpSceneLoad()")));
    EXPECT_FALSE(source.contains(QStringLiteral("QFutureWatcher<PointCloudLoadResult>")));
    EXPECT_FALSE(pump_block.contains(QStringLiteral("QMetaObject::invokeMethod(self.data()")));

    EXPECT_TRUE(header.contains(QStringLiteral(
        "std::optional<TiePointMetadataRequest> _pendingTiePointMetadataLoad")));
    EXPECT_TRUE(header.contains(QStringLiteral(
        "bool _tiePointMetadataWorkerActive = false")));
    const qsizetype metadata_pump_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::pumpTiePointMetadataLoad"));
    const qsizetype fit_view_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::fitViewToLoadedGeometry"),
        metadata_pump_start);
    ASSERT_GE(metadata_pump_start, 0);
    ASSERT_GT(fit_view_start, metadata_pump_start);
    const QString metadata_pump_block = source.mid(
        metadata_pump_start, fit_view_start - metadata_pump_start);
    EXPECT_TRUE(metadata_pump_block.contains(QStringLiteral(
        "if (_tiePointMetadataWorkerActive || !_pendingTiePointMetadataLoad)")));
    EXPECT_TRUE(metadata_pump_block.contains(QStringLiteral(
        "cancellation->load(std::memory_order_relaxed)")));
    EXPECT_TRUE(metadata_pump_block.contains(QStringLiteral(
        "self->pumpTiePointMetadataLoad()")));
}

TEST(CameraSceneRenderContractTest, ResourceUpdateBatchRollsBackUntilSubmitted)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype render_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::render(QRhiCommandBuffer *cb)"));
    const qsizetype render_end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay"), render_start);
    ASSERT_GE(render_start, 0);
    ASSERT_GT(render_end, render_start);
    const QString render_block = source.mid(render_start, render_end - render_start);
    const qsizetype batch_release = render_block.indexOf(QStringLiteral("updates->release()"));
    const qsizetype rollback = render_block.indexOf(QStringLiteral("rollbackResourceUpdateState()"));
    const qsizetype begin_pass = render_block.indexOf(QStringLiteral("cb->beginPass("));
    const qsizetype commit = render_block.indexOf(QStringLiteral("commitResourceUpdateState()"));

    ASSERT_GE(batch_release, 0);
    ASSERT_GT(rollback, batch_release);
    ASSERT_GE(begin_pass, 0);
    ASSERT_GT(commit, begin_pass);
    EXPECT_TRUE(render_block.contains(QStringLiteral("abort_update_batch()")));
    EXPECT_TRUE(header.contains(
        QStringLiteral("QSet<int> _thumbnailPoseIndicesPendingCommit")));
    EXPECT_TRUE(header.contains(
        QStringLiteral("QSet<QString> _thumbnailCacheKeysPendingCommit")));

    const qsizetype thumbnail_start = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype thumbnail_end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), thumbnail_start);
    ASSERT_GE(thumbnail_start, 0);
    ASSERT_GT(thumbnail_end, thumbnail_start);
    const QString thumbnail_block = source.mid(
        thumbnail_start, thumbnail_end - thumbnail_start);
    EXPECT_TRUE(thumbnail_block.contains(
        QStringLiteral("_thumbnailCacheKeysPendingCommit.insert")));
    EXPECT_FALSE(thumbnail_block.contains(QStringLiteral("_cameraImageCache.remove")));
}

TEST(CameraSceneRenderContractTest, ManualPointSelectionAndEditsRunInBackgroundHelpers)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString selectionSource =
        readProjectFile(QStringLiteral("src/gui/views/SceneGeometryPreparation.cpp"));
    const QString editHeader =
        readProjectFile(QStringLiteral("src/gui/views/PointCloudEditPreparation.h"));
    const QString editSource =
        readProjectFile(QStringLiteral("src/gui/views/PointCloudEditPreparation.cpp"));
    const QString snapshotSource =
        readProjectFile(QStringLiteral("src/gui/views/PointCloudSnapshotIO.cpp"));
    const qsizetype start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::startManualPointSelection"));
    const qsizetype end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::pushManualUndoDelta"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("const QMatrix4x4 clip_matrix")));
    EXPECT_TRUE(block.contains(QStringLiteral("runGuardedWithOutcome(")));
    EXPECT_TRUE(block.contains(QStringLiteral("preparePointSelection(request.vertexData")));
    EXPECT_TRUE(block.contains(QStringLiteral("load_generation == self->_loadGen")));
    EXPECT_TRUE(block.contains(QStringLiteral("cancellation.get()")));
    EXPECT_TRUE(block.contains(QStringLiteral("pumpManualPointSelection()")));
    EXPECT_TRUE(header.contains(
        QStringLiteral("std::optional<ManualSelectionRequest> _pendingManualSelection")));
    EXPECT_TRUE(header.contains(
        QStringLiteral("bool _manualSelectionWorkerActive = false")));
    EXPECT_TRUE(header.contains(QStringLiteral("std::vector<PointVertexIndex>")));
    EXPECT_TRUE(selectionSource.contains(QStringLiteral(
        "kMaximumInitialSelectionReserve")));
    EXPECT_FALSE(block.contains(QStringLiteral("for (std::size_t")));
    EXPECT_TRUE(selectionSource.contains(QStringLiteral("clipMatrix * QVector4D")));
    EXPECT_TRUE(editHeader.contains(QStringLiteral("struct PointCloudEditDelta")));
    EXPECT_TRUE(editSource.contains(QStringLiteral("filterPointCloudWithDelta(")));
    EXPECT_TRUE(editSource.contains(QStringLiteral("restorePointCloudFromDelta(")));
    EXPECT_TRUE(source.contains(QStringLiteral("filterPointCloudWithDelta(")));
    EXPECT_TRUE(source.contains(QStringLiteral("restorePointCloudFromDelta(")));
    EXPECT_TRUE(source.contains(QStringLiteral("stagePointCloudSnapshot(")));
    EXPECT_TRUE(header.contains(QStringLiteral("_manualEditCancellation")));
    EXPECT_TRUE(editSource.contains(QStringLiteral("isCancelled(cancellationFlag)")));
    EXPECT_TRUE(snapshotSource.contains(QStringLiteral(
        "cancellationFlag->load(std::memory_order_relaxed)")));
    EXPECT_TRUE(source.contains(QStringLiteral("task_result.save.commit(&save_error)")));
    EXPECT_TRUE(source.contains(QStringLiteral("outcome.value->save.discard()")));
    EXPECT_TRUE(snapshotSource.contains(QStringLiteral("plapoint::io::writeObj<float>")));
    EXPECT_TRUE(snapshotSource.contains(QStringLiteral("PlyFormat::BinaryLE")));

    const qsizetype applyStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::startManualPruneApply"));
    const qsizetype undoStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::undoLastManualPrune"), applyStart);
    ASSERT_GE(applyStart, 0);
    ASSERT_GT(undoStart, applyStart);
    const QString applyBlock = source.mid(applyStart, undoStart - applyStart);
    const qsizetype generationCheck = applyBlock.indexOf(
        QStringLiteral("load_generation != self->_loadGen"));
    const qsizetype snapshotCommit = applyBlock.indexOf(
        QStringLiteral("task_result.save.commit(&save_error)"));
    ASSERT_GE(generationCheck, 0);
    ASSERT_GE(snapshotCommit, 0);
    EXPECT_LT(generationCheck, snapshotCommit);
    EXPECT_TRUE(applyBlock.contains(QStringLiteral(
        "image_counts,\n                cancellation.get()")));
    EXPECT_TRUE(applyBlock.contains(QStringLiteral(
        "path, *task_result.edit.cloud, cancellation.get()")));

    const qsizetype cancel_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::cancelPendingLoad"));
    const qsizetype clear_prepared_start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::clearPreparedGeometry"), cancel_start);
    ASSERT_GE(cancel_start, 0);
    ASSERT_GT(clear_prepared_start, cancel_start);
    const QString cancel_block = source.mid(
        cancel_start, clear_prepared_start - cancel_start);
    EXPECT_TRUE(cancel_block.contains(QStringLiteral(
        "_manualEditCancellation->store(true")));
}

TEST(CameraSceneRenderContractTest, PointScalarsAndSelectionHighlightStayOnGpu)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString vertexShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_point.vert"));
    const QString fragmentShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_point.frag"));

    const qsizetype pipelineStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensurePointPipeline"));
    const qsizetype pipelineEnd = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureTexturedMeshPipeline"), pipelineStart);
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPointCloud"));
    const qsizetype drawEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawTexturedMesh"), drawStart);
    const qsizetype overlayStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::paintOverlay"));
    const qsizetype overlayEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawPlyLoadProgressOverlay"), overlayStart);
    ASSERT_GE(pipelineStart, 0);
    ASSERT_GT(pipelineEnd, pipelineStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(drawEnd, drawStart);
    ASSERT_GE(overlayStart, 0);
    ASSERT_GT(overlayEnd, overlayStart);

    const QString pipelineBlock = source.mid(pipelineStart, pipelineEnd - pipelineStart);
    const QString drawBlock = source.mid(drawStart, drawEnd - drawStart);
    const QString overlayBlock = source.mid(overlayStart, overlayEnd - overlayStart);
    EXPECT_TRUE(header.contains(QStringLiteral("RhiBufferSet _pointScalarBuffer")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiBufferSet _manualHighlightPointBuffer")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiBufferSet _manualHighlightScalarBuffer")));
    EXPECT_TRUE(header.contains(QStringLiteral("_manualHighlightBuffersReleasePending")));
    EXPECT_TRUE(header.contains(QStringLiteral("RhiPipelineSet _highlightPointPipeline")));
    EXPECT_GE(pipelineBlock.count(QStringLiteral("QRhiVertexInputBinding::PerInstance")), 2);
    EXPECT_TRUE(pipelineBlock.contains(QStringLiteral(
        "QRhiVertexInputAttribute(1, 3, QRhiVertexInputAttribute::Float")));
    EXPECT_TRUE(pipelineBlock.contains(QStringLiteral("setDepthTest(!highlightOnly)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("_pointScalarBuffer.vertexBuffer")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("cb->setVertexInput(0, 2, vertex_inputs)")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("_highlightPointPipeline")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("renderModeFlags[2] = 1.0f")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("renderModeFlags[3] = 1.0f")));
    EXPECT_TRUE(source.contains(QStringLiteral("kMaximumCompactSelectionPoints")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "screenRect.normalized().intersected(rect())")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("highlightCap")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("manual_clip_matrix")));
    EXPECT_FALSE(overlayBlock.contains(QStringLiteral("painter.drawEllipse(screenPoint")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("layout(location = 3) in float aImageCount")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("vSelected")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("uRenderModeFlags.w > 0.5")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("scalarRamp(aImageCount")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("if (vSelected < 0.5)")));
}

TEST(CameraSceneRenderContractTest, DeduplicatesCameraPosesByImageIdentity)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setCameraPoses"));
    const qsizetype end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setShowGizmo"), start);

    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);
    EXPECT_TRUE(block.contains(QStringLiteral("QSet<QString> cameraKeys")));
    EXPECT_TRUE(block.contains(QStringLiteral("cameraKeys.contains(cameraKey)")));
    EXPECT_TRUE(block.contains(QStringLiteral("deduplicated_poses.push_back(pose)")));
}

TEST(CameraSceneRenderContractTest, MetadataResultUpdatesDoNotReloadCameraImages)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    const qsizetype signature_start = source.indexOf(
        QStringLiteral("QJsonArray cameraPoseMetadataFromImages"));
    const qsizetype signature_end = source.indexOf(
        QStringLiteral("} // namespace"), signature_start);
    ASSERT_GE(signature_start, 0);
    ASSERT_GT(signature_end, signature_start);
    const QString signature_block = source.mid(
        signature_start, signature_end - signature_start);
    EXPECT_TRUE(signature_block.contains(QStringLiteral("QStringLiteral(\"path\")")));
    EXPECT_TRUE(signature_block.contains(QStringLiteral("QStringLiteral(\"image_path\")")));
    EXPECT_TRUE(signature_block.contains(QStringLiteral("QStringLiteral(\"camera\")")));
    EXPECT_FALSE(signature_block.contains(QStringLiteral("depth_map_results")));

    EXPECT_TRUE(header.contains(QStringLiteral("QJsonArray _cameraPoseMetadata")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "if (_cameraPoseMetadata == camera_pose_metadata)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraPoseMetadata = QJsonArray()")));
}

TEST(CameraSceneRenderContractTest, RepeatedCameraPosesPreserveLoadedThumbnailResources)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype start = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setCameraPoses"));
    const qsizetype end = source.indexOf(
        QStringLiteral("void CameraSceneWidget::setShowGizmo"), start);
    ASSERT_GE(start, 0);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("if (poses_unchanged)")));
    EXPECT_TRUE(block.contains(QStringLiteral("return;")));
    EXPECT_TRUE(block.contains(QStringLiteral("reusable_image_sequence")));
    EXPECT_TRUE(block.contains(QStringLiteral("clearManualPointSelection()")));
    EXPECT_TRUE(block.contains(QStringLiteral("if (!reusable_image_sequence)")));
    EXPECT_TRUE(block.contains(QStringLiteral("_cameraImageCache.clear()")));
    EXPECT_TRUE(block.contains(QStringLiteral("_thumbnailPipeline.instancesDirty = true")));
}

TEST(CameraSceneRenderContractTest, CachesImageLoadFailuresToPreventRetryStorm)
{
    const QString header = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("_cameraImageLoadFailures")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.contains(key)")));
    EXPECT_TRUE(source.contains(QStringLiteral("_cameraImageLoadFailures.insert(key)")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "_cameraImageLoadFailures.contains(thumbnail_key)")));
    EXPECT_TRUE(source.contains(QStringLiteral("++_cameraThumbnailLoadCompleted")));
}

TEST(CameraSceneRenderContractTest, RefreshedCameraImagesInvalidateGpuUploadState)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral(
        "std::array<float, 16> mvp{};")));
    EXPECT_TRUE(header.contains(QStringLiteral(
        "static_assert(sizeof(ImagePlaneUniforms) == 16 * sizeof(float));")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "if (_imagePipeline.uploadedImageKey == key)")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "page->uploadedPoseIndices.remove(pose_index)")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "--_cameraThumbnailLoadCompleted")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "std::copy_n(mvp.constData(), 16, uniforms.mvp.begin())")));
}

TEST(CameraSceneRenderContractTest, HiddenCameraResourcesKeepIndependentPipelineState)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_TRUE(header.contains(QStringLiteral("bool pipelineDirty = true")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool pipelinesDirty = true")));
    EXPECT_TRUE(source.contains(QStringLiteral("_imagePipeline.pipelineDirty")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.pipelinesDirty")));
    EXPECT_TRUE(source.contains(QStringLiteral("_imagePipeline.pipelineDirty = false")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.pipelinesDirty = false")));
}

TEST(CameraSceneRenderContractTest, InactiveGpuResourcesAreReleasedOnTheRenderThread)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype vertexStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureRhiBuffer"));
    const qsizetype indexStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureRhiIndexBuffer"), vertexStart);
    const qsizetype pipelineStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensurePipeline"), indexStart);
    ASSERT_GE(vertexStart, 0);
    ASSERT_GT(indexStart, vertexStart);
    ASSERT_GT(pipelineStart, indexStart);

    const QString vertexBlock = source.mid(vertexStart, indexStart - vertexStart);
    const QString indexBlock = source.mid(indexStart, pipelineStart - indexStart);
    EXPECT_TRUE(vertexBlock.contains(QStringLiteral("buffer->vertexBuffer.reset()")));
    EXPECT_TRUE(indexBlock.contains(QStringLiteral("buffer->indexBuffer.reset()")));
    EXPECT_GE(vertexBlock.count(QStringLiteral("buffer->vertexBuffer.reset()")), 2);
    EXPECT_GE(indexBlock.count(QStringLiteral("buffer->indexBuffer.reset()")), 2);

    const qsizetype texturedStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureTexturedMeshPipeline"));
    const qsizetype texturedEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawRhiBuffer"), texturedStart);
    const qsizetype imageStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureImagePipeline"));
    const qsizetype imageEnd = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureSolidCameraBatchResource"), imageStart);
    const qsizetype thumbnailStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype thumbnailEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), thumbnailStart);
    ASSERT_GE(texturedStart, 0);
    ASSERT_GT(texturedEnd, texturedStart);
    ASSERT_GE(imageStart, 0);
    ASSERT_GT(imageEnd, imageStart);
    ASSERT_GE(thumbnailStart, 0);
    ASSERT_GT(thumbnailEnd, thumbnailStart);
    EXPECT_TRUE(source.mid(texturedStart, texturedEnd - texturedStart).contains(QStringLiteral(
        "releaseTexturedMeshPipelineResources();")));
    EXPECT_TRUE(source.mid(imageStart, imageEnd - imageStart).contains(QStringLiteral(
        "releaseImagePipelineResources();")));
    EXPECT_TRUE(source.mid(thumbnailStart, thumbnailEnd - thumbnailStart).contains(QStringLiteral(
        "releaseCameraThumbnailPipelineResources();")));
    EXPECT_GE(source.mid(texturedStart, texturedEnd - texturedStart).count(QStringLiteral(
        "releaseTexturedMeshPipelineResources();")), 2);
    EXPECT_GE(source.mid(imageStart, imageEnd - imageStart).count(QStringLiteral(
        "releaseImagePipelineResources();")), 2);
    EXPECT_GE(source.mid(thumbnailStart, thumbnailEnd - thumbnailStart).count(QStringLiteral(
        "releaseCameraThumbnailPipelineResources();")), 2);
    EXPECT_TRUE(source.contains(QStringLiteral("_texturedMeshPipeline.texture.reset()")));
    EXPECT_TRUE(source.contains(QStringLiteral("_imagePipeline.texture.reset()")));
    EXPECT_TRUE(source.contains(QStringLiteral("_thumbnailPipeline.atlasPages.clear()")));
}

TEST(CameraSceneRenderContractTest, HiddenThumbnailsStopQueuedDecodeAndDiscardResults)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype discardStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::discardQueuedCameraThumbnails"));
    const qsizetype requestStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::requestCameraPlaneImage"), discardStart);
    const qsizetype applyStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::applyCameraPlaneImage"), requestStart);
    ASSERT_GE(discardStart, 0);
    ASSERT_GT(requestStart, discardStart);
    ASSERT_GT(applyStart, requestStart);

    const QString discardBlock = source.mid(discardStart, requestStart - discardStart);
    const QString requestBlock = source.mid(requestStart, applyStart - requestStart);
    EXPECT_TRUE(discardBlock.contains(QStringLiteral("_cameraImageLoadQueue.dequeue()")));
    EXPECT_TRUE(discardBlock.contains(QStringLiteral("_cameraImageCache.erase(it)")));
    EXPECT_TRUE(requestBlock.contains(
        QStringLiteral("(!_showCameras || !_showCameraThumbnails)")));
    EXPECT_TRUE(source.contains(QStringLiteral("current_request_path")));
}

TEST(CameraSceneRenderContractTest, TiePointMetadataWaitsForCloudAndBlocksConcurrentPrune)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "const bool cloud_not_ready = point_count == 0")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "if (cloud_not_ready || metadata_matches_cloud)")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "const bool tie_point_metadata_matches")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "_isTiePointCloud && _tiePointMetadataLoading")));
}

TEST(CameraSceneRenderContractTest, ImageModeKeepsTheFreeOrbitViewMatrix)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype start = source.indexOf(QStringLiteral(
        "CameraSceneWidget::SceneMatrices CameraSceneWidget::sceneMatrices() const"));
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
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_TRUE(vertexShader.contains(QStringLiteral("layout(location = 0) in vec3 position")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("uniform ImagePlaneUniforms")));
    EXPECT_TRUE(vertexShader.contains(QStringLiteral("uMVP * vec4(position, 1.0)")));
    EXPECT_TRUE(source.contains(QStringLiteral("calibratedImagePlaneCorners")));
    EXPECT_FALSE(source.contains(QStringLiteral("const float image_half_extent")));
    EXPECT_FALSE(source.contains(QStringLiteral("thumbnail_half_extent * 5.2f")));
    EXPECT_FALSE(source.contains(QStringLiteral("activeCameraImageViewportScale")));
}

TEST(CameraSceneRenderContractTest, RenderPipelinesConsumePreformattedImages)
{
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    const qsizetype texturedStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureTexturedMeshPipeline"));
    const qsizetype texturedEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawRhiBuffer"), texturedStart);
    const qsizetype imageStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureImagePipeline"));
    const qsizetype imageEnd = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureSolidCameraBatchResource"), imageStart);
    const qsizetype thumbnailStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureCameraThumbnailPipeline"));
    const qsizetype thumbnailEnd = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawCameraThumbnails"), thumbnailStart);
    ASSERT_GE(texturedStart, 0);
    ASSERT_GT(texturedEnd, texturedStart);
    ASSERT_GE(imageStart, 0);
    ASSERT_GT(imageEnd, imageStart);
    ASSERT_GE(thumbnailStart, 0);
    ASSERT_GT(thumbnailEnd, thumbnailStart);

    EXPECT_FALSE(source.mid(texturedStart, texturedEnd - texturedStart)
                     .contains(QStringLiteral("convertToFormat")));
    EXPECT_FALSE(source.mid(imageStart, imageEnd - imageStart)
                     .contains(QStringLiteral("convertToFormat")));
    EXPECT_FALSE(source.mid(thumbnailStart, thumbnailEnd - thumbnailStart)
                     .contains(QStringLiteral("convertToFormat")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "image.convertToFormat(QImage::Format_RGBX8888)")));
    EXPECT_TRUE(source.contains(QStringLiteral(
        "result.textureImage.convertToFormat(QImage::Format_RGBA8888)")));
}

TEST(CameraSceneRenderContractTest, ActiveImageGeometryIsUploadedOnlyWhenDirtyAndCornersAreCached)
{
    const QString header =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString source =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    const qsizetype ensureStart = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureImagePipeline"));
    const qsizetype ensureEnd = source.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensureSolidCameraBatchResource"), ensureStart);
    const qsizetype drawStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawActiveCameraImage"));
    const qsizetype cornersStart = source.indexOf(
        QStringLiteral("QVector<QVector3D> CameraSceneWidget::displayedCameraImagePlaneCorners"),
        drawStart);
    const qsizetype foregroundStart = source.indexOf(
        QStringLiteral("QPainterPath CameraSceneWidget::foregroundCameraImageOcclusionPath"),
        cornersStart);
    const qsizetype sceneStart = source.indexOf(
        QStringLiteral("void CameraSceneWidget::drawSceneGeometry"), foregroundStart);
    ASSERT_GE(ensureStart, 0);
    ASSERT_GT(ensureEnd, ensureStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(cornersStart, drawStart);
    ASSERT_GT(foregroundStart, cornersStart);
    ASSERT_GT(sceneStart, foregroundStart);

    const QString ensureBlock = source.mid(ensureStart, ensureEnd - ensureStart);
    const QString drawBlock = source.mid(drawStart, cornersStart - drawStart);
    const QString cornersBlock = source.mid(cornersStart, foregroundStart - cornersStart);
    const QString foregroundBlock = source.mid(foregroundStart, sceneStart - foregroundStart);
    EXPECT_TRUE(header.contains(QStringLiteral("QString uploadedGeometryKey")));
    EXPECT_TRUE(header.contains(QStringLiteral("QVector<QVector3D> planeCorners")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool geometryDirty = true")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("_imagePipeline.geometryDirty")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("updates->updateDynamicBuffer(")));
    EXPECT_TRUE(ensureBlock.contains(QStringLiteral("_imagePipeline.vertexBuffer.data()")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("updateDynamicBuffer(")));
    EXPECT_FALSE(drawBlock.contains(QStringLiteral("displayedCameraImagePlaneCorners()")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("_imagePipeline.uploadedGeometryKey")));
    EXPECT_TRUE(cornersBlock.contains(QStringLiteral("!_imagePipeline.geometryDirty")));
    EXPECT_TRUE(cornersBlock.contains(QStringLiteral("return _imagePipeline.planeCorners")));
    EXPECT_TRUE(foregroundBlock.contains(QStringLiteral("displayedCameraImagePlaneCorners()")));
}

TEST(CameraSceneRenderContractTest, AutomaticImageModeHasNoFirstPhotoFallback)
{
    const QString source = readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("drawPointCloudOverlay")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral(
        "void CameraSceneWidget::drawPointCloudOverlay")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_tiePointPipeline")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::Elevation")));
    EXPECT_TRUE(sceneSource.contains(
        QStringLiteral("TiePointColorMode::ImageCount")));
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
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "drawPointCloud(cb, uniforms, clipMatrix);")));

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
    EXPECT_FALSE(thumbnailPipelineBlock.contains(QStringLiteral("convertToFormat")));

    const QString renderBlock = sceneSource.mid(renderStart);
    const qsizetype geometryCall = renderBlock.indexOf(
        QStringLiteral("drawSceneGeometry(cb, uniforms, mvp);"));
    const qsizetype thumbnailsCall = renderBlock.indexOf(
        QStringLiteral("drawCameraThumbnails(cb, mvp, mv);"));
    ASSERT_GE(geometryCall, 0);
    ASSERT_GT(thumbnailsCall, geometryCall);
}

TEST(CameraSceneRenderContractTest, TiePointLoadFitsViewToLoadedGeometry)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

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
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("readBinaryPlyPreview")));

    const qsizetype plyLoadStart =
        sceneSource.indexOf(QStringLiteral("void CameraSceneWidget::loadModelFromPlyInternal"));
    const qsizetype requestLoadStart =
        sceneSource.indexOf(QStringLiteral("void CameraSceneWidget::requestSceneLoad"), plyLoadStart);
    const qsizetype guardedLoadStart =
        sceneSource.indexOf(QStringLiteral("runGuardedWithOutcome("), requestLoadStart);
    const qsizetype plyTiePointState =
        sceneSource.indexOf(QStringLiteral("_isTiePointCloud = request.tiePointCloud;"), requestLoadStart);
    const qsizetype plyFitState =
        sceneSource.indexOf(QStringLiteral("_fitViewAfterLoad = request.fitAfterLoad;"), requestLoadStart);
    ASSERT_GE(plyLoadStart, 0);
    ASSERT_GT(requestLoadStart, plyLoadStart);
    ASSERT_GT(guardedLoadStart, requestLoadStart);
    EXPECT_GT(plyTiePointState, requestLoadStart);
    EXPECT_LT(plyTiePointState, guardedLoadStart);
    EXPECT_GT(plyFitState, requestLoadStart);
    EXPECT_LT(plyFitState, guardedLoadStart);

    EXPECT_TRUE(sceneSource.contains(QStringLiteral("if (request.fitAfterLoad)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("self->fitViewToLoadedGeometry();")));
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
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
        QStringLiteral("void CameraSceneWidget::pumpSceneLoad"));
    const qsizetype tiePointLoaderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::loadTiePointCloudFromFile"), objLoaderStart);
    ASSERT_GE(objLoaderStart, 0);
    ASSERT_GT(tiePointLoaderStart, objLoaderStart);
    const QString objLoaderBlock = sceneSource.mid(
        objLoaderStart,
        tiePointLoaderStart - objLoaderStart);
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral("if (!request.pointCloudResource)")));
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral("plapoint::io::readObj<float>")));
    EXPECT_TRUE(objLoaderBlock.contains(QStringLiteral(
        "&& !request.pointCloudResource\n"
        "                        && !result.textureWarning.isEmpty())")));

    const qsizetype uploadStart = sceneSource.indexOf(
        QStringLiteral("bool CameraSceneWidget::uploadGpuData"));
    const qsizetype pipelineStart = sceneSource.indexOf(
        QStringLiteral("bool CameraSceneWidget::ensurePipeline"), uploadStart);
    ASSERT_GE(uploadStart, 0);
    ASSERT_GT(pipelineStart, uploadStart);
    const QString uploadBlock = sceneSource.mid(uploadStart, pipelineStart - uploadStart);
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral("use_prepared_point_buffer")));
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral("_preparedPointVertexData")));
    EXPECT_TRUE(uploadBlock.contains(QStringLiteral("不会在 GUI 线程回退重建")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("preparePointRenderData(")));
    EXPECT_FALSE(uploadBlock.contains(QStringLiteral("for (std::size_t")));
    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("_modelPointBuffer")));
    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("_modelPointPipeline")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_preferModelPointRender")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "drawPointCloud(cb, uniforms, clipMatrix);")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("drawPointCloudOverlay(painter);")));
}

TEST(CameraSceneRenderContractTest, PointCloudUsesOwnBoundsAndPixelSizedFloorPivot)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("bool       _hasCloudBounds = false;")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "const QVector<QVector3D> &vertices = _cachedCloudBoxVertices;")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "constexpr qreal half_size_pixels = 7.0;")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral(
        "qMax(0.25f, _cachedRadius * 0.045f)")));
}

TEST(CameraSceneRenderContractTest, ModelMenuEnablesTextureAndGatesUnsupportedModes)
{
    const QString menuSource = readProjectFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
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
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral(
        "void modelColorModeChanged(ModelColorMode mode);")));
    EXPECT_TRUE(menuBindingsSource.contains(QStringLiteral("setModelColorMode")));
    EXPECT_TRUE(menuBindingsSource.contains(QStringLiteral(
        "&CameraSceneWidget::modelColorModeChanged")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "self->setModelColorMode(ModelColorMode::Texture)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "self->setModelColorMode(ModelColorMode::Solid)")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral(
        "self->setModelColorMode(ModelColorMode::Shaded)")));
}

TEST(CameraSceneRenderContractTest, ColorModeSwitchesOnlyUpdateUniformDrivenState)
{
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const qsizetype tiePointStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::setTiePointColorMode"));
    const qsizetype modelStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::setModelColorMode"), tiePointStart);
    const qsizetype metadataStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::startTiePointMetadataLoad"), modelStart);
    const qsizetype drawStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::drawSceneGeometry"));
    const qsizetype renderStart = sceneSource.indexOf(
        QStringLiteral("void CameraSceneWidget::render("), drawStart);
    ASSERT_GE(tiePointStart, 0);
    ASSERT_GT(modelStart, tiePointStart);
    ASSERT_GT(metadataStart, modelStart);
    ASSERT_GE(drawStart, 0);
    ASSERT_GT(renderStart, drawStart);

    const QString tiePointBlock = sceneSource.mid(tiePointStart, modelStart - tiePointStart);
    const QString modelBlock = sceneSource.mid(modelStart, metadataStart - modelStart);
    const QString drawBlock = sceneSource.mid(drawStart, renderStart - drawStart);
    EXPECT_TRUE(tiePointBlock.contains(QStringLiteral("_tiePointColorMode = mode")));
    EXPECT_TRUE(modelBlock.contains(QStringLiteral("_modelColorMode = mode")));
    EXPECT_FALSE(tiePointBlock.contains(QStringLiteral("_gpuDirty")));
    EXPECT_FALSE(tiePointBlock.contains(QStringLiteral("_pipelinesDirty")));
    EXPECT_FALSE(modelBlock.contains(QStringLiteral("_gpuDirty")));
    EXPECT_FALSE(modelBlock.contains(QStringLiteral("_pipelinesDirty")));
    EXPECT_FALSE(modelBlock.contains(QStringLiteral("prepareObjRenderData")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "float(static_cast<int>(point_mode))")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral(
        "float(static_cast<int>(_modelColorMode))")));
    EXPECT_TRUE(drawBlock.contains(QStringLiteral("uniforms.renderModeFlags")));
}

TEST(CameraSceneRenderContractTest, MeshModesReuseStaticVerticesAndUInt32IndexBuffers)
{
    const QString sceneHeader =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.h"));
    const QString sceneSource =
        readProjectFile(QStringLiteral("src/gui/views/CameraSceneWidget.cpp"));
    const QString preparationHeader =
        readProjectFile(QStringLiteral("src/gui/views/ObjRenderPreparation.h"));
    const QString preparationSource =
        readProjectFile(QStringLiteral("src/gui/views/ObjRenderPreparation.cpp"));
    const QString fragmentShader =
        readProjectFile(QStringLiteral("src/gui/shaders/camera_scene_mesh.frag"));

    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("struct RhiIndexBufferSet")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("RhiBufferSet _meshBuffer")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("RhiIndexBufferSet _meshTriangleIndices")));
    EXPECT_TRUE(sceneHeader.contains(QStringLiteral("RhiIndexBufferSet _meshWireframeIndices")));
    EXPECT_TRUE(preparationHeader.contains(QStringLiteral("QByteArray triangleIndexData")));
    EXPECT_TRUE(preparationHeader.contains(QStringLiteral("QByteArray wireframeIndexData")));
    EXPECT_TRUE(preparationSource.contains(QStringLiteral("std::sort(edges.begin(), edges.end())")));
    EXPECT_TRUE(preparationSource.contains(QStringLiteral(
        "std::unique(edges.begin(), edges.end())")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("QRhiBuffer::Static")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("QRhiBuffer::IndexBuffer")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("QRhiCommandBuffer::IndexUInt32")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("cb->drawIndexed(")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("&_meshTriangleIndices")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("&_meshWireframeIndices")));
    EXPECT_FALSE(sceneHeader.contains(QStringLiteral("ModelVisualizationManager")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("ModelVisualizationManager")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("buildGeometry(")));
    EXPECT_FALSE(sceneSource.contains(QStringLiteral("_modelWireframeBuffer")));
    EXPECT_TRUE(sceneSource.contains(QStringLiteral("drawModelLegend(painter)")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("uRenderModeFlags")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("scalarRamp(vElevation")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "cross(dFdx(vViewPosition), dFdy(vViewPosition))")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("isNeutralSurface")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("neutralShape")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral("dot(normal, viewDirection) < 0.0")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "vec3 baseLinear = srgbToLinear(baseColor);")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "float keyDiffuse = max(dot(normal, lightDirection), 0.0);")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "fragColor = vec4(linearToSrgb(baseLinear * shape), 1.0);")));
    EXPECT_TRUE(fragmentShader.contains(QStringLiteral(
        "0.86 + 0.10 * keyDiffuse + 0.04 * headDiffuse")));
    EXPECT_FALSE(fragmentShader.contains(QStringLiteral("0.18 + 0.72 * diffuse")));
}
