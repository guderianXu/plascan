// =============================================================================
// 文件: CameraSceneWidget.h
// 功能: 可复用相机三维场景控件声明
// 职责:
//   - CameraSceneWidget: 基于 Qt RHI 的三维交互场景控件，
//                        支持相机姿态、点云、PLY 网格和 OBJ/MTL 纹理模型渲染
// =============================================================================

#pragma once

#include <QByteArray>
#include <QFutureWatcher>
#include <QHash>
#include <QImage>
#include <QLineF>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QQueue>
#include <QRect>
#include <QRhiWidget>
#include <QScopedPointer>
#include <QSharedPointer>
#include <QSet>
#include <QSize>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <atomic>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <plapoint/core/point_cloud.h>

#include "CameraSceneViewMath.h"
#include "ModelVisualization.h"
#include "ObjRenderPreparation.h"
#include "PointCloudEditPreparation.h"
#include "SceneGeometryPreparation.h"
#include "TiePointVisualization.h"

/// 渲染用点云类型别名
using RenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

class QWidget;
class QPainter;
class QPainterPath;
class CameraSceneOverlayWidget;
class QRhiBuffer;
class QRhiCommandBuffer;
class QRhiGraphicsPipeline;
class QRhiResourceUpdateBatch;
class QRhiSampler;
class QRhiShaderResourceBindings;
class QRhiTexture;
class QResizeEvent;
class QTimer;

// =============================================================================
// CameraSceneWidget
// 继承自 QRhiWidget，提供跨 Vulkan、D3D11、Metal 和 OpenGL 后端的三维场景渲染控件。
// 使用 QRhiBuffer + .qsb shader（顶点色、Phong 光照和 UV 纹理 shader）
// 功能包括：
//   - 渲染相机姿态（位置+相机卡片）、点云（xyz）、PLY 网格和 OBJ/MTL 纹理模型
//   - 画布导航：左键/ Ctrl+左键稳定环绕，右键或中键平移，滚轮缩放
//   - 中央轨迹球 Arcball 自由旋转和 X/Y/Z 单轴环旋转
//   - 实时显示坐标轴指示器和欧拉角信息
// =============================================================================
class CameraSceneWidget : public QRhiWidget
{
    Q_OBJECT
public:
    using TiePointColorMode = xjw::gui::tie_points::ColorMode;
    using TiePointQualityCriterion = xjw::gui::tie_points::QualityCriterion;
    using TiePointPrunePreviewQuery = xjw::gui::tie_points::PrunePreviewQuery;
    using ModelColorMode = xjw::gui::model_views::ColorMode;

    // -------------------------------------------------------------------------
    // CameraPose：单个相机的姿态信息
    //   name     - 相机/影像名称（用于 3D 场景中的文字标注）
    //   center   - 相机光心在世界坐标系中的位置（单位与项目坐标一致）
    //   rotation - 3×3 camera-to-world 旋转矩阵 R（行主序），列向量依次为
    //              camera +x / camera +y / physical forward。照片平面的视觉上轴
    //              还必须结合 vAxisSign 换算。
    // -------------------------------------------------------------------------
    struct CameraPose {
        QString name;       // 相机名称（用于标注）
        QString imagePath;  // 影像路径（用于与影像列表联动高亮）
        QVector3D center;   // 相机光心世界坐标
        QMatrix3x3 rotation; // 相机旋转矩阵（camera +x/+y/forward 列向量）
        float focalX = 0.0f;
        float focalY = 0.0f;
        float principalX = 0.0f;
        float principalY = 0.0f;
        int imageWidth = 0;
        int imageHeight = 0;
        int uAxisSign = 1;
        int vAxisSign = 1;
        bool depthAxisFlipped = false;
    };

    enum class CameraImagePlaneMode
    {
        Solid,
        Thumbnail,
        Image
    };

    enum class CameraImageDisplayLayer
    {
        Background,
        Foreground
    };

    // 构造函数，初始化 RHI 渲染控件并设置默认视角
    explicit CameraSceneWidget(QWidget *parent = nullptr);
    ~CameraSceneWidget() override;

    // 设置要渲染的相机姿态列表，触发重绘
    void setCameraPoses(const QVector<CameraPose> &poses);
    // 从 XYZ 文本文件加载点云（每行 "x y z [r g b ...]"）
    void loadPointCloudFromXyz(const QString &xyzPath);
    // 从 PLY 点云加载并在完成后自动适应视图。
    void loadPointCloudFromPly(const QString &plyPath);
    // 从 OBJ 顶点点云加载；不执行模型材质、UV 和三角面渲染准备。
    void loadPointCloudFromObj(const QString &objPath);

    // 从 PLY 文件加载三角网格或点云模型
    void loadModelFromPly(const QString &plyPath);
    // 从 OBJ 文件加载三角网格或点云模型
    void loadModelFromObj(const QString &objPath);
    // 加载稀疏连接点，并同步读取逐点影像观测数。
    void loadTiePointCloudFromFile(const QString &pointCloudPath,
                                   const QString &sidecarPath = QString());
    void setTiePointColorMode(TiePointColorMode mode);
    void setModelColorMode(ModelColorMode mode);
    bool hasTiePointQualityMetadata(TiePointQualityCriterion criterion) const;
    xjw::gui::tie_points::ScalarRange tiePointQualityRange(
        TiePointQualityCriterion criterion) const;
    int tiePointQualityPointCount() const;
    bool requestTiePointPrunePreview(
        const TiePointPrunePreviewQuery &query,
        QString *errorMessage = nullptr);
    void clearTiePointPrunePreview();
    int tiePointPrunePreviewCandidateCount() const;

    bool setManualPruneModeEnabled(bool enabled, QString *errorMessage = nullptr);
    bool isManualPruneModeEnabled() const { return _manualPruneMode; }
    bool undoLastManualPrune(QString *errorMessage = nullptr);

    /// 设置是否显示操控球（Gizmo 旋转环），用于减少视线遮挡
    void setShowGizmo(bool show);
    /// 设置是否显示相机平面卡片和文件名标签
    void setShowCameras(bool show);
    void setShowCameraThumbnails(bool show);
    /// 设置是否在每个相机光心显示其局部 X/Y/Z 坐标轴。
    void setShowCameraLocalAxes(bool show);
    void setShowCameraImage(bool show);
    void setCameraImageDisplayLayer(CameraImageDisplayLayer layer);
    void setCameraImageLocked(bool locked);
    void setHighlightedCameraPath(const QString &imagePath);
    void clearHighlightedCamera();
    // 清空当前项目的相机、模型和异步加载状态，并恢复默认观察视图。
    void clearProjectScene();
    void zoomIn();
    void zoomOut();
    void resetView();

signals:
    void plyLoadProgressChanged(int generation, int percent, const QString &statusText);
    void modelColorModeChanged(ModelColorMode mode);
    void tiePointQualityMetadataReady(bool ready);
    void tiePointPrunePreviewChanged(int candidateCount);
    void manualPruneApplied(int removedCount, int remainingCount);
    void manualPruneUndone(int restoredCount);
    void manualPruneSaved(const QString &path, int remainingCount);
    void manualPruneSaveFailed(const QString &errorMessage);

protected:
    // RHI 初始化：创建当前图形后端的渲染资源和管线。
    void initialize(QRhiCommandBuffer *cb) override;

    // 主渲染函数：清屏 → 绘制点云/模型与相机图像层 → 更新覆盖层。
    void render(QRhiCommandBuffer *cb) override;

    // 释放 RHI 资源。
    void releaseResources() override;

    // 视口尺寸变化时标记资源和投影参数需要更新。
    void resizeEvent(QResizeEvent *event) override;

    // 鼠标按下：区分画布平移/环绕、Gizmo 旋转与显式框选工具
    void mousePressEvent(QMouseEvent *event) override;

    // 鼠标移动：画布平移/环绕、Arcball、单轴旋转和悬停高亮检测
    void mouseMoveEvent(QMouseEvent *event) override;

    // 鼠标释放：清除拖拽状态，恢复光标
    void mouseReleaseEvent(QMouseEvent *event) override;

    // 滚轮事件：缩放场景（调整 _zoomScale）
    void wheelEvent(QWheelEvent *event) override;

    void keyPressEvent(QKeyEvent *event) override;

private:
    friend class CameraSceneOverlayWidget;

    // -------------------------------------------------------------------------
    // HoverAxis：鼠标悬停/拖拽时激活的旋转轴
    //   None - 无激活轴（使用 Arcball 自由旋转）
    //   X/Y/Z - 绕对应本地轴进行环旋转
    // -------------------------------------------------------------------------
    enum class HoverAxis {
        None, // 无轴（Arcball 自由旋转）
        X,    // 绕 X 轴旋转（红色环）
        Y,    // 绕 Y 轴旋转（绿色环）
        Z     // 绕 Z 轴旋转（蓝色环）
    };
    enum class LeftDragMode
    {
        None,
        Orbit,
        GizmoOrbit,
        GizmoAxis
    };
    struct SceneMatrices
    {
        QMatrix4x4 modelView;
        QMatrix4x4 projection;
    };
    struct CameraPlaneImageResult;
    void applyZoomFactor(double factor);

    // 将三维世界点投影到屏幕像素坐标（考虑当前视图旋转、投影、平移偏移）
    // ok 为 nullptr 或 false 时表示点在裁剪空间外
    QPointF projectToScreen(const QVector3D &p, bool *ok = nullptr) const;
    SceneMatrices sceneMatrices() const;
    bool cameraAlignmentActive() const;
    QSize displayedCameraImageSize() const;
    QRectF cameraAlignmentViewport(const QSize &renderSize) const;

    // 将向量从本地空间旋转到当前视图空间（应用 _viewRot）
    QVector3D applyViewRotation(const QVector3D &v) const;

    // 返回当前视图四元数对应的欧拉角（度数，顺序：pitch/yaw/roll）
    QVector3D eulerAnglesDeg() const;

    // 计算操控球（Gizmo 旋转环）在屏幕上的半径（像素），随窗口大小自适应
    qreal manipRadiusPx() const;

    // 返回操控球在屏幕空间的中心点（固定为窗口中心）
    QPointF manipCenterScreen() const;

    // 计算场景中所有点（相机、点云、模型顶点）的质心
    QVector3D sceneCenter() const;

    // 计算场景中所有点到质心的最大距离（用于自适应缩放）
    float sceneRadius() const;

    // 根据鼠标位置检测当前最近的旋转轴环（距离 <12px 判定为悬停）
    HoverAxis pickHoverAxis(const QPoint &mousePos) const;

    // 中央 Gizmo 仅在自身圆形范围内接管左键，画布其余区域使用导航手势。
    bool isInsideRotationGizmo(const QPoint &mousePos) const;

    // 计算指定轴环在鼠标附近切线方向（单位向量，用于单轴旋转的拖拽量计算）
    QVector2D pickAxisTangent(const QPoint &mousePos, HoverAxis axis) const;

    // Arcball：将屏幕位置映射到单位球面（xy=屏幕坐标系，z指向观察者）
    // 若鼠标在球内则返回球面点，否则投影到赤道圆
    QVector3D arcballVector(const QPoint &mousePos) const;

    // 画布环绕不依赖 Gizmo 半径，使用稳定的屏幕水平/垂直旋转。
    void applyOrbitDrag(const QPoint &pixelDelta);

    bool isNavigationDragging() const;

    // 将场景平移偏移量限制在合理范围（±45% 画布尺寸），避免场景移出屏幕
    void clampSceneOffset();

    // 根据当前拖拽状态和悬停轴更新鼠标光标样式
    void updateCursor();

    int maxVisibleCameraLabels() const;
    float cameraImagePlaneHalfExtent(const CameraPose &pose,
                                     const QMatrix4x4 &worldToView) const;
    bool isCameraHighlighted(const CameraPose &pose) const;
    QString normalizedCameraPath(const QString &imagePath) const;
    QString cameraPlaneImageKey(const QString &imagePath, CameraImagePlaneMode mode) const;
    QImage cachedCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode) const;
    void requestCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode);
    void applyCameraPlaneImage(const CameraPlaneImageResult &result);
    void discardQueuedCameraThumbnails();
    int displayedCameraImagePoseIndex() const;
    void updateActiveCameraForView();
    void refreshLockedCameraImage();
    void drawFloorPivotCross(QPainter &painter) const;
    bool cameraDirectionLeaderSegment(const CameraPose &pose,
                                      float planeHalfExtent,
                                      QVector3D *start,
                                      QVector3D *end) const;

    // 在普通透明 QWidget 覆盖层中绘制 2D 标注：
    //   - 操控球 Gizmo（旋转环）
    //   - 相机平面卡片和名称标注
    //   - 右下角坐标轴指示器和欧拉角文字
    void updateCameraOverlay();
    void requestOverlayUpdate();
    void paintOverlay(QPainter &painter);
    void drawPlyLoadProgressOverlay(QPainter &painter);
    void drawCameraThumbnailProgressOverlay(QPainter &painter) const;
    void drawRotationGizmo(QPainter &painter) const;
    void drawTiePointLegend(QPainter &painter) const;
    void drawModelLegend(QPainter &painter) const;
    struct TiePointMetadataRequest
    {
        QString sidecarPath;
        int generation = 0;
    };
    void startTiePointMetadataLoad(const QString &sidecarPath, int generation);
    void pumpTiePointMetadataLoad();
    struct TiePointPrunePreviewRequest
    {
        QByteArray vertexData;
        QByteArray scalarData;
        xjw::gui::tie_points::QualityMetadata metadata;
        TiePointPrunePreviewQuery query;
        int strideBytes = 0;
        int generation = 0;
        int loadGeneration = 0;
        int pointCount = 0;
    };
    void pumpTiePointPrunePreview();
    void loadPointCloudFromXyzInternal(const QString &xyzPath,
                                       bool tiePointCloud,
                                       bool fitAfterLoad);
    void loadModelFromPlyInternal(const QString &plyPath,
                                  bool tiePointCloud,
                                  bool fitAfterLoad,
                                  bool pointCloudResource);
    void loadModelFromObjInternal(const QString &objPath,
                                  bool tiePointCloud,
                                  bool fitAfterLoad,
                                  bool pointCloudResource);
    enum class SceneLoadFormat
    {
        Xyz,
        Ply,
        Obj
    };
    struct SceneLoadRequest
    {
        SceneLoadFormat format = SceneLoadFormat::Xyz;
        QString path;
        bool tiePointCloud = false;
        bool fitAfterLoad = false;
        bool pointCloudResource = true;
        int generation = 0;
    };
    void requestSceneLoad(SceneLoadRequest request);
    void pumpSceneLoad();
    void fitViewToLoadedGeometry();

    struct RhiPipelineSet;

    // 将点云和模型数据整理为 RHI 顶点缓冲，在 render 中按需上传。
    bool uploadGpuData();
    bool failGeometryUpload(const QString &message);
    void releaseGeometryBufferResources();
    static void releasePipelineResources(RhiPipelineSet *pipeline);
    void clearPreparedGeometry();
    void applyPointPreparation(PointRenderPreparation preparation);
    void applyPointChunkScalarData(const QByteArray &scalarData);
    void applyCloudSpatialSummary(const CloudSpatialSummary &summary);
    // 点云使用单采样，三角网格使用 4x MSAA；切换资源时重建匹配的管线。
    void updateSampleCountForGeometry();

    // 当点云/模型数据变更后调用，标记缓存失效并重新计算
    void invalidateCache() const;

    void startManualPointSelection(const QRect &screenRect);
    struct ManualSelectionRequest
    {
        QByteArray vertexData;
        QByteArray scalarData;
        QRect screenRect;
        QMatrix4x4 clipMatrix;
        QSize viewportSize;
        QPointF sceneOffset;
        int strideBytes = 0;
        int generation = 0;
        int loadGeneration = 0;
    };
    void pumpManualPointSelection();
    void startManualPruneApply();
    void clearManualPointSelection();
    void pushManualUndoDelta(PointCloudEditDelta delta);
    void applyManualPointCloudResult(
        PointCloudEditResult result,
        const xjw::gui::tie_points::ScalarRange &imageCountRange);

    // 取消未完成的异步加载并等待结束（在新加载开始前调用）
    void cancelPendingLoad();
    void cancelMeshTexturePreparation();
    void prepareMeshTextureUploadImage(int maximumTextureSize);

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
        QVector<PointVertexIndex> sourceIndices;
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
        bool geometryDirty = true;
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
    static_assert(offsetof(SceneUniforms, mvp) == 0);
    static_assert(offsetof(SceneUniforms, modelView) == 16 * sizeof(float));
    static_assert(offsetof(SceneUniforms, normalMatrix) == 32 * sizeof(float));
    static_assert(offsetof(SceneUniforms, lightDirPointSize) == 48 * sizeof(float));
    static_assert(offsetof(SceneUniforms, viewportSize) == 52 * sizeof(float));
    static_assert(offsetof(SceneUniforms, renderModeFlags) == 56 * sizeof(float));
    static_assert(offsetof(SceneUniforms, scalarRange) == 60 * sizeof(float));
    static_assert(sizeof(SceneUniforms) == 64 * sizeof(float));

    struct alignas(16) ImagePlaneUniforms
    {
        std::array<float, 16> mvp{};
        std::array<float, 4> composition{};
    };
    static_assert(sizeof(ImagePlaneUniforms) == 20 * sizeof(float));

    struct alignas(16) CameraPlaneUniforms
    {
        std::array<float, 16> mvp{};
        std::array<float, 16> modelView{};
        std::array<float, 4> viewportZoom{};
    };
    static_assert(sizeof(CameraPlaneUniforms) == 36 * sizeof(float));

    struct CameraPlaneImageResult
    {
        QString path;
        QImage image;
        QSize originalSize;
        QString errorMessage;
        CameraImagePlaneMode mode = CameraImagePlaneMode::Solid;
        int generation = 0;
        bool loaded = false;
    };

    struct CameraPlaneImageRequest
    {
        QString path;
        CameraImagePlaneMode mode = CameraImagePlaneMode::Solid;
        int generation = 0;
    };

    static CameraPlaneImageResult loadCameraPlaneImage(const QString &imagePath,
                                                       CameraImagePlaneMode mode,
                                                       int generation);
    void pumpCameraPlaneImageLoads();

    bool _gpuDirty = true;  // 顶点缓冲需要重新上传
    bool _rhiReady = false;
    bool _backendFallbackScheduled = false;
    bool _backendFallbackActive = false;
    bool _pipelinesDirty = true;
    QString _renderError;
    QString _renderWarning;
    QString _geometryUploadError;
    bool _texturedMeshResourceFailed = false;
    bool _cameraResourceFailed = false;
    bool _activeImageResourceFailed = false;
    CameraSceneOverlayWidget *_overlayWidget = nullptr;

    bool ensureRhiBuffer(RhiBufferSet *buffer, QRhiResourceUpdateBatch *updates);
    bool ensureRhiIndexBuffer(RhiIndexBufferSet *buffer,
                              QRhiResourceUpdateBatch *updates);
    bool ensurePipeline(RhiPipelineSet *pipeline,
                        int topology,
                        int strideBytes,
                        bool hasNormals,
                        bool depthWrite = true);
    bool ensurePointPipeline(RhiPipelineSet *pipeline,
                             bool highlightOnly);
    bool ensureTexturedMeshPipeline(QRhiResourceUpdateBatch *updates);
    bool ensureImagePipeline(QRhiResourceUpdateBatch *updates);
    bool ensureCameraThumbnailPipeline(QRhiResourceUpdateBatch *updates);
    void releaseTexturedMeshPipelineResources();
    void releaseImagePipelineResources();
    void releaseCameraThumbnailPipelineResources();
    void rollbackResourceUpdateState();
    void commitResourceUpdateState();
    bool ensureSolidCameraBatchResource(
        QSharedPointer<RhiCameraThumbnailResource> *resource,
        const QColor &color,
        QRhiResourceUpdateBatch *updates);
    bool ensureCameraThumbnailAtlasPage(int page_index,
                                        QRhiResourceUpdateBatch *updates);
    void drawRhiBuffer(QRhiCommandBuffer *cb,
                       RhiBufferSet *buffer,
                       RhiPipelineSet *pipeline,
                       const SceneUniforms &uniforms);
    void drawIndexedRhiBuffer(QRhiCommandBuffer *cb,
                              RhiBufferSet *vertices,
                              RhiIndexBufferSet *indices,
                              RhiPipelineSet *pipeline,
                              const SceneUniforms &uniforms);
    void drawPointCloud(QRhiCommandBuffer *cb,
                        const SceneUniforms &uniforms,
                        const QMatrix4x4 &clipMatrix);
    void drawTexturedMesh(QRhiCommandBuffer *cb, const SceneUniforms &uniforms);
    void drawActiveCameraImage(QRhiCommandBuffer *cb,
                               const QMatrix4x4 &mvp,
                               float opacity);
    void drawCameraThumbnails(QRhiCommandBuffer *cb,
                              const QMatrix4x4 &mvp,
                              const QMatrix4x4 &model_view);
    void drawSceneGeometry(QRhiCommandBuffer *cb,
                           SceneUniforms &uniforms,
                           const QMatrix4x4 &clipMatrix);

    // 点云 GPU 资源
    RhiBufferSet _pointBuffer;
    RhiBufferSet _pointScalarBuffer;
    QVector<QSharedPointer<RhiPointChunk>> _pointChunks;
    RhiBufferSet _manualHighlightPointBuffer;
    RhiBufferSet _manualHighlightScalarBuffer;
    bool _manualHighlightBuffersReleasePending = false;
    RhiBufferSet _prunePreviewPointBuffer;
    RhiBufferSet _prunePreviewScalarBuffer;
    bool _prunePreviewBuffersReleasePending = false;
    int _pointCount = 0;
    float _pointCloudPointSize = 2.4f;

    // 网格 GPU 资源：非纹理模式共享静态 VBO/IBO，UV 接缝单独展开。
    RhiBufferSet _meshBuffer;
    RhiBufferSet _texturedMeshBuffer;
    RhiIndexBufferSet _meshTriangleIndices;
    RhiIndexBufferSet _meshWireframeIndices;
    QVector<QSharedPointer<RhiMeshPointPreviewChunk>> _meshPointPreviewChunks;
    int _meshVertCount = 0;
    bool _meshHasFaces = true;  ///< false 时用点图元绘制含法向量点云
    bool _meshIsPointPreview = false;
    ObjRenderPreparation _preparedMesh;
    bool _preparedPointBuffer = false;
    QByteArray _preparedPointVertexData;
    int _preparedPointVertexCount = 0;
    QVector<PointRenderChunkPreparation> _preparedPointChunks;

    // 点云包围盒 GPU 资源。
    RhiBufferSet _lineBuffer;
    int _lineCount = 0;

    RhiPipelineSet _colorPointPipeline;
    RhiPipelineSet _prunePreviewPointPipeline;
    RhiPipelineSet _highlightPointPipeline;
    RhiPipelineSet _colorLinePipeline;
    RhiPipelineSet _meshTrianglePipeline;
    RhiPipelineSet _meshWireframePipeline;
    RhiPipelineSet _meshPointPreviewPipeline;
    RhiTexturedMeshPipelineSet _texturedMeshPipeline;
    RhiImagePipelineSet _imagePipeline;
    RhiCameraThumbnailPipelineSet _thumbnailPipeline;

    QVector<CameraPose> _poses;             // 当前相机姿态列表
    QVector<xjw::gui::camera_scene::CameraViewCandidate> _cameraViewCandidates;
    RenderCloud _cloud;     // 当前显示的点云或网格（源自文件或外部调用）
    bool _isTiePointCloud = false;
    TiePointColorMode _tiePointColorMode = TiePointColorMode::Color;
    ModelColorMode _modelColorMode = ModelColorMode::Shaded;
    QVector<int> _tiePointImageCounts;
    xjw::gui::tie_points::QualityMetadata _tiePointQualityMetadata;
    QByteArray _tiePointScalarData;
    xjw::gui::tie_points::ScalarRange _tiePointElevationRange;
    xjw::gui::tie_points::ScalarRange _tiePointImageCountRange;
    xjw::gui::tie_points::ScalarRange _modelElevationRange;
    bool _tiePointMetadataLoading = false;
    QString _tiePointMetadataError;
    std::optional<TiePointMetadataRequest> _pendingTiePointMetadataLoad;
    std::shared_ptr<std::atomic_bool> _tiePointMetadataCancellation;
    bool _tiePointMetadataWorkerActive = false;
    std::optional<TiePointPrunePreviewRequest> _pendingTiePointPrunePreview;
    std::shared_ptr<std::atomic_bool> _tiePointPrunePreviewCancellation;
    bool _tiePointPrunePreviewWorkerActive = false;
    int _tiePointPrunePreviewGeneration = 0;
    int _tiePointPrunePreviewCandidateCount = 0;
    QImage _meshTextureImage;
    QImage _meshTextureUploadImage;
    QString _meshTexturePath;
    bool _meshHasTexture = false;
    std::shared_ptr<std::atomic_bool> _meshTexturePreparationCancellation;
    bool _meshTexturePreparationActive = false;
    int _meshTexturePreparationMaximumSize = 0;

    // 场景中心/半径/包围盒缓存（避免每帧遍历大量点）
    mutable QVector3D  _cachedCenter;
    mutable float      _cachedRadius = 10.0f;
    mutable QVector3D  _cachedAABBMin;
    mutable QVector3D  _cachedAABBMax;
    mutable QVector3D  _cachedCloudAABBMin;
    mutable QVector3D  _cachedCloudAABBMax;
    mutable QVector<QVector3D> _cachedCloudBoxVertices;
    mutable bool       _hasCloudBounds = false;
    mutable bool       _cacheDirty = true;
    CloudSpatialSummary _cloudSpatialSummary;
    QVector3D _focusedGeometryCenter;
    float _focusedGeometryRadius = 1.0f;
    bool _hasFocusedGeometryBounds = false;

    // 异步加载状态
    bool   _loading      = false;
    int    _loadGen      = 0;    ///< 每次发起新加载时递增，用于丢弃过期回调
    std::optional<SceneLoadRequest> _pendingSceneLoad;
    std::shared_ptr<std::atomic_bool> _sceneLoadCancellation;
    bool _sceneLoadWorkerActive = false;
    int    _plyLoadProgressPercent = -1;
    QString _plyLoadProgressText;
    QTimer *_loadProgressAnimationTimer = nullptr;
    bool _fitViewAfterLoad = false;
    QQuaternion _viewRot;                     // 当前视图旋转四元数
    double _zoomScale = 1.0;                  // 当前缩放系数（影响相机到场景中心的距离）
    QPointF _sceneOffsetPx = QPointF(0.0, 0.0); // 场景在屏幕空间的平移偏移（像素）
    QPoint _lastMousePos;                     // 上一帧鼠标位置（用于增量计算）
    HoverAxis _hoverAxis = HoverAxis::None;   // 当前鼠标悬停的轴（高亮显示）
    HoverAxis _dragAxis = HoverAxis::None;    // 当前拖拽激活的轴
    bool _leftDragging = false;               // 左键导航或 Gizmo 拖拽
    bool _middleDragging = false;             // 中键平移
    bool _rightDragging = false;              // 右键画布平移
    LeftDragMode _leftDragMode = LeftDragMode::None;
    QVector2D _dragAxisDir;                   // 单轴旋转时的切线方向（屏幕空间）
    // Arcball 按下时记录的状态（用于从初始旋转叠加增量，避免浮点漂移）
    QVector3D  _arcballPressVector;  // 按下时球面坐标向量
    QQuaternion _viewRotAtPress;     // 按下时的视图旋转快照
    bool _showGizmo = true;                       // 操控球是否可见（默认可见）
    bool _showCameras = true;                     // 相机平面卡片和文件名标签是否可见（默认可见）
    bool _showCameraThumbnails = true;
    bool _showCameraLocalAxes = false;
    bool _showCameraImage = false;
    CameraImageDisplayLayer _cameraImageDisplayLayer = CameraImageDisplayLayer::Foreground;
    bool _cameraImageLocked = false;
    int _activeCameraImagePoseIndex = -1;
    int _lockedCameraImagePoseIndex = -1;
    QString _lockedCameraImagePath;
    QString _lockedCameraImageName;
    QHash<QString, QImage> _cameraImageCache;
    QHash<QString, int> _poseIndexByNormalizedPath;
    QQueue<QString> _fullImageCacheLru;
    qint64 _fullImageCacheBytes = 0;
    QSet<int> _pendingThumbnailPoseIndices;
    QSet<QString> _cameraImageLoadsInFlight;
    QSet<QString> _cameraImageLoadsQueued;
    QQueue<CameraPlaneImageRequest> _cameraImageLoadQueue;
    QSet<QString> _cameraImageLoadFailures;
    int _maximumCameraImageLoads = 4;
    int _cameraImageLoadGeneration = 0;
    int _cameraThumbnailLoadTotal = 0;
    int _cameraThumbnailLoadCompleted = 0;
    bool _thumbnailUpdateScheduled = false;
    QSet<int> _thumbnailPoseIndicesPendingCommit;
    QSet<QString> _thumbnailCacheKeysPendingCommit;
    QString _highlightedCameraPath;
    bool _manualPruneMode = false;
    bool _manualSelecting = false;
    QPoint _manualSelectStart;
    QRect _manualSelectRect;
    std::vector<PointVertexIndex> _manualPreviewIndices;
    bool _manualPreviewValid = false;
    bool _manualPreviewUsesScreenRect = false;
    bool _manualSelectionRunning = false;
    bool _manualEditRunning = false;
    std::shared_ptr<std::atomic_bool> _manualEditCancellation;
    bool _manualDeletePending = false;
    int _manualSelectionGeneration = 0;
    std::optional<ManualSelectionRequest> _pendingManualSelection;
    std::shared_ptr<std::atomic_bool> _manualSelectionCancellation;
    bool _manualSelectionWorkerActive = false;
    QString _currentCloudPath;
    std::vector<PointCloudEditDelta> _manualUndoStack;
    int _manualUndoLimit = 10;
};
