// =============================================================================
// 文件: CameraModel3DDialog.h
// 功能: 相机三维模型可视化对话框声明
// 职责:
//   - CameraSceneWidget: 基于 Qt RHI/Vulkan 的三维交互场景控件，
//                        支持相机姿态、点云、PLY 网格和 OBJ/MTL 纹理模型渲染
//   - CameraModel3DDialog: 封装 CameraSceneWidget 的对话框，从项目元数据读取
//                          相机信息并在三维场景中展示
// =============================================================================

#pragma once

#include <QByteArray>
#include <QDialog>
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
#include <QThreadPool>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <array>
#include <cstddef>
#include <vector>
#include <plapoint/core/point_cloud.h>

#include "TiePointVisualization.h"
#include "ModelVisualization.h"

/// 渲染用点云类型别名
using RenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

// 前向声明：项目管理器（提供当前项目的相机元数据）
class ProjectManager;
class QWidget;
class QLabel;
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

// =============================================================================
// CameraSceneWidget
// 继承自 QRhiWidget，提供基于 Vulkan 后端的三维场景渲染控件。
// 使用 QRhiBuffer + .qsb shader（顶点色、Phong 光照和 UV 纹理 shader）
// 功能包括：
//   - 渲染相机姿态（位置+相机卡片）、点云（xyz）、PLY 网格和 OBJ/MTL 纹理模型
//   - Arcball 自由旋转、单轴环旋转（X/Y/Z）
//   - 滚轮缩放、中键平移
//   - 实时显示坐标轴指示器和欧拉角信息
// =============================================================================
class CameraSceneWidget : public QRhiWidget
{
    Q_OBJECT
public:
    using TiePointColorMode = xjw::gui::tie_points::ColorMode;
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

    // 构造函数，初始化 Vulkan 渲染控件并设置默认视角
    explicit CameraSceneWidget(QWidget *parent = nullptr);
    ~CameraSceneWidget() override;

    // 设置要渲染的相机姿态列表，触发重绘
    void setCameraPoses(const QVector<CameraPose> &poses);
    // 直接设置点云或网格模型数据（不启动异步 IO）。
    // cloud.hasFaces() == false 时作点云渲染，true 时作 Phong 网格渲染。
    void setPointCloud(const RenderCloud &cloud);

    // setPointCloud 的网格语义别名（cloud 应包含面片）
    void setMesh(const RenderCloud &mesh);
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
    TiePointColorMode tiePointColorMode() const { return _tiePointColorMode; }
    void setModelColorMode(ModelColorMode mode);
    ModelColorMode modelColorMode() const { return _modelColorMode; }

    bool setManualPruneModeEnabled(bool enabled, QString *errorMessage = nullptr);
    bool isManualPruneModeEnabled() const { return _manualPruneMode; }
    bool saveCurrentPointCloudToSource(QString *errorMessage = nullptr);
    bool undoLastManualPrune(QString *errorMessage = nullptr);

    /// 设置是否显示操控球（Gizmo 旋转环），用于减少视线遮挡
    void setShowGizmo(bool show);
    /// 查询当前操控球是否可见
    bool isGizmoVisible() const { return _showGizmo; }
    /// 设置是否显示相机平面卡片和文件名标签
    void setShowCameras(bool show);
    void setShowCameraThumbnails(bool show);
    /// 设置是否在每个相机光心显示其局部 X/Y/Z 坐标轴。
    void setShowCameraLocalAxes(bool show);
    void setShowCameraImage(bool show);
    void setCameraImagePlaneMode(CameraImagePlaneMode mode);
    void setCameraImageDisplayLayer(CameraImageDisplayLayer layer);
    void setCameraImageLocked(bool locked);
    void setHighlightedCameraPath(const QString &imagePath);
    void setHighlightedCameraName(const QString &imageName);
    void clearHighlightedCamera();
    // 清空当前项目的相机、模型和异步加载状态，并恢复默认观察视图。
    void clearProjectScene();
    /// 查询当前相机覆盖层是否可见
    bool areCamerasVisible() const { return _showCameras; }
    CameraImagePlaneMode cameraImagePlaneMode() const { return _cameraImagePlaneMode; }
    bool areCameraThumbnailsVisible() const { return _showCameraThumbnails; }
    bool areCameraLocalAxesVisible() const { return _showCameraLocalAxes; }
    bool isCameraImageVisible() const { return _showCameraImage; }
    CameraImageDisplayLayer cameraImageDisplayLayer() const { return _cameraImageDisplayLayer; }
    bool isCameraImageLocked() const { return _cameraImageLocked; }
    void zoomIn();
    void zoomOut();
    void resetView();

signals:
    void plyLoadProgressChanged(int generation, int percent, const QString &statusText);
    void manualPruneModeChanged(bool enabled);
    void manualPruneApplied(int removedCount, int remainingCount);
    void manualPruneUndone(int restoredCount);
    void manualPruneSaved(const QString &path, int remainingCount);
    void manualPruneSaveFailed(const QString &errorMessage);

protected:
    // RHI 初始化：创建 Vulkan 渲染资源和管线。
    void initialize(QRhiCommandBuffer *cb) override;

    // 主渲染函数：清屏 → 绘制点云/模型与相机图像层 → 更新覆盖层。
    void render(QRhiCommandBuffer *cb) override;

    // 释放 RHI 资源。
    void releaseResources() override;

    // 视口尺寸变化时标记资源和投影参数需要更新。
    void resizeEvent(QResizeEvent *event) override;

    // 鼠标按下：记录初始旋转状态，区分左键旋转与中键平移
    void mousePressEvent(QMouseEvent *event) override;

    // 鼠标移动：Arcball 自由旋转 / 单轴环旋转 / 中键平移 / 悬停高亮检测
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

    // 将向量从本地空间旋转到当前视图空间（应用 _viewRot）
    QVector3D applyViewRotation(const QVector3D &v) const;

    // 返回当前视图四元数对应的欧拉角（度数，顺序：pitch/yaw/roll）
    QVector3D eulerAnglesDeg() const;

    // 计算操控球（Gizmo 旋转环）在屏幕上的半径（像素），随窗口大小自适应
    qreal manipRadiusPx() const;

    // 返回操控球在屏幕空间的中心点（固定为窗口中心）
    QPointF manipCenterScreen() const;

    // 返回操控球在世界空间的中心点（等同于场景中心）
    QVector3D manipCenterWorld() const;

    // 计算场景中所有点（相机、点云、模型顶点）的质心
    QVector3D sceneCenter() const;

    // 计算场景中所有点到质心的最大距离（用于自适应缩放）
    float sceneRadius() const;

    // 根据鼠标位置检测当前最近的旋转轴环（距离 <12px 判定为悬停）
    HoverAxis pickHoverAxis(const QPoint &mousePos) const;

    // 计算指定轴环在鼠标附近切线方向（单位向量，用于单轴旋转的拖拽量计算）
    QVector2D pickAxisTangent(const QPoint &mousePos, HoverAxis axis) const;

    // Arcball：将屏幕位置映射到单位球面（xy=屏幕坐标系，z指向观察者）
    // 若鼠标在球内则返回球面点，否则投影到赤道圆
    QVector3D arcballVector(const QPoint &mousePos) const;

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
    int displayedCameraImagePoseIndex() const;
    QVector<QVector3D> displayedCameraImagePlaneCorners() const;
    QPainterPath foregroundCameraImageOcclusionPath() const;
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
    void drawPointCloudOverlay(QPainter &painter) const;
    void drawRotationGizmo(QPainter &painter) const;
    void drawTiePointLegend(QPainter &painter) const;
    void drawModelLegend(QPainter &painter) const;
    void startTiePointMetadataLoad(const QString &sidecarPath, int generation);
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
    void fitViewToLoadedGeometry();

    // 将点云和模型数据整理为 RHI 顶点缓冲，在 render 中按需上传。
    void uploadGpuData();
    // 点云使用单采样，三角网格使用 4x MSAA；切换资源时重建匹配的管线。
    void updateSampleCountForGeometry();

    // 当点云/模型数据变更后调用，标记缓存失效并重新计算
    void invalidateCache() const;

    int removePointsInScreenRect(const QRect &screenRect);
    void pushManualUndoSnapshot(RenderCloud snapshot);
    void collectPointIndicesInScreenRect(const QRect &screenRect, std::vector<std::size_t> *indices) const;
    void queueCurrentPointCloudSave();
    void startCurrentPointCloudSave();

    // 取消未完成的异步加载并等待结束（在新加载开始前调用）
    void cancelPendingLoad();

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
        QScopedPointer<QRhiBuffer> vertexBuffer;
        QScopedPointer<QRhiTexture> texture;
        QScopedPointer<QRhiShaderResourceBindings> bindings;
        int vertexCapacity = 0;
    };

    struct RhiCameraThumbnailAtlasPage
    {
        QScopedPointer<QRhiBuffer> vertexBuffer;
        QScopedPointer<QRhiTexture> texture;
        QScopedPointer<QRhiShaderResourceBindings> bindings;
        QSet<int> uploadedPoseIndices;
        QHash<int, QSize> imageSizes;
        int vertexCapacity = 0;
    };

    struct RhiCameraThumbnailPipelineSet
    {
        QScopedPointer<QRhiBuffer> uniformBuffer;
        QScopedPointer<QRhiSampler> sampler;
        QScopedPointer<QRhiGraphicsPipeline> pipeline;
        QVector<QSharedPointer<RhiCameraThumbnailAtlasPage>> atlasPages;
        QSharedPointer<RhiCameraThumbnailResource> solidResource;
        QSharedPointer<RhiCameraThumbnailResource> highlightedSolidResource;
        QScopedPointer<QRhiBuffer> leaderBuffer;
        int atlasSize = 0;
        bool resourcesDirty = true;
    };

    struct alignas(16) SceneUniforms
    {
        std::array<float, 16> mvp{};
        std::array<float, 16> modelView{};
        std::array<float, 16> normalMatrix{};
        std::array<float, 4> lightDirPointSize{};
        std::array<float, 4> viewportSize{};
    };
    static_assert(offsetof(SceneUniforms, mvp) == 0);
    static_assert(offsetof(SceneUniforms, modelView) == 16 * sizeof(float));
    static_assert(offsetof(SceneUniforms, normalMatrix) == 32 * sizeof(float));
    static_assert(offsetof(SceneUniforms, lightDirPointSize) == 48 * sizeof(float));
    static_assert(offsetof(SceneUniforms, viewportSize) == 52 * sizeof(float));
    static_assert(sizeof(SceneUniforms) == 56 * sizeof(float));

    struct ImagePlaneUniforms
    {
        QMatrix4x4 mvp;
    };

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
    bool _pipelinesDirty = true;
    QString _renderError;
    CameraSceneOverlayWidget *_overlayWidget = nullptr;

    bool ensureRhiBuffer(RhiBufferSet *buffer, QRhiResourceUpdateBatch *updates);
    bool ensurePipeline(RhiPipelineSet *pipeline,
                        int topology,
                        int strideBytes,
                        bool hasNormals,
                        bool depthWrite = true);
    bool ensurePointPipeline();
    bool ensureTexturedMeshPipeline(QRhiResourceUpdateBatch *updates);
    bool ensureImagePipeline(QRhiResourceUpdateBatch *updates);
    bool ensureCameraThumbnailPipeline(QRhiResourceUpdateBatch *updates);
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
    void drawPointCloud(QRhiCommandBuffer *cb, const SceneUniforms &uniforms);
    void drawTexturedMesh(QRhiCommandBuffer *cb, const SceneUniforms &uniforms);
    void drawActiveCameraImage(QRhiCommandBuffer *cb, const QMatrix4x4 &mvp);
    void drawCameraThumbnails(QRhiCommandBuffer *cb,
                              const QMatrix4x4 &mvp,
                              const SceneUniforms &sceneUniforms);
    void drawSceneGeometry(QRhiCommandBuffer *cb, SceneUniforms &uniforms);

    // 点云 GPU 资源
    RhiBufferSet _pointBuffer;
    int _pointCount = 0;
    float _pointCloudPointSize = 2.4f;

    // 网格（三角面展开）GPU 资源
    RhiBufferSet _meshBuffer;
    RhiBufferSet _modelWireframeBuffer;
    int _meshVertCount = 0;
    int _modelWireframeVertCount = 0;
    bool _meshHasFaces = true;  ///< false 时用点图元绘制含法向量点云
    bool _preparedObjMeshBuffer = false;
    bool _preparedObjMeshHasTexture = false;
    QByteArray _preparedObjVertexData;
    int _preparedObjVertexCount = 0;
    int _preparedObjStrideBytes = 0;

    // 点云包围盒 GPU 资源。
    RhiBufferSet _lineBuffer;
    int _lineCount = 0;

    RhiPipelineSet _colorPointPipeline;
    RhiPipelineSet _colorLinePipeline;
    RhiPipelineSet _cameraLeaderPipeline;
    RhiPipelineSet _meshTrianglePipeline;
    RhiPipelineSet _meshPointPipeline;
    RhiTexturedMeshPipelineSet _texturedMeshPipeline;
    RhiImagePipelineSet _imagePipeline;
    RhiCameraThumbnailPipelineSet _thumbnailPipeline;

    QVector<CameraPose> _poses;             // 当前相机姿态列表
    RenderCloud _cloud;     // 当前显示的点云或网格（源自文件或外部调用）
    bool _isTiePointCloud = false;
    TiePointColorMode _tiePointColorMode = TiePointColorMode::Color;
    ModelColorMode _modelColorMode = ModelColorMode::Shaded;
    xjw::gui::model_views::ModelVisualizationManager _modelVisualization;
    QVector<int> _tiePointImageCounts;
    xjw::gui::tie_points::ScalarRange _tiePointElevationRange;
    xjw::gui::tie_points::ScalarRange _tiePointImageCountRange;
    xjw::gui::tie_points::ScalarRange _modelElevationRange;
    bool _tiePointMetadataLoading = false;
    QString _tiePointMetadataError;
    QImage _meshTextureImage;
    QString _meshTexturePath;
    bool _meshHasTexture = false;

    // 场景中心/半径/包围盒缓存（避免每帧遍历大量点）
    mutable QVector3D  _cachedCenter;
    mutable float      _cachedRadius = 10.0f;
    mutable QVector3D  _cachedAABBMin;
    mutable QVector3D  _cachedAABBMax;
    mutable QVector3D  _cachedCloudAABBMin;
    mutable QVector3D  _cachedCloudAABBMax;
    mutable bool       _hasCloudBounds = false;
    mutable bool       _cacheDirty = true;
    QVector3D _focusedGeometryCenter;
    float _focusedGeometryRadius = 1.0f;
    bool _hasFocusedGeometryBounds = false;

    // 异步加载状态
    bool   _loading      = false;
    int    _loadGen      = 0;    ///< 每次发起新加载时递增，用于丢弃过期回调
    int    _plyLoadProgressPercent = -1;
    QString _plyLoadProgressText;
    bool _fitViewAfterLoad = false;
    QQuaternion _viewRot;                     // 当前视图旋转四元数
    double _zoomScale = 1.0;                  // 当前缩放系数（影响相机到场景中心的距离）
    QPointF _sceneOffsetPx = QPointF(0.0, 0.0); // 场景在屏幕空间的平移偏移（像素）
    QPoint _lastMousePos;                     // 上一帧鼠标位置（用于增量计算）
    HoverAxis _hoverAxis = HoverAxis::None;   // 当前鼠标悬停的轴（高亮显示）
    HoverAxis _dragAxis = HoverAxis::None;    // 当前拖拽激活的轴
    bool _leftDragging = false;               // 左键是否正在拖拽
    bool _middleDragging = false;             // 中键是否正在拖拽（平移）
    QVector2D _dragAxisDir;                   // 单轴旋转时的切线方向（屏幕空间）
    // Arcball 按下时记录的状态（用于从初始旋转叠加增量，避免浮点漂移）
    QVector3D  _arcballPressVector;  // 按下时球面坐标向量
    QQuaternion _viewRotAtPress;     // 按下时的视图旋转快照
    bool _showGizmo = true;                       // 操控球是否可见（默认可见）
    bool _showCameras = true;                     // 相机平面卡片和文件名标签是否可见（默认可见）
    CameraImagePlaneMode _cameraImagePlaneMode = CameraImagePlaneMode::Thumbnail;
    bool _showCameraThumbnails = true;
    bool _showCameraLocalAxes = false;
    bool _showCameraImage = false;
    CameraImageDisplayLayer _cameraImageDisplayLayer = CameraImageDisplayLayer::Foreground;
    bool _cameraImageLocked = false;
    int _activeCameraImagePoseIndex = -1;
    QString _lockedCameraImagePath;
    QString _lockedCameraImageName;
    QHash<QString, QImage> _cameraImageCache;
    QSet<QString> _cameraImageLoadsInFlight;
    QSet<QString> _cameraImageLoadsQueued;
    QQueue<CameraPlaneImageRequest> _cameraImageLoadQueue;
    QSet<QString> _cameraImageLoadFailures;
    QThreadPool _cameraImageLoadPool;
    int _cameraImageLoadGeneration = 0;
    int _cameraThumbnailLoadTotal = 0;
    int _cameraThumbnailLoadCompleted = 0;
    QString _highlightedCameraPath;
    QString _highlightedCameraName;
    bool _manualPruneMode = false;
    bool _manualSelecting = false;
    QPoint _manualSelectStart;
    QRect _manualSelectRect;
    std::vector<std::size_t> _manualPreviewIndices;
    bool _manualPreviewValid = false;
    QString _currentCloudPath;
    std::vector<RenderCloud> _manualUndoStack;
    int _manualUndoLimit = 10;
    bool _manualSaveRunning = false;
    bool _manualSavePending = false;
};

// =============================================================================
// CameraModel3DDialog
// 封装 CameraSceneWidget 的模态/非模态对话框。
// 从 ProjectManager 读取当前项目的相机姿态信息，在三维场景中可视化展示。
// 提供"刷新"按钮以重新从项目元数据加载数据。
// =============================================================================
class CameraModel3DDialog : public QDialog
{
    Q_OBJECT
public:
    // 构造函数，传入 ProjectManager 以便读取相机元数据
    explicit CameraModel3DDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

private slots:
    // 重新从项目元数据读取相机姿态并刷新场景（点击"刷新"按钮时触发）
    void reloadFromProject();

private:
    // 从 ProjectManager 中解析 JSON 格式的相机元数据，返回姿态列表
    // JSON 结构: { "images": [ { "camera": { "C": [x,y,z], "R": [3x3] }, "name": "..." }, ... ] }
    QVector<CameraSceneWidget::CameraPose> readCamerasFromMeta() const;

    ProjectManager *_projectManager = nullptr;  // 项目管理器（提供相机 JSON 元数据）
    CameraSceneWidget *_scene = nullptr;         // 三维场景渲染控件
    QLabel *_summaryLabel = nullptr;             // 底部摘要标签（显示相机数量及操作提示）
};
