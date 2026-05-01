// =============================================================================
// 文件: CameraModel3DDialog.h
// 功能: 相机三维模型可视化对话框声明
// 职责:
//   - CameraSceneWidget: 基于 OpenGL 4.x Core Profile 的三维交互场景控件，
//                        支持相机姿态、点云、PLY 网格模型的渲染与交互旋转/缩放/平移
//   - CameraModel3DDialog: 封装 CameraSceneWidget 的对话框，从项目元数据读取
//                          相机信息并在三维场景中展示
// =============================================================================

#pragma once

#include <QDialog>
#include <QVector3D>
#include <QVector2D>
#include <QQuaternion>
#include <QMatrix3x3>
#include <QVector>
#include <array>
#include <QOpenGLWidget>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QFutureWatcher>
#include <QRect>
#include <vector>
#include "data/PointCloud.h"

/// XYZ 文件异步加载结果（坐标 + 可选颜色）
/// 已代替为 xjw::pointcloud::PointCloud，保留此别名以兼容旧调用点
using XyzCloudData = xjw::pointcloud::PointCloud;

// Qt OpenGL 4.3 Core Profile 函数集（用于可编程管线渲染）
class QOpenGLFunctions_4_3_Core;

// 前向声明：项目管理器（提供当前项目的相机元数据）
class ProjectManager;
class QWidget;
class QLabel;

// =============================================================================
// CameraSceneWidget
// 继承自 QOpenGLWidget，提供基于 OpenGL 4.x Core Profile 可编程管线的三维场景渲染控件。
// 使用 VAO/VBO + GLSL shader（顶点色 shader + Phong 光照 shader）
// 功能包括：
//   - 渲染相机姿态（位置+视锥体）、点云（xyz 文件）、网格模型（PLY 文件）
//   - Arcball 自由旋转、单轴环旋转（X/Y/Z）
//   - 滚轮缩放、中键平移
//   - 实时显示坐标轴指示器和欧拉角信息
// =============================================================================
class CameraSceneWidget : public QOpenGLWidget
{
    Q_OBJECT
public:
    // -------------------------------------------------------------------------
    // CameraPose：单个相机的姿态信息
    //   name     - 相机/影像名称（用于 3D 场景中的文字标注）
    //   center   - 相机光心在世界坐标系中的位置（单位与项目坐标一致）
    //   rotation - 3×3 旋转矩阵 R（行主序），列向量依次为 right/up/forward
    // -------------------------------------------------------------------------
    struct CameraPose {
        QString name;       // 相机名称（用于标注）
        QVector3D center;   // 相机光心世界坐标
        QMatrix3x3 rotation; // 相机旋转矩阵（right/up/forward 列向量）
    };

    // 构造函数，初始化 OpenGL 控件并设置默认视角
    explicit CameraSceneWidget(QWidget *parent = nullptr);

    // 设置要渲染的相机姿态列表，触发重绘
    void setCameraPoses(const QVector<CameraPose> &poses);
    // 直接设置点云或网格模型数据（不启动异步 IO）。
    // cloud.hasFaces() == false 时作点云渲染，true 时作 Phong 网格渲染。
    void setPointCloud(const xjw::pointcloud::PointCloud &cloud);

    // setPointCloud 的网格语义别名（cloud 应包含面片）
    void setMesh(const xjw::pointcloud::PointCloud &mesh);
    // 从 XYZ 文本文件加载点云（每行 "x y z [r g b ...]"）
    void loadPointCloudFromXyz(const QString &xyzPath);

    // 从 PLY 文件加载三角网格或点云模型
    void loadModelFromPly(const QString &plyPath);

    bool setManualPruneModeEnabled(bool enabled, QString *errorMessage = nullptr);
    bool isManualPruneModeEnabled() const { return m_manualPruneMode; }
    bool saveCurrentPointCloudToSource(QString *errorMessage = nullptr);
    bool undoLastManualPrune(QString *errorMessage = nullptr);

    /// 设置是否显示操控球（Gizmo 旋转环），用于减少视线遮挡
    void setShowGizmo(bool show);
    /// 查询当前操控球是否可见
    bool isGizmoVisible() const { return m_showGizmo; }

signals:
    void manualPruneModeChanged(bool enabled);
    void manualPruneApplied(int removedCount, int remainingCount);
    void manualPruneUndone(int restoredCount);
    void manualPruneSaved(const QString &path, int remainingCount);
    void manualPruneSaveFailed(const QString &errorMessage);

protected:
    // OpenGL 初始化：获取函数对象、启用深度测试/混合/点平滑
    void initializeGL() override;

    // 视口尺寸变化时更新 glViewport
    void resizeGL(int w, int h) override;

    // 主渲染函数：清屏 → 绘制点云 → 绘制模型 → 绘制包围盒 → 绘制覆盖层（QPainter）
    void paintGL() override;

    // 鼠标按下：记录初始旋转状态，区分左键旋转与中键平移
    void mousePressEvent(QMouseEvent *event) override;

    // 鼠标移动：Arcball 自由旋转 / 单轴环旋转 / 中键平移 / 悬停高亮检测
    void mouseMoveEvent(QMouseEvent *event) override;

    // 鼠标释放：清除拖拽状态，恢复光标
    void mouseReleaseEvent(QMouseEvent *event) override;

    // 滚轮事件：缩放场景（调整 m_zoomScale）
    void wheelEvent(QWheelEvent *event) override;

    void keyPressEvent(QKeyEvent *event) override;

private:
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

    // 将三维世界点投影到屏幕像素坐标（考虑当前视图旋转、投影、平移偏移）
    // ok 为 nullptr 或 false 时表示点在裁剪空间外
    QPointF projectToScreen(const QVector3D &p, bool *ok = nullptr) const;

    // 将向量从本地空间旋转到当前视图空间（应用 m_viewRot）
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

    // 在 OpenGL 渲染完成后，用 QPainter 绘制 2D 覆盖层：
    //   - 操控球 Gizmo（旋转环）
    //   - 相机视锥体和名称标注
    //   - 世界原点标记
    //   - 右下角坐标轴指示器和欧拉角文字
    void drawOverlay();

    // 将点云/模型/包围盒数据上传到 GPU（VBO/VAO），在 paintGL 中按需调用
    void uploadGpuData();

    // 当点云/模型数据变更后调用，标记缓存失效并重新计算
    void invalidateCache() const;

    int removePointsInScreenRect(const QRect &screenRect);
    void pushManualUndoSnapshot(const xjw::pointcloud::PointCloud &snapshot);
    void collectPointIndicesInScreenRect(const QRect &screenRect, std::vector<std::size_t> *indices) const;

    // 取消未完成的异步加载并等待结束（在新加载开始前调用）
    void cancelPendingLoad();

    // OpenGL 4.3 Core Profile 函数对象
    QOpenGLFunctions_4_3_Core *m_gl = nullptr;
    bool m_gpuDirty = true;  // VBO 需要重新上传

    // Shader 程序
    QOpenGLShaderProgram *m_colorProgram = nullptr; // 点云/线框：pos+color 直通
    QOpenGLShaderProgram *m_meshProgram  = nullptr; // 网格：pos+normal+color Phong

    // 点云 GPU 资源
    QOpenGLVertexArrayObject m_pointVao;
    QOpenGLBuffer m_pointVbo{QOpenGLBuffer::VertexBuffer};
    int m_pointCount = 0;

    // 网格（三角面展开）GPU 资源
    QOpenGLVertexArrayObject m_meshVao;
    QOpenGLBuffer m_meshVbo{QOpenGLBuffer::VertexBuffer};
    int m_meshVertCount = 0;
    bool m_meshHasFaces = true;  ///< false 时用 GL_POINTS 绘制含法向量点云

    // 无面片模型（作为点云绘制）GPU 资源
    QOpenGLVertexArrayObject m_modelPtVao;
    QOpenGLBuffer m_modelPtVbo{QOpenGLBuffer::VertexBuffer};
    int m_modelPtCount = 0;
    float m_modelPointSize = 3.5f;

    // 包围盒线框 GPU 资源
    QOpenGLVertexArrayObject m_lineVao;
    QOpenGLBuffer m_lineVbo{QOpenGLBuffer::VertexBuffer};
    int m_lineCount = 0;

    QVector<CameraPose> m_poses;             // 当前相机姿态列表
    xjw::pointcloud::PointCloud m_cloud;     // 当前显示的点云或网格（源自文件或外部调用）

    // 场景中心/半径/包围盒缓存（避免每帧遍历大量点）
    mutable QVector3D  m_cachedCenter;
    mutable float      m_cachedRadius = 10.0f;
    mutable QVector3D  m_cachedAABBMin;
    mutable QVector3D  m_cachedAABBMax;
    mutable bool       m_cacheDirty = true;

    // 异步加载状态
    bool   m_loading      = false;
    int    m_loadGen      = 0;    ///< 每次发起新加载时递增，用于丢弃过期回调
    bool m_preferModelPointRender = false;
    QQuaternion m_viewRot;                     // 当前视图旋转四元数
    float m_zoomScale = 1.0f;                  // 当前缩放系数（影响相机到场景中心的距离）
    QPointF m_sceneOffsetPx = QPointF(0.0, 0.0); // 场景在屏幕空间的平移偏移（像素）
    QPoint m_lastMousePos;                     // 上一帧鼠标位置（用于增量计算）
    HoverAxis m_hoverAxis = HoverAxis::None;   // 当前鼠标悬停的轴（高亮显示）
    HoverAxis m_dragAxis = HoverAxis::None;    // 当前拖拽激活的轴
    bool m_leftDragging = false;               // 左键是否正在拖拽
    bool m_middleDragging = false;             // 中键是否正在拖拽（平移）
    QVector2D m_dragAxisDir;                   // 单轴旋转时的切线方向（屏幕空间）
    // Arcball 按下时记录的状态（用于从初始旋转叠加增量，避免浮点漂移）
    QVector3D  m_arcballPressVector;  // 按下时球面坐标向量
    QQuaternion m_viewRotAtPress;     // 按下时的视图旋转快照
    bool m_showGizmo = true;                       // 操控球是否可见（默认可见）
    bool m_manualPruneMode = false;
    bool m_manualSelecting = false;
    QPoint m_manualSelectStart;
    QRect m_manualSelectRect;
    std::vector<std::size_t> m_manualPreviewIndices;
    QString m_currentCloudPath;
    std::vector<xjw::pointcloud::PointCloud> m_manualUndoStack;
    int m_manualUndoLimit = 10;
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

    ProjectManager *m_projectManager = nullptr;  // 项目管理器（提供相机 JSON 元数据）
    CameraSceneWidget *m_scene = nullptr;         // 三维场景渲染控件
    QLabel *m_summaryLabel = nullptr;             // 底部摘要标签（显示相机数量及操作提示）
};
