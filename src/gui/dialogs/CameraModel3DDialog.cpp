// =============================================================================
// 文件: CameraModel3DDialog.cpp
// 功能: 相机三维模型可视化对话框实现
// 内容:
//   - CameraSceneWidget：OpenGL 4.x Core Profile 可编程管线渲染控件
//       · 点云 / PLY 模型 / 相机视锥体渲染（VAO/VBO + GLSL shader）
//       · Arcball 自由旋转 + 单轴环旋转（X/Y/Z Gizmo）
//       · 中键平移、滚轮缩放
//       · QPainter 覆盖层（Gizmo 环、坐标轴、相机视锥体、欧拉角）
//   - CameraModel3DDialog：对话框 UI + 从 ProjectManager 读取相机姿态
// =============================================================================
#include "CameraModel3DDialog.h"
#include "ui_CameraModel3DDialog.h"

#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include "Logger.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCursor>
#include <QPixmap>
#include <QVector2D>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLVersionFunctionsFactory>
#include <QSurfaceFormat>
#include <QJsonObject>
#include <QJsonArray>
#include <QMatrix4x4>
#include <QtMath>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QFileInfo>
#include <QKeyEvent>
#include <QKeySequence>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <plapoint/io/xyz_io.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/obj_io.h>

namespace
{

RenderCloud cloneRenderCloud(const RenderCloud &src)
{
    RenderCloud dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        for (int d = 0; d < 3; ++d)
            dst.points()(static_cast<plamatrix::Index>(i), d) = src.points()(static_cast<plamatrix::Index>(i), d);
    if (src.hasColors())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> c(src.size(), 3);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 3; ++d)
                c(static_cast<plamatrix::Index>(i), d) = src.colors()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setColors(std::move(c));
    }
    if (src.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> n(src.size(), 3);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 3; ++d)
                n(static_cast<plamatrix::Index>(i), d) = src.normals()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setNormals(std::move(n));
    }
    if (src.hasTextureCoords())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> t(src.size(), 2);
        for (size_t i = 0; i < src.size(); ++i)
            for (int d = 0; d < 2; ++d)
                t(static_cast<plamatrix::Index>(i), d) = src.textureCoords()->getValue(static_cast<plamatrix::Index>(i), d);
        dst.setTextureCoords(std::move(t));
    }
    if (src.hasFaces())
    {
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> f(src.faces()->rows(), 3);
        for (int i = 0; i < src.faces()->rows(); ++i)
            for (int d = 0; d < 3; ++d)
                f(i, d) = src.faces()->getValue(i, d);
        dst.setFaces(std::move(f));
    }
    dst.setMaterialLibraryFile(src.materialLibraryFile());
    dst.setTextureImageFile(src.textureImageFile());
    return dst;
}

} // namespace

// 构造函数：设置最小尺寸，启用鼠标追踪（悬停检测需要），
// 设置默认视角为俯仰 -25°、偏航 35°（斜上方看向场景）
CameraSceneWidget::CameraSceneWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    setFormat(format);

    setMinimumSize(760, 520);
    setMouseTracking(true); // 启用鼠标追踪，以便在无按键时检测悬停轴
    m_viewRot = QQuaternion::fromEulerAngles(-25.0f, 35.0f, 0.0f); // 默认斜视角
    setFocusPolicy(Qt::StrongFocus);
    updateCursor();
}

// 设置要渲染的相机姿态列表。
// 调用后触发重绘，场景中每个姿态点将绘制光心标记和视锥体框线。
void CameraSceneWidget::setCameraPoses(const QVector<CameraPose> &poses)
{
    m_poses = poses;
    m_cacheDirty = true; // 相机位置变更，缓存失效
    update(); // 触发 paintGL 重绘
}

void CameraSceneWidget::setShowGizmo(bool show)
{
    if (m_showGizmo != show) 
    {
        m_showGizmo = show;
        update(); // 触发重绘
    }
}

// 取消未完成的加载（递增 generation 令旧回调自行失效）
void CameraSceneWidget::cancelPendingLoad()
{
    ++m_loadGen;
}

// 标记缓存脏 + 重算（在加载完成后或场景数据变更后调用）
void CameraSceneWidget::invalidateCache() const
{
    bool has = false;
    QVector3D acc(0, 0, 0);
    int count = 0;
    QVector3D mn(0, 0, 0), mx(0, 0, 0);

    auto accum = [&](const QVector3D &p)
    {
        acc += p; ++count;
        if (!has) { mn = p; mx = p; has = true; return; }
        mn.setX(qMin(mn.x(), p.x())); mn.setY(qMin(mn.y(), p.y())); mn.setZ(qMin(mn.z(), p.z()));
        mx.setX(qMax(mx.x(), p.x())); mx.setY(qMax(mx.y(), p.y())); mx.setZ(qMax(mx.z(), p.z()));
    };

    for (const auto &p : m_poses)             accum(p.center);
    for (size_t i = 0; i < m_cloud.size(); ++i)
    {
        accum(QVector3D(m_cloud.points()(static_cast<plamatrix::Index>(i), 0),
                        m_cloud.points()(static_cast<plamatrix::Index>(i), 1),
                        m_cloud.points()(static_cast<plamatrix::Index>(i), 2)));
    }

    if (count <= 0)
    {
        m_cachedCenter  = QVector3D(0, 0, 0);
        m_cachedRadius  = 10.0f;
        m_cachedCameraFrustumBase = 0.6f;
        m_cachedAABBMin = QVector3D(-10, -10, -10);
        m_cachedAABBMax = QVector3D( 10,  10,  10);
    }
    else
    {
        m_cachedCenter  = acc / float(count);
        m_cachedAABBMin = mn;
        m_cachedAABBMax = mx;
        // 使用 95th 百分位距离作为场景半径，避免离群点把相机推太远
        std::vector<float> dists;
        dists.reserve(m_poses.size() + m_cloud.size());
        for (const auto &p : m_poses)             dists.push_back((p.center - m_cachedCenter).length());
        for (size_t i = 0; i < m_cloud.size(); ++i)
        {
            const QVector3D qp(m_cloud.points()(static_cast<plamatrix::Index>(i), 0),
                               m_cloud.points()(static_cast<plamatrix::Index>(i), 1),
                               m_cloud.points()(static_cast<plamatrix::Index>(i), 2));
            dists.push_back((qp - m_cachedCenter).length());
        }
        if (!dists.empty())
        {
            std::sort(dists.begin(), dists.end());
            const int p95 = std::min((int)dists.size() - 1, (int)(dists.size() * 0.95));
            m_cachedRadius = qMax(1.0f, dists[p95] * 1.15f);
        }
        else
        {
            m_cachedRadius = 1.0f;
        }

        m_cachedCameraFrustumBase = qMax(0.1f, m_cachedRadius * 0.02f);
        if (m_poses.size() > 1)
        {
            std::vector<float> nearestDistances;
            nearestDistances.reserve(static_cast<size_t>(m_poses.size()));
            for (qsizetype i = 0; i < m_poses.size(); ++i)
            {
                float nearest = std::numeric_limits<float>::max();
                for (qsizetype j = 0; j < m_poses.size(); ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }
                    nearest = qMin(nearest, (m_poses[i].center - m_poses[j].center).length());
                }
                if (std::isfinite(nearest))
                {
                    nearestDistances.push_back(nearest);
                }
            }
            if (!nearestDistances.empty())
            {
                const size_t medianIndex = nearestDistances.size() / 2;
                std::nth_element(nearestDistances.begin(),
                                 nearestDistances.begin() + static_cast<std::ptrdiff_t>(medianIndex),
                                 nearestDistances.end());
                const float spacingBase = nearestDistances[medianIndex] * 0.25f;
                m_cachedCameraFrustumBase = qMax(0.1f, qMin(m_cachedCameraFrustumBase, spacingBase));
            }
        }
    }
    m_cacheDirty = false;
}

// 直接设置点云或网格（cloud.hasFaces() 决定渲染模式）
void CameraSceneWidget::setPointCloud(const RenderCloud &cloud)
{
    cancelPendingLoad();
    m_cloud = cloneRenderCloud(cloud);
    m_currentCloudPath.clear();
    m_preferModelPointRender = false;
    m_cacheDirty = true;
    m_gpuDirty   = true;
    update();
}

void CameraSceneWidget::setMesh(const RenderCloud &mesh)
{
    setPointCloud(mesh); // 实现相同，语义区分（mesh 应含面片）
}

// 从 XYZ 格式文本文件异步加载点云数据，使用 plapoint IO 解析。
void CameraSceneWidget::loadPointCloudFromXyz(const QString &xyzPath)
{
    cancelPendingLoad();
    m_currentCloudPath = xyzPath;
    m_cloud = RenderCloud();
    m_preferModelPointRender = false;
    m_cacheDirty = true;
    m_gpuDirty   = true;
    LOG_INFO(QStringLiteral("[3D] 正在加载点云: %1").arg(xyzPath));

    const int gen = m_loadGen;
    auto *watcher = new QFutureWatcher<std::shared_ptr<RenderCloud>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,
            this, [this, watcher, gen]()
    {
        if (gen == m_loadGen)
        {
            auto result = watcher->result();
            if (result)
            {
                m_cloud = std::move(*result);
                LOG_INFO(QStringLiteral("[3D] 点云加载完成，共 %1 点")
                    .arg(m_cloud.size()));
                invalidateCache();
                m_gpuDirty = true;
                update();
            }
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([xyzPath]() -> std::shared_ptr<RenderCloud>
    {
        try
        {
            return plapoint::io::readXyz<float>(xyzPath.toStdString());
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] XYZ 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        return nullptr;
    }));
}

// 从 PLY 文件异步加载网格模型或点云。
void CameraSceneWidget::loadModelFromPly(const QString &plyPath)
{
    cancelPendingLoad();
    m_currentCloudPath = plyPath;
    m_cloud = RenderCloud();
    m_preferModelPointRender = true;
    m_cacheDirty = true;
    m_gpuDirty   = true;
    LOG_INFO(QStringLiteral("[3D] 正在加载模型: %1").arg(plyPath));

    const int gen = m_loadGen;
    auto *watcher = new QFutureWatcher<std::shared_ptr<RenderCloud>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,
            this, [this, watcher, gen]()
    {
        if (gen == m_loadGen)
        {
            auto result = watcher->result();
            if (result) m_cloud = std::move(*result);
            m_preferModelPointRender = !m_cloud.hasFaces();
            LOG_INFO(QStringLiteral("[3D] 模型加载完成，共 %1 顶点 / %2 面%3")
                     .arg(m_cloud.size())
                     .arg(m_cloud.hasFaces() ? static_cast<int>(m_cloud.faces()->rows()) : 0)
                     .arg(m_cloud.hasColors() ? QStringLiteral("（含RGB颜色）")
                                              : QStringLiteral("（无颜色）")));
            invalidateCache();
            m_gpuDirty = true;
            update();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([plyPath]() -> std::shared_ptr<RenderCloud>
    {
        try
        {
            return plapoint::io::readPly<float>(plyPath.toStdString());
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] PLY 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        return nullptr;
    }));
}

void CameraSceneWidget::loadModelFromObj(const QString &objPath)
{
    cancelPendingLoad();
    m_currentCloudPath = objPath;
    m_cloud = RenderCloud();
    m_preferModelPointRender = false;
    m_cacheDirty = true;
    m_gpuDirty = true;
    LOG_INFO(QStringLiteral("[3D] 正在加载 OBJ 模型: %1").arg(objPath));

    const int gen = m_loadGen;
    auto *watcher = new QFutureWatcher<std::shared_ptr<RenderCloud>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<RenderCloud>>::finished,
            this, [this, watcher, gen]()
    {
        if (gen == m_loadGen)
        {
            auto result = watcher->result();
            if (result)
            {
                m_cloud = std::move(*result);
            }
            m_preferModelPointRender = !m_cloud.hasFaces();
            LOG_INFO(QStringLiteral("[3D] OBJ 模型加载完成，共 %1 顶点 / %2 面")
                         .arg(m_cloud.size())
                         .arg(m_cloud.hasFaces() ? static_cast<int>(m_cloud.faces()->rows()) : 0));
            invalidateCache();
            m_gpuDirty = true;
            update();
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([objPath]() -> std::shared_ptr<RenderCloud>
    {
        try
        {
            return plapoint::io::readObj<float>(objPath.toStdString());
        }
        catch (const std::exception &e)
        {
            LOG_ERROR(QStringLiteral("[3D] OBJ 加载失败: %1").arg(QString::fromStdString(e.what())));
        }
        return nullptr;
    }));
}

// 计算场景中所有点（相机光心、点云、模型顶点）的质心作为场景中心。
// 使用缓存，仅在数据变更后重新计算。
QVector3D CameraSceneWidget::sceneCenter() const
{
    if (m_cacheDirty) invalidateCache();
    return m_cachedCenter;
}

// 计算场景中所有点到质心的最大距离，用于自适应相机距离、投影远裁平面等。
// 使用缓存，仅在数据变更后重新计算。
float CameraSceneWidget::sceneRadius() const
{
    if (m_cacheDirty) invalidateCache();
    return m_cachedRadius;
}

QPointF CameraSceneWidget::projectToScreen(const QVector3D &p, bool *ok) const
{
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = qMax(radius * 0.001f, radius * (3.2f / qMax(0.1f, m_zoomScale)));

    // 从 +Z 方向看向原点：世界X→屏幕右，世界Y→屏幕上，与overlay/arcball坐标系一致
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);

    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));

    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(m_viewRot);
    model.translate(-center);

    const float nearPlane = qMax(1e-4f, distance * 0.001f);
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);
    QMatrix4x4 proj;
    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    proj.perspective(45.0f, aspect, nearPlane, farPlane);

    QVector4D clip = proj * view * model * QVector4D(p, 1.0f);
    if (clip.w() <= 1e-6f) {
        if (ok) *ok = false;
        return QPointF();
    }
    QVector3D ndc(clip.x() / clip.w(), clip.y() / clip.w(), clip.z() / clip.w());
    if (ok) *ok = true;
    const float sx = (ndc.x() * 0.5f + 0.5f) * width() + float(m_sceneOffsetPx.x());
    const float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * height() + float(m_sceneOffsetPx.y());
    return QPointF(sx, sy);
}

// 将向量从副本局部空间旋转到当前视图空间（应用 m_viewRot）
QVector3D CameraSceneWidget::applyViewRotation(const QVector3D &v) const
{
    return m_viewRot.rotatedVector(v);
}

// 返回当前视图四元数对应的欧拉角（度， x=pitch, y=yaw, z=roll）
QVector3D CameraSceneWidget::eulerAnglesDeg() const
{
    return m_viewRot.toEulerAngles();
}

// 根据窗口尺寸自适应计算 Gizmo 操控球屏幕半径（px）
// 范围：[30, min(w,h)*0.24]，基准为 min(w,h)*0.11
qreal CameraSceneWidget::manipRadiusPx() const
{
    const qreal base = qMin(width(), height()) * 0.11;
    return qBound<qreal>(30.0, base, qMin(width(), height()) * 0.24);
}

int CameraSceneWidget::maxVisibleCameraLabels() const
{
    return 40;
}

float CameraSceneWidget::cameraFrustumBase() const
{
    if (m_cacheDirty) invalidateCache();
    return m_cachedCameraFrustumBase;
}

// Gizmo 操控球的世界中心点（等于场景质心）
QVector3D CameraSceneWidget::manipCenterWorld() const
{
    return sceneCenter();
}

// Gizmo 操控球的屏幕中心点（固定为窗口中心）
QPointF CameraSceneWidget::manipCenterScreen() const
{
    return QPointF(width() * 0.5, height() * 0.5);
}

// 根据鼠标位置检测鼠标当前悬停的 Gizmo 展向轴环。
// 原理：遍历 X/Y/Z 三个轴环的散列点，计算鼠标到环上最近线段的距离，
// 距离小于阈値 12px 时判定为悬停在该轴环上。
CameraSceneWidget::HoverAxis CameraSceneWidget::pickHoverAxis(const QPoint &mousePos) const
{
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 计算鼠标到指定轴环的最近距离（只考虑正面可见的弧段）
    auto minDistToCircle = [&](HoverAxis axis) {
        qreal best = 1e9;
        QPointF prev;
        bool hasPrev = false;
        bool prevVisible = false;
        for (int i = 0; i <= 96; ++i) {
            const qreal t = (2.0 * M_PI * i) / 96.0;
            QVector3D pLocal;
            // 根据轴选择环面上的局部点
            if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
            else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
            else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
            QVector3D pView = applyViewRotation(pLocal);
            const bool currVisible = (pView.z() > 0.0f); // z>0 表示正面对观察者
            QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
            if (hasPrev && prevVisible && currVisible) {
                // 计算鼠标到线段 [prev, curr] 的最近距离
                const QPointF ab = curr - prev;
                const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
                if (ab2 > 1e-6) {
                    qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                    u = qBound<qreal>(0.0, u, 1.0);
                    QPointF proj = prev + ab * u;
                    best = qMin(best, QLineF(QPointF(mousePos), proj).length());
                }
            }
            prev = curr;
            prevVisible = currVisible;
            hasPrev = true;
        }
        return best;
    };

    const qreal th = 12.0; // 距离阈値（像素）
    const qreal dx = minDistToCircle(HoverAxis::X);
    const qreal dy = minDistToCircle(HoverAxis::Y);
    const qreal dz = minDistToCircle(HoverAxis::Z);
    const qreal dmin = qMin(dx, qMin(dy, dz));
    if (dmin > th) return HoverAxis::None; // 距离过远，无悬停
    // 返回距离最小的轴
    if (dx <= dy && dx <= dz) return HoverAxis::X;
    if (dy <= dx && dy <= dz) return HoverAxis::Y;
    return HoverAxis::Z;
}

// 计算鼠标附近指定轴环的切线方向（屏幕空间单位向量）。
// 用于将鼠标拖拽距离投影到切线方向以计算旋转角度。
QVector2D CameraSceneWidget::pickAxisTangent(const QPoint &mousePos, HoverAxis axis) const
{
    if (axis == HoverAxis::None) return QVector2D(1.0f, 0.0f);
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 遍历环面散列点，找到鼠标最近的线段 [bestA, bestB]
    QPointF bestA;
    QPointF bestB;
    qreal bestDist = 1e12;
    QPointF prev;
    bool hasPrev = false;
    bool prevVisible = false;
    for (int i = 0; i <= 128; ++i) 
    {
        const qreal t = (2.0 * M_PI * i) / 128.0;
        QVector3D pLocal;
        if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
        else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
        else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
        const QVector3D pView = applyViewRotation(pLocal);
        const bool currVisible = (pView.z() > 0.0f);
        const QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
        if (hasPrev && prevVisible && currVisible) 
        {
            const QPointF ab = curr - prev;
            const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (ab2 > 1e-9) 
            {
                qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                u = qBound<qreal>(0.0, u, 1.0);
                const QPointF proj = prev + ab * u;
                const qreal d = QLineF(QPointF(mousePos), proj).length();
                if (d < bestDist) 
                {
                    bestDist = d;
                    bestA = prev;  // 最近线段起点
                    bestB = curr;  // 最近线段终点
                }
            }
        }
        prev = curr;
        prevVisible = currVisible;
        hasPrev = true;
    }

    // 计算并归一化切线方向向量
    QVector2D tanDir(float(bestB.x() - bestA.x()), float(bestB.y() - bestA.y()));
    if (tanDir.lengthSquared() < 1e-8f) tanDir = QVector2D(1.0f, 0.0f); // 防止除以零
    return tanDir.normalized();
}

// ---------------------------------------------------------------------------
// Arcball 球面投影
//   屏幕投影规则： pView.x → 屏幕右， pView.y → 屏幕上（y已翻转）， pView.z → 朝向观察者
//   注：与 drawGreatCircle 里 cursor2d + (pView.x*r, -pView.y*r) 保持一致
// ---------------------------------------------------------------------------
QVector3D CameraSceneWidget::arcballVector(const QPoint &mousePos) const
{
    const QPointF c = manipCenterScreen();
    const float r = float(manipRadiusPx());
    if (r < 1.0f) return QVector3D(0.0f, 0.0f, 1.0f);

    const float x =  float(mousePos.x() - c.x()) / r;
    const float y = -float(mousePos.y() - c.y()) / r;  // 屏幕 Y 选转为那数学 Y
    const float len2 = x * x + y * y;
    if (len2 <= 1.0f) {
        return QVector3D(x, y, std::sqrt(1.0f - len2));   // 在球面上
    }
    // 在球外：投影到赤道圆
    const float len = std::sqrt(len2);
    return QVector3D(x / len, y / len, 0.0f);
}

// 平移偏移量无限制（允许将模型拖出视口外）
void CameraSceneWidget::clampSceneOffset()
{
    // 不做任何限制
}

// 根据当前的悬停轴/拖拽状态更新鼠标光标样式：
//   - 中键拖拽中        → 四向移动光标
//   - 左键自由旋转中    → 闭合手型光标
//   - 悬停/拖拽 X/Y/Z 环 → 带颜色和字母的自定义圆形光标
//   - 默认              → 开放手型光标
void CameraSceneWidget::updateCursor()
{
    if (m_manualPruneMode && m_manualSelecting)
    {
        setCursor(Qt::CrossCursor);
        return;
    }

    // 创建带颜色和轴标签的自定义光标（24x24 像素，圆心热点）
    auto axisCursor = [](const QColor &color, const QString &label) {
        QPixmap pm(24, 24);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(color, 2));
        p.drawEllipse(QPointF(12, 12), 8, 8);
        p.drawText(QRect(0, 0, 24, 24), Qt::AlignCenter, label);
        return QCursor(pm, 12, 12);
    };

    if (m_middleDragging) {
        setCursor(Qt::SizeAllCursor); // 中键平移中
        return;
    }
    if (m_leftDragging && m_dragAxis == HoverAxis::None) {
        setCursor(Qt::ClosedHandCursor); // Arcball 自由旋转中
        return;
    }

    // 优先显示当前拖拽轴（若无则显示悬停轴）
    HoverAxis axis = (m_leftDragging ? m_dragAxis : m_hoverAxis);
    if (axis == HoverAxis::X) {
        setCursor(axisCursor(QColor(255, 120, 120), QStringLiteral("X")));
    } else if (axis == HoverAxis::Y) {
        setCursor(axisCursor(QColor(120, 255, 120), QStringLiteral("Y")));
    } else if (axis == HoverAxis::Z) {
        setCursor(axisCursor(QColor(120, 180, 255), QStringLiteral("Z")));
    } else {
        setCursor(Qt::OpenHandCursor); // 空闲时显示开放手型
    }
}

// OpenGL 初始化：获取 GL 4.3 Core Profile 函数对象，编译 shader，创建 VAO/VBO
void CameraSceneWidget::initializeGL()
{
    m_gl = context() ? QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_3_Core>(context()) : nullptr;
    if (!m_gl) return;
    m_gl->initializeOpenGLFunctions();
    m_gl->glEnable(GL_DEPTH_TEST);
    m_gl->glEnable(GL_BLEND);
    m_gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    m_gl->glEnable(GL_PROGRAM_POINT_SIZE); // 允许 vertex shader 通过 gl_PointSize 控制点大小

    // ── 顶点色直通 shader（点云 + 线框）────────────────────────────────────
    static const char *colorVert =
        "#version 430 core\n"
        "layout(location=0) in vec3 aPos;\n"
        "layout(location=1) in vec3 aColor;\n"
        "uniform mat4 uMVP;\n"
        "uniform float uPointSize;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position  = uMVP * vec4(aPos, 1.0);\n"
        "    gl_PointSize = uPointSize;\n"
        "    vColor = aColor;\n"
        "}\n";
    static const char *colorFrag =
        "#version 430 core\n"
        "in vec3 vColor;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    fragColor = vec4(vColor, 1.0);\n"
        "}\n";

    m_colorProgram = new QOpenGLShaderProgram(this);
    m_colorProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,   colorVert);
    m_colorProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, colorFrag);
    m_colorProgram->link();

    // ── Phong 光照 shader（三角网格）────────────────────────────────────────
    static const char *meshVert =
        "#version 430 core\n"
        "layout(location=0) in vec3 aPos;\n"
        "layout(location=1) in vec3 aNormal;\n"
        "layout(location=2) in vec3 aColor;\n"
        "uniform mat4 uMVP;\n"
        "uniform mat3 uNormalMat;\n"
        "uniform float uPointSize;\n"
        "out vec3 vNormal;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position = uMVP * vec4(aPos, 1.0);\n"
        "    gl_PointSize = uPointSize;\n"
        "    vNormal = uNormalMat * aNormal;\n"
        "    vColor  = aColor;\n"
        "}\n";
    static const char *meshFrag =
        "#version 430 core\n"
        "in vec3 vNormal;\n"
        "in vec3 vColor;\n"
        "uniform vec3 uLightDir;\n"
        "out vec4 fragColor;\n"
        "vec3 srgbToLinear(vec3 c) {\n"
        "    return pow(max(c, vec3(0.0)), vec3(2.2));\n"
        "}\n"
        "vec3 linearToSrgb(vec3 c) {\n"
        "    return pow(clamp(c, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));\n"
        "}\n"
        "void main() {\n"
        "    vec3 n = normalize(vNormal);\n"
        "    if (!gl_FrontFacing) n = -n;\n"
        "    vec3 L = normalize(uLightDir);\n"
        "    float diff = max(dot(n, L), 0.0);\n"
        "    vec3 R = reflect(-L, n);\n"
        "    float spec = pow(max(R.z, 0.0), 32.0) * 0.25;\n"
        "    vec3 baseLinear = srgbToLinear(vColor);\n"
        "    vec3 litLinear = baseLinear * (0.55 + 0.75 * diff) + vec3(spec);\n"
        "    fragColor = vec4(linearToSrgb(litLinear), 1.0);\n"
        "}\n";

    m_meshProgram = new QOpenGLShaderProgram(this);
    m_meshProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,   meshVert);
    m_meshProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, meshFrag);
    m_meshProgram->link();

    // 创建 VAO/VBO（GL 资源，需在 context 活跃时创建）
    m_pointVao.create();   m_pointVbo.create();
    m_meshVao.create();    m_meshVbo.create();
    m_modelPtVao.create(); m_modelPtVbo.create();
    m_lineVao.create();    m_lineVbo.create();

    m_gpuDirty = true;
}

// 视口尺寸变化时更新 OpenGL 的视口矩形
void CameraSceneWidget::resizeGL(int w, int h)
{
    if (!m_gl) return;
    m_gl->glViewport(0, 0, w, h);
    m_gpuDirty = true; // 包围盒线框依赖 AABB，重建一次以防万一
}

// 将点云/模型/包围盒数据上传到 GPU（VBO/VAO）。
// 在 paintGL 检测到 m_gpuDirty 时调用，避免每帧重复上传。
void CameraSceneWidget::uploadGpuData()
{
    if (!m_gl) return;

    // ── 辅助：建立颜色直通 VAO（stride=6 floats: xyz rgb）────────────────
    auto setupColorVao = [this](QOpenGLVertexArrayObject &vao,
                                QOpenGLBuffer &vbo,
                                const QVector<float> &data)
    {
        vao.bind();
        vbo.bind();
        vbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 6 * sizeof(float);
        m_gl->glEnableVertexAttribArray(0);
        m_gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        m_gl->glEnableVertexAttribArray(1);
        m_gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        vbo.release();
        vao.release();
    };

    // ── 1. 点云（m_cloud，无面片且无法向量，颜色直通）──────────────────────────
    m_pointCount = 0;
    m_modelPtCount = 0;
    if (!(m_cloud.size() == 0) && !m_cloud.hasFaces() && !m_cloud.hasNormals()) {
        const bool hasColors = m_cloud.hasColors();
        QVector<float> data;
        data.reserve((int)m_cloud.size() * 6);
        for (std::size_t i = 0; i < m_cloud.size(); ++i) {
            data << m_cloud.points()(static_cast<plamatrix::Index>(i), 0)
                 << m_cloud.points()(static_cast<plamatrix::Index>(i), 1)
                 << m_cloud.points()(static_cast<plamatrix::Index>(i), 2);
            if (hasColors) {
                data << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 0) / 255.f
                     << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 1) / 255.f
                     << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 2) / 255.f;
            } else {
                data << 0.45f << 0.45f << 0.50f;
            }
        }
        if (m_preferModelPointRender) {
            setupColorVao(m_modelPtVao, m_modelPtVbo, data);
            m_modelPtCount = (int)m_cloud.size();
        } else {
            setupColorVao(m_pointVao, m_pointVbo, data);
            m_pointCount = (int)m_cloud.size();
        }
    }

    // ── 2. 网格（hasFaces）或含法向量点云（!hasFaces && hasNormals）──────────
    m_meshVertCount = 0;
    m_meshHasFaces = false;
    if (!(m_cloud.size() == 0) && m_cloud.hasFaces()) {
        m_meshHasFaces = true;
        const bool hasVertCol = m_cloud.hasColors();
        const bool hasNrm     = m_cloud.hasNormals();
        const std::size_t Nv  = m_cloud.size();

        // 计算（或复用）逐顶点法向量
        std::vector<QVector3D> vNormals(Nv);
        if (hasNrm) {
            for (std::size_t i = 0; i < Nv; ++i) {
                vNormals[i] = QVector3D(
                    m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0),
                    m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1),
                    m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2));
            }
        } else {
            const auto nF = static_cast<std::size_t>(m_cloud.faces()->rows());
            for (std::size_t fi = 0; fi < nF; ++fi) {
                const std::size_t i0 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
                const std::size_t i1 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
                const std::size_t i2 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
                if (i0 >= Nv || i1 >= Nv || i2 >= Nv) continue;
                float p0x = m_cloud.points()(static_cast<plamatrix::Index>(i0), 0);
                float p0y = m_cloud.points()(static_cast<plamatrix::Index>(i0), 1);
                float p0z = m_cloud.points()(static_cast<plamatrix::Index>(i0), 2);
                float p1x = m_cloud.points()(static_cast<plamatrix::Index>(i1), 0);
                float p1y = m_cloud.points()(static_cast<plamatrix::Index>(i1), 1);
                float p1z = m_cloud.points()(static_cast<plamatrix::Index>(i1), 2);
                float p2x = m_cloud.points()(static_cast<plamatrix::Index>(i2), 0);
                float p2y = m_cloud.points()(static_cast<plamatrix::Index>(i2), 1);
                float p2z = m_cloud.points()(static_cast<plamatrix::Index>(i2), 2);
                const QVector3D fn = QVector3D::crossProduct(
                    QVector3D(p1x - p0x, p1y - p0y, p1z - p0z),
                    QVector3D(p2x - p0x, p2y - p0y, p2z - p0z));
                vNormals[i0] += fn;
                vNormals[i1] += fn;
                vNormals[i2] += fn;
            }
            for (auto &n : vNormals) n.normalize();
        }

        // 展开面片为平坦顶点数组（stride=9 floats: xyz nxnynz rgb）
        QVector<float> data;
        data.reserve(static_cast<int>(m_cloud.faces()->rows()) * 3 * 9);
        const auto nFaces = static_cast<std::size_t>(m_cloud.faces()->rows());
        for (std::size_t fi = 0; fi < nFaces; ++fi) {
            const std::size_t i0 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 0));
            const std::size_t i1 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 1));
            const std::size_t i2 = static_cast<std::size_t>(m_cloud.faces()->getValue(static_cast<plamatrix::Index>(fi), 2));
            if (i0 >= Nv || i1 >= Nv || i2 >= Nv) continue;
            const std::size_t indices[3] = {i0, i1, i2};
            for (int vi = 0; vi < 3; ++vi) {
                const std::size_t idx = indices[vi];
                data << m_cloud.points()(static_cast<plamatrix::Index>(idx), 0)
                     << m_cloud.points()(static_cast<plamatrix::Index>(idx), 1)
                     << m_cloud.points()(static_cast<plamatrix::Index>(idx), 2);
                data << vNormals[idx].x() << vNormals[idx].y() << vNormals[idx].z();
                if (hasVertCol) {
                    data << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 0) / 255.f
                         << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 1) / 255.f
                         << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(idx), 2) / 255.f;
                } else {
                    data << 0.55f << 0.55f << 0.58f;
                }
            }
        }
        m_meshVao.bind();
        m_meshVbo.bind();
        m_meshVbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 9 * sizeof(float);
        m_gl->glEnableVertexAttribArray(0);
        m_gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        m_gl->glEnableVertexAttribArray(1);
        m_gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        m_gl->glEnableVertexAttribArray(2);
        m_gl->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(6 * sizeof(float)));
        m_meshVbo.release();
        m_meshVao.release();
        m_meshVertCount = data.size() / 9;
    } else if (!(m_cloud.size() == 0) && !m_cloud.hasFaces() && m_cloud.hasNormals()) {
        // 含法向量但无面片 → GL_POINTS + Phong 光照（稠密点云等）
        const bool hasColors = m_cloud.hasColors();
        const std::size_t Nv = m_cloud.size();
        QVector<float> data;
        data.reserve((int)Nv * 9);
        for (std::size_t i = 0; i < Nv; ++i) {
            data << m_cloud.points()(static_cast<plamatrix::Index>(i), 0)
                 << m_cloud.points()(static_cast<plamatrix::Index>(i), 1)
                 << m_cloud.points()(static_cast<plamatrix::Index>(i), 2);
            data << m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0)
                 << m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1)
                 << m_cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2);
            if (hasColors) {
                data << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 0) / 255.f
                     << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 1) / 255.f
                     << m_cloud.colors()->getValue(static_cast<plamatrix::Index>(i), 2) / 255.f;
            } else {
                data << 0.55f << 0.55f << 0.58f;
            }
        }
        m_meshVao.bind();
        m_meshVbo.bind();
        m_meshVbo.allocate(data.constData(), data.size() * sizeof(float));
        const int stride = 9 * sizeof(float);
        m_gl->glEnableVertexAttribArray(0);
        m_gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        m_gl->glEnableVertexAttribArray(1);
        m_gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(3 * sizeof(float)));
        m_gl->glEnableVertexAttribArray(2);
        m_gl->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                                    reinterpret_cast<void*>(6 * sizeof(float)));
        m_meshVbo.release();
        m_meshVao.release();
        m_meshVertCount = (int)Nv;
    }

    // ── 3. 包围盒线框（12 条边 × 2 顶点）────────────────────────────────
    m_lineCount = 0;
    {
        if (m_cacheDirty) invalidateCache();
        const bool empty = m_poses.isEmpty() && (m_cloud.size() == 0);
        if (!empty) {
            const QVector3D &mn = m_cachedAABBMin;
            const QVector3D &mx = m_cachedAABBMax;
            const QVector3D v000(mn.x(), mn.y(), mn.z());
            const QVector3D v001(mn.x(), mn.y(), mx.z());
            const QVector3D v010(mn.x(), mx.y(), mn.z());
            const QVector3D v011(mn.x(), mx.y(), mx.z());
            const QVector3D v100(mx.x(), mn.y(), mn.z());
            const QVector3D v101(mx.x(), mn.y(), mx.z());
            const QVector3D v110(mx.x(), mx.y(), mn.z());
            const QVector3D v111(mx.x(), mx.y(), mx.z());

            constexpr float r = 0.72f, g = 0.72f, b = 0.76f;
            auto addEdge = [&](QVector<float> &d, const QVector3D &a, const QVector3D &b2) {
                d << a.x() << a.y() << a.z() << r << g << b;
                d << b2.x() << b2.y() << b2.z() << r << g << b;
            };
            QVector<float> data;
            data.reserve(24 * 6);
            addEdge(data, v000, v001); addEdge(data, v000, v010); addEdge(data, v000, v100);
            addEdge(data, v111, v110); addEdge(data, v111, v101); addEdge(data, v111, v011);
            addEdge(data, v001, v011); addEdge(data, v001, v101);
            addEdge(data, v010, v011); addEdge(data, v010, v110);
            addEdge(data, v100, v101); addEdge(data, v100, v110);
            setupColorVao(m_lineVao, m_lineVbo, data);
            m_lineCount = 24; // 12 条边 × 2 顶点
        }
    }

    m_gpuDirty = false;
}



void CameraSceneWidget::paintGL()
{
    if (!m_gl || !m_colorProgram || !m_meshProgram) {
        QPainter p(this);
        p.fillRect(rect(), Qt::white);
        return;
    }

    // 按需上传 GPU 数据
    if (m_gpuDirty) uploadGpuData();

    m_gl->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    m_gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ── 构建 MVP 矩阵（与旧代码逻辑保持一致）───────────────────────────────
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = qMax(radius * 0.001f, radius * (3.2f / qMax(0.1f, m_zoomScale)));
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);

    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));

    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(m_viewRot);
    model.translate(-center);

    const float nearPlane = qMax(1e-4f, distance * 0.001f);
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);
    QMatrix4x4 proj;
    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    proj.perspective(45.0f, aspect, nearPlane, farPlane);

    QMatrix4x4 shift;
    shift.setToIdentity();
    shift.translate(float(2.0 * m_sceneOffsetPx.x() / qMax(1, width())),
                    float(-2.0 * m_sceneOffsetPx.y() / qMax(1, height())),
                    0.0f);
    const QMatrix4x4 mv  = view * model;
    const QMatrix4x4 mvp = shift * proj * mv;

    auto drawOpaquePointSet = [this, &mvp](QOpenGLVertexArrayObject &vao,
                                           int pointCount,
                                           float pointSize)
    {
        if (pointCount <= 0) {
            return;
        }

        const float depthOnlyPointSize = qMax(pointSize + 1.25f, pointSize * 1.35f);

        m_gl->glEnable(GL_DEPTH_TEST);
        m_gl->glDepthMask(GL_TRUE);
        m_gl->glDepthFunc(GL_LEQUAL);
        m_gl->glDisable(GL_BLEND);

        // 先写入稍大一点的深度轮廓，减少后层点从前层点之间漏出来。
        m_gl->glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        m_colorProgram->bind();
        m_colorProgram->setUniformValue("uMVP", mvp);
        m_colorProgram->setUniformValue("uPointSize", depthOnlyPointSize);
        vao.bind();
        m_gl->glDrawArrays(GL_POINTS, 0, pointCount);
        vao.release();
        m_colorProgram->release();

        m_gl->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        m_colorProgram->bind();
        m_colorProgram->setUniformValue("uMVP", mvp);
        m_colorProgram->setUniformValue("uPointSize", pointSize);
        vao.bind();
        m_gl->glDrawArrays(GL_POINTS, 0, pointCount);
        vao.release();
        m_colorProgram->release();

        m_gl->glEnable(GL_BLEND);
        m_gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    };

    // ── 点云（颜色直通）──────────────────────────────────────────────────
    if (m_pointCount > 0) {
        drawOpaquePointSet(m_pointVao, m_pointCount, 2.3f);
    }

    // ── 三角网格（Phong 双面光照）────────────────────────────────────────
    if (m_meshVertCount > 0) {
        m_gl->glDisable(GL_BLEND);
        m_gl->glEnable(GL_DEPTH_TEST);
        m_gl->glDepthMask(GL_TRUE);
        m_gl->glDepthFunc(GL_LEQUAL);
        m_meshProgram->bind();
        m_meshProgram->setUniformValue("uMVP",       mvp);
        m_meshProgram->setUniformValue("uNormalMat", mv.normalMatrix());
        // 固定方向光：视角坐标系左上前方（与旧 GL_LIGHT0 位置一致）
        m_meshProgram->setUniformValue("uLightDir",  QVector3D(0.5f, 0.8f, 0.6f));
        m_meshVao.bind();
        if (m_meshHasFaces) {
            m_gl->glDrawArrays(GL_TRIANGLES, 0, m_meshVertCount);
        } else {
            m_meshProgram->setUniformValue("uPointSize", 1.8f);
            m_gl->glDrawArrays(GL_POINTS, 0, m_meshVertCount);
        }
        m_meshVao.release();
        m_meshProgram->release();
        m_gl->glEnable(GL_BLEND);
        m_gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // ── 模型顶点（无面片时作为点云）──────────────────────────────────────
    if (m_modelPtCount > 0) {
        drawOpaquePointSet(m_modelPtVao, m_modelPtCount, qMax(m_modelPointSize, 4.0f));
    }

    // ── 包围盒线框 ────────────────────────────────────────────────────────
    if (m_lineCount > 0) {
        m_colorProgram->bind();
        m_colorProgram->setUniformValue("uMVP",       mvp);
        m_colorProgram->setUniformValue("uPointSize", 1.0f);
        m_lineVao.bind();
        m_gl->glLineWidth(1.0f);
        m_gl->glDrawArrays(GL_LINES, 0, m_lineCount);
        m_lineVao.release();
        m_colorProgram->release();
    }

    drawOverlay();
}

void CameraSceneWidget::drawOverlay()
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const float frustumBase = cameraFrustumBase();
    const int labelBudget = maxVisibleCameraLabels();
    const int cameraCount = static_cast<int>(m_poses.size());
    const bool drawAllCameraLabels = cameraCount <= labelBudget;
    const int cameraLabelStride = drawAllCameraLabels
        ? 1
        : qMax(1, static_cast<int>(std::ceil(double(cameraCount) / double(qMax(1, labelBudget)))));
    const QColor frustumColor = cameraCount > 200
        ? QColor(180, 130, 50, 105)
        : QColor(180, 130, 50, 210);
    const qreal frustumLineWidth = cameraCount > 200 ? 0.8 : 1.5;
    auto drawLine3D = [&](const QVector3D &a, const QVector3D &b, const QPen &pen) {
        bool okA = false;
        bool okB = false;
        const QPointF pa = projectToScreen(a, &okA);
        const QPointF pb = projectToScreen(b, &okB);
        if (!okA || !okB) return;
        painter.setPen(pen);
        painter.drawLine(pa, pb);
    };

    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // ── 操控球（Gizmo）：仅当 m_showGizmo 为 true 时绘制 ───────────────
    if (m_showGizmo) {
    QRadialGradient grad(center2d - QPointF(radiusPx * 0.18, radiusPx * 0.18), radiusPx * 1.25);
    grad.setColorAt(0.0, QColor(245, 245, 248, 40));
    grad.setColorAt(1.0, QColor(175, 178, 186, 28));
    painter.setPen(QPen(QColor(210, 210, 216, 44), 1.0));
    painter.setBrush(grad);
    painter.drawEllipse(center2d, radiusPx, radiusPx);

    auto axisPen = [&](HoverAxis axis, const QColor &base) {
        const bool hl = (m_hoverAxis == axis) || (m_dragAxis == axis && m_leftDragging);
        QColor cc = base;
        if (hl) cc = cc.lighter(150);
        return QPen(cc, hl ? 4.0 : 2.0);
    };
    auto drawGreatCircle = [&](HoverAxis axis, const QColor &color) {
        painter.setPen(axisPen(axis, color));
        QPointF prev;
        QPointF first;
        bool hasPrev = false;
        bool prevVisible = false;
        bool firstVisible = false;
        for (int i = 0; i <= 128; ++i) {
            const qreal t = (2.0 * M_PI * i) / 128.0;
            QVector3D pLocal;
            if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
            else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
            else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
            QVector3D pView = applyViewRotation(pLocal);
            const bool currVisible = (pView.z() > 0.0f);
            QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
            if (!hasPrev) {
                first = curr;
                firstVisible = currVisible;
            } else {
                if (prevVisible && currVisible) painter.drawLine(prev, curr);
            }
            prev = curr;
            prevVisible = currVisible;
            hasPrev = true;
        }
        if (hasPrev && firstVisible && prevVisible) painter.drawLine(prev, first);
    };
    drawGreatCircle(HoverAxis::X, QColor(255, 110, 110, 52));
    drawGreatCircle(HoverAxis::Y, QColor(110, 255, 150, 52));
    drawGreatCircle(HoverAxis::Z, QColor(110, 170, 255, 52));
    } // end if (m_showGizmo)

    if (m_poses.isEmpty()) {
        painter.setPen(QColor(120, 120, 120));
        painter.drawText(rect(), Qt::AlignCenter, tr("暂无相机参数，显示默认模型球"));
    }

    painter.setBrush(QColor(220, 100, 40));
    painter.setPen(Qt::NoPen);
    for (qsizetype poseIndex = 0; poseIndex < m_poses.size(); ++poseIndex) {
        const CameraPose &pose = m_poses.at(poseIndex);
        bool ok = false;
        const QPointF pc = projectToScreen(pose.center, &ok);
        if (!ok) continue;
        painter.drawEllipse(pc, 4.5, 4.5);

        const QVector3D right(pose.rotation(0, 0), pose.rotation(1, 0), pose.rotation(2, 0));
        const QVector3D up(pose.rotation(0, 1), pose.rotation(1, 1), pose.rotation(2, 1));
        const QVector3D forward(pose.rotation(0, 2), pose.rotation(1, 2), pose.rotation(2, 2));
        const float base = frustumBase;
        const QVector3D fc = pose.center + forward * (base * 2.2f);
        const QVector3D p1 = fc + right * base + up * base;
        const QVector3D p2 = fc - right * base + up * base;
        const QVector3D p3 = fc - right * base - up * base;
        const QVector3D p4 = fc + right * base - up * base;
        const QPen frustumPen(frustumColor, frustumLineWidth);
        drawLine3D(pose.center, p1, frustumPen);
        drawLine3D(pose.center, p2, frustumPen);
        drawLine3D(pose.center, p3, frustumPen);
        drawLine3D(pose.center, p4, frustumPen);
        drawLine3D(p1, p2, frustumPen);
        drawLine3D(p2, p3, frustumPen);
        drawLine3D(p3, p4, frustumPen);
        drawLine3D(p4, p1, frustumPen);
        const bool drawCameraLabel = drawAllCameraLabels
            || (poseIndex == 0)
            || (poseIndex == m_poses.size() - 1)
            || (static_cast<int>(poseIndex) % cameraLabelStride == 0);
        if (drawCameraLabel)
        {
            painter.setPen(QColor(60, 60, 60));
            painter.drawText(pc + QPointF(7.0, -7.0), QFileInfo(pose.name).fileName());
        }
    }

    bool okO = false;
    const QPointF o2d = projectToScreen(QVector3D(0, 0, 0), &okO);
    if (okO) {
        painter.setPen(QPen(QColor(80, 80, 80), 1.5));
        painter.drawLine(o2d + QPointF(-5, 0), o2d + QPointF(5, 0));
        painter.drawLine(o2d + QPointF(0, -5), o2d + QPointF(0, 5));
        painter.drawText(o2d + QPointF(6, -6), QStringLiteral("XYZ(0,0,0)"));
    }

    const QPoint origin(width() - 64, height() - 64);
    const QVector3D ex = applyViewRotation(QVector3D(1, 0, 0)).normalized();
    const QVector3D ey = applyViewRotation(QVector3D(0, 1, 0)).normalized();
    const QVector3D ez = applyViewRotation(QVector3D(0, 0, 1)).normalized();
    auto drawMiniAxis = [&](const QVector3D &dir, const QColor &color, const QString &label) {
        const QPoint end(origin.x() + int(dir.x() * 28.0f), origin.y() - int(dir.y() * 28.0f));
        painter.setPen(QPen(color, 2));
        painter.drawLine(origin, end);
        painter.setPen(color);
        painter.drawText(end + QPoint(4, -2), label);
    };
    painter.setPen(QPen(QColor(80, 80, 80), 1));
    painter.setBrush(QColor(80, 80, 80));
    painter.drawEllipse(QPointF(origin), 2.5, 2.5);
    drawMiniAxis(ex, QColor(210, 50, 50), QStringLiteral("X"));
    drawMiniAxis(ey, QColor(30, 160, 60), QStringLiteral("Y"));
    drawMiniAxis(ez, QColor(40, 100, 220), QStringLiteral("Z"));
    painter.setPen(QColor(100, 100, 110));
    const QVector3D euler = eulerAnglesDeg();
    painter.drawText(origin + QPoint(-84, 26),
                     QStringLiteral("Yaw %1°  Pitch %2°  Roll %3°")
                         .arg(QString::number(euler.y(), 'f', 1))
                         .arg(QString::number(euler.x(), 'f', 1))
                         .arg(QString::number(euler.z(), 'f', 1)));

    if (m_manualPruneMode)
    {
        painter.setPen(QPen(QColor(255, 90, 90, 220), 1.5, Qt::DashLine));
        painter.setBrush(QColor(255, 90, 90, 40));
        if (!m_manualSelectRect.isNull())
        {
            painter.drawRect(m_manualSelectRect.normalized());
        }

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 230, 90, 170));
        const int highlightCap = 12000;
        const int drawCount = std::min<int>(static_cast<int>(m_manualPreviewIndices.size()), highlightCap);
        for (int i = 0; i < drawCount; ++i)
        {
            const std::size_t pointIndex = m_manualPreviewIndices[static_cast<std::size_t>(i)];
            if (pointIndex >= m_cloud.size())
            {
                continue;
            }
            bool ok = false;
            const QPointF screenPoint = projectToScreen(QVector3D(
                m_cloud.points()(static_cast<plamatrix::Index>(pointIndex), 0),
                m_cloud.points()(static_cast<plamatrix::Index>(pointIndex), 1),
                m_cloud.points()(static_cast<plamatrix::Index>(pointIndex), 2)), &ok);
            if (ok)
            {
                painter.drawEllipse(screenPoint, 1.5, 1.5);
            }
        }

        painter.setPen(QColor(235, 80, 80));
        painter.drawText(QPointF(14.0, 24.0),
                         tr("手动剔除模式：右键框选高亮，前进侧键删除，Ctrl+Z 撤销（已选 %1）")
                             .arg(static_cast<int>(m_manualPreviewIndices.size())));
    }
}

bool CameraSceneWidget::setManualPruneModeEnabled(bool enabled, QString *errorMessage)
{
    if (enabled)
    {
        if ((m_cloud.size() == 0))
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前未加载点云数据。");
            }
            return false;
        }
        if (m_cloud.hasFaces())
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前为网格模型，手动剔除仅支持点云。");
            }
            return false;
        }
    }

    m_manualPruneMode = enabled;
    m_manualSelecting = false;
    m_manualSelectRect = QRect();
    m_manualPreviewIndices.clear();
    if (!enabled)
    {
        m_manualUndoStack.clear();
    }
    emit manualPruneModeChanged(m_manualPruneMode);
    updateCursor();
    update();
    return true;
}

void CameraSceneWidget::pushManualUndoSnapshot(RenderCloud snapshot)
{
    if (!m_manualPruneMode)
    {
        return;
    }
    m_manualUndoStack.push_back(std::move(snapshot));
    if (static_cast<int>(m_manualUndoStack.size()) > m_manualUndoLimit)
    {
        m_manualUndoStack.erase(m_manualUndoStack.begin());
    }
}

bool CameraSceneWidget::undoLastManualPrune(QString *errorMessage)
{
    if (!m_manualPruneMode)
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前未处于手动剔除模式。");
        }
        return false;
    }
    if (m_manualUndoStack.empty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("没有可撤销的删除操作。");
        }
        return false;
    }

    m_cloud = std::move(m_manualUndoStack.back());
    m_manualUndoStack.pop_back();
    m_cacheDirty = true;
    m_gpuDirty = true;
    update();

    QString saveError;
    if (!saveCurrentPointCloudToSource(&saveError))
    {
        if (errorMessage)
        {
            *errorMessage = saveError;
        }
        emit manualPruneSaveFailed(saveError);
        return false;
    }

    emit manualPruneUndone(static_cast<int>(m_cloud.size()));
    emit manualPruneSaved(m_currentCloudPath, static_cast<int>(m_cloud.size()));
    return true;
}

int CameraSceneWidget::removePointsInScreenRect(const QRect &screenRect)
{
    const QRect rect = screenRect.normalized();
    if (rect.width() < 3 || rect.height() < 3 || (m_cloud.size() == 0))
    {
        return 0;
    }

    std::vector<std::size_t> selectedIndices;
    collectPointIndicesInScreenRect(rect, &selectedIndices);
    const std::size_t pointCount = m_cloud.size();
    std::vector<bool> removeMask(pointCount, false);
    for (const std::size_t index : selectedIndices)
    {
        if (index < pointCount)
        {
            removeMask[index] = true;
        }
    }
    const int removedCount = static_cast<int>(selectedIndices.size());

    if (removedCount <= 0)
    {
        return 0;
    }

    // Build kept indices list
    const std::size_t keepCount = pointCount - static_cast<std::size_t>(removedCount);
    std::vector<plamatrix::Index> kept;
    kept.reserve(keepCount);
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        if (!removeMask[index])
            kept.push_back(static_cast<plamatrix::Index>(index));
    }

    // Build new point cloud from kept indices
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newPts(keepCount, 3);
    for (std::size_t i = 0; i < keepCount; ++i)
    {
        plamatrix::Index src = kept[i];
        for (int d = 0; d < 3; ++d)
            newPts(static_cast<plamatrix::Index>(i), d) = m_cloud.points()(src, d);
    }

    RenderCloud filtered(std::move(newPts));
    filtered.setMaterialLibraryFile(m_cloud.materialLibraryFile());
    filtered.setTextureImageFile(m_cloud.textureImageFile());

    // Copy colors if present
    if (m_cloud.hasColors())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> newCol(keepCount, 3);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 3; ++d)
                newCol(static_cast<plamatrix::Index>(i), d) = m_cloud.colors()->getValue(src, d);
        }
        filtered.setColors(std::move(newCol));
    }

    // Copy normals if present
    if (m_cloud.hasNormals())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newNrm(keepCount, 3);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 3; ++d)
                newNrm(static_cast<plamatrix::Index>(i), d) = m_cloud.normals()->getValue(src, d);
        }
        filtered.setNormals(std::move(newNrm));
    }

    // Copy texture coords if present
    if (m_cloud.hasTextureCoords())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> newTex(keepCount, 2);
        for (std::size_t i = 0; i < keepCount; ++i)
        {
            plamatrix::Index src = kept[i];
            for (int d = 0; d < 2; ++d)
                newTex(static_cast<plamatrix::Index>(i), d) = m_cloud.textureCoords()->getValue(src, d);
        }
        filtered.setTextureCoords(std::move(newTex));
    }

    m_cloud = std::move(filtered);
    m_cacheDirty = true;
    m_gpuDirty = true;
    update();
    return removedCount;
}

void CameraSceneWidget::collectPointIndicesInScreenRect(const QRect &screenRect,
                                                        std::vector<std::size_t> *indices) const
{
    if (!indices)
    {
        return;
    }
    indices->clear();

    const QRect rect = screenRect.normalized();
    if (rect.width() < 3 || rect.height() < 3 || (m_cloud.size() == 0))
    {
        return;
    }

    const std::size_t pointCount = m_cloud.size();
    indices->reserve(pointCount / 8);
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        bool projected = false;
        const QPointF screenPoint = projectToScreen(QVector3D(
            m_cloud.points()(static_cast<plamatrix::Index>(index), 0),
            m_cloud.points()(static_cast<plamatrix::Index>(index), 1),
            m_cloud.points()(static_cast<plamatrix::Index>(index), 2)), &projected);
        if (projected && rect.contains(screenPoint.toPoint()))
        {
            indices->push_back(index);
        }
    }
}

bool CameraSceneWidget::saveCurrentPointCloudToSource(QString *errorMessage)
{
    if (m_currentCloudPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前点云来源未知，无法覆盖保存。");
        }
        return false;
    }

    // Determine file format from extension
    const std::string stdPath = m_currentCloudPath.toStdString();
    const bool isPly = (stdPath.size() >= 4 &&
        (stdPath.substr(stdPath.size() - 4) == ".ply" || stdPath.substr(stdPath.size() - 4) == ".PLY"));

    try
    {
        if (isPly)
        {
            plapoint::io::writePly<float>(stdPath, m_cloud);
        }
        else
        {
            plapoint::io::writeXyz<float>(stdPath, m_cloud);
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMessage)
        {
            *errorMessage = QString::fromStdString(e.what());
        }
        return false;
    }
}

void CameraSceneWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();

    if (m_manualPruneMode && event->button() == Qt::RightButton)
    {
        m_manualSelecting = true;
        m_manualSelectStart = event->pos();
        m_manualSelectRect = QRect(m_manualSelectStart, m_manualSelectStart);
        updateCursor();
        event->accept();
        update();
        return;
    }

    if (m_manualPruneMode && event->button() == Qt::ForwardButton)
    {
        const QRect selectionRect = m_manualSelectRect.normalized();
        RenderCloud snapshot = cloneRenderCloud(m_cloud);
        const int removedCount = removePointsInScreenRect(selectionRect);
        if (removedCount > 0)
        {
            pushManualUndoSnapshot(std::move(snapshot));
            emit manualPruneApplied(removedCount, static_cast<int>(m_cloud.size()));
            QString saveError;
            if (saveCurrentPointCloudToSource(&saveError))
            {
                emit manualPruneSaved(m_currentCloudPath, static_cast<int>(m_cloud.size()));
            }
            else
            {
                emit manualPruneSaveFailed(saveError);
            }
            collectPointIndicesInScreenRect(m_manualSelectRect, &m_manualPreviewIndices);
        }
        event->accept();
        update();
        return;
    }

    m_lastMousePos = event->pos();
    if (event->button() == Qt::LeftButton) {
        m_leftDragging = true;
        m_dragAxis = m_hoverAxis;
        // 无论单轴还是自由旋转，均记录按下时的旋转状态
        m_viewRotAtPress = m_viewRot;
        if (m_dragAxis != HoverAxis::None) {
            m_dragAxisDir = pickAxisTangent(event->pos(), m_dragAxis);
            // 注意：Y 轴环的屏幕切线方向在参数化时与鼠标拖拽方向有符号差，
            // 在多数情况下需要翻转切线方向以使鼠标向右/上时视图按直觉旋转。
            // 仅对 Y 轴做翻转修正，避免对 X/Z 轴产生副作用。
            if (m_dragAxis == HoverAxis::Y) m_dragAxisDir = -m_dragAxisDir;
        } else {
            // Arcball 自由旋转：记录按下那一刻球面坐标
            m_arcballPressVector = arcballVector(event->pos());
        }
        updateCursor();
        event->accept();
        return;
    }
    if (event->button() == Qt::MiddleButton) {
        m_middleDragging = true;
        updateCursor();
        event->accept();
        return;
    }
    QOpenGLWidget::mousePressEvent(event);
}

void CameraSceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_manualPruneMode && m_manualSelecting && (event->buttons() & Qt::RightButton))
    {
        m_manualSelectRect = QRect(m_manualSelectStart, event->pos()).normalized();
        collectPointIndicesInScreenRect(m_manualSelectRect, &m_manualPreviewIndices);
        event->accept();
        update();
        return;
    }

    const QPoint delta = event->pos() - m_lastMousePos;

    if (m_leftDragging && (event->buttons() & Qt::LeftButton)) {
        if (m_dragAxis == HoverAxis::None) {
            // ── Arcball 自由旋转 ──────────────────────────────────────────────
            // 将当前鼠标投影到球面，计算从按下点到当前点的旋转，
            // 再䈛到按下时保存的初始视图旋转上——
            // 这样球面上最始点击处就会一直跟随鼠标移动。
            const QVector3D v2 = arcballVector(event->pos());
            const QVector3D axis = QVector3D::crossProduct(m_arcballPressVector, v2);
            if (axis.lengthSquared() > 1e-10f) {
                const float dot = qBound(-1.0f,
                    QVector3D::dotProduct(m_arcballPressVector, v2), 1.0f);
                const float angleDeg = qRadiansToDegrees(std::acos(dot));
                const QQuaternion delta_q =
                    QQuaternion::fromAxisAndAngle(axis.normalized(), angleDeg);
                // 应用到按下时的初始旋转（非增量式，避免浮点漂移）
                m_viewRot = (delta_q * m_viewRotAtPress).normalized();
            }
        } else {
            // ── 单轴环旋转 ─────────────────────────────────────────────────────
            // 目标：环面的法向方向（即 X/Y/Z 轴在当前世界空间中的指向）固定不动，
            //       环只在自身平面内"自旋"，看起来像环面始终保持水平/垂直。
            // 实现：将本地轴转换到世界空间 axisView，绕 axisView 前乘旋转。
            //       前乘（世界空间旋转）效果：环法向不变，环平面姿态不变，
            //       场景内容（相机等）绕该轴旋转。
            const QVector2D d(float(delta.x()), float(delta.y()));
            const float scalar = QVector2D::dotProduct(d, m_dragAxisDir);
            const float ang = scalar * 0.35f;
            // 取该环的本地法向轴，转换为当前视图下的世界方向
            QVector3D localAxis;
            if (m_dragAxis == HoverAxis::X)      localAxis = QVector3D(1.0f, 0.0f, 0.0f);
            else if (m_dragAxis == HoverAxis::Y) localAxis = QVector3D(0.0f, 1.0f, 0.0f);
            else                                  localAxis = QVector3D(0.0f, 0.0f, 1.0f);
            const QVector3D axisWorld = applyViewRotation(localAxis).normalized();
            // 绕世界空间轴前乘：new_rot = qa_world * old_rot
            // 这样环的法向量方向(axisWorld)在此次旋转后保持恒定
            const QQuaternion qa = QQuaternion::fromAxisAndAngle(axisWorld, ang);
            m_viewRot = (qa * m_viewRot).normalized();
        }
        update();
    } else if (m_middleDragging && (event->buttons() & Qt::MiddleButton)) {
        // 中键平移：1:1 映射鼠标像素，无论缩放倍率如何，拖拽同量始终移动同距离
        m_sceneOffsetPx += QPointF(delta.x(), delta.y());
        clampSceneOffset();
        update();
    } else {
        const HoverAxis newHover = pickHoverAxis(event->pos());
        if (newHover != m_hoverAxis) {
            m_hoverAxis = newHover;
            updateCursor();
            update();
        }
    }

    m_lastMousePos = event->pos();
    QOpenGLWidget::mouseMoveEvent(event);
}

void CameraSceneWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (m_manualPruneMode && event->button() == Qt::RightButton)
    {
        m_manualSelecting = false;
        m_manualSelectRect = m_manualSelectRect.normalized();
        collectPointIndicesInScreenRect(m_manualSelectRect, &m_manualPreviewIndices);
        updateCursor();
        update();
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        m_leftDragging = false;
        m_dragAxis = HoverAxis::None;
    }
    if (event->button() == Qt::MiddleButton) {
        m_middleDragging = false;
    }
    updateCursor();
    QOpenGLWidget::mouseReleaseEvent(event);
}

void CameraSceneWidget::wheelEvent(QWheelEvent *event)
{
    if (m_leftDragging || m_middleDragging) {
        event->ignore();
        return;
    }

    const QPoint angle = event->angleDelta();
    if (!angle.isNull()) {
        const float factor = (angle.y() > 0) ? 1.10f : 0.90f;
        m_zoomScale = m_zoomScale * factor;   // 无缩放上下限
        clampSceneOffset();
        update();
    }
    event->accept();
}

void CameraSceneWidget::keyPressEvent(QKeyEvent *event)
{
    if (m_manualPruneMode && event->matches(QKeySequence::Undo))
    {
        QString errorMessage;
        undoLastManualPrune(&errorMessage);
        event->accept();
        return;
    }
    QOpenGLWidget::keyPressEvent(event);
}

CameraModel3DDialog::CameraModel3DDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    Ui::CameraModel3DDialog form;
    form.setupUi(this);

    m_scene = form.m_scene;
    m_summaryLabel = form.m_summaryLabel;

    connect(form.reloadButton, &QPushButton::clicked, this, &CameraModel3DDialog::reloadFromProject);
    connect(form.closeButton, &QPushButton::clicked, this, &QDialog::accept);

    reloadFromProject();
}

QVector<CameraSceneWidget::CameraPose> CameraModel3DDialog::readCamerasFromMeta() const
{
    QVector<CameraSceneWidget::CameraPose> poses;
    if (!m_projectManager)
    {
        return poses;
    }

    const QJsonArray images = xjw::gui::project::projectImageEntries(m_projectManager->currentMeta());
    for (const QJsonValue &imageValue : images)
    {
        const QJsonObject imageObject = imageValue.toObject();
        xjw::Camera camera;
        if (!xjw::gui::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();

        CameraSceneWidget::CameraPose pose;
        pose.name = imageObject.value(QStringLiteral("name")).toString();
        if (pose.name.isEmpty())
        {
            pose.name = imageObject.value(QStringLiteral("path")).toString();
        }
        pose.center = QVector3D(
            float(cameraCenter[0]),
            float(cameraCenter[1]),
            float(cameraCenter[2]));

        QMatrix3x3 rot;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rot(row, col) = float(cameraToWorldRotation[row * 3 + col]);
            }
        }
        pose.rotation = rot;
        poses.push_back(pose);
    }

    return poses;
}

void CameraModel3DDialog::reloadFromProject()
{
    const QVector<CameraSceneWidget::CameraPose> poses = readCamerasFromMeta();
    m_scene->setCameraPoses(poses);
    const QString labelHint = poses.size() > 40
        ? tr("，相机名称已抽样显示")
        : QString();
    m_summaryLabel->setText(tr("相机数量: %1%2（左键旋转，滚轮缩放）").arg(poses.size()).arg(labelHint));
}
