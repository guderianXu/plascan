#pragma once

#include <QLineF>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QPointF>
#include <QQuaternion>
#include <QVector>
#include <QVector3D>

namespace xjw::gui::camera_scene
{

inline constexpr double MinimumSceneZoomScale = 1.0e-3;
inline constexpr double MaximumSceneZoomScale = 1.0e3;

struct CameraViewCandidate
{
    int index = -1;
    QVector3D forward;
    QVector3D center;
    bool imageAvailable = false;
};

struct CameraImagePlaneAxes
{
    QVector3D right;
    QVector3D up;
};

struct CameraLocalAxes
{
    QVector3D x;
    QVector3D y;
    QVector3D z;
};

struct PointCloudPrincipalAxes
{
    QVector3D center;
    QVector3D first;
    QVector3D second;
    QVector3D third;
    bool valid = false;
};

QVector<int> farToNearCameraIndices(const QVector<QVector3D> &centers,
                                    const QMatrix4x4 &worldToView);

// 返回真实视口宽高比。竖向窗口必须保留小于 1 的 aspect，不能按横向
// 窗口处理，否则模型和相机平面会被横向压缩。
float sceneViewportAspectRatio(int width, int height);

// 应用一次缩放并限制到可计算的范围，避免极端滚轮输入把观察距离或
// near plane 推到 0、无穷或 NaN。
double boundedSceneZoomScale(
    double currentScale,
    double factor,
    double minimumScale = MinimumSceneZoomScale,
    double maximumScale = MaximumSceneZoomScale);

// 相机卡片的目标屏幕半尺寸只由全局缩放倍率决定。
// 缩远场景时持续增大，缩近时持续减小，但使用缓增长曲线避免
// 相机卡片比场景本身更快占满视口；不设置视觉范围夹紧。
double cameraPlaneScreenHalfExtentPixels(
    double zoomScale,
    double normalHalfExtentPixels = 34.0);

// 先按单个相机的观察深度抵消透视缩放，再只用全局缩放倍率决定
// 目标屏幕尺寸。这样旋转视图不会产生“近大远小”，而缩远整个
// 场景时，相机卡片会按设计变大。
float cameraPlaneHalfExtentForScreenSize(
    const QVector3D &center,
    const QMatrix4x4 &worldToView,
    int viewportHeight,
    double zoomScale,
    float verticalFieldOfViewDegrees = 45.0f,
    double normalHalfExtentPixels = 34.0);

// 生成固定屏幕像素长度的方向引线。探针只提供方向，探针距离
// 和单个相机的透视深度都不会改变引线长度。引线会跳过照片平面
// 占据的起始区间，避免从照片中心穿出。
QLineF cameraPlaneLeaderLine(const QPointF &center,
                             const QPointF &directionProbe,
                             qreal startOffsetPixels,
                             qreal lengthPixels);

// 判断一个 NDC 点是否位于投影四边形内并处在该平面之后。
// 四边形顶点顺序必须与相机平面的两个三角形一致。
bool pointIsBehindProjectedQuad(const QVector3D &pointNdc,
                                const QVector<QVector3D> &quadNdc,
                                float depthEpsilon = 1.0e-5f);

int selectCameraForView(const QVector<CameraViewCandidate> &candidates,
                        const QVector3D &worldViewDirection,
                        const QVector3D &sceneCenter,
                        float maximumViewAngleDegrees = 30.0f);

QVector3D cameraForwardDirection(const QMatrix3x3 &cameraToWorld,
                                 bool depthAxisFlipped);

// 返回相机坐标系的 X/Y/Z 轴在世界坐标系中的方向。
// Z 轴遵循深度轴翻转标志，确保其始终指向实际拍摄方向。
CameraLocalAxes cameraLocalAxes(const QMatrix3x3 &cameraToWorld,
                                bool depthAxisFlipped);

// 将相机像素坐标的 +u/+v 方向转换为三维照片平面的右/上轴。
// Qt 纹理的 v=0 位于图像上边，因此 up 必须与相机像素 +v 方向相反。
CameraImagePlaneAxes cameraImagePlaneAxes(const QMatrix3x3 &cameraToWorld,
                                          int uAxisSign,
                                          int vAxisSign);

QVector3D currentWorldViewDirection(const QQuaternion &viewRotation);

QMatrix4x4 calibratedProjection(float focalX,
                                float focalY,
                                float principalX,
                                float principalY,
                                int imageWidth,
                                int imageHeight,
                                float nearPlane,
                                float farPlane,
                                int uAxisSign,
                                int vAxisSign);

QVector<QVector3D> cameraImagePlaneCorners(const QVector3D &center,
                                           const QVector3D &right,
                                           const QVector3D &up,
                                           float halfWidth,
                                           float halfHeight);

QVector<QVector3D> axisAlignedBoundingBoxLineVertices(
    const QVector3D &minimum,
    const QVector3D &maximum);

// 先由协方差估计点云支撑平面，再在该平面投影的凸包上寻找最小
// 面积矩形方向。只有整体尺度有效时才返回 valid。
PointCloudPrincipalAxes pointCloudPrincipalAxes(
    const QVector<QVector3D> &points);

// minimum/maximum 是相对于 axes.center、沿三个主轴的局部坐标范围。
QVector<QVector3D> orientedBoundingBoxLineVertices(
    const PointCloudPrincipalAxes &axes,
    const QVector3D &minimum,
    const QVector3D &maximum);

QVector<QVector3D> calibratedImagePlaneCorners(const QVector3D &cameraCenter,
                                               const QVector3D &forward,
                                               const QVector3D &right,
                                               const QVector3D &up,
                                               const QVector3D &sceneCenter,
                                               float focalX,
                                               float focalY,
                                               float principalX,
                                               float principalY,
                                               int imageWidth,
                                               int imageHeight);

} // namespace xjw::gui::camera_scene
