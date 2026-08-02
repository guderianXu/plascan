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

class CameraImageSelectionState
{
public:
    void setActiveIndex(int index) { _activeIndex = index; }
    int activeIndex() const { return _activeIndex; }
    void setLocked(bool locked) { _locked = locked; }
    bool isLocked() const { return _locked; }
    int resolveAutomaticIndex(int automaticIndex) const
    {
        return _locked && _activeIndex >= 0 ? _activeIndex : automaticIndex;
    }

private:
    int _activeIndex = -1;
    bool _locked = false;
};

QVector<int> farToNearCameraIndices(const QVector<QVector3D> &centers,
                                    const QMatrix4x4 &worldToView);

// 相机卡片的目标屏幕半尺寸只由全局缩放倍率决定。
// 缩远场景时返回更大的像素尺寸，且不受单个相机观察深度影响。
float cameraPlaneScreenHalfExtentPixels(
    float zoomScale,
    float normalHalfExtentPixels = 34.0f,
    float minimumHalfExtentPixels = 18.0f,
    float maximumHalfExtentPixels = 72.0f);

// 先按单个相机的观察深度抵消透视缩放，再只用全局缩放倍率决定
// 目标屏幕尺寸。这样旋转视图不会产生“近大远小”，而缩远整个
// 场景时，相机卡片会按设计变大。
float cameraPlaneHalfExtentForScreenSize(
    const QVector3D &center,
    const QMatrix4x4 &worldToView,
    int viewportHeight,
    float zoomScale,
    float verticalFieldOfViewDegrees = 45.0f,
    float normalHalfExtentPixels = 34.0f,
    float minimumHalfExtentPixels = 18.0f,
    float maximumHalfExtentPixels = 72.0f);

// 生成固定屏幕像素长度的方向引线。探针只提供方向，探针距离
// 和单个相机的透视深度都不会改变引线长度。
QLineF cameraPlaneLeaderLine(const QPointF &center,
                             const QPointF &directionProbe,
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
