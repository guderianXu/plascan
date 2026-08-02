#pragma once

#include <QMatrix3x3>
#include <QMatrix4x4>
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

// 将目标屏幕尺寸换算成世界尺寸，并相对场景中心增强深度差异，
// 使远处相机卡片在屏幕上大于近处卡片。
float cameraPlaneHalfExtentForViewDepth(
    const QVector3D &center,
    const QVector3D &referenceCenter,
    const QMatrix4x4 &worldToView,
    int viewportHeight,
    float verticalFieldOfViewDegrees = 45.0f,
    float targetHalfExtentPixels = 28.0f,
    float depthEmphasisExponent = 2.0f);

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
