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

int selectCameraForView(const QVector<CameraViewCandidate> &candidates,
                        const QVector3D &worldViewDirection,
                        const QVector3D &sceneCenter);

QVector3D cameraForwardDirection(const QMatrix3x3 &cameraToWorld,
                                 bool depthAxisFlipped);

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

} // namespace xjw::gui::camera_scene
