#include "CameraSceneViewMath.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QVector4D>

namespace xjw::gui::camera_scene
{
namespace
{

constexpr float ScoreEpsilon = 1e-6f;

QVector3D normalizedOrZero(const QVector3D &value)
{
    return value.lengthSquared() > ScoreEpsilon * ScoreEpsilon
        ? value.normalized()
        : QVector3D();
}

} // namespace

QVector<int> farToNearCameraIndices(const QVector<QVector3D> &centers,
                                    const QMatrix4x4 &worldToView)
{
    QVector<int> indices;
    indices.reserve(centers.size());
    for (qsizetype index = 0; index < centers.size(); ++index)
    {
        indices.push_back(static_cast<int>(index));
    }

    std::stable_sort(indices.begin(), indices.end(), [&](int lhs, int rhs)
    {
        const float lhs_depth = -(worldToView * QVector4D(centers.at(lhs), 1.0f)).z();
        const float rhs_depth = -(worldToView * QVector4D(centers.at(rhs), 1.0f)).z();
        return lhs_depth > rhs_depth;
    });
    return indices;
}

int selectCameraForView(const QVector<CameraViewCandidate> &candidates,
                        const QVector3D &worldViewDirection,
                        const QVector3D &sceneCenter)
{
    const QVector3D view_direction = normalizedOrZero(worldViewDirection);
    if (view_direction.isNull())
    {
        return -1;
    }

    int best_index = -1;
    float best_direction_score = -std::numeric_limits<float>::infinity();
    float best_center_score = -std::numeric_limits<float>::infinity();
    for (const CameraViewCandidate &candidate : candidates)
    {
        if (!candidate.imageAvailable || candidate.index < 0)
        {
            continue;
        }

        const QVector3D forward = normalizedOrZero(candidate.forward);
        if (forward.isNull())
        {
            continue;
        }

        const float direction_score = QVector3D::dotProduct(forward, view_direction);
        const QVector3D toward_scene = normalizedOrZero(sceneCenter - candidate.center);
        const float center_score = toward_scene.isNull()
            ? 0.0f
            : QVector3D::dotProduct(forward, toward_scene);
        const bool better_direction = direction_score > best_direction_score + ScoreEpsilon;
        const bool same_direction = std::abs(direction_score - best_direction_score) <= ScoreEpsilon;
        const bool better_center = center_score > best_center_score + ScoreEpsilon;
        const bool same_center = std::abs(center_score - best_center_score) <= ScoreEpsilon;
        if (better_direction
            || (same_direction && better_center)
            || (same_direction && same_center && (best_index < 0 || candidate.index < best_index)))
        {
            best_index = candidate.index;
            best_direction_score = direction_score;
            best_center_score = center_score;
        }
    }
    return best_index;
}

QVector3D cameraForwardDirection(const QMatrix3x3 &cameraToWorld,
                                 bool depthAxisFlipped)
{
    QVector3D forward(cameraToWorld(0, 2),
                      cameraToWorld(1, 2),
                      cameraToWorld(2, 2));
    if (depthAxisFlipped)
    {
        forward = -forward;
    }
    return normalizedOrZero(forward);
}

QVector3D currentWorldViewDirection(const QQuaternion &viewRotation)
{
    return normalizedOrZero(viewRotation.conjugated().rotatedVector(QVector3D(0.0f, 0.0f, -1.0f)));
}

QMatrix4x4 calibratedProjection(float focalX,
                                float focalY,
                                float principalX,
                                float principalY,
                                int imageWidth,
                                int imageHeight,
                                float nearPlane,
                                float farPlane,
                                int uAxisSign,
                                int vAxisSign)
{
    const float near_plane = std::max(1e-4f, nearPlane);
    const float far_plane = std::max(near_plane + 1.0f, farPlane);
    QMatrix4x4 projection;
    if (!(std::isfinite(focalX) && std::isfinite(focalY)
          && std::isfinite(principalX) && std::isfinite(principalY))
        || focalX <= 0.0f || focalY <= 0.0f || imageWidth <= 0 || imageHeight <= 0)
    {
        const float aspect = imageWidth > 0 && imageHeight > 0
            ? static_cast<float>(imageWidth) / static_cast<float>(imageHeight)
            : 1.0f;
        projection.perspective(45.0f, aspect, near_plane, far_plane);
        return projection;
    }

    const float left = -principalX * near_plane / focalX;
    const float right = (static_cast<float>(imageWidth) - principalX) * near_plane / focalX;
    const float bottom = -(static_cast<float>(imageHeight) - principalY) * near_plane / focalY;
    const float top = principalY * near_plane / focalY;
    projection.frustum(left, right, bottom, top, near_plane, far_plane);
    projection(0, 0) *= uAxisSign < 0 ? -1.0f : 1.0f;
    projection(1, 1) *= vAxisSign < 0 ? 1.0f : -1.0f;
    return projection;
}

QVector<QVector3D> cameraImagePlaneCorners(const QVector3D &center,
                                           const QVector3D &right,
                                           const QVector3D &up,
                                           float halfWidth,
                                           float halfHeight)
{
    const QVector3D horizontal = normalizedOrZero(right) * std::max(0.0f, halfWidth);
    const QVector3D vertical = normalizedOrZero(up) * std::max(0.0f, halfHeight);
    return {
        center + horizontal + vertical,
        center - horizontal + vertical,
        center - horizontal - vertical,
        center + horizontal - vertical,
    };
}

} // namespace xjw::gui::camera_scene
