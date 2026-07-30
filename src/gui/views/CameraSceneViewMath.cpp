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
                        const QVector3D &sceneCenter,
                        float maximumViewAngleDegrees)
{
    const QVector3D view_direction = normalizedOrZero(worldViewDirection);
    if (view_direction.isNull())
    {
        return -1;
    }

    int best_index = -1;
    bool has_best_position = false;
    float best_position_score = -std::numeric_limits<float>::infinity();
    float best_direction_score = -std::numeric_limits<float>::infinity();
    float best_optical_axis_score = -std::numeric_limits<float>::infinity();
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

        // 三维视图的观察方向首先由相机光心在模型周围的方位决定。
        // 不能仅比较光轴：局部外参方向存在噪声或翻转时，会把圆轨道对侧
        // 的照片误判为当前观察方向的照片。
        const QVector3D toward_scene = normalizedOrZero(sceneCenter - candidate.center);
        const bool has_position = !toward_scene.isNull();
        const float position_score = has_position
            ? QVector3D::dotProduct(toward_scene, view_direction)
            : 0.0f;
        const float direction_score = QVector3D::dotProduct(forward, view_direction);
        const float optical_axis_score = toward_scene.isNull()
            ? 0.0f
            : QVector3D::dotProduct(forward, toward_scene);
        const bool better_position = has_position
            && (!has_best_position || position_score > best_position_score + ScoreEpsilon);
        const bool same_position = has_position && has_best_position
            && std::abs(position_score - best_position_score) <= ScoreEpsilon;
        const bool both_without_position = !has_position && !has_best_position;
        const bool better_direction = direction_score > best_direction_score + ScoreEpsilon;
        const bool same_direction = std::abs(direction_score - best_direction_score) <= ScoreEpsilon;
        const bool better_optical_axis = optical_axis_score > best_optical_axis_score + ScoreEpsilon;
        const bool same_optical_axis = std::abs(optical_axis_score - best_optical_axis_score) <= ScoreEpsilon;
        if (better_position
            || ((same_position || both_without_position) && better_direction)
            || ((same_position || both_without_position) && same_direction && better_optical_axis)
            || ((same_position || both_without_position) && same_direction && same_optical_axis
                && (best_index < 0 || candidate.index < best_index)))
        {
            best_index = candidate.index;
            has_best_position = has_position;
            best_position_score = position_score;
            best_direction_score = direction_score;
            best_optical_axis_score = optical_axis_score;
        }
    }
    if (best_index < 0)
    {
        return -1;
    }

    constexpr float pi = 3.14159265358979323846f;
    const float maximum_angle = std::clamp(maximumViewAngleDegrees, 0.0f, 180.0f);
    const float minimum_match_score = std::cos(maximum_angle * pi / 180.0f);
    const float best_match_score = has_best_position
        ? best_position_score
        : best_direction_score;
    return best_match_score + ScoreEpsilon >= minimum_match_score
        ? best_index
        : -1;
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

CameraLocalAxes cameraLocalAxes(const QMatrix3x3 &cameraToWorld,
                                bool depthAxisFlipped)
{
    CameraLocalAxes axes;
    axes.x = normalizedOrZero(QVector3D(cameraToWorld(0, 0),
                                        cameraToWorld(1, 0),
                                        cameraToWorld(2, 0)));
    axes.y = normalizedOrZero(QVector3D(cameraToWorld(0, 1),
                                        cameraToWorld(1, 1),
                                        cameraToWorld(2, 1)));
    axes.z = cameraForwardDirection(cameraToWorld, depthAxisFlipped);
    return axes;
}

CameraImagePlaneAxes cameraImagePlaneAxes(const QMatrix3x3 &cameraToWorld,
                                          int uAxisSign,
                                          int vAxisSign)
{
    const QVector3D camera_x(cameraToWorld(0, 0),
                             cameraToWorld(1, 0),
                             cameraToWorld(2, 0));
    const QVector3D camera_y(cameraToWorld(0, 1),
                             cameraToWorld(1, 1),
                             cameraToWorld(2, 1));
    const float u_sign = uAxisSign < 0 ? -1.0f : 1.0f;
    const float v_sign = vAxisSign < 0 ? -1.0f : 1.0f;

    CameraImagePlaneAxes axes;
    axes.right = normalizedOrZero(camera_x * u_sign);
    axes.up = normalizedOrZero(camera_y * -v_sign);
    return axes;
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
                                               int imageHeight)
{
    const QVector3D forward_axis = normalizedOrZero(forward);
    const QVector3D right_axis = normalizedOrZero(right);
    const QVector3D up_axis = normalizedOrZero(up);
    if (forward_axis.isNull() || right_axis.isNull() || up_axis.isNull())
    {
        return {};
    }

    const float model_depth = QVector3D::dotProduct(sceneCenter - cameraCenter, forward_axis);
    if (!std::isfinite(model_depth) || model_depth <= ScoreEpsilon)
    {
        return {};
    }

    const QVector3D optical_axis_center = cameraCenter + forward_axis * model_depth;
    const bool valid_intrinsics = std::isfinite(focalX)
        && std::isfinite(focalY)
        && std::isfinite(principalX)
        && std::isfinite(principalY)
        && focalX > ScoreEpsilon
        && focalY > ScoreEpsilon
        && imageWidth > 0
        && imageHeight > 0;
    if (!valid_intrinsics)
    {
        constexpr float half_fov_radians = 22.5f * 3.14159265358979323846f / 180.0f;
        const float aspect = imageWidth > 0 && imageHeight > 0
            ? static_cast<float>(imageWidth) / static_cast<float>(imageHeight)
            : 4.0f / 3.0f;
        const float half_height = model_depth * std::tan(half_fov_radians);
        const float half_width = half_height * aspect;
        return cameraImagePlaneCorners(
            optical_axis_center, right_axis, up_axis, half_width, half_height);
    }

    const float left_extent = principalX * model_depth / focalX;
    const float right_extent =
        (static_cast<float>(imageWidth) - principalX) * model_depth / focalX;
    const float top_extent = principalY * model_depth / focalY;
    const float bottom_extent =
        (static_cast<float>(imageHeight) - principalY) * model_depth / focalY;
    return {
        optical_axis_center + right_axis * right_extent + up_axis * top_extent,
        optical_axis_center - right_axis * left_extent + up_axis * top_extent,
        optical_axis_center - right_axis * left_extent - up_axis * bottom_extent,
        optical_axis_center + right_axis * right_extent - up_axis * bottom_extent,
    };
}

} // namespace xjw::gui::camera_scene
