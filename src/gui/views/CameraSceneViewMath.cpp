#include "CameraSceneViewMath.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QVector2D>
#include <QVector4D>

namespace xjw::gui::camera_scene
{
namespace
{

constexpr float ScoreEpsilon = 1e-6f;
constexpr double CameraCardZoomResponseExponent = 0.25;

QVector3D normalizedOrZero(const QVector3D &value)
{
    return value.lengthSquared() > ScoreEpsilon * ScoreEpsilon
        ? value.normalized()
        : QVector3D();
}

bool interpolatedTriangleDepth(const QVector3D &point,
                               const QVector3D &a,
                               const QVector3D &b,
                               const QVector3D &c,
                               float *depth)
{
    const float denominator =
        (b.y() - c.y()) * (a.x() - c.x())
        + (c.x() - b.x()) * (a.y() - c.y());
    if (std::abs(denominator) <= ScoreEpsilon)
    {
        return false;
    }

    const float weightA =
        ((b.y() - c.y()) * (point.x() - c.x())
         + (c.x() - b.x()) * (point.y() - c.y()))
        / denominator;
    const float weightB =
        ((c.y() - a.y()) * (point.x() - c.x())
         + (a.x() - c.x()) * (point.y() - c.y()))
        / denominator;
    const float weightC = 1.0f - weightA - weightB;
    constexpr float edgeTolerance = 1.0e-5f;
    if (weightA < -edgeTolerance
        || weightB < -edgeTolerance
        || weightC < -edgeTolerance)
    {
        return false;
    }

    if (depth)
    {
        *depth = weightA * a.z() + weightB * b.z() + weightC * c.z();
    }
    return true;
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

double cameraPlaneScreenHalfExtentPixels(double zoomScale,
                                         double normalHalfExtentPixels)
{
    if (!std::isfinite(zoomScale)
        || !std::isfinite(normalHalfExtentPixels)
        || zoomScale <= 0.0
        || normalHalfExtentPixels <= 0.0)
    {
        return 0.0;
    }

    const double zoom_response = std::pow(
        zoomScale,
        -CameraCardZoomResponseExponent);
    const double half_extent_pixels = normalHalfExtentPixels * zoom_response;
    return std::isfinite(half_extent_pixels) ? half_extent_pixels : 0.0;
}

float cameraPlaneHalfExtentForScreenSize(
    const QVector3D &center,
    const QMatrix4x4 &worldToView,
    int viewportHeight,
    double zoomScale,
    float verticalFieldOfViewDegrees,
    double normalHalfExtentPixels)
{
    const double target_half_extent_pixels = cameraPlaneScreenHalfExtentPixels(
        zoomScale,
        normalHalfExtentPixels);
    if (viewportHeight <= 0
        || !std::isfinite(verticalFieldOfViewDegrees)
        || verticalFieldOfViewDegrees <= 0.0f
        || verticalFieldOfViewDegrees >= 179.0f
        || target_half_extent_pixels <= 0.0)
    {
        return 0.0f;
    }

    const float depth = -(worldToView * QVector4D(center, 1.0f)).z();
    if (!std::isfinite(depth) || depth <= ScoreEpsilon)
    {
        return 0.0f;
    }

    constexpr float pi = 3.14159265358979323846f;
    const float halfFovRadians =
        verticalFieldOfViewDegrees * pi / 360.0f;
    const float worldHeightAtDepth =
        2.0f * depth * std::tan(halfFovRadians);
    const double world_half_extent = static_cast<double>(worldHeightAtDepth)
        * target_half_extent_pixels
        / static_cast<double>(viewportHeight);
    if (!std::isfinite(world_half_extent)
        || world_half_extent > std::numeric_limits<float>::max())
    {
        return 0.0f;
    }
    return static_cast<float>(world_half_extent);
}

QLineF cameraPlaneLeaderLine(const QPointF &center,
                             const QPointF &directionProbe,
                             qreal startOffsetPixels,
                             qreal lengthPixels)
{
    if (!std::isfinite(center.x())
        || !std::isfinite(center.y())
        || !std::isfinite(directionProbe.x())
        || !std::isfinite(directionProbe.y())
        || !std::isfinite(startOffsetPixels)
        || !std::isfinite(lengthPixels)
        || startOffsetPixels < 0.0
        || lengthPixels <= 0.0)
    {
        return {};
    }

    const QPointF direction = directionProbe - center;
    const qreal directionLength = QLineF(QPointF(), direction).length();
    if (directionLength < 1.0e-6)
    {
        return {};
    }

    const QPointF unitDirection = direction / directionLength;
    const QPointF start = center + unitDirection * startOffsetPixels;
    return QLineF(start, start + unitDirection * lengthPixels);
}

bool pointIsBehindProjectedQuad(const QVector3D &pointNdc,
                                const QVector<QVector3D> &quadNdc,
                                float depthEpsilon)
{
    if (quadNdc.size() != 4
        || !std::isfinite(pointNdc.x())
        || !std::isfinite(pointNdc.y())
        || !std::isfinite(pointNdc.z())
        || !std::isfinite(depthEpsilon)
        || depthEpsilon < 0.0f)
    {
        return false;
    }

    float planeDepth = 0.0f;
    const bool insideFirst = interpolatedTriangleDepth(
        pointNdc, quadNdc.at(0), quadNdc.at(1), quadNdc.at(2), &planeDepth);
    const bool insideSecond = insideFirst || interpolatedTriangleDepth(
        pointNdc, quadNdc.at(0), quadNdc.at(2), quadNdc.at(3), &planeDepth);
    return insideSecond && pointNdc.z() >= planeDepth - depthEpsilon;
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

QVector<QVector3D> axisAlignedBoundingBoxLineVertices(
    const QVector3D &minimum,
    const QVector3D &maximum)
{
    const auto is_finite = [](const QVector3D &point)
    {
        return std::isfinite(point.x())
            && std::isfinite(point.y())
            && std::isfinite(point.z());
    };
    if (!is_finite(minimum) || !is_finite(maximum))
    {
        return {};
    }

    const QVector3D lower(
        std::min(minimum.x(), maximum.x()),
        std::min(minimum.y(), maximum.y()),
        std::min(minimum.z(), maximum.z()));
    const QVector3D upper(
        std::max(minimum.x(), maximum.x()),
        std::max(minimum.y(), maximum.y()),
        std::max(minimum.z(), maximum.z()));
    const QVector<QVector3D> corners{
        {lower.x(), lower.y(), lower.z()},
        {upper.x(), lower.y(), lower.z()},
        {upper.x(), upper.y(), lower.z()},
        {lower.x(), upper.y(), lower.z()},
        {lower.x(), lower.y(), upper.z()},
        {upper.x(), lower.y(), upper.z()},
        {upper.x(), upper.y(), upper.z()},
        {lower.x(), upper.y(), upper.z()},
    };
    constexpr int edges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };

    QVector<QVector3D> vertices;
    vertices.reserve(24);
    for (const auto &edge : edges)
    {
        vertices.push_back(corners.at(edge[0]));
        vertices.push_back(corners.at(edge[1]));
    }
    return vertices;
}

PointCloudPrincipalAxes pointCloudPrincipalAxes(
    const QVector<QVector3D> &points)
{
    PointCloudPrincipalAxes result;
    if (points.size() < 3)
    {
        return result;
    }

    QVector3D center;
    int finite_count = 0;
    for (const QVector3D &point : points)
    {
        if (!std::isfinite(point.x()) || !std::isfinite(point.y())
            || !std::isfinite(point.z()))
        {
            continue;
        }
        center += point;
        ++finite_count;
    }
    if (finite_count < 3)
    {
        return result;
    }
    center /= static_cast<float>(finite_count);

    double covariance[3][3]{};
    for (const QVector3D &point : points)
    {
        if (!std::isfinite(point.x()) || !std::isfinite(point.y())
            || !std::isfinite(point.z()))
        {
            continue;
        }
        const QVector3D offset = point - center;
        const double value[3]{offset.x(), offset.y(), offset.z()};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = row; column < 3; ++column)
            {
                covariance[row][column] += value[row] * value[column];
                covariance[column][row] = covariance[row][column];
            }
        }
    }

    double eigenvectors[3][3]{{1.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0},
                              {0.0, 0.0, 1.0}};
    for (int iteration = 0; iteration < 24; ++iteration)
    {
        int p = 0;
        int q = 1;
        double largest = std::abs(covariance[p][q]);
        for (int row = 0; row < 3; ++row)
        {
            for (int column = row + 1; column < 3; ++column)
            {
                if (std::abs(covariance[row][column]) > largest)
                {
                    p = row;
                    q = column;
                    largest = std::abs(covariance[row][column]);
                }
            }
        }
        const double diagonal_scale = std::max(
            1.0,
            std::abs(covariance[0][0]) + std::abs(covariance[1][1])
                + std::abs(covariance[2][2]));
        if (largest <= diagonal_scale * 1.0e-12)
        {
            break;
        }

        const double angle = 0.5 * std::atan2(
            2.0 * covariance[p][q],
            covariance[q][q] - covariance[p][p]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const double app = covariance[p][p];
        const double aqq = covariance[q][q];
        const double apq = covariance[p][q];
        covariance[p][p] = cosine * cosine * app
            - 2.0 * sine * cosine * apq + sine * sine * aqq;
        covariance[q][q] = sine * sine * app
            + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        covariance[p][q] = 0.0;
        covariance[q][p] = 0.0;
        for (int index = 0; index < 3; ++index)
        {
            if (index != p && index != q)
            {
                const double aip = covariance[index][p];
                const double aiq = covariance[index][q];
                covariance[index][p] = cosine * aip - sine * aiq;
                covariance[p][index] = covariance[index][p];
                covariance[index][q] = sine * aip + cosine * aiq;
                covariance[q][index] = covariance[index][q];
            }
            const double vip = eigenvectors[index][p];
            const double viq = eigenvectors[index][q];
            eigenvectors[index][p] = cosine * vip - sine * viq;
            eigenvectors[index][q] = sine * vip + cosine * viq;
        }
    }

    int order[3]{0, 1, 2};
    std::sort(std::begin(order), std::end(order), [&](int left, int right)
    {
        return covariance[left][left] > covariance[right][right];
    });
    const double largest_eigenvalue = covariance[order[0]][order[0]];
    if (!std::isfinite(largest_eigenvalue) || largest_eigenvalue <= 1.0e-12)
    {
        return result;
    }

    const auto eigenvector = [&](int column)
    {
        return QVector3D(static_cast<float>(eigenvectors[0][column]),
                         static_cast<float>(eigenvectors[1][column]),
                         static_cast<float>(eigenvectors[2][column])).normalized();
    };
    QVector3D first = eigenvector(order[0]);
    QVector3D second = eigenvector(order[1]);
    QVector3D third = QVector3D::crossProduct(first, second).normalized();
    second = QVector3D::crossProduct(third, first).normalized();
    if (first.isNull() || second.isNull() || third.isNull())
    {
        return result;
    }

    QVector<QVector2D> projected_points;
    projected_points.reserve(finite_count);
    for (const QVector3D &point : points)
    {
        if (!std::isfinite(point.x()) || !std::isfinite(point.y())
            || !std::isfinite(point.z()))
        {
            continue;
        }
        const QVector3D offset = point - center;
        projected_points.push_back(QVector2D(
            QVector3D::dotProduct(offset, first),
            QVector3D::dotProduct(offset, second)));
    }
    std::sort(projected_points.begin(), projected_points.end(),
              [](const QVector2D &left, const QVector2D &right)
    {
        return left.x() < right.x()
            || (left.x() == right.x() && left.y() < right.y());
    });
    projected_points.erase(
        std::unique(projected_points.begin(), projected_points.end()),
        projected_points.end());

    const auto cross = [](const QVector2D &origin,
                          const QVector2D &left,
                          const QVector2D &right)
    {
        return static_cast<double>(left.x() - origin.x())
                * static_cast<double>(right.y() - origin.y())
            - static_cast<double>(left.y() - origin.y())
                * static_cast<double>(right.x() - origin.x());
    };
    QVector<QVector2D> hull;
    hull.reserve(projected_points.size() * 2);
    for (const QVector2D &point : projected_points)
    {
        while (hull.size() >= 2
               && cross(hull.at(hull.size() - 2), hull.back(), point) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    const qsizetype lower_size = hull.size();
    for (qsizetype index = projected_points.size() - 2; index >= 0; --index)
    {
        const QVector2D point = projected_points.at(index);
        while (hull.size() > lower_size
               && cross(hull.at(hull.size() - 2), hull.back(), point) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    if (hull.size() > 1)
    {
        hull.pop_back();
    }

    // A projected sphere can have a very large sampled hull.  A bounded,
    // uniform hull subset keeps this display-only fit predictable; the caller
    // still projects every cloud point to compute the final enclosing bounds.
    constexpr qsizetype MaximumHullFitPoints = 4096;
    if (hull.size() > MaximumHullFitPoints)
    {
        QVector<QVector2D> reduced_hull;
        reduced_hull.reserve(MaximumHullFitPoints);
        for (qsizetype index = 0; index < MaximumHullFitPoints; ++index)
        {
            reduced_hull.push_back(hull.at(
                index * hull.size() / MaximumHullFitPoints));
        }
        hull = std::move(reduced_hull);
    }

    double best_area = std::numeric_limits<double>::infinity();
    QVector2D best_axis;
    double best_first_extent = 0.0;
    double best_second_extent = 0.0;
    constexpr qsizetype MaximumCandidateEdges = 1024;
    const qsizetype candidate_count = std::min(
        hull.size(), MaximumCandidateEdges);
    for (qsizetype candidate = 0; candidate < candidate_count; ++candidate)
    {
        const qsizetype index = candidate * hull.size() / candidate_count;
        QVector2D axis = hull.at((index + 1) % hull.size()) - hull.at(index);
        if (axis.lengthSquared() <= ScoreEpsilon * ScoreEpsilon)
        {
            continue;
        }
        axis.normalize();
        const QVector2D perpendicular(-axis.y(), axis.x());
        double minimum_first = std::numeric_limits<double>::infinity();
        double maximum_first = -std::numeric_limits<double>::infinity();
        double minimum_second = std::numeric_limits<double>::infinity();
        double maximum_second = -std::numeric_limits<double>::infinity();
        for (const QVector2D &point : hull)
        {
            const double along = QVector2D::dotProduct(point, axis);
            const double across = QVector2D::dotProduct(point, perpendicular);
            minimum_first = std::min(minimum_first, along);
            maximum_first = std::max(maximum_first, along);
            minimum_second = std::min(minimum_second, across);
            maximum_second = std::max(maximum_second, across);
        }
        const double first_extent = maximum_first - minimum_first;
        const double second_extent = maximum_second - minimum_second;
        const double area = first_extent * second_extent;
        if (area < best_area)
        {
            best_area = area;
            best_axis = axis;
            best_first_extent = first_extent;
            best_second_extent = second_extent;
        }
    }
    if (std::isfinite(best_area) && best_area > 0.0)
    {
        QVector3D footprint_first =
            (first * best_axis.x() + second * best_axis.y()).normalized();
        QVector3D footprint_second =
            QVector3D::crossProduct(third, footprint_first).normalized();
        if (best_second_extent > best_first_extent)
        {
            std::swap(footprint_first, footprint_second);
        }
        first = footprint_first;
        second = QVector3D::crossProduct(third, first).normalized();
    }

    const auto canonicalize = [](QVector3D axis)
    {
        const float values[3]{axis.x(), axis.y(), axis.z()};
        int largest_index = 0;
        for (int index = 1; index < 3; ++index)
        {
            if (std::abs(values[index]) > std::abs(values[largest_index]))
            {
                largest_index = index;
            }
        }
        return values[largest_index] < 0.0f ? -axis : axis;
    };
    first = canonicalize(first);
    second = canonicalize(second);
    third = QVector3D::crossProduct(first, second).normalized();
    second = QVector3D::crossProduct(third, first).normalized();

    result.center = center;
    result.first = first;
    result.second = second;
    result.third = third;
    result.valid = true;
    return result;
}

QVector<QVector3D> orientedBoundingBoxLineVertices(
    const PointCloudPrincipalAxes &axes,
    const QVector3D &minimum,
    const QVector3D &maximum)
{
    if (!axes.valid)
    {
        return {};
    }
    const QVector3D lower(std::min(minimum.x(), maximum.x()),
                          std::min(minimum.y(), maximum.y()),
                          std::min(minimum.z(), maximum.z()));
    const QVector3D upper(std::max(minimum.x(), maximum.x()),
                          std::max(minimum.y(), maximum.y()),
                          std::max(minimum.z(), maximum.z()));
    QVector<QVector3D> local_corners{
        {lower.x(), lower.y(), lower.z()}, {upper.x(), lower.y(), lower.z()},
        {upper.x(), upper.y(), lower.z()}, {lower.x(), upper.y(), lower.z()},
        {lower.x(), lower.y(), upper.z()}, {upper.x(), lower.y(), upper.z()},
        {upper.x(), upper.y(), upper.z()}, {lower.x(), upper.y(), upper.z()},
    };
    QVector<QVector3D> world_corners;
    world_corners.reserve(local_corners.size());
    for (const QVector3D &corner : local_corners)
    {
        world_corners.push_back(axes.center + axes.first * corner.x()
                                + axes.second * corner.y()
                                + axes.third * corner.z());
    }
    constexpr int edges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    QVector<QVector3D> vertices;
    vertices.reserve(24);
    for (const auto &edge : edges)
    {
        vertices.push_back(world_corners.at(edge[0]));
        vertices.push_back(world_corners.at(edge[1]));
    }
    return vertices;
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
