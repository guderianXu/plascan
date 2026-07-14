#include "MarkerGeometry.h"

#include <QLineF>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw::control_points
{

namespace
{

struct Observation
{
    const MarkerProjection *projection = nullptr;
    const MarkerCamera *camera = nullptr;
};

const MarkerCamera *findCamera(const QVector<MarkerCamera> &cameras, const QString &imageId)
{
    const auto it = std::find_if(cameras.cbegin(), cameras.cend(), [&imageId](const MarkerCamera &camera)
    {
        return camera.imageId == imageId;
    });
    return it == cameras.cend() ? nullptr : &*it;
}

cv::Point3d dltPoint(const QVector<Observation> &observations)
{
    cv::Mat design(static_cast<int>(observations.size()) * 2, 4, CV_64F);
    for (int index = 0; index < observations.size(); ++index)
    {
        const cv::Matx34d projection = observations.at(index).camera->projectionMatrix();
        const double x = observations.at(index).projection->xy.x();
        const double y = observations.at(index).projection->xy.y();
        for (int column = 0; column < 4; ++column)
        {
            design.at<double>(index * 2, column) = x * projection(2, column) - projection(0, column);
            design.at<double>(index * 2 + 1, column) = y * projection(2, column) - projection(1, column);
        }
    }

    cv::SVD decomposition(design, cv::SVD::FULL_UV);
    const cv::Mat homogeneous = decomposition.vt.row(3);
    const double scale = homogeneous.at<double>(0, 3);
    if (!std::isfinite(scale) || std::abs(scale) < 1.0e-12)
    {
        return cv::Point3d(std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN());
    }
    return cv::Point3d(homogeneous.at<double>(0, 0) / scale,
                       homogeneous.at<double>(0, 1) / scale,
                       homogeneous.at<double>(0, 2) / scale);
}

double reprojectionError(const Observation &observation, const cv::Point3d &point)
{
    const QPointF projected = observation.camera->project(point);
    return QLineF(projected, observation.projection->xy).length();
}

double median(QVector<double> values)
{
    if (values.isEmpty()) return 0.0;
    std::sort(values.begin(), values.end());
    const int middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values.at(middle - 1) + values.at(middle))
        : values.at(middle);
}

double minimumIntersectionAngle(const QVector<Observation> &observations, const cv::Point3d &point)
{
    double minimum = 180.0;
    for (int first = 0; first < observations.size(); ++first)
    {
        const cv::Point3d first_center = observations.at(first).camera->center();
        cv::Vec3d first_ray(point.x - first_center.x,
                            point.y - first_center.y,
                            point.z - first_center.z);
        first_ray /= cv::norm(first_ray);
        for (int second = first + 1; second < observations.size(); ++second)
        {
            const cv::Point3d second_center = observations.at(second).camera->center();
            cv::Vec3d second_ray(point.x - second_center.x,
                                 point.y - second_center.y,
                                 point.z - second_center.z);
            second_ray /= cv::norm(second_ray);
            const double cosine = std::clamp(first_ray.dot(second_ray), -1.0, 1.0);
            minimum = std::min(minimum, std::acos(cosine) * 180.0 / CV_PI);
        }
    }
    return observations.size() >= 2 ? minimum : 0.0;
}

cv::Matx33d skew(const cv::Vec3d &value)
{
    return cv::Matx33d(0.0, -value[2], value[1],
                       value[2], 0.0, -value[0],
                       -value[1], value[0], 0.0);
}

} // namespace

cv::Matx34d MarkerCamera::projectionMatrix() const
{
    const cv::Matx34d extrinsics(rotation(0, 0), rotation(0, 1), rotation(0, 2), translation[0],
                                 rotation(1, 0), rotation(1, 1), rotation(1, 2), translation[1],
                                 rotation(2, 0), rotation(2, 1), rotation(2, 2), translation[2]);
    return intrinsics * extrinsics;
}

cv::Point3d MarkerCamera::center() const
{
    const cv::Vec3d value = -(rotation.t() * translation);
    return cv::Point3d(value[0], value[1], value[2]);
}

double MarkerCamera::depth(const cv::Point3d &point) const
{
    const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + translation;
    return camera_point[2];
}

QPointF MarkerCamera::project(const cv::Point3d &point) const
{
    const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + translation;
    const cv::Vec3d pixel = intrinsics * camera_point;
    if (std::abs(pixel[2]) < 1.0e-12)
    {
        return QPointF(std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN());
    }
    return QPointF(pixel[0] / pixel[2], pixel[1] / pixel[2]);
}

bool MarkerCamera::contains(const QPointF &pixel) const
{
    return std::isfinite(pixel.x()) && std::isfinite(pixel.y())
        && pixel.x() >= 0.0 && pixel.y() >= 0.0
        && pixel.x() < imageSize.width() && pixel.y() < imageSize.height();
}

double EpipolarBand::distanceTo(const QPointF &pixel) const
{
    if (!valid) return std::numeric_limits<double>::infinity();
    return std::abs(a * pixel.x() + b * pixel.y() + c);
}

bool EpipolarBand::contains(const QPointF &pixel) const
{
    return distanceTo(pixel) <= halfWidthPx;
}

MarkerTriangulation triangulateMarker(const Marker &marker,
                                      const QVector<MarkerCamera> &cameras,
                                      const MarkerTriangulationOptions &options)
{
    MarkerTriangulation result;
    QVector<Observation> observations;
    for (const MarkerProjection &projection : marker.projections)
    {
        if (!projectionParticipatesInAdjustment(projection.state)) continue;
        const MarkerCamera *camera = findCamera(cameras, projection.imageId);
        if (camera) observations.push_back({&projection, camera});
    }
    if (observations.size() < 2)
    {
        result.error = QStringLiteral("至少需要两个有效标记投影才能三角化");
        return result;
    }

    cv::Point3d point = dltPoint(observations);
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
    {
        result.error = QStringLiteral("标记 DLT 三角化退化");
        return result;
    }

    QVector<double> residuals;
    residuals.reserve(observations.size());
    for (const Observation &observation : observations)
    {
        residuals.push_back(reprojectionError(observation, point));
    }
    const double robust_limit = std::min(options.maximumReprojectionErrorPx,
                                         std::max(1.0, median(residuals) * 2.5));
    QVector<Observation> inliers;
    for (int index = 0; index < observations.size(); ++index)
    {
        if (residuals.at(index) <= robust_limit) inliers.push_back(observations.at(index));
    }
    if (inliers.size() >= 2 && inliers.size() != observations.size())
    {
        observations = inliers;
        point = dltPoint(observations);
    }

    double squared_error = 0.0;
    for (const Observation &observation : observations)
    {
        if (observation.camera->depth(point) <= 0.0)
        {
            result.error = QStringLiteral("三角化标记未通过正深度检查");
            return result;
        }
        const double error = reprojectionError(observation, point);
        result.residualByImage.insert(observation.camera->imageId, error);
        result.usedImageIds.push_back(observation.camera->imageId);
        squared_error += error * error;
    }
    result.rmsReprojectionPx = std::sqrt(squared_error / observations.size());
    if (result.rmsReprojectionPx > options.maximumReprojectionErrorPx)
    {
        result.error = QStringLiteral("标记重投影 RMS 超限: %1 px").arg(result.rmsReprojectionPx, 0, 'f', 3);
        return result;
    }
    result.minimumIntersectionAngleDegrees = minimumIntersectionAngle(observations, point);
    if (result.minimumIntersectionAngleDegrees < options.minimumIntersectionAngleDegrees)
    {
        result.error = QStringLiteral("标记最小交会角不足: %1°")
                           .arg(result.minimumIntersectionAngleDegrees, 0, 'f', 3);
        return result;
    }
    result.point = point;
    result.success = true;
    return result;
}

EpipolarBand epipolarSearchBand(const QPointF &sourcePixel,
                                const MarkerCamera &sourceCamera,
                                const MarkerCamera &targetCamera,
                                double halfWidthPx)
{
    EpipolarBand band;
    const cv::Matx33d relative_rotation = targetCamera.rotation * sourceCamera.rotation.t();
    const cv::Vec3d relative_translation = targetCamera.translation
        - relative_rotation * sourceCamera.translation;
    const cv::Matx33d essential = skew(relative_translation) * relative_rotation;
    const cv::Matx33d fundamental = targetCamera.intrinsics.inv().t()
        * essential * sourceCamera.intrinsics.inv();
    const cv::Vec3d line = fundamental * cv::Vec3d(sourcePixel.x(), sourcePixel.y(), 1.0);
    const double norm = std::hypot(line[0], line[1]);
    if (!std::isfinite(norm) || norm < 1.0e-12) return band;
    band.a = line[0] / norm;
    band.b = line[1] / norm;
    band.c = line[2] / norm;
    band.halfWidthPx = std::max(0.0, halfWidthPx);
    band.valid = true;
    return band;
}

} // namespace xjw::control_points
