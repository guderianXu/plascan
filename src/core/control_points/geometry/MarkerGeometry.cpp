#include "MarkerGeometry.h"

#include <QLineF>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace xjw::control_points
{
    namespace
    {

        struct Observation
        {
            const MarkerProjection* projection = nullptr;
            const MarkerImageView* view = nullptr;
            placamera::ImagingLocus locus;
        };

        QString imageId(const MarkerImageView& view)
        {
            return QString::fromStdString(std::string(view.camera->imageId().value()));
        }

        const MarkerImageView* findView(const QVector<MarkerImageView>& views, const QString& image_id)
        {
            const auto it = std::find_if(views.cbegin(),
                                         views.cend(),
                                         [&image_id](const MarkerImageView& view)
                                         { return view.camera && imageId(view) == image_id; });
            return it == views.cend() ? nullptr : &*it;
        }

        placamera::GroundCoordinate groundPoint(const placamera::FrameId& frame, const cv::Point3d& point)
        {
            return {frame, {point.x, point.y, point.z}};
        }

        std::optional<QPointF> projectPoint(const MarkerImageView& view, const cv::Point3d& point)
        {
            const auto projected = view.camera->groundToImage(groundPoint(view.camera->groundFrame(), point));
            if (!projected)
            {
                return std::nullopt;
            }
            return QPointF(projected.value().image.sample, projected.value().image.line);
        }

        cv::Point3d intersectRays(const QVector<Observation>& observations)
        {
            cv::Mat normal(3, 3, CV_64F, cv::Scalar(0));
            cv::Mat right_hand_side(3, 1, CV_64F, cv::Scalar(0));
            for (const Observation& observation : observations)
            {
                const auto& origin = observation.locus.origin.position;
                const auto& direction = observation.locus.direction;
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        const double projector = (row == column ? 1.0 : 0.0) - direction[row] * direction[column];
                        normal.at<double>(row, column) += projector;
                        right_hand_side.at<double>(row) += projector * origin[column];
                    }
                }
            }

            cv::Mat solution;
            if (!cv::solve(normal, right_hand_side, solution, cv::DECOMP_SVD))
            {
                const double invalid = std::numeric_limits<double>::quiet_NaN();
                return {invalid, invalid, invalid};
            }
            return {solution.at<double>(0), solution.at<double>(1), solution.at<double>(2)};
        }

        double reprojectionError(const Observation& observation, const cv::Point3d& point)
        {
            const auto projected = projectPoint(*observation.view, point);
            return projected ? QLineF(*projected, observation.projection->xy).length()
                             : std::numeric_limits<double>::infinity();
        }

        double median(QVector<double> values)
        {
            if (values.isEmpty())
            {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            const int middle = values.size() / 2;
            return values.size() % 2 == 0 ? 0.5 * (values.at(middle - 1) + values.at(middle)) : values.at(middle);
        }

        double minimumIntersectionAngle(const QVector<Observation>& observations, const cv::Point3d& point)
        {
            double minimum = 180.0;
            for (int first = 0; first < observations.size(); ++first)
            {
                const auto& first_center = observations.at(first).locus.origin.position;
                cv::Vec3d first_ray(point.x - first_center[0], point.y - first_center[1], point.z - first_center[2]);
                const double first_length = cv::norm(first_ray);
                if (!(first_length > 0.0))
                {
                    return 0.0;
                }
                first_ray /= first_length;
                for (int second = first + 1; second < observations.size(); ++second)
                {
                    const auto& second_center = observations.at(second).locus.origin.position;
                    cv::Vec3d second_ray(
                        point.x - second_center[0], point.y - second_center[1], point.z - second_center[2]);
                    const double second_length = cv::norm(second_ray);
                    if (!(second_length > 0.0))
                    {
                        return 0.0;
                    }
                    second_ray /= second_length;
                    const double cosine = std::clamp(first_ray.dot(second_ray), -1.0, 1.0);
                    minimum = std::min(minimum, std::acos(cosine) * 180.0 / CV_PI);
                }
            }
            return observations.size() >= 2 ? minimum : 0.0;
        }

        placamera::GroundCoordinate pointAlongRay(const placamera::ImagingLocus& locus, double distance)
        {
            placamera::GroundCoordinate point = locus.origin;
            for (int axis = 0; axis < 3; ++axis)
            {
                point.position[axis] += distance * locus.direction[axis];
            }
            return point;
        }

    } // namespace

    double EpipolarBand::distanceTo(const QPointF& pixel) const
    {
        return valid ? std::abs(a * pixel.x() + b * pixel.y() + c) : std::numeric_limits<double>::infinity();
    }

    bool EpipolarBand::contains(const QPointF& pixel) const
    {
        return distanceTo(pixel) <= halfWidthPx;
    }

    MarkerTriangulation triangulateMarker(const Marker& marker,
                                          const QVector<MarkerImageView>& views,
                                          const MarkerTriangulationOptions& options)
    {
        MarkerTriangulation result;
        QVector<Observation> observations;
        std::optional<placamera::FrameId> frame;
        for (const MarkerProjection& projection : marker.projections)
        {
            if (!projectionParticipatesInAdjustment(projection.state))
            {
                continue;
            }
            const MarkerImageView* view = findView(views, projection.imageId);
            if (!view)
            {
                continue;
            }
            if (frame && *frame != view->camera->groundFrame())
            {
                result.error = QStringLiteral("标记观测相机不在同一世界坐标系");
                return result;
            }
            const auto locus =
                view->camera->imageToImagingLocus(placamera::ImageCoordinate{projection.xy.x(), projection.xy.y()});
            if (!locus)
            {
                continue;
            }
            frame = view->camera->groundFrame();
            observations.push_back(Observation{&projection, view, locus.value()});
        }
        if (observations.size() < 2)
        {
            result.error = QStringLiteral("至少需要两个有效标记投影才能三角化");
            return result;
        }

        cv::Point3d point = intersectRays(observations);
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            result.error = QStringLiteral("标记成像射线交会退化");
            return result;
        }

        QVector<double> residuals;
        residuals.reserve(observations.size());
        for (const Observation& observation : observations)
        {
            residuals.push_back(reprojectionError(observation, point));
        }
        const double robust_limit =
            std::min(options.maximumReprojectionErrorPx, std::max(1.0, median(residuals) * 2.5));
        QVector<Observation> inliers;
        for (int index = 0; index < observations.size(); ++index)
        {
            if (residuals.at(index) <= robust_limit)
            {
                inliers.push_back(observations.at(index));
            }
        }
        if (inliers.size() >= 2 && inliers.size() != observations.size())
        {
            observations = inliers;
            point = intersectRays(observations);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                result.error = QStringLiteral("标记成像射线交会退化");
                return result;
            }
        }

        double squared_error = 0.0;
        for (const Observation& observation : observations)
        {
            const double error = reprojectionError(observation, point);
            if (!std::isfinite(error))
            {
                result.error = QStringLiteral("三角化标记未通过正深度检查");
                return result;
            }
            result.residualByImage.insert(imageId(*observation.view), error);
            result.usedImageIds.push_back(imageId(*observation.view));
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
            result.error =
                QStringLiteral("标记最小交会角不足: %1°").arg(result.minimumIntersectionAngleDegrees, 0, 'f', 3);
            return result;
        }
        result.point = point;
        result.groundFrame = frame;
        result.success = true;
        return result;
    }

    EpipolarBand epipolarSearchBand(const QPointF& sourcePixel,
                                    const MarkerImageView& sourceView,
                                    const MarkerImageView& targetView,
                                    double halfWidthPx)
    {
        EpipolarBand band;
        if (!sourceView.camera || !targetView.camera ||
            sourceView.camera->groundFrame() != targetView.camera->groundFrame())
        {
            return band;
        }
        const auto locus =
            sourceView.camera->imageToImagingLocus(placamera::ImageCoordinate{sourcePixel.x(), sourcePixel.y()});
        if (!locus)
        {
            return band;
        }
        const auto& source_center = sourceView.camera->pose().center;
        const auto& target_center = targetView.camera->pose().center;
        const double baseline =
            std::hypot(std::hypot(source_center[0] - target_center[0], source_center[1] - target_center[1]),
                       source_center[2] - target_center[2]);
        const double near_distance = std::max(1.0, baseline * 2.0);
        const auto near_pixel = targetView.camera->groundToImage(pointAlongRay(locus.value(), near_distance));
        const auto far_pixel = targetView.camera->groundToImage(pointAlongRay(locus.value(), near_distance * 1000.0));
        if (!near_pixel || !far_pixel)
        {
            return band;
        }
        const double near_sample = near_pixel.value().image.sample;
        const double near_line = near_pixel.value().image.line;
        const double far_sample = far_pixel.value().image.sample;
        const double far_line = far_pixel.value().image.line;
        const double a = near_line - far_line;
        const double b = far_sample - near_sample;
        const double norm = std::hypot(a, b);
        if (!std::isfinite(norm) || norm < 1.0e-12)
        {
            return band;
        }
        band.a = a / norm;
        band.b = b / norm;
        band.c = (near_sample * far_line - far_sample * near_line) / norm;
        band.halfWidthPx = std::max(0.0, halfWidthPx);
        band.valid = true;
        return band;
    }

} // namespace xjw::control_points
