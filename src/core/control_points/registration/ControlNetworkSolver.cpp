#include "ControlNetworkSolver.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::control_points
{

namespace
{

using Point3 = std::array<double, 3>;

bool finitePoint(const Point3 &point)
{
    return std::all_of(point.cbegin(), point.cend(), [](double value)
    {
        return std::isfinite(value);
    });
}

Point3 subtract(const Point3 &left, const Point3 &right)
{
    return {{left[0] - right[0], left[1] - right[1], left[2] - right[2]}};
}

double norm(const Point3 &point)
{
    return std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
}

double distance(const Point3 &left, const Point3 &right)
{
    return norm(subtract(left, right));
}

double sigmaRms(const ControlNetworkPoint &point)
{
    double sum = 0.0;
    int count = 0;
    for (double sigma : point.sigma)
    {
        if (std::isfinite(sigma) && sigma > 0.0)
        {
            sum += sigma * sigma;
            ++count;
        }
    }
    return count > 0 ? std::sqrt(sum / static_cast<double>(count)) : 1.0;
}

double pointWeight(const ControlNetworkPoint &point)
{
    const double sigma = std::max(1.0e-9, sigmaRms(point));
    return 1.0 / (sigma * sigma);
}

double triangleAreaRatio(const Point3 &first, const Point3 &second, const Point3 &third)
{
    const Point3 a = subtract(second, first);
    const Point3 b = subtract(third, first);
    const Point3 cross{{a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]}};
    const double maximum_edge = std::max({distance(first, second),
                                          distance(first, third),
                                          distance(second, third)});
    if (!(maximum_edge > 1.0e-12)) return 0.0;
    return norm(cross) / (maximum_edge * maximum_edge);
}

bool nonCollinear(const std::vector<const ControlNetworkPoint *> &sample,
                  double minimumRatio)
{
    return sample.size() == 3
        && triangleAreaRatio(sample[0]->estimatedPoint,
                             sample[1]->estimatedPoint,
                             sample[2]->estimatedPoint) >= minimumRatio
        && triangleAreaRatio(sample[0]->referencePoint,
                             sample[1]->referencePoint,
                             sample[2]->referencePoint) >= minimumRatio;
}

SimilarityTransform3D weightedUmeyama(
    const std::vector<const ControlNetworkPoint *> &points)
{
    SimilarityTransform3D transform;
    if (points.size() < 3) return transform;

    Point3 source_mean{{0.0, 0.0, 0.0}};
    Point3 target_mean{{0.0, 0.0, 0.0}};
    double weight_sum = 0.0;
    for (const ControlNetworkPoint *point : points)
    {
        const double weight = pointWeight(*point);
        weight_sum += weight;
        for (int axis = 0; axis < 3; ++axis)
        {
            source_mean[axis] += weight * point->estimatedPoint[axis];
            target_mean[axis] += weight * point->referencePoint[axis];
        }
    }
    if (!(weight_sum > 0.0)) return transform;
    for (int axis = 0; axis < 3; ++axis)
    {
        source_mean[axis] /= weight_sum;
        target_mean[axis] /= weight_sum;
    }

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    double source_variance = 0.0;
    for (const ControlNetworkPoint *point : points)
    {
        const double weight = pointWeight(*point);
        const Point3 source = subtract(point->estimatedPoint, source_mean);
        const Point3 target = subtract(point->referencePoint, target_mean);
        source_variance += weight * (source[0] * source[0]
                                   + source[1] * source[1]
                                   + source[2] * source[2]);
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                covariance.at<double>(row, column) += weight * target[row] * source[column];
            }
        }
    }
    source_variance /= weight_sum;
    covariance /= weight_sum;
    if (!(source_variance > 1.0e-18)) return transform;

    cv::SVD decomposition(covariance, cv::SVD::FULL_UV);
    cv::Mat sign = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rotation = decomposition.u * decomposition.vt;
    if (cv::determinant(rotation) < 0.0)
    {
        sign.at<double>(2, 2) = -1.0;
        rotation = decomposition.u * sign * decomposition.vt;
    }

    double scale_numerator = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        scale_numerator += decomposition.w.at<double>(axis) * sign.at<double>(axis, axis);
    }
    transform.scale = scale_numerator / source_variance;
    if (!(transform.scale > 0.0) || !std::isfinite(transform.scale)) return {};

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            transform.rotation[static_cast<std::size_t>(row * 3 + column)] =
                rotation.at<double>(row, column);
        }
    }
    transform.valid = true;
    const Point3 mapped_mean = transform.apply(source_mean);
    for (int axis = 0; axis < 3; ++axis)
    {
        transform.translation[axis] = target_mean[axis] - mapped_mean[axis];
    }
    return transform;
}

double inlierThreshold(const ControlNetworkPoint &point, const ControlNetworkOptions &options)
{
    return std::max(options.inlierThreshold, options.sigmaMultiplier * sigmaRms(point));
}

MarkerResidual residualFor(const ControlNetworkPoint &point,
                           const SimilarityTransform3D &transform,
                           const ControlNetworkOptions &options)
{
    MarkerResidual residual;
    residual.markerId = point.markerId;
    residual.role = point.role;
    residual.delta = subtract(transform.apply(point.estimatedPoint), point.referencePoint);
    residual.total = norm(residual.delta);
    residual.normalized = residual.total / std::max(1.0e-9, sigmaRms(point));
    residual.inlier = point.role == MarkerRole::ControlPoint
        && residual.total <= inlierThreshold(point, options);
    return residual;
}

} // namespace

std::array<double, 3> SimilarityTransform3D::apply(const std::array<double, 3> &point) const
{
    std::array<double, 3> result = translation;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            result[row] += scale
                * rotation[static_cast<std::size_t>(row * 3 + column)]
                * point[column];
        }
    }
    return result;
}

std::array<double, 9> SimilarityTransform3D::rotate(
    const std::array<double, 9> &cameraToWorldRotation) const
{
    std::array<double, 9> result{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                result[static_cast<std::size_t>(row * 3 + column)] +=
                    rotation[static_cast<std::size_t>(row * 3 + inner)]
                    * cameraToWorldRotation[static_cast<std::size_t>(inner * 3 + column)];
            }
        }
    }
    return result;
}

ControlNetworkResult solveControlNetwork(const ControlNetworkInput &input)
{
    ControlNetworkResult result;
    std::vector<const ControlNetworkPoint *> controls;
    for (const ControlNetworkPoint &point : input.points)
    {
        if (point.enabled && point.role == MarkerRole::ControlPoint
            && finitePoint(point.estimatedPoint) && finitePoint(point.referencePoint))
        {
            controls.push_back(&point);
        }
    }
    if (controls.size() < 3)
    {
        result.error = QStringLiteral("绝对定向至少需要三个有效控制点");
        return result;
    }

    SimilarityTransform3D best;
    std::vector<const ControlNetworkPoint *> best_inliers;
    double best_weight = -1.0;
    double best_weighted_error = std::numeric_limits<double>::infinity();
    int hypotheses = 0;
    bool found_non_collinear_sample = false;
    for (std::size_t first = 0; first + 2 < controls.size(); ++first)
    {
        for (std::size_t second = first + 1; second + 1 < controls.size(); ++second)
        {
            for (std::size_t third = second + 1; third < controls.size(); ++third)
            {
                if (hypotheses++ >= std::max(1, input.options.maximumHypotheses)) break;
                const std::vector<const ControlNetworkPoint *> sample{
                    controls[first], controls[second], controls[third]};
                if (!nonCollinear(sample, input.options.minimumTriangleAreaRatio)) continue;
                found_non_collinear_sample = true;
                const SimilarityTransform3D candidate = weightedUmeyama(sample);
                if (!candidate.valid) continue;

                std::vector<const ControlNetworkPoint *> inliers;
                double inlier_weight = 0.0;
                double weighted_error = 0.0;
                for (const ControlNetworkPoint *control : controls)
                {
                    const double residual = distance(candidate.apply(control->estimatedPoint),
                                                     control->referencePoint);
                    if (residual <= inlierThreshold(*control, input.options))
                    {
                        const double weight = pointWeight(*control);
                        inliers.push_back(control);
                        inlier_weight += weight;
                        weighted_error += weight * residual * residual;
                    }
                }
                if (inliers.size() < 3) continue;
                if (inlier_weight > best_weight
                    || (std::abs(inlier_weight - best_weight) <= 1.0e-12
                        && weighted_error < best_weighted_error))
                {
                    best = candidate;
                    best_inliers = std::move(inliers);
                    best_weight = inlier_weight;
                    best_weighted_error = weighted_error;
                }
            }
            if (hypotheses >= std::max(1, input.options.maximumHypotheses)) break;
        }
        if (hypotheses >= std::max(1, input.options.maximumHypotheses)) break;
    }

    if (!found_non_collinear_sample)
    {
        result.error = QStringLiteral("控制点几何退化：控制点共线或物方坐标共线");
        return result;
    }
    if (best_inliers.size() < 3)
    {
        result.error = QStringLiteral("控制网 RANSAC 未找到至少三个一致控制点");
        return result;
    }

    best = weightedUmeyama(best_inliers);
    if (!best.valid)
    {
        result.error = QStringLiteral("控制网加权 Umeyama 求解失败");
        return result;
    }
    result.transform = best;
    double inlier_sum_squared = 0.0;
    for (const ControlNetworkPoint &point : input.points)
    {
        if (!point.enabled || !finitePoint(point.estimatedPoint) || !finitePoint(point.referencePoint)) continue;
        MarkerResidual residual = residualFor(point, best, input.options);
        if (point.role == MarkerRole::ControlPoint)
        {
            if (residual.inlier)
            {
                ++result.controlInlierCount;
                inlier_sum_squared += residual.total * residual.total;
            }
            result.controlResiduals.push_back(residual);
        }
        else if (point.role == MarkerRole::CheckPoint)
        {
            result.checkPointResiduals.push_back(residual);
        }
    }
    result.controlInlierRms = result.controlInlierCount > 0
        ? std::sqrt(inlier_sum_squared / static_cast<double>(result.controlInlierCount))
        : 0.0;
    result.ok = result.controlInlierCount >= 3;
    if (!result.ok) result.error = QStringLiteral("绝对定向后有效控制点少于三个");
    return result;
}

} // namespace xjw::control_points
