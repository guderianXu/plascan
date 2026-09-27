#include "TriangulationQuality.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace xjw
{

    namespace
    {

        constexpr double kMaximumReconstructionUncertainty = 1.0e6;

        bool pointProjectionJacobian(const placamera::FramePinholeModel& camera,
                                     const std::array<double, 3>& worldPoint,
                                     cv::Matx<double, 2, 3>* jacobian)
        {
            if (!jacobian)
            {
                return false;
            }

            const std::array<double, 3>& center = camera.pose().center;
            const double range =
                std::hypot(worldPoint[0] - center[0], worldPoint[1] - center[1], worldPoint[2] - center[2]);
            if (!std::isfinite(range) || range <= 1e-9)
            {
                return false;
            }

            const double step = std::max(1e-7, range * 1e-6);
            for (int axis = 0; axis < 3; ++axis)
            {
                std::array<double, 3> plus = worldPoint;
                std::array<double, 3> minus = worldPoint;
                plus[static_cast<std::size_t>(axis)] += step;
                minus[static_cast<std::size_t>(axis)] -= step;
                const auto projectedPlus = camera.groundToImageSigned({camera.groundFrame(), plus});
                const auto projectedMinus = camera.groundToImageSigned({camera.groundFrame(), minus});
                if (!projectedPlus || !projectedMinus)
                {
                    return false;
                }
                const std::array<double, 2> positivePixel{projectedPlus.value().image.sample,
                                                          projectedPlus.value().image.line};
                const std::array<double, 2> negativePixel{projectedMinus.value().image.sample,
                                                          projectedMinus.value().image.line};
                for (int pixelAxis = 0; pixelAxis < 2; ++pixelAxis)
                {
                    const double derivative = (positivePixel[static_cast<std::size_t>(pixelAxis)] -
                                               negativePixel[static_cast<std::size_t>(pixelAxis)]) /
                                              (2.0 * step);
                    if (!std::isfinite(derivative))
                    {
                        return false;
                    }
                    (*jacobian)(pixelAxis, axis) = derivative;
                }
            }
            return true;
        }

        bool cleanTiePointProjectionJacobian(const placamera::FramePinholeModel& camera,
                                             const std::array<double, 3>& worldPoint,
                                             cv::Matx<double, 2, 3>* jacobian)
        {
            if (!jacobian)
            {
                return false;
            }

            if (!camera.groundToImage({camera.groundFrame(), worldPoint}))
            {
                return false;
            }

            const std::array<double, 9>& rotation = camera.pose().cameraToWorldRotation;
            const std::array<double, 3>& center = camera.pose().center;
            const std::array<double, 3> delta{
                worldPoint[0] - center[0], worldPoint[1] - center[1], worldPoint[2] - center[2]};
            std::array<double, 3> local{};
            for (int axis = 0; axis < 3; ++axis)
            {
                for (int worldAxis = 0; worldAxis < 3; ++worldAxis)
                {
                    local[static_cast<std::size_t>(axis)] += rotation[static_cast<std::size_t>(worldAxis * 3 + axis)] *
                                                             delta[static_cast<std::size_t>(worldAxis)];
                }
            }
            if (!std::isfinite(local[0]) || !std::isfinite(local[1]) || !std::isfinite(local[2]) ||
                std::abs(local[2]) <= 1.0e-12)
            {
                return false;
            }

            const double inverse_z = 1.0 / local[2];
            const double x = local[0] * inverse_z;
            const double y = local[1] * inverse_z;
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const placamera::BrownConradyDistortion& distortion = camera.pinholeDefinition().distortion();
            const double radial =
                1.0 + distortion.radialK1 * r2 + distortion.radialK2 * r4 + distortion.radialK3 * r4 * r2;
            const double radial_derivative =
                distortion.radialK1 + 2.0 * distortion.radialK2 * r2 + 3.0 * distortion.radialK3 * r4;
            const double radial_x = 2.0 * x * radial_derivative;
            const double radial_y = 2.0 * y * radial_derivative;
            const double distorted_x_x =
                radial + x * radial_x + 2.0 * distortion.tangentialP1 * y + 6.0 * distortion.tangentialP2 * x;
            const double distorted_x_y =
                x * radial_y + 2.0 * distortion.tangentialP1 * x + 2.0 * distortion.tangentialP2 * y;
            const double distorted_y_x =
                y * radial_x + 2.0 * distortion.tangentialP1 * x + 2.0 * distortion.tangentialP2 * y;
            const double distorted_y_y =
                radial + y * radial_y + 6.0 * distortion.tangentialP1 * y + 2.0 * distortion.tangentialP2 * x;

            const placamera::FrameIntrinsics& intrinsics = camera.pinholeDefinition().intrinsics();
            const double u_scale = static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX;
            const double v_scale = static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY;
            const std::array<double, 3> local_u{u_scale * distorted_x_x * inverse_z,
                                                u_scale * distorted_x_y * inverse_z,
                                                -u_scale * (distorted_x_x * x + distorted_x_y * y) * inverse_z};
            const std::array<double, 3> local_v{v_scale * distorted_y_x * inverse_z,
                                                v_scale * distorted_y_y * inverse_z,
                                                -v_scale * (distorted_y_x * x + distorted_y_y * y) * inverse_z};
            for (int world_axis = 0; world_axis < 3; ++world_axis)
            {
                double derivative_u = 0.0;
                double derivative_v = 0.0;
                for (int local_axis = 0; local_axis < 3; ++local_axis)
                {
                    const double world_to_camera = rotation[static_cast<std::size_t>(world_axis * 3 + local_axis)];
                    derivative_u += local_u[static_cast<std::size_t>(local_axis)] * world_to_camera;
                    derivative_v += local_v[static_cast<std::size_t>(local_axis)] * world_to_camera;
                }
                if (!std::isfinite(derivative_u) || !std::isfinite(derivative_v))
                {
                    return false;
                }
                (*jacobian)(0, world_axis) = derivative_u;
                (*jacobian)(1, world_axis) = derivative_v;
            }
            return cv::norm(cv::Mat(*jacobian), cv::NORM_L2SQR) > 0.0;
        }

    } // namespace

    CleanTiePointQuality evaluateCleanTiePointQuality(const std::vector<TiePointQualityObservation>& observations,
                                                      const std::array<double, 3>& worldPoint)
    {
        CleanTiePointQuality result;
        double scale_sum = 0.0;
        cv::Matx33d information = cv::Matx33d::zeros();
        int geometry_count = 0;

        for (const TiePointQualityObservation& observation : observations)
        {
            ++result.imageCount;
            scale_sum += observation.measurementScale;
            if (!observation.camera)
            {
                continue;
            }

            const auto projected = observation.camera->groundToImage({observation.camera->groundFrame(), worldPoint});
            if (projected)
            {
                const double scale = observation.measurementScale == 0.0 ? 1.0 : observation.measurementScale;
                const double residual_x = projected.value().image.sample - observation.imagePoint[0];
                const double residual_y = projected.value().image.line - observation.imagePoint[1];
                const double normalized_residual = std::hypot(residual_x, residual_y) / scale;
                if (std::isfinite(normalized_residual))
                {
                    result.reprojectionError = std::max(result.reprojectionError, normalized_residual);
                }
            }

            cv::Matx<double, 2, 3> jacobian;
            if (!cleanTiePointProjectionJacobian(*observation.camera, worldPoint, &jacobian))
            {
                continue;
            }
            information += jacobian.t() * jacobian;
            ++geometry_count;
        }

        if (result.imageCount != 0)
        {
            result.projectionAccuracy = scale_sum / static_cast<double>(result.imageCount);
        }
        if (geometry_count < 2)
        {
            return result;
        }

        cv::Mat singular_values;
        cv::SVD::compute(cv::Mat(information), singular_values);
        if (singular_values.total() != 3)
        {
            return result;
        }
        const double first = singular_values.at<double>(0);
        const double second = singular_values.at<double>(1);
        const double third = singular_values.at<double>(2);
        const double minimum = std::min({first, second, third});
        const double maximum = std::max({first, second, third});
        if (minimum > 0.0 && std::isfinite(minimum) && std::isfinite(maximum))
        {
            result.reconstructionUncertainty = std::sqrt(maximum / minimum);
            result.hasProjectionGeometry = std::isfinite(result.reconstructionUncertainty);
        }
        else if (maximum > 0.0)
        {
            result.reconstructionUncertainty = std::numeric_limits<double>::infinity();
            result.hasProjectionGeometry = true;
        }
        return result;
    }

    double reconstructionUncertainty(const std::vector<TiePointQualityObservation>& observations,
                                     const std::array<double, 3>& worldPoint)
    {
        cv::Matx33d information = cv::Matx33d::zeros();
        int validObservationCount = 0;
        for (const TiePointQualityObservation& observation : observations)
        {
            if (!observation.camera)
            {
                continue;
            }
            cv::Matx<double, 2, 3> jacobian;
            if (!pointProjectionJacobian(*observation.camera, worldPoint, &jacobian))
            {
                continue;
            }
            const double scale = std::isfinite(observation.measurementScale) && observation.measurementScale > 0.0
                                     ? observation.measurementScale
                                     : 1.0;
            information += (jacobian.t() * jacobian) / (scale * scale);
            ++validObservationCount;
        }
        if (validObservationCount < 2)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        cv::Mat eigenvalues;
        if (!cv::eigen(cv::Mat(information), eigenvalues) || eigenvalues.total() != 3)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double largest = eigenvalues.at<double>(0);
        const double smallest = eigenvalues.at<double>(2);
        if (!std::isfinite(largest) || !std::isfinite(smallest) || largest <= 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double stableSmallest =
            std::max(smallest, largest / (kMaximumReconstructionUncertainty * kMaximumReconstructionUncertainty));
        return std::sqrt(largest / stableSmallest);
    }

    double projectionAccuracy(const std::vector<TiePointQualityObservation>& observations)
    {
        if (observations.empty())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double sum = 0.0;
        for (const TiePointQualityObservation& observation : observations)
        {
            if (!std::isfinite(observation.measurementScale) || observation.measurementScale <= 0.0)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            sum += observation.measurementScale;
        }
        return sum / static_cast<double>(observations.size());
    }

} // namespace xjw
