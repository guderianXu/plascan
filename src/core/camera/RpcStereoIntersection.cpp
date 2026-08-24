#include "RpcStereoIntersection.h"

#include <algorithm>
#include <cmath>

namespace xjw
{
    namespace
    {

        using Vector3 = std::array<double, 3>;

        double dot(const Vector3& first, const Vector3& second)
        {
            return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
        }

        Vector3 subtract(const Vector3& first, const Vector3& second)
        {
            return {{first[0] - second[0], first[1] - second[1], first[2] - second[2]}};
        }

        Vector3 pointOnRay(const CameraImagingRay& ray, double distance)
        {
            return {{ray.originMeters[0] + distance * ray.direction[0],
                     ray.originMeters[1] + distance * ray.direction[1],
                     ray.originMeters[2] + distance * ray.direction[2]}};
        }

        bool initialPoint(const RpcCameraModel& firstCamera,
                          const CameraImageCoordinate& firstObservation,
                          const RpcCameraModel& secondCamera,
                          const CameraImageCoordinate& secondObservation,
                          Vector3* point)
        {
            CameraImagingRay first_ray;
            CameraImagingRay second_ray;
            if (!firstCamera.rayForPixel(firstObservation, &first_ray) ||
                !secondCamera.rayForPixel(secondObservation, &second_ray))
            {
                return false;
            }
            const Vector3 between_origins = subtract(first_ray.originMeters, second_ray.originMeters);
            const double direction_dot = dot(first_ray.direction, second_ray.direction);
            const double first_projection = dot(first_ray.direction, between_origins);
            const double second_projection = dot(second_ray.direction, between_origins);
            const double denominator = 1.0 - direction_dot * direction_dot;
            if (std::abs(denominator) < 1.0e-12)
            {
                return false;
            }
            const double first_distance = (direction_dot * second_projection - first_projection) / denominator;
            const double second_distance = (second_projection - direction_dot * first_projection) / denominator;
            const Vector3 first_point = pointOnRay(first_ray, first_distance);
            const Vector3 second_point = pointOnRay(second_ray, second_distance);
            *point = {{0.5 * (first_point[0] + second_point[0]),
                       0.5 * (first_point[1] + second_point[1]),
                       0.5 * (first_point[2] + second_point[2])}};
            return true;
        }

        bool residuals(const RpcCameraModel& firstCamera,
                       const CameraImageCoordinate& firstObservation,
                       const RpcCameraModel& secondCamera,
                       const CameraImageCoordinate& secondObservation,
                       const Vector3& point,
                       std::array<double, 4>* values)
        {
            CameraGroundProjection first_projection;
            CameraGroundProjection second_projection;
            if (!firstCamera.groundToImage(point, &first_projection) ||
                !secondCamera.groundToImage(point, &second_projection))
            {
                return false;
            }
            *values = {{firstObservation.sample - first_projection.image.sample,
                        firstObservation.line - first_projection.image.line,
                        secondObservation.sample - second_projection.image.sample,
                        secondObservation.line - second_projection.image.line}};
            return true;
        }

        bool solve3x3(double matrix[3][3], double rightHandSide[3], Vector3* solution)
        {
            double augmented[3][4]{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    augmented[row][column] = matrix[row][column];
                }
                augmented[row][3] = rightHandSide[row];
            }
            for (int pivot = 0; pivot < 3; ++pivot)
            {
                int best_row = pivot;
                for (int row = pivot + 1; row < 3; ++row)
                {
                    if (std::abs(augmented[row][pivot]) > std::abs(augmented[best_row][pivot]))
                    {
                        best_row = row;
                    }
                }
                if (std::abs(augmented[best_row][pivot]) < 1.0e-20)
                {
                    return false;
                }
                for (int column = pivot; column < 4; ++column)
                {
                    std::swap(augmented[pivot][column], augmented[best_row][column]);
                }
                for (int row = pivot + 1; row < 3; ++row)
                {
                    const double factor = augmented[row][pivot] / augmented[pivot][pivot];
                    for (int column = pivot; column < 4; ++column)
                    {
                        augmented[row][column] -= factor * augmented[pivot][column];
                    }
                }
            }
            for (int row = 2; row >= 0; --row)
            {
                double value = augmented[row][3];
                for (int column = row + 1; column < 3; ++column)
                {
                    value -= augmented[row][column] * (*solution)[column];
                }
                (*solution)[row] = value / augmented[row][row];
            }
            return true;
        }

        double rms(const std::array<double, 4>& values)
        {
            double sum = 0.0;
            for (double value : values)
            {
                sum += value * value;
            }
            return std::sqrt(sum / static_cast<double>(values.size()));
        }

    } // namespace

    bool intersectRpcObservations(const RpcCameraModel& firstCamera,
                                  const CameraImageCoordinate& firstObservation,
                                  const RpcCameraModel& secondCamera,
                                  const CameraImageCoordinate& secondObservation,
                                  RpcStereoIntersectionResult* result,
                                  std::string* errorMessage)
    {
        return intersectRpcObservations(firstCamera,
                                        firstObservation,
                                        secondCamera,
                                        secondObservation,
                                        result,
                                        RpcStereoIntersectionOptions{},
                                        errorMessage);
    }

    bool intersectRpcObservations(const RpcCameraModel& firstCamera,
                                  const CameraImageCoordinate& firstObservation,
                                  const RpcCameraModel& secondCamera,
                                  const CameraImageCoordinate& secondObservation,
                                  RpcStereoIntersectionResult* result,
                                  const RpcStereoIntersectionOptions& options,
                                  std::string* errorMessage)
    {
        if (!result || !firstCamera.isValid() || !secondCamera.isValid() || options.pixelTolerance <= 0.0 ||
            options.positionToleranceMeters <= 0.0 || options.maximumIterations <= 0)
        {
            if (errorMessage)
            {
                *errorMessage = "RPC intersection requires valid cameras, output and positive tolerances";
            }
            return false;
        }

        Vector3 point;
        if (!initialPoint(firstCamera, firstObservation, secondCamera, secondObservation, &point))
        {
            if (errorMessage)
            {
                *errorMessage = "RPC observation rays are invalid or nearly parallel";
            }
            return false;
        }

        std::array<double, 4> current_residuals;
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            if (!residuals(firstCamera, firstObservation, secondCamera, secondObservation, point, &current_residuals))
            {
                if (errorMessage)
                {
                    *errorMessage = "RPC projection failed during stereo intersection";
                }
                return false;
            }
            const double current_rms = rms(current_residuals);
            result->iterations = iteration + 1;
            if (current_rms <= options.pixelTolerance)
            {
                result->ecefMeters = point;
                result->reprojectionRmsPixels = current_rms;
                if (!RpcCameraModel::ecefToGeodetic(point, &result->geodetic))
                {
                    return false;
                }
                if (errorMessage)
                {
                    errorMessage->clear();
                }
                return true;
            }

            constexpr double derivative_step_meters = 0.5;
            double jacobian[4][3]{};
            for (int axis = 0; axis < 3; ++axis)
            {
                Vector3 plus = point;
                Vector3 minus = point;
                plus[axis] += derivative_step_meters;
                minus[axis] -= derivative_step_meters;
                std::array<double, 4> plus_residuals;
                std::array<double, 4> minus_residuals;
                if (!residuals(firstCamera, firstObservation, secondCamera, secondObservation, plus, &plus_residuals) ||
                    !residuals(firstCamera, firstObservation, secondCamera, secondObservation, minus, &minus_residuals))
                {
                    return false;
                }
                for (int row = 0; row < 4; ++row)
                {
                    jacobian[row][axis] = (plus_residuals[row] - minus_residuals[row]) / (2.0 * derivative_step_meters);
                }
            }

            double normal[3][3]{};
            double right_hand_side[3]{};
            for (int row = 0; row < 4; ++row)
            {
                for (int first_axis = 0; first_axis < 3; ++first_axis)
                {
                    right_hand_side[first_axis] -= jacobian[row][first_axis] * current_residuals[row];
                    for (int second_axis = 0; second_axis < 3; ++second_axis)
                    {
                        normal[first_axis][second_axis] += jacobian[row][first_axis] * jacobian[row][second_axis];
                    }
                }
            }
            const double diagonal_scale = std::max({normal[0][0], normal[1][1], normal[2][2], 1.0e-20});
            for (int axis = 0; axis < 3; ++axis)
            {
                normal[axis][axis] += diagonal_scale * 1.0e-10;
            }
            Vector3 update{};
            if (!solve3x3(normal, right_hand_side, &update))
            {
                if (errorMessage)
                {
                    *errorMessage = "RPC stereo normal matrix is singular";
                }
                return false;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                point[axis] += update[axis];
            }
            if (std::hypot(update[0], std::hypot(update[1], update[2])) <= options.positionToleranceMeters)
            {
                break;
            }
        }

        if (!residuals(firstCamera, firstObservation, secondCamera, secondObservation, point, &current_residuals))
        {
            return false;
        }
        result->ecefMeters = point;
        result->reprojectionRmsPixels = rms(current_residuals);
        RpcCameraModel::ecefToGeodetic(point, &result->geodetic);
        if (result->reprojectionRmsPixels <= options.pixelTolerance)
        {
            return true;
        }
        if (errorMessage)
        {
            *errorMessage = "RPC stereo intersection did not converge to the requested pixel tolerance";
        }
        return false;
    }

} // namespace xjw
