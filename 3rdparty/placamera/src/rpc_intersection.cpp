#include "placamera/rpc_adjustment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace placamera
{

    namespace
    {

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        double dot(const Vector3& first, const Vector3& second) noexcept
        {
            return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
        }

        Vector3 subtract(const Vector3& first, const Vector3& second) noexcept
        {
            return {{first[0] - second[0], first[1] - second[1], first[2] - second[2]}};
        }

        Vector3 pointOnRay(const ImagingLocus& ray, double distance) noexcept
        {
            return {{ray.origin.position[0] + distance * ray.direction[0],
                     ray.origin.position[1] + distance * ray.direction[1],
                     ray.origin.position[2] + distance * ray.direction[2]}};
        }

        EvaluationResult<Vector3> initialIntersection(const RpcModel& first,
                                                      const ImageCoordinate& firstImage,
                                                      const RpcModel& second,
                                                      const ImageCoordinate& secondImage,
                                                      const EvaluationOptions& evaluationOptions)
        {
            const auto first_ray = first.imageToImagingLocus(firstImage, evaluationOptions);
            const auto second_ray = second.imageToImagingLocus(secondImage, evaluationOptions);
            if (!first_ray || !second_ray)
            {
                return EvaluationResult<Vector3>::failure(CameraErrorCode::NonConvergence,
                                                          "RPC observation rays could not be evaluated");
            }
            const Vector3 between_origins =
                subtract(first_ray.value().origin.position, second_ray.value().origin.position);
            const double direction_dot = dot(first_ray.value().direction, second_ray.value().direction);
            const double first_projection = dot(first_ray.value().direction, between_origins);
            const double second_projection = dot(second_ray.value().direction, between_origins);
            const double denominator = 1.0 - direction_dot * direction_dot;
            if (std::abs(denominator) < 1.0e-12)
            {
                return EvaluationResult<Vector3>::failure(CameraErrorCode::NonConvergence,
                                                          "RPC observation rays are nearly parallel");
            }
            const double first_distance = (direction_dot * second_projection - first_projection) / denominator;
            const double second_distance = (second_projection - direction_dot * first_projection) / denominator;
            const Vector3 first_point = pointOnRay(first_ray.value(), first_distance);
            const Vector3 second_point = pointOnRay(second_ray.value(), second_distance);
            return EvaluationResult<Vector3>::success({{0.5 * (first_point[0] + second_point[0]),
                                                        0.5 * (first_point[1] + second_point[1]),
                                                        0.5 * (first_point[2] + second_point[2])}});
        }

        EvaluationResult<std::array<double, 4>> rpcResiduals(const RpcModel& first,
                                                             const ImageCoordinate& firstImage,
                                                             const RpcModel& second,
                                                             const ImageCoordinate& secondImage,
                                                             const Vector3& point,
                                                             const EvaluationOptions& evaluationOptions)
        {
            const GroundCoordinate ground{first.groundFrame(), point};
            const auto first_projection = first.groundToImage(ground, evaluationOptions);
            const auto second_projection = second.groundToImage(ground, evaluationOptions);
            if (!first_projection || !second_projection)
            {
                return EvaluationResult<std::array<double, 4>>::failure(
                    CameraErrorCode::OutsideModelDomain, "RPC projection failed during stereo intersection");
            }
            return EvaluationResult<std::array<double, 4>>::success(
                {{firstImage.sample - first_projection.value().image.sample,
                  firstImage.line - first_projection.value().image.line,
                  secondImage.sample - second_projection.value().image.sample,
                  secondImage.line - second_projection.value().image.line}});
        }

        bool solve3x3(double matrix[3][3], double rightHandSide[3], Vector3* solution) noexcept
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

        double residualRms(const std::array<double, 4>& values) noexcept
        {
            double sum = 0.0;
            for (const double value : values)
            {
                sum += value * value;
            }
            return std::sqrt(sum / static_cast<double>(values.size()));
        }

    } // namespace

    EvaluationResult<RpcIntersectionResult> intersectRpc(const RpcModel& first,
                                                         const ImageCoordinate& firstImage,
                                                         const RpcModel& second,
                                                         const ImageCoordinate& secondImage,
                                                         const RpcIntersectionOptions& options)
    {
        if (first.groundFrame() != second.groundFrame())
        {
            return EvaluationResult<RpcIntersectionResult>::failure(
                CameraErrorCode::FrameMismatch, "RPC stereo models must use the same Cartesian ground frame");
        }
        const ReferenceEllipsoid& first_ellipsoid = first.rpcDefinition().ellipsoid();
        const ReferenceEllipsoid& second_ellipsoid = second.rpcDefinition().ellipsoid();
        if (first_ellipsoid.semiMajorAxisMeters != second_ellipsoid.semiMajorAxisMeters ||
            first_ellipsoid.inverseFlattening != second_ellipsoid.inverseFlattening)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(
                CameraErrorCode::InvalidArgument, "RPC stereo models must use the same reference ellipsoid");
        }
        if (!finite(options.pixelTolerance) || options.pixelTolerance <= 0.0 ||
            !finite(options.positionToleranceMeters) || options.positionToleranceMeters <= 0.0 ||
            !finite(options.derivativeStepMeters) || options.derivativeStepMeters <= 0.0 ||
            options.maximumIterations <= 0)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(
                CameraErrorCode::InvalidArgument, "RPC intersection tolerances and iteration count must be positive");
        }

        EvaluationOptions evaluation_options;
        evaluation_options.desiredPrecisionPixels = std::min(options.pixelTolerance, 1.0e-7);
        evaluation_options.maximumIterations = std::max(options.maximumIterations, 30);
        const auto initial = initialIntersection(first, firstImage, second, secondImage, evaluation_options);
        if (!initial)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(initial.errorCode(), initial.message());
        }

        Vector3 point = initial.value();
        std::array<double, 4> current_residuals{};
        int completed_iterations = 0;
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            const auto residuals = rpcResiduals(first, firstImage, second, secondImage, point, evaluation_options);
            if (!residuals)
            {
                return EvaluationResult<RpcIntersectionResult>::failure(residuals.errorCode(), residuals.message());
            }
            current_residuals = residuals.value();
            completed_iterations = iteration + 1;
            const double current_rms = residualRms(current_residuals);
            if (current_rms <= options.pixelTolerance)
            {
                const auto geodetic = cartesianToGeodetic(point, first_ellipsoid);
                if (!geodetic)
                {
                    return EvaluationResult<RpcIntersectionResult>::failure(geodetic.errorCode(), geodetic.message());
                }
                RpcIntersectionResult result{
                    {first.groundFrame(), point}, geodetic.value(), current_rms, completed_iterations};
                return EvaluationResult<RpcIntersectionResult>::success(result, current_rms);
            }

            double jacobian[4][3]{};
            for (int axis = 0; axis < 3; ++axis)
            {
                Vector3 plus = point;
                Vector3 minus = point;
                plus[axis] += options.derivativeStepMeters;
                minus[axis] -= options.derivativeStepMeters;
                const auto plus_residuals =
                    rpcResiduals(first, firstImage, second, secondImage, plus, evaluation_options);
                const auto minus_residuals =
                    rpcResiduals(first, firstImage, second, secondImage, minus, evaluation_options);
                if (!plus_residuals || !minus_residuals)
                {
                    return EvaluationResult<RpcIntersectionResult>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "RPC stereo derivatives could not be evaluated around the current point");
                }
                for (int row = 0; row < 4; ++row)
                {
                    jacobian[row][axis] = (plus_residuals.value()[row] - minus_residuals.value()[row]) /
                                          (2.0 * options.derivativeStepMeters);
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
                return EvaluationResult<RpcIntersectionResult>::failure(CameraErrorCode::NonConvergence,
                                                                        "RPC stereo normal matrix is singular");
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

        const auto final_residuals = rpcResiduals(first, firstImage, second, secondImage, point, evaluation_options);
        if (!final_residuals)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(final_residuals.errorCode(),
                                                                    final_residuals.message());
        }
        const double final_rms = residualRms(final_residuals.value());
        if (final_rms > options.pixelTolerance)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(
                CameraErrorCode::NonConvergence, "RPC stereo intersection did not reach the requested pixel tolerance");
        }
        const auto geodetic = cartesianToGeodetic(point, first_ellipsoid);
        if (!geodetic)
        {
            return EvaluationResult<RpcIntersectionResult>::failure(geodetic.errorCode(), geodetic.message());
        }
        RpcIntersectionResult result{{first.groundFrame(), point}, geodetic.value(), final_rms, completed_iterations};
        return EvaluationResult<RpcIntersectionResult>::success(result, final_rms);
    }

} // namespace placamera
