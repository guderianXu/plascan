#include "RpcIntersectionService.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw::camera_models::rpc
{
    namespace
    {

        double dot(const EcefCoordinate& first, const EcefCoordinate& second)
        {
            return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
        }

        EcefCoordinate subtract(const EcefCoordinate& first, const EcefCoordinate& second)
        {
            return {{first[0] - second[0], first[1] - second[1], first[2] - second[2]}};
        }

        EcefCoordinate pointOnRay(const RpcRay& ray, double distance)
        {
            return {{ray.origin[0] + distance * ray.direction[0],
                     ray.origin[1] + distance * ray.direction[1],
                     ray.origin[2] + distance * ray.direction[2]}};
        }

        bool initialPoint(const RpcInstance& first,
                          const ImagePoint& firstImage,
                          const RpcInstance& second,
                          const ImagePoint& secondImage,
                          EcefCoordinate* point)
        {
            RpcRay firstRay;
            RpcRay secondRay;
            if (!RpcProjection::ray(first, firstImage, &firstRay) ||
                !RpcProjection::ray(second, secondImage, &secondRay))
            {
                return false;
            }
            const EcefCoordinate betweenOrigins = subtract(firstRay.origin, secondRay.origin);
            const double directionDot = dot(firstRay.direction, secondRay.direction);
            const double firstProjection = dot(firstRay.direction, betweenOrigins);
            const double secondProjection = dot(secondRay.direction, betweenOrigins);
            const double denominator = 1.0 - directionDot * directionDot;
            if (std::abs(denominator) < 1.0e-12)
            {
                return false;
            }
            const double firstDistance = (directionDot * secondProjection - firstProjection) / denominator;
            const double secondDistance = (secondProjection - directionDot * firstProjection) / denominator;
            const EcefCoordinate firstPoint = pointOnRay(firstRay, firstDistance);
            const EcefCoordinate secondPoint = pointOnRay(secondRay, secondDistance);
            *point = {{0.5 * (firstPoint[0] + secondPoint[0]),
                       0.5 * (firstPoint[1] + secondPoint[1]),
                       0.5 * (firstPoint[2] + secondPoint[2])}};
            return true;
        }

        bool residuals(const RpcInstance& first,
                       const ImagePoint& firstImage,
                       const RpcInstance& second,
                       const ImagePoint& secondImage,
                       const EcefCoordinate& point,
                       std::array<double, 4>* values)
        {
            ImagePoint firstProjection;
            ImagePoint secondProjection;
            if (!RpcProjection::groundToImageEcef(first, point, &firstProjection) ||
                !RpcProjection::groundToImageEcef(second, point, &secondProjection))
            {
                return false;
            }
            *values = {{firstImage.sample - firstProjection.sample,
                        firstImage.line - firstProjection.line,
                        secondImage.sample - secondProjection.sample,
                        secondImage.line - secondProjection.line}};
            return true;
        }

        bool solve3x3(double matrix[3][3], double rightHandSide[3], EcefCoordinate* solution)
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
                int bestRow = pivot;
                for (int row = pivot + 1; row < 3; ++row)
                {
                    if (std::abs(augmented[row][pivot]) > std::abs(augmented[bestRow][pivot]))
                    {
                        bestRow = row;
                    }
                }
                if (std::abs(augmented[bestRow][pivot]) < 1.0e-20)
                {
                    return false;
                }
                for (int column = pivot; column < 4; ++column)
                {
                    std::swap(augmented[pivot][column], augmented[bestRow][column]);
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

        void setError(std::string* error, const char* message)
        {
            if (error)
            {
                *error = message;
            }
        }

    } // namespace

    bool RpcIntersectionService::intersect(const RpcInstance& first,
                                           const ImagePoint& firstImage,
                                           const RpcInstance& second,
                                           const ImagePoint& secondImage,
                                           RpcIntersectionResult* result,
                                           std::string* error)
    {
        return intersect(first, firstImage, second, secondImage, result, RpcIntersectionOptions{}, error);
    }

    bool RpcIntersectionService::intersect(const RpcInstance& first,
                                           const ImagePoint& firstImage,
                                           const RpcInstance& second,
                                           const ImagePoint& secondImage,
                                           RpcIntersectionResult* result,
                                           const RpcIntersectionOptions& options,
                                           std::string* error)
    {
        if (!result || options.pixelTolerance <= 0.0 || options.positionToleranceMeters <= 0.0 ||
            options.maximumIterations <= 0)
        {
            setError(error, "RPC intersection requires an output and positive tolerances");
            return false;
        }

        EcefCoordinate point;
        if (!initialPoint(first, firstImage, second, secondImage, &point))
        {
            setError(error, "RPC observation rays are invalid or nearly parallel");
            return false;
        }

        std::array<double, 4> currentResiduals;
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            if (!residuals(first, firstImage, second, secondImage, point, &currentResiduals))
            {
                setError(error, "RPC projection failed during stereo intersection");
                return false;
            }
            const double currentRms = rms(currentResiduals);
            result->iterations = iteration + 1;
            if (currentRms <= options.pixelTolerance)
            {
                result->ecefMeters = point;
                result->reprojectionRmsPixels = currentRms;
                if (!RpcProjection::ecefToGeodetic(point, &result->geodetic))
                {
                    return false;
                }
                if (error)
                {
                    error->clear();
                }
                return true;
            }

            constexpr double derivativeStepMeters = 0.5;
            double jacobian[4][3]{};
            for (int axis = 0; axis < 3; ++axis)
            {
                EcefCoordinate plus = point;
                EcefCoordinate minus = point;
                plus[axis] += derivativeStepMeters;
                minus[axis] -= derivativeStepMeters;
                std::array<double, 4> plusResiduals;
                std::array<double, 4> minusResiduals;
                if (!residuals(first, firstImage, second, secondImage, plus, &plusResiduals) ||
                    !residuals(first, firstImage, second, secondImage, minus, &minusResiduals))
                {
                    return false;
                }
                for (int row = 0; row < 4; ++row)
                {
                    jacobian[row][axis] =
                        (plusResiduals[row] - minusResiduals[row]) / (2.0 * derivativeStepMeters);
                }
            }

            double normal[3][3]{};
            double rightHandSide[3]{};
            for (int row = 0; row < 4; ++row)
            {
                for (int firstAxis = 0; firstAxis < 3; ++firstAxis)
                {
                    rightHandSide[firstAxis] -= jacobian[row][firstAxis] * currentResiduals[row];
                    for (int secondAxis = 0; secondAxis < 3; ++secondAxis)
                    {
                        normal[firstAxis][secondAxis] += jacobian[row][firstAxis] * jacobian[row][secondAxis];
                    }
                }
            }
            const double diagonalScale = std::max({normal[0][0], normal[1][1], normal[2][2], 1.0e-20});
            for (int axis = 0; axis < 3; ++axis)
            {
                normal[axis][axis] += diagonalScale * 1.0e-10;
            }
            EcefCoordinate update{};
            if (!solve3x3(normal, rightHandSide, &update))
            {
                setError(error, "RPC stereo normal matrix is singular");
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

        if (!residuals(first, firstImage, second, secondImage, point, &currentResiduals))
        {
            return false;
        }
        result->ecefMeters = point;
        result->reprojectionRmsPixels = rms(currentResiduals);
        RpcProjection::ecefToGeodetic(point, &result->geodetic);
        if (result->reprojectionRmsPixels <= options.pixelTolerance)
        {
            if (error)
            {
                error->clear();
            }
            return true;
        }
        setError(error, "RPC stereo intersection did not converge to the requested pixel tolerance");
        return false;
    }

} // namespace xjw::camera_models::rpc
