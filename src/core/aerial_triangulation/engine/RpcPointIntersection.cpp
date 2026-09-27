#include "engine/RpcEngineInternals.h"
#include <placamera/rpc_adjustment.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::aerial_triangulation::engine
{
    namespace
    {
        bool solve3x3(double matrix[3][3], double values[3], std::array<double, 3>* solution)
        {
            double augmented[3][4]{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    augmented[row][column] = matrix[row][column];
                }
                augmented[row][3] = values[row];
            }
            for (int column = 0; column < 3; ++column)
            {
                int pivot = column;
                for (int row = column + 1; row < 3; ++row)
                {
                    if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                    {
                        pivot = row;
                    }
                }
                if (!std::isfinite(augmented[pivot][column]) || std::abs(augmented[pivot][column]) < 1.0e-20)
                {
                    return false;
                }
                for (int entry = column; entry < 4; ++entry)
                {
                    std::swap(augmented[column][entry], augmented[pivot][entry]);
                }
                for (int row = column + 1; row < 3; ++row)
                {
                    const double factor = augmented[row][column] / augmented[column][column];
                    for (int entry = column; entry < 4; ++entry)
                    {
                        augmented[row][entry] -= factor * augmented[column][entry];
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
            return std::isfinite((*solution)[0]) && std::isfinite((*solution)[1]) && std::isfinite((*solution)[2]);
        }

        bool rpcResiduals(const TiePointGraph& graph,
                          const std::map<ImageId, RpcCamera>& cameras,
                          const std::vector<RpcObservation>& observations,
                          const RpcCartesianCoordinate& ecef,
                          std::vector<double>* residuals)
        {
            residuals->clear();
            residuals->reserve(observations.size() * 2);
            for (const RpcObservation& observation : observations)
            {
                const auto camera = cameras.find(observation.imageId);
                if (camera == cameras.cend() || !camera->second)
                {
                    return false;
                }
                const auto projected = camera->second->groundToImage(
                    placamera::GroundCoordinate{camera->second->groundFrame(), ecef});
                if (!projected)
                {
                    return false;
                }
                const RpcImagePoint measured = rpcImageCoordinate(graph, observation);
                residuals->push_back(projected.value().image.sample - measured.sample);
                residuals->push_back(projected.value().image.line - measured.line);
            }
            return true;
        }

        double residualRms(const std::vector<double>& residuals)
        {
            if (residuals.empty())
            {
                return std::numeric_limits<double>::infinity();
            }
            double squaredSum = 0.0;
            for (double residual : residuals)
            {
                squaredSum += residual * residual;
            }
            return std::sqrt(squaredSum / residuals.size());
        }

        bool refineRpcPoint(const TiePointGraph& graph,
                            const std::map<ImageId, RpcCamera>& cameras,
                            const std::vector<RpcObservation>& observations,
                            RpcCartesianCoordinate* ecef)
        {
            std::vector<double> currentResiduals;
            if (!rpcResiduals(graph, cameras, observations, *ecef, &currentResiduals))
            {
                return false;
            }
            double currentRms = residualRms(currentResiduals);
            constexpr double derivativeStepMeters = 0.5;
            for (int iteration = 0; iteration < 15; ++iteration)
            {
                std::vector<std::array<double, 3>> jacobian(currentResiduals.size());
                for (int axis = 0; axis < 3; ++axis)
                {
                    RpcCartesianCoordinate plus = *ecef;
                    RpcCartesianCoordinate minus = *ecef;
                    plus[axis] += derivativeStepMeters;
                    minus[axis] -= derivativeStepMeters;
                    std::vector<double> plusResiduals;
                    std::vector<double> minusResiduals;
                    if (!rpcResiduals(graph, cameras, observations, plus, &plusResiduals) ||
                        !rpcResiduals(graph, cameras, observations, minus, &minusResiduals))
                    {
                        return false;
                    }
                    for (std::size_t row = 0; row < currentResiduals.size(); ++row)
                    {
                        jacobian[row][axis] = (plusResiduals[row] - minusResiduals[row]) / (2.0 * derivativeStepMeters);
                    }
                }

                double normal[3][3]{};
                double rightHandSide[3]{};
                for (std::size_t row = 0; row < currentResiduals.size(); ++row)
                {
                    const double residualMagnitude = std::abs(currentResiduals[row]);
                    const double weight = residualMagnitude <= 2.0 ? 1.0 : 2.0 / residualMagnitude;
                    for (int firstAxis = 0; firstAxis < 3; ++firstAxis)
                    {
                        rightHandSide[firstAxis] -= weight * jacobian[row][firstAxis] * currentResiduals[row];
                        for (int secondAxis = 0; secondAxis < 3; ++secondAxis)
                        {
                            normal[firstAxis][secondAxis] +=
                                weight * jacobian[row][firstAxis] * jacobian[row][secondAxis];
                        }
                    }
                }
                const double diagonalScale = std::max({normal[0][0], normal[1][1], normal[2][2], 1.0e-20});
                for (int axis = 0; axis < 3; ++axis)
                {
                    normal[axis][axis] += diagonalScale * 1.0e-8;
                }
                std::array<double, 3> update{};
                if (!solve3x3(normal, rightHandSide, &update))
                {
                    break;
                }
                const double updateLength = std::hypot(update[0], std::hypot(update[1], update[2]));
                const double updateScale = updateLength > 500.0 ? 500.0 / updateLength : 1.0;
                RpcCartesianCoordinate candidate{(*ecef)[0] + updateScale * update[0],
                                                 (*ecef)[1] + updateScale * update[1],
                                                 (*ecef)[2] + updateScale * update[2]};
                std::vector<double> candidateResiduals;
                if (!rpcResiduals(graph, cameras, observations, candidate, &candidateResiduals))
                {
                    break;
                }
                const double candidateRms = residualRms(candidateResiduals);
                if (candidateRms > currentRms + 1.0e-12)
                {
                    break;
                }
                *ecef = candidate;
                currentResiduals = std::move(candidateResiduals);
                currentRms = candidateRms;
                if (updateLength * updateScale <= 1.0e-3)
                {
                    break;
                }
            }
            return true;
        }

    } // namespace
    namespace detail
    {
        bool intersectRpcTrack(const TiePointGraph& graph,
                               const std::map<ImageId, RpcCamera>& cameras,
                               const std::vector<RpcObservation>& observations,
                               double maximumRmsPixels,
                               RpcPoint* point,
                               std::map<ImageId, CameraResidualAccumulator>* cameraResiduals)
        {
            placamera::RpcIntersectionOptions options;
            options.pixelTolerance = 1.0e-4;
            options.positionToleranceMeters = 1.0e-3;
            options.maximumIterations = 40;

            std::optional<placamera::RpcIntersectionResult> best;
            bool hasBest = false;
            for (std::size_t first = 0; first + 1 < observations.size(); ++first)
            {
                for (std::size_t second = first + 1; second < observations.size(); ++second)
                {
                    const RpcObservation& firstRpcObservation = observations[first];
                    const RpcObservation& secondRpcObservation = observations[second];
                    const auto firstCamera = cameras.find(firstRpcObservation.imageId);
                    const auto secondCamera = cameras.find(secondRpcObservation.imageId);
                    if (firstCamera == cameras.cend() || secondCamera == cameras.cend() || !firstCamera->second ||
                        !secondCamera->second)
                    {
                        continue;
                    }
                    const auto candidate = placamera::intersectRpc(
                        *firstCamera->second,
                        placamera::ImageCoordinate{rpcImageCoordinate(graph, firstRpcObservation).sample,
                                                   rpcImageCoordinate(graph, firstRpcObservation).line},
                        *secondCamera->second,
                        placamera::ImageCoordinate{rpcImageCoordinate(graph, secondRpcObservation).sample,
                                                   rpcImageCoordinate(graph, secondRpcObservation).line},
                        options);
                    if (!candidate || !std::isfinite(candidate.value().reprojectionRmsPixels))
                    {
                        continue;
                    }
                    if (!hasBest || candidate.value().reprojectionRmsPixels < best->reprojectionRmsPixels)
                    {
                        best = candidate.value();
                        hasBest = true;
                    }
                }
            }
            if (!hasBest)
            {
                return false;
            }

            RpcCartesianCoordinate ecef = best->cartesian.position;
            if (!refineRpcPoint(graph, cameras, observations, &ecef))
            {
                return false;
            }
            const auto firstCamera = cameras.find(observations.front().imageId);
            if (firstCamera == cameras.cend() || !firstCamera->second)
            {
                return false;
            }
            const auto geodetic = placamera::cartesianToGeodetic(ecef, firstCamera->second->rpcDefinition().ellipsoid());
            if (!geodetic)
            {
                return false;
            }
            const RpcGeodeticCoordinate geodeticArray{
                geodetic.value().longitudeDegrees, geodetic.value().latitudeDegrees, geodetic.value().heightMeters};

            double squaredErrorSum = 0.0;
            double maximumResidual = 0.0;
            for (const RpcObservation& observation : observations)
            {
                const auto camera = cameras.find(observation.imageId);
                if (camera == cameras.cend() || !camera->second)
                {
                    return false;
                }
                const auto projection = camera->second->groundToImage(
                    placamera::GroundCoordinate{camera->second->groundFrame(), ecef});
                if (!projection)
                {
                    return false;
                }
                const RpcImagePoint measured = rpcImageCoordinate(graph, observation);
                const double residual =
                    std::hypot(projection.value().image.sample - measured.sample,
                               projection.value().image.line - measured.line);
                if (!std::isfinite(residual))
                {
                    return false;
                }
                squaredErrorSum += residual * residual;
                maximumResidual = std::max(maximumResidual, residual);
            }
            const double rms = std::sqrt(squaredErrorSum / observations.size());
            if (!std::isfinite(rms) || rms > maximumRmsPixels || maximumResidual > maximumRmsPixels * 2.0 ||
                !std::isfinite(geodeticArray[0]) || !std::isfinite(geodeticArray[1]) ||
                !std::isfinite(geodeticArray[2]) || std::abs(geodeticArray[0]) > 180.0 ||
                std::abs(geodeticArray[1]) > 90.0)
            {
                return false;
            }

            point->ecef = ecef;
            point->geodetic = geodeticArray;
            point->observations = observations;
            point->rmsPixels = rms;
            point->maximumResidualPixels = maximumResidual;

            if (cameraResiduals)
            {
                for (const RpcObservation& observation : observations)
                {
                    const auto projection = cameras.at(observation.imageId)->groundToImage(
                        placamera::GroundCoordinate{cameras.at(observation.imageId)->groundFrame(), ecef});
                    if (!projection)
                    {
                        return false;
                    }
                    const RpcImagePoint measured = rpcImageCoordinate(graph, observation);
                    const double residual =
                        std::hypot(projection.value().image.sample - measured.sample,
                                   projection.value().image.line - measured.line);
                    CameraResidualAccumulator& accumulator = (*cameraResiduals)[observation.imageId];
                    ++accumulator.observationCount;
                    accumulator.squaredErrorSum += residual * residual;
                    accumulator.maximumResidual = std::max(accumulator.maximumResidual, residual);
                }
            }
            return true;
        }

    } // namespace detail
} // namespace xjw::aerial_triangulation::engine
