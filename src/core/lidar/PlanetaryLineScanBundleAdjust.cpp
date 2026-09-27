#include "PlanetaryLineScanBundleAdjust.h"

#include <plabundle/linescan.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace xjw
{
    namespace lidar
    {
        namespace
        {

            using Vector3 = std::array<double, 3>;
            constexpr double kIsisToCsmPixelCenterOffset = 0.5;

            void setError(std::string* errorMessage, const std::string& message)
            {
                if (errorMessage)
                {
                    *errorMessage = message;
                }
            }

            double dot(const Vector3& left, const Vector3& right)
            {
                return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
            }

            Vector3 subtract(const Vector3& left, const Vector3& right)
            {
                return {{left[0] - right[0], left[1] - right[1], left[2] - right[2]}};
            }

            double norm(const Vector3& value)
            {
                return std::sqrt(dot(value, value));
            }

            bool covarianceToSqrtInformation(const std::array<double, 9>& covariance,
                                             std::array<double, 9>* sqrtInformation)
            {
                if (!sqrtInformation)
                {
                    return false;
                }
                const double scale =
                    std::max({1.0, std::abs(covariance[0]), std::abs(covariance[4]), std::abs(covariance[8])});
                const double tolerance = 1.0e-14 * scale;
                const double c00 = covariance[0];
                const double c10 = 0.5 * (covariance[3] + covariance[1]);
                const double c20 = 0.5 * (covariance[6] + covariance[2]);
                const double c11 = covariance[4];
                const double c21 = 0.5 * (covariance[7] + covariance[5]);
                const double c22 = covariance[8];
                if (!(c00 > tolerance))
                {
                    return false;
                }
                const double l00 = std::sqrt(c00);
                const double l10 = c10 / l00;
                const double l20 = c20 / l00;
                const double pivot11 = c11 - l10 * l10;
                if (!(pivot11 > tolerance))
                {
                    return false;
                }
                const double l11 = std::sqrt(pivot11);
                const double l21 = (c21 - l20 * l10) / l11;
                const double pivot22 = c22 - l20 * l20 - l21 * l21;
                if (!(pivot22 > tolerance))
                {
                    return false;
                }
                const double l22 = std::sqrt(pivot22);
                *sqrtInformation = {{
                    1.0 / l00,
                    0.0,
                    0.0,
                    -l10 / (l00 * l11),
                    1.0 / l11,
                    0.0,
                    (l10 * l21 - l20 * l11) / (l00 * l11 * l22),
                    -l21 / (l11 * l22),
                    1.0 / l22,
                }};
                return std::all_of(sqrtInformation->begin(),
                                   sqrtInformation->end(),
                                   [](double value) { return std::isfinite(value); });
            }

            struct MappedMeasure
            {
                int cameraIndex = -1;
                const IsisControlMeasure* measure = nullptr;
                placamera::ImagingLocus ray;
            };

            bool chooseTriangulationPair(const std::vector<MappedMeasure>& measures, int* firstIndex, int* secondIndex)
            {
                double bestScore = -1.0;
                for (std::size_t first = 0; first < measures.size(); ++first)
                {
                    for (std::size_t second = first + 1; second < measures.size(); ++second)
                    {
                        const double score =
                            1.0 - std::abs(dot(measures[first].ray.direction, measures[second].ray.direction));
                        if (score > bestScore)
                        {
                            bestScore = score;
                            *firstIndex = static_cast<int>(first);
                            *secondIndex = static_cast<int>(second);
                        }
                    }
                }
                return bestScore > 1.0e-12;
            }

            bool usedLaserTime(const PlanetaryLineScanBaCamera& camera,
                               const PlanetaryLaserShot& shot,
                               PlanetaryLaserLineScanTimeMode mode,
                               double* ephemerisTimeSeconds)
            {
                if (mode == PlanetaryLaserLineScanTimeMode::ShotEphemerisTime)
                {
                    *ephemerisTimeSeconds = shot.ephemerisTimeSeconds;
                    return true;
                }
                const auto measure = std::find_if(shot.imageMeasures.begin(),
                                                  shot.imageMeasures.end(),
                                                  [&camera](const PlanetaryLaserImageMeasure& candidate)
                                                  { return candidate.imageId == camera.serialNumber; });
                if (measure == shot.imageMeasures.end())
                {
                    return false;
                }
                const auto time = camera.instance->timeForLine(measure->linePixels - kIsisToCsmPixelCenterOffset);
                if (!time)
                {
                    return false;
                }
                *ephemerisTimeSeconds = time.value().seconds;
                return true;
            }

            bool centerAtTime(const placamera::LineScanModel& instance, double ephemerisTimeSeconds, Vector3* center)
            {
                if (!center || !std::isfinite(ephemerisTimeSeconds))
                {
                    return false;
                }
                const auto pose = instance.trajectory().poseAt(
                    placamera::TimeReference::create(placamera::TimeScale::Tdb, ephemerisTimeSeconds),
                    instance.groundFrame());
                if (!pose)
                {
                    return false;
                }
                *center = pose.value().center;
                return true;
            }

            bool isSupportedLineScanBackend(plabundle::Backend backend)
            {
                return backend == plabundle::Backend::Auto || backend == plabundle::Backend::PlaMatrixCpu ||
                       backend == plabundle::Backend::PlaMatrixCuda || backend == plabundle::Backend::PlaMatrixVulkan ||
                       backend == plabundle::Backend::PlaMatrixOpenCl;
            }
        } // namespace

        bool triangulatePlanetaryLineScanRays(const placamera::ImagingLocus& first,
                                              const placamera::ImagingLocus& second,
                                              std::array<double, 3>* pointBodyFixedMeters,
                                              double* raySeparationMeters)
        {
            if (!pointBodyFixedMeters)
            {
                return false;
            }
            if (first.origin.frame != second.origin.frame)
            {
                return false;
            }
            const Vector3 offset = subtract(first.origin.position, second.origin.position);
            const double a = dot(first.direction, first.direction);
            const double b = dot(first.direction, second.direction);
            const double c = dot(second.direction, second.direction);
            const double d = dot(first.direction, offset);
            const double e = dot(second.direction, offset);
            const double denominator = a * c - b * b;
            if (!(a > 0.0) || !(c > 0.0) || std::abs(denominator) < 1.0e-14)
            {
                return false;
            }
            const double firstDistance = (b * e - c * d) / denominator;
            const double secondDistance = (a * e - b * d) / denominator;
            if (!(firstDistance > 0.0) || !(secondDistance > 0.0))
            {
                return false;
            }
            Vector3 firstPoint{};
            Vector3 secondPoint{};
            for (int axis = 0; axis < 3; ++axis)
            {
                firstPoint[axis] = first.origin.position[axis] + firstDistance * first.direction[axis];
                secondPoint[axis] = second.origin.position[axis] + secondDistance * second.direction[axis];
                (*pointBodyFixedMeters)[axis] = 0.5 * (firstPoint[axis] + secondPoint[axis]);
            }
            if (raySeparationMeters)
            {
                *raySeparationMeters = norm(subtract(firstPoint, secondPoint));
            }
            return std::all_of(pointBodyFixedMeters->begin(),
                               pointBodyFixedMeters->end(),
                               [](double value) { return std::isfinite(value); });
        }

        const char* planetaryLaserLineScanTimeModeName(PlanetaryLaserLineScanTimeMode mode)
        {
            return mode == PlanetaryLaserLineScanTimeMode::ShotEphemerisTime ? "shot_et"
                                                                             : "isis_simultaneous_measure_line";
        }

        bool runPlanetaryLineScanBundleAdjust(const std::vector<PlanetaryLineScanBaCamera>& cameras,
                                              const IsisControlNetwork& controlNetwork,
                                              const PlanetaryLaserDataset* laserDataset,
                                              const PlanetaryLineScanBaOptions& options,
                                              PlanetaryLineScanBaResult* result,
                                              std::string* errorMessage)
        {
            if (!result)
            {
                setError(errorMessage, "output planetary line-scan BA result pointer is null");
                return false;
            }
            *result = PlanetaryLineScanBaResult{};
            result->requestedBackend = options.backend;
            result->laserConstraintsEnabled = options.enableLaserRangeConstraints;
            if (cameras.size() < 2 || !controlNetwork.validate(errorMessage))
            {
                setError(errorMessage,
                         cameras.size() < 2 ? "planetary line-scan BA requires at least two cameras"
                                            : (errorMessage ? *errorMessage : "invalid ISIS control network"));
                return false;
            }
            const auto finitePositive = [](double value) { return std::isfinite(value) && value > 0.0; };
            const auto finiteNonNegative = [](double value) { return std::isfinite(value) && value >= 0.0; };
            if (!finitePositive(options.imageSigmaPixels) || !finitePositive(options.cameraPositionSigmaMeters) ||
                !finitePositive(options.cameraAngleSigmaDegrees) || !finitePositive(options.laserRangeWeight) ||
                !finitePositive(options.finiteDifferencePointStepMeters) ||
                !finitePositive(options.finiteDifferencePositionStepMeters) ||
                !finitePositive(options.finiteDifferenceAngleStepRadians) ||
                !finitePositive(options.maximumCameraTranslationMeters) ||
                !finitePositive(options.maximumCameraAngleDegrees) ||
                !finiteNonNegative(options.imageHuberDeltaPixels) ||
                !finiteNonNegative(options.laserRangeHuberDeltaSigma) || options.maximumIterations <= 0 ||
                options.threadCount < 0 || options.plaMatrixDevice < 0 || options.minPlaMatrixCudaCameras <= 0 ||
                options.minPlaMatrixCudaObservations <= 0 || options.minPlaMatrixVulkanCameras <= 0 ||
                options.minPlaMatrixVulkanObservations <= 0 || options.minPlaMatrixOpenClCameras <= 0 ||
                options.minPlaMatrixOpenClObservations <= 0 || options.minPlaMatrixDenseCameras <= 0 ||
                options.minPlaMatrixCudaDenseObservations <= 0 || options.minPlaMatrixVulkanDenseObservations <= 0 ||
                options.minPlaMatrixOpenClDenseObservations <= 0 || !isSupportedLineScanBackend(options.backend))
            {
                setError(errorMessage, "planetary line-scan BA options contain invalid sigma or iteration values");
                return false;
            }

            std::unordered_map<std::string, int> cameraBySerial;
            plabundle::linescan::Problem workingSet;
            workingSet.cameraParameters.resize(cameras.size());
            workingSet.projection = [&cameras](const plabundle::linescan::ImageObservation& observation,
                                               const std::array<double, 6>& parameters,
                                               const std::array<double, 3>& point,
                                               std::array<double, 2>* residual)
            {
                const auto& camera = *cameras[observation.cameraIndex].instance;
                placamera::LineScanTrajectoryBias bias = camera.trajectoryBias();
                for (int axis = 0; axis < 3; ++axis)
                {
                    bias.translationMeters[axis] += parameters[axis];
                    bias.rotationVectorRadians[axis] += parameters[axis + 3];
                }
                const placamera::GroundCoordinate ground{camera.groundFrame(), point};
                const auto projection = camera.projectAtLine(ground, observation.linePixels, bias);
                if (!projection)
                {
                    return false;
                }
                (*residual)[0] = projection.value().projection.image.sample - observation.samplePixels;
                (*residual)[1] = projection.value().lineResidualPixels;
                return std::isfinite((*residual)[0]) && std::isfinite((*residual)[1]);
            };
            result->cameras.resize(cameras.size());
            for (std::size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
            {
                const auto& camera = cameras[cameraIndex];
                if (camera.serialNumber.empty() || !camera.instance ||
                    !cameraBySerial.emplace(camera.serialNumber, static_cast<int>(cameraIndex)).second)
                {
                    setError(errorMessage, "line-scan camera has an empty/duplicate serial or invalid ISD model");
                    return false;
                }
                if (camera.instance->groundFrame().value() != "MOON_ME" ||
                    (!controlNetwork.targetName.empty() && controlNetwork.targetName != "MOON"))
                {
                    setError(errorMessage, "line-scan ISD, control network, and solver must share MOON/MOON_ME");
                    return false;
                }
                result->cameras[cameraIndex].serialNumber = camera.serialNumber;
            }

            for (const IsisControlPoint& controlPoint : controlNetwork.points)
            {
                if (controlPoint.ignored)
                {
                    continue;
                }
                if (controlPoint.type != IsisControlPointType::Free)
                {
                    setError(errorMessage,
                             "P0 line-scan BA supports only Free ISIS control points; point " + controlPoint.id +
                                 " must not lose its ground prior silently");
                    return false;
                }
                std::vector<MappedMeasure> measures;
                for (const IsisControlMeasure& measure : controlPoint.measures)
                {
                    if (measure.ignored)
                    {
                        continue;
                    }
                    const auto camera = cameraBySerial.find(measure.serialNumber);
                    if (camera == cameraBySerial.end())
                    {
                        setError(errorMessage,
                                 "unmapped ISIS camera serial in control network: " + measure.serialNumber);
                        return false;
                    }
                    const double csmSample = measure.samplePixels - kIsisToCsmPixelCenterOffset;
                    const double csmLine = measure.linePixels - kIsisToCsmPixelCenterOffset;
                    const auto ray = cameras[camera->second].instance->imageToImagingLocus({csmSample, csmLine});
                    if (!ray)
                    {
                        setError(errorMessage,
                                 "failed to construct line-scan ray for control point " + controlPoint.id);
                        return false;
                    }
                    measures.push_back({camera->second, &measure, ray.value()});
                }
                int first = -1;
                int second = -1;
                if (measures.size() < 2 || !chooseTriangulationPair(measures, &first, &second))
                {
                    setError(errorMessage, "control point has insufficient intersecting rays: " + controlPoint.id);
                    return false;
                }
                Vector3 initialPoint{};
                double separation = 0.0;
                if (!triangulatePlanetaryLineScanRays(
                        measures[first].ray, measures[second].ray, &initialPoint, &separation))
                {
                    setError(errorMessage, "failed to triangulate control point " + controlPoint.id);
                    return false;
                }
                const int pointIndex = static_cast<int>(workingSet.tiePoints.size());
                workingSet.tiePoints.push_back(initialPoint);
                result->points.push_back(
                    {controlPoint.id, initialPoint, initialPoint, static_cast<int>(measures.size()), separation});
                for (const MappedMeasure& measure : measures)
                {
                    workingSet.imageObservations.push_back({measure.cameraIndex,
                                                            pointIndex,
                                                            measure.measure->samplePixels - kIsisToCsmPixelCenterOffset,
                                                            measure.measure->linePixels - kIsisToCsmPixelCenterOffset});
                }
            }

            if (workingSet.tiePoints.empty())
            {
                setError(errorMessage, "ISIS control network produced no usable tie points");
                return false;
            }

            if (laserDataset)
            {
                std::string validationError;
                if (!laserDataset->validate(&validationError) ||
                    laserDataset->sensorModel != PlanetaryLaserSensorModel::LineScan ||
                    laserDataset->rangeType != PlanetaryLaserRangeType::OneWay ||
                    laserDataset->reference.targetName != "MOON" ||
                    laserDataset->reference.bodyFixedFrame != "MOON_ME" ||
                    (!controlNetwork.targetName.empty() &&
                     controlNetwork.targetName != laserDataset->reference.targetName))
                {
                    setError(errorMessage, "invalid MOON_ME line-scan one-way laser dataset: " + validationError);
                    return false;
                }
                for (const PlanetaryLaserShot& shot : laserDataset->shots)
                {
                    if (norm(shot.leverArmSensorMeters) > 1.0e-12 || shot.simultaneousImageIds.size() != 1)
                    {
                        setError(errorMessage,
                                 "P0 line-scan BA requires zero LOLA lever arm and one simultaneous image per shot");
                        return false;
                    }
                    const std::string& serial = shot.simultaneousImageIds.front();
                    const auto mappedCamera = cameraBySerial.find(serial);
                    if (mappedCamera == cameraBySerial.end())
                    {
                        setError(errorMessage, "unmapped simultaneous laser image serial: " + serial);
                        return false;
                    }
                    double usedEt = 0.0;
                    const auto& camera = cameras[mappedCamera->second];
                    Vector3 center{};
                    if (!usedLaserTime(camera, shot, options.laserTimeMode, &usedEt) ||
                        !centerAtTime(*camera.instance, usedEt, &center))
                    {
                        setError(errorMessage, "failed to evaluate laser shot time/position: " + shot.id);
                        return false;
                    }
                    plabundle::linescan::LaserPoint laserPoint;
                    laserPoint.initialMeters = shot.pointBodyFixedMeters;
                    laserPoint.refinedMeters = shot.pointBodyFixedMeters;
                    laserPoint.mode = shot.pointMode == PlanetaryLaserPointMode::Constrained
                                          ? plabundle::linescan::LaserPointMode::Constrained
                                          : plabundle::linescan::LaserPointMode::Fixed;
                    if (shot.pointMode == PlanetaryLaserPointMode::Constrained &&
                        (!shot.pointCovarianceBodyFixedMetersSquared ||
                         !covarianceToSqrtInformation(*shot.pointCovarianceBodyFixedMetersSquared,
                                                      &laserPoint.sqrtInformation)))
                    {
                        setError(errorMessage, "invalid laser point covariance for shot " + shot.id);
                        return false;
                    }
                    if (options.enableLaserRangeConstraints && shot.pointMode == PlanetaryLaserPointMode::Free)
                    {
                        setError(errorMessage, "P0 line-scan BA requires fixed or constrained laser points");
                        return false;
                    }
                    const int laserPointIndex = static_cast<int>(workingSet.laserPoints.size());
                    workingSet.laserPoints.push_back(laserPoint);
                    workingSet.laserObservations.push_back({mappedCamera->second,
                                                            laserPointIndex,
                                                            center,
                                                            shot.observedRangeMeters,
                                                            shot.rangeSigmaMeters});
                    PlanetaryLineScanBaLaserShotResult shotResult;
                    shotResult.id = shot.id;
                    shotResult.simultaneousImageId = serial;
                    shotResult.shotEphemerisTimeSeconds = shot.ephemerisTimeSeconds;
                    shotResult.usedEphemerisTimeSeconds = usedEt;
                    shotResult.imageLineTimeMinusShotTimeSeconds = usedEt - shot.ephemerisTimeSeconds;
                    shotResult.observedRangeMeters = shot.observedRangeMeters;
                    shotResult.initialPointBodyFixedMeters = shot.pointBodyFixedMeters;
                    shotResult.refinedPointBodyFixedMeters = shot.pointBodyFixedMeters;
                    result->laserShots.push_back(std::move(shotResult));
                }
            }
            else if (options.enableLaserRangeConstraints)
            {
                setError(errorMessage, "laser constraints requested without a planetary laser dataset");
                return false;
            }

            result->controlPointCount = static_cast<int>(workingSet.tiePoints.size());
            result->imageObservationCount = static_cast<int>(workingSet.imageObservations.size());
            result->activeLaserRangeCount =
                options.enableLaserRangeConstraints ? static_cast<int>(workingSet.laserObservations.size()) : 0;
            plabundle::linescan::Options numericOptions;
            numericOptions.backend = options.backend;
            numericOptions.plaMatrixDevice = options.plaMatrixDevice;
            numericOptions.minPlaMatrixCudaCameras = options.minPlaMatrixCudaCameras;
            numericOptions.minPlaMatrixCudaObservations = options.minPlaMatrixCudaObservations;
            numericOptions.minPlaMatrixVulkanCameras = options.minPlaMatrixVulkanCameras;
            numericOptions.minPlaMatrixVulkanObservations = options.minPlaMatrixVulkanObservations;
            numericOptions.minPlaMatrixOpenClCameras = options.minPlaMatrixOpenClCameras;
            numericOptions.minPlaMatrixOpenClObservations = options.minPlaMatrixOpenClObservations;
            numericOptions.minPlaMatrixDenseCameras = options.minPlaMatrixDenseCameras;
            numericOptions.minPlaMatrixCudaDenseObservations = options.minPlaMatrixCudaDenseObservations;
            numericOptions.minPlaMatrixVulkanDenseObservations = options.minPlaMatrixVulkanDenseObservations;
            numericOptions.minPlaMatrixOpenClDenseObservations = options.minPlaMatrixOpenClDenseObservations;
            numericOptions.maxDenseSchurCameras = options.maxDenseSchurCameras;
            numericOptions.allowBackendFallback = options.allowBackendFallback;
            numericOptions.cancelFlag = options.cancelFlag;
            numericOptions.enableLaserRangeConstraints = options.enableLaserRangeConstraints;
            numericOptions.maximumIterations = options.maximumIterations;
            numericOptions.threadCount = options.threadCount;
            numericOptions.imageSigmaPixels = options.imageSigmaPixels;
            numericOptions.imageHuberDeltaPixels = options.imageHuberDeltaPixels;
            numericOptions.cameraPositionSigmaMeters = options.cameraPositionSigmaMeters;
            numericOptions.cameraAngleSigmaDegrees = options.cameraAngleSigmaDegrees;
            numericOptions.laserRangeWeight = options.laserRangeWeight;
            numericOptions.laserRangeHuberDeltaSigma = options.laserRangeHuberDeltaSigma;
            numericOptions.finiteDifferencePointStepMeters = options.finiteDifferencePointStepMeters;
            numericOptions.finiteDifferencePositionStepMeters = options.finiteDifferencePositionStepMeters;
            numericOptions.finiteDifferenceAngleStepRadians = options.finiteDifferenceAngleStepRadians;
            numericOptions.maximumCameraTranslationMeters = options.maximumCameraTranslationMeters;
            numericOptions.maximumCameraAngleDegrees = options.maximumCameraAngleDegrees;
            const plabundle::linescan::Result numericResult = plabundle::linescan::solve(workingSet, numericOptions);
            result->solutionUsable = numericResult.solutionUsable;
            result->converged = numericResult.converged;
            result->backendFallback = numericResult.backendFallback;
            result->usedGpu = numericResult.usedGpu;
            result->usedBackend = numericResult.usedBackend;
            result->terminationType = numericResult.terminationType;
            result->message = numericResult.message;
            result->backendMessage = numericResult.backendMessage;
            result->linearSolverName = numericResult.linearSolverName;
            result->deviceName = numericResult.deviceName;
            result->solverBriefReport = numericResult.solverBriefReport;
            result->iterations = numericResult.iterations;
            result->initialImageRmsPixels = numericResult.initialImageRmsPixels;
            result->refinedImageRmsPixels = numericResult.refinedImageRmsPixels;
            result->initialLaserRangeRmsMeters = numericResult.initialLaserRangeRmsMeters;
            result->refinedLaserRangeRmsMeters = numericResult.refinedLaserRangeRmsMeters;
            if (!numericResult.usable())
            {
                setError(errorMessage,
                         numericResult.message.empty() ? numericResult.backendMessage : numericResult.message);
                return false;
            }
            for (std::size_t index = 0; index < numericResult.refinedCameraParameters.size(); ++index)
            {
                const auto& parameters = numericResult.refinedCameraParameters[index];
                std::copy_n(parameters.begin(), 3, result->cameras[index].translationBodyFixedMeters.begin());
                std::copy_n(parameters.begin() + 3, 3, result->cameras[index].angleAxisBodyFixedRadians.begin());
            }
            for (std::size_t index = 0; index < numericResult.refinedTiePoints.size(); ++index)
            {
                result->points[index].refinedBodyFixedMeters = numericResult.refinedTiePoints[index];
            }
            for (std::size_t index = 0; index < numericResult.refinedLaserPoints.size(); ++index)
            {
                auto& shotResult = result->laserShots[index];
                shotResult.refinedPointBodyFixedMeters = numericResult.refinedLaserPoints[index];
                const auto& observation = workingSet.laserObservations[index];
                const auto rangeResidual = [&observation](const Vector3& point, const std::array<double, 6>& camera)
                {
                    Vector3 delta{{point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
                                   point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
                                   point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
                    return norm(delta) - observation.observedRangeMeters;
                };
                const std::array<double, 6> zero{};
                shotResult.initialComputedMinusObservedMeters =
                    rangeResidual(workingSet.laserPoints[index].initialMeters, zero);
                shotResult.refinedComputedMinusObservedMeters =
                    rangeResidual(numericResult.refinedLaserPoints[index],
                                  numericResult.refinedCameraParameters[observation.cameraIndex]);
            }
            result->success = true;
            return true;
        }

    } // namespace lidar
} // namespace xjw
