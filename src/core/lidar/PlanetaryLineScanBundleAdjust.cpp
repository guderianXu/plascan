#include "PlanetaryLineScanBundleAdjust.h"

#include "PlanetaryLineScanBundleAdjustInternal.h"

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

using Vector3 = PlanetaryLineScanCamera::Vector3;

constexpr double kIsisToCsmPixelCenterOffset = 0.5;

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

double dot(const Vector3 &left, const Vector3 &right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vector3 subtract(const Vector3 &left, const Vector3 &right)
{
    return {{left[0] - right[0], left[1] - right[1], left[2] - right[2]}};
}

double norm(const Vector3 &value)
{
    return std::sqrt(dot(value, value));
}

bool covarianceToSqrtInformation(const std::array<double, 9> &covariance,
                                 std::array<double, 9> *sqrtInformation)
{
    if (!sqrtInformation)
    {
        return false;
    }
    const double scale = std::max(
        {1.0, std::abs(covariance[0]), std::abs(covariance[4]),
         std::abs(covariance[8])});
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
        1.0 / l00, 0.0, 0.0,
        -l10 / (l00 * l11), 1.0 / l11, 0.0,
        (l10 * l21 - l20 * l11) / (l00 * l11 * l22),
        -l21 / (l11 * l22), 1.0 / l22,
    }};
    return std::all_of(sqrtInformation->begin(), sqrtInformation->end(), [](double value)
    {
        return std::isfinite(value);
    });
}

struct MappedMeasure
{
    int cameraIndex = -1;
    const IsisControlMeasure *measure = nullptr;
    PlanetaryLineScanCamera::ImagingRay ray;
};

bool chooseTriangulationPair(const std::vector<MappedMeasure> &measures,
                             int *firstIndex,
                             int *secondIndex)
{
    double bestScore = -1.0;
    for (std::size_t first = 0; first < measures.size(); ++first)
    {
        for (std::size_t second = first + 1; second < measures.size(); ++second)
        {
            const double score = 1.0 - std::abs(dot(
                measures[first].ray.directionBodyFixed,
                measures[second].ray.directionBodyFixed));
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

bool usedLaserTime(const PlanetaryLineScanBaCamera &camera,
                   const PlanetaryLaserShot &shot,
                   PlanetaryLaserLineScanTimeMode mode,
                   double *ephemerisTimeSeconds)
{
    if (mode == PlanetaryLaserLineScanTimeMode::ShotEphemerisTime)
    {
        *ephemerisTimeSeconds = shot.ephemerisTimeSeconds;
        return true;
    }
    const auto measure = std::find_if(
        shot.imageMeasures.begin(), shot.imageMeasures.end(),
        [&camera](const PlanetaryLaserImageMeasure &candidate)
        {
            return candidate.imageId == camera.serialNumber;
        });
    return measure != shot.imageMeasures.end() &&
        camera.model.absoluteEtForLine(
            measure->linePixels - kIsisToCsmPixelCenterOffset,
            PlanetaryLineScanCamera::PixelConvention::CsmPixelCenter,
            ephemerisTimeSeconds);
}

bool isSupportedLineScanBackend(BABackend backend)
{
    return backend == BABackend::Auto || backend == BABackend::PlaMatrixCpu || backend == BABackend::PlaMatrixCuda ||
           backend == BABackend::PlaMatrixOpenCl;
}

BABackend selectLineScanBackend(const PlanetaryLineScanBaOptions& options, int cameraCount, int observationCount)
{
    if (options.backend != BABackend::Auto)
    {
        return options.backend;
    }
    BAOptions auto_options;
    auto_options.minPlaMatrixCudaCameras = options.minPlaMatrixCudaCameras;
    auto_options.minPlaMatrixCudaObservations = options.minPlaMatrixCudaObservations;
    auto_options.minPlaMatrixOpenClCameras = options.minPlaMatrixOpenClCameras;
    auto_options.minPlaMatrixOpenClObservations = options.minPlaMatrixOpenClObservations;
    auto_options.minPlaMatrixDenseCameras = options.minPlaMatrixDenseCameras;
    auto_options.minPlaMatrixCudaDenseObservations = options.minPlaMatrixCudaDenseObservations;
    auto_options.minPlaMatrixOpenClDenseObservations = options.minPlaMatrixOpenClDenseObservations;
    BAProblemStats stats;
    stats.cameraCount = cameraCount;
    stats.observationCount = observationCount;
    if (BundleAdjust::autoBackendMeetsScaleThreshold(BABackend::PlaMatrixCuda, stats, auto_options) &&
        BundleAdjust::isBackendAvailable(BABackend::PlaMatrixCuda))
    {
        return BABackend::PlaMatrixCuda;
    }
    if (BundleAdjust::autoBackendMeetsScaleThreshold(BABackend::PlaMatrixOpenCl, stats, auto_options) &&
        BundleAdjust::isBackendAvailable(BABackend::PlaMatrixOpenCl))
    {
        return BABackend::PlaMatrixOpenCl;
    }
    return BABackend::PlaMatrixCpu;
}

} // namespace

bool triangulatePlanetaryLineScanRays(const PlanetaryLineScanCamera::ImagingRay& first,
                                      const PlanetaryLineScanCamera::ImagingRay& second,
                                      std::array<double, 3>* pointBodyFixedMeters,
                                      double* raySeparationMeters)
{
    if (!pointBodyFixedMeters)
    {
        return false;
    }
    const Vector3 offset = subtract(first.centerBodyFixedMeters,
                                    second.centerBodyFixedMeters);
    const double a = dot(first.directionBodyFixed, first.directionBodyFixed);
    const double b = dot(first.directionBodyFixed, second.directionBodyFixed);
    const double c = dot(second.directionBodyFixed, second.directionBodyFixed);
    const double d = dot(first.directionBodyFixed, offset);
    const double e = dot(second.directionBodyFixed, offset);
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
        firstPoint[axis] = first.centerBodyFixedMeters[axis] +
                           firstDistance * first.directionBodyFixed[axis];
        secondPoint[axis] = second.centerBodyFixedMeters[axis] +
                            secondDistance * second.directionBodyFixed[axis];
        (*pointBodyFixedMeters)[axis] = 0.5 * (firstPoint[axis] + secondPoint[axis]);
    }
    if (raySeparationMeters)
    {
        *raySeparationMeters = norm(subtract(firstPoint, secondPoint));
    }
    return std::all_of(pointBodyFixedMeters->begin(), pointBodyFixedMeters->end(), [](double value)
    {
        return std::isfinite(value);
    });
}

const char *planetaryLaserLineScanTimeModeName(PlanetaryLaserLineScanTimeMode mode)
{
    return mode == PlanetaryLaserLineScanTimeMode::ShotEphemerisTime
        ? "shot_et"
        : "isis_simultaneous_measure_line";
}

bool runPlanetaryLineScanBundleAdjust(
    const std::vector<PlanetaryLineScanBaCamera> &cameras,
    const IsisControlNetwork &controlNetwork,
    const PlanetaryLaserDataset *laserDataset,
    const PlanetaryLineScanBaOptions &options,
    PlanetaryLineScanBaResult *result,
    std::string *errorMessage)
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
        setError(errorMessage, cameras.size() < 2
            ? "planetary line-scan BA requires at least two cameras"
            : (errorMessage ? *errorMessage : "invalid ISIS control network"));
        return false;
    }
    const auto finitePositive = [](double value)
    {
        return std::isfinite(value) && value > 0.0;
    };
    const auto finiteNonNegative = [](double value) { return std::isfinite(value) && value >= 0.0; };
    if (!finitePositive(options.imageSigmaPixels) || !finitePositive(options.cameraPositionSigmaMeters) ||
        !finitePositive(options.cameraAngleSigmaDegrees) || !finitePositive(options.laserRangeWeight) ||
        !finitePositive(options.finiteDifferencePointStepMeters) ||
        !finitePositive(options.finiteDifferencePositionStepMeters) ||
        !finitePositive(options.finiteDifferenceAngleStepRadians) ||
        !finitePositive(options.maximumCameraTranslationMeters) || !finitePositive(options.maximumCameraAngleDegrees) ||
        !finiteNonNegative(options.imageHuberDeltaPixels) || !finiteNonNegative(options.laserRangeHuberDeltaSigma) ||
        options.maximumIterations <= 0 || options.threadCount < 0 || options.plaMatrixDevice < 0 ||
        options.minPlaMatrixCudaCameras <= 0 || options.minPlaMatrixCudaObservations <= 0 ||
        options.minPlaMatrixOpenClCameras <= 0 || options.minPlaMatrixOpenClObservations <= 0 ||
        options.minPlaMatrixDenseCameras <= 0 || options.minPlaMatrixCudaDenseObservations <= 0 ||
        options.minPlaMatrixOpenClDenseObservations <= 0 || !isSupportedLineScanBackend(options.backend))
    {
        setError(errorMessage, "planetary line-scan BA options contain invalid sigma or iteration values");
        return false;
    }

    std::unordered_map<std::string, int> cameraBySerial;
    detail::PlanetaryLineScanBaWorkingSet workingSet;
    workingSet.cameraParameters.resize(cameras.size());
    result->cameras.resize(cameras.size());
    for (std::size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
    {
        const auto &camera = cameras[cameraIndex];
        if (camera.serialNumber.empty() || !camera.model.isValid() ||
            !cameraBySerial.emplace(camera.serialNumber, static_cast<int>(cameraIndex)).second)
        {
            setError(errorMessage, "line-scan camera has an empty/duplicate serial or invalid ISD model");
            return false;
        }
        if (camera.model.targetName() != "MOON" ||
            camera.model.bodyFixedFrameName() != "MOON_ME" ||
            (!controlNetwork.targetName.empty() &&
             controlNetwork.targetName != camera.model.targetName()))
        {
            setError(errorMessage,
                     "line-scan ISD, control network, and solver must share MOON/MOON_ME");
            return false;
        }
        workingSet.cameraModels.push_back(&camera.model);
        result->cameras[cameraIndex].serialNumber = camera.serialNumber;
    }

    for (const IsisControlPoint &controlPoint : controlNetwork.points)
    {
        if (controlPoint.ignored)
        {
            continue;
        }
        if (controlPoint.type != IsisControlPointType::Free)
        {
            setError(errorMessage,
                     "P0 line-scan BA supports only Free ISIS control points; point " +
                         controlPoint.id + " must not lose its ground prior silently");
            return false;
        }
        std::vector<MappedMeasure> measures;
        for (const IsisControlMeasure &measure : controlPoint.measures)
        {
            if (measure.ignored)
            {
                continue;
            }
            const auto camera = cameraBySerial.find(measure.serialNumber);
            if (camera == cameraBySerial.end())
            {
                setError(errorMessage, "unmapped ISIS camera serial in control network: " +
                                           measure.serialNumber);
                return false;
            }
            MappedMeasure mapped;
            mapped.cameraIndex = camera->second;
            mapped.measure = &measure;
            const double csmSample =
                measure.samplePixels - kIsisToCsmPixelCenterOffset;
            const double csmLine =
                measure.linePixels - kIsisToCsmPixelCenterOffset;
            if (!cameras[mapped.cameraIndex].model.pixelRayBodyFixed(
                    csmSample, csmLine,
                    PlanetaryLineScanCamera::PixelConvention::CsmPixelCenter,
                    &mapped.ray))
            {
                setError(errorMessage, "failed to construct line-scan ray for control point " +
                                           controlPoint.id);
                return false;
            }
            measures.push_back(mapped);
        }
        int first = -1;
        int second = -1;
        if (measures.size() < 2 || !chooseTriangulationPair(measures, &first, &second))
        {
            setError(errorMessage, "control point has insufficient intersecting rays: " +
                                       controlPoint.id);
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
        result->points.push_back({controlPoint.id, initialPoint, initialPoint,
                                  static_cast<int>(measures.size()), separation});
        for (const MappedMeasure &measure : measures)
        {
            workingSet.imageObservations.push_back({
                measure.cameraIndex, pointIndex,
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
            setError(errorMessage,
                     "invalid MOON_ME line-scan one-way laser dataset: " + validationError);
            return false;
        }
        for (const PlanetaryLaserShot &shot : laserDataset->shots)
        {
            if (norm(shot.leverArmSensorMeters) > 1.0e-12 ||
                shot.simultaneousImageIds.size() != 1)
            {
                setError(errorMessage,
                         "P0 line-scan BA requires zero LOLA lever arm and one simultaneous image per shot");
                return false;
            }
            const std::string &serial = shot.simultaneousImageIds.front();
            const auto mappedCamera = cameraBySerial.find(serial);
            if (mappedCamera == cameraBySerial.end())
            {
                setError(errorMessage, "unmapped simultaneous laser image serial: " + serial);
                return false;
            }
            double usedEt = 0.0;
            const auto &camera = cameras[mappedCamera->second];
            Vector3 center{};
            if (!usedLaserTime(camera, shot, options.laserTimeMode, &usedEt) ||
                !camera.model.sensorCenterBodyFixedAtEt(usedEt, &center))
            {
                setError(errorMessage, "failed to evaluate laser shot time/position: " + shot.id);
                return false;
            }
            detail::LineScanLaserPoint laserPoint;
            laserPoint.initialBodyFixedMeters = shot.pointBodyFixedMeters;
            laserPoint.refinedBodyFixedMeters = shot.pointBodyFixedMeters;
            laserPoint.pointMode = shot.pointMode;
            if (shot.pointMode == PlanetaryLaserPointMode::Constrained &&
                (!shot.pointCovarianceBodyFixedMetersSquared ||
                 !covarianceToSqrtInformation(*shot.pointCovarianceBodyFixedMetersSquared,
                                              &laserPoint.sqrtInformation)))
            {
                setError(errorMessage, "invalid laser point covariance for shot " + shot.id);
                return false;
            }
            if (options.enableLaserRangeConstraints &&
                shot.pointMode == PlanetaryLaserPointMode::Free)
            {
                setError(errorMessage, "P0 line-scan BA requires fixed or constrained laser points");
                return false;
            }
            const int laserPointIndex = static_cast<int>(workingSet.laserPoints.size());
            workingSet.laserPoints.push_back(laserPoint);
            workingSet.laserObservations.push_back({mappedCamera->second, laserPointIndex, center,
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
    result->activeLaserRangeCount = options.enableLaserRangeConstraints
        ? static_cast<int>(workingSet.laserObservations.size())
        : 0;
    result->initialImageRmsPixels = detail::lineScanImageRms(workingSet);
    result->initialLaserRangeRmsMeters = detail::lineScanLaserRangeRms(workingSet);
    const detail::PlanetaryLineScanBaWorkingSet initialWorkingSet = workingSet;
    PlanetaryLineScanBaOptions solverOptions = options;
    solverOptions.backend = selectLineScanBackend(
        options,
        static_cast<int>(workingSet.cameraParameters.size()),
        static_cast<int>(workingSet.imageObservations.size()));
    result->usedBackend = solverOptions.backend;
    bool solved = detail::solvePlanetaryLineScanBundleAdjustPlaMatrix(
        &workingSet, solverOptions, result, errorMessage);
    if (!solved &&
        (solverOptions.backend == BABackend::PlaMatrixCuda ||
         solverOptions.backend == BABackend::PlaMatrixOpenCl) &&
        result->terminationType != "CANCELLED" &&
        options.allowBackendFallback)
    {
        const std::string plaMatrixError = errorMessage ? *errorMessage : std::string{};
        workingSet = initialWorkingSet;
        solverOptions.backend = BABackend::PlaMatrixCpu;
        result->backendFallback = true;
        result->usedBackend = BABackend::PlaMatrixCpu;
        solved = detail::solvePlanetaryLineScanBundleAdjustPlaMatrix(
            &workingSet, solverOptions, result, errorMessage);
        if (solved)
        {
            result->backendMessage = "PlaMatrix line-scan BA failed: " +
                plaMatrixError + "; fell back to plamatrix_cpu";
        }
    }
    if (!solved)
    {
        return false;
    }
    if (result->backendMessage.empty())
    {
        result->backendMessage = std::string("line-scan BA used ") +
            BundleAdjust::backendName(result->usedBackend);
    }
    result->refinedImageRmsPixels = detail::lineScanImageRms(workingSet);
    result->refinedLaserRangeRmsMeters = detail::lineScanLaserRangeRms(workingSet);
    if (!std::isfinite(result->refinedImageRmsPixels) ||
        !std::isfinite(result->refinedLaserRangeRmsMeters) ||
        result->refinedImageRmsPixels > result->initialImageRmsPixels +
            std::max(0.25, 0.5 * result->initialImageRmsPixels))
    {
        setError(errorMessage,
                 "line-scan BA quality gate rejected non-finite or degraded residuals");
        return false;
    }
    for (std::size_t index = 0; index < workingSet.cameraParameters.size(); ++index)
    {
        const auto &parameters = workingSet.cameraParameters[index];
        const Vector3 translation{{parameters[0], parameters[1], parameters[2]}};
        const Vector3 angleAxis{{parameters[3], parameters[4], parameters[5]}};
        const double angleDegrees = norm(angleAxis) * 180.0 / std::acos(-1.0);
        if (!std::all_of(parameters.begin(), parameters.end(), [](double value)
            {
                return std::isfinite(value);
            }) ||
            norm(translation) > options.maximumCameraTranslationMeters ||
            angleDegrees > options.maximumCameraAngleDegrees)
        {
            setError(errorMessage,
                     "line-scan BA quality gate rejected an excessive camera correction");
            return false;
        }
        std::copy_n(parameters.begin(), 3,
                    result->cameras[index].translationBodyFixedMeters.begin());
        std::copy_n(parameters.begin() + 3, 3,
                    result->cameras[index].angleAxisBodyFixedRadians.begin());
    }
    for (std::size_t index = 0; index < workingSet.tiePoints.size(); ++index)
    {
        if (!std::all_of(workingSet.tiePoints[index].begin(),
                         workingSet.tiePoints[index].end(), [](double value)
            {
                return std::isfinite(value);
            }))
        {
            setError(errorMessage, "line-scan BA produced a non-finite control point");
            return false;
        }
        result->points[index].refinedBodyFixedMeters = workingSet.tiePoints[index];
    }
    for (std::size_t index = 0; index < workingSet.laserPoints.size(); ++index)
    {
        auto &shotResult = result->laserShots[index];
        shotResult.refinedPointBodyFixedMeters =
            workingSet.laserPoints[index].refinedBodyFixedMeters;
        const auto &observation = workingSet.laserObservations[index];
        const auto rangeResidual = [&observation](const Vector3 &point,
                                                  const std::array<double, 6> &camera)
        {
            Vector3 delta{{point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
                           point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
                           point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
            return norm(delta) - observation.observedRangeMeters;
        };
        const std::array<double, 6> zero{};
        shotResult.initialComputedMinusObservedMeters =
            rangeResidual(workingSet.laserPoints[index].initialBodyFixedMeters, zero);
        shotResult.refinedComputedMinusObservedMeters =
            rangeResidual(workingSet.laserPoints[index].refinedBodyFixedMeters,
                          workingSet.cameraParameters[observation.cameraIndex]);
    }
    result->success = true;
    return true;
}

} // namespace lidar
} // namespace xjw
