#pragma once

#include "BundleAdjustSolver.h"
#include "IsisControlNetworkPvl.h"
#include "PlanetaryLaserShot.h"
#include "PlanetaryLineScanCamera.h"

#include <array>
#include <string>
#include <vector>

namespace xjw
{
namespace lidar
{

enum class PlanetaryLaserLineScanTimeMode
{
    ShotEphemerisTime,
    IsisSimultaneousMeasureLine
};

struct PlanetaryLineScanBaCamera
{
    std::string serialNumber;
    PlanetaryLineScanCamera model;
};

struct PlanetaryLineScanBaOptions
{
    BABackend backend = BABackend::Auto;
    int plaMatrixDevice = 0;
    int minPlaMatrixCudaCameras = BAOptions::kDefaultMinPlaMatrixCudaCameras;
    int minPlaMatrixCudaObservations = BAOptions::kDefaultMinPlaMatrixCudaObservations;
    int minPlaMatrixOpenClCameras = BAOptions::kDefaultMinPlaMatrixOpenClCameras;
    int minPlaMatrixOpenClObservations = BAOptions::kDefaultMinPlaMatrixOpenClObservations;
    int minPlaMatrixDenseCameras = BAOptions::kDefaultMinPlaMatrixDenseCameras;
    int minPlaMatrixCudaDenseObservations = BAOptions::kDefaultMinPlaMatrixCudaDenseObservations;
    int minPlaMatrixOpenClDenseObservations = BAOptions::kDefaultMinPlaMatrixOpenClDenseObservations;
    int maxDenseSchurCameras = 200;
    bool allowBackendFallback = true;
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    bool enableLaserRangeConstraints = false;
    PlanetaryLaserLineScanTimeMode laserTimeMode = PlanetaryLaserLineScanTimeMode::ShotEphemerisTime;
    int maximumIterations = 50;
    int threadCount = 0;
    double imageSigmaPixels = 1.0;
    double imageHuberDeltaPixels = 3.0;
    double cameraPositionSigmaMeters = 1000.0;
    double cameraAngleSigmaDegrees = 2.0;
    double laserRangeWeight = 1.0;
    double laserRangeHuberDeltaSigma = 3.0;
    double finiteDifferencePointStepMeters = 0.05;
    double finiteDifferencePositionStepMeters = 0.05;
    double finiteDifferenceAngleStepRadians = 1.0e-7;
    double maximumCameraTranslationMeters = 5000.0;
    double maximumCameraAngleDegrees = 5.0;
};

struct PlanetaryLineScanBaCameraResult
{
    std::string serialNumber;
    std::array<double, 3> translationBodyFixedMeters{{0.0, 0.0, 0.0}};
    std::array<double, 3> angleAxisBodyFixedRadians{{0.0, 0.0, 0.0}};
};

struct PlanetaryLineScanBaPointResult
{
    std::string id;
    std::array<double, 3> initialBodyFixedMeters{{0.0, 0.0, 0.0}};
    std::array<double, 3> refinedBodyFixedMeters{{0.0, 0.0, 0.0}};
    int observationCount = 0;
    double initialRaySeparationMeters = 0.0;
};

struct PlanetaryLineScanBaLaserShotResult
{
    std::string id;
    std::string simultaneousImageId;
    double shotEphemerisTimeSeconds = 0.0;
    double usedEphemerisTimeSeconds = 0.0;
    double imageLineTimeMinusShotTimeSeconds = 0.0;
    double observedRangeMeters = 0.0;
    double initialComputedMinusObservedMeters = 0.0;
    double refinedComputedMinusObservedMeters = 0.0;
    std::array<double, 3> initialPointBodyFixedMeters{{0.0, 0.0, 0.0}};
    std::array<double, 3> refinedPointBodyFixedMeters{{0.0, 0.0, 0.0}};
};

struct PlanetaryLineScanBaResult
{
    bool success = false;
    bool solutionUsable = false;
    bool converged = false;
    bool backendFallback = false;
    bool usedGpu = false;
    bool laserConstraintsEnabled = false;
    BABackend requestedBackend = BABackend::Auto;
    BABackend usedBackend = BABackend::Auto;
    std::string terminationType;
    std::string message;
    std::string backendMessage;
    std::string linearSolverName;
    std::string deviceName;
    std::string solverBriefReport;
    int controlPointCount = 0;
    int imageObservationCount = 0;
    int activeLaserRangeCount = 0;
    int iterations = 0;
    double initialImageRmsPixels = 0.0;
    double refinedImageRmsPixels = 0.0;
    double initialLaserRangeRmsMeters = 0.0;
    double refinedLaserRangeRmsMeters = 0.0;
    std::vector<PlanetaryLineScanBaCameraResult> cameras;
    std::vector<PlanetaryLineScanBaPointResult> points;
    std::vector<PlanetaryLineScanBaLaserShotResult> laserShots;
};

/**
 * @brief Triangulate the midpoint of the shortest segment between two rays.
 */
bool triangulatePlanetaryLineScanRays(
    const PlanetaryLineScanCamera::ImagingRay &first,
    const PlanetaryLineScanCamera::ImagingRay &second,
    std::array<double, 3> *pointBodyFixedMeters,
    double *raySeparationMeters = nullptr);

/**
 * @brief Run sparse pushbroom bundle adjustment with optional laser ranges.
 *
 * ISIS image measures are converted from the (1, 1) upper-left pixel centre
 * to CSM's (0.5, 0.5) convention before projection. The per-image correction
 * is deliberately low-dimensional: one body-fixed
 * translation and one left-multiplicative small-angle rotation. Laser virtual
 * image measures are used only to choose ISIS-compatible line time; they are
 * never added as measured reprojection observations.
 */
bool runPlanetaryLineScanBundleAdjust(
    const std::vector<PlanetaryLineScanBaCamera> &cameras,
    const IsisControlNetwork &controlNetwork,
    const PlanetaryLaserDataset *laserDataset,
    const PlanetaryLineScanBaOptions &options,
    PlanetaryLineScanBaResult *result,
    std::string *errorMessage = nullptr);

const char *planetaryLaserLineScanTimeModeName(PlanetaryLaserLineScanTimeMode mode);

} // namespace lidar
} // namespace xjw
