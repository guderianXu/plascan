#pragma once

#include "PlanetaryLineScanBundleAdjust.h"

#include <array>
#include <vector>

namespace xjw
{
namespace lidar
{
namespace detail
{

struct LineScanImageObservation
{
    int cameraIndex = -1;
    int pointIndex = -1;
    // CSM pixel-centre convention: the upper-left pixel centre is (0.5, 0.5).
    double samplePixels = 0.0;
    double linePixels = 0.0;
};

struct LineScanLaserPoint
{
    std::array<double, 3> initialBodyFixedMeters{{0.0, 0.0, 0.0}};
    std::array<double, 3> refinedBodyFixedMeters{{0.0, 0.0, 0.0}};
    std::array<double, 9> sqrtInformation{{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}};
    PlanetaryLaserPointMode pointMode = PlanetaryLaserPointMode::Unspecified;
};

struct LineScanLaserObservation
{
    int cameraIndex = -1;
    int laserPointIndex = -1;
    std::array<double, 3> nominalSensorCenterMeters{{0.0, 0.0, 0.0}};
    double observedRangeMeters = 0.0;
    double sigmaMeters = 1.0;
};

struct PlanetaryLineScanBaWorkingSet
{
    std::vector<const PlanetaryLineScanCamera *> cameraModels;
    std::vector<std::array<double, 6>> cameraParameters;
    std::vector<std::array<double, 3>> tiePoints;
    std::vector<LineScanImageObservation> imageObservations;
    std::vector<LineScanLaserPoint> laserPoints;
    std::vector<LineScanLaserObservation> laserObservations;
};

PlanetaryLineScanCamera::PoseBias lineScanPoseBias(
    const std::array<double, 6> &parameters);

bool evaluateLineScanImageObservation(
    const PlanetaryLineScanCamera &camera,
    const LineScanImageObservation &observation,
    const double *cameraParameters,
    const double *point,
    double imageSigmaPixels,
    double *residuals);

double lineScanImageRms(const PlanetaryLineScanBaWorkingSet &workingSet);
double lineScanLaserRangeRms(const PlanetaryLineScanBaWorkingSet &workingSet);

bool solvePlanetaryLineScanBundleAdjustPlaMatrix(
    PlanetaryLineScanBaWorkingSet *workingSet,
    const PlanetaryLineScanBaOptions &options,
    PlanetaryLineScanBaResult *result,
    std::string *errorMessage);

} // namespace detail
} // namespace lidar
} // namespace xjw
