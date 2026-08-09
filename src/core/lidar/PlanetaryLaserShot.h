#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace xjw
{
namespace lidar
{

enum class PlanetaryLaserSourceFormat
{
    Unknown,
    PlaScanSiJsonV1,
    IsisLidarDataJson
};

enum class PlanetaryLaserTimeSystem
{
    Unknown,
    TdbEtSeconds
};

enum class PlanetaryLaserImageMeasureKind
{
    Measured,
    ProjectedVirtual
};

enum class PlanetaryLaserSensorModel
{
    Unknown,
    Frame,
    LineScan
};

enum class PlanetaryLaserRangeType
{
    Unknown,
    OneWay,
    RoundTrip
};

enum class PlanetaryLaserPointMode
{
    Unspecified,
    Fixed,
    Constrained,
    Free
};

struct PlanetaryLaserReferenceSystem
{
    std::string targetName;
    std::string bodyFixedFrame;
    std::string laserFrame;
    PlanetaryLaserTimeSystem timeSystem = PlanetaryLaserTimeSystem::Unknown;
    std::string latitudeType;
    std::string longitudeDirection;
};

struct PlanetaryLaserImageMeasure
{
    std::string imageId;
    double samplePixels = 0.0;
    double linePixels = 0.0;
    PlanetaryLaserImageMeasureKind kind = PlanetaryLaserImageMeasureKind::ProjectedVirtual;
    std::optional<std::array<double, 4>> covariancePixelsSquared;
};

struct PlanetaryLaserShot
{
    std::string id;
    PlanetaryLaserPointMode pointMode = PlanetaryLaserPointMode::Unspecified;
    double ephemerisTimeSeconds = 0.0;
    double observedRangeMeters = 0.0;
    double rangeSigmaMeters = 0.0;
    std::array<double, 3> pointBodyFixedMeters{{0.0, 0.0, 0.0}};
    // Full row-major XYZ covariance in the declared body-fixed frame, in m^2.
    // ISIS spherical (latitude rad, longitude rad, radius m) covariance is
    // transformed through the local spherical-to-XYZ Jacobian on import.
    std::optional<std::array<double, 9>> pointCovarianceBodyFixedMetersSquared;
    std::vector<std::string> simultaneousImageIds;
    std::vector<PlanetaryLaserImageMeasure> imageMeasures;
    std::array<double, 3> leverArmSensorMeters{{0.0, 0.0, 0.0}};
    bool leverArmSpecified = false;
};

struct PlanetaryLaserDataset
{
    PlanetaryLaserSourceFormat sourceFormat = PlanetaryLaserSourceFormat::Unknown;
    PlanetaryLaserSensorModel sensorModel = PlanetaryLaserSensorModel::Unknown;
    PlanetaryLaserRangeType rangeType = PlanetaryLaserRangeType::Unknown;
    PlanetaryLaserReferenceSystem reference;
    std::vector<PlanetaryLaserShot> shots;

    bool validate(std::string *errorMessage = nullptr) const;
};

std::array<double, 3> planetocentricToBodyFixedMeters(double latitudeDegrees,
                                                       double longitudeDegrees,
                                                       double radiusMeters);

const char *planetaryLaserTimeSystemName(PlanetaryLaserTimeSystem timeSystem);
const char *planetaryLaserImageMeasureKindName(PlanetaryLaserImageMeasureKind kind);
const char *planetaryLaserSensorModelName(PlanetaryLaserSensorModel sensorModel);
const char *planetaryLaserRangeTypeName(PlanetaryLaserRangeType rangeType);
const char *planetaryLaserPointModeName(PlanetaryLaserPointMode pointMode);

} // namespace lidar
} // namespace xjw
