#pragma once

#include "PlanetaryLaserShot.h"

#include <array>
#include <optional>
#include <string>

namespace xjw
{
namespace lidar
{

struct PlanetaryLaserIsisContext
{
    PlanetaryLaserReferenceSystem reference;
    PlanetaryLaserSensorModel sensorModel = PlanetaryLaserSensorModel::Unknown;
    PlanetaryLaserRangeType rangeType = PlanetaryLaserRangeType::Unknown;
    // A zero lever arm is meaningful and must be supplied explicitly. ISIS
    // LidarData JSON does not serialize instrument extrinsics.
    std::optional<std::array<double, 3>> leverArmSensorMeters;
};

struct PlanetaryLaserJsonParseOptions
{
    // ISIS LidarData JSON does not carry target/frame/time-system metadata, so
    // importing that format is rejected unless the caller supplies this context.
    std::optional<PlanetaryLaserIsisContext> isisContext;
};

/**
 * Parses either PlaScan SI JSON v1 or ISIS LidarData JSON.
 *
 * PlaScan SI JSON v1 requires schema="plascan.planetary_laser_dataset",
 * version=1, SI/degree/pixel unit declarations, explicit reference metadata,
 * and a shots array. ISIS input is detected only by its top-level points array
 * and requires PlanetaryLaserJsonParseOptions::isisContext.
 */
bool parsePlanetaryLaserJson(const std::string &json,
                             const PlanetaryLaserJsonParseOptions &options,
                             PlanetaryLaserDataset *dataset,
                             std::string *errorMessage = nullptr);

bool loadPlanetaryLaserJsonFile(const std::string &path,
                                const PlanetaryLaserJsonParseOptions &options,
                                PlanetaryLaserDataset *dataset,
                                std::string *errorMessage = nullptr);

} // namespace lidar
} // namespace xjw
