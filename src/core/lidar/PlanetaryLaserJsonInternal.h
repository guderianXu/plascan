#pragma once

#include "PlanetaryLaserJson.h"

class QJsonObject;

namespace xjw
{
namespace lidar
{

bool parseIsisLidarDataJson(const QJsonObject &root,
                            const PlanetaryLaserIsisContext &context,
                            PlanetaryLaserDataset *dataset,
                            std::string *errorMessage);

} // namespace lidar
} // namespace xjw
