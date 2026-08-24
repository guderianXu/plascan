#pragma once

#include "DemDomTypes.h"
#include "RpcCameraModel.h"

#include <QString>

#include <array>
#include <vector>

namespace xjw::stereo_dem
{

    struct ProjectedCoordinateSystem
    {
        int epsg = 0;
        QString name;
        QString wkt;
    };

    bool createLocalUtm(double longitudeDegrees,
                        double latitudeDegrees,
                        ProjectedCoordinateSystem* coordinateSystem,
                        QString* errorMessage);

    bool geodeticToProjected(const std::vector<RpcCameraModel::GeodeticCoordinate>& geodetic,
                             const ProjectedCoordinateSystem& coordinateSystem,
                             std::vector<std::array<double, 3>>* projected,
                             QString* errorMessage);

    bool projectedRowToGeodetic(const DemGridData& dem,
                                int row,
                                std::vector<RpcCameraModel::GeodeticCoordinate>* geodetic,
                                QString* errorMessage);

} // namespace xjw::stereo_dem
