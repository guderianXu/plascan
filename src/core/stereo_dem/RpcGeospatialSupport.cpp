#include "RpcGeospatialSupport.h"

#include <ogr_spatialref.h>
#include <cpl_conv.h>

#include <algorithm>
#include <memory>

namespace xjw::stereo_dem
{
    namespace
    {

        struct CoordinateTransformationDeleter
        {
            void operator()(OGRCoordinateTransformation* transformation) const
            {
                if (transformation)
                {
                    OCTDestroyCoordinateTransformation(transformation);
                }
            }
        };

        using CoordinateTransformationPtr =
            std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>;

        void configureTraditionalAxisOrder(OGRSpatialReference* spatialReference)
        {
            if (spatialReference)
            {
                spatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            }
        }

        bool setWgs84(OGRSpatialReference* spatialReference, QString* errorMessage)
        {
            if (!spatialReference || spatialReference->SetWellKnownGeogCS("WGS84") != OGRERR_NONE)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法创建 WGS84 坐标系");
                }
                return false;
            }
            configureTraditionalAxisOrder(spatialReference);
            return true;
        }

        bool importProjected(const ProjectedCoordinateSystem& coordinateSystem,
                             OGRSpatialReference* spatialReference,
                             QString* errorMessage)
        {
            if (!spatialReference || coordinateSystem.wkt.trimmed().isEmpty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("目标投影坐标系为空");
                }
                return false;
            }
            const QByteArray wkt = coordinateSystem.wkt.toUtf8();
            if (spatialReference->SetFromUserInput(wkt.constData()) != OGRERR_NONE)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法解析目标投影坐标系");
                }
                return false;
            }
            configureTraditionalAxisOrder(spatialReference);
            return true;
        }

    } // namespace

    bool createLocalUtm(double longitudeDegrees,
                        double latitudeDegrees,
                        ProjectedCoordinateSystem* coordinateSystem,
                        QString* errorMessage)
    {
        if (!coordinateSystem || longitudeDegrees < -180.0 || longitudeDegrees > 180.0 || latitudeDegrees < -80.0 ||
            latitudeDegrees > 84.0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法为当前经纬度创建 UTM 坐标系");
            }
            return false;
        }

        const int zone = std::clamp(static_cast<int>((longitudeDegrees + 180.0) / 6.0) + 1, 1, 60);
        const bool north = latitudeDegrees >= 0.0;
        OGRSpatialReference target;
        configureTraditionalAxisOrder(&target);
        if (target.SetWellKnownGeogCS("WGS84") != OGRERR_NONE ||
            target.SetUTM(zone, north ? TRUE : FALSE) != OGRERR_NONE)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("创建 UTM 投影失败");
            }
            return false;
        }

        char* exportedWkt = nullptr;
        if (target.exportToWkt(&exportedWkt) != OGRERR_NONE || !exportedWkt)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("导出 UTM WKT 失败");
            }
            return false;
        }

        coordinateSystem->epsg = (north ? 32600 : 32700) + zone;
        coordinateSystem->name = QStringLiteral("EPSG:%1").arg(coordinateSystem->epsg);
        coordinateSystem->wkt = QString::fromUtf8(exportedWkt);
        CPLFree(exportedWkt);
        return true;
    }

    bool geodeticToProjected(const std::vector<RpcCameraModel::GeodeticCoordinate>& geodetic,
                             const ProjectedCoordinateSystem& coordinateSystem,
                             std::vector<std::array<double, 3>>* projected,
                             QString* errorMessage)
    {
        if (!projected || geodetic.empty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("待投影的地面点为空");
            }
            return false;
        }

        OGRSpatialReference source;
        OGRSpatialReference target;
        if (!setWgs84(&source, errorMessage) || !importProjected(coordinateSystem, &target, errorMessage))
        {
            return false;
        }
        CoordinateTransformationPtr transformation(OGRCreateCoordinateTransformation(&source, &target));
        if (!transformation)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法创建 WGS84 到目标投影的坐标转换");
            }
            return false;
        }

        std::vector<double> x(geodetic.size());
        std::vector<double> y(geodetic.size());
        std::vector<double> z(geodetic.size());
        for (std::size_t index = 0; index < geodetic.size(); ++index)
        {
            x[index] = geodetic[index][0];
            y[index] = geodetic[index][1];
            z[index] = geodetic[index][2];
        }
        if (!transformation->Transform(static_cast<int>(geodetic.size()), x.data(), y.data(), z.data()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("WGS84 地面点转换到目标投影失败");
            }
            return false;
        }

        projected->resize(geodetic.size());
        for (std::size_t index = 0; index < geodetic.size(); ++index)
        {
            (*projected)[index] = {x[index], y[index], geodetic[index][2]};
        }
        return true;
    }

    bool projectedRowToGeodetic(const DemGridData& dem,
                                int row,
                                std::vector<RpcCameraModel::GeodeticCoordinate>* geodetic,
                                QString* errorMessage)
    {
        if (!geodetic || row < 0 || row >= dem.height || dem.width <= 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 行坐标无效");
            }
            return false;
        }

        ProjectedCoordinateSystem projectedSystem;
        projectedSystem.wkt = dem.projection.projectionWkt;
        OGRSpatialReference source;
        OGRSpatialReference target;
        if (!importProjected(projectedSystem, &source, errorMessage) || !setWgs84(&target, errorMessage))
        {
            return false;
        }
        CoordinateTransformationPtr transformation(OGRCreateCoordinateTransformation(&source, &target));
        if (!transformation)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法创建 DEM 投影到 WGS84 的坐标转换");
            }
            return false;
        }

        std::vector<double> x(static_cast<std::size_t>(dem.width));
        std::vector<double> y(static_cast<std::size_t>(dem.width));
        std::vector<double> z(static_cast<std::size_t>(dem.width));
        for (int col = 0; col < dem.width; ++col)
        {
            x[static_cast<std::size_t>(col)] = dem.minX + dem.stepX * col;
            y[static_cast<std::size_t>(col)] = dem.minY + dem.stepY * row;
            z[static_cast<std::size_t>(col)] = dem.elevation.at<float>(row, col);
        }
        if (!transformation->Transform(dem.width, x.data(), y.data(), z.data()))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 网格坐标转换到 WGS84 失败");
            }
            return false;
        }

        geodetic->resize(static_cast<std::size_t>(dem.width));
        for (int col = 0; col < dem.width; ++col)
        {
            const std::size_t index = static_cast<std::size_t>(col);
            (*geodetic)[index] = {x[index], y[index], static_cast<double>(dem.elevation.at<float>(row, col))};
        }
        return true;
    }

} // namespace xjw::stereo_dem
