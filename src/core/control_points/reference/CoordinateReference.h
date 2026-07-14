#pragma once

#include "model/MarkerTypes.h"

#include <QString>

#include <array>

namespace xjw::control_points
{

enum class AxisOrder
{
    TraditionalGis,
    AuthorityCompliant,
    LongitudeLatitude,
    LatitudeLongitude
};

enum class CoordinateUnit
{
    Unknown,
    Degree,
    Metre,
    InternationalFoot,
    UsSurveyFoot,
    OtherLinear,
    OtherAngular
};

QString axisOrderName(AxisOrder order);
bool axisOrderFromName(const QString &name, AxisOrder *order);
QString coordinateUnitName(CoordinateUnit unit);
CoordinateUnit coordinateUnitFromName(const QString &name);
double coordinateUnitToMetres(CoordinateUnit unit);

class CoordinateReference
{
public:
    static CoordinateReference fromEpsg(int epsg,
                                        AxisOrder order = AxisOrder::TraditionalGis);
    static CoordinateReference fromWkt(const QString &wkt,
                                       AxisOrder order = AxisOrder::TraditionalGis);
    static CoordinateReference fromUserInput(const QString &definition,
                                             AxisOrder order = AxisOrder::TraditionalGis);

    bool isValid() const noexcept;
    bool isGeographic() const noexcept;
    bool isProjected() const noexcept;
    bool isGeocentric() const noexcept;
    int axisCount() const noexcept;
    AxisOrder axisOrder() const noexcept;
    CoordinateUnit horizontalUnit() const noexcept;
    double horizontalUnitToMetres() const noexcept;
    QString definition() const;
    QString wkt() const;
    QString authority() const;
    QString error() const;

private:
    static CoordinateReference build(const QString &definition, AxisOrder order);

    bool _valid = false;
    bool _geographic = false;
    bool _projected = false;
    bool _geocentric = false;
    int _axisCount = 0;
    AxisOrder _axisOrder = AxisOrder::TraditionalGis;
    CoordinateUnit _horizontalUnit = CoordinateUnit::Unknown;
    double _horizontalUnitToMetres = 0.0;
    QString _definition;
    QString _wkt;
    QString _authority;
    QString _error;
};

struct CoordinateTransformResult
{
    bool ok = false;
    std::array<double, 3> xyz{0.0, 0.0, 0.0};
    QString error;
};

CoordinateTransformResult transformCoordinate(const std::array<double, 3> &xyz,
                                              const CoordinateReference &source,
                                              const CoordinateReference &target);

struct ReferenceCoordinateAssessment
{
    bool usable = false;
    QString error;
};

ReferenceCoordinateAssessment assessReferenceCoordinate(const ReferenceCoordinate &coordinate);

} // namespace xjw::control_points

