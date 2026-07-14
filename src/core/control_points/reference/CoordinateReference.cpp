#include "CoordinateReference.h"

#include <cpl_conv.h>
#include <cpl_error.h>
#include <ogr_spatialref.h>

#include <cmath>
#include <memory>
#include <utility>

namespace xjw::control_points
{

namespace
{

class GdalQuietErrorScope
{
public:
    GdalQuietErrorScope() { CPLPushErrorHandler(CPLQuietErrorHandler); }
    ~GdalQuietErrorScope() { CPLPopErrorHandler(); }
};

QString normalizedToken(QString value)
{
    value = value.trimmed().toLower();
    value.replace(QLatin1Char('-'), QLatin1Char('_'));
    value.replace(QLatin1Char(' '), QLatin1Char('_'));
    return value;
}

void applyAxisMapping(OGRSpatialReference *reference, AxisOrder order)
{
    if (!reference) return;
    const bool authority_compliant = order == AxisOrder::AuthorityCompliant;
    reference->SetAxisMappingStrategy(authority_compliant
                                          ? OAMS_AUTHORITY_COMPLIANT
                                          : OAMS_TRADITIONAL_GIS_ORDER);
}

QString lastGdalError(const QString &fallback)
{
    const QString error = QString::fromUtf8(CPLGetLastErrorMsg()).trimmed();
    return error.isEmpty() ? fallback : error;
}

CoordinateUnit classifyUnit(bool angular, const char *name, double factor)
{
    const QString normalized = normalizedToken(QString::fromUtf8(name ? name : ""));
    if (angular)
    {
        if (std::abs(factor - M_PI / 180.0) < 1.0e-12 || normalized.contains(QStringLiteral("degree")))
        {
            return CoordinateUnit::Degree;
        }
        return factor > 0.0 ? CoordinateUnit::OtherAngular : CoordinateUnit::Unknown;
    }

    if (std::abs(factor - 1.0) < 1.0e-12) return CoordinateUnit::Metre;
    if (std::abs(factor - 0.3048) < 1.0e-12) return CoordinateUnit::InternationalFoot;
    if (std::abs(factor - 0.3048006096012192) < 1.0e-12
        || normalized.contains(QStringLiteral("survey_foot"))
        || normalized.contains(QStringLiteral("foot_us")))
    {
        return CoordinateUnit::UsSurveyFoot;
    }
    return factor > 0.0 ? CoordinateUnit::OtherLinear : CoordinateUnit::Unknown;
}

bool makeSpatialReference(const CoordinateReference &reference,
                          OGRSpatialReference *spatialReference,
                          QString *error)
{
    if (!spatialReference || !reference.isValid())
    {
        if (error) *error = reference.error().isEmpty()
            ? QStringLiteral("CRS 无效") : reference.error();
        return false;
    }

    const QByteArray wkt = reference.wkt().toUtf8();
    const GdalQuietErrorScope quiet_errors;
    CPLErrorReset();
    if (spatialReference->SetFromUserInput(wkt.constData()) != OGRERR_NONE)
    {
        if (error) *error = lastGdalError(QStringLiteral("无法解析 CRS WKT"));
        return false;
    }
    applyAxisMapping(spatialReference, reference.axisOrder());
    return true;
}

struct CoordinateTransformationDeleter
{
    void operator()(OGRCoordinateTransformation *transformation) const
    {
        if (transformation) OGRCoordinateTransformation::DestroyCT(transformation);
    }
};

} // namespace

QString axisOrderName(AxisOrder order)
{
    switch (order)
    {
    case AxisOrder::TraditionalGis: return QStringLiteral("traditional_gis");
    case AxisOrder::AuthorityCompliant: return QStringLiteral("authority_compliant");
    case AxisOrder::LongitudeLatitude: return QStringLiteral("longitude_latitude");
    case AxisOrder::LatitudeLongitude: return QStringLiteral("latitude_longitude");
    }
    return QStringLiteral("traditional_gis");
}

bool axisOrderFromName(const QString &name, AxisOrder *order)
{
    if (!order) return false;
    const QString normalized = normalizedToken(name);
    if (normalized.isEmpty() || normalized == QLatin1String("traditional_gis")
        || normalized == QLatin1String("xy") || normalized == QLatin1String("easting_northing"))
    {
        *order = AxisOrder::TraditionalGis;
    }
    else if (normalized == QLatin1String("authority_compliant")
             || normalized == QLatin1String("authority"))
    {
        *order = AxisOrder::AuthorityCompliant;
    }
    else if (normalized == QLatin1String("longitude_latitude")
             || normalized == QLatin1String("lon_lat"))
    {
        *order = AxisOrder::LongitudeLatitude;
    }
    else if (normalized == QLatin1String("latitude_longitude")
             || normalized == QLatin1String("lat_lon"))
    {
        *order = AxisOrder::LatitudeLongitude;
    }
    else
    {
        return false;
    }
    return true;
}

QString coordinateUnitName(CoordinateUnit unit)
{
    switch (unit)
    {
    case CoordinateUnit::Degree: return QStringLiteral("degree");
    case CoordinateUnit::Metre: return QStringLiteral("m");
    case CoordinateUnit::InternationalFoot: return QStringLiteral("ft");
    case CoordinateUnit::UsSurveyFoot: return QStringLiteral("us_survey_ft");
    case CoordinateUnit::OtherLinear: return QStringLiteral("other_linear");
    case CoordinateUnit::OtherAngular: return QStringLiteral("other_angular");
    case CoordinateUnit::Unknown: break;
    }
    return QStringLiteral("unknown");
}

CoordinateUnit coordinateUnitFromName(const QString &name)
{
    const QString normalized = normalizedToken(name);
    if (normalized == QLatin1String("m") || normalized == QLatin1String("meter")
        || normalized == QLatin1String("metre") || normalized == QLatin1String("meters")
        || normalized == QLatin1String("metres"))
    {
        return CoordinateUnit::Metre;
    }
    if (normalized == QLatin1String("ft") || normalized == QLatin1String("foot")
        || normalized == QLatin1String("feet") || normalized == QLatin1String("international_foot"))
    {
        return CoordinateUnit::InternationalFoot;
    }
    if (normalized == QLatin1String("us_survey_ft") || normalized == QLatin1String("us_survey_foot")
        || normalized == QLatin1String("ftus"))
    {
        return CoordinateUnit::UsSurveyFoot;
    }
    if (normalized == QLatin1String("degree") || normalized == QLatin1String("degrees"))
    {
        return CoordinateUnit::Degree;
    }
    return CoordinateUnit::Unknown;
}

double coordinateUnitToMetres(CoordinateUnit unit)
{
    switch (unit)
    {
    case CoordinateUnit::Metre: return 1.0;
    case CoordinateUnit::InternationalFoot: return 0.3048;
    case CoordinateUnit::UsSurveyFoot: return 0.3048006096012192;
    default: return 0.0;
    }
}

CoordinateReference CoordinateReference::fromEpsg(int epsg, AxisOrder order)
{
    return build(QStringLiteral("EPSG:%1").arg(epsg), order);
}

CoordinateReference CoordinateReference::fromWkt(const QString &wkt, AxisOrder order)
{
    return build(wkt, order);
}

CoordinateReference CoordinateReference::fromUserInput(const QString &definition, AxisOrder order)
{
    return build(definition, order);
}

CoordinateReference CoordinateReference::build(const QString &definition, AxisOrder order)
{
    CoordinateReference result;
    result._definition = definition.trimmed();
    result._axisOrder = order;
    if (result._definition.isEmpty())
    {
        result._error = QStringLiteral("CRS 定义为空");
        return result;
    }

    OGRSpatialReference reference;
    const QByteArray encoded = result._definition.toUtf8();
    const GdalQuietErrorScope quiet_errors;
    CPLErrorReset();
    if (reference.SetFromUserInput(encoded.constData()) != OGRERR_NONE)
    {
        result._error = QStringLiteral("CRS 无法解析: %1").arg(
            lastGdalError(result._definition));
        return result;
    }
    applyAxisMapping(&reference, order);

    char *wkt = nullptr;
    if (reference.exportToWkt(&wkt) != OGRERR_NONE || !wkt)
    {
        result._error = QStringLiteral("CRS 无法导出 WKT: %1").arg(
            lastGdalError(result._definition));
        CPLFree(wkt);
        return result;
    }
    result._wkt = QString::fromUtf8(wkt);
    CPLFree(wkt);

    result._geographic = reference.IsGeographic();
    result._projected = reference.IsProjected();
    result._geocentric = reference.IsGeocentric();
    result._axisCount = reference.GetAxesCount();

    const char *unit_name = nullptr;
    const bool angular = result._geographic && !result._geocentric;
    const double unit_factor = angular
        ? reference.GetAngularUnits(&unit_name)
        : reference.GetLinearUnits(&unit_name);
    result._horizontalUnit = classifyUnit(angular, unit_name, unit_factor);
    result._horizontalUnitToMetres = angular ? 0.0 : unit_factor;

    const char *authority_name = reference.GetAuthorityName(nullptr);
    const char *authority_code = reference.GetAuthorityCode(nullptr);
    if (authority_name && authority_code)
    {
        result._authority = QStringLiteral("%1:%2")
                                .arg(QString::fromLatin1(authority_name),
                                     QString::fromLatin1(authority_code));
    }
    result._valid = true;
    return result;
}

bool CoordinateReference::isValid() const noexcept { return _valid; }
bool CoordinateReference::isGeographic() const noexcept { return _geographic; }
bool CoordinateReference::isProjected() const noexcept { return _projected; }
bool CoordinateReference::isGeocentric() const noexcept { return _geocentric; }
int CoordinateReference::axisCount() const noexcept { return _axisCount; }
AxisOrder CoordinateReference::axisOrder() const noexcept { return _axisOrder; }
CoordinateUnit CoordinateReference::horizontalUnit() const noexcept { return _horizontalUnit; }
double CoordinateReference::horizontalUnitToMetres() const noexcept { return _horizontalUnitToMetres; }
QString CoordinateReference::definition() const { return _definition; }
QString CoordinateReference::wkt() const { return _wkt; }
QString CoordinateReference::authority() const { return _authority; }
QString CoordinateReference::error() const { return _error; }

CoordinateTransformResult transformCoordinate(const std::array<double, 3> &xyz,
                                              const CoordinateReference &source,
                                              const CoordinateReference &target)
{
    CoordinateTransformResult result;
    OGRSpatialReference source_reference;
    OGRSpatialReference target_reference;
    if (!makeSpatialReference(source, &source_reference, &result.error)
        || !makeSpatialReference(target, &target_reference, &result.error))
    {
        return result;
    }

    const GdalQuietErrorScope quiet_errors;
    CPLErrorReset();
    std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter> transformation(
        OGRCreateCoordinateTransformation(&source_reference, &target_reference));
    if (!transformation)
    {
        result.error = QStringLiteral("无法创建 CRS 转换: %1 -> %2: %3")
                           .arg(source.definition(), target.definition(),
                                lastGdalError(QStringLiteral("GDAL 未返回详细错误")));
        return result;
    }

    result.xyz = xyz;
    if (source.axisOrder() == AxisOrder::LatitudeLongitude)
    {
        std::swap(result.xyz[0], result.xyz[1]);
    }
    if (!transformation->Transform(1, &result.xyz[0], &result.xyz[1], &result.xyz[2]))
    {
        result.error = QStringLiteral("CRS 坐标转换失败: %1 -> %2: %3")
                           .arg(source.definition(), target.definition(),
                                lastGdalError(QStringLiteral("坐标超出 CRS 有效范围")));
        return result;
    }
    if (target.axisOrder() == AxisOrder::LatitudeLongitude)
    {
        std::swap(result.xyz[0], result.xyz[1]);
    }
    if (!std::isfinite(result.xyz[0]) || !std::isfinite(result.xyz[1])
        || !std::isfinite(result.xyz[2]))
    {
        result.error = QStringLiteral("CRS 转换产生非有限坐标");
        return result;
    }
    result.ok = true;
    return result;
}

ReferenceCoordinateAssessment assessReferenceCoordinate(const ReferenceCoordinate &coordinate)
{
    ReferenceCoordinateAssessment result;
    if (!std::isfinite(coordinate.x) || !std::isfinite(coordinate.y) || !std::isfinite(coordinate.z)
        || !std::isfinite(coordinate.sigmaX) || coordinate.sigmaX <= 0.0
        || !std::isfinite(coordinate.sigmaY) || coordinate.sigmaY <= 0.0
        || !std::isfinite(coordinate.sigmaZ) || coordinate.sigmaZ <= 0.0)
    {
        result.error = QStringLiteral("参考坐标或 XY/Z 精度无效");
        return result;
    }

    AxisOrder order = AxisOrder::TraditionalGis;
    if (!axisOrderFromName(coordinate.axisOrder, &order))
    {
        result.error = QStringLiteral("参考坐标轴顺序无效: %1").arg(coordinate.axisOrder);
        return result;
    }
    const CoordinateReference reference =
        CoordinateReference::fromUserInput(coordinate.sourceCrs, order);
    if (!reference.isValid())
    {
        result.error = QStringLiteral("参考坐标 CRS 不可用: %1").arg(reference.error());
        return result;
    }
    if (reference.horizontalUnit() == CoordinateUnit::Unknown)
    {
        result.error = QStringLiteral("参考坐标 CRS 的水平单位无法识别");
        return result;
    }

    const bool crs_contains_height = reference.isGeocentric() || reference.axisCount() >= 3;
    if (!crs_contains_height)
    {
        if (coordinate.verticalDatum.trimmed().isEmpty())
        {
            result.error = QStringLiteral("二维 CRS 缺少垂直基准，参考坐标不能进入 BA");
            return result;
        }
        if (coordinateUnitFromName(coordinate.verticalUnit) == CoordinateUnit::Unknown)
        {
            result.error = QStringLiteral("垂直单位无法识别，参考坐标不能进入 BA");
            return result;
        }
    }

    result.usable = true;
    return result;
}

} // namespace xjw::control_points
