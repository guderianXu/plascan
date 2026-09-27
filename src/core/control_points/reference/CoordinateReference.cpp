#include "CoordinateReference.h"

#include <placoordinate/context/CoordinateContext.h>
#include <placoordinate/gdal/GdalCoordinateTransform.h>

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
            GdalQuietErrorScope()
            {
                CPLPushErrorHandler(CPLQuietErrorHandler);
            }
            ~GdalQuietErrorScope()
            {
                CPLPopErrorHandler();
            }
        };

        QString normalizedToken(QString value)
        {
            value = value.trimmed().toLower();
            value.replace(QLatin1Char('-'), QLatin1Char('_'));
            value.replace(QLatin1Char(' '), QLatin1Char('_'));
            return value;
        }

        void applyAxisMapping(OGRSpatialReference* reference, AxisOrder order)
        {
            if (!reference)
                return;
            const bool authority_compliant = order == AxisOrder::AuthorityCompliant;
            reference->SetAxisMappingStrategy(authority_compliant ? OAMS_AUTHORITY_COMPLIANT
                                                                  : OAMS_TRADITIONAL_GIS_ORDER);
        }

        QString lastGdalError(const QString& fallback)
        {
            const QString error = QString::fromUtf8(CPLGetLastErrorMsg()).trimmed();
            return error.isEmpty() ? fallback : error;
        }

        CoordinateUnit classifyUnit(bool angular, const char* name, double factor)
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

            if (std::abs(factor - 1.0) < 1.0e-12)
                return CoordinateUnit::Metre;
            if (std::abs(factor - 0.3048) < 1.0e-12)
                return CoordinateUnit::InternationalFoot;
            if (std::abs(factor - 0.3048006096012192) < 1.0e-12 || normalized.contains(QStringLiteral("survey_foot")) ||
                normalized.contains(QStringLiteral("foot_us")))
            {
                return CoordinateUnit::UsSurveyFoot;
            }
            return factor > 0.0 ? CoordinateUnit::OtherLinear : CoordinateUnit::Unknown;
        }

        bool makeSpatialReference(const CoordinateReference& reference,
                                  OGRSpatialReference* spatialReference,
                                  QString* error)
        {
            if (!spatialReference || !reference.isValid())
            {
                if (error)
                    *error = reference.error().isEmpty() ? QStringLiteral("CRS 无效") : reference.error();
                return false;
            }

            const QByteArray wkt = reference.wkt().toUtf8();
            const GdalQuietErrorScope quiet_errors;
            CPLErrorReset();
            if (spatialReference->SetFromUserInput(wkt.constData()) != OGRERR_NONE)
            {
                if (error)
                    *error = lastGdalError(QStringLiteral("无法解析 CRS WKT"));
                return false;
            }
            applyAxisMapping(spatialReference, reference.axisOrder());
            return true;
        }

        struct CoordinateTransformationDeleter
        {
            void operator()(OGRCoordinateTransformation* transformation) const
            {
                if (transformation)
                    OGRCoordinateTransformation::DestroyCT(transformation);
            }
        };

        placoordinate::CoordinateAxisOrder coordinateAxisOrder(AxisOrder order)
        {
            using placoordinate::CoordinateAxisOrder;
            switch (order)
            {
            case AxisOrder::TraditionalGis:
                return CoordinateAxisOrder::TraditionalGis;
            case AxisOrder::AuthorityCompliant:
                return CoordinateAxisOrder::AuthorityCompliant;
            case AxisOrder::LongitudeLatitude:
                return CoordinateAxisOrder::LongitudeLatitude;
            case AxisOrder::LatitudeLongitude:
                return CoordinateAxisOrder::LatitudeLongitude;
            }
            return CoordinateAxisOrder::TraditionalGis;
        }

        placoordinate::VerticalReference verticalReference(const ReferenceCoordinate& coordinate,
                                                                    const CoordinateReference& source)
        {
            using placoordinate::VerticalReference;
            if (source.isGeocentric())
            {
                return VerticalReference::NotApplicable;
            }
            const QString datum = normalizedToken(coordinate.verticalDatum);
            if (datum.contains(QStringLiteral("ellipsoid")))
            {
                return VerticalReference::Ellipsoidal;
            }
            if (datum.contains(QStringLiteral("orthometric")) || datum.contains(QStringLiteral("geoid")) ||
                datum.startsWith(QStringLiteral("navd")) || datum.startsWith(QStringLiteral("egm")))
            {
                return VerticalReference::Orthometric;
            }
            if (datum.contains(QStringLiteral("planet")) || datum.contains(QStringLiteral("radius")))
            {
                return VerticalReference::PlanetaryRadius;
            }
            if (datum.contains(QStringLiteral("relative")) || datum.contains(QStringLiteral("local")))
            {
                return VerticalReference::Relative;
            }
            return VerticalReference::Unknown;
        }

        MetricReferenceCoordinateResult resolveWithContext(const ReferenceCoordinate& coordinate,
                                                           AxisOrder axisOrder,
                                                           const CoordinateReference& source,
                                                           const placoordinate::CoordinateContext& context)
        {
            using namespace placoordinate;
            MetricReferenceCoordinateResult result;
            if (!context.solverFrame().hasMetricScale())
            {
                result.error = "工程坐标上下文的 solver frame 尚未解析为米制";
                return result;
            }
            const SpatialReferenceDefinition* solver_reference = context.solverSpatialReference();
            if (!solver_reference || !solver_reference->isMetricCartesian())
            {
                result.error = "工程坐标上下文缺少米制笛卡尔 solver reference";
                return result;
            }
            if (source.isGeographic() && source.axisCount() < 3)
            {
                result.error = "二维地理 CRS 的外部高程尚不能由当前坐标上下文完整转换";
                return result;
            }

            const VerticalReference source_vertical = verticalReference(coordinate, source);
            const GdalSpatialReferenceResult normalized =
                normalizeGdalSpatialReference(SpatialReferenceId("marker-source-probe"),
                                              CoordinateFrameId("marker-source-probe-frame"),
                                              source.wkt().toStdString(),
                                              source_vertical);
            if (!normalized.ok())
            {
                result.error = "参考坐标 CRS 无法规范化: " + normalized.error;
                return result;
            }
            const SpatialReferenceDefinition* registered = context.findSpatialReferenceByDefinitionHash(
                normalized.reference->canonicalDefinitionHash(), source_vertical);
            if (!registered)
            {
                result.error =
                    context.findSpatialReferenceByDefinitionHash(normalized.reference->canonicalDefinitionHash())
                        ? "参考坐标垂直基准与工程坐标上下文不一致"
                        : "参考坐标 CRS 未注册到当前工程坐标上下文";
                return result;
            }
            const GdalCoordinateUncertaintyResult transformed = transformGdalCoordinateWithDiagonalUncertainty(
                {coordinate.x, coordinate.y, coordinate.z},
                {coordinate.sigmaX, coordinate.sigmaY, coordinate.sigmaZ},
                *registered,
                coordinateAxisOrder(axisOrder),
                *solver_reference);
            if (!transformed.ok)
            {
                result.error = "参考坐标转换到 solver reference 失败: " + transformed.error;
                return result;
            }
            result.pointMetres = transformed.coordinate;
            result.sigmaMetres = transformed.standardDeviation;
            result.contextId = context.id().value();
            result.contextHash = context.contextHash();
            result.solverFrameId = context.solverFrame().frameId.value();
            result.solverSpatialReferenceId = solver_reference->id().value();
            result.normalizationHash = context.solverFrame().normalizationHash;
            result.referenceKey = context.contextHash() + "\nsolver_reference=" + solver_reference->id().value() +
                                  "\nnormalization=" + context.solverFrame().normalizationHash;
            result.ok = true;
            return result;
        }

    } // namespace

    QString axisOrderName(AxisOrder order)
    {
        switch (order)
        {
        case AxisOrder::TraditionalGis:
            return QStringLiteral("traditional_gis");
        case AxisOrder::AuthorityCompliant:
            return QStringLiteral("authority_compliant");
        case AxisOrder::LongitudeLatitude:
            return QStringLiteral("longitude_latitude");
        case AxisOrder::LatitudeLongitude:
            return QStringLiteral("latitude_longitude");
        }
        return QStringLiteral("traditional_gis");
    }

    bool axisOrderFromName(const QString& name, AxisOrder* order)
    {
        if (!order)
            return false;
        const QString normalized = normalizedToken(name);
        if (normalized.isEmpty() || normalized == QLatin1String("traditional_gis") ||
            normalized == QLatin1String("xy") || normalized == QLatin1String("easting_northing"))
        {
            *order = AxisOrder::TraditionalGis;
        }
        else if (normalized == QLatin1String("authority_compliant") || normalized == QLatin1String("authority"))
        {
            *order = AxisOrder::AuthorityCompliant;
        }
        else if (normalized == QLatin1String("longitude_latitude") || normalized == QLatin1String("lon_lat"))
        {
            *order = AxisOrder::LongitudeLatitude;
        }
        else if (normalized == QLatin1String("latitude_longitude") || normalized == QLatin1String("lat_lon"))
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
        case CoordinateUnit::Degree:
            return QStringLiteral("degree");
        case CoordinateUnit::Metre:
            return QStringLiteral("m");
        case CoordinateUnit::InternationalFoot:
            return QStringLiteral("ft");
        case CoordinateUnit::UsSurveyFoot:
            return QStringLiteral("us_survey_ft");
        case CoordinateUnit::OtherLinear:
            return QStringLiteral("other_linear");
        case CoordinateUnit::OtherAngular:
            return QStringLiteral("other_angular");
        case CoordinateUnit::Unknown:
            break;
        }
        return QStringLiteral("unknown");
    }

    CoordinateUnit coordinateUnitFromName(const QString& name)
    {
        const QString normalized = normalizedToken(name);
        if (normalized == QLatin1String("m") || normalized == QLatin1String("meter") ||
            normalized == QLatin1String("metre") || normalized == QLatin1String("meters") ||
            normalized == QLatin1String("metres"))
        {
            return CoordinateUnit::Metre;
        }
        if (normalized == QLatin1String("ft") || normalized == QLatin1String("foot") ||
            normalized == QLatin1String("feet") || normalized == QLatin1String("international_foot"))
        {
            return CoordinateUnit::InternationalFoot;
        }
        if (normalized == QLatin1String("us_survey_ft") || normalized == QLatin1String("us_survey_foot") ||
            normalized == QLatin1String("ftus"))
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
        case CoordinateUnit::Metre:
            return 1.0;
        case CoordinateUnit::InternationalFoot:
            return 0.3048;
        case CoordinateUnit::UsSurveyFoot:
            return 0.3048006096012192;
        default:
            return 0.0;
        }
    }

    CoordinateReference CoordinateReference::fromEpsg(int epsg, AxisOrder order)
    {
        return build(QStringLiteral("EPSG:%1").arg(epsg), order);
    }

    CoordinateReference CoordinateReference::fromWkt(const QString& wkt, AxisOrder order)
    {
        return build(wkt, order);
    }

    CoordinateReference CoordinateReference::fromUserInput(const QString& definition, AxisOrder order)
    {
        return build(definition, order);
    }

    CoordinateReference CoordinateReference::build(const QString& definition, AxisOrder order)
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
            result._error = QStringLiteral("CRS 无法解析: %1").arg(lastGdalError(result._definition));
            return result;
        }
        applyAxisMapping(&reference, order);

        char* wkt = nullptr;
        if (reference.exportToWkt(&wkt) != OGRERR_NONE || !wkt)
        {
            result._error = QStringLiteral("CRS 无法导出 WKT: %1").arg(lastGdalError(result._definition));
            CPLFree(wkt);
            return result;
        }
        result._wkt = QString::fromUtf8(wkt);
        CPLFree(wkt);

        result._geographic = reference.IsGeographic();
        result._projected = reference.IsProjected();
        result._geocentric = reference.IsGeocentric();
        result._axisCount = reference.GetAxesCount();

        const char* unit_name = nullptr;
        const bool angular = result._geographic && !result._geocentric;
        const double unit_factor =
            angular ? reference.GetAngularUnits(&unit_name) : reference.GetLinearUnits(&unit_name);
        result._horizontalUnit = classifyUnit(angular, unit_name, unit_factor);
        result._horizontalUnitToMetres = angular ? 0.0 : unit_factor;

        const char* authority_name = reference.GetAuthorityName(nullptr);
        const char* authority_code = reference.GetAuthorityCode(nullptr);
        if (authority_name && authority_code)
        {
            result._authority =
                QStringLiteral("%1:%2").arg(QString::fromLatin1(authority_name), QString::fromLatin1(authority_code));
        }
        result._valid = true;
        return result;
    }

    bool CoordinateReference::isValid() const noexcept
    {
        return _valid;
    }
    bool CoordinateReference::isGeographic() const noexcept
    {
        return _geographic;
    }
    bool CoordinateReference::isProjected() const noexcept
    {
        return _projected;
    }
    bool CoordinateReference::isGeocentric() const noexcept
    {
        return _geocentric;
    }
    int CoordinateReference::axisCount() const noexcept
    {
        return _axisCount;
    }
    AxisOrder CoordinateReference::axisOrder() const noexcept
    {
        return _axisOrder;
    }
    CoordinateUnit CoordinateReference::horizontalUnit() const noexcept
    {
        return _horizontalUnit;
    }
    double CoordinateReference::horizontalUnitToMetres() const noexcept
    {
        return _horizontalUnitToMetres;
    }
    QString CoordinateReference::definition() const
    {
        return _definition;
    }
    QString CoordinateReference::wkt() const
    {
        return _wkt;
    }
    QString CoordinateReference::authority() const
    {
        return _authority;
    }
    QString CoordinateReference::error() const
    {
        return _error;
    }

    CoordinateTransformResult transformCoordinate(const std::array<double, 3>& xyz,
                                                  const CoordinateReference& source,
                                                  const CoordinateReference& target)
    {
        CoordinateTransformResult result;
        OGRSpatialReference source_reference;
        OGRSpatialReference target_reference;
        if (!makeSpatialReference(source, &source_reference, &result.error) ||
            !makeSpatialReference(target, &target_reference, &result.error))
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
                               .arg(source.definition(),
                                    target.definition(),
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
                               .arg(source.definition(),
                                    target.definition(),
                                    lastGdalError(QStringLiteral("坐标超出 CRS 有效范围")));
            return result;
        }
        if (target.axisOrder() == AxisOrder::LatitudeLongitude)
        {
            std::swap(result.xyz[0], result.xyz[1]);
        }
        if (!std::isfinite(result.xyz[0]) || !std::isfinite(result.xyz[1]) || !std::isfinite(result.xyz[2]))
        {
            result.error = QStringLiteral("CRS 转换产生非有限坐标");
            return result;
        }
        result.ok = true;
        return result;
    }

    MetricReferenceCoordinateResult
    resolveMetricReferenceCoordinate(const ReferenceCoordinate& coordinate,
                                     const placoordinate::CoordinateContext* context)
    {
        MetricReferenceCoordinateResult result;
        const ReferenceCoordinateAssessment assessment = assessReferenceCoordinate(coordinate);
        if (!assessment.usable)
        {
            result.error = assessment.error.toStdString();
            return result;
        }

        AxisOrder axis_order = AxisOrder::TraditionalGis;
        if (!axisOrderFromName(coordinate.axisOrder, &axis_order))
        {
            result.error = QStringLiteral("参考坐标轴顺序无效: %1").arg(coordinate.axisOrder).toStdString();
            return result;
        }

        const CoordinateReference source = CoordinateReference::fromUserInput(coordinate.sourceCrs, axis_order);
        if (!source.isValid())
        {
            result.error = QStringLiteral("参考坐标 CRS 不可用: %1").arg(source.error()).toStdString();
            return result;
        }
        if (context)
        {
            return resolveWithContext(coordinate, axis_order, source, *context);
        }
        if (source.isGeographic())
        {
            result.error =
                QStringLiteral("地理角坐标不能直接作为米制 BA 坐标；请先通过工程坐标上下文转换到 solver frame")
                    .toStdString();
            return result;
        }
        if (!source.isProjected() && !source.isGeocentric())
        {
            result.error = QStringLiteral("参考坐标 CRS 不是可用于 BA 的投影或地心笛卡尔坐标系").toStdString();
            return result;
        }

        const double horizontal_to_metres = source.horizontalUnitToMetres();
        if (!std::isfinite(horizontal_to_metres) || horizontal_to_metres <= 0.0)
        {
            result.error = QStringLiteral("参考坐标 CRS 的线性单位无法换算为米").toStdString();
            return result;
        }

        const CoordinateReference canonical = CoordinateReference::fromWkt(source.wkt(), AxisOrder::TraditionalGis);
        const CoordinateTransformResult canonicalized =
            transformCoordinate({coordinate.x, coordinate.y, coordinate.z}, source, canonical);
        if (!canonicalized.ok)
        {
            result.error = canonicalized.error.toStdString();
            return result;
        }

        double vertical_to_metres = horizontal_to_metres;
        const bool crs_contains_height = source.isGeocentric() || source.axisCount() >= 3;
        if (!crs_contains_height)
        {
            vertical_to_metres = coordinateUnitToMetres(coordinateUnitFromName(coordinate.verticalUnit));
            if (!std::isfinite(vertical_to_metres) || vertical_to_metres <= 0.0)
            {
                result.error = QStringLiteral("垂直单位无法换算为米，参考坐标不能进入 BA").toStdString();
                return result;
            }
        }

        result.pointMetres = {canonicalized.xyz[0] * horizontal_to_metres,
                              canonicalized.xyz[1] * horizontal_to_metres,
                              canonicalized.xyz[2] * vertical_to_metres};
        result.sigmaMetres = {coordinate.sigmaX * horizontal_to_metres,
                              coordinate.sigmaY * horizontal_to_metres,
                              coordinate.sigmaZ * vertical_to_metres};
        result.referenceKey = canonical.wkt().toStdString();
        if (!crs_contains_height)
        {
            result.referenceKey += QStringLiteral("\nvertical_datum=%1;vertical_unit=metre")
                                       .arg(coordinate.verticalDatum.trimmed().toLower())
                                       .toStdString();
        }
        result.ok = true;
        return result;
    }

    ReferenceCoordinateAssessment assessReferenceCoordinate(const ReferenceCoordinate& coordinate)
    {
        ReferenceCoordinateAssessment result;
        if (!std::isfinite(coordinate.x) || !std::isfinite(coordinate.y) || !std::isfinite(coordinate.z) ||
            !std::isfinite(coordinate.sigmaX) || coordinate.sigmaX <= 0.0 || !std::isfinite(coordinate.sigmaY) ||
            coordinate.sigmaY <= 0.0 || !std::isfinite(coordinate.sigmaZ) || coordinate.sigmaZ <= 0.0)
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
        const CoordinateReference reference = CoordinateReference::fromUserInput(coordinate.sourceCrs, order);
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
