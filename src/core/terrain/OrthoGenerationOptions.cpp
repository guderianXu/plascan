#include "OrthoGenerationOptions.h"

#include <QString>

#include <cmath>

namespace xjw
{

namespace
{

bool finiteValue(double value)
{
    return std::isfinite(value);
}

bool parseProjectionType(const QString &token,
                         OrthoProjectionType *type,
                         QString *errorMsg)
{
    if (token == QLatin1String("dem_grid"))
    {
        *type = OrthoProjectionType::DemGrid;
        return true;
    }
    if (token == QLatin1String("planar"))
    {
        *type = OrthoProjectionType::Planar;
        return true;
    }
    if (token == QLatin1String("cylindrical")
        || token == QLatin1String("simple_cylindrical"))
    {
        *type = OrthoProjectionType::SimpleCylindrical;
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的正射投影类型: %1").arg(token);
    }
    return false;
}

bool parseSurfaceType(const QString &token,
                      OrthoSurfaceType *type,
                      QString *errorMsg)
{
    if (token == QLatin1String("dem"))
    {
        *type = OrthoSurfaceType::Dem;
        return true;
    }
    if (token == QLatin1String("point_cloud"))
    {
        *type = OrthoSurfaceType::PointCloud;
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的正射表面类型: %1").arg(token);
    }
    return false;
}

bool parseColorSource(const QString &token,
                      OrthoColorSource *source,
                      QString *errorMsg)
{
    if (token == QLatin1String("images"))
    {
        *source = OrthoColorSource::Images;
        return true;
    }
    if (token == QLatin1String("point_colors"))
    {
        *source = OrthoColorSource::PointColors;
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的正射颜色来源: %1").arg(token);
    }
    return false;
}

bool parseBlendMode(const QString &token,
                    OrthoBlendMode *mode,
                    QString *errorMsg)
{
    if (token == QLatin1String("mosaic"))
    {
        *mode = OrthoBlendMode::Mosaic;
        return true;
    }
    if (token == QLatin1String("weighted_average"))
    {
        *mode = OrthoBlendMode::WeightedAverage;
        return true;
    }
    if (token == QLatin1String("first_valid"))
    {
        *mode = OrthoBlendMode::FirstValid;
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的正射混合模式: %1").arg(token);
    }
    return false;
}

bool parseSizingMode(const QString &token,
                     OrthoSizingMode *mode,
                     QString *errorMsg)
{
    if (token == QLatin1String("pixel_size"))
    {
        *mode = OrthoSizingMode::PixelSize;
        return true;
    }
    if (token == QLatin1String("maximum_dimension"))
    {
        *mode = OrthoSizingMode::MaximumDimension;
        return true;
    }
    if (errorMsg)
    {
        *errorMsg = QStringLiteral("不支持的正射尺寸模式: %1").arg(token);
    }
    return false;
}

} // namespace

QString orthoBlendModeToken(OrthoBlendMode mode)
{
    switch (mode)
    {
    case OrthoBlendMode::Mosaic:
        return QStringLiteral("mosaic");
    case OrthoBlendMode::WeightedAverage:
        return QStringLiteral("weighted_average");
    case OrthoBlendMode::FirstValid:
        return QStringLiteral("first_valid");
    }
    return QStringLiteral("mosaic");
}

QString orthoSizingModeToken(OrthoSizingMode mode)
{
    return mode == OrthoSizingMode::MaximumDimension
        ? QStringLiteral("maximum_dimension")
        : QStringLiteral("pixel_size");
}

QString orthoProjectionTypeToken(OrthoProjectionType type)
{
    switch (type)
    {
    case OrthoProjectionType::DemGrid:
        return QStringLiteral("dem_grid");
    case OrthoProjectionType::Planar:
        return QStringLiteral("planar");
    case OrthoProjectionType::SimpleCylindrical:
        return QStringLiteral("cylindrical");
    }
    return QStringLiteral("dem_grid");
}

QString orthoSurfaceTypeToken(OrthoSurfaceType type)
{
    return type == OrthoSurfaceType::PointCloud
        ? QStringLiteral("point_cloud")
        : QStringLiteral("dem");
}

QString orthoColorSourceToken(OrthoColorSource source)
{
    return source == OrthoColorSource::PointColors
        ? QStringLiteral("point_colors")
        : QStringLiteral("images");
}

bool OrthoGenerationOptions::fromJson(const QJsonObject &settings,
                                      OrthoGenerationOptions *options,
                                      QString *errorMsg)
{
    if (!options)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("正射参数输出对象为空");
        }
        return false;
    }

    QString sizing_mode =
        settings.value(QStringLiteral("sizing_mode")).toString().trimmed();
    if (sizing_mode.isEmpty())
    {
        sizing_mode = QStringLiteral("pixel_size");
    }

    OrthoGenerationOptions parsed;
    if (!parseProjectionType(
            settings.value(QStringLiteral("projection_type")).toString(QStringLiteral("dem_grid")),
            &parsed.projectionType,
            errorMsg)
        || !parseSurfaceType(
            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("dem")),
            &parsed.surfaceType,
            errorMsg)
        || !parseColorSource(
            settings.value(QStringLiteral("color_source")).toString(QStringLiteral("images")),
            &parsed.colorSource,
            errorMsg)
        || !parseBlendMode(
            settings.value(QStringLiteral("blend_mode")).toString(QStringLiteral("mosaic")),
            &parsed.blendMode,
            errorMsg)
        || !parseSizingMode(
            sizing_mode,
            &parsed.sizingMode,
            errorMsg))
    {
        return false;
    }

    parsed.pixelSizeX = settings.value(QStringLiteral("pixel_size_x")).toDouble();
    parsed.pixelSizeY = settings.value(QStringLiteral("pixel_size_y")).toDouble();
    parsed.maximumDimension =
        settings.value(QStringLiteral("maximum_dimension")).toInt(parsed.maximumDimension);
    parsed.bounds.enabled = settings.value(QStringLiteral("bounds_enabled")).toBool(false);
    parsed.bounds.minX = settings.value(QStringLiteral("min_x")).toDouble();
    parsed.bounds.minY = settings.value(QStringLiteral("min_y")).toDouble();
    parsed.bounds.maxX = settings.value(QStringLiteral("max_x")).toDouble();
    parsed.bounds.maxY = settings.value(QStringLiteral("max_y")).toDouble();
    parsed.colorCorrection = settings.value(QStringLiteral("color_correction")).toBool(true);
    parsed.sharpnessWeighting =
        settings.value(QStringLiteral("sharpness_weighting")).toBool(false);
    parsed.ghostFilter = settings.value(QStringLiteral("ghost_filter")).toBool(false);
    parsed.fillHoles = settings.value(QStringLiteral("fill_holes")).toBool(false);
    parsed.holeFillMaxArea =
        settings.value(QStringLiteral("hole_fill_max_area")).toInt(parsed.holeFillMaxArea);
    parsed.holeFillRadius =
        settings.value(QStringLiteral("hole_fill_radius")).toDouble(parsed.holeFillRadius);
    parsed.useProjectMasks =
        settings.value(QStringLiteral("use_project_masks")).toBool(false);
    parsed.maximumPixelCount =
        static_cast<qint64>(settings.value(QStringLiteral("maximum_pixel_count"))
                                .toDouble(static_cast<double>(parsed.maximumPixelCount)));
    parsed.bodyReferenceAuto =
        settings.value(QStringLiteral("body_reference_auto")).toBool(true);
    parsed.bodyCenterX = settings.value(QStringLiteral("body_center_x")).toDouble();
    parsed.bodyCenterY = settings.value(QStringLiteral("body_center_y")).toDouble();
    parsed.bodyCenterZ = settings.value(QStringLiteral("body_center_z")).toDouble();
    parsed.referenceRadius = settings.value(QStringLiteral("reference_radius")).toDouble();
    parsed.centralMeridian =
        settings.value(QStringLiteral("central_meridian")).toDouble();

    if (!parsed.validate(errorMsg))
    {
        return false;
    }
    *options = parsed;
    return true;
}

bool OrthoGenerationOptions::validate(QString *errorMsg) const
{
    QString message;
    const bool demCombination = projectionType == OrthoProjectionType::DemGrid
        && surfaceType == OrthoSurfaceType::Dem
        && colorSource == OrthoColorSource::Images;
    const bool pointCloudCombination = projectionType != OrthoProjectionType::DemGrid
        && surfaceType == OrthoSurfaceType::PointCloud
        && colorSource == OrthoColorSource::PointColors;
    if (!demCombination && !pointCloudCombination)
    {
        message = QStringLiteral("投影、表面与颜色来源组合无效");
    }
    else if (!finiteValue(pixelSizeX) || !finiteValue(pixelSizeY)
        || pixelSizeX < 0.0 || pixelSizeY < 0.0)
    {
        message = QStringLiteral("正射像元大小必须为有限的非负数");
    }
    else if (sizingMode == OrthoSizingMode::MaximumDimension
             && maximumDimension <= 0)
    {
        message = QStringLiteral("正射最大尺寸必须大于 0");
    }
    else if (bounds.enabled
             && (!finiteValue(bounds.minX) || !finiteValue(bounds.minY)
                 || !finiteValue(bounds.maxX) || !finiteValue(bounds.maxY)
                 || bounds.minX >= bounds.maxX || bounds.minY >= bounds.maxY))
    {
        message = QStringLiteral("正射区域边界无效，必须满足 min_x < max_x 且 min_y < max_y");
    }
    else if (fillHoles
             && (holeFillMaxArea <= 0
                 || !finiteValue(holeFillRadius)
                 || holeFillRadius <= 0.0))
    {
        message = QStringLiteral("正射孔洞填充面积和半径必须大于 0");
    }
    else if (maximumPixelCount <= 0)
    {
        message = QStringLiteral("正射最大像素数必须大于 0");
    }
    else if (!finiteValue(bodyCenterX) || !finiteValue(bodyCenterY)
             || !finiteValue(bodyCenterZ) || !finiteValue(referenceRadius)
             || !finiteValue(centralMeridian))
    {
        message = QStringLiteral("小天体参考参数必须为有限数值");
    }
    else if (projectionType == OrthoProjectionType::SimpleCylindrical
             && !bodyReferenceAuto && referenceRadius <= 0.0)
    {
        message = QStringLiteral("全球圆柱投影的参考半径必须大于 0");
    }

    if (!message.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = message;
        }
        return false;
    }
    return true;
}

QJsonObject OrthoGenerationOptions::toResolvedJson() const
{
    QJsonObject object;
    object[QStringLiteral("projection_type")] = orthoProjectionTypeToken(projectionType);
    object[QStringLiteral("surface_type")] = orthoSurfaceTypeToken(surfaceType);
    object[QStringLiteral("color_source")] = orthoColorSourceToken(colorSource);
    object[QStringLiteral("blend_mode")] = orthoBlendModeToken(blendMode);
    object[QStringLiteral("sizing_mode")] = orthoSizingModeToken(sizingMode);
    object[QStringLiteral("pixel_size_x")] = pixelSizeX;
    object[QStringLiteral("pixel_size_y")] = pixelSizeY;
    object[QStringLiteral("maximum_dimension")] = maximumDimension;
    object[QStringLiteral("bounds_enabled")] = bounds.enabled;
    object[QStringLiteral("min_x")] = bounds.minX;
    object[QStringLiteral("min_y")] = bounds.minY;
    object[QStringLiteral("max_x")] = bounds.maxX;
    object[QStringLiteral("max_y")] = bounds.maxY;
    object[QStringLiteral("color_correction")] = colorCorrection;
    object[QStringLiteral("sharpness_weighting")] = sharpnessWeighting;
    object[QStringLiteral("ghost_filter")] = ghostFilter;
    object[QStringLiteral("fill_holes")] = fillHoles;
    object[QStringLiteral("hole_fill_max_area")] = holeFillMaxArea;
    object[QStringLiteral("hole_fill_radius")] = holeFillRadius;
    object[QStringLiteral("use_project_masks")] = useProjectMasks;
    object[QStringLiteral("maximum_pixel_count")] =
        static_cast<double>(maximumPixelCount);
    object[QStringLiteral("body_reference_auto")] = bodyReferenceAuto;
    object[QStringLiteral("body_center_x")] = bodyCenterX;
    object[QStringLiteral("body_center_y")] = bodyCenterY;
    object[QStringLiteral("body_center_z")] = bodyCenterZ;
    object[QStringLiteral("reference_radius")] = referenceRadius;
    object[QStringLiteral("central_meridian")] = centralMeridian;
    return object;
}

} // namespace xjw
