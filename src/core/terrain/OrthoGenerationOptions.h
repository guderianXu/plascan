#pragma once

#include <QJsonObject>
#include <QtGlobal>

class QString;

namespace xjw
{

enum class OrthoProjectionType
{
    DemGrid,
    Planar,
    SimpleCylindrical
};

enum class OrthoSurfaceType
{
    Dem,
    PointCloud
};

enum class OrthoColorSource
{
    Images,
    PointColors
};

enum class OrthoBlendMode
{
    Mosaic,
    WeightedAverage,
    FirstValid
};

enum class OrthoSizingMode
{
    PixelSize,
    MaximumDimension
};

struct OrthoBounds
{
    bool enabled = false;
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
};

struct OrthoGenerationOptions
{
    OrthoProjectionType projectionType = OrthoProjectionType::DemGrid;
    OrthoSurfaceType surfaceType = OrthoSurfaceType::Dem;
    OrthoColorSource colorSource = OrthoColorSource::Images;
    OrthoBlendMode blendMode = OrthoBlendMode::Mosaic;
    OrthoSizingMode sizingMode = OrthoSizingMode::PixelSize;
    double pixelSizeX = 0.0;
    double pixelSizeY = 0.0;
    int maximumDimension = 4096;
    OrthoBounds bounds;
    bool colorCorrection = true;
    bool sharpnessWeighting = false;
    bool ghostFilter = false;
    bool fillHoles = false;
    int holeFillMaxArea = 256;
    double holeFillRadius = 3.0;
    bool useProjectMasks = false;
    qint64 maximumPixelCount = 100000000;
    bool bodyReferenceAuto = true;
    double bodyCenterX = 0.0;
    double bodyCenterY = 0.0;
    double bodyCenterZ = 0.0;
    double referenceRadius = 0.0;
    double centralMeridian = 0.0;

    static bool fromJson(const QJsonObject &settings,
                         OrthoGenerationOptions *options,
                         QString *errorMsg = nullptr);

    bool validate(QString *errorMsg = nullptr) const;
    QJsonObject toResolvedJson() const;
};

QString orthoBlendModeToken(OrthoBlendMode mode);
QString orthoSizingModeToken(OrthoSizingMode mode);
QString orthoProjectionTypeToken(OrthoProjectionType type);
QString orthoSurfaceTypeToken(OrthoSurfaceType type);
QString orthoColorSourceToken(OrthoColorSource source);

} // namespace xjw
