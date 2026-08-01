#pragma once

#include <QJsonObject>
#include <QtGlobal>

class QString;

namespace xjw
{

enum class OrthoProjectionType
{
    DemGrid
};

enum class OrthoSurfaceType
{
    Dem
};

enum class OrthoColorSource
{
    Images
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

    static bool fromJson(const QJsonObject &settings,
                         OrthoGenerationOptions *options,
                         QString *errorMsg = nullptr,
                         double legacyResolution = 0.0);

    bool validate(QString *errorMsg = nullptr) const;
    QJsonObject toResolvedJson() const;
};

QString orthoBlendModeToken(OrthoBlendMode mode);
QString orthoSizingModeToken(OrthoSizingMode mode);

} // namespace xjw
