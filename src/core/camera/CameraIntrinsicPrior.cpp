#include "CameraIntrinsicPrior.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace
{

struct FixedLensCameraSpec
{
    const char *make;
    const char *model;
    double focalLengthMm;
    double sensorWidthMm;
};

// 这里只收录固定镜头且型号可唯一确定传感器和镜头的相机。可换镜头机身不能
// 仅按机型推断焦距，否则会把错误先验强加给整个摄影块。
constexpr std::array<FixedLensCameraSpec, 2> kFixedLensCameraSpecs{{
    {"sony", "dsc-rx1", 35.0, 35.8},
    {"sony", "dsc-rx1r", 35.0, 35.8},
}};

QString normalizedIdentifier(QString value)
{
    value = value.trimmed().toLower();
    return value.simplified();
}
bool validImageSize(int width, int height)
{
    return width > 0 && height > 0;
}

std::optional<CameraIntrinsicPrior> makePrior(double focalPixels,
                                              int imageWidth,
                                              int imageHeight,
                                              const QString &source,
                                              const xjw::common::io::ImageExifMetadata &metadata)
{
    const double longestSide = static_cast<double>(std::max(imageWidth, imageHeight));
    if (!std::isfinite(focalPixels) || focalPixels <= 0.0 || longestSide <= 0.0)
    {
        return std::nullopt;
    }
    return CameraIntrinsicPrior{
        focalPixels,
        focalPixels / longestSide,
        source,
        metadata.make.trimmed(),
        metadata.model.trimmed(),
        true,
    };
}

} // namespace

std::optional<CameraIntrinsicPrior> estimateCameraIntrinsicPrior(
    const xjw::common::io::ImageExifMetadata &metadata,
    int imageWidth,
    int imageHeight)
{
    if (!validImageSize(imageWidth, imageHeight))
    {
        return std::nullopt;
    }

    // 35 mm 等效焦距按全画幅对角线定义。使用影像对角线换算可同时支持横片和竖片。
    if (metadata.focalLength35Mm.has_value() &&
        std::isfinite(*metadata.focalLength35Mm) && *metadata.focalLength35Mm > 0.0)
    {
        constexpr double kFullFrameDiagonalMm = 43.2666153056;
        const double imageDiagonal = std::hypot(static_cast<double>(imageWidth),
                                                static_cast<double>(imageHeight));
        return makePrior(*metadata.focalLength35Mm / kFullFrameDiagonalMm * imageDiagonal,
                         imageWidth,
                         imageHeight,
                         QStringLiteral("exif_focal_length_35mm"),
                         metadata);
    }

    const QString make = normalizedIdentifier(metadata.make);
    const QString model = normalizedIdentifier(metadata.model);
    for (const FixedLensCameraSpec &spec : kFixedLensCameraSpecs)
    {
        if (make == QLatin1String(spec.make) && model == QLatin1String(spec.model))
        {
            const double physicalFocal = metadata.focalLengthMm.value_or(spec.focalLengthMm);
            const QString source = metadata.focalLengthMm.has_value()
                ? QStringLiteral("exif_focal_length_and_camera_sensor_catalog")
                : QStringLiteral("fixed_lens_camera_catalog");
            return makePrior(physicalFocal / spec.sensorWidthMm * imageWidth,
                             imageWidth,
                             imageHeight,
                             source,
                             metadata);
        }
    }

    // 只有物理焦距而不知道传感器尺寸时不能换算成像素焦距。此处明确拒绝，
    // 避免把毫米值误当作 35 mm 等效焦距。
    return std::nullopt;
}
