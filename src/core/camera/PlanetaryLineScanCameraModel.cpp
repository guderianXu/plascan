#include "PlanetaryLineScanCamera.h"

namespace xjw
{

CameraModelType PlanetaryLineScanCamera::modelType() const noexcept
{
    return CameraModelType::PlanetaryLineScan;
}

std::optional<CameraImageSize> PlanetaryLineScanCamera::imageSize() const noexcept
{
    if (_imageSamples <= 0 || _imageLines <= 0)
    {
        return std::nullopt;
    }
    return CameraImageSize{_imageSamples, _imageLines};
}

std::string_view PlanetaryLineScanCamera::worldFrameName() const noexcept
{
    return _bodyFixedFrameName;
}

bool PlanetaryLineScanCamera::rayForPixel(const CameraImageCoordinate &pixel,
                                          CameraImagingRay *ray) const
{
    if (ray == nullptr)
    {
        return false;
    }

    ImagingRay source;
    if (!pixelRayBodyFixed(pixel.sample,
                           pixel.line,
                           PixelConvention::OpenCvZeroBased,
                           &source))
    {
        return false;
    }

    CameraImagingRay result;
    result.originMeters = source.centerBodyFixedMeters;
    result.direction = source.directionBodyFixed;
    result.ephemerisTimeSeconds = source.ephemerisTimeSeconds;
    *ray = result;
    return true;
}

bool PlanetaryLineScanCamera::groundToImage(
    const Vector3 &groundBodyFixedMeters,
    CameraGroundProjection *projection) const
{
    if (projection == nullptr)
    {
        return false;
    }

    ImageCoordinate image;
    if (!groundToImage(groundBodyFixedMeters,
                       PixelConvention::OpenCvZeroBased,
                       &image))
    {
        return false;
    }

    FixedLineProjection fixed_line;
    if (!projectAtObservedLine(groundBodyFixedMeters,
                               image.line,
                               PixelConvention::OpenCvZeroBased,
                               &fixed_line))
    {
        return false;
    }

    CameraGroundProjection result;
    result.image = {image.sample, image.line};
    result.positiveDepthMeters = fixed_line.sensorDepthMeters;
    result.ephemerisTimeSeconds = fixed_line.ephemerisTimeSeconds;
    *projection = result;
    return true;
}

} // namespace xjw
