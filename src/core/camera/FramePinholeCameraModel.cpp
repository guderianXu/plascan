#include "FramePinholeCamera.h"

#include <cmath>
#include <utility>

namespace xjw
{

CameraModelType FramePinholeCamera::modelType() const noexcept
{
    return CameraModelType::FramePinhole;
}

std::optional<CameraImageSize> FramePinholeCamera::imageSize() const noexcept
{
    return _imageSize;
}

std::string_view FramePinholeCamera::worldFrameName() const noexcept
{
    return _worldFrameName;
}

void FramePinholeCamera::setImageSize(std::optional<CameraImageSize> imageSize)
{
    _imageSize = imageSize;
    if (_imageSize.has_value()
        && (_imageSize->samples <= 0 || _imageSize->lines <= 0))
    {
        _imageSize.reset();
    }
}

void FramePinholeCamera::setWorldFrameName(std::string worldFrameName)
{
    _worldFrameName = std::move(worldFrameName);
}

bool FramePinholeCamera::rayForPixel(const CameraImageCoordinate &pixel,
                                     CameraImagingRay *ray) const
{
    if (ray == nullptr)
    {
        return false;
    }

    const double source_pixel[2] = {pixel.sample, pixel.line};
    double point_at_unit_depth[3] = {0.0, 0.0, 0.0};
    if (!unprojectPixel(source_pixel, 1.0, point_at_unit_depth))
    {
        return false;
    }

    CameraImagingRay result;
    result.originMeters = cameraCenter();
    double squared_norm = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        result.direction[axis] = point_at_unit_depth[axis] - result.originMeters[axis];
        squared_norm += result.direction[axis] * result.direction[axis];
    }
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm))
    {
        return false;
    }

    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (double &component : result.direction)
    {
        component *= inverse_norm;
        if (!std::isfinite(component))
        {
            return false;
        }
    }
    result.ephemerisTimeSeconds.reset();
    *ray = result;
    return true;
}

bool FramePinholeCamera::groundToImage(
    const std::array<double, 3> &groundMeters,
    CameraGroundProjection *projection) const
{
    if (projection == nullptr)
    {
        return false;
    }

    double pixel[2] = {0.0, 0.0};
    double positive_depth = 0.0;
    if (!projectWorldPointWithDepth(groundMeters.data(), pixel, positive_depth))
    {
        return false;
    }

    CameraGroundProjection result;
    result.image = {pixel[0], pixel[1]};
    result.positiveDepthMeters = positive_depth;
    result.ephemerisTimeSeconds.reset();
    *projection = result;
    return true;
}

} // namespace xjw
