#pragma once

#include "Camera.h"
#include "CameraModel.h"

#include <optional>
#include <string>

namespace xjw
{

/**
 * @brief Static frame/pinhole implementation of the common camera geometry.
 *
 * This class owns the existing Tsai Camera value so it can be passed through a
 * CameraModel pointer without changing the established frame SfM/BA APIs.
 */
class FramePinholeCamera final : public CameraModel
{
public:
    FramePinholeCamera() = default;
    explicit FramePinholeCamera(Camera camera,
                                std::optional<CameraImageSize> imageSize = std::nullopt,
                                std::string worldFrameName = {});

    CameraModelType modelType() const noexcept override;
    bool isValid() const noexcept override;
    std::optional<CameraImageSize> imageSize() const noexcept override;
    std::string_view worldFrameName() const noexcept override;

    bool rayForPixel(const CameraImageCoordinate &pixel,
                     CameraImagingRay *ray) const override;
    bool groundToImage(const std::array<double, 3> &groundMeters,
                       CameraGroundProjection *projection) const override;

    const Camera &camera() const noexcept { return _camera; }

private:
    Camera _camera;
    std::optional<CameraImageSize> _imageSize;
    std::string _worldFrameName;
};

} // namespace xjw
