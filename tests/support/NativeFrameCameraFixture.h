#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>

#include <placamera/frame_camera.h>

// Mutable camera data used by GUI report fixtures. Geometry is evaluated by PlaCamera.
class NativeFrameCameraFixture
{
public:
    void setIntrinsics(double focal_x, double focal_y, double principal_x, double principal_y)
    {
        _intrinsics.focalX = focal_x;
        _intrinsics.focalY = focal_y;
        _intrinsics.principalX = principal_x;
        _intrinsics.principalY = principal_y;
    }

    void setPixelPitch(double pitch)
    {
        _intrinsics.pixelPitch = pitch;
    }

    void setAxisDirections(int u_sign, int v_sign)
    {
        _intrinsics.uAxisSign = u_sign;
        _intrinsics.vAxisSign = v_sign;
    }

    void setDepthAxisFlipped(bool flipped)
    {
        _depthAxisFlipped = flipped;
    }

    void setDistortion(double k1, double k2, double k3, double p1, double p2)
    {
        _distortion = {k1, k2, k3, p1, p2};
    }

    void setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center)
    {
        _rotation = rotation;
        _center = center;
    }

    void setCameraCenter(const std::array<double, 3>& center)
    {
        _center = center;
    }

    void setImageSize(placamera::ImageSize size)
    {
        _imageSize = size;
    }

    void bind(placamera::CameraInstanceId instance_id, placamera::ImageId image_id, placamera::FrameId world_frame)
    {
        _instanceId = std::move(instance_id);
        _imageId = std::move(image_id);
        _worldFrame = std::move(world_frame);
    }

    bool hasBoundIdentity() const noexcept
    {
        return _instanceId.has_value() && _imageId.has_value();
    }
    const placamera::CameraInstanceId& instanceId() const
    {
        return _instanceId.value();
    }
    const placamera::ImageId& imageId() const
    {
        return _imageId.value();
    }
    const placamera::FrameId& worldFrame() const noexcept
    {
        return _worldFrame;
    }
    const std::optional<placamera::TimeReference>& captureTime() const noexcept
    {
        return _captureTime;
    }
    std::optional<placamera::ImageSize> imageSize() const noexcept
    {
        return _imageSize;
    }
    const placamera::BrownConradyDistortion& distortion() const noexcept
    {
        return _distortion;
    }
    double focalX() const noexcept
    {
        return _intrinsics.focalX;
    }
    double focalY() const noexcept
    {
        return _intrinsics.focalY;
    }
    double principalX() const noexcept
    {
        return _intrinsics.principalX;
    }
    double principalY() const noexcept
    {
        return _intrinsics.principalY;
    }
    double pixelPitch() const noexcept
    {
        return _intrinsics.pixelPitch;
    }
    double focalXMillimeters() const noexcept
    {
        return focalX() * pixelPitch();
    }
    double focalYMillimeters() const noexcept
    {
        return focalY() * pixelPitch();
    }
    double principalXMillimeters() const noexcept
    {
        return principalX() * pixelPitch();
    }
    double principalYMillimeters() const noexcept
    {
        return principalY() * pixelPitch();
    }
    int uAxisSign() const noexcept
    {
        return _intrinsics.uAxisSign;
    }
    int vAxisSign() const noexcept
    {
        return _intrinsics.vAxisSign;
    }
    bool depthAxisFlipped() const noexcept
    {
        return _depthAxisFlipped;
    }
    const std::array<double, 9>& cameraToWorldRotation() const noexcept
    {
        return _rotation;
    }
    const std::array<double, 3>& cameraCenter() const noexcept
    {
        return _center;
    }

    std::array<double, 9> worldToCameraRotation() const noexcept
    {
        return {_rotation[0],
                _rotation[3],
                _rotation[6],
                _rotation[1],
                _rotation[4],
                _rotation[7],
                _rotation[2],
                _rotation[5],
                _rotation[8]};
    }

    std::array<double, 3> worldToCameraTranslation() const noexcept
    {
        const auto rotation = worldToCameraRotation();
        return {-(rotation[0] * _center[0] + rotation[1] * _center[1] + rotation[2] * _center[2]),
                -(rotation[3] * _center[0] + rotation[4] * _center[1] + rotation[5] * _center[2]),
                -(rotation[6] * _center[0] + rotation[7] * _center[1] + rotation[8] * _center[2])};
    }

    placamera::FramePinholeModel makeModel(placamera::CameraInstanceId instance_id,
                                           placamera::ImageId image_id,
                                           placamera::ImageSize image_size) const
    {
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("gui-fixture-definition-" + image_id.value()),
            _intrinsics,
            _distortion,
            placamera::PixelConvention::PixelCenter,
            _worldFrame,
            _depthAxisFlipped);
        return placamera::FramePinholeModel::create(std::move(instance_id),
                                                    std::move(image_id),
                                                    definition,
                                                    image_size,
                                                    placamera::Pose::create(_worldFrame, _center, _rotation),
                                                    _captureTime);
    }

    std::optional<placamera::FramePinholeModel> toModel() const
    {
        if (!hasBoundIdentity() || !_imageSize)
        {
            return std::nullopt;
        }
        return makeModel(*_instanceId, *_imageId, *_imageSize);
    }

private:
    placamera::FrameIntrinsics _intrinsics;
    placamera::BrownConradyDistortion _distortion;
    std::array<double, 9> _rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::array<double, 3> _center{0.0, 0.0, 0.0};
    placamera::FrameId _worldFrame{"project-world"};
    std::optional<placamera::CameraInstanceId> _instanceId;
    std::optional<placamera::ImageId> _imageId;
    std::optional<placamera::ImageSize> _imageSize;
    std::optional<placamera::TimeReference> _captureTime;
    bool _depthAxisFlipped = false;
};
