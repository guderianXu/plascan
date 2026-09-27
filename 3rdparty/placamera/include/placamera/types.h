#pragma once

#include "placamera/errors.h"

#include <placoordinate/types/CoordinateIds.h>
#include <placoordinate/types/TimeReference.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace placamera
{

    template <typename Tag> class StrongId
    {
    public:
        explicit StrongId(std::string value) : _value(std::move(value))
        {
            const bool has_non_whitespace =
                std::any_of(_value.begin(), _value.end(), [](unsigned char value) { return std::isspace(value) == 0; });
            if (_value.empty() || !has_non_whitespace)
            {
                throw CameraValidationError(CameraErrorCode::EmptyIdentifier, "camera identifier must not be empty");
            }
        }

        const std::string& value() const noexcept
        {
            return _value;
        }

        friend bool operator==(const StrongId&, const StrongId&) = default;

    private:
        std::string _value;
    };

    struct CameraDefinitionIdTag;
    struct CameraInstanceIdTag;
    struct ImageIdTag;
    struct CaptureGroupIdTag;

    using CameraDefinitionId = StrongId<CameraDefinitionIdTag>;
    using CameraInstanceId = StrongId<CameraInstanceIdTag>;
    using ImageId = StrongId<ImageIdTag>;
    using CaptureGroupId = StrongId<CaptureGroupIdTag>;
    using FrameId = placoordinate::CoordinateFrameId;
    using TimeScale = placoordinate::TimeScale;
    using TimeReference = placoordinate::TimeReference;

    using Vector3 = std::array<double, 3>;
    using RotationMatrix = std::array<double, 9>;

    struct ImageSize
    {
        int samples = 0;
        int lines = 0;

        bool isValid() const noexcept
        {
            return samples > 0 && lines > 0;
        }
    };

    struct ImageCoordinate
    {
        double sample = 0.0;
        double line = 0.0;
    };

    struct GroundCoordinate
    {
        FrameId frame;
        Vector3 position{{0.0, 0.0, 0.0}};
    };

    struct Pose
    {
        FrameId frame;
        Vector3 center{{0.0, 0.0, 0.0}};
        RotationMatrix cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        static Pose create(FrameId frame, Vector3 center, RotationMatrix cameraToWorldRotation);
    };

} // namespace placamera

namespace std
{

    template <typename Tag> struct hash<placamera::StrongId<Tag>>
    {
        std::size_t operator()(const placamera::StrongId<Tag>& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

} // namespace std
