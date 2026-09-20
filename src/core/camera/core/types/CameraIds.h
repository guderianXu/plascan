#pragma once

#include "CameraErrors.h"

#include <functional>
#include <cctype>
#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace xjw::camera_core
{

    template <typename Tag> class StrongId
    {
    public:
        explicit StrongId(std::string value) : _value(std::move(value))
        {
            const bool hasNonWhitespace =
                std::any_of(_value.begin(), _value.end(), [](unsigned char value) { return std::isspace(value) == 0; });
            if (_value.empty() || !hasNonWhitespace)
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
    struct ReferenceSourceIdTag;

    using CameraDefinitionId = StrongId<CameraDefinitionIdTag>;
    using CameraInstanceId = StrongId<CameraInstanceIdTag>;
    using ImageId = StrongId<ImageIdTag>;
    using ReferenceSourceId = StrongId<ReferenceSourceIdTag>;

} // namespace xjw::camera_core

namespace std
{

    template <typename Tag> struct hash<xjw::camera_core::StrongId<Tag>>
    {
        std::size_t operator()(const xjw::camera_core::StrongId<Tag>& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

} // namespace std
