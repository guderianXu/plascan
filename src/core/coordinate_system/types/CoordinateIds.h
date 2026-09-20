#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace xjw::coordinate_system
{

    class CoordinateFrameId
    {
    public:
        explicit CoordinateFrameId(std::string value);

        const std::string& value() const noexcept;

        friend bool operator==(const CoordinateFrameId&, const CoordinateFrameId&) = default;

    private:
        std::string _value;
    };

    class SpatialReferenceId
    {
    public:
        explicit SpatialReferenceId(std::string value);

        const std::string& value() const noexcept;

        friend bool operator==(const SpatialReferenceId&, const SpatialReferenceId&) = default;

    private:
        std::string _value;
    };

    class CoordinateContextId
    {
    public:
        explicit CoordinateContextId(std::string value);

        const std::string& value() const noexcept;

        friend bool operator==(const CoordinateContextId&, const CoordinateContextId&) = default;

    private:
        std::string _value;
    };

} // namespace xjw::coordinate_system

namespace std
{

    template <> struct hash<xjw::coordinate_system::CoordinateFrameId>
    {
        std::size_t operator()(const xjw::coordinate_system::CoordinateFrameId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

    template <> struct hash<xjw::coordinate_system::SpatialReferenceId>
    {
        std::size_t operator()(const xjw::coordinate_system::SpatialReferenceId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

    template <> struct hash<xjw::coordinate_system::CoordinateContextId>
    {
        std::size_t operator()(const xjw::coordinate_system::CoordinateContextId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

} // namespace std
