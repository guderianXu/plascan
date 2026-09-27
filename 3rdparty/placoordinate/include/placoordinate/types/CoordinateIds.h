#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace placoordinate
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

} // namespace placoordinate

namespace std
{

    template <> struct hash<placoordinate::CoordinateFrameId>
    {
        std::size_t operator()(const placoordinate::CoordinateFrameId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

    template <> struct hash<placoordinate::SpatialReferenceId>
    {
        std::size_t operator()(const placoordinate::SpatialReferenceId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

    template <> struct hash<placoordinate::CoordinateContextId>
    {
        std::size_t operator()(const placoordinate::CoordinateContextId& id) const noexcept
        {
            return std::hash<std::string_view>{}(id.value());
        }
    };

} // namespace std
