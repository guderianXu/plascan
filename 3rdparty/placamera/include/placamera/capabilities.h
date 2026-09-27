#pragma once

#include <cstdint>
#include <initializer_list>

namespace placamera
{

    enum class CapabilityKind : std::uint32_t
    {
        Projection = 1U << 0U,
        InverseProjection = 1U << 1U,
        ImagingLocus = 1U << 2U,
        StaticPose = 1U << 3U,
        Trajectory = 1U << 4U,
        ImageCorrection = 1U << 5U,
        Optimization = 1U << 6U,
    };

    class CapabilitySet
    {
    public:
        CapabilitySet() = default;
        CapabilitySet(std::initializer_list<CapabilityKind> values) noexcept;

        bool contains(CapabilityKind value) const noexcept;
        void add(CapabilityKind value) noexcept;
        std::uint32_t bits() const noexcept;

    private:
        std::uint32_t _bits = 0U;
    };

    const char* capabilityName(CapabilityKind capability) noexcept;

} // namespace placamera
