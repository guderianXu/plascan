#include "placamera/capabilities.h"

namespace placamera
{

    namespace
    {

        constexpr std::uint32_t bit(CapabilityKind value) noexcept
        {
            return static_cast<std::uint32_t>(value);
        }

    } // namespace

    CapabilitySet::CapabilitySet(std::initializer_list<CapabilityKind> values) noexcept
    {
        for (const CapabilityKind value : values)
        {
            add(value);
        }
    }

    bool CapabilitySet::contains(CapabilityKind value) const noexcept
    {
        return (_bits & bit(value)) != 0U;
    }

    void CapabilitySet::add(CapabilityKind value) noexcept
    {
        _bits |= bit(value);
    }

    std::uint32_t CapabilitySet::bits() const noexcept
    {
        return _bits;
    }

    const char* capabilityName(CapabilityKind capability) noexcept
    {
        switch (capability)
        {
        case CapabilityKind::Projection:
            return "projection";
        case CapabilityKind::InverseProjection:
            return "inverse_projection";
        case CapabilityKind::ImagingLocus:
            return "imaging_locus";
        case CapabilityKind::StaticPose:
            return "static_pose";
        case CapabilityKind::Trajectory:
            return "trajectory";
        case CapabilityKind::ImageCorrection:
            return "image_correction";
        case CapabilityKind::Optimization:
            return "optimization";
        }
        return "unknown";
    }

} // namespace placamera
