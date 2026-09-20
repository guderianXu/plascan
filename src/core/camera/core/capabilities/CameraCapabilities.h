#pragma once

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

namespace xjw::camera_core
{

    enum class CapabilityKind
    {
        Projection,
        InverseProjection,
        Ray,
        StaticPose,
        Trajectory,
        ImageCorrection,
        Optimization,
    };

    class CapabilitySet
    {
    public:
        CapabilitySet() = default;
        CapabilitySet(std::initializer_list<CapabilityKind> values) : _values(values)
        {
        }

        bool contains(CapabilityKind value) const noexcept
        {
            return std::find(_values.begin(), _values.end(), value) != _values.end();
        }

        void add(CapabilityKind value)
        {
            if (!contains(value))
            {
                _values.push_back(value);
            }
        }

        const std::vector<CapabilityKind>& values() const noexcept
        {
            return _values;
        }

    private:
        std::vector<CapabilityKind> _values;
    };

    const char* capabilityName(CapabilityKind capability) noexcept;

} // namespace xjw::camera_core
