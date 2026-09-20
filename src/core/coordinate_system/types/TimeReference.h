#pragma once

namespace xjw::coordinate_system
{

    enum class TimeScale
    {
        Utc,
        Tai,
        Tdb,
        Relative,
    };

    struct TimeReference
    {
        TimeScale scale = TimeScale::Relative;
        double seconds = 0.0;

        static TimeReference create(TimeScale timeScale, double value);
    };

} // namespace xjw::coordinate_system
