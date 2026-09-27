#pragma once

namespace placoordinate
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

        friend bool operator==(const TimeReference&, const TimeReference&) = default;
    };

} // namespace placoordinate
