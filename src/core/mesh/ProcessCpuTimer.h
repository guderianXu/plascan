#pragma once

#include <ctime>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace xjw::mesh::detail
{

inline double processCpuTimeMilliseconds()
{
#ifdef _WIN32
    FILETIME creation_time{};
    FILETIME exit_time{};
    FILETIME kernel_time{};
    FILETIME user_time{};
    if (!GetProcessTimes(GetCurrentProcess(),
                         &creation_time,
                         &exit_time,
                         &kernel_time,
                         &user_time))
    {
        return 0.0;
    }
    ULARGE_INTEGER kernel_value{};
    kernel_value.LowPart = kernel_time.dwLowDateTime;
    kernel_value.HighPart = kernel_time.dwHighDateTime;
    ULARGE_INTEGER user_value{};
    user_value.LowPart = user_time.dwLowDateTime;
    user_value.HighPart = user_time.dwHighDateTime;
    constexpr double kHundredNanosecondsPerMillisecond = 10000.0;
    return static_cast<double>(kernel_value.QuadPart + user_value.QuadPart) /
        kHundredNanosecondsPerMillisecond;
#else
    return 1000.0 * static_cast<double>(std::clock()) /
        static_cast<double>(CLOCKS_PER_SEC);
#endif
}

} // namespace xjw::mesh::detail
