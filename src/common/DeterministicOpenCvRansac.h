#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <mutex>
#include <utility>

namespace xjw::opencv_compat
{

/**
 * @brief 根据稳定的影像/阶段编号生成 OpenCV RANSAC 种子。
 */
inline int stableRansacSeed(std::uint32_t value0,
                            std::uint32_t value1 = 0,
                            std::uint32_t value2 = 0)
{
    std::uint32_t hash = 2166136261u;
    auto mix = [&hash](std::uint32_t value)
    {
        hash ^= value;
        hash *= 16777619u;
    };
    mix(value0);
    mix(value1);
    mix(value2);
    return static_cast<int>(hash & 0x7fffffffu);
}

inline std::mutex &deterministicRansacMutex()
{
    static std::mutex mutex;
    return mutex;
}

/**
 * @brief 在短临界区内设置稳定种子并调用依赖 OpenCV 默认 RNG 的 RANSAC。
 *
 * OpenCV 的部分鲁棒估计接口使用进程/线程默认 RNG。并行评估多个 SfM
 * 候选时，如果不把“设种子 + 求解”绑定为一个原子操作，正式解会依赖线程调度。
 */
template <typename Callable>
decltype(auto) runDeterministicRansac(int seed, Callable &&callable)
{
    std::lock_guard<std::mutex> lock(deterministicRansacMutex());
    cv::setRNGSeed(seed);
    return std::forward<Callable>(callable)();
}

} // namespace xjw::opencv_compat
