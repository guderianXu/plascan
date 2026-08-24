#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>

namespace xjw::opencv_utils
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

class ScopedOpenCvRngSeed
{
public:
    explicit ScopedOpenCvRngSeed(int seed)
        : _previousRng(cv::theRNG())
    {
        cv::setRNGSeed(seed);
    }

    ~ScopedOpenCvRngSeed()
    {
        cv::theRNG() = _previousRng;
    }

    ScopedOpenCvRngSeed(const ScopedOpenCvRngSeed &) = delete;
    ScopedOpenCvRngSeed &operator=(const ScopedOpenCvRngSeed &) = delete;

private:
    cv::RNG _previousRng;
};

/**
 * @brief 在线程局部 RNG 上设置稳定种子并调用依赖 OpenCV 默认 RNG 的 RANSAC。
 *
 * 支持的 OpenCV 版本通过 TLS 实现 cv::theRNG()。保存并恢复调用线程的 RNG
 * 状态既能隔离外部随机状态，也允许多个 SfM 候选真正并行执行鲁棒估计。
 */
template <typename Callable>
decltype(auto) runDeterministicRansac(int seed, Callable &&callable)
{
    const ScopedOpenCvRngSeed scopedSeed(seed);
    return std::forward<Callable>(callable)();
}

} // namespace xjw::opencv_utils
