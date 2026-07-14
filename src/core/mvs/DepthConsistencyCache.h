#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xjw
{
namespace mvs
{

struct DepthConsistencyFrame
{
    int frameIndex = -1;
    cv::Mat depth;
    cv::Mat confidence;

    std::size_t byteSize() const;
};

class DepthConsistencyCache
{
public:
    using Loader = std::function<bool(int,
                                      DepthConsistencyFrame &,
                                      std::string *)>;
    using FrameHandle = std::shared_ptr<const DepthConsistencyFrame>;

    DepthConsistencyCache(Loader loader, std::size_t memoryBudgetBytes);

    FrameHandle acquire(int frameIndex, std::string *errorMessage = nullptr);

    std::size_t currentBytes() const;
    std::size_t peakBytes() const;
    std::size_t memoryBudgetBytes() const;

private:
    struct CacheEntry
    {
        std::shared_ptr<DepthConsistencyFrame> frame;
        std::uint64_t lastUse = 0;
    };

    void evictFor(std::size_t requiredBytes);

    Loader _loader;
    std::size_t _memoryBudgetBytes = 0;
    std::size_t _currentBytes = 0;
    std::size_t _peakBytes = 0;
    std::uint64_t _clock = 0;
    std::unordered_map<int, CacheEntry> _entries;
    mutable std::mutex _mutex;
};

} // namespace mvs
} // namespace xjw
