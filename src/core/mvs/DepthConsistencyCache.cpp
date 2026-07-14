#include "DepthConsistencyCache.h"

#include <algorithm>
#include <limits>

namespace xjw
{
namespace mvs
{

std::size_t DepthConsistencyFrame::byteSize() const
{
    return depth.total() * depth.elemSize() +
           confidence.total() * confidence.elemSize();
}

DepthConsistencyCache::DepthConsistencyCache(Loader loader,
                                             std::size_t memoryBudgetBytes)
    : _loader(std::move(loader)),
      _memoryBudgetBytes(std::max<std::size_t>(1, memoryBudgetBytes))
{
}

DepthConsistencyCache::FrameHandle DepthConsistencyCache::acquire(
    int frameIndex,
    std::string *errorMessage)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const auto cached = _entries.find(frameIndex);
    if (cached != _entries.end())
    {
        cached->second.lastUse = ++_clock;
        return cached->second.frame;
    }

    if (!_loader)
    {
        if (errorMessage)
        {
            *errorMessage = "depth consistency cache has no frame loader";
        }
        return {};
    }

    auto loaded = std::make_shared<DepthConsistencyFrame>();
    if (!_loader(frameIndex, *loaded, errorMessage) || loaded->depth.empty())
    {
        return {};
    }
    loaded->frameIndex = frameIndex;
    const std::size_t bytes = std::max<std::size_t>(1, loaded->byteSize());
    evictFor(bytes);

    CacheEntry entry;
    entry.frame = loaded;
    entry.lastUse = ++_clock;
    _entries.insert_or_assign(frameIndex, std::move(entry));
    _currentBytes += bytes;
    _peakBytes = std::max(_peakBytes, _currentBytes);
    return loaded;
}

std::size_t DepthConsistencyCache::currentBytes() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _currentBytes;
}

std::size_t DepthConsistencyCache::peakBytes() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _peakBytes;
}

std::size_t DepthConsistencyCache::memoryBudgetBytes() const
{
    return _memoryBudgetBytes;
}

void DepthConsistencyCache::evictFor(std::size_t requiredBytes)
{
    while (!_entries.empty() &&
           _currentBytes + requiredBytes > _memoryBudgetBytes)
    {
        auto victim = _entries.end();
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (auto candidate = _entries.begin(); candidate != _entries.end(); ++candidate)
        {
            if (candidate->second.frame.use_count() == 1 &&
                candidate->second.lastUse < oldest)
            {
                oldest = candidate->second.lastUse;
                victim = candidate;
            }
        }
        if (victim == _entries.end())
        {
            break;
        }

        _currentBytes -= std::max<std::size_t>(1, victim->second.frame->byteSize());
        _entries.erase(victim);
    }
}

} // namespace mvs
} // namespace xjw
