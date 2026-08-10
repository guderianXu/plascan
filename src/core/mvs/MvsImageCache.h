#pragma once

#include "camera/Camera.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace xjw::mvs
{

struct MvsImageFrame
{
    cv::Mat gray;
    cv::Mat preparedGray;
    Camera preparedCamera;
    cv::Mat validMask;
    bool projectMaskLoaded = false;

    std::uint64_t residentBytes() const noexcept;
};

class MvsImageCache
{
private:
    struct LeaseControl;

public:
    using Loader = std::function<bool(
        int,
        const std::atomic_bool *,
        MvsImageFrame *,
        std::string *)>;

    class ImageLease
    {
    public:
        ImageLease() = default;
        ~ImageLease();

        ImageLease(const ImageLease &) = delete;
        ImageLease &operator=(const ImageLease &) = delete;
        ImageLease(ImageLease &&other) noexcept;
        ImageLease &operator=(ImageLease &&other) noexcept;

        explicit operator bool() const noexcept;
        const MvsImageFrame *operator->() const noexcept;
        const MvsImageFrame &frame() const;
        int frameIndex() const noexcept;
        void reset() noexcept;

    private:
        friend class MvsImageCache;

        ImageLease(std::shared_ptr<LeaseControl> control,
                   int frameIndex,
                   std::shared_ptr<const MvsImageFrame> frame);

        std::shared_ptr<LeaseControl> _control;
        int _frameIndex = -1;
        std::shared_ptr<const MvsImageFrame> _frame;
    };

    MvsImageCache(std::size_t frameCount,
                  std::size_t capacity,
                  Loader loader);
    ~MvsImageCache();

    MvsImageCache(const MvsImageCache &) = delete;
    MvsImageCache &operator=(const MvsImageCache &) = delete;

    ImageLease acquire(int frameIndex,
                       const std::atomic_bool *cancelFlag,
                       std::string *errorMessage = nullptr);

    bool preloadAll(int workerCount,
                    const std::atomic_bool *cancelFlag,
                    std::string *errorMessage = nullptr);

    void clear();
    bool contains(int frameIndex) const;
    std::size_t frameCount() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t residentCount() const;
    std::uint64_t residentBytes() const;
    std::size_t loadAttemptCount(int frameIndex) const;

private:
    struct LeaseControl
    {
        std::mutex mutex;
        MvsImageCache *owner = nullptr;
    };

    enum class EntryState
    {
        Empty,
        Loading,
        Ready,
        Failed
    };

    struct Entry
    {
        EntryState state = EntryState::Empty;
        std::shared_ptr<const MvsImageFrame> frame;
        std::size_t pinCount = 0;
        std::uint64_t lastUse = 0;
        std::size_t loadAttemptCount = 0;
        std::string error;
    };

    void release(int frameIndex) noexcept;
    bool evictOneUnlocked(int excludedFrameIndex);
    bool cancelled(const std::atomic_bool *cancelFlag) const noexcept;

    std::vector<Entry> _entries;
    std::shared_ptr<LeaseControl> _leaseControl;
    std::size_t _capacity = 1;
    Loader _loader;
    mutable std::mutex _mutex;
    std::condition_variable _condition;
    std::size_t _reservedEntryCount = 0;
    std::uint64_t _clock = 0;
};

} // namespace xjw::mvs
