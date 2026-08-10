#include "MvsImageCache.h"

#include "concurrency/SafeWorkerGroup.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <stdexcept>
#include <stop_token>
#include <utility>

namespace xjw::mvs
{
namespace
{

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

} // namespace

MvsImageCache::ImageLease::ImageLease(
    std::shared_ptr<LeaseControl> control,
    int frameIndex,
    std::shared_ptr<const MvsImageFrame> frame)
    : _control(std::move(control))
    , _frameIndex(frameIndex)
    , _frame(std::move(frame))
{
}

MvsImageCache::ImageLease::~ImageLease()
{
    reset();
}

MvsImageCache::ImageLease::ImageLease(ImageLease &&other) noexcept
    : _control(std::move(other._control))
    , _frameIndex(other._frameIndex)
    , _frame(std::move(other._frame))
{
    other._frameIndex = -1;
}

MvsImageCache::ImageLease &MvsImageCache::ImageLease::operator=(
    ImageLease &&other) noexcept
{
    if (this != &other)
    {
        reset();
        _control = std::move(other._control);
        _frameIndex = other._frameIndex;
        _frame = std::move(other._frame);
        other._frameIndex = -1;
    }
    return *this;
}

MvsImageCache::ImageLease::operator bool() const noexcept
{
    return static_cast<bool>(_frame);
}

const MvsImageFrame *MvsImageCache::ImageLease::operator->() const noexcept
{
    return _frame.get();
}

const MvsImageFrame &MvsImageCache::ImageLease::frame() const
{
    if (!_frame)
    {
        throw std::logic_error("MVS image lease is empty");
    }
    return *_frame;
}

int MvsImageCache::ImageLease::frameIndex() const noexcept
{
    return _frameIndex;
}

void MvsImageCache::ImageLease::reset() noexcept
{
    if (_control)
    {
        const std::shared_ptr<LeaseControl> control = std::move(_control);
        const int frameIndex = _frameIndex;
        _frameIndex = -1;
        _frame.reset();
        std::lock_guard<std::mutex> lock(control->mutex);
        if (control->owner)
        {
            control->owner->release(frameIndex);
        }
    }
    else
    {
        _frame.reset();
    }
}

MvsImageCache::MvsImageCache(std::size_t frameCount,
                             std::size_t capacity,
                             Loader loader)
    : _entries(frameCount)
    , _leaseControl(std::make_shared<LeaseControl>())
    , _capacity(std::clamp<std::size_t>(capacity, 1, std::max<std::size_t>(1, frameCount)))
    , _loader(std::move(loader))
{
    if (!_loader)
    {
        throw std::invalid_argument("MVS image cache loader is empty");
    }
    _leaseControl->owner = this;
}

MvsImageCache::~MvsImageCache()
{
    std::lock_guard<std::mutex> lock(_leaseControl->mutex);
    _leaseControl->owner = nullptr;
}

bool MvsImageCache::cancelled(const std::atomic_bool *cancelFlag) const noexcept
{
    return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
}

bool MvsImageCache::evictOneUnlocked(int excludedFrameIndex)
{
    int victimIndex = -1;
    std::uint64_t oldestUse = std::numeric_limits<std::uint64_t>::max();
    for (int index = 0; index < static_cast<int>(_entries.size()); ++index)
    {
        const Entry &entry = _entries[static_cast<std::size_t>(index)];
        if (index == excludedFrameIndex
            || entry.state != EntryState::Ready
            || entry.pinCount != 0)
        {
            continue;
        }
        if (entry.lastUse < oldestUse)
        {
            oldestUse = entry.lastUse;
            victimIndex = index;
        }
    }
    if (victimIndex < 0)
    {
        return false;
    }

    Entry &victim = _entries[static_cast<std::size_t>(victimIndex)];
    victim = Entry{};
    --_reservedEntryCount;
    return true;
}

MvsImageCache::ImageLease MvsImageCache::acquire(
    int frameIndex,
    const std::atomic_bool *cancelFlag,
    std::string *errorMessage)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(_entries.size()))
    {
        setError(errorMessage, "MVS image cache frame index is out of range");
        return {};
    }

    std::unique_lock<std::mutex> lock(_mutex);
    Entry &entry = _entries[static_cast<std::size_t>(frameIndex)];
    for (;;)
    {
        if (cancelled(cancelFlag))
        {
            setError(errorMessage, "MVS image cache wait cancelled");
            return {};
        }
        if (entry.state == EntryState::Ready)
        {
            ++entry.pinCount;
            entry.lastUse = ++_clock;
            if (errorMessage)
            {
                errorMessage->clear();
            }
            return ImageLease(_leaseControl, frameIndex, entry.frame);
        }
        if (entry.state == EntryState::Failed)
        {
            setError(errorMessage, entry.error);
            return {};
        }
        if (entry.state == EntryState::Loading)
        {
            _condition.wait_for(lock, std::chrono::milliseconds(50));
            continue;
        }
        if (_reservedEntryCount >= _capacity && !evictOneUnlocked(frameIndex))
        {
            _condition.wait_for(lock, std::chrono::milliseconds(50));
            continue;
        }

        entry.state = EntryState::Loading;
        ++entry.loadAttemptCount;
        ++_reservedEntryCount;
        break;
    }

    lock.unlock();
    std::shared_ptr<MvsImageFrame> loaded;
    try
    {
        loaded = std::make_shared<MvsImageFrame>();
    }
    catch (...)
    {
        lock.lock();
        entry.state = EntryState::Empty;
        --_reservedEntryCount;
        _condition.notify_all();
        throw;
    }

    std::string loadError;
    bool loadedSuccessfully = false;
    std::exception_ptr loadException;
    try
    {
        loadedSuccessfully = _loader(
            frameIndex, cancelFlag, loaded.get(), &loadError);
    }
    catch (...)
    {
        loadException = std::current_exception();
    }

    lock.lock();
    if (loadException)
    {
        entry.state = EntryState::Empty;
        entry.frame.reset();
        --_reservedEntryCount;
        _condition.notify_all();
        lock.unlock();
        try
        {
            std::rethrow_exception(loadException);
        }
        catch (const std::exception &error)
        {
            setError(errorMessage,
                     std::string("MVS image loader failed: ") + error.what());
        }
        catch (...)
        {
            setError(errorMessage, "MVS image loader threw an unknown exception");
        }
        return {};
    }
    if (cancelled(cancelFlag))
    {
        entry.state = EntryState::Empty;
        entry.frame.reset();
        --_reservedEntryCount;
        _condition.notify_all();
        setError(errorMessage, "MVS image cache load cancelled");
        return {};
    }
    if (!loadedSuccessfully || loaded->gray.empty() || loaded->preparedGray.empty())
    {
        entry.state = EntryState::Failed;
        entry.error = loadError.empty()
            ? "MVS image loader returned an empty frame"
            : std::move(loadError);
        --_reservedEntryCount;
        _condition.notify_all();
        setError(errorMessage, entry.error);
        return {};
    }

    entry.frame = std::move(loaded);
    entry.state = EntryState::Ready;
    entry.pinCount = 1;
    entry.lastUse = ++_clock;
    entry.error.clear();
    _condition.notify_all();
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return ImageLease(_leaseControl, frameIndex, entry.frame);
}

bool MvsImageCache::preloadAll(int workerCount,
                               const std::atomic_bool *cancelFlag,
                               std::string *errorMessage)
{
    if (_entries.empty())
    {
        return true;
    }
    if (_capacity < _entries.size())
    {
        setError(errorMessage, "bounded MVS image cache cannot preload every frame");
        return false;
    }

    std::atomic_size_t nextIndex{0};
    std::atomic_bool failed{false};
    std::mutex errorMutex;
    std::string firstError;
    const std::size_t workers = std::min(
        _entries.size(),
        static_cast<std::size_t>(std::max(1, workerCount)));
    xjw::common::concurrency::runWorkerGroup(
        workers,
        [&](std::stop_token stopToken)
        {
            while (!stopToken.stop_requested()
                   && !failed.load(std::memory_order_relaxed)
                   && !cancelled(cancelFlag))
            {
                const std::size_t index = nextIndex.fetch_add(
                    1, std::memory_order_relaxed);
                if (index >= _entries.size())
                {
                    break;
                }
                std::string localError;
                ImageLease lease = acquire(
                    static_cast<int>(index), cancelFlag, &localError);
                if (!lease)
                {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> errorLock(errorMutex);
                    if (firstError.empty())
                    {
                        firstError = std::move(localError);
                    }
                    break;
                }
            }
        });

    if (cancelled(cancelFlag))
    {
        setError(errorMessage, "MVS image preload cancelled");
        return false;
    }
    if (failed.load(std::memory_order_relaxed))
    {
        setError(errorMessage, firstError);
        return false;
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

void MvsImageCache::release(int frameIndex) noexcept
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (frameIndex < 0 || frameIndex >= static_cast<int>(_entries.size()))
    {
        return;
    }
    Entry &entry = _entries[static_cast<std::size_t>(frameIndex)];
    if (entry.pinCount > 0)
    {
        --entry.pinCount;
        entry.lastUse = ++_clock;
    }
    _condition.notify_all();
}

void MvsImageCache::clear()
{
    std::lock_guard<std::mutex> lock(_mutex);
    for (Entry &entry : _entries)
    {
        if (entry.pinCount == 0 && entry.state != EntryState::Loading)
        {
            if (entry.state == EntryState::Ready)
            {
                --_reservedEntryCount;
            }
            entry = Entry{};
        }
    }
    _condition.notify_all();
}

bool MvsImageCache::contains(int frameIndex) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return frameIndex >= 0
        && frameIndex < static_cast<int>(_entries.size())
        && _entries[static_cast<std::size_t>(frameIndex)].state == EntryState::Ready;
}

std::size_t MvsImageCache::frameCount() const noexcept
{
    return _entries.size();
}

std::size_t MvsImageCache::capacity() const noexcept
{
    return _capacity;
}

std::size_t MvsImageCache::residentCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return static_cast<std::size_t>(std::count_if(
        _entries.begin(), _entries.end(), [](const Entry &entry)
        {
            return entry.state == EntryState::Ready;
        }));
}

std::uint64_t MvsImageCache::residentBytes() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    std::uint64_t bytes = 0;
    for (const Entry &entry : _entries)
    {
        if (entry.state == EntryState::Ready && entry.frame)
        {
            bytes += entry.frame->residentBytes();
        }
    }
    return bytes;
}

std::size_t MvsImageCache::loadAttemptCount(int frameIndex) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (frameIndex < 0 || frameIndex >= static_cast<int>(_entries.size()))
    {
        return 0;
    }
    return _entries[static_cast<std::size_t>(frameIndex)].loadAttemptCount;
}

} // namespace xjw::mvs
