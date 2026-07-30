#include "FeaturePreparationQueue.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

namespace xjw
{
namespace matchphotos
{

FeaturePreparationQueue::FeaturePreparationQueue(
    std::vector<FeaturePreparationRequest> requests,
    int capacity,
    std::atomic_bool *cancelFlag,
    FeaturePrepareFunction prepareFunction)
    : _requests(std::move(requests)),
      _capacity(std::max(1, capacity)),
      _cancelFlag(cancelFlag),
      _prepareFunction(std::move(prepareFunction)),
      _producer(&FeaturePreparationQueue::produce, this)
{
}

FeaturePreparationQueue::~FeaturePreparationQueue()
{
    stop();
}

bool FeaturePreparationQueue::take(PreparedFeatureImage *prepared)
{
    if (!prepared)
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    while (_buffer.empty() && !_finished && !_stopRequested)
    {
        if (cancellationRequested())
        {
            _stopRequested = true;
            _notFull.notify_all();
            return false;
        }
        _notEmpty.wait_for(lock, std::chrono::milliseconds(10));
    }

    if (cancellationRequested())
    {
        _stopRequested = true;
        _buffer.clear();
        _notFull.notify_all();
        return false;
    }
    if (_buffer.empty())
    {
        return false;
    }

    *prepared = std::move(_buffer.front());
    _buffer.pop_front();
    _notFull.notify_one();
    return true;
}

void FeaturePreparationQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopRequested = true;
        _buffer.clear();
    }
    _notEmpty.notify_all();
    _notFull.notify_all();
    if (_producer.joinable())
    {
        _producer.join();
    }
}

int FeaturePreparationQueue::capacity() const
{
    return _capacity;
}

int FeaturePreparationQueue::peakBufferedCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _peakBufferedCount;
}

bool FeaturePreparationQueue::cancellationRequested() const
{
    return _cancelFlag && _cancelFlag->load();
}

void FeaturePreparationQueue::produce()
{
    for (const FeaturePreparationRequest &request : _requests)
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _notFull.wait(lock,
                          [this]()
                          {
                              return _stopRequested ||
                                  cancellationRequested() ||
                                  static_cast<int>(_buffer.size()) < _capacity;
                          });
            if (_stopRequested || cancellationRequested())
            {
                break;
            }
        }

        PreparedFeatureImage prepared;
        try
        {
            prepared = _prepareFunction(request);
        }
        catch (const std::exception &exception)
        {
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            prepared.featurePath = request.featurePath;
            prepared.errorMessage = QString::fromUtf8(exception.what());
        }
        catch (...)
        {
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            prepared.featurePath = request.featurePath;
            prepared.errorMessage = QStringLiteral("未知的影像预取错误");
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopRequested || cancellationRequested())
            {
                break;
            }
            _buffer.push_back(std::move(prepared));
            _peakBufferedCount =
                std::max(_peakBufferedCount, static_cast<int>(_buffer.size()));
        }
        _notEmpty.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _finished = true;
    }
    _notEmpty.notify_all();
}

} // namespace matchphotos
} // namespace xjw
