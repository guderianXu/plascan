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
    FeaturePrepareFunction prepareFunction,
    int workerCount)
    : _requests(std::move(requests)),
      _capacity(std::max(1, capacity)),
      _workerCount(std::clamp(workerCount, 1, std::max(1, capacity))),
      _cancelFlag(cancelFlag),
      _prepareFunction(std::move(prepareFunction)),
      _activeProducerCount(_workerCount)
{
    _producers.reserve(static_cast<std::size_t>(_workerCount));
    try
    {
        for (int worker = 0; worker < _workerCount; ++worker)
        {
            _producers.emplace_back(&FeaturePreparationQueue::produce, this);
        }
    }
    catch (...)
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopRequested = true;
        }
        _notFull.notify_all();
        for (std::thread &producer : _producers)
        {
            if (producer.joinable())
            {
                producer.join();
            }
        }
        throw;
    }
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
    while (_preparedByPosition.find(_nextTakePosition) == _preparedByPosition.end() &&
           !_finished && !_stopRequested)
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
        _preparedByPosition.clear();
        _notFull.notify_all();
        return false;
    }
    const auto preparedIt = _preparedByPosition.find(_nextTakePosition);
    if (preparedIt == _preparedByPosition.end())
    {
        const std::exception_ptr producerFailure = _producerFailure;
        lock.unlock();
        if (producerFailure)
        {
            std::rethrow_exception(producerFailure);
        }
        return false;
    }

    *prepared = std::move(preparedIt->second);
    _preparedByPosition.erase(preparedIt);
    ++_nextTakePosition;
    _notFull.notify_all();
    return true;
}

void FeaturePreparationQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopRequested = true;
        _preparedByPosition.clear();
    }
    _notEmpty.notify_all();
    _notFull.notify_all();
    for (std::thread &producer : _producers)
    {
        if (producer.joinable())
        {
            producer.join();
        }
    }
}

int FeaturePreparationQueue::capacity() const
{
    return _capacity;
}

int FeaturePreparationQueue::workerCount() const
{
    return _workerCount;
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
    try
    {
        while (true)
        {
            std::size_t requestPosition = 0;
            FeaturePreparationRequest request;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _notFull.wait(lock,
                              [this]()
                              {
                                  return _stopRequested ||
                                      cancellationRequested() ||
                                      _nextRequestPosition >= _requests.size() ||
                                      _nextRequestPosition - _nextTakePosition <
                                          static_cast<std::size_t>(_capacity);
                              });
                if (_stopRequested || cancellationRequested() ||
                    _nextRequestPosition >= _requests.size())
                {
                    break;
                }
                requestPosition = _nextRequestPosition++;
                request = _requests[requestPosition];
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
                _preparedByPosition.emplace(requestPosition, std::move(prepared));
                _peakBufferedCount =
                    std::max(_peakBufferedCount, static_cast<int>(_preparedByPosition.size()));
            }
            _notEmpty.notify_all();
        }
    }
    catch (...)
    {
        // No exception may escape the std::thread entry point. The consumer
        // rethrows at the owning stage boundary after already-buffered work.
        std::lock_guard<std::mutex> lock(_mutex);
        _producerFailure = std::current_exception();
        _stopRequested = true;
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        --_activeProducerCount;
        _finished = _activeProducerCount == 0;
    }
    _notEmpty.notify_all();
    _notFull.notify_all();
}

} // namespace matchphotos
} // namespace xjw
