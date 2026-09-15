#pragma once
#include "MvsPipelineInternals.h"
namespace xjw::mvs::pipeline_detail
{

    class DepthFrameArtifactSaveQueue
    {
    public:
        using SaveFn = std::function<bool(int, const DepthFrameResult&, const QString&)>;

        class ProducerReservation
        {
        public:
            ProducerReservation() = default;

            ProducerReservation(const ProducerReservation&) = delete;
            ProducerReservation& operator=(const ProducerReservation&) = delete;

            ProducerReservation(ProducerReservation&& other) noexcept
                : _owner(other._owner), _residentBytes(other._residentBytes)
            {
                other.disarm();
            }

            ProducerReservation& operator=(ProducerReservation&& other) noexcept
            {
                if (this != &other)
                {
                    reset();
                    _owner = other._owner;
                    _residentBytes = other._residentBytes;
                    other.disarm();
                }
                return *this;
            }

            ~ProducerReservation()
            {
                reset();
            }

            explicit operator bool() const
            {
                return _owner != nullptr;
            }

            void reset()
            {
                if (_owner == nullptr)
                {
                    return;
                }

                DepthFrameArtifactSaveQueue* owner = _owner;
                const uint64_t resident_bytes = _residentBytes;
                disarm();
                owner->releaseProducerReservation(resident_bytes);
            }

        private:
            friend class DepthFrameArtifactSaveQueue;

            ProducerReservation(DepthFrameArtifactSaveQueue* owner, uint64_t residentBytes)
                : _owner(owner), _residentBytes(residentBytes)
            {
            }

            void disarm()
            {
                _owner = nullptr;
                _residentBytes = 0;
            }

            DepthFrameArtifactSaveQueue* _owner = nullptr;
            uint64_t _residentBytes = 0;
        };

        explicit DepthFrameArtifactSaveQueue(SaveFn saveFn,
                                             size_t workerCount,
                                             size_t maxResidentTasks,
                                             uint64_t maxResidentBytes,
                                             uint64_t producerReservationBytes)
            : _saveFn(std::move(saveFn)), _workerCount(std::clamp<size_t>(workerCount, 1, 2)),
              _maxResidentTasks(std::max<size_t>(1, maxResidentTasks)),
              _maxResidentBytes(std::max<uint64_t>(1, maxResidentBytes)),
              _producerReservationBytes(std::max<uint64_t>(1, producerReservationBytes))
        {
            _workers.reserve(_workerCount);
            try
            {
                for (size_t worker_index = 0; worker_index < _workerCount; ++worker_index)
                {
                    _workers.emplace_back(&DepthFrameArtifactSaveQueue::run, this);
                }
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stopping = true;
                }
                _cv.notify_all();
                for (std::thread& worker : _workers)
                {
                    if (worker.joinable())
                    {
                        worker.join();
                    }
                }
                throw;
            }
        }

        ~DepthFrameArtifactSaveQueue()
        {
            stop();
        }

        ProducerReservation reserveProducer(const std::atomic<bool>* cancelFlag = nullptr)
        {
            const auto wait_start = std::chrono::steady_clock::now();
            {
                std::unique_lock<std::mutex> lock(_mutex);
                while (!_stopping && !(cancelFlag && cancelFlag->load()) && !canAcceptLocked(_producerReservationBytes))
                {
                    _capacityCv.wait_for(lock, std::chrono::milliseconds(25));
                }
                const auto reservation_time = std::chrono::steady_clock::now();
                const auto reservation_wait = reservation_time - wait_start;
                _enqueueWait += reservation_wait;
                _maxEnqueueWait = std::max(_maxEnqueueWait, reservation_wait);
                if (_stopping || (cancelFlag && cancelFlag->load()))
                {
                    return {};
                }

                if (!_wallStarted)
                {
                    _wallStarted = true;
                    _wallStart = reservation_time;
                }
                _lastActivity = reservation_time;
                ++_producerReservations;
                ++_residentTasks;
                _residentBytes += _producerReservationBytes;
                _peakResidentTasks = std::max(_peakResidentTasks, _residentTasks);
                _peakResidentBytes = std::max(_peakResidentBytes, _residentBytes);
            }
            return ProducerReservation(this, _producerReservationBytes);
        }

        void
        enqueue(ProducerReservation&& reservation, int frameIndex, const DepthFrameResult& result, QString stageLabel)
        {
            if (reservation._owner != this)
            {
                _failed = true;
                LOG_ERROR(QStringLiteral("[MVS] 深度产物保存队列收到无效的生产者内存预约"));
                return;
            }

            uint64_t reserved_bytes_for_log = 0;
            uint64_t resident_bytes = 0;
            bool queued = false;
            bool reservation_too_small = false;
            QString enqueue_exception;
            try
            {
                resident_bytes = depthFrameResultResidentBytes(result);
                SaveTask task{frameIndex, result, std::move(stageLabel), resident_bytes};

                std::lock_guard<std::mutex> lock(_mutex);
                const uint64_t reserved_bytes = reservation._residentBytes;
                reserved_bytes_for_log = reserved_bytes;
                if (_stopping)
                {
                    releaseProducerReservationLocked(reserved_bytes);
                    reservation.disarm();
                }
                else if (resident_bytes > reserved_bytes)
                {
                    // The reservation is a conservative upper bound for all matrices
                    // currently retained by DepthFrameResult. Failing closed here keeps
                    // the configured/4 GiB limit a real upper bound if that structure is
                    // extended without updating the estimate.
                    releaseProducerReservationLocked(reserved_bytes);
                    reservation.disarm();
                    _failed = true;
                    ++_failedTasks;
                    reservation_too_small = true;
                }
                else
                {
                    // Commit the queue insertion before transferring reservation
                    // accounting. std::deque::push_back has a strong exception
                    // guarantee, so an allocation failure leaves the armed RAII
                    // reservation to release the producer slot and bytes.
                    _tasks.push_back(std::move(task));
                    --_producerReservations;
                    _residentBytes -= std::min(_residentBytes, reserved_bytes);
                    _residentBytes += resident_bytes;
                    reservation.disarm();
                    _lastActivity = std::chrono::steady_clock::now();
                    queued = true;
                }
            }
            catch (const std::exception& exception)
            {
                enqueue_exception = QString::fromUtf8(exception.what());
            }
            catch (...)
            {
                enqueue_exception = QStringLiteral("未知异常");
            }
            if (!enqueue_exception.isEmpty())
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _failed = true;
                    ++_failedTasks;
                }
                LOG_ERROR(
                    QStringLiteral("[MVS] 深度帧 %1 保存任务入队异常：%2").arg(frameIndex).arg(enqueue_exception));
            }
            _capacityCv.notify_all();
            if (reservation_too_small)
            {
                LOG_ERROR(QStringLiteral("[MVS] 深度帧 %1 保存预约不足: reserved=%2 MiB actual=%3 MiB；"
                                         "为保持保存队列内存硬上限，本帧未入队")
                              .arg(frameIndex)
                              .arg(static_cast<double>(reserved_bytes_for_log) / (1024.0 * 1024.0), 0, 'f', 1)
                              .arg(static_cast<double>(resident_bytes) / (1024.0 * 1024.0), 0, 'f', 1));
            }
            if (queued)
            {
                _cv.notify_one();
            }
        }

        bool waitUntilIdle(const std::atomic<bool>* cancelFlag = nullptr)
        {
            std::unique_lock<std::mutex> lock(_mutex);
            const auto idle = [this]() { return _tasks.empty() && _activeTasks == 0 && _producerReservations == 0; };
            while (!idle())
            {
                if (cancelFlag && cancelFlag->load())
                {
                    return false;
                }
                if (cancelFlag)
                {
                    _idleCv.wait_for(lock, std::chrono::milliseconds(25));
                }
                else
                {
                    _idleCv.wait(lock);
                }
            }
            return true;
        }

        void cancel()
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _dropPendingTasks = true;
                _stopping = true;
                dropPendingTasksLocked();
                if (_wallStarted)
                {
                    _lastActivity = std::chrono::steady_clock::now();
                }
                if (_activeTasks == 0 && _producerReservations == 0)
                {
                    _idleCv.notify_all();
                }
            }
            _capacityCv.notify_all();
            _cv.notify_all();
        }

        void stop()
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stopping = true;
            }
            _capacityCv.notify_all();
            _cv.notify_all();
            for (std::thread& worker : _workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
            logSummaryOnce();
        }

        bool failed() const
        {
            return _failed.load();
        }

    private:
        struct MatAllocationSpan
        {
            std::uintptr_t begin = 0;
            std::uintptr_t end = 0;
        };

        struct SaveTask
        {
            int frameIndex = -1;
            DepthFrameResult result;
            QString stageLabel;
            uint64_t residentBytes = 0;
        };

        static void appendMatAllocationSpan(const cv::Mat& matrix, std::vector<MatAllocationSpan>& spans)
        {
            if (matrix.empty() || matrix.datastart == nullptr || matrix.dataend == nullptr)
            {
                return;
            }

            const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(matrix.datastart);
            const std::uintptr_t end = reinterpret_cast<std::uintptr_t>(matrix.dataend);
            if (end > begin)
            {
                spans.push_back({begin, end});
            }
        }

        static void appendSharedMatAllocationSpan(const QSharedPointer<cv::Mat>& matrix,
                                                  std::vector<MatAllocationSpan>& spans)
        {
            if (matrix)
            {
                appendMatAllocationSpan(*matrix, spans);
            }
        }

        static uint64_t depthFrameResultResidentBytes(const DepthFrameResult& result)
        {
            std::vector<MatAllocationSpan> spans;
            spans.reserve(32 + result.intermediatePyramidLevels.size() * 6);
            appendSharedMatAllocationSpan(result.depthMap, spans);
            appendSharedMatAllocationSpan(result.confidence, spans);
            appendSharedMatAllocationSpan(result.normalMap, spans);
            appendSharedMatAllocationSpan(result.supportCount, spans);
            appendSharedMatAllocationSpan(result.photometricSourceMask, spans);
            appendSharedMatAllocationSpan(result.geometrySupportCount, spans);
            appendSharedMatAllocationSpan(result.geometrySourceMask, spans);
            appendSharedMatAllocationSpan(result.inverseDepthMean, spans);
            appendSharedMatAllocationSpan(result.inverseDepthRelativeSpread, spans);
            appendSharedMatAllocationSpan(result.adaptiveGeometrySupportWeight, spans);
            appendSharedMatAllocationSpan(result.adaptiveGeometryEffectiveViewCount, spans);
            appendSharedMatAllocationSpan(result.adaptiveGeometryConflictRatio, spans);
            appendSharedMatAllocationSpan(result.crossViewRepairedMask, spans);
            appendSharedMatAllocationSpan(result.targetedGapRecoveredMask, spans);
            appendSharedMatAllocationSpan(result.residualReestimatedMask, spans);
            appendSharedMatAllocationSpan(result.depthProvenance, spans);
            appendSharedMatAllocationSpan(result.missingReasonMap, spans);
            appendSharedMatAllocationSpan(result.validMask, spans);
            appendSharedMatAllocationSpan(result.supportRegionMask, spans);
            for (const DepthLevelResult& level : result.intermediatePyramidLevels)
            {
                appendMatAllocationSpan(level.depth, spans);
                appendMatAllocationSpan(level.normalMap, spans);
                appendMatAllocationSpan(level.confidence, spans);
                appendMatAllocationSpan(level.supportCount, spans);
                appendMatAllocationSpan(level.uncertainty, spans);
                appendMatAllocationSpan(level.validMask, spans);
            }

            if (spans.empty())
            {
                return 0;
            }
            std::sort(spans.begin(),
                      spans.end(),
                      [](const MatAllocationSpan& left, const MatAllocationSpan& right)
                      { return left.begin < right.begin || (left.begin == right.begin && left.end < right.end); });

            uint64_t resident_bytes = 0;
            std::uintptr_t allocation_begin = spans.front().begin;
            std::uintptr_t allocation_end = spans.front().end;
            for (size_t index = 1; index < spans.size(); ++index)
            {
                const MatAllocationSpan& span = spans[index];
                if (span.begin <= allocation_end)
                {
                    allocation_end = std::max(allocation_end, span.end);
                    continue;
                }
                resident_bytes += static_cast<uint64_t>(allocation_end - allocation_begin);
                allocation_begin = span.begin;
                allocation_end = span.end;
            }
            resident_bytes += static_cast<uint64_t>(allocation_end - allocation_begin);
            return resident_bytes;
        }

        bool canAcceptLocked(uint64_t taskResidentBytes) const
        {
            if (taskResidentBytes > _maxResidentBytes || _residentTasks >= _maxResidentTasks ||
                _residentBytes >= _maxResidentBytes)
            {
                return false;
            }
            return taskResidentBytes <= _maxResidentBytes - _residentBytes;
        }

        void releaseProducerReservationLocked(uint64_t residentBytes)
        {
            if (_producerReservations == 0 || _residentTasks == 0)
            {
                return;
            }
            --_producerReservations;
            --_residentTasks;
            _residentBytes -= std::min(_residentBytes, residentBytes);
            _lastActivity = std::chrono::steady_clock::now();
            if (_tasks.empty() && _activeTasks == 0 && _producerReservations == 0)
            {
                _idleCv.notify_all();
            }
        }

        void releaseProducerReservation(uint64_t residentBytes)
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                releaseProducerReservationLocked(residentBytes);
            }
            _capacityCv.notify_all();
        }

        void dropPendingTasksLocked()
        {
            for (const SaveTask& task : _tasks)
            {
                _residentBytes -= std::min(_residentBytes, task.residentBytes);
            }
            _residentTasks -= std::min(_residentTasks, _tasks.size());
            _tasks.clear();
        }

        void logSummaryOnce()
        {
            std::chrono::steady_clock::duration wall;
            std::chrono::steady_clock::duration enqueue_wait;
            std::chrono::steady_clock::duration max_enqueue_wait;
            std::chrono::steady_clock::duration busy;
            uint64_t peak_resident_bytes = 0;
            size_t peak_resident_tasks = 0;
            size_t saved_tasks = 0;
            size_t failed_tasks = 0;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_summaryLogged)
                {
                    return;
                }
                _summaryLogged = true;
                if (_wallStarted && _lastActivity >= _wallStart)
                {
                    wall = _lastActivity - _wallStart;
                }
                enqueue_wait = _enqueueWait;
                max_enqueue_wait = _maxEnqueueWait;
                busy = _busy;
                peak_resident_bytes = _peakResidentBytes;
                peak_resident_tasks = _peakResidentTasks;
                saved_tasks = _savedTasks;
                failed_tasks = _failedTasks;
            }

            const double wall_ms = std::chrono::duration<double, std::milli>(wall).count();
            const double enqueue_wait_ms = std::chrono::duration<double, std::milli>(enqueue_wait).count();
            const double max_enqueue_wait_ms = std::chrono::duration<double, std::milli>(max_enqueue_wait).count();
            const double busy_ms = std::chrono::duration<double, std::milli>(busy).count();
            LOG_INFO(QStringLiteral("[MVS] 深度产物保存队列统计: workers=%1 task_limit=%2 "
                                    "byte_limit=%3 MiB wall=%4 ms enqueue_wait_total=%5 ms "
                                    "enqueue_wait_max=%6 ms busy=%7 ms peak_resident_tasks=%8 "
                                    "peak_resident=%9 MiB saved=%10 failed=%11")
                         .arg(_workerCount)
                         .arg(_maxResidentTasks)
                         .arg(static_cast<double>(_maxResidentBytes) / (1024.0 * 1024.0), 0, 'f', 1)
                         .arg(wall_ms, 0, 'f', 1)
                         .arg(enqueue_wait_ms, 0, 'f', 1)
                         .arg(max_enqueue_wait_ms, 0, 'f', 1)
                         .arg(busy_ms, 0, 'f', 1)
                         .arg(peak_resident_tasks)
                         .arg(static_cast<double>(peak_resident_bytes) / (1024.0 * 1024.0), 0, 'f', 1)
                         .arg(saved_tasks)
                         .arg(failed_tasks));
        }

        void run()
        {
            for (;;)
            {
                SaveTask task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    _cv.wait(lock, [this]() { return _stopping || !_tasks.empty(); });

                    if (_tasks.empty())
                    {
                        if (_stopping)
                        {
                            break;
                        }
                        continue;
                    }

                    if (_dropPendingTasks)
                    {
                        dropPendingTasksLocked();
                        if (_activeTasks == 0 && _producerReservations == 0)
                        {
                            _idleCv.notify_all();
                        }
                        if (_stopping)
                        {
                            break;
                        }
                        continue;
                    }

                    task = std::move(_tasks.front());
                    _tasks.pop_front();
                    ++_activeTasks;
                }

                const auto busy_start = std::chrono::steady_clock::now();
                bool saved = false;
                bool save_exception_occurred = false;
                std::array<char, 512> save_exception{};
                try
                {
                    saved = _saveFn(task.frameIndex, task.result, task.stageLabel);
                }
                catch (const std::exception& exception)
                {
                    save_exception_occurred = true;
                    std::snprintf(save_exception.data(), save_exception.size(), "%s", exception.what());
                }
                catch (...)
                {
                    save_exception_occurred = true;
                    std::snprintf(save_exception.data(), save_exception.size(), "%s", "unknown exception");
                }
                const auto busy_end = std::chrono::steady_clock::now();
                const auto busy_elapsed = busy_end - busy_start;
                if (!saved)
                {
                    _failed = true;
                }

                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _busy += busy_elapsed;
                    if (saved)
                    {
                        ++_savedTasks;
                    }
                    else
                    {
                        ++_failedTasks;
                    }
                    --_activeTasks;
                    --_residentTasks;
                    _residentBytes -= std::min(_residentBytes, task.residentBytes);
                    _lastActivity = std::max(_lastActivity, busy_end);
                    if (_tasks.empty() && _activeTasks == 0 && _producerReservations == 0)
                    {
                        _idleCv.notify_all();
                    }
                }
                _capacityCv.notify_all();
                if (save_exception_occurred)
                {
                    // Accounting is already complete. Keep diagnostic formatting
                    // behind a second exception boundary so low-memory logging can
                    // never escape this std::thread.
                    try
                    {
                        LOG_ERROR(QStringLiteral("[MVS] 深度帧 %1 保存线程异常：%2")
                                      .arg(task.frameIndex)
                                      .arg(QString::fromUtf8(save_exception.data())));
                    }
                    catch (...)
                    {
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_tasks.empty() && _activeTasks == 0 && _producerReservations == 0)
                {
                    _idleCv.notify_all();
                }
            }
            _capacityCv.notify_all();
        }

        SaveFn _saveFn;
        std::deque<SaveTask> _tasks;
        mutable std::mutex _mutex;
        std::condition_variable _cv;
        std::condition_variable _capacityCv;
        std::condition_variable _idleCv;
        std::vector<std::thread> _workers;
        std::atomic<bool> _failed{false};
        size_t _workerCount = 1;
        size_t _maxResidentTasks = 1;
        uint64_t _maxResidentBytes = 1;
        uint64_t _producerReservationBytes = 1;
        uint64_t _residentBytes = 0;
        uint64_t _peakResidentBytes = 0;
        size_t _residentTasks = 0;
        size_t _producerReservations = 0;
        size_t _peakResidentTasks = 0;
        size_t _savedTasks = 0;
        size_t _failedTasks = 0;
        std::chrono::steady_clock::duration _enqueueWait{};
        std::chrono::steady_clock::duration _maxEnqueueWait{};
        std::chrono::steady_clock::duration _busy{};
        std::chrono::steady_clock::time_point _wallStart{};
        std::chrono::steady_clock::time_point _lastActivity{};
        bool _stopping = false;
        bool _dropPendingTasks = false;
        bool _summaryLogged = false;
        bool _wallStarted = false;
        int _activeTasks = 0;
    };
} // namespace xjw::mvs::pipeline_detail
