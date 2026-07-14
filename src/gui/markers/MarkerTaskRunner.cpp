#include "MarkerTaskRunner.h"

#include "detection/MarkerDetectorFactory.h"

#include <QtConcurrent>

#include <QFileInfo>
#include <QImageReader>
#include <QMutex>
#include <QMutexLocker>
#include <QPromise>
#include <QQueue>
#include <QRunnable>
#include <QSemaphore>
#include <QThread>
#include <QThreadPool>

#include <algorithm>
#include <stdexcept>

namespace xjw::gui::markers
{

struct MarkerTaskSharedState
{
    std::atomic_bool cancelRequested = false;
    std::atomic_int candidatesDetected = 0;
};

namespace
{

struct ImageDetectionResult
{
    QVector<control_points::MarkerDetectionObservation> observations;
    QString error;
};

QVector<control_points::MarkerDetection> detectFamily(
    control_points::MarkerTargetFamily family,
    const QImage &image,
    const QImage &mask,
    const control_points::MarkerDetectionOptions &options)
{
    const std::unique_ptr<control_points::MarkerDetector> detector =
        control_points::MarkerDetectorFactory::create(family);
    return detector->detect(image, mask, options);
}

QImage readImage(const QString &path)
{
    QImageReader reader(path);
    reader.setAutoTransform(false);
    return reader.read();
}

ImageDetectionResult detectImage(const MarkerDetectionImage &input,
                                 const MarkerDetectionJob &job,
                                 const std::shared_ptr<MarkerTaskSharedState> &state)
{
    ImageDetectionResult result;
    if (state->cancelRequested.load(std::memory_order_relaxed))
    {
        return result;
    }

    const QImage image = readImage(input.imagePath);
    if (image.isNull())
    {
        result.error = QStringLiteral("无法读取检测影像: %1").arg(input.imagePath);
        return result;
    }
    QImage mask;
    if (!input.maskPath.isEmpty())
    {
        mask = readImage(input.maskPath);
        if (mask.isNull())
        {
            result.error = QStringLiteral("无法读取检测蒙版: %1").arg(input.maskPath);
            return result;
        }
    }

    try
    {
        control_points::MarkerDetectionOptions options = job.detectorOptions;
        options.cancelRequested = &state->cancelRequested;
        // 外层已按影像并行，每个 AprilTag 实例使用单线程可避免 CPU 过度订阅。
        options.threadCount = 1;
        for (const control_points::MarkerTargetFamily family : job.targetFamilies)
        {
            if (state->cancelRequested.load(std::memory_order_relaxed))
            {
                return {};
            }
            const auto detections = detectFamily(family, image, mask, options);
            for (const control_points::MarkerDetection &detection : detections)
            {
                result.observations.push_back({input.imageId,
                                               input.imagePath,
                                               input.imageContentSignature,
                                               detection});
            }
            state->candidatesDetected.fetch_add(detections.size(), std::memory_order_relaxed);
        }
    }
    catch (const std::exception &exception)
    {
        result.error = QStringLiteral("检测影像 %1 失败: %2")
                           .arg(input.imagePath, QString::fromUtf8(exception.what()));
    }
    return result;
}

void runDetectionJob(QPromise<MarkerDetectionTaskResult> &promise,
                     MarkerDetectionJob job,
                     std::shared_ptr<MarkerTaskSharedState> state)
{
    MarkerDetectionTaskResult result;
    result.baseRevision = job.baseRevision;
    promise.setProgressRange(0, job.images.size());
    promise.setProgressValue(0);
    if (job.images.isEmpty())
    {
        promise.addResult(result);
        return;
    }

    const int hardware_threads = std::max(1, QThread::idealThreadCount());
    const int worker_count = std::clamp(job.maxConcurrentImages > 0
                                            ? job.maxConcurrentImages
                                            : hardware_threads,
                                        1,
                                        static_cast<int>(job.images.size()));
    QThreadPool pool;
    pool.setMaxThreadCount(worker_count);
    QVector<ImageDetectionResult> image_results(job.images.size());
    QSemaphore completed;
    QMutex completed_mutex;
    QQueue<int> completed_indices;

    for (int index = 0; index < job.images.size(); ++index)
    {
        pool.start(QRunnable::create([&, index]()
        {
            ImageDetectionResult detected = detectImage(job.images[index], job, state);
            {
                QMutexLocker locker(&completed_mutex);
                image_results[index] = std::move(detected);
                completed_indices.enqueue(index);
            }
            completed.release();
        }));
    }

    for (int completed_count = 1; completed_count <= job.images.size(); ++completed_count)
    {
        completed.acquire();
        int completed_index = -1;
        {
            QMutexLocker locker(&completed_mutex);
            completed_index = completed_indices.dequeue();
        }
        promise.setProgressValueAndText(completed_count, job.images[completed_index].imagePath);
    }
    pool.waitForDone();

    result.cancelled = state->cancelRequested.load(std::memory_order_relaxed);
    if (!result.cancelled)
    {
        for (ImageDetectionResult &image_result : image_results)
        {
            result.observations += image_result.observations;
            if (!image_result.error.isEmpty())
            {
                result.errors.push_back(image_result.error);
            }
        }
    }
    promise.addResult(result);
}

} // namespace

MarkerTaskRunner::MarkerTaskRunner(QObject *parent)
    : QObject(parent)
{
    connect(&_watcher, &QFutureWatcher<MarkerDetectionTaskResult>::progressValueChanged,
            this, [this](int value)
    {
        MarkerDetectionProgress progress;
        progress.imagesCompleted = value;
        progress.imageCount = _watcher.progressMaximum();
        progress.candidatesDetected = _state
            ? _state->candidatesDetected.load(std::memory_order_relaxed)
            : 0;
        progress.currentImage = _watcher.progressText();
        emit progressChanged(progress);
    });
    connect(&_watcher, &QFutureWatcher<MarkerDetectionTaskResult>::finished,
            this, [this]()
    {
        const MarkerDetectionTaskResult result = _watcher.result();
        _running = false;
        emit finished(result);
    });
}

MarkerTaskRunner::~MarkerTaskRunner()
{
    cancel();
}

bool MarkerTaskRunner::start(const MarkerDetectionJob &job)
{
    if (_running || job.targetFamilies.isEmpty())
    {
        return false;
    }
    _state = std::make_shared<MarkerTaskSharedState>();
    _running = true;
    _watcher.setFuture(QtConcurrent::run(runDetectionJob, job, _state));
    return true;
}

void MarkerTaskRunner::cancel()
{
    if (_state)
    {
        _state->cancelRequested.store(true, std::memory_order_relaxed);
    }
}

bool MarkerTaskRunner::isRunning() const noexcept
{
    return _running;
}

} // namespace xjw::gui::markers
