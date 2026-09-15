#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    PatchMatchConfig
    MvsPipelineService::nextCudaRetryPatchMatchConfig(const PatchMatchConfig& config, int imageWidth, int imageHeight)
    {
        PatchMatchConfig retryConfig = config;
        const int current = std::max(1, retryConfig.downsampleFactor);
        int next = current < 4 ? current + 1 : current + 2;

        const int maxDim = std::max(imageWidth, imageHeight);
        if (maxDim >= 5000 && current <= 2)
        {
            next = std::max(next, 3);
        }

        retryConfig.downsampleFactor = std::min(next, 12);
        retryConfig.numIterations = std::max(1, retryConfig.numIterations - 1);
        retryConfig.patchHalf = std::max(3, retryConfig.patchHalf - 1);
        return retryConfig;
    }

    // =============================================================================
    void MvsPipelineService::setViews(const std::vector<CameraView>& views)
    {
        _views = views;
        _skipFrameMask.assign(_views.size(), 0);
        {
            std::lock_guard<std::mutex> lock(_preparedRasterArtifactsMutex);
            _preparedRasterArtifacts.assign(_views.size(), MvsPreparedRasterArtifact{});
        }
        clearFrameCaches();
    }

    void MvsPipelineService::setSparseCloud(const SparseCloud& sparse)
    {
        _sparse = sparse;
        clearFrameCaches();
    }

    void MvsPipelineService::setConfig(const DepthGenConfig& config)
    {
        _config = config;
    }

    void MvsPipelineService::setSkippedFrameIndices(const std::vector<int>& indices)
    {
        _skipFrameMask.assign(_views.size(), 0);
        for (int index : indices)
        {
            if (index >= 0 && index < static_cast<int>(_skipFrameMask.size()))
            {
                _skipFrameMask[static_cast<size_t>(index)] = 1;
            }
        }
    }

    void MvsPipelineService::emitFinishedOnce(bool success)
    {
        bool expected = false;
        if (_finishedEmitted.compare_exchange_strong(expected, true))
        {
            finished(success);
        }
    }

    void MvsPipelineService::clearRuntimeCachesAfterFailure()
    {
        _imageCache.reset();
        clearFrameCaches();
        releaseStoredDepthFramePixelStorage(_depthFrames);

        std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
        _filteredDepths.clear();
    }

    void MvsPipelineService::runInBackground()
    {
        detail::runDepthMapBackgroundTaskWithExceptionBoundary([this]() { runInBackgroundImpl(); },
                                                               [this](const QString& message)
                                                               {
                                                                   clearRuntimeCachesAfterFailure();
                                                                   if (_cancelled.load(std::memory_order_relaxed))
                                                                   {
                                                                       emitFinishedOnce(false);
                                                                       return;
                                                                   }
                                                                   LOG_ERROR(QStringLiteral("[MVS] %1").arg(message));
                                                                   errorOccurred(message);
                                                                   emitFinishedOnce(false);
                                                               });
    }
} // namespace xjw::mvs
#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    MvsPipelineService::MvsPipelineService(task_runtime::WorkflowControl control)
        : _cancellation(control.cancellation ? control.cancellation : std::make_shared<std::atomic_bool>(false)),
          _cancelled(*_cancellation), _control(std::move(control))
    {
    }
    MvsPipelineService::~MvsPipelineService()
    {
        PatchMatchDepthEstimator::cleanupGpuImageCache();
        PatchMatchDepthEstimator::cleanupOpenClResources();
    }
    void MvsPipelineService::setEvents(MvsPipelineEvents events)
    {
        _events = std::move(events);
    }
    task_runtime::WorkflowOutcome MvsPipelineService::execute()
    {
        std::unique_lock<std::mutex> lock(_executionMutex, std::try_to_lock);
        if (!lock.owns_lock())
        {
            return {task_runtime::WorkflowStatus::Failed,
                    {"already_running", "MVS pipeline is already running", "start", {}}};
        }
        _finishedEmitted = false;
        _depthFrames.clear();
        _outcome = {};
        if (_cancelled.load())
            emitFinishedOnce(false);
        else
            runInBackground();
        return _outcome;
    }
    void MvsPipelineService::depthMapReady(DepthFrameResult result)
    {
        if (_events.depthMapReady)
            _events.depthMapReady(std::move(result));
    }
    void MvsPipelineService::depthMapSaved(QString path, int width, int height, QString image)
    {
        if (_events.depthMapSaved)
            _events.depthMapSaved(std::move(path), width, height, std::move(image));
    }
    void MvsPipelineService::depthMapArtifactSaved(QJsonObject artifact)
    {
        if (_events.depthMapArtifactSaved)
            _events.depthMapArtifactSaved(std::move(artifact));
    }
    void MvsPipelineService::pointCloudReady(std::vector<DensePoint> cloud)
    {
        if (_events.pointCloudReady)
            _events.pointCloudReady(std::move(cloud));
    }
    void MvsPipelineService::progressChanged(QString stage, float ratio)
    {
        _control.reportProgress(stage.toUtf8().toStdString(), ratio);
        if (_events.progressChanged)
            _events.progressChanged(std::move(stage), ratio);
    }
    void MvsPipelineService::errorOccurred(QString message)
    {
        _outcome.error = {"mvs_failed", message.toUtf8().toStdString(), "depth", _outputDir};
        if (_events.errorOccurred)
            _events.errorOccurred(std::move(message));
    }
    void MvsPipelineService::finished(bool success)
    {
        _outcome.status = success ? task_runtime::WorkflowStatus::Succeeded
                                  : (_cancelled.load() ? task_runtime::WorkflowStatus::Cancelled
                                                       : task_runtime::WorkflowStatus::Failed);
        if (_events.finished)
            _events.finished(success);
    }

} // namespace xjw::mvs
