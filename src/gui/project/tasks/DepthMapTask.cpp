#include "DepthMapTask.h"
#include <QtConcurrent/QtConcurrent>

namespace xjw::gui::tasks
{
    using mvs::MvsPipelineEvents;
    DepthMapTask::DepthMapTask(QObject* parent) : QObject(parent), _service(std::make_unique<MvsPipelineService>())
    {
        qRegisterMetaType<DepthFrameResult>("DepthFrameResult");
        qRegisterMetaType<QSharedPointer<cv::Mat>>("QSharedPointer<cv::Mat>");
        qRegisterMetaType<std::vector<DensePoint>>("std::vector<DensePoint>");
        MvsPipelineEvents events;
        events.depthMapReady = [this](DepthFrameResult result) { emit depthMapReady(std::move(result)); };
        events.depthMapSaved = [this](QString path, int w, int h, QString image)
        { emit depthMapSaved(std::move(path), w, h, std::move(image)); };
        events.depthMapArtifactSaved = [this](QJsonObject artifact)
        { emit depthMapArtifactSaved(std::move(artifact)); };
        events.pointCloudReady = [this](std::vector<DensePoint> cloud) { emit pointCloudReady(std::move(cloud)); };
        events.progressChanged = [this](QString stage, float ratio) { emit progressChanged(std::move(stage), ratio); };
        events.errorOccurred = [this](QString error) { emit errorOccurred(std::move(error)); };
        events.finished = [this](bool success) { emit finished(success); };
        _service->setEvents(std::move(events));
    }
    DepthMapTask::~DepthMapTask()
    {
        requestCancel();
        if (_backgroundFuture.isRunning())
        {
            _backgroundFuture.waitForFinished();
        }
    }
    void DepthMapTask::start()
    {
        if (_backgroundFuture.isRunning())
        {
            emit errorOccurred(QStringLiteral("深度图生成任务已经在运行，忽略重复启动请求"));
            return;
        }
        _service->resetCancellation();
        _backgroundFuture = QtConcurrent::run([this]() { _service->execute(); });
    }
    void DepthMapTask::setViews(const std::vector<CameraView>& value)
    {
        if (_backgroundFuture.isRunning())
        {
            emit errorOccurred(QStringLiteral("MVS 运行时不能修改输入或配置"));
            return;
        }
        _service->setViews(value);
    }
    void DepthMapTask::setSparseCloud(const SparseCloud& value)
    {
        if (_backgroundFuture.isRunning())
        {
            emit errorOccurred(QStringLiteral("MVS 运行时不能修改输入或配置"));
            return;
        }
        _service->setSparseCloud(value);
    }
    void DepthMapTask::setConfig(const DepthGenConfig& value)
    {
        if (_backgroundFuture.isRunning())
        {
            emit errorOccurred(QStringLiteral("MVS 运行时不能修改输入或配置"));
            return;
        }
        _service->setConfig(value);
    }
    void DepthMapTask::setSkippedFrameIndices(const std::vector<int>& value)
    {
        if (_backgroundFuture.isRunning())
        {
            emit errorOccurred(QStringLiteral("MVS 运行时不能修改输入或配置"));
            return;
        }
        _service->setSkippedFrameIndices(value);
    }
} // namespace xjw::gui::tasks
