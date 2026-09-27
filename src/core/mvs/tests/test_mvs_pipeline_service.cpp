#include "MvsPipelineService.h"

#include <gtest/gtest.h>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>

using namespace xjw::mvs;
using xjw::task_runtime::WorkflowStatus;

namespace
{
    std::shared_ptr<const placamera::FramePinholeModel> makeCamera(const char* instanceId,
                                                                    const char* imageId,
                                                                    const char* frameId)
    {
        const placamera::FrameId frame(frameId);
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId(std::string(instanceId) + "-definition"),
            placamera::FrameIntrinsics{100.0, 100.0, 32.0, 24.0},
            {},
            placamera::PixelConvention::PixelCenter,
            frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(instanceId),
            placamera::ImageId(imageId),
            definition,
            {64, 48},
            placamera::Pose::create(frame,
                                    {0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }
}

TEST(MvsPipelineServiceContract, SynchronousFailureNeedsNoQObjectAndFinishesOnce)
{
    MvsPipelineService service;
    DepthGenConfig config;
    config.runDepthEstimation = false;
    service.setConfig(config);
    int finished_count = 0;
    int error_count = 0;
    const auto caller = std::this_thread::get_id();
    MvsPipelineEvents events;
    events.errorOccurred = [&](QString error)
    {
        EXPECT_FALSE(error.isEmpty());
        EXPECT_EQ(std::this_thread::get_id(), caller);
        ++error_count;
    };
    events.finished = [&](bool success)
    {
        EXPECT_FALSE(success);
        ++finished_count;
    };
    service.setEvents(std::move(events));
    const auto outcome = service.execute();
    EXPECT_EQ(outcome.status, WorkflowStatus::Failed);
    EXPECT_FALSE(outcome.error.message.empty());
    EXPECT_EQ(error_count, 1);
    EXPECT_EQ(finished_count, 1);
    service.execute();
    EXPECT_EQ(finished_count, 2);
}

TEST(MvsPipelineServiceContract, SharedCancellationIsNotClearedBySynchronousExecution)
{
    xjw::task_runtime::WorkflowControl control;
    control.cancellation = std::make_shared<std::atomic_bool>(true);
    MvsPipelineService service(control);
    int finished_count = 0;
    MvsPipelineEvents events;
    events.finished = [&](bool success)
    {
        EXPECT_FALSE(success);
        ++finished_count;
    };
    service.setEvents(std::move(events));
    EXPECT_EQ(service.execute().status, WorkflowStatus::Cancelled);
    EXPECT_TRUE(control.isCancelled());
    EXPECT_EQ(finished_count, 1);
}

TEST(MvsPipelineServiceContract, RejectsConcurrentExecutionWithoutChangingActiveOutcome)
{
    MvsPipelineService service;
    DepthGenConfig config;
    config.runDepthEstimation = false;
    service.setConfig(config);
    std::promise<void> entered;
    std::promise<void> release;
    auto release_future = release.get_future();
    MvsPipelineEvents events;
    events.errorOccurred = [&](QString)
    {
        entered.set_value();
        release_future.wait();
    };
    service.setEvents(std::move(events));
    auto first = std::async(std::launch::async, [&]() { return service.execute(); });
    entered.get_future().wait();
    const auto second = service.execute();
    EXPECT_EQ(second.status, WorkflowStatus::Failed);
    EXPECT_EQ(second.error.code, "already_running");
    release.set_value();
    EXPECT_EQ(first.get().error.code, "mvs_failed");
}

TEST(MvsPipelineServiceContract, CancellationAndExplicitResetRemainReusable)
{
    MvsPipelineService service;
    service.requestCancel();
    EXPECT_EQ(service.execute().status, WorkflowStatus::Cancelled);
    service.resetCancellation();
    DepthGenConfig config;
    config.runDepthEstimation = false;
    service.setConfig(config);
    EXPECT_EQ(service.execute().status, WorkflowStatus::Failed);
}

TEST(MvsPipelineServiceContract, UnsupportedBackendsFailBeforeImagePreparation)
{
    for (const PatchMatchBackend backend : {PatchMatchBackend::Cpu, PatchMatchBackend::OpenCl})
    {
        MvsPipelineService service;
        DepthGenConfig config;
        config.patchMatch.backend = backend;
        service.setConfig(config);
        CameraView view;
        view.imagePath = "missing-input-must-not-be-read.tif";
        view.camera = makeCamera("backend-instance", "backend-image", "backend-world");
        service.setViews({view});
        int finished_count = 0;
        int artifact_count = 0;
        MvsPipelineEvents events;
        events.finished = [&](bool success)
        {
            EXPECT_FALSE(success);
            ++finished_count;
        };
        events.depthMapArtifactSaved = [&](QJsonObject) { ++artifact_count; };
        service.setEvents(std::move(events));
        const auto outcome = service.execute();
        EXPECT_EQ(outcome.status, WorkflowStatus::Failed);
        EXPECT_TRUE(QString::fromStdString(outcome.error.message).contains(QStringLiteral("仅支持 CUDA")));
        EXPECT_EQ(finished_count, 1);
        EXPECT_EQ(artifact_count, 0);
    }
}

TEST(MvsPipelineServiceContract, InvalidExplicitCudaDeviceFailsWithoutFallback)
{
    MvsPipelineService service;
    DepthGenConfig config;
    config.patchMatch.backend = PatchMatchBackend::Cuda;
    config.patchMatch.cudaDeviceIndex = std::numeric_limits<int>::max();
    service.setConfig(config);
    const auto outcome = service.execute();
    EXPECT_EQ(outcome.status, WorkflowStatus::Failed);
    EXPECT_TRUE(QString::fromStdString(outcome.error.message).contains(QStringLiteral("设备编号无效")));
    EXPECT_TRUE(QString::fromStdString(outcome.error.message).contains(QStringLiteral("不会自动切换")));
    EXPECT_EQ(service.config().patchMatch.backend, PatchMatchBackend::Cuda);
}

TEST(MvsPipelineServiceContract, RejectsInvalidCameraBeforeImagePreparation)
{
    MvsPipelineService service;
    DepthGenConfig config;
    config.patchMatch.backend = PatchMatchBackend::Cpu;
    service.setConfig(config);

    CameraView view;
    view.imagePath = "missing-input-must-not-be-read.tif";
    service.setViews({view});

    int finished_count = 0;
    int progress_count = 0;
    QString error_message;
    MvsPipelineEvents events;
    events.errorOccurred = [&](QString message) { error_message = std::move(message); };
    events.progressChanged = [&](QString, float) { ++progress_count; };
    events.finished = [&](bool success)
    {
        EXPECT_FALSE(success);
        ++finished_count;
    };
    service.setEvents(std::move(events));

    const auto outcome = service.execute();
    EXPECT_EQ(outcome.status, WorkflowStatus::Failed);
    EXPECT_TRUE(error_message.contains(QStringLiteral("PlaCamera 模型缺失"))) << error_message.toStdString();
    EXPECT_EQ(finished_count, 1);
    EXPECT_EQ(progress_count, 0);
}

TEST(MvsPipelineServiceContract, RejectsMixedWorldFramesBeforeImagePreparation)
{
    MvsPipelineService service;
    DepthGenConfig config;
    config.patchMatch.backend = PatchMatchBackend::Cpu;
    service.setConfig(config);

    const auto make_view = [](const char* instance_id, const char* image_id, const char* frame_id)
    {
        CameraView view;
        view.imagePath = "missing-input-must-not-be-read.tif";
        view.camera = makeCamera(instance_id, image_id, frame_id);
        return view;
    };

    service.setViews({make_view("instance-a", "image-a", "world-a"),
                      make_view("instance-b", "image-b", "world-b")});

    int finished_count = 0;
    int progress_count = 0;
    QString error_message;
    MvsPipelineEvents events;
    events.errorOccurred = [&](QString message) { error_message = std::move(message); };
    events.progressChanged = [&](QString, float) { ++progress_count; };
    events.finished = [&](bool success)
    {
        EXPECT_FALSE(success);
        ++finished_count;
    };
    service.setEvents(std::move(events));

    const auto outcome = service.execute();
    EXPECT_EQ(outcome.status, WorkflowStatus::Failed);
    EXPECT_TRUE(error_message.contains(QStringLiteral("混用 world frame"))) << error_message.toStdString();
    EXPECT_EQ(finished_count, 1);
    EXPECT_EQ(progress_count, 0);
}
