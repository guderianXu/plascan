#include "MvsPipelineService.h"

#include <gtest/gtest.h>
#include <future>
#include <thread>

using namespace xjw::mvs;
using xjw::task_runtime::WorkflowStatus;

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
