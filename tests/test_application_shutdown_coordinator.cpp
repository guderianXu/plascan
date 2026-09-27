#include "ApplicationShutdownCoordinator.h"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QThread>
#include <QTimer>

#include <functional>

namespace
{

    bool spinUntil(const std::function<bool()>& condition, int timeoutMs = 3000)
    {
        QElapsedTimer timer;
        timer.start();
        while (!condition() && timer.elapsed() < timeoutMs)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(2);
        }
        return condition();
    }

    TEST(ApplicationShutdownCoordinatorTest, KeepsEventLoopResponsiveAndRunsEachShutdownStageOnce)
    {
        bool heartbeat = false;
        bool project_operations_idle = false;
        bool gate_acquired = false;
        bool runtime_shutdown_requested = false;
        bool runtime_shutdown_complete = false;
        bool ready_to_close = false;
        int gate_attempts = 0;
        int drain_starts = 0;
        int runtime_shutdown_requests = 0;
        std::function<void()> finish_drain;

        xjw::gui::main_window::ApplicationShutdownCoordinator::Hooks hooks;
        hooks.requestResourceCancellation = [] {};
        hooks.projectOperationsIdle = [&project_operations_idle] { return project_operations_idle; };
        hooks.tryAcquireShutdownGate = [&]
        {
            ++gate_attempts;
            gate_acquired = true;
            return true;
        };
        hooks.projectTaskDrainInProgress = [] { return false; };
        hooks.startProjectTaskDrain = [&](std::function<void()> continuation)
        {
            ++drain_starts;
            finish_drain = std::move(continuation);
            return true;
        };
        hooks.requestRuntimeShutdown = [&]
        {
            ++runtime_shutdown_requests;
            runtime_shutdown_requested = true;
        };
        hooks.runtimeShutdownComplete = [&runtime_shutdown_complete] { return runtime_shutdown_complete; };

        xjw::gui::main_window::ApplicationShutdownCoordinator coordinator(std::move(hooks));
        QObject::connect(&coordinator,
                         &xjw::gui::main_window::ApplicationShutdownCoordinator::readyToClose,
                         [&ready_to_close] { ready_to_close = true; });

        ASSERT_TRUE(coordinator.requestShutdown());
        EXPECT_FALSE(coordinator.requestShutdown());
        QTimer::singleShot(0,
                           [&]
                           {
                               heartbeat = true;
                               project_operations_idle = true;
                           });

        ASSERT_TRUE(spinUntil([&] { return heartbeat && gate_acquired && static_cast<bool>(finish_drain); }));
        EXPECT_FALSE(runtime_shutdown_requested);
        EXPECT_FALSE(ready_to_close);

        finish_drain();
        ASSERT_TRUE(spinUntil([&runtime_shutdown_requested] { return runtime_shutdown_requested; }));
        EXPECT_FALSE(ready_to_close);

        QTimer::singleShot(0, [&runtime_shutdown_complete] { runtime_shutdown_complete = true; });
        ASSERT_TRUE(spinUntil([&ready_to_close] { return ready_to_close; }));
        EXPECT_TRUE(coordinator.isReadyToClose());
        EXPECT_FALSE(coordinator.isShutdownInProgress());
        EXPECT_EQ(gate_attempts, 1);
        EXPECT_EQ(drain_starts, 1);
        EXPECT_EQ(runtime_shutdown_requests, 1);
    }

} // namespace
