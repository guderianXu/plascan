#include "TaskControl.h"
#include "TaskJournal.h"
#include "TaskScheduler.h"
#include "TaskTypes.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{

    using namespace std::chrono_literals;
    using xjw::task_runtime::IProjectEpochGuard;
    using xjw::task_runtime::ITaskExecutor;
    using xjw::task_runtime::ProjectAccess;
    using xjw::task_runtime::TaskCheckpoint;
    using xjw::task_runtime::TaskControlDecision;
    using xjw::task_runtime::TaskControlSource;
    using xjw::task_runtime::TaskControlToken;
    using xjw::task_runtime::TaskDefinition;
    using xjw::task_runtime::TaskExecutionContext;
    using xjw::task_runtime::TaskExecutionOutcome;
    using xjw::task_runtime::TaskExecutionStatus;
    using xjw::task_runtime::TaskJournal;
    using xjw::task_runtime::TaskProgress;
    using xjw::task_runtime::TaskResult;
    using xjw::task_runtime::TaskRunSnapshot;
    using xjw::task_runtime::TaskScheduler;
    using xjw::task_runtime::TaskSchedulerLimits;
    using xjw::task_runtime::TaskState;

    TaskDefinition makeTask(std::string id, int priority = 0)
    {
        TaskDefinition definition;
        definition.taskId = std::move(id);
        definition.kind = "test";
        definition.displayName = definition.taskId;
        definition.priority = priority;
        definition.capabilities.canPause = true;
        definition.capabilities.canCheckpoint = true;
        return definition;
    }

    class RecordingExecutor final : public ITaskExecutor
    {
    public:
        explicit RecordingExecutor(std::vector<std::string>* order = nullptr) : _order(order)
        {
        }

        TaskExecutionOutcome execute(const TaskDefinition& definition, TaskExecutionContext&) override
        {
            if (_order)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _order->push_back(definition.taskId);
            }
            return {TaskExecutionStatus::Succeeded, std::nullopt, TaskResult{"done", {}}, std::nullopt};
        }

    private:
        std::vector<std::string>* _order = nullptr;
        std::mutex _mutex;
    };

    class CheckpointExecutor final : public ITaskExecutor
    {
    public:
        TaskExecutionOutcome execute(const TaskDefinition&, TaskExecutionContext& context) override
        {
            std::uint64_t completed = context.checkpoint ? context.checkpoint->completedUnits : 0;
            for (; completed < 20; ++completed)
            {
                const TaskControlDecision decision = context.control.pollAtSafePoint("unit_boundary");
                if (decision == TaskControlDecision::Cancel)
                {
                    return {TaskExecutionStatus::Cancelled};
                }
                if (decision == TaskControlDecision::Pause)
                {
                    TaskCheckpoint checkpoint;
                    checkpoint.location = "memory://checkpoint";
                    checkpoint.inputSignature = "input-v1";
                    checkpoint.completedUnits = completed;
                    checkpoint.totalUnits = 20;
                    context.saveCheckpoint(checkpoint);
                    return {TaskExecutionStatus::Paused, checkpoint};
                }
                context.reportProgress(
                    TaskProgress{"unit", completed + 1, 20, static_cast<double>(completed + 1) / 20.0});
                std::this_thread::sleep_for(3ms);
            }
            return {
                TaskExecutionStatus::Succeeded, std::nullopt, TaskResult{"complete", "memory://result"}, std::nullopt};
        }
    };

    class ConcurrencyExecutor final : public ITaskExecutor
    {
    public:
        TaskExecutionOutcome execute(const TaskDefinition&, TaskExecutionContext&) override
        {
            const int active = _active.fetch_add(1) + 1;
            int observed = _maximum.load();
            while (active > observed && !_maximum.compare_exchange_weak(observed, active))
            {
            }
            std::this_thread::sleep_for(30ms);
            _active.fetch_sub(1);
            return {TaskExecutionStatus::Succeeded};
        }

        int maximum() const
        {
            return _maximum.load();
        }

    private:
        std::atomic_int _active{0};
        std::atomic_int _maximum{0};
    };

    class RejectingEpochGuard final : public IProjectEpochGuard
    {
    public:
        bool isCurrent(const TaskDefinition&) const override
        {
            return false;
        }
    };

    TEST(TaskTypesTest, ValidatesExplicitStateTransitions)
    {
        using xjw::task_runtime::canTransitionTaskState;
        using xjw::task_runtime::isTerminalTaskState;

        EXPECT_TRUE(canTransitionTaskState(TaskState::Queued, TaskState::Running));
        EXPECT_TRUE(canTransitionTaskState(TaskState::Running, TaskState::PauseRequested));
        EXPECT_TRUE(canTransitionTaskState(TaskState::PauseRequested, TaskState::Paused));
        EXPECT_TRUE(canTransitionTaskState(TaskState::Paused, TaskState::Queued));
        EXPECT_FALSE(canTransitionTaskState(TaskState::Succeeded, TaskState::Running));
        EXPECT_FALSE(canTransitionTaskState(TaskState::Running, TaskState::Queued));
        EXPECT_TRUE(isTerminalTaskState(TaskState::Succeeded));
        EXPECT_FALSE(isTerminalTaskState(TaskState::Interrupted));
    }

    TEST(TaskControlTest, CancellationWakesLegacyPausedWait)
    {
        TaskControlToken token;
        TaskControlSource source(token);
        source.requestPause();
        std::atomic<TaskControlDecision> decision{TaskControlDecision::Continue};
        std::thread waiter([&token, &decision] { decision = token.waitIfPaused("legacy_boundary"); });
        std::this_thread::sleep_for(10ms);
        source.requestCancellation();
        waiter.join();
        EXPECT_EQ(decision.load(), TaskControlDecision::Cancel);
    }

    TEST(TaskSchedulerTest, RunsDependenciesBeforeHigherPriorityDependentTask)
    {
        std::vector<std::string> order;
        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("test", std::make_shared<RecordingExecutor>(&order));

        TaskDefinition first = makeTask("prepare", 1);
        TaskDefinition second = makeTask("publish", 100);
        second.dependencies.push_back("prepare");
        const auto submitted = scheduler.submitBatch({second, first});
        ASSERT_TRUE(submitted.accepted) << submitted.error;
        ASSERT_EQ(submitted.runIds.size(), 2u);
        EXPECT_TRUE(scheduler.waitForState(submitted.runIds[0], TaskState::Succeeded, 2s));
        ASSERT_EQ(order.size(), 2u);
        EXPECT_EQ(order[0], "prepare");
        EXPECT_EQ(order[1], "publish");
    }

    TEST(TaskSchedulerTest, PausesWithCheckpointReleasesWorkerAndResumesSameRun)
    {
        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("test", std::make_shared<CheckpointExecutor>());
        const auto submitted = scheduler.submit(makeTask("checkpointed"));
        ASSERT_TRUE(submitted.accepted);
        const std::string run_id = submitted.runIds.front();
        ASSERT_TRUE(scheduler.waitForState(run_id, TaskState::Running, 1s));

        const auto pause = scheduler.requestPause(run_id);
        ASSERT_TRUE(pause.accepted) << pause.error;
        ASSERT_TRUE(scheduler.waitForState(run_id, TaskState::Paused, 2s));
        const auto paused = scheduler.snapshot(run_id);
        ASSERT_TRUE(paused.has_value());
        ASSERT_TRUE(paused->checkpoint.has_value());
        EXPECT_LT(paused->checkpoint->completedUnits, 20u);

        const auto resumed = scheduler.resume(run_id, paused->revision);
        ASSERT_TRUE(resumed.accepted) << resumed.error;
        EXPECT_TRUE(scheduler.waitForState(run_id, TaskState::Succeeded, 2s));
        const auto completed = scheduler.snapshot(run_id);
        ASSERT_TRUE(completed.has_value());
        EXPECT_EQ(completed->attemptId, 1u);
        ASSERT_TRUE(completed->result.has_value());
        EXPECT_EQ(completed->result->outputLocation, "memory://result");
    }

    TEST(TaskSchedulerTest, RejectsStaleCommandRevision)
    {
        TaskScheduler scheduler({1, {}});
        TaskDefinition task = makeTask("no-executor");
        task.kind = "missing";
        const auto submitted = scheduler.submit(task);
        ASSERT_TRUE(submitted.accepted);
        const auto queued = scheduler.snapshot(submitted.runIds.front());
        ASSERT_TRUE(queued.has_value());

        const auto changed = scheduler.setPriority(queued->runId, 4, queued->revision);
        ASSERT_TRUE(changed.accepted);
        const auto stale = scheduler.setPriority(queued->runId, 5, queued->revision);
        EXPECT_FALSE(stale.accepted);
        EXPECT_EQ(stale.error, "revision_conflict");
    }

    TEST(TaskSchedulerTest, RejectsDependencyCyclesAtomically)
    {
        TaskScheduler scheduler({1, {}});
        TaskDefinition first = makeTask("first");
        TaskDefinition second = makeTask("second");
        first.dependencies.push_back("second");
        second.dependencies.push_back("first");
        const auto submitted = scheduler.submitBatch({first, second});
        EXPECT_FALSE(submitted.accepted);
        EXPECT_TRUE(scheduler.snapshots().empty());
    }

    TEST(TaskSchedulerTest, PropagatesDependencyFailureThroughPendingChain)
    {
        class FailingExecutor final : public ITaskExecutor
        {
        public:
            TaskExecutionOutcome execute(const TaskDefinition&, TaskExecutionContext&) override
            {
                return {TaskExecutionStatus::Failed};
            }
        };

        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("fail", std::make_shared<FailingExecutor>());
        TaskDefinition first = makeTask("first");
        first.kind = "fail";
        TaskDefinition second = makeTask("second");
        second.dependencies = {"first"};
        TaskDefinition third = makeTask("third");
        third.dependencies = {"second"};
        const auto submitted = scheduler.submitBatch({third, second, first});
        ASSERT_TRUE(submitted.accepted) << submitted.error;
        ASSERT_TRUE(scheduler.waitForState(submitted.runIds[0], TaskState::Failed, 2s));
        const auto failed = scheduler.snapshot(submitted.runIds[0]);
        ASSERT_TRUE(failed.has_value());
        ASSERT_TRUE(failed->error.has_value());
        EXPECT_EQ(failed->error->code, "dependency_failed");
    }

    TEST(TaskSchedulerTest, KeepsSuccessfulHistoryNeededByActiveDependent)
    {
        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("test", std::make_shared<RecordingExecutor>());
        TaskDefinition first = makeTask("first");
        TaskDefinition second = makeTask("second");
        second.kind = "missing";
        second.dependencies = {"first"};
        const auto submitted = scheduler.submitBatch({second, first});
        ASSERT_TRUE(submitted.accepted) << submitted.error;
        ASSERT_TRUE(scheduler.waitForState(submitted.runIds[1], TaskState::Succeeded, 2s));
        EXPECT_EQ(scheduler.clearTerminalRuns(), 0u);
        EXPECT_TRUE(scheduler.snapshot(submitted.runIds[1]).has_value());
    }

    TEST(TaskSchedulerTest, RejectsResultFromObsoleteProjectGeneration)
    {
        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("test", std::make_shared<RecordingExecutor>());
        scheduler.setProjectEpochGuard(std::make_shared<RejectingEpochGuard>());
        TaskDefinition task = makeTask("stale-project");
        task.projectKey = "/tmp/project.plascan";
        task.chunkId = "chunk-a";
        task.projectGeneration = 3;
        const auto submitted = scheduler.submit(task);
        ASSERT_TRUE(submitted.accepted);
        ASSERT_TRUE(scheduler.waitForState(submitted.runIds.front(), TaskState::Failed, 2s));
        const auto failed = scheduler.snapshot(submitted.runIds.front());
        ASSERT_TRUE(failed.has_value());
        ASSERT_TRUE(failed->error.has_value());
        EXPECT_EQ(failed->error->code, "stale_project_generation");
    }

    TEST(TaskSchedulerTest, ClearsOnlyTerminalHistory)
    {
        TaskScheduler scheduler({1, {}});
        scheduler.registerExecutor("test", std::make_shared<RecordingExecutor>());
        const auto completed = scheduler.submit(makeTask("completed"));
        ASSERT_TRUE(completed.accepted);
        ASSERT_TRUE(scheduler.waitForState(completed.runIds.front(), TaskState::Succeeded, 2s));
        TaskDefinition waiting = makeTask("waiting");
        waiting.kind = "missing";
        const auto blocked = scheduler.submit(waiting);
        ASSERT_TRUE(blocked.accepted);
        EXPECT_EQ(scheduler.clearTerminalRuns(), 1u);
        EXPECT_FALSE(scheduler.snapshot(completed.runIds.front()).has_value());
        EXPECT_TRUE(scheduler.snapshot(blocked.runIds.front()).has_value());
    }

    TEST(TaskSchedulerTest, RejectsDuplicateTaskIdsDuringRestore)
    {
        TaskRunSnapshot first;
        first.runId = "run-a";
        first.definition = makeTask("same-task");
        TaskRunSnapshot second = first;
        second.runId = "run-b";
        TaskScheduler scheduler({1, {}});
        const auto restored = scheduler.restore({first, second});
        EXPECT_FALSE(restored.accepted);
        EXPECT_TRUE(scheduler.snapshots().empty());
    }

    TEST(TaskSchedulerTest, SerializesConflictingProjectWrites)
    {
        auto executor = std::make_shared<ConcurrencyExecutor>();
        TaskScheduler scheduler(TaskSchedulerLimits{2, {}});
        scheduler.registerExecutor("test", executor);
        TaskDefinition first = makeTask("write-a");
        TaskDefinition second = makeTask("write-b");
        first.projectKey = "project-a";
        second.projectKey = "project-a";
        first.resources.projectAccess = ProjectAccess::Write;
        second.resources.projectAccess = ProjectAccess::Write;
        const auto submitted = scheduler.submitBatch({first, second});
        ASSERT_TRUE(submitted.accepted);
        EXPECT_TRUE(scheduler.waitForState(submitted.runIds[0], TaskState::Succeeded, 2s));
        EXPECT_TRUE(scheduler.waitForState(submitted.runIds[1], TaskState::Succeeded, 2s));
        EXPECT_EQ(executor->maximum(), 1);
    }

    TEST(TaskJournalTest, RoundTripsCheckpointAndRestoresRunningAsInterrupted)
    {
        TaskRunSnapshot snapshot;
        snapshot.runId = "depth-run-7";
        snapshot.attemptId = 2;
        snapshot.definition = makeTask("depth");
        snapshot.definition.displayName = "深度图\t阶段";
        snapshot.definition.payload = "line1\nline2,100%";
        snapshot.definition.dependencies = {"prepare,a", "camera%b"};
        snapshot.state = TaskState::Running;
        snapshot.revision = 9;
        snapshot.queueSequence = 12;
        snapshot.progress = {"frame", 3, 10, 0.3};
        snapshot.checkpoint = TaskCheckpoint{2, "mvs/manifest.json", "hash-v2", 3, 10};
        snapshot.result = TaskResult{"已完成 3 帧", "mvs/depth"};
        snapshot.error = xjw::task_runtime::TaskError{"partial_warning", "保留检查点", true};
        snapshot.submittedAt = std::chrono::system_clock::now();

        const std::filesystem::path directory = std::filesystem::temp_directory_path() / "plascan-task-runtime-test";
        const std::filesystem::path journal = directory / "tasks.journal";
        std::string error;
        ASSERT_TRUE(TaskJournal::save(journal, {snapshot}, &error)) << error;
        const auto loaded = TaskJournal::load(journal);
        ASSERT_TRUE(loaded.succeeded) << loaded.error;
        ASSERT_EQ(loaded.snapshots.size(), 1u);
        EXPECT_EQ(loaded.snapshots[0].definition.displayName, "深度图\t阶段");
        EXPECT_EQ(loaded.snapshots[0].definition.payload, "line1\nline2,100%");
        EXPECT_EQ(loaded.snapshots[0].definition.dependencies, (std::vector<std::string>{"prepare,a", "camera%b"}));
        ASSERT_TRUE(loaded.snapshots[0].checkpoint.has_value());
        EXPECT_EQ(loaded.snapshots[0].checkpoint->completedUnits, 3u);
        ASSERT_TRUE(loaded.snapshots[0].result.has_value());
        EXPECT_EQ(loaded.snapshots[0].result->outputLocation, "mvs/depth");
        ASSERT_TRUE(loaded.snapshots[0].error.has_value());
        EXPECT_EQ(loaded.snapshots[0].error->code, "partial_warning");
        EXPECT_TRUE(loaded.snapshots[0].error->retryable);

        std::vector<TaskRunSnapshot> restore_snapshots = loaded.snapshots;
        restore_snapshots.front().definition.dependencies.clear();
        TaskScheduler scheduler({1, {}});
        const auto restored = scheduler.restore(std::move(restore_snapshots));
        ASSERT_TRUE(restored.accepted) << restored.error;
        const auto interrupted = scheduler.snapshot(snapshot.runId);
        ASSERT_TRUE(interrupted.has_value());
        EXPECT_EQ(interrupted->state, TaskState::Interrupted);
        EXPECT_EQ(interrupted->blockedReason, "process_interrupted");
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

} // namespace
