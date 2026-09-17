#include "WorkflowExecution.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace xjw::task_runtime;

TEST(WorkflowExecutionContract, ProgressClampsInvalidAndOutOfRangeRatios)
{
    WorkflowControl control;
    std::vector<double> values;
    control.progress = [&](const WorkflowProgress& progress)
    {
        EXPECT_EQ(progress.stage, "stage");
        values.push_back(progress.ratio);
    };
    control.reportProgress("stage", -1.0);
    control.reportProgress("stage", 2.0);
    control.reportProgress("stage", std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(values, (std::vector<double>{0.0, 1.0, 0.0}));
}

TEST(WorkflowExecutionContract, SharedFlagAndTaskCancellationUseOneControl)
{
    WorkflowControl control;
    control.cancellation = std::make_shared<std::atomic_bool>(false);
    bool task_cancelled = false;
    control.cancellationCheck = [&]() { return task_cancelled; };
    EXPECT_FALSE(control.isCancelled());
    task_cancelled = true;
    EXPECT_TRUE(control.isCancelled());
    task_cancelled = false;
    control.cancellation->store(true);
    EXPECT_TRUE(control.isCancelled());
    EXPECT_FALSE(WorkflowControl{}.isCancelled());
}

TEST(WorkflowExecutionContract, SuccessIsExplicitAndErrorsKeepLocation)
{
    WorkflowOutcome result;
    EXPECT_FALSE(result.succeeded());
    result.error = {"read_failed", "missing depth", "load", "frame.depth"};
    EXPECT_EQ(result.error.path, "frame.depth");
    result.status = WorkflowStatus::Succeeded;
    EXPECT_TRUE(result.succeeded());
}
