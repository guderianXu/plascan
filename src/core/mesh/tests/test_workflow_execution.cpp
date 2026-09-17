#include "DepthTsdfSurfaceBuilder.h"
#include "ModelWorkflowService.h"
#include "../workflow/ModelWorkflowInternals.h"

#include <gtest/gtest.h>

using xjw::task_runtime::WorkflowStatus;

TEST(ModelWorkflowExecutionContract, NestedRequestsReportOnceAndUseCanonicalCancellation)
{
    xjw::mesh::workflow::ModelBuildRequest request;
    request.execution.cancellation = std::make_shared<std::atomic_bool>(false);
    int shared_count = 0;
    bool task_cancelled = false;
    request.execution.progress = [&](const xjw::task_runtime::WorkflowProgress& progress)
    {
        EXPECT_EQ(progress.stage, QStringLiteral("网格清理").toUtf8().toStdString());
        EXPECT_DOUBLE_EQ(progress.ratio, 0.25);
        ++shared_count;
    };
    request.execution.cancellationCheck = [&] { return task_cancelled; };
    const auto once = xjw::mesh::workflow::workflow_detail::bindWorkflowCallbacks(request);
    xjw::mesh::workflow::DepthMapMeshBuildRequest child;
    child.execution = once.execution;
    const auto nested = xjw::mesh::workflow::workflow_detail::bindWorkflowCallbacks(child);
    nested.progress(QStringLiteral("网格清理"), 25);
    EXPECT_EQ(shared_count, 1);
    EXPECT_FALSE(nested.isCancelled());
    request.execution.cancellation->store(true);
    EXPECT_TRUE(nested.isCancelled());
    request.execution.cancellation->store(false);
    task_cancelled = true;
    EXPECT_TRUE(nested.isCancelled());
}

TEST(ModelWorkflowExecutionContract, SharedCancellationStopsBeforeInputValidation)
{
    xjw::mesh::workflow::ModelBuildRequest request;
    request.execution.cancellation = std::make_shared<std::atomic_bool>(true);
    const auto result = xjw::mesh::workflow::buildModel(request);
    EXPECT_EQ(result.outcome().status, WorkflowStatus::Cancelled);
    EXPECT_TRUE(result.payload.value(QStringLiteral("cancelled")).toBool());
}

TEST(ModelWorkflowExecutionContract, TaskContextCancellationStopsBeforeValidation)
{
    xjw::mesh::workflow::ModelBuildRequest request;
    request.execution.cancellationCheck = [] { return true; };
    EXPECT_EQ(xjw::mesh::workflow::buildModel(request).outcome().status, WorkflowStatus::Cancelled);
}

TEST(ModelWorkflowExecutionContract, AllIndependentModelEntriesHonorSharedCancellation)
{
    xjw::task_runtime::WorkflowControl control;
    control.cancellation = std::make_shared<std::atomic_bool>(true);
    xjw::mesh::workflow::MeshBuildRequest mesh;
    mesh.execution = control;
    EXPECT_EQ(xjw::mesh::workflow::buildMeshAndOptionalTexture(mesh).outcome().status, WorkflowStatus::Cancelled);
    xjw::mesh::workflow::DepthMapMeshBuildRequest depth;
    depth.execution = control;
    EXPECT_EQ(xjw::mesh::workflow::buildMeshFromDepthMaps(depth).outcome().status, WorkflowStatus::Cancelled);
    xjw::mesh::workflow::TextureBuildRequest texture;
    texture.execution = control;
    EXPECT_EQ(xjw::mesh::workflow::buildTextureOnly(texture).outcome().status, WorkflowStatus::Cancelled);
}

TEST(TsdfWorkflowExecutionContract, SharedCancellationStopsBeforeAllocation)
{
    xjw::mesh::DepthTsdfOptions options;
    options.execution.cancellation = std::make_shared<std::atomic_bool>(true);
    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build({}, options);
    EXPECT_EQ(result.outcome().status, WorkflowStatus::Cancelled);
    EXPECT_TRUE(result.mesh.empty());
    EXPECT_EQ(result.layout.sampleCount, 0);
}

TEST(TsdfWorkflowExecutionContract, InvalidInputProducesFailureNotCancellation)
{
    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build({}, {});
    EXPECT_EQ(result.outcome().status, WorkflowStatus::Failed);
    EXPECT_FALSE(result.errorMessage.isEmpty());
}

TEST(TsdfWorkflowExecutionContract, TaskCancellationStopsWithoutSharedFlagOrProgress)
{
    xjw::mesh::DepthTsdfOptions options;
    options.execution.cancellationCheck = [] { return true; };
    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build({}, options);
    EXPECT_EQ(result.outcome().status, WorkflowStatus::Cancelled);
    EXPECT_TRUE(result.mesh.empty());
    EXPECT_EQ(result.layout.sampleCount, 0);
}

TEST(ModelWorkflowExecutionContract, SuccessDoesNotExposeStaleErrors)
{
    xjw::mesh::workflow::WorkflowResult result;
    result.ok = true;
    result.errorMessage = QStringLiteral("stale");
    EXPECT_TRUE(result.outcome().succeeded());
    EXPECT_TRUE(result.outcome().error.message.empty());
}
