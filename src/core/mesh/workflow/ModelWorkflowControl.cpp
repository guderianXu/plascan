#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    bool cancellationRequested(const std::function<bool()>& isCancelled)
    {
        return isCancelled && isCancelled();
    }

    WorkflowResult cancelledWorkflowResult()
    {
        WorkflowResult result;
        result.errorMessage = QStringLiteral("模型生成已取消");
        result.payload[QStringLiteral("cancelled")] = true;
        return result;
    }
} // namespace xjw::mesh::workflow::workflow_detail
