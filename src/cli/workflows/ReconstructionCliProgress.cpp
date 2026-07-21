#include "ReconstructionCliProgress.h"

// 工作流程 CLI 的稳定进度输出协议。

#include "CliConsole.h"

namespace xjw::cli
{

void printPipelineStage(int currentStage, int totalStages, const QString &message)
{
    printUtf8(stdout, QStringLiteral("[%1/%2] %3").arg(currentStage).arg(totalStages).arg(message));
}

void printScopedProgress(const QString &scope, int percent, const QString &message)
{
    printUtf8(stdout,
              QStringLiteral("  [%1 %2%] %3")
                  .arg(scope)
                  .arg(percent, 3)
                  .arg(message));
}

} // namespace xjw::cli
