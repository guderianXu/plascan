#pragma once

// 工作流程 CLI 的进度报告接口。

#include <QString>

namespace xjw::cli
{

void printPipelineStage(int currentStage, int totalStages, const QString &message);
void printScopedProgress(const QString &scope, int percent, const QString &message);

} // namespace xjw::cli
