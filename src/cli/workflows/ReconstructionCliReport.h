#pragma once

// 工作流程 CLI 的机器可读报告接口。

#include <QJsonObject>
#include <QString>

namespace xjw::cli
{

bool writeReconstructionReport(const QString &outputDirectory,
                               const QJsonObject &report,
                               QJsonObject *writtenReport,
                               QString *errorMessage);

} // namespace xjw::cli
