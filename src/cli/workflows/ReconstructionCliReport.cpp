#include "ReconstructionCliReport.h"

// 工作流程 CLI 的机器可读报告实现。

#include "CliJsonIO.h"

#include <QDir>

namespace xjw::cli
{

bool writeReconstructionReport(const QString &outputDirectory,
                               const QJsonObject &report,
                               QJsonObject *writtenReport,
                               QString *errorMessage)
{
    const QString reportPath = QDir(outputDirectory).filePath(QStringLiteral("report.json"));
    QJsonObject output = report;
    output[QStringLiteral("report_json")] = reportPath;
    if (!writeJsonFile(reportPath, output, errorMessage))
    {
        return false;
    }
    if (writtenReport)
    {
        *writtenReport = output;
    }
    return true;
}

} // namespace xjw::cli
