#include "ReconstructionPipelineReportContext.h"

#include "ReconstructionCliReport.h"
#include "project/ProjectSession.h"

#include <cstdio>
#include <utility>

namespace xjw::cli
{

ReconstructionPipelineReportContext::ReconstructionPipelineReportContext(
    QString reportsRoot,
    xjw::common::project::ProjectSession &projectSession,
    QJsonObject initialReport)
    : _reportsRoot(std::move(reportsRoot))
    , _projectSession(projectSession)
    , _report(std::move(initialReport))
{
}

QJsonObject &ReconstructionPipelineReportContext::report()
{
    return _report;
}

QJsonObject &ReconstructionPipelineReportContext::timings()
{
    return _timings;
}

double ReconstructionPipelineReportContext::recordTiming(
    const QString &key,
    const std::chrono::steady_clock::time_point &start)
{
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    _timings[key] = elapsedMs;
    return elapsedMs;
}

void ReconstructionPipelineReportContext::markSkippedStage(
    const QString &stage,
    const QString &reason)
{
    QJsonObject skippedStages =
        _report.value(QStringLiteral("skipped_stages")).toObject();
    skippedStages[stage] = reason;
    _report[QStringLiteral("skipped_stages")] = skippedStages;
}

bool ReconstructionPipelineReportContext::writeFinalReport(
    QJsonObject *finalReport)
{
    QString reportError;
    if (!writeReconstructionReport(
            _reportsRoot, _report, finalReport, &reportError))
    {
        std::fprintf(stderr,
                     "报告写入失败: %s\n",
                     qUtf8Printable(reportError));
        return false;
    }

    QJsonObject reportRecord = _report;
    reportRecord[QStringLiteral("kind")] =
        QStringLiteral("reconstruction_pipeline_cli");
    reportRecord[QStringLiteral("path")] =
        finalReport->value(QStringLiteral("report_json")).toString();
    _projectSession.upsertResultByPath(
        QStringLiteral("report_results"),
        QStringLiteral("path"),
        reportRecord);
    if (!_projectSession.save(&reportError))
    {
        std::fprintf(stderr,
                     "报告已生成，但 Chunk 写回失败: %s\n",
                     qUtf8Printable(reportError));
        return false;
    }
    return true;
}

} // namespace xjw::cli
