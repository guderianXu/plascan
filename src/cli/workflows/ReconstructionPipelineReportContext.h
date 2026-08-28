#pragma once

#include <QJsonObject>
#include <QString>

#include <chrono>

namespace xjw::common::project
{
class ProjectSession;
}

namespace xjw::cli
{

class ReconstructionPipelineReportContext
{
public:
    ReconstructionPipelineReportContext(
        QString reportsRoot,
        xjw::common::project::ProjectSession &projectSession,
        QJsonObject initialReport);

    QJsonObject &report();
    QJsonObject &timings();

    double recordTiming(
        const QString &key,
        const std::chrono::steady_clock::time_point &start);
    void markSkippedStage(const QString &stage, const QString &reason);
    bool writeFinalReport(QJsonObject *finalReport);

private:
    QString _reportsRoot;
    xjw::common::project::ProjectSession &_projectSession;
    QJsonObject _report;
    QJsonObject _timings;
};

} // namespace xjw::cli
