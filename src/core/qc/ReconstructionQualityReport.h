#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::qc
{

struct ReconstructionQualityReportWriteResult
{
    bool ok = false;
    QString error;
    QString jsonPath;
    QString csvPath;
    QJsonObject report;
};

class ReconstructionQualityReport
{
public:
    static QJsonObject buildFromProjectMeta(const QJsonObject &projectMeta);

    static ReconstructionQualityReportWriteResult writeFromProjectMeta(const QJsonObject &projectMeta,
                                                                       const QString &outputDir,
                                                                       const QString &baseName);
};

} // namespace xjw::qc
