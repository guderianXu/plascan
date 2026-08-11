#pragma once

#include "MvsTypes.h"

#include <QJsonObject>
#include <QString>

#include <vector>

namespace xjw::mvs
{

struct MvsPairAuditSummary
{
    int auditedPairCount = 0;
    int verifiedPairCount = 0;
    int failedPairCount = 0;
    int missingStatisticsPairCount = 0;
};

bool cameraFromMvsWorkspaceJson(const QJsonObject &object, FramePinholeCamera *camera);

bool loadMvsReplayViews(const QString &manifestPath,
                        const QString &maskDirectory,
                        std::vector<CameraView> *views,
                        QString *errorMessage = nullptr);

bool loadMvsPairAuditReport(
    const QString &reportPath,
    std::vector<MvsSourcePairQuality> *qualities,
    MvsPairAuditSummary *summary = nullptr,
    QString *errorMessage = nullptr);

} // namespace xjw::mvs
