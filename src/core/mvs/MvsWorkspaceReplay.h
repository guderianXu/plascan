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

/**
 * Decode the MVS-specific numeric camera record.  Storage-only callers must
 * explicitly pass false when they intentionally need a temporary numeric
 * value; replay and fusion paths pass true so the camera cannot lose its
 * project image and world-frame binding at the persistence boundary.
 */
bool cameraFromMvsWorkspaceJson(const QJsonObject& object,
                                xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera,
                                bool requireBoundIdentity);

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
