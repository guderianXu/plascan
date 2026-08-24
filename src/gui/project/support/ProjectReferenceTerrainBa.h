#pragma once

#include "BundleAdjustSolver.h"
#include "BundleAdjustService.h"
#include "ReferenceDatasetWorkflow.h"

#include <QJsonObject>
#include <QString>

#include <vector>

namespace xjw::gui::project {

struct ReferenceTerrainBaApplyResult
{
    bool success = false;
    QString errorMessage;
    QJsonObject summary;
};

ReferenceTerrainBaApplyResult applyReferenceTerrainPriorToBundleAdjust(
    std::vector<xjw::BATrack> *tracks,
    xjw::gui::BaServiceOptions *options);

} // namespace xjw::gui::project
