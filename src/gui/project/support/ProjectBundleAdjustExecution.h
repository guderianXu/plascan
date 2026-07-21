#pragma once

#include "project/BaInputBuilder.h"
#include "BundleAdjustService.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::gui::project {

using xjw::core::project::BaInputBuildResult;
using xjw::core::project::BaInputBuildStatus;
using xjw::core::project::buildBaInputFromMeta;

struct BundleAdjustExecutionResult
{
    BaInputBuildStatus buildStatus = BaInputBuildStatus::Ok;
    xjw::gui::BaServiceResult serviceResult;
    QMap<QString, QJsonObject> beforeCamMeta;
};

BundleAdjustExecutionResult runBundleAdjustExecution(const QJsonObject &coreData,
                                                     const QString &plascanPath,
                                                     const QStringList &selectedImages,
                                                     int minMatches,
                                                     xjw::gui::BaServiceOptions options);

} // namespace xjw::gui::project
