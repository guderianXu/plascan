#pragma once

#include "BundleAdjustService.h"
#include "ProjectBaInputBuilder.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::gui::project {

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