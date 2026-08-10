#pragma once

#include "ModelOutputPolicy.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

inline constexpr int kModelResultSchemaVersion = 2;

struct DefaultModelResult
{
    bool ok = false;
    int index = -1;
    QString meshPath;
    QJsonObject modelRecord;
    QString errorMessage;
};

DefaultModelResult resolveDefaultModelResult(const QJsonObject &metadata);

bool registerCompletedModelRun(
    QJsonObject *metadata,
    QJsonObject modelRecord,
    xjw::mesh::workflow::ModelOutputPolicy policy,
    QString *errorMessage = nullptr);

bool updateCompletedModelRun(QJsonObject *metadata,
                             const QJsonObject &modelRecord,
                             QString *errorMessage = nullptr);

} // namespace xjw::gui::project
