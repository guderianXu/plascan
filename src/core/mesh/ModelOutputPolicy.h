#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::mesh::workflow
{

enum class ModelOutputPolicy
{
    CreateVersionedResult,
    ReplaceDefault
};

QString modelOutputPolicyName(ModelOutputPolicy policy);
ModelOutputPolicy modelOutputPolicyFromSettings(const QJsonObject &settings);
QString createModelRunId();

bool createModelRunOutputDirectory(const QString &baseOutputRoot,
                                   const QString &requestedRunId,
                                   QString *runId,
                                   QString *runOutputRoot,
                                   QString *errorMessage = nullptr);

bool createTextureRunOutputDirectory(const QString &modelOutputRoot,
                                     const QString &requestedRunId,
                                     QString *runId,
                                     QString *runOutputRoot,
                                     QString *errorMessage = nullptr);

bool removeUnpublishedModelRunDirectory(const QString &baseOutputRoot,
                                        const QString &runId,
                                        const QString &runOutputRoot,
                                        QString *errorMessage = nullptr);

bool removeUnpublishedTextureRunDirectory(const QString &modelOutputRoot,
                                          const QString &runId,
                                          const QString &runOutputRoot,
                                          QString *errorMessage = nullptr);

} // namespace xjw::mesh::workflow
