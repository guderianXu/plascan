#pragma once

#include "ProjectWorkflowUtils.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project {

enum class SparsePointWorkflowKind
{
    OutlierRemoval,
    LocalOptim,
    Refine,
};

struct SparsePointWorkflowSpec
{
    QString title;
    QString projectOpenMessage;
    QString progressMessage;
    QString outputDirPrefix;
    QString successVerb;
};

struct SparsePointWorkflowResult
{
    OperationResult status;
    SparsePointOperationResult operation;
};

SparsePointWorkflowSpec sparsePointWorkflowSpec(SparsePointWorkflowKind kind);

bool runSparsePointWorkflow(SparsePointWorkflowKind kind,
                            const SparsePointContext &context,
                            const QJsonObject &settings,
                            const QString &outputDir,
                            SparsePointOperationResult *result,
                            QString *errorMessage);

SparsePointWorkflowResult runSparsePointWorkflowResult(SparsePointWorkflowKind kind,
                                                       const SparsePointContext &context,
                                                       const QJsonObject &settings,
                                                       const QString &outputDir);

QString buildSparsePointWorkflowSuccessMessage(const SparsePointWorkflowSpec &spec,
                                               const SparsePointOperationResult &result);

} // namespace xjw::gui::project