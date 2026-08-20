#pragma once

#include "ProjectWorkflowOperations.h"

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
    xjw::common::OperationResult status;
    xjw::core::project::SparsePointOperationResult operation;
};

SparsePointWorkflowSpec sparsePointWorkflowSpec(SparsePointWorkflowKind kind);

bool runSparsePointWorkflow(SparsePointWorkflowKind kind,
                            const xjw::core::project::SparsePointContext &context,
                            const QJsonObject &settings,
                            const QString &outputDir,
                            xjw::core::project::SparsePointOperationResult *result,
                            QString *errorMessage);

SparsePointWorkflowResult runSparsePointWorkflowResult(SparsePointWorkflowKind kind,
                                                       const xjw::core::project::SparsePointContext &context,
                                                       const QJsonObject &settings,
                                                       const QString &outputDir);

QString buildSparsePointWorkflowSuccessMessage(const SparsePointWorkflowSpec &spec,
                                               const xjw::core::project::SparsePointOperationResult &result);

} // namespace xjw::gui::project
