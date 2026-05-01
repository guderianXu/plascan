#include "ProjectSparseWorkflow.h"

namespace xjw::gui::project {

SparsePointWorkflowSpec sparsePointWorkflowSpec(SparsePointWorkflowKind kind)
{
    switch (kind)
    {
    case SparsePointWorkflowKind::OutlierRemoval:
        return {QStringLiteral("离群点分层剔除"),
                QStringLiteral("请先打开项目后再执行稀疏点云离群点剔除"),
                QStringLiteral("正在执行离群点分层剔除..."),
                QStringLiteral("outlier_removal"),
                QStringLiteral("处理完成")};
    case SparsePointWorkflowKind::LocalOptim:
        return {QStringLiteral("稀疏点云空间清理"),
                QStringLiteral("请先打开项目后再执行稀疏点云空间清理"),
                QStringLiteral("正在执行稀疏点云空间清理..."),
                QStringLiteral("sparse_spatial_cleanup"),
                QStringLiteral("处理完成")};
    case SparsePointWorkflowKind::Refine:
        return {QStringLiteral("稀疏点云精修"),
                QStringLiteral("请先打开项目后再执行稀疏点云精修"),
                QStringLiteral("正在执行稀疏点云精修..."),
                QStringLiteral("sparse_refine"),
                QStringLiteral("精修完成")};
    }
    return {};
}

bool runSparsePointWorkflow(SparsePointWorkflowKind kind,
                            const SparsePointContext &context,
                            const QJsonObject &settings,
                            const QString &outputDir,
                            SparsePointOperationResult *result,
                            QString *errorMessage)
{
    switch (kind)
    {
    case SparsePointWorkflowKind::OutlierRemoval:
        return runSparsePointOutlierRemoval(context, settings, outputDir, result, errorMessage);
    case SparsePointWorkflowKind::LocalOptim:
        return runSparsePointLocalOptim(context, settings, outputDir, result, errorMessage);
    case SparsePointWorkflowKind::Refine:
        return runSparsePointRefine(context, settings, outputDir, result, errorMessage);
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("未知的稀疏点工作流类型");
    }
    return false;
}

SparsePointWorkflowResult runSparsePointWorkflowResult(SparsePointWorkflowKind kind,
                                                       const SparsePointContext &context,
                                                       const QJsonObject &settings,
                                                       const QString &outputDir)
{
    SparsePointWorkflowResult result;
    QString errorMessage;
    result.status.ok = runSparsePointWorkflow(kind,
                                              context,
                                              settings,
                                              outputDir,
                                              &result.operation,
                                              &errorMessage);
    result.status.errorMessage = errorMessage;
    return result;
}

QString buildSparsePointWorkflowSuccessMessage(const SparsePointWorkflowSpec &spec,
                                               const SparsePointOperationResult &result)
{
    return QStringLiteral("%1。\n输入点数: %2\n输出点数: %3\n输出文件: %4")
        .arg(spec.successVerb)
        .arg(result.inputCount)
        .arg(result.outputCount)
        .arg(result.sparseCloudPath);
}

} // namespace xjw::gui::project