#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    QString sha256ForFile(const QString& path, QString* error_message)
    {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
        {
            if (error_message)
            {
                *error_message = file.errorString();
            }
            return {};
        }

        QCryptographicHash hash(QCryptographicHash::Sha256);
        while (!file.atEnd())
        {
            const QByteArray chunk = file.read(1024 * 1024);
            if (chunk.isEmpty() && file.error() != QFileDevice::NoError)
            {
                if (error_message)
                {
                    *error_message = file.errorString();
                }
                return {};
            }
            hash.addData(chunk);
        }
        return QString::fromLatin1(hash.result().toHex());
    }

    bool isNonEmptyFile(const QString& path)
    {
        const QFileInfo info(path);
        return !path.trimmed().isEmpty() && info.isFile() && info.size() > 0;
    }

    bool validateModelRunArtifacts(const QJsonObject& payload, QString* errorMessage)
    {
        const QString modelPly = payload.value(QStringLiteral("model_ply")).toString();
        if (!isNonEmptyFile(modelPly))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("模型运行未生成完整的 PLY 产物：%1").arg(modelPly);
            }
            return false;
        }

        const QString finalModel = payload.value(QStringLiteral("final_model_path")).toString();
        if (!isNonEmptyFile(finalModel))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("模型运行的最终产物不存在或为空：%1").arg(finalModel);
            }
            return false;
        }

        for (const QString& key :
             {QStringLiteral("model_obj"), QStringLiteral("model_mtl"), QStringLiteral("texture_png")})
        {
            const QString path = payload.value(key).toString().trimmed();
            if (!path.isEmpty() && !isNonEmptyFile(path))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("模型运行产物不完整（%1）：%2").arg(key, path);
                }
                return false;
            }
        }
        return true;
    }

    bool writeRunDiagnostics(WorkflowResult* result, const QString& diagnosticsPath, const QString& diagnosticsType)
    {
        QJsonObject diagnostics = result->payload;
        diagnostics[QStringLiteral("schema_version")] = 1;
        diagnostics[QStringLiteral("diagnostics_type")] = diagnosticsType;
        diagnostics[QStringLiteral("ok")] = true;
        const QByteArray diagnosticsBytes = QJsonDocument(diagnostics).toJson(QJsonDocument::Indented);

        QSaveFile diagnosticsFile(diagnosticsPath);
        if (!diagnosticsFile.open(QIODevice::WriteOnly) ||
            diagnosticsFile.write(diagnosticsBytes) != diagnosticsBytes.size() || !diagnosticsFile.commit() ||
            !isNonEmptyFile(diagnosticsPath))
        {
            result->ok = false;
            result->errorMessage = QStringLiteral("无法原子写入模型运行诊断：%1（%2）")
                                       .arg(diagnosticsPath, diagnosticsFile.errorString());
            return false;
        }
        return true;
    }

    bool finalizeModelRun(WorkflowResult* result,
                          ModelOutputPolicy policy,
                          const QString& runId,
                          const QString& runOutputRoot)
    {
        if (!result || !result->ok)
        {
            return false;
        }

        result->payload[QStringLiteral("model_run_id")] = runId;
        result->payload[QStringLiteral("model_output_policy")] = modelOutputPolicyName(policy);
        result->payload[QStringLiteral("model_run_directory")] = runOutputRoot;
        result->payload[QStringLiteral("model_artifact_directory")] =
            QDir(runOutputRoot).filePath(QStringLiteral("products"));

        QString validationError;
        if (!validateModelRunArtifacts(result->payload, &validationError))
        {
            result->ok = false;
            result->errorMessage = validationError;
            return false;
        }

        const QString diagnosticsPath = QDir(runOutputRoot).filePath(QStringLiteral("model_result.json"));
        result->payload[QStringLiteral("model_diagnostics_path")] = diagnosticsPath;
        return writeRunDiagnostics(result, diagnosticsPath, QStringLiteral("model"));
    }

    bool finalizeTextureRun(WorkflowResult* result, const QString& runId, const QString& runOutputRoot)
    {
        if (!result || !result->ok)
        {
            return false;
        }

        result->payload[QStringLiteral("texture_run_id")] = runId;
        result->payload[QStringLiteral("texture_run_directory")] = runOutputRoot;
        for (const QString& key :
             {QStringLiteral("model_obj"), QStringLiteral("model_mtl"), QStringLiteral("texture_png")})
        {
            const QString path = result->payload.value(key).toString().trimmed();
            if (!isNonEmptyFile(path))
            {
                result->ok = false;
                result->errorMessage = QStringLiteral("纹理运行产物不完整（%1）：%2").arg(key, path);
                return false;
            }
        }

        const QString diagnosticsPath = QDir(runOutputRoot).filePath(QStringLiteral("texture_result.json"));
        result->payload[QStringLiteral("texture_diagnostics_path")] = diagnosticsPath;
        return writeRunDiagnostics(result, diagnosticsPath, QStringLiteral("texture"));
    }

    void assignFinalModelFields(QJsonObject* result, bool exportObj)
    {
        if (!result)
        {
            return;
        }

        (*result)[QStringLiteral("requested_export_format")] =
            exportObj ? QStringLiteral("OBJ") : QStringLiteral("PLY");

        if (exportObj)
        {
            const QString objPath = result->value(QStringLiteral("model_obj")).toString();
            if (!objPath.isEmpty())
            {
                (*result)[QStringLiteral("final_model_format")] = QStringLiteral("OBJ");
                (*result)[QStringLiteral("final_model_path")] = objPath;
                return;
            }
        }

        const QString plyPath = result->value(QStringLiteral("model_ply")).toString();
        (*result)[QStringLiteral("final_model_format")] = QStringLiteral("PLY");
        (*result)[QStringLiteral("final_model_path")] = plyPath;
    }

    void mergePayload(const QJsonObject& source, QJsonObject* target)
    {
        if (!target)
        {
            return;
        }
        for (auto it = source.constBegin(); it != source.constEnd(); ++it)
        {
            (*target)[it.key()] = it.value();
        }
    }
} // namespace xjw::mesh::workflow::workflow_detail
