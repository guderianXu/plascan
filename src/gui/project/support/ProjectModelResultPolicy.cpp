#include "ProjectModelResultPolicy.h"
#include "ProjectRunArtifactValidator.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include <algorithm>

namespace xjw::gui::project
{
namespace
{

QString modelMeshPath(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("model_ply")).toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("mesh_ply")).toString();
    }
    path = path.trimmed();
    return path.isEmpty() ? QString() : QDir::cleanPath(path);
}

bool isExistingFile(const QString &path)
{
    const QFileInfo info(path);
    return !path.isEmpty() && info.isFile() && info.size() > 0;
}

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

int currentDefaultIndex(const QJsonArray &records)
{
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = records.at(index).toObject();
        if (record.value(QStringLiteral("is_default_model")).toBool(false)
            && isExistingFile(modelMeshPath(record)))
        {
            return index;
        }
    }
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = records.at(index).toObject();
        if (!record.contains(QStringLiteral("is_default_model"))
            && isExistingFile(modelMeshPath(record)))
        {
            return index;
        }
    }
    return -1;
}

bool validateModelArtifacts(const QJsonObject &record,
                            bool requireRunMetadata,
                            QString *errorMessage)
{
    const QString runId = record.value(QStringLiteral("model_run_id"))
        .toString().trimmed();
    if (requireRunMetadata && runId.isEmpty())
    {
        setError(errorMessage, QStringLiteral("模型结果缺少 model_run_id"));
        return false;
    }
    if (!isExistingFile(modelMeshPath(record)))
    {
        setError(errorMessage,
                 QStringLiteral("模型运行的 PLY 产物不存在或为空：%1")
                     .arg(modelMeshPath(record)));
        return false;
    }
    const QString finalModelPath = record.value(
        QStringLiteral("final_model_path")).toString().trimmed();
    if ((requireRunMetadata || !finalModelPath.isEmpty())
        && !isExistingFile(finalModelPath))
    {
        setError(errorMessage,
                 QStringLiteral("模型运行的最终产物不存在或为空：%1")
                     .arg(finalModelPath));
        return false;
    }
    const bool textured = record.value(QStringLiteral("textured"))
        .toBool(false);
    const bool hasTextureRun =
        !record.value(QStringLiteral("texture_run_id")).toString().isEmpty()
        || !record.value(QStringLiteral("texture_run_directory"))
                .toString().isEmpty()
        || !record.value(QStringLiteral("texture_diagnostics_path"))
                .toString().isEmpty();
    for (const QString &key : {
             QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png")})
    {
        const QString path = record.value(key).toString().trimmed();
        if ((textured || !path.isEmpty()) && !isExistingFile(path))
        {
            setError(errorMessage,
                     QStringLiteral("模型运行产物不完整（%1）：%2")
                         .arg(key, path));
            return false;
        }
    }
    const QString modelDiagnosticsPath = record.value(
        QStringLiteral("model_diagnostics_path")).toString().trimmed();
    if (!requireRunMetadata && !modelDiagnosticsPath.isEmpty()
        && !isExistingFile(modelDiagnosticsPath))
    {
        setError(errorMessage,
                 QStringLiteral("模型运行诊断不存在或为空：%1")
                     .arg(modelDiagnosticsPath));
        return false;
    }

    if (requireRunMetadata)
    {
        const QString artifactDirectory = record.value(
            QStringLiteral("model_artifact_directory")).toString().trimmed();
        const QString runDirectory = record.value(
            QStringLiteral("model_run_directory")).toString().trimmed();
        if (!isPhysicalDirectory(artifactDirectory)
            || !pathBelongsToDirectory(artifactDirectory, runDirectory))
        {
            setError(errorMessage,
                     QStringLiteral("模型产物目录不属于当前 run：%1")
                         .arg(artifactDirectory));
            return false;
        }
        QStringList modelMatchingPathKeys{
            QStringLiteral("model_ply"),
            QStringLiteral("mesh_ply"),
            QStringLiteral("model_artifact_directory"),
            QStringLiteral("model_diagnostics_path"),
            QStringLiteral("depth_tsdf_detail_candidate_path"),
            QStringLiteral("boundary_attribution_debug_ply"),
            QStringLiteral("acquisition_gap_report")
        };
        if (textured && !hasTextureRun)
        {
            modelMatchingPathKeys.append({
                QStringLiteral("model_obj"),
                QStringLiteral("model_mtl"),
                QStringLiteral("texture_png"),
                QStringLiteral("texture_image")
            });
        }
        const RunDiagnosticsSpec modelSpec{
            QStringLiteral("模型"),
            QStringLiteral("model"),
            QStringLiteral("model_run_id"),
            QStringLiteral("model_run_directory"),
            QStringLiteral("model_diagnostics_path"),
            {QStringLiteral("model_ply"),
             QStringLiteral("mesh_ply"),
             QStringLiteral("final_model_path"),
             QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png"),
             QStringLiteral("texture_image"),
             QStringLiteral("model_diagnostics_path"),
             QStringLiteral("texture_diagnostics_path"),
             QStringLiteral("depth_tsdf_detail_candidate_path"),
             QStringLiteral("boundary_attribution_debug_ply"),
             QStringLiteral("acquisition_gap_report")},
            modelMatchingPathKeys
        };
        if (!validateRunDiagnostics(record, modelSpec, errorMessage))
        {
            return false;
        }
        if (!textured
            && !sameExistingPath(finalModelPath, modelMeshPath(record)))
        {
            setError(errorMessage,
                     QStringLiteral("未纹理模型的最终产物不是当前 run 的 PLY"));
            return false;
        }
    }

    if (textured && !sameExistingPath(finalModelPath, record.value(
            QStringLiteral("model_obj")).toString()))
    {
        setError(errorMessage,
                 QStringLiteral("纹理模型的最终产物不是 OBJ"));
        return false;
    }
    if (hasTextureRun && !textured)
    {
        setError(errorMessage,
                 QStringLiteral("纹理 run 未标记为完整纹理模型"));
        return false;
    }
    if (hasTextureRun)
    {
        const RunDiagnosticsSpec textureSpec{
            QStringLiteral("纹理"),
            QStringLiteral("texture"),
            QStringLiteral("texture_run_id"),
            QStringLiteral("texture_run_directory"),
            QStringLiteral("texture_diagnostics_path"),
            {QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png"),
             QStringLiteral("texture_image"),
             QStringLiteral("final_model_path"),
             QStringLiteral("texture_diagnostics_path")},
            {QStringLiteral("model_obj"),
             QStringLiteral("model_mtl"),
             QStringLiteral("texture_png"),
             QStringLiteral("texture_image"),
             QStringLiteral("texture_diagnostics_path")}
        };
        if (!validateRunDiagnostics(record, textureSpec, errorMessage))
        {
            return false;
        }
        if (requireRunMetadata
            && !pathBelongsToDirectory(
                record.value(QStringLiteral("texture_run_directory")).toString(),
                record.value(QStringLiteral("model_run_directory")).toString()))
        {
            setError(errorMessage,
                     QStringLiteral("纹理 run 不属于当前模型 run"));
            return false;
        }
    }
    return true;
}

} // namespace

DefaultModelResult resolveDefaultModelResult(const QJsonObject &metadata)
{
    DefaultModelResult result;
    const QJsonArray records = metadata.value(
        QStringLiteral("model_results")).toArray();
    const int index = currentDefaultIndex(records);
    if (index < 0)
    {
        result.errorMessage = QStringLiteral(
            "未找到可用的默认网格模型，请先执行网格重建");
        return result;
    }

    result.ok = true;
    result.index = index;
    result.modelRecord = records.at(index).toObject();
    result.meshPath = modelMeshPath(result.modelRecord);
    return result;
}

bool registerCompletedModelRun(
    QJsonObject *metadata,
    QJsonObject modelRecord,
    xjw::mesh::workflow::ModelOutputPolicy policy,
    QString *errorMessage)
{
    if (!metadata)
    {
        setError(errorMessage, QStringLiteral("项目元数据为空"));
        return false;
    }
    if (!validateModelArtifacts(modelRecord, true, errorMessage))
    {
        return false;
    }

    QJsonArray records = metadata->value(
        QStringLiteral("model_results")).toArray();
    const QString runId = modelRecord.value(
        QStringLiteral("model_run_id")).toString().trimmed();
    modelRecord[QStringLiteral("model_run_id")] = runId;
    for (const QJsonValue &value : records)
    {
        if (value.toObject().value(QStringLiteral("model_run_id")).toString()
                .trimmed() == runId)
        {
            setError(errorMessage,
                     QStringLiteral("模型运行记录已存在：%1").arg(runId));
            return false;
        }
    }

    const int defaultIndex = currentDefaultIndex(records);
    for (int index = 0; index < records.size(); ++index)
    {
        QJsonObject record = records.at(index).toObject();
        record[QStringLiteral("is_default_model")] =
            policy != xjw::mesh::workflow::ModelOutputPolicy::ReplaceDefault
            && index == defaultIndex;
        records[index] = record;
    }
    if (policy == xjw::mesh::workflow::ModelOutputPolicy::ReplaceDefault)
    {
        modelRecord[QStringLiteral("is_default_model")] = true;
    }
    else if (defaultIndex >= 0)
    {
        modelRecord[QStringLiteral("is_default_model")] = false;
    }
    else
    {
        modelRecord[QStringLiteral("is_default_model")] = true;
    }

    modelRecord[QStringLiteral("model_output_policy")] =
        xjw::mesh::workflow::modelOutputPolicyName(policy);
    modelRecord[QStringLiteral("model_property_schema_version")] =
        kModelResultSchemaVersion;
    records.append(modelRecord);
    (*metadata)[QStringLiteral("model_results")] = records;
    return true;
}

bool updateCompletedModelRun(QJsonObject *metadata,
                             const QJsonObject &modelRecord,
                             QString *errorMessage)
{
    if (!metadata)
    {
        setError(errorMessage, QStringLiteral("项目元数据为空"));
        return false;
    }

    QJsonArray records = metadata->value(
        QStringLiteral("model_results")).toArray();
    const QString runId = modelRecord.value(
        QStringLiteral("model_run_id")).toString().trimmed();
    if (!validateModelArtifacts(modelRecord, !runId.isEmpty(), errorMessage))
    {
        return false;
    }
    const QString meshPath = modelMeshPath(modelRecord);
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject existing = records.at(index).toObject();
        const bool sameRun = !runId.isEmpty()
            && existing.value(QStringLiteral("model_run_id")).toString().trimmed()
                == runId;
        const bool sameLegacyPath = runId.isEmpty()
            && modelMeshPath(existing) == meshPath;
        if (!sameRun && !sameLegacyPath)
        {
            continue;
        }

        QJsonObject updated = modelRecord;
        if (!runId.isEmpty())
        {
            updated[QStringLiteral("model_run_id")] = runId;
        }
        if (!updated.contains(QStringLiteral("is_default_model"))
            && existing.contains(QStringLiteral("is_default_model")))
        {
            updated[QStringLiteral("is_default_model")] = existing.value(
                QStringLiteral("is_default_model"));
        }
        records[index] = updated;
        (*metadata)[QStringLiteral("model_results")] = records;
        return true;
    }

    setError(errorMessage, QStringLiteral("未找到待更新的模型运行记录"));
    return false;
}

} // namespace xjw::gui::project
