#pragma once

#include "ProjectCameraIO.h"
#include "ProjectMetadata.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <array>

namespace xjw::common::project
{

struct LatestModelMeshLookupResult
{
    bool ok = false;
    QString meshPath;
    QJsonObject modelRecord;
    QString errorMessage;
};

inline LatestModelMeshLookupResult resolveLatestModelMeshRecord(const QJsonObject &metadata)
{
    LatestModelMeshLookupResult result;

    const QJsonArray modelResults = metadata.value(QStringLiteral("model_results")).toArray();
    for (int index = modelResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = modelResults.at(index).toObject();
        const QString candidate = record.value(QStringLiteral("model_ply")).toString();
        if (!candidate.isEmpty() && QFileInfo::exists(candidate))
        {
            result.ok = true;
            result.meshPath = candidate;
            result.modelRecord = record;
            return result;
        }
    }

    result.errorMessage = QStringLiteral("未找到可用的网格模型，请先执行网格重建");
    return result;
}

inline QJsonObject enrichModelResultFromTerrain(const QJsonObject &baseModelResult,
                                                const QJsonObject &terrainResult)
{
    QJsonObject enrichedModelResult = baseModelResult;

    const QStringList extraKeys = {
        QStringLiteral("mesh_ply"),
        QStringLiteral("mesh_algorithm"),
        QStringLiteral("model_obj"),
        QStringLiteral("model_mtl"),
        QStringLiteral("final_model_path"),
        QStringLiteral("final_model_format"),
        QStringLiteral("requested_export_format"),
        QStringLiteral("texture_png"),
        QStringLiteral("texture_image"),
        QStringLiteral("texture_size"),
        QStringLiteral("texture_algorithm"),
        QStringLiteral("uv_method"),
        QStringLiteral("blend_method"),
        QStringLiteral("texture_warning")
    };

    for (const QString &key : extraKeys)
    {
        const QJsonValue value = terrainResult.value(key);
        if (!value.isUndefined() && !value.isNull())
        {
            enrichedModelResult.insert(key, value);
        }
    }

    if (terrainResult.value(QStringLiteral("final_model_format")).toString() == QStringLiteral("OBJ"))
    {
        enrichedModelResult[QStringLiteral("textured")] = true;
    }

    return enrichedModelResult;
}

} // namespace xjw::common::project
