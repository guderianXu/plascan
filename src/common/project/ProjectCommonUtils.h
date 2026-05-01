#pragma once

#include "Camera.h"

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

inline QString normalizePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

inline bool pathTokenMatchesImage(const QString &token, const QString &imagePath)
{
    if (token.isEmpty() || imagePath.isEmpty())
    {
        return false;
    }

    const QString normalizedToken = normalizePath(token);
    const QString normalizedImage = normalizePath(imagePath);
    if (normalizedToken == normalizedImage)
    {
        return true;
    }

    if (QFileInfo(token).fileName() == QFileInfo(imagePath).fileName())
    {
        return true;
    }

    return QFileInfo(token).completeBaseName() == QFileInfo(imagePath).completeBaseName();
}

inline bool cameraFromJson(const QJsonObject &cameraObject, xjw::Camera *camera)
{
    if (!camera || cameraObject.isEmpty())
    {
        return false;
    }

    const QJsonArray centerArray = cameraObject.value(QStringLiteral("C")).toArray();
    const QJsonArray rotationArray = cameraObject.value(QStringLiteral("R")).toArray();
    if (centerArray.size() < 3 || rotationArray.size() < 9)
    {
        return false;
    }

    std::array<double, 3> center{{centerArray.at(0).toDouble(),
                                  centerArray.at(1).toDouble(),
                                  centerArray.at(2).toDouble()}};
    if (cameraObject.value(QStringLiteral("camera_center_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive)
        == 0)
    {
        center[0] /= 1000.0;
        center[1] /= 1000.0;
        center[2] /= 1000.0;
    }

    std::array<double, 9> rotation{};
    for (int index = 0; index < 9; ++index)
    {
        rotation[index] = rotationArray.at(index).toDouble();
    }

    const double pitch = cameraObject.value(QStringLiteral("pitch")).toDouble(1.0);
    const bool intrinsicsInMillimeters =
        cameraObject.value(QStringLiteral("intrinsics_unit"))
            .toString()
            .compare(QStringLiteral("mm"), Qt::CaseInsensitive)
        == 0;
    const double fu = cameraObject.value(QStringLiteral("fu")).toDouble();
    const double fv = cameraObject.value(QStringLiteral("fv")).toDouble();
    const double cu = cameraObject.value(QStringLiteral("cu")).toDouble();
    const double cv = cameraObject.value(QStringLiteral("cv")).toDouble();

    if (intrinsicsInMillimeters)
    {
        camera->setIntrinsicsMillimeters(fu, fv, cu, cv, pitch);
    }
    else
    {
        camera->setPixelPitch(pitch);
        camera->setIntrinsics(fu,
                              fv,
                              cu,
                              cv);
    }

    camera->setAxisDirections(cameraObject.value(QStringLiteral("u_direction")).toInt(1),
                              cameraObject.value(QStringLiteral("v_direction")).toInt(1));
    camera->setDepthAxisFlipped(cameraObject.value(QStringLiteral("depth_axis_flipped")).toBool(false));
    camera->setDistortion(cameraObject.value(QStringLiteral("k1")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("k2")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("k3")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("p1")).toDouble(0.0),
                          cameraObject.value(QStringLiteral("p2")).toDouble(0.0));
    camera->setPose(rotation, center);

    return true;
}

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
