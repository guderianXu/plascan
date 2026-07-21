#pragma once

#include "model/AerialTriangulationResolvedConfig.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>

namespace xjw::aerial_triangulation
{

struct AerialTriangulationReconstructionResult
{
    bool success = false;
    QString errorMessage;
    QString summary;
    int numRegisteredImages = 0;
    int numPoints3D = 0;
    double meanReprojError = 0.0;
    QMap<QString, QJsonObject> pendingCamUpdates;
    QString sparseCloudPath;
    QJsonObject qualityMetadata;
    QJsonObject resultRecordExtra;
    QJsonObject sfmDiagnostics;
    double baRmsBefore = 0.0;
    double baRmsAfter = 0.0;
    int baTracksTotal = 0;
    int baTracksOptimized = 0;
    int baTracksFiltered = 0;
    double durationSeconds = -1.0;
    QJsonArray perCameraResiduals;
};

struct AerialTriangulationResult
{
    AerialTriangulationResolvedConfig config;
    AerialTriangulationReconstructionResult reconstructionResult;
    bool tiePointPreparationExecuted = false;
    matchphotos::MatchPhotosResult tiePointResult;
};

} // namespace xjw::aerial_triangulation
