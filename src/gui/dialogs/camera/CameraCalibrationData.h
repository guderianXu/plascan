#pragma once

#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

namespace xjw::gui::camera_calibration
{

struct CameraCalibrationRecord
{
    QString path;
    QString name;
    QString model;
    int imageWidth = 0;
    int imageHeight = 0;
    QJsonObject initial;
    QJsonObject adjusted;
    bool hasInitial = false;
    bool hasAdjusted = false;
    QString initialSource;
    QString adjustmentStatus;
    QStringList optimizedParameters;
    bool requiresReview = false;
};

QJsonArray buildCameraCalibrationComparison(
    const QJsonObject &projectMetadata,
    const QMap<QString, QJsonObject> &adjustedCameras,
    const QJsonObject &sfmDiagnostics);

QVector<CameraCalibrationRecord> buildCameraCalibrationRecords(
    const QJsonObject &projectMetadata,
    const QJsonObject &bundleAdjustReport);

QJsonObject readLatestCameraCalibrationReport(const QString &projectAssetsDir,
                                              QString *errorMessage = nullptr);

} // namespace xjw::gui::camera_calibration
