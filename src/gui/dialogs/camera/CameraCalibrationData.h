#pragma once

#include <QJsonObject>
#include <QString>
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
};

QVector<CameraCalibrationRecord> buildCameraCalibrationRecords(
    const QJsonObject &projectMetadata,
    const QJsonObject &bundleAdjustReport);

QJsonObject readLatestCameraCalibrationReport(const QString &projectAssetsDir,
                                              QString *errorMessage = nullptr);

} // namespace xjw::gui::camera_calibration
