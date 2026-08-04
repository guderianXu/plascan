#pragma once

#include "CameraCalibrationData.h"

#include <QDialog>
#include <QVector>

class QLabel;
class QListWidget;
class QTableWidget;
class QTabWidget;

class CameraCalibrationDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit CameraCalibrationDialog(const QJsonObject &projectMetadata,
                                     const QString &projectAssetsDir,
                                     QWidget *parent = nullptr);

private slots:
    void showSelectedCameraGroup(int row);

private:
    struct CameraGroup
    {
        QString label;
        QVector<int> recordIndices;
    };

    void buildInterface();
    void buildGroups();
    void populateParameterTables(const CameraGroup &group);
    void populatePhotoTable(const CameraGroup &group);
    void showEmptyState();

    QVector<xjw::gui::camera_calibration::CameraCalibrationRecord> _records;
    QVector<CameraGroup> _groups;
    QString _reportTimestamp;
    QString _reportError;
    QListWidget *_cameraGroups = nullptr;
    QTabWidget *_calibrationTabs = nullptr;
    QTableWidget *_initialParameters = nullptr;
    QTableWidget *_adjustedParameters = nullptr;
    QTableWidget *_photoTable = nullptr;
    QLabel *_summaryLabel = nullptr;
};
