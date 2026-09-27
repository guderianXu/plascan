#pragma once

#include "CameraCalibrationData.h"

#include <QDialog>
#include <QJsonObject>
#include <QVector>

class QLabel;
class QListWidget;
class QPushButton;
class QTableWidget;
class QTabWidget;

class CameraCalibrationDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit CameraCalibrationDialog(const QJsonObject& projectMetadata,
                                     const QString& projectAssetsDir,
                                     QWidget* parent = nullptr);
    void setCameraTaskRunning(bool running);

signals:
    void importCameraForImageRequested(const QString& imagePath);
    void importCameraProjectRequested();
    void initializeIntrinsicsRequested(const QJsonObject& settings);
    void clearCamerasRequested(const QStringList& imagePaths);

private slots:
    void showSelectedCameraGroup(int row);
    void updateCameraActionAvailability();
    void requestImportForSelectedPhoto();
    void requestCameraProjectImport();
    void requestInitializeIntrinsics();
    void requestClearSelectedCameras();

private:
    struct CameraGroup
    {
        QString label;
        QVector<int> recordIndices;
    };

    void buildInterface();
    void buildGroups();
    void populateParameterTables(const CameraGroup& group);
    void populatePhotoTable(const CameraGroup& group);
    void showEmptyState();
    QStringList selectedPhotoPaths() const;
    QStringList selectedConfiguredPhotoPaths() const;

    QVector<xjw::gui::camera_calibration::CameraCalibrationRecord> _records;
    QVector<CameraGroup> _groups;
    bool _hasProject = false;
    bool _hasProjectImages = false;
    bool _cameraTaskRunning = false;
    QString _reportTimestamp;
    QString _reportError;
    QListWidget* _cameraGroups = nullptr;
    QTabWidget* _calibrationTabs = nullptr;
    QTableWidget* _initialParameters = nullptr;
    QTableWidget* _adjustedParameters = nullptr;
    QTableWidget* _photoTable = nullptr;
    QLabel* _summaryLabel = nullptr;
    QPushButton* _importSelectedButton = nullptr;
    QPushButton* _importProjectButton = nullptr;
    QPushButton* _initializeIntrinsicsButton = nullptr;
    QPushButton* _clearSelectedButton = nullptr;
};
