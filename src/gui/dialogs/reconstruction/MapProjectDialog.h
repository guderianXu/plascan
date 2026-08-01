#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QCheckBox;
class QCloseEvent;
class QComboBox;
class QDialogButtonBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QScrollArea;
class QSpinBox;
class QTimer;
class QToolButton;
class QWidget;

class MapProjectDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MapProjectDialog(QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setProjectRoot(const QString &projectRoot);
    void setDefaultDemPath(const QString &demPath);
    void setImageReadiness(const QStringList &cameraReadyImages, int maskCount);

signals:
    void requestRunMapProject(const QJsonObject &settings);
    void requestCancelMapProject();
    void settingsChanged(const QJsonObject &settings);

public slots:
    void applySettings(const QJsonObject &settings);
    void onPipelineStarted();
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineFinished(bool success, const QString &message);
    void reject() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onChooseDem();
    void onChooseOutput();
    void onRun();
    void onCancelRequested();
    void onSettingsModified();
    void onDemPathChanged();
    void onPixelSizeEdited();
    void onBoundsEdited();
    void onResolutionModeChanged();
    void onBoundsEnabledChanged();
    void onImageListToggled(bool expanded);
    void onImageSelectionChanged();
    void restoreDemPixelSize();
    void resetBoundsToDem();
    void estimateNow();

private:
    void setupUi();
    void connectUi();
    void scheduleEstimate();
    bool runEstimate(bool reportError);
    void invalidateDemEstimate();
    void setRunning(bool running);
    void updateControlAvailability();
    void updateImageSummary();
    void updateEstimateSummary(const QJsonObject &estimate);
    void updateLocalEstimateSummary();
    void applyImageReadiness();
    void applyPendingImageSelection();
    QStringList selectedImages() const;
    QJsonObject currentSettings() const;
    bool validateSettings(const QJsonObject &settings, QString *errorMessage) const;

    QScrollArea *_contentScrollArea = nullptr;
    QGroupBox *_projectionGroup = nullptr;
    QGroupBox *_parametersGroup = nullptr;
    QGroupBox *_regionGroup = nullptr;
    QGroupBox *_outputGroup = nullptr;
    QGroupBox *_progressGroup = nullptr;

    QRadioButton *_demGridProjectionRadio = nullptr;
    QRadioButton *_planarProjectionRadio = nullptr;
    QRadioButton *_cylindricalProjectionRadio = nullptr;
    QLabel *_coordinateSystemLabel = nullptr;

    QComboBox *_surfaceCombo = nullptr;
    QLineEdit *_demEdit = nullptr;
    QPushButton *_demBrowseButton = nullptr;
    QComboBox *_blendCombo = nullptr;
    QComboBox *_colorSourceCombo = nullptr;
    QCheckBox *_refineSeamsCheck = nullptr;
    QCheckBox *_fillHolesCheck = nullptr;
    QCheckBox *_ghostFilterCheck = nullptr;
    QCheckBox *_colorCorrectionCheck = nullptr;
    QCheckBox *_sharpnessWeightingCheck = nullptr;
    QCheckBox *_useProjectMasksCheck = nullptr;
    QLabel *_imageReadinessLabel = nullptr;
    QToolButton *_imageToggleButton = nullptr;
    QWidget *_imagePanel = nullptr;
    QListWidget *_imageList = nullptr;

    QRadioButton *_pixelSizeRadio = nullptr;
    QRadioButton *_maximumDimensionRadio = nullptr;
    QDoubleSpinBox *_pixelSizeXSpin = nullptr;
    QDoubleSpinBox *_pixelSizeYSpin = nullptr;
    QPushButton *_restorePixelSizeButton = nullptr;
    QSpinBox *_maximumDimensionSpin = nullptr;
    QCheckBox *_boundsEnabledCheck = nullptr;
    QDoubleSpinBox *_minXSpin = nullptr;
    QDoubleSpinBox *_maxXSpin = nullptr;
    QDoubleSpinBox *_minYSpin = nullptr;
    QDoubleSpinBox *_maxYSpin = nullptr;
    QPushButton *_resetBoundsButton = nullptr;
    QPushButton *_estimateButton = nullptr;
    QLabel *_totalSizeLabel = nullptr;
    QLabel *_memoryEstimateLabel = nullptr;

    QLineEdit *_outputEdit = nullptr;
    QPushButton *_outputBrowseButton = nullptr;

    QLabel *_stageLabel = nullptr;
    QLabel *_statusLabel = nullptr;
    QProgressBar *_progressBar = nullptr;
    QDialogButtonBox *_buttonBox = nullptr;
    QPushButton *_runButton = nullptr;
    QPushButton *_cancelButton = nullptr;
    QTimer *_estimateTimer = nullptr;

    QString _projectRoot;
    QStringList _cameraReadyImages;
    QStringList _pendingSelectedImages;
    QJsonObject _lastEstimate;
    double _demPixelSizeX = 0.0;
    double _demPixelSizeY = 0.0;
    double _demMinX = 0.0;
    double _demMaxX = 0.0;
    double _demMinY = 0.0;
    double _demMaxY = 0.0;
    int _cameraReadyCount = 0;
    int _maskCount = 0;
    bool _hasPendingImageSelection = false;
    bool _imageReadinessSet = false;
    bool _requestedUseProjectMasks = false;
    bool _hasDemEstimate = false;
    bool _pixelSizeUserEdited = false;
    bool _boundsUserEdited = false;
    bool _applyingSettings = false;
    bool _updatingEstimate = false;
    bool _running = false;
    bool _cancelRequested = false;
    bool _hasRunFinished = false;
};
