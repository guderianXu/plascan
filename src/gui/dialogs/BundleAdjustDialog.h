#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QListWidget;
class QLineEdit;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QTableWidget;
class QToolButton;

class BundleAdjustDialog : public QDialog
{
    Q_OBJECT
public:
    explicit BundleAdjustDialog(QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setDefaultOutputDir(const QString &dirPath);
    void applySettings(const QJsonObject &settings);
    void setRunResult(const QJsonObject &result);

signals:
    void requestRunBundleAdjust(const QStringList &images,
                                const QString &outputDir,
                                int threads,
                                bool dryRun,
                                const QJsonObject &options);
    void requestApplyBundleAdjustResult();
    void requestDiscardBundleAdjustResult();
    void settingsChanged(const QJsonObject &settings);
    void requestRestore();

private slots:
    void onChooseOutputDir();
    void onChooseLaserConstraintCloud();
    void onRun();
    void onApplyResult();
    void onDiscardResult();
    void updateLaserControls();
    void emitSettingsNow();

private:
    // 统一更新结果区域按钮状态。
    void updateResultButtons();
    QStringList selectedImages() const;

    QListWidget *_imageList = nullptr;
    QLineEdit *_outputDirEdit = nullptr;
    QSpinBox *_threadsSpin = nullptr;
    QSpinBox *_chunkSizeSpin = nullptr;
    QSpinBox *_maxIterationsSpin = nullptr;
    QSpinBox *_maxPointItersSpin = nullptr;
    QSpinBox *_maxCameraItersSpin = nullptr;
    QSpinBox *_minMatchesSpin = nullptr;
    QDoubleSpinBox *_huberDeltaSpin = nullptr;
    QDoubleSpinBox *_dampingSpin = nullptr;
    QDoubleSpinBox *_finiteDiffSpin = nullptr;
    QDoubleSpinBox *_stepTolSpin = nullptr;
    QCheckBox *_refinePoseCheck = nullptr;
    QCheckBox *_dryRunCheck = nullptr;

    QCheckBox *_enableLaserConstraintsCheck = nullptr;
    QLineEdit *_laserConstraintCloudEdit = nullptr;
    QToolButton *_chooseLaserConstraintCloudBtn = nullptr;
    QDoubleSpinBox *_laserAssociationMaxDistanceSpin = nullptr;
    QDoubleSpinBox *_laserVoxelSizeSpin = nullptr;
    QDoubleSpinBox *_laserMaxCurvatureSpin = nullptr;
    QSpinBox *_laserMaxSamplesSpin = nullptr;
    QCheckBox *_laserMissingNormalsAsHeightPlanesCheck = nullptr;
    QDoubleSpinBox *_laserWeightSpin = nullptr;
    QDoubleSpinBox *_laserHuberDeltaSpin = nullptr;

    // 调试参数：控制输出哪些文件与可视化评估图。
    QCheckBox *_exportTsaiCheck = nullptr;
    QCheckBox *_exportSummaryTxtCheck = nullptr;
    QCheckBox *_exportPointsCsvCheck = nullptr;
    QCheckBox *_exportCameraCsvCheck = nullptr;
    QCheckBox *_exportRunJsonCheck = nullptr;
    QCheckBox *_exportEvalPlotCheck = nullptr;

    QLabel *_resultSummaryLabel = nullptr;
    QTableWidget *_resultCameraTable = nullptr;
    QToolButton *_applyResultBtn = nullptr;
    QToolButton *_discardResultBtn = nullptr;
    QStringList _savedSelectedImages;
    bool _hasPendingResult = false;
    bool _suppressSettingsChanged = false;
};
