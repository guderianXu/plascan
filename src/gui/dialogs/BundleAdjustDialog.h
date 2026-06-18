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

    QListWidget *m_imageList = nullptr;
    QLineEdit *m_outputDirEdit = nullptr;
    QSpinBox *m_threadsSpin = nullptr;
    QSpinBox *m_chunkSizeSpin = nullptr;
    QSpinBox *m_maxIterationsSpin = nullptr;
    QSpinBox *m_maxPointItersSpin = nullptr;
    QSpinBox *m_maxCameraItersSpin = nullptr;
    QSpinBox *m_minMatchesSpin = nullptr;
    QDoubleSpinBox *m_huberDeltaSpin = nullptr;
    QDoubleSpinBox *m_dampingSpin = nullptr;
    QDoubleSpinBox *m_finiteDiffSpin = nullptr;
    QDoubleSpinBox *m_stepTolSpin = nullptr;
    QCheckBox *m_refinePoseCheck = nullptr;
    QCheckBox *m_dryRunCheck = nullptr;

    QCheckBox *m_enableLaserConstraintsCheck = nullptr;
    QLineEdit *m_laserConstraintCloudEdit = nullptr;
    QToolButton *m_chooseLaserConstraintCloudBtn = nullptr;
    QDoubleSpinBox *m_laserAssociationMaxDistanceSpin = nullptr;
    QDoubleSpinBox *m_laserVoxelSizeSpin = nullptr;
    QDoubleSpinBox *m_laserMaxCurvatureSpin = nullptr;
    QSpinBox *m_laserMaxSamplesSpin = nullptr;
    QDoubleSpinBox *m_laserWeightSpin = nullptr;
    QDoubleSpinBox *m_laserHuberDeltaSpin = nullptr;

    // 调试参数：控制输出哪些文件与可视化评估图。
    QCheckBox *m_exportTsaiCheck = nullptr;
    QCheckBox *m_exportSummaryTxtCheck = nullptr;
    QCheckBox *m_exportPointsCsvCheck = nullptr;
    QCheckBox *m_exportCameraCsvCheck = nullptr;
    QCheckBox *m_exportRunJsonCheck = nullptr;
    QCheckBox *m_exportEvalPlotCheck = nullptr;

    QLabel *m_resultSummaryLabel = nullptr;
    QTableWidget *m_resultCameraTable = nullptr;
    QToolButton *m_applyResultBtn = nullptr;
    QToolButton *m_discardResultBtn = nullptr;
    QStringList m_savedSelectedImages;
    bool m_hasPendingResult = false;
};
