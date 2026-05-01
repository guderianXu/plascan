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
class QWidget;
class QCheckBox;

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
    void onRun();
    void onApplyResult();
    void onDiscardResult();
    void emitSettingsNow();

private:
    // 创建一个可折叠分组（基础/高级/系统/调试）。
    // title: 分组标题；expandedByDefault: 是否默认展开；contentOut: 返回可放置控件的内容容器。
    QWidget* createCollapsibleGroup(const QString &title, bool expandedByDefault, QWidget **contentOut);
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
