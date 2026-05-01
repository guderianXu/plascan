// =============================================================================
// 文件: DenseMatchDialog.h
// 功能: 密集匹配参数配置对话框
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QListWidget;
class QTableWidget;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLineEdit;
class QPushButton;
class QGroupBox;
class QLabel;
class ProjectManager;

class DenseMatchDialog : public QDialog
{
    Q_OBJECT
public:
    explicit DenseMatchDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

    void applySettings(const QJsonObject &settings);
    QJsonObject collectSettings() const;
    void onProcessingFinished();

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void onRun();
    void onBrowseOutput();
    void onAlgorithmChanged(int index);
    void onSelectAll();
    void onDeselectAll();
    void onImageSelectionChanged();
    void emitSettingsNow();

private:
    void setupUi();
    void loadProjectImages();
    void refreshMatchPairs();

    struct MatchPairInfo
    {
        QString imgA;
        QString imgB;
        QString matchFile;
        int     numMatches = 0;
    };

    ProjectManager *m_projectManager = nullptr;

    QListWidget   *m_imageList       = nullptr;
    QPushButton   *m_selectAllBtn    = nullptr;
    QPushButton   *m_deselectAllBtn  = nullptr;
    QStringList    m_allImages;

    QTableWidget  *m_matchTable      = nullptr;
    QLabel        *m_matchCountLabel = nullptr;
    QList<MatchPairInfo> m_matchPairs;

    QLineEdit     *m_outputEdit      = nullptr;

    QComboBox     *m_algorithmCombo  = nullptr;
    QComboBox     *m_costFuncCombo   = nullptr;
    QComboBox     *m_subpixelCombo   = nullptr;
    QSpinBox      *m_minDispSpin     = nullptr;
    QSpinBox      *m_maxDispSpin     = nullptr;
    QSpinBox      *m_kernelWSpin     = nullptr;
    QSpinBox      *m_kernelHSpin     = nullptr;

    QSpinBox      *m_p1Spin          = nullptr;
    QSpinBox      *m_p2Spin          = nullptr;
    QSpinBox      *m_directionsSpin  = nullptr;
    QSpinBox      *m_pyramidSpin     = nullptr;

    QCheckBox     *m_useCudaChk      = nullptr;
    QSpinBox      *m_deviceSpin      = nullptr;
    QSpinBox      *m_threadsSpin     = nullptr;
    QCheckBox     *m_opencvCompareChk = nullptr;

    QDoubleSpinBox *m_lrThresholdSpin = nullptr;
    QSpinBox      *m_medianFilterSpin = nullptr;

    QPushButton   *m_runBtn    = nullptr;
    QPushButton   *m_cancelBtn = nullptr;
    QPushButton   *m_resetBtn  = nullptr;
};
