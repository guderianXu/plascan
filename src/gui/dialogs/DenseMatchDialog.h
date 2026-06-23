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

    ProjectManager *_projectManager = nullptr;

    QListWidget *_imageList = nullptr;
    QPushButton *_selectAllBtn = nullptr;
    QPushButton *_deselectAllBtn = nullptr;
    QStringList _allImages;

    QTableWidget *_matchTable = nullptr;
    QLabel *_matchCountLabel = nullptr;
    QList<MatchPairInfo> _matchPairs;

    QLineEdit *_outputEdit = nullptr;

    QComboBox *_algorithmCombo = nullptr;
    QComboBox *_costFuncCombo = nullptr;
    QComboBox *_subpixelCombo = nullptr;
    QSpinBox *_minDispSpin = nullptr;
    QSpinBox *_maxDispSpin = nullptr;
    QSpinBox *_kernelWSpin = nullptr;
    QSpinBox *_kernelHSpin = nullptr;

    QSpinBox *_p1Spin = nullptr;
    QSpinBox *_p2Spin = nullptr;
    QSpinBox *_directionsSpin = nullptr;
    QSpinBox *_pyramidSpin = nullptr;

    QCheckBox *_useCudaChk = nullptr;
    QSpinBox *_deviceSpin = nullptr;
    QSpinBox *_threadsSpin = nullptr;
    QCheckBox *_opencvCompareChk = nullptr;

    QDoubleSpinBox *_lrThresholdSpin = nullptr;
    QSpinBox *_medianFilterSpin = nullptr;

    QPushButton *_runBtn = nullptr;
    QPushButton *_cancelBtn = nullptr;
    QPushButton *_resetBtn = nullptr;
};
