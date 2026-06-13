#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

#include "VocabularyOverlapRetriever.h"

#include <atomic>
#include <memory>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
template <typename T>
class QFutureWatcher;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QGroupBox;
class QSpinBox;
class QTableWidget;
class ProjectManager;

namespace Ui { class VocabularyOverlapDialog; }

class VocabularyOverlapDialog : public QDialog
{
    Q_OBJECT

public:
    struct RunResult;

    explicit VocabularyOverlapDialog(ProjectManager *projectManager, QWidget *parent = nullptr);
    ~VocabularyOverlapDialog() override;

signals:
    void settingsChanged(const QJsonObject &settings);
    void generatedPairsReady(const QStringList &pairs, const QJsonObject &settings);
    void overlapProgressChanged(const QString &stage, int percent);
    void overlapFinished(bool success);
    void overlapCancelRequested();

public slots:
    void applySettings(const QJsonObject &settings);
    void setProjectImages(const QStringList &paths);
    void cancelRun();

private slots:
    void onBrowseFeatureDir();
    void onAutoDetectFeatureDir();
    void onRun();
    void onExportLis();
    void onApplyToMatching();
    void onResetDefaults();
    void handleProgress(const QString &stage, int percent);

private:
    void setupUi();
    void setupConnections();
    void emitSettingsNow();
    void refreshFeatureStatus();
    QJsonObject collectSettings() const;
    QString selectedFeatureSuffix() const;
    QStringList checkedImages() const;
    QString pairTokenForImage(const QString &imagePath) const;
    bool loadFeatures(std::vector<xjw::VocabularyImageFeatures> *features, QString *errorMsg);
    bool writeOutputs(const QJsonObject &settings, const QStringList &pairs, QString *errorMsg) const;
    void populatePairTable();
    void setUiBusy(bool busy, const QString &message = QString());
    void handleRunFinished(QFutureWatcher<RunResult> *watcher);
    void updateMethodUi(bool refreshSummary = true);

    ProjectManager *m_projectManager = nullptr;
    QFutureWatcher<RunResult> *m_runWatcher = nullptr;
    std::shared_ptr<std::atomic_bool> m_cancelFlag;
    QStringList m_projectImages;
    QStringList m_generatedPairs;
    QJsonObject m_lastRunSettings;
    std::vector<xjw::VocabularyOverlapPairResult> m_candidatePairs;

    QListWidget *m_imageList = nullptr;
    QPushButton *m_selectAllBtn = nullptr;
    QPushButton *m_clearSelectionBtn = nullptr;
    QComboBox *m_overlapMethodCombo = nullptr;
    QComboBox *m_referenceBodyCombo = nullptr;
    QCheckBox *m_autoReferenceElevationCheck = nullptr;
    QDoubleSpinBox *m_referenceElevationSpin = nullptr;
    QDoubleSpinBox *m_cameraNeighborFactorSpin = nullptr;
    QComboBox *m_featureAlgorithmCombo = nullptr;
    QLineEdit *m_featureDirEdit = nullptr;
    QPushButton *m_autoDetectFeatureDirBtn = nullptr;
    QPushButton *m_browseFeatureDirBtn = nullptr;
    QSpinBox *m_branchFactorSpin = nullptr;
    QSpinBox *m_treeDepthSpin = nullptr;
    QSpinBox *m_samplePerImageSpin = nullptr;
    QSpinBox *m_maxTrainingDescriptorsSpin = nullptr;
    QSpinBox *m_topKSpin = nullptr;
    QDoubleSpinBox *m_minSimilaritySpin = nullptr;
    QCheckBox *m_useTfidfCheck = nullptr;
    QCheckBox *m_mutualTopKCheck = nullptr;
    QCheckBox *m_enableGeometryCheck = nullptr;
    QSpinBox *m_minInliersSpin = nullptr;
    QDoubleSpinBox *m_ransacThresholdSpin = nullptr;
    QComboBox *m_geometryModelCombo = nullptr;
    QSpinBox *m_overlapThreadsSpin = nullptr;
    QCheckBox *m_useFlannAssignmentCheck = nullptr;
    QCheckBox *m_useInvertedIndexCheck = nullptr;
    QCheckBox *m_useCudaOverlapCheck = nullptr;
    QSpinBox *m_geometryMaxDescriptorsSpin = nullptr;
    QSpinBox *m_geometryMaxPairsSpin = nullptr;
    QLineEdit *m_outputJsonEdit = nullptr;
    QLineEdit *m_outputLisEdit = nullptr;
    QCheckBox *m_applyToMatchingCheck = nullptr;
    QTableWidget *m_pairTable = nullptr;
    QLabel *m_summaryLabel = nullptr;
    QPushButton *m_resetBtn = nullptr;
    QPushButton *m_exportLisBtn = nullptr;
    QPushButton *m_applyToMatchingBtn = nullptr;
    QPushButton *m_runBtn = nullptr;
    QPushButton *m_closeBtn = nullptr;
};
