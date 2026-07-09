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
    QString defaultFeatureSuffix() const;
    QString selectedFeatureSuffix() const;
    QStringList checkedImages() const;
    QString pairTokenForImage(const QString &imagePath) const;
    bool loadFeatures(std::vector<xjw::VocabularyImageFeatures> *features, QString *errorMsg);
    bool writeOutputs(const QJsonObject &settings, const QStringList &pairs, QString *errorMsg) const;
    void populatePairTable();
    void setUiBusy(bool busy, const QString &message = QString());
    void handleRunFinished(QFutureWatcher<RunResult> *watcher);
    void updateMethodUi(bool refreshSummary = true);

    ProjectManager *_projectManager = nullptr;
    QFutureWatcher<RunResult> *_runWatcher = nullptr;
    std::shared_ptr<std::atomic_bool> _cancelFlag;
    QStringList _projectImages;
    QStringList _generatedPairs;
    QJsonObject _lastRunSettings;
    std::vector<xjw::VocabularyOverlapPairResult> _candidatePairs;

    QListWidget *_imageList = nullptr;
    QPushButton *_selectAllBtn = nullptr;
    QPushButton *_clearSelectionBtn = nullptr;
    QComboBox *_overlapMethodCombo = nullptr;
    QComboBox *_referenceBodyCombo = nullptr;
    QCheckBox *_autoReferenceElevationCheck = nullptr;
    QDoubleSpinBox *_referenceElevationSpin = nullptr;
    QDoubleSpinBox *_cameraNeighborFactorSpin = nullptr;
    QComboBox *_featureAlgorithmCombo = nullptr;
    QLineEdit *_featureDirEdit = nullptr;
    QPushButton *_autoDetectFeatureDirBtn = nullptr;
    QPushButton *_browseFeatureDirBtn = nullptr;
    QSpinBox *_branchFactorSpin = nullptr;
    QSpinBox *_treeDepthSpin = nullptr;
    QSpinBox *_samplePerImageSpin = nullptr;
    QSpinBox *_maxTrainingDescriptorsSpin = nullptr;
    QSpinBox *_topKSpin = nullptr;
    QDoubleSpinBox *_minSimilaritySpin = nullptr;
    QCheckBox *_useTfidfCheck = nullptr;
    QCheckBox *_mutualTopKCheck = nullptr;
    QCheckBox *_enableGeometryCheck = nullptr;
    QSpinBox *_minInliersSpin = nullptr;
    QDoubleSpinBox *_ransacThresholdSpin = nullptr;
    QComboBox *_geometryModelCombo = nullptr;
    QSpinBox *_overlapThreadsSpin = nullptr;
    QCheckBox *_useFlannAssignmentCheck = nullptr;
    QCheckBox *_useInvertedIndexCheck = nullptr;
    QCheckBox *_useCudaOverlapCheck = nullptr;
    QSpinBox *_geometryMaxDescriptorsSpin = nullptr;
    QSpinBox *_geometryMaxPairsSpin = nullptr;
    QLineEdit *_outputJsonEdit = nullptr;
    QLineEdit *_outputLisEdit = nullptr;
    QCheckBox *_applyToMatchingCheck = nullptr;
    QTableWidget *_pairTable = nullptr;
    QLabel *_summaryLabel = nullptr;
    QPushButton *_resetBtn = nullptr;
    QPushButton *_exportLisBtn = nullptr;
    QPushButton *_applyToMatchingBtn = nullptr;
    QPushButton *_runBtn = nullptr;
    QPushButton *_closeBtn = nullptr;
};
