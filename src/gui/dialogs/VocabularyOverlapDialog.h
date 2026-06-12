#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

#include "VocabularyOverlapRetriever.h"

#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QTableWidget;
class ProjectManager;

namespace Ui { class VocabularyOverlapDialog; }

class VocabularyOverlapDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VocabularyOverlapDialog(ProjectManager *projectManager, QWidget *parent = nullptr);
    ~VocabularyOverlapDialog() override;

signals:
    void settingsChanged(const QJsonObject &settings);
    void generatedPairsReady(const QStringList &pairs, const QJsonObject &settings);

public slots:
    void applySettings(const QJsonObject &settings);
    void setProjectImages(const QStringList &paths);

private slots:
    void onBrowseFeatureDir();
    void onAutoDetectFeatureDir();
    void onRun();
    void onExportLis();
    void onApplyToMatching();
    void onResetDefaults();

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

    ProjectManager *m_projectManager = nullptr;
    QStringList m_projectImages;
    QStringList m_generatedPairs;
    QJsonObject m_lastRunSettings;
    std::vector<xjw::VocabularyOverlapPairResult> m_candidatePairs;

    QListWidget *m_imageList = nullptr;
    QPushButton *m_selectAllBtn = nullptr;
    QPushButton *m_clearSelectionBtn = nullptr;
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
