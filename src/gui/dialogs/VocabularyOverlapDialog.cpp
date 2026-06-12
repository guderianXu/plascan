#include "compat/QtTorchMacroGuard.h"

#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#include "VocabularyOverlapDialog.h"
#include "ui_VocabularyOverlapDialog.h"

#include "ProjectIO.h"
#include "ProjectManager.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextStream>

#include <cstring>

namespace
{

QString defaultOverlapOutputDir(const QString &projectPath)
{
    if (projectPath.isEmpty())
    {
        return QString();
    }
    return QDir(ProjectIO::projectAssetsDir(projectPath)).filePath(QStringLiteral("overlap"));
}

QString featurePathInDir(const QString &featureDir, const QString &imagePath, const QString &suffix)
{
    if (featureDir.trimmed().isEmpty() || suffix.trimmed().isEmpty())
    {
        return QString();
    }
    return QDir(featureDir).filePath(QFileInfo(imagePath).completeBaseName() + suffix);
}

QJsonArray stringListToJsonArray(const QStringList &values)
{
    QJsonArray array;
    for (const QString &value : values)
    {
        array.append(value);
    }
    return array;
}

cv::Mat tensorToCvMat(const torch::Tensor &tensor)
{
    if (!tensor.defined() || tensor.numel() <= 0 || tensor.dim() != 2)
    {
        return cv::Mat();
    }

    torch::Tensor cpu = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const int rows = static_cast<int>(cpu.size(0));
    const int cols = static_cast<int>(cpu.size(1));
    cv::Mat descriptors(rows, cols, CV_32F);
    std::memcpy(descriptors.ptr<float>(0), cpu.data_ptr<float>(), static_cast<std::size_t>(rows * cols) * sizeof(float));
    return descriptors;
}

void setTableItem(QTableWidget *table, int row, int column, const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    table->setItem(row, column, item);
}

} // namespace

VocabularyOverlapDialog::VocabularyOverlapDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setupUi();
    setupConnections();
    onResetDefaults();
    if (m_projectManager)
    {
        setProjectImages(m_projectManager->getAllImages());
    }
}

VocabularyOverlapDialog::~VocabularyOverlapDialog() = default;

void VocabularyOverlapDialog::setupUi()
{
    Ui::VocabularyOverlapDialog ui;
    ui.setupUi(this);

    m_imageList = ui.m_imageList;
    m_selectAllBtn = ui.m_selectAllBtn;
    m_clearSelectionBtn = ui.m_clearSelectionBtn;
    m_featureAlgorithmCombo = ui.m_featureAlgorithmCombo;
    m_featureDirEdit = ui.m_featureDirEdit;
    m_autoDetectFeatureDirBtn = ui.m_autoDetectFeatureDirBtn;
    m_browseFeatureDirBtn = ui.m_browseFeatureDirBtn;
    m_branchFactorSpin = ui.m_branchFactorSpin;
    m_treeDepthSpin = ui.m_treeDepthSpin;
    m_samplePerImageSpin = ui.m_samplePerImageSpin;
    m_maxTrainingDescriptorsSpin = ui.m_maxTrainingDescriptorsSpin;
    m_topKSpin = ui.m_topKSpin;
    m_minSimilaritySpin = ui.m_minSimilaritySpin;
    m_useTfidfCheck = ui.m_useTfidfCheck;
    m_mutualTopKCheck = ui.m_mutualTopKCheck;
    m_enableGeometryCheck = ui.m_enableGeometryCheck;
    m_minInliersSpin = ui.m_minInliersSpin;
    m_ransacThresholdSpin = ui.m_ransacThresholdSpin;
    m_geometryModelCombo = ui.m_geometryModelCombo;
    m_outputJsonEdit = ui.m_outputJsonEdit;
    m_outputLisEdit = ui.m_outputLisEdit;
    m_applyToMatchingCheck = ui.m_applyToMatchingCheck;
    m_pairTable = ui.m_pairTable;
    m_summaryLabel = ui.m_summaryLabel;
    m_resetBtn = ui.m_resetBtn;
    m_exportLisBtn = ui.m_exportLisBtn;
    m_applyToMatchingBtn = ui.m_applyToMatchingBtn;
    m_runBtn = ui.m_runBtn;
    m_closeBtn = ui.m_closeBtn;

    ui.topSplit->setStretchFactor(0, 3);
    ui.topSplit->setStretchFactor(1, 2);

    m_featureAlgorithmCombo->clear();
    m_featureAlgorithmCombo->addItem(QStringLiteral("DISK (.dsk)"), QStringLiteral(".dsk"));
    m_featureAlgorithmCombo->addItem(QStringLiteral("ALIKED (.alk)"), QStringLiteral(".alk"));
    m_featureAlgorithmCombo->addItem(QStringLiteral("SuperPoint (.sp)"), QStringLiteral(".sp"));
    m_featureAlgorithmCombo->addItem(QStringLiteral("SIFT (.sift)"), QStringLiteral(".sift"));
    m_featureAlgorithmCombo->addItem(QStringLiteral("ORB (.orb)"), QStringLiteral(".orb"));

    m_geometryModelCombo->clear();
    m_geometryModelCombo->addItem(QStringLiteral("Fundamental Matrix"), QStringLiteral("fundamental"));

    m_pairTable->setColumnCount(6);
    m_pairTable->setHorizontalHeaderLabels(QStringList()
                                           << QStringLiteral("影像 1")
                                           << QStringLiteral("影像 2")
                                           << QStringLiteral("相似度")
                                           << QStringLiteral("共享词")
                                           << QStringLiteral("几何内点")
                                           << QStringLiteral("状态"));
    m_pairTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_pairTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
}

void VocabularyOverlapDialog::setupConnections()
{
    connect(m_selectAllBtn, &QPushButton::clicked, this, [this]()
    {
        for (int i = 0; i < m_imageList->count(); ++i)
        {
            m_imageList->item(i)->setCheckState(Qt::Checked);
        }
        emitSettingsNow();
    });

    connect(m_clearSelectionBtn, &QPushButton::clicked, this, [this]()
    {
        for (int i = 0; i < m_imageList->count(); ++i)
        {
            m_imageList->item(i)->setCheckState(Qt::Unchecked);
        }
        emitSettingsNow();
    });

    connect(m_featureAlgorithmCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]()
    {
        refreshFeatureStatus();
        emitSettingsNow();
    });
    connect(m_featureDirEdit, &QLineEdit::editingFinished, this, [this]()
    {
        refreshFeatureStatus();
        emitSettingsNow();
    });
    connect(m_browseFeatureDirBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onBrowseFeatureDir);
    connect(m_autoDetectFeatureDirBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onAutoDetectFeatureDir);
    connect(m_runBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onRun);
    connect(m_exportLisBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onExportLis);
    connect(m_applyToMatchingBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onApplyToMatching);
    connect(m_resetBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onResetDefaults);
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    const auto emitNow = [this]() { emitSettingsNow(); };
    connect(m_branchFactorSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_treeDepthSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_samplePerImageSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_maxTrainingDescriptorsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_topKSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_minSimilaritySpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(m_useTfidfCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_mutualTopKCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_enableGeometryCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_minInliersSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_ransacThresholdSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(m_outputJsonEdit, &QLineEdit::editingFinished, this, emitNow);
    connect(m_outputLisEdit, &QLineEdit::editingFinished, this, emitNow);
    connect(m_applyToMatchingCheck, &QCheckBox::toggled, this, emitNow);
}

void VocabularyOverlapDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty())
    {
        return;
    }

    QSignalBlocker blockAlgorithm(m_featureAlgorithmCombo);
    QSignalBlocker blockFeatureDir(m_featureDirEdit);

    const QString suffix = settings.value(QStringLiteral("feature_suffix")).toString(QStringLiteral(".dsk"));
    const int algorithmIndex = m_featureAlgorithmCombo->findData(suffix);
    if (algorithmIndex >= 0)
    {
        m_featureAlgorithmCombo->setCurrentIndex(algorithmIndex);
    }

    m_featureDirEdit->setText(settings.value(QStringLiteral("feature_dir")).toString(m_featureDirEdit->text()));
    m_branchFactorSpin->setValue(settings.value(QStringLiteral("branch_factor")).toInt(m_branchFactorSpin->value()));
    m_treeDepthSpin->setValue(settings.value(QStringLiteral("tree_depth")).toInt(m_treeDepthSpin->value()));
    m_samplePerImageSpin->setValue(settings.value(QStringLiteral("sample_per_image")).toInt(m_samplePerImageSpin->value()));
    m_maxTrainingDescriptorsSpin->setValue(
        settings.value(QStringLiteral("max_training_descriptors")).toInt(m_maxTrainingDescriptorsSpin->value()));
    m_topKSpin->setValue(settings.value(QStringLiteral("top_k")).toInt(m_topKSpin->value()));
    m_minSimilaritySpin->setValue(settings.value(QStringLiteral("min_similarity")).toDouble(m_minSimilaritySpin->value()));
    m_useTfidfCheck->setChecked(settings.value(QStringLiteral("use_tfidf")).toBool(m_useTfidfCheck->isChecked()));
    m_mutualTopKCheck->setChecked(settings.value(QStringLiteral("mutual_top_k")).toBool(m_mutualTopKCheck->isChecked()));
    m_enableGeometryCheck->setChecked(
        settings.value(QStringLiteral("geometry_check")).toBool(m_enableGeometryCheck->isChecked()));
    m_minInliersSpin->setValue(settings.value(QStringLiteral("min_inliers")).toInt(m_minInliersSpin->value()));
    m_ransacThresholdSpin->setValue(
        settings.value(QStringLiteral("ransac_threshold")).toDouble(m_ransacThresholdSpin->value()));
    m_outputJsonEdit->setText(settings.value(QStringLiteral("output_json")).toString(m_outputJsonEdit->text()));
    m_outputLisEdit->setText(settings.value(QStringLiteral("output_lis")).toString(m_outputLisEdit->text()));
    m_applyToMatchingCheck->setChecked(
        settings.value(QStringLiteral("apply_to_matching")).toBool(m_applyToMatchingCheck->isChecked()));

    m_generatedPairs.clear();
    const QJsonArray generatedPairs = settings.value(QStringLiteral("generated_pairs")).toArray();
    for (const QJsonValue &value : generatedPairs)
    {
        const QString pair = value.toString().trimmed();
        if (!pair.isEmpty())
        {
            m_generatedPairs.append(pair);
        }
    }

    refreshFeatureStatus();
}

void VocabularyOverlapDialog::setProjectImages(const QStringList &paths)
{
    m_projectImages = paths;
    m_imageList->clear();

    for (const QString &path : paths)
    {
        auto *item = new QListWidgetItem(path, m_imageList);
        item->setData(Qt::UserRole, path);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
    }

    refreshFeatureStatus();
}

void VocabularyOverlapDialog::onBrowseFeatureDir()
{
    const QString dir = QFileDialog::getExistingDirectory(this,
                                                          QStringLiteral("选择特征目录"),
                                                          m_featureDirEdit->text());
    if (dir.isEmpty())
    {
        return;
    }
    m_featureDirEdit->setText(QDir::cleanPath(dir));
    refreshFeatureStatus();
    emitSettingsNow();
}

void VocabularyOverlapDialog::onAutoDetectFeatureDir()
{
    if (!m_projectManager)
    {
        return;
    }
    const QString projectPath = m_projectManager->currentProjectPath();
    m_featureDirEdit->setText(ProjectIO::ipfindOutputDir(projectPath));
    refreshFeatureStatus();
    emitSettingsNow();
}

void VocabularyOverlapDialog::onRun()
{
    std::vector<xjw::VocabularyImageFeatures> features;
    QString error;
    if (!loadFeatures(&features, &error))
    {
        QMessageBox::warning(this, QStringLiteral("获取重叠对失败"), error);
        return;
    }

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = m_branchFactorSpin->value();
    config.treeDepth = m_treeDepthSpin->value();
    config.samplePerImage = m_samplePerImageSpin->value();
    config.maxTrainingDescriptors = m_maxTrainingDescriptorsSpin->value();
    config.topK = m_topKSpin->value();
    config.minSimilarity = m_minSimilaritySpin->value();
    config.useTfidf = m_useTfidfCheck->isChecked();
    config.mutualTopK = m_mutualTopKCheck->isChecked();
    config.geometryCheck = m_enableGeometryCheck->isChecked();
    config.minInliers = m_minInliersSpin->value();
    config.ransacThreshold = m_ransacThresholdSpin->value();

    xjw::VocabularyOverlapResult result;
    std::string coreError;
    if (!xjw::VocabularyOverlapRetriever::retrieve(features, config, &result, &coreError))
    {
        QMessageBox::warning(this, QStringLiteral("获取重叠对失败"), QString::fromStdString(coreError));
        return;
    }

    m_candidatePairs = result.candidates;
    m_generatedPairs.clear();
    for (const xjw::VocabularyOverlapPairResult &pair : result.acceptedPairs)
    {
        const QString imageA = QString::fromStdString(pair.imagePathA);
        const QString imageB = QString::fromStdString(pair.imagePathB);
        m_generatedPairs.append(pairTokenForImage(imageA) + QStringLiteral("__") + pairTokenForImage(imageB));
    }

    m_lastRunSettings = collectSettings();
    m_lastRunSettings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(m_generatedPairs));
    m_lastRunSettings.insert(QStringLiteral("vocabulary_size"), result.vocabularySize);
    m_lastRunSettings.insert(QStringLiteral("detail"), QString::fromStdString(result.detail));

    if (!writeOutputs(m_lastRunSettings, m_generatedPairs, &error))
    {
        QMessageBox::warning(this, QStringLiteral("写出重叠对失败"), error);
        return;
    }

    populatePairTable();
    m_summaryLabel->setText(QStringLiteral("候选 %1，对外输出 %2，词汇数 %3")
                                .arg(result.candidates.size())
                                .arg(m_generatedPairs.size())
                                .arg(result.vocabularySize));

    emit settingsChanged(m_lastRunSettings);
    if (m_applyToMatchingCheck->isChecked())
    {
        emit generatedPairsReady(m_generatedPairs, m_lastRunSettings);
    }
}

void VocabularyOverlapDialog::onExportLis()
{
    if (m_generatedPairs.isEmpty())
    {
        QMessageBox::information(this, QStringLiteral("导出 LIS"), QStringLiteral("当前没有可导出的重叠对。"));
        return;
    }

    QString error;
    const QJsonObject settings = m_lastRunSettings.isEmpty() ? collectSettings() : m_lastRunSettings;
    if (!writeOutputs(settings, m_generatedPairs, &error))
    {
        QMessageBox::warning(this, QStringLiteral("导出 LIS 失败"), error);
    }
}

void VocabularyOverlapDialog::onApplyToMatching()
{
    if (m_generatedPairs.isEmpty())
    {
        QMessageBox::information(this, QStringLiteral("应用到匹配"), QStringLiteral("请先运行获取重叠对。"));
        return;
    }

    QJsonObject settings = m_lastRunSettings.isEmpty() ? collectSettings() : m_lastRunSettings;
    settings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(m_generatedPairs));
    emit generatedPairsReady(m_generatedPairs, settings);
    emit settingsChanged(settings);
}

void VocabularyOverlapDialog::onResetDefaults()
{
    const QString projectPath = m_projectManager ? m_projectManager->currentProjectPath() : QString();
    const QString outputDir = defaultOverlapOutputDir(projectPath);

    const int diskIndex = m_featureAlgorithmCombo->findData(QStringLiteral(".dsk"));
    m_featureAlgorithmCombo->setCurrentIndex(diskIndex >= 0 ? diskIndex : 0);
    m_featureDirEdit->setText(projectPath.isEmpty() ? QString() : ProjectIO::ipfindOutputDir(projectPath));
    m_branchFactorSpin->setValue(10);
    m_treeDepthSpin->setValue(3);
    m_samplePerImageSpin->setValue(500);
    m_maxTrainingDescriptorsSpin->setValue(50000);
    m_topKSpin->setValue(8);
    m_minSimilaritySpin->setValue(0.05);
    m_useTfidfCheck->setChecked(true);
    m_mutualTopKCheck->setChecked(true);
    m_enableGeometryCheck->setChecked(false);
    m_minInliersSpin->setValue(30);
    m_ransacThresholdSpin->setValue(3.0);
    m_applyToMatchingCheck->setChecked(true);

    if (!outputDir.isEmpty())
    {
        m_outputJsonEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.json")));
        m_outputLisEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.lis")));
    }

    refreshFeatureStatus();
    emitSettingsNow();
}

void VocabularyOverlapDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void VocabularyOverlapDialog::refreshFeatureStatus()
{
    const QString suffix = selectedFeatureSuffix();
    const QString featureDir = m_featureDirEdit->text().trimmed();
    int existing = 0;

    for (int i = 0; i < m_imageList->count(); ++i)
    {
        QListWidgetItem *item = m_imageList->item(i);
        const QString imagePath = item->data(Qt::UserRole).toString();
        QString featurePath = featurePathInDir(featureDir, imagePath, suffix);
        if (!QFile::exists(featurePath) && m_projectManager)
        {
            featurePath = ProjectIO::featureFileForSuffix(m_projectManager->currentProjectPath(), imagePath, suffix);
        }
        const bool found = QFile::exists(featurePath);
        if (found)
        {
            ++existing;
        }
        item->setText(found
                          ? QStringLiteral("%1  [%2]").arg(imagePath, QFileInfo(featurePath).fileName())
                          : QStringLiteral("%1  [缺少%2]").arg(imagePath, suffix));
        item->setData(Qt::UserRole + 1, featurePath);
    }

    m_summaryLabel->setText(QStringLiteral("特征文件 %1/%2，当前算法 %3")
                                .arg(existing)
                                .arg(m_imageList->count())
                                .arg(suffix));
}

QJsonObject VocabularyOverlapDialog::collectSettings() const
{
    QJsonObject settings;
    settings.insert(QStringLiteral("feature_suffix"), selectedFeatureSuffix());
    settings.insert(QStringLiteral("feature_algorithm"), m_featureAlgorithmCombo->currentText());
    settings.insert(QStringLiteral("feature_dir"), m_featureDirEdit->text().trimmed());
    settings.insert(QStringLiteral("branch_factor"), m_branchFactorSpin->value());
    settings.insert(QStringLiteral("tree_depth"), m_treeDepthSpin->value());
    settings.insert(QStringLiteral("sample_per_image"), m_samplePerImageSpin->value());
    settings.insert(QStringLiteral("max_training_descriptors"), m_maxTrainingDescriptorsSpin->value());
    settings.insert(QStringLiteral("top_k"), m_topKSpin->value());
    settings.insert(QStringLiteral("min_similarity"), m_minSimilaritySpin->value());
    settings.insert(QStringLiteral("use_tfidf"), m_useTfidfCheck->isChecked());
    settings.insert(QStringLiteral("mutual_top_k"), m_mutualTopKCheck->isChecked());
    settings.insert(QStringLiteral("geometry_check"), m_enableGeometryCheck->isChecked());
    settings.insert(QStringLiteral("geometry_model"), m_geometryModelCombo->currentData().toString());
    settings.insert(QStringLiteral("min_inliers"), m_minInliersSpin->value());
    settings.insert(QStringLiteral("ransac_threshold"), m_ransacThresholdSpin->value());
    settings.insert(QStringLiteral("output_json"), m_outputJsonEdit->text().trimmed());
    settings.insert(QStringLiteral("output_lis"), m_outputLisEdit->text().trimmed());
    settings.insert(QStringLiteral("apply_to_matching"), m_applyToMatchingCheck->isChecked());
    settings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(m_generatedPairs));
    return settings;
}

QString VocabularyOverlapDialog::selectedFeatureSuffix() const
{
    const QString suffix = m_featureAlgorithmCombo->currentData().toString();
    return suffix.isEmpty() ? QStringLiteral(".dsk") : suffix;
}

QStringList VocabularyOverlapDialog::checkedImages() const
{
    QStringList images;
    for (int i = 0; i < m_imageList->count(); ++i)
    {
        const QListWidgetItem *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked)
        {
            images.append(item->data(Qt::UserRole).toString());
        }
    }
    return images;
}

QString VocabularyOverlapDialog::pairTokenForImage(const QString &imagePath) const
{
    return QFileInfo(imagePath).completeBaseName();
}

bool VocabularyOverlapDialog::loadFeatures(std::vector<xjw::VocabularyImageFeatures> *features, QString *errorMsg)
{
    if (!features)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("内部错误：特征输出指针为空。");
        }
        return false;
    }

    features->clear();
    const QStringList images = checkedImages();
    if (images.size() < 2)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("至少选择两张影像。");
        }
        return false;
    }

    const QString suffix = selectedFeatureSuffix();
    const QString featureDir = m_featureDirEdit->text().trimmed();
    for (const QString &imagePath : images)
    {
        QString featurePath = featurePathInDir(featureDir, imagePath, suffix);
        if (!QFile::exists(featurePath) && m_projectManager)
        {
            featurePath = ProjectIO::featureFileForSuffix(m_projectManager->currentProjectPath(), imagePath, suffix);
        }
        if (!QFile::exists(featurePath))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("缺少特征文件：%1\n期望后缀：%2").arg(imagePath, suffix);
            }
            return false;
        }

        QString storedImageName;
        FeatureOutput output;
        if (!FeatureFileIO::read(featurePath, storedImageName, output))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("无法读取特征文件：%1").arg(featurePath);
            }
            return false;
        }

        cv::Mat descriptors = tensorToCvMat(output.descriptors);
        if (descriptors.empty() || output.keypoints.size() != static_cast<std::size_t>(descriptors.rows))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("特征文件缺少有效描述子或关键点数量不一致：%1").arg(featurePath);
            }
            return false;
        }

        xjw::VocabularyImageFeatures imageFeatures;
        imageFeatures.imagePath = imagePath.toStdString();
        imageFeatures.keypoints = output.keypoints;
        imageFeatures.descriptors = descriptors;
        features->push_back(std::move(imageFeatures));
    }

    return true;
}

bool VocabularyOverlapDialog::writeOutputs(const QJsonObject &settings,
                                           const QStringList &pairs,
                                           QString *errorMsg) const
{
    const QString jsonPath = settings.value(QStringLiteral("output_json")).toString().trimmed();
    const QString lisPath = settings.value(QStringLiteral("output_lis")).toString().trimmed();

    if (!jsonPath.isEmpty())
    {
        QDir().mkpath(QFileInfo(jsonPath).absolutePath());
        QJsonObject root = settings;
        QJsonArray candidates;
        for (const xjw::VocabularyOverlapPairResult &pair : m_candidatePairs)
        {
            QJsonObject object;
            object.insert(QStringLiteral("image_a"), QString::fromStdString(pair.imagePathA));
            object.insert(QStringLiteral("image_b"), QString::fromStdString(pair.imagePathB));
            object.insert(QStringLiteral("pair_token"),
                          pairTokenForImage(QString::fromStdString(pair.imagePathA)) + QStringLiteral("__") +
                              pairTokenForImage(QString::fromStdString(pair.imagePathB)));
            object.insert(QStringLiteral("bow_score"), pair.bowScore);
            object.insert(QStringLiteral("shared_word_count"), pair.sharedWordCount);
            object.insert(QStringLiteral("geometric_inliers"), pair.geometricInliers);
            object.insert(QStringLiteral("accepted"), pair.accepted);
            object.insert(QStringLiteral("reject_reason"), QString::fromStdString(pair.rejectReason));
            candidates.append(object);
        }
        root.insert(QStringLiteral("candidates"), candidates);

        QFile jsonFile(jsonPath);
        if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("无法写入 JSON：%1").arg(jsonPath);
            }
            return false;
        }
        jsonFile.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        jsonFile.close();
    }

    if (!lisPath.isEmpty())
    {
        QDir().mkpath(QFileInfo(lisPath).absolutePath());
        QFile lisFile(lisPath);
        if (!lisFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("无法写入 LIS：%1").arg(lisPath);
            }
            return false;
        }
        QTextStream stream(&lisFile);
        if (!m_candidatePairs.empty())
        {
            for (const xjw::VocabularyOverlapPairResult &pair : m_candidatePairs)
            {
                if (!pair.accepted)
                {
                    continue;
                }
                stream << QString::fromStdString(pair.imagePathA) << ' '
                       << QString::fromStdString(pair.imagePathB) << '\n';
            }
        }
        else
        {
            for (const QString &pair : pairs)
            {
                const QStringList parts = pair.split(QStringLiteral("__"));
                if (parts.size() == 2)
                {
                    stream << parts[0] << ' ' << parts[1] << '\n';
                }
            }
        }
        lisFile.close();
    }

    return true;
}

void VocabularyOverlapDialog::populatePairTable()
{
    m_pairTable->setRowCount(static_cast<int>(m_candidatePairs.size()));
    for (int row = 0; row < static_cast<int>(m_candidatePairs.size()); ++row)
    {
        const xjw::VocabularyOverlapPairResult &pair = m_candidatePairs[static_cast<std::size_t>(row)];
        setTableItem(m_pairTable, row, 0, QFileInfo(QString::fromStdString(pair.imagePathA)).fileName());
        setTableItem(m_pairTable, row, 1, QFileInfo(QString::fromStdString(pair.imagePathB)).fileName());
        setTableItem(m_pairTable, row, 2, QString::number(pair.bowScore, 'f', 4));
        setTableItem(m_pairTable, row, 3, QString::number(pair.sharedWordCount));
        setTableItem(m_pairTable, row, 4, QString::number(pair.geometricInliers));
        setTableItem(m_pairTable, row, 5,
                     pair.accepted ? QStringLiteral("保留") : QStringLiteral("剔除：%1").arg(
                         QString::fromStdString(pair.rejectReason)));
    }
    m_pairTable->resizeColumnsToContents();
}
