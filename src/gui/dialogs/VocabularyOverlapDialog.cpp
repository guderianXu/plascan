#include "compat/QtTorchMacroGuard.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#include "VocabularyOverlapDialog.h"
#include "ui_VocabularyOverlapDialog.h"

#include "ProjectIO.h"
#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include "OverlapAnalyzer.h"
#include "io/PathIO.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>

struct VocabularyOverlapDialog::RunResult
{
    bool ok = false;
    bool applyToMatching = false;
    int vocabularySize = 0;
    QString method;
    QString errorTitle;
    QString errorMessage;
    QString detail;
    QStringList generatedPairs;
    QJsonObject settings;
    std::vector<xjw::VocabularyOverlapPairResult> candidates;
};

namespace
{

struct VocabularyOverlapRunRequest
{
    QString method;
    QStringList images;
    QString suffix;
    QString featureDir;
    QString projectPath;
    QJsonObject projectMeta;
    xjw::VocabularyOverlapConfig config;
    QJsonObject settings;
    bool applyToMatching = false;
    xjw::ReferenceBody referenceBody = xjw::ReferenceBody::Earth;
    double referenceRadiusMeters = 6378137.0;
    double referenceElevationMeters = 0.0;
    bool autoReferenceElevation = true;
    double cameraNeighborFactor = 2.0;
    std::shared_ptr<std::atomic_bool> cancelFlag;
    std::function<bool(const QString &stage, int percent)> progress;
};

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

QString pairTokenForPath(const QString &imagePath)
{
    return QFileInfo(imagePath).completeBaseName();
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

QString referenceBodyId(xjw::ReferenceBody body)
{
    switch (body)
    {
    case xjw::ReferenceBody::Moon:
        return QStringLiteral("moon");
    case xjw::ReferenceBody::Mars:
        return QStringLiteral("mars");
    case xjw::ReferenceBody::Earth:
    default:
        return QStringLiteral("earth");
    }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

xjw::ReferenceBody referenceBodyFromId(const QString &id)
{
    const QString normalized = id.trimmed().toLower();
    if (normalized == QStringLiteral("moon"))
    {
        return xjw::ReferenceBody::Moon;
    }
    if (normalized == QStringLiteral("mars"))
    {
        return xjw::ReferenceBody::Mars;
    }
    return xjw::ReferenceBody::Earth;
}

bool requestProgress(const VocabularyOverlapRunRequest &request, const QString &stage, int percent)
{
    if (request.cancelFlag && request.cancelFlag->load(std::memory_order_relaxed))
    {
        return false;
    }
    if (request.progress)
    {
        return request.progress(stage, std::clamp(percent, 0, 100));
    }
    return true;
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
    const auto descriptor_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(float);
    std::memcpy(descriptors.ptr<float>(0), cpu.data_ptr<float>(), descriptor_bytes);
    return descriptors;
}

void setTableItem(QTableWidget *table, int row, int column, const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    table->setItem(row, column, item);
}

bool loadCameraInputsForRequest(const VocabularyOverlapRunRequest &request,
                                std::vector<xjw::OverlapImageInput> *inputs,
                                QString *errorMsg)
{
    if (!inputs)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("内部错误：相机输入指针为空。");
        }
        return false;
    }

    inputs->clear();
    if (request.images.size() < 2)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("至少选择两张影像。");
        }
        return false;
    }

    const QMap<QString, QJsonObject> metaByPath =
        xjw::gui::project::projectImageMetaByPath(request.projectMeta, true);

    for (const QString &imagePath : request.images)
    {
        const QString normalized = xjw::gui::project::normalizePath(imagePath);
        const QJsonObject imageMeta = metaByPath.value(normalized);
        if (imageMeta.isEmpty())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("项目元数据中找不到影像：%1").arg(imagePath);
            }
            return false;
        }

        xjw::Camera camera;
        if (!xjw::gui::project::imageCameraFromEntry(imageMeta, &camera))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("影像缺少可用相机模型：%1").arg(imagePath);
            }
            return false;
        }

        QImageReader reader(imagePath);
        const QSize size = reader.size();
        if (!size.isValid() || size.width() <= 0 || size.height() <= 0)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("无法读取影像尺寸：%1").arg(imagePath);
            }
            return false;
        }

        xjw::OverlapImageInput input;
        input.imagePath = xjw::common::io::toUtf8Path(imagePath);
        input.camera = camera;
        input.width = size.width();
        input.height = size.height();
        inputs->push_back(std::move(input));
    }

    return true;
}

bool loadFeaturesForRequest(const VocabularyOverlapRunRequest &request,
                            std::vector<xjw::VocabularyImageFeatures> *features,
                            QString *errorMsg)
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
    if (request.images.size() < 2)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("至少选择两张影像。");
        }
        return false;
    }

    for (const QString &imagePath : request.images)
    {
        QString featurePath = featurePathInDir(request.featureDir, imagePath, request.suffix);
        if (!QFile::exists(featurePath) && !request.projectPath.isEmpty())
        {
            featurePath = ProjectIO::featureFileForSuffix(request.projectPath, imagePath, request.suffix);
        }
        if (!QFile::exists(featurePath))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("缺少特征文件：%1\n期望后缀：%2").arg(imagePath, request.suffix);
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
        imageFeatures.imagePath = xjw::common::io::toUtf8Path(imagePath);
        imageFeatures.keypoints = output.keypoints;
        imageFeatures.descriptors = descriptors;
        features->push_back(std::move(imageFeatures));
    }

    return true;
}

bool containsImageIndex(const std::vector<int> &indices, int value)
{
    return std::find(indices.begin(), indices.end(), value) != indices.end();
}

QStringList acceptedPairTokens(const std::vector<xjw::VocabularyOverlapPairResult> &pairs)
{
    QStringList generatedPairs;
    for (const xjw::VocabularyOverlapPairResult &pair : pairs)
    {
        if (!pair.accepted)
        {
            continue;
        }
        const QString imageA = xjw::common::io::fromUtf8Path(pair.imagePathA);
        const QString imageB = xjw::common::io::fromUtf8Path(pair.imagePathB);
        generatedPairs.append(pairTokenForPath(imageA) + QStringLiteral("__") + pairTokenForPath(imageB));
    }
    return generatedPairs;
}

bool buildCameraOverlapCandidates(const VocabularyOverlapRunRequest &request,
                                  std::vector<xjw::VocabularyOverlapPairResult> *candidates,
                                  QString *detail,
                                  QString *errorMsg)
{
    if (!requestProgress(request, QStringLiteral("读取相机模型"), 10))
    {
        if (errorMsg) *errorMsg = QStringLiteral("用户取消获取重叠对");
        return false;
    }

    std::vector<xjw::OverlapImageInput> inputs;
    if (!loadCameraInputsForRequest(request, &inputs, errorMsg))
    {
        return false;
    }

    if (!requestProgress(request, QStringLiteral("计算相机覆盖"), 35))
    {
        if (errorMsg) *errorMsg = QStringLiteral("用户取消获取重叠对");
        return false;
    }

    xjw::OverlapAnalysisResult overlapResult;
    std::string coreError;
    xjw::OverlapAnalysisOptions options;
    options.groundModel = xjw::OverlapGroundModel::ReferenceSphere;
    options.neighborFactor = request.cameraNeighborFactor;
    options.referenceSphere.body = request.referenceBody;
    options.referenceSphere.radiusMeters = request.referenceRadiusMeters;
    options.referenceSphere.elevationMeters = request.referenceElevationMeters;
    options.referenceSphere.autoLocalTangentHeight = request.autoReferenceElevation;
    options.referenceSphere.centerMode = xjw::ReferenceSphereCenterMode::Auto;
    if (!xjw::OverlapAnalyzer::analyze(inputs, options, &overlapResult, &coreError))
    {
        if (errorMsg)
        {
            *errorMsg = QString::fromStdString(coreError);
        }
        return false;
    }

    if (!requestProgress(request, QStringLiteral("筛选相机重叠对"), 75))
    {
        if (errorMsg) *errorMsg = QStringLiteral("用户取消获取重叠对");
        return false;
    }

    std::vector<xjw::VocabularyOverlapPairResult> output;
    output.reserve(overlapResult.pairs.size());
    const int imageCount = static_cast<int>(inputs.size());
    std::vector<std::vector<std::pair<double, int>>> scoredByImage(static_cast<std::size_t>(imageCount));

    for (const xjw::OverlapPairResult &overlapPair : overlapResult.pairs)
    {
        xjw::VocabularyOverlapPairResult pair;
        pair.indexA = overlapPair.indexA;
        pair.indexB = overlapPair.indexB;
        pair.imagePathA = inputs[static_cast<std::size_t>(overlapPair.indexA)].imagePath;
        pair.imagePathB = inputs[static_cast<std::size_t>(overlapPair.indexB)].imagePath;
        pair.bowScore = overlapPair.overlapScore;
        pair.sharedWordCount = 0;
        pair.geometricInliers = 0;
        pair.accepted = overlapPair.overlapScore >= request.config.minSimilarity;
        if (!pair.accepted)
        {
            pair.rejectReason = "相机重叠评分低于阈值";
        }
        else
        {
            scoredByImage[static_cast<std::size_t>(pair.indexA)].emplace_back(pair.bowScore, pair.indexB);
            scoredByImage[static_cast<std::size_t>(pair.indexB)].emplace_back(pair.bowScore, pair.indexA);
        }
        output.push_back(std::move(pair));
    }

    const int topK = std::max(1, request.config.topK);
    std::vector<std::vector<int>> topIndices(static_cast<std::size_t>(imageCount));
    for (int imageIndex = 0; imageIndex < imageCount; ++imageIndex)
    {
        auto &scored = scoredByImage[static_cast<std::size_t>(imageIndex)];
        std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first == rhs.first)
            {
                return lhs.second < rhs.second;
            }
            return lhs.first > rhs.first;
        });
        const int keep = std::min(topK, static_cast<int>(scored.size()));
        topIndices[static_cast<std::size_t>(imageIndex)].reserve(static_cast<std::size_t>(keep));
        for (int i = 0; i < keep; ++i)
        {
            topIndices[static_cast<std::size_t>(imageIndex)].push_back(scored[static_cast<std::size_t>(i)].second);
        }
    }

    for (xjw::VocabularyOverlapPairResult &pair : output)
    {
        if (!pair.accepted)
        {
            continue;
        }
        const bool aHasB = containsImageIndex(topIndices[static_cast<std::size_t>(pair.indexA)], pair.indexB);
        const bool bHasA = containsImageIndex(topIndices[static_cast<std::size_t>(pair.indexB)], pair.indexA);
        if (request.config.mutualTopK ? !(aHasB && bHasA) : !(aHasB || bHasA))
        {
            pair.accepted = false;
            pair.rejectReason = request.config.mutualTopK ? "非互选 Top-K" : "不在 Top-K";
        }
    }

    std::sort(output.begin(), output.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.bowScore == rhs.bowScore)
        {
            return std::tie(lhs.indexA, lhs.indexB) < std::tie(rhs.indexA, rhs.indexB);
        }
        return lhs.bowScore > rhs.bowScore;
    });

    if (candidates)
    {
        *candidates = std::move(output);
    }
    if (detail)
    {
        const int acceptedCount = candidates
            ? static_cast<int>(std::count_if(candidates->begin(), candidates->end(), [](const auto &pair) {
                  return pair.accepted;
              }))
            : 0;
        const QString detail_template =
            QStringLiteral("method=camera images=%1 candidates=%2 accepted=%3 neighbor=%4 reference_body=%5 ")
            + QStringLiteral("radius_m=%6 auto_elevation=%7 elevation_m=%8");
        *detail = detail_template.arg(imageCount)
                      .arg(overlapResult.pairs.size())
                      .arg(acceptedCount)
                      .arg(request.cameraNeighborFactor)
                      .arg(referenceBodyId(request.referenceBody))
                      .arg(request.referenceRadiusMeters)
                      .arg(request.autoReferenceElevation)
                      .arg(request.referenceElevationMeters);
    }
    return true;
}

bool writeOverlapOutputs(const QJsonObject &settings,
                         const QStringList &pairs,
                         const std::vector<xjw::VocabularyOverlapPairResult> &candidatePairs,
                         QString *errorMsg)
{
    const QString jsonPath = settings.value(QStringLiteral("output_json")).toString().trimmed();
    const QString lisPath = settings.value(QStringLiteral("output_lis")).toString().trimmed();

    if (!jsonPath.isEmpty())
    {
        QDir().mkpath(QFileInfo(jsonPath).absolutePath());
        QJsonObject root = settings;
        QJsonArray candidates;
        for (const xjw::VocabularyOverlapPairResult &pair : candidatePairs)
        {
            QJsonObject object;
            const QString imageA = QString::fromStdString(pair.imagePathA);
            const QString imageB = QString::fromStdString(pair.imagePathB);
            object.insert(QStringLiteral("image_a"), imageA);
            object.insert(QStringLiteral("image_b"), imageB);
            object.insert(QStringLiteral("pair_token"),
                          pairTokenForPath(imageA) + QStringLiteral("__") + pairTokenForPath(imageB));
            object.insert(QStringLiteral("bow_score"), pair.bowScore);
            object.insert(QStringLiteral("overlap_score"), pair.bowScore);
            object.insert(QStringLiteral("shared_word_count"), pair.sharedWordCount);
            object.insert(QStringLiteral("geometric_inliers"), pair.geometricInliers);
            object.insert(QStringLiteral("accepted"), pair.accepted);
            object.insert(QStringLiteral("reject_reason"), QString::fromStdString(pair.rejectReason));
            QJsonArray sourceTypes;
            for (const std::string &sourceType : pair.sourceTypes)
            {
                sourceTypes.append(QString::fromStdString(sourceType));
            }
            object.insert(QStringLiteral("source_types"), sourceTypes);
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
        if (!candidatePairs.empty())
        {
            for (const xjw::VocabularyOverlapPairResult &pair : candidatePairs)
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

VocabularyOverlapDialog::RunResult runVocabularyOverlapRequest(const VocabularyOverlapRunRequest &request)
{
    VocabularyOverlapDialog::RunResult runResult;
    runResult.applyToMatching = request.applyToMatching;
    runResult.method = request.method;

    try
    {
        if (!requestProgress(request, QStringLiteral("准备获取重叠对"), 1))
        {
            runResult.errorTitle = QStringLiteral("获取重叠对已取消");
            runResult.errorMessage = QStringLiteral("用户取消获取重叠对");
            return runResult;
        }

        QString error;
        std::vector<xjw::VocabularyOverlapPairResult> candidates;
        int vocabularySize = 0;
        QString detail;

        if (request.method == QStringLiteral("camera"))
        {
            if (!buildCameraOverlapCandidates(request, &candidates, &detail, &error))
            {
                runResult.errorTitle = QStringLiteral("获取重叠对失败");
                runResult.errorMessage = error;
                return runResult;
            }
        }
        else
        {
            if (!requestProgress(request, QStringLiteral("读取特征文件"), 5))
            {
                runResult.errorTitle = QStringLiteral("获取重叠对已取消");
                runResult.errorMessage = QStringLiteral("用户取消获取重叠对");
                return runResult;
            }

            std::vector<xjw::VocabularyImageFeatures> features;
            if (!loadFeaturesForRequest(request, &features, &error))
            {
                runResult.errorTitle = QStringLiteral("获取重叠对失败");
                runResult.errorMessage = error;
                return runResult;
            }

            xjw::VocabularyOverlapResult result;
            std::string coreError;
            if (!xjw::VocabularyOverlapRetriever::retrieve(features, request.config, &result, &coreError))
            {
                runResult.errorTitle = coreError == "用户取消获取重叠对"
                    ? QStringLiteral("获取重叠对已取消")
                    : QStringLiteral("获取重叠对失败");
                runResult.errorMessage = QString::fromStdString(coreError);
                return runResult;
            }
            vocabularySize = result.vocabularySize;
            detail = QString::fromStdString(result.detail);
            candidates = std::move(result.candidates);
        }

        QStringList generatedPairs = acceptedPairTokens(candidates);
        QJsonObject settings = request.settings;
        settings.insert(QStringLiteral("overlap_method"), request.method);
        settings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(generatedPairs));
        settings.insert(QStringLiteral("vocabulary_size"), vocabularySize);
        settings.insert(QStringLiteral("detail"), detail);

        if (!writeOverlapOutputs(settings, generatedPairs, candidates, &error))
        {
            runResult.errorTitle = QStringLiteral("写出重叠对失败");
            runResult.errorMessage = error;
            return runResult;
        }

        runResult.ok = true;
        runResult.vocabularySize = vocabularySize;
        runResult.detail = detail;
        runResult.generatedPairs = generatedPairs;
        runResult.settings = settings;
        runResult.candidates = std::move(candidates);
        requestProgress(request, QStringLiteral("完成"), 100);
    }
    catch (const std::exception &ex)
    {
        runResult.errorTitle = QStringLiteral("获取重叠对失败");
        runResult.errorMessage = QString::fromUtf8(ex.what());
    }
    catch (...)
    {
        runResult.errorTitle = QStringLiteral("获取重叠对失败");
        runResult.errorMessage = QStringLiteral("未知异常");
    }

    return runResult;
}

} // namespace

VocabularyOverlapDialog::VocabularyOverlapDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
{
    setupUi();
    setupConnections();
    onResetDefaults();
    if (_projectManager)
    {
        setProjectImages(_projectManager->getAllImages());
    }
}

VocabularyOverlapDialog::~VocabularyOverlapDialog() = default;

void VocabularyOverlapDialog::setupUi()
{
    Ui::VocabularyOverlapDialog ui;
    ui.setupUi(this);

    _imageList = ui.m_imageList;
    _selectAllBtn = ui.m_selectAllBtn;
    _clearSelectionBtn = ui.m_clearSelectionBtn;
    _overlapMethodCombo = ui.m_overlapMethodCombo;
    _referenceBodyCombo = ui.m_referenceBodyCombo;
    _autoReferenceElevationCheck = ui.m_autoReferenceElevationCheck;
    _referenceElevationSpin = ui.m_referenceElevationSpin;
    _cameraNeighborFactorSpin = ui.m_cameraNeighborFactorSpin;
    _featureAlgorithmCombo = ui.m_featureAlgorithmCombo;
    _featureDirEdit = ui.m_featureDirEdit;
    _autoDetectFeatureDirBtn = ui.m_autoDetectFeatureDirBtn;
    _browseFeatureDirBtn = ui.m_browseFeatureDirBtn;
    _branchFactorSpin = ui.m_branchFactorSpin;
    _treeDepthSpin = ui.m_treeDepthSpin;
    _samplePerImageSpin = ui.m_samplePerImageSpin;
    _maxTrainingDescriptorsSpin = ui.m_maxTrainingDescriptorsSpin;
    _topKSpin = ui.m_topKSpin;
    _minSimilaritySpin = ui.m_minSimilaritySpin;
    _useTfidfCheck = ui.m_useTfidfCheck;
    _mutualTopKCheck = ui.m_mutualTopKCheck;
    _enableGeometryCheck = ui.m_enableGeometryCheck;
    _minInliersSpin = ui.m_minInliersSpin;
    _ransacThresholdSpin = ui.m_ransacThresholdSpin;
    _geometryModelCombo = ui.m_geometryModelCombo;
    _overlapThreadsSpin = ui.m_overlapThreadsSpin;
    _useFlannAssignmentCheck = ui.m_useFlannAssignmentCheck;
    _useInvertedIndexCheck = ui.m_useInvertedIndexCheck;
    _useCudaOverlapCheck = ui.m_useCudaOverlapCheck;
    _geometryMaxDescriptorsSpin = ui.m_geometryMaxDescriptorsSpin;
    _geometryMaxPairsSpin = ui.m_geometryMaxPairsSpin;
    _outputJsonEdit = ui.m_outputJsonEdit;
    _outputLisEdit = ui.m_outputLisEdit;
    _applyToMatchingCheck = ui.m_applyToMatchingCheck;
    _pairTable = ui.m_pairTable;
    _summaryLabel = ui.m_summaryLabel;
    _resetBtn = ui.m_resetBtn;
    _exportLisBtn = ui.m_exportLisBtn;
    _applyToMatchingBtn = ui.m_applyToMatchingBtn;
    _runBtn = ui.m_runBtn;
    _closeBtn = ui.m_closeBtn;

    ui.topSplit->setStretchFactor(0, 3);
    ui.topSplit->setStretchFactor(1, 2);

    _overlapMethodCombo->clear();
    _overlapMethodCombo->addItem(QStringLiteral("特征词汇（已有特征）"), QStringLiteral("vocabulary"));
    _overlapMethodCombo->addItem(QStringLiteral("相机模型（已有相机）"), QStringLiteral("camera"));

    _referenceBodyCombo->clear();
    _referenceBodyCombo->addItem(QStringLiteral("地球 (WGS84 6378137 m)"), QStringLiteral("earth"));
    _referenceBodyCombo->addItem(QStringLiteral("月球 (1737400 m)"), QStringLiteral("moon"));
    _referenceBodyCombo->addItem(QStringLiteral("火星 (3389500 m)"), QStringLiteral("mars"));

    _featureAlgorithmCombo->clear();
    _featureAlgorithmCombo->addItem(QStringLiteral("SIFT (.sift)"), QStringLiteral(".sift"));
    _featureAlgorithmCombo->addItem(QStringLiteral("DISK (.dsk)"), QStringLiteral(".dsk"));
    _featureAlgorithmCombo->addItem(QStringLiteral("ALIKED (.alk)"), QStringLiteral(".alk"));
    _featureAlgorithmCombo->addItem(QStringLiteral("SuperPoint (.sp)"), QStringLiteral(".sp"));
    _featureAlgorithmCombo->addItem(QStringLiteral("ORB (.orb)"), QStringLiteral(".orb"));

    _geometryModelCombo->clear();
    _geometryModelCombo->addItem(QStringLiteral("Fundamental Matrix"), QStringLiteral("fundamental"));

    _pairTable->setColumnCount(6);
    _pairTable->setHorizontalHeaderLabels(QStringList()
                                           << QStringLiteral("影像 1")
                                           << QStringLiteral("影像 2")
                                           << QStringLiteral("相似度")
                                           << QStringLiteral("共享词")
                                           << QStringLiteral("几何内点")
                                           << QStringLiteral("状态"));
    _pairTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _pairTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _pairTable->setMinimumHeight(180);
    _pairTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

void VocabularyOverlapDialog::setupConnections()
{
    connect(_selectAllBtn, &QPushButton::clicked, this, [this]()
    {
        for (int i = 0; i < _imageList->count(); ++i)
        {
            _imageList->item(i)->setCheckState(Qt::Checked);
        }
        emitSettingsNow();
    });

    connect(_clearSelectionBtn, &QPushButton::clicked, this, [this]()
    {
        for (int i = 0; i < _imageList->count(); ++i)
        {
            _imageList->item(i)->setCheckState(Qt::Unchecked);
        }
        emitSettingsNow();
    });

    connect(_featureAlgorithmCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]()
    {
        refreshFeatureStatus();
        emitSettingsNow();
    });
    connect(_overlapMethodCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]()
    {
        updateMethodUi();
        emitSettingsNow();
    });
    connect(_featureDirEdit, &QLineEdit::editingFinished, this, [this]()
    {
        refreshFeatureStatus();
        emitSettingsNow();
    });
    connect(_browseFeatureDirBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onBrowseFeatureDir);
    connect(_autoDetectFeatureDirBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onAutoDetectFeatureDir);
    connect(_runBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onRun);
    connect(_exportLisBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onExportLis);
    connect(_applyToMatchingBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onApplyToMatching);
    connect(_resetBtn, &QPushButton::clicked, this, &VocabularyOverlapDialog::onResetDefaults);
    connect(_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    const auto emitNow = [this]() { emitSettingsNow(); };
    connect(_branchFactorSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_treeDepthSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_samplePerImageSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_maxTrainingDescriptorsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_topKSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_minSimilaritySpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(_referenceBodyCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, emitNow);
    connect(_autoReferenceElevationCheck, &QCheckBox::toggled, this, [this](bool checked)
    {
        _referenceElevationSpin->setEnabled(!checked &&
                                             _overlapMethodCombo->currentData().toString() == QStringLiteral("camera"));
        emitSettingsNow();
    });
    connect(_referenceElevationSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(_cameraNeighborFactorSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(_useTfidfCheck, &QCheckBox::toggled, this, emitNow);
    connect(_mutualTopKCheck, &QCheckBox::toggled, this, emitNow);
    connect(_enableGeometryCheck, &QCheckBox::toggled, this, emitNow);
    connect(_minInliersSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_ransacThresholdSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(_overlapThreadsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_useFlannAssignmentCheck, &QCheckBox::toggled, this, emitNow);
    connect(_useInvertedIndexCheck, &QCheckBox::toggled, this, emitNow);
    connect(_useCudaOverlapCheck, &QCheckBox::toggled, this, emitNow);
    connect(_geometryMaxDescriptorsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_geometryMaxPairsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(_outputJsonEdit, &QLineEdit::editingFinished, this, emitNow);
    connect(_outputLisEdit, &QLineEdit::editingFinished, this, emitNow);
    connect(_applyToMatchingCheck, &QCheckBox::toggled, this, emitNow);
}

void VocabularyOverlapDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty())
    {
        return;
    }

    QSignalBlocker blockAlgorithm(_featureAlgorithmCombo);
    QSignalBlocker blockMethod(_overlapMethodCombo);
    QSignalBlocker blockFeatureDir(_featureDirEdit);
    QSignalBlocker blockReferenceBody(_referenceBodyCombo);
    QSignalBlocker blockReferenceAuto(_autoReferenceElevationCheck);
    QSignalBlocker blockReferenceElevation(_referenceElevationSpin);

    const QString method = settings.value(QStringLiteral("overlap_method")).toString(QStringLiteral("vocabulary"));
    const int methodIndex = _overlapMethodCombo->findData(method);
    if (methodIndex >= 0)
    {
        _overlapMethodCombo->setCurrentIndex(methodIndex);
    }

    const QString suffix = settings.value(QStringLiteral("feature_suffix")).toString(defaultFeatureSuffix());
    const int algorithmIndex = _featureAlgorithmCombo->findData(suffix);
    if (algorithmIndex >= 0)
    {
        _featureAlgorithmCombo->setCurrentIndex(algorithmIndex);
    }

    _featureDirEdit->setText(settings.value(QStringLiteral("feature_dir")).toString(_featureDirEdit->text()));
    _branchFactorSpin->setValue(
        settings.value(QStringLiteral("branch_factor")).toInt(_branchFactorSpin->value()));
    _treeDepthSpin->setValue(settings.value(QStringLiteral("tree_depth")).toInt(_treeDepthSpin->value()));
    _samplePerImageSpin->setValue(
        settings.value(QStringLiteral("sample_per_image")).toInt(_samplePerImageSpin->value()));
    _maxTrainingDescriptorsSpin->setValue(
        settings.value(QStringLiteral("max_training_descriptors")).toInt(_maxTrainingDescriptorsSpin->value()));
    _topKSpin->setValue(settings.value(QStringLiteral("top_k")).toInt(_topKSpin->value()));
    _minSimilaritySpin->setValue(
        settings.value(QStringLiteral("min_similarity")).toDouble(_minSimilaritySpin->value()));
    const QString referenceBody = settings.value(QStringLiteral("reference_body")).toString(QStringLiteral("earth"));
    const int referenceBodyIndex = _referenceBodyCombo->findData(referenceBody);
    if (referenceBodyIndex >= 0)
    {
        _referenceBodyCombo->setCurrentIndex(referenceBodyIndex);
    }
    const double referenceElevation = settings.contains(QStringLiteral("reference_elevation_m"))
        ? settings.value(QStringLiteral("reference_elevation_m")).toDouble(_referenceElevationSpin->value())
        : settings.value(QStringLiteral("fixed_ground_z")).toDouble(_referenceElevationSpin->value());
    _referenceElevationSpin->setValue(referenceElevation);
    _autoReferenceElevationCheck->setChecked(
        settings.value(QStringLiteral("auto_reference_elevation")).toBool(_autoReferenceElevationCheck->isChecked()));
    _cameraNeighborFactorSpin->setValue(
        settings.value(QStringLiteral("camera_neighbor_factor")).toDouble(_cameraNeighborFactorSpin->value()));
    _useTfidfCheck->setChecked(settings.value(QStringLiteral("use_tfidf")).toBool(_useTfidfCheck->isChecked()));
    _mutualTopKCheck->setChecked(settings.value(QStringLiteral("mutual_top_k")).toBool(_mutualTopKCheck->isChecked()));
    _enableGeometryCheck->setChecked(
        settings.value(QStringLiteral("geometry_check")).toBool(_enableGeometryCheck->isChecked()));
    _minInliersSpin->setValue(settings.value(QStringLiteral("min_inliers")).toInt(_minInliersSpin->value()));
    _ransacThresholdSpin->setValue(
        settings.value(QStringLiteral("ransac_threshold")).toDouble(_ransacThresholdSpin->value()));
    _overlapThreadsSpin->setValue(
        settings.value(QStringLiteral("overlap_threads")).toInt(_overlapThreadsSpin->value()));
    _useFlannAssignmentCheck->setChecked(
        settings.value(QStringLiteral("use_flann_assignment")).toBool(_useFlannAssignmentCheck->isChecked()));
    _useInvertedIndexCheck->setChecked(
        settings.value(QStringLiteral("use_inverted_index")).toBool(_useInvertedIndexCheck->isChecked()));
    _useCudaOverlapCheck->setChecked(
        settings.value(QStringLiteral("use_cuda_overlap")).toBool(_useCudaOverlapCheck->isChecked()));
    _geometryMaxDescriptorsSpin->setValue(
        settings.value(QStringLiteral("geometry_max_descriptors")).toInt(_geometryMaxDescriptorsSpin->value()));
    _geometryMaxPairsSpin->setValue(
        settings.value(QStringLiteral("geometry_max_pairs")).toInt(_geometryMaxPairsSpin->value()));
    _outputJsonEdit->setText(settings.value(QStringLiteral("output_json")).toString(_outputJsonEdit->text()));
    _outputLisEdit->setText(settings.value(QStringLiteral("output_lis")).toString(_outputLisEdit->text()));
    _applyToMatchingCheck->setChecked(
        settings.value(QStringLiteral("apply_to_matching")).toBool(_applyToMatchingCheck->isChecked()));

    _generatedPairs.clear();
    const QJsonArray generatedPairs = settings.value(QStringLiteral("generated_pairs")).toArray();
    for (const QJsonValue &value : generatedPairs)
    {
        const QString pair = value.toString().trimmed();
        if (!pair.isEmpty())
        {
            _generatedPairs.append(pair);
        }
    }

    refreshFeatureStatus();
    updateMethodUi();
}

void VocabularyOverlapDialog::setProjectImages(const QStringList &paths)
{
    _projectImages = paths;
    _imageList->clear();

    for (const QString &path : paths)
    {
        auto *item = new QListWidgetItem(path, _imageList);
        item->setData(Qt::UserRole, path);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
    }

    refreshFeatureStatus();
}

void VocabularyOverlapDialog::cancelRun()
{
    if (!_runWatcher || !_runWatcher->isRunning())
    {
        return;
    }
    if (_cancelFlag)
    {
        _cancelFlag->store(true, std::memory_order_relaxed);
    }
    _runBtn->setEnabled(false);
    _runBtn->setText(QStringLiteral("正在取消..."));
    _summaryLabel->setText(QStringLiteral("正在取消获取重叠对..."));
}

void VocabularyOverlapDialog::onBrowseFeatureDir()
{
    const QString dir = QFileDialog::getExistingDirectory(this,
                                                          QStringLiteral("选择特征目录"),
                                                          _featureDirEdit->text());
    if (dir.isEmpty())
    {
        return;
    }
    _featureDirEdit->setText(QDir::cleanPath(dir));
    refreshFeatureStatus();
    emitSettingsNow();
}

void VocabularyOverlapDialog::onAutoDetectFeatureDir()
{
    if (!_projectManager)
    {
        return;
    }
    const QString projectPath = _projectManager->currentProjectPath();
    _featureDirEdit->setText(ProjectIO::ipfindOutputDir(projectPath));
    refreshFeatureStatus();
    emitSettingsNow();
}

void VocabularyOverlapDialog::onRun()
{
    if (_runWatcher && _runWatcher->isRunning())
    {
        cancelRun();
        emit overlapCancelRequested();
        return;
    }

    VocabularyOverlapRunRequest request;
    request.method = _overlapMethodCombo->currentData().toString();
    if (request.method.isEmpty())
    {
        request.method = QStringLiteral("vocabulary");
    }
    request.images = checkedImages();
    request.suffix = selectedFeatureSuffix();
    request.featureDir = _featureDirEdit->text().trimmed();
    request.projectPath = _projectManager ? _projectManager->currentProjectPath() : QString();
    request.projectMeta = _projectManager ? _projectManager->currentMeta() : QJsonObject();
    request.settings = collectSettings();
    request.applyToMatching = _applyToMatchingCheck->isChecked();
    request.referenceBody = referenceBodyFromId(_referenceBodyCombo->currentData().toString());
    request.referenceRadiusMeters = xjw::referenceBodyRadiusMeters(request.referenceBody);
    request.referenceElevationMeters = _referenceElevationSpin->value();
    request.autoReferenceElevation = _autoReferenceElevationCheck->isChecked();
    request.cameraNeighborFactor = _cameraNeighborFactorSpin->value();
    request.cancelFlag = std::make_shared<std::atomic_bool>(false);
    request.config.branchFactor = _branchFactorSpin->value();
    request.config.treeDepth = _treeDepthSpin->value();
    request.config.samplePerImage = _samplePerImageSpin->value();
    request.config.maxTrainingDescriptors = _maxTrainingDescriptorsSpin->value();
    request.config.topK = _topKSpin->value();
    request.config.minSimilarity = _minSimilaritySpin->value();
    request.config.useTfidf = _useTfidfCheck->isChecked();
    request.config.mutualTopK = _mutualTopKCheck->isChecked();
    request.config.geometryCheck = _enableGeometryCheck->isChecked();
    request.config.minInliers = _minInliersSpin->value();
    request.config.ransacThreshold = _ransacThresholdSpin->value();
    request.config.numThreads = _overlapThreadsSpin->value();
    request.config.useFlannAssignment = _useFlannAssignmentCheck->isChecked();
    request.config.useInvertedIndex = _useInvertedIndexCheck->isChecked();
    request.config.useCuda = _useCudaOverlapCheck->isChecked();
    request.config.geometryMaxDescriptors = _geometryMaxDescriptorsSpin->value();
    request.config.geometryMaxCandidatePairs = _geometryMaxPairsSpin->value();

    QPointer<VocabularyOverlapDialog> self(this);
    const std::shared_ptr<std::atomic_bool> cancelFlag = request.cancelFlag;
    request.progress = [self, cancelFlag](const QString &stage, int percent) -> bool
    {
        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
        {
            return false;
        }
        if (self)
        {
            QMetaObject::invokeMethod(self.data(), [self, stage, percent]()
            {
                if (self)
                {
                    self->handleProgress(stage, percent);
                }
            }, Qt::QueuedConnection);
        }
        return !(cancelFlag && cancelFlag->load(std::memory_order_relaxed));
    };
    request.config.progressCallback = [progress = request.progress](const std::string &stage, int percent) -> bool
    {
        return progress(QString::fromStdString(stage), percent);
    };

    _cancelFlag = request.cancelFlag;
    setUiBusy(true, QStringLiteral("正在获取重叠对..."));
    emit overlapProgressChanged(QStringLiteral("正在获取重叠对"), 0);

    auto *watcher = new QFutureWatcher<RunResult>(this);
    _runWatcher = watcher;
    connect(watcher, &QFutureWatcher<RunResult>::finished, watcher, [self, watcher]()
    {
        if (!self)
        {
            return;
        }
        self->handleRunFinished(watcher);
    });
    watcher->setFuture(QtConcurrent::run([request]()
    {
        return runVocabularyOverlapRequest(request);
    }));
}

void VocabularyOverlapDialog::onExportLis()
{
    if (_generatedPairs.isEmpty())
    {
        QMessageBox::information(this, QStringLiteral("导出 LIS"), QStringLiteral("当前没有可导出的重叠对。"));
        return;
    }

    QString error;
    const QJsonObject settings = _lastRunSettings.isEmpty() ? collectSettings() : _lastRunSettings;
    if (!writeOutputs(settings, _generatedPairs, &error))
    {
        QMessageBox::warning(this, QStringLiteral("导出 LIS 失败"), error);
    }
}

void VocabularyOverlapDialog::onApplyToMatching()
{
    if (_generatedPairs.isEmpty())
    {
        QMessageBox::information(this, QStringLiteral("应用到匹配"), QStringLiteral("请先运行获取重叠对。"));
        return;
    }

    QJsonObject settings = _lastRunSettings.isEmpty() ? collectSettings() : _lastRunSettings;
    settings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(_generatedPairs));
    emit generatedPairsReady(_generatedPairs, settings);
    emit settingsChanged(settings);
}

void VocabularyOverlapDialog::onResetDefaults()
{
    const QString projectPath = _projectManager ? _projectManager->currentProjectPath() : QString();
    const QString outputDir = defaultOverlapOutputDir(projectPath);

    const QString defaultSuffix = defaultFeatureSuffix();
    const int defaultFeatureIndex = _featureAlgorithmCombo->findData(defaultSuffix);
    const int vocabIndex = _overlapMethodCombo->findData(QStringLiteral("vocabulary"));
    const int earthIndex = _referenceBodyCombo->findData(QStringLiteral("earth"));
    _overlapMethodCombo->setCurrentIndex(vocabIndex >= 0 ? vocabIndex : 0);
    _referenceBodyCombo->setCurrentIndex(earthIndex >= 0 ? earthIndex : 0);
    _autoReferenceElevationCheck->setChecked(true);
    _referenceElevationSpin->setValue(0.0);
    _cameraNeighborFactorSpin->setValue(2.0);
    _featureAlgorithmCombo->setCurrentIndex(defaultFeatureIndex >= 0 ? defaultFeatureIndex : 0);
    _featureDirEdit->setText(projectPath.isEmpty() ? QString() : ProjectIO::ipfindOutputDir(projectPath));
    _branchFactorSpin->setValue(10);
    _treeDepthSpin->setValue(3);
    _samplePerImageSpin->setValue(500);
    _maxTrainingDescriptorsSpin->setValue(50000);
    _topKSpin->setValue(8);
    _minSimilaritySpin->setValue(0.05);
    _useTfidfCheck->setChecked(true);
    _mutualTopKCheck->setChecked(true);
    _enableGeometryCheck->setChecked(false);
    _minInliersSpin->setValue(30);
    _ransacThresholdSpin->setValue(3.0);
    _overlapThreadsSpin->setValue(0);
    _useFlannAssignmentCheck->setChecked(true);
    _useInvertedIndexCheck->setChecked(true);
    _useCudaOverlapCheck->setChecked(false);
    _geometryMaxDescriptorsSpin->setValue(2048);
    _geometryMaxPairsSpin->setValue(2000);
    _applyToMatchingCheck->setChecked(true);

    if (!outputDir.isEmpty())
    {
        _outputJsonEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.json")));
        _outputLisEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.lis")));
    }

    refreshFeatureStatus();
    updateMethodUi();
    emitSettingsNow();
}

void VocabularyOverlapDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void VocabularyOverlapDialog::refreshFeatureStatus()
{
    const QString suffix = selectedFeatureSuffix();
    const QString featureDir = _featureDirEdit->text().trimmed();
    int existing = 0;

    for (int i = 0; i < _imageList->count(); ++i)
    {
        QListWidgetItem *item = _imageList->item(i);
        const QString imagePath = item->data(Qt::UserRole).toString();
        QString featurePath = featurePathInDir(featureDir, imagePath, suffix);
        if (!QFile::exists(featurePath) && _projectManager)
        {
            featurePath = ProjectIO::featureFileForSuffix(_projectManager->currentProjectPath(), imagePath, suffix);
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

    _summaryLabel->setText(QStringLiteral("特征文件 %1/%2，当前算法 %3")
                                .arg(existing)
                                .arg(_imageList->count())
                                .arg(suffix));
}

QJsonObject VocabularyOverlapDialog::collectSettings() const
{
    QJsonObject settings;
    settings.insert(QStringLiteral("overlap_method"), _overlapMethodCombo->currentData().toString());
    settings.insert(QStringLiteral("feature_suffix"), selectedFeatureSuffix());
    settings.insert(QStringLiteral("feature_algorithm"), _featureAlgorithmCombo->currentText());
    settings.insert(QStringLiteral("feature_dir"), _featureDirEdit->text().trimmed());
    settings.insert(QStringLiteral("branch_factor"), _branchFactorSpin->value());
    settings.insert(QStringLiteral("tree_depth"), _treeDepthSpin->value());
    settings.insert(QStringLiteral("sample_per_image"), _samplePerImageSpin->value());
    settings.insert(QStringLiteral("max_training_descriptors"), _maxTrainingDescriptorsSpin->value());
    settings.insert(QStringLiteral("top_k"), _topKSpin->value());
    settings.insert(QStringLiteral("min_similarity"), _minSimilaritySpin->value());
    const xjw::ReferenceBody body = referenceBodyFromId(_referenceBodyCombo->currentData().toString());
    settings.insert(QStringLiteral("reference_body"), referenceBodyId(body));
    settings.insert(QStringLiteral("reference_radius_m"), xjw::referenceBodyRadiusMeters(body));
    settings.insert(QStringLiteral("reference_elevation_m"), _referenceElevationSpin->value());
    settings.insert(QStringLiteral("auto_reference_elevation"), _autoReferenceElevationCheck->isChecked());
    settings.insert(QStringLiteral("fixed_ground_z"), _referenceElevationSpin->value());
    settings.insert(QStringLiteral("camera_neighbor_factor"), _cameraNeighborFactorSpin->value());
    settings.insert(QStringLiteral("use_tfidf"), _useTfidfCheck->isChecked());
    settings.insert(QStringLiteral("mutual_top_k"), _mutualTopKCheck->isChecked());
    settings.insert(QStringLiteral("geometry_check"), _enableGeometryCheck->isChecked());
    settings.insert(QStringLiteral("geometry_model"), _geometryModelCombo->currentData().toString());
    settings.insert(QStringLiteral("min_inliers"), _minInliersSpin->value());
    settings.insert(QStringLiteral("ransac_threshold"), _ransacThresholdSpin->value());
    settings.insert(QStringLiteral("overlap_threads"), _overlapThreadsSpin->value());
    settings.insert(QStringLiteral("use_flann_assignment"), _useFlannAssignmentCheck->isChecked());
    settings.insert(QStringLiteral("use_inverted_index"), _useInvertedIndexCheck->isChecked());
    settings.insert(QStringLiteral("use_cuda_overlap"), _useCudaOverlapCheck->isChecked());
    settings.insert(QStringLiteral("geometry_max_descriptors"), _geometryMaxDescriptorsSpin->value());
    settings.insert(QStringLiteral("geometry_max_pairs"), _geometryMaxPairsSpin->value());
    settings.insert(QStringLiteral("output_json"), _outputJsonEdit->text().trimmed());
    settings.insert(QStringLiteral("output_lis"), _outputLisEdit->text().trimmed());
    settings.insert(QStringLiteral("apply_to_matching"), _applyToMatchingCheck->isChecked());
    settings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(_generatedPairs));
    return settings;
}

QString VocabularyOverlapDialog::defaultFeatureSuffix() const
{
    const QString projectPath = _projectManager ? _projectManager->currentProjectPath() : QString();
    const QJsonObject projectMeta = _projectManager ? _projectManager->currentMeta() : QJsonObject();
    const QString suffix = xjw::gui::project::inferPreferredFeatureSuffix(projectPath, projectMeta);
    return suffix.isEmpty() ? QStringLiteral(".sift") : suffix;
}

QString VocabularyOverlapDialog::selectedFeatureSuffix() const
{
    const QString suffix = _featureAlgorithmCombo->currentData().toString();
    return suffix.isEmpty() ? defaultFeatureSuffix() : suffix;
}

QStringList VocabularyOverlapDialog::checkedImages() const
{
    QStringList images;
    for (int i = 0; i < _imageList->count(); ++i)
    {
        const QListWidgetItem *item = _imageList->item(i);
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
    const QString featureDir = _featureDirEdit->text().trimmed();
    for (const QString &imagePath : images)
    {
        QString featurePath = featurePathInDir(featureDir, imagePath, suffix);
        if (!QFile::exists(featurePath) && _projectManager)
        {
            featurePath = ProjectIO::featureFileForSuffix(_projectManager->currentProjectPath(), imagePath, suffix);
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
        imageFeatures.imagePath = xjw::common::io::toUtf8Path(imagePath);
        imageFeatures.keypoints = output.keypoints;
        imageFeatures.descriptors = descriptors;
        features->push_back(std::move(imageFeatures));
    }

    return true;
}

void VocabularyOverlapDialog::setUiBusy(bool busy, const QString &message)
{
    _runBtn->setEnabled(true);
    _runBtn->setText(busy ? QStringLiteral("取消") : QStringLiteral("运行"));
    _selectAllBtn->setEnabled(!busy);
    _clearSelectionBtn->setEnabled(!busy);
    _overlapMethodCombo->setEnabled(!busy);
    _referenceBodyCombo->setEnabled(!busy);
    _autoReferenceElevationCheck->setEnabled(!busy);
    _referenceElevationSpin->setEnabled(!busy && !_autoReferenceElevationCheck->isChecked());
    _cameraNeighborFactorSpin->setEnabled(!busy);
    _featureAlgorithmCombo->setEnabled(!busy);
    _featureDirEdit->setEnabled(!busy);
    _autoDetectFeatureDirBtn->setEnabled(!busy);
    _browseFeatureDirBtn->setEnabled(!busy);
    _branchFactorSpin->setEnabled(!busy);
    _treeDepthSpin->setEnabled(!busy);
    _samplePerImageSpin->setEnabled(!busy);
    _maxTrainingDescriptorsSpin->setEnabled(!busy);
    _topKSpin->setEnabled(!busy);
    _minSimilaritySpin->setEnabled(!busy);
    _useTfidfCheck->setEnabled(!busy);
    _mutualTopKCheck->setEnabled(!busy);
    _enableGeometryCheck->setEnabled(!busy);
    _minInliersSpin->setEnabled(!busy);
    _ransacThresholdSpin->setEnabled(!busy);
    _geometryModelCombo->setEnabled(!busy);
    _overlapThreadsSpin->setEnabled(!busy);
    _useFlannAssignmentCheck->setEnabled(!busy);
    _useInvertedIndexCheck->setEnabled(!busy);
    _useCudaOverlapCheck->setEnabled(!busy);
    _geometryMaxDescriptorsSpin->setEnabled(!busy);
    _geometryMaxPairsSpin->setEnabled(!busy);
    _outputJsonEdit->setEnabled(!busy);
    _outputLisEdit->setEnabled(!busy);
    _applyToMatchingCheck->setEnabled(!busy);
    _resetBtn->setEnabled(!busy);
    _exportLisBtn->setEnabled(!busy && !_generatedPairs.isEmpty());
    _applyToMatchingBtn->setEnabled(!busy && !_generatedPairs.isEmpty());
    _closeBtn->setEnabled(!busy);

    if (!message.isEmpty())
    {
        _summaryLabel->setText(message);
    }
    if (!busy)
    {
        updateMethodUi(false);
    }
}

void VocabularyOverlapDialog::handleProgress(const QString &stage, int percent)
{
    if (!_runWatcher || !_runWatcher->isRunning())
    {
        return;
    }

    const QString text = percent > 0 && percent < 100
        ? QStringLiteral("%1 %2%").arg(stage).arg(percent)
        : stage;
    _summaryLabel->setText(text);
    emit overlapProgressChanged(stage, std::clamp(percent, 0, 100));
}

void VocabularyOverlapDialog::handleRunFinished(QFutureWatcher<RunResult> *watcher)
{
    if (!watcher)
    {
        return;
    }

    const bool isCurrentWatcher = (watcher == _runWatcher);
    const RunResult runResult = watcher->result();
    if (isCurrentWatcher)
    {
        _runWatcher = nullptr;
    }
    _cancelFlag.reset();
    watcher->deleteLater();

    if (!isCurrentWatcher)
    {
        return;
    }

    if (!runResult.ok)
    {
        setUiBusy(false);
        emit overlapFinished(false);
        QMessageBox::warning(this,
                             runResult.errorTitle.isEmpty()
                                 ? QStringLiteral("获取重叠对失败")
                                 : runResult.errorTitle,
                             runResult.errorMessage);
        refreshFeatureStatus();
        return;
    }

    _candidatePairs = runResult.candidates;
    _generatedPairs = runResult.generatedPairs;
    _lastRunSettings = runResult.settings;

    populatePairTable();
    _summaryLabel->setText(QStringLiteral("候选 %1，对外输出 %2，词汇数 %3")
                                .arg(_candidatePairs.size())
                                .arg(_generatedPairs.size())
                                .arg(runResult.vocabularySize));
    setUiBusy(false);
    emit overlapFinished(true);

    emit settingsChanged(_lastRunSettings);
    if (runResult.applyToMatching)
    {
        emit generatedPairsReady(_generatedPairs, _lastRunSettings);
    }
}

bool VocabularyOverlapDialog::writeOutputs(const QJsonObject &settings,
                                           const QStringList &pairs,
                                           QString *errorMsg) const
{
    return writeOverlapOutputs(settings, pairs, _candidatePairs, errorMsg);
}

void VocabularyOverlapDialog::populatePairTable()
{
    _pairTable->setRowCount(static_cast<int>(_candidatePairs.size()));
    for (int row = 0; row < static_cast<int>(_candidatePairs.size()); ++row)
    {
        const xjw::VocabularyOverlapPairResult &pair = _candidatePairs[static_cast<std::size_t>(row)];
        setTableItem(_pairTable, row, 0, QFileInfo(QString::fromStdString(pair.imagePathA)).fileName());
        setTableItem(_pairTable, row, 1, QFileInfo(QString::fromStdString(pair.imagePathB)).fileName());
        setTableItem(_pairTable, row, 2, QString::number(pair.bowScore, 'f', 4));
        setTableItem(_pairTable, row, 3, QString::number(pair.sharedWordCount));
        setTableItem(_pairTable, row, 4, QString::number(pair.geometricInliers));
        setTableItem(_pairTable, row, 5,
                     pair.accepted ? QStringLiteral("保留") : QStringLiteral("剔除：%1").arg(
                         QString::fromStdString(pair.rejectReason)));
    }
    _pairTable->resizeColumnsToContents();
    _pairTable->setVisible(true);
    _pairTable->viewport()->update();
}

void VocabularyOverlapDialog::updateMethodUi(bool refreshSummary)
{
    const bool cameraMode = _overlapMethodCombo &&
        _overlapMethodCombo->currentData().toString() == QStringLiteral("camera");

    if (auto *featureGroup = findChild<QGroupBox *>(QStringLiteral("featureGroup")))
    {
        featureGroup->setVisible(!cameraMode);
    }
    if (auto *vocabularyGroup = findChild<QGroupBox *>(QStringLiteral("vocabularyGroup")))
    {
        vocabularyGroup->setVisible(!cameraMode);
    }
    if (auto *geometryGroup = findChild<QGroupBox *>(QStringLiteral("geometryGroup")))
    {
        geometryGroup->setVisible(!cameraMode);
    }

    _referenceBodyCombo->setEnabled(cameraMode);
    _autoReferenceElevationCheck->setEnabled(cameraMode);
    _referenceElevationSpin->setEnabled(cameraMode && !_autoReferenceElevationCheck->isChecked());
    _cameraNeighborFactorSpin->setEnabled(cameraMode);
    _useTfidfCheck->setEnabled(!cameraMode);
    _useFlannAssignmentCheck->setEnabled(!cameraMode);
    _useInvertedIndexCheck->setEnabled(!cameraMode);
    _useCudaOverlapCheck->setEnabled(!cameraMode);
    _geometryMaxDescriptorsSpin->setEnabled(!cameraMode);
    _geometryMaxPairsSpin->setEnabled(!cameraMode);

    if (!refreshSummary)
    {
        return;
    }

    if (cameraMode)
    {
        _summaryLabel->setText(QStringLiteral("相机模型模式：使用项目中的相机参数、影像尺寸和基准球面计算地面重叠对"));
    }
    else
    {
        refreshFeatureStatus();
    }
}
