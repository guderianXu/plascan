#include "compat/QtTorchMacroGuard.h"

#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#include "VocabularyOverlapDialog.h"
#include "ui_VocabularyOverlapDialog.h"

#include "ProjectIO.h"
#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include "OverlapAnalyzer.h"

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
    std::memcpy(descriptors.ptr<float>(0), cpu.data_ptr<float>(), static_cast<std::size_t>(rows * cols) * sizeof(float));
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
        input.imagePath = imagePath.toStdString();
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
        imageFeatures.imagePath = imagePath.toStdString();
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
        const QString imageA = QString::fromStdString(pair.imagePathA);
        const QString imageB = QString::fromStdString(pair.imagePathB);
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
        *detail = QStringLiteral("method=camera images=%1 candidates=%2 accepted=%3 neighbor=%4 reference_body=%5 radius_m=%6 auto_elevation=%7 elevation_m=%8")
                      .arg(imageCount)
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
    m_overlapMethodCombo = ui.m_overlapMethodCombo;
    m_referenceBodyCombo = ui.m_referenceBodyCombo;
    m_autoReferenceElevationCheck = ui.m_autoReferenceElevationCheck;
    m_referenceElevationSpin = ui.m_referenceElevationSpin;
    m_cameraNeighborFactorSpin = ui.m_cameraNeighborFactorSpin;
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
    m_overlapThreadsSpin = ui.m_overlapThreadsSpin;
    m_useFlannAssignmentCheck = ui.m_useFlannAssignmentCheck;
    m_useInvertedIndexCheck = ui.m_useInvertedIndexCheck;
    m_useCudaOverlapCheck = ui.m_useCudaOverlapCheck;
    m_geometryMaxDescriptorsSpin = ui.m_geometryMaxDescriptorsSpin;
    m_geometryMaxPairsSpin = ui.m_geometryMaxPairsSpin;
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

    m_overlapMethodCombo->clear();
    m_overlapMethodCombo->addItem(QStringLiteral("特征词汇（已有特征）"), QStringLiteral("vocabulary"));
    m_overlapMethodCombo->addItem(QStringLiteral("相机模型（已有相机）"), QStringLiteral("camera"));

    m_referenceBodyCombo->clear();
    m_referenceBodyCombo->addItem(QStringLiteral("地球 (WGS84 6378137 m)"), QStringLiteral("earth"));
    m_referenceBodyCombo->addItem(QStringLiteral("月球 (1737400 m)"), QStringLiteral("moon"));
    m_referenceBodyCombo->addItem(QStringLiteral("火星 (3389500 m)"), QStringLiteral("mars"));

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
    m_pairTable->setMinimumHeight(180);
    m_pairTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
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
    connect(m_overlapMethodCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]()
    {
        updateMethodUi();
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
    connect(m_referenceBodyCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, emitNow);
    connect(m_autoReferenceElevationCheck, &QCheckBox::toggled, this, [this](bool checked)
    {
        m_referenceElevationSpin->setEnabled(!checked &&
                                             m_overlapMethodCombo->currentData().toString() == QStringLiteral("camera"));
        emitSettingsNow();
    });
    connect(m_referenceElevationSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(m_cameraNeighborFactorSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(m_useTfidfCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_mutualTopKCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_enableGeometryCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_minInliersSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_ransacThresholdSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, emitNow);
    connect(m_overlapThreadsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_useFlannAssignmentCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_useInvertedIndexCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_useCudaOverlapCheck, &QCheckBox::toggled, this, emitNow);
    connect(m_geometryMaxDescriptorsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
    connect(m_geometryMaxPairsSpin, qOverload<int>(&QSpinBox::valueChanged), this, emitNow);
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
    QSignalBlocker blockMethod(m_overlapMethodCombo);
    QSignalBlocker blockFeatureDir(m_featureDirEdit);
    QSignalBlocker blockReferenceBody(m_referenceBodyCombo);
    QSignalBlocker blockReferenceAuto(m_autoReferenceElevationCheck);
    QSignalBlocker blockReferenceElevation(m_referenceElevationSpin);

    const QString method = settings.value(QStringLiteral("overlap_method")).toString(QStringLiteral("vocabulary"));
    const int methodIndex = m_overlapMethodCombo->findData(method);
    if (methodIndex >= 0)
    {
        m_overlapMethodCombo->setCurrentIndex(methodIndex);
    }

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
    const QString referenceBody = settings.value(QStringLiteral("reference_body")).toString(QStringLiteral("earth"));
    const int referenceBodyIndex = m_referenceBodyCombo->findData(referenceBody);
    if (referenceBodyIndex >= 0)
    {
        m_referenceBodyCombo->setCurrentIndex(referenceBodyIndex);
    }
    const double referenceElevation = settings.contains(QStringLiteral("reference_elevation_m"))
        ? settings.value(QStringLiteral("reference_elevation_m")).toDouble(m_referenceElevationSpin->value())
        : settings.value(QStringLiteral("fixed_ground_z")).toDouble(m_referenceElevationSpin->value());
    m_referenceElevationSpin->setValue(referenceElevation);
    m_autoReferenceElevationCheck->setChecked(
        settings.value(QStringLiteral("auto_reference_elevation")).toBool(m_autoReferenceElevationCheck->isChecked()));
    m_cameraNeighborFactorSpin->setValue(
        settings.value(QStringLiteral("camera_neighbor_factor")).toDouble(m_cameraNeighborFactorSpin->value()));
    m_useTfidfCheck->setChecked(settings.value(QStringLiteral("use_tfidf")).toBool(m_useTfidfCheck->isChecked()));
    m_mutualTopKCheck->setChecked(settings.value(QStringLiteral("mutual_top_k")).toBool(m_mutualTopKCheck->isChecked()));
    m_enableGeometryCheck->setChecked(
        settings.value(QStringLiteral("geometry_check")).toBool(m_enableGeometryCheck->isChecked()));
    m_minInliersSpin->setValue(settings.value(QStringLiteral("min_inliers")).toInt(m_minInliersSpin->value()));
    m_ransacThresholdSpin->setValue(
        settings.value(QStringLiteral("ransac_threshold")).toDouble(m_ransacThresholdSpin->value()));
    m_overlapThreadsSpin->setValue(settings.value(QStringLiteral("overlap_threads")).toInt(m_overlapThreadsSpin->value()));
    m_useFlannAssignmentCheck->setChecked(
        settings.value(QStringLiteral("use_flann_assignment")).toBool(m_useFlannAssignmentCheck->isChecked()));
    m_useInvertedIndexCheck->setChecked(
        settings.value(QStringLiteral("use_inverted_index")).toBool(m_useInvertedIndexCheck->isChecked()));
    m_useCudaOverlapCheck->setChecked(
        settings.value(QStringLiteral("use_cuda_overlap")).toBool(m_useCudaOverlapCheck->isChecked()));
    m_geometryMaxDescriptorsSpin->setValue(
        settings.value(QStringLiteral("geometry_max_descriptors")).toInt(m_geometryMaxDescriptorsSpin->value()));
    m_geometryMaxPairsSpin->setValue(
        settings.value(QStringLiteral("geometry_max_pairs")).toInt(m_geometryMaxPairsSpin->value()));
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
    updateMethodUi();
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

void VocabularyOverlapDialog::cancelRun()
{
    if (!m_runWatcher || !m_runWatcher->isRunning())
    {
        return;
    }
    if (m_cancelFlag)
    {
        m_cancelFlag->store(true, std::memory_order_relaxed);
    }
    m_runBtn->setEnabled(false);
    m_runBtn->setText(QStringLiteral("正在取消..."));
    m_summaryLabel->setText(QStringLiteral("正在取消获取重叠对..."));
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
    if (m_runWatcher && m_runWatcher->isRunning())
    {
        cancelRun();
        emit overlapCancelRequested();
        return;
    }

    VocabularyOverlapRunRequest request;
    request.method = m_overlapMethodCombo->currentData().toString();
    if (request.method.isEmpty())
    {
        request.method = QStringLiteral("vocabulary");
    }
    request.images = checkedImages();
    request.suffix = selectedFeatureSuffix();
    request.featureDir = m_featureDirEdit->text().trimmed();
    request.projectPath = m_projectManager ? m_projectManager->currentProjectPath() : QString();
    request.projectMeta = m_projectManager ? m_projectManager->currentMeta() : QJsonObject();
    request.settings = collectSettings();
    request.applyToMatching = m_applyToMatchingCheck->isChecked();
    request.referenceBody = referenceBodyFromId(m_referenceBodyCombo->currentData().toString());
    request.referenceRadiusMeters = xjw::referenceBodyRadiusMeters(request.referenceBody);
    request.referenceElevationMeters = m_referenceElevationSpin->value();
    request.autoReferenceElevation = m_autoReferenceElevationCheck->isChecked();
    request.cameraNeighborFactor = m_cameraNeighborFactorSpin->value();
    request.cancelFlag = std::make_shared<std::atomic_bool>(false);
    request.config.branchFactor = m_branchFactorSpin->value();
    request.config.treeDepth = m_treeDepthSpin->value();
    request.config.samplePerImage = m_samplePerImageSpin->value();
    request.config.maxTrainingDescriptors = m_maxTrainingDescriptorsSpin->value();
    request.config.topK = m_topKSpin->value();
    request.config.minSimilarity = m_minSimilaritySpin->value();
    request.config.useTfidf = m_useTfidfCheck->isChecked();
    request.config.mutualTopK = m_mutualTopKCheck->isChecked();
    request.config.geometryCheck = m_enableGeometryCheck->isChecked();
    request.config.minInliers = m_minInliersSpin->value();
    request.config.ransacThreshold = m_ransacThresholdSpin->value();
    request.config.numThreads = m_overlapThreadsSpin->value();
    request.config.useFlannAssignment = m_useFlannAssignmentCheck->isChecked();
    request.config.useInvertedIndex = m_useInvertedIndexCheck->isChecked();
    request.config.useCuda = m_useCudaOverlapCheck->isChecked();
    request.config.geometryMaxDescriptors = m_geometryMaxDescriptorsSpin->value();
    request.config.geometryMaxCandidatePairs = m_geometryMaxPairsSpin->value();

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

    m_cancelFlag = request.cancelFlag;
    setUiBusy(true, QStringLiteral("正在获取重叠对..."));
    emit overlapProgressChanged(QStringLiteral("正在获取重叠对"), 0);

    auto *watcher = new QFutureWatcher<RunResult>(this);
    m_runWatcher = watcher;
    connect(watcher, &QFutureWatcher<RunResult>::finished, this, [this, watcher]()
    {
        handleRunFinished(watcher);
    });
    watcher->setFuture(QtConcurrent::run([request]()
    {
        return runVocabularyOverlapRequest(request);
    }));
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
    const int vocabIndex = m_overlapMethodCombo->findData(QStringLiteral("vocabulary"));
    const int earthIndex = m_referenceBodyCombo->findData(QStringLiteral("earth"));
    m_overlapMethodCombo->setCurrentIndex(vocabIndex >= 0 ? vocabIndex : 0);
    m_referenceBodyCombo->setCurrentIndex(earthIndex >= 0 ? earthIndex : 0);
    m_autoReferenceElevationCheck->setChecked(true);
    m_referenceElevationSpin->setValue(0.0);
    m_cameraNeighborFactorSpin->setValue(2.0);
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
    m_overlapThreadsSpin->setValue(0);
    m_useFlannAssignmentCheck->setChecked(true);
    m_useInvertedIndexCheck->setChecked(true);
    m_useCudaOverlapCheck->setChecked(false);
    m_geometryMaxDescriptorsSpin->setValue(2048);
    m_geometryMaxPairsSpin->setValue(2000);
    m_applyToMatchingCheck->setChecked(true);

    if (!outputDir.isEmpty())
    {
        m_outputJsonEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.json")));
        m_outputLisEdit->setText(QDir(outputDir).filePath(QStringLiteral("vocabulary_overlap_pairs.lis")));
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
    settings.insert(QStringLiteral("overlap_method"), m_overlapMethodCombo->currentData().toString());
    settings.insert(QStringLiteral("feature_suffix"), selectedFeatureSuffix());
    settings.insert(QStringLiteral("feature_algorithm"), m_featureAlgorithmCombo->currentText());
    settings.insert(QStringLiteral("feature_dir"), m_featureDirEdit->text().trimmed());
    settings.insert(QStringLiteral("branch_factor"), m_branchFactorSpin->value());
    settings.insert(QStringLiteral("tree_depth"), m_treeDepthSpin->value());
    settings.insert(QStringLiteral("sample_per_image"), m_samplePerImageSpin->value());
    settings.insert(QStringLiteral("max_training_descriptors"), m_maxTrainingDescriptorsSpin->value());
    settings.insert(QStringLiteral("top_k"), m_topKSpin->value());
    settings.insert(QStringLiteral("min_similarity"), m_minSimilaritySpin->value());
    const xjw::ReferenceBody body = referenceBodyFromId(m_referenceBodyCombo->currentData().toString());
    settings.insert(QStringLiteral("reference_body"), referenceBodyId(body));
    settings.insert(QStringLiteral("reference_radius_m"), xjw::referenceBodyRadiusMeters(body));
    settings.insert(QStringLiteral("reference_elevation_m"), m_referenceElevationSpin->value());
    settings.insert(QStringLiteral("auto_reference_elevation"), m_autoReferenceElevationCheck->isChecked());
    settings.insert(QStringLiteral("fixed_ground_z"), m_referenceElevationSpin->value());
    settings.insert(QStringLiteral("camera_neighbor_factor"), m_cameraNeighborFactorSpin->value());
    settings.insert(QStringLiteral("use_tfidf"), m_useTfidfCheck->isChecked());
    settings.insert(QStringLiteral("mutual_top_k"), m_mutualTopKCheck->isChecked());
    settings.insert(QStringLiteral("geometry_check"), m_enableGeometryCheck->isChecked());
    settings.insert(QStringLiteral("geometry_model"), m_geometryModelCombo->currentData().toString());
    settings.insert(QStringLiteral("min_inliers"), m_minInliersSpin->value());
    settings.insert(QStringLiteral("ransac_threshold"), m_ransacThresholdSpin->value());
    settings.insert(QStringLiteral("overlap_threads"), m_overlapThreadsSpin->value());
    settings.insert(QStringLiteral("use_flann_assignment"), m_useFlannAssignmentCheck->isChecked());
    settings.insert(QStringLiteral("use_inverted_index"), m_useInvertedIndexCheck->isChecked());
    settings.insert(QStringLiteral("use_cuda_overlap"), m_useCudaOverlapCheck->isChecked());
    settings.insert(QStringLiteral("geometry_max_descriptors"), m_geometryMaxDescriptorsSpin->value());
    settings.insert(QStringLiteral("geometry_max_pairs"), m_geometryMaxPairsSpin->value());
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

void VocabularyOverlapDialog::setUiBusy(bool busy, const QString &message)
{
    m_runBtn->setEnabled(true);
    m_runBtn->setText(busy ? QStringLiteral("取消") : QStringLiteral("运行"));
    m_selectAllBtn->setEnabled(!busy);
    m_clearSelectionBtn->setEnabled(!busy);
    m_overlapMethodCombo->setEnabled(!busy);
    m_referenceBodyCombo->setEnabled(!busy);
    m_autoReferenceElevationCheck->setEnabled(!busy);
    m_referenceElevationSpin->setEnabled(!busy && !m_autoReferenceElevationCheck->isChecked());
    m_cameraNeighborFactorSpin->setEnabled(!busy);
    m_featureAlgorithmCombo->setEnabled(!busy);
    m_featureDirEdit->setEnabled(!busy);
    m_autoDetectFeatureDirBtn->setEnabled(!busy);
    m_browseFeatureDirBtn->setEnabled(!busy);
    m_branchFactorSpin->setEnabled(!busy);
    m_treeDepthSpin->setEnabled(!busy);
    m_samplePerImageSpin->setEnabled(!busy);
    m_maxTrainingDescriptorsSpin->setEnabled(!busy);
    m_topKSpin->setEnabled(!busy);
    m_minSimilaritySpin->setEnabled(!busy);
    m_useTfidfCheck->setEnabled(!busy);
    m_mutualTopKCheck->setEnabled(!busy);
    m_enableGeometryCheck->setEnabled(!busy);
    m_minInliersSpin->setEnabled(!busy);
    m_ransacThresholdSpin->setEnabled(!busy);
    m_geometryModelCombo->setEnabled(!busy);
    m_overlapThreadsSpin->setEnabled(!busy);
    m_useFlannAssignmentCheck->setEnabled(!busy);
    m_useInvertedIndexCheck->setEnabled(!busy);
    m_useCudaOverlapCheck->setEnabled(!busy);
    m_geometryMaxDescriptorsSpin->setEnabled(!busy);
    m_geometryMaxPairsSpin->setEnabled(!busy);
    m_outputJsonEdit->setEnabled(!busy);
    m_outputLisEdit->setEnabled(!busy);
    m_applyToMatchingCheck->setEnabled(!busy);
    m_resetBtn->setEnabled(!busy);
    m_exportLisBtn->setEnabled(!busy && !m_generatedPairs.isEmpty());
    m_applyToMatchingBtn->setEnabled(!busy && !m_generatedPairs.isEmpty());
    m_closeBtn->setEnabled(!busy);

    if (!message.isEmpty())
    {
        m_summaryLabel->setText(message);
    }
    if (!busy)
    {
        updateMethodUi(false);
    }
}

void VocabularyOverlapDialog::handleProgress(const QString &stage, int percent)
{
    if (!m_runWatcher || !m_runWatcher->isRunning())
    {
        return;
    }

    const QString text = percent > 0 && percent < 100
        ? QStringLiteral("%1 %2%").arg(stage).arg(percent)
        : stage;
    m_summaryLabel->setText(text);
    emit overlapProgressChanged(stage, std::clamp(percent, 0, 100));
}

void VocabularyOverlapDialog::handleRunFinished(QFutureWatcher<RunResult> *watcher)
{
    if (!watcher)
    {
        return;
    }

    const bool isCurrentWatcher = (watcher == m_runWatcher);
    const RunResult runResult = watcher->result();
    if (isCurrentWatcher)
    {
        m_runWatcher = nullptr;
    }
    m_cancelFlag.reset();
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

    m_candidatePairs = runResult.candidates;
    m_generatedPairs = runResult.generatedPairs;
    m_lastRunSettings = runResult.settings;

    populatePairTable();
    m_summaryLabel->setText(QStringLiteral("候选 %1，对外输出 %2，词汇数 %3")
                                .arg(m_candidatePairs.size())
                                .arg(m_generatedPairs.size())
                                .arg(runResult.vocabularySize));
    setUiBusy(false);
    emit overlapFinished(true);

    emit settingsChanged(m_lastRunSettings);
    if (runResult.applyToMatching)
    {
        emit generatedPairsReady(m_generatedPairs, m_lastRunSettings);
    }
}

bool VocabularyOverlapDialog::writeOutputs(const QJsonObject &settings,
                                           const QStringList &pairs,
                                           QString *errorMsg) const
{
    return writeOverlapOutputs(settings, pairs, m_candidatePairs, errorMsg);
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
    m_pairTable->setVisible(true);
    m_pairTable->viewport()->update();
}

void VocabularyOverlapDialog::updateMethodUi(bool refreshSummary)
{
    const bool cameraMode = m_overlapMethodCombo &&
        m_overlapMethodCombo->currentData().toString() == QStringLiteral("camera");

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

    m_referenceBodyCombo->setEnabled(cameraMode);
    m_autoReferenceElevationCheck->setEnabled(cameraMode);
    m_referenceElevationSpin->setEnabled(cameraMode && !m_autoReferenceElevationCheck->isChecked());
    m_cameraNeighborFactorSpin->setEnabled(cameraMode);
    m_useTfidfCheck->setEnabled(!cameraMode);
    m_useFlannAssignmentCheck->setEnabled(!cameraMode);
    m_useInvertedIndexCheck->setEnabled(!cameraMode);
    m_useCudaOverlapCheck->setEnabled(!cameraMode);
    m_geometryMaxDescriptorsSpin->setEnabled(!cameraMode);
    m_geometryMaxPairsSpin->setEnabled(!cameraMode);

    if (!refreshSummary)
    {
        return;
    }

    if (cameraMode)
    {
        m_summaryLabel->setText(QStringLiteral("相机模型模式：使用项目中的相机参数、影像尺寸和基准球面计算地面重叠对"));
    }
    else
    {
        refreshFeatureStatus();
    }
}
