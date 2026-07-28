#include "MenuWorkflowController.h"

#include "ProjectManager.h"
#include "project/ProjectIO.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "workflow/AerialTriangulationWorkflow.h"
#include "preparation/MatchResultCatalog.h"
#include "preparation/ReconstructionPrerequisiteReport.h"
#include "Logger.h"
#include "project/SparseResultQuality.h"

#include "FeatureExtractionDialog.h"
#include "VocabularyOverlapDialog.h"
#include "FeatureExtractionRunner.h"
#include "GuiTaskRunner.h"
#include "FeaturePointVisualizationDialog.h"
#include "CanvasWidget.h"
#include "MainWindow.h"
#include "MainMenu.h"
#include "MatchPairSelectorDialog.h"
#include "AerialTriangulationDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "OverlapAnalysisDialog.h"
#include "CreateDemDialog.h"
#include "MapProjectDialog.h"
#include "WorkflowReportDialog.h"
#include "CameraConvertDialog.h"

#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"

#include <QColor>
#include <QCheckBox>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QSet>
#include <QSettings>
#include <QTimer>
#include <QAction>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <numeric>

namespace
{

bool isSequenceReferencePreselection(const QJsonObject &settings)
{
    if (!settings.value(QStringLiteral("reference_preselection")).toBool(false))
    {
        return false;
    }

    const QString source = settings.value(QStringLiteral("reference_preselection_source"))
                               .toString(QStringLiteral("source_code"))
                               .trimmed()
                               .toLower();
    return source == QStringLiteral("sequence") ||
           source == QStringLiteral("sequential") ||
           source == QStringLiteral("photo_sequence");
}

bool shouldUseStoredGeneratedPairConstraints(const QJsonObject &settings)
{
    if (isSequenceReferencePreselection(settings))
    {
        // 照片序列是用户显式选择的配对先验。历史 generated_pairs 多来自词汇树/重叠预选，
        // 不能在空三启动时继续作为硬白名单，否则会漏掉序列边和首尾附近的闭环边。
        return false;
    }
    return true;
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

float normalizedFeatureGrayscaleMin(const QJsonObject &settings)
{
    if (settings.contains(QStringLiteral("feature_grayscale_min_px")))
    {
        const int px = std::clamp(settings.value(QStringLiteral("feature_grayscale_min_px")).toInt(5), 0, 255);
        return static_cast<float>(px) / 255.0f;
    }

    double value = settings.value(QStringLiteral("feature_grayscale_min")).toDouble(5.0 / 255.0);
    if (value > 1.0)
    {
        value /= 255.0;
    }
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
}

int sfmQualityLevelFromWorkflowQuality(const QString &quality)
{
    if (quality == QStringLiteral("fast"))
    {
        return 0;
    }
    if (quality == QStringLiteral("quality"))
    {
        return 2;
    }
    return 1;
}

/// 将最新报告写入 latest 文件，并把同一份报告追加到历史数组文件中。
bool writeLatestAndAppendHistoryReport(const QString &reportsDir,
                                       const QString &latestFileName,
                                       const QString &historyFileName,
                                       const QJsonObject &report)
{
    if (reportsDir.isEmpty() || report.isEmpty())
    {
        return false;
    }

    QDir().mkpath(reportsDir);

    QFile latestFile(QDir(reportsDir).filePath(latestFileName));
    if (!latestFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    latestFile.write(QJsonDocument(report).toJson(QJsonDocument::Compact));
    latestFile.close();

    QJsonArray history;
    QFile historyFile(QDir(reportsDir).filePath(historyFileName));
    if (historyFile.open(QIODevice::ReadOnly))
    {
        const QJsonDocument oldDoc = QJsonDocument::fromJson(historyFile.readAll());
        if (oldDoc.isArray())
        {
            history = oldDoc.array();
        }
        historyFile.close();
    }
    history.append(report);

    if (!historyFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    historyFile.write(QJsonDocument(history).toJson(QJsonDocument::Indented));
    historyFile.close();
    return true;
}

/// 从特征匹配对话框设置中读取已生成的配对约束，并检测其是否覆盖当前选图。
QStringList loadGeneratedPairConstraints(const QString &projectPath,
                                         const QJsonObject &projectMeta,
                                         const QStringList &selectedImages,
                                         bool *usedStoredPairs,
                                         bool *storedPairsStale)
{
    if (usedStoredPairs)
    {
        *usedStoredPairs = false;
    }
    if (storedPairsStale)
    {
        *storedPairsStale = false;
    }
    if (projectPath.isEmpty())
    {
        return {};
    }

    DialogSettingStore store(DialogSettingKeys::FeatureMatching, nullptr);
    store.setProjectPath(projectPath);
    const QJsonObject saved = store.load();
    const QJsonArray generatedPairs = saved.value(QStringLiteral("generated_pairs")).toArray();
    if (generatedPairs.isEmpty())
    {
        return {};
    }

    if (usedStoredPairs)
    {
        *usedStoredPairs = true;
    }

    QSet<QString> selectedSet;
    QSet<QString> coveredSelectedImages;
    for (const QString &imagePath : selectedImages)
    {
        selectedSet.insert(xjw::common::project::normalizePath(imagePath));
    }

    QStringList allowedPairs;
    QSet<QString> seenPairs;
    for (const QJsonValue &value : generatedPairs)
    {
        const QString pairText = value.toString().trimmed();
        const int separator = pairText.indexOf(QStringLiteral("__"));
        if (separator <= 0)
        {
            continue;
        }

        const QString tokenA = pairText.left(separator);
        const QString tokenB = pairText.mid(separator + 2);
        const QString imageA = xjw::common::project::resolveProjectImagePathFromToken(tokenA, projectMeta);
        const QString imageB = xjw::common::project::resolveProjectImagePathFromToken(tokenB, projectMeta);
        if (imageA.isEmpty() || imageB.isEmpty())
        {
            continue;
        }

        const QString normA = xjw::common::project::normalizePath(imageA);
        const QString normB = xjw::common::project::normalizePath(imageB);
        if (!selectedSet.contains(normA) || !selectedSet.contains(normB))
        {
            continue;
        }

        const QString pairKey = xjw::common::project::canonicalImagePairKey(
            normA, normB, QStringLiteral("\n"));
        if (pairKey.isEmpty() || seenPairs.contains(pairKey))
        {
            continue;
        }

        seenPairs.insert(pairKey);
        allowedPairs.append(pairKey);
        coveredSelectedImages.insert(normA);
        coveredSelectedImages.insert(normB);
    }

    if (!selectedSet.isEmpty() && coveredSelectedImages.size() != selectedSet.size())
    {
        if (storedPairsStale)
        {
            *storedPairsStale = true;
        }
        return {};
    }

    if (usedStoredPairs)
    {
        *usedStoredPairs = !allowedPairs.isEmpty();
    }

    return allowedPairs;
}

} // namespace

MenuWorkflowController::MenuWorkflowController(QMainWindow *mainWindow, QObject *parent)
    : QObject(parent)
    , _mainWindow(mainWindow)
{
}

void MenuWorkflowController::setProjectManager(ProjectManager *projectManager)
{
    _projectManager = projectManager;
}

void MenuWorkflowController::bindActions(MainMenu *mainMenu)
{
    if (!mainMenu)
    {
        return;
    }

    auto connectAction = [this](QAction *action, void (MenuWorkflowController::*slot)())
    {
        if (action)
        {
            connect(action, &QAction::triggered, this, slot, Qt::UniqueConnection);
        }
    };

    connectAction(mainMenu->detectFeaturesAction(), &MenuWorkflowController::openFeatureExtractionDialog);
    connectAction(mainMenu->vocabularyOverlapAction(), &MenuWorkflowController::openVocabularyOverlapDialog);
    connectAction(mainMenu->workflowAerialTriangulationAction(),
                  &MenuWorkflowController::openWorkflowAerialTriangulationDialog);
    connectAction(mainMenu->aerialTriangulationAction(), &MenuWorkflowController::openAerialTriangulationDialog);
    connectAction(mainMenu->featureVisualizationAction(), &MenuWorkflowController::openFeaturePointVisualizationDialog);
    connectAction(mainMenu->threeDReconstructionAction(), &MenuWorkflowController::openThreeDReconstructionDialog);
    connectAction(mainMenu->overlapAnalysisAction(), &MenuWorkflowController::openOverlapAnalysisDialog);
    connectAction(mainMenu->createDEMAction(), &MenuWorkflowController::openCreateDemDialog);
    connectAction(mainMenu->generateOrthoAction(), &MenuWorkflowController::openMapProjectDialog);
    connectAction(mainMenu->viewWorkflowReportAction(), &MenuWorkflowController::openWorkflowReportDialog);
    connectAction(mainMenu->cameraConvertAction(), &MenuWorkflowController::openCameraConvertDialog);

    if (!_projectManager)
    {
        return;
    }

    auto connectProjectAction = [this](QAction *action, void (ProjectManager::*slot)())
    {
        if (action)
        {
            connect(action, &QAction::triggered, _projectManager, slot, Qt::UniqueConnection);
        }
    };

    connectProjectAction(mainMenu->importReferenceDatasetAction(), &ProjectManager::importReferenceDataset);
    connectProjectAction(mainMenu->surveyControlAction(), &ProjectManager::openSurveyControlDialog);
    connectProjectAction(mainMenu->generateMaskAction(), &ProjectManager::openGenerateMaskDialog);
    connectProjectAction(mainMenu->referenceQualityCheckAction(), &ProjectManager::runReferenceQualityCheck);
    connectProjectAction(mainMenu->referenceTerrainBundleAdjustAction(),
                         &ProjectManager::prepareReferenceTerrainBundleAdjust);
}

QJsonObject MenuWorkflowController::colorToJson(const QColor &c)
{
    QJsonObject o;
    o.insert("r", c.red());
    o.insert("g", c.green());
    o.insert("b", c.blue());
    return o;
}

QStringList MenuWorkflowController::getProjectImages() const
{
    if (!_projectManager)
    {
        return QStringList();
    }

    QStringList images = _projectManager->getImagesByCategory(QStringLiteral("源数据"));
    if (images.isEmpty()) images = _projectManager->getImagesByCategory(QStringLiteral("照片"));
    if (images.isEmpty()) images = _projectManager->getImagesByCategory(QStringLiteral("Photos"));
    if (images.isEmpty()) images = _projectManager->getAllImages();
    return images;
}

MenuWorkflowController::SparsePrerequisiteSummary
MenuWorkflowController::summarizeSparsePrerequisites(const QStringList &images,
                                                     const QJsonObject &meta,
                                                     const QString &projectPath,
                                                     const QString &featureAlgorithm,
                                                     const QString &matchAlgorithm)
{
    SparsePrerequisiteSummary summary;
    summary.imageCount = images.size();
    const QString selectedFeatureAlgorithm = featureAlgorithm.trimmed().toLower();
    const QString selectedMatchAlgorithm = matchAlgorithm.trimmed().toLower();

    int availableFeatureCount = 0;
    for (const QString &imagePath : images)
    {
        if (!xjw::common::project::ProjectIO::findFeatureForImage(projectPath, imagePath).isEmpty())
        {
            ++availableFeatureCount;
        }
    }
    summary.hasFeatures = images.isEmpty() || availableFeatureCount == images.size();

    auto nameAliases = [](const QString &pathOrName) -> QStringList
    {
        const QFileInfo info(pathOrName);
        QStringList aliases;
        const QString fileName = info.fileName().toCaseFolded();
        const QString baseName = info.completeBaseName().toCaseFolded();
        const QString rawName = pathOrName.trimmed().toCaseFolded();
        if (!fileName.isEmpty())
        {
            aliases.append(fileName);
        }
        if (!baseName.isEmpty() && !aliases.contains(baseName))
        {
            aliases.append(baseName);
        }
        if (!rawName.isEmpty() && !aliases.contains(rawName))
        {
            aliases.append(rawName);
        }
        return aliases;
    };

    auto matchPairKey = [](const QString &left, const QString &right) -> QString
    {
        if (left.isEmpty() || right.isEmpty() || left == right)
        {
            return QString();
        }
        return (left < right)
            ? (left + QStringLiteral("\n") + right)
            : (right + QStringLiteral("\n") + left);
    };

    auto recordAlgorithmMatches = [&](const QJsonObject &object) -> bool
    {
        QString recordFeatureAlgorithm =
            object.value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
        if (recordFeatureAlgorithm.isEmpty())
        {
            recordFeatureAlgorithm = object.value(QStringLiteral("settings")).toObject()
                                         .value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower();
        }

        QString recordMatchAlgorithm =
            object.value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
        if (recordMatchAlgorithm.isEmpty())
        {
            recordMatchAlgorithm = object.value(QStringLiteral("settings")).toObject()
                                      .value(QStringLiteral("match_algorithm")).toString().trimmed().toLower();
        }

        if (!selectedFeatureAlgorithm.isEmpty() && recordFeatureAlgorithm != selectedFeatureAlgorithm)
        {
            return false;
        }
        if (!selectedMatchAlgorithm.isEmpty() && recordMatchAlgorithm != selectedMatchAlgorithm)
        {
            return false;
        }
        return true;
    };

    auto variantAlgorithmMatches = [&](const xjw::aerial_triangulation::MatchVariant &variant) -> bool
    {
        if (!variant.compatible)
        {
            return false;
        }
        if (!selectedFeatureAlgorithm.isEmpty() &&
            variant.featureAlgorithm.trimmed().toLower() != selectedFeatureAlgorithm)
        {
            return false;
        }
        if (!selectedMatchAlgorithm.isEmpty() &&
            variant.matchAlgorithm.trimmed().toLower() != selectedMatchAlgorithm)
        {
            return false;
        }
        return true;
    };

    QSet<QString> matchedPairKeys;
    QVector<QPair<QString, QString>> matchedPairs;

    auto appendPreflightPair = [&](const QString &leftToken, const QString &rightToken)
    {
        const QStringList leftAliases = nameAliases(leftToken);
        const QStringList rightAliases = nameAliases(rightToken);
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                const QString key = matchPairKey(left, right);
                if (!key.isEmpty() && !matchedPairKeys.contains(key))
                {
                    matchedPairKeys.insert(key);
                    matchedPairs.append(qMakePair(leftToken, rightToken));
                    return;
                }
            }
        }
    };

    if (!projectPath.isEmpty())
    {
        xjw::aerial_triangulation::MatchResultCatalogConfig catalogConfig;
        catalogConfig.matchDirectory = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
        catalogConfig.targetImagePaths = images;
        const xjw::aerial_triangulation::MatchResultCatalogSummary catalogSummary =
            xjw::aerial_triangulation::MatchResultCatalog(catalogConfig).scan();
        for (const xjw::aerial_triangulation::MatchPairGroup &group : catalogSummary.pairGroups)
        {
            for (const xjw::aerial_triangulation::MatchVariant &variant : group.variants)
            {
                if (!variantAlgorithmMatches(variant))
                {
                    continue;
                }

                appendPreflightPair(variant.imageA.isEmpty() ? group.imageA : variant.imageA,
                                    variant.imageB.isEmpty() ? group.imageB : variant.imageB);
                break;
            }
        }
    }

    const QJsonArray ipmatchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &resultValue : ipmatchResults)
    {
        const QJsonObject resultObject = resultValue.toObject();
        if (!recordAlgorithmMatches(resultObject))
        {
            continue;
        }

        QString image0Name = resultObject.value(QStringLiteral("image0_name")).toString();
        QString image1Name = resultObject.value(QStringLiteral("image1_name")).toString();

        if (image0Name.isEmpty())
        {
            image0Name = QFileInfo(resultObject.value(QStringLiteral("image0")).toString()).completeBaseName();
        }
        if (image1Name.isEmpty())
        {
            image1Name = QFileInfo(resultObject.value(QStringLiteral("image1")).toString()).completeBaseName();
        }

        appendPreflightPair(image0Name, image1Name);
    }

    for (const auto &pair : matchedPairs)
    {
        const QStringList leftAliases = nameAliases(pair.first);
        const QStringList rightAliases = nameAliases(pair.second);
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                const QString key = matchPairKey(left, right);
                if (!key.isEmpty())
                {
                    matchedPairKeys.insert(key);
                }
            }
        }
    }

    QSet<QString> processedPairKeys = matchedPairKeys;
    QVector<QPair<QString, QString>> settledNoMatchPairs;
    QSet<QString> settledNoMatchKeys;
    auto appendSettledNoMatchPair = [&](const QString &leftToken, const QString &rightToken)
    {
        const QStringList leftAliases = nameAliases(leftToken);
        const QStringList rightAliases = nameAliases(rightToken);
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                const QString key = matchPairKey(left, right);
                if (!key.isEmpty() && !settledNoMatchKeys.contains(key))
                {
                    settledNoMatchKeys.insert(key);
                    settledNoMatchPairs.append(qMakePair(leftToken, rightToken));
                    return;
                }
            }
        }
    };

    if (!projectPath.isEmpty())
    {
        QFile noMatchFile(QDir(xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath))
                              .filePath(QStringLiteral("no_match_pairs.json")));
        if (noMatchFile.open(QIODevice::ReadOnly))
        {
            const QJsonArray records = QJsonDocument::fromJson(noMatchFile.readAll()).array();
            for (const QJsonValue &value : records)
            {
                const QJsonObject object = value.toObject();
                if (!recordAlgorithmMatches(object))
                {
                    continue;
                }
                appendSettledNoMatchPair(object.value(QStringLiteral("image0")).toString(),
                                         object.value(QStringLiteral("image1")).toString());
            }
        }
    }

    for (const auto &pair : settledNoMatchPairs)
    {
        const QStringList leftAliases = nameAliases(pair.first);
        const QStringList rightAliases = nameAliases(pair.second);
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                const QString key = matchPairKey(left, right);
                if (!key.isEmpty())
                {
                    processedPairKeys.insert(key);
                }
            }
        }
    }

    auto pairCoveredByAliases = [&](const QStringList &leftAliases, const QStringList &rightAliases) -> bool
    {
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                if (matchedPairKeys.contains(matchPairKey(left, right)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto pairProcessedByAliases = [&](const QStringList &leftAliases, const QStringList &rightAliases) -> bool
    {
        for (const QString &left : leftAliases)
        {
            for (const QString &right : rightAliases)
            {
                if (processedPairKeys.contains(matchPairKey(left, right)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto imagePairCovered = [&](const QString &leftImage, const QString &rightImage) -> bool
    {
        return pairCoveredByAliases(nameAliases(leftImage), nameAliases(rightImage));
    };

    auto imagePairProcessed = [&](const QString &leftImage, const QString &rightImage) -> bool
    {
        return pairProcessedByAliases(nameAliases(leftImage), nameAliases(rightImage));
    };

    bool usedStoredPairs = false;
    bool storedPairsStale = false;
    const QStringList generatedPairs = loadGeneratedPairConstraints(projectPath,
                                                                    meta,
                                                                    images,
                                                                    &usedStoredPairs,
                                                                    &storedPairsStale);
    int generatedPairRequiredCount = 0;
    int generatedPairCoveredCount = 0;
    int generatedPairProcessedCount = 0;

    struct MatchGraphStats
    {
        int matchedEdgeCount = 0;
        int matchedImageCount = 0;
        int componentCount = 0;
        int largestComponentSize = 0;
        bool allImagesCovered = true;
        bool connected = true;
    };

    auto matchGraphStats = [&]() -> MatchGraphStats
    {
        MatchGraphStats stats;
        if (images.size() < 2)
        {
            stats.matchedImageCount = images.size();
            stats.componentCount = images.isEmpty() ? 0 : 1;
            stats.largestComponentSize = images.size();
            return stats;
        }

        QHash<QString, int> imageIndexByAlias;
        for (int i = 0; i < images.size(); ++i)
        {
            const QStringList aliases = nameAliases(images.at(i));
            for (const QString &alias : aliases)
            {
                if (!imageIndexByAlias.contains(alias))
                {
                    imageIndexByAlias.insert(alias, i);
                }
            }
        }

        QVector<int> parent(images.size());
        std::iota(parent.begin(), parent.end(), 0);
        auto findRoot = [&parent](int index) -> int
        {
            int root = index;
            while (parent[root] != root)
            {
                root = parent[root];
            }
            while (parent[index] != index)
            {
                const int next = parent[index];
                parent[index] = root;
                index = next;
            }
            return root;
        };

        auto resolveImageIndex = [&](const QString &pathOrName) -> int
        {
            const QStringList aliases = nameAliases(pathOrName);
            for (const QString &alias : aliases)
            {
                const auto it = imageIndexByAlias.constFind(alias);
                if (it != imageIndexByAlias.constEnd())
                {
                    return it.value();
                }
            }
            return -1;
        };

        QSet<int> matchedImageIndices;
        for (const auto &pair : matchedPairs)
        {
            const int leftIndex = resolveImageIndex(pair.first);
            const int rightIndex = resolveImageIndex(pair.second);
            if (leftIndex < 0 || rightIndex < 0 || leftIndex == rightIndex)
            {
                continue;
            }

            const int leftRoot = findRoot(leftIndex);
            const int rightRoot = findRoot(rightIndex);
            if (leftRoot != rightRoot)
            {
                parent[rightRoot] = leftRoot;
            }
            matchedImageIndices.insert(leftIndex);
            matchedImageIndices.insert(rightIndex);
            ++stats.matchedEdgeCount;
        }

        stats.matchedImageCount = matchedImageIndices.size();
        stats.allImagesCovered = (stats.matchedImageCount == images.size());
        if (stats.matchedEdgeCount <= 0)
        {
            stats.connected = false;
            stats.componentCount = images.size();
            stats.largestComponentSize = images.isEmpty() ? 0 : 1;
            return stats;
        }

        QHash<int, int> componentSizes;
        for (int i = 0; i < images.size(); ++i)
        {
            const int root = findRoot(i);
            componentSizes[root] = componentSizes.value(root) + 1;
        }
        stats.componentCount = componentSizes.size();
        for (auto it = componentSizes.constBegin(); it != componentSizes.constEnd(); ++it)
        {
            stats.largestComponentSize = std::max(stats.largestComponentSize, it.value());
        }
        stats.connected = (stats.componentCount <= 1);
        return stats;
    };

    const MatchGraphStats matchStats = matchGraphStats();

    if (usedStoredPairs && !storedPairsStale)
    {
        for (const QString &pairKey : generatedPairs)
        {
            const QStringList parts = pairKey.split(QStringLiteral("\n"));
            if (parts.size() != 2)
            {
                continue;
            }

            ++generatedPairRequiredCount;
            if (imagePairCovered(parts.at(0), parts.at(1)))
            {
                ++generatedPairCoveredCount;
            }
            if (imagePairProcessed(parts.at(0), parts.at(1)))
            {
                ++generatedPairProcessedCount;
            }
        }
    }
    const bool generatedPlanHasNoProcessedPairs = usedStoredPairs
        && !storedPairsStale
        && generatedPairRequiredCount > 0
        && generatedPairProcessedCount == 0;
    summary.hasMatches = images.size() < 2 || (!generatedPlanHasNoProcessedPairs && matchStats.matchedEdgeCount > 0);

    xjw::aerial_triangulation::ReconstructionPrerequisiteReport prerequisiteReport;
    prerequisiteReport.imageCount = static_cast<int>(images.size());
    prerequisiteReport.plannedPairCount = generatedPairRequiredCount;
    prerequisiteReport.validMatchPairCount = matchStats.matchedEdgeCount;
    prerequisiteReport.settledNoMatchPairCount = settledNoMatchPairs.size();
    prerequisiteReport.missingFeaturePairCount =
        std::max(0, static_cast<int>(images.size()) - availableFeatureCount);
    prerequisiteReport.missingMatchPairCount =
        (usedStoredPairs && !storedPairsStale && generatedPairRequiredCount > 0)
            ? std::max(0, generatedPairRequiredCount - generatedPairProcessedCount)
            : 0;
    prerequisiteReport.failedGeometryPairCount = settledNoMatchPairs.size();
    summary.prerequisiteReport = prerequisiteReport.toJson();

    const auto recommendedAction = prerequisiteReport.recommendedAction();
    const bool matchingProducedNoUsableEdges =
        recommendedAction == xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality;
    summary.blockOnMatchQuality = matchingProducedNoUsableEdges;

    switch (recommendedAction)
    {
    case xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::RunSfmWithExistingMatches:
        LOG_INFO(QStringLiteral("空三上游数据就绪：复用已有匹配 %1 对，已确认无匹配 %2 对")
                     .arg(prerequisiteReport.validMatchPairCount)
                     .arg(prerequisiteReport.settledNoMatchPairCount));
        break;
    case xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::FillMissingMatchesOnly:
        LOG_INFO(QStringLiteral("空三缺少部分匹配：只补齐缺失 pair %1 对，复用已有匹配 %2 对")
                     .arg(prerequisiteReport.missingMatchPairCount)
                     .arg(prerequisiteReport.validMatchPairCount));
        break;
    case xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::PrepareFeaturesAndMatches:
        LOG_INFO(QStringLiteral("空三缺少连接点输入：创建连接点流程将自动提取特征并匹配"));
        break;
    case xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality:
        LOG_WARN(QStringLiteral("匹配阶段已完成，但没有可用于空三的连接边；请检查匹配参数、重叠对和几何验证报告。"));
        break;
    }

    if (matchStats.matchedEdgeCount > 0)
    {
        LOG_INFO(QStringLiteral("空中三角测量预检: 匹配边=%1 覆盖影像=%2/%3 连通分量=%4 最大分量=%5 已确认无匹配=%6")
                     .arg(matchStats.matchedEdgeCount)
                     .arg(matchStats.matchedImageCount)
                     .arg(images.size())
                     .arg(matchStats.componentCount)
                     .arg(matchStats.largestComponentSize)
                     .arg(settledNoMatchPairs.size()));
    }
    else if (images.size() >= 2)
    {
        LOG_WARN(QStringLiteral("空中三角测量预检: 未找到已完成的影像匹配结果"));
    }

    if (!summary.hasFeatures)
    {
        summary.missingMessages.append(
            QStringLiteral("缺少连接点输入：当前影像特征不完整，将通过创建连接点流程自动补齐。"));
    }
    if (!summary.hasMatches && matchingProducedNoUsableEdges)
    {
        summary.warningMessages.append(
            QStringLiteral("匹配阶段已完成，但没有可用于空三的连接边；请检查匹配参数、重叠对和几何验证报告。"));
    }
    else if (!summary.hasMatches)
    {
        summary.missingMessages.append(
            QStringLiteral("缺少连接点：当前影像对匹配不完整，将通过创建连接点流程自动补齐。"));
    }
    if (summary.hasMatches && images.size() >= 2 && matchStats.matchedEdgeCount > 0)
    {
        if (!matchStats.allImagesCovered)
        {
            summary.warningMessages.append(
                QStringLiteral("匹配网络未覆盖全部影像：已覆盖 %1/%2 张，空三可能只注册部分影像。")
                    .arg(matchStats.matchedImageCount)
                    .arg(images.size()));
        }
        if (!matchStats.connected)
        {
            summary.warningMessages.append(
                QStringLiteral("匹配网络不连通：%1 个连通分量，最大分量 %2/%3 张；空三可能只从最大分量开始扩展。")
                    .arg(matchStats.componentCount)
                    .arg(matchStats.largestComponentSize)
                    .arg(images.size()));
        }
    }
    for (const QString &warning : summary.warningMessages)
    {
        LOG_WARN(QStringLiteral("空中三角测量预检: %1").arg(warning));
    }
    return summary;
}

void MenuWorkflowController::openFeatureExtractionDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dlg = new FeatureExtractionDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (_projectManager)
    {
        if (!_featureExtractionSetting)
        {
            _featureExtractionSetting = new DialogSettingStore(DialogSettingKeys::FeatureExtraction, this);
        }
        _featureExtractionSetting->setProjectPath(_projectManager->currentProjectPath());

        // 从 project_dialog.json 加载之前保存的设置
        const QJsonObject saved = _featureExtractionSetting->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }

        // 自动获取项目中的照片数据作为输入选择
        QStringList images = getProjectImages();
        if (!images.isEmpty())
        {
            dlg->setProjectImages(images);
        }

        // 设置默认输出目录
        const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
        if (!assetsDir.isEmpty() && saved.value(QStringLiteral("output_dir")).toString().isEmpty())
        {
            QJsonObject defaultOutput = saved;
            defaultOutput.insert("output_dir", QDir(assetsDir).filePath(QStringLiteral("ip")));
            dlg->applySettings(defaultOutput);
        }
    }
    else
    {
        LOG_ERROR(QStringLiteral("无法运行特征提取：项目管理器未初始化"));
    }

    // 连接设置变更信号，实时保存到 project_dialog.json
    connect(dlg, &FeatureExtractionDialog::settingsChanged, this, [this](const QJsonObject &s)
    {
        if (_featureExtractionSetting)
        {
            _featureExtractionSetting->save(s);
        }
    });

    // 连接运行请求信号
    connect(dlg, &FeatureExtractionDialog::runRequested, this,
        [this](const QJsonObject &config, const QStringList &inputs)
    {
        if (!_projectManager)
        {
            LOG_ERROR(QStringLiteral("无法运行特征提取：项目管理器未初始化"));
            return;
        }

        runFeatureExtraction(config, inputs);
    });

    dlg->exec();
}

void MenuWorkflowController::openVocabularyOverlapDialog()
{
    if (!_mainWindow)
    {
        return;
    }
    if (!_projectManager || _projectManager->currentProjectPath().isEmpty())
    {
        LOG_ERROR(QStringLiteral("无法获取重叠对：请先打开或创建项目"));
        QMessageBox::warning(_mainWindow, QStringLiteral("获取重叠对"), QStringLiteral("请先打开或创建项目。"));
        return;
    }

    if (!_vocabOverlapSetting)
    {
        _vocabOverlapSetting = new DialogSettingStore(DialogSettingKeys::VocabularyOverlap, this);
    }
    _vocabOverlapSetting->setProjectPath(_projectManager->currentProjectPath());

    auto *dlg = new VocabularyOverlapDialog(_projectManager, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setProjectImages(getProjectImages());

    auto *mainWin = qobject_cast<MainWindow *>(_mainWindow.data());
    QMetaObject::Connection overlapCancelConn;
    if (mainWin)
    {
        connect(dlg, &VocabularyOverlapDialog::overlapProgressChanged,
                mainWin, &MainWindow::onOverlapProgress);
        connect(dlg, &VocabularyOverlapDialog::overlapFinished,
                mainWin, &MainWindow::onOverlapFinished);
        overlapCancelConn = connect(mainWin, &MainWindow::overlapCancelRequested,
                                    dlg, &VocabularyOverlapDialog::cancelRun);
        connect(dlg, &QObject::destroyed, mainWin, [overlapCancelConn]()
        {
            QObject::disconnect(overlapCancelConn);
        });
    }

    const QJsonObject saved = _vocabOverlapSetting->load();
    if (!saved.isEmpty())
    {
        dlg->applySettings(saved);
    }

    connect(dlg, &VocabularyOverlapDialog::settingsChanged, this, [this](const QJsonObject &settings)
    {
        if (_vocabOverlapSetting)
        {
            _vocabOverlapSetting->save(settings);
        }
    });

    connect(dlg,
            &VocabularyOverlapDialog::generatedPairsReady,
            this,
            [this](const QStringList &pairs, const QJsonObject &settings)
    {
        if (!_projectManager || _projectManager->currentProjectPath().isEmpty())
        {
            return;
        }

        DialogSettingStore matchStore(DialogSettingKeys::FeatureMatching, this);
        matchStore.setProjectPath(_projectManager->currentProjectPath());

        QJsonObject matchingSettings = matchStore.load();
        matchingSettings.insert(QStringLiteral("generated_pairs"), stringListToJsonArray(pairs));
        matchingSettings.insert(QStringLiteral("pair_source"), QStringLiteral("vocabulary_overlap"));
        matchingSettings.insert(QStringLiteral("feature_suffix"), settings.value(QStringLiteral("feature_suffix")));
        matchingSettings.insert(QStringLiteral("lis_file"), settings.value(QStringLiteral("output_lis")));
        matchingSettings.insert(QStringLiteral("overlap_lis"), settings.value(QStringLiteral("output_lis")));
        matchingSettings.insert(QStringLiteral("overlap_json"), settings.value(QStringLiteral("output_json")));
        matchStore.save(matchingSettings);

        LOG_INFO(QStringLiteral("词汇重叠对已应用到特征匹配设置：%1 对").arg(pairs.size()));
    });

    dlg->exec();
}

void MenuWorkflowController::openFeaturePointVisualizationDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    // 收集当前可用的特征文件后缀
    QStringList availableSuffixes;
    QString currentSuffix;
    auto *mainWin = qobject_cast<MainWindow*>(_mainWindow.data());
    auto *canvas = mainWin ? mainWin->canvas() : nullptr;
    if (canvas)
    {
        availableSuffixes = canvas->availableFeatureSuffixes();
        currentSuffix = canvas->activeFeatureSuffix();
    }
    if (availableSuffixes.isEmpty())
        availableSuffixes << QStringLiteral(".sp") << QStringLiteral(".dsk") << QStringLiteral(".alk")
                           << QStringLiteral(".sift") << QStringLiteral(".orb") << QStringLiteral(".akz");

    QJsonObject sv;
    if (_projectManager)
    {
        if (!_featurePointVisualizationSetting)
        {
            _featurePointVisualizationSetting =
                new DialogSettingStore(DialogSettingKeys::FeaturePointVisualization, this);
        }
        _featurePointVisualizationSetting->setProjectPath(_projectManager->currentProjectPath());
        sv = _featurePointVisualizationSetting->load();

        const QString savedSuffix = sv.value(QStringLiteral("feature_suffix")).toString().trimmed();
        if (!savedSuffix.isEmpty())
        {
            currentSuffix = savedSuffix;
        }
    }

    auto *dlg = new FeaturePointVisualizationDialog(availableSuffixes, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    if (!currentSuffix.isEmpty())
    {
        dlg->setCurrentSuffix(currentSuffix);
    }
    const QString effectiveSuffix = dlg->currentSuffix();
    if (canvas && !effectiveSuffix.isEmpty())
    {
        canvas->setActiveFeatureSuffix(effectiveSuffix);
    }

    // 切换特征文件后缀 → CanvasWidget 重新加载
    if (canvas)
    {
        connect(dlg, &FeaturePointVisualizationDialog::featureSuffixChanged,
                this, [this, canvas](const QString &suffix)
        {
            canvas->setActiveFeatureSuffix(suffix);
            if (_featurePointVisualizationSetting)
            {
                QJsonObject sv = _featurePointVisualizationSetting->load();
                sv[QStringLiteral("feature_suffix")] = suffix;
                _featurePointVisualizationSetting->save(sv);
            }
        });
    }

    // 懒初始化可视化记忆化管理器并加载保存的设置
    if (_projectManager)
    {
        if (!sv.isEmpty())
        {
            LayerRenderer::FeatureDisplayOptions opts;
            opts.showPoints = sv.value("showPoints").toBool(opts.showPoints);
            opts.showScale = sv.value("showScale").toBool(opts.showScale);
            opts.showOrientation = sv.value("showOrientation").toBool(opts.showOrientation);
            opts.showResiduals = sv.value("showResiduals").toBool(opts.showResiduals);
            opts.residualScale = sv.value("residualScale").toDouble(opts.residualScale);
            opts.minimumResidualPx = sv.value("minimumResidualPx").toDouble(opts.minimumResidualPx);
            opts.maximumResidualLengthPx =
                sv.value("maximumResidualLengthPx").toDouble(opts.maximumResidualLengthPx);
            opts.useFill = sv.value("useFill").toBool(opts.useFill);
            opts.pointSize = sv.value("pointSize").toInt(opts.pointSize);
            opts.scaleMultiplier = sv.value("scaleMultiplier").toDouble(opts.scaleMultiplier);
            opts.opacity = sv.value("opacity").toInt(opts.opacity);
            opts.markerShape = sv.value("markerShape").toString(opts.markerShape);
            opts.maxDisplayCount = sv.value("maxDisplayCount").toInt(opts.maxDisplayCount);
            opts.showTopScores = sv.value("showTopScores").toBool(opts.showTopScores);

            QJsonObject pc = sv.value("pointColor").toObject();
            if (!pc.isEmpty())
            {
                opts.pointColor = QColor(pc["r"].toInt(), pc["g"].toInt(), pc["b"].toInt());
            }
            QJsonObject sc = sv.value("scaleColor").toObject();
            if (!sc.isEmpty())
            {
                opts.scaleColor = QColor(sc["r"].toInt(), sc["g"].toInt(), sc["b"].toInt());
            }
            QJsonObject oc = sv.value("orientColor").toObject();
            if (!oc.isEmpty())
            {
                opts.orientColor = QColor(oc["r"].toInt(), oc["g"].toInt(), oc["b"].toInt());
            }
            QJsonObject rc = sv.value("residualColor").toObject();
            if (!rc.isEmpty())
            {
                opts.residualColor = QColor(rc["r"].toInt(), rc["g"].toInt(), rc["b"].toInt());
            }

            dlg->setDisplayOptions(opts);
        }
    }

    if (canvas)
    {
        LayerRenderer::FeatureDisplayOptions currentOptions = dlg->getDisplayOptions();
        currentOptions.showPoints = canvas->showsInterestPoints();
        currentOptions.showResiduals = canvas->showsFeatureResiduals();
        dlg->setDisplayOptions(currentOptions);
    }

    // 连接实时更新信号
    connect(dlg, &FeaturePointVisualizationDialog::displayOptionsChanged, this,
        [this, dlg](const LayerRenderer::FeatureDisplayOptions &opts)
        {
            // 发送信号给MainWindow应用到CanvasWidget
            emit requestApplyFeatureDisplayOptions(opts);

            // 保存到 project_dialog.json
            if (_featurePointVisualizationSetting)
            {
                QJsonObject sv;
                sv["showPoints"] = opts.showPoints;
                sv["showScale"] = opts.showScale;
                sv["showOrientation"] = opts.showOrientation;
                sv["showResiduals"] = opts.showResiduals;
                sv["residualScale"] = opts.residualScale;
                sv["minimumResidualPx"] = opts.minimumResidualPx;
                sv["maximumResidualLengthPx"] = opts.maximumResidualLengthPx;
                sv["useFill"] = opts.useFill;
                sv["pointSize"] = opts.pointSize;
                sv["scaleMultiplier"] = opts.scaleMultiplier;
                sv["opacity"] = opts.opacity;
                sv["markerShape"] = opts.markerShape;
                sv["maxDisplayCount"] = opts.maxDisplayCount;
                sv["showTopScores"] = opts.showTopScores;
                sv[QStringLiteral("feature_suffix")] = dlg->currentSuffix();
                sv["pointColor"] = colorToJson(opts.pointColor);
                sv["scaleColor"] = colorToJson(opts.scaleColor);
                sv["orientColor"] = colorToJson(opts.orientColor);
                sv["residualColor"] = colorToJson(opts.residualColor);

                _featurePointVisualizationSetting->save(sv);
            }
        });

    dlg->show();
}

void MenuWorkflowController::applySavedFeatureDisplayOptions(const QJsonObject &ui)
{
    if (!_projectManager)
    {
        return;
    }

    // 优先从 project_dialog.json 加载
    if (!_featurePointVisualizationSetting)
    {
        _featurePointVisualizationSetting =
            new DialogSettingStore(DialogSettingKeys::FeaturePointVisualization, this);
    }
    _featurePointVisualizationSetting->setProjectPath(_projectManager->currentProjectPath());
    QJsonObject sv = _featurePointVisualizationSetting->load();

    // 兼容旧版本：若新文件中无数据则尝试从传入的旧 ui 设置中读取
    if (sv.isEmpty() && ui.contains(QStringLiteral("superpoint_visualization")))
    {
        sv = ui.value(QStringLiteral("superpoint_visualization")).toObject();
    }
    auto *mainWin = qobject_cast<MainWindow*>(_mainWindow.data());
    auto *canvas = mainWin ? mainWin->canvas() : nullptr;
    const QString savedSuffix = sv.value(QStringLiteral("feature_suffix")).toString().trimmed();
    if (canvas)
    {
        const QString inferredSuffix = xjw::common::project::inferPreferredFeatureSuffix(
            _projectManager->currentProjectPath(), _projectManager->currentMeta());
        const bool savedSuffixUsable = !savedSuffix.isEmpty()
            && (inferredSuffix.isEmpty()
                || xjw::common::project::projectHasFeatureSuffix(
                    _projectManager->currentProjectPath(), _projectManager->currentMeta(), savedSuffix));
        if (savedSuffixUsable)
        {
            canvas->setActiveFeatureSuffix(savedSuffix);
        }
        else if (!inferredSuffix.isEmpty())
        {
            canvas->setActiveFeatureSuffix(inferredSuffix);
        }
    }

    if (sv.isEmpty())
    {
        return;
    }

    LayerRenderer::FeatureDisplayOptions opts;
    opts.showPoints = sv.value("showPoints").toBool(opts.showPoints);
    opts.showScale = sv.value("showScale").toBool(opts.showScale);
    opts.showOrientation = sv.value("showOrientation").toBool(opts.showOrientation);
    opts.showResiduals = sv.value("showResiduals").toBool(opts.showResiduals);
    opts.residualScale = sv.value("residualScale").toDouble(opts.residualScale);
    opts.minimumResidualPx = sv.value("minimumResidualPx").toDouble(opts.minimumResidualPx);
    opts.maximumResidualLengthPx =
        sv.value("maximumResidualLengthPx").toDouble(opts.maximumResidualLengthPx);
    opts.useFill = sv.value("useFill").toBool(opts.useFill);
    opts.pointSize = sv.value("pointSize").toInt(opts.pointSize);
    opts.scaleMultiplier = sv.value("scaleMultiplier").toDouble(opts.scaleMultiplier);
    opts.opacity = sv.value("opacity").toInt(opts.opacity);
    opts.markerShape = sv.value("markerShape").toString(opts.markerShape);
    opts.maxDisplayCount = sv.value("maxDisplayCount").toInt(opts.maxDisplayCount);
    opts.showTopScores = sv.value("showTopScores").toBool(opts.showTopScores);
    QJsonObject pc = sv.value("pointColor").toObject();
    if (!pc.isEmpty())
    {
        opts.pointColor = QColor(pc["r"].toInt(), pc["g"].toInt(), pc["b"].toInt());
    }
    QJsonObject sc = sv.value("scaleColor").toObject();
    if (!sc.isEmpty())
    {
        opts.scaleColor = QColor(sc["r"].toInt(), sc["g"].toInt(), sc["b"].toInt());
    }
    QJsonObject oc = sv.value("orientColor").toObject();
    if (!oc.isEmpty())
    {
        opts.orientColor = QColor(oc["r"].toInt(), oc["g"].toInt(), oc["b"].toInt());
    }
    QJsonObject rc = sv.value("residualColor").toObject();
    if (!rc.isEmpty())
    {
        opts.residualColor = QColor(rc["r"].toInt(), rc["g"].toInt(), rc["b"].toInt());
    }

    emit requestApplyFeatureDisplayOptions(opts);
}

void MenuWorkflowController::openThreeDReconstructionDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dlg = new ThreeDReconstructionDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    const QStringList images = getProjectImages();
    dlg->setImageCount(images.size());

    if (!_threeDSetting)
    {
        _threeDSetting = new DialogSettingStore(DialogSettingKeys::ThreeDReconstruction, this);
    }

    if (_projectManager)
    {
        const QString projectPath = _projectManager->currentProjectPath();
        const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("three_d_reconstruction")));
        }
        _threeDSetting->setProjectPath(projectPath);
        dlg->applySettings(_threeDSetting->load());
    }

    connect(dlg, &ThreeDReconstructionDialog::settingsChanged, this, [this](const QJsonObject &settings) {
        if (_threeDSetting)
        {
            _threeDSetting->save(settings);
        }
    });
    connect(dlg, &ThreeDReconstructionDialog::runRequested, this, [this](const QJsonObject &settings) {
        if (_threeDSetting)
        {
            _threeDSetting->save(settings);
        }
        startThreeDReconstructionWorkflow(settings);
    }, Qt::QueuedConnection);

    dlg->exec();
}

void MenuWorkflowController::openAerialTriangulationDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    AerialTriangulationDialog dlg(_mainWindow);

    const QStringList images = getProjectImages();
    dlg.setImageCount(images.size());
    bool hasAllReferenceCameras = false;
    const int cameraCount = _projectManager
        ? _projectManager->getCamerasForImages(images, &hasAllReferenceCameras).size()
        : 0;
    dlg.setReferencePreselectionAvailable(
        hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2,
        cameraCount,
        images.size());

    if (!_aerialTriangulationSetting)
    {
        _aerialTriangulationSetting = new DialogSettingStore(DialogSettingKeys::AerialTriangulation, this);
    }

    if (_projectManager)
    {
        const QString projectPath = _projectManager->currentProjectPath();
        _aerialTriangulationSetting->setProjectPath(projectPath);
        dlg.applySettings(_aerialTriangulationSetting->load());
    }

    connect(&dlg, &AerialTriangulationDialog::settingsChanged, this, [this](const QJsonObject &settings)
    {
        if (_aerialTriangulationSetting)
        {
            _aerialTriangulationSetting->save(settings);
        }
    });

    if (dlg.exec() == QDialog::Accepted)
    {
        const QJsonObject settings = dlg.collectSettings();
        if (_aerialTriangulationSetting)
        {
            _aerialTriangulationSetting->save(settings);
        }
        startAerialTriangulationWorkflow(settings);
    }
}

void MenuWorkflowController::openWorkflowAerialTriangulationDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    AerialTriangulationDialog dlg(_mainWindow);

    const QStringList images = getProjectImages();
    dlg.setImageCount(images.size());
    bool hasAllReferenceCameras = false;
    const int cameraCount = _projectManager
        ? _projectManager->getCamerasForImages(images, &hasAllReferenceCameras).size()
        : 0;
    dlg.setReferencePreselectionAvailable(
        hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2,
        cameraCount,
        images.size());

    if (!_aerialTriangulationSetting)
    {
        _aerialTriangulationSetting = new DialogSettingStore(DialogSettingKeys::AerialTriangulation, this);
    }

    if (_projectManager)
    {
        const QString projectPath = _projectManager->currentProjectPath();
        _aerialTriangulationSetting->setProjectPath(projectPath);
        dlg.applySettings(_aerialTriangulationSetting->load());
    }

    connect(&dlg, &AerialTriangulationDialog::settingsChanged, this, [this](const QJsonObject &settings) {
        if (_aerialTriangulationSetting)
        {
            _aerialTriangulationSetting->save(settings);
        }
    });

    if (dlg.exec() == QDialog::Accepted)
    {
        const QJsonObject settings = dlg.collectSettings();
        if (_aerialTriangulationSetting)
        {
            _aerialTriangulationSetting->save(settings);
        }
        startAerialTriangulationWorkflow(settings);
    }
}

QJsonObject MenuWorkflowController::sanitizeAerialTriangulationReferencePreselection(
    const QJsonObject &requestedSettings,
    const QStringList &images) const
{
    QJsonObject settings = requestedSettings;
    if (!settings.value(QStringLiteral("reference_preselection")).toBool(false))
    {
        return settings;
    }
    if (isSequenceReferencePreselection(settings))
    {
        return settings;
    }

    bool hasAllReferenceCameras = false;
    const int cameraCount = _projectManager
        ? _projectManager->getCamerasForImages(images, &hasAllReferenceCameras).size()
        : 0;
    const bool available =
        hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2;
    if (!available)
    {
        settings[QStringLiteral("reference_preselection")] = false;
        LOG_WARN(QStringLiteral("空中三角测量: 项目相机文件不完整，参考预选已关闭（相机 %1/%2）。")
                     .arg(cameraCount)
                     .arg(images.size()));
    }
    return settings;
}

void MenuWorkflowController::startAerialTriangulationWorkflow(const QJsonObject &settings)
{
    if (!_projectManager)
    {
        QMessageBox::warning(_mainWindow, QStringLiteral("空中三角测量"), QStringLiteral("请先打开项目"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(_mainWindow,
                             QStringLiteral("空中三角测量"),
                             QStringLiteral("至少需要 2 张影像才能进行空中三角测量。"));
        return;
    }

    const QJsonObject projectMeta = _projectManager->currentMeta();
    const bool hasDepthMaps = !projectMeta.value(QStringLiteral("depth_map_results")).toArray().isEmpty();
    QSettings warningSettings(QStringLiteral("PlaScan"), QStringLiteral("plascan_gui"));
    const bool suppressDepthInvalidationWarning = warningSettings.value(
        QStringLiteral("Warnings/suppressDepthMapInvalidationBeforeAerialTriangulation"),
        false).toBool();
    if (hasDepthMaps && !suppressDepthInvalidationWarning)
    {
        QMessageBox confirmation(_mainWindow);
        confirmation.setWindowTitle(QStringLiteral("空中三角测量"));
        confirmation.setIcon(QMessageBox::Warning);
        confirmation.setText(
            QStringLiteral("当前深度图将在空中三角测量成功后失效并从项目中移除。是否继续？"));
        confirmation.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        confirmation.setDefaultButton(QMessageBox::No);
        auto *dontShowAgain = new QCheckBox(QStringLiteral("不再显示该信息"), &confirmation);
        confirmation.setCheckBox(dontShowAgain);
        const auto answer = static_cast<QMessageBox::StandardButton>(confirmation.exec());
        if (answer != QMessageBox::Yes)
        {
            return;
        }
        if (dontShowAgain->isChecked())
        {
            warningSettings.setValue(
                QStringLiteral("Warnings/suppressDepthMapInvalidationBeforeAerialTriangulation"),
                true);
        }
    }

    auto *pm = _projectManager;
    const QString projectPath = pm->currentProjectPath();
    QString outputRoot = settings.value(QStringLiteral("output_dir")).toString().trimmed();
    if (outputRoot.isEmpty())
    {
        const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
        outputRoot = QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation"));
    }
    outputRoot = QDir::cleanPath(outputRoot);
    QDir().mkpath(outputRoot);

    QJsonObject runSettings = settings;
    runSettings[QStringLiteral("output_dir")] = outputRoot;
    runSettings = sanitizeAerialTriangulationReferencePreselection(runSettings, images);
    const QString selectedFeatureAlgorithm =
        runSettings.value(QStringLiteral("feature_algorithm")).toString(QStringLiteral("sift"));
    const QString selectedMatchAlgorithm =
        runSettings.value(QStringLiteral("match_algorithm")).toString(QStringLiteral("lightglue"));

    emit pm->atProgressChanged(QStringLiteral("空中三角测量: 检查上游数据..."), 0);

    QPointer<ProjectManager> pmGuard(pm);
    xjw::gui::tasks::runGuarded(
        this,
        [images, projectMeta, projectPath, selectedFeatureAlgorithm, selectedMatchAlgorithm]()
        {
            return MenuWorkflowController::summarizeSparsePrerequisites(images,
                                                                        projectMeta,
                                                                        projectPath,
                                                                        selectedFeatureAlgorithm,
                                                                        selectedMatchAlgorithm);
        },
        [pmGuard, runSettings, images, projectPath, projectMeta, outputRoot](
            MenuWorkflowController *controller, const SparsePrerequisiteSummary &prereq)
        {
            if (!pmGuard)
            {
                return;
            }
            if (pmGuard->currentProjectPath() != projectPath)
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral("项目已切换，本次空三启动已取消。"));
                return;
            }

            if (!prereq.prerequisiteReport.isEmpty())
            {
                LOG_INFO(QStringLiteral("空三前置报告: %1")
                             .arg(QString::fromUtf8(
                                 QJsonDocument(prereq.prerequisiteReport).toJson(QJsonDocument::Compact))));
            }

            bool autoFillMissing = false;
            const bool reuseExistingMatches =
                runSettings.value(QStringLiteral("reuse_existing_matches")).toBool(true);
            if (prereq.blockOnMatchQuality && reuseExistingMatches)
            {
                emit pmGuard->atProgressFinished(false);
                const QString details = prereq.warningMessages.isEmpty()
                    ? QStringLiteral("匹配阶段已完成，但没有可用于空三的连接边；请检查匹配参数、重叠对和几何验证报告。")
                    : prereq.warningMessages.join(QStringLiteral("\n"));
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral(
                                         "%1\n\n当前已选择“重用现有匹配”，不会自动重新跑完整匹配。"
                                         "如需重建匹配，请在高级设置中取消该选项。")
                                         .arg(details));
                return;
            }
            if (!prereq.missingMessages.isEmpty())
            {
                autoFillMissing = true;
                LOG_INFO(QStringLiteral("空中三角测量: 检测到缺少连接点输入，直接运行创建连接点流程；该流程会自动提取特征并匹配。"));
            }

            controller->runUnifiedAerialTriangulation(runSettings,
                                                      images,
                                                      projectPath,
                                                      projectMeta,
                                                      outputRoot,
                                                      autoFillMissing);
        });
}

void MenuWorkflowController::runUnifiedAerialTriangulation(const QJsonObject &settings,
                                                           const QStringList &images,
                                                           const QString &projectPath,
                                                           const QJsonObject &projectMeta,
                                                           const QString &outputRoot,
                                                           bool fillMissingTiePoints)
{
    if (!_projectManager)
    {
        return;
    }

    auto *pm = _projectManager;
    if (pm->currentProjectPath() != projectPath)
    {
        emit pm->atProgressFinished(false);
        QMessageBox::warning(_mainWindow,
                             QStringLiteral("空中三角测量"),
                             QStringLiteral("项目已切换，本次空三启动已取消。"));
        return;
    }

    const bool resetCurrentAlignment =
        settings.value(QStringLiteral("reset_current_alignment")).toBool(true);

    const int workflowThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    xjw::aerial_triangulation::AerialTriangulationOptions workflowOptions;
    workflowOptions.images = images;
    workflowOptions.projectPath = projectPath;
    workflowOptions.projectMeta = projectMeta;
    workflowOptions.outputDir = outputRoot;
    workflowOptions.quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("high"));
    workflowOptions.genericPreselection = settings.value(QStringLiteral("generic_preselection")).toBool(true);
    workflowOptions.referencePreselection = settings.value(QStringLiteral("reference_preselection")).toBool(false);
    workflowOptions.referenceMode =
        settings.value(QStringLiteral("reference_preselection_source")).toString(QStringLiteral("source_code"));
    workflowOptions.resetAlignment = settings.value(QStringLiteral("reset_current_alignment")).toBool(true);
    workflowOptions.reuseExistingMatches =
        settings.value(QStringLiteral("reuse_existing_matches")).toBool(true);
    workflowOptions.lockInputCameraPoses =
        settings.value(QStringLiteral("lock_input_camera_poses")).toBool(false);
    workflowOptions.saveAfterEachStep = settings.value(QStringLiteral("save_project_after_each_step")).toBool(false);
    workflowOptions.keypointLimit = settings.value(QStringLiteral("keypoint_limit")).toInt(40000);
    workflowOptions.tiepointLimit = settings.value(QStringLiteral("tiepoint_limit")).toInt(4000);
    workflowOptions.maskApplyMode =
        settings.value(QStringLiteral("mask_apply_mode")).toString(QStringLiteral("keypoints"));
    workflowOptions.excludeFixedTiePoints = settings.value(QStringLiteral("exclude_fixed_tie_points")).toBool(true);
    workflowOptions.guidedImageMatching = settings.value(QStringLiteral("guided_image_matching")).toBool(false);
    workflowOptions.adaptiveCameraModelFitting =
        settings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool(true);
    workflowOptions.featureAlgorithm = settings.value(QStringLiteral("feature_algorithm"))
                                           .toString(QStringLiteral("sift"))
                                           .trimmed()
                                           .toLower();
    workflowOptions.matchAlgorithm = settings.value(QStringLiteral("match_algorithm"))
                                         .toString(QStringLiteral("lightglue"))
                                         .trimmed()
                                         .toLower();
    workflowOptions.matchPipeline = settings.value(QStringLiteral("match_pipeline")).toString().trimmed().toLower();
    workflowOptions.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    workflowOptions.threads = workflowThreads;
    workflowOptions.autoGenerateMissingMatches = fillMissingTiePoints;
    workflowOptions.assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    workflowOptions.featureDir = xjw::common::project::ProjectIO::ipfindOutputDir(projectPath);
    workflowOptions.matchDir = xjw::common::project::ProjectIO::ipmatchOutputDir(projectPath);
    workflowOptions.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(projectPath, images);
    workflowOptions.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    workflowOptions.featureGrayscaleMax = 1.0f;

    if (workflowOptions.referencePreselection &&
        workflowOptions.referenceMode.trimmed().toLower() != QStringLiteral("sequence"))
    {
        bool hasAllReferenceCameras = false;
        workflowOptions.referenceCameras = pm->getCamerasForImages(images, &hasAllReferenceCameras);
        if (!hasAllReferenceCameras)
        {
            LOG_WARN(QStringLiteral("空中三角测量: 参考预选已启用，但项目相机参考不完整"));
        }
    }

    bool usedStoredPairs = false;
    bool storedPairsStale = false;
    const bool useStoredGeneratedPairs = shouldUseStoredGeneratedPairConstraints(settings);
    const QStringList allowedPairs = useStoredGeneratedPairs
        ? loadGeneratedPairConstraints(projectPath,
                                       projectMeta,
                                       images,
                                       &usedStoredPairs,
                                       &storedPairsStale)
        : QStringList();
    if (!allowedPairs.isEmpty())
    {
        workflowOptions.restrictPairs = true;
        workflowOptions.allowedPairs = allowedPairs;
        LOG_INFO(QStringLiteral("空中三角测量: 使用已生成候选配对约束 %1 对").arg(allowedPairs.size()));
    }
    else if (!useStoredGeneratedPairs)
    {
        LOG_INFO(QStringLiteral("空中三角测量: 照片序列预选已启用，SfM 跳过历史候选配对约束"));
    }
    else if (storedPairsStale)
    {
        LOG_WARN(QStringLiteral("空中三角测量: 已生成候选配对与当前影像集合不一致，改用自动配对规划"));
    }

    QPointer<ProjectManager> pmGuard(pm);
    workflowOptions.progressFn = [pmGuard](const QString &stage, int percent)
    {
        if (!pmGuard)
        {
            return;
        }
        xjw::gui::tasks::postGuarded(pmGuard.data(), [stage, percent](ProjectManager *manager)
        {
            emit manager->atProgressChanged(QStringLiteral("空中三角测量: %1").arg(stage), percent);
        });
    };
    workflowOptions.pairMatchedFn = [pmGuard](const QString &img0,
                                              const QString &img1,
                                              const QString &matchPath,
                                              int numMatches)
    {
        if (!pmGuard)
        {
            return;
        }
        xjw::gui::tasks::postGuarded(pmGuard.data(),
                                     [img0, img1, matchPath, numMatches](ProjectManager *manager)
        {
            emit manager->matchPairReady(img0, img1, matchPath, numMatches);
        });
    };

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    workflowOptions.cancelFlag = cancelFlag;
    const xjw::aerial_triangulation::AerialTriangulationResolvedConfig resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(workflowOptions);

    emit pm->atProgressChanged(resolved.prepareTiePoints
                                    ? QStringLiteral("空中三角测量: 准备连接点...")
                                    : QStringLiteral("空中三角测量: 启动 SfM/BA..."),
                                0);

    const QStringList sfmImages = images;
    const QString sfmOutputDir = resolved.pipelineInput.outputDir;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    xjw::gui::tasks::runGuarded(
        this,
        [runWorkflowOptions = std::move(workflowOptions), sfmImages, sfmOutputDir, assetsDir]() mutable
        {
            xjw::aerial_triangulation::AerialTriangulationResult workflowResult =
                xjw::aerial_triangulation::AerialTriangulationWorkflow::run(runWorkflowOptions);
            const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result = workflowResult.reconstructionResult;
            if (result.success && !assetsDir.isEmpty())
            {
                QJsonObject report;
                report[QStringLiteral("type")] = QStringLiteral("aerial_triangulation_sfm");
                report[QStringLiteral("mode")] = QStringLiteral("sfm");
                report[QStringLiteral("source")] = QStringLiteral("workflow_aerial_triangulation");
                report[QStringLiteral("timestamp")] =
                    QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
                report[QStringLiteral("num_images")] = sfmImages.size();
                report[QStringLiteral("num_registered")] = result.numRegisteredImages;
                report[QStringLiteral("num_points_3d")] = result.numPoints3D;
                report[QStringLiteral("mean_reproj_error_px")] = result.meanReprojError;
                report[QStringLiteral("ba_rms_before")] = result.baRmsBefore;
                report[QStringLiteral("ba_rms_after")] = result.baRmsAfter;
                report[QStringLiteral("ba_tracks_total")] = result.baTracksTotal;
                report[QStringLiteral("ba_tracks_optimized")] = result.baTracksOptimized;
                report[QStringLiteral("ba_tracks_filtered")] = result.baTracksFiltered;
                report[QStringLiteral("duration_s")] = result.durationSeconds;
                report[QStringLiteral("output_dir")] = sfmOutputDir;
                report[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
                report[QStringLiteral("per_camera")] = result.perCameraResiduals;
                report[QStringLiteral("sfm_diagnostics")] = result.sfmDiagnostics;
                report[QStringLiteral("resolved_settings")] = workflowResult.config.resolvedSettings;
                report[QStringLiteral("tie_point_preparation_executed")] =
                    workflowResult.tiePointPreparationExecuted;
                report[QStringLiteral("tie_point_track_count")] =
                    workflowResult.tiePointResult.trackCount;
                writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                  QStringLiteral("at_report.json"),
                                                  QStringLiteral("at_report_history.json"),
                                                  report);
                writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                  QStringLiteral("aerial_triangulation_sfm_report.json"),
                                                  QStringLiteral("aerial_triangulation_sfm_report_history.json"),
                                                  report);
            }
            return workflowResult;
        },
        [pmGuard,
         cancelFlag,
         sfmImages,
         sfmOutputDir,
         projectPath,
         resetCurrentAlignment](MenuWorkflowController *controller,
                                xjw::aerial_triangulation::AerialTriangulationResult workflowResult) mutable {
            if (!pmGuard)
            {
                return;
            }
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result = workflowResult.reconstructionResult;
            pmGuard->clearAtCancelFlag(cancelFlag);
            if (pmGuard->currentProjectPath() != projectPath)
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral("项目已切换，本次空三结果未写回。"));
                return;
            }

            const bool wasCanceled = cancelFlag->load(std::memory_order_relaxed);
            if (wasCanceled)
            {
                emit pmGuard->atProgressFinished(false);
                return;
            }

            if (workflowResult.tiePointPreparationExecuted)
            {
                for (const xjw::matchphotos::MatchPhotosFeatureRecord &feature :
                     workflowResult.tiePointResult.features)
                {
                    pmGuard->appendIpfindResult(feature.imagePath,
                                                feature.featurePath,
                                                feature.settings);
                }
                for (const xjw::matchphotos::MatchPhotosMatchRecord &match :
                     workflowResult.tiePointResult.matches)
                {
                    pmGuard->appendIpmatchResult(QStringList{match.matchPath}, match.settings);
                }
            }

            if (!result.success)
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     result.errorMessage.isEmpty()
                                         ? QStringLiteral("空中三角测量失败。")
                                         : result.errorMessage);
                return;
            }

            int registeredImageCount = result.numRegisteredImages;
            QJsonObject resultRecordExtra = result.resultRecordExtra;
            QString sparseBlockingReason = QStringLiteral("SFM 未生成可用的正式稀疏点云。");
            QStringList registeredImages;
            if (result.success && !result.sparseCloudPath.isEmpty())
            {
                const QStringList registeredCameraKeys = result.pendingCamUpdates.keys();
                registeredImages.reserve(sfmImages.size());
                for (const QString &imagePath : sfmImages)
                {
                    const QString normalized = xjw::common::project::normalizePath(imagePath);
                    if (registeredCameraKeys.contains(normalized))
                    {
                        registeredImages.append(normalized);
                    }
                }
                if (registeredImages.size() < registeredCameraKeys.size())
                {
                    for (const QString &imagePath : registeredCameraKeys)
                    {
                        if (!registeredImages.contains(imagePath))
                        {
                            registeredImages.append(imagePath);
                        }
                    }
                }
                registeredImageCount = registeredImages.size();

                resultRecordExtra[QStringLiteral("source")] = QStringLiteral("aerial_triangulation");
                sparseBlockingReason = xjw::gui::project::sparseResultBlockingReason(resultRecordExtra);
            }

            if (!xjw::gui::project::isProductionSparseResult(resultRecordExtra))
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     sparseBlockingReason.isEmpty()
                                         ? QStringLiteral("当前 SfM/BA 稀疏点云质量不足。")
                                         : sparseBlockingReason);
                return;
            }

            // 只有通过 MVS 入口质量门控的正式 SfM 结果才写回工程，避免失败候选污染相机状态。
            if (resetCurrentAlignment)
            {
                int updated = 0;
                int cleared = 0;
                QString err;
                if (!pmGuard->replaceImageCameras(sfmImages,
                                                  result.pendingCamUpdates,
                                                  &updated,
                                                  &cleared,
                                                  &err))
                {
                    LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
                }
                else
                {
                    LOG_INFO(QStringLiteral("空中三角测量: 相机对齐状态已刷新，注册 %1，清除旧位姿 %2")
                                 .arg(updated)
                                 .arg(cleared));
                }
            }
            else if (!result.pendingCamUpdates.isEmpty())
            {
                int updated = 0;
                QString err;
                if (!pmGuard->setImageCameras(result.pendingCamUpdates, &updated, &err))
                {
                    LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
                }
            }

            if (!pmGuard->replaceTiePointResult(result.sparseCloudPath,
                                                result.numPoints3D,
                                                registeredImages,
                                                sfmOutputDir,
                                                resultRecordExtra))
            {
                emit pmGuard->atProgressFinished(false);
                return;
            }

            emit pmGuard->atProgressFinished(true);
            QMessageBox::information(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral("正式 SfM/BA 稀疏云已生成。\n注册影像: %1\n点数: %2\n路径: %3")
                                         .arg(registeredImageCount)
                                         .arg(result.numPoints3D)
                                         .arg(result.sparseCloudPath));
        });
}

void MenuWorkflowController::startThreeDReconstructionWorkflow(const QJsonObject &settings)
{
    if (!_projectManager)
    {
        QMessageBox::warning(_mainWindow, QStringLiteral("三维重建"), QStringLiteral("请先打开项目"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(_mainWindow,
                             QStringLiteral("三维重建"),
                             QStringLiteral("至少需要 2 张影像才能进行三维重建。"));
        return;
    }

    QString outputRoot = settings.value(QStringLiteral("output_dir")).toString().trimmed();
    if (outputRoot.isEmpty())
    {
        const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
        outputRoot = QDir(assetsDir).filePath(QStringLiteral("three_d_reconstruction"));
    }
    outputRoot = QDir::cleanPath(outputRoot);
    QDir().mkpath(outputRoot);

    QJsonObject runSettings = settings;
    runSettings[QStringLiteral("output_dir")] = outputRoot;

    auto *pm = _projectManager;
    xjw::aerial_triangulation::AerialTriangulationOptions workflowOptions;
    workflowOptions.images = images;
    workflowOptions.projectPath = pm->currentProjectPath();
    workflowOptions.projectMeta = pm->coreProjectMeta();
    workflowOptions.outputDir = outputRoot;
    const int workflowThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    workflowOptions.threads = workflowThreads;
    workflowOptions.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    workflowOptions.featureAlgorithm = settings.value(QStringLiteral("feature_algorithm"))
                                 .toString(QStringLiteral("disk"))
                                 .trimmed()
                                 .toLower();
    workflowOptions.matchAlgorithm = settings.value(QStringLiteral("match_algorithm"))
                              .toString(QStringLiteral("lightglue"))
                              .trimmed()
                              .toLower();
    workflowOptions.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    workflowOptions.featureGrayscaleMax = 1.0f;
    workflowOptions.resetAlignment = settings.value(QStringLiteral("reset_current_alignment")).toBool(true);
    workflowOptions.reuseExistingMatches =
        settings.value(QStringLiteral("reuse_existing_matches")).toBool(true);
    workflowOptions.lockInputCameraPoses =
        settings.value(QStringLiteral("lock_input_camera_poses")).toBool(false);
    workflowOptions.autoGenerateMissingMatches = true;

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    workflowOptions.quality = quality;

    QPointer<ProjectManager> pmGuard(pm);
    workflowOptions.progressFn = [pmGuard](const QString &stage, int percent) {
        if (!pmGuard)
        {
            return;
        }
        xjw::gui::tasks::postGuarded(pmGuard.data(), [stage, percent](ProjectManager *manager) {
            emit manager->atProgressChanged(QStringLiteral("三维重建/空三: %1").arg(stage), percent);
        });
    };
    workflowOptions.pairMatchedFn = [pmGuard](const QString &img0,
                                   const QString &img1,
                                   const QString &matchPath,
                                   int numMatches) {
        if (!pmGuard)
        {
            return;
        }
        xjw::gui::tasks::postGuarded(pmGuard.data(),
                                     [img0, img1, matchPath, numMatches](ProjectManager *manager) {
            emit manager->matchPairReady(img0, img1, matchPath, numMatches);
        });
    };

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    workflowOptions.cancelFlag = cancelFlag;

    emit pm->atProgressChanged(QStringLiteral("三维重建: 启动空中三角测量..."), 0);

    const QStringList sfmImages = images;
    const QString sfmOutputDir =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(workflowOptions)
            .pipelineInput.outputDir;
    const QString projectPath = pm->currentProjectPath();
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    xjw::gui::tasks::runGuarded(
        this,
        [runWorkflowOptions = std::move(workflowOptions), sfmImages, sfmOutputDir, assetsDir]() mutable {
            xjw::aerial_triangulation::AerialTriangulationResult workflowResult =
                xjw::aerial_triangulation::AerialTriangulationWorkflow::run(runWorkflowOptions);
            const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result =
                workflowResult.reconstructionResult;

            if (result.success && !assetsDir.isEmpty())
            {
                QJsonObject report;
                report[QStringLiteral("type")] = QStringLiteral("three_d_reconstruction_sfm");
                report[QStringLiteral("mode")] = QStringLiteral("sfm");
                report[QStringLiteral("source")] = QStringLiteral("three_d_reconstruction_sfm");
                report[QStringLiteral("timestamp")] =
                    QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
                report[QStringLiteral("num_images")] = sfmImages.size();
                report[QStringLiteral("num_registered")] = result.numRegisteredImages;
                report[QStringLiteral("num_points_3d")] = result.numPoints3D;
                report[QStringLiteral("mean_reproj_error_px")] = result.meanReprojError;
                report[QStringLiteral("ba_rms_before")] = result.baRmsBefore;
                report[QStringLiteral("ba_rms_after")] = result.baRmsAfter;
                report[QStringLiteral("ba_tracks_total")] = result.baTracksTotal;
                report[QStringLiteral("ba_tracks_optimized")] = result.baTracksOptimized;
                report[QStringLiteral("ba_tracks_filtered")] = result.baTracksFiltered;
                report[QStringLiteral("duration_s")] = result.durationSeconds;
                report[QStringLiteral("output_dir")] = sfmOutputDir;
                report[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
                report[QStringLiteral("per_camera")] = result.perCameraResiduals;
                report[QStringLiteral("sfm_diagnostics")] = result.sfmDiagnostics;
                writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                  QStringLiteral("at_report.json"),
                                                  QStringLiteral("at_report_history.json"),
                                                  report);
                writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                  QStringLiteral("three_d_reconstruction_sfm_report.json"),
                                                  QStringLiteral("three_d_reconstruction_sfm_report_history.json"),
                                                  report);
            }
            return workflowResult;
        },
        [pmGuard,
         cancelFlag,
         projectPath,
         runSettings,
         sfmImages,
         sfmOutputDir](MenuWorkflowController *controller,
                       xjw::aerial_triangulation::AerialTriangulationResult workflowResult) mutable {
            if (!pmGuard)
            {
                return;
            }
            pmGuard->clearAtCancelFlag(cancelFlag);
            if (pmGuard->currentProjectPath() != projectPath)
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("三维重建"),
                                     QStringLiteral("项目已切换，本次三维重建空三结果未写回。"));
                return;
            }

            const bool wasCanceled = cancelFlag->load(std::memory_order_relaxed);
            if (wasCanceled)
            {
                emit pmGuard->atProgressFinished(false);
                return;
            }

            xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result =
                workflowResult.reconstructionResult;
            for (const xjw::matchphotos::MatchPhotosFeatureRecord &feature :
                 workflowResult.tiePointResult.features)
            {
                pmGuard->appendIpfindResult(feature.imagePath, feature.featurePath, feature.settings);
            }
            for (const xjw::matchphotos::MatchPhotosMatchRecord &match :
                 workflowResult.tiePointResult.matches)
            {
                pmGuard->appendIpmatchResult(QStringList{match.matchPath}, match.settings);
            }

            const bool resetCurrentAlignment =
                runSettings.value(QStringLiteral("reset_current_alignment")).toBool(true);
            if (result.success && resetCurrentAlignment)
            {
                int updated = 0;
                int cleared = 0;
                QString err;
                if (!pmGuard->replaceImageCameras(sfmImages,
                                                  result.pendingCamUpdates,
                                                  &updated,
                                                  &cleared,
                                                  &err))
                {
                    LOG_WARN(QStringLiteral("三维重建: SFM 相机写回失败: %1").arg(err));
                }
            }
            else if (!result.pendingCamUpdates.isEmpty())
            {
                int updated = 0;
                QString err;
                if (!pmGuard->setImageCameras(result.pendingCamUpdates, &updated, &err))
                {
                    LOG_WARN(QStringLiteral("三维重建: SFM 相机写回失败: %1").arg(err));
                }
            }

            int registeredImageCount = result.numRegisteredImages;
            int sfmAtIndex = -1;
            bool currentSfmIsProduction = false;
            QString currentSfmBlockingReason = QStringLiteral("SFM 未生成可用的正式稀疏点云，已停止后续 MVS 流程。");
            if (result.success && !result.sparseCloudPath.isEmpty())
            {
                const QStringList registeredCameraKeys = result.pendingCamUpdates.keys();
                QStringList registeredImages;
                registeredImages.reserve(sfmImages.size());
                for (const QString &imagePath : sfmImages)
                {
                    const QString normalized = xjw::common::project::normalizePath(imagePath);
                    if (registeredCameraKeys.contains(normalized))
                    {
                        registeredImages.append(normalized);
                    }
                }
                if (registeredImages.size() < registeredCameraKeys.size())
                {
                    for (const QString &imagePath : registeredCameraKeys)
                    {
                        if (!registeredImages.contains(imagePath))
                        {
                            registeredImages.append(imagePath);
                        }
                    }
                }
                registeredImageCount = registeredImages.size();

                QJsonObject resultRecordExtra = result.resultRecordExtra;
                resultRecordExtra[QStringLiteral("source")] = QStringLiteral("three_d_reconstruction");
                sfmAtIndex = 0;
                if (!pmGuard->replaceTiePointResult(result.sparseCloudPath,
                                                    result.numPoints3D,
                                                    registeredImages,
                                                    sfmOutputDir,
                                                    resultRecordExtra))
                {
                    emit pmGuard->atProgressFinished(false);
                    return;
                }
                currentSfmIsProduction = xjw::gui::project::isProductionSparseResult(resultRecordExtra);
                currentSfmBlockingReason = xjw::gui::project::sparseResultBlockingReason(resultRecordExtra);
            }

            emit pmGuard->atProgressFinished(result.success);
            if (!result.success)
            {
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("三维重建"),
                                     result.errorMessage.isEmpty()
                                         ? QStringLiteral("空中三角测量失败。")
                                         : result.errorMessage);
                return;
            }

            if (!currentSfmIsProduction)
            {
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("三维重建"),
                                     currentSfmBlockingReason.isEmpty()
                                         ? QStringLiteral("当前 SFM 稀疏点云质量不足，已停止后续 MVS 流程。")
                                         : currentSfmBlockingReason);
                return;
            }

            QJsonObject denseRunSettings = runSettings;
            denseRunSettings[QStringLiteral("registered_image_count")] = registeredImageCount;
            denseRunSettings[QStringLiteral("sfm_at_index")] = sfmAtIndex;
            controller->startThreeDReconstructionDenseStage(denseRunSettings);
        });
}

void MenuWorkflowController::startThreeDReconstructionDenseStage(const QJsonObject &settings)
{
    if (!_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(_projectManager);
    QPointer<MenuWorkflowController> self(this);
    connect(_projectManager, &ProjectManager::mvsProgressFinished, ctx,
            [self, ctx, settings](bool success) {
        ctx->deleteLater();
        if (!success || !self)
        {
            return;
        }

        self->startThreeDReconstructionDenseRefineStage(settings);
    });

    const QString outputRoot = settings.value(QStringLiteral("output_dir")).toString();
    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));

    QJsonObject denseSettings;
    denseSettings[QStringLiteral("pipeline_mode")] = true;
    denseSettings[QStringLiteral("at_index")] = settings.value(QStringLiteral("sfm_at_index")).toInt(-1);
    denseSettings[QStringLiteral("output_dir")] = QDir(outputRoot).filePath(QStringLiteral("mvs"));
    const int denseThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    const bool useCuda = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto")) != QStringLiteral("cpu");
    denseSettings[QStringLiteral("threads")] = denseThreads;
    denseSettings[QStringLiteral("cuda")] = useCuda;
    denseSettings[QStringLiteral("gpu_frame_workers")] = useCuda
        ? std::clamp(std::max(1, denseThreads / 4), 1, 2)
        : 0;
    denseSettings[QStringLiteral("cpu_frame_workers")] = useCuda
        ? 0
        : std::clamp(std::max(1, denseThreads / 4), 1, 4);
    denseSettings[QStringLiteral("keepColor")] = true;
    denseSettings[QStringLiteral("keepNormals")] = true;
    const int registeredImageCount = settings.value(QStringLiteral("registered_image_count")).toInt(0);
    const int denseMinViewCount = registeredImageCount > 0
        ? std::clamp(registeredImageCount, 2, 3)
        : 2;
    denseSettings[QStringLiteral("minConsistentViews")] = denseMinViewCount;
    denseSettings[QStringLiteral("minViews")] = denseMinViewCount;
    denseSettings[QStringLiteral("patchSize")] = 11;
    denseSettings[QStringLiteral("confidence")] = 0.20;
    denseSettings[QStringLiteral("minConfidence")] = 0.50;
    denseSettings[QStringLiteral("depthConsistency")] = 1.0;
    denseSettings[QStringLiteral("maxReprojError")] = 2.0;
    if (quality == QStringLiteral("fast"))
    {
        denseSettings[QStringLiteral("resScale")] = 0.25;
        denseSettings[QStringLiteral("iterations")] = 4;
    }
    else if (quality == QStringLiteral("quality"))
    {
        denseSettings[QStringLiteral("resScale")] = 0.5;
        denseSettings[QStringLiteral("iterations")] = 10;
    }
    else
    {
        denseSettings[QStringLiteral("resScale")] = 0.5;
        denseSettings[QStringLiteral("iterations")] = 6;
    }

    _projectManager->startGenerateDenseCloudAsync(denseSettings);
}

void MenuWorkflowController::startThreeDReconstructionDenseRefineStage(const QJsonObject &settings)
{
    if (!_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(_projectManager);
    QPointer<MenuWorkflowController> self(this);
    connect(_projectManager, &ProjectManager::mvsProgressFinished, ctx,
            [self, ctx, settings](bool success) {
        ctx->deleteLater();
        if (!success || !self)
        {
            return;
        }

        self->startThreeDReconstructionMeshStage(settings);
    });

    const bool useCuda = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto")) != QStringLiteral("cpu");
    QJsonObject refineSettings;
    refineSettings[QStringLiteral("pipeline_mode")] = true;
    refineSettings[QStringLiteral("sorEnabled")] = true;
    refineSettings[QStringLiteral("sorK")] = 30;
    refineSettings[QStringLiteral("sorStdDev")] = 2.0;
    refineSettings[QStringLiteral("voxelEnabled")] = false;
    refineSettings[QStringLiteral("voxelSize")] = 0.005;
    refineSettings[QStringLiteral("normalsEnabled")] = true;
    refineSettings[QStringLiteral("normalK")] = 30;
    refineSettings[QStringLiteral("smoothIter")] = 2;
    refineSettings[QStringLiteral("colorEnabled")] = false;
    refineSettings[QStringLiteral("threads")] = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    refineSettings[QStringLiteral("processingDevice")] = useCuda ? QStringLiteral("gpu") : QStringLiteral("cpu");

    _projectManager->startDenseCloudRefineAsync(refineSettings);
}

void MenuWorkflowController::startThreeDReconstructionMeshStage(const QJsonObject &settings)
{
    if (!_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(_projectManager);
    connect(_projectManager, &ProjectManager::meshProgressFinished, ctx,
            [this, ctx](bool success) {
        ctx->deleteLater();
        QMessageBox::information(_mainWindow,
                                 QStringLiteral("三维重建"),
                                 success ? QStringLiteral("三维模型生成完成。")
                                         : QStringLiteral("三维模型生成失败。"));
    });

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    QJsonObject meshSettings;
    meshSettings[QStringLiteral("pipeline_mode")] = true;
    meshSettings[QStringLiteral("method")] = QStringLiteral("Poisson Surface");
    meshSettings[QStringLiteral("export_format")] =
        settings.value(QStringLiteral("export_obj")).toBool(false) ? QStringLiteral("OBJ") : QStringLiteral("PLY");
    meshSettings[QStringLiteral("smoothIter")] = 2;
    meshSettings[QStringLiteral("holeFill")] = true;
    meshSettings[QStringLiteral("cleanSmall")] = true;
    meshSettings[QStringLiteral("threads")] = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    if (quality == QStringLiteral("fast"))
    {
        meshSettings[QStringLiteral("qualityProfile")] = QStringLiteral("lite");
        meshSettings[QStringLiteral("voxelDensity")] = QStringLiteral("coarse");
        meshSettings[QStringLiteral("resolution")] = 160;
        meshSettings[QStringLiteral("octreeDepth")] = 8;
    }
    else if (quality == QStringLiteral("quality"))
    {
        meshSettings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");
        meshSettings[QStringLiteral("voxelDensity")] = QStringLiteral("fine");
        meshSettings[QStringLiteral("resolution")] = 320;
        meshSettings[QStringLiteral("octreeDepth")] = 10;
    }
    else
    {
        meshSettings[QStringLiteral("qualityProfile")] = QStringLiteral("balanced");
        meshSettings[QStringLiteral("voxelDensity")] = QStringLiteral("medium");
        meshSettings[QStringLiteral("resolution")] = 224;
        meshSettings[QStringLiteral("octreeDepth")] = 9;
    }

    _projectManager->startMeshReconstructionAsync(meshSettings);
}

void MenuWorkflowController::openOverlapAnalysisDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dlg = new OverlapAnalysisDialog(_projectManager, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MenuWorkflowController::openCreateDemDialog()
{
    if (!_mainWindow)
    {
        return;
    }
    if (!_projectManager)
    {
        QMessageBox::warning(_mainWindow, QStringLiteral("生成 DEM"), QStringLiteral("请先打开项目"));
        return;
    }

    auto *dlg = new CreateDemDialog(_projectManager, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    QStringList images = getProjectImages();
    if (!images.isEmpty())
    {
        dlg->setAvailableImages(images);
    }

    // 自动模式：完整流水线
    connect(dlg, &CreateDemDialog::requestRunFullPipeline, this,
        [this](const QStringList &images, const QString &outputDir, const QJsonObject &pipelineSettings)
    {
        if (!_projectManager)
        {
            return;
        }
        QPointer<ProjectManager> pmGuard(_projectManager);
        const QString projectPath = pmGuard->currentProjectPath();
        QTimer::singleShot(0, pmGuard.data(),
            [pmGuard, projectPath, images, outputDir, pipelineSettings]()
            {
                if (!pmGuard || pmGuard->currentProjectPath() != projectPath)
                {
                    return;
                }
                pmGuard->startFullDemPipelineAsync(images, outputDir, pipelineSettings);
            });
    });

    // 手动模式：从密集点云生成 DEM
    connect(dlg, &CreateDemDialog::requestRunFromDenseCloud, this,
        [this](const QString &denseCloudPath, const QString &outputDir, double demResolution, const QString &demType)
    {
        if (!_projectManager)
        {
            return;
        }
        QPointer<ProjectManager> pmGuard(_projectManager);
        const QString projectPath = pmGuard->currentProjectPath();
        QTimer::singleShot(0, pmGuard.data(),
            [pmGuard, projectPath, denseCloudPath, outputDir, demResolution, demType]()
            {
                if (!pmGuard || pmGuard->currentProjectPath() != projectPath)
                {
                    return;
                }
                pmGuard->startDemFromDenseCloudAsync(denseCloudPath, outputDir, demResolution, demType);
            });
    });

    // 进度反馈 → 对话框内显示
    if (_projectManager)
    {
        connect(_projectManager, &ProjectManager::demPipelineProgressChanged,
                dlg, &CreateDemDialog::onPipelineProgress);
        connect(_projectManager, &ProjectManager::demPipelineFinished,
                dlg, &CreateDemDialog::onPipelineFinished);
    }

    dlg->show();
}

void MenuWorkflowController::openMapProjectDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dlg = new MapProjectDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (_projectManager)
    {
        QStringList images = getProjectImages();
        if (!images.isEmpty())
        {
            dlg->setAvailableImages(images);
        }

        QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(_projectManager->currentProjectPath());
        if (!projectRoot.isEmpty())
        {
            dlg->setProjectRoot(projectRoot);
        }

        const QJsonArray demResults = _projectManager->currentMeta().value(QStringLiteral("dem_results")).toArray();
        QString latestRelativeDem;
        QString latestAnyDem;
        for (int index = demResults.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demResults.at(index).toObject();
            const QString candidate = record.value(QStringLiteral("dem_tif")).toString();
            if (candidate.isEmpty())
            {
                continue;
            }
            if (latestAnyDem.isEmpty())
            {
                latestAnyDem = candidate;
            }
            if (record.value(QStringLiteral("dem_reference")).toString() == QStringLiteral("relative"))
            {
                latestRelativeDem = candidate;
                break;
            }
        }
        dlg->setDefaultDemPath(!latestRelativeDem.isEmpty() ? latestRelativeDem : latestAnyDem);

        // 懒初始化 MapProject 记忆化管理器
        if (!_mapSetting)
        {
            _mapSetting = new DialogSettingStore(DialogSettingKeys::MapProject, this);
        }
        _mapSetting->setProjectPath(_projectManager->currentProjectPath());
        const QJsonObject saved = _mapSetting->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }
    }

    connect(dlg, &MapProjectDialog::settingsChanged, this, [this](const QJsonObject &s)
    {
        if (_mapSetting)
        {
            _mapSetting->save(s);
        }
    });

    connect(dlg, &MapProjectDialog::requestRunMapProject, this,
        [this](const QStringList &images, const QString &demPath, const QString &outputPath, double res)
        {
        if (!_projectManager)
        {
            LOG_WARN(QStringLiteral("MapProject: 未找到 ProjectManager"));
            return;
        }
        QPointer<ProjectManager> pmGuard(_projectManager);
        const QString projectPath = pmGuard->currentProjectPath();
        QTimer::singleShot(0, pmGuard.data(),
            [pmGuard, projectPath, images, demPath, outputPath, res]()
            {
                if (!pmGuard || pmGuard->currentProjectPath() != projectPath)
                {
                    return;
                }
                pmGuard->startMapProjectAsync(images, demPath, outputPath, res);
            });
    });

    dlg->exec();
}

void MenuWorkflowController::runFeatureExtraction(const QJsonObject &config, const QStringList &inputs)
{
    const QString featureAlgorithm = config.value(QStringLiteral("feature_algorithm")).toString(QStringLiteral("disk")).toUpper();
    LOG_INFO(QStringLiteral("开始在后台线程执行 %1 特征提取...").arg(featureAlgorithm));

    QPointer<MainWindow> mainWin(qobject_cast<MainWindow *>(_mainWindow.data()));

    auto cancelFlag    = std::make_shared<std::atomic<bool>>(false);
    auto progressCount = std::make_shared<std::atomic<int>>(0);
    const int total    = inputs.size();

    if (mainWin)
    {
        mainWin->showSpProgress(total);
    }

    // 取消按钮临时连接
    QMetaObject::Connection cancelConn;
    if (mainWin)
    {
        cancelConn = connect(mainWin, &MainWindow::spCancelRequested,
                             [cancelFlag]()
                             {
                                 cancelFlag->store(true);
                             });
    }

    // 定时轮询进度（100ms）
    auto *timer = new QTimer(_mainWindow);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, timer, [mainWin, progressCount]()
    {
        if (mainWin)
        {
            mainWin->updateSpProgress(progressCount->load());
        }
    });
    timer->start();

    auto *watcher = new QFutureWatcher<bool>(_mainWindow);
    connect(watcher, &QFutureWatcher<bool>::finished, watcher,
            [mainWin, cancelFlag, timer, watcher, cancelConn]()
    {
        timer->stop();
        timer->deleteLater();
        if (mainWin)
        {
            QObject::disconnect(cancelConn);
            const bool cancelled = cancelFlag->load();
            const bool taskOk = !cancelled && watcher->result();
            mainWin->hideSpProgress(taskOk);
        }
        watcher->deleteLater();
    });

    QPointer<ProjectManager> pmGuard(_projectManager);
    watcher->setFuture(QtConcurrent::run(
        [config, inputs, pmGuard, cancelFlag, progressCount]() -> bool
        {
            return FeatureExtractionRunner::run(config, inputs, pmGuard, *cancelFlag, *progressCount);
        }));
}

void MenuWorkflowController::openWorkflowReportDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    QString assetsDir;
    if (_projectManager)
    {
        assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
    }

    auto *dlg = new WorkflowReportDialog(assetsDir, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MenuWorkflowController::openCameraConvertDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dlg = new CameraConvertDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}
