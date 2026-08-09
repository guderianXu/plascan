#include "MenuWorkflowController.h"

#include "ProjectManager.h"
#include "project/ProjectIO.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "workflow/AerialTriangulationWorkflow.h"
#include "preparation/MatchResultCatalog.h"
#include "preparation/ReconstructionPrerequisiteReport.h"
#include "Logger.h"
#include "project/SparseResultQuality.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowReports.h"

#include "GuiTaskRunner.h"
#include "FeatureVisualizationController.h"
#include "MainMenu.h"
#include "tie_points/MatchPairSelectorDialog.h"
#include "reconstruction/AerialTriangulationDialog.h"
#include "application/WorkflowSettingsDialog.h"
#include "tie_points/OverlapAnalysisDialog.h"
#include "reconstruction/CreateDemDialog.h"
#include "reconstruction/MapProjectDialog.h"
#include "application/WorkflowReportDialog.h"
#include "camera/CameraCalibrationDialog.h"
#include "camera/CameraConvertDialog.h"

#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"

#include <QCheckBox>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
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

QString normalizedReferencePreselectionSource(const QJsonObject &settings)
{
    const QString source = settings.value(QStringLiteral("reference_preselection_source"))
                               .toString(QStringLiteral("source_code"))
                               .trimmed()
                               .toLower();
    return source == QStringLiteral("estimated_pose") ? QStringLiteral("estimated") : source;
}

QMap<QString, xjw::Camera> referenceCamerasForMode(ProjectManager *projectManager,
                                                   const QStringList &images,
                                                   const QJsonObject &projectMeta,
                                                   const QString &requestedMode,
                                                   bool *hasCamerasForAll)
{
    if (hasCamerasForAll)
    {
        *hasCamerasForAll = false;
    }
    if (!projectManager)
    {
        return {};
    }

    bool loadedAll = false;
    const QMap<QString, xjw::Camera> allCameras =
        projectManager->getCamerasForImages(images, &loadedAll);
    if (!loadedAll)
    {
        return {};
    }

    const QString mode = requestedMode.trimmed().toLower() == QStringLiteral("estimated_pose")
        ? QStringLiteral("estimated")
        : requestedMode.trimmed().toLower();
    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::common::project::projectImageMetaByPath(projectMeta, true);
    const QJsonArray imageEntries = xjw::common::project::projectImageEntries(projectMeta);
    QMap<QString, xjw::Camera> filtered;
    for (const QString &imagePath : images)
    {
        const QString normalized = xjw::common::project::normalizePath(imagePath);
        const auto cameraIt = allCameras.constFind(normalized);
        if (cameraIt == allCameras.constEnd())
        {
            continue;
        }

        QJsonObject imageMeta = imageMetaByPath.value(normalized);
        if (imageMeta.isEmpty())
        {
            // 运行时影像通常是已解析绝对路径，而持久化元数据可能仍是 plascan URI。
            for (const QJsonValue &value : imageEntries)
            {
                const QJsonObject entry = value.toObject();
                if (xjw::common::project::pathTokenMatchesImage(
                        entry.value(QStringLiteral("path")).toString(), imagePath))
                {
                    imageMeta = entry;
                    break;
                }
            }
        }
        const QString poseSource = imageMeta.value(QStringLiteral("camera"))
                                       .toObject()
                                       .value(QStringLiteral("pose_source"))
                                       .toString()
                                       .trimmed()
                                       .toLower();
        const bool sfmEstimated = poseSource == QStringLiteral("sfm_estimated");
        if ((mode == QStringLiteral("estimated") && !sfmEstimated) ||
            (mode == QStringLiteral("source_code") && sfmEstimated))
        {
            continue;
        }
        filtered.insert(normalized, cameraIt.value());
    }

    if (hasCamerasForAll)
    {
        *hasCamerasForAll = filtered.size() == images.size();
    }
    return filtered;
}

bool shouldUseStoredGeneratedPairConstraints(const QJsonObject &settings)
{
    if (settings.value(QStringLiteral("reference_preselection")).toBool(false))
    {
        // 参考来源是用户本次显式选择的配对先验。历史 generated_pairs 多来自另一次
        // 词汇树/重叠配置，不能继续作为 ManualOnly 白名单覆盖当前位姿或序列策略。
        return false;
    }
    return true;
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
    , _featureVisualizationController(new FeatureVisualizationController(mainWindow, this))
{
    connect(_featureVisualizationController,
            &FeatureVisualizationController::optionsChanged,
            this,
            &MenuWorkflowController::requestApplyFeatureDisplayOptions);
}

void MenuWorkflowController::setProjectManager(ProjectManager *projectManager)
{
    _projectManager = projectManager;
    _featureVisualizationController->setProjectManager(projectManager);
}

DialogSettingStore *MenuWorkflowController::createDialogSettingStore(
    const QString &settingKey)
{
    auto *store = new DialogSettingStore(settingKey, this);
    store->setChangeCallback([this]()
    {
        if (_projectManager)
        {
            _projectManager->markWorkspaceDirty();
        }
    });
    return store;
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

    connectAction(mainMenu->workflowAerialTriangulationAction(),
                  &MenuWorkflowController::openWorkflowAerialTriangulationDialog);
    connectAction(mainMenu->workflowSettingsAction(),
                  &MenuWorkflowController::openWorkflowSettingsDialog);
    if (mainMenu->featureVisualizationAction())
    {
        connect(mainMenu->featureVisualizationAction(),
                &QAction::triggered,
                _featureVisualizationController,
                &FeatureVisualizationController::openDialog,
                Qt::UniqueConnection);
    }
    connectAction(mainMenu->overlapAnalysisAction(), &MenuWorkflowController::openOverlapAnalysisDialog);
    connectAction(mainMenu->createDEMAction(), &MenuWorkflowController::openCreateDemDialog);
    connectAction(mainMenu->generateOrthoAction(), &MenuWorkflowController::openMapProjectDialog);
    connectAction(mainMenu->viewWorkflowReportAction(), &MenuWorkflowController::openWorkflowReportDialog);
    connectAction(mainMenu->cameraCalibrationAction(), &MenuWorkflowController::openCameraCalibrationDialog);
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
                                                     const QString &algorithmId,
                                                     const std::function<void(int, int)> &progressCallback)
{
    SparsePrerequisiteSummary summary;
    summary.imageCount = images.size();
    // 新匹配链路只有组合算法标识，不再把“特征算法”和“匹配算法”拆成两个
    // 自由字符串。SIFT 描述子只驻留任务内存，预检也不再查找特征中间文件。
    const QString selectedAlgorithmId = algorithmId.trimmed().isEmpty()
        ? QStringLiteral("sift_lightglue")
        : algorithmId.trimmed().toLower();

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

    auto variantAlgorithmMatches = [&](const xjw::aerial_triangulation::MatchVariant &variant) -> bool
    {
        if (!variant.compatible)
        {
            return false;
        }
        return variant.algorithmId.trimmed().toLower() == selectedAlgorithmId;
    };

    QSet<QString> matchedPairKeys;
    QVector<QPair<QString, QString>> matchedPairs;

    xjw::aerial_triangulation::MatchResultCatalogSummary catalogSummary;
    if (!projectPath.isEmpty())
    {
        xjw::aerial_triangulation::MatchResultCatalogConfig catalogConfig;
        catalogConfig.matchDirectory =
            xjw::common::project::ProjectIO::imageMatchOutputDir(projectPath);
        catalogConfig.targetImagePaths = images;
        catalogConfig.progressCallback = progressCallback;
        catalogSummary = xjw::aerial_triangulation::MatchResultCatalog(catalogConfig).scan();
        LOG_INFO(QStringLiteral(
                     "空三上游索引: 分片=%1 内存命中=%2 持久索引命中=%3 首次重建=%4 损坏=%5")
                     .arg(catalogSummary.matchFileCount)
                     .arg(catalogSummary.memoryIndexHitCount)
                     .arg(catalogSummary.persistentIndexHitCount)
                     .arg(catalogSummary.rebuiltIndexCount)
                     .arg(catalogSummary.incompatibleVariantCount));
    }

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

    if (!catalogSummary.pairGroups.isEmpty())
    {
        for (const xjw::aerial_triangulation::MatchPairGroup &group : catalogSummary.pairGroups)
        {
            for (const xjw::aerial_triangulation::MatchVariant &variant : group.variants)
            {
                if (!variantAlgorithmMatches(variant))
                {
                    continue;
                }

                if (variant.geometryPassed && variant.geometricVerifiedInliers > 0)
                {
                    appendPreflightPair(variant.imageA.isEmpty() ? group.imageA : variant.imageA,
                                        variant.imageB.isEmpty() ? group.imageB : variant.imageB);
                }
                break;
            }
        }
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

    if (!catalogSummary.pairGroups.isEmpty())
    {
        // 无匹配或几何失败也是 `.pimatch` 中的一种确定结果。它与有效匹配
        // 共用同一格式和版本，不再维护容易失同步的 no_match_pairs.json。
        for (const auto &group : catalogSummary.pairGroups)
        {
            const auto settled = std::find_if(
                group.variants.cbegin(), group.variants.cend(),
                [&](const xjw::aerial_triangulation::MatchVariant &variant)
                {
                    return variant.compatible &&
                        variant.algorithmId.trimmed().toLower() == selectedAlgorithmId &&
                        (!variant.geometryPassed || variant.geometricVerifiedInliers <= 0);
                });
            if (settled != group.variants.cend())
            {
                appendSettledNoMatchPair(group.imageA, group.imageB);
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
    case xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::PrepareImageMatches:
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

void MenuWorkflowController::applySavedFeatureDisplayOptions(const QJsonObject &uiSettings)
{
    _featureVisualizationController->applySavedOptions(uiSettings);
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
        _aerialTriangulationSetting =
            createDialogSettingStore(DialogSettingKeys::AerialTriangulation);
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
        const QJsonObject dialogSettings = dlg.collectSettings();
        if (_aerialTriangulationSetting)
        {
            _aerialTriangulationSetting->save(dialogSettings);
        }
        startAerialTriangulationWorkflow(
            mergeAerialTriangulationSettings(dialogSettings));
    }
}

void MenuWorkflowController::openWorkflowSettingsDialog()
{
    if (!_mainWindow)
    {
        return;
    }
    if (!_projectManager || _projectManager->currentProjectPath().trimmed().isEmpty())
    {
        QMessageBox::warning(_mainWindow,
                             QStringLiteral("工作流程设置"),
                             QStringLiteral("请先打开项目。工作流程设置按项目保存。"));
        return;
    }

    if (!_workflowSettingsStore)
    {
        _workflowSettingsStore = createDialogSettingStore(
            DialogSettingKeys::WorkflowSettings);
    }
    _workflowSettingsStore->setProjectPath(_projectManager->currentProjectPath());

    WorkflowSettingsDialog dialog(_mainWindow);
    dialog.applySettings(_workflowSettingsStore->load());
    if (dialog.exec() == QDialog::Accepted)
    {
        QString saveError;
        if (!_workflowSettingsStore->save(dialog.collectSettings(), &saveError))
        {
            QMessageBox::warning(_mainWindow,
                                 QStringLiteral("工作流程设置"),
                                 saveError.isEmpty()
                                     ? QStringLiteral("无法保存工作流程设置。")
                                     : saveError);
        }
    }
}

QJsonObject MenuWorkflowController::mergeAerialTriangulationSettings(
    const QJsonObject &dialogSettings)
{
    QJsonObject merged;
    const QJsonObject defaultAerialSettings =
        WorkflowSettingsDialog::aerialTriangulationSettings(
            WorkflowSettingsDialog::defaultSettings());
    for (auto it = defaultAerialSettings.constBegin();
         it != defaultAerialSettings.constEnd(); ++it)
    {
        merged.insert(it.key(), it.value());
    }
    if (_projectManager)
    {
        if (!_workflowSettingsStore)
        {
            _workflowSettingsStore = createDialogSettingStore(
                DialogSettingKeys::WorkflowSettings);
        }
        _workflowSettingsStore->setProjectPath(_projectManager->currentProjectPath());
        const QJsonObject savedAerialSettings =
            WorkflowSettingsDialog::aerialTriangulationSettings(
                _workflowSettingsStore->load());
        for (auto it = savedAerialSettings.constBegin();
             it != savedAerialSettings.constEnd(); ++it)
        {
            merged.insert(it.key(), it.value());
        }
    }

    // 空三主对话框拥有质量、预选、蒙版和连接点总配额等高频字段；若未来
    // 两个对话框出现同名字段，应以用户本次确认的主对话框值为准。
    for (auto it = dialogSettings.constBegin(); it != dialogSettings.constEnd(); ++it)
    {
        merged.insert(it.key(), it.value());
    }
    return merged;
}

QJsonObject MenuWorkflowController::sanitizeAerialTriangulationReferencePreselection(
    const QJsonObject &requestedSettings,
    const QStringList &images,
    const QJsonObject &projectMeta) const
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
    const QString referenceMode = normalizedReferencePreselectionSource(settings);
    const int cameraCount = referenceCamerasForMode(_projectManager,
                                                    images,
                                                    projectMeta,
                                                    referenceMode,
                                                    &hasAllReferenceCameras).size();
    const bool available =
        hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2;
    if (!available)
    {
        settings[QStringLiteral("reference_preselection")] = false;
        LOG_WARN(QStringLiteral("空中三角测量: %1 参考位姿不完整，参考预选已关闭（相机 %2/%3）。")
                     .arg(referenceMode)
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

    auto *pm = _projectManager;
    if (pm->hasActiveAtTask())
    {
        QMessageBox::information(
            _mainWindow,
            QStringLiteral("空中三角测量"),
            QStringLiteral("已有空三或光束法平差任务正在运行，请等待其结束或先取消当前任务。"));
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

    const auto session = pm->currentSessionContext();
    const QString projectPath = session.projectPath;
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
    runSettings = sanitizeAerialTriangulationReferencePreselection(runSettings, images, projectMeta);
    const QString selectedAlgorithmId =
        runSettings.value(QStringLiteral("algorithm_id"))
            .toString(QStringLiteral("sift_lightglue"))
            .trimmed()
            .toLower();

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    emit pm->atProgressChanged(QStringLiteral("空中三角测量: 检查上游数据..."), 0);

    QPointer<ProjectManager> pmGuard(pm);
    const auto preflightProgress = [pmGuard, session, cancelFlag](int processed, int total)
    {
        if (!pmGuard || cancelFlag->load(std::memory_order_relaxed))
        {
            return;
        }
        const int percent = total <= 0
            ? 100
            : qBound(0, static_cast<int>((static_cast<qint64>(processed) * 100) / total), 100);
        xjw::gui::tasks::postGuarded(pmGuard,
            [session, cancelFlag, processed, total, percent](ProjectManager *manager)
        {
            if (manager->ownsAtCancelFlag(cancelFlag)
                && manager->isCurrentSession(session)
                && !cancelFlag->load(std::memory_order_relaxed))
            {
                emit manager->atProgressChanged(
                    QStringLiteral("空中三角测量: 检查上游匹配索引 %1/%2")
                        .arg(processed)
                        .arg(total),
                    percent);
            }
        });
    };
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [images, projectMeta, projectPath, selectedAlgorithmId, preflightProgress]()
        {
            return MenuWorkflowController::summarizeSparsePrerequisites(images,
                                                                        projectMeta,
                                                                        projectPath,
                                                                        selectedAlgorithmId,
                                                                        preflightProgress);
        },
        [pmGuard, cancelFlag, runSettings, images, session, projectMeta, outputRoot](
            MenuWorkflowController *controller,
            xjw::gui::tasks::TaskOutcome<SparsePrerequisiteSummary> outcome)
        {
            if (!pmGuard)
            {
                return;
            }

            const bool ownsTask = pmGuard->ownsAtCancelFlag(cancelFlag);
            const bool currentSession = pmGuard->isCurrentSession(session);
            if (!ownsTask || !currentSession)
            {
                pmGuard->clearAtCancelFlag(cancelFlag);
                return;
            }
            if (cancelFlag->load(std::memory_order_relaxed))
            {
                pmGuard->clearAtCancelFlag(cancelFlag);
                emit pmGuard->atProgressFinished(false);
                return;
            }

            if (!outcome.succeeded())
            {
                pmGuard->clearAtCancelFlag(cancelFlag);
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     outcome.errorMessage);
                return;
            }

            const auto &prereq = *outcome.value;
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
                pmGuard->clearAtCancelFlag(cancelFlag);
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
                                                      session,
                                                      projectMeta,
                                                      outputRoot,
                                                      autoFillMissing,
                                                      cancelFlag);
        });
}

void MenuWorkflowController::runUnifiedAerialTriangulation(const QJsonObject &settings,
                                                            const QStringList &images,
                                                            const xjw::gui::project::ProjectSessionContext &session,
                                                            const QJsonObject &projectMeta,
                                                            const QString &outputRoot,
                                                            bool fillMissingTiePoints,
                                                            const std::shared_ptr<std::atomic<bool>> &cancelFlag)
{
    if (!_projectManager)
    {
        return;
    }

    auto *pm = _projectManager;
    if (!pm->isCurrentSession(session)
        || !pm->ownsAtCancelFlag(cancelFlag)
        || cancelFlag->load(std::memory_order_relaxed))
    {
        pm->clearAtCancelFlag(cancelFlag);
        return;
    }

    const QString projectPath = session.projectPath;

    const bool resetCurrentAlignment =
        settings.value(QStringLiteral("reset_current_alignment")).toBool(true);

    // 线程数属于机器运行时能力，不能复用项目里由另一台电脑保存的历史固定值。
    // 统一传 0，由核心层在每次启动时按当前机器逻辑线程数解析。
    constexpr int workflowThreads = 0;
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
    workflowOptions.matchingAlgorithmId =
        settings.value(QStringLiteral("algorithm_id"))
            .toString(QStringLiteral("sift_lightglue"))
            .trimmed()
            .toLower();
    workflowOptions.lightGlueTensorRtEnginePath =
        settings.value(QStringLiteral("lightglue_tensorrt_engine")).toString().trimmed();
    workflowOptions.lomaRTensorRtPackagePath =
        settings.value(QStringLiteral("loma_r_tensorrt_package")).toString().trimmed();
    workflowOptions.lomaRKeypointBudget =
        settings.value(QStringLiteral("loma_r_keypoint_budget")).toInt(0);
    workflowOptions.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    workflowOptions.threads = workflowThreads;
    workflowOptions.cudaDevice = std::max(
        0, settings.value(QStringLiteral("cuda_device")).toInt(0));
    workflowOptions.featureMaxImageDim = std::max(
        0, settings.value(QStringLiteral("feature_max_image_dim")).toInt(0));
    workflowOptions.cudaParallelPairs = std::max(
        0, settings.value(QStringLiteral("cuda_parallel_pairs")).toInt(0));
    workflowOptions.featurePrefetchDepth = std::clamp(
        settings.value(QStringLiteral("feature_prefetch_depth")).toInt(2), 1, 4);
    workflowOptions.matchThreshold = static_cast<float>(std::clamp(
        settings.value(QStringLiteral("match_threshold")).toDouble(0.15), 0.0, 1.0));
    workflowOptions.geometryReprojThreshold = std::max(
        0.1, settings.value(QStringLiteral("geometry_reprojection_threshold_px")).toDouble(1.5));
    workflowOptions.geometryMinInliers = std::max(
        8, settings.value(QStringLiteral("geometry_min_inliers")).toInt(20));
    workflowOptions.geometryMaxIterations = std::max(
        100, settings.value(QStringLiteral("geometry_max_iterations")).toInt(10000));
    workflowOptions.tiePointGridColumns = std::clamp(
        settings.value(QStringLiteral("tie_point_grid_columns")).toInt(8), 1, 64);
    workflowOptions.tiePointGridRows = std::clamp(
        settings.value(QStringLiteral("tie_point_grid_rows")).toInt(8), 1, 64);
    workflowOptions.maxTiePointsPerGridCell = std::max(
        0, settings.value(QStringLiteral("tie_point_grid_cell_limit")).toInt(0));
    workflowOptions.stationaryTiePointMaxPixelMotion = static_cast<float>(std::max(
        0.0,
        settings.value(QStringLiteral("stationary_tie_point_max_pixel_motion")).toDouble(1.0)));
    workflowOptions.autoGenerateMissingMatches = fillMissingTiePoints;
    workflowOptions.assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    workflowOptions.matchDir = xjw::common::project::ProjectIO::imageMatchOutputDir(projectPath);
    workflowOptions.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(projectPath, images);
    workflowOptions.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    workflowOptions.featureGrayscaleMax = 1.0f;

    if (workflowOptions.referencePreselection &&
        workflowOptions.referenceMode.trimmed().toLower() != QStringLiteral("sequence"))
    {
        bool hasAllReferenceCameras = false;
        workflowOptions.referenceCameras = referenceCamerasForMode(
            pm,
            images,
            projectMeta,
            workflowOptions.referenceMode,
            &hasAllReferenceCameras);
        if (!hasAllReferenceCameras)
        {
            LOG_WARN(QStringLiteral("空中三角测量: 参考预选已启用，但 %1 位姿不完整")
                         .arg(workflowOptions.referenceMode));
        }
        else
        {
            LOG_INFO(QStringLiteral("空中三角测量: 已加载 %1 个 %2 参考位姿用于候选对规划")
                         .arg(workflowOptions.referenceCameras.size())
                         .arg(workflowOptions.referenceMode));
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
        LOG_INFO(QStringLiteral("空中三角测量: 参考预选已启用，跳过历史候选配对约束并按本次来源重新规划"));
    }
    else if (storedPairsStale)
    {
        LOG_WARN(QStringLiteral("空中三角测量: 已生成候选配对与当前影像集合不一致，改用自动配对规划"));
    }

    QPointer<ProjectManager> pmGuard(pm);
    workflowOptions.progressFn = [pmGuard, session, cancelFlag](const QString &stage, int percent)
    {
        if (!pmGuard || cancelFlag->load(std::memory_order_relaxed))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(
            pmGuard,
            [session, cancelFlag, stage, percent](ProjectManager *manager)
        {
            if (manager->ownsAtCancelFlag(cancelFlag)
                && manager->isCurrentSession(session)
                && !cancelFlag->load(std::memory_order_relaxed))
            {
                emit manager->atProgressChanged(
                    QStringLiteral("空中三角测量: %1").arg(stage), percent);
            }
        });
    };
    workflowOptions.pairMatchedFn = [pmGuard, session, cancelFlag](const QString &img0,
                                                                  const QString &img1,
                                                                  const QString &matchPath,
                                                                  int numMatches)
    {
        if (!pmGuard || cancelFlag->load(std::memory_order_relaxed))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(
            pmGuard,
            [session, cancelFlag, img0, img1, matchPath, numMatches](ProjectManager *manager)
        {
            if (manager->ownsAtCancelFlag(cancelFlag)
                && manager->isCurrentSession(session)
                && !cancelFlag->load(std::memory_order_relaxed))
            {
                emit manager->matchPairReady(img0, img1, matchPath, numMatches);
            }
        });
    };

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
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [runWorkflowOptions = std::move(workflowOptions)]() mutable
        {
            return xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
                runWorkflowOptions);
        },
        [pmGuard,
         cancelFlag,
         sfmImages,
         sfmOutputDir,
         assetsDir,
         projectMeta,
         session,
         resetCurrentAlignment](MenuWorkflowController *controller,
                                xjw::gui::tasks::TaskOutcome<
                                    xjw::aerial_triangulation::AerialTriangulationResult> outcome) mutable {
            if (!pmGuard)
            {
                return;
            }
            if (!pmGuard->ownsAtCancelFlag(cancelFlag)
                || !pmGuard->isCurrentSession(session))
            {
                return;
            }

            pmGuard->clearAtCancelFlag(cancelFlag);
            if (!outcome.succeeded())
            {
                emit pmGuard->atProgressFinished(false);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     outcome.errorMessage);
                return;
            }
            auto workflowResult = std::move(*outcome.value);
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result =
                workflowResult.reconstructionResult;
            const bool wasCanceled = cancelFlag->load(std::memory_order_relaxed);
            if (wasCanceled)
            {
                emit pmGuard->atProgressFinished(false);
                return;
            }

            if (workflowResult.tiePointPreparationExecuted)
            {
                pmGuard->appendImageMatchResults(
                    xjw::gui::project::makeImageMatchResultRecords(
                        workflowResult.tiePointResult));
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

            if (!assetsDir.isEmpty())
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
                report[QStringLiteral("camera_comparison")] =
                    xjw::gui::camera_calibration::buildCameraCalibrationComparison(
                        projectMeta,
                        result.pendingCamUpdates,
                        result.sfmDiagnostics);
                report[QStringLiteral("camera_calibration_semantics")] = QJsonObject{
                    {QStringLiteral("initial"), QStringLiteral("intrinsics_at_alignment_start")},
                    {QStringLiteral("adjusted"), QStringLiteral("intrinsics_after_bundle_adjustment")},
                    {QStringLiteral("principal_point"), QStringLiteral("offset_from_image_center")},
                    {QStringLiteral("excludes_extrinsics"), true}};
                report[QStringLiteral("resolved_settings")] = workflowResult.config.resolvedSettings;
                report[QStringLiteral("tie_point_preparation_executed")] =
                    workflowResult.tiePointPreparationExecuted;
                report[QStringLiteral("tie_point_track_count")] =
                    workflowResult.tiePointResult.trackCount;
                const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
                xjw::gui::project::writeLatestAndAppendHistoryReport(
                    reportsDir,
                    QStringLiteral("at_report.json"),
                    QStringLiteral("at_report_history.json"),
                    report);
                xjw::gui::project::writeLatestAndAppendHistoryReport(
                    reportsDir,
                    QStringLiteral("aerial_triangulation_sfm_report.json"),
                    QStringLiteral("aerial_triangulation_sfm_report_history.json"),
                    report);
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

    auto *dlg = new CreateDemDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    connect(dlg, &CreateDemDialog::requestRun, this,
        [this](const xjw::gui::project::DemGenerationRequest &request)
    {
        if (!_projectManager)
        {
            return;
        }
        QPointer<ProjectManager> pmGuard(_projectManager);
        const auto session = pmGuard->currentSessionContext();
        QTimer::singleShot(0, pmGuard.data(),
            [pmGuard, session, request]()
            {
                if (!pmGuard || !pmGuard->isCurrentSession(session))
                {
                    return;
                }
                pmGuard->startDemFromPointCloudAsync(request);
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

        const QString projectPath = _projectManager->currentProjectPath();
        const QString projectRoot =
            xjw::common::project::ProjectIO::projectRootFromPlascan(projectPath);
        if (!projectRoot.isEmpty())
        {
            dlg->setProjectRoot(projectRoot);
        }

        const QMap<QString, xjw::Camera> cameraMap =
            _projectManager->getCamerasForImages(images);
        int maskReadyCount = 0;
        for (const QString &imagePath : images)
        {
            if (!xjw::common::project::ProjectIO::findMaskForImage(projectPath, imagePath).isEmpty())
            {
                ++maskReadyCount;
            }
        }
        dlg->setImageReadiness(cameraMap.keys(), maskReadyCount);

        const QJsonArray demResults = _projectManager->currentMeta().value(QStringLiteral("dem_results")).toArray();
        QString latestRelativeDem;
        QString latestAnyDem;
        for (int index = demResults.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demResults.at(index).toObject();
            const QString candidate =
                xjw::common::project::ProjectIO::resolveProjectResourcePath(
                    projectPath,
                    record.value(QStringLiteral("dem_tif")).toString());
            const QFileInfo candidateInfo(candidate);
            if (candidate.isEmpty() || !candidateInfo.exists() || !candidateInfo.isFile())
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

        const QJsonArray denseResults =
            _projectManager->currentMeta().value(QStringLiteral("dense_cloud_results")).toArray();
        for (int index = denseResults.size() - 1; index >= 0; --index)
        {
            const QString candidate =
                xjw::common::project::ProjectIO::resolveProjectResourcePath(
                    projectPath,
                    denseResults.at(index).toObject()
                        .value(QStringLiteral("dense_cloud_xyz")).toString());
            const QFileInfo candidateInfo(candidate);
            if (!candidate.isEmpty() && candidateInfo.exists() && candidateInfo.isFile())
            {
                dlg->setDefaultPointCloudPath(candidate);
                break;
            }
        }

        // 懒初始化 MapProject 记忆化管理器
        if (!_mapSetting)
        {
            _mapSetting = createDialogSettingStore(DialogSettingKeys::MapProject);
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

    connect(dlg, &MapProjectDialog::requestRunMapProject, dlg,
        [this, dialog = QPointer<MapProjectDialog>(dlg)](const QJsonObject &settings)
        {
        if (!_projectManager)
        {
            LOG_WARN(QStringLiteral("MapProject: 未找到 ProjectManager"));
            if (dialog)
            {
                dialog->onPipelineFinished(
                    false, QStringLiteral("项目管理器不可用，无法启动正射影像任务"));
            }
            return;
        }
        xjw::gui::project::OrthoGenerationRequest request;
        QString requestError;
        if (!xjw::gui::project::OrthoGenerationRequest::fromJson(
                settings, &request, &requestError))
        {
            if (dialog)
            {
                dialog->onPipelineFinished(false, requestError);
            }
            return;
        }
        _projectManager->startMapProjectAsync(request);
    });

    connect(dlg, &MapProjectDialog::requestCancelMapProject, this, [this]()
    {
        if (_projectManager)
        {
            _projectManager->cancelMapProject();
        }
    });

    if (_projectManager)
    {
        connect(_projectManager, &ProjectManager::orthoPipelineStarted,
                dlg, &MapProjectDialog::onPipelineStarted);
        connect(_projectManager, &ProjectManager::orthoPipelineProgressChanged,
                dlg, &MapProjectDialog::onPipelineProgress);
        connect(_projectManager, &ProjectManager::orthoPipelineFinished,
                dlg, &MapProjectDialog::onPipelineFinished);
    }

    dlg->exec();
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

void MenuWorkflowController::openCameraCalibrationDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    const QJsonObject metadata = _projectManager
        ? _projectManager->currentMeta()
        : QJsonObject();
    const QString assetsDir = _projectManager
        ? xjw::common::project::ProjectIO::projectAssetsDir(
              _projectManager->currentProjectPath())
        : QString();
    auto *dialog = new CameraCalibrationDialog(metadata, assetsDir, _mainWindow);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->show();
}
