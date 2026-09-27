#include "MenuWorkflowController.h"

#include "project/services/ProjectResourceService.h"
#include "project/services/ProjectSession.h"
#include "project/tasks/ProjectTaskOrchestrator.h"
#include "project/ProjectIO.h"
#include "project/ProjectCameraIO.h"
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

#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"

#include <QCheckBox>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFuture>
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
#include <QApplication>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

    std::vector<placamera::ImageId>
    cameraImageIdsForPaths(const QJsonObject& projectMeta, const QStringList& imagePaths, bool* allResolved)
    {
        if (allResolved)
        {
            *allResolved = true;
        }
        const QJsonArray entries = xjw::common::project::projectImageEntries(projectMeta);
        std::vector<placamera::ImageId> result;
        result.reserve(static_cast<std::size_t>(imagePaths.size()));
        QSet<QString> seen;
        for (const QString& imagePath : imagePaths)
        {
            QString imageId;
            for (const QJsonValue& value : entries)
            {
                const QJsonObject image = value.toObject();
                if (!xjw::common::project::pathTokenMatchesImage(image.value(QStringLiteral("path")).toString(),
                                                                 imagePath))
                {
                    continue;
                }
                imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
                break;
            }
            if (imageId.isEmpty() || seen.contains(imageId))
            {
                if (allResolved)
                {
                    *allResolved = false;
                }
                continue;
            }
            try
            {
                result.emplace_back(imageId.toStdString());
            }
            catch (const std::exception& exception)
            {
                if (allResolved)
                {
                    *allResolved = false;
                }
                LOG_WARN(QStringLiteral("空中三角测量: 影像 %1 的 ImageId 无效: %2")
                             .arg(imagePath, QString::fromUtf8(exception.what())));
                continue;
            }
            seen.insert(imageId);
        }
        if (allResolved && result.size() != static_cast<std::size_t>(imagePaths.size()))
        {
            *allResolved = false;
        }
        return result;
    }

    QStringList imagePathsForCameraInstances(const QJsonObject& projectMeta,
                                             const QStringList& candidatePaths,
                                             const placamera::CameraInstanceSet& instances)
    {
        QSet<QString> updateIds;
        for (const auto& camera : instances.values())
        {
            updateIds.insert(QString::fromStdString(camera->imageId().value()).trimmed());
        }

        const QJsonArray entries = xjw::common::project::projectImageEntries(projectMeta);
        QStringList result;
        for (const QString& candidate : candidatePaths)
        {
            for (const QJsonValue& value : entries)
            {
                const QJsonObject image = value.toObject();
                if (!xjw::common::project::pathTokenMatchesImage(image.value(QStringLiteral("path")).toString(),
                                                                 candidate))
                {
                    continue;
                }
                const QString imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
                if (updateIds.contains(imageId))
                {
                    result.append(xjw::common::project::normalizePath(candidate));
                }
                break;
            }
        }
        return result;
    }

    bool isSequenceReferencePreselection(const QJsonObject& settings)
    {
        if (!settings.value(QStringLiteral("reference_preselection")).toBool(false))
        {
            return false;
        }

        const QString source = settings.value(QStringLiteral("reference_preselection_source"))
                                   .toString(QStringLiteral("source_code"))
                                   .trimmed()
                                   .toLower();
        return source == QStringLiteral("sequence") || source == QStringLiteral("sequential") ||
               source == QStringLiteral("photo_sequence");
    }

    QString normalizedReferencePreselectionSource(const QJsonObject& settings)
    {
        const QString source = settings.value(QStringLiteral("reference_preselection_source"))
                                   .toString(QStringLiteral("source_code"))
                                   .trimmed()
                                   .toLower();
        return source == QStringLiteral("estimated_pose") ? QStringLiteral("estimated") : source;
    }

    placamera::reference::ReferenceCameraGeometryMap
    referenceCameraGeometriesForMode(xjw::gui::project::ProjectSession* session,
                                     const QStringList& images,
                                     const QJsonObject& projectMeta,
                                     const QString& requestedMode,
                                     bool* hasCamerasForAll)
    {
        if (hasCamerasForAll)
        {
            *hasCamerasForAll = false;
        }
        if (!session)
        {
            return {};
        }

        bool loadedAll = false;
        const placamera::reference::ReferenceCameraGeometryMap allGeometries =
            session->getReferenceCameraGeometriesForImages(images, &loadedAll);
        bool resolvedAllImageIds = false;
        const std::vector<placamera::ImageId> imageIds = session->getImageIdsForImages(images, &resolvedAllImageIds);

        const QString mode = requestedMode.trimmed().toLower() == QStringLiteral("estimated_pose")
                                 ? QStringLiteral("estimated")
                                 : requestedMode.trimmed().toLower();
        QMap<QString, QString> pose_sources;
        for (const QJsonValue& value : xjw::common::project::projectFilesRootObject(projectMeta)
                                           .value(QStringLiteral("camera_instances"))
                                           .toArray())
        {
            const QJsonObject instance = value.toObject();
            const QJsonObject state = instance.value(QStringLiteral("state")).toObject();
            QString source = state.value(QStringLiteral("pose_source")).toString();
            if (source.isEmpty())
            {
                source =
                    state.value(QStringLiteral("metadata")).toObject().value(QStringLiteral("pose_source")).toString();
            }
            pose_sources.insert(instance.value(QStringLiteral("image_uuid")).toString(), source.trimmed().toLower());
        }
        placamera::reference::ReferenceCameraGeometryMap filtered;
        if (!placamera::reference::validateReferenceCameraGeometryMap(allGeometries))
        {
            return filtered;
        }
        for (int index = 0; index < images.size(); ++index)
        {
            if (static_cast<std::size_t>(index) >= imageIds.size())
            {
                continue;
            }
            const auto cameraIt = allGeometries.find(imageIds[static_cast<std::size_t>(index)]);
            if (cameraIt == allGeometries.cend())
            {
                continue;
            }

            const QString poseSource =
                pose_sources.value(QString::fromStdString(imageIds[static_cast<std::size_t>(index)].value()));
            const bool sfmEstimated = poseSource == QStringLiteral("sfm_estimated");
            const bool sourceMode = mode == QStringLiteral("source") || mode == QStringLiteral("source_code");
            if ((mode == QStringLiteral("estimated") && !sfmEstimated) || (sourceMode && sfmEstimated))
            {
                continue;
            }
            filtered.emplace(cameraIt->first, cameraIt->second);
        }

        if (hasCamerasForAll)
        {
            *hasCamerasForAll =
                loadedAll && resolvedAllImageIds && filtered.size() == static_cast<std::size_t>(images.size());
        }
        return filtered;
    }

    bool shouldUseStoredGeneratedPairConstraints(const QJsonObject& settings)
    {
        if (settings.value(QStringLiteral("reference_preselection")).toBool(false))
        {
            // 参考来源是用户本次显式选择的配对先验。历史 generated_pairs 多来自另一次
            // 词汇树/重叠配置，不能继续作为 ManualOnly 白名单覆盖当前位姿或序列策略。
            return false;
        }
        return true;
    }

    float normalizedFeatureGrayscaleMin(const QJsonObject& settings)
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
    QStringList loadGeneratedPairConstraints(const QString& projectPath,
                                             const QJsonObject& projectMeta,
                                             const QStringList& selectedImages,
                                             bool* usedStoredPairs,
                                             bool* storedPairsStale)
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
        for (const QString& imagePath : selectedImages)
        {
            selectedSet.insert(xjw::common::project::normalizePath(imagePath));
        }

        QStringList allowedPairs;
        QSet<QString> seenPairs;
        for (const QJsonValue& value : generatedPairs)
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

            const QString pairKey = xjw::common::project::canonicalImagePairKey(normA, normB, QStringLiteral("\n"));
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

MenuWorkflowController::MenuWorkflowController(QMainWindow* mainWindow, QObject* parent)
    : QObject(parent), _mainWindow(mainWindow),
      _featureVisualizationController(new FeatureVisualizationController(mainWindow, this))
{
    connect(_featureVisualizationController,
            &FeatureVisualizationController::optionsChanged,
            this,
            &MenuWorkflowController::requestApplyFeatureDisplayOptions);
}

void MenuWorkflowController::setProjectServices(xjw::gui::project::ProjectSession* session,
                                                xjw::gui::project::ProjectTaskOrchestrator* tasks,
                                                xjw::gui::project::ProjectResourceService* resources)
{
    _session = session;
    _tasks = tasks;
    _resources = resources;
    _featureVisualizationController->setProjectSession(session);
}

DialogSettingStore* MenuWorkflowController::createDialogSettingStore(const QString& settingKey)
{
    auto* store = new DialogSettingStore(settingKey, this);
    store->setChangeCallback(
        [this]()
        {
            if (_session)
            {
                _session->markWorkspaceDirty();
            }
        });
    return store;
}

void MenuWorkflowController::bindActions(MainMenu* mainMenu)
{
    if (!mainMenu)
    {
        return;
    }

    auto connectAction = [this](QAction* action, void (MenuWorkflowController::*slot)())
    {
        if (action)
        {
            connect(action, &QAction::triggered, this, slot, Qt::UniqueConnection);
        }
    };

    connectAction(mainMenu->workflowAerialTriangulationAction(),
                  &MenuWorkflowController::openWorkflowAerialTriangulationDialog);
    connectAction(mainMenu->workflowSettingsAction(), &MenuWorkflowController::openWorkflowSettingsDialog);
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
    connectAction(mainMenu->importCameraAction(), &MenuWorkflowController::openCameraCalibrationDialog);
    connectAction(mainMenu->cameraCalibrationAction(), &MenuWorkflowController::openCameraCalibrationDialog);

    if (!_session || !_tasks || !_resources)
    {
        return;
    }

    auto connectProjectAction = [this](QAction* action, const std::function<void()>& slot)
    {
        if (action)
        {
            connect(action, &QAction::triggered, this, slot, Qt::UniqueConnection);
        }
    };

    connectProjectAction(mainMenu->importReferenceDatasetAction(), [this]() { _resources->importReferenceDataset(); });
    connectProjectAction(mainMenu->surveyControlAction(), [this]() { _resources->openSurveyControlDialog(); });
    connectProjectAction(mainMenu->generateMaskAction(), [this]() { _tasks->openGenerateMaskDialog(); });
    connectProjectAction(mainMenu->referenceQualityCheckAction(), [this]() { _resources->runReferenceQualityCheck(); });
    connectProjectAction(mainMenu->referenceTerrainBundleAdjustAction(),
                         [this]() { _resources->prepareReferenceTerrainBundleAdjust(); });
}

QStringList MenuWorkflowController::getProjectImages() const
{
    if (!_session)
    {
        return QStringList();
    }

    QStringList images = _session->imagesByCategory(QStringLiteral("源数据"));
    if (images.isEmpty())
        images = _session->imagesByCategory(QStringLiteral("照片"));
    if (images.isEmpty())
        images = _session->imagesByCategory(QStringLiteral("Photos"));
    if (images.isEmpty())
        images = _session->allImages();
    return images;
}

MenuWorkflowController::SparsePrerequisiteSummary
MenuWorkflowController::summarizeSparsePrerequisites(const QStringList& images,
                                                     const QJsonObject& meta,
                                                     const QString& projectPath,
                                                     const QString& algorithmId,
                                                     const std::function<void(int, int)>& progressCallback)
{
    SparsePrerequisiteSummary summary;
    summary.imageCount = images.size();
    // 新匹配链路只有组合算法标识，不再把“特征算法”和“匹配算法”拆成两个
    // 自由字符串。SIFT 描述子只驻留任务内存，预检也不再查找特征中间文件。
    const QString selectedAlgorithmId =
        algorithmId.trimmed().isEmpty() ? QStringLiteral("plamatch_hct") : algorithmId.trimmed().toLower();

    auto nameAliases = [](const QString& pathOrName) -> QStringList
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

    auto matchPairKey = [](const QString& left, const QString& right) -> QString
    {
        if (left.isEmpty() || right.isEmpty() || left == right)
        {
            return QString();
        }
        return (left < right) ? (left + QStringLiteral("\n") + right) : (right + QStringLiteral("\n") + left);
    };

    auto variantAlgorithmMatches = [&](const xjw::aerial_triangulation::MatchVariant& variant) -> bool
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
        catalogConfig.matchDirectory = xjw::common::project::ProjectIO::imageMatchOutputDir(projectPath);
        catalogConfig.targetImagePaths = images;
        catalogConfig.progressCallback = progressCallback;
        catalogSummary = xjw::aerial_triangulation::MatchResultCatalog(catalogConfig).scan();
        LOG_INFO(QStringLiteral("空三上游索引: 分片=%1 内存命中=%2 持久索引命中=%3 首次重建=%4 损坏=%5")
                     .arg(catalogSummary.matchFileCount)
                     .arg(catalogSummary.memoryIndexHitCount)
                     .arg(catalogSummary.persistentIndexHitCount)
                     .arg(catalogSummary.rebuiltIndexCount)
                     .arg(catalogSummary.incompatibleVariantCount));
    }

    auto appendPreflightPair = [&](const QString& leftToken, const QString& rightToken)
    {
        const QStringList leftAliases = nameAliases(leftToken);
        const QStringList rightAliases = nameAliases(rightToken);
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
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
        for (const xjw::aerial_triangulation::MatchPairGroup& group : catalogSummary.pairGroups)
        {
            for (const xjw::aerial_triangulation::MatchVariant& variant : group.variants)
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

    for (const auto& pair : matchedPairs)
    {
        const QStringList leftAliases = nameAliases(pair.first);
        const QStringList rightAliases = nameAliases(pair.second);
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
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
    auto appendSettledNoMatchPair = [&](const QString& leftToken, const QString& rightToken)
    {
        const QStringList leftAliases = nameAliases(leftToken);
        const QStringList rightAliases = nameAliases(rightToken);
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
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
        for (const auto& group : catalogSummary.pairGroups)
        {
            const auto settled =
                std::find_if(group.variants.cbegin(),
                             group.variants.cend(),
                             [&](const xjw::aerial_triangulation::MatchVariant& variant)
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

    for (const auto& pair : settledNoMatchPairs)
    {
        const QStringList leftAliases = nameAliases(pair.first);
        const QStringList rightAliases = nameAliases(pair.second);
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
            {
                const QString key = matchPairKey(left, right);
                if (!key.isEmpty())
                {
                    processedPairKeys.insert(key);
                }
            }
        }
    }

    auto pairCoveredByAliases = [&](const QStringList& leftAliases, const QStringList& rightAliases) -> bool
    {
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
            {
                if (matchedPairKeys.contains(matchPairKey(left, right)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto pairProcessedByAliases = [&](const QStringList& leftAliases, const QStringList& rightAliases) -> bool
    {
        for (const QString& left : leftAliases)
        {
            for (const QString& right : rightAliases)
            {
                if (processedPairKeys.contains(matchPairKey(left, right)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto imagePairCovered = [&](const QString& leftImage, const QString& rightImage) -> bool
    { return pairCoveredByAliases(nameAliases(leftImage), nameAliases(rightImage)); };

    auto imagePairProcessed = [&](const QString& leftImage, const QString& rightImage) -> bool
    { return pairProcessedByAliases(nameAliases(leftImage), nameAliases(rightImage)); };

    bool usedStoredPairs = false;
    bool storedPairsStale = false;
    const QStringList generatedPairs =
        loadGeneratedPairConstraints(projectPath, meta, images, &usedStoredPairs, &storedPairsStale);
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
            for (const QString& alias : aliases)
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

        auto resolveImageIndex = [&](const QString& pathOrName) -> int
        {
            const QStringList aliases = nameAliases(pathOrName);
            for (const QString& alias : aliases)
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
        for (const auto& pair : matchedPairs)
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
        for (const QString& pairKey : generatedPairs)
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
    const bool generatedPlanHasNoProcessedPairs =
        usedStoredPairs && !storedPairsStale && generatedPairRequiredCount > 0 && generatedPairProcessedCount == 0;
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
        recommendedAction ==
        xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction::InspectMatchQuality;
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
    for (const QString& warning : summary.warningMessages)
    {
        LOG_WARN(QStringLiteral("空中三角测量预检: %1").arg(warning));
    }
    return summary;
}

void MenuWorkflowController::applySavedFeatureDisplayOptions(const QJsonObject& uiSettings)
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
    const int cameraCount =
        _session
            ? static_cast<int>(_session->getReferenceCameraGeometriesForImages(images, &hasAllReferenceCameras).size())
            : 0;
    dlg.setReferencePreselectionAvailable(
        hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2, cameraCount, images.size());

    if (!_aerialTriangulationSetting)
    {
        _aerialTriangulationSetting = createDialogSettingStore(DialogSettingKeys::AerialTriangulation);
    }

    if (_session)
    {
        const QString projectPath = _session->projectPath();
        _aerialTriangulationSetting->setProjectPath(projectPath);
        dlg.applySettings(_aerialTriangulationSetting->load());
        const QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(projectPath);
        const QString tiePointPath =
            QDir(projectRoot).filePath(QStringLiteral("assets/tie_points/latest_tie_points.json"));
        const std::optional<int> cachedLimit =
            xjw::aerial_triangulation::AerialTriangulationWorkflow::storedTiePointLimit(tiePointPath);
        dlg.setCachedTiePointLimit(QFileInfo::exists(tiePointPath), cachedLimit.value_or(-1));
    }

    connect(&dlg,
            &AerialTriangulationDialog::settingsChanged,
            this,
            [this](const QJsonObject& settings)
            {
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
        startAerialTriangulationWorkflow(mergeAerialTriangulationSettings(dialogSettings));
    }
}

void MenuWorkflowController::openWorkflowSettingsDialog()
{
    if (!_mainWindow)
    {
        return;
    }
    if (!_session || _session->projectPath().trimmed().isEmpty())
    {
        QMessageBox::warning(
            _mainWindow, QStringLiteral("工作流程设置"), QStringLiteral("请先打开项目。工作流程设置按项目保存。"));
        return;
    }

    if (!_workflowSettingsStore)
    {
        _workflowSettingsStore = createDialogSettingStore(DialogSettingKeys::WorkflowSettings);
    }
    _workflowSettingsStore->setProjectPath(_session->projectPath());

    WorkflowSettingsDialog dialog(_mainWindow);
    dialog.applySettings(_workflowSettingsStore->load());
    if (dialog.exec() == QDialog::Accepted)
    {
        QString saveError;
        if (!_workflowSettingsStore->save(dialog.collectSettings(), &saveError))
        {
            QMessageBox::warning(_mainWindow,
                                 QStringLiteral("工作流程设置"),
                                 saveError.isEmpty() ? QStringLiteral("无法保存工作流程设置。") : saveError);
        }
    }
}

QJsonObject MenuWorkflowController::mergeAerialTriangulationSettings(const QJsonObject& dialogSettings)
{
    QJsonObject merged;
    const QJsonObject defaultAerialSettings =
        WorkflowSettingsDialog::aerialTriangulationSettings(WorkflowSettingsDialog::defaultSettings());
    for (auto it = defaultAerialSettings.constBegin(); it != defaultAerialSettings.constEnd(); ++it)
    {
        merged.insert(it.key(), it.value());
    }
    if (_session)
    {
        if (!_workflowSettingsStore)
        {
            _workflowSettingsStore = createDialogSettingStore(DialogSettingKeys::WorkflowSettings);
        }
        _workflowSettingsStore->setProjectPath(_session->projectPath());
        const QJsonObject savedAerialSettings =
            WorkflowSettingsDialog::aerialTriangulationSettings(_workflowSettingsStore->load());
        for (auto it = savedAerialSettings.constBegin(); it != savedAerialSettings.constEnd(); ++it)
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
    const QJsonObject& requestedSettings, const QStringList& images, const QJsonObject& projectMeta) const
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
    const int cameraCount =
        referenceCameraGeometriesForMode(_session, images, projectMeta, referenceMode, &hasAllReferenceCameras).size();
    const bool available = hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2;
    if (!available)
    {
        const QString algorithmId =
            settings.value(QStringLiteral("algorithm_id")).toString(QStringLiteral("plamatch_hct")).trimmed().toLower();
        if (algorithmId == QStringLiteral("plamatch_hct"))
        {
            LOG_INFO(QStringLiteral("空中三角测量: %1 参考位姿为 %2/%3；PlaMatch 将使用已有坐标，"
                                    "坐标集合为空时回退索引邻域。")
                         .arg(referenceMode)
                         .arg(cameraCount)
                         .arg(images.size()));
            return settings;
        }
        settings[QStringLiteral("reference_preselection")] = false;
        LOG_WARN(QStringLiteral("空中三角测量: %1 参考位姿不完整，参考预选已关闭（相机 %2/%3）。")
                     .arg(referenceMode)
                     .arg(cameraCount)
                     .arg(images.size()));
    }
    return settings;
}

void MenuWorkflowController::startAerialTriangulationWorkflow(const QJsonObject& settings)
{
    if (!_session || !_tasks || _session->projectPath().trimmed().isEmpty())
    {
        QMessageBox::warning(_mainWindow, QStringLiteral("空中三角测量"), QStringLiteral("请先打开项目"));
        return;
    }

    if (_tasks->hasActiveTask())
    {
        QMessageBox::information(_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 QStringLiteral("已有空三或光束法平差任务正在运行，请等待其结束或先取消当前任务。"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(
            _mainWindow, QStringLiteral("空中三角测量"), QStringLiteral("至少需要 2 张影像才能进行空中三角测量。"));
        return;
    }

    const QJsonObject projectMeta = _session->metadata();
    const bool hasDepthMaps = !projectMeta.value(QStringLiteral("depth_map_results")).toArray().isEmpty();
    QSettings warningSettings(QStringLiteral("PlaScan"), QStringLiteral("plascan_gui"));
    const bool suppressDepthInvalidationWarning =
        warningSettings.value(QStringLiteral("Warnings/suppressDepthMapInvalidationBeforeAerialTriangulation"), false)
            .toBool();
    if (hasDepthMaps && !suppressDepthInvalidationWarning)
    {
        QMessageBox confirmation(_mainWindow);
        confirmation.setWindowTitle(QStringLiteral("空中三角测量"));
        confirmation.setIcon(QMessageBox::Warning);
        confirmation.setText(QStringLiteral("当前深度图将在空中三角测量成功后失效并从项目中移除。是否继续？"));
        confirmation.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        confirmation.setDefaultButton(QMessageBox::No);
        auto* dontShowAgain = new QCheckBox(QStringLiteral("不再显示该信息"), &confirmation);
        confirmation.setCheckBox(dontShowAgain);
        const auto answer = static_cast<QMessageBox::StandardButton>(confirmation.exec());
        if (answer != QMessageBox::Yes)
        {
            return;
        }
        if (dontShowAgain->isChecked())
        {
            warningSettings.setValue(QStringLiteral("Warnings/suppressDepthMapInvalidationBeforeAerialTriangulation"),
                                     true);
        }
    }

    const auto session = _session->context();
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
        runSettings.value(QStringLiteral("algorithm_id")).toString(QStringLiteral("plamatch_hct")).trimmed().toLower();

    const QString task_id = QStringLiteral("aerial_triangulation");
    if (!_tasks->beginTask(task_id))
    {
        QMessageBox::information(_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 QStringLiteral("已有空三或光束法平差任务正在运行，请等待其结束或先取消当前任务。"));
        return;
    }
    const xjw::gui::project::ProjectTaskContext task_context = _tasks->context(task_id);
    if (!task_context.cancelFlag)
    {
        _tasks->finishTask(task_context, false, QStringLiteral("无法创建空三任务上下文"));
        return;
    }
    _tasks->reportSparseProgress(task_context, QStringLiteral("空中三角测量: 检查上游数据..."), 0);

    QPointer<xjw::gui::project::ProjectTaskOrchestrator> task_guard(_tasks);
    const auto preflightProgress = [task_guard, task_context](int processed, int total)
    {
        if (!task_guard || !task_guard->isTaskActive(task_context))
        {
            return;
        }
        const int percent =
            total <= 0 ? 100 : qBound(0, static_cast<int>((static_cast<qint64>(processed) * 100) / total), 100);
        xjw::gui::tasks::postGuarded(
            task_guard,
            [task_context, processed, total, percent](xjw::gui::project::ProjectTaskOrchestrator* tasks)
            {
                tasks->reportSparseProgress(
                    task_context,
                    QStringLiteral("空中三角测量: 检查上游匹配索引 %1/%2").arg(processed).arg(total),
                    percent);
            });
    };
    QFuture<void> preflight_future = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [images, projectMeta, projectPath, selectedAlgorithmId, preflightProgress]()
        {
            return MenuWorkflowController::summarizeSparsePrerequisites(
                images, projectMeta, projectPath, selectedAlgorithmId, preflightProgress);
        },
        [task_guard, task_context, runSettings, images, projectMeta, outputRoot](
            MenuWorkflowController* controller, xjw::gui::tasks::TaskOutcome<SparsePrerequisiteSummary> outcome)
        {
            if (!task_guard || !task_guard->isTaskActive(task_context))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                task_guard->finishTask(task_context, false, outcome.errorMessage);
                QMessageBox::warning(controller->_mainWindow, QStringLiteral("空中三角测量"), outcome.errorMessage);
                return;
            }

            const auto& prereq = *outcome.value;
            if (!prereq.prerequisiteReport.isEmpty())
            {
                LOG_INFO(QStringLiteral("空三前置报告: %1")
                             .arg(QString::fromUtf8(
                                 QJsonDocument(prereq.prerequisiteReport).toJson(QJsonDocument::Compact))));
            }

            bool autoFillMissing = false;
            const bool reuseExistingMatches = runSettings.value(QStringLiteral("reuse_existing_matches")).toBool(true);
            if (prereq.blockOnMatchQuality && reuseExistingMatches)
            {
                task_guard->finishTask(task_context, false);
                const QString details =
                    prereq.warningMessages.isEmpty()
                        ? QStringLiteral(
                              "匹配阶段已完成，但没有可用于空三的连接边；请检查匹配参数、重叠对和几何验证报告。")
                        : prereq.warningMessages.join(QStringLiteral("\n"));
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral("%1\n\n当前已选择“重用现有匹配”，不会自动重新跑完整匹配。"
                                                    "如需重建匹配，请在高级设置中取消该选项。")
                                         .arg(details));
                return;
            }
            if (!prereq.missingMessages.isEmpty())
            {
                autoFillMissing = true;
                LOG_INFO(QStringLiteral(
                    "空中三角测量: 检测到缺少连接点输入，直接运行创建连接点流程；该流程会自动提取特征并匹配。"));
            }

            controller->runUnifiedAerialTriangulation(
                runSettings, images, task_context, projectMeta, outputRoot, autoFillMissing);
        });
    _tasks->trackExternalSparseFuture(task_context, std::move(preflight_future));
}

void MenuWorkflowController::runUnifiedAerialTriangulation(const QJsonObject& settings,
                                                           const QStringList& images,
                                                           const xjw::gui::project::ProjectTaskContext& taskContext,
                                                           const QJsonObject& projectMeta,
                                                           const QString& outputRoot,
                                                           bool fillMissingTiePoints)
{
    if (!_session || !_tasks || !_tasks->isTaskActive(taskContext))
    {
        return;
    }

    const QString projectPath = taskContext.session.projectPath;

    const bool resetCurrentAlignment = settings.value(QStringLiteral("reset_current_alignment")).toBool(true);

    // 线程数属于机器运行时能力，不能复用项目里由另一台电脑保存的历史固定值。
    // 统一传 0，由核心层在每次启动时按当前机器逻辑线程数解析。
    constexpr int workflowThreads = 0;
    xjw::aerial_triangulation::AerialTriangulationOptions workflowOptions;
    workflowOptions.images = images;
    bool allImageIdsResolved = false;
    workflowOptions.imageIds = _session->getImageIdsForImages(images, &allImageIdsResolved);
    if (!allImageIdsResolved)
    {
        LOG_WARN(QStringLiteral("空中三角测量: 当前影像缺少稳定 ImageId，参考几何将被拒绝"));
    }
    workflowOptions.projectPath = projectPath;
    workflowOptions.projectMeta = projectMeta;
    workflowOptions.outputDir = outputRoot;
    workflowOptions.quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("high"));
    workflowOptions.genericPreselection = settings.value(QStringLiteral("generic_preselection")).toBool(true);
    workflowOptions.referencePreselection = settings.value(QStringLiteral("reference_preselection")).toBool(false);
    workflowOptions.referenceMode =
        settings.value(QStringLiteral("reference_preselection_source")).toString(QStringLiteral("source_code"));
    workflowOptions.resetAlignment = settings.value(QStringLiteral("reset_current_alignment")).toBool(true);
    workflowOptions.reuseExistingMatches = settings.value(QStringLiteral("reuse_existing_matches")).toBool(true);
    workflowOptions.lockInputCameraPoses = settings.value(QStringLiteral("lock_input_camera_poses")).toBool(false);
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
        settings.value(QStringLiteral("algorithm_id")).toString(QStringLiteral("plamatch_hct")).trimmed().toLower();
    workflowOptions.lightGlueTensorRtEnginePath =
        settings.value(QStringLiteral("lightglue_tensorrt_engine")).toString().trimmed();
    workflowOptions.lomaRTensorRtPackagePath =
        settings.value(QStringLiteral("loma_r_tensorrt_package")).toString().trimmed();
    workflowOptions.lomaRKeypointBudget = settings.value(QStringLiteral("loma_r_keypoint_budget")).toInt(0);
    workflowOptions.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    workflowOptions.threads = workflowThreads;
    workflowOptions.cudaDevice = std::max(0, settings.value(QStringLiteral("cuda_device")).toInt(0));
    workflowOptions.featureMaxImageDim = std::max(0, settings.value(QStringLiteral("feature_max_image_dim")).toInt(0));
    workflowOptions.cudaParallelPairs = std::max(0, settings.value(QStringLiteral("cuda_parallel_pairs")).toInt(0));
    workflowOptions.featurePrefetchDepth =
        std::clamp(settings.value(QStringLiteral("feature_prefetch_depth")).toInt(2), 1, 4);
    workflowOptions.matchThreshold =
        static_cast<float>(std::clamp(settings.value(QStringLiteral("match_threshold")).toDouble(0.15), 0.0, 1.0));
    workflowOptions.siftMaximumRatio =
        static_cast<float>(std::clamp(settings.value(QStringLiteral("sift_maximum_ratio")).toDouble(0.98), 0.0, 1.0));
    workflowOptions.siftMinimumAdaptiveRatio =
        static_cast<float>(std::clamp(settings.value(QStringLiteral("sift_minimum_adaptive_ratio")).toDouble(0.78),
                                      0.0,
                                      static_cast<double>(workflowOptions.siftMaximumRatio)));
    workflowOptions.adaptiveSiftRatio = settings.value(QStringLiteral("adaptive_sift_ratio")).toBool(true);
    workflowOptions.geometryReprojThreshold =
        std::max(0.1, settings.value(QStringLiteral("geometry_reprojection_threshold_px")).toDouble(1.5));
    workflowOptions.geometryMinInliers = std::max(8, settings.value(QStringLiteral("geometry_min_inliers")).toInt(20));
    workflowOptions.geometryMinInlierRatio =
        std::clamp(settings.value(QStringLiteral("geometry_min_inlier_ratio")).toDouble(0.18), 0.01, 0.95);
    workflowOptions.geometryMinGridCoverage =
        std::clamp(settings.value(QStringLiteral("geometry_min_grid_coverage")).toDouble(0.12), 0.01, 1.0);
    workflowOptions.geometryMaxIterations =
        std::max(100, settings.value(QStringLiteral("geometry_max_iterations")).toInt(10000));
    workflowOptions.tiePointGridColumns =
        std::clamp(settings.value(QStringLiteral("tie_point_grid_columns")).toInt(8), 1, 64);
    workflowOptions.tiePointGridRows =
        std::clamp(settings.value(QStringLiteral("tie_point_grid_rows")).toInt(8), 1, 64);
    workflowOptions.maxTiePointsPerGridCell =
        std::max(0, settings.value(QStringLiteral("tie_point_grid_cell_limit")).toInt(0));
    workflowOptions.stationaryTiePointMaxPixelMotion = static_cast<float>(
        std::max(0.0, settings.value(QStringLiteral("stationary_tie_point_max_pixel_motion")).toDouble(1.0)));
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
        workflowOptions.referenceCameraGeometries = referenceCameraGeometriesForMode(
            _session, images, projectMeta, workflowOptions.referenceMode, &hasAllReferenceCameras);
        if (!hasAllReferenceCameras)
        {
            LOG_WARN(
                QStringLiteral("空中三角测量: 参考预选已启用，但 %1 位姿不完整").arg(workflowOptions.referenceMode));
        }
        else
        {
            LOG_INFO(QStringLiteral("空中三角测量: 已加载 %1 个 %2 参考位姿用于候选对规划")
                         .arg(static_cast<int>(workflowOptions.referenceCameraGeometries.size()))
                         .arg(workflowOptions.referenceMode));
        }
    }

    bool usedStoredPairs = false;
    bool storedPairsStale = false;
    const bool useStoredGeneratedPairs = shouldUseStoredGeneratedPairConstraints(settings);
    const QStringList allowedPairs =
        useStoredGeneratedPairs
            ? loadGeneratedPairConstraints(projectPath, projectMeta, images, &usedStoredPairs, &storedPairsStale)
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

    QPointer<xjw::gui::project::ProjectTaskOrchestrator> task_guard(_tasks);
    QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    workflowOptions.progressFn = [task_guard, taskContext](const QString& stage, int percent)
    {
        if (!task_guard || !task_guard->isTaskActive(taskContext))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(
            task_guard,
            [taskContext, stage, percent](xjw::gui::project::ProjectTaskOrchestrator* tasks)
            { tasks->reportSparseProgress(taskContext, QStringLiteral("空中三角测量: %1").arg(stage), percent); });
    };
    workflowOptions.computeDeviceFn = [task_guard, taskContext](const QString& displayName)
    {
        if (!task_guard || !task_guard->isTaskActive(taskContext))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(task_guard,
                                     [taskContext, displayName](xjw::gui::project::ProjectTaskOrchestrator* tasks)
                                     { tasks->reportSparseComputeDevice(taskContext, displayName); });
    };
    workflowOptions.pairMatchedFn =
        [task_guard, taskContext](const QString& img0, const QString& img1, const QString& matchPath, int numMatches)
    {
        if (!task_guard || !task_guard->isTaskActive(taskContext))
        {
            return;
        }
        xjw::gui::tasks::postGuarded(
            task_guard,
            [taskContext, img0, img1, matchPath, numMatches](xjw::gui::project::ProjectTaskOrchestrator* tasks)
            { tasks->reportSparseMatchPair(taskContext, img0, img1, matchPath, numMatches); });
    };

    workflowOptions.cancelFlag = taskContext.cancelFlag;
    const xjw::aerial_triangulation::AerialTriangulationResolvedConfig resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(workflowOptions);

    _tasks->reportSparseProgress(taskContext,
                                 resolved.prepareTiePoints ? QStringLiteral("空中三角测量: 准备连接点...")
                                                           : QStringLiteral("空中三角测量: 启动 SfM/BA..."),
                                 0);

    const QStringList sfmImages = images;
    const QString sfmOutputDir = resolved.pipelineInput.outputDir;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    QFuture<void> workflow_future = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [runWorkflowOptions = std::move(workflowOptions)]() mutable
        { return xjw::aerial_triangulation::AerialTriangulationWorkflow::run(runWorkflowOptions); },
        [task_guard,
         session_guard,
         taskContext,
         sfmImages,
         sfmOutputDir,
         assetsDir,
         projectMeta,
         resetCurrentAlignment](
            MenuWorkflowController* controller,
            xjw::gui::tasks::TaskOutcome<xjw::aerial_triangulation::AerialTriangulationResult> outcome) mutable
        {
            if (!task_guard || !session_guard || !task_guard->isTaskActive(taskContext))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                task_guard->finishTask(taskContext, false, outcome.errorMessage);
                QMessageBox::warning(controller->_mainWindow, QStringLiteral("空中三角测量"), outcome.errorMessage);
                return;
            }
            auto workflowResult = std::move(*outcome.value);
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult& result =
                workflowResult.reconstructionResult;
            const bool wasCanceled = taskContext.cancelFlag->load(std::memory_order_relaxed);
            if (wasCanceled)
            {
                task_guard->finishTask(taskContext, false);
                return;
            }

            if (workflowResult.tiePointPreparationExecuted)
            {
                QString match_write_error;
                if (!session_guard->appendImageMatchResults(
                        taskContext.session,
                        xjw::gui::project::makeImageMatchResultRecords(workflowResult.tiePointResult),
                        &match_write_error))
                {
                    task_guard->finishTask(taskContext, false, match_write_error);
                    QMessageBox::warning(controller->_mainWindow,
                                         QStringLiteral("空中三角测量"),
                                         match_write_error.isEmpty() ? QStringLiteral("连接点匹配结果写回失败。")
                                                                     : match_write_error);
                    return;
                }
            }

            if (!result.success)
            {
                task_guard->finishTask(taskContext, false, result.errorMessage);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     result.errorMessage.isEmpty() ? QStringLiteral("空中三角测量失败。")
                                                                   : result.errorMessage);
                return;
            }

            int registeredImageCount = result.numRegisteredImages;
            QJsonObject resultRecordExtra = result.resultRecordExtra;
            QString sparseBlockingReason = QStringLiteral("SFM 未生成可用的正式稀疏点云。");
            QStringList registeredImages;
            if (result.success && !result.sparseCloudPath.isEmpty())
            {
                registeredImages = imagePathsForCameraInstances(projectMeta, sfmImages, result.cameraInstances);
                registeredImageCount = registeredImages.size();

                resultRecordExtra[QStringLiteral("source")] = QStringLiteral("aerial_triangulation");
                sparseBlockingReason = xjw::gui::project::sparseResultBlockingReason(resultRecordExtra);
            }

            if (!xjw::gui::project::isProductionSparseResult(resultRecordExtra))
            {
                task_guard->finishTask(taskContext, false, sparseBlockingReason);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     sparseBlockingReason.isEmpty() ? QStringLiteral("当前 SfM/BA 稀疏点云质量不足。")
                                                                    : sparseBlockingReason);
                return;
            }

            const bool rpcAerialTriangulation =
                resultRecordExtra.value(QStringLiteral("camera_model")).toString() == QStringLiteral("rpc00b");

            // 只有正式空三结果才写回工程，避免失败候选污染相机状态。
            bool allTargetImageIdsResolved = false;
            const std::vector<placamera::ImageId> targetImageIds =
                cameraImageIdsForPaths(projectMeta, sfmImages, &allTargetImageIdsResolved);
            if (!allTargetImageIdsResolved)
            {
                LOG_WARN(QStringLiteral("空中三角测量: 工程影像身份不完整，拒绝按路径回退写回相机"));
                if (!result.cameraInstances.empty())
                {
                    task_guard->finishTask(taskContext, false, QStringLiteral("工程影像身份不完整"));
                    QMessageBox::warning(
                        controller->_mainWindow,
                        QStringLiteral("空中三角测量"),
                        QStringLiteral("工程影像身份不完整，已拒绝提交空三相机结果；请重新加载工程后重试。"));
                    return;
                }
            }
            bool cameraWritebackOk = true;
            if (resetCurrentAlignment)
            {
                int updated = 0;
                int cleared = 0;
                QString err;
                if (allTargetImageIdsResolved &&
                    !session_guard->replaceNativeCameraInstances(taskContext.session,
                                                                 targetImageIds,
                                                                 result.cameraInstances,
                                                                 result.cameraAnnotationsByImageId,
                                                                 &updated,
                                                                 &cleared,
                                                                 &err))
                {
                    LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
                    cameraWritebackOk = false;
                }
                else if (allTargetImageIdsResolved)
                {
                    LOG_INFO(QStringLiteral("空中三角测量: 相机对齐状态已刷新，注册 %1，清除旧位姿 %2")
                                 .arg(updated)
                                 .arg(cleared));
                }
            }
            else if (allTargetImageIdsResolved && !result.cameraInstances.empty())
            {
                int updated = 0;
                QString err;
                if (!session_guard->upsertNativeCameraInstances(
                        taskContext.session, result.cameraInstances, result.cameraAnnotationsByImageId, &updated, &err))
                {
                    LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
                    cameraWritebackOk = false;
                }
            }
            if (!cameraWritebackOk)
            {
                task_guard->finishTask(taskContext, false, QStringLiteral("相机结果写回失败"));
                QMessageBox::warning(
                    controller->_mainWindow,
                    QStringLiteral("空中三角测量"),
                    QStringLiteral("相机结果未能通过 canonical identity/frame 校验，已拒绝提交空三结果。"));
                return;
            }

            const xjw::gui::project::TiePointMutationResult tie_point_result =
                session_guard->replaceTiePointResult(taskContext.session,
                                                     result.sparseCloudPath,
                                                     result.numPoints3D,
                                                     registeredImages,
                                                     sfmOutputDir,
                                                     resultRecordExtra);
            if (!tie_point_result.success)
            {
                task_guard->finishTask(taskContext, false, tie_point_result.errorMessage);
                QMessageBox::warning(controller->_mainWindow,
                                     QStringLiteral("空中三角测量"),
                                     tie_point_result.errorMessage.isEmpty() ? QStringLiteral("连接点结果写回失败。")
                                                                             : tie_point_result.errorMessage);
                return;
            }
            if (!tie_point_result.cleanupWarnings.isEmpty())
            {
                LOG_WARN(QStringLiteral("当前连接点已更新，但旧文件清理失败: %1")
                             .arg(tie_point_result.cleanupWarnings.join(QStringLiteral("；"))));
            }

            if (!assetsDir.isEmpty())
            {
                QJsonObject report;
                report[QStringLiteral("type")] = QStringLiteral("aerial_triangulation_sfm");
                report[QStringLiteral("mode")] = rpcAerialTriangulation ? QStringLiteral("rpc") : QStringLiteral("sfm");
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
                        projectMeta, result.cameraInstances, result.sfmDiagnostics);
                report[QStringLiteral("camera_calibration_semantics")] =
                    QJsonObject{{QStringLiteral("initial"), QStringLiteral("intrinsics_at_alignment_start")},
                                {QStringLiteral("adjusted"), QStringLiteral("intrinsics_after_bundle_adjustment")},
                                {QStringLiteral("principal_point"), QStringLiteral("offset_from_image_center")},
                                {QStringLiteral("excludes_extrinsics"), true}};
                report[QStringLiteral("resolved_settings")] = workflowResult.config.resolvedSettings;
                report[QStringLiteral("tie_point_preparation_executed")] = workflowResult.tiePointPreparationExecuted;
                report[QStringLiteral("tie_point_track_count")] = workflowResult.tiePointResult.trackCount;
                const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
                xjw::gui::project::writeLatestAndAppendHistoryReport(
                    reportsDir, QStringLiteral("at_report.json"), QStringLiteral("at_report_history.json"), report);
                xjw::gui::project::writeLatestAndAppendHistoryReport(
                    reportsDir,
                    QStringLiteral("aerial_triangulation_sfm_report.json"),
                    QStringLiteral("aerial_triangulation_sfm_report_history.json"),
                    report);
            }

            const QString sidecar_path = resultRecordExtra.value(QStringLiteral("files"))
                                             .toObject()
                                             .value(QStringLiteral("sparse_cloud_points_json"))
                                             .toString();
            if (!task_guard->reportSparseTiePointResult(taskContext, result.sparseCloudPath, sidecar_path) ||
                !task_guard->finishTask(taskContext, true))
            {
                return;
            }
            QMessageBox::information(
                controller->_mainWindow,
                QStringLiteral("空中三角测量"),
                (rpcAerialTriangulation
                     ? QStringLiteral("RPC 空三稀疏云已生成。\n注册影像: %1\n地面点: %2\n路径: %3")
                     : QStringLiteral("正式 SfM/BA 稀疏云已生成。\n注册影像: %1\n点数: %2\n路径: %3"))
                    .arg(registeredImageCount)
                    .arg(result.numPoints3D)
                    .arg(result.sparseCloudPath));
        });
    _tasks->trackExternalSparseFuture(taskContext, std::move(workflow_future));
}

void MenuWorkflowController::openOverlapAnalysisDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto* dlg = new OverlapAnalysisDialog(_session, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MenuWorkflowController::openCreateDemDialog()
{
    if (!_mainWindow)
    {
        return;
    }
    if (!_session || !_tasks || _session->projectPath().trimmed().isEmpty())
    {
        QMessageBox::warning(_mainWindow, QStringLiteral("生成 DEM"), QStringLiteral("请先打开项目"));
        return;
    }

    if (_createDemDialog)
    {
        _createDemDialog->show();
        _createDemDialog->raise();
        _createDemDialog->activateWindow();
        return;
    }

    for (QWidget* widget : QApplication::topLevelWidgets())
    {
        auto* existing_dialog = qobject_cast<CreateDemDialog*>(widget);
        if (!existing_dialog || existing_dialog->parentWidget() != _mainWindow)
        {
            continue;
        }
        _createDemDialog = existing_dialog;
        existing_dialog->show();
        existing_dialog->raise();
        existing_dialog->activateWindow();
        return;
    }

    auto* dlg = new CreateDemDialog(_mainWindow);
    dlg->setAvailableImages(getProjectImages());
    _createDemDialog = dlg;
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    connect(dlg, &QObject::destroyed, this, [this]() { _createDemDialog = nullptr; });

    connect(dlg,
            &CreateDemDialog::requestRun,
            this,
            [this](const xjw::gui::project::DemGenerationRequest& request)
            {
                if (!_session || !_tasks)
                {
                    return;
                }
                QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
                QPointer<xjw::gui::project::ProjectTaskOrchestrator> task_guard(_tasks);
                const auto session = session_guard->context();
                QTimer::singleShot(0,
                                   task_guard.data(),
                                   [session_guard, task_guard, session, request]()
                                   {
                                       if (!session_guard || !task_guard || !session_guard->isCurrent(session))
                                       {
                                           return;
                                       }
                                       task_guard->startDemFromPointCloudAsync(request);
                                   });
            });
    connect(dlg,
            &CreateDemDialog::requestCancel,
            this,
            [this]()
            {
                if (_tasks)
                {
                    _tasks->cancelDemGeneration();
                }
            });

    // 进度反馈 → 对话框内显示
    if (_session && _tasks)
    {
        connect(_session, &xjw::gui::project::ProjectSession::sessionChanged, dlg, &QObject::deleteLater);
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::demPipelineProgressChanged,
                dlg,
                &CreateDemDialog::onPipelineProgress);
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished,
                dlg,
                &CreateDemDialog::onPipelineFinished);
    }

    dlg->show();
}

void MenuWorkflowController::openMapProjectDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto* dlg = new MapProjectDialog(_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (_session)
    {
        QStringList images = getProjectImages();
        if (!images.isEmpty())
        {
            dlg->setAvailableImages(images);
        }

        const QString projectPath = _session->projectPath();
        const QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(projectPath);
        if (!projectRoot.isEmpty())
        {
            dlg->setProjectRoot(projectRoot);
        }

        const auto pinhole_models = _session->getPinholeModelsForImages(images);
        int maskReadyCount = 0;
        for (const QString& imagePath : images)
        {
            if (!xjw::common::project::ProjectIO::findMaskForImage(projectPath, imagePath).isEmpty())
            {
                ++maskReadyCount;
            }
        }
        dlg->setImageReadiness(pinhole_models.keys(), maskReadyCount);
        dlg->setRpcImageReadiness(_session->getRpcCameraImagePaths(images));

        const QJsonArray demResults = _session->metadata().value(QStringLiteral("dem_results")).toArray();
        QString latestRelativeDem;
        QString latestRpcDem;
        QString latestAnyDem;
        for (int index = demResults.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demResults.at(index).toObject();
            if (record.value(QStringLiteral("terrain_mode")).toString() == QLatin1String("small_body_global"))
            {
                continue;
            }
            const QString candidate = xjw::common::project::ProjectIO::resolveProjectResourcePath(
                projectPath, record.value(QStringLiteral("dem_path")).toString());
            const QFileInfo candidateInfo(candidate);
            if (candidate.isEmpty() || !candidateInfo.exists() || !candidateInfo.isFile())
            {
                continue;
            }
            if (latestAnyDem.isEmpty())
            {
                latestAnyDem = candidate;
            }
            if (latestRpcDem.isEmpty() &&
                record.value(QStringLiteral("terrain_mode")).toString() == QLatin1String("rpc_stereo"))
            {
                latestRpcDem = candidate;
            }
            if (latestRelativeDem.isEmpty() &&
                record.value(QStringLiteral("dem_reference")).toString() == QStringLiteral("relative"))
            {
                latestRelativeDem = candidate;
            }
        }
        dlg->setDefaultDemPath(!latestRelativeDem.isEmpty() ? latestRelativeDem : latestAnyDem);
        dlg->setDefaultRpcDemPath(latestRpcDem);

        const QJsonArray denseResults = _session->metadata().value(QStringLiteral("dense_cloud_results")).toArray();
        for (int index = denseResults.size() - 1; index >= 0; --index)
        {
            const QString candidate = xjw::common::project::ProjectIO::resolveProjectResourcePath(
                projectPath, denseResults.at(index).toObject().value(QStringLiteral("dense_cloud_xyz")).toString());
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
        _mapSetting->setProjectPath(_session->projectPath());
        const QJsonObject saved = _mapSetting->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }
    }

    connect(dlg,
            &MapProjectDialog::settingsChanged,
            this,
            [this](const QJsonObject& s)
            {
                if (_mapSetting)
                {
                    _mapSetting->save(s);
                }
            });

    connect(dlg,
            &MapProjectDialog::requestRunMapProject,
            dlg,
            [this, dialog = QPointer<MapProjectDialog>(dlg)](const QJsonObject& settings)
            {
                if (!_tasks)
                {
                    LOG_WARN(QStringLiteral("MapProject: 未找到项目任务服务"));
                    if (dialog)
                    {
                        dialog->onPipelineFinished(false, QStringLiteral("项目管理器不可用，无法启动正射影像任务"));
                    }
                    return;
                }
                xjw::gui::project::OrthoGenerationRequest request;
                QString requestError;
                if (!xjw::gui::project::OrthoGenerationRequest::fromJson(settings, &request, &requestError))
                {
                    if (dialog)
                    {
                        dialog->onPipelineFinished(false, requestError);
                    }
                    return;
                }
                _tasks->startMapProjectAsync(request);
            });

    connect(dlg,
            &MapProjectDialog::requestCancelMapProject,
            this,
            [this]()
            {
                if (_tasks)
                {
                    _tasks->cancelMapProject();
                }
            });

    if (_tasks)
    {
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineStarted,
                dlg,
                &MapProjectDialog::onPipelineStarted);
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineProgressChanged,
                dlg,
                &MapProjectDialog::onPipelineProgress);
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineFinished,
                dlg,
                &MapProjectDialog::onPipelineFinished);
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
    if (_session)
    {
        assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(_session->projectPath());
    }

    const QJsonObject metadata = _session ? _session->metadata() : QJsonObject();
    auto* dlg = new WorkflowReportDialog(assetsDir, metadata, _mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    if (_session)
    {
        connect(_session, &xjw::gui::project::ProjectSession::sessionChanged, dlg, &QObject::deleteLater);
        connect(_session,
                &xjw::gui::project::ProjectSession::metadataChanged,
                dlg,
                [dlg](const QJsonObject& updatedMetadata)
                {
                    dlg->setProjectMetadata(updatedMetadata);
                    dlg->refresh();
                });
    }
    dlg->show();
}

void MenuWorkflowController::openCameraCalibrationDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    const auto session = _session ? _session->context() : xjw::gui::project::ProjectSessionContext{};
    const QJsonObject metadata = _session ? _session->metadata() : QJsonObject();
    const QString assetsDir =
        _session ? xjw::common::project::ProjectIO::projectAssetsDir(_session->projectPath()) : QString();
    auto* dialog = new CameraCalibrationDialog(metadata, assetsDir, _mainWindow);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if (_session)
    {
        connect(_session, &xjw::gui::project::ProjectSession::sessionChanged, dialog, &QDialog::close);
    }
    auto sessionIsCurrent = [this, session]() { return _session && _session->isCurrent(session); };
    auto reopenAfterChange = [this, dialog, sessionIsCurrent](bool changed)
    {
        if (!changed || !sessionIsCurrent())
        {
            return;
        }
        dialog->close();
        QTimer::singleShot(0, this, &MenuWorkflowController::openCameraCalibrationDialog);
    };
    const auto camera_task_pending = std::make_shared<bool>(false);
    auto markCameraTaskStarted = [dialog, camera_task_pending](bool started)
    {
        if (started)
        {
            *camera_task_pending = true;
            dialog->setCameraTaskRunning(true);
        }
    };
    if (_tasks)
    {
        dialog->setCameraTaskRunning(_tasks->hasRunningCameraTask());
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::cameraProgressChanged,
                dialog,
                [dialog](const QString&, int) { dialog->setCameraTaskRunning(true); });
        connect(_tasks,
                &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished,
                dialog,
                [this, dialog, session, sessionIsCurrent, camera_task_pending](bool success)
                {
                    const bool reopen_on_success = *camera_task_pending;
                    *camera_task_pending = false;
                    dialog->setCameraTaskRunning(false);
                    if (!reopen_on_success || !success || !sessionIsCurrent() || !_tasks ||
                        _tasks->isSessionDrainInProgress())
                    {
                        return;
                    }
                    dialog->close();
                    QTimer::singleShot(0,
                                       this,
                                       [this, session]()
                                       {
                                           if (_session && _session->isCurrent(session) && _tasks &&
                                               !_tasks->isSessionDrainInProgress())
                                           {
                                               openCameraCalibrationDialog();
                                           }
                                       });
                });
    }
    connect(dialog,
            &CameraCalibrationDialog::importCameraForImageRequested,
            this,
            [this, markCameraTaskStarted, sessionIsCurrent](const QString& imagePath)
            {
                if (sessionIsCurrent())
                {
                    markCameraTaskStarted(_tasks && _tasks->importCameraForImage(imagePath));
                }
            });
    connect(dialog,
            &CameraCalibrationDialog::importCameraProjectRequested,
            this,
            [this, markCameraTaskStarted, sessionIsCurrent]()
            {
                if (sessionIsCurrent())
                {
                    markCameraTaskStarted(_tasks && _tasks->importCameraProject());
                }
            });
    connect(dialog,
            &CameraCalibrationDialog::initializeIntrinsicsRequested,
            this,
            [this, markCameraTaskStarted, sessionIsCurrent](const QJsonObject& settings)
            {
                if (sessionIsCurrent())
                {
                    markCameraTaskStarted(_tasks && _tasks->initializeCamerasFromIntrinsics(settings));
                }
            });
    connect(dialog,
            &CameraCalibrationDialog::clearCamerasRequested,
            this,
            [this, reopenAfterChange, session, sessionIsCurrent](const QStringList& imagePaths)
            {
                if (!sessionIsCurrent())
                {
                    return;
                }
                int clearedCount = 0;
                QString error;
                bool allResolved = false;
                const auto image_ids = cameraImageIdsForPaths(_session->metadata(), imagePaths, &allResolved);
                bool changed = false;
                if (!allResolved)
                {
                    error = tr("所选影像无法唯一对应项目 ImageId，未清除相机");
                }
                else
                {
                    changed = _session->replaceNativeCameraInstances(
                        session, image_ids, {}, {}, nullptr, &clearedCount, &error);
                }
                if (!changed && !error.isEmpty())
                {
                    QMessageBox::warning(_mainWindow, tr("清除相机"), error);
                }
                reopenAfterChange(changed && clearedCount > 0);
            });
    dialog->show();
}
