#include "MenuWorkflowController.h"

#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "SFMService.h"
#include "Logger.h"
#include "project/SparseResultQuality.h"

#include "FeatureExtractionDialog.h"
#include "VocabularyOverlapDialog.h"
#include "FeatureExtractionRunner.h"
#include "FeaturePointVisualizationDialog.h"
#include "CanvasWidget.h"
#include "MainWindow.h"
#include "MatchPairSelectorDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "OverlapAnalysisDialog.h"
#include "CreateDemDialog.h"
#include "MapProjectDialog.h"
#include "WorkflowReportDialog.h"
#include "CameraConvertDialog.h"

#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"

#include <QColor>
#include <QDateTime>
#include <QDebug>
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
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <numeric>

namespace
{

/// 生成稳定的影像对 key，用于匹配约束和历史结果索引。
QString canonicalPairKey(const QString &imageA, const QString &imageB)
{
    const QString normA = xjw::gui::project::normalizePath(imageA);
    const QString normB = xjw::gui::project::normalizePath(imageB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return (normA < normB)
        ? (normA + QStringLiteral("\n") + normB)
        : (normB + QStringLiteral("\n") + normA);
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
        selectedSet.insert(xjw::gui::project::normalizePath(imagePath));
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
        const QString imageA = xjw::gui::project::resolveProjectImagePathFromToken(tokenA, projectMeta);
        const QString imageB = xjw::gui::project::resolveProjectImagePathFromToken(tokenB, projectMeta);
        if (imageA.isEmpty() || imageB.isEmpty())
        {
            continue;
        }

        const QString normA = xjw::gui::project::normalizePath(imageA);
        const QString normB = xjw::gui::project::normalizePath(imageB);
        if (!selectedSet.contains(normA) || !selectedSet.contains(normB))
        {
            continue;
        }

        const QString pairKey = canonicalPairKey(normA, normB);
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
    , m_mainWindow(mainWindow)
{
}

void MenuWorkflowController::setProjectManager(ProjectManager *projectManager)
{
    m_projectManager = projectManager;
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
    if (!m_projectManager)
    {
        return QStringList();
    }

    QStringList images = m_projectManager->getImagesByCategory(QStringLiteral("源数据"));
    if (images.isEmpty()) images = m_projectManager->getImagesByCategory(QStringLiteral("照片"));
    if (images.isEmpty()) images = m_projectManager->getImagesByCategory(QStringLiteral("Photos"));
    if (images.isEmpty()) images = m_projectManager->getAllImages();
    return images;
}

MenuWorkflowController::SparsePrerequisiteSummary
MenuWorkflowController::summarizeSparsePrerequisites(const QStringList &images,
                                                     const QJsonObject &meta,
                                                     const QString &projectPath)
{
    SparsePrerequisiteSummary summary;
    summary.imageCount = images.size();

    int availableFeatureCount = 0;
    for (const QString &imagePath : images)
    {
        if (!ProjectIO::findFeatureForImage(projectPath, imagePath).isEmpty())
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

    QSet<QString> matchedPairKeys;
    const QVector<QPair<QString, QString>> matchedPairs =
        xjw::gui::project::collectMatchedImageNamePairs(projectPath, meta);
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

    auto imagePairCovered = [&](const QString &leftImage, const QString &rightImage) -> bool
    {
        return pairCoveredByAliases(nameAliases(leftImage), nameAliases(rightImage));
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

    auto currentMatchGraphIsUsable = [&]() -> bool
    {
        if (images.size() < 2)
        {
            return true;
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

        int matchedEdgeCount = 0;
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
            ++matchedEdgeCount;
        }

        if (matchedEdgeCount <= 0 || matchedImageIndices.size() != images.size())
        {
            return false;
        }

        const int firstRoot = findRoot(0);
        for (int i = 1; i < images.size(); ++i)
        {
            if (findRoot(i) != firstRoot)
            {
                return false;
            }
        }
        return true;
    };

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
        }
    }
    const bool generatedPlanHasNoCoveredPairs = usedStoredPairs
        && !storedPairsStale
        && generatedPairRequiredCount > 0
        && generatedPairCoveredCount == 0;
    summary.hasMatches = !generatedPlanHasNoCoveredPairs && currentMatchGraphIsUsable();

    if (!summary.hasFeatures)
    {
        summary.missingMessages.append(QStringLiteral("缺少特征：当前影像特征不完整，无法直接用于空三。"));
    }
    if (!summary.hasMatches)
    {
        summary.missingMessages.append(QStringLiteral("缺少连接点：当前影像对匹配不完整，无法直接用于空三。"));
    }
    return summary;
}

bool MenuWorkflowController::confirmAutoFillMissingSparseInputs(const SparsePrerequisiteSummary &summary) const
{
    if (summary.missingMessages.isEmpty())
    {
        return true;
    }

    const QString message = QStringLiteral("空中三角测量缺少上游数据：\n\n%1\n\n是否自动补齐缺失步骤？")
        .arg(summary.missingMessages.join(QStringLiteral("\n")));
    QMessageBox box(m_mainWindow);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(QStringLiteral("空中三角测量"));
    box.setText(message);
    QPushButton *autoFill = box.addButton(QStringLiteral("自动补齐缺失步骤"), QMessageBox::AcceptRole);
    box.addButton(QStringLiteral("返回手动处理"), QMessageBox::RejectRole);
    box.exec();
    return box.clickedButton() == autoFill;
}

void MenuWorkflowController::openFeatureExtractionDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new FeatureExtractionDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (m_projectManager)
    {
        if (!m_featureExtractionSetting)
        {
            m_featureExtractionSetting = new DialogSettingStore(DialogSettingKeys::FeatureExtraction, this);
        }
        m_featureExtractionSetting->setProjectPath(m_projectManager->currentProjectPath());

        // 从 project_dialog.json 加载之前保存的设置
        const QJsonObject saved = m_featureExtractionSetting->load();
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
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        if (!assetsDir.isEmpty())
        {
            QJsonObject defaultOutput;
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
        if (m_featureExtractionSetting)
        {
            m_featureExtractionSetting->save(s);
        }
    });

    // 连接运行请求信号
    connect(dlg, &FeatureExtractionDialog::runRequested, this,
        [this](const QJsonObject &config, const QStringList &inputs)
    {
        if (!m_projectManager)
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
    if (!m_mainWindow)
    {
        return;
    }
    if (!m_projectManager || m_projectManager->currentProjectPath().isEmpty())
    {
        LOG_ERROR(QStringLiteral("无法获取重叠对：请先打开或创建项目"));
        QMessageBox::warning(m_mainWindow, QStringLiteral("获取重叠对"), QStringLiteral("请先打开或创建项目。"));
        return;
    }

    if (!m_vocabOverlapSetting)
    {
        m_vocabOverlapSetting = new DialogSettingStore(DialogSettingKeys::VocabularyOverlap, this);
    }
    m_vocabOverlapSetting->setProjectPath(m_projectManager->currentProjectPath());

    auto *dlg = new VocabularyOverlapDialog(m_projectManager, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setProjectImages(getProjectImages());

    auto *mainWin = qobject_cast<MainWindow *>(m_mainWindow.data());
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

    const QJsonObject saved = m_vocabOverlapSetting->load();
    if (!saved.isEmpty())
    {
        dlg->applySettings(saved);
    }

    connect(dlg, &VocabularyOverlapDialog::settingsChanged, this, [this](const QJsonObject &settings)
    {
        if (m_vocabOverlapSetting)
        {
            m_vocabOverlapSetting->save(settings);
        }
    });

    connect(dlg,
            &VocabularyOverlapDialog::generatedPairsReady,
            this,
            [this](const QStringList &pairs, const QJsonObject &settings)
    {
        if (!m_projectManager || m_projectManager->currentProjectPath().isEmpty())
        {
            return;
        }

        DialogSettingStore matchStore(DialogSettingKeys::FeatureMatching, this);
        matchStore.setProjectPath(m_projectManager->currentProjectPath());

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
    if (!m_mainWindow)
    {
        return;
    }

    // 收集当前可用的特征文件后缀
    QStringList availableSuffixes;
    QString currentSuffix;
    auto *mainWin = qobject_cast<MainWindow*>(m_mainWindow.data());
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
    if (m_projectManager)
    {
        if (!m_featurePointVisualizationSetting)
        {
            m_featurePointVisualizationSetting =
                new DialogSettingStore(DialogSettingKeys::FeaturePointVisualization, this);
        }
        m_featurePointVisualizationSetting->setProjectPath(m_projectManager->currentProjectPath());
        sv = m_featurePointVisualizationSetting->load();

        const QString savedSuffix = sv.value(QStringLiteral("feature_suffix")).toString().trimmed();
        if (!savedSuffix.isEmpty())
        {
            currentSuffix = savedSuffix;
        }
    }

    auto *dlg = new FeaturePointVisualizationDialog(availableSuffixes, m_mainWindow);
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
            if (m_featurePointVisualizationSetting)
            {
                QJsonObject sv = m_featurePointVisualizationSetting->load();
                sv[QStringLiteral("feature_suffix")] = suffix;
                m_featurePointVisualizationSetting->save(sv);
            }
        });
    }

    // 懒初始化可视化记忆化管理器并加载保存的设置
    if (m_projectManager)
    {
        if (!sv.isEmpty())
        {
            LayerRenderer::FeatureDisplayOptions opts;
            opts.showPoints = sv.value("showPoints").toBool(opts.showPoints);
            opts.showScale = sv.value("showScale").toBool(opts.showScale);
            opts.showOrientation = sv.value("showOrientation").toBool(opts.showOrientation);
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

            dlg->setDisplayOptions(opts);
        }
    }

    // 连接实时更新信号
    connect(dlg, &FeaturePointVisualizationDialog::displayOptionsChanged, this,
        [this, dlg](const LayerRenderer::FeatureDisplayOptions &opts)
        {
            // 发送信号给MainWindow应用到CanvasWidget
            emit requestApplyFeatureDisplayOptions(opts);

            // 保存到 project_dialog.json
            if (m_featurePointVisualizationSetting)
            {
                QJsonObject sv;
                sv["showPoints"] = opts.showPoints;
                sv["showScale"] = opts.showScale;
                sv["showOrientation"] = opts.showOrientation;
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

                m_featurePointVisualizationSetting->save(sv);
            }
        });

    dlg->show();
}

void MenuWorkflowController::applySavedFeatureDisplayOptions(const QJsonObject &ui)
{
    if (!m_projectManager)
    {
        return;
    }

    // 优先从 project_dialog.json 加载
    if (!m_featurePointVisualizationSetting)
    {
        m_featurePointVisualizationSetting =
            new DialogSettingStore(DialogSettingKeys::FeaturePointVisualization, this);
    }
    m_featurePointVisualizationSetting->setProjectPath(m_projectManager->currentProjectPath());
    QJsonObject sv = m_featurePointVisualizationSetting->load();

    // 兼容旧版本：若新文件中无数据则尝试从传入的旧 ui 设置中读取
    if (sv.isEmpty() && ui.contains(QStringLiteral("superpoint_visualization")))
    {
        sv = ui.value(QStringLiteral("superpoint_visualization")).toObject();
    }
    auto *mainWin = qobject_cast<MainWindow*>(m_mainWindow.data());
    auto *canvas = mainWin ? mainWin->canvas() : nullptr;
    const QString savedSuffix = sv.value(QStringLiteral("feature_suffix")).toString().trimmed();
    if (canvas)
    {
        const QString inferredSuffix = xjw::gui::project::inferPreferredFeatureSuffix(
            m_projectManager->currentProjectPath(), m_projectManager->currentMeta());
        const bool savedSuffixUsable = !savedSuffix.isEmpty()
            && (inferredSuffix.isEmpty()
                || xjw::gui::project::projectHasFeatureSuffix(
                    m_projectManager->currentProjectPath(), m_projectManager->currentMeta(), savedSuffix));
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

    emit requestApplyFeatureDisplayOptions(opts);
}

void MenuWorkflowController::openThreeDReconstructionDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new ThreeDReconstructionDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    const QStringList images = getProjectImages();
    dlg->setImageCount(images.size());

    if (!m_threeDSetting)
    {
        m_threeDSetting = new DialogSettingStore(DialogSettingKeys::ThreeDReconstruction, this);
    }

    if (m_projectManager)
    {
        const QString projectPath = m_projectManager->currentProjectPath();
        const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("three_d_reconstruction")));
        }
        m_threeDSetting->setProjectPath(projectPath);
        dlg->applySettings(m_threeDSetting->load());
    }

    connect(dlg, &ThreeDReconstructionDialog::settingsChanged, this, [this](const QJsonObject &settings) {
        if (m_threeDSetting)
        {
            m_threeDSetting->save(settings);
        }
    });
    connect(dlg, &ThreeDReconstructionDialog::runRequested, this, [this](const QJsonObject &settings) {
        if (m_threeDSetting)
        {
            m_threeDSetting->save(settings);
        }
        startThreeDReconstructionWorkflow(settings);
    }, Qt::QueuedConnection);

    dlg->exec();
}

void MenuWorkflowController::openAerialTriangulationDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new ThreeDReconstructionDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation);

    const QStringList images = getProjectImages();
    dlg->setImageCount(images.size());

    if (!m_aerialTriangulationSetting)
    {
        m_aerialTriangulationSetting = new DialogSettingStore(DialogSettingKeys::AerialTriangulation, this);
    }

    if (m_projectManager)
    {
        const QString projectPath = m_projectManager->currentProjectPath();
        const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation")));
        }
        m_aerialTriangulationSetting->setProjectPath(projectPath);
        dlg->applySettings(m_aerialTriangulationSetting->load());
    }

    connect(dlg, &ThreeDReconstructionDialog::settingsChanged, this, [this](const QJsonObject &settings) {
        if (m_aerialTriangulationSetting)
        {
            m_aerialTriangulationSetting->save(settings);
        }
    });
    connect(dlg, &ThreeDReconstructionDialog::runRequested, this, [this](const QJsonObject &settings) {
        if (m_aerialTriangulationSetting)
        {
            m_aerialTriangulationSetting->save(settings);
        }
        startAerialTriangulationWorkflow(settings);
    }, Qt::QueuedConnection);

    dlg->exec();
}

void MenuWorkflowController::startAerialTriangulationWorkflow(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        QMessageBox::warning(m_mainWindow, QStringLiteral("空中三角测量"), QStringLiteral("请先打开项目"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(m_mainWindow,
                             QStringLiteral("空中三角测量"),
                             QStringLiteral("至少需要 2 张影像才能进行空中三角测量。"));
        return;
    }

    auto *pm = m_projectManager;
    const QString projectPath = pm->currentProjectPath();
    const QJsonObject projectMeta = pm->currentMeta();
    QString outputRoot = settings.value(QStringLiteral("output_dir")).toString().trimmed();
    if (outputRoot.isEmpty())
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
        outputRoot = QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation"));
    }
    outputRoot = QDir::cleanPath(outputRoot);
    QDir().mkpath(outputRoot);

    QJsonObject runSettings = settings;
    runSettings[QStringLiteral("output_dir")] = outputRoot;

    emit pm->atProgressChanged(QStringLiteral("空中三角测量: 检查上游数据..."), 0);

    auto *watcher = new QFutureWatcher<SparsePrerequisiteSummary>(this);
    QPointer<MenuWorkflowController> self(this);
    QPointer<ProjectManager> pmGuard(pm);
    connect(watcher, &QFutureWatcher<SparsePrerequisiteSummary>::finished, this,
            [self, pmGuard, watcher, runSettings, images, projectPath, projectMeta, outputRoot]() {
        const SparsePrerequisiteSummary prereq = watcher->result();
        watcher->deleteLater();

        if (!self || !pmGuard)
        {
            return;
        }
        if (pmGuard->currentProjectPath() != projectPath)
        {
            emit pmGuard->atProgressFinished(false);
            QMessageBox::warning(self->m_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 QStringLiteral("项目已切换，本次空三启动已取消。"));
            return;
        }

        const bool autoFillMissing = self->confirmAutoFillMissingSparseInputs(prereq);
        if (!autoFillMissing && !prereq.missingMessages.isEmpty())
        {
            emit pmGuard->atProgressFinished(false);
            return;
        }

        self->launchAerialTriangulationSfm(runSettings,
                                           images,
                                           projectPath,
                                           projectMeta,
                                           outputRoot,
                                           autoFillMissing);
    });
    watcher->setFuture(QtConcurrent::run([images, projectMeta, projectPath]() {
        return MenuWorkflowController::summarizeSparsePrerequisites(images, projectMeta, projectPath);
    }));
}

void MenuWorkflowController::launchAerialTriangulationSfm(const QJsonObject &settings,
                                                         const QStringList &images,
                                                         const QString &projectPath,
                                                         const QJsonObject &projectMeta,
                                                         const QString &outputRoot,
                                                         bool autoFillMissing)
{
    if (!m_projectManager)
    {
        return;
    }

    auto *pm = m_projectManager;
    if (pm->currentProjectPath() != projectPath)
    {
        emit pm->atProgressFinished(false);
        QMessageBox::warning(m_mainWindow,
                             QStringLiteral("空中三角测量"),
                             QStringLiteral("项目已切换，本次空三启动已取消。"));
        return;
    }

    xjw::gui::SFMServiceOptions opts;
    opts.autoGenerateMissingMatches = autoFillMissing;
    opts.images = images;
    opts.plascanPath = projectPath;
    opts.projectMeta = projectMeta;
    opts.outputDir = QDir(outputRoot).filePath(QStringLiteral("sfm_sparse"));
    const int workflowThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    opts.threads = workflowThreads;
    opts.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    opts.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    opts.featureGrayscaleMax = 1.0f;
    opts.cudaParallelPairs = opts.device == QStringLiteral("cpu")
        ? 1
        : std::clamp(std::max(1, workflowThreads / 4), 1, 2);

    bool usedStoredPairs = false;
    bool storedPairsStale = false;
    const QStringList allowedPairs = loadGeneratedPairConstraints(projectPath,
                                                                  projectMeta,
                                                                  images,
                                                                  &usedStoredPairs,
                                                                  &storedPairsStale);
    if (!allowedPairs.isEmpty())
    {
        opts.restrictPairs = true;
        opts.allowedPairs = allowedPairs;
        LOG_INFO(QStringLiteral("空中三角测量: 使用已生成候选配对约束 %1 对").arg(allowedPairs.size()));
    }
    else if (storedPairsStale)
    {
        LOG_WARN(QStringLiteral("空中三角测量: 已生成候选配对与当前影像集合不一致，改用自动配对规划"));
    }

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    opts.quality = sfmQualityLevelFromWorkflowQuality(quality);

    QPointer<ProjectManager> pmGuard(pm);
    opts.progressFn = [pmGuard](const QString &stage, int percent) {
        if (!pmGuard)
        {
            return;
        }
        QMetaObject::invokeMethod(pmGuard, [pmGuard, stage, percent]() {
            if (!pmGuard)
            {
                return;
            }
            emit pmGuard->atProgressChanged(QStringLiteral("空中三角测量: %1").arg(stage), percent);
        }, Qt::QueuedConnection);
    };
    opts.pairMatchedFn = [pmGuard](const QString &img0,
                                   const QString &img1,
                                   const QString &matchPath,
                                   int numMatches) {
        if (!pmGuard)
        {
            return;
        }
        QMetaObject::invokeMethod(pmGuard, [pmGuard, img0, img1, matchPath, numMatches]() {
            if (!pmGuard)
            {
                return;
            }
            emit pmGuard->matchPairReady(img0, img1, matchPath, numMatches);
        }, Qt::QueuedConnection);
    };

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    opts.cancelFlag = cancelFlag;

    emit pm->atProgressChanged(QStringLiteral("空中三角测量: 启动 SfM/BA..."), 0);

    const QStringList sfmImages = images;
    const QString sfmOutputDir = opts.outputDir;
    const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
    auto *watcher = new QFutureWatcher<xjw::gui::SFMServiceResult>(this);
    connect(watcher, &QFutureWatcher<xjw::gui::SFMServiceResult>::finished, this,
            [this, watcher, pmGuard, cancelFlag, sfmImages, sfmOutputDir, projectPath]() mutable {
        xjw::gui::SFMServiceResult result = watcher->result();
        watcher->deleteLater();

        if (!pmGuard)
        {
            return;
        }
        pmGuard->clearAtCancelFlag(cancelFlag);
        if (pmGuard->currentProjectPath() != projectPath)
        {
            emit pmGuard->atProgressFinished(false);
            QMessageBox::warning(m_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 QStringLiteral("项目已切换，本次空三结果未写回。"));
            return;
        }

        const bool wasCanceled = cancelFlag->load(std::memory_order_relaxed);
        for (const auto &sp : result.newFeatureFiles)
        {
            pmGuard->appendIpfindResult(sp.imagePath, sp.featurePath, QJsonObject());
        }
        for (const auto &match : result.newMatchFiles)
        {
            pmGuard->appendIpmatchResult(QStringList{match.matchPath}, match.settings);
        }

        if (!result.pendingCamUpdates.isEmpty())
        {
            int updated = 0;
            QString err;
            if (!pmGuard->setImageCameras(result.pendingCamUpdates, &updated, &err))
            {
                LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
            }
        }

        int registeredImageCount = result.numRegisteredImages;
        QJsonObject resultRecordExtra = result.resultRecordExtra;
        QString sparseBlockingReason = QStringLiteral("SFM 未生成可用的正式稀疏点云。");
        if (result.success && !result.sparseCloudPath.isEmpty())
        {
            const QStringList registeredCameraKeys = result.pendingCamUpdates.keys();
            QStringList registeredImages;
            registeredImages.reserve(sfmImages.size());
            for (const QString &imagePath : sfmImages)
            {
                const QString normalized = xjw::gui::project::normalizePath(imagePath);
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
            pmGuard->appendAtResult(result.sparseCloudPath,
                                    result.numPoints3D,
                                    registeredImages,
                                    sfmOutputDir,
                                    resultRecordExtra);
            sparseBlockingReason = xjw::gui::project::sparseResultBlockingReason(resultRecordExtra);
        }

        emit pmGuard->atProgressFinished(result.success);
        if (wasCanceled)
        {
            return;
        }
        if (!result.success)
        {
            QMessageBox::warning(m_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 result.errorMessage.isEmpty()
                                     ? QStringLiteral("空中三角测量失败。")
                                     : result.errorMessage);
            return;
        }

        if (!xjw::gui::project::isProductionSparseResult(resultRecordExtra))
        {
            QMessageBox::warning(m_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 sparseBlockingReason.isEmpty()
                                     ? QStringLiteral("当前 SfM/BA 稀疏点云质量不足。")
                                     : sparseBlockingReason);
            return;
        }

        QMessageBox::information(m_mainWindow,
                                 QStringLiteral("空中三角测量"),
                                 QStringLiteral("正式 SfM/BA 稀疏云已生成。\n注册影像: %1\n点数: %2\n路径: %3")
                                     .arg(registeredImageCount)
                                     .arg(result.numPoints3D)
                                     .arg(result.sparseCloudPath));
    });
    watcher->setFuture(QtConcurrent::run([runOpts = std::move(opts), sfmImages, sfmOutputDir, assetsDir]() mutable {
        xjw::gui::SFMServiceResult result = xjw::gui::SFMService::run(runOpts);
        if (result.success && !assetsDir.isEmpty())
        {
            QJsonObject report;
            report[QStringLiteral("type")] = QStringLiteral("aerial_triangulation_sfm");
            report[QStringLiteral("mode")] = QStringLiteral("sfm");
            report[QStringLiteral("source")] = QStringLiteral("workflow_aerial_triangulation");
            report[QStringLiteral("timestamp")] = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
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
                                              QStringLiteral("aerial_triangulation_sfm_report.json"),
                                              QStringLiteral("aerial_triangulation_sfm_report_history.json"),
                                              report);
        }
        return result;
    }));
}

void MenuWorkflowController::startThreeDReconstructionWorkflow(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        QMessageBox::warning(m_mainWindow, QStringLiteral("三维重建"), QStringLiteral("请先打开项目"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(m_mainWindow,
                             QStringLiteral("三维重建"),
                             QStringLiteral("至少需要 2 张影像才能进行三维重建。"));
        return;
    }

    QString outputRoot = settings.value(QStringLiteral("output_dir")).toString().trimmed();
    if (outputRoot.isEmpty())
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        outputRoot = QDir(assetsDir).filePath(QStringLiteral("three_d_reconstruction"));
    }
    outputRoot = QDir::cleanPath(outputRoot);
    QDir().mkpath(outputRoot);

    QJsonObject runSettings = settings;
    runSettings[QStringLiteral("output_dir")] = outputRoot;

    auto *pm = m_projectManager;
    xjw::gui::SFMServiceOptions opts;
    opts.images = images;
    opts.plascanPath = pm->currentProjectPath();
    opts.projectMeta = pm->coreProjectMeta();
    opts.outputDir = QDir(outputRoot).filePath(QStringLiteral("sparse"));
    const int workflowThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    opts.threads = workflowThreads;
    opts.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    opts.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    opts.featureGrayscaleMax = 1.0f;
    opts.cudaParallelPairs = opts.device == QStringLiteral("cpu")
        ? 1
        : std::clamp(std::max(1, workflowThreads / 4), 1, 2);

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    opts.quality = sfmQualityLevelFromWorkflowQuality(quality);

    opts.progressFn = [pm](const QString &stage, int percent) {
        QMetaObject::invokeMethod(pm, [pm, stage, percent]() {
            emit pm->atProgressChanged(QStringLiteral("三维重建/空三: %1").arg(stage), percent);
        }, Qt::QueuedConnection);
    };
    opts.pairMatchedFn = [pm](const QString &img0,
                              const QString &img1,
                              const QString &matchPath,
                              int numMatches) {
        QMetaObject::invokeMethod(pm, [pm, img0, img1, matchPath, numMatches]() {
            emit pm->matchPairReady(img0, img1, matchPath, numMatches);
        }, Qt::QueuedConnection);
    };

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    opts.cancelFlag = cancelFlag;

    emit pm->atProgressChanged(QStringLiteral("三维重建: 启动空中三角测量..."), 0);

    QPointer<MenuWorkflowController> self(this);
    const QStringList sfmImages = images;
    const QString sfmOutputDir = opts.outputDir;
    const QString assetsDir = ProjectIO::projectAssetsDir(pm->currentProjectPath());
    (void)QtConcurrent::run([self, pm, opts, runSettings, sfmImages, sfmOutputDir, assetsDir]() {
        xjw::gui::SFMServiceResult result = xjw::gui::SFMService::run(opts);

        if (result.success && !assetsDir.isEmpty())
        {
            QJsonObject report;
            report[QStringLiteral("type")] = QStringLiteral("three_d_reconstruction_sfm");
            report[QStringLiteral("mode")] = QStringLiteral("sfm");
            report[QStringLiteral("source")] = QStringLiteral("three_d_reconstruction_sfm");
            report[QStringLiteral("timestamp")] = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
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

        QMetaObject::invokeMethod(pm, [self, pm, result = std::move(result), runSettings, sfmImages, sfmOutputDir]() mutable {
            for (const auto &sp : result.newFeatureFiles)
            {
                pm->appendIpfindResult(sp.imagePath, sp.featurePath, QJsonObject());
            }
            for (const auto &match : result.newMatchFiles)
            {
                pm->appendIpmatchResult(QStringList{match.matchPath}, match.settings);
            }

            if (!result.pendingCamUpdates.isEmpty())
            {
                int updated = 0;
                QString err;
                if (!pm->setImageCameras(result.pendingCamUpdates, &updated, &err))
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
                    const QString normalized = xjw::gui::project::normalizePath(imagePath);
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
                sfmAtIndex = pm->currentMeta()
                                 .value(QStringLiteral("aerial_triangulation_results"))
                                 .toArray()
                                 .size();
                pm->appendAtResult(result.sparseCloudPath,
                                   result.numPoints3D,
                                   registeredImages,
                                   sfmOutputDir,
                                   resultRecordExtra);
                currentSfmIsProduction = xjw::gui::project::isProductionSparseResult(resultRecordExtra);
                currentSfmBlockingReason = xjw::gui::project::sparseResultBlockingReason(resultRecordExtra);
            }

            emit pm->atProgressFinished(result.success);
            if (!result.success)
            {
                QMessageBox::warning(nullptr,
                                     QStringLiteral("三维重建"),
                                     result.errorMessage.isEmpty()
                                         ? QStringLiteral("空中三角测量失败。")
                                         : result.errorMessage);
                return;
            }

            if (!currentSfmIsProduction)
            {
                QMessageBox::warning(nullptr,
                                     QStringLiteral("三维重建"),
                                     currentSfmBlockingReason.isEmpty()
                                         ? QStringLiteral("当前 SFM 稀疏点云质量不足，已停止后续 MVS 流程。")
                                         : currentSfmBlockingReason);
                return;
            }

            if (self)
            {
                QJsonObject denseRunSettings = runSettings;
                denseRunSettings[QStringLiteral("registered_image_count")] = registeredImageCount;
                denseRunSettings[QStringLiteral("sfm_at_index")] = sfmAtIndex;
                self->startThreeDReconstructionDenseStage(denseRunSettings);
            }
        }, Qt::QueuedConnection);
    });
}

void MenuWorkflowController::startThreeDReconstructionDenseStage(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(m_projectManager);
    QPointer<MenuWorkflowController> self(this);
    connect(m_projectManager, &ProjectManager::mvsProgressFinished, ctx,
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

    m_projectManager->startGenerateDenseCloudAsync(denseSettings);
}

void MenuWorkflowController::startThreeDReconstructionDenseRefineStage(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(m_projectManager);
    QPointer<MenuWorkflowController> self(this);
    connect(m_projectManager, &ProjectManager::mvsProgressFinished, ctx,
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

    m_projectManager->startDenseCloudRefineAsync(refineSettings);
}

void MenuWorkflowController::startThreeDReconstructionMeshStage(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        return;
    }

    QObject *ctx = new QObject(m_projectManager);
    connect(m_projectManager, &ProjectManager::meshProgressFinished, ctx,
            [this, ctx](bool success) {
        ctx->deleteLater();
        QMessageBox::information(m_mainWindow,
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

    m_projectManager->startMeshReconstructionAsync(meshSettings);
}

void MenuWorkflowController::openOverlapAnalysisDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new OverlapAnalysisDialog(m_projectManager, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MenuWorkflowController::openCreateDemDialog()
{
    if (!m_mainWindow)
    {
        return;
    }
    if (!m_projectManager)
    {
        QMessageBox::warning(m_mainWindow, QStringLiteral("生成 DEM"), QStringLiteral("请先打开项目"));
        return;
    }

    auto *dlg = new CreateDemDialog(m_projectManager, m_mainWindow);
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
        if (!m_projectManager)
            return;
        QMetaObject::invokeMethod(m_projectManager, "startFullDemPipelineAsync", Qt::QueuedConnection,
            Q_ARG(QStringList, images), Q_ARG(QString, outputDir), Q_ARG(QJsonObject, pipelineSettings));
    });

    // 手动模式：从密集点云生成 DEM
    connect(dlg, &CreateDemDialog::requestRunFromDenseCloud, this,
        [this](const QString &denseCloudPath, const QString &outputDir, double demResolution, const QString &demType)
    {
        if (!m_projectManager)
            return;
        QMetaObject::invokeMethod(m_projectManager, "startDemFromDenseCloudAsync", Qt::QueuedConnection,
            Q_ARG(QString, denseCloudPath), Q_ARG(QString, outputDir),
            Q_ARG(double, demResolution), Q_ARG(QString, demType));
    });

    // 进度反馈 → 对话框内显示
    if (m_projectManager)
    {
        connect(m_projectManager, &ProjectManager::demPipelineProgressChanged,
                dlg, &CreateDemDialog::onPipelineProgress);
        connect(m_projectManager, &ProjectManager::demPipelineFinished,
                dlg, &CreateDemDialog::onPipelineFinished);
    }

    dlg->show();
}

void MenuWorkflowController::openMapProjectDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new MapProjectDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (m_projectManager)
    {
        QStringList images = getProjectImages();
        if (!images.isEmpty())
        {
            dlg->setAvailableImages(images);
        }

        QString projectRoot = ProjectIO::projectRootFromPlascan(m_projectManager->currentProjectPath());
        if (!projectRoot.isEmpty())
        {
            dlg->setProjectRoot(projectRoot);
        }

        const QJsonArray demResults = m_projectManager->currentMeta().value(QStringLiteral("dem_results")).toArray();
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
        if (!m_mapSetting)
        {
            m_mapSetting = new DialogSettingStore(DialogSettingKeys::MapProject, this);
        }
        m_mapSetting->setProjectPath(m_projectManager->currentProjectPath());
        const QJsonObject saved = m_mapSetting->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }
    }

    connect(dlg, &MapProjectDialog::settingsChanged, this, [this](const QJsonObject &s)
    {
        if (m_mapSetting)
        {
            m_mapSetting->save(s);
        }
    });

    connect(dlg, &MapProjectDialog::requestRunMapProject, this,
        [this](const QStringList &images, const QString &demPath, const QString &outputPath, double res)
        {
        if (!m_projectManager)
        {
            LOG_WARN(QStringLiteral("MapProject: 未找到 ProjectManager"));
            return;
        }
        QMetaObject::invokeMethod(m_projectManager, "startMapProjectAsync", Qt::QueuedConnection,
            Q_ARG(QStringList, images), Q_ARG(QString, demPath), Q_ARG(QString, outputPath), Q_ARG(double, res));
    });

    dlg->exec();
}

void MenuWorkflowController::runFeatureExtraction(const QJsonObject &config, const QStringList &inputs)
{
    const QString featureAlgorithm = config.value(QStringLiteral("feature_algorithm")).toString(QStringLiteral("disk")).toUpper();
    LOG_INFO(QStringLiteral("开始在后台线程执行 %1 特征提取...").arg(featureAlgorithm));

    auto *mainWin = qobject_cast<MainWindow *>(m_mainWindow.data());

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
    auto *timer = new QTimer(m_mainWindow);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, [mainWin, progressCount]()
    {
        if (mainWin)
        {
            mainWin->updateSpProgress(progressCount->load());
        }
    });
    timer->start();

    auto *watcher = new QFutureWatcher<bool>(m_mainWindow);
    connect(watcher, &QFutureWatcher<bool>::finished,
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

    watcher->setFuture(QtConcurrent::run(
        [config, inputs, pm = m_projectManager, cancelFlag, progressCount]() -> bool
        {
            return FeatureExtractionRunner::run(config, inputs, pm, *cancelFlag, *progressCount);
        }));
}

void MenuWorkflowController::openWorkflowReportDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    QString assetsDir;
    if (m_projectManager)
    {
        assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
    }

    auto *dlg = new WorkflowReportDialog(assetsDir, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MenuWorkflowController::openCameraConvertDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new CameraConvertDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}
