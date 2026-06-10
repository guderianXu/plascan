#include "MenuWorkflowController.h"

#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "SFMService.h"
#include "Logger.h"

#include "FeatureExtractionDialog.h"
#include "SuperPointRunner.h"
#include "SuperPointVisualizationDialog.h"
#include "CanvasWidget.h"
#include "MainWindow.h"
#include "MatchPairSelectorDialog.h"
#include "ThreeDReconstructionDialog.h"
#include "OverlapAnalysisDialog.h"
#include "CreateDemDialog.h"
#include "MapProjectDialog.h"
#include "WorkflowReportDialog.h"

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
#include <QJsonDocument>
#include <QMainWindow>
#include <QMessageBox>
#include <QSet>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>

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

/// 从 SuperGlue 对话框设置中读取已生成的配对约束，并检测其是否覆盖当前选图。
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

    DialogSettingStore store(DialogSettingKeys::SuperGlue, nullptr);
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
        if (!m_spSetting)
        {
            m_spSetting = new DialogSettingStore(DialogSettingKeys::SuperPoint, this);
        }
        m_spSetting->setProjectPath(m_projectManager->currentProjectPath());

        // 从 project_dialog.json 加载之前保存的设置
        const QJsonObject saved = m_spSetting->load();
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
        if (m_spSetting)
        {
            m_spSetting->save(s);
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

        runSuperPointExtraction(config, inputs);
    });

    dlg->exec();
}

void MenuWorkflowController::openSuperPointVisualizationDialog()
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

    auto *dlg = new SuperPointVisualizationDialog(availableSuffixes, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    if (!currentSuffix.isEmpty())
        dlg->setCurrentSuffix(currentSuffix);

    // 切换特征文件后缀 → CanvasWidget 重新加载
    if (canvas)
    {
        connect(dlg, &SuperPointVisualizationDialog::featureSuffixChanged,
                canvas, &CanvasWidget::setActiveFeatureSuffix);
    }

    // 懒初始化可视化记忆化管理器并加载保存的设置
    if (m_projectManager)
    {
        if (!m_spVisSetting)
        {
            m_spVisSetting = new DialogSettingStore(DialogSettingKeys::SuperPointVisualization, this);
        }
        m_spVisSetting->setProjectPath(m_projectManager->currentProjectPath());

        const QJsonObject sv = m_spVisSetting->load();
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
    connect(dlg, &SuperPointVisualizationDialog::displayOptionsChanged, this,
        [this](const LayerRenderer::FeatureDisplayOptions &opts)
        {
            // 发送信号给MainWindow应用到CanvasWidget
            emit requestApplyFeatureDisplayOptions(opts);

            // 保存到 project_dialog.json
            if (m_spVisSetting)
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
                sv["pointColor"] = colorToJson(opts.pointColor);
                sv["scaleColor"] = colorToJson(opts.scaleColor);
                sv["orientColor"] = colorToJson(opts.orientColor);

                m_spVisSetting->save(sv);
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
    if (!m_spVisSetting)
    {
        m_spVisSetting = new DialogSettingStore(DialogSettingKeys::SuperPointVisualization, this);
    }
    m_spVisSetting->setProjectPath(m_projectManager->currentProjectPath());
    QJsonObject sv = m_spVisSetting->load();

    // 兼容旧版本：若新文件中无数据则尝试从传入的旧 ui 设置中读取
    if (sv.isEmpty() && ui.contains(QStringLiteral("superpoint_visualization")))
    {
        sv = ui.value(QStringLiteral("superpoint_visualization")).toObject();
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
    });

    dlg->exec();
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
    opts.threads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    opts.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    opts.cudaParallelPairs = 1;

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    if (quality == QStringLiteral("fast"))
    {
        opts.quality = 1;
    }
    else
    {
        opts.quality = 3;
    }

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
            report[QStringLiteral("timestamp")] = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
            report[QStringLiteral("num_images")] = sfmImages.size();
            report[QStringLiteral("num_registered")] = result.numRegisteredImages;
            report[QStringLiteral("num_points_3d")] = result.numPoints3D;
            report[QStringLiteral("mean_reproj_error_px")] = result.meanReprojError;
            report[QStringLiteral("output_dir")] = sfmOutputDir;
            report[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
            writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                              QStringLiteral("three_d_reconstruction_sfm_report.json"),
                                              QStringLiteral("three_d_reconstruction_sfm_report_history.json"),
                                              report);
        }

        QMetaObject::invokeMethod(pm, [self, pm, result = std::move(result), runSettings, sfmImages, sfmOutputDir]() mutable {
            for (const auto &sp : result.newSpFiles)
            {
                pm->appendIpfindResult(sp.imagePath, sp.spPath, QJsonObject());
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

            if (result.success && !result.sparseCloudPath.isEmpty())
            {
                pm->appendAtResult(result.sparseCloudPath,
                                   result.numPoints3D,
                                   sfmImages,
                                   sfmOutputDir,
                                   QJsonObject{{QStringLiteral("source"), QStringLiteral("three_d_reconstruction")}});
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

            if (self)
            {
                self->startThreeDReconstructionDenseStage(runSettings);
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

    auto latestDensePath = [](const QJsonObject &meta) -> QString {
        const QJsonArray arr = meta.value(QStringLiteral("dense_cloud_results")).toArray();
        if (arr.isEmpty())
        {
            return QString();
        }
        return arr.last().toObject().value(QStringLiteral("dense_cloud_xyz")).toString();
    };

    const QString beforePath = latestDensePath(m_projectManager->currentMeta());
    QObject *ctx = new QObject(m_projectManager);
    QPointer<MenuWorkflowController> self(this);
    connect(m_projectManager, &ProjectManager::projectMetadataChanged, ctx,
            [self, ctx, beforePath, settings, latestDensePath](const QJsonObject &meta) {
        const QString densePath = latestDensePath(meta);
        if (densePath.isEmpty() || densePath == beforePath)
        {
            return;
        }
        ctx->deleteLater();
        if (self)
        {
            self->startThreeDReconstructionMeshStage(settings);
        }
    });
    connect(m_projectManager, &ProjectManager::mvsProgressFinished, ctx,
            [ctx](bool success) {
        if (!success)
        {
            ctx->deleteLater();
        }
    });

    const QString outputRoot = settings.value(QStringLiteral("output_dir")).toString();
    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));

    QJsonObject denseSettings;
    denseSettings[QStringLiteral("pipeline_mode")] = true;
    denseSettings[QStringLiteral("at_index")] = -1;
    denseSettings[QStringLiteral("output_dir")] = QDir(outputRoot).filePath(QStringLiteral("mvs"));
    denseSettings[QStringLiteral("threads")] = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    denseSettings[QStringLiteral("cuda")] = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto")) != QStringLiteral("cpu");
    denseSettings[QStringLiteral("keepColor")] = true;
    denseSettings[QStringLiteral("keepNormals")] = false;
    denseSettings[QStringLiteral("minConsistentViews")] = 2;
    denseSettings[QStringLiteral("minViews")] = 3;
    denseSettings[QStringLiteral("patchSize")] = 11;
    denseSettings[QStringLiteral("confidence")] = 0.20;
    denseSettings[QStringLiteral("minConfidence")] = 0.20;
    denseSettings[QStringLiteral("depthConsistency")] = 2.0;
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
        return;

    auto *dlg = new CreateDemDialog(m_projectManager, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (m_projectManager)
    {
        QStringList images = getProjectImages();
        if (!images.isEmpty())
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

void MenuWorkflowController::runSuperPointExtraction(const QJsonObject &config, const QStringList &inputs)
{
    LOG_INFO(QStringLiteral("开始在后台线程执行 SuperPoint..."));

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
            return SuperPointRunner::run(config, inputs, pm, *cancelFlag, *progressCount);
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
