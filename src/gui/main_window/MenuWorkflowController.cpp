#include "MenuWorkflowController.h"

#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "SFMService.h"
#include "Logger.h"

#include "SuperPointDialog.h"
#include "SuperPointRunner.h"
#include "SuperPointVisualizationDialog.h"
#include "CanvasWidget.h"
#include "MainWindow.h"
#include "MatchPairSelectorDialog.h"
#include "AerialTriangulationDialog.h"
#include "SimplePointCloudDialog.h"
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

void MenuWorkflowController::openSuperPointDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new SuperPointDialog(m_mainWindow);
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
        LOG_ERROR(QStringLiteral("无法运行 SuperPoint：项目管理器未初始化"));
    }

    // 连接设置变更信号，实时保存到 project_dialog.json
    connect(dlg, &SuperPointDialog::settingsChanged, this, [this](const QJsonObject &s)
    {
        if (m_spSetting)
        {
            m_spSetting->save(s);
        }
    });

    // 连接运行请求信号
    connect(dlg, &SuperPointDialog::runRequested, this,
        [this](const QJsonObject &config, const QStringList &inputs)
    {
        if (!m_projectManager)
        {
            LOG_ERROR(QStringLiteral("无法运行 SuperPoint：项目管理器未初始化"));
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

void MenuWorkflowController::openAerialTriangulationDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new AerialTriangulationDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (!m_atSetting)
    {
        m_atSetting = new DialogSettingStore(DialogSettingKeys::AerialTriangulation, this);
    }

    if (m_projectManager)
    {
        const QStringList images = getProjectImages();
        if (!images.isEmpty())
        {
            dlg->setAvailableImages(images);
        }

        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation")));
        }

        m_atSetting->setProjectPath(m_projectManager->currentProjectPath());
        const QJsonObject saved = m_atSetting->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }
    }

    connect(dlg, &AerialTriangulationDialog::settingsChanged, this, [this](const QJsonObject &s)
    {
        if (m_atSetting)
        {
            m_atSetting->save(s);
        }
    });

    connect(dlg, &AerialTriangulationDialog::runRequested, this, [this](const QJsonObject &settings)
    {
        if (!m_projectManager)
        {
            QMessageBox::warning(m_mainWindow, QStringLiteral("提示"), QStringLiteral("请先打开项目"));
            return;
        }

        // 解析参数
        const QJsonArray imageArr = settings.value(QStringLiteral("images")).toArray();
        QStringList images;
        images.reserve(imageArr.size());
        for (const QJsonValue &v : imageArr)
        {
            images.append(v.toString());
        }

        const QString outputDir = settings.value(QStringLiteral("output_dir")).toString();
        const int threads = settings.value(QStringLiteral("threads")).toInt(4);
        const int cameraOptMode = settings.value(QStringLiteral("camera_optimization_mode")).toInt(2);
        const QString projectPath = m_projectManager->currentProjectPath();
        const QJsonObject projectMeta = m_projectManager->coreProjectMeta();
        bool useStoredPairs = false;
        bool storedPairsStale = false;
        const QStringList allowedPairs = loadGeneratedPairConstraints(projectPath,
                                                                     projectMeta,
                                                                     images,
                                                                     &useStoredPairs,
                                                                     &storedPairsStale);
        // cameraOptMode: 0=SFM, 1=光束法平差（自动化全流程）, 2=自动

        if (storedPairsStale)
        {
            LOG_WARN(QStringLiteral("空三: 已保存的连接点配对未覆盖当前全部选中影像，本次将自动扩展匹配范围以纳入新影像"));
        }
        else if (useStoredPairs)
        {
            LOG_INFO(QStringLiteral("空三: 复用已生成的匹配对约束，共 %1 对").arg(allowedPairs.size()));
        }

        bool needSfm            = false;
        bool needBaWithAutoPrep = false;

        if (cameraOptMode == 0)
        {
            // 用户明确选择 SFM
            needSfm = true;
        }
        else if (cameraOptMode == 1)
        {
            // 光束法平差模式：先自动确保特征/匹配存在，再对已有相机参数执行 BA
            needBaWithAutoPrep = true;
            LOG_INFO(QStringLiteral("空三(BA): 自动准备特征/匹配，然后执行光束法平差"));
        }
        else if (cameraOptMode == 2)
        {
            // 自动模式：检查选中影像是否都有相机参数
            bool hasCamerasForAll = false;
            m_projectManager->getCamerasForImages(images, &hasCamerasForAll);
            needSfm = !hasCamerasForAll;
            if (needSfm)
            {
                LOG_INFO(QStringLiteral("空三(自动): 部分影像缺少相机参数，将执行 SFM 恢复相机"));
            }
            else
            {
                LOG_INFO(QStringLiteral("空三(自动): 所有影像已有相机参数，执行含 BA 的全流水线"));
                needSfm = true;  // 统一走 SFMService 以保证特征/匹配存在
            }
        }

        if (needSfm)
        {
            // ── 执行增量式 SFM 全自动流水线（后台线程）──
            LOG_INFO(QStringLiteral("空三: 启动 SFM 全自动流水线 (%1 张影像)").arg(images.size()));

            xjw::gui::SFMServiceOptions opts;
            opts.images      = images;
            opts.plascanPath = projectPath;
            opts.projectMeta = projectMeta;
            opts.quality     = settings.value(QStringLiteral("quality")).toInt(3);
            opts.threads     = threads;
            opts.restrictPairs = useStoredPairs;
            opts.allowedPairs = allowedPairs;

            // 用户提供的内方位元素
            if (settings.contains(QStringLiteral("intrinsics")))
            {
                const QJsonObject intr = settings.value(QStringLiteral("intrinsics")).toObject();
                opts.userFu = intr.value(QStringLiteral("fu")).toDouble();
                opts.userFv = intr.value(QStringLiteral("fv")).toDouble();
                opts.userCu = intr.value(QStringLiteral("cu")).toDouble();
                opts.userCv = intr.value(QStringLiteral("cv")).toDouble();
                opts.userPitch = intr.value(QStringLiteral("pitch")).toDouble();
            }

            auto *pm = m_projectManager;
            const QStringList sfmImages   = images;
            const QString sfmOutputDir    = outputDir;
            const QString assetsDir       = ProjectIO::projectAssetsDir(pm->currentProjectPath());

            QMap<QString, QJsonObject> beforeCameras;
            {
                const QJsonObject meta = pm->coreProjectMeta();  // 仅需 images/camera，无需加载 results
                const QJsonArray  imgs = meta.value(QStringLiteral("images")).toArray();
                for (const QJsonValue &v : imgs)
                {
                    const QJsonObject imgObj = v.toObject();
                    const QString normPath = QDir::cleanPath(
                        QFileInfo(imgObj.value(QStringLiteral("path")).toString()).absoluteFilePath());
                    const QJsonObject camObj = imgObj.value(QStringLiteral("camera")).toObject();
                    if (!normPath.isEmpty() && !camObj.isEmpty())
                        beforeCameras.insert(normPath, camObj);
                }
            }

            // 设置进度回调：从工作线程跨回主线程发出 atProgressChanged 信号
            opts.progressFn = [pm](const QString &stage, int pct)
            {
                QMetaObject::invokeMethod(pm, [pm, stage, pct]()
                {
                    emit pm->atProgressChanged(stage, pct);
                }, Qt::QueuedConnection);
            };

            // 创建取消标志并存储到 ProjectManager 中供 UI 层调用
            auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
            pm->setAtCancelFlag(cancelFlag);
            opts.cancelFlag = cancelFlag;

            // 成功匹配一对后立即通知 UI（用于匹配查看器实时刷新）
            opts.pairMatchedFn = [pm](const QString &img0, const QString &img1,
                                      const QString &matchPath, int numMatches)
            {
                Q_UNUSED(numMatches)
                QMetaObject::invokeMethod(pm, [pm, img0, img1, matchPath, numMatches]()
                {
                    emit pm->matchPairReady(img0, img1, matchPath, numMatches);
                }, Qt::QueuedConnection);
            };

            // CUDA 并行对数
            if (settings.contains(QStringLiteral("cuda_parallel_pairs")))
            {
                opts.cudaParallelPairs = settings.value(QStringLiteral("cuda_parallel_pairs")).toInt(1);
            }

            emit pm->atProgressChanged(tr("启动空三..."), 0);

            (void)QtConcurrent::run([opts, pm, sfmImages, sfmOutputDir, assetsDir,
                                     beforeCameras = std::move(beforeCameras)]()
            {
                xjw::gui::SFMServiceResult result = xjw::gui::SFMService::run(opts);

                // ── AT 报告生成：纯文件 I/O，在后台线程完成，不阻塞 UI ──────────
                if (result.success && !assetsDir.isEmpty())
                {
                    QDir(assetsDir).mkpath(QStringLiteral("reports"));
                    QJsonObject rep;
                    rep[QStringLiteral("type")]              = QStringLiteral("aerial_triangulation");
                    rep[QStringLiteral("timestamp")]         = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
                    rep[QStringLiteral("num_images")]        = sfmImages.size();
                    rep[QStringLiteral("num_registered")]    = result.numRegisteredImages;
                    rep[QStringLiteral("num_points_3d")]     = result.numPoints3D;
                    rep[QStringLiteral("mean_reproj_error_px")] = result.meanReprojError;
                    rep[QStringLiteral("ba_rms_before")]     = result.baRmsBefore;
                    rep[QStringLiteral("ba_rms_after")]      = result.baRmsAfter;
                    rep[QStringLiteral("ba_tracks_total")]   = result.baTracksTotal;
                    rep[QStringLiteral("ba_tracks_optimized")] = result.baTracksOptimized;
                    rep[QStringLiteral("ba_tracks_filtered")] = result.baTracksFiltered;
                    rep[QStringLiteral("duration_s")]        = result.durationSeconds;
                    rep[QStringLiteral("output_dir")]        = sfmOutputDir;
                    rep[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
                    rep[QStringLiteral("per_camera")]        = result.perCameraResiduals;

                    QJsonArray camComp;
                    for (auto it = result.pendingCamUpdates.constBegin();
                        it != result.pendingCamUpdates.constEnd(); ++it)
                    {
                        const QString &normPath = it.key();
                        const QJsonObject &after = it.value();
                        const QJsonObject &before = beforeCameras.value(normPath);
                        const QJsonArray cA = after.value(QStringLiteral("C")).toArray();
                        const QJsonArray cB = before.value(QStringLiteral("C")).toArray();
                        double posDelta = -1.0;
                        if (cA.size() == 3 && cB.size() == 3)
                        {
                            double dx = cA[0].toDouble() - cB[0].toDouble();
                            double dy = cA[1].toDouble() - cB[1].toDouble();
                            double dz = cA[2].toDouble() - cB[2].toDouble();
                            posDelta = std::sqrt(dx*dx + dy*dy + dz*dz);
                        }
                        QJsonObject entry;
                        entry[QStringLiteral("name")]         = QFileInfo(normPath).fileName();
                        entry[QStringLiteral("path")]         = normPath;
                        entry[QStringLiteral("had_before")]   = !before.isEmpty();
                        entry[QStringLiteral("fu_before")]    = before.value(QStringLiteral("fu")).toDouble();
                        entry[QStringLiteral("fu_after")]     = after.value(QStringLiteral("fu")).toDouble();
                        entry[QStringLiteral("fv_before")]    = before.value(QStringLiteral("fv")).toDouble();
                        entry[QStringLiteral("fv_after")]     = after.value(QStringLiteral("fv")).toDouble();
                        entry[QStringLiteral("cu_before")]    = before.value(QStringLiteral("cu")).toDouble();
                        entry[QStringLiteral("cu_after")]     = after.value(QStringLiteral("cu")).toDouble();
                        entry[QStringLiteral("cv_before")]    = before.value(QStringLiteral("cv")).toDouble();
                        entry[QStringLiteral("cv_after")]     = after.value(QStringLiteral("cv")).toDouble();
                        entry[QStringLiteral("C_before")]     = cB;
                        entry[QStringLiteral("C_after")]      = cA;
                        entry[QStringLiteral("yaw_before")]   = before.value(QStringLiteral("yaw_deg")).toDouble();
                        entry[QStringLiteral("yaw_after")]    = after.value(QStringLiteral("yaw_deg")).toDouble();
                        entry[QStringLiteral("pitch_before")] = before.value(QStringLiteral("pitch_deg")).toDouble();
                        entry[QStringLiteral("pitch_after")]  = after.value(QStringLiteral("pitch_deg")).toDouble();
                        entry[QStringLiteral("roll_before")]  = before.value(QStringLiteral("roll_deg")).toDouble();
                        entry[QStringLiteral("roll_after")]   = after.value(QStringLiteral("roll_deg")).toDouble();
                        entry[QStringLiteral("pos_delta")]    = posDelta;
                        camComp.append(entry);
                    }
                    rep[QStringLiteral("camera_comparison")] = camComp;
                    rep[QStringLiteral("source")]            = QStringLiteral("workflow_aerial_triangulation");

                    if (!writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                           QStringLiteral("at_report.json"),
                                                           QStringLiteral("at_report_history.json"),
                                                           rep))
                    {
                        LOG_WARN(QStringLiteral("空三: 保存项目报告失败"));
                    }
                }

                // ── 将项目元数据更新派发到主线程（仅内存操作，ZIP 写由防抖定时器合批）──
                QMetaObject::invokeMethod(pm, [pm, result = std::move(result), sfmImages,
                                               sfmOutputDir]()
                {
                    // 记录自动提取的特征文件
                    for (const auto &sp : result.newSpFiles)
                    {
                        pm->appendIpfindResult(sp.imagePath, sp.spPath, QJsonObject());
                    }
                    // 记录自动生成的匹配文件
                    for (const auto &mr : result.newMatchFiles)
                    {
                        pm->appendIpmatchResult(QStringList{mr.matchPath}, mr.settings);
                    }
                    // 更新相机参数
                    if (!result.pendingCamUpdates.isEmpty())
                    {
                        int count = 0;
                        QString err;
                        if (pm->setImageCameras(result.pendingCamUpdates, &count, &err))
                        {
                            LOG_INFO(QStringLiteral("SFM: 更新了 %1 个相机参数到项目").arg(count));
                        }
                        else
                        {
                            LOG_WARN(QStringLiteral("SFM: 写回相机参数失败: %1").arg(err));
                        }
                    }
                    // 写入空三结果到 aerial_triangulation_results（刷新 DataTree "连接点"）
                    if (result.success && !result.sparseCloudPath.isEmpty())
                    {
                        pm->appendAtResult(result.sparseCloudPath,
                                           result.numPoints3D,
                                           sfmImages,
                                           sfmOutputDir);
                    }
                    if (result.success)
                    {
                        LOG_INFO(result.summary);
                    }
                    // 发出 AT 完成信号（成功或失败均发出）
                    emit pm->atProgressFinished(result.success);
                }, Qt::QueuedConnection);
            });
        }
        else if (needBaWithAutoPrep)
        {
            // ── 光束法平差：先用 baOnly 确保特征/匹配，然后执行 BA ──
            LOG_INFO(QStringLiteral("空三(BA): 启动特征/匹配预处理 (%1 张影像)").arg(images.size()));

            xjw::gui::SFMServiceOptions baOpts;
            baOpts.images      = images;
            baOpts.plascanPath = projectPath;
            baOpts.projectMeta = projectMeta;
            baOpts.quality     = settings.value(QStringLiteral("quality")).toInt(3);
            baOpts.threads     = threads;
            baOpts.baOnly      = true;
            baOpts.restrictPairs = useStoredPairs;
            baOpts.allowedPairs = allowedPairs;

            if (settings.contains(QStringLiteral("intrinsics")))
            {
                const QJsonObject intr = settings.value(QStringLiteral("intrinsics")).toObject();
                baOpts.userFu    = intr.value(QStringLiteral("fu")).toDouble();
                baOpts.userFv    = intr.value(QStringLiteral("fv")).toDouble();
                baOpts.userCu    = intr.value(QStringLiteral("cu")).toDouble();
                baOpts.userCv    = intr.value(QStringLiteral("cv")).toDouble();
                baOpts.userPitch = intr.value(QStringLiteral("pitch")).toDouble();
            }

            auto *pm = m_projectManager;
            const QStringList sfmImages = images;
            const QString sfmOutputDir  = outputDir;
            const QString assetsDir     = ProjectIO::projectAssetsDir(pm->currentProjectPath());

            // 记录 BA 前的相机参数（供报告展示平差前后对比）
            QMap<QString, QJsonObject> beforeCameras;
            {
                const QJsonObject meta = pm->coreProjectMeta();  // 仅需 images/camera，无需加载 results
                const QJsonArray  imgs = meta.value(QStringLiteral("images")).toArray();
                for (const QJsonValue &v : imgs)
                {
                    const QJsonObject imgObj = v.toObject();
                    const QString normPath = QDir::cleanPath(
                        QFileInfo(imgObj.value(QStringLiteral("path")).toString()).absoluteFilePath());
                    const QJsonObject camObj = imgObj.value(QStringLiteral("camera")).toObject();
                    if (!normPath.isEmpty() && !camObj.isEmpty())
                        beforeCameras.insert(normPath, camObj);
                }
            }

            baOpts.progressFn = [pm](const QString &stage, int pct)
            {
                QMetaObject::invokeMethod(pm, [pm, stage, pct]()
                {
                    emit pm->atProgressChanged(stage, pct);
                }, Qt::QueuedConnection);
            };

            // 创建取消标志并存储到 ProjectManager 中供 UI 层调用
            auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
            pm->setAtCancelFlag(cancelFlag);
            baOpts.cancelFlag = cancelFlag;

            emit pm->atProgressChanged(tr("启动光束法平差..."), 0);

            // 从 settings 提取 BA 参数
            QJsonObject baExtra;
            baExtra[QStringLiteral("refine_camera_pose")]      = settings.value(QStringLiteral("ba_refine_pose")).toBool(true);
            baExtra[QStringLiteral("max_iterations")]          = settings.value(QStringLiteral("ba_max_iterations")).toInt(20);
            baExtra[QStringLiteral("huber_delta")]             = settings.value(QStringLiteral("ba_huber_delta")).toDouble(2.0);
            baExtra[QStringLiteral("filter_max_reproj_error")] = settings.value(QStringLiteral("ba_filter_reproj")).toDouble(4.0);
            baExtra[QStringLiteral("damping")]                 = settings.value(QStringLiteral("ba_damping")).toDouble(1e-6);

            const int threadsVal = threads;
            (void)QtConcurrent::run([baOpts, pm, sfmImages, sfmOutputDir, assetsDir,
                                     beforeCameras = std::move(beforeCameras),
                                     baExtra, threadsVal]()
            {
                xjw::gui::SFMServiceResult prepResult = xjw::gui::SFMService::run(baOpts);

                QMetaObject::invokeMethod(pm, [pm, prepResult = std::move(prepResult),
                                               sfmImages, sfmOutputDir, assetsDir,
                                               beforeCameras, baExtra, threadsVal]() mutable {
                    // 记录自动提取的特征/匹配文件
                    for (const auto &sp : prepResult.newSpFiles)
                    {
                        pm->appendIpfindResult(sp.imagePath, sp.spPath, QJsonObject());
                    }
                    for (const auto &mr : prepResult.newMatchFiles)
                    {
                        pm->appendIpmatchResult(QStringList{mr.matchPath}, mr.settings);
                    }

                    if (!prepResult.featureMatchesReady)
                    {
                        const QString errMsg = prepResult.errorMessage.isEmpty()
                            ? QStringLiteral("特征/匹配准备失败")
                            : prepResult.errorMessage;
                        LOG_WARN(QStringLiteral("BA 前处理失败: %1").arg(errMsg));
                        emit pm->atProgressFinished(false);
                        return;
                    }

                    // 特征/匹配已就绪：注册一次性 BA 结果接收，然后执行光束法平差
                    emit pm->atProgressChanged(tr("执行光束法平差..."), 50);
                    LOG_INFO(QStringLiteral("BA: 特征/匹配准备完毕，启动光束法平差"));

                    // 以 ctx 为发射上下文，确保只触发一次
                    QObject *ctx = new QObject(pm);
                    QObject::connect(
                        pm, &ProjectManager::bundleAdjustPreviewReady,
                        ctx,
                        [pm, ctx, sfmImages, sfmOutputDir, assetsDir,
                         beforeCameras = std::move(beforeCameras)](const QJsonObject &) mutable {
                            ctx->deleteLater();  // 自动断开，只处理一次
                            pm->applyBundleAdjustForAt(assetsDir, sfmImages,
                                                       sfmOutputDir, beforeCameras);
                        },
                        Qt::DirectConnection);

                    pm->startBundleAdjustAsync(sfmImages, sfmOutputDir, threadsVal, false, baExtra);
                }, Qt::QueuedConnection);
            });
        }
    });

    dlg->exec();
}

void MenuWorkflowController::startGenerateModelWorkflow()
{
    if (!m_projectManager)
    {
        LOG_WARN(QStringLiteral("生成模型: 未找到 ProjectManager"));
        return;
    }
    QMetaObject::invokeMethod(m_projectManager, "startGenerateModelAsync", Qt::QueuedConnection);
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

// 工作流程菜单←税硾式一键创建稠密点云
void MenuWorkflowController::openCreatePointCloudDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new SimplePointCloudDialog(m_projectManager, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (m_projectManager)
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("dense_cloud")));
        }
    }

    connect(dlg, &SimplePointCloudDialog::runRequested,
            this, [this](const QJsonObject &settings)
    {
        if (!m_projectManager)
        {
            LOG_WARN(QStringLiteral("创建点云: 未找到 ProjectManager"));
            return;
        }
        QMetaObject::invokeMethod(m_projectManager,
                                  "startGenerateDenseCloudAsync",
                                  Qt::QueuedConnection,
                                  Q_ARG(QJsonObject, settings));
    });

    dlg->exec();
}

// 打开工作流程历史报告对话框（AT / 稠密点云 / 三维模型）
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