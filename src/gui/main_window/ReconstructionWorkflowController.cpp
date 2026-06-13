// =============================================================================
// 文件: ReconstructionWorkflowController.cpp
// 模块: main_window
// =============================================================================

#include "ReconstructionWorkflowController.h"
#include "MainWindow.h"
#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "graph/ObservationNetworkBuilder.h"

// ── 重建对话框头文件 ──
#include "ObservationNetworkDialog.h"
#include "InitCameraPoseDialog.h"
#include "TriangulationDialog.h"
#include "BundleAdjustDialog.h"
#include "SparseCloudPostProcessDialog.h"
#include "DenseMatchDialog.h"
#include "DepthMapEstimateDialog.h"
#include "DepthFusionDialog.h"
#include "DenseCloudRefineDialog.h"
#include "MeshReconstructionDialog.h"
#include "TextureMappingDialog.h"
#include "ModelExportDialog.h"

#include <QFile>
#include <QDir>
#include <QJsonDocument>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QDateTime>
#include <QtEndian>

// ---------------------------------------------------------------------------
// 构造 / 注入
// ---------------------------------------------------------------------------

ReconstructionWorkflowController::ReconstructionWorkflowController(
    QMainWindow *mainWindow, QObject *parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
{
}

void ReconstructionWorkflowController::setProjectManager(ProjectManager *pm)
{
    m_projectManager = pm;
}

QString ReconstructionWorkflowController::projectPath() const
{
    return m_projectManager ? m_projectManager->currentProjectPath() : QString();
}

// ============================================================================
// 稀疏重建
// ============================================================================

// ── 读取 SGMT 二进制匹配文件的匹配数和图像名（均可选）────────────────────────
static int readSgmtMatchCount(const QString &matchFile,
                              QString *img0BaseName = nullptr,
                              QString *img1BaseName = nullptr)
{
    QFile f(matchFile);
    if (!f.open(QIODevice::ReadOnly))
    {
        return -1;
    }

    // ── 尝试 SGMT 格式（新版 GUI 工具生成）─────────────────────────────────
    {
        QDataStream in(&f);
        in.setVersion(QDataStream::Qt_5_15);
        char magic[4];
        if (in.readRawData(magic, 4) == 4 && strncmp(magic, "SGMT", 4) == 0)
        {
            quint32 ver = 0;
            in >> ver;
            if (ver == 1)
            {
                quint32 n0 = 0;
                in >> n0;
                QByteArray b0(n0, 0);
                if (in.readRawData(b0.data(), n0) == static_cast<int>(n0))
                {
                    quint32 n1 = 0;
                    in >> n1;
                    QByteArray b1(n1, 0);
                    if (in.readRawData(b1.data(), n1) == static_cast<int>(n1))
                    {
                        if (img0BaseName)
                        {
                            *img0BaseName = QFileInfo(QString::fromUtf8(b0)).completeBaseName();
                        }
                        if (img1BaseName)
                        {
                            *img1BaseName = QFileInfo(QString::fromUtf8(b1)).completeBaseName();
                        }

                        qint32 nm = 0;
                        in >> nm;
                        return static_cast<int>(nm);
                    }
                }
            }
        }
    }

    // ── 回退：ASP/VisionWorkbench 格式（旧版工具生成）──────────────────────
    // 格式：uint64 LE × 2（两幅图匹配点数），数量相同即为实际匹配数
    f.seek(0);
    {
        QByteArray header = f.read(16);
        if (header.size() == 16)
        {
            quint64 cnt1 = 0, cnt2 = 0;
            memcpy(&cnt1, header.constData(),     8);
            memcpy(&cnt2, header.constData() + 8, 8);
            cnt1 = qFromLittleEndian<quint64>(cnt1);
            cnt2 = qFromLittleEndian<quint64>(cnt2);
            // 若两端计数相等且合理，则为匹配点数
            if (cnt1 == cnt2 && cnt1 > 0 && cnt1 < 1000000)
            {
                return static_cast<int>(cnt1);
            }
        }
    }

    return -1;
}

// ── 从 ipmatch_results 元数据构建 MatchEdge 列表的辅助函数 ──────────────────
static QVector<xjw::MatchEdge> buildMatchEdges(
    const QJsonArray &results,
    QStringList      &outImageNames)
{
    // 1) 收集所有图像名称，建立 name → index 映射
    QMap<QString, int> nameToIdx;
    auto getIdx = [&](const QString &name) -> int {
        if (name.isEmpty())
        {
            return -1;
        }

        auto it = nameToIdx.find(name);
        if (it != nameToIdx.end())
        {
            return it.value();
        }

        int idx = outImageNames.size();
        outImageNames.append(name);
        nameToIdx.insert(name, idx);
        return idx;
    };

    QVector<xjw::MatchEdge> edges;
    edges.reserve(results.size());

    for (const QJsonValue &val : results)
    {
        if (!val.isObject())
        {
            continue;
        }

        const QJsonObject rec = val.toObject();

        // 新格式: image0_name / image1_name（SuperGlue sidecar）
        QString name0 = rec.value(QStringLiteral("image0_name")).toString();
        QString name1 = rec.value(QStringLiteral("image1_name")).toString();

        // 兼容旧格式: settings.image_files[]
        if (name0.isEmpty() || name1.isEmpty())
        {
            const QJsonObject settings = rec.value(QStringLiteral("settings")).toObject();
            const QJsonArray  files    = settings.value(QStringLiteral("image_files")).toArray();
            if (files.size() >= 2)
            {
                if (name0.isEmpty())
                {
                    name0 = QFileInfo(files.at(0).toString()).completeBaseName();
                }
                if (name1.isEmpty())
                {
                    name1 = QFileInfo(files.at(1).toString()).completeBaseName();
                }
            }
        }
        // 兼容新版精简格式: image0/image1 = 影像全路径，取 basename
        if (name0.isEmpty())
        {
            name0 = QFileInfo(rec.value(QStringLiteral("image0")).toString()).completeBaseName();
        }
        if (name1.isEmpty())
        {
            name1 = QFileInfo(rec.value(QStringLiteral("image1")).toString()).completeBaseName();
        }
        // 旧版兜底
        if (name0.isEmpty())
        {
            name0 = QFileInfo(rec.value(QStringLiteral("image0_path")).toString()).completeBaseName();
        }
        if (name1.isEmpty())
        {
            name1 = QFileInfo(rec.value(QStringLiteral("image1_path")).toString()).completeBaseName();
        }

        if (name0.isEmpty() || name1.isEmpty())
        {
            continue;
        }

        const int idx0 = getIdx(name0);
        const int idx1 = getIdx(name1);
        // 读取匹配数：优先用记录里存好的值，否则直接解析 SGMT 二进制文件
        int nm = rec.value(QStringLiteral("num_matches")).toInt(-1);
        if (nm < 0)
        {
            // 兼容新格式 "output"（字符串）和旧格式 "outputs"（数组）
            QString matchPath = rec.value(QStringLiteral("output")).toString();
            if (matchPath.isEmpty())
            {
                const QJsonArray outs = rec.value(QStringLiteral("outputs")).toArray();
                if (!outs.isEmpty())
                {
                    matchPath = outs.at(0).toString();
                }
            }
            if (!matchPath.isEmpty())
            {
                QString sgmtN0, sgmtN1;
                nm = readSgmtMatchCount(matchPath,
                                        name0.isEmpty() ? &sgmtN0 : nullptr,
                                        name1.isEmpty() ? &sgmtN1 : nullptr);
                if (name0.isEmpty() && !sgmtN0.isEmpty())
                {
                    name0 = sgmtN0;
                }
                if (name1.isEmpty() && !sgmtN1.isEmpty())
                {
                    name1 = sgmtN1;
                }
            }
        }
        if (nm < 0)
        {
            nm = 0;
        }

        xjw::MatchEdge e;
        e.idx0        = idx0;
        e.idx1        = idx1;
        e.numMatches  = nm;
        e.overlapScore = 0.0;
        e.inlierRatio  = 1.0;   // 暂无内点率字段，视为全部有效
        edges.append(e);
    }
    return edges;
}

void ReconstructionWorkflowController::openObservationNetworkDialog()
{
    auto *dlg = prepareDialog<ObservationNetworkDialog>(
        DialogSettingKeys::ObservationNetwork, m_obsNetStore);
    if (!dlg)
    {
        return;
    }

    // ── 注入匹配数据 ────────────────────────────────────────────────────────
    QVector<xjw::MatchEdge> allEdges;
    QStringList             allImageNames;

    if (m_projectManager)
    {
        QJsonObject meta = m_projectManager->currentMeta();
        QJsonArray results = meta.value(QStringLiteral("ipmatch_results")).toArray();
        allEdges = buildMatchEdges(results, allImageNames);
        dlg->setMatchEdges(allEdges, allImageNames);
    }

    // ── 收到"执行构建"信号 → 后台运行算法 ───────────────────────────────
    auto *pm = m_projectManager;
    auto *mw = m_mainWindow.data();

    connect(
        dlg,
        &ObservationNetworkDialog::runRequested,
        this,
        [this, pm, mw, allEdges, allImageNames](const QJsonObject &settings)
        {
            if (!pm)
            {
                return;
            }

            // 1) 发出进度: 准备中 0%
            QMetaObject::invokeMethod(
                pm,
                [pm]()
                {
                    emit pm->obsNetProgressChanged(QStringLiteral("准备中"), 0);
                },
                Qt::QueuedConnection);

            // 2) 将 Qt 数据转换为 std 类型（在主线程完成，避免隐式共享竞争）
            std::vector<std::string> nodeNames;
            nodeNames.reserve(allImageNames.size());
            for (const QString &n : allImageNames)
            {
                nodeNames.push_back(n.toStdString());
            }

            std::vector<xjw::MatchEdge> edges(allEdges.begin(), allEdges.end());

            // 3) 组装配置
            xjw::ObservationNetworkConfig cfg;
            const int algoIndex = settings.contains(QStringLiteral("graphAlgoIndex"))
                ? settings.value(QStringLiteral("graphAlgoIndex")).toInt()
                : 1;
            switch (algoIndex)
            {
                case 0:
                    cfg.algorithm = xjw::ObservationNetworkConfig::Complete;
                    break;
                case 1:
                    cfg.algorithm = xjw::ObservationNetworkConfig::KNN;
                    break;
                case 2:
                    cfg.algorithm = xjw::ObservationNetworkConfig::MST;
                    break;
                case 3:
                    cfg.algorithm = xjw::ObservationNetworkConfig::Spatial;
                    break;
                case 4:
                    cfg.algorithm = xjw::ObservationNetworkConfig::KDTree;
                    break;
                default:
                    cfg.algorithm = xjw::ObservationNetworkConfig::KNN;
                    break;
            }

            const QString algoStr = settings.value(QStringLiteral("graphAlgorithm")).toString();
            cfg.k = settings.value(QStringLiteral("maxNeighbors")).toInt(20);
            cfg.minMatches = settings.value(QStringLiteral("minMatchCount")).toInt(30);
            cfg.minOverlap = settings.value(QStringLiteral("minOverlap")).toDouble(0.10);
            cfg.pruneWeak = settings.value(QStringLiteral("pruneWeak")).toBool(true);
            cfg.pruneThresh = settings.value(QStringLiteral("pruneThreshold")).toDouble(0.15);

            // 4) GPS 坐标（暂无时传空向量）
            std::vector<xjw::GpsCoord> gps;

            // 5) 启动后台任务
            using WatcherT = QFutureWatcher<xjw::ObservationNetwork>;
            auto *watcher = new WatcherT(mw ? static_cast<QObject *>(mw) : this);

            connect(
                watcher,
                &WatcherT::finished,
                this,
                [pm, watcher, algoStr]()
                {
                    try
                    {
                        xjw::ObservationNetwork net = watcher->result();
                        watcher->deleteLater();
                        LOG_INFO(
                            QStringLiteral("观测网络构建完成: %1 节点, %2 边")
                                .arg(net.numNodes()).arg(net.numEdges()));

                        QJsonObject extra;
                        QJsonArray edgesArr;
                        for (const auto &edge : net.edges)
                        {
                            QJsonObject edgeJson;
                            edgeJson[QStringLiteral("i")] = edge.idx0;
                            edgeJson[QStringLiteral("j")] = edge.idx1;
                            edgeJson[QStringLiteral("w")] = edge.weight;
                            edgeJson[QStringLiteral("n")] = edge.numMatches;
                            edgesArr.append(edgeJson);
                        }

                        QJsonArray namesArr;
                        for (const auto &nodeName : net.nodeNames)
                        {
                            namesArr.append(QString::fromStdString(nodeName));
                        }
                        extra[QStringLiteral("edges")] = edgesArr;
                        extra[QStringLiteral("node_names")] = namesArr;

                        const QString plascanPath = pm->currentProjectPath();
                        if (!plascanPath.isEmpty())
                        {
                            const QString assetsDir = ProjectIO::projectAssetsDir(plascanPath);
                            const QString obsnetDir = QDir(assetsDir).filePath(QStringLiteral("obsnet"));
                            QDir().mkpath(obsnetDir);
                            const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
                            const QString netFile = QDir(obsnetDir).filePath(
                                QStringLiteral("%1_%2.json").arg(algoStr, ts));

                            QJsonObject netObj;
                            netObj[QStringLiteral("algorithm")] = algoStr;
                            netObj[QStringLiteral("node_names")] = namesArr;
                            netObj[QStringLiteral("edges")] = edgesArr;

                            QFile networkFile(netFile);
                            if (networkFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
                            {
                                networkFile.write(QJsonDocument(netObj).toJson(QJsonDocument::Compact));
                                networkFile.close();
                                extra[QStringLiteral("network_file")] = netFile;
                                LOG_INFO(
                                    QStringLiteral("观测网络已保存: %1").arg(netFile));
                            }
                        }

                        pm->appendObsNetResult(net.numNodes(), net.numEdges(), algoStr, extra);
                        QMetaObject::invokeMethod(
                            pm,
                            [pm]()
                            {
                                emit pm->obsNetProgressFinished(true);
                            },
                            Qt::QueuedConnection);
                    }
                    catch (const std::exception &ex)
                    {
                        watcher->deleteLater();
                        LOG_ERROR(
                            QStringLiteral("观测网络构建异常: %1")
                                .arg(QString::fromStdString(ex.what())));
                        QMetaObject::invokeMethod(
                            pm,
                            [pm]()
                            {
                                emit pm->obsNetProgressFinished(false);
                            },
                            Qt::QueuedConnection);
                    }
                });

            auto progressFn = [pm](const QString &stage, int pct)
            {
                QMetaObject::invokeMethod(
                    pm,
                    [pm, stage, pct]()
                    {
                        emit pm->obsNetProgressChanged(stage, pct);
                    },
                    Qt::QueuedConnection);
            };

            watcher->setFuture(QtConcurrent::run(
                [nodeNames, edges, gps, cfg, progressFn]() -> xjw::ObservationNetwork
                {
                    progressFn(QStringLiteral("过滤匹配对"), 20);
                    xjw::ObservationNetwork net =
                        xjw::ObservationNetworkBuilder::build(nodeNames, edges, gps, cfg);
                    progressFn(QStringLiteral("构建完成"), 100);
                    return net;
                }));

            LOG_INFO(
                QStringLiteral("观测网络构建已启动: algo=%1 k=%2 minMatches=%3")
                    .arg(algoStr).arg(cfg.k).arg(cfg.minMatches));
        });

    dlg->exec();
}

void ReconstructionWorkflowController::openInitCameraPoseDialog()
{
    auto *dlg = prepareDialog<InitCameraPoseDialog>(
        DialogSettingKeys::InitCameraPose, m_initPoseStore);
    if (!dlg)
    {
        return;
    }

    QStringList imagePaths;
    if (m_projectManager)
    {
        imagePaths = xjw::gui::project::projectImagePaths(m_projectManager->currentMeta());
        dlg->setAvailableFeatureSuffixes(
            xjw::gui::project::projectFeatureSuffixes(m_projectManager->currentProjectPath(),
                                                       m_projectManager->currentMeta()));
    }
    dlg->setAvailableImages(imagePaths);
    if (m_initPoseStore)
    {
        const QJsonObject saved = m_initPoseStore->load();
        if (!saved.isEmpty())
        {
            dlg->applySettings(saved);
        }
    }

    connect(
        dlg,
        &InitCameraPoseDialog::runRequested,
        this,
        [this, dlg](const QJsonObject &s)
        {
            const int mode = s.value(QStringLiteral("mode")).toInt();

            if (!m_projectManager)
            {
                return;
            }

            if (mode == 2)
            {
                const int importMode = s.value(QStringLiteral("cameraImportMode")).toInt(0);
                bool imported = false;
                if (importMode == 0)
                {
                    const QString imagePath = s.value(QStringLiteral("targetImagePath")).toString();
                    if (imagePath.isEmpty())
                    {
                        QMessageBox::warning(
                            m_mainWindow,
                            QStringLiteral("初始化相机位姿"),
                            QStringLiteral("请先选择需要导入相机文件的影像。"));
                        return;
                    }
                    imported = m_projectManager->importCameraForImage(imagePath);
                }
                else
                {
                    imported = m_projectManager->importCamerasByFilenameBatch();
                }

                if (imported)
                {
                    LOG_INFO(
                        QStringLiteral("初始化相机位姿: 复用相机导入流程 %1")
                            .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));
                    dlg->accept();
                }
                return;
            }

            bool ok = false;
            if (mode == 0 || mode == 1)
            {
                ok = m_projectManager->initializeCameraPosesWithSFM(s);
            }

            if (ok)
            {
                LOG_INFO(
                    QStringLiteral("初始化相机位姿: %1")
                        .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));
                dlg->accept();
            }
        });

    dlg->exec();
}

void ReconstructionWorkflowController::openTriangulationDialog()
{
    auto *dlg = prepareDialog<TriangulationDialog>(
        DialogSettingKeys::Triangulation, m_triStore);
    if (!dlg)
    {
        return;
    }

    connect(dlg, &TriangulationDialog::runRequested, this, [this, dlg](const QJsonObject &s)
    {
        if (!m_projectManager)
        {
            return;
        }

        LOG_INFO(QStringLiteral("三角化: %1")
            .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));
        m_projectManager->startTriangulationAsync(s);
        dlg->accept();
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openReconBundleAdjustDialog()
{
    auto *dlg = prepareDialog<BundleAdjustDialog>(
        DialogSettingKeys::ReconBundleAdjust, m_reconBaStore);
    if (!dlg)
    {
        return;
    }

    if (m_projectManager)
    {
        const QStringList images = m_projectManager->getAllImages();
        if (!images.isEmpty())
        {
            dlg->setAvailableImages(images);
        }

        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("ba")));
        }

        if (m_reconBaStore)
        {
            const QJsonObject saved = m_reconBaStore->load();
            if (!saved.isEmpty())
            {
                dlg->applySettings(saved);
            }
        }
    }

    connect(
        dlg,
        &BundleAdjustDialog::requestRestore,
        this,
        [this, dlg]()
        {
            if (!m_reconBaStore || !dlg)
            {
                return;
            }
            const QJsonObject saved = m_reconBaStore->load();
            if (!saved.isEmpty())
            {
                dlg->applySettings(saved);
            }
        });

    // BundleAdjustDialog 有自己的 run 信号: requestRunBundleAdjust
    connect(
        dlg,
        &BundleAdjustDialog::requestRunBundleAdjust,
        this,
        [this](const QStringList &images,
               const QString &outputDir,
               int threads,
               bool dryRun,
               const QJsonObject &extra)
        {
            if (!m_projectManager)
            {
                return;
            }
            m_projectManager->startBundleAdjustAsync(images, outputDir, threads, dryRun, extra);
        });

    connect(
        dlg,
        &BundleAdjustDialog::requestApplyBundleAdjustResult,
        this,
        [this]()
        {
            if (!m_projectManager)
            {
                return;
            }
            QString err;
            if (!m_projectManager->acceptBundleAdjustPreview(&err) && !err.isEmpty())
            {
                LOG_WARN(QStringLiteral("BA 应用失败: %1").arg(err));
            }
        });

    connect(
        dlg,
        &BundleAdjustDialog::requestDiscardBundleAdjustResult,
        this,
        [this]()
        {
            if (m_projectManager)
            {
                m_projectManager->discardBundleAdjustPreview();
            }
        });

    if (m_projectManager)
    {
        connect(m_projectManager, &ProjectManager::bundleAdjustPreviewReady,
                dlg, &BundleAdjustDialog::setRunResult);
    }

    dlg->show();
}

// ============================================================================
// 稀疏点云后处理
// ============================================================================

void ReconstructionWorkflowController::openSparseCloudPostProcessDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new SparseCloudPostProcessDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    if (m_projectManager)
    {
        if (!m_sparsePostStore)
        {
            m_sparsePostStore = new DialogSettingStore(DialogSettingKeys::SparseCloudPostProcess, this);
        }
        m_sparsePostStore->setProjectPath(projectPath());

        const QJsonObject saved = m_sparsePostStore->load();
        dlg->applySettings(saved);

        connect(dlg, &SparseCloudPostProcessDialog::settingsChanged, this, [this](const QJsonObject &settings)
        {
            if (m_sparsePostStore)
            {
                m_sparsePostStore->save(settings);
            }
        });

        dlg->setAvailableSparseClouds(m_projectManager->getAvailableAtResults());
    }

    connect(dlg, &SparseCloudPostProcessDialog::runRequested, this, [this](const QJsonObject &settings)
    {
        const QString mode = settings.value(QStringLiteral("mode")).toString(QStringLiteral("outlier_removal"));
        LOG_INFO(QStringLiteral("稀疏点云后处理[%1]: %2")
            .arg(mode)
            .arg(QString::fromUtf8(QJsonDocument(settings).toJson(QJsonDocument::Compact))));

        if (!m_projectManager)
        {
            return;
        }

        if (mode == QLatin1String("refine"))
        {
            m_projectManager->startSparseCloudRefineAsync(settings);
        }
        else if (mode == QLatin1String("spatial_cleanup"))
        {
            m_projectManager->startSparseCloudLocalOptimAsync(settings);
        }
        else
        {
            m_projectManager->startSparseCloudOutlierRemovalAsync(settings);
        }
    });

    dlg->exec();
}

// ============================================================================
// 密集重建
// ============================================================================

void ReconstructionWorkflowController::openDenseMatchDialog()
{
    if (!m_mainWindow || !m_projectManager)
    {
        return;
    }

    auto *dlg = new DenseMatchDialog(m_projectManager, m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    // 恢复记忆化设置
    if (!m_denseMatchStore)
    {
        m_denseMatchStore = new DialogSettingStore(
            DialogSettingKeys::DenseMatch, this);
    }
    m_denseMatchStore->setProjectPath(projectPath());
    const QJsonObject saved = m_denseMatchStore->load();
    if (!saved.isEmpty())
    {
        dlg->applySettings(saved);
    }

    // 实时持久化
    connect(dlg, &DenseMatchDialog::settingsChanged, this,
            [this](const QJsonObject &s)
    {
        if (m_denseMatchStore)
            m_denseMatchStore->save(s);
    });

    // 运行（进度条在主窗口状态栏右下角）
    auto *mw = qobject_cast<MainWindow*>(m_mainWindow);
    connect(dlg, &DenseMatchDialog::runRequested, this,
            [this, mw, dlg_ptr = QPointer<DenseMatchDialog>(dlg)]
            (const QJsonObject &settings)
    {
        if (!m_projectManager) return;

        const QJsonArray pairs = settings.value(QStringLiteral("match_pairs")).toArray();
        const int totalPairs = pairs.size();
        if (totalPairs == 0) return;

        // 每对影像 5 个步骤: 加载+代价计算+匹配+验证+保存
        const int totalSteps = totalPairs * 5;

        LOG_INFO(QStringLiteral("密集匹配启动: %1 个匹配对, 共 %2 步")
                     .arg(totalPairs).arg(totalSteps));

        if (mw) mw->showDmProgress(totalSteps);

        auto progress = std::make_shared<std::atomic<int>>(0);
        auto dmCancelFlag = std::make_shared<std::atomic<bool>>(false);
        auto *watcher = new QFutureWatcher<void>(this);
        auto *timer = new QTimer(this);

        QMetaObject::Connection cancelConn;
        if (mw)
        {
            cancelConn = connect(mw, &MainWindow::dmCancelRequested,
                                 this, [dmCancelFlag]()
            {
                dmCancelFlag->store(true);
            });
        }

        connect(timer, &QTimer::timeout, this,
                [mw, progress, totalSteps]()
        {
            if (mw) mw->updateDmProgress(progress->load());
        });
        timer->start(150);

        connect(watcher, &QFutureWatcher<void>::finished, this,
                [timer, watcher, mw, dlg_ptr, progress, dmCancelFlag, cancelConn]()
        {
            timer->stop();
            timer->deleteLater();
            watcher->deleteLater();
            if (mw)
            {
                QObject::disconnect(cancelConn);
                const bool cancelled = dmCancelFlag->load();
                mw->hideDmProgress(!cancelled);
            }
            if (dlg_ptr)
            {
                dlg_ptr->onProcessingFinished();
            }
        });

        watcher->setFuture(QtConcurrent::run(
            [this, settings, progress, dmCancelFlag]()
        {
            m_projectManager->startDenseMatchAsyncWithProgress(settings, progress, dmCancelFlag);
        }));
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openDepthMapEstimateDialog()
{
    auto *dlg = prepareDialog<DepthMapEstimateDialog>(
        DialogSettingKeys::DepthMapEstimate, m_depthEstStore);
    if (!dlg)
    {
        return;
    }

    if (m_projectManager)
    {
        dlg->setAvailableAtResults(m_projectManager->getAvailableAtResults());
        if (m_depthEstStore)
        {
            const QJsonObject saved = m_depthEstStore->load();
            if (!saved.isEmpty())
            {
                dlg->applySettings(saved);
            }
        }
    }

    connect(dlg, &DepthMapEstimateDialog::runRequested, this, [this](const QJsonObject &depthSettings)
    {
        if (!m_projectManager)
        {
            return;
        }

        LOG_INFO(QStringLiteral("深度图估计: resScale=%1 iterations=%2 confidence=%3")
            .arg(depthSettings.value(QStringLiteral("resScale")).toDouble())
            .arg(depthSettings.value(QStringLiteral("iterations")).toInt())
            .arg(depthSettings.value(QStringLiteral("confidence")).toDouble()));

        m_projectManager->startEstimateDepthMapsAsync(depthSettings);
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openDepthFusionDialog()
{
    auto *dlg = prepareDialog<DepthFusionDialog>(
        DialogSettingKeys::DepthFusion, m_depthFuseStore);
    if (!dlg)
    {
        return;
    }

    connect(dlg, &DepthFusionDialog::runRequested, this, [this](const QJsonObject &fuseSettings)
    {
        if (!m_projectManager)
        {
            return;
        }

        LOG_INFO(QStringLiteral("深度图融合: minConsistentViews=%1 minConfidence=%2")
            .arg(fuseSettings.value(QStringLiteral("minConsistentViews")).toInt())
            .arg(fuseSettings.value(QStringLiteral("minConfidence")).toDouble()));

        m_projectManager->startFuseDepthMapsAsync(fuseSettings);
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openDenseCloudRefineDialog()
{
    auto *dlg = prepareDialog<DenseCloudRefineDialog>(
        DialogSettingKeys::DenseCloudRefine, m_denseRefStore);
    if (!dlg)
    {
        return;
    }

    connect(dlg, &DenseCloudRefineDialog::runRequested, this, [this](const QJsonObject &settings)
    {
        if (!m_projectManager)
        {
            return;
        }

        LOG_INFO(QStringLiteral("密集点云后处理: sorEnabled=%1 voxelEnabled=%2 normalsEnabled=%3")
            .arg(settings.value(QStringLiteral("sorEnabled")).toBool())
            .arg(settings.value(QStringLiteral("voxelEnabled")).toBool())
            .arg(settings.value(QStringLiteral("normalsEnabled")).toBool()));

        m_projectManager->startDenseCloudRefineAsync(settings);
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openMeshReconstructionDialog()
{
    auto *dlg = prepareDialog<MeshReconstructionDialog>(
        DialogSettingKeys::MeshReconstruction, m_meshStore);
    if (!dlg)
    {
        return;
    }

    if (m_projectManager)
    {
        QStringList denseCandidates;
        const QJsonArray denseResults = m_projectManager->currentMeta().value(QStringLiteral("dense_cloud_results")).toArray();
        for (int index = denseResults.size() - 1; index >= 0; --index)
        {
            const QString candidate = QDir::cleanPath(
                denseResults.at(index).toObject().value(QStringLiteral("dense_cloud_xyz")).toString());
            if (candidate.isEmpty() || !QFileInfo::exists(candidate) || denseCandidates.contains(candidate))
            {
                continue;
            }
            denseCandidates.push_back(candidate);
        }

        dlg->setDenseCloudCandidates(denseCandidates);
    }

    connect(dlg, &MeshReconstructionDialog::runRequested, this, [this](const QJsonObject &s)
    {
        LOG_INFO(QStringLiteral("网格重建: %1")
            .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));

        if (!m_projectManager)
        {
            return;
        }

        m_projectManager->startMeshReconstructionAsync(s);
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openTextureMappingDialog()
{
    auto *dlg = prepareDialog<TextureMappingDialog>(
        DialogSettingKeys::TextureMapping, m_texStore);
    if (!dlg)
    {
        return;
    }

    connect(dlg, &TextureMappingDialog::runRequested, this, [this](const QJsonObject &s)
    {
        LOG_INFO(QStringLiteral("纹理映射: %1")
            .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));

        if (!m_projectManager)
        {
            return;
        }

        m_projectManager->startTextureMappingAsync(s);
    });

    dlg->exec();
}

void ReconstructionWorkflowController::openModelExportDialog()
{
    auto *dlg = prepareDialog<ModelExportDialog>(
        DialogSettingKeys::ModelExport, m_exportStore);
    if (!dlg)
    {
        return;
    }

    connect(dlg, &ModelExportDialog::runRequested, this, [](const QJsonObject &s)
    {
        LOG_INFO(QStringLiteral("模型导出: %1")
            .arg(QString::fromUtf8(QJsonDocument(s).toJson(QJsonDocument::Compact))));
    });

    dlg->exec();
}
