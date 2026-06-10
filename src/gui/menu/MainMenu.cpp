/**
 * @file MainMenu.cpp
 * @brief MainMenu 的实现文件。
 *
 * 包含菜单栏构建、最近项目子菜单的动态重建，以及全部访问器方法的实现。
 * 本文件不包含任何业务逻辑，所有 QAction 的 triggered 信号由主窗口负责连接。
 */
#include "MainMenu.h"
#include "WindowPanel.h"

#include <QDir>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QToolBar>
#include <QApplication>
#include <QStatusBar>
#include <QWidgetAction>
#include <QFileInfo>
#include <QSize>
#include <QStyle>
#include <Qt>

// ============================================================
//  构造函数 — 创建完整的菜单结构
// ============================================================

/**
 * @brief 构造函数：在 mainWindow 上创建所有菜单项和工具栏。
 *
 * 菜单创建顺序（与菜单栏从左到右排列一致）：
 *   1. 项目菜单（新建、打开、最近打开子菜单、保存、退出）
 *   2. 视图菜单（缩放、可视化设置、窗口面板子菜单）
 *   3. 工作流程菜单（空三、模型、DEM、正射影像）
 *   4. 工具菜单（重叠度、交汇、报告）
 *   5. 帮助菜单（关于）
 *   6. 主工具栏
 *
 * @param mainWindow 目标主窗口，不可为 nullptr。
 */
MainMenu::MainMenu(QMainWindow *mainWindow)
    : QObject(mainWindow), m_mainWindow(mainWindow)
{
    if (!m_mainWindow) return;

    // ---- 项目菜单 ----
    // 创建顶级"项目"菜单并依次添加固定动作
    m_fileMenu = m_mainWindow->menuBar()->addMenu(tr("项目"));
    m_newAct   = m_fileMenu->addAction(tr("新建项目"));
    m_openAct  = m_fileMenu->addAction(tr("打开项目"));
    m_saveAct  = m_fileMenu->addAction(tr("保存项目"));

    // "最近打开"子菜单插入到"保存项目"之前，保持菜单项顺序符合直觉
    m_recentMenu = new QMenu(tr("最近打开"), m_fileMenu);
    m_fileMenu->insertMenu(m_saveAct, m_recentMenu);

    auto *exportMenu = m_fileMenu->addMenu(tr("导出"));
    m_exportMatchedPairsAct = exportMenu->addAction(tr("导出匹配对(.lis)"));

    m_fileMenu->addSeparator();
    m_minimizeAct = m_fileMenu->addAction(tr("最小化"));
    // 退出动作直接连接到 QCoreApplication::quit，无需额外连接
    m_exitAct = m_fileMenu->addAction(tr("退出"), qApp, &QCoreApplication::quit);

    // ---- 视图菜单 ----
    auto *viewMenu = m_mainWindow->menuBar()->addMenu(tr("视图"));
    m_zoomInAct    = viewMenu->addAction(tr("放大"));
    m_zoomOutAct   = viewMenu->addAction(tr("缩小"));
    m_resetViewAct = viewMenu->addAction(tr("重置视图"));
    viewMenu->addSeparator();
    // 操控球显示/隐藏切换
    m_toggleGizmoAct = new QAction(tr("显示操控球"), viewMenu);
    m_toggleGizmoAct->setCheckable(true);
    m_toggleGizmoAct->setChecked(true);  // 默认显示
    m_toggleGizmoAct->setToolTip(tr("显示或隐藏 3D 视图中的旋转操控球"));
    viewMenu->addAction(m_toggleGizmoAct);
    viewMenu->addSeparator();
    // 特征点可视化设置对话框入口
    m_featureVisualizationAct = viewMenu->addAction(tr("特征点 可视化设置..."));
    viewMenu->addSeparator();

    // 窗口面板子菜单：使用 QWidgetAction + WindowPanel 实现带复选框的面板开关列表
    auto *windowMenu = viewMenu->addMenu(tr("窗口"));
    m_toggleLogAct = new QAction(tr("日志"), windowMenu);
    m_toggleLogAct->setCheckable(true);  // 可切换：勾选时面板可见
    m_toggleLogAct->setChecked(true);    // 默认显示日志面板
    m_featureInfoAct = new QAction(tr("兴趣点信息"), windowMenu);
    m_featureInfoAct->setCheckable(true); // 可切换：勾选时面板可见

    // 将动作列表传给 WindowPanel 组件，以列表形式展示在子菜单中
    QList<QAction*> windowActs = { m_toggleLogAct, m_featureInfoAct };
    auto *panelAct = new QWidgetAction(windowMenu);
    auto *wp = new WindowPanel(windowMenu);
    panelAct->setDefaultWidget(wp);
    windowMenu->addAction(panelAct);
    wp->setActions(windowActs);

    // ---- 工作流程菜单 ----
    // 提供高层一键式处理流程入口，适合不需要分步调试的普通用户
    auto *workflowMenu = m_mainWindow->menuBar()->addMenu(tr("工作流程"));
    m_addPhotoAct       = workflowMenu->addAction(tr("添加 照片"));
    m_addFolderAct      = workflowMenu->addAction(tr("添加 文件夹"));
    workflowMenu->addSeparator();
    m_threeDReconstructionAct = workflowMenu->addAction(tr("三维重建"));     // 一键完整建模流程
    m_createDEMAct      = workflowMenu->addAction(tr("创建 DEM"));          // DEM 完整流程
    m_generateOrthoAct  = workflowMenu->addAction(tr("生成 正射影像"));     // 正射影像完整流程

    // ---- 重建菜单 ----
    // 三级菜单结构：稀疏重建 / 密集重建 / 模型生成
    auto *reconMenu = m_mainWindow->menuBar()->addMenu(tr("重建"));

    // ── 稀疏重建 ──
    auto *sparseReconMenu = reconMenu->addMenu(tr("稀疏重建"));
    m_detectFeaturesAct = sparseReconMenu->addAction(tr("特征点提取"));
    m_matchFeaturesAct  = sparseReconMenu->addAction(tr("创建连接点"));
    sparseReconMenu->addSeparator();
    m_buildObsNetworkAct     = sparseReconMenu->addAction(tr("构建观测网络模型..."));
    m_initCameraPoseAct      = sparseReconMenu->addAction(tr("初始化相机位姿..."));
    m_triangulateAct         = sparseReconMenu->addAction(tr("生成初始稀疏点云..."));
    m_reconBundleAdjustAct   = sparseReconMenu->addAction(tr("光束法平差优化..."));
    m_sparseCloudPostProcessAct = sparseReconMenu->addAction(tr("稀疏点云后处理..."));

    // ── 密集重建 ──
    auto *denseReconMenu = reconMenu->addMenu(tr("密集重建"));
    m_denseMatchAct       = denseReconMenu->addAction(tr("密集匹配..."));
    m_depthMapEstimateAct = denseReconMenu->addAction(tr("深度图估计..."));
    m_fuseDepthMapsAct    = denseReconMenu->addAction(tr("深度图融合生成密集点云..."));
    m_refineDenseCloudAct = denseReconMenu->addAction(tr("密集点云后处理..."));

    // ── 模型生成 ──
    auto *modelGenMenu = reconMenu->addMenu(tr("模型生成"));
    m_meshReconstructAct = modelGenMenu->addAction(tr("网格重建..."));
    m_textureMappingAct  = modelGenMenu->addAction(tr("纹理映射..."));
    m_exportModelAct     = modelGenMenu->addAction(tr("模型导出..."));

    // ---- 工具菜单 ----
    // 提供细粒度的单步工具入口，供高级用户和调试场景使用
    auto *toolsMenu = m_mainWindow->menuBar()->addMenu(tr("工具"));

    // 质量检查工具
    m_overlapAnalysisAct = toolsMenu->addAction(tr("重叠度获取"));
    QMenu *intersectionMenu = toolsMenu->addMenu(tr("前方交汇精度检验"));
    m_intersectionCheckAct = intersectionMenu->addAction(tr("检测交汇"));
    m_intersectionViewResultsAct = intersectionMenu->addAction(tr("查看结果"));
    toolsMenu->addSeparator();
    m_manualPointCloudPruneAct = toolsMenu->addAction(tr("手动点云剔除"));
    toolsMenu->addSeparator();
    m_viewMatchesAct = toolsMenu->addAction(tr("连接点查看"));

    // 报告：查看各工作流程的历史统计报告
    toolsMenu->addSeparator();
    m_viewWorkflowReportAct = toolsMenu->addAction(tr("查看工作流程报告..."));

    // ---- 帮助菜单 ----
    auto *helpMenu = m_mainWindow->menuBar()->addMenu(tr("帮助"));
    helpMenu->addAction(tr("关于"), m_mainWindow, [mw = m_mainWindow]() {
        // 在状态栏短暂显示应用说明，3 秒后自动清除
        mw->statusBar()->showMessage(tr("PlaScan: 行星表面摄影测量处理系统"), 3000);
    });

    // ---- 主工具栏 ----
    m_toolBar = m_mainWindow->addToolBar(tr("工具"));
    if (m_toolBar)
    {
        m_toolBar->setMovable(false);
        m_toolBar->setFloatable(false);
        m_toolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_toolBar->setIconSize(QSize(24, 24));

        if (m_saveAct)
        {
            m_saveAct->setIcon(m_mainWindow->style()->standardIcon(QStyle::SP_DialogSaveButton));
            m_toolBar->addAction(m_saveAct);
        }
        if (m_manualPointCloudPruneAct)
        {
            m_manualPointCloudPruneAct->setIcon(m_mainWindow->style()->standardIcon(QStyle::SP_CommandLink));
            m_toolBar->addAction(m_manualPointCloudPruneAct);
        }
    }
}

/** @brief 析构函数（默认实现，所有成员由 Qt 对象树释放）。 */
MainMenu::~MainMenu() = default;

// ============================================================
//  最近打开子菜单的动态重建
// ============================================================

/**
 * @brief 根据路径列表动态重建"最近打开"子菜单。
 *
 * 每次调用时先清空旧菜单项，然后：
 * 1. 为每条路径创建编号动作，点击时 emit recentProjectSelected；
 * 2. 若列表为空则添加灰色"（无）"占位项；
 * 3. 末尾添加"清空最近打开"分隔项，可用时（列表非空）才启用。
 *
 * @param paths 最近打开的项目路径列表。
 */
void MainMenu::setRecentProjects(const QStringList &paths)
{
    if (!m_recentMenu) return;
    m_recentMenu->clear(); // 清除旧的菜单项

    int idx = 0;
    for (const QString &p : paths) {
        ++idx;
        // 转换为平台原生路径分隔符，并取绝对路径，使显示更友好
        QString abs = QDir::toNativeSeparators(QFileInfo(p).absoluteFilePath());
        // 菜单项格式：序号. 路径（例如 "1. /home/user/project.plascan"）
        QAction *a = m_recentMenu->addAction(QStringLiteral("%1. %2").arg(idx).arg(abs));
        a->setToolTip(abs); // 鼠标悬停时显示完整路径
        // 点击时以 lambda 捕获路径，emit 信号通知主窗口打开项目
        connect(a, &QAction::triggered, this, [this, abs]() { emit recentProjectSelected(abs); });
    }

    if (paths.isEmpty()) {
        // 无记录时显示灰色占位项，防止子菜单空白
        auto *na = m_recentMenu->addAction(tr("（无）"));
        na->setEnabled(false);
    }

    m_recentMenu->addSeparator();
    auto *clearAct = m_recentMenu->addAction(tr("清空最近打开"));
    // 只有列表非空时才允许清空操作，防止用户对空列表执行无意义操作
    clearAct->setEnabled(!paths.isEmpty());
    connect(clearAct, &QAction::triggered, this, [this]() { emit clearRecentRequested(); });
}

// ============================================================
//  访问器方法实现（均为简单的指针返回，无逻辑）
// ============================================================

QAction *MainMenu::toggleLogAction() const   { return m_toggleLogAct; }
QAction *MainMenu::featureInfoAction() const { return m_featureInfoAct; }
QToolBar *MainMenu::toolBar() const          { return m_toolBar; }

QAction *MainMenu::newAction() const  { return m_newAct; }
QAction *MainMenu::openAction() const { return m_openAct; }
QAction *MainMenu::saveAction() const { return m_saveAct; }
QAction *MainMenu::minimizeAction() const { return m_minimizeAct; }
QAction *MainMenu::exitAction() const { return m_exitAct; }

QAction *MainMenu::zoomInAction() const    { return m_zoomInAct; }
QAction *MainMenu::zoomOutAction() const   { return m_zoomOutAct; }
QAction *MainMenu::resetViewAction() const { return m_resetViewAct; }
QAction *MainMenu::toggleGizmoAction() const { return m_toggleGizmoAct; }

QAction *MainMenu::addPhotoAction() const       { return m_addPhotoAct; }
QAction *MainMenu::addFolderAction() const      { return m_addFolderAct; }
QAction *MainMenu::detectFeaturesAction() const { return m_detectFeaturesAct; }
QAction *MainMenu::featureVisualizationAction() const { return m_featureVisualizationAct; }
QAction *MainMenu::matchFeaturesAction() const  { return m_matchFeaturesAct; }
QAction *MainMenu::viewMatchesAction() const    { return m_viewMatchesAct; }
QAction *MainMenu::threeDReconstructionAction() const { return m_threeDReconstructionAct; }
QAction *MainMenu::overlapAnalysisAction() const { return m_overlapAnalysisAct; }
QAction *MainMenu::intersectionCheckAction() const { return m_intersectionCheckAct; }
QAction *MainMenu::intersectionViewResultsAction() const { return m_intersectionViewResultsAct; }
QAction *MainMenu::createDEMAction() const      { return m_createDEMAct; }
QAction *MainMenu::generateOrthoAction() const  { return m_generateOrthoAct; }

QAction *MainMenu::viewWorkflowReportAction() const         { return m_viewWorkflowReportAct; }
QAction *MainMenu::manualPointCloudPruneAction() const      { return m_manualPointCloudPruneAct; }

QAction *MainMenu::buildObsNetworkAction() const     { return m_buildObsNetworkAct; }
QAction *MainMenu::initCameraPoseAction() const      { return m_initCameraPoseAct; }
QAction *MainMenu::triangulateAction() const         { return m_triangulateAct; }
QAction *MainMenu::reconBundleAdjustAction() const   { return m_reconBundleAdjustAct; }
QAction *MainMenu::sparseCloudPostProcessAction() const { return m_sparseCloudPostProcessAct; }
QAction *MainMenu::depthMapEstimateAction() const    { return m_depthMapEstimateAct; }
QAction *MainMenu::fuseDepthMapsAction() const       { return m_fuseDepthMapsAct; }
QAction *MainMenu::refineDenseCloudAction() const    { return m_refineDenseCloudAct; }
QAction *MainMenu::meshReconstructAction() const     { return m_meshReconstructAct; }
QAction *MainMenu::textureMappingAction() const      { return m_textureMappingAct; }
QAction *MainMenu::exportModelAction() const         { return m_exportModelAct; }
QAction *MainMenu::exportMatchedPairsAction() const  { return m_exportMatchedPairsAct; }

QAction *MainMenu::denseMatchAction() const { return m_denseMatchAct; }
