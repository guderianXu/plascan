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

namespace {

template <typename T>
T *findNamedChild(QObject *root, const char *name)
{
    return root ? root->findChild<T *>(QString::fromLatin1(name)) : nullptr;
}

} // namespace

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
    : QObject(mainWindow), _mainWindow(mainWindow)
{
    if (!_mainWindow) return;

    if (findNamedChild<QAction>(_mainWindow, "actionNewProject"))
    {
        _fileMenu = findNamedChild<QMenu>(_mainWindow, "menuProject");
        _recentMenu = findNamedChild<QMenu>(_mainWindow, "menuRecentProjects");
        auto *viewMenu = findNamedChild<QMenu>(_mainWindow, "menuView");
        auto *windowMenu = findNamedChild<QMenu>(_mainWindow, "menuWindow");
        auto *toolsMenu = findNamedChild<QMenu>(_mainWindow, "menuTools");

        _newAct = findNamedChild<QAction>(_mainWindow, "actionNewProject");
        _openAct = findNamedChild<QAction>(_mainWindow, "actionOpenProject");
        _saveAct = findNamedChild<QAction>(_mainWindow, "actionSaveProject");
        _exportMatchedPairsAct = findNamedChild<QAction>(_mainWindow, "actionExportMatchedPairs");
        _minimizeAct = findNamedChild<QAction>(_mainWindow, "actionMinimize");
        _exitAct = findNamedChild<QAction>(_mainWindow, "actionExit");

        _zoomInAct = findNamedChild<QAction>(_mainWindow, "actionZoomIn");
        _zoomOutAct = findNamedChild<QAction>(_mainWindow, "actionZoomOut");
        _resetViewAct = findNamedChild<QAction>(_mainWindow, "actionResetView");
        _toggleGizmoAct = findNamedChild<QAction>(_mainWindow, "actionToggleGizmo");
        _toggleCamerasAct = findNamedChild<QAction>(_mainWindow, "actionToggleCameras");
        _featureVisualizationAct = findNamedChild<QAction>(_mainWindow, "actionFeatureVisualization");
        _toggleLogAct = findNamedChild<QAction>(_mainWindow, "actionToggleLog");
        _featureInfoAct = findNamedChild<QAction>(_mainWindow, "actionFeatureInfo");

        _addPhotoAct = findNamedChild<QAction>(_mainWindow, "actionAddPhoto");
        _addFolderAct = findNamedChild<QAction>(_mainWindow, "actionAddFolder");
        _threeDReconstructionAct = findNamedChild<QAction>(_mainWindow, "actionThreeDReconstruction");
        _createDEMAct = findNamedChild<QAction>(_mainWindow, "actionCreateDEM");
        _generateOrthoAct = findNamedChild<QAction>(_mainWindow, "actionGenerateOrtho");

        _detectFeaturesAct = findNamedChild<QAction>(_mainWindow, "actionDetectFeatures");
        _vocabularyOverlapAct = findNamedChild<QAction>(_mainWindow, "actionVocabularyOverlap");
        _matchFeaturesAct = findNamedChild<QAction>(_mainWindow, "actionMatchFeatures");
        _aerialTriangulationAct = findNamedChild<QAction>(_mainWindow, "actionAerialTriangulation");
        _buildObsNetworkAct = findNamedChild<QAction>(_mainWindow, "actionBuildObsNetwork");
        _initCameraPoseAct = findNamedChild<QAction>(_mainWindow, "actionInitCameraPose");
        _triangulateAct = findNamedChild<QAction>(_mainWindow, "actionTriangulate");
        _reconBundleAdjustAct = findNamedChild<QAction>(_mainWindow, "actionReconBundleAdjust");
        _sparseCloudPostProcessAct = findNamedChild<QAction>(_mainWindow, "actionSparseCloudPostProcess");
        _denseMatchAct = findNamedChild<QAction>(_mainWindow, "actionDenseMatch");
        _depthMapEstimateAct = findNamedChild<QAction>(_mainWindow, "actionDepthMapEstimate");
        _fuseDepthMapsAct = findNamedChild<QAction>(_mainWindow, "actionFuseDepthMaps");
        _refineDenseCloudAct = findNamedChild<QAction>(_mainWindow, "actionRefineDenseCloud");
        _meshReconstructAct = findNamedChild<QAction>(_mainWindow, "actionMeshReconstruct");
        _textureMappingAct = findNamedChild<QAction>(_mainWindow, "actionTextureMapping");
        _exportModelAct = findNamedChild<QAction>(_mainWindow, "actionExportModel");

        _overlapAnalysisAct = findNamedChild<QAction>(_mainWindow, "actionOverlapAnalysis");
        _intersectionCheckAct = findNamedChild<QAction>(_mainWindow, "actionIntersectionCheck");
        _intersectionViewResultsAct = findNamedChild<QAction>(_mainWindow, "actionIntersectionViewResults");
        _manualPointCloudPruneAct = findNamedChild<QAction>(_mainWindow, "actionManualPointCloudPrune");
        _viewMatchesAct = findNamedChild<QAction>(_mainWindow, "actionViewMatches");
        _viewWorkflowReportAct = findNamedChild<QAction>(_mainWindow, "actionViewWorkflowReport");
        _cameraConvertAct = findNamedChild<QAction>(_mainWindow, "actionCameraConvert");
        _surveyControlAct = findNamedChild<QAction>(_mainWindow, "actionSurveyControl");
        _importReferenceDatasetAct = findNamedChild<QAction>(_mainWindow, "actionImportReferenceDataset");
        _referenceQualityCheckAct = findNamedChild<QAction>(_mainWindow, "actionReferenceQualityCheck");
        _referenceTerrainBundleAdjustAct = findNamedChild<QAction>(_mainWindow, "actionReferenceTerrainBundleAdjust");
        if (!_toggleCamerasAct)
        {
            QObject *actionParent = viewMenu
                ? static_cast<QObject *>(viewMenu)
                : static_cast<QObject *>(_mainWindow);
            _toggleCamerasAct = new QAction(tr("显示相机"), actionParent);
            _toggleCamerasAct->setObjectName(QStringLiteral("actionToggleCameras"));
            _toggleCamerasAct->setCheckable(true);
            _toggleCamerasAct->setChecked(true);
            _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
            if (viewMenu)
            {
                QAction *before = _featureVisualizationAct ? _featureVisualizationAct : nullptr;
                if (before)
                {
                    viewMenu->insertAction(before, _toggleCamerasAct);
                }
                else
                {
                    viewMenu->addAction(_toggleCamerasAct);
                }
            }
        }
        else
        {
            _toggleCamerasAct->setCheckable(true);
            _toggleCamerasAct->setChecked(true);
            _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
        }
        if (!_cameraConvertAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _cameraConvertAct = new QAction(tr("相机格式转换..."), actionParent);
            _cameraConvertAct->setObjectName(QStringLiteral("actionCameraConvert"));
            _cameraConvertAct->setToolTip(tr("将外部相机文件转换为 PlaScan tsai 和 image_camera.lis"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _cameraConvertAct);
                    toolsMenu->insertSeparator(_viewWorkflowReportAct);
                }
                else
                {
                    toolsMenu->addAction(_cameraConvertAct);
                }
            }
        }
        if (!_importReferenceDatasetAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _importReferenceDatasetAct = new QAction(tr("导入参考 DEM/LiDAR..."), actionParent);
            _importReferenceDatasetAct->setObjectName(QStringLiteral("actionImportReferenceDataset"));
            _importReferenceDatasetAct->setToolTip(tr("以外部引用方式导入 DEM、LAS/LAZ/COPC 或点云文件，用于精度检查和后续软约束"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _importReferenceDatasetAct);
                }
                else
                {
                    toolsMenu->addAction(_importReferenceDatasetAct);
                }
            }
        }
        if (!_surveyControlAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _surveyControlAct = new QAction(tr("测绘控制..."), actionParent);
            _surveyControlAct->setObjectName(QStringLiteral("actionSurveyControl"));
            _surveyControlAct->setToolTip(tr("导入并查看控制点、检查点和比例尺约束"));
            if (toolsMenu)
            {
                if (_importReferenceDatasetAct)
                {
                    toolsMenu->insertAction(_importReferenceDatasetAct, _surveyControlAct);
                }
                else if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _surveyControlAct);
                }
                else
                {
                    toolsMenu->addAction(_surveyControlAct);
                }
            }
        }
        if (!_referenceQualityCheckAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _referenceQualityCheckAct = new QAction(tr("点云/DEM 精度检查..."), actionParent);
            _referenceQualityCheckAct->setObjectName(QStringLiteral("actionReferenceQualityCheck"));
            _referenceQualityCheckAct->setToolTip(tr("根据已导入参考 DEM/LiDAR 和当前 DEM/点云成果生成精度检查报告"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _referenceQualityCheckAct);
                }
                else
                {
                    toolsMenu->addAction(_referenceQualityCheckAct);
                }
            }
        }
        if (!_referenceTerrainBundleAdjustAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _referenceTerrainBundleAdjustAct = new QAction(tr("参考地形约束重新平差..."), actionParent);
            _referenceTerrainBundleAdjustAct->setObjectName(QStringLiteral("actionReferenceTerrainBundleAdjust"));
            _referenceTerrainBundleAdjustAct->setToolTip(tr("检查参考 DEM/LiDAR 是否可作为 BA 软约束，并生成前置检查报告"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _referenceTerrainBundleAdjustAct);
                }
                else
                {
                    toolsMenu->addAction(_referenceTerrainBundleAdjustAct);
                }
            }
        }

        if (_exitAct)
        {
            connect(_exitAct, &QAction::triggered, qApp, &QCoreApplication::quit);
        }
        if (auto *aboutAct = findNamedChild<QAction>(_mainWindow, "actionAbout"))
        {
            connect(aboutAct, &QAction::triggered, _mainWindow, [mw = _mainWindow]() {
                mw->statusBar()->showMessage(tr("PlaScan: 行星表面摄影测量处理系统"), 3000);
            });
        }

        if (windowMenu)
        {
            QList<QAction*> windowActs = { _toggleLogAct, _featureInfoAct };
            auto *panelAct = new QWidgetAction(windowMenu);
            auto *wp = new WindowPanel(windowMenu);
            panelAct->setDefaultWidget(wp);
            windowMenu->addAction(panelAct);
            wp->setActions(windowActs);
        }

        _toolBar = findNamedChild<QToolBar>(_mainWindow, "mainToolBar");
        if (_toolBar)
        {
            _toolBar->setMovable(false);
            _toolBar->setFloatable(false);
            _toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
            _toolBar->setIconSize(QSize(18, 18));
        }
        if (_saveAct)
        {
            _saveAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DialogSaveButton));
        }
        if (_addPhotoAct)
        {
            _addPhotoAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_FileIcon));
        }
        if (_addFolderAct)
        {
            _addFolderAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DirOpenIcon));
        }
        if (_threeDReconstructionAct)
        {
            _threeDReconstructionAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_MediaPlay));
        }
        if (_createDEMAct)
        {
            _createDEMAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DriveHDIcon));
        }
        if (_generateOrthoAct)
        {
            _generateOrthoAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DesktopIcon));
        }
        if (_manualPointCloudPruneAct)
        {
            _manualPointCloudPruneAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_CommandLink));
        }

        return;
    }

    // ---- 项目菜单 ----
    // 创建顶级"项目"菜单并依次添加固定动作
    _fileMenu = _mainWindow->menuBar()->addMenu(tr("项目"));
    _newAct   = _fileMenu->addAction(tr("新建项目"));
    _openAct  = _fileMenu->addAction(tr("打开项目"));
    _saveAct  = _fileMenu->addAction(tr("保存项目"));

    // "最近打开"子菜单插入到"保存项目"之前，保持菜单项顺序符合直觉
    _recentMenu = new QMenu(tr("最近打开"), _fileMenu);
    _fileMenu->insertMenu(_saveAct, _recentMenu);

    auto *exportMenu = _fileMenu->addMenu(tr("导出"));
    _exportMatchedPairsAct = exportMenu->addAction(tr("导出匹配对(.lis)"));

    _fileMenu->addSeparator();
    _minimizeAct = _fileMenu->addAction(tr("最小化"));
    // 退出动作直接连接到 QCoreApplication::quit，无需额外连接
    _exitAct = _fileMenu->addAction(tr("退出"), qApp, &QCoreApplication::quit);

    // ---- 视图菜单 ----
    auto *viewMenu = _mainWindow->menuBar()->addMenu(tr("视图"));
    _zoomInAct    = viewMenu->addAction(tr("放大"));
    _zoomOutAct   = viewMenu->addAction(tr("缩小"));
    _resetViewAct = viewMenu->addAction(tr("重置视图"));
    viewMenu->addSeparator();
    // 操控球显示/隐藏切换
    _toggleGizmoAct = new QAction(tr("显示操控球"), viewMenu);
    _toggleGizmoAct->setCheckable(true);
    _toggleGizmoAct->setChecked(true);  // 默认显示
    _toggleGizmoAct->setToolTip(tr("显示或隐藏 3D 视图中的旋转操控球"));
    viewMenu->addAction(_toggleGizmoAct);
    _toggleCamerasAct = new QAction(tr("显示相机"), viewMenu);
    _toggleCamerasAct->setObjectName(QStringLiteral("actionToggleCameras"));
    _toggleCamerasAct->setCheckable(true);
    _toggleCamerasAct->setChecked(true);
    _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
    viewMenu->addAction(_toggleCamerasAct);
    viewMenu->addSeparator();
    // 特征点可视化设置对话框入口
    _featureVisualizationAct = viewMenu->addAction(tr("特征点 可视化设置..."));
    viewMenu->addSeparator();

    // 窗口面板子菜单：使用 QWidgetAction + WindowPanel 实现带复选框的面板开关列表
    auto *windowMenu = viewMenu->addMenu(tr("窗口"));
    _toggleLogAct = new QAction(tr("日志"), windowMenu);
    _toggleLogAct->setCheckable(true);  // 可切换：勾选时面板可见
    _toggleLogAct->setChecked(true);    // 默认显示日志面板
    _featureInfoAct = new QAction(tr("兴趣点信息"), windowMenu);
    _featureInfoAct->setCheckable(true); // 可切换：勾选时面板可见

    // 将动作列表传给 WindowPanel 组件，以列表形式展示在子菜单中
    QList<QAction*> windowActs = { _toggleLogAct, _featureInfoAct };
    auto *panelAct = new QWidgetAction(windowMenu);
    auto *wp = new WindowPanel(windowMenu);
    panelAct->setDefaultWidget(wp);
    windowMenu->addAction(panelAct);
    wp->setActions(windowActs);

    // ---- 工作流程菜单 ----
    // 提供高层一键式处理流程入口，适合不需要分步调试的普通用户
    auto *workflowMenu = _mainWindow->menuBar()->addMenu(tr("工作流程"));
    _addPhotoAct       = workflowMenu->addAction(tr("添加 照片"));
    _addFolderAct      = workflowMenu->addAction(tr("添加 文件夹"));
    workflowMenu->addSeparator();
    _threeDReconstructionAct = workflowMenu->addAction(tr("三维重建"));     // 一键完整建模流程
    _createDEMAct      = workflowMenu->addAction(tr("创建 DEM"));          // DEM 完整流程
    _generateOrthoAct  = workflowMenu->addAction(tr("生成 正射影像"));     // 正射影像完整流程

    // ---- 重建菜单 ----
    // 三级菜单结构：稀疏重建 / 密集重建 / 模型生成
    auto *reconMenu = _mainWindow->menuBar()->addMenu(tr("重建"));

    // ── 稀疏重建 ──
    auto *sparseReconMenu = reconMenu->addMenu(tr("稀疏重建"));
    _detectFeaturesAct = sparseReconMenu->addAction(tr("特征点提取"));
    _vocabularyOverlapAct = sparseReconMenu->addAction(tr("重叠对规划..."));
    _matchFeaturesAct  = sparseReconMenu->addAction(tr("连接点匹配"));
    _aerialTriangulationAct = sparseReconMenu->addAction(tr("空中三角测量..."));
    _sparseCloudPostProcessAct = sparseReconMenu->addAction(tr("稀疏点云后处理..."));

    sparseReconMenu->addSeparator();
    auto *advancedSparseMenu = sparseReconMenu->addMenu(tr("高级工具"));
    _viewMatchesAct = advancedSparseMenu->addAction(tr("查看匹配"));
    _buildObsNetworkAct = advancedSparseMenu->addAction(tr("构建观测网络..."));
    _initCameraPoseAct = advancedSparseMenu->addAction(tr("初始化相机位姿..."));
    _triangulateAct = advancedSparseMenu->addAction(tr("生成两视预览云..."));
    _reconBundleAdjustAct = advancedSparseMenu->addAction(tr("单独光束法平差..."));

    // ── 密集重建 ──
    auto *denseReconMenu = reconMenu->addMenu(tr("密集重建"));
    _denseMatchAct       = denseReconMenu->addAction(tr("密集匹配..."));
    _depthMapEstimateAct = denseReconMenu->addAction(tr("深度图估计..."));
    _fuseDepthMapsAct    = denseReconMenu->addAction(tr("深度图融合生成密集点云..."));
    _refineDenseCloudAct = denseReconMenu->addAction(tr("密集点云后处理..."));

    // ── 模型生成 ──
    auto *modelGenMenu = reconMenu->addMenu(tr("模型生成"));
    _meshReconstructAct = modelGenMenu->addAction(tr("网格重建..."));
    _textureMappingAct  = modelGenMenu->addAction(tr("纹理映射..."));
    _exportModelAct     = modelGenMenu->addAction(tr("模型导出..."));

    // ---- 工具菜单 ----
    // 提供细粒度的单步工具入口，供高级用户和调试场景使用
    auto *toolsMenu = _mainWindow->menuBar()->addMenu(tr("工具"));

    // 质量检查工具
    _overlapAnalysisAct = toolsMenu->addAction(tr("重叠度获取"));
    QMenu *intersectionMenu = toolsMenu->addMenu(tr("前方交汇精度检验"));
    _intersectionCheckAct = intersectionMenu->addAction(tr("检测交汇"));
    _intersectionViewResultsAct = intersectionMenu->addAction(tr("查看结果"));
    toolsMenu->addSeparator();
    _manualPointCloudPruneAct = toolsMenu->addAction(tr("手动点云剔除"));

    // 相机格式转换：外部 benchmark/摄影测量相机 -> tsai + image_camera.lis
    toolsMenu->addSeparator();
    _cameraConvertAct = toolsMenu->addAction(tr("相机格式转换..."));

    // 参考数据：外部 DEM/LiDAR 只登记引用，用于精度检查和后续 BA 软约束
    _surveyControlAct = toolsMenu->addAction(tr("测绘控制..."));
    _surveyControlAct->setObjectName(QStringLiteral("actionSurveyControl"));
    _surveyControlAct->setToolTip(tr("导入并查看控制点、检查点和比例尺约束"));
    _importReferenceDatasetAct = toolsMenu->addAction(tr("导入参考 DEM/LiDAR..."));
    _importReferenceDatasetAct->setObjectName(QStringLiteral("actionImportReferenceDataset"));
    _importReferenceDatasetAct->setToolTip(tr("以外部引用方式导入 DEM、LAS/LAZ/COPC 或点云文件，用于精度检查和后续软约束"));
    _referenceQualityCheckAct = toolsMenu->addAction(tr("点云/DEM 精度检查..."));
    _referenceQualityCheckAct->setObjectName(QStringLiteral("actionReferenceQualityCheck"));
    _referenceQualityCheckAct->setToolTip(tr("根据已导入参考 DEM/LiDAR 和当前 DEM/点云成果生成精度检查报告"));
    _referenceTerrainBundleAdjustAct = toolsMenu->addAction(tr("参考地形约束重新平差..."));
    _referenceTerrainBundleAdjustAct->setObjectName(QStringLiteral("actionReferenceTerrainBundleAdjust"));
    _referenceTerrainBundleAdjustAct->setToolTip(tr("检查参考 DEM/LiDAR 是否可作为 BA 软约束，并生成前置检查报告"));

    // 报告：查看各工作流程的历史统计报告
    toolsMenu->addSeparator();
    _viewWorkflowReportAct = toolsMenu->addAction(tr("查看工作流程报告..."));

    // ---- 帮助菜单 ----
    auto *helpMenu = _mainWindow->menuBar()->addMenu(tr("帮助"));
    helpMenu->addAction(tr("关于"), _mainWindow, [mw = _mainWindow]() {
        // 在状态栏短暂显示应用说明，3 秒后自动清除
        mw->statusBar()->showMessage(tr("PlaScan: 行星表面摄影测量处理系统"), 3000);
    });

    // ---- 主工具栏 ----
    _toolBar = _mainWindow->addToolBar(tr("工具"));
    if (_toolBar)
    {
        _toolBar->setMovable(false);
        _toolBar->setFloatable(false);
        _toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        _toolBar->setIconSize(QSize(18, 18));

        if (_saveAct)
        {
            _saveAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DialogSaveButton));
            _toolBar->addAction(_saveAct);
        }
        _toolBar->addSeparator();
        if (_addPhotoAct)
        {
            _addPhotoAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_FileIcon));
            _toolBar->addAction(_addPhotoAct);
        }
        if (_addFolderAct)
        {
            _addFolderAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DirOpenIcon));
            _toolBar->addAction(_addFolderAct);
        }
        _toolBar->addSeparator();
        if (_threeDReconstructionAct)
        {
            _threeDReconstructionAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_MediaPlay));
            _toolBar->addAction(_threeDReconstructionAct);
        }
        if (_createDEMAct)
        {
            _createDEMAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DriveHDIcon));
            _toolBar->addAction(_createDEMAct);
        }
        if (_generateOrthoAct)
        {
            _generateOrthoAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DesktopIcon));
            _toolBar->addAction(_generateOrthoAct);
        }
        _toolBar->addSeparator();
        if (_manualPointCloudPruneAct)
        {
            _manualPointCloudPruneAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_CommandLink));
            _toolBar->addAction(_manualPointCloudPruneAct);
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
    if (!_recentMenu) return;
    _recentMenu->clear(); // 清除旧的菜单项

    int idx = 0;
    for (const QString &p : paths) {
        ++idx;
        // 转换为平台原生路径分隔符，并取绝对路径，使显示更友好
        QString abs = QDir::toNativeSeparators(QFileInfo(p).absoluteFilePath());
        // 菜单项格式：序号. 路径（例如 "1. /home/user/project.plascan"）
        QAction *a = _recentMenu->addAction(QStringLiteral("%1. %2").arg(idx).arg(abs));
        a->setToolTip(abs); // 鼠标悬停时显示完整路径
        // 点击时以 lambda 捕获路径，emit 信号通知主窗口打开项目
        connect(a, &QAction::triggered, this, [this, abs]() { emit recentProjectSelected(abs); });
    }

    if (paths.isEmpty()) {
        // 无记录时显示灰色占位项，防止子菜单空白
        auto *na = _recentMenu->addAction(tr("（无）"));
        na->setEnabled(false);
    }

    _recentMenu->addSeparator();
    auto *clearAct = _recentMenu->addAction(tr("清空最近打开"));
    // 只有列表非空时才允许清空操作，防止用户对空列表执行无意义操作
    clearAct->setEnabled(!paths.isEmpty());
    connect(clearAct, &QAction::triggered, this, [this]() { emit clearRecentRequested(); });
}

// ============================================================
//  访问器方法实现（均为简单的指针返回，无逻辑）
// ============================================================

QAction *MainMenu::toggleLogAction() const   { return _toggleLogAct; }
QAction *MainMenu::featureInfoAction() const { return _featureInfoAct; }
QToolBar *MainMenu::toolBar() const          { return _toolBar; }

QAction *MainMenu::newAction() const  { return _newAct; }
QAction *MainMenu::openAction() const { return _openAct; }
QAction *MainMenu::saveAction() const { return _saveAct; }
QAction *MainMenu::minimizeAction() const { return _minimizeAct; }
QAction *MainMenu::exitAction() const { return _exitAct; }

QAction *MainMenu::zoomInAction() const    { return _zoomInAct; }
QAction *MainMenu::zoomOutAction() const   { return _zoomOutAct; }
QAction *MainMenu::resetViewAction() const { return _resetViewAct; }
QAction *MainMenu::toggleGizmoAction() const { return _toggleGizmoAct; }
QAction *MainMenu::toggleCamerasAction() const { return _toggleCamerasAct; }

QAction *MainMenu::addPhotoAction() const       { return _addPhotoAct; }
QAction *MainMenu::addFolderAction() const      { return _addFolderAct; }
QAction *MainMenu::detectFeaturesAction() const { return _detectFeaturesAct; }
QAction *MainMenu::vocabularyOverlapAction() const { return _vocabularyOverlapAct; }
QAction *MainMenu::featureVisualizationAction() const { return _featureVisualizationAct; }
QAction *MainMenu::matchFeaturesAction() const  { return _matchFeaturesAct; }
QAction *MainMenu::viewMatchesAction() const    { return _viewMatchesAct; }
QAction *MainMenu::threeDReconstructionAction() const { return _threeDReconstructionAct; }
QAction *MainMenu::overlapAnalysisAction() const { return _overlapAnalysisAct; }
QAction *MainMenu::intersectionCheckAction() const { return _intersectionCheckAct; }
QAction *MainMenu::intersectionViewResultsAction() const { return _intersectionViewResultsAct; }
QAction *MainMenu::createDEMAction() const      { return _createDEMAct; }
QAction *MainMenu::generateOrthoAction() const  { return _generateOrthoAct; }

QAction *MainMenu::viewWorkflowReportAction() const         { return _viewWorkflowReportAct; }
QAction *MainMenu::manualPointCloudPruneAction() const      { return _manualPointCloudPruneAct; }
QAction *MainMenu::cameraConvertAction() const              { return _cameraConvertAct; }
QAction *MainMenu::surveyControlAction() const              { return _surveyControlAct; }
QAction *MainMenu::importReferenceDatasetAction() const     { return _importReferenceDatasetAct; }
QAction *MainMenu::referenceQualityCheckAction() const      { return _referenceQualityCheckAct; }
QAction *MainMenu::referenceTerrainBundleAdjustAction() const { return _referenceTerrainBundleAdjustAct; }

QAction *MainMenu::buildObsNetworkAction() const     { return _buildObsNetworkAct; }
QAction *MainMenu::initCameraPoseAction() const      { return _initCameraPoseAct; }
QAction *MainMenu::aerialTriangulationAction() const { return _aerialTriangulationAct; }
QAction *MainMenu::triangulateAction() const         { return _triangulateAct; }
QAction *MainMenu::reconBundleAdjustAction() const   { return _reconBundleAdjustAct; }
QAction *MainMenu::sparseCloudPostProcessAction() const { return _sparseCloudPostProcessAct; }
QAction *MainMenu::depthMapEstimateAction() const    { return _depthMapEstimateAct; }
QAction *MainMenu::fuseDepthMapsAction() const       { return _fuseDepthMapsAct; }
QAction *MainMenu::refineDenseCloudAction() const    { return _refineDenseCloudAct; }
QAction *MainMenu::meshReconstructAction() const     { return _meshReconstructAct; }
QAction *MainMenu::textureMappingAction() const      { return _textureMappingAct; }
QAction *MainMenu::exportModelAction() const         { return _exportModelAct; }
QAction *MainMenu::exportMatchedPairsAction() const  { return _exportMatchedPairsAct; }

QAction *MainMenu::denseMatchAction() const { return _denseMatchAct; }
