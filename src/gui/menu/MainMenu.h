#pragma once

/**
 * @file MainMenu.h
 * @brief 主菜单栏与工具栏管理器的声明文件。
 *
 * MainMenu 封装了应用程序所有菜单项（QAction）的访问与动态内容维护，
 * 遵循"只负责构建 UI 结构，不承载业务逻辑"的原则：
 *   - 优先绑定 MainWindow.ui 中定义的菜单和动作；
 *   - 在裸 QMainWindow 测试场景下回退为代码创建菜单和动作；
 *   - 通过公开的访问器方法（xxxAction()）将各 QAction 暴露给主窗口；
 *   - 主窗口负责将这些 QAction 的 triggered 信号连接到实际的业务槽函数。
 *
 * 菜单结构：
 *   - 项目：新建、打开、最近打开、保存、退出
 *   - 视图：放大、缩小、重置视图、特征点可视化、窗口面板开关
 *   - 工作流程：空三、生成模型、DEM、正射影像等高层操作
 *   - 工具：连接点、重叠度、前方交汇、工作流程报告
 *   - 帮助：更新 Python 环境、关于
 */

#include <QObject>
#include <QList>

class QMainWindow;
class QAction;
class QToolBar;
class QMenu;
class QActionGroup;

/**
 * @class MainMenu
 * @brief 绑定并管理应用程序菜单栏与工具栏的所有动作。
 *
 * 正常运行时 QAction 来自 MainWindow.ui，外部通过访问器方法获取指针并连接信号。
 * 最近打开项目和窗口面板等动态菜单内容仍由本类维护。
 */
class MainMenu : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数：绑定或创建指定主窗口上的菜单系统与工具栏。
     *
     * 优先绑定 MainWindow.ui 中的项目、视图、工作流程、工具、帮助菜单和主工具栏；
     * 若主窗口未通过 .ui 初始化，则回退到代码创建，保持单元测试可独立运行。
     *
     * @param mainWindow 要在其上添加菜单/工具栏的主窗口，不可为 nullptr。
     */
    explicit MainMenu(QMainWindow *mainWindow);

    /** @brief 析构函数（所有子对象由 Qt 对象树自动释放）。 */
    ~MainMenu() override;

    // ==== 面板开关动作 ====

    /** @brief 返回"日志面板"显示/隐藏切换动作（可检查状态的 QAction）。 */
    QAction *toggleLogAction() const;

    /** @brief 返回"左侧工作区"显示/隐藏切换动作（可检查状态的 QAction）。 */
    QAction *toggleWorkspaceAction() const;

    /** @brief 返回"属性面板"显示/隐藏切换动作（可检查状态的 QAction）。 */
    QAction *togglePropertiesAction() const;

    /** @brief 返回"照片面板"显示/隐藏切换动作（可检查状态的 QAction）。 */
    QAction *togglePhotosAction() const;

    /** @brief 返回主工具栏显示/隐藏切换动作。 */
    QAction *toggleMainToolbarAction() const;

    /** @brief 返回“恢复默认窗口布局”动作。 */
    QAction *restoreDefaultWindowLayoutAction() const;

    /** @brief 使用面板注册表中的动作重建“视图 > 窗口”菜单。 */
    void setManagedWindowActions(const QList<QAction *> &dockActions,
                                 const QList<QAction *> &toolBarActions);

    // ==== 工具栏 ====

    /** @brief 返回主工具栏指针，外部可向其添加额外的操作按钮。 */
    QToolBar *toolBar() const;

    /** @brief 根据中央工作区类型切换三维与影像专属工具组。 */
    void setContextualToolbarVisibility(bool showModelTools, bool showImageTools);

    /** @brief 更新当前影像是否已完成解码并可执行影像操作。 */
    void setImageDisplayReady(bool ready);

    /** @brief 更新当前影像是否存在可显示的深度数据。 */
    void setDepthOverlayAvailable(bool available);

    /** @brief 分别更新最终层与 Level 1/2/3 深度栅格是否可显示。 */
    void setDepthOverlayLevelsAvailable(bool finalAvailable,
                                        bool level1Available,
                                        bool level2Available,
                                        bool level3Available,
                                        const QString &finalReason = {},
                                        const QString &level1Reason = {},
                                        const QString &level2Reason = {},
                                        const QString &level3Reason = {});

    // ==== 文件菜单动作 ====

    /** @brief 返回"新建项目"动作。 */
    QAction *newAction() const;

    /** @brief 返回"打开项目"动作。 */
    QAction *openAction() const;

    /** @brief 返回"保存项目"动作。 */
    QAction *saveAction() const;

    /** @brief 返回"最小化"动作。 */
    QAction *minimizeAction() const;

    /** @brief 返回"退出"动作（已连接到 QCoreApplication::quit）。 */
    QAction *exitAction() const;

    /** @brief 返回“帮助 / 更新 Python 环境”动作。 */
    QAction *updatePythonRuntimeAction() const;

    // ==== 视图菜单动作 ====

    /** @brief 返回"放大"动作。 */
    QAction *zoomInAction() const;

    /** @brief 返回"缩小"动作。 */
    QAction *zoomOutAction() const;

    /** @brief 返回"重置视图"动作。 */
    QAction *resetViewAction() const;

    /** @brief 返回全屏显示切换动作。 */
    QAction *toggleFullScreenAction() const;

    /** @brief 返回当前影像向左旋转 90 度的动作。 */
    QAction *rotateImageLeftAction() const;

    /** @brief 返回当前影像向右旋转 90 度的动作。 */
    QAction *rotateImageRightAction() const;
    QAction *showFeaturePointsAction() const;
    QAction *showFeatureResidualsAction() const;
    QAction *showMaskOverlayAction() const;
    QAction *showDepthOverlayAction() const;
    QAction *depthOverlayAllLevelsAction() const;
    QAction *depthOverlayLevel1Action() const;
    QAction *depthOverlayLevel2Action() const;
    QAction *depthOverlayLevel3Action() const;
    QAction *showDepthIntensityAction() const;

    /** @brief 返回"显示轨迹球"切换动作（可检查状态）。 */
    QAction *toggleGizmoAction() const;

    /** @brief 返回"显示相机"切换动作（可检查状态）。 */
    QAction *toggleCamerasAction() const;

    /** @brief 返回"显示从属相机"动作（当前未启用，保留 Metashape 风格菜单占位）。 */
    QAction *toggleDependentCamerasAction() const;

    /** @brief 返回"显示缩略图"切换动作，用于控制相机影像平面显示模式。 */
    QAction *toggleCameraThumbnailsAction() const;

    /** @brief 返回"显示本地轴"切换动作，用于工具栏相机菜单，状态与轨迹球显示同步。 */
    QAction *toggleLocalAxesAction() const;

    /** @brief 返回"显示图像"切换动作，用于控制当前相机图像平面。 */
    QAction *toggleCameraImagesAction() const;

    QAction *showCameraImagesInForegroundAction() const;
    QAction *showCameraImagesInBackgroundAction() const;
    QAction *lockCameraImageAction() const;
    QAction *tiePointColorModeAction() const;
    QAction *tiePointElevationModeAction() const;
    QAction *tiePointImageCountModeAction() const;
    QAction *modelTextureModeAction() const;
    QAction *modelShadedModeAction() const;
    QAction *modelSolidModeAction() const;
    QAction *modelWireframeModeAction() const;
    QAction *modelElevationModeAction() const;
    QAction *modelConfidenceModeAction() const;
    QAction *modelAssignedImageModeAction() const;

    /** @brief 返回"河南大学校徽"显示/隐藏切换动作（可检查状态）。 */
    QAction *toggleHenanUniversityBrandAction() const;

    // ==== 工作流程菜单动作 ====

    /** @brief 返回"添加照片"动作。 */
    QAction *addPhotoAction() const;

    /** @brief 返回"添加文件夹"动作（批量导入目录内的所有图片）。 */
    QAction *addFolderAction() const;

    /** @brief 返回“文件 / 导入 / 导入点云”动作。 */
    QAction *importPointCloudAction() const;

    /** @brief 返回“文件 / 导入 / 导入模型”动作。 */
    QAction *importModelAction() const;

    /** @brief 返回"特征点可视化设置"对话框动作。 */
    QAction *featureVisualizationAction() const;

    /** @brief 返回工作流程中的"空中三角测量"参数对话框动作。 */
    QAction *workflowAerialTriangulationAction() const;

    /** @brief 返回工作流程级高级设置动作。 */
    QAction *workflowSettingsAction() const;

    /** @brief 返回"重叠度获取"分析动作。 */
    QAction *overlapAnalysisAction() const;

    /** @brief 返回"前方交汇检测"动作。 */
    QAction *intersectionCheckAction() const;

    /** @brief 返回"前方交汇查看结果"动作。 */
    QAction *intersectionViewResultsAction() const;

    /** @brief 返回"创建 DEM"动作。 */
    QAction *createDEMAction() const;

    /** @brief 返回"生成正射影像"动作。 */
    QAction *generateOrthoAction() const;

    /** @brief 返回"创建点云"动作。 */
    QAction *createPointCloudAction() const;

    /** @brief 返回"生成模型"动作。 */
    QAction *generateModelAction() const;

    /** @brief 返回"生成纹理"动作。 */
    QAction *generateTextureAction() const;

    /** @brief 返回"查看工作流程报告"动作。 */
    QAction *viewWorkflowReportAction() const;
    QAction *createTiePointsAction() const;
    QAction *thinTiePointsAction() const;
    QAction *cleanTiePointsAction() const;
    QAction *viewTiePointMatchesAction() const;
    QAction *manualPointCloudPruneAction() const;
    QAction *cameraConvertAction() const;
    QAction *generateMaskAction() const;
    QAction *surveyControlAction() const;
    QAction *detectMarkersAction() const;
    QAction *reviewMarkerDetectionsAction() const;
    QAction *printMarkersAction() const;
    QAction *importReferenceDatasetAction() const;
    QAction *referenceQualityCheckAction() const;
    QAction *referenceTerrainBundleAdjustAction() const;

    /** @brief 返回“导出匹配对(.lis)”动作。 */
    QAction *exportMatchedPairsAction() const;

    // ==== 最近打开项目 ====

    /**
     * @brief 根据最近项目路径列表重建"最近打开"子菜单。
     *
     * 每条路径生成一个编号 QAction，触发时发出 recentProjectSelected 信号；
     * 列表末尾添加"清空最近打开"选项；列表为空时显示灰色"（无）"占位项。
     *
     * @param paths 最近打开的项目路径列表（最多 10 条，最新的排在最前）。
     */
    void setRecentProjects(const QStringList &paths);

signals:
    /**
     * @brief 用户点击"最近打开"子菜单中某个项目时发出。
     * @param plascanPath 被选中的 .plascan 文件绝对路径。
     */
    void recentProjectSelected(const QString &plascanPath);

    /** @brief 用户点击"清空最近打开"时发出，供外部调用清空逻辑。 */
    void clearRecentRequested();

private:
    void updateImageActionAvailability();

    /** @brief 目标主窗口指针，用于 addMenu/addToolBar 等操作。 */
    QMainWindow *_mainWindow{};

    /** @brief 主工具栏，供外部追加快捷操作按钮。 */
    QToolBar *_toolBar{};

    /** @brief “文件”顶级菜单。 */
    QMenu *_fileMenu{};

    /** @brief "最近打开"子菜单，内容由 setRecentProjects 动态重建。 */
    QMenu *_recentMenu{};

    /** @brief "模型"顶级菜单，承载 3D 模型视图相关显示控制。 */
    QMenu *_modelMenu{};

    /** @brief "模型 / 显示/隐藏项目"子菜单。 */
    QMenu *_modelDisplayHideMenu{};

    /** @brief "视图 / 窗口"子菜单。 */
    QMenu *_windowMenu{};

    /** @brief "视图 / 影像显示"子菜单。 */
    QMenu *_imageDisplayMenu{};

    /** @brief "工具 / 标记"子菜单。 */
    QMenu *_markersMenu{};

    // ---- 文件菜单中的固定动作 ----
    QAction *_newAct{};   ///< 新建项目
    QAction *_openAct{};  ///< 打开项目
    QAction *_saveAct{};  ///< 保存项目
    QAction *_importPointCloudAct{}; ///< 导入 Metashape/通用点云
    QAction *_importModelAct{}; ///< 导入 Metashape/通用模型
    QAction *_minimizeAct{}; ///< 最小化窗口
    QAction *_exitAct{};  ///< 退出应用
    QAction *_updatePythonRuntimeAct{}; ///< 下载或更新 PlaScan 管理的 Python 环境

    // ---- 视图菜单动作 ----
    QAction *_zoomInAct{};    ///< 放大视图
    QAction *_zoomOutAct{};   ///< 缩小视图
    QAction *_resetViewAct{}; ///< 重置视图到原始比例
    QAction *_saveToolbarWidgetAct{}; ///< 保存项目按钮的工具栏包装动作
    QAction *_zoomInToolbarWidgetAct{}; ///< 放大按钮的工具栏包装动作
    QAction *_zoomOutToolbarWidgetAct{}; ///< 缩小按钮的工具栏包装动作
    QAction *_rotateImageLeftAct{}; ///< 当前影像向左旋转 90 度
    QAction *_rotateImageRightAct{}; ///< 当前影像向右旋转 90 度
    QAction *_showFeaturePointsAct{};
    QAction *_showFeatureResidualsAct{};
    QAction *_showMaskOverlayAct{};
    QAction *_showDepthOverlayAct{};
    QAction *_depthOverlayAllLevelsAct{};
    QAction *_depthOverlayLevel1Act{};
    QAction *_depthOverlayLevel2Act{};
    QAction *_depthOverlayLevel3Act{};
    QAction *_showDepthIntensityAct{};
    QActionGroup *_depthOverlayLevelGroup{};
    QAction *_cameraToolbarWidgetAct{}; ///< 三维相机工具按钮的工具栏包装动作
    QAction *_cameraImageToolbarWidgetAct{}; ///< 三维图像工具按钮的工具栏包装动作
    QAction *_rotateImageLeftToolbarWidgetAct{}; ///< 左转按钮的工具栏包装动作
    QAction *_rotateImageRightToolbarWidgetAct{}; ///< 右转按钮的工具栏包装动作
    QAction *_showFeaturePointsToolbarWidgetAct{};
    QAction *_showMaskOverlayToolbarWidgetAct{};
    QAction *_showDepthOverlayToolbarWidgetAct{};
    QAction *_resetImageViewToolbarWidgetAct{};
    QAction *_toolbarEditingSeparatorAct{}; ///< 视图操作组与编辑操作组之间的分隔符
    QAction *_manualPointCloudPruneToolbarWidgetAct{}; ///< 点云剔除按钮的工具栏包装动作
    QAction *_toggleGizmoAct{}; ///< 显示/隐藏轨迹球
    QAction *_toggleCamerasAct{}; ///< 显示/隐藏 3D 相机覆盖层
    QAction *_toggleDependentCamerasAct{}; ///< 显示/隐藏从属相机（当前仅保留菜单占位）
    QAction *_toggleCameraThumbnailsAct{}; ///< 相机影像平面显示缩略图
    QAction *_toggleLocalAxesAct{}; ///< 工具栏相机菜单中的本地轴显示开关
    QAction *_toggleCameraImagesAct{}; ///< 显示当前相机图像平面
    QAction *_showCameraImagesInForegroundAct{}; ///< 在前景中显示当前相机图像
    QAction *_showCameraImagesInBackgroundAct{}; ///< 在后景中显示当前相机图像
    QAction *_lockCameraImageAct{}; ///< 锁定当前显示的相机图像
    QAction *_tiePointColorModeAct{}; ///< 连接点原始 RGB 颜色
    QAction *_tiePointElevationModeAct{}; ///< 连接点按高程着色
    QAction *_tiePointImageCountModeAct{}; ///< 连接点按影像观测数着色
    QAction *_modelTextureModeAct{}; ///< 模型纹理
    QAction *_modelShadedModeAct{}; ///< 模型平滑阴影
    QAction *_modelSolidModeAct{}; ///< 模型实体
    QAction *_modelWireframeModeAct{}; ///< 模型线框
    QAction *_modelElevationModeAct{}; ///< 模型按高程着色
    QAction *_modelConfidenceModeAct{}; ///< 模型按视角支持可信度着色
    QAction *_modelAssignedImageModeAct{}; ///< 模型按指定影像着色
    QAction *_toggleHenanUniversityBrandAct{}; ///< 显示/隐藏河南大学校徽
    // ---- 面板开关动作 ----
    QAction *_toggleWorkspaceAct{}; ///< 左侧工作区显示/隐藏
    QAction *_togglePropertiesAct{}; ///< 属性面板显示/隐藏
    QAction *_togglePhotosAct{}; ///< 照片面板显示/隐藏
    QAction *_toggleLogAct{}; ///< 日志面板显示/隐藏
    QAction *_toggleMainToolbarAct{}; ///< 主工具栏显示/隐藏
    QAction *_restoreDefaultWindowLayoutAct{}; ///< 恢复默认 Dock 与工具栏布局
    QAction *_toggleFullScreenAct{}; ///< 全屏显示切换
    bool _imageToolsVisible{};
    bool _imageDisplayReady{};
    bool _depthOverlayAvailable{};
    bool _depthOverlayFinalAvailable{};
    bool _depthOverlayLevel1Available{};
    bool _depthOverlayLevel2Available{};
    bool _depthOverlayLevel3Available{};

    // ---- 工作流程菜单动作 ----
    QAction *_addPhotoAct{};            ///< 添加单张照片
    QAction *_addFolderAct{};           ///< 批量添加文件夹中的图片
    QAction *_featureVisualizationAct{};///< 特征点可视化设置对话框
    QAction *_workflowAerialTriangulationAct{}; ///< 工作流程中的对齐照片参数对话框
    QAction *_workflowSettingsAct{};    ///< 工作流程级设备、并行度和数值门限设置
    QAction *_createPointCloudAct{};     ///< 从深度图创建密集点云
    QAction *_generateModelAct{};       ///< 生成模型（源数据选择）
    QAction *_generateTextureAct{};     ///< 生成纹理（已有模型投影纹理）
    QAction *_overlapAnalysisAct{};     ///< 重叠度分析
    QAction *_intersectionCheckAct{};   ///< 前方交汇精度检验
    QAction *_intersectionViewResultsAct{}; ///< 查看前方交汇结果
    QAction *_createDEMAct{};           ///< 创建数字高程模型（DEM）
    QAction *_generateOrthoAct{};       ///< 生成正射影像

    QAction *_viewWorkflowReportAct{};        ///< 查看工作流程历史报告
    QAction *_createTiePointsAct{};           ///< 工具菜单中创建连接点参数入口
    QAction *_thinTiePointsAct{};             ///< 工具菜单中稀释连接点参数入口
    QAction *_cleanTiePointsAct{};            ///< 工具菜单中清理连接点参数入口
    QAction *_viewTiePointMatchesAct{};       ///< 工具菜单中查看匹配入口
    QAction *_manualPointCloudPruneAct{};     ///< 手动点云剔除
    QAction *_cameraConvertAct{};             ///< 通用相机格式转换
    QAction *_generateMaskAct{};              ///< 生成照片蒙版
    QAction *_surveyControlAct{};             ///< 控制点/检查点/比例尺管理
    QAction *_detectMarkersAct{};              ///< 自动检测编码或非编码标靶
    QAction *_reviewMarkerDetectionsAct{};     ///< 人工复核检测候选与身份冲突
    QAction *_printMarkersAct{};               ///< 生成物理尺寸明确的标靶 PDF
    QAction *_importReferenceDatasetAct{};    ///< 导入外部 DEM/LiDAR 参考数据
    QAction *_referenceQualityCheckAct{};     ///< 使用参考数据生成精度检查报告
    QAction *_referenceTerrainBundleAdjustAct{}; ///< 参考地形软约束 BA 前置检查

    QAction *_exportMatchedPairsAct{};      ///< 导出匹配影像对 .lis
};
