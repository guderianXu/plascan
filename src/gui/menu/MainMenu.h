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
 *   - 帮助：关于
 */

#include <QObject>

class QMainWindow;
class QAction;
class QToolBar;
class QMenu;

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
     * 优先绑定 MainWindow.ui 中的项目、视图、工作流程、重建、工具、帮助菜单和主工具栏；
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

    /** @brief 返回"兴趣点信息面板"显示/隐藏切换动作（可检查状态的 QAction）。 */
    QAction *featureInfoAction() const;

    // ==== 工具栏 ====

    /** @brief 返回主工具栏指针，外部可向其添加额外的操作按钮。 */
    QToolBar *toolBar() const;

    // ==== 文件/项目菜单动作 ====

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

    // ==== 视图菜单动作 ====

    /** @brief 返回"放大"动作。 */
    QAction *zoomInAction() const;

    /** @brief 返回"缩小"动作。 */
    QAction *zoomOutAction() const;

    /** @brief 返回"重置视图"动作。 */
    QAction *resetViewAction() const;

    /** @brief 返回"显示操控球"切换动作（可检查状态）。 */
    QAction *toggleGizmoAction() const;

    /** @brief 返回"显示相机"切换动作（可检查状态）。 */
    QAction *toggleCamerasAction() const;

    /** @brief 返回"河南大学校徽"显示/隐藏切换动作（可检查状态）。 */
    QAction *toggleHenanUniversityBrandAction() const;

    // ==== 工作流程菜单动作 ====

    /** @brief 返回"添加照片"动作。 */
    QAction *addPhotoAction() const;

    /** @brief 返回"添加文件夹"动作（批量导入目录内的所有图片）。 */
    QAction *addFolderAction() const;

    /** @brief 返回"特征点查找"（通用特征检测）动作。 */
    QAction *detectFeaturesAction() const;

    /** @brief 返回"获取重叠对"动作。 */
    QAction *vocabularyOverlapAction() const;

    /** @brief 返回"特征点可视化设置"对话框动作。 */
    QAction *featureVisualizationAction() const;

    /** @brief 返回"创建连接点"（通用特征匹配）动作。 */
    QAction *matchFeaturesAction() const;

    /** @brief 返回"查看匹配"（匹配结果可视化）动作。 */
    QAction *viewMatchesAction() const;
    QAction *denseMatchAction() const;

    /** @brief 返回"三维重建"（一键完整建模）动作。 */
    QAction *threeDReconstructionAction() const;

    /** @brief 返回工作流程中的"空中三角测量"参数对话框动作。 */
    QAction *workflowAerialTriangulationAction() const;

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

    /** @brief 返回"生成模型"动作。 */
    QAction *generateModelAction() const;

    /** @brief 返回"查看工作流程报告"动作。 */
    QAction *viewWorkflowReportAction() const;
    QAction *createTiePointsAction() const;
    QAction *thinTiePointsAction() const;
    QAction *cleanTiePointsAction() const;
    QAction *viewTiePointMatchesAction() const;
    QAction *manualPointCloudPruneAction() const;
    QAction *cameraConvertAction() const;
    QAction *surveyControlAction() const;
    QAction *importReferenceDatasetAction() const;
    QAction *referenceQualityCheckAction() const;
    QAction *referenceTerrainBundleAdjustAction() const;

    // ==== 重建菜单动作（稀疏/密集/模型） ====

    QAction *buildObsNetworkAction() const;
    QAction *initCameraPoseAction() const;
    QAction *aerialTriangulationAction() const;
    QAction *triangulateAction() const;
    QAction *reconBundleAdjustAction() const;
    QAction *sparseCloudPostProcessAction() const;
    QAction *depthMapEstimateAction() const;
    QAction *fuseDepthMapsAction() const;
    QAction *refineDenseCloudAction() const;
    QAction *meshReconstructAction() const;
    QAction *textureMappingAction() const;
    QAction *exportModelAction() const;

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
    /** @brief 目标主窗口指针，用于 addMenu/addToolBar 等操作。 */
    QMainWindow *_mainWindow{};

    /** @brief 主工具栏，供外部追加快捷操作按钮。 */
    QToolBar *_toolBar{};

    /** @brief "项目"顶级菜单（原名"文件"）。 */
    QMenu *_fileMenu{};

    /** @brief "最近打开"子菜单，内容由 setRecentProjects 动态重建。 */
    QMenu *_recentMenu{};

    // ---- 项目菜单中的固定动作 ----
    QAction *_newAct{};   ///< 新建项目
    QAction *_openAct{};  ///< 打开项目
    QAction *_saveAct{};  ///< 保存项目
    QAction *_minimizeAct{}; ///< 最小化窗口
    QAction *_exitAct{};  ///< 退出应用

    // ---- 视图菜单动作 ----
    QAction *_zoomInAct{};    ///< 放大视图
    QAction *_zoomOutAct{};   ///< 缩小视图
    QAction *_resetViewAct{}; ///< 重置视图到原始比例
    QAction *_toggleGizmoAct{}; ///< 显示/隐藏操控球
    QAction *_toggleCamerasAct{}; ///< 显示/隐藏 3D 相机覆盖层
    QAction *_toggleHenanUniversityBrandAct{}; ///< 显示/隐藏河南大学校徽
    QAction *_featureInfoAct{}; ///< 兴趣点信息面板开关

    // ---- 面板开关动作 ----
    QAction *_toggleWorkspaceAct{}; ///< 左侧工作区显示/隐藏
    QAction *_togglePropertiesAct{}; ///< 属性面板显示/隐藏
    QAction *_togglePhotosAct{}; ///< 照片面板显示/隐藏
    QAction *_toggleLogAct{}; ///< 日志面板显示/隐藏

    // ---- 工作流程菜单动作 ----
    QAction *_addPhotoAct{};            ///< 添加单张照片
    QAction *_addFolderAct{};           ///< 批量添加文件夹中的图片
    QAction *_detectFeaturesAct{};      ///< 特征点检测
    QAction *_vocabularyOverlapAct{};   ///< 基于特征词汇预获取重叠影像对
    QAction *_featureVisualizationAct{};///< 特征点可视化设置对话框
    QAction *_matchFeaturesAct{};       ///< 特征点匹配生成连接点
    QAction *_viewMatchesAct{};         ///< 查看匹配结果
    QAction *_denseMatchAct{};          ///< 密集匹配
    QAction *_workflowAerialTriangulationAct{}; ///< 工作流程中的对齐照片参数对话框
    QAction *_threeDReconstructionAct{}; ///< 三维重建（一键完整建模）
    QAction *_generateModelAct{};       ///< 生成模型（源数据选择）
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
    QAction *_surveyControlAct{};             ///< 控制点/检查点/比例尺管理
    QAction *_importReferenceDatasetAct{};    ///< 导入外部 DEM/LiDAR 参考数据
    QAction *_referenceQualityCheckAct{};     ///< 使用参考数据生成精度检查报告
    QAction *_referenceTerrainBundleAdjustAct{}; ///< 参考地形软约束 BA 前置检查

    // ---- 重建菜单动作 ----
    QAction *_buildObsNetworkAct{};         ///< 构建观测网络模型
    QAction *_initCameraPoseAct{};          ///< 初始化相机位姿
    QAction *_aerialTriangulationAct{};     ///< 正式空中三角测量（SfM + BA）
    QAction *_triangulateAct{};             ///< 两视预览云三角化
    QAction *_reconBundleAdjustAct{};       ///< 光束法平差（重建菜单）
    QAction *_sparseCloudPostProcessAct{};  ///< 稀疏点云后处理
    QAction *_depthMapEstimateAct{};        ///< 深度图估计
    QAction *_fuseDepthMapsAct{};           ///< 深度图融合
    QAction *_refineDenseCloudAct{};        ///< 密集点云后处理
    QAction *_meshReconstructAct{};         ///< 网格重建
    QAction *_textureMappingAct{};          ///< 纹理映射
    QAction *_exportModelAct{};             ///< 模型导出
    QAction *_exportMatchedPairsAct{};      ///< 导出匹配影像对 .lis
};
