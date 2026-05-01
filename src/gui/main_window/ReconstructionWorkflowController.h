// =============================================================================
// 文件: ReconstructionWorkflowController.h
// 模块: main_window
// 说明:
//   重建菜单业务控制器。管理「重建」顶层菜单中所有对话框的创建、
//   设置加载/保存（记忆化）以及运行请求的转发。
//
//   设计原则:
//   - 从 MenuWorkflowController 独立出来，避免单类过大
//   - 使用模板 prepareDialog<T>() 复用对话框创建/设置恢复/保存的公共逻辑
//   - 每个 open*() 槽仅包含该对话框特有的「运行」逻辑
// =============================================================================

#pragma once

#include <QObject>
#include <QPointer>
#include <QJsonObject>

#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"
#include "Logger.h"

class QMainWindow;
class ProjectManager;

class ReconstructionWorkflowController : public QObject
{
    Q_OBJECT

public:
    /// 构造重建菜单业务控制器。
    /// @param mainWindow 父主窗口。
    /// @param parent QObject 父对象。
    explicit ReconstructionWorkflowController(QMainWindow *mainWindow, QObject *parent = nullptr);

    /// 注入项目管理器，供各重建步骤查询项目状态和提交任务。
    /// @param pm 当前项目管理器，非拥有引用。
    void setProjectManager(ProjectManager *pm);

public slots:
    // ── 稀疏重建 ──
    /// 打开观测网络构建对话框。
    void openObservationNetworkDialog();

    /// 打开初始相机位姿恢复对话框。
    void openInitCameraPoseDialog();

    /// 打开初始稀疏点云三角化对话框。
    void openTriangulationDialog();

    /// 打开稀疏重建阶段的光束法平差对话框。
    void openReconBundleAdjustDialog();

    /// 打开统一稀疏点云后处理对话框。
    void openSparseCloudPostProcessDialog();

    // ── 密集重建 ──
    void openDenseMatchDialog();
    void openDepthMapEstimateDialog();
    void openDepthFusionDialog();
    void openDenseCloudRefineDialog();

    // ── 模型生成 ──
    void openMeshReconstructionDialog();
    void openTextureMappingDialog();
    void openModelExportDialog();

private:
    /**
     * @brief 通用对话框创建/设置恢复辅助模板。
     *
     * 完成与每个重建对话框相同的公共步骤:
     *   1. new DialogT(parent), WA_DeleteOnClose
     *   2. 从 DialogSettingStore 加载并恢复上次保存的参数
     *   3. 连接 settingsChanged → store.save()
     *
     * @tparam DialogT    对话框类型（须有 applySettings / settingsChanged 信号）
     * @param  settingKey DialogSettingKeys 中对应的键
     * @param  store      对应的 DialogSettingStore* 成员（引用，延迟初始化）
     * @return 已准备就绪的对话框指针；调用方只需连接 runRequested 并 exec()
     */
    template<typename DialogT>
    DialogT *prepareDialog(const QString &settingKey, DialogSettingStore *&store)
    {
        if (!m_mainWindow)
        {
            return nullptr;
        }

        auto *dlg = new DialogT(m_mainWindow);
        dlg->setAttribute(Qt::WA_DeleteOnClose);

        if (m_projectManager)
        {
            if (!store)
            {
                store = new DialogSettingStore(settingKey, this);
            }
            store->setProjectPath(projectPath());

            const QJsonObject saved = store->load();
            if (!saved.isEmpty())
            {
                dlg->applySettings(saved);
            }
        }

        // 实时持久化参数变更
        auto *storePtr = store;
        connect(dlg, &DialogT::settingsChanged, this, [storePtr](const QJsonObject &s)
        {
            if (storePtr)
            {
                storePtr->save(s);
            }
        });

        return dlg;
    }

    /// 获取当前项目路径，避免重复查询逻辑。
    QString projectPath() const;

    QPointer<QMainWindow> m_mainWindow;
    ProjectManager       *m_projectManager = nullptr;

    // ── 各对话框的设置存储（延迟初始化） ──
    DialogSettingStore *m_obsNetStore       = nullptr;
    DialogSettingStore *m_initPoseStore     = nullptr;
    DialogSettingStore *m_triStore          = nullptr;
    DialogSettingStore *m_reconBaStore      = nullptr;
    DialogSettingStore *m_sparsePostStore   = nullptr;
    DialogSettingStore *m_denseMatchStore   = nullptr;
    DialogSettingStore *m_depthEstStore     = nullptr;
    DialogSettingStore *m_depthFuseStore    = nullptr;
    DialogSettingStore *m_denseRefStore     = nullptr;
    DialogSettingStore *m_meshStore         = nullptr;
    DialogSettingStore *m_texStore          = nullptr;
    DialogSettingStore *m_exportStore       = nullptr;
};
