// =============================================================================
// 文件: ReconstructionWorkflowController.h
// 模块: main_window
// 说明:
//   管理工作流程菜单中的点云、模型生成与纹理生成对话框，
//   负责设置加载/保存（记忆化）以及运行请求的转发。
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

    /// 注入项目管理器，供模型工作流程查询项目状态和提交任务。
    /// @param pm 当前项目管理器，非拥有引用。
    void setProjectManager(ProjectManager *pm);

public slots:
    void openCreatePointCloudDialog();
    void openGenerateModelDialog();
    void openTextureMappingDialog();

private:
    /**
     * @brief 通用对话框创建/设置恢复辅助模板。
     *
     * 完成模型工作流程对话框相同的公共步骤:
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
        if (!_mainWindow)
        {
            return nullptr;
        }

        auto *dlg = new DialogT(_mainWindow);
        dlg->setAttribute(Qt::WA_DeleteOnClose);

        if (_projectManager)
        {
            if (!store)
            {
                store = new DialogSettingStore(settingKey, this);
                store->setChangeCallback([this]()
                {
                    markProjectWorkspaceDirty();
                });
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
    void markProjectWorkspaceDirty();

    QPointer<QMainWindow> _mainWindow;
    ProjectManager       *_projectManager = nullptr;

    DialogSettingStore *_createPointCloudStore = nullptr;
    DialogSettingStore *_generateModelStore = nullptr;
    DialogSettingStore *_texStore          = nullptr;
    DialogSettingStore *_workflowSettingsStore = nullptr;
};
