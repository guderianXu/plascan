#pragma once

/**
 * @file AppConfigManager.h
 * @brief 应用级配置管理器的声明文件。
 *
 * AppConfigManager 作为顶层聚合入口，将所有与"应用全局状态"相关的
 * 子管理器集中在一起，便于主窗口或其他模块通过统一接口访问：
 *   - 窗口几何/状态持久化（WindowStateManager）
 *   - 最近打开项目列表（RecentProjectsManager）
 *   - 文件对话框最近目录（FileDialogStateManager）
 *
 * 所有子管理器均以 QObject 为父节点托管在本对象中，生命周期由 Qt 对象树管理。
 */

#include <QObject>

#include "settings/WindowStateManager.h"
#include "settings/RecentProjectsManager.h"
#include "settings/FileDialogStateManager.h"

/**
 * @class AppConfigManager
 * @brief 聚合应用级配置管理器。
 *
 * 本类不直接读写任何配置，而是将职责委托给各专职子管理器。
 * 外部代码只需持有一个 AppConfigManager 实例，即可访问全部应用级持久化状态。
 */
class AppConfigManager : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数，初始化所有子管理器并将其挂载到本对象的 Qt 对象树。
     * @param parent 父对象指针，用于 Qt 内存管理（通常为 nullptr 或主窗口）。
     */
    explicit AppConfigManager(QObject *parent = nullptr);

    /** @brief 返回窗口几何/状态持久化管理器的指针。 */
    WindowStateManager *windowState() { return &_windowState; }

    /** @brief 返回最近打开项目列表管理器的指针。 */
    RecentProjectsManager *recentProjects() { return &_recentProjects; }

    /** @brief 返回文件对话框最近目录管理器的指针。 */
    FileDialogStateManager *fileDialogs() { return &_fileDialogs; }

private:
    /** @brief 负责保存/恢复主窗口尺寸、位置和停靠布局。 */
    WindowStateManager _windowState;

    /** @brief 维护最近打开的 .plascan 项目路径列表（上限 10 条）。 */
    RecentProjectsManager _recentProjects;

    /** @brief 为各文件对话框记住上次浏览的目录，提升操作效率。 */
    FileDialogStateManager _fileDialogs;
};
