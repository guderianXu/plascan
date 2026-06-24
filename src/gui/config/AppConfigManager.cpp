/**
 * @file AppConfigManager.cpp
 * @brief AppConfigManager 的实现文件。
 *
 * 本文件仅包含构造函数，所有实际逻辑均由各子管理器负责实现。
 */
#include "AppConfigManager.h"

/**
 * @brief 构造函数。
 *
 * 通过成员初始化列表将三个子管理器的父对象设置为 this，
 * 使其生命周期与 AppConfigManager 完全绑定；当 AppConfigManager
 * 被销毁时，Qt 对象树会自动递归销毁所有子对象，无需手动 delete。
 */
AppConfigManager::AppConfigManager(QObject *parent)
    : QObject(parent)
    , _windowState(this)      // 窗口状态管理器，父对象为本实例
    , _recentProjects(this)   // 最近项目管理器，父对象为本实例
    , _fileDialogs(this)      // 文件对话框状态管理器，父对象为本实例
{
}
