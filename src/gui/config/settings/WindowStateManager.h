#pragma once

/**
 * @file WindowStateManager.h
 * @brief 主窗口几何状态持久化管理器的声明文件。
 *
 * WindowStateManager 负责在应用退出时保存、在下次启动时恢复主窗口的：
 *   - 几何信息（位置、大小）：通过 QMainWindow::saveGeometry/restoreGeometry；
 *   - 最大化/全屏状态：单独以布尔值存储，因为这些状态不能只依赖 geometry。
 *
 * 特殊逻辑：
 *   首次运行（"hasRunBefore" 键不存在）时自动最大化，
 *   为新用户提供最佳的初始视图体验，同时写入该标志避免后续重复触发。
 *
 * 数据通过 QSettings("PlaScan", "plascan_gui") 持久化。
 */

class QMainWindow;

/**
 * @class WindowStateManager
 * @brief 负责主窗口几何状态的保存与恢复。
 */
class WindowStateManager
{
public:
    /**
     * @brief 从 QSettings 恢复主窗口的几何尺寸和最大化/全屏状态。
     *
     * 应在主窗口 show() 之前（或之后立即）调用，以确保窗口以上次状态呈现。
     * 首次运行时跳过恢复并设置最大化，写入 "hasRunBefore" = true 标志。
     *
     * @param mainWindow 需要恢复状态的主窗口指针；为 nullptr 时函数直接返回。
     */
    void load(QMainWindow *mainWindow);

    /**
     * @brief 将主窗口当前的几何尺寸和窗口状态保存到 QSettings。
     *
     * 应在主窗口关闭事件（closeEvent）中调用，确保在进程退出前数据已落盘。
     *
     * @param mainWindow 需要保存状态的主窗口指针；为 nullptr 时函数直接返回。
     */
    void save(QMainWindow *mainWindow);
};
