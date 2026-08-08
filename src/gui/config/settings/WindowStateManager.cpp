/**
 * @file WindowStateManager.cpp
 * @brief WindowStateManager 的实现文件。
 *
 * 实现主窗口状态的加载与保存逻辑，使用 QSettings 持久化所有数据。
 * QSettings 组织名为 "PlaScan"，应用名为 "plascan_gui"。
 */
#include "WindowStateManager.h"

#include "GuiSettingsStore.h"

#include <QMainWindow>

/**
 * @brief 从 QSettings 恢复主窗口的几何尺寸和最大化状态。
 *
 * 执行步骤：
 * 1. 检查 "hasRunBefore" 标志；若为 false（首次运行），
 *    则直接设置全屏并写入标志后返回，不读取旧几何数据（因为不存在）。
 * 2. 非首次运行时，依次恢复：
 *    a. 窗口几何（位置与大小）——通过 restoreGeometry；
 *    b. 窗口状态标志——若上次退出时处于全屏或最大化则恢复为最大化。
 *
 * @note 最大化恢复放在 geometry 之后，防止普通窗口尺寸覆盖用户期望的屏幕占用状态。
 *
 * @param mainWindow 需要恢复状态的主窗口；为 nullptr 时立即返回。
 */
void WindowStateManager::load(QMainWindow *mainWindow)
{
    if (!mainWindow) return;

    QSettings settings = xjw::gui::settings::createGuiSettings();

    // 检查是否为首次运行
    bool hasRun = settings.value("hasRunBefore", false).toBool();
    if (!hasRun) {
        // 首次运行：使用最大化而不是全屏，保留系统标题栏与最小化/关闭按钮。
        mainWindow->setWindowState(mainWindow->windowState() | Qt::WindowMaximized);
        // 写入标志，确保后续运行不再触发首次全屏逻辑
        settings.setValue("hasRunBefore", true);
        return;
    }

    const QByteArray geo   = settings.value("MainWindow/geometry").toByteArray();
    if (!geo.isEmpty())   mainWindow->restoreGeometry(geo);

    // 恢复窗口状态：将历史全屏回退为最大化，确保窗口控制按钮可用。
    const bool wasFull = settings.value("MainWindow/isFullScreen", false).toBool();
    const bool wasMaximized = settings.value("MainWindow/isMaximized", false).toBool();
    if (wasFull || wasMaximized)
    {
        mainWindow->setWindowState((mainWindow->windowState() & ~Qt::WindowFullScreen) | Qt::WindowMaximized);
    }
}

/**
 * @brief 将主窗口当前状态持久化到 QSettings。
 *
 * 保存内容：
 * - "MainWindow/geometry"     : 当前窗口几何（经 saveGeometry 序列化）；
 * - "MainWindow/isFullScreen" : 当前是否处于全屏模式的布尔标志；
 * - "MainWindow/isMaximized"  : 当前是否处于最大化模式的布尔标志。
 *
 * 应在 closeEvent 中调用，以保证数据在进程退出前已写入磁盘。
 *
 * @param mainWindow 需要保存状态的主窗口；为 nullptr 时立即返回。
 */
void WindowStateManager::save(QMainWindow *mainWindow)
{
    if (!mainWindow) return;

    QSettings settings = xjw::gui::settings::createGuiSettings();
    settings.setValue("MainWindow/geometry",     mainWindow->saveGeometry());
    settings.setValue("MainWindow/isFullScreen", mainWindow->isFullScreen());
    settings.setValue("MainWindow/isMaximized",  mainWindow->isMaximized());
}
