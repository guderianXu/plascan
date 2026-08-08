/**
 * @file FileDialogStateManager.cpp
 * @brief FileDialogStateManager 的实现文件。
 *
 * 所有数据均通过 QSettings("PlaScan", "plascan_gui") 持久化，
 * 在 Linux 下通常存储在 ~/.config/PlaScan/plascan_gui.conf。
 */
#include "FileDialogStateManager.h"

#include "GuiSettingsStore.h"

#include <QDir>

/**
 * @brief 获取指定 key 对应对话框的上次浏览目录。
 *
 * 实现细节：
 * 1. 将 key 去除首尾空格；若为空则使用字面量 "default" 作为键名，
 *    防止 QSettings 键路径异常。
 * 2. 从 "Dialogs/lastDir/<key>" 读取值；若不存在则回退到用户家目录，
 *    使对话框至少有一个合理的起始目录。
 *
 * @param key  对话框类型标识符。
 * @return     目录路径字符串，保证非空。
 */
QString FileDialogStateManager::lastDir(const QString &key) const
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    // 空 key 统一存为 "default"，避免生成无意义的空键名
    const QString k = key.trimmed().isEmpty() ? QStringLiteral("default") : key.trimmed();
    return settings.value(QStringLiteral("Dialogs/lastDir/%1").arg(k), QDir::homePath()).toString();
}

/**
 * @brief 记录指定 key 对应对话框的最近浏览目录。
 *
 * 实现细节：同 lastDir 一样对 key 做 trim 与空值处理，
 * 然后将 dir 写入 QSettings 对应键路径，立即持久化到磁盘。
 *
 * @param key  对话框类型标识符。
 * @param dir  需要持久化的目录路径。
 */
void FileDialogStateManager::setLastDir(const QString &key, const QString &dir)
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    // 同 lastDir，空 key 统一为 "default"
    const QString k = key.trimmed().isEmpty() ? QStringLiteral("default") : key.trimmed();
    settings.setValue(QStringLiteral("Dialogs/lastDir/%1").arg(k), dir);
}
