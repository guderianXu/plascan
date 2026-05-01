/**
 * @file RecentProjectsManager.cpp
 * @brief RecentProjectsManager 的实现文件。
 *
 * 所有数据均通过 QSettings("PlaScan", "plascan_gui") 以键
 * "Project/recent" 持久化；每次操作都直接读写 QSettings，
 * 不在内存中预缓存，以保证多处调用时数据的一致性。
 */
#include "RecentProjectsManager.h"

#include <QSettings>

/**
 * @brief 构造函数，无额外初始化逻辑。
 */
RecentProjectsManager::RecentProjectsManager(QObject *parent)
    : QObject(parent)
{
}

/**
 * @brief 获取最近打开的项目路径列表。
 *
 * 直接从 QSettings 读取，确保反映磁盘的最新状态。
 *
 * @return 路径字符串列表（最多 10 条），最新的排在最前；无记录时返回空列表。
 */
QStringList RecentProjectsManager::recentProjects() const
{
    QSettings settings("PlaScan", "plascan_gui");
    return settings.value("Project/recent", QStringList()).toStringList();
}

/**
 * @brief 将指定路径添加到最近打开列表的首位并持久化。
 *
 * @param plascanPath .plascan 项目文件的绝对路径；空字符串时直接忽略。
 */
void RecentProjectsManager::addRecentProject(const QString &plascanPath)
{
    // 忽略空路径，防止写入无意义记录
    if (plascanPath.trimmed().isEmpty()) return;

    QSettings settings("PlaScan", "plascan_gui");
    QStringList list = settings.value("Project/recent", QStringList()).toStringList();

    // 先移除列表中已有的同路径条目，避免重复
    list.removeAll(plascanPath);

    // 将新路径插入首位，使"最近打开"菜单按最新顺序排列
    list.prepend(plascanPath);

    // 保持列表条目数不超过 10，超出时从尾部删除最旧记录
    while (list.size() > 10) list.removeLast();

    settings.setValue("Project/recent", list);
}

/**
 * @brief 清空全部最近打开记录。
 *
 * 将 "Project/recent" 键写为空列表，下次读取时返回空。
 */
void RecentProjectsManager::clearRecentProjects()
{
    QSettings settings("PlaScan", "plascan_gui");
    settings.setValue("Project/recent", QStringList());
}
